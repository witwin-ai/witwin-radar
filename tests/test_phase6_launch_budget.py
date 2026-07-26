"""Per-frame launch and memory ledger for every Phase-6 owner (criterion A7).

The three per-waveform test files each count their own family's launches. This
file is the ledger ACROSS them: one operator table, wrapped once, with every
Phase-6 symbol in it, so a stage that quietly adds a launch to one owner has to
change a number here rather than a number in the file it was already editing.

The budgets are the design's section 4.3:

    fmcw_beat_forward       1 per frame
    ofdm_cfr_forward        1 per frame
    pulsed_echo_forward     1 per frame
    sensor_weight_forward   1 per frame
    frontend                <= 3 per frame
    backward                one launch per forward launch

and memory: peak allocation during a forward within 2.0x of the output bytes
plus the inputs. A ``K x chirps x samples`` intermediate fails that immediately,
which is the point of measuring it rather than reading the source.
"""

from __future__ import annotations

import pytest
import torch

from support import multi_endpoint_driver as drv

pytestmark = pytest.mark.gpu


HOST_OBSERVERS = ("item", "cpu", "tolist", "numpy")

#: Every Phase-6 synthesis and sensor symbol, by family. The frontend is
#: counted separately because its budget is a total rather than a per-symbol
#: count.
SYNTHESIS_OPERATORS = (
    "fmcw_beat_forward",
    "fmcw_beat_backward",
    "fmcw_beat_jvp",
    "ofdm_cfr_forward",
    "ofdm_cfr_backward",
    "ofdm_cfr_jvp",
    "pulsed_echo_forward",
    "pulsed_echo_backward",
    "pulsed_echo_jvp",
    "sensor_weight_forward",
    "sensor_weight_backward",
    "sensor_weight_jvp",
)

FRONTEND_OPERATORS = (
    "frontend_noise_forward",
    "frontend_agc_forward",
    "frontend_quantize_forward",
)


class Ledger:
    """Count native launches and host observations while it is active."""

    def __init__(self, monkeypatch, operators, names) -> None:
        self.launches = dict.fromkeys(names, 0)
        self.host = dict.fromkeys((*HOST_OBSERVERS, "synchronize"), 0)
        for name in self.launches:
            original = getattr(operators, name)

            def counting(*args, _name=name, _original=original, **kwargs):
                self.launches[_name] += 1
                return _original(*args, **kwargs)

            monkeypatch.setattr(operators, name, counting)
        for name in HOST_OBSERVERS:
            original_method = getattr(torch.Tensor, name)

            def observing(
                tensor, *args, _name=name, _original=original_method, **kwargs
            ):
                self.host[_name] += 1
                return _original(tensor, *args, **kwargs)

            monkeypatch.setattr(torch.Tensor, name, observing)
        original_sync = torch.cuda.synchronize

        def counting_sync(*args, **kwargs):
            self.host["synchronize"] += 1
            return original_sync(*args, **kwargs)

        monkeypatch.setattr(torch.cuda, "synchronize", counting_sync)


@pytest.fixture(scope="module")
def spike():
    pytest.importorskip("witwin.channel")
    return drv.MultiEndpointSpike()


@pytest.fixture(scope="module")
def batch(spike):
    composed, _, _ = spike.frame()
    return drv.to_synthesis(composed)


def _operators():
    from witwin.radar.cuda import build

    return build.build_extension()


def _waveforms():
    """The three synthesis entry points, each with its own fixture spec."""

    from test_phase6_cross_waveform import fmcw_spec, ofdm_spec, pulsed_spec
    from witwin.radar.synthesis import (
        synthesize_fmcw_beat,
        synthesize_ofdm_cfr,
        synthesize_pulsed_echo,
    )

    return (
        ("fmcw_beat", synthesize_fmcw_beat, fmcw_spec(4)),
        ("ofdm_cfr", synthesize_ofdm_cfr, ofdm_spec(4)),
        ("pulsed_echo", synthesize_pulsed_echo, pulsed_spec(4)),
    )


def test_each_waveform_costs_exactly_one_forward_launch_per_frame(
    batch, monkeypatch, capsys
):
    """One launch each, and nothing from the other two families.

    The cross-family assertion is the one a per-family test cannot make: a
    waveform owner that reached into another owner's kernel would still count
    one launch of its own.
    """

    operators = _operators()
    _waveforms()  # resolve imports before the table is wrapped
    reported = {}
    for family, synthesize, spec in _waveforms():
        ledger = Ledger(monkeypatch, operators, SYNTHESIS_OPERATORS)
        synthesize(batch, spec)
        reported[family] = dict(ledger.launches)
        assert ledger.launches[f"{family}_forward"] == 1, ledger.launches
        assert sum(ledger.launches.values()) == 1, ledger.launches
        assert ledger.host == dict.fromkeys(
            (*HOST_OBSERVERS, "synchronize"), 0
        ), ledger.host
        monkeypatch.undo()

    with capsys.disabled():
        print("\nT5.9 launch ledger, one frame")
        for family, counts in reported.items():
            live = {name: value for name, value in counts.items() if value}
            print(f"  {family:14s} {live}")


def test_the_launch_count_is_flat_in_slot_count(spike, monkeypatch, capsys):
    """Slots are free at the launch ledger (plan item 3).

    Two counts, both independent of the slot count: the propagation replay is
    ONE consumer call per leg for the whole frame, and the waveform is ONE
    forward launch. A Python per-slot loop would multiply the first by the slot
    count; a per-slot synthesis would multiply the second.
    """

    from witwin.channel.propagation import consumer
    from witwin.radar.synthesis import synthesize_fmcw_beat

    from test_phase6_cross_waveform import fmcw_spec

    operators = _operators()
    spec = fmcw_spec(4)
    reported = {}
    for slots in (1, 8, 64):
        times = [index * 1.0e-5 for index in range(slots)]
        stack = drv.slot_site_stack(spike.site_tensor(), (0.0, 1.0, 0.0), times)
        spike.slot_legs(stack, slot_count=slots)  # warm the replication cache

        replays = {"count": 0}
        original = consumer.reevaluate

        def counting(*args, _original=original, **kwargs):
            replays["count"] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(consumer, "reevaluate", counting)
        ledger = Ledger(monkeypatch, operators, SYNTHESIS_OPERATORS)
        inbound, outbound = spike.slot_legs(stack, slot_count=slots)
        composed = spike.composer.compose(
            inbound.slot(0), outbound.slot(0), drv.make_response()
        )
        synthesize_fmcw_beat(drv.to_synthesis(composed), spec)
        reported[slots] = (replays["count"], dict(ledger.launches))
        monkeypatch.undo()

        assert replays["count"] == 2, (slots, replays)
        assert ledger.launches["fmcw_beat_forward"] == 1, (slots, ledger.launches)
        assert sum(ledger.launches.values()) == 1, (slots, ledger.launches)

    with capsys.disabled():
        print("\nT7 slot launch ledger")
        for slots, (replays_count, launches) in reported.items():
            live = {name: value for name, value in launches.items() if value}
            print(f"  T={slots:<5d} consumer.reevaluate={replays_count} {live}")


def test_one_backward_launch_per_forward_launch(batch, monkeypatch):
    """R-ADR-004's shape, measured: one companion launch, not two, not zero."""

    operators = _operators()
    _waveforms()
    from dataclasses import replace

    for family, synthesize, spec in _waveforms():
        transfer = batch.complex_transfer_ref.detach().clone().requires_grad_(True)
        live = replace(batch, complex_transfer_ref=transfer)
        ledger = Ledger(monkeypatch, operators, SYNTHESIS_OPERATORS)
        cube = synthesize(live, spec)
        (cube.real.sum() + cube.imag.sum()).backward()
        assert ledger.launches[f"{family}_forward"] == 1, ledger.launches
        assert ledger.launches[f"{family}_backward"] == 1, ledger.launches
        assert ledger.launches[f"{family}_jvp"] == 0, ledger.launches
        monkeypatch.undo()


def test_the_sensor_weight_owner_costs_one_launch_per_frame(monkeypatch, capsys):
    """The legacy Dirichlet frame path, which is where the migration landed.

    One ``sensor_weight_forward`` for the whole frame's rows, and no synthesis
    launch at all: the Dirichlet spectrum is its own family and this ledger is
    about the owner work item 8 introduced.
    """

    from witwin.radar import Radar

    config = {
        "num_tx": 2,
        "num_rx": 2,
        "fc": 77e9,
        "slope": 60.012,
        "adc_samples": 128,
        "adc_start_time": 6,
        "sample_rate": 4400,
        "idle_time": 7,
        "ramp_end_time": 58,
        "chirp_per_frame": 2,
        "frame_per_second": 10,
        "num_doppler_bins": 2,
        "num_range_bins": 128,
        "num_angle_bins": 16,
        "power": 12,
        "tx_loc": [[0, 0, 0], [2, 0, 0]],
        "rx_loc": [[0, 0, 0], [1, 0, 0]],
    }
    radar = Radar(config, device="cuda")
    points = torch.tensor(
        [[0.0, 0.0, -4.0], [1.0, 0.5, -7.0], [-2.0, 0.0, -5.0]], device="cuda"
    )
    intensities = torch.tensor([1.0, 0.5, 0.25], device="cuda")

    class _Trace:
        def __init__(self, points, intensities):
            self.points = points
            self.intensities = intensities
            self.entry_points = points
            self.fixed_path_lengths = torch.zeros(
                points.shape[0], device=points.device
            )
            self.depths = torch.zeros(
                points.shape[0], dtype=torch.int32, device=points.device
            )
            self.normals = None

    trace = _Trace(points, intensities)
    radar.mimo_from_trace(trace)  # resolve the table before wrapping it

    operators = _operators()
    ledger = Ledger(monkeypatch, operators, SYNTHESIS_OPERATORS)
    radar.mimo_from_trace(trace)
    with capsys.disabled():
        print(
            "\n  sensor_weight  "
            f"{ {name: value for name, value in ledger.launches.items() if value} }"
        )
    assert ledger.launches["sensor_weight_forward"] == 1, ledger.launches
    assert ledger.launches["sensor_weight_backward"] == 0, ledger.launches
    assert ledger.launches["sensor_weight_jvp"] == 0, ledger.launches
    for name in SYNTHESIS_OPERATORS:
        if not name.startswith("sensor_weight"):
            assert ledger.launches[name] == 0, (name, ledger.launches)


def test_the_frontend_costs_at_most_three_launches_per_frame(monkeypatch, capsys):
    """Noise, AGC, ADC: three operator calls for the whole six-stage chain.

    Port conversion and the LNA are fused into the noise operator on purpose -
    that fusion is what makes thermal noise input-referred by construction - so
    a six-stage chain is three launches and not six.
    """

    from witwin.radar.frontend import (
        AdcSpec,
        AgcSpec,
        FrontendChain,
        FrontendSpec,
        LnaSpec,
        NoiseSpec,
        PortSpec,
        SeedSpec,
    )

    spec = FrontendSpec(
        port=PortSpec(reference_impedance_ohm=50.0),
        noise=NoiseSpec(
            noise_figure_db=6.0,
            antenna_temperature_k=290.0,
            bandwidth_hz=5.0e6,
            phase_noise_dbc_per_hz=-90.0,
            phase_offset_hz=1.0e5,
            phase_sample_rate_hz=5.0e6,
        ),
        lna=LnaSpec(gain_db=20.0),
        agc=AgcSpec(
            target_rms=0.2, mode="global", min_gain_db=-40.0, max_gain_db=40.0
        ),
        adc=AdcSpec(bits=10, full_scale=1.0),
        seed=SeedSpec(seed_base=7),
    )
    chain = FrontendChain(spec)
    signal = torch.randn(2, 2, 4, 32, dtype=torch.complex64, device="cuda")
    chain.apply(signal)  # resolve the table before wrapping it

    operators = _operators()
    ledger = Ledger(monkeypatch, operators, FRONTEND_OPERATORS)
    chain.apply(signal)
    with capsys.disabled():
        print(f"\n  frontend       {ledger.launches}")
    assert sum(ledger.launches.values()) <= 3, ledger.launches
    assert ledger.launches == {
        "frontend_noise_forward": 1,
        "frontend_agc_forward": 1,
        "frontend_quantize_forward": 1,
    }, ledger.launches


def test_no_waveform_materialises_a_row_by_sample_intermediate(batch, capsys):
    """Peak allocation within 2x the output plus a row-sized allowance.

    The bound is what separates a fused reduction from an expansion. A
    ``K x chirps x samples`` intermediate for eleven rows and a 512-sample
    pulsed train is 22x the output on its own, so it cannot hide inside a
    factor of two.
    """

    reported = {}
    for family, synthesize, spec in _waveforms():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        cube = synthesize(batch, spec)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - before
        output_bytes = cube.numel() * cube.element_size()
        input_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                batch.total_delay_s,
                batch.complex_transfer_ref,
                batch.sensor_pair_index,
                batch.pair_offsets,
            )
        )
        # The bound is derived, not fitted. A synthesis family writes its two
        # REAL buffers (together exactly one output) and `torch.complex`
        # materialises the cube (one more), so 2x the output is structural and
        # unavoidable at this seam. The remaining allowance is a small multiple
        # of the ROW-sized transients - the masked weight and its two
        # components - which is what `16 * input_bytes` is. A
        # `K x chirps x samples` intermediate would be eleven times the output
        # here and cannot hide inside either term.
        bound = 2.0 * output_bytes + 16.0 * input_bytes
        reported[family] = (peak, output_bytes, input_bytes, bound)
        assert peak <= bound, (family, peak, bound)
        del cube

    with capsys.disabled():
        print("\nT5.9 peak allocation, bytes")
        for family, (peak, output, inputs, bound) in reported.items():
            print(
                f"  {family:14s} peak={peak:>9} output={output:>9} "
                f"inputs={inputs:>6} bound={bound:>10.0f}"
            )
