"""Per-frame launch and memory ledger for every Phase-6 owner (criterion A7).

The three per-waveform test files each count their own family's launches. This
file is the ledger ACROSS them: one operator table, wrapped once, with every
Phase-6 symbol in it, so a stage that quietly adds a launch to one owner has to
change a number here rather than a number in the file it was already editing.

The budgets are the design's section 4.3:

    fmcw_spectrum_forward   1 per default FMCW frame
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
from support.dsp_ledger import DspLedger
from support.multi_endpoint_specs import fmcw_spec, ofdm_spec, pulsed_spec

pytestmark = pytest.mark.gpu

#: Every synthesis and sensor symbol, by family. The frontend is
#: counted separately because its budget is a total rather than a per-symbol
#: count.
SYNTHESIS_OPERATORS = (
    "fmcw_beat_forward",
    "fmcw_beat_backward",
    "fmcw_beat_jvp",
    "fmcw_spectrum_forward",
    "fmcw_spectrum_backward",
    "fmcw_spectrum_jvp",
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

FRONTEND_OPERATORS = ("frontend_noise_forward", "frontend_agc_forward", "frontend_quantize_forward")


@pytest.fixture(scope="module")
def spike():
    pytest.importorskip("witwin.channel")
    return drv.MultiEndpointSpike()


@pytest.fixture(scope="module")
def batch(spike):
    composed, _, _ = spike.frame()
    return drv.to_synthesis(composed)


def _operators():
    from witwin.radar.cuda import runtime

    return runtime.build_extension()


def _waveforms():
    """The three synthesis entry points, each with its own fixture spec."""

    from witwin.radar.synthesis.fmcw import synthesize_fmcw
    from witwin.radar.synthesis.ofdm import synthesize_ofdm
    from witwin.radar.synthesis.pulsed import synthesize_pulsed

    return (
        ("fmcw_beat", synthesize_fmcw, fmcw_spec(4)),
        ("ofdm_cfr", synthesize_ofdm, ofdm_spec(4)),
        ("pulsed_echo", synthesize_pulsed, pulsed_spec(4)),
    )


def test_each_waveform_costs_exactly_one_forward_launch_per_frame(batch, monkeypatch):
    """One launch each, and nothing from the other two families.

    The cross-family assertion is the one a per-family test cannot make: a
    waveform owner that reached into another owner's kernel would still count
    one launch of its own.
    """

    operators = _operators()
    _waveforms()  # resolve imports before the table is wrapped
    for family, synthesize, spec in _waveforms():
        ledger = DspLedger(monkeypatch, SYNTHESIS_OPERATORS, operators)
        synthesize(batch, spec)
        assert ledger.launches[f"{family}_forward"] == 1, ledger.launches
        assert ledger.transform_count == 1, ledger.launches
        assert ledger.host_observation_count == 0, ledger.host
        monkeypatch.undo()


def test_the_launch_count_is_flat_in_slot_count(spike, monkeypatch):
    """Slots are free at the launch ledger (plan item 3).

    Two counts, both independent of the slot count: the propagation replay is
    ONE consumer call per leg for the whole frame, and the waveform is ONE
    forward launch. A Python per-slot loop would multiply the first by the slot
    count; a per-slot synthesis would multiply the second.
    """

    from witwin.channel.propagation import consumer

    from witwin.radar.synthesis.fmcw import synthesize_fmcw

    operators = _operators()
    spec = fmcw_spec(4)
    for slots in (1, 8, 64):
        times = [index * 1.0e-5 for index in range(slots)]
        stack = drv.slot_site_stack(spike.site_tensor(), (0.0, 1.0, 0.0), times)
        spike.slot_legs(stack, slot_count=slots)  # warm the replication cache

        replays = {"count": 0}
        original = consumer.reevaluate

        def counting(*args, _original=original, _replays=replays, **kwargs):
            _replays["count"] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(consumer, "reevaluate", counting)
        ledger = DspLedger(monkeypatch, SYNTHESIS_OPERATORS, operators)
        inbound, outbound = spike.slot_legs(stack, slot_count=slots)
        composed = spike.composer.compose(inbound.slot(0), outbound.slot(0), drv.make_response())
        synthesize_fmcw(drv.to_synthesis(composed), spec)
        monkeypatch.undo()

        assert replays["count"] == 2, (slots, replays)
        assert ledger.launches["fmcw_beat_forward"] == 1, (slots, ledger.launches)
        assert ledger.transform_count == 1, (slots, ledger.launches)


def test_one_backward_launch_per_forward_launch(batch, monkeypatch):
    """R-ADR-004's shape, measured: one companion launch, not two, not zero."""

    operators = _operators()
    _waveforms()
    from dataclasses import replace

    for family, synthesize, spec in _waveforms():
        transfer = batch.complex_transfer_ref.detach().clone().requires_grad_(True)
        live = replace(batch, complex_transfer_ref=transfer)
        ledger = DspLedger(monkeypatch, SYNTHESIS_OPERATORS, operators)
        cube = synthesize(live, spec)
        (cube.real.sum() + cube.imag.sum()).backward()
        assert ledger.launches[f"{family}_forward"] == 1, ledger.launches
        assert ledger.launches[f"{family}_backward"] == 1, ledger.launches
        assert ledger.launches[f"{family}_jvp"] == 0, ledger.launches
        monkeypatch.undo()


def test_the_sensor_weight_owner_costs_one_launch_per_frame(monkeypatch):
    """One ``sensor_weight_forward`` per frame, on the scene-driven route.

    ``sensors.py`` applies the pattern inside ``Radar.simulate``, and ONE launch
    covers the whole frame's composed rows however many pairs and sites they
    span.

    This route also synthesizes a waveform, so ``fmcw_spectrum_forward`` is
    asserted at the same count rather than required to be zero - the claim is
    that the pattern stage adds exactly one launch on top of the frame the
    entry already pays for. The run is TWO frames, which is what makes this a
    per-frame budget: a stage that rebuilt its row tables inside the frame loop
    would still launch once per frame, but one that re-applied the pattern per
    site or per pair would not, and a single frame cannot tell those apart from
    the total.

    The pattern is the isotropic table so that this is a launch count rather
    than a physics change; the stage runs either way.
    """

    from support import multi_endpoint_geometry as geo
    from support import multi_endpoint_world as world

    from witwin.radar import Pattern, PointTargets, Radar

    radar = Radar.from_dict(
        dict(geo.FIXTURE_RADAR_CONFIG), pattern=Pattern.isotropic(), position=(0.0, 0.0, 0.0), look_at=(1.0, 0.0, 0.0)
    )
    scene, mesh = world.make_scene()
    world.assert_world_coordinates_survived(mesh)
    targets = PointTargets(
        positions=torch.tensor(
            (geo.SITE_P_POSITION_M, geo.SITE_Q_POSITION_M), dtype=torch.float32, device=radar.device
        ),
        amplitude=drv.FIXTURE_AMPLITUDE,
        phase=drv.FIXTURE_PHASE_RAD,
    )

    def simulate(times):
        return radar.simulate(scene, targets, times=times)

    simulate((0.0,))  # resolve every lazy import and table before wrapping

    frames = 2
    operators = _operators()
    ledger = DspLedger(monkeypatch, SYNTHESIS_OPERATORS, operators)
    simulate(tuple(index * 1.0e-3 for index in range(frames)))
    assert ledger.launches["sensor_weight_forward"] == frames, ledger.launches
    assert ledger.launches["sensor_weight_backward"] == 0, ledger.launches
    assert ledger.launches["sensor_weight_jvp"] == 0, ledger.launches
    assert ledger.launches["fmcw_spectrum_forward"] == frames, ledger.launches
    for name in SYNTHESIS_OPERATORS:
        if not name.startswith("sensor_weight") and name != "fmcw_spectrum_forward":
            assert ledger.launches[name] == 0, (name, ledger.launches)


def test_the_frontend_costs_at_most_three_launches_per_frame(monkeypatch):
    """Noise, AGC, ADC: three operator calls for the whole six-stage chain.

    Port conversion and the LNA are fused into the noise operator on purpose -
    that fusion is what makes thermal noise input-referred by construction - so
    a six-stage chain is three launches and not six.
    """

    from witwin.radar import Adc, Agc, Noise
    from witwin.radar.frontend import FrontendChain, FrontendSpec

    spec = FrontendSpec(
        impedance=50.0,
        noise=Noise(
            figure=6.0,
            antenna_temperature=290.0,
            bandwidth=5.0e6,
            phase_density=-90.0,
            phase_offset=1.0e5,
            phase_sample_rate=5.0e6,
        ),
        lna=20.0,
        agc=Agc(target_rms=0.2, mode="global", min_gain=-40.0, max_gain=40.0),
        adc=Adc(bits=10, full_scale=1.0),
        seed=7,
    )
    chain = FrontendChain(spec)
    signal = torch.randn(2, 2, 4, 32, dtype=torch.complex64, device="cuda")
    chain.apply(signal)  # resolve the table before wrapping it

    operators = _operators()
    ledger = DspLedger(monkeypatch, FRONTEND_OPERATORS, operators)
    chain.apply(signal)
    assert ledger.transform_count <= 3, ledger.launches
    assert ledger.launches == {
        "frontend_noise_forward": 1,
        "frontend_agc_forward": 1,
        "frontend_quantize_forward": 1,
    }, ledger.launches


def test_no_waveform_materialises_a_row_by_sample_intermediate(batch):
    """Peak allocation within 2x the output plus a row-sized allowance.

    The bound is what separates a fused reduction from an expansion. A
    ``K x chirps x samples`` intermediate for eleven rows and a 512-sample
    pulsed train is 22x the output on its own, so it cannot hide inside a
    factor of two.
    """

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
            for tensor in (batch.total_delay_s, batch.complex_transfer_ref, batch.sensor_pair_index, batch.pair_offsets)
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
        assert peak <= bound, (
            family,
            f"peak {peak} B, output {output_bytes} B, inputs {input_bytes} B, bound {bound:.0f} B",
        )
        del cube
