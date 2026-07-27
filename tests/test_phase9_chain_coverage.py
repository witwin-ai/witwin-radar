"""The four AD chains that were reachable in production and covered nowhere.

Each of these routes exists, is differentiable, and had no gradient test of its
own. They are gathered here rather than scattered because they share the
scenario the rest of Phase 9 uses and because each one is small:

1. **frontend -> loss, behind a real cube.** Frontend AD is tested on a
   standalone random tensor. Nothing ever put the port, the phase noise, the
   thermal noise and the LNA behind a synthesized cube and asked for the
   endpoint gradient. The ADC is off throughout: it refuses a derivative by
   design and its row lives in the wall's file.
2. **slot-batched ``reevaluate_slots`` -> loss.** ``test_phase7_slot_batching.py``
   pins the batched replay's row identity and its host-observation budget. Not
   its derivative.
3. **wideband band -> loss.** S1 drove an endpoint leaf into ``_compose_band``
   and stopped at the published ``[K, F]`` band. Nothing carried it into an OFDM
   cube.
4. **sensor weight -> waveform -> loss.** ``evaluate_sensor_weights`` is
   validated against its own finite difference and its own adjoint. Its
   production consumer is the Dirichlet spectrum synthesis inside
   ``Radar.mimo_from_trace``, and that composition had no AD test.

**Noise reproducibility under AD** closes the file. Noise is off by default in
every physics test in this tree; where it is on, the realisation is a
counter-based Philox draw keyed by ``(seed_base, stage, index)``, so the same
seed must give a BITWISE identical gradient and a different seed must give a
different one. An RNG-stream reordering is invisible to every other test.

**Two exact oracles do most of the work here**, and they are stronger than a
difference. The port and the LNA are a constant multiplicative factor on the
signal, so terminating the chain in ``sum |signal|^2`` scales the endpoint
gradient by exactly ``R * g_lna^2`` - 500.0 for a 50 ohm port and 10 dB of gain,
measured 500.003. And three identical slots of a static scene make the batched
gradient exactly three times the single-frame one.
"""

from __future__ import annotations

import pytest
import torch
import torch.autograd.forward_ad as forward_ad

pytest.importorskip("witwin.channel")

from support import ad_matrix as mx  # noqa: E402
from support import fd  # noqa: E402
from support import multi_endpoint_driver as drv  # noqa: E402
from support import multi_endpoint_geometry as geo  # noqa: E402
from support import waveform_chains as wc  # noqa: E402
from support.synthesis_batch import to_synthesis  # noqa: E402

from witwin.radar.frontend import (  # noqa: E402
    FrontendChain,
    FrontendSpec,
    LnaSpec,
    NoiseSpec,
    PortSpec,
    SeedSpec,
)
from witwin.radar.propagation.channel_consumer import (  # noqa: E402
    ChannelPropagationAdapter,
)
from witwin.radar.synthesis import OfdmCfrSpec, synthesize_ofdm_cfr  # noqa: E402


pytestmark = pytest.mark.gpu


PORT_OHM = 50.0
LNA_GAIN_DB = 10.0

#: Metres, fourth-order stencil. Swept on this chain at 2e-5, 5e-5, 1e-4, 2e-4
#: and 5e-4: 4.1%, 1.6%, 2.2%, 1.6%, 9.0%. The window is shallow because the
#: loss carries the 77 GHz propagation phase and turns a full cycle every
#: 1.95 mm of round-trip path, and the frontend does not change that - the
#: same sweep on the bare cube gives 4.1%, 1.6%, 2.2%, 1.6%, 9.0%, which is the
#: measurement that says the conditioning belongs to the chain rather than to
#: the noise.
STEP_M = 5.0e-5
FD_RTOL = 5.0e-2

#: Wideband, same stencil: 0.38%, 0.33%, 1.8% at 5e-5, 1e-4 and 2e-4.
WIDEBAND_STEP_M = 1.0e-4
WIDEBAND_FD_RTOL = 1.0e-2

SLOT_COUNT = 3

ZERO_FLOOR = 1.0e-30

#: How much of the ungated gradient may survive a global AGC. See
#: ``test_a_global_agc_makes_a_magnitude_loss_exactly_constant``; measured
#: 7.8e-5, and the ungated gradient itself is what a backward that dropped the
#: AGC's own rms term would return.
AGC_RESIDUAL_BOUND = 1.0e-3


@pytest.fixture(scope="module")
def spike():
    return drv.MultiEndpointSpike()


@pytest.fixture(scope="module")
def values(spike):
    return mx.base_values(spike)


def _noise(**overrides):
    fields = dict(
        noise_figure_db=3.0,
        bandwidth_hz=1.0e6,
        phase_noise_dbc_per_hz=-80.0,
        phase_offset_hz=1.0e5,
        phase_sample_rate_hz=1.0e6,
    )
    fields.update(overrides)
    return NoiseSpec(**fields)


def _frontend(*, quiet: bool = False, seed: int = 5) -> FrontendSpec:
    """Port, noise, LNA - and deliberately no AGC and no ADC.

    The ADC refuses a derivative by design. The AGC is left out for a subtler
    reason that ``test_a_global_agc_makes_a_magnitude_loss_exactly_constant``
    states and measures: a global AGC normalises the frame to a fixed RMS, so
    ``sum |signal|^2`` becomes a CONSTANT and every gradient through it is
    correctly zero. That is a real property worth pinning, and a terrible
    fixture for a chain test.
    """

    return FrontendSpec(
        port=PortSpec(PORT_OHM),
        noise=(
            _noise(
                noise_figure_db=0.0,
                antenna_temperature_k=0.0,
                bandwidth_hz=0.0,
                phase_noise_dbc_per_hz=None,
                phase_offset_hz=None,
            )
            if quiet
            else _noise()
        ),
        lna=LnaSpec(gain_db=LNA_GAIN_DB),
        agc=None,
        seed=SeedSpec(seed),
    )


def _cube(spike, values, *, ad_mode="none", kind="fmcw"):
    composed = mx.replay(spike, values, ad_mode=ad_mode)
    return wc.synthesize(kind, composed, wc.make_spec(kind))


def _bare_loss(spike, values, *, ad_mode="none", kind="fmcw"):
    return _cube(spike, values, ad_mode=ad_mode, kind=kind).abs().square().sum()


def _frontend_loss(
    spike, values, *, ad_mode="none", kind="fmcw", frontend=None, seed_base=None
):
    cube = _cube(spike, values, ad_mode=ad_mode, kind=kind)
    chain = FrontendChain(_frontend() if frontend is None else frontend)
    return chain.apply(cube.contiguous(), seed_base=seed_base).signal.abs().square().sum()


def _site_gradient(loss_fn, values, **kwargs):
    live = mx.marked(values, ("sites",))
    loss = loss_fn(live, **kwargs)
    loss.backward()
    return live["sites"].grad.detach().clone(), float(loss.detach())


# --------------------------------------------------------------------------
# 1. frontend -> loss, behind a real cube
# --------------------------------------------------------------------------


def test_the_noiseless_frontend_scales_the_endpoint_gradient_by_exactly_r_g_squared(
    spike, values
):
    """An exact analytic oracle, and it pins the port convention as well.

    With the noise stages off the frontend is ``x -> g_lna * sqrt(R) * x``, a
    real constant. A loss quadratic in the signal therefore scales by the square
    of it: ``R * g_lna^2 = 50 * 10 = 500``. Measured 500.003 per component,
    which is float32 and nothing else.

    This is worth more than a finite difference here: it says the port's
    ``sqrt(W) -> volts`` conversion is applied EXACTLY once, which is the one
    thing the frontend's own docstring warns about and which a magnitude plot
    could never show.
    """

    bare, _ = _site_gradient(
        lambda live: _bare_loss(spike, live, ad_mode="vjp"), values
    )
    through, _ = _site_gradient(
        lambda live: _frontend_loss(
            spike, live, ad_mode="vjp", frontend=_frontend(quiet=True)
        ),
        values,
    )
    assert float(bare.abs().max()) > 0.0
    expected = PORT_OHM * (10.0 ** (LNA_GAIN_DB / 10.0))
    live = bare.abs() > 0.0
    ratio = (through[live] / bare[live]).tolist()
    for value in ratio:
        assert abs(value - expected) < 1.0e-3 * expected, (value, expected)


def test_the_noisy_frontend_gradient_matches_a_fourth_order_difference(
    spike, values
):
    """The whole chain with phase noise, thermal noise and the LNA live.

    The difference is taken with the SAME seed at every sample, so the noise
    realisation is identical across the stencil and cancels out of the
    difference exactly as it cancels out of the derivative. A seed that moved
    per sample would be differencing four different noise draws and would
    measure nothing.
    """

    gradient, _ = _site_gradient(
        lambda live: _frontend_loss(spike, live, ad_mode="vjp"), values
    )
    direction = mx.direction("sites", values["sites"], gradient)
    analytic = float((gradient * direction).sum())
    assert abs(analytic) > 0.0

    samples = {
        offset: float(
            _frontend_loss(
                spike,
                mx.perturbed(values, {"sites": direction}, ("sites",), offset, STEP_M),
            )
        )
        for offset in (-2, -1, 1, 2)
    }
    measured = fd.fourth_order_difference(samples, STEP_M)
    assert fd.relative_error(measured, analytic, floor=ZERO_FLOOR) < FD_RTOL, (
        measured,
        analytic,
    )


def test_the_frontend_forward_tangent_reproduces_the_reverse_gradient(
    spike, values
):
    """Both AD modes through the frontend, on one frozen topology."""

    gradient, _ = _site_gradient(
        lambda live: _frontend_loss(spike, live, ad_mode="vjp"), values
    )
    direction = mx.direction("sites", values["sites"], gradient)
    analytic = float((gradient * direction).sum())

    with forward_ad.dual_level():
        duals = dict(values)
        for name in ("sites", "transmitters", "receivers"):
            seed = direction if name == "sites" else torch.zeros_like(values[name])
            duals[name] = forward_ad.make_dual(values[name].clone(), seed)
        loss = _frontend_loss(spike, duals, ad_mode="jvp")
        tangent = forward_ad.unpack_dual(loss).tangent
        assert tangent is not None
        measured = float(tangent)
    assert fd.relative_error(measured, analytic, floor=ZERO_FLOOR) < 1.0e-4


def test_a_global_agc_makes_a_magnitude_loss_exactly_constant(spike, values):
    """A ZERO cell that is a real property of the stage, and its falsifier.

    A global AGC divides the frame by its own RMS and multiplies by a target, so
    ``sum |signal|^2`` is ``N * target^2`` for ANY input - measured exactly, to
    the last bit of the float32 product. The correct derivative is therefore
    zero, and the AGC backward's ``inner`` term is what makes it so: a backward
    that treated the measured gain as a constant would return the full 500x
    gradient the noiseless chain above produces, which is 1e7 times what is
    measured here.
    """

    from witwin.radar.frontend import AgcSpec

    target_rms = 1.0e-3
    spec = FrontendSpec(
        port=PortSpec(PORT_OHM),
        noise=_noise(),
        lna=LnaSpec(gain_db=LNA_GAIN_DB),
        agc=AgcSpec(target_rms=target_rms, mode="global"),
        seed=SeedSpec(5),
    )
    gated, loss = _site_gradient(
        lambda live: _frontend_loss(spike, live, ad_mode="vjp", frontend=spec),
        values,
    )
    ungated, _ = _site_gradient(
        lambda live: _frontend_loss(spike, live, ad_mode="vjp"), values
    )
    elements = _cube(spike, values).numel()
    assert loss == pytest.approx(elements * target_rms * target_rms, rel=1e-6)
    # Measured 7.8e-5 of the ungated gradient. It is not exactly zero and could
    # not be: the derivative that cancels is a sum of 2048 terms each about 1e5
    # times the residual, so what survives is float32 cancellation of a sum
    # whose true value is zero, not a derivative the backward failed to remove.
    assert float(gated.abs().sum()) < AGC_RESIDUAL_BOUND * float(
        ungated.abs().sum()
    )


# --------------------------------------------------------------------------
# 2. The noise realisation is reproducible under AD
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", mx.WAVEFORMS)
def test_the_same_seed_replays_a_bitwise_identical_gradient(spike, values, kind):
    """Same seed, same gradient, to the last bit; a different seed, a different one.

    The realisation is a counter-based Philox draw keyed by
    ``(seed_base, stage, linear index)``, so this is a statement about the DRAW
    ORDER as much as about the seed: a refactor that fused two draws into one,
    or that reordered the real and imaginary components, would reproduce every
    distributional test in the tree and fail here.
    """

    first, _ = _site_gradient(
        lambda live: _frontend_loss(spike, live, ad_mode="vjp", kind=kind, seed_base=17),
        values,
    )
    replayed, _ = _site_gradient(
        lambda live: _frontend_loss(spike, live, ad_mode="vjp", kind=kind, seed_base=17),
        values,
    )
    other, _ = _site_gradient(
        lambda live: _frontend_loss(spike, live, ad_mode="vjp", kind=kind, seed_base=18),
        values,
    )
    assert float(first.abs().sum()) > 0.0
    assert torch.equal(first, replayed)
    assert not torch.equal(first, other)


def test_the_physics_chain_itself_has_no_noise_to_reproduce(spike, values):
    """Noise is OFF by default, which is why the tests above have to ask for it.

    ``RadarConfig`` carries no ``FrontendSpec`` unless a caller builds one, so
    every other Phase-9 chain is deterministic by construction rather than by a
    seed. Asserted directly so the default cannot drift.
    """

    first = float(_bare_loss(spike, values))
    second = float(_bare_loss(spike, values))
    assert first == second
    assert drv.make_spec(num_chirps=2).__class__.__name__ == "FmcwBeatSpec"
    from witwin.radar import RadarConfig

    config = RadarConfig.from_dict(dict(geo.FIXTURE_RADAR_CONFIG))
    assert getattr(config, "frontend", None) is None


# --------------------------------------------------------------------------
# 3. Slot-batched replay -> loss
# --------------------------------------------------------------------------


def _slot_loss(spike, values, *, ad_mode="none", kind="fmcw"):
    stacked = spike.stacked(values["sites"], SLOT_COUNT)
    inbound, outbound = spike.slot_legs(
        stacked, slot_count=SLOT_COUNT, ad_mode=ad_mode
    )
    spec = wc.make_spec(kind)
    frames = spike.slot_frames(
        inbound, outbound, mx.response_of(values), include_delay_rate=False
    )
    return sum(
        wc.synthesize(kind, frame, spec).abs().square().sum() for frame in frames
    )


def test_a_slot_batched_replay_carries_the_single_frame_gradient_exactly(
    spike, values
):
    """Three identical slots of a static scene, and an exact factor of three.

    ``reevaluate_slots`` is ONE consumer call for the whole frame - one launch
    per bucket, one validation copy, one synchronisation - against three
    separate replays. Its row identity and its host-observation budget are
    pinned by ``test_phase7_slot_batching.py``; its DERIVATIVE was not pinned
    anywhere. With a static scene the three slots are the same frame, so the
    batched gradient must be exactly three times the single-frame one, and a
    batched backward that lost a slot or reduced across slot boundaries could
    not produce that.
    """

    batched, _ = _site_gradient(
        lambda live: _slot_loss(spike, live, ad_mode="vjp"), values
    )
    single, _ = _site_gradient(
        lambda live: _bare_loss(spike, live, ad_mode="vjp"), values
    )
    assert float(single.abs().max()) > 0.0
    live = single.abs() > 0.0
    ratio = (batched[live] / single[live]).tolist()
    for value in ratio:
        assert value == pytest.approx(float(SLOT_COUNT), rel=1.0e-5), ratio


def test_the_slot_batched_gradient_matches_a_fourth_order_difference(spike, values):
    """And the factor of three is not the only thing that is right about it."""

    gradient, _ = _site_gradient(
        lambda live: _slot_loss(spike, live, ad_mode="vjp"), values
    )
    direction = mx.direction("sites", values["sites"], gradient)
    analytic = float((gradient * direction).sum())
    samples = {
        offset: float(
            _slot_loss(
                spike,
                mx.perturbed(values, {"sites": direction}, ("sites",), offset, STEP_M),
            )
        )
        for offset in (-2, -1, 1, 2)
    }
    measured = fd.fourth_order_difference(samples, STEP_M)
    assert fd.relative_error(measured, analytic, floor=ZERO_FLOOR) < FD_RTOL, (
        measured,
        analytic,
    )


# --------------------------------------------------------------------------
# 4. The wideband band, all the way to a cube
# --------------------------------------------------------------------------


WIDEBAND_SPEC = OfdmCfrSpec(
    num_subcarriers=8,
    num_symbols=2,
    subcarrier_spacing_hz=25.0e6,
    cyclic_prefix_s=1.0e-6,
    reference_frequency_hz=geo.REFERENCE_FREQUENCY_HZ,
    max_expected_delay_s=5.0e-7,
    carrier_hz=0.0,
    carrier_rate_hz=geo.REFERENCE_FREQUENCY_HZ,
)


@pytest.fixture(scope="module")
def banded(spike):
    """A banded spike over the SAME compiled scene as the narrowband one.

    Sharing the compile removes the question of whether any difference between
    the two is a band effect or a compile effect.
    """

    adapter = ChannelPropagationAdapter(
        spike.compiled,
        reference_frequency_hz=geo.REFERENCE_FREQUENCY_HZ,
        components=drv.MULTIPATH_COMPONENTS,
        max_depth=1,
        frequency_offsets_hz=WIDEBAND_SPEC.frequency_offsets_hz,
    )
    return drv.MultiEndpointSpike(compiled=spike.compiled, adapter=adapter)


def _wideband_loss(banded, values, *, ad_mode="none"):
    composed = mx.replay(banded, values, ad_mode=ad_mode)
    assert composed.frequency_response is not None, "this spike declared a band"
    cube = synthesize_ofdm_cfr(to_synthesis(composed), WIDEBAND_SPEC)
    return cube.abs().square().sum()


def test_a_wideband_endpoint_gradient_reaches_a_synthesized_cube(banded, values):
    """S1 stopped at the published band; this carries it into the cube.

    The per-subcarrier route consumes a different column of the band in every
    output bin, so a chain that reached the band correctly and then indexed it
    wrongly would pass every S1 test and fail here.
    """

    gradient, _ = _site_gradient(
        lambda live: _wideband_loss(banded, live, ad_mode="vjp"), values
    )
    direction = mx.direction("sites", values["sites"], gradient)
    analytic = float((gradient * direction).sum())
    assert abs(analytic) > 0.0

    samples = {
        offset: float(
            _wideband_loss(
                banded,
                mx.perturbed(
                    values, {"sites": direction}, ("sites",), offset, WIDEBAND_STEP_M
                ),
            )
        )
        for offset in (-2, -1, 1, 2)
    }
    measured = fd.fourth_order_difference(samples, WIDEBAND_STEP_M)
    assert (
        fd.relative_error(measured, analytic, floor=ZERO_FLOOR) < WIDEBAND_FD_RTOL
    ), (measured, analytic)


def test_the_wideband_cube_is_not_the_narrowband_one(spike, banded, values):
    """The falsifier: without it the test above could be a narrowband test.

    A band whose columns were all the reference column would reproduce every
    magnitude the narrowband route produces and would still pass a difference
    against itself.
    """

    wide = float(_wideband_loss(banded, values))
    narrow_composed = mx.replay(spike, values)
    narrow = float(
        synthesize_ofdm_cfr(to_synthesis(narrow_composed), WIDEBAND_SPEC)
        .abs()
        .square()
        .sum()
    )
    assert narrow != wide
    assert abs(wide - narrow) > 1.0e-6 * abs(wide)


# --------------------------------------------------------------------------
# 5. Sensor weight -> waveform -> loss
# --------------------------------------------------------------------------


#: The 1 TX x 1 RX legacy configuration ``tests/solvers/test_mimo_cross.py``
#: uses for the same entry point, spelled out rather than imported: a test
#: module is not a fixture owner, and importing one from another is how two
#: files end up sharing a constant neither of them owns.
MIMO_CONFIG = {
    "num_tx": 1,
    "num_rx": 1,
    "fc": 77e9,
    "slope": 60.012,
    "adc_samples": 256,
    "adc_start_time": 6,
    "sample_rate": 4400,
    "idle_time": 7,
    "ramp_end_time": 65,
    "chirp_per_frame": 4,
    "frame_per_second": 10,
    "num_doppler_bins": 4,
    "num_range_bins": 256,
    "num_angle_bins": 64,
    "power": 15,
    "tx_loc": [[0, 0, 0]],
    "rx_loc": [[0, 0, 0]],
}


def _mimo_config():
    from witwin.radar import RadarConfig

    return RadarConfig.from_dict(dict(MIMO_CONFIG))


def _mimo_loss(points: torch.Tensor) -> torch.Tensor:
    """``evaluate_sensor_weights -> Dirichlet spectrum -> cube -> scalar``.

    ``Radar.mimo_from_trace`` is the production composition: one sensor-weight
    launch produces the delay, the rate and the complex weight, and the
    Dirichlet spectrum synthesis consumes all three. Building the chain by hand
    here would test a composition nothing runs.
    """

    from witwin.radar import Radar, TraceResult

    radar = Radar(_mimo_config())
    intensities = torch.tensor([0.9], dtype=torch.float32, device="cuda")
    frame = radar.mimo_from_trace(TraceResult(points, intensities), t0=0.0)
    return frame.abs().square().sum()


#: Metres. The Dirichlet route has no 77 GHz reference phase in the loss - the
#: legacy carrier lives in the spectrum - so the loss is far smoother in the
#: target position than the Channel chain is, and a 1e-3 m step is inside the
#: window rather than several cycles past it. Measured relative agreement at
#: 1e-4, 5e-4, 1e-3 and 5e-3 is reported in the stage record.
MIMO_STEP_M = 1.0e-3
MIMO_FD_RTOL = 2.0e-2


def test_a_sensor_weight_gradient_reaches_a_synthesized_dirichlet_cube():
    """The sensor weight's own AD, composed with a waveform for the first time.

    ``test_phase6_sensor_weight.py`` validates the weight kernel's jvp against a
    central difference and its vjp against that jvp. Neither says anything about
    what happens when the weight is multiplied into a spectrum and summed: the
    weight is complex, the spectrum conjugates nothing, and a chain that dropped
    the imaginary half would still pass every operator-level test.
    """

    base = torch.tensor(
        [[0.0, 0.0, -3.0]], dtype=torch.float32, device="cuda"
    )
    live = base.clone().requires_grad_(True)
    loss = _mimo_loss(live)
    loss.backward()
    gradient = live.grad.detach()
    assert gradient is not None
    assert float(gradient.abs().sum()) > 0.0

    # Radial: the target sits on the boresight axis, so the z component is the
    # range derivative and the transverse ones are structurally small. The
    # difference is taken along z for that reason.
    direction = torch.tensor(
        [[0.0, 0.0, 1.0]], dtype=torch.float32, device="cuda"
    )
    analytic = float((gradient * direction).sum())
    assert abs(analytic) > 0.0
    samples = {
        offset: float(_mimo_loss(base + (offset * MIMO_STEP_M) * direction))
        for offset in (-2, -1, 1, 2)
    }
    measured = fd.fourth_order_difference(samples, MIMO_STEP_M)
    assert (
        fd.relative_error(measured, analytic, floor=ZERO_FLOOR) < MIMO_FD_RTOL
    ), (measured, analytic)
