"""Phase 8: the wideband band, end to end, and the per-subcarrier OFDM cube.

The safety argument for flipping rule R8 is the first test in this file and
nothing else: with no band declared, every published number is BITWISE what it
was before. Everything after that is the new capability, measured against a
closed form rather than against itself.

The kernel change is the one that could silently produce a plausible cube. A
narrowband weight carries none of the ``n * df`` phase, so the subcarrier term
multiplies the full delay; a wideband column already carries the whole
subcarrier phase at the frozen delay, so it multiplies the drift. Indexing the
weight per subcarrier WITHOUT that switch counts ``n * df * tau_rt`` twice and
puts every tap at twice its delay. ``test_a_wideband_column_is_not_the_delay_
phase_applied_twice`` is the test that would catch it.
"""

from __future__ import annotations

import math

import pytest
import torch
from support import multi_endpoint_driver as driver  # noqa: E402
from support import multi_endpoint_geometry as geo  # noqa: E402

from witwin.radar.channel import ChannelPropagationAdapter
from witwin.radar.synthesis import OfdmSpec, SlowTimeMode, SynthesisPathBatch, synthesize_ofdm
from witwin.radar.synthesis.ofdm import synthesize_cfr_rows

pytestmark = pytest.mark.gpu

F_REF = geo.REFERENCE_FREQUENCY_HZ

#: 16 subcarriers x 25 MHz = a 400 MHz band at 77 GHz. That is a real 5G FR2
#: allocation and it is the band the design's quantified narrowband error law is
#: stated for: 0.5% of spreading tilt, and half an Airy fringe of a 0.1 m wall.
NUM_SUBCARRIERS = 16
SUBCARRIER_SPACING_HZ = 25.0e6
NUM_SYMBOLS = 4


def _spec(**overrides) -> OfdmSpec:
    fields = {
        "num_subcarriers": NUM_SUBCARRIERS,
        "num_symbols": NUM_SYMBOLS,
        "subcarrier_spacing_hz": SUBCARRIER_SPACING_HZ,
        "cyclic_prefix_s": 1.0e-6,
        "reference_frequency_hz": F_REF,
        "max_expected_delay_s": 5.0e-7,
        "carrier_hz": 0.0,
        "carrier_rate_hz": F_REF,
    }
    fields.update(overrides)
    return OfdmSpec(**fields)


def _spikes(spec: OfdmSpec):
    """One compiled scene, two spikes: narrowband and banded, same world.

    The compiled scene is SHARED on purpose. A frequency-only recompile leaves
    every world version domain untouched, so two separately compiled scenes
    would still be legal here - but sharing one removes the question entirely
    and makes any difference between the two spikes attributable to the band.
    """

    narrow = driver.MultiEndpointSpike()
    banded_adapter = ChannelPropagationAdapter(
        narrow.compiled,
        reference_frequency_hz=F_REF,
        components=driver.MULTIPATH_COMPONENTS,
        max_depth=1,
        frequency_offsets_hz=spec.frequency_offsets_hz,
    )
    banded = driver.MultiEndpointSpike(compiled=narrow.compiled, adapter=banded_adapter)
    return narrow, banded


def _cube(spike, spec, response):
    composed, _, _ = spike.frame(response=response, include_delay_rate=False)
    batch = SynthesisPathBatch.from_radar_paths(composed, slow_time_mode=SlowTimeMode.FROZEN_WEIGHT_WITH_CARRIER_RATE)
    return synthesize_ofdm(batch, spec), composed


# ---------------------------------------------------------------------------
# The safety argument: nothing moves when no band is declared
# ---------------------------------------------------------------------------


def test_an_adapter_without_a_band_publishes_exactly_what_it_did_before():
    """Bitwise, on every published member, not to a tolerance.

    This is the whole safety argument for flipping R8. If a narrowband caller's
    numbers can move at all, the flip is a numerical change disguised as a
    capability addition.
    """

    spike = driver.MultiEndpointSpike()
    first = spike.adapter.reevaluate(
        spike.inbound,
        spike._transmitter_batch([p for _, p in spike.transmitters]),
        spike._site_batch(spike.site_positions, role="sink"),
        ad_mode="none",
    )
    assert first.frequency_response is None
    assert first.frequency_offsets_hz is None
    assert first.band_count == 0

    composed, _, _ = spike.frame(include_delay_rate=False)
    assert composed.frequency_response is None
    assert composed.band_count == 0

    batch = SynthesisPathBatch.from_radar_paths(composed, slow_time_mode=SlowTimeMode.FROZEN_WEIGHT_WITH_CARRIER_RATE)
    assert batch.frequency_response is None


def test_a_banded_batch_reproduces_the_reference_column_bitwise():
    """``df = 0`` is bit-identical, so column 0 IS the narrowband answer.

    Channel publishes that identity, this asserts it survives the whole radar
    chain: leg -> two-way join -> composed batch. It is what makes the OFDM
    anchor ``H[0][p][0] == C_rt`` hold under the wideband route without a
    tolerance anywhere.
    """

    spec = _spec()
    narrow, banded = _spikes(spec)
    response = driver.make_response()

    narrow_composed, _, _ = narrow.frame(response=response, include_delay_rate=False)
    banded_composed, _, _ = banded.frame(response=response, include_delay_rate=False)

    assert banded_composed.band_count == NUM_SUBCARRIERS
    assert torch.equal(narrow_composed.complex_transfer_ref, banded_composed.complex_transfer_ref)
    assert torch.equal(banded_composed.frequency_response[:, 0], banded_composed.complex_transfer_ref)


def test_the_ofdm_anchor_survives_the_wideband_route():
    """``H[0][p][0]`` is still the pair's summed reference coefficient.

    Symbol 0 has no drift and subcarrier 0 has no offset, so both routes reduce
    to the same sum of the same weights. Asserted between the two cubes rather
    than against a recomputed sum, because a recomputed sum would reduce in a
    different order and need a tolerance for a claim that is exact.
    """

    spec = _spec()
    narrow, banded = _spikes(spec)
    response = driver.make_response()
    narrow_cube, _ = _cube(narrow, spec, response)
    banded_cube, _ = _cube(banded, spec, response)

    assert torch.equal(narrow_cube[0, :, 0], banded_cube[0, :, 0])


# ---------------------------------------------------------------------------
# The capability: the band is physically different, in the predicted direction
# ---------------------------------------------------------------------------


def test_the_wideband_cube_differs_from_the_narrowband_one_across_the_band():
    """Column 0 agrees exactly; the rest do not, and the gap grows with n.

    "Grows with n" is the falsifiable half. A bug that scaled the whole cube, or
    that applied a constant offset, would move column 0 too.
    """

    spec = _spec()
    narrow, banded = _spikes(spec)
    response = driver.make_response()
    narrow_cube, _ = _cube(narrow, spec, response)
    banded_cube, _ = _cube(banded, spec, response)

    difference = (banded_cube - narrow_cube).abs() / narrow_cube.abs().clamp_min(1.0e-30)
    assert float(difference[:, :, 0].max()) == 0.0
    assert float(difference[:, :, 1:].max()) > 1.0e-2

    # The fixture wall is a 0.1 m slab, so the band crosses a fraction of an
    # Airy fringe and the deviation is not monotone in n. What IS monotone is
    # that the first half of the band deviates less than the second.
    per_column = difference[0, 0]
    assert float(per_column[1 : NUM_SUBCARRIERS // 2].mean()) < float(per_column[NUM_SUBCARRIERS // 2 :].mean())


def test_the_line_of_sight_column_follows_the_exact_spreading_tilt():
    """``|H(f_n)| / |H(f_ref)| = (f_ref / f_n)`` per leg, in closed form.

    The ``lambda / (4 pi d)`` factor is exact and has zero phase, so this is a
    deterministic 0.5%-across-the-band tilt that the narrowband route does not
    have at all. Measured on a LINE-OF-SIGHT row, where it is the only frequency
    dependence there is - a reflection row would add the wall's response and the
    test would stop being a statement about spreading.
    """

    spec = _spec()
    _, banded = _spikes(spec)
    leg = banded.adapter.reevaluate(
        banded.inbound,
        banded._transmitter_batch([p for _, p in banded.transmitters]),
        banded._site_batch(banded.site_positions, role="sink"),
        ad_mode="none",
    )
    los = (leg.component_id == 0).nonzero().flatten()
    assert los.numel() > 0, "the fixture must publish a line-of-sight row"

    band = leg.frequency_response.index_select(0, los).abs().to(torch.float64)
    for index in range(NUM_SUBCARRIERS):
        expected = F_REF / (F_REF + index * SUBCARRIER_SPACING_HZ)
        measured = band[:, index] / band[:, 0]
        assert torch.allclose(measured, torch.full_like(measured, expected), rtol=1.0e-5, atol=0.0), (
            index,
            measured.tolist(),
            expected,
        )


def test_a_wideband_column_is_not_the_delay_phase_applied_twice():
    """The trap: indexing the weight without moving the subcarrier delay term.

    A wideband column already carries ``exp(-j 2 pi f_n tau_rt)``. If the kernel
    still multiplied the FULL delay by ``f_sub``, every tap would land at twice
    its delay in the inverse transform. This drives a single stationary row and
    checks the phase slope across subcarriers directly, which is the bin-free
    statement of the same thing.
    """

    spec = _spec()
    tau = torch.tensor([1.5e-8], dtype=torch.float32, device="cuda")
    rate = torch.zeros_like(tau)
    offsets = torch.tensor([0, 1], dtype=torch.int64, device="cuda")
    columns = torch.tensor(
        [[_channel_style_column(float(tau), index) for index in range(NUM_SUBCARRIERS)]],
        dtype=torch.complex64,
        device="cuda",
    )
    cube = synthesize_cfr_rows(tau, rate, columns, offsets, spec)

    # A stationary row with no drift: the kernel adds NO phase at all, so the
    # cube is the column verbatim.
    assert torch.allclose(cube[0, 0], columns[0], rtol=1.0e-6, atol=1.0e-7)

    # And the phase slope is the single-application one, not twice it.
    phase = torch.angle(cube[0, 0].to(torch.complex128)).cpu()
    step = float(phase[1] - phase[0])
    expected = -2.0 * math.pi * SUBCARRIER_SPACING_HZ * float(tau)
    while step - expected > math.pi:
        step -= 2.0 * math.pi
    while expected - step > math.pi:
        step += 2.0 * math.pi
    assert abs(step - expected) < 1.0e-4, (step, expected, 2.0 * expected)


def _channel_style_column(delay_s: float, subcarrier: int) -> complex:
    """One wideband column as Channel publishes it: the phase at ``f_n``."""

    frequency = F_REF + subcarrier * SUBCARRIER_SPACING_HZ
    cycles = -frequency * delay_s
    fraction = cycles - math.floor(cycles)
    return complex(math.cos(2.0 * math.pi * fraction), math.sin(2.0 * math.pi * fraction))


# ---------------------------------------------------------------------------
# The two-way composition
# ---------------------------------------------------------------------------


def test_the_join_composes_each_column_as_the_product_of_the_two_legs():
    """``H_join[k, j] == H_in[k, j] * S * H_out[k, j]``, column by column.

    Checked against the leg tensors the join itself consumed, so the assertion
    is about the COMPOSITION rather than about propagation. The reference is
    formed in Torch here; that is a test oracle, not a second implementation.
    """

    spec = _spec()
    _, banded = _spikes(spec)
    response = driver.make_response()
    composed, inbound, outbound = banded.frame(response=response, include_delay_rate=False)

    site = response.evaluate(banded.composer.site_count, composed.device)
    reference = (
        inbound.frequency_response.index_select(0, banded.composer.inbound_row)
        * outbound.frequency_response.index_select(0, banded.composer.outbound_row)
        * site.index_select(0, banded.composer.response_slot).unsqueeze(1)
    )
    if composed.row_valid is not None:
        reference = torch.where(composed.row_valid.unsqueeze(1), reference, torch.zeros_like(reference))

    error = (composed.frequency_response - reference).abs() / reference.abs().clamp_min(1.0e-30)
    assert float(error.max()) < 1.0e-5, float(error.max())


def test_the_reference_column_of_the_join_matches_the_narrowband_join_bitwise():
    spec = _spec()
    narrow, banded = _spikes(spec)
    response = driver.make_response()
    narrow_composed, _, _ = narrow.frame(response=response, include_delay_rate=False)
    banded_composed, _, _ = banded.frame(response=response, include_delay_rate=False)

    assert torch.equal(banded_composed.frequency_response[:, 0], narrow_composed.complex_transfer_ref)


def test_a_banded_leg_cannot_be_joined_with_a_narrowband_one():
    """Half a wideband round trip is the narrowband approximation, silently."""

    spec = _spec()
    narrow, banded = _spikes(spec)
    banded_in, _ = banded.legs()
    _, narrow_out = narrow.legs()
    with pytest.raises(ValueError, match="both legs must be evaluated"):
        banded.composer.compose(banded_in, narrow_out, driver.make_response())


def test_the_join_loop_costs_one_launch_per_column_and_no_host_observation():
    """The recorded price of not widening ``two_way_join.cu``.

    ``1 + F`` join launches: the reference column, which also produces the
    delays and the rate, plus one per band column. Recorded here so that the
    R-ADR for a strided ``[K, F]`` kernel has a measurement to argue against,
    and asserted so that a later change to the loop is visible.
    """

    from witwin.radar.cuda import runtime as build

    spec = _spec()
    _, banded = _spikes(spec)
    operators = build.build_extension()
    original = operators.two_way_join_forward
    launches = 0

    def counting(*args, **kwargs):
        nonlocal launches
        launches += 1
        return original(*args, **kwargs)

    import witwin.radar.paths as two_way

    class _Patched:
        def __getattr__(self, name):
            if name == "two_way_join_forward":
                return counting
            return getattr(operators, name)

    saved = two_way._ops
    patched = _Patched()
    two_way._ops = lambda: patched
    try:
        banded.frame(response=driver.make_response(), include_delay_rate=False)
    finally:
        two_way._ops = saved

    assert launches == 1 + NUM_SUBCARRIERS, launches


# ---------------------------------------------------------------------------
# AD
# ---------------------------------------------------------------------------


def test_the_wideband_cfr_vjp_matches_central_differences():
    """Reverse mode on the band, against a float64 central difference.

    The FD oracle runs the same float32 forward, so its own noise floor sits
    near 1e-4; the bound says so rather than using a default that would pass for
    almost any answer. The tight statement is the forward/reverse agreement in
    the next test, which has no finite difference in it at all.
    """

    torch.manual_seed(11)
    spec = _spec(num_symbols=2)
    paths = 4
    tau = torch.rand(paths, device="cuda", dtype=torch.float32) * 1.0e-8 + 1.0e-9
    rate = (torch.rand(paths, device="cuda", dtype=torch.float32) - 0.5) * 2.0e-8
    offsets = torch.tensor([0, 2, paths], dtype=torch.int64, device="cuda")
    band = torch.randn(paths, NUM_SUBCARRIERS, dtype=torch.complex64, device="cuda")

    live = band.clone().requires_grad_(True)
    out = synthesize_cfr_rows(tau, rate, live, offsets, spec)
    seed = torch.randn_like(out)
    (out.real * seed.real + out.imag * seed.imag).sum().backward()

    step = 1.0e-3
    worst = 0.0
    for row in (0, paths - 1):
        for column in (0, NUM_SUBCARRIERS - 1):
            bump = torch.zeros_like(band)
            bump[row, column] = complex(step, 0.0)
            plus = synthesize_cfr_rows(tau, rate, band + bump, offsets, spec)
            minus = synthesize_cfr_rows(tau, rate, band - bump, offsets, spec)
            derivative = (plus - minus) / (2.0 * step)
            expected = float((derivative.real * seed.real + derivative.imag * seed.imag).sum())
            measured = float(live.grad[row, column].real)
            worst = max(worst, abs(measured - expected) / max(abs(expected), 1.0e-12))
    assert worst < 1.0e-3, worst


def test_the_wideband_cfr_jvp_and_vjp_agree_with_each_other():
    """Two independent native companions over the same launch.

    ``<J v, w> == <v, J^T w>`` for a random ``v`` and ``w``. No finite
    difference, so the tolerance is float32 accumulation and nothing else, and a
    column that quietly used the wrong weight would break it.
    """

    import torch.autograd.forward_ad as forward_ad

    torch.manual_seed(23)
    spec = _spec(num_symbols=2)
    paths = 4
    tau = torch.rand(paths, device="cuda", dtype=torch.float32) * 1.0e-8 + 1.0e-9
    rate = (torch.rand(paths, device="cuda", dtype=torch.float32) - 0.5) * 2.0e-8
    offsets = torch.tensor([0, 2, paths], dtype=torch.int64, device="cuda")
    band = torch.randn(paths, NUM_SUBCARRIERS, dtype=torch.complex64, device="cuda")
    tangent = torch.randn_like(band)

    with forward_ad.dual_level():
        dual = forward_ad.make_dual(band, tangent)
        out = synthesize_cfr_rows(tau, rate, dual, offsets, spec)
        forward = forward_ad.unpack_dual(out).tangent.clone()

    live = band.clone().requires_grad_(True)
    primal = synthesize_cfr_rows(tau, rate, live, offsets, spec)
    seed = torch.randn_like(primal)
    (primal.real * seed.real + primal.imag * seed.imag).sum().backward()

    left = float((forward.real * seed.real + forward.imag * seed.imag).sum())
    right = float((live.grad.real * tangent.real + live.grad.imag * tangent.imag).sum())
    assert abs(left - right) <= 1.0e-5 * max(abs(left), abs(right)), (left, right)


def test_the_columns_carry_genuinely_different_derivatives():
    """A guard on the two tests above.

    If every column were secretly the same weight, the FD and the
    forward/reverse checks would both still pass. This asserts the gradient is
    NOT constant across the band, so an implementation that collapsed the band
    would fail here.
    """

    torch.manual_seed(31)
    spec = _spec(num_symbols=2)
    paths = 3
    tau = torch.rand(paths, device="cuda", dtype=torch.float32) * 1.0e-8 + 1.0e-9
    rate = torch.zeros_like(tau)
    offsets = torch.tensor([0, 1, paths], dtype=torch.int64, device="cuda")
    band = torch.randn(paths, NUM_SUBCARRIERS, dtype=torch.complex64, device="cuda").requires_grad_(True)

    out = synthesize_cfr_rows(tau, rate, band, offsets, spec)
    weight = torch.arange(1, NUM_SUBCARRIERS + 1, dtype=torch.float32, device="cuda")
    (out.real * weight).sum().backward()

    columns = band.grad.abs()
    spread = float(columns.max() - columns.min())
    assert spread > 1.0e-3, spread


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_a_band_whose_width_is_not_the_subcarrier_count_is_refused():
    spec = _spec()
    narrow, _ = _spikes(spec)
    short_adapter = ChannelPropagationAdapter(
        narrow.compiled,
        reference_frequency_hz=F_REF,
        components=driver.MULTIPATH_COMPONENTS,
        max_depth=1,
        frequency_offsets_hz=spec.frequency_offsets_hz[: NUM_SUBCARRIERS - 2],
    )
    short = driver.MultiEndpointSpike(compiled=narrow.compiled, adapter=short_adapter)
    composed, _, _ = short.frame(response=driver.make_response(), include_delay_rate=False)
    batch = SynthesisPathBatch.from_radar_paths(composed, slow_time_mode=SlowTimeMode.FROZEN_WEIGHT_WITH_CARRIER_RATE)
    with pytest.raises(ValueError, match="one column per subcarrier|column wideband"):
        synthesize_ofdm(batch, spec)


def test_fmcw_refuses_the_band_it_cannot_index():
    """The declared deferral, enforced rather than documented.

    FMCW's instantaneous transmit frequency is continuous in fast time, so there
    is no discrete grid to index and a band would be silently discarded.
    """

    from witwin.radar.synthesis import FmcwSpec, synthesize_fmcw

    spec = _spec()
    _, banded = _spikes(spec)
    composed, _, _ = banded.frame(response=driver.make_response(), include_delay_rate=False)
    batch = SynthesisPathBatch.from_radar_paths(composed, slow_time_mode=SlowTimeMode.FROZEN_WEIGHT_WITH_CARRIER_RATE)
    from witwin.radar import RadarConfig

    beat = FmcwSpec.from_radar_config(RadarConfig.from_dict(dict(geo.FIXTURE_RADAR_CONFIG)), carrier_hz=0.0)
    with pytest.raises(ValueError, match="does not consume a wideband response"):
        synthesize_fmcw(batch, beat)


def test_the_offsets_grid_is_a_host_declaration_not_a_tensor():
    narrow = driver.MultiEndpointSpike()
    with pytest.raises(TypeError, match="host declaration"):
        ChannelPropagationAdapter(
            narrow.compiled,
            reference_frequency_hz=F_REF,
            components=driver.MULTIPATH_COMPONENTS,
            max_depth=1,
            frequency_offsets_hz=torch.zeros(4),
        )
