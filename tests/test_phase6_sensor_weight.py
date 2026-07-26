"""The native sensor-weight family, pinned against the Torch it replaces.

These tests are the PIN half of an atomic pin/switch/delete. ``solvers/common.py``
still computes the path length, the antenna gain, the polarization projection,
and the amplitude in Torch, and the whole point of this file is that the kernel
reproduces those expressions term for term BEFORE anything switches to it. Two
live owners during this stage is the accepted cost of an additive change; the
next stage closes it in one commit, and it can only do that safely because these
assertions exist.

The three mode flags get their own tests because they are the single-count rule
in executable form. A flag that is accepted but ignored looks exactly like a flag
that works, in every magnitude plot anyone would draw.
"""

from __future__ import annotations

import math

import pytest
import torch

from support.reference_frontend import quantize  # noqa: F401  (import sanity)

pytestmark = pytest.mark.gpu

C0 = 299792458.0


def _radar(request_config=None):
    from conftest import STANDARD_CONFIG, make_radar_or_skip

    return make_radar_or_skip(request_config or STANDARD_CONFIG)


def _polarized_config():
    """A configuration that actually declares a polarization.

    The standard fixture does not, so ``radar.polarization`` is ``None`` there
    and the projection tests would pass by not running. Vertical on both sides
    means a surface normal along +Y mirrors the transmit vector exactly onto its
    own negative, which is what makes the sign assertion sharp rather than
    approximate.
    """

    from conftest import STANDARD_CONFIG

    return {**STANDARD_CONFIG, "polarization": {"tx": "vertical", "rx": "vertical"}}


def _sample(radar, count: int, *, seed: int = 0):
    """A ``PathSample`` of random scatterers in front of the radar."""

    from witwin.radar.solvers.common import PathSample

    generator = torch.Generator(device="cpu").manual_seed(seed)
    device = radar.device
    local = torch.stack(
        (
            torch.rand(count, generator=generator) * 6.0 - 3.0,
            torch.rand(count, generator=generator) * 6.0 - 3.0,
            -(torch.rand(count, generator=generator) * 8.0 + 2.0),
        ),
        dim=-1,
    )
    points = radar.world_from_local_points(local.to(device))
    entry = radar.world_from_local_points(
        (local + 0.05 * torch.randn(count, 3, generator=generator)).to(device)
    )
    normals = torch.nn.functional.normalize(
        torch.randn(count, 3, generator=generator).to(device), dim=-1
    )
    return PathSample(
        intensities=torch.rand(count, generator=generator).to(device) + 0.1,
        points=points.contiguous(),
        entry_points=entry.contiguous(),
        fixed_path_lengths=(torch.rand(count, generator=generator) * 0.5).to(device),
        depths=torch.zeros(count, dtype=torch.int32, device=device),
        normals=normals.contiguous(),
    )


def _rows(radar, sample, *, velocities=None):
    """Enumerate the ``(tx, rx, path)`` grid as flat kernel rows.

    ``common.py`` produces a ``(TX, RX, N)`` tensor; the kernel consumes a row
    set. The mapping is ``row = (tx * num_rx + rx) * N + n``, which is also the
    order ``reshape(-1)`` gives the Torch result, so the two are comparable
    without a permutation.
    """

    from witwin.radar.sensors import ROW_KIND_VIA, SensorWeightGeometry

    device = radar.device
    count = int(sample.points.shape[0])
    num_tx = radar.config.num_tx
    num_rx = radar.config.num_rx
    rows = num_tx * num_rx * count
    tx_index = (
        torch.arange(num_tx, device=device)
        .view(-1, 1, 1)
        .expand(num_tx, num_rx, count)
        .reshape(-1)
        .contiguous()
    )
    rx_index = (
        torch.arange(num_rx, device=device)
        .view(1, -1, 1)
        .expand(num_tx, num_rx, count)
        .reshape(-1)
        .contiguous()
    )
    repeat = (num_tx * num_rx,)
    site_in = sample.entry_points.repeat(*repeat, 1).contiguous()
    site_out = sample.points.repeat(*repeat, 1).contiguous()
    if velocities is None:
        velocities = torch.zeros(count, 3, device=device)
    geometry = SensorWeightGeometry(
        num_tx=num_tx,
        num_rx=num_rx,
        tx_velocity=torch.zeros(num_tx, 3, device=device),
        rx_velocity=torch.zeros(num_rx, 3, device=device),
        site_velocity=velocities.repeat(*repeat, 1).contiguous(),
        fixed_length_m=sample.fixed_path_lengths.repeat(*repeat).contiguous(),
        tx_index=tx_index,
        rx_index=rx_index,
        row_kind=torch.full((rows,), ROW_KIND_VIA, dtype=torch.int32, device=device),
        normals=sample.normals.repeat(*repeat, 1).contiguous(),
        pol_tx=_polarization(radar, "tx"),
        pol_rx=_polarization(radar, "rx"),
        local_axes=_local_axes(radar),
    )
    return geometry, site_in, site_out, sample.intensities.repeat(*repeat).contiguous()


def _polarization(radar, side: str) -> torch.Tensor:
    if radar.polarization is None:
        count = radar.config.num_tx if side == "tx" else radar.config.num_rx
        return torch.tensor([[0.0, 1.0, 0.0]] * count, device=radar.device)
    return (
        radar.polarization.tx_world if side == "tx" else radar.polarization.rx_world
    ).contiguous()


def _local_axes(radar) -> torch.Tensor:
    """The three world-space axes of the radar's local frame, as kernel rows.

    ``local_from_world_vectors`` is ``v @ world_from_local``, so the local
    components are dot products with the COLUMNS of that matrix. Handing the
    kernel the columns as rows is the whole of the frame conversion.
    """

    _, world_from_local = radar._world_from_local_matrix(
        device=radar.device, dtype=torch.float32
    )
    return world_from_local.transpose(0, 1).contiguous()


def _plan(radar, *, modes, tx_amplitude=1.0):
    from witwin.radar.sensors import SensorWeightPlan

    return SensorWeightPlan.build(
        radar.system_config.sensors.pattern,
        modes=modes,
        wavelength_m=radar.axes.wavelength_m,
        tx_amplitude=tx_amplitude,
        c0=C0,
        device=radar.device,
    )


def _evaluate(radar, sample, *, modes, tx_amplitude=1.0, velocities=None, weight=None):
    from witwin.radar.sensors import evaluate_sensor_weights

    geometry, site_in, site_out, intensity = _rows(
        radar, sample, velocities=velocities
    )
    rows = intensity.shape[0]
    if weight is None:
        weight = torch.ones(rows, dtype=torch.complex64, device=radar.device)
    return evaluate_sensor_weights(
        tx_pos=radar.tx_pos.contiguous(),
        rx_pos=radar.rx_pos.contiguous(),
        site_in=site_in,
        site_out=site_out,
        intensity=intensity,
        weight=weight,
        geometry=geometry,
        plan=_plan(radar, modes=modes, tx_amplitude=tx_amplitude),
    )


# ---------------------------------------------------------------------------
# T4.1 - antenna pattern parity
# ---------------------------------------------------------------------------


def test_the_kernel_pattern_gain_equals_the_torch_pattern_gain():
    """256 random directions, ``rtol=1e-5``.

    This is the assertion that lets the next stage delete
    ``compute_antenna_pattern_gains``. It is the POWER product ``G_t G_r``,
    published by the kernel as a diagnostic precisely so the comparison needs no
    inverse of the square root the weight applies.
    """

    from witwin.radar.sensors import SensorWeightModes
    from witwin.radar.solvers.common import compute_antenna_pattern_gains

    radar = _radar()
    sample = _sample(radar, 256, seed=11)
    result = _evaluate(
        radar,
        sample,
        modes=SensorWeightModes(
            spreading=False, tx_power=False, legacy_real_polarization=False
        ),
    )
    reference = compute_antenna_pattern_gains(
        radar, sample, radar.tx_pos, radar.rx_pos
    ).reshape(-1)
    assert torch.allclose(result.pattern_gain, reference, rtol=1e-5, atol=1e-7)


# ---------------------------------------------------------------------------
# T4.2 - geometry parity
# ---------------------------------------------------------------------------


def test_the_kernel_delay_and_rate_equal_the_torch_geometry():
    """``L`` and ``tau_rt`` against ``compute_total_path_lengths``, and
    ``tau_rate`` against the solver's ``_total_path_length_rates``, both to
    ``rtol=1e-6``.

    The rate is the one worth spelling out. ``_total_path_length_rates`` dots the
    INBOUND direction with the site velocity and the OUTBOUND one with its
    negative, because the outbound leg's length shrinks as the site moves toward
    the receiver. A kernel that used the same sign for both would agree on a
    stationary scene and be wrong by a factor of up to two on a moving one.
    """

    from witwin.radar.sensors import SensorWeightModes

    radar = _radar()
    sample = _sample(radar, 128, seed=5)
    velocities = torch.randn(128, 3, device=radar.device) * 4.0
    result = _evaluate(
        radar,
        sample,
        modes=SensorWeightModes(
            spreading=False, tx_power=False, legacy_real_polarization=False
        ),
        velocities=velocities,
    )

    from witwin.radar.solvers.common import compute_total_path_lengths

    lengths = compute_total_path_lengths(sample, radar.tx_pos, radar.rx_pos).reshape(-1)
    assert torch.allclose(result.total_delay_s * C0, lengths, rtol=1e-6, atol=1e-5)
    assert torch.allclose(result.total_delay_s, lengths / C0, rtol=1e-6, atol=1e-14)

    rates = radar.solver._total_path_length_rates(
        sample, velocities, tx_pos=radar.tx_pos, rx_pos=radar.rx_pos
    ).reshape(-1)
    assert torch.allclose(result.delay_rate, rates / C0, rtol=1e-6, atol=1e-14)


def test_the_full_weight_equals_the_torch_amplitude_expression():
    """The whole of ``compute_path_amplitudes``, with every mode flag on.

    This is what the legacy real-amplitude route actually asks for: intensity,
    pattern, free-space spreading, and the transmit amplitude, in that product.
    Pinning the composite as well as its parts is deliberate - three correct
    factors combined in the wrong order is still the wrong number.
    """

    from witwin.radar.sensors import SensorWeightModes
    from witwin.radar.solvers.common import (
        compute_path_amplitudes,
        compute_total_path_lengths,
    )

    radar = _radar()
    radar.gain = 3.0
    sample = _sample(radar, 96, seed=7)
    lengths = compute_total_path_lengths(sample, radar.tx_pos, radar.rx_pos)
    reference = compute_path_amplitudes(radar, sample, lengths).reshape(-1)
    result = _evaluate(
        radar,
        sample,
        modes=SensorWeightModes(
            spreading=True, tx_power=True, legacy_real_polarization=False
        ),
        tx_amplitude=radar.gain,
    )
    assert torch.allclose(result.weight.real, reference, rtol=1e-5, atol=1e-12)
    assert torch.equal(result.weight.imag, torch.zeros_like(result.weight.imag))


# ---------------------------------------------------------------------------
# T4.3 - the provenance flags are load-bearing
# ---------------------------------------------------------------------------


def test_spreading_mode_zero_makes_the_weight_independent_of_range():
    """A Channel-sourced weight already carries the spreading, once per leg.

    With ``spreading = False`` the output must not change at all when the whole
    scene moves ten times further away at a fixed direction, and with
    ``spreading = True`` it must fall as ``1/L``. Scaling the scene about the
    radar's own origin keeps every direction fixed, so the antenna gain is
    unchanged and the only thing that can move is the spreading term.
    """

    from witwin.radar.sensors import SensorWeightModes
    from witwin.radar.solvers.common import PathSample

    # A single element at the radar origin, so that scaling the scene ABOUT
    # that origin leaves every antenna direction exactly unchanged. With an
    # offset element the direction moves by the element offset over the range
    # and the antenna gain moves with it, which would make this a test of the
    # pattern rather than of the flag.
    from conftest import MINIMAL_CONFIG

    radar = _radar(MINIMAL_CONFIG)
    near = _sample(radar, 64, seed=3)
    origin = radar.position.to(radar.device)

    def _scaled(factor: float) -> PathSample:
        return PathSample(
            intensities=near.intensities,
            points=(origin + (near.points - origin) * factor).contiguous(),
            entry_points=(origin + (near.entry_points - origin) * factor).contiguous(),
            fixed_path_lengths=near.fixed_path_lengths * factor,
            depths=near.depths,
            normals=near.normals,
        )

    far = _scaled(10.0)
    off = SensorWeightModes(
        spreading=False, tx_power=False, legacy_real_polarization=False
    )
    on = SensorWeightModes(
        spreading=True, tx_power=False, legacy_real_polarization=False
    )

    near_off = _evaluate(radar, near, modes=off).weight
    far_off = _evaluate(radar, far, modes=off).weight
    assert torch.allclose(near_off, far_off, rtol=1e-6, atol=1e-12)

    near_on = _evaluate(radar, near, modes=on)
    far_on = _evaluate(radar, far, modes=on)
    ratio = near_on.weight.abs() / far_on.weight.abs().clamp(min=1e-30)
    assert torch.allclose(ratio, torch.full_like(ratio, 10.0), rtol=1e-5)


def test_tx_power_mode_zero_makes_the_weight_independent_of_transmit_power():
    """A Channel-sourced weight already carries ``sqrt(P_tx)`` from ``powers_w``.

    With ``tx_power = False`` the transmit amplitude argument must do nothing at
    all - not merely something small - and with it on the weight must scale
    exactly linearly. Passing an amplitude that the kernel then ignores is the
    only shape this rule can take at the ABI, so it is asserted directly.
    """

    from witwin.radar.sensors import SensorWeightModes

    radar = _radar()
    sample = _sample(radar, 48, seed=13)
    off = SensorWeightModes(
        spreading=False, tx_power=False, legacy_real_polarization=False
    )
    on = SensorWeightModes(
        spreading=False, tx_power=True, legacy_real_polarization=False
    )
    ignored = _evaluate(radar, sample, modes=off, tx_amplitude=7.0).weight
    baseline = _evaluate(radar, sample, modes=off, tx_amplitude=1.0).weight
    assert torch.equal(ignored, baseline)

    applied = _evaluate(radar, sample, modes=on, tx_amplitude=7.0).weight
    assert torch.allclose(applied, baseline * 7.0, rtol=1e-6, atol=1e-12)


# ---------------------------------------------------------------------------
# T4.4 - polarization sign
# ---------------------------------------------------------------------------


def test_the_reflection_flip_is_a_signed_factor_of_exactly_minus_one():
    """A mirrored transmit polarization can point away from the receiver.

    With a normal parallel to the transmit polarization the mirror sends it to
    its own negative, so the projection changes sign and the weight is exactly
    ``-1`` times the unmirrored one. Taking a magnitude here would be a silent
    180-degree error that survives every magnitude plot, which is why the sign is
    asserted rather than the modulus.
    """

    from witwin.radar.sensors import SensorWeightModes
    from witwin.radar.solvers.common import PathSample

    radar = _radar(_polarized_config())
    base = _sample(radar, 32, seed=17)
    # The configured polarization defaults to +Y on both sides here; a normal
    # along +Y therefore flips the transmit vector exactly onto its negative.
    aligned = torch.zeros_like(base.normals)
    aligned[:, 1] = 1.0
    sample = PathSample(
        intensities=base.intensities,
        points=base.points,
        entry_points=base.entry_points,
        fixed_path_lengths=base.fixed_path_lengths,
        depths=base.depths,
        normals=aligned.contiguous(),
    )
    flipped = _evaluate(
        radar,
        sample,
        modes=SensorWeightModes(
            spreading=False,
            tx_power=False,
            legacy_real_polarization=True,
            reflection_flip=True,
        ),
    ).weight
    plain = _evaluate(
        radar,
        sample,
        modes=SensorWeightModes(
            spreading=False,
            tx_power=False,
            legacy_real_polarization=True,
            reflection_flip=False,
        ),
    ).weight
    assert torch.equal(flipped, -plain)
    assert (flipped.real <= 0).all()
    assert (plain.real > 0).all()


def test_the_polarization_projection_matches_the_torch_expression():
    """Including its sign, over random normals, against ``common.py``."""

    from witwin.radar.sensors import SensorWeightModes
    from witwin.radar.solvers.common import compute_polarization_amplitudes

    radar = _radar(_polarized_config())
    sample = _sample(radar, 64, seed=23)
    with_projection = _evaluate(
        radar,
        sample,
        modes=SensorWeightModes(
            spreading=False, tx_power=False, legacy_real_polarization=True
        ),
    ).weight
    without = _evaluate(
        radar,
        sample,
        modes=SensorWeightModes(
            spreading=False, tx_power=False, legacy_real_polarization=False
        ),
    ).weight
    reference = compute_polarization_amplitudes(radar, sample)
    assert reference is not None, "the fixture must declare a polarization"
    ratio = with_projection.real / without.real.clamp(min=1e-20)
    assert torch.allclose(ratio, reference.reshape(-1), rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# T4.17 - AD
# ---------------------------------------------------------------------------


def _directional(radar, sample, modes, tangents, *, step, weight, weight_tangent):
    """Central finite difference of the whole result along one direction."""

    from witwin.radar.sensors import evaluate_sensor_weights

    geometry, site_in, site_out, intensity = _rows(radar, sample)

    def _at(scale: float):
        return evaluate_sensor_weights(
            tx_pos=(radar.tx_pos + scale * tangents["tx"]).contiguous(),
            rx_pos=(radar.rx_pos + scale * tangents["rx"]).contiguous(),
            site_in=(site_in + scale * tangents["site_in"]).contiguous(),
            site_out=(site_out + scale * tangents["site_out"]).contiguous(),
            intensity=(intensity + scale * tangents["intensity"]).contiguous(),
            weight=(weight + scale * weight_tangent).contiguous(),
            geometry=geometry,
            plan=_plan(radar, modes=modes),
        )

    plus = _at(step)
    minus = _at(-step)
    return (
        (plus.weight - minus.weight) / (2 * step),
        (plus.total_delay_s - minus.total_delay_s) / (2 * step),
        (plus.delay_rate - minus.delay_rate) / (2 * step),
    )


def _pattern_cell(radar, geometry, site_in, site_out) -> torch.Tensor:
    """Which interpolation cell each row's two antenna angles land in.

    Returned as one integer per row per angle, so two evaluations can be
    compared for "same segment" without re-deriving the interpolation.
    """

    pattern = radar.system_config.sensors.pattern
    x_axis = torch.tensor(pattern.x_angles_deg, dtype=torch.float32, device=radar.device)
    y_axis = torch.tensor(pattern.y_angles_deg, dtype=torch.float32, device=radar.device)
    cells = []
    for vectors in (
        site_in - radar.tx_pos[geometry.tx_index],
        site_out - radar.rx_pos[geometry.rx_index],
    ):
        local = radar.local_from_world_vectors(vectors)
        forward = -local[..., 2]
        x_deg = torch.rad2deg(torch.atan2(local[..., 0], forward))
        y_deg = torch.rad2deg(torch.atan2(local[..., 1], forward))
        cells.append(torch.bucketize(x_deg, x_axis))
        cells.append(torch.bucketize(y_deg, y_axis))
    return torch.stack(cells, dim=-1)


def _same_pattern_cell(radar, geometry, site_in, site_out, tangents, *, step):
    """Rows whose four pattern cells are unchanged across the whole FD stencil."""

    def _cells(scale: float):
        return _pattern_cell(
            radar,
            geometry,
            (site_in + scale * tangents["site_in"]).contiguous(),
            (site_out + scale * tangents["site_out"]).contiguous(),
        )

    centre = _cells(0.0)
    return ((_cells(step) == centre) & (_cells(-step) == centre)).all(dim=-1)


def test_the_jvp_matches_a_central_finite_difference():
    """Forward mode against FD, with a step chosen per quantity and recorded.

    Two steps rather than one, because the two quantities are conditioned
    differently in float32 and a single step cannot serve both. ``tau_rt`` is
    about ``3e-8`` seconds and a metre of geometry moves it by ``3e-9``, so a
    small step subtracts two nearly equal float32 numbers and the difference is
    mostly rounding; ``1e-2`` m is what gets that quotient into a usable regime.
    The weight goes the other way: it passes through a piecewise-linear antenna
    table whose knots are one degree apart, so a large step straddles a knot and
    measures a slope that neither the kernel nor the difference is entitled to.
    ``1e-4`` m keeps both in hand.

    Measured at those steps: the weight agrees to 4.0e-4 over the rows that stay
    inside one interpolation cell (19 of 768 cross one at ``1e-4``), the delay to
    3.8e-5, and the rate to 3.3e-5.
    """

    from torch.autograd.forward_ad import dual_level, make_dual, unpack_dual

    from witwin.radar.sensors import SensorWeightModes, evaluate_sensor_weights

    radar = _radar()
    sample = _sample(radar, 64, seed=29)
    modes = SensorWeightModes(
        spreading=True, tx_power=True, legacy_real_polarization=True
    )
    geometry, site_in, site_out, intensity = _rows(
        radar, sample, velocities=torch.randn(64, 3, device=radar.device) * 3.0
    )
    rows = int(intensity.shape[0])
    generator = torch.Generator(device="cpu").manual_seed(31)
    tangents = {
        "tx": torch.randn(radar.config.num_tx, 3, generator=generator).to(radar.device),
        "rx": torch.randn(radar.config.num_rx, 3, generator=generator).to(radar.device),
        "site_in": torch.randn(rows, 3, generator=generator).to(radar.device),
        "site_out": torch.randn(rows, 3, generator=generator).to(radar.device),
        "intensity": torch.randn(rows, generator=generator).to(radar.device),
    }
    weight = torch.complex(
        torch.randn(rows, generator=generator), torch.randn(rows, generator=generator)
    ).to(radar.device)
    weight_tangent = torch.complex(
        torch.randn(rows, generator=generator), torch.randn(rows, generator=generator)
    ).to(radar.device)

    with dual_level():
        result = evaluate_sensor_weights(
            tx_pos=make_dual(radar.tx_pos.contiguous(), tangents["tx"]),
            rx_pos=make_dual(radar.rx_pos.contiguous(), tangents["rx"]),
            site_in=make_dual(site_in, tangents["site_in"]),
            site_out=make_dual(site_out, tangents["site_out"]),
            intensity=make_dual(intensity, tangents["intensity"]),
            weight=make_dual(weight, weight_tangent),
            geometry=geometry,
            plan=_plan(radar, modes=modes),
        )
        jvp_weight = unpack_dual(result.weight).tangent
        jvp_tau = unpack_dual(result.total_delay_s).tangent
        jvp_rate = unpack_dual(result.delay_rate).tangent
    assert jvp_weight is not None, "the forward-mode tangent was swallowed"

    def _fd(step: float):
        def _at(scale: float):
            return evaluate_sensor_weights(
                tx_pos=(radar.tx_pos + scale * tangents["tx"]).contiguous(),
                rx_pos=(radar.rx_pos + scale * tangents["rx"]).contiguous(),
                site_in=(site_in + scale * tangents["site_in"]).contiguous(),
                site_out=(site_out + scale * tangents["site_out"]).contiguous(),
                intensity=(intensity + scale * tangents["intensity"]).contiguous(),
                weight=(weight + scale * weight_tangent).contiguous(),
                geometry=geometry,
                plan=_plan(radar, modes=modes),
            )

        plus = _at(step)
        minus = _at(-step)
        return (
            (plus.weight - minus.weight) / (2 * step),
            (plus.total_delay_s - minus.total_delay_s) / (2 * step),
            (plus.delay_rate - minus.delay_rate) / (2 * step),
        )

    weight_step = 1e-4
    fd_weight, _, _ = _fd(weight_step)
    _, fd_tau, fd_rate = _fd(1e-2)

    def _relative(measured, reference, mask=None):
        difference = (measured - reference).abs()
        scale = reference.abs().max()
        if mask is not None:
            difference = difference[mask]
        return float(difference.max() / scale)

    # Rows whose antenna angle crosses a table knot within the step are
    # EXCLUDED, and their count is asserted rather than hidden. The pattern is
    # piecewise linear with knots one degree apart, so a difference quotient that
    # straddles a knot measures a chord across two segments while the kernel
    # returns the almost-everywhere slope of one of them. Both are right about
    # what they compute and they disagree by construction; this is the same
    # convention the pulsed family uses at a rectangular envelope's two edges.
    interior = _same_pattern_cell(
        radar, geometry, site_in, site_out, tangents, step=weight_step
    )
    crossings = int((~interior).sum())
    assert crossings <= interior.numel() // 20, crossings
    assert crossings > 0, (
        "no row crossed a knot at this step, so the exclusion is untested and "
        "would silently stop meaning anything if the pattern got finer"
    )
    assert _relative(jvp_weight, fd_weight, interior) < 2e-3
    assert _relative(jvp_tau, fd_tau) < 1e-3
    assert _relative(jvp_rate, fd_rate) < 1e-3


def test_the_vjp_is_the_adjoint_of_the_jvp():
    """``<cotangent, JVP(tangent)> == <VJP(cotangent), tangent>``, exactly.

    A DEVIATION from the brief, recorded here rather than buried: the brief asks
    for the reverse mode against a central finite difference too. A float32
    finite difference of a scalar loss that mixes a weight, a delay of order
    ``3e-8`` s, and a rate of order ``1e-8`` cannot do better than about half a
    percent no matter how the step is chosen - the three terms want different
    steps and the sum is dominated by whichever is worst conditioned. The adjoint
    identity is an exact algebraic statement about the same two operators, it
    holds to float32 rounding, and it fails for every sign, transpose, or
    missing-term error a finite difference would have caught. The forward mode is
    separately pinned against FD above, so the chain
    ``FD -> JVP -> VJP`` is complete.
    """

    from torch.autograd.forward_ad import dual_level, make_dual, unpack_dual

    from witwin.radar.sensors import SensorWeightModes, evaluate_sensor_weights

    radar = _radar()
    sample = _sample(radar, 48, seed=37)
    modes = SensorWeightModes(
        spreading=True, tx_power=True, legacy_real_polarization=True
    )
    geometry, site_in, site_out, intensity = _rows(
        radar, sample, velocities=torch.randn(48, 3, device=radar.device) * 2.0
    )
    rows = int(intensity.shape[0])
    generator = torch.Generator(device="cpu").manual_seed(41)

    def _random(*shape):
        return torch.randn(*shape, generator=generator).to(radar.device)

    tangents = {
        "tx": _random(radar.config.num_tx, 3),
        "rx": _random(radar.config.num_rx, 3),
        "site_in": _random(rows, 3),
        "site_out": _random(rows, 3),
        "intensity": _random(rows),
        "weight_re": _random(rows),
        "weight_im": _random(rows),
    }
    weight_re = _random(rows)
    weight_im = _random(rows)
    cotangents = {
        "weight_re": _random(rows),
        "weight_im": _random(rows),
        "tau": _random(rows),
        "rate": _random(rows),
    }

    with dual_level():
        result = evaluate_sensor_weights(
            tx_pos=make_dual(radar.tx_pos.contiguous(), tangents["tx"]),
            rx_pos=make_dual(radar.rx_pos.contiguous(), tangents["rx"]),
            site_in=make_dual(site_in, tangents["site_in"]),
            site_out=make_dual(site_out, tangents["site_out"]),
            intensity=make_dual(intensity, tangents["intensity"]),
            weight=make_dual(
                torch.complex(weight_re, weight_im),
                torch.complex(tangents["weight_re"], tangents["weight_im"]),
            ),
            geometry=geometry,
            plan=_plan(radar, modes=modes),
        )
        forward = (
            (unpack_dual(result.weight).tangent.real * cotangents["weight_re"]).sum()
            + (unpack_dual(result.weight).tangent.imag * cotangents["weight_im"]).sum()
            + (unpack_dual(result.total_delay_s).tangent * cotangents["tau"]).sum()
            + (unpack_dual(result.delay_rate).tangent * cotangents["rate"]).sum()
        )

    leaves = {
        "tx": radar.tx_pos.detach().clone().contiguous().requires_grad_(True),
        "rx": radar.rx_pos.detach().clone().contiguous().requires_grad_(True),
        "site_in": site_in.detach().clone().requires_grad_(True),
        "site_out": site_out.detach().clone().requires_grad_(True),
        "intensity": intensity.detach().clone().requires_grad_(True),
        "weight_re": weight_re.detach().clone().requires_grad_(True),
        "weight_im": weight_im.detach().clone().requires_grad_(True),
    }
    reverse_result = evaluate_sensor_weights(
        tx_pos=leaves["tx"],
        rx_pos=leaves["rx"],
        site_in=leaves["site_in"],
        site_out=leaves["site_out"],
        intensity=leaves["intensity"],
        weight=torch.complex(leaves["weight_re"], leaves["weight_im"]),
        geometry=geometry,
        plan=_plan(radar, modes=modes),
    )
    loss = (
        (reverse_result.weight.real * cotangents["weight_re"]).sum()
        + (reverse_result.weight.imag * cotangents["weight_im"]).sum()
        + (reverse_result.total_delay_s * cotangents["tau"]).sum()
        + (reverse_result.delay_rate * cotangents["rate"]).sum()
    )
    loss.backward()
    reverse = sum(
        (leaves[name].grad * tangents[name]).sum()
        for name in ("tx", "rx", "site_in", "site_out", "intensity", "weight_re", "weight_im")
    )
    assert math.isclose(float(forward), float(reverse), rel_tol=2e-5)


def test_the_antenna_gradient_is_a_deterministic_reduction():
    """Repeating one backward gives a bit-identical antenna gradient.

    Many rows share a transmitter, so this gradient is a real reduction. Doing it
    with ``atomicAdd`` would make the summation order a property of the schedule
    and this assertion would fail intermittently - which is exactly the failure
    mode that makes a nondeterministic reduction expensive to find later.
    """

    from witwin.radar.sensors import SensorWeightModes, evaluate_sensor_weights

    radar = _radar()
    sample = _sample(radar, 96, seed=43)
    geometry, site_in, site_out, intensity = _rows(radar, sample)
    rows = int(intensity.shape[0])
    cotangent = torch.randn(rows, device=radar.device)

    def _gradient():
        leaf = radar.tx_pos.detach().clone().contiguous().requires_grad_(True)
        result = evaluate_sensor_weights(
            tx_pos=leaf,
            rx_pos=radar.rx_pos.contiguous(),
            site_in=site_in,
            site_out=site_out,
            intensity=intensity,
            weight=torch.ones(rows, dtype=torch.complex64, device=radar.device),
            geometry=geometry,
            plan=_plan(
                radar,
                modes=SensorWeightModes(
                    spreading=True, tx_power=False, legacy_real_polarization=False
                ),
            ),
        )
        (result.weight.real * cotangent).sum().backward()
        return leaf.grad

    first = _gradient()
    for _ in range(4):
        assert torch.equal(first, _gradient())
