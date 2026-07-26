"""The aspect-dependent scatter response kernel, against its closed form.

Direct contract coverage for ``scatter_response_aspect_forward``, plus the
composition path that makes it a production symbol rather than cleanup debt,
plus the negative case that proves there is no fallback behind it.

The law is written once, in ``support/aspect_oracle.py``, in float64 on the
host. The kernel accumulates in double and rounds once to float32, so the two
agree to a few float32 ULPs; the tolerances below say so rather than using a
default that would pass for almost any answer.
"""

from __future__ import annotations

import math

import pytest
import torch

from witwin.radar.paths import TwoWayComposer
from witwin.radar.scattering import AspectScatterResponse, ScalarRcsResponse

from support import aspect_oracle as oracle  # noqa: E402
from support import join_fixture as fx  # noqa: E402


pytestmark = pytest.mark.gpu

SOURCES = [10, 11]
SINKS = [30, 31]
SITES = [20, 21]
REFERENCE_FREQUENCY_HZ = 77.0e9
EXPONENT = 2.0

#: One coherent interval of the fixture radar, and short enough that the
#: declared aspect rate below is comfortably inside the budget.
COHERENT_INTERVAL_S = 1.0e-3

#: Magnitude agreement, relative. The kernel rounds float64 to float32 once.
MAGNITUDE_RTOL = 1.0e-5

#: Phase agreement, in radians.
PHASE_ATOL_RAD = 1.0e-5


def _ops():
    from witwin.radar.cuda import build

    return build.build_extension()


def _composer(device: str = "cuda", *, outbound_components=(0,)) -> TwoWayComposer:
    """A two-site join whose OUTBOUND leg is line of sight only.

    The aspect response needs the departure direction at the site, and a leg
    publishes its final segment's direction; those coincide only for a
    line-of-sight outbound row. The fabricated fixture aliases depth to the
    component, so a single component ``0`` outbound leg is a depth-0 leg.
    """

    inbound = fx.frozen_leg(fx.leg_rows(SOURCES, SITES, (0, 1)), device=device)
    outbound = fx.frozen_leg(
        fx.leg_rows(SITES, SINKS, outbound_components), device=device
    )
    return TwoWayComposer.freeze(
        inbound,
        outbound,
        torch.tensor(SITES, dtype=torch.int64, device=device),
        radar_source_ids=SOURCES,
        radar_sink_ids=SINKS,
        reference_frequency_hz=REFERENCE_FREQUENCY_HZ,
    )


def _directions(
    rows: int, *, seed: int, device: str = "cuda", into_site: bool = False
) -> torch.Tensor:
    """Unit directions in a cone comfortably clear of the clamp boundary.

    A test that straddles the clamp cannot distinguish the lobe from the
    clamp, and its finite differences would be meaningless. The cone here keeps
    every cosine above 0.3, and one deliberate back-facing case is built by
    hand where it is the subject.

    ``into_site`` flips the cone to the ``-x`` half space, which is what an
    INBOUND direction is: it propagates INTO the site, so its cosine against
    the outward ``+x``-ish aspect axis is positive only when the vector points
    the other way. Building both cones from one generator is what keeps that
    sign visible in the test rather than hidden in a fixture.
    """

    generator = torch.Generator().manual_seed(seed)
    raw = torch.rand(rows, 3, generator=generator, dtype=torch.float64)
    vectors = torch.stack(
        [
            0.6 + 0.4 * raw[:, 0],
            0.6 * (raw[:, 1] - 0.5),
            0.6 * (raw[:, 2] - 0.5),
        ],
        dim=1,
    )
    vectors = vectors / torch.linalg.vector_norm(vectors, dim=1, keepdim=True)
    if into_site:
        vectors = -vectors
    return vectors.to(device=device, dtype=torch.float32).contiguous()


def _parameters(sites: int, *, device: str = "cuda", requires_grad: bool = False):
    axis = oracle.unit_rows([(1.0, 0.15 * index, -0.1 * index) for index in range(sites)])
    amplitude = [1.5 + 0.5 * index for index in range(sites)]
    phase = [0.4 + 0.3 * index for index in range(sites)]
    return AspectScatterResponse(
        axis=axis.to(device=device, dtype=torch.float32)
        .contiguous()
        .requires_grad_(requires_grad),
        amplitude=torch.tensor(amplitude, dtype=torch.float32, device=device)
        .requires_grad_(requires_grad),
        phase_rad=torch.tensor(phase, dtype=torch.float32, device=device)
        .requires_grad_(requires_grad),
        exponent=EXPONENT,
        coherent_interval_s=COHERENT_INTERVAL_S,
    )


def _launch(composer, response, dir_in, dir_out, flags):
    rows = composer.path_count
    s_re = torch.empty(rows, dtype=torch.float32, device="cuda")
    s_im = torch.empty_like(s_re)
    _ops().scatter_response_aspect_forward(
        dir_in,
        dir_out,
        composer.inbound_row,
        composer.outbound_row,
        composer.response_slot,
        response.axis,
        response.amplitude,
        response.phase_rad,
        flags,
        s_re,
        s_im,
        response.exponent,
        rows,
    )
    return s_re, s_im


def test_the_aspect_kernel_matches_a_closed_form():
    """``S = A cos^n(theta_in) cos^n(theta_out) exp(-i phi_0)``, with n = 2.

    Magnitude to 1e-5 relative and phase to 1e-5 radians, both against a
    float64 host evaluation of the same law. The phase is checked separately
    from the magnitude because a magnitude-only comparison passes with the
    Channel-to-beat conjugation inverted.
    """

    composer = _composer()
    response = _parameters(composer.site_count)
    dir_in = _directions(composer.inbound_row_count, seed=41, into_site=True)
    dir_out = _directions(composer.outbound_row_count, seed=42)
    flags = torch.ones(composer.path_count, dtype=torch.int32, device="cuda")

    s_re, s_im = _launch(composer, response, dir_in, dir_out, flags)
    measured = torch.complex(s_re.double(), s_im.double()).cpu()
    expected = oracle.aspect_response(
        dir_in,
        dir_out,
        composer.inbound_row.tolist(),
        composer.outbound_row.tolist(),
        composer.response_slot.tolist(),
        response.axis,
        response.amplitude,
        response.phase_rad,
        EXPONENT,
    )

    assert composer.path_count == len(SOURCES) * len(SINKS) * len(SITES) * 2
    assert float(expected.abs().min()) > 0.0, "the reference must not be vacuous"
    torch.testing.assert_close(
        measured.abs(), expected.abs(), rtol=MAGNITUDE_RTOL, atol=0.0
    )
    for index in range(len(expected)):
        difference = math.remainder(
            float(torch.angle(measured[index]) - torch.angle(expected[index])),
            2.0 * math.pi,
        )
        assert abs(difference) < PHASE_ATOL_RAD, index


def test_a_back_facing_direction_publishes_exactly_zero():
    """The clamp is physical, and it is exact rather than small.

    A direction on the far side of the aspect plane is not illuminated by a
    separable forward lobe. Publishing a tiny negative-power value there would
    be a plausible number in the sidelobe region of a plot.
    """

    composer = _composer()
    response = _parameters(composer.site_count)
    dir_in = _directions(composer.inbound_row_count, seed=41, into_site=True)
    # Flip the FIRST outbound row behind the aspect plane and leave the rest.
    dir_out = _directions(composer.outbound_row_count, seed=42)
    dir_out = dir_out.clone()
    dir_out[0] = torch.tensor([-1.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
    flags = torch.ones(composer.path_count, dtype=torch.int32, device="cuda")

    s_re, s_im = _launch(composer, response, dir_in, dir_out, flags)
    affected = [
        row
        for row in range(composer.path_count)
        if int(composer.outbound_row[row]) == 0
    ]
    assert affected, "the flipped row must own at least one composed row"
    for row in affected:
        assert float(s_re[row]) == 0.0 and float(s_im[row]) == 0.0, row
    others = [row for row in range(composer.path_count) if row not in affected]
    assert any(float(s_re[row]) != 0.0 for row in others)


def test_a_dead_row_publishes_exactly_zero():
    """``row_valid`` is the sole authority and a dead row is inert.

    A dead row can carry a stale direction, and a NaN in the response would
    reach a LIVE row through nothing but the shared response tensor.
    """

    composer = _composer()
    response = _parameters(composer.site_count)
    dir_in = _directions(composer.inbound_row_count, seed=41, into_site=True)
    dir_out = _directions(composer.outbound_row_count, seed=42)
    flags = torch.ones(composer.path_count, dtype=torch.int32, device="cuda")
    flags[1] = 0
    dir_in = dir_in.clone()
    dir_in[int(composer.inbound_row[1])] = torch.tensor(
        [float("nan")] * 3, dtype=torch.float32, device="cuda"
    )

    s_re, s_im = _launch(composer, response, dir_in, dir_out, flags)
    assert float(s_re[1]) == 0.0 and float(s_im[1]) == 0.0
    live = [
        row
        for row in range(composer.path_count)
        if row != 1 and int(composer.inbound_row[row]) != int(composer.inbound_row[1])
    ]
    assert live
    assert torch.isfinite(s_re[live]).all() and torch.isfinite(s_im[live]).all()


def test_the_composed_round_trip_carries_the_row_response():
    """The production end-to-end caller: ``compose`` with the aspect response.

    ``C_rt = (C_out * S_row) * C_in`` with ``S_row`` the per-composed-row
    response, and the join is the SAME kernel that carries a per-site response -
    it is handed an identity site index and an identity CSR. So this asserts
    both that the response reached the join and that routing it through the
    identity family did not change the composition.
    """

    composer = _composer()
    response = _parameters(composer.site_count)
    dir_in = _directions(composer.inbound_row_count, seed=41, into_site=True)
    dir_out = _directions(composer.outbound_row_count, seed=42)
    tau_in, _, c_in = fx.payload(composer.inbound_row_count, seed=11)
    tau_out, _, c_out = fx.payload(composer.outbound_row_count, seed=12)
    inbound = fx.leg_batch(
        tau_in.float(), c_in.to(torch.complex64), direction=dir_in
    )
    outbound = fx.leg_batch(
        tau_out.float(), c_out.to(torch.complex64), direction=dir_out
    )

    composed = composer.compose(inbound, outbound, response)

    expected_response = oracle.aspect_response(
        dir_in,
        dir_out,
        composer.inbound_row.tolist(),
        composer.outbound_row.tolist(),
        composer.response_slot.tolist(),
        response.axis,
        response.amplitude,
        response.phase_rad,
        EXPONENT,
    )
    idx_in = composer.inbound_row.tolist()
    idx_out = composer.outbound_row.tolist()
    reference = torch.stack(
        [
            (
                c_out.cpu().to(torch.complex128)[idx_out[row]]
                * expected_response[row]
            )
            * c_in.cpu().to(torch.complex128)[idx_in[row]]
            for row in range(composer.path_count)
        ]
    )
    measured = composed.complex_transfer_ref.detach().cpu().to(torch.complex128)
    torch.testing.assert_close(measured, reference, rtol=2.0e-5, atol=0.0)


def test_a_line_of_sight_only_response_still_serves_the_per_site_route():
    """The narrowing is additive: a per-site response is unchanged.

    Same join, same legs, the ordinary ``ScalarRcsResponse``. If the response
    dispatch had started routing everything through the identity family the
    per-site gradient reduction would silently become a per-row one.
    """

    composer = _composer()
    tau_in, _, c_in = fx.payload(composer.inbound_row_count, seed=11)
    tau_out, _, c_out = fx.payload(composer.outbound_row_count, seed=12)
    inbound = fx.leg_batch(tau_in.float(), c_in.to(torch.complex64))
    outbound = fx.leg_batch(tau_out.float(), c_out.to(torch.complex64))
    scalar = ScalarRcsResponse.from_values(2.0, 0.3, device="cuda")

    composed = composer.compose(inbound, outbound, scalar)
    site = complex(2.0 * math.cos(0.3), -2.0 * math.sin(0.3))
    idx_in = composer.inbound_row.tolist()
    idx_out = composer.outbound_row.tolist()
    reference = torch.stack(
        [
            (c_out.cpu().to(torch.complex128)[idx_out[row]] * site)
            * c_in.cpu().to(torch.complex128)[idx_in[row]]
            for row in range(composer.path_count)
        ]
    )
    measured = composed.complex_transfer_ref.detach().cpu().to(torch.complex128)
    torch.testing.assert_close(measured, reference, rtol=2.0e-5, atol=0.0)


class _HandWrittenGeometryResponse:
    """A geometry-dependent response that is NOT the native one.

    It even has an ``evaluate_rows`` method, because the composer's check is
    against a declared owner NAME rather than against a protocol: a protocol
    check can see a method's name and not what runs behind it, and this class
    is exactly the thing it would wave through.
    """

    is_geometry_dependent = True

    def evaluate(self, row_count, device):  # pragma: no cover - never reached
        raise AssertionError("the composer must refuse before evaluating")

    def evaluate_rows(self, composer, inbound, outbound, row_valid):
        # A Torch expression in place of a kernel - the exact thing the
        # refusal exists to stop.
        rows = composer.path_count
        real = torch.ones(rows, dtype=torch.float32, device=inbound.delay_s.device)
        return real, real.clone()


def test_the_composer_still_refuses_an_unowned_geometry_dependent_response():
    """The guard narrows; it does not disappear."""

    composer = _composer()
    tau_in, _, c_in = fx.payload(composer.inbound_row_count, seed=11)
    tau_out, _, c_out = fx.payload(composer.outbound_row_count, seed=12)
    inbound = fx.leg_batch(tau_in.float(), c_in.to(torch.complex64))
    outbound = fx.leg_batch(tau_out.float(), c_out.to(torch.complex64))

    with pytest.raises(NotImplementedError, match="must be evaluated in a native kernel"):
        composer.compose(inbound, outbound, _HandWrittenGeometryResponse())


def test_the_response_refuses_a_leg_without_a_direction():
    """No fallback: a fabricated leg has no geometry and gets no guess.

    A response that invented a direction here - a default axis, a zero vector,
    the previous frame's - would produce a plausible lobe from nothing.
    """

    composer = _composer()
    response = _parameters(composer.site_count)
    tau_in, _, c_in = fx.payload(composer.inbound_row_count, seed=11)
    tau_out, _, c_out = fx.payload(composer.outbound_row_count, seed=12)
    inbound = fx.leg_batch(tau_in.float(), c_in.to(torch.complex64))
    outbound = fx.leg_batch(tau_out.float(), c_out.to(torch.complex64))

    with pytest.raises(ValueError, match="carries no field_direction"):
        composer.compose(inbound, outbound, response)


def test_the_response_refuses_a_higher_order_outbound_leg():
    """The one honest limitation of reading the published direction basis.

    A leg publishes the direction of its FINAL segment. For the outbound leg
    that is the arrival direction at the receiver, which equals the departure
    direction from the site only when the row is line of sight. Reading it as a
    departure direction for a reflected row is wrong by the reflection angle
    and entirely plausible.
    """

    composer = _composer(outbound_components=(0, 1))
    assert composer.outbound_max_depth == 1
    response = _parameters(composer.site_count)
    dir_in = _directions(composer.inbound_row_count, seed=41, into_site=True)
    dir_out = _directions(composer.outbound_row_count, seed=42)
    tau_in, _, c_in = fx.payload(composer.inbound_row_count, seed=11)
    tau_out, _, c_out = fx.payload(composer.outbound_row_count, seed=12)
    inbound = fx.leg_batch(tau_in.float(), c_in.to(torch.complex64), direction=dir_in)
    outbound = fx.leg_batch(
        tau_out.float(), c_out.to(torch.complex64), direction=dir_out
    )

    with pytest.raises(NotImplementedError, match="DEPARTURE direction"):
        composer.compose(inbound, outbound, response)


def test_the_kernel_refuses_a_host_tensor_instead_of_computing_on_it():
    """The no-fallback negative: there is no CPU route behind this symbol."""

    composer = _composer()
    response = _parameters(composer.site_count)
    dir_in = _directions(composer.inbound_row_count, seed=41, into_site=True).cpu()
    dir_out = _directions(composer.outbound_row_count, seed=42)
    flags = torch.ones(composer.path_count, dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError, match="must be a CUDA tensor"):
        _launch(composer, response, dir_in, dir_out, flags)


def test_the_response_refuses_a_non_unit_axis():
    """No silent renormalisation: the gradient would be of another parameter."""

    with pytest.raises(ValueError, match="unit vectors"):
        AspectScatterResponse.from_values(
            [(2.0, 0.0, 0.0)],
            [1.0],
            [0.0],
            exponent=EXPONENT,
            coherent_interval_s=COHERENT_INTERVAL_S,
        )


def test_the_aspect_rate_guard_refuses_a_fast_aspect_change():
    """A named refusal at freeze time, not a silent approximation.

    ``|d(arg S)/dt| * T_frame`` must stay below ``ASPECT_PHASE_BUDGET_RAD``.
    The join publishes ``tan_rate_rt = 0`` and carries the whole rate in
    ``tau_rt``, so an aspect phase that walks is simply dropped, and no output
    reports that it was.
    """

    from witwin.radar.synthesis.contracts import ASPECT_PHASE_BUDGET_RAD

    over = 1.01 * ASPECT_PHASE_BUDGET_RAD / COHERENT_INTERVAL_S
    with pytest.raises(ValueError, match="unmodelled aspect Doppler"):
        AspectScatterResponse.from_values(
            [(1.0, 0.0, 0.0)],
            [1.0],
            [0.0],
            exponent=EXPONENT,
            coherent_interval_s=COHERENT_INTERVAL_S,
            aspect_phase_rate_rad_per_s=over,
        )
    # And the boundary is where it says it is: just inside is accepted.
    under = 0.99 * ASPECT_PHASE_BUDGET_RAD / COHERENT_INTERVAL_S
    AspectScatterResponse.from_values(
        [(1.0, 0.0, 0.0)],
        [1.0],
        [0.0],
        exponent=EXPONENT,
        coherent_interval_s=COHERENT_INTERVAL_S,
        aspect_phase_rate_rad_per_s=under,
    )
