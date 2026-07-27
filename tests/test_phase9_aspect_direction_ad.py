"""``field_direction`` reaches a Radar endpoint leaf, through the aspect response.

Phase 7 shipped ``AspectScatterResponse`` with a native JVP and a native
backward that both compute ``grad_dir_in`` / ``grad_dir_out``, and the module
docstring claimed that "a gradient taken here reaches the endpoint positions".
Until ADR-043 that claim was FALSE from end to end: Channel marked
``PropagationGeometry.field_direction`` non-differentiable in both branches of
both field-transport setup contexts, so the two direction cotangents the kernel
produced were handed to an autograd graph that had nothing to give them to, and
a forward tangent never arrived at all. Nothing failed. The Phase-9 survey
measured it: ``field_direction.requires_grad`` was ``False`` and its forward
tangent ``None`` while every other geometry column was live.

ADR-043 (Channel ``CONTRACT_VERSION`` 6) seeds ``grad_direction`` into the
free-space and reflection backward kernels and publishes ``direction.d`` from
their JVP kernels, for ``{los, reflection}`` under a frozen topology. This
module is the Radar half of that claim, executed rather than asserted.

**How the fixture isolates the direction term.** The aspect law is

    ``S = amplitude * max(-dot(dir_in, axis), 0)^n``
    ``    * max(dot(dir_out, axis), 0)^n * exp(-i * phase_rad)``

and its ONLY geometry inputs are the two legs' ``field_direction`` columns. So
a test that differentiates ``S`` alone with respect to an endpoint position has
exactly one route from the leaf to the loss, and a detached ``field_direction``
makes that gradient EXACTLY zero rather than merely small. That falsifier is
:func:`test_the_aspect_gradient_is_exactly_zero_when_field_direction_is_detached`
and it is what makes the finite differences below mean "the direction companion
is right" instead of "some derivative arrived".

**Why the outbound leg is frozen line of sight only.** A leg publishes its FINAL
segment's direction. For the outbound leg that is the arrival direction at the
receiver, which equals the departure direction from the site only at depth
zero, and ``AspectScatterResponse`` refuses anything deeper by name. The inbound
leg keeps its reflection rows, so a REFLECTION ``field_direction`` is what
drives the response on one of the two live rows - the half of ADR-043 that was
the risk item, since the reflection cotangent splits over the final segment and
over the last bounce's reflected direction.

**Why the two live rows are attributable.** One aspect axis per site cannot
illuminate both a line-of-sight and a reflected arrival at the same site: the
two arrive from opposite half spaces, so the lobe clamps one of them to zero.
The fixture uses that: site P is aimed at its REFLECTED arrival and site Q at
its LINE-OF-SIGHT arrival, and the receivers are placed so each site has exactly
one live outbound row. Site P's whole gradient is therefore a reflection-row
gradient and site Q's is a line-of-sight-row gradient, with no mixing to hide
behind.

Finite differences are the test oracle and never a production route.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch
import torch.autograd.forward_ad as forward_ad

pytest.importorskip("witwin.channel")

from support import fd  # noqa: E402
from support import multi_endpoint_driver as drv  # noqa: E402
from support.synthesis_batch import to_synthesis  # noqa: E402

from witwin.radar.scattering import AspectScatterResponse  # noqa: E402
from witwin.radar.synthesis import synthesize_fmcw_beat  # noqa: E402


pytestmark = pytest.mark.gpu


#: ``RX_C`` sits at ``(6, 3, 0)``: on the FAR side of the wall plane from the
#: sites, so the outbound line from site P leaves along ``+x`` and back-scatters
#: the reflected arrival. Its line crosses ``x = 4`` at ``y = 1.8``, well past
#: the facet half width of 1.2, so it is not occluded. ``RX_A`` is the fixture's
#: own receiver and back-scatters site Q's line-of-sight arrival.
RECEIVERS = ((32, (6.0, 3.0, 0.0)), (30, (0.15, 0.0, 0.0)))

#: 15 degrees off the exact back-scatter direction of each site's live arrival.
#: Exactly on it would put the lobe at its peak, where the derivative with
#: respect to a UNIT direction is zero by construction and a finite difference
#: would be comparing two zeros.
_AXIS_RAW = (
    (0.9659258, 0.2588190, 0.0),
    (-0.4188792, -0.9080614, 0.0),
)

ASPECT_EXPONENT = 2.0
COHERENT_INTERVAL_S = 1.0e-3

#: The composed row whose INBOUND leg is the reflection row, and the one whose
#: inbound leg is line of sight. Asserted, not assumed, by
#: :func:`test_the_fixture_puts_one_reflection_row_and_one_line_of_sight_row_in_the_lobe`.
REFLECTION_ROW = 1
LINE_OF_SIGHT_ROW = 5

#: Metres. Measured, not guessed: across steps of 5e-5, 1e-4, 5e-4, 1e-3, 5e-3
#: and 1e-2 the worst component's agreement with the analytic gradient is
#: 6.1e-2, 2.7e-2, 1.3e-2, 8.8e-4, 1.8e-4 and 3.5e-4. Below 1e-3 the float32
#: primal noise dominates; above 1e-2 the lobe's curvature does.
ASPECT_STEP_M = 5.0e-3

#: The realized agreement at that step is 1.8e-4; a factor of ~30 of margin.
ASPECT_FD_RTOL = 5.0e-3

#: The synthesized-cube chain runs the same gradient through the join, the beat
#: kernel and a float32 cube whose value is ~2.8e-6, so its smallest gradient
#: component (8.0e-9, three orders below the dominant 3.5e-6) sits close to the
#: float32 cancellation floor. That component is what forces the FOURTH-order
#: stencil here: with the second-order one it reads 15-20% low at 1e-4 and
#: 2e-4 m. Fourth order, worst component across steps of 1e-4, 2e-4, 4e-4, 8e-4
#: and 1.6e-3 m: 15.2%, 19.9%, 1.07%, 1.07% and 1.37%.
CUBE_STEP_M = 8.0e-4
CUBE_FD_RTOL = 3.0e-2

#: Below this a component is structurally zero rather than a small number.
ZERO_FLOOR = 1.0e-9


# --------------------------------------------------------------------------
# Fixture
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def spike():
    """2 TX x 2 sites x 2 RX, inbound multipath, outbound line of sight."""

    return drv.MultiEndpointSpike(
        receivers=RECEIVERS,
        outbound_components=frozenset({"los"}),
        outbound_max_depth=0,
    )


def _axis(device: str = "cuda") -> torch.Tensor:
    raw = torch.tensor(_AXIS_RAW, dtype=torch.float64)
    unit = raw / torch.linalg.vector_norm(raw, dim=1, keepdim=True)
    return unit.to(dtype=torch.float32, device=device)


def _response(device: str = "cuda") -> AspectScatterResponse:
    return AspectScatterResponse(
        axis=_axis(device),
        amplitude=torch.tensor([1.0e5, 0.8e5], dtype=torch.float32, device=device),
        phase_rad=torch.tensor([0.7, -0.3], dtype=torch.float32, device=device),
        exponent=ASPECT_EXPONENT,
        coherent_interval_s=COHERENT_INTERVAL_S,
    )


def _flags(spike) -> torch.Tensor:
    return torch.ones(spike.composer.path_count, dtype=torch.int32, device="cuda")


def _cotangent(rows: int, *, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A deterministic, sign-mixed loss weight. Never all ones."""

    generator = torch.Generator().manual_seed(seed)
    raw = torch.rand(2, rows, generator=generator, dtype=torch.float32)
    pair = 2.0 * raw - 1.0
    return pair[0].cuda(), pair[1].cuda()


def _aspect_rows(spike, sites=None, transmitters=None, *, ad_mode: str = "none"):
    inbound, outbound = spike.legs(
        sites, transmitters=transmitters, ad_mode=ad_mode
    )
    rows = _response().evaluate_rows(
        spike.composer, inbound, outbound, _flags(spike)
    )
    return rows, inbound, outbound


def _aspect_loss(spike, weights, sites=None, transmitters=None, *, ad_mode="none"):
    (real, imaginary), _, _ = _aspect_rows(
        spike, sites, transmitters, ad_mode=ad_mode
    )
    w_re, w_im = weights
    return (w_re * real + w_im * imaginary).sum()


# --------------------------------------------------------------------------
# 1. The Channel-side claim, observed from Radar
# --------------------------------------------------------------------------


def test_the_frozen_leg_publishes_a_graph_bearing_field_direction(spike):
    """ADR-043, seen through the production adapter rather than the consumer.

    Both AD modes, and the primal is asserted BIT FOR BIT against the
    ``ad_mode='none'`` answer. A contract change that made the direction live by
    perturbing the value it publishes would be a numerical change wearing an AD
    label, and this is what refuses it.
    """

    reference, _ = spike.legs(ad_mode="none")
    assert reference.field_direction is not None
    assert not reference.field_direction.requires_grad

    sites = spike.site_tensor(requires_grad=True)
    inbound, outbound = spike.legs(sites, ad_mode="vjp")
    for name, leg in (("inbound", inbound), ("outbound", outbound)):
        assert leg.field_direction is not None, name
        assert leg.field_direction.requires_grad, name
        assert leg.field_direction.grad_fn is not None, name
    assert torch.equal(inbound.field_direction, reference.field_direction)

    direction = torch.ones_like(spike.site_tensor())
    with forward_ad.dual_level():
        dual = forward_ad.make_dual(spike.site_tensor(), direction)
        inbound, outbound = spike.legs(dual, ad_mode="jvp")
        for name, leg in (("inbound", inbound), ("outbound", outbound)):
            tangent = forward_ad.unpack_dual(leg.field_direction).tangent
            assert tangent is not None, name
            assert float(tangent.abs().max()) > 0.0, name
        primal = forward_ad.unpack_dual(inbound.field_direction).primal
        assert torch.equal(primal, reference.field_direction)


def test_the_adapter_only_ever_freezes_direction_differentiable_components(spike):
    """The invariant that makes a silent zero unreachable from Radar.

    The adapter narrows every frozen topology to
    ``fixed_topology_components``. ADR-043 decides ``field_direction`` liveness
    ONCE for the whole result from ``direction_differentiable_components``, so a
    request that mixed in an undeclared component would publish a fully detached
    direction - a complete answer by contract, and a silent zero for this
    module. It cannot happen while this subset relation holds, and if Channel
    ever narrows the direction set this fails instead of the aspect gradient
    quietly going to zero.
    """

    from witwin.channel.propagation import consumer

    capabilities = consumer.capabilities()
    assert capabilities.fixed_topology_components.issubset(
        capabilities.direction_differentiable_components
    )
    assert frozenset({"los", "reflection"}).issubset(
        capabilities.direction_differentiable_components
    )


# --------------------------------------------------------------------------
# 2. The fixture premise
# --------------------------------------------------------------------------


def test_the_fixture_puts_one_reflection_row_and_one_line_of_sight_row_in_the_lobe(
    spike,
):
    """Exactly two live composed rows, and they carry different components.

    Everything below attributes site P's gradient to a reflection arrival and
    site Q's to a line-of-sight one. That attribution is only legitimate if the
    OTHER four rows really are identically zero, so it is measured here first.
    """

    (real, imaginary), inbound, outbound = _aspect_rows(spike)
    assert spike.composer.outbound_max_depth == 0
    assert spike.composer.path_count == 6

    magnitude = torch.sqrt(real * real + imaginary * imaginary)
    live = torch.nonzero(magnitude > 0.0).flatten().tolist()
    assert live == [REFLECTION_ROW, LINE_OF_SIGHT_ROW], live

    inbound_row = spike.composer.inbound_row
    depth = inbound.depth
    assert int(depth[int(inbound_row[REFLECTION_ROW])]) == 1
    assert int(depth[int(inbound_row[LINE_OF_SIGHT_ROW])]) == 0
    assert int(outbound.depth.max()) == 0

    # And the two live rows belong to different sites, so a per-site gradient
    # separates them without any masking.
    slots = spike.composer.response_slot
    assert int(slots[REFLECTION_ROW]) != int(slots[LINE_OF_SIGHT_ROW])


# --------------------------------------------------------------------------
# 3. Reverse mode, against finite differences
# --------------------------------------------------------------------------


def test_a_reverse_aspect_gradient_reaches_the_site_positions(spike):
    """``d(S)/d(site)``, whose only route is ``field_direction``."""

    weights = _cotangent(spike.composer.path_count, seed=901)
    sites = spike.site_tensor(requires_grad=True)
    _aspect_loss(spike, weights, sites, ad_mode="vjp").backward()

    gradient = sites.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    # Both sites move the loss: the reflection row and the line-of-sight row
    # each carry a real derivative.
    assert float(gradient[0, :2].abs().min()) > ZERO_FLOOR
    assert float(gradient[1, :2].abs().min()) > ZERO_FLOOR
    # z is structurally zero: every endpoint and the wall are coplanar in z,
    # and the lobe axes have no z component.
    torch.testing.assert_close(
        gradient[:, 2], torch.zeros_like(gradient[:, 2]), rtol=0.0, atol=0.0
    )

    base = spike.site_tensor()
    for site in range(base.shape[0]):
        for axis in range(2):
            plus = base.clone()
            minus = base.clone()
            plus[site, axis] += ASPECT_STEP_M
            minus[site, axis] -= ASPECT_STEP_M
            realized = float(plus[site, axis] - minus[site, axis]) / 2.0
            measured = (
                float(_aspect_loss(spike, weights, plus))
                - float(_aspect_loss(spike, weights, minus))
            ) / (2.0 * realized)
            expected = float(gradient[site, axis])
            assert abs(measured - expected) <= ASPECT_FD_RTOL * max(
                abs(expected), ZERO_FLOOR
            ), (site, axis, measured, expected)


def test_the_aspect_gradient_is_exactly_zero_when_field_direction_is_detached(
    spike,
):
    """The falsifier the whole module rests on.

    ``evaluate_rows`` reads the two legs' ``field_direction`` and nothing else
    that could carry a leaf. Detaching that one column must therefore take the
    gradient to EXACTLY zero - not small, zero - which is precisely the state
    the tree was in before ADR-043 and precisely what a regression would look
    like.
    """

    weights = _cotangent(spike.composer.path_count, seed=901)
    sites = spike.site_tensor(requires_grad=True)
    inbound, outbound = spike.legs(sites, ad_mode="vjp")
    severed = tuple(
        dataclasses.replace(leg, field_direction=leg.field_direction.detach())
        for leg in (inbound, outbound)
    )
    real, imaginary = _response().evaluate_rows(
        spike.composer, severed[0], severed[1], _flags(spike)
    )
    w_re, w_im = weights
    loss = (w_re * real + w_im * imaginary).sum()

    assert not loss.requires_grad
    assert sites.grad is None


def test_a_reverse_aspect_gradient_reaches_the_transmitter_through_a_reflection_row(
    spike,
):
    """The reflection half of ADR-043, on its own.

    The loss is a one-hot on the reflection row, and the leaf is the
    TRANSMITTER - which enters that row's published direction only through the
    specular point on the wall. A free-space direction derivative cannot produce
    this number, so it is the reflection cotangent split (over the final segment
    AND over the last bounce's reflected direction) that is being measured.
    """

    rows = spike.composer.path_count
    w_re = torch.zeros(rows, dtype=torch.float32, device="cuda")
    w_im = torch.zeros(rows, dtype=torch.float32, device="cuda")
    w_re[REFLECTION_ROW] = 1.0
    w_im[REFLECTION_ROW] = -0.5
    weights = (w_re, w_im)

    transmitters = spike.transmitter_tensor().requires_grad_(True)
    _aspect_loss(
        spike, weights, transmitters=transmitters, ad_mode="vjp"
    ).backward()

    gradient = transmitters.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    # TX_A owns the reflection row; TX_B publishes no rows at all, so its
    # gradient is exactly zero and that is the correct complete answer.
    assert float(gradient[0, :2].abs().min()) > ZERO_FLOOR
    torch.testing.assert_close(
        gradient[1], torch.zeros_like(gradient[1]), rtol=0.0, atol=0.0
    )

    base = spike.transmitter_tensor()
    for axis in range(2):
        plus = base.clone()
        minus = base.clone()
        plus[0, axis] += ASPECT_STEP_M
        minus[0, axis] -= ASPECT_STEP_M
        realized = float(plus[0, axis] - minus[0, axis]) / 2.0
        measured = (
            float(_aspect_loss(spike, weights, transmitters=plus))
            - float(_aspect_loss(spike, weights, transmitters=minus))
        ) / (2.0 * realized)
        expected = float(gradient[0, axis])
        assert abs(measured - expected) <= ASPECT_FD_RTOL * max(
            abs(expected), ZERO_FLOOR
        ), (axis, measured, expected)


# --------------------------------------------------------------------------
# 4. Forward mode
# --------------------------------------------------------------------------


def test_a_forward_tangent_on_the_sites_reaches_the_aspect_response(spike):
    """The same derivative three ways: JVP, VJP projected, and a difference.

    An ADR-038 forward-only dual: no ``requires_grad`` on the dual primal, so a
    facade that short-circuited autograd on ``requires_grad`` would publish a
    tangent-free result and this would catch it.
    """

    weights = _cotangent(spike.composer.path_count, seed=901)
    direction = torch.tensor(
        [[0.3, -0.9, 0.0], [0.7, 0.4, 0.0]], dtype=torch.float32, device="cuda"
    )

    base = spike.site_tensor()
    with forward_ad.dual_level():
        dual = forward_ad.make_dual(base.clone(), direction)
        assert not dual.requires_grad
        value = _aspect_loss(spike, weights, dual, ad_mode="jvp")
        tangent = forward_ad.unpack_dual(value).tangent
        assert tangent is not None
        forward = float(tangent)

    sites = spike.site_tensor(requires_grad=True)
    _aspect_loss(spike, weights, sites, ad_mode="vjp").backward()
    projected = float((sites.grad * direction).sum())

    differenced = (
        float(_aspect_loss(spike, weights, base + ASPECT_STEP_M * direction))
        - float(_aspect_loss(spike, weights, base - ASPECT_STEP_M * direction))
    ) / (2.0 * ASPECT_STEP_M)

    assert abs(forward) > ZERO_FLOOR
    assert abs(forward - projected) <= 1.0e-4 * abs(projected)
    assert abs(differenced - forward) <= ASPECT_FD_RTOL * abs(forward)


# --------------------------------------------------------------------------
# 5. All the way to a synthesized cube
# --------------------------------------------------------------------------


def _cube_loss(spike, spec, sites, *, ad_mode: str = "none"):
    composed, _, _ = spike.frame(
        sites, _response(), ad_mode=ad_mode, include_delay_rate=False
    )
    cube = synthesize_fmcw_beat(to_synthesis(composed), spec)
    return cube.abs().square().sum()


@pytest.fixture(scope="module")
def spec():
    return drv.make_spec(num_chirps=2)


def test_the_aspect_direction_gradient_reaches_a_synthesized_fmcw_loss(spike, spec):
    """Core leaf -> propagation -> direction -> aspect -> join -> IQ -> scalar.

    The whole chain, both modes, against a difference of the production chain
    at perturbed positions. This gradient MIXES the delay term and the direction
    term, which is the point: it is the number an inverse-design caller actually
    reads, and the direction term is a real part of it (see the companion test
    below).
    """

    base = spike.site_tensor()
    sites = base.clone().requires_grad_(True)
    loss = _cube_loss(spike, spec, sites, ad_mode="vjp")
    assert loss.requires_grad
    loss.backward()
    gradient = sites.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert float(gradient.abs().max()) > 0.0

    for site in range(base.shape[0]):
        for axis in range(2):
            samples = {}
            realized = {}
            for offset in (-2, -1, 1, 2):
                moved = base.clone()
                moved[site, axis] += offset * CUBE_STEP_M
                samples[offset] = float(_cube_loss(spike, spec, moved))
                realized[offset] = float(moved[site, axis])
            measured = fd.fourth_order_difference(
                samples, (realized[1] - realized[-1]) / 2.0
            )
            expected = float(gradient[site, axis])
            assert fd.relative_error(
                measured, expected, floor=ZERO_FLOOR
            ) < CUBE_FD_RTOL, (site, axis, measured, expected)

    direction = torch.tensor(
        [[0.3, -0.9, 0.0], [0.7, 0.4, 0.0]], dtype=torch.float32, device="cuda"
    )
    with forward_ad.dual_level():
        dual = forward_ad.make_dual(base.clone(), direction)
        tangent = forward_ad.unpack_dual(
            _cube_loss(spike, spec, dual, ad_mode="jvp")
        ).tangent
        assert tangent is not None
        forward = float(tangent)
    projected = float((gradient * direction).sum())
    assert abs(forward - projected) <= 1.0e-3 * abs(projected)


def test_the_direction_term_is_load_bearing_in_the_synthesized_loss(spike, spec):
    """Severing ``field_direction`` changes the cube gradient materially.

    Without this the previous test would still pass if the direction companion
    published nothing, because the delay term alone reproduces its own finite
    difference. Measured: severing moves the dominant component by 38% of its
    own magnitude, against an FD tolerance of 3%. The floor below is 20%, which
    is a factor of ~7 above the tolerance and a factor of ~2 below the measured
    effect.
    """

    base = spike.site_tensor()

    def gradient(*, sever: bool) -> torch.Tensor:
        sites = base.clone().requires_grad_(True)
        inbound, outbound = spike.legs(sites, ad_mode="vjp")
        if sever:
            inbound, outbound = (
                dataclasses.replace(
                    leg, field_direction=leg.field_direction.detach()
                )
                for leg in (inbound, outbound)
            )
        composed = spike.composer.compose(
            inbound, outbound, _response(), include_delay_rate=False
        )
        cube = synthesize_fmcw_beat(to_synthesis(composed), spec)
        cube.abs().square().sum().backward()
        return sites.grad

    live = gradient(sever=False)
    severed = gradient(sever=True)
    scale = float(live.abs().max())
    assert scale > 0.0
    assert float((live - severed).abs().max()) / scale > 0.20
