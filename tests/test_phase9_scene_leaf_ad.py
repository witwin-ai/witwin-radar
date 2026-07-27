"""Mesh vertices and material permittivity, all the way to a synthesized cube.

The Phase-9 survey's single biggest coverage hole. Channel's fixed-topology
reflection route has supported a differentiable mesh and differentiable compiled
material fields since Stage I, and Radar has consumed that route since Phase 5,
but no Radar test ever marked a scene leaf: every end-to-end AD test drives
endpoint positions and the RCS scalar. Both cells worked when the surveyor
probed them and neither was pinned, so a regression anywhere between
``witwin.core.Mesh``, the Channel compile, the reflection reevaluation and the
join would have been invisible.

These are the two leaves an inverse-design caller reaches for after the
endpoints: WHERE a wall is and WHAT it is made of.

**The fixture.** The multi-endpoint world - 2 TX x 2 sites x 2 RX, a narrow
concrete facet at ``x = 4``, one transmitter that publishes zero rows and
therefore empty pair segments, and per-pair row counts that differ. A scene leaf
has to survive all of that, and a single-pair fixture cannot show it.

**Recompiling is the finite difference.** A vertex tensor and a permittivity are
baked into the compiled scene, so a differenced sample is a fresh compile, a
fresh discovery and a fresh frame - the whole production chain, not a replay.
That is what makes these differences an oracle rather than a self-comparison. It
costs about 0.15 s per sample.

**Why the fourth-order stencil and why the step window is narrow.** The loss is
``|IQ|^2`` summed over a cube whose rows interfere, at a 77 GHz reference. A
wall displacement of ``dx`` moves the reflected path by ``2 dx``, which is a
full phase turn every 1.95 mm. Measured relative agreement of the fourth-order
difference with the analytic directional derivative:

    step (m)   1e-4     2e-4     5e-4     1e-3
    plane      0.69%    0.51%    6.8%     71%
    tilt       3.4%     2.5%     0.23%    16%

and for the permittivity, at steps of 2e-4, 5e-4, 1e-3 and 5e-3: 0.17%, 0.04%,
0.07% and 0.10%. The vertex window is ``[1e-4, 5e-4]`` and 2e-4 sits in it for
both directions; the permittivity is smooth and its step barely matters.

Finite differences are the test oracle and never a production route.
"""

from __future__ import annotations

import pytest
import torch
import torch.autograd.forward_ad as forward_ad

pytest.importorskip("witwin.channel")

from support import fd  # noqa: E402
from support import multi_endpoint_driver as drv  # noqa: E402
from support import multi_endpoint_geometry as geo  # noqa: E402
from support import multi_endpoint_world as world  # noqa: E402
from support.synthesis_batch import to_synthesis  # noqa: E402

from witwin.radar.synthesis import synthesize_fmcw_beat  # noqa: E402


pytestmark = pytest.mark.gpu


BASE_VERTICES = torch.tensor(geo.WALL_VERTICES_M, dtype=torch.float32)
BASE_EPS_R = float(geo.WALL_EPS_R)

#: Translate the whole facet along its normal. Every vertex moves, so this is
#: the plane displacement the reflection geometry actually depends on.
PLANE_DIRECTION = torch.tensor(
    [[1.0, 0.0, 0.0]] * 4, dtype=torch.float32
)

#: Tilt: three vertices move along the normal by different amounts, so the two
#: triangles stop being coplanar and each specular point moves differently. The
#: plane direction alone could be reproduced by a model that only knew the
#: wall's offset.
TILT_DIRECTION = torch.tensor(
    [[1.0, 0.0, 0.0], [-0.5, 0.0, 0.0], [0.3, 0.0, 0.0], [0.0, 0.0, 0.0]],
    dtype=torch.float32,
)

#: Metres, and the permittivity is dimensionless. See the module docstring.
VERTEX_STEP_M = 2.0e-4
VERTEX_FD_RTOL = 5.0e-2
EPS_R_STEP = 1.0e-3
EPS_R_FD_RTOL = 5.0e-3

#: The combined perturbation is chosen so its three parts ADD rather than
#: cancel: with the site direction reversed the vertex and site terms are
#: 5.6e-4 and -4.4e-4 and their sum is a 4x amplification of each one's noise.
#: Measured at 2e-4: sum-of-parts against all-at-once 8.4e-5 relative, analytic
#: against all-at-once 0.27%.
SITE_DIRECTION = torch.tensor(
    [[-0.4, 0.8, 0.0], [0.6, -0.3, 0.0]], dtype=torch.float32, device="cuda"
)
COMBINED_STEP = 2.0e-4
COMBINED_FD_RTOL = 2.0e-2

ZERO_FLOOR = 1.0e-12


@pytest.fixture(scope="module")
def spec():
    return drv.make_spec(num_chirps=2)


def _chain(spec, *, vertices=None, eps_r=None, sites=None, ad_mode="none"):
    """compile -> freeze -> reevaluate -> compose -> synthesize -> scalar."""

    compiled = world.compile_fixture_scene(vertices=vertices, eps_r=eps_r)
    spike = drv.MultiEndpointSpike(compiled=compiled)
    composed, _, _ = spike.frame(
        sites, drv.make_response(), ad_mode=ad_mode, include_delay_rate=False
    )
    cube = synthesize_fmcw_beat(to_synthesis(composed), spec)
    return cube.abs().square().sum(), spike


def _value(spec, **kwargs) -> float:
    return float(_chain(spec, **kwargs)[0])


@pytest.fixture(scope="module")
def site_base(spec):
    return _chain(spec)[1].site_tensor()


# --------------------------------------------------------------------------
# 1. The fixture premise
# --------------------------------------------------------------------------


def test_the_scene_leaf_fixture_keeps_its_multi_pair_shape(spec):
    """A live vertex tensor did not turn the fixture into a single-pair one.

    ``make_scene`` now passes a caller-supplied vertex tensor through untouched,
    and ``Mesh`` defaults ``recenter=True``, which would rewrite the authored
    world coordinates and quietly move the wall plane. The elementwise check in
    ``assert_world_coordinates_survived`` is what refuses that; this asserts the
    fixture's own shape survived alongside it.
    """

    vertices = BASE_VERTICES.clone().requires_grad_(True)
    _, spike = _chain(spec, vertices=vertices)
    assert spike.composer.path_count == 11
    # TX_B publishes zero rows, so at least one pair segment is empty.
    offsets = spike.composer.pair_offsets.tolist()
    widths = [offsets[i + 1] - offsets[i] for i in range(len(offsets) - 1)]
    assert 0 in widths, widths
    assert len(set(widths)) > 1, widths


# --------------------------------------------------------------------------
# 2. Mesh vertices
# --------------------------------------------------------------------------


def test_a_mesh_vertex_gradient_reaches_a_synthesized_fmcw_loss(spec):
    """``d(|IQ|^2)/d(wall vertices)``, against two directional differences."""

    vertices = BASE_VERTICES.clone().requires_grad_(True)
    loss, _ = _chain(spec, vertices=vertices, ad_mode="vjp")
    assert loss.requires_grad
    loss.backward()
    gradient = vertices.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert float(gradient[:, 0].abs().min()) > ZERO_FLOOR

    for name, direction in (
        ("plane", PLANE_DIRECTION),
        ("tilt", TILT_DIRECTION),
    ):
        expected = float((gradient * direction).sum())
        samples = {
            offset: _value(
                spec, vertices=BASE_VERTICES + offset * VERTEX_STEP_M * direction
            )
            for offset in (-2, -1, 1, 2)
        }
        measured = fd.fourth_order_difference(samples, VERTEX_STEP_M)
        assert abs(expected) > ZERO_FLOOR, name
        assert fd.relative_error(
            measured, expected, floor=ZERO_FLOOR
        ) < VERTEX_FD_RTOL, (name, measured, expected)


def test_the_in_plane_vertex_gradient_is_exactly_zero_and_that_is_correct(spec):
    """A ZERO cell, pinned as EXACT and shown to be a fact about the physics.

    Every wall vertex has ``x = 4``, so moving a vertex's ``y`` or ``z`` changes
    the facet's EXTENT and leaves its plane alone. A specular reflection off a
    plane depends on the plane; the extent only enters through whether the
    specular point lands inside the facet. Exact zero is therefore the complete
    answer, not a severed wire.

    The falsifier is the second half: narrow the facet until a specular point
    falls outside it and the loss MOVES. Measured, the surviving reflection
    rows' specular points all sit inside ``|y| < 0.5``: at a half width of 0.7
    the composed row count is still 11 and the loss is bit for bit unchanged,
    and at 0.5 it drops to 10 and the loss moves by 1.9%. So the zero is a
    statement about a smooth neighbourhood rather than about a gradient that
    never arrives, and the neighbourhood is 0.2 m wide - four hundred times the
    finite-difference step used above.
    """

    vertices = BASE_VERTICES.clone().requires_grad_(True)
    loss, _ = _chain(spec, vertices=vertices, ad_mode="vjp")
    loss.backward()
    gradient = vertices.grad
    torch.testing.assert_close(
        gradient[:, 1:], torch.zeros_like(gradient[:, 1:]), rtol=0.0, atol=0.0
    )

    baseline = _value(spec, vertices=BASE_VERTICES)

    def narrowed(half_width: float) -> torch.Tensor:
        moved = BASE_VERTICES.clone()
        moved[:, 1] = torch.sign(moved[:, 1]) * half_width
        return moved

    # Still inside the smooth neighbourhood: bit for bit unchanged.
    assert _value(spec, vertices=narrowed(0.7)) == baseline
    # Past it: a row stops existing and the answer moves.
    assert abs(_value(spec, vertices=narrowed(0.5)) - baseline) > 1.0e-3 * abs(
        baseline
    )


def test_a_forward_tangent_on_the_wall_matches_the_reverse_gradient(spec):
    """JVP against the projected VJP, on a scene leaf rather than an endpoint.

    A moved wall moves the reflected delay, so the adapter's dead-tangent guard
    is satisfied by this leaf on its own - unlike the permittivity below.
    """

    vertices = BASE_VERTICES.clone().requires_grad_(True)
    loss, _ = _chain(spec, vertices=vertices, ad_mode="vjp")
    loss.backward()
    projected = float((vertices.grad * PLANE_DIRECTION).sum())

    with forward_ad.dual_level():
        dual = forward_ad.make_dual(BASE_VERTICES.clone(), PLANE_DIRECTION)
        assert not dual.requires_grad
        tangent = forward_ad.unpack_dual(
            _chain(spec, vertices=dual, ad_mode="jvp")[0]
        ).tangent
        assert tangent is not None
        forward = float(tangent)

    assert abs(projected) > ZERO_FLOOR
    assert abs(forward - projected) <= 1.0e-4 * abs(projected)


# --------------------------------------------------------------------------
# 3. Material permittivity
# --------------------------------------------------------------------------


def test_a_material_permittivity_gradient_reaches_a_synthesized_fmcw_loss(spec):
    """``d(|IQ|^2)/d(eps_r)``, the canonical "what is it made of" leaf.

    It reaches the loss only through the reflection rows' Fresnel coefficient,
    so this is a pure material derivative with no geometry term in it: the
    delays do not move at all.
    """

    eps_r = torch.tensor(BASE_EPS_R).requires_grad_(True)
    loss, _ = _chain(spec, eps_r=eps_r, ad_mode="vjp")
    assert loss.requires_grad
    loss.backward()
    expected = float(eps_r.grad)
    assert abs(expected) > ZERO_FLOOR

    samples = {
        offset: _value(spec, eps_r=BASE_EPS_R + offset * EPS_R_STEP)
        for offset in (-2, -1, 1, 2)
    }
    measured = fd.fourth_order_difference(samples, EPS_R_STEP)
    assert fd.relative_error(
        measured, expected, floor=ZERO_FLOOR
    ) < EPS_R_FD_RTOL, (measured, expected)

    # The delays really are untouched: a material leaf moves the coefficient
    # only, which is what makes the forward-mode cell below refuse.
    reference, spike_ref = _chain(spec)
    del reference
    moved, spike_moved = _chain(spec, eps_r=BASE_EPS_R + 1.0)
    del moved
    inbound_ref, _ = spike_ref.legs()
    inbound_moved, _ = spike_moved.legs()
    assert torch.equal(inbound_ref.delay_s, inbound_moved.delay_s)
    assert not torch.equal(inbound_ref.coefficient, inbound_moved.coefficient)


def test_a_material_only_forward_dual_is_refused_by_the_dead_tangent_guard(spec):
    """A REF cell, and an honest one: the guard is the adapter's, not Channel's.

    ``ChannelPropagationAdapter._delay_rate`` requires a ``delay_s`` tangent
    whenever ``ad_mode='jvp'``, because a dead forward tangent publishes
    ``delay_rate = 0``, which is indistinguishable from a correct stationary
    answer. A material-only dual moves the coefficient and not the delay, so it
    trips that guard even though the material tangent itself is live.

    That is a real narrowing of the forward-mode material cell and it is
    recorded rather than worked around: a material forward tangent IS available
    when the same call also carries an endpoint tangent, which the combined test
    below measures. Loosening the guard to accept a coefficient-only tangent is
    a separate decision about what ``delay_rate = None`` means to a caller.
    """

    with forward_ad.dual_level():
        dual = forward_ad.make_dual(
            torch.tensor(BASE_EPS_R), torch.tensor(1.0)
        )
        with pytest.raises(RuntimeError) as raised:
            _chain(spec, eps_r=dual, ad_mode="jvp")
    assert "produced no delay_s tangent" in str(raised.value)


# --------------------------------------------------------------------------
# 4. All three leaves at once
# --------------------------------------------------------------------------


def test_vertices_permittivity_and_endpoints_are_live_in_one_call(
    spec, site_base
):
    """The combined-input test: three leaves at once against three differences.

    A scene leaf, a material leaf and an endpoint leaf reach the same scalar
    through three different parts of the chain - the specular geometry, the
    Fresnel coefficient, and the endpoint positions. Marking one at a time
    cannot catch a reduction that wrote one leaf's cotangent into another's
    slot; perturbing all three at once and comparing against the SUM of the
    single-leaf differences can.
    """

    vertices = BASE_VERTICES.clone().requires_grad_(True)
    eps_r = torch.tensor(BASE_EPS_R).requires_grad_(True)
    sites = site_base.clone().requires_grad_(True)
    loss, _ = _chain(
        spec, vertices=vertices, eps_r=eps_r, sites=sites, ad_mode="vjp"
    )
    loss.backward()
    parts = (
        float((vertices.grad * PLANE_DIRECTION).sum()),
        float(eps_r.grad),
        float((sites.grad * SITE_DIRECTION).sum()),
    )
    combined = sum(parts)
    # Every part is a real contribution and they ADD rather than cancel, so the
    # sum is not one term plus two noise floors.
    for part in parts:
        assert abs(part) > ZERO_FLOOR
    assert abs(combined) > 0.5 * max(abs(part) for part in parts)

    def difference(vertex_on: float, eps_on: float, site_on: float) -> float:
        samples = {}
        for offset in (-2, -1, 1, 2):
            step = offset * COMBINED_STEP
            samples[offset] = _value(
                spec,
                vertices=BASE_VERTICES + step * vertex_on * PLANE_DIRECTION,
                eps_r=BASE_EPS_R + step * eps_on,
                sites=site_base + step * site_on * SITE_DIRECTION,
            )
        return fd.fourth_order_difference(samples, COMBINED_STEP)

    singles = (
        difference(1.0, 0.0, 0.0),
        difference(0.0, 1.0, 0.0),
        difference(0.0, 0.0, 1.0),
    )
    all_at_once = difference(1.0, 1.0, 1.0)
    assert fd.relative_error(
        sum(singles), all_at_once, floor=ZERO_FLOOR
    ) < COMBINED_FD_RTOL
    assert fd.relative_error(
        combined, all_at_once, floor=ZERO_FLOOR
    ) < COMBINED_FD_RTOL

    with forward_ad.dual_level():
        tangent = forward_ad.unpack_dual(
            _chain(
                spec,
                vertices=forward_ad.make_dual(
                    BASE_VERTICES.clone(), PLANE_DIRECTION
                ),
                eps_r=forward_ad.make_dual(
                    torch.tensor(BASE_EPS_R), torch.tensor(1.0)
                ),
                sites=forward_ad.make_dual(site_base.clone(), SITE_DIRECTION),
                ad_mode="jvp",
            )[0]
        ).tangent
        assert tangent is not None
        forward = float(tangent)
    assert abs(forward - combined) <= 1.0e-4 * abs(combined)
