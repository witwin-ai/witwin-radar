"""Every supported leaf, one backward, one frozen topology, three waveforms.

Phase 9's acceptance criterion is that "primal, jvp and vjp share compact path
identity, row mapping and numerical convention". A test that builds a fresh
topology per AD mode proves nothing about row mapping: it compares two answers
to two questions. So every group below drives the SAME
``PreparedFixedTopology`` - the same compiled scene, the same two frozen legs,
the same frozen join - through all three modes, and the scene-leaf group, which
cannot share a compile, asserts topology identity elementwise instead of
assuming it.

Three groups:

**G1, the combined backward.** All eight supported leaves marked at once, in
ONE ``backward()``. Every gradient must be nonzero, and each must equal the
gradient the same leaf gets on its own. That equality is the whole point: a
combined backward that silently drops a leaf, or that writes one leaf's
cotangent into another's slot, reproduces every single-leaf test in the tree.
Measured, the agreement is BITWISE on all eight leaves and all three waveforms,
so the assertion is exact rather than approximate.

**G2, the adjoint identity.** ``<grad, v> == <u, J v>`` for a random cotangent
``u`` over the whole complex cube and a random direction ``v``, on one frozen
topology. The cotangent is a full cube rather than the scalar loss's implicit
one, so this is a genuine adjoint statement about the Jacobian and not a second
reading of the same scalar.

**G3, compact path identity across modes.** Identical ``K``, identical
identity-key columns elementwise, identical row order, bitwise identical
primal, and - the part nothing pinned at this boundary before -
``ad_mode="none"`` publishing no graph and no tangent on any output.

**The finite-difference policy.** The step is 1e-4 with a fourth-order stencil,
chosen from the sweep recorded in ``support/ad_matrix.py``; the tolerance is 5e-2
against a worst measured 0.94 percent. Differences are the test oracle here and
never a production route.
"""

from __future__ import annotations

import contextlib

import pytest
import torch
import torch.autograd.forward_ad as forward_ad

pytest.importorskip("witwin.channel")

from support import ad_matrix as mx  # noqa: E402
from support import fd  # noqa: E402
from support import multi_endpoint_driver as drv  # noqa: E402
from support import waveform_chains as wc  # noqa: E402


pytestmark = pytest.mark.gpu


#: Relative tolerance for the combined difference. Worst measured across the
#: three waveforms at ``FD_STEP``: 0.94 percent analytic-against-all-at-once and
#: 0.71 percent sum-of-singles-against-all-at-once.
COMBINED_FD_RTOL = 5.0e-2

#: The two leaves this stage introduces and no earlier stage validated against a
#: difference of its own. Their steps are far larger than ``FD_STEP`` on
#: purpose: each contributes a small part of the loss, so a step sized for the
#: geometry leaves would difference two float32 numbers that agree to seven
#: digits. Swept in the module report; ``sigma_e`` agrees to 6.9e-5 / 4.2e-4 /
#: 2.4e-3 at 1e-3 and ``phase_rad`` to 6.9e-5 / 1.0e-4 / 4.6e-5 at 5e-3.
SIGMA_E_STEP = 1.0e-3
SIGMA_E_FD_RTOL = 1.0e-2
PHASE_STEP = 5.0e-3
PHASE_FD_RTOL = 5.0e-3

ZERO_FLOOR = 1.0e-30

#: A magnitude-only loss cannot see a global response phase; this bounds how
#: much of it survives as float32 roundoff, measured at 1e-7 of the loss.
PHASE_INVARIANCE_BOUND = 1.0e-6


@pytest.fixture(scope="module")
def spike():
    """ONE compiled scene, ONE pair of frozen legs, ONE frozen join."""

    return drv.MultiEndpointSpike()


@pytest.fixture(scope="module")
def values(spike):
    return mx.base_values(spike)


def _reverse(kind, values, names, *, spike=None):
    """One backward with ``names`` marked; returns the loss and the gradients."""

    live = mx.marked(values, names)
    loss = mx.loss_of(kind, live, ad_mode="vjp", spike=spike)
    loss.backward()
    return float(loss.detach()), {name: live[name].grad.detach() for name in names}


# --------------------------------------------------------------------------
# 0. The premise: one scenario, one topology, three waveforms
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", mx.WAVEFORMS)
def test_the_three_waveforms_share_one_frozen_topology(spike, values, kind):
    """The scenario's shape, asserted before anything is differentiated.

    Eleven composed rows over 2 TX x 2 sites x 2 RX, with ``TX_B`` publishing
    none at all so at least one pair segment is empty and the per-pair widths
    differ. The waveform is chosen after the topology is frozen and cannot move
    it, which is what lets one scenario answer for all three families.
    """

    composed = mx.replay(spike, values)
    assert composed.path_count == 11
    offsets = spike.composer.pair_offsets.tolist()
    widths = [offsets[i + 1] - offsets[i] for i in range(len(offsets) - 1)]
    assert 0 in widths, widths
    assert len(set(widths)) > 1, widths
    cube = wc.synthesize(kind, composed, wc.make_spec(kind))
    assert cube.dtype == torch.complex64
    assert float(cube.abs().max()) > 0.0


def test_the_scene_leaf_scenario_is_the_same_topology_as_the_shared_one(
    spike, values
):
    """A scene leaf forces its own compile; this says it is the same world.

    ``vertices``, ``eps_r`` and ``sigma_e`` are baked into the compiled scene,
    so a scenario that marks them is necessarily a different ``MultiEndpointSpike``
    object with its own ``PreparedFixedTopology``. What makes the combined
    backward below a statement about ONE topology is that the two agree on
    every identity key, on the row order and on the primal - checked here, once,
    rather than assumed by every test that follows.
    """

    shared = mx.replay(spike, values)
    own, own_spike = mx.frame(values)

    assert own.path_count == shared.path_count
    assert drv.composed_keys(own_spike, own) == drv.composed_keys(spike, shared)
    assert torch.equal(own.total_delay_s, shared.total_delay_s)
    assert torch.equal(own.complex_transfer_ref, shared.complex_transfer_ref)
    assert torch.equal(own.row_valid, shared.row_valid)
    assert own_spike.inbound.prepared is not spike.inbound.prepared


# --------------------------------------------------------------------------
# 1. G1 - every supported leaf live in one backward
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", mx.WAVEFORMS)
def test_every_supported_leaf_is_live_in_one_combined_backward(values, kind):
    """Eight leaves, one ``backward()``, and not one of them comes back empty.

    A dropped leaf shows up here as ``grad is None`` or as an exact zero, and
    both are the defect class this phase exists to kill: the surveyor measured
    all eight working and NO test covering them together.
    """

    _, gradients = _reverse(kind, values, mx.LEAF_NAMES)
    for name in mx.LEAF_NAMES:
        gradient = gradients[name]
        assert gradient is not None, name
        assert torch.isfinite(gradient).all(), name
        assert float(gradient.abs().sum()) > 0.0, name


@pytest.mark.parametrize("kind", mx.WAVEFORMS)
def test_each_combined_gradient_equals_its_single_leaf_gradient(values, kind):
    """The combined backward against eight separate ones, BITWISE.

    Exact rather than approximate, and that is a measurement rather than an
    aspiration: the reductions are per-leaf and deterministic, so marking seven
    extra leaves must not perturb the eighth by a single ulp. An approximate
    tolerance here would accept a reduction that had started mixing cotangents
    at the last digit, which is precisely how this defect begins.

    The primal is asserted equal too. Without it the comparison could be
    between two different scenes: every single-leaf run recompiles.
    """

    combined_loss, combined = _reverse(kind, values, mx.LEAF_NAMES)
    for name in mx.LEAF_NAMES:
        single_loss, single = _reverse(kind, values, (name,))
        assert single_loss == combined_loss, name
        assert torch.equal(single[name], combined[name]), (
            name,
            float((single[name] - combined[name]).abs().max()),
        )


@pytest.mark.parametrize("kind", mx.WAVEFORMS)
def test_the_combined_difference_equals_the_sum_of_the_single_leaf_differences(
    values, kind
):
    """Perturb all eight at once against the sum of the eight single differences.

    Two statements in one. The sum-of-singles comparison is a linearity check
    on the difference itself and is insensitive to the stencil's truncation,
    because both sides carry the same truncation. The analytic comparison is
    the one that can fail for a real reason, and it is the reason the tolerance
    is 5 percent rather than tighter.
    """

    _, gradients = _reverse(kind, values, mx.LEAF_NAMES)
    directions = {
        name: mx.direction(name, values[name], gradients[name])
        for name in mx.LEAF_NAMES
    }
    parts = {
        name: float((gradients[name] * directions[name]).sum())
        for name in mx.LEAF_NAMES
    }
    for name, part in parts.items():
        assert abs(part) > 0.0, name
    # Signed by the gradient, so every leaf pushes the loss the same way and
    # the total is the sum of magnitudes rather than a near cancellation.
    assert all(part > 0.0 for part in parts.values()) or all(
        part < 0.0 for part in parts.values()
    ), parts
    analytic = sum(parts.values())

    def difference(active):
        samples = {
            offset: float(
                mx.loss_of(
                    kind,
                    mx.perturbed(values, directions, active, offset, mx.FD_STEP),
                )
            )
            for offset in (-2, -1, 1, 2)
        }
        return fd.fourth_order_difference(samples, mx.FD_STEP)

    singles = sum(difference((name,)) for name in mx.LEAF_NAMES)
    all_at_once = difference(mx.LEAF_NAMES)
    assert (
        fd.relative_error(singles, all_at_once, floor=ZERO_FLOOR)
        < COMBINED_FD_RTOL
    ), (singles, all_at_once)
    assert (
        fd.relative_error(analytic, all_at_once, floor=ZERO_FLOOR)
        < COMBINED_FD_RTOL
    ), (analytic, all_at_once)


@pytest.mark.parametrize("kind", mx.WAVEFORMS)
def test_the_conductivity_gradient_matches_a_central_difference(values, kind):
    """``sigma_e``, the leaf this stage adds, against a difference of its own.

    S1 pinned ``eps_r``. The conductivity enters the same complex permittivity
    through ``sigma_e / (2 pi f eps_0)``, so a chain that reached one and not
    the other would be differentiating half a material and every ``eps_r`` test
    would still pass.
    """

    _, gradients = _reverse(kind, values, ("sigma_e",))
    expected = float(gradients["sigma_e"])
    assert abs(expected) > 0.0

    def at(offset: float) -> float:
        moved = dict(values)
        moved["sigma_e"] = values["sigma_e"] + offset * SIGMA_E_STEP
        return float(mx.loss_of(kind, moved))

    measured = fd.fourth_order_difference(
        {offset: at(offset) for offset in (-2, -1, 1, 2)}, SIGMA_E_STEP
    )
    assert (
        fd.relative_error(measured, expected, floor=ZERO_FLOOR) < SIGMA_E_FD_RTOL
    ), (measured, expected)


@pytest.mark.parametrize("kind", mx.WAVEFORMS)
def test_the_response_phase_gradient_matches_a_central_difference(
    spike, values, kind
):
    """``phase_rad`` through the conjugation boundary, against a difference.

    The FMCW beat cube is CONJUGATED relative to Channel's convention and the
    other two are not, so a phase gradient that lost the conjugation would come
    back with the wrong sign on exactly one of the three waveforms. This is
    parametrized over all three for that reason.
    """

    _, gradients = _reverse(kind, values, ("phase_rad",), spike=spike)
    expected = float(gradients["phase_rad"])
    assert abs(expected) > 0.0

    def at(offset: float) -> float:
        moved = dict(values)
        moved["phase_rad"] = values["phase_rad"] + offset * PHASE_STEP
        return float(mx.loss_of(kind, moved, spike=spike))

    measured = fd.fourth_order_difference(
        {offset: at(offset) for offset in (-2, -1, 1, 2)}, PHASE_STEP
    )
    assert (
        fd.relative_error(measured, expected, floor=ZERO_FLOOR) < PHASE_FD_RTOL
    ), (measured, expected)


@pytest.mark.parametrize("kind", mx.WAVEFORMS)
def test_a_magnitude_only_loss_cannot_see_the_response_phase(spike, values, kind):
    """Why the scenario's loss has a second term, asserted rather than asserted about.

    ONE ``ScalarRcsResponse`` multiplies every composed row, so its phase is a
    GLOBAL rotation and ``sum |cube|^2`` is exactly invariant under it. The
    derivative that survives is float32 roundoff, measured at 1e-7 of the loss,
    and a test built on the magnitude loss alone would have been asserting that
    a rounding error was a gradient. ``ad_matrix.combined_loss`` adds
    ``sum Re(cube^2)``, which a global rotation multiplies by ``exp(2 j theta)``.
    """

    live = mx.marked(values, ("phase_rad",))
    cube = mx.cube_of(kind, live, ad_mode="vjp", spike=spike)
    magnitude = cube.abs().square().sum()
    (magnitude_gradient,) = torch.autograd.grad(
        magnitude, live["phase_rad"], retain_graph=True
    )
    assert abs(float(magnitude_gradient)) < PHASE_INVARIANCE_BOUND * abs(
        float(magnitude.detach())
    )

    both = mx.combined_loss(cube)
    (combined_gradient,) = torch.autograd.grad(both, live["phase_rad"])
    assert abs(float(combined_gradient)) > 100.0 * abs(float(magnitude_gradient))


# --------------------------------------------------------------------------
# 2. G2 - the adjoint identity, on one PreparedFixedTopology
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cotangent():
    """A deterministic sign-mixed cotangent over the complex cube, never ones.

    All-ones would be blind to a per-element permutation, and a real-only one
    would be blind to a conjugation error in the imaginary half.
    """

    generator = torch.Generator().manual_seed(9091)

    def build(shape):
        raw = torch.rand(2, *shape, generator=generator, dtype=torch.float32)
        return (2.0 * raw[0] - 1.0).cuda(), (2.0 * raw[1] - 1.0).cuda()

    return build


@pytest.mark.parametrize("kind", mx.WAVEFORMS)
def test_the_jvp_is_the_adjoint_of_the_vjp_on_one_frozen_topology(
    spike, values, cotangent, kind
):
    """``<grad, v> == <u, J v>`` over the whole cube, one topology, one order.

    The two passes go through the SAME ``MultiEndpointSpike``, so the same
    ``PreparedFixedTopology``, the same frozen legs and the same frozen join -
    asserted by object identity below rather than left to inference. The
    cotangent covers the whole complex cube, so this measures the Jacobian
    rather than one scalar functional of it, and a mode that reordered rows or
    flipped a conjugation could not satisfy it.
    """

    prepared = (spike.inbound.prepared, spike.outbound.prepared)

    generator = torch.Generator().manual_seed(613)
    raw = torch.rand(
        values["sites"].shape, generator=generator, dtype=torch.float32
    )
    tangent = (2.0 * raw - 1.0).cuda()
    # The out-of-plane column is structurally dead in this fixture, so a
    # direction with one would only dilute the identity.
    tangent[:, 2] = 0.0

    live = mx.marked(values, ("sites",))
    cube = mx.cube_of(kind, live, ad_mode="vjp", spike=spike)
    u_re, u_im = cotangent(tuple(cube.shape))
    (cube.real * u_re + cube.imag * u_im).sum().backward()
    reverse = float((live["sites"].grad * tangent).sum())

    with forward_ad.dual_level():
        duals = dict(values)
        for name in ("sites", "transmitters", "receivers"):
            seed = tangent if name == "sites" else torch.zeros_like(values[name])
            duals[name] = forward_ad.make_dual(values[name].clone(), seed)
        forward_cube = mx.cube_of(kind, duals, ad_mode="jvp", spike=spike)
        jacobian = forward_ad.unpack_dual(forward_cube).tangent
        assert jacobian is not None
        forward = float((jacobian.real * u_re + jacobian.imag * u_im).sum())

    assert (spike.inbound.prepared, spike.outbound.prepared) == prepared
    assert abs(reverse) > 0.0
    assert fd.relative_error(forward, reverse, floor=ZERO_FLOOR) < 1.0e-5, (
        forward,
        reverse,
    )


# --------------------------------------------------------------------------
# 3. G3 - compact path identity across none / jvp / vjp
# --------------------------------------------------------------------------


#: Every column of ``RadarPathTopology``. These are the five the join builds its
#: canonical order from and the five a caller traces a row back through, so
#: comparing all of them elementwise is the complete row-mapping statement.
IDENTITY_COLUMNS = (
    "radar_source_id",
    "site_id",
    "radar_sink_id",
    "inbound_row",
    "outbound_row",
)


@contextlib.contextmanager
def _modes(spike, values):
    """One composed result per AD mode, over the same frozen topology.

    The forward-mode result is produced INSIDE a ``dual_level`` and the block
    stays open for the caller, because a dual that outlives its level loses its
    tangent and the comparison would then be against a primal-only answer that
    happened to look right.
    """

    results = {
        "none": mx.replay(spike, values, ad_mode="none"),
        "vjp": mx.replay(spike, mx.marked(values, ("sites",)), ad_mode="vjp"),
    }
    with forward_ad.dual_level():
        duals = dict(values)
        # Sites only. Seeding all three endpoint sets with ones is a RIGID
        # TRANSLATION and publishes an exactly zero tangent on every
        # line-of-sight row, which would make the forward result agree with the
        # others for the wrong reason.
        for name in ("sites", "transmitters", "receivers"):
            seed = (
                torch.ones_like(values[name])
                if name == "sites"
                else torch.zeros_like(values[name])
            )
            duals[name] = forward_ad.make_dual(values[name].clone(), seed)
        results["jvp"] = mx.replay(spike, duals, ad_mode="jvp")
        yield results


@pytest.mark.parametrize("kind", mx.WAVEFORMS)
def test_the_three_ad_modes_publish_the_same_compact_rows(spike, values, kind):
    """Identical ``K``, identical identity keys, identical row ORDER.

    The identity-key columns are the ones ``test_phase6_identity_key_columns.py``
    proved the join's order is built from, read here off the composed topology
    rather than rebuilt. Comparing them elementwise is what makes this a
    row-mapping statement instead of a row-count one: two results can agree on
    ``K`` and on the multiset of keys and still be permutations of each other.
    """

    with _modes(spike, values) as results:
        assert {name: r.path_count for name, r in results.items()} == {
            name: 11 for name in results
        }
        keys = {
            name: drv.composed_keys(spike, result)
            for name, result in results.items()
        }
        assert keys["none"] == keys["vjp"] == keys["jvp"]
        for column in IDENTITY_COLUMNS:
            reference = getattr(results["none"].topology, column)
            for name in ("vjp", "jvp"):
                measured = getattr(results[name].topology, column)
                assert torch.equal(measured, reference), (column, name)
        for name, result in results.items():
            assert torch.equal(result.row_valid, results["none"].row_valid), name


@pytest.mark.parametrize("kind", mx.WAVEFORMS)
def test_the_primal_is_bitwise_identical_in_all_three_ad_modes(
    spike, values, kind
):
    """The numerical-convention half: same rows, same numbers, to the last bit.

    Exact, not close. An AD mode is a request for extra outputs, never a
    different computation of the primal, so anything short of bitwise means one
    of the three modes took a different path through the kernel - which is
    exactly the situation in which a gradient can be right about the wrong
    function.
    """

    with _modes(spike, values) as results:
        spec = wc.make_spec(kind)
        primal = {}
        for name, result in results.items():
            cube = wc.synthesize(kind, result, spec)
            primal[name] = (
                forward_ad.unpack_dual(cube).primal if name == "jvp" else cube
            )
            delay = result.total_delay_s
            transfer = result.complex_transfer_ref
            if name == "jvp":
                delay = forward_ad.unpack_dual(delay).primal
                transfer = forward_ad.unpack_dual(transfer).primal
            assert torch.equal(
                delay.detach(), results["none"].total_delay_s
            ), name
            assert torch.equal(
                transfer.detach(), results["none"].complex_transfer_ref
            ), name
        for name in ("vjp", "jvp"):
            assert torch.equal(primal[name].detach(), primal["none"]), name


@pytest.mark.parametrize("kind", mx.WAVEFORMS)
def test_ad_mode_none_publishes_no_graph_and_no_tangent(spike, values, kind):
    """The zero-overhead primal contract, at the Radar boundary.

    Pinned today only at the Channel solver boundary and never at this one. It
    is asserted INSIDE a live ``dual_level`` as well, because the failure it
    guards against is a stage that captures a tangent from an ambient level
    rather than from its own inputs - and outside a dual level there is no
    tangent to capture and the test would pass vacuously.
    """

    composed = mx.replay(spike, values, ad_mode="none")
    cube = wc.synthesize(kind, composed, wc.make_spec(kind))
    published = (
        composed.total_delay_s,
        composed.complex_transfer_ref,
        composed.row_valid,
        cube,
    )
    for tensor in published:
        assert not tensor.requires_grad
        assert tensor.grad_fn is None
        assert forward_ad.unpack_dual(tensor).tangent is None
    assert composed.delay_rate is None

    with forward_ad.dual_level():
        inside = mx.replay(spike, values, ad_mode="none")
        inside_cube = wc.synthesize(kind, inside, wc.make_spec(kind))
        for tensor in (
            inside.total_delay_s,
            inside.complex_transfer_ref,
            inside_cube,
        ):
            assert forward_ad.unpack_dual(tensor).tangent is None
            assert not tensor.requires_grad
        assert torch.equal(inside_cube, cube)
