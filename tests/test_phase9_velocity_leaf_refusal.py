"""A velocity is a tangent direction, never a leaf, and now it says so.

Under ADR-038 every velocity in ``witwin.radar.propagation`` goes
into the TANGENT slot of ``torch.autograd.forward_ad.make_dual``. A tangent is
consumed by the forward pass and never differentiated, so ``d(loss)/d(velocity)``
does not exist in either AD mode. It is not zero and it is not small; there is
no such derivative to return.

The Phase-9 survey found that nothing said so. A ``requires_grad`` velocity was
accepted by ``Kinematics``, the whole Doppler chain ran to a scalar loss, and
``velocity.grad`` came back ``None`` - the exact silent-severance shape this
phase exists to remove, and on the one quantity the module is named after.

Three authoring points now refuse it before any object is built:
``Kinematics``, ``LinearDeformation`` and any ``DeformationVelocity``
implementation reached through ``deformation_kinematics``. The fourth shape -
a velocity DERIVED from a grad-carrying position by ``rigid_site_velocities`` -
is the same defect one step removed and is refused by the same check, because
``omega x (p - c)`` is a differentiable expression of ``p``.

Every test here asserts the refusal AND that no object was produced, and the
last one asserts that the SUPPORTED route - the same velocity as a tangent
direction - still works, so the refusal is a narrowing of a defect rather than a
removal of a capability.
"""

from __future__ import annotations

import pytest
import torch
import torch.autograd.forward_ad as forward_ad

import witwin.radar.propagation as kin


#: The refusal names the decision that makes it structural, not a preference.
ADR = "ADR-038"

POSITIONS = ((2.0, 0.6, 0.0), (2.0, 2.4, 0.0))
VELOCITIES = ((-12.0, 0.0, 0.0), (0.0, 5.0, 0.0))


def _tensor(values, *, requires_grad: bool = False) -> torch.Tensor:
    return torch.tensor(
        list(values), dtype=torch.float32
    ).requires_grad_(requires_grad)


# --------------------------------------------------------------------------
# The four shapes
# --------------------------------------------------------------------------


def test_a_kinematics_velocity_that_requires_grad_is_refused():
    """The direct request, refused before a ``Kinematics`` exists."""

    with pytest.raises(RuntimeError) as raised:
        kin.Kinematics(
            positions_m=_tensor(POSITIONS),
            velocities_m_per_s=_tensor(VELOCITIES, requires_grad=True),
        )
    message = str(raised.value)
    assert "Kinematics.velocities_m_per_s" in message
    assert ADR in message
    assert "requires_grad" in message
    # The supported use is named in the same breath, so a reader who hits this
    # learns what to do instead rather than only what not to do.
    assert "tangent" in message


def test_a_kinematics_velocity_carrying_a_forward_dual_is_refused():
    """A tangent ON a tangent is a second-order forward request."""

    with forward_ad.dual_level():
        dual = forward_ad.make_dual(
            _tensor(VELOCITIES), torch.ones(len(VELOCITIES), 3)
        )
        assert not dual.requires_grad
        with pytest.raises(RuntimeError) as raised:
            kin.Kinematics(
                positions_m=_tensor(POSITIONS), velocities_m_per_s=dual
            )
    message = str(raised.value)
    assert "forward tangent" in message
    assert ADR in message


def test_a_linear_deformation_velocity_that_requires_grad_is_refused():
    """The descriptor that is both a Core deformation and a Radar tangent.

    It is refused at construction rather than at ``velocity_at``, so a caller
    cannot build one, compile a moving scene against its ``at(t)``, and only
    then discover that the tangent half was never differentiable.
    """

    with pytest.raises(RuntimeError) as raised:
        kin.LinearDeformation(
            vertices_m=_tensor(POSITIONS),
            velocities_m_per_s=_tensor(VELOCITIES, requires_grad=True),
        )
    assert "LinearDeformation.velocities_m_per_s" in str(raised.value)
    assert ADR in str(raised.value)


def test_a_custom_deformation_velocity_is_refused_and_blamed_by_name():
    """A third-party ``DeformationVelocity`` gets its own name in the message."""

    class _GradCarryingHinge:
        def __init__(self) -> None:
            self.velocities = _tensor(VELOCITIES, requires_grad=True)

        def velocity_at(self, time_s: float) -> torch.Tensor:
            return self.velocities * float(1.0 + time_s)

    with pytest.raises(RuntimeError) as raised:
        kin.deformation_kinematics(
            _tensor(POSITIONS), _GradCarryingHinge(), 0.5
        )
    assert "_GradCarryingHinge.velocity_at" in str(raised.value)
    assert ADR in str(raised.value)


def test_a_velocity_derived_from_a_grad_carrying_position_is_refused():
    """The subtle one: ``omega x (p - c)`` inherits the position's graph.

    Nothing was marked as a velocity here. The caller marked POSITIONS, which is
    a supported leaf everywhere else in the package, and the rigid-body seam
    turned them into a graph-carrying tangent. Accepting it would send the
    position gradient down a branch autograd never traverses, and the number the
    caller read back would be short by the Doppler term with nothing to say so.
    """

    positions = _tensor(POSITIONS, requires_grad=True)
    velocities = kin.rigid_site_velocities(
        positions, velocity=(1.0, 0.0, 0.0), angular_velocity=(0.0, 0.0, 3.0)
    )
    assert velocities.requires_grad, "the premise: the graph really is inherited"

    with pytest.raises(RuntimeError) as raised:
        kin.Kinematics(positions_m=positions, velocities_m_per_s=velocities)
    assert "detached copy" in str(raised.value)

    # And the workflow the message prescribes is accepted: derive the velocity
    # from a detached copy, dual the live positions with it.
    from_detached = kin.rigid_site_velocities(
        positions.detach(),
        velocity=(1.0, 0.0, 0.0),
        angular_velocity=(0.0, 0.0, 3.0),
    )
    accepted = kin.Kinematics(
        positions_m=positions, velocities_m_per_s=from_detached
    )
    assert accepted.positions_m.requires_grad
    assert not accepted.velocities_m_per_s.requires_grad
    torch.testing.assert_close(
        from_detached, velocities.detach(), rtol=0.0, atol=0.0
    )


def test_a_stationary_structure_state_still_builds_exact_zeros():
    """The refusal did not turn a legitimate zero velocity into a failure."""

    class _State:
        rigid_motion = None

    kinematics = kin.structure_site_kinematics(_State(), _tensor(POSITIONS))
    assert torch.equal(
        kinematics.velocities_m_per_s, torch.zeros(len(POSITIONS), 3)
    )


# --------------------------------------------------------------------------
# The supported route is untouched
# --------------------------------------------------------------------------


def test_the_same_velocity_as_a_tangent_direction_stays_supported():
    """What the refusal message promises, executed.

    ``two_way_duals`` puts the very velocities that cannot be leaves into the
    tangent slot of all three position tensors, and the tangent is live inside
    the level. That is the whole supported contract, and it is what publishes
    ``delay_rate`` downstream.
    """

    sites = kin.Kinematics(
        positions_m=_tensor(POSITIONS), velocities_m_per_s=_tensor(VELOCITIES)
    )
    transmitters = kin.Kinematics(
        positions_m=_tensor([(0.0, 0.0, 0.0)]),
        velocities_m_per_s=_tensor([(0.0, 3.0, 0.0)]),
    )
    with kin.two_way_duals(sites=sites, transmitters=transmitters) as duals:
        for tensor, source in (
            (duals.sites, sites),
            (duals.transmitters, transmitters),
        ):
            primal, tangent = forward_ad.unpack_dual(tensor)
            assert tangent is not None
            assert torch.equal(primal, source.positions_m)
            assert torch.equal(tangent, source.velocities_m_per_s)


def test_a_position_leaf_and_a_velocity_tangent_coexist_in_one_dual():
    """The combination the refusal exists to keep separable.

    A ``requires_grad`` POSITION carrying a velocity TANGENT is the shape an
    inverse-design caller wants: optimise where the target is while the frame
    still has Doppler. ADR-043 pins the equivalent statement on the Channel
    side; this is the Radar-side seam agreeing with it.
    """

    positions = _tensor(POSITIONS, requires_grad=True)
    sites = kin.Kinematics(
        positions_m=positions, velocities_m_per_s=_tensor(VELOCITIES)
    )
    with kin.two_way_duals(sites=sites) as duals:
        primal, tangent = forward_ad.unpack_dual(duals.sites)
        assert tangent is not None
        assert primal.requires_grad
        assert torch.equal(tangent, sites.velocities_m_per_s)
