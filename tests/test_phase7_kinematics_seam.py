"""The Core-kinematics-to-forward-dual seam, pinned primitive by primitive.

``RigidMotion.velocity`` and ``RigidMotion.angular_velocity`` had zero consumers
anywhere in the platform before ``witwin.radar.propagation`` existed:
Channel's compiler reads only ``rotation`` and ``translation``, and every
Doppler test built its tangent from a literal. So this file pins the two halves
of the seam separately.

* the ARITHMETIC - ``v = v_cm + omega x (p - c)``, the rotation centre, the
  endpoint composition, the deformation contract - against hand-computed values
  and against a finite-difference oracle of the exact composition Channel
  applies. Finite differences are permitted here and nowhere else: this is a
  test, and the quantity being differenced is a closed-form pose, not a solver.
* the DUAL DISCIPLINE - that a tangent reaches the tensors the seam claims it
  reaches, and that the failure mode when it does not is visible. That second
  half is the whole reason this module exists as a production owner instead of
  as three lines in each caller: ``make_dual`` followed by a rebuild from Python
  values produces an ordinary tensor with no tangent, the chain then publishes
  ``delay_rate = 0``, and zero is exactly what a correct stationary scene
  publishes.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.autograd.forward_ad as forward_ad

import witwin.radar.propagation as kin

# --------------------------------------------------------------------------
# v = v_cm + omega x (p - c)
# --------------------------------------------------------------------------


def test_angular_velocity_produces_the_expected_site_velocity():
    """The cross product, against a hand-computed value.

    ``omega = (0, 0, 2)`` about ``c = (2, 0, 0)`` sends the point ``(2, 0.6, 0)``
    to ``omega x (0, 0.6, 0) = (-1.2, 0, 0)`` and the diametrically opposite
    point to ``(+1.2, 0, 0)``. Equal and opposite, which is the rotor-blade
    signature: a model carrying only ``v_cm`` would give both points the same
    velocity and could not produce a Doppler SPREAD at all, only a shift.
    """

    positions = torch.tensor([[2.0, 0.6, 0.0], [2.0, -0.6, 0.0], [2.9, 0.0, 0.0]], dtype=torch.float32)
    velocities = kin.rigid_site_velocities(
        positions, velocity=(0.5, 0.0, -0.25), angular_velocity=(0.0, 0.0, 2.0), centre_m=(2.0, 0.0, 0.0)
    )
    expected = torch.tensor([[0.5 - 1.2, 0.0, -0.25], [0.5 + 1.2, 0.0, -0.25], [0.5, 1.8, -0.25]], dtype=torch.float32)
    torch.testing.assert_close(velocities, expected, rtol=1e-6, atol=1e-7)

    # The spin term alone is antisymmetric about the centre, which is the
    # statement the blade-flash spread rests on.
    spin = velocities - torch.tensor([0.5, 0.0, -0.25])
    torch.testing.assert_close(spin[0], -spin[1], rtol=1e-6, atol=1e-7)


def test_a_body_with_no_declared_motion_is_exactly_stationary():
    """``None`` means "not part of this motion", not "argument missing".

    A Core ``RigidMotion`` that declares only a translation describes a purely
    translating body, and a structure state with no motion at all describes a
    stationary one. Both must produce EXACT zeros rather than a small number,
    because a downstream ``delay_rate`` of exactly zero is how a stationary
    scene is recognised.
    """

    positions = torch.tensor([[1.0, 2.0, 3.0], [-4.0, 0.5, 0.0]], dtype=torch.float32)
    velocities = kin.rigid_site_velocities(positions)
    assert torch.equal(velocities, torch.zeros_like(positions))

    class _State:
        rigid_motion = None

    kinematics = kin.structure_site_kinematics(_State(), positions)
    assert torch.equal(kinematics.velocities_m_per_s, torch.zeros_like(positions))
    assert torch.equal(kinematics.positions_m, positions)


def test_the_rotation_centre_is_the_translation_not_the_authored_pose():
    """The centre that matches how Channel actually composes a snapshot.

    Channel builds a moved structure as ``vertices @ R.T + t``: the authored
    WORLD vertices are rotated about the world ORIGIN and the translation is
    applied afterwards. Differentiating that gives
    ``dp/dt = omega x (p - t) + t_dot``, so the instantaneous rotation centre is
    the current translation. The intuitive answer - the authored pose position -
    is wrong, and wrong in a way that hides: it adds an ``omega x (t - pose)``
    offset that is UNIFORM over the body and therefore looks exactly like a
    platform velocity rather than like a bug.

    The oracle is a central difference of that same composition. A finite
    difference is forbidden in production and is exactly the right instrument
    here, because the thing being differenced is a closed-form pose.
    """

    authored = torch.tensor([[3.0, 1.0, 0.0], [3.0, -1.0, 0.5]], dtype=torch.float64)
    omega = 0.75
    origin = torch.tensor([2.0, -0.5, 0.0], dtype=torch.float64)
    speed = torch.tensor([0.4, 0.1, -0.2], dtype=torch.float64)

    def posed(time_s: float) -> torch.Tensor:
        angle = omega * time_s
        rotation = torch.tensor(
            [[math.cos(angle), -math.sin(angle), 0.0], [math.sin(angle), math.cos(angle), 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float64,
        )
        return authored @ rotation.T + (origin + speed * time_s)

    step = 1.0e-5
    reference = (posed(step) - posed(-step)) / (2.0 * step)

    class _Motion:
        translation = origin.to(torch.float32)
        velocity = speed.to(torch.float32)
        angular_velocity = torch.tensor([0.0, 0.0, omega], dtype=torch.float32)

    class _State:
        rigid_motion = _Motion()

    centre = kin.rotation_centre_m(_Motion())
    torch.testing.assert_close(centre, origin.to(torch.float32), rtol=1e-6, atol=1e-7)

    measured = kin.structure_site_kinematics(_State(), posed(0.0).to(torch.float32)).velocities_m_per_s
    torch.testing.assert_close(measured.to(torch.float64), reference, rtol=1e-5, atol=1e-7)

    # And the authored pose centre is a DIFFERENT, wrong answer here, so the
    # assertion above is a choice rather than a coincidence.
    wrong = kin.rigid_site_velocities(
        posed(0.0).to(torch.float32),
        velocity=speed.to(torch.float32),
        angular_velocity=torch.tensor([0.0, 0.0, omega], dtype=torch.float32),
        centre_m=(0.0, 0.0, 0.0),
    )
    assert float((wrong - measured).abs().max()) > 0.3


# --------------------------------------------------------------------------
# Endpoint kinematics off a Core snapshot
# --------------------------------------------------------------------------


def _dynamic_endpoint_snapshot(time_s: float):
    from witwin.core import AntennaState, Scene
    from witwin.core.dynamics import DynamicScene, LinearTrajectory
    from witwin.core.identity import reserve_antenna_id

    moving = AntennaState(reserve_antenna_id(77201), "tx", torch.tensor([1.0, 2.0, 3.0]))
    still = AntennaState(reserve_antenna_id(77202), "rx", torch.tensor([-1.0, 0.0, 0.5]))
    scene = Scene(structures=(), endpoints=[moving, still])
    dynamic = DynamicScene(
        scene, endpoint_trajectories={77201: LinearTrajectory(origin=(0.0, 0.0, 0.0), velocity=(4.0, 0.0, -1.0))}
    )
    return dynamic.at(time_s)


def test_endpoint_velocity_comes_from_the_core_rigid_motion():
    """The first consumer ``RigidMotion.velocity`` has ever had.

    Position follows Core's own composition - the authored antenna position
    plus the snapshot's additional world-frame translation - and velocity is
    the declared ``velocity`` verbatim. An endpoint with no trajectory is
    exactly stationary at exactly its authored position.
    """

    snapshot = _dynamic_endpoint_snapshot(2.0)
    kinematics = kin.endpoint_kinematics(snapshot, (77201, 77202), device="cpu")
    torch.testing.assert_close(
        kinematics.positions_m,
        torch.tensor([[9.0, 2.0, 1.0], [-1.0, 0.0, 0.5]], dtype=torch.float32),
        rtol=1e-6,
        atol=1e-7,
    )
    torch.testing.assert_close(
        kinematics.velocities_m_per_s,
        torch.tensor([[4.0, 0.0, -1.0], [0.0, 0.0, 0.0]], dtype=torch.float32),
        rtol=1e-6,
        atol=1e-7,
    )
    assert torch.equal(kinematics.velocities_m_per_s[1], torch.zeros(3, dtype=torch.float32))

    # The declared order IS the endpoint batch order, so reversing it reverses
    # both tensors together. Reversing only one is the failure this pins.
    reversed_kinematics = kin.endpoint_kinematics(snapshot, (77202, 77201), device="cpu")
    assert torch.equal(reversed_kinematics.positions_m, kinematics.positions_m.flip(0))
    assert torch.equal(reversed_kinematics.velocities_m_per_s, kinematics.velocities_m_per_s.flip(0))


def test_an_endpoint_the_snapshot_does_not_declare_is_named():
    snapshot = _dynamic_endpoint_snapshot(0.0)
    with pytest.raises(KeyError, match="77999"):
        kin.endpoint_kinematics(snapshot, (77201, 77999), device="cpu")


def test_the_position_and_the_tangent_must_name_the_same_endpoints():
    with pytest.raises(ValueError, match="same endpoints in the same order"):
        kin.Kinematics(positions_m=torch.zeros(3, 3), velocities_m_per_s=torch.zeros(2, 3))


# --------------------------------------------------------------------------
# The deformation contract
# --------------------------------------------------------------------------


class _HingeVelocity:
    """A hinge: vertex ``k`` moves ``k`` times as fast as the root.

    Analytic, which is the point. Core's ``DeformationState`` states where the
    vertices are and never how fast they are moving, so a deforming structure
    has no time derivative anywhere in Core; differencing two snapshots is a
    finite difference and is forbidden in production. An implementation of the
    protocol is the supported way to supply the missing derivative.
    """

    def __init__(self, tip_speed: float, vertices: int) -> None:
        self._tip_speed = float(tip_speed)
        self._vertices = int(vertices)

    def velocity_at(self, time_s: float) -> torch.Tensor:
        scale = self._tip_speed * math.cos(float(time_s))
        rows = torch.arange(self._vertices, dtype=torch.float32)
        return torch.stack([torch.zeros_like(rows), rows * scale, torch.zeros_like(rows)], dim=1)


def test_the_deformation_protocol_supplies_the_velocity():
    descriptor = _HingeVelocity(tip_speed=3.0, vertices=5)
    assert isinstance(descriptor, kin.DeformationVelocity)

    positions = torch.tensor([[2.0, 0.2, 0.0], [2.0, 1.0, 0.0], [2.0, 1.8, 0.0]], dtype=torch.float32)
    tracked = torch.tensor([1, 2, 4], dtype=torch.int64)
    kinematics = kin.deformation_kinematics(positions, descriptor, 0.0, vertex_index=tracked)
    torch.testing.assert_close(
        kinematics.velocities_m_per_s,
        torch.tensor([[0.0, 3.0, 0.0], [0.0, 6.0, 0.0], [0.0, 12.0, 0.0]], dtype=torch.float32),
        rtol=1e-6,
        atol=1e-7,
    )
    assert kinematics.velocities_m_per_s.is_contiguous()

    # The velocity is a function of time, so a later instant is a different
    # answer: a descriptor that ignored ``time_s`` would pass every shape check
    # and freeze the deformation rate at whatever it was authored with.
    later = kin.deformation_kinematics(positions, descriptor, math.pi / 3.0, vertex_index=tracked)
    torch.testing.assert_close(later.velocities_m_per_s, kinematics.velocities_m_per_s * 0.5, rtol=1e-6, atol=1e-7)


def test_a_descriptor_that_returns_the_wrong_type_is_refused():
    class _Broken:
        def velocity_at(self, time_s: float):
            return [(0.0, 0.0, 0.0)]

    with pytest.raises(TypeError, match="velocity_at must return"):
        kin.deformation_kinematics(torch.zeros(1, 3, dtype=torch.float32), _Broken(), 0.0)


# --------------------------------------------------------------------------
# The dual discipline
# --------------------------------------------------------------------------


def test_slot_replication_keeps_the_tangent_alive():
    """``index_select`` on the dual, not a rebuild from values.

    Slot-major replication is what turns one endpoint set into a whole TDM
    frame. Doing it by reading the positions back into Python and building a
    fresh tensor produces a stack with no tangent, which then publishes
    ``delay_rate = 0`` for every slot - a perfectly plausible stationary frame.
    """

    positions = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=torch.float32)
    velocities = torch.tensor([[-3.0, 0.0, 0.0], [0.0, 0.5, 0.0]], dtype=torch.float32)
    with forward_ad.dual_level():
        dual = forward_ad.make_dual(positions, velocities)
        stacked = kin.replicate_slots(dual, 3)
        assert tuple(stacked.shape) == (6, 3)
        primal, tangent = forward_ad.unpack_dual(stacked)
        assert tangent is not None
        assert torch.equal(primal, positions.repeat(3, 1))
        assert torch.equal(tangent, velocities.repeat(3, 1))

        # Slot major, not endpoint major: slot ``t`` owns a contiguous block.
        assert torch.equal(primal[0:2], positions)
        assert torch.equal(primal[2:4], positions)

        # And the rebuild-from-values route is the trap, demonstrated rather
        # than described.
        rebuilt = torch.tensor(stacked.detach().tolist(), dtype=torch.float32)
        assert forward_ad.unpack_dual(rebuilt).tangent is None

    assert kin.replicate_slots(positions, 1) is positions


def test_all_three_endpoint_tensors_are_dualised_in_one_level():
    """One level covering TX, site and RX, because a round trip needs all three.

    The inbound leg's rate is ``d|p_site - p_tx|/dt`` and needs both tangents
    live at once; the outbound leg's needs the site and the receiver. A level
    per tensor would let each leg see one moving end and one frozen end and
    would publish a round trip whose halves describe different worlds.
    """

    def track(values, rates):
        return kin.Kinematics(
            positions_m=torch.tensor(values, dtype=torch.float32),
            velocities_m_per_s=torch.tensor(rates, dtype=torch.float32),
        )

    transmitters = track([[0.0, 0.0, 0.0]], [[0.0, 3.0, 0.0]])
    sites = track([[2.0, 0.6, 0.0]], [[-12.0, 0.0, 0.0]])
    receivers = track([[0.15, 0.0, 0.0]], [[0.0, -3.0, 0.0]])

    with kin.two_way_duals(sites=sites, transmitters=transmitters, receivers=receivers, slot_count=2) as duals:
        for tensor, source in ((duals.transmitters, transmitters), (duals.sites, sites), (duals.receivers, receivers)):
            primal, tangent = forward_ad.unpack_dual(tensor)
            assert tangent is not None
            assert torch.equal(primal, source.positions_m.repeat(2, 1))
            assert torch.equal(tangent, source.velocities_m_per_s.repeat(2, 1))
        assert duals.slot_count == 2
        held = duals.sites

    # The level is closed on exit, which is why the adapter clones the delay
    # tangent INSIDE it rather than after.
    assert forward_ad.unpack_dual(held).tangent is None


def test_a_static_front_end_is_declared_by_omission():
    sites = kin.Kinematics(
        positions_m=torch.tensor([[2.0, 0.6, 0.0]], dtype=torch.float32),
        velocities_m_per_s=torch.tensor([[0.0, 12.0, 0.0]], dtype=torch.float32),
    )
    with kin.two_way_duals(sites=sites) as duals:
        assert duals.transmitters is None
        assert duals.receivers is None
        assert forward_ad.unpack_dual(duals.sites).tangent is not None

    with pytest.raises(TypeError, match="sites must be a Kinematics"):
        with kin.two_way_duals(sites=sites.positions_m):
            pass


# --------------------------------------------------------------------------
# The two traps, against the real producer
# --------------------------------------------------------------------------


@pytest.mark.gpu
def test_a_missing_tangent_is_a_hard_error():
    """``ad_mode='jvp'`` outside a dual level raises rather than publishing zero.

    The adapter already refuses this; what is new is that the refusal is pinned
    against the seam that is now supposed to open the level. Publishing
    ``delay_rate = 0`` here would be indistinguishable from a correct
    stationary answer.
    """

    pytest.importorskip("witwin.channel")
    from support import multi_endpoint_driver as drv

    spike = drv.MultiEndpointSpike()
    with pytest.raises(RuntimeError, match="dual_level"):
        spike.legs(spike.site_tensor(), ad_mode="jvp")


@pytest.mark.gpu
def test_a_dead_tangent_is_detectable():
    """The trap is real, the fixture can see it, and the seam does not fall in.

    The trap is PARTIAL, and that is what makes it silent. Killing EVERY tangent
    is caught by the adapter, which refuses a jvp replay that produced no delay
    tangent at all (``test_a_missing_tangent_is_a_hard_error``). Killing exactly
    ONE of the three tensors is not: the remaining duals keep the delay tangent
    alive, nothing raises, and the rate that comes back is missing the whole
    contribution of the endpoint that was rebuilt. Here the site is the only
    mover, so the trap's answer is a clean, plausible, entirely wrong ZERO.

    Three measurements in one place, because any two of them alone are
    consistent with a broken fixture:

    1. the radial reference rate is NOT zero, so a zero result means something;
    2. a site tensor rebuilt from Python values inside the very same dual level
       publishes ``delay_rate`` of exactly zero and raises nothing;
    3. the tensor :func:`two_way_duals` yields carries a live tangent, so the
       production seam is not taking route 2.
    """

    pytest.importorskip("witwin.channel")
    from support import multi_endpoint_driver as drv
    from support import multi_endpoint_geometry as geo

    spike = drv.MultiEndpointSpike()

    def still(positions):
        return kin.Kinematics(positions_m=positions, velocities_m_per_s=torch.zeros_like(positions))

    sites = kin.Kinematics(
        positions_m=spike.site_tensor(),
        velocities_m_per_s=torch.tensor(
            [geo.SITE_P_RADIAL_VELOCITY_M_PER_S, geo.STATIONARY], dtype=torch.float32, device="cuda"
        ),
    )
    with kin.two_way_duals(
        sites=sites, transmitters=still(spike.transmitter_tensor()), receivers=still(spike.receiver_tensor())
    ) as duals:
        assert forward_ad.unpack_dual(duals.sites).tangent is not None
        live, _, _ = spike.frame(duals.sites, transmitters=duals.transmitters, receivers=duals.receivers, ad_mode="jvp")
        live_rate = live.delay_rate.clone()

        rebuilt = spike.site_tensor()
        assert forward_ad.unpack_dual(rebuilt).tangent is None
        dead, _, _ = spike.frame(rebuilt, transmitters=duals.transmitters, receivers=duals.receivers, ad_mode="jvp")
        dead_rate = dead.delay_rate.clone()

    assert float(live_rate.abs().max()) > 1.0e-9
    assert torch.equal(dead_rate, torch.zeros_like(dead_rate))
    assert not torch.equal(live_rate, dead_rate)
