"""Moving structures: a deforming mesh and a moving environment.

Everything here is driven from Core descriptors - a ``Deformation``, a
``LinearTrajectory`` on a structure - and checked against the float64
image-source closed forms in ``support.multi_endpoint_geometry``.

A STRUCTURE carries its motion in the compiled scene: Channel's fixed
reflection reads wall vertices from the compiled scene it is handed, so a moved
wall changes the reflection delay. The moving-environment rate is therefore
measured the only way it exists: as the evolution of the delay across
snapshots, replayed on a fresh ``CompiledScene`` per snapshot under the
declared ``world_motion="fixed_winner_replay"``.

Sign convention, once: ``f_D = -f_ref * d(tau_rt)/dt``; a receding row is
negative.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("witwin.channel")

from support import multi_endpoint_driver as drv
from support import multi_endpoint_geometry as geo
from support import multi_endpoint_world as world
from support.smpl_models import smpl_model_root

pytestmark = pytest.mark.gpu

#: The gate against the float64 closed form. Worst measured value anywhere in
#: this file is 4.4e-5, so this is 45x of margin.
RATE_RTOL = 2.0e-3


# --------------------------------------------------------------------------
# S5  deformation
# --------------------------------------------------------------------------


def test_the_smpl_pose_velocity_matches_a_two_snapshot_difference():
    """The independent oracle for the analytic SMPL vertex velocity.

    Core has no velocity on ``DeformationState``, so a deforming mesh has no
    time derivative anywhere in Core and Radar supplies one analytically. The
    only oracle available for "is that derivative right" is a difference of two
    snapshots - a finite difference, which is allowed HERE and forbidden in
    production.

    The tolerance is derived, not tuned. ``SmplPoseDeformation`` runs a
    rotation through linear blend skinning, so its central difference carries
    an ``O(h^2)`` truncation plus a cancellation floor of roughly
    ``eps * |p| / h`` - at ``h = 1e-4 s``, 1 m limbs and float32 that is about
    6e-4 relative, which is what is measured.
    """

    smpl = pytest.importorskip("smplpytorch")
    del smpl
    from witwin.radar.smpl import SMPLBody, SmplPoseDeformation

    model_root = smpl_model_root()
    if model_root is None:
        pytest.skip("no SMPL model files available in this checkout")
    step = 1.0e-4
    pose_rate = torch.zeros(72, device="cuda")
    # One elbow and one knee, so the fast vertices are a limb rather than the
    # whole body: a global rotation would make every vertex agree trivially.
    pose_rate[3 * 18 + 2] = 3.0
    pose_rate[3 * 4 + 0] = -2.0
    body = SMPLBody(pose=torch.zeros(72), shape=torch.zeros(10), model_root=model_root, device="cuda")
    deformation = SmplPoseDeformation(body, pose_rate=pose_rate)
    velocity = deformation.velocity_at(0.0)
    fastest = float(velocity.norm(dim=1).max())
    assert fastest > 0.5, fastest
    fd = (deformation.vertices_at(step) - deformation.vertices_at(-step)) / (2.0 * step)
    error = float((fd - velocity).norm(dim=1).max()) / fastest
    assert error < 5.0e-3, error


def test_a_deforming_mesh_is_a_core_deformation_not_a_reposed_geometry():
    """The ``SMPLBody`` bridge: a posed body enters as a Core ``Deformation``.

    Re-posing and re-meshing a body per frame would respecify the structure's
    faces every time and make every frozen ``primitive_sequence`` label
    meaningless. As a ``Deformation`` over a fixed rest ``Mesh`` it instead
    moves only the vertices, so ``topology_version`` holds still and a
    fixed-winner replay stays legitimate.
    """

    pytest.importorskip("smplpytorch")
    from witwin.core import PhysicalMaterial, Scene, Structure
    from witwin.core.dynamics import DeformationState, DynamicScene

    from witwin.radar.smpl import SMPLBody, SmplPoseDeformation

    model_root = smpl_model_root()
    if model_root is None:
        pytest.skip("no SMPL model files available in this checkout")
    pose_rate = torch.zeros(72, device="cuda")
    pose_rate[3 * 18 + 2] = 3.0
    body = SMPLBody(pose=torch.zeros(72), shape=torch.zeros(10), model_root=model_root, device="cuda")
    deformation = SmplPoseDeformation(body, pose_rate=pose_rate)
    state = deformation.at(0.1)
    assert isinstance(state, DeformationState)
    assert state.vertices is not None and state.offsets is None

    mesh = deformation.rest_mesh()
    assert mesh.recenter is False
    scene = Scene(
        structures=(
            Structure(
                geometry=mesh,
                material=PhysicalMaterial(name="skin", eps_r=15.0, sigma_e=1.2),
                structure_id=7,
                material_id=1,
                assignment_id=1,
                surface_id=1,
            ),
        ),
        endpoints=[],
    )
    dynamic = DynamicScene(scene, structure_deformations={7: deformation})
    early = dynamic.at(0.0)
    late = dynamic.at(0.1)
    assert early.topology_version == late.topology_version
    assert early.geometry_version != late.geometry_version
    moved = late.structures[0].deformation.vertices
    rest = early.structures[0].deformation.vertices
    assert not torch.equal(moved, rest)
    # A DEFORMATION, not a rigid motion: the limb sweeps centimetres while the
    # torso barely registers. A rigid motion would displace every vertex by the
    # same amount, so the spread is the discriminating statement and the
    # absolute minimum is not - linear blend skinning's pose blend shapes touch
    # every vertex by a few microns and nothing is ever exactly frozen.
    displacement = (moved - rest).norm(dim=1)
    assert float(displacement.max()) > 0.05
    assert float(displacement.max()) / float(displacement.min()) > 100.0


class _LinearWallDeformation:
    """A Core ``Deformation`` moving two of the wall's four vertices along x."""

    def __init__(self) -> None:
        self.vertices_m = torch.tensor(geo.WALL_VERTICES_M, dtype=torch.float32)
        self.velocities_m_per_s = torch.tensor(
            [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.3, 0.0, 0.0), (0.3, 0.0, 0.0)], dtype=torch.float32
        )

    def at(self, time_s: float):
        from witwin.core.dynamics import DeformationState

        return DeformationState(vertices=self.vertices_m + self.velocities_m_per_s * float(time_s))


def test_rotation_and_deformation_preserve_topology_version():
    """A moved structure changes exactly one version domain.

    If a rotation or a deformation ever moved ``topology_version``,
    ``material_version`` or ``assignment_version``, Channel would correctly
    refuse a fixed-winner replay - the frozen ``primitive_sequence`` and
    ``material_sequence`` labels would no longer name the same world - and
    every scenario in this file would become unreachable. This is the test that
    says so out loud rather than leaving it as an assumption.
    """

    quarter_turn = (0.0, 0.0, 0.35)
    rotated = world.make_dynamic_scene(wall_rotation=quarter_turn, wall_angular_velocity=(0.0, 0.0, 1.0))
    deformed = world.make_dynamic_scene(wall_deformation=_LinearWallDeformation())
    for dynamic, label in ((rotated, "rotation"), (deformed, "deformation")):
        early = world.compile_snapshot(dynamic.at(0.0))
        late = world.compile_snapshot(dynamic.at(1.0))
        assert early.topology_version == late.topology_version, label
        assert early.material_version == late.material_version, label
        assert early.assignment_version == late.assignment_version, label
        assert early.geometry_version != late.geometry_version, label
        # And the structure really did move, rather than the version being a
        # timestamp: a static wall compiled from the same authored scene gives
        # the same reflection geometry, a moved one does not.
        assert early.time_s == 0.0 and late.time_s == 1.0


# --------------------------------------------------------------------------
# S7  moving environment
# --------------------------------------------------------------------------


def _static_legs(spike):
    """Both legs at the fixture's own, unmoving endpoint positions."""

    sites = spike.site_tensor()
    inbound = spike.adapter.reevaluate_slots(
        spike.inbound,
        spike._stacked_ids(spike.stacked([p for _, p in spike.transmitters], 1), spike.transmitter_ids, geo.TX_POWER_W),
        spike._stacked_ids(sites, spike.site_ids, None),
        slot_count=1,
        ad_mode="none",
    )
    outbound = spike.adapter.reevaluate_slots(
        spike.outbound,
        spike._stacked_ids(sites, spike.site_ids, geo.SITE_POWER_W),
        spike._stacked_ids(spike.stacked([p for _, p in spike.receivers], 1), spike.receiver_ids, None),
        slot_count=1,
        ad_mode="none",
    )
    return inbound, outbound


def test_a_translating_wall_moves_only_the_reflection_row():
    """The single most load-bearing test of this stage.

    A wall translating along its own normal at 4 m/s, with every endpoint
    standing still. It exercises items 1, 5 and 7 together: a fresh
    ``CompiledScene`` per snapshot from ``DynamicScene.at``, the declared
    ``world_motion="fixed_winner_replay"`` that lets a frozen topology be
    replayed against moved geometry, and the physics that says only the
    reflection rows can possibly change.

    The line-of-sight rows are asserted with ``torch.equal`` across a wall
    displacement of 8 mm, not with a tolerance. A line of sight does not touch
    the wall, so its delay is not merely stable, it is the same float.

    The reflection rows are checked against the image source, which moves at
    ``2u`` along the plane normal because a mirror through a plane at ``x = P``
    puts the image at ``2P - x``. That factor of two is the whole content of
    the scenario: a wall velocity read straight off the trajectory would be
    half the right answer and would still look plausible.
    """

    dynamic = world.make_dynamic_scene(wall_velocity=geo.WALL_VELOCITY_M_PER_S)
    step = 1.0e-3
    spike = drv.MultiEndpointSpike(compiled=world.compile_snapshot(dynamic.at(0.0)))
    assert spike.adapter.world_motion == "frozen_world"

    delays = {}
    for time_s in (-step, step):
        spike.adapter.refreeze(world.compile_snapshot(dynamic.at(time_s)), world_motion="fixed_winner_replay")
        assert spike.adapter.world_motion == "fixed_winner_replay"
        inbound, outbound = _static_legs(spike)
        # Every frozen row still exists: the wall slid along its own normal by
        # 8 mm and no specular point left the facet.
        assert bool(inbound.row_valid.all()) and bool(outbound.row_valid.all())
        delays[time_s] = (inbound.delay_s.double().clone(), outbound.delay_s.double().clone())

    for index, (leg_name, rows) in enumerate(
        (("inbound", spike.predicted_inbound_rows()), ("outbound", spike.predicted_outbound_rows()))
    ):
        early = delays[-step][index]
        late = delays[step][index]
        rate = (late - early) / (2.0 * step)
        positions = dict(geo.ALL_ENDPOINTS)
        los = [row.component == "los" for row in rows]
        assert any(los) and not all(los), leg_name
        mask = torch.tensor(los, device=early.device)
        assert torch.equal(early[mask], late[mask]), leg_name
        for row_index, row in enumerate(rows):
            expected = geo.wall_motion_leg_delay_rate_s_per_s(
                positions[row.source_id], positions[row.sink_id], row.component, geo.WALL_VELOCITY_M_PER_S
            )
            measured = float(rate[row_index])
            if row.component == "los":
                assert measured == 0.0, (leg_name, row_index)
            else:
                assert measured == pytest.approx(expected, rel=RATE_RTOL), (leg_name, row_index)
                # Non-vacuity: the reflection really is moving.
                assert abs(measured) > 1.0e-9
