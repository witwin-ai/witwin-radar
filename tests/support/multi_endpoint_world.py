"""Build the multi-endpoint fixture world with witwin.core and compile it.

Provisional dependency note (owner-approved deviation, R-ADR-008): like the
Phase-4 fixture, this consumes ``witwin-channel`` and ``witwin`` from source
checkouts rather than from pinned release wheels.

This is a sibling of ``phase4_world`` rather than an extension of it. The
Phase-4 ``endpoint_spec`` builds exactly ONE row (``positions.reshape(1, 3)``,
a single stable ID) and every Phase-4/5 expectation depends on that; batching it
would have meant changing a fixture whose numbers are frozen. The batched
builder lives here instead.
"""

from __future__ import annotations

import torch

from . import multi_endpoint_geometry as geo


def make_scene(*, transmitter_positions=None, vertices=None, eps_r=None, sigma_e=None):
    """One narrow concrete wall plus one registered antenna endpoint.

    ``vertices``, ``eps_r`` and ``sigma_e`` accept a LIVE tensor and are passed
    through untouched, so a caller can mark any of them as an AD leaf and have
    the graph reach the compiled scene. They are the scene-owned leaves
    Channel's fixed-topology reflection route supports besides the endpoints,
    and until Phase 9 no Radar test drove any of them. ``sigma_e`` is the
    conductivity half of the Fresnel coefficient: it enters the same complex
    permittivity ``eps_r`` does, through ``sigma_e / (2 pi f eps_0)``, so a
    chain that reached one and not the other would be differentiating half a
    material.

    ``Mesh`` defaults ``recenter=True`` and silently subtracts the bounding-box
    centre from authored vertices, which would move the wall plane away from
    ``x = 4`` and make every closed form in ``multi_endpoint_geometry`` wrong
    without raising anything. ``recenter=False`` is therefore mandatory and
    ``assert_world_coordinates_survived`` re-checks it. R-ADR-009 proposes
    fixing that default upstream; this fixture does not patch another
    repository to work around it.

    ``transmitter_positions`` only affects the registered ``AntennaState``,
    which is scene metadata; the legs are driven by the endpoint batches passed
    to the adapter, not by this.
    """

    from witwin.core import AntennaState, Mesh, PhysicalMaterial, Scene, Structure
    from witwin.core.identity import reserve_antenna_id

    mesh = Mesh(
        vertices=(torch.tensor(geo.WALL_VERTICES_M, dtype=torch.float32) if vertices is None else vertices),
        faces=torch.tensor(geo.WALL_FACES, dtype=torch.int64),
        recenter=False,
        fill_mode="surface",
        topology_diagnostics=False,
    )
    wall = Structure(
        geometry=mesh,
        material=PhysicalMaterial(
            name="concrete",
            eps_r=geo.WALL_EPS_R if eps_r is None else eps_r,
            sigma_e=geo.WALL_SIGMA_E if sigma_e is None else sigma_e,
        ),
        structure_id=1,
        material_id=1,
        assignment_id=1,
        surface_id=1,
    )
    anchor = geo.TX_A_POSITION_M if transmitter_positions is None else transmitter_positions[0]
    scene = Scene(
        structures=(wall,),
        endpoints=[AntennaState(reserve_antenna_id(77101), "tx", torch.tensor(anchor, dtype=torch.float32))],
    )
    return scene, mesh


def assert_world_coordinates_survived(mesh, authored=None) -> None:
    """Defensive check that authored world coordinates were not recentred.

    ``authored`` is the vertex tensor the caller handed in. When it is given the
    check is elementwise against it, which is the direct statement ("the mesh
    kept what I wrote") and is what a perturbed fixture needs; the fixture
    constants below are the same statement specialised to the default wall, and
    they are what catches a caller who forgot to pass anything at all.
    """

    vertices = mesh.vertices.detach().to(dtype=torch.float64).cpu()
    if authored is not None:
        expected = authored.detach().to(dtype=torch.float64).cpu()
        if not torch.equal(vertices, expected):
            raise AssertionError(
                "the authored vertices were rewritten between Mesh construction "
                f"and mesh.vertices; max change "
                f"{float((vertices - expected).abs().max())}"
            )
        return
    plane_x = float(vertices[:, 0].min())
    if abs(plane_x - geo.WALL_PLANE_X_M) > 1e-6:
        raise AssertionError(
            f"authored wall plane x={geo.WALL_PLANE_X_M} was rewritten to "
            f"{plane_x}; witwin.core.Mesh recentred the geometry"
        )
    half_y = float(vertices[:, 1].max())
    if abs(half_y - geo.WALL_HALF_Y_M) > 1e-6:
        raise AssertionError(
            f"authored wall half-width y={geo.WALL_HALF_Y_M} was rewritten to "
            f"{half_y}; the facet extent is this fixture's only design knob"
        )


def compile_fixture_scene(*, vertices=None, eps_r=None, sigma_e=None):
    """Compile the fixture world at the fixture reference frequency."""

    from witwin.channel.scene import compile as compile_scene

    scene, mesh = make_scene(vertices=vertices, eps_r=eps_r, sigma_e=sigma_e)
    assert_world_coordinates_survived(mesh, authored=vertices)
    return compile_scene(scene, reference_frequency_hz=geo.REFERENCE_FREQUENCY_HZ)


WALL_STRUCTURE_ID = 1


def make_dynamic_scene(
    *,
    wall_velocity=None,
    wall_origin=None,
    wall_rotation=None,
    wall_angular_velocity=None,
    wall_deformation=None,
    endpoint_trajectories=None,
):
    """The fixture world with the wall put in motion.

    ONE ``Scene`` under ONE ``DynamicScene``, never two independently built
    scenes. The four Core version domains are content hashes that fold tensor
    identity, so two separately constructed scenes of the same world differ in
    ``topology_version`` as well as in ``geometry_version`` - and a test about
    a moved wall would then be refused for the wrong reason and would pass
    while proving nothing.

    ``wall_deformation`` is any ``witwin.core.dynamics.Deformation``. It is how
    a deforming mesh enters, and it moves ``geometry_version`` while leaving
    the face indexing, the material and the assignment alone, which is exactly
    the condition a fixed-winner replay needs.
    """

    from witwin.core.dynamics import DynamicScene, LinearTrajectory

    scene, mesh = make_scene()
    assert_world_coordinates_survived(mesh)
    trajectories = {}
    if any(value is not None for value in (wall_velocity, wall_origin, wall_rotation, wall_angular_velocity)):
        trajectories[WALL_STRUCTURE_ID] = LinearTrajectory(
            origin=torch.tensor((0.0, 0.0, 0.0) if wall_origin is None else wall_origin, dtype=torch.float32),
            velocity=torch.tensor((0.0, 0.0, 0.0) if wall_velocity is None else wall_velocity, dtype=torch.float32),
            rotation=(None if wall_rotation is None else torch.tensor(wall_rotation, dtype=torch.float32)),
            angular_velocity=(
                None if wall_angular_velocity is None else torch.tensor(wall_angular_velocity, dtype=torch.float32)
            ),
        )
    return DynamicScene(
        scene,
        structure_trajectories=trajectories or None,
        structure_deformations=(None if wall_deformation is None else {WALL_STRUCTURE_ID: wall_deformation}),
        endpoint_trajectories=endpoint_trajectories,
    )


def compile_snapshot(snapshot, *, reference_frequency_hz=geo.REFERENCE_FREQUENCY_HZ):
    """Compile one ``SceneSnapshot`` at the fixture reference frequency.

    The keyword is spelled out so that this function is directly usable as
    ``SceneEpochLoop(compile_scene=...)``, which calls it exactly this way.
    """

    from witwin.channel.scene import compile as compile_scene

    return compile_scene(snapshot, reference_frequency_hz=reference_frequency_hz)


def endpoint_batch(positions, stable_ids, *, power_w=None, device="cuda"):
    """An N-row Radar endpoint spec.

    ``positions`` is either a sequence of ``(x, y, z)`` tuples or a live
    ``(N, 3)`` tensor. A tensor is passed through untouched so that a
    ``requires_grad`` leaf or a forward-AD dual keeps its tape; this is what
    lets ONE site tensor be the sink of the inbound leg and the source of the
    outbound leg and accumulate gradient from both.
    """

    from witwin.radar.propagation import RadarEndpointSpec

    ids = list(stable_ids)
    if isinstance(positions, torch.Tensor):
        values = positions
    else:
        values = torch.tensor(list(positions), dtype=torch.float32, device=device)
    if values.ndim != 2 or int(values.shape[0]) != len(ids):
        raise ValueError(
            f"positions carries {tuple(values.shape)} but {len(ids)} stable IDs "
            "were given; the endpoint batch order IS the array order and the "
            "two must be permuted together"
        )
    rows = len(ids)
    powers = None if power_w is None else torch.full((rows,), float(power_w), dtype=torch.float32, device=device)
    return RadarEndpointSpec(
        stable_ids=torch.tensor(ids, dtype=torch.int64, device=device),
        positions_m=values,
        polarizations=torch.tensor([geo.POLARIZATION] * rows, dtype=torch.float32, device=device),
        powers_w=powers,
    )


def split(endpoints):
    """Split a sequence of ``(stable_id, position)`` into ids and positions."""

    ids = tuple(stable_id for stable_id, _ in endpoints)
    positions = tuple(position for _, position in endpoints)
    return ids, positions


__all__ = [
    "WALL_STRUCTURE_ID",
    "assert_world_coordinates_survived",
    "compile_fixture_scene",
    "compile_snapshot",
    "make_dynamic_scene",
    "endpoint_batch",
    "make_scene",
    "split",
]
