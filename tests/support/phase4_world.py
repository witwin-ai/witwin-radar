"""Build the Phase-4 fixture world with witwin.core and compile it.

Provisional dependency note (owner-approved deviation, R-ADR-008): Phase 4
consumes ``witwin-channel`` and ``witwin`` from source checkouts rather than
from pinned release wheels, because the release artifacts were still building.
The artifact pin plus a required-consumer CI job is the recorded follow-up.
"""

from __future__ import annotations

import torch

from . import phase4_geometry as geo


def make_scene(*, rough: bool = False):
    """A single concrete wall plus one registered antenna endpoint.

    ``rough=True`` gives the wall a Gaussian surface roughness. That is not a
    variant anyone simulates here; it exists so a test can prove that the
    consumer's refusal to reevaluate a frozen reflection topology on a rough
    scene reaches the caller instead of being swallowed into a quietly smooth
    answer.

    ``Mesh`` defaults ``recenter=True`` and silently subtracts the bounding-box
    centre from authored vertices, so a caller that writes world coordinates and
    omits the keyword gets relocated geometry and physically wrong results that
    never raise. Every Mesh here passes ``recenter=False`` deliberately.
    R-ADR-009 proposes fixing that default upstream; this spike does not patch
    another repository to work around it.
    """

    from witwin.core import AntennaState, Mesh, PhysicalMaterial, Scene, Structure
    from witwin.core.identity import reserve_antenna_id
    from witwin.core.material import SurfaceRoughness

    mesh = Mesh(
        vertices=torch.tensor(geo.WALL_VERTICES_M, dtype=torch.float32),
        faces=torch.tensor(geo.WALL_FACES, dtype=torch.int64),
        recenter=False,
        fill_mode="surface",
        topology_diagnostics=False,
    )
    roughness = (
        SurfaceRoughness(
            rms_height_m=1.0e-3,
            correlation_length_x_m=1.0e-2,
            correlation_length_y_m=1.0e-2,
        )
        if rough
        else None
    )
    wall = Structure(
        geometry=mesh,
        material=PhysicalMaterial(
            name="concrete",
            eps_r=geo.WALL_EPS_R,
            sigma_e=geo.WALL_SIGMA_E,
            roughness_front=roughness,
        ),
        structure_id=1,
        material_id=1,
        assignment_id=1,
        surface_id=1,
    )
    scene = Scene(
        structures=(wall,),
        endpoints=[
            AntennaState(
                reserve_antenna_id(77001),
                "tx",
                torch.tensor(geo.TX_POSITION_M, dtype=torch.float32),
            )
        ],
    )
    return scene, mesh


def assert_world_coordinates_survived(mesh) -> None:
    """Defensive check that authored world coordinates were not recentred.

    R-ADR-009 is Proposed, not implemented, so this asserts the property the
    spike actually depends on instead of trusting an upstream default.
    """

    vertices = mesh.vertices.detach().to(dtype=torch.float64).cpu()
    plane_x = float(vertices[:, 0].min())
    if abs(plane_x - geo.WALL_PLANE_X_M) > 1e-6:
        raise AssertionError(
            f"authored wall plane x={geo.WALL_PLANE_X_M} was rewritten to "
            f"{plane_x}; witwin.core.Mesh recentred the geometry"
        )


def compile_fixture_scene(*, rough: bool = False):
    """Compile the fixture world at the fixture reference frequency."""

    from witwin.channel.scene import compile as compile_scene

    scene, mesh = make_scene(rough=rough)
    assert_world_coordinates_survived(mesh)
    return compile_scene(scene, reference_frequency_hz=geo.REFERENCE_FREQUENCY_HZ)


def endpoint_spec(position, stable_id, *, power_w=None, device="cuda"):
    """One-row Radar endpoint spec at ``position``.

    ``position`` may be a tuple, or a live tensor that carries ``requires_grad``
    or a forward-AD tangent; a tensor is passed through untouched so the tape
    survives.
    """

    from witwin.radar.propagation import RadarEndpointSpec

    if isinstance(position, torch.Tensor):
        positions = position.reshape(1, 3)
    else:
        positions = torch.tensor([position], dtype=torch.float32, device=device)
    powers = (
        None
        if power_w is None
        else torch.full((1,), float(power_w), dtype=torch.float32, device=device)
    )
    return RadarEndpointSpec(
        stable_ids=torch.tensor([stable_id], dtype=torch.int64, device=device),
        positions_m=positions,
        polarizations=torch.tensor(
            [geo.POLARIZATION], dtype=torch.float32, device=device
        ),
        powers_w=powers,
    )


__all__ = [
    "assert_world_coordinates_survived",
    "compile_fixture_scene",
    "endpoint_spec",
    "make_scene",
]
