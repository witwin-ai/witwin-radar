"""Build the single-site fixture world with witwin.core and compile it.

The multi-endpoint world builder does the work; this module hands it the
single-site wall and adds the one-row endpoint spec the single-site tests
declare their three endpoints with.
"""

from __future__ import annotations

import torch

from . import multi_endpoint_world
from . import single_site_geometry as geo


def make_scene(*, rough: bool = False):
    """The single-site wall plus one registered antenna endpoint.

    ``rough=True`` gives the wall a Gaussian surface roughness. That is not a
    variant anyone simulates here; it exists so a test can prove that the
    consumer's refusal to reevaluate a frozen reflection topology on a rough
    scene reaches the caller instead of being swallowed into a quietly smooth
    answer.
    """

    from witwin.core.material import SurfaceRoughness

    roughness = (
        SurfaceRoughness(rms_height_m=1.0e-3, correlation_length_x_m=1.0e-2, correlation_length_y_m=1.0e-2)
        if rough
        else None
    )
    return multi_endpoint_world.make_scene(vertices=_wall_vertices(), roughness=roughness)


def _wall_vertices() -> torch.Tensor:
    return torch.tensor(geo.WALL_VERTICES_M, dtype=torch.float32)


def assert_world_coordinates_survived(mesh) -> None:
    """The authored wall was not recentred; see ``multi_endpoint_world``."""

    multi_endpoint_world.assert_world_coordinates_survived(mesh, authored=_wall_vertices())


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

    positions = position.reshape(1, 3) if isinstance(position, torch.Tensor) else (position,)
    return multi_endpoint_world.endpoint_batch(positions, (stable_id,), power_w=power_w, device=device)


__all__ = ["assert_world_coordinates_survived", "compile_fixture_scene", "endpoint_spec", "make_scene"]
