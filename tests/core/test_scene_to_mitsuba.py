"""CPU-side coverage for shared Mitsuba scene conversion helpers."""

from __future__ import annotations

import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))


def _make_box_arrays():
    import witwin.radar as wr

    return wr.Box(position=(0.0, -0.1, -3.0), size=(0.8, 1.6, 0.4)).to_mesh()


def _dummy_sensor_dict(T):
    return {
        "type": "perspective",
        "to_world": T.look_at(origin=(0, 0, 0), target=(0, 0, -5), up=(0, 1, 0)),
        "fov": 60,
        "film": {
            "type": "hdrfilm",
            "width": 8,
            "height": 8,
            "pixel_format": "luminance",
            "component_format": "float32",
        },
        "sampler": {"type": "independent", "sample_count": 1},
    }


def _param_vertices(params, key: str) -> np.ndarray:
    return np.array(params[key], dtype=np.float32).reshape(-1, 3)


def _build_geometry_state(geometry):
    mi = pytest.importorskip("mitsuba")
    mi.set_variant("scalar_rgb")
    from mitsuba.scalar_rgb import Transform4f as T
    from witwin.core.scene_to_mitsuba import build_mitsuba_scene

    vertices, faces = geometry.to_mesh()
    state = build_mitsuba_scene(
        sensor=_dummy_sensor_dict(T),
        renderables={
            "target": SimpleNamespace(vertices=vertices, faces=faces, bsdf=None),
        },
        variant="scalar_rgb",
    )
    return mi, state


def test_static_mesh_arrays_promote_to_core_mesh():
    import witwin.radar as wr

    vertices, faces = _make_box_arrays()
    vertices_np = vertices.detach().cpu().numpy()
    scene = wr.Scene(device="cpu").add_mesh(
        name="target",
        vertices=vertices_np,
        faces=faces,
    )

    structure = scene.structures[0]
    assert isinstance(structure, wr.Structure)
    assert isinstance(structure.geometry, wr.Mesh)
    assert isinstance(structure.material, wr.Material)
    assert structure.metadata == {}


def test_tensor_mesh_inputs_promote_to_core_mesh_and_keep_dynamic_metadata():
    import witwin.radar as wr

    vertices, faces = _make_box_arrays()
    scene = wr.Scene(device="cpu").add_mesh(
        name="target",
        vertices=torch.as_tensor(vertices, dtype=torch.float32),
        faces=torch.tensor(faces, dtype=torch.int64),
        dynamic=True,
    )

    structure = scene.structures[0]
    assert isinstance(structure, wr.Structure)
    assert isinstance(structure.geometry, wr.Mesh)
    assert bool(structure.metadata["dynamic"]) is True


def test_scene_accepts_shared_structure_directly():
    import witwin.radar as wr

    structure = wr.Structure(
        geometry=wr.Box(position=(0.0, 0.0, -3.0), size=(1.0, 1.0, 1.0)),
        material=wr.Material(eps_r=3.0),
        name="target",
    )
    scene = wr.Scene(device="cpu").add_structure(structure)

    compiled = scene.compile_renderables()
    assert "target" in compiled
    assert compiled["target"].vertices.shape[1] == 3


def test_add_smpl_builds_shared_structure_and_dynamic_metadata():
    import witwin.radar as wr
    import witwin.core as wc

    scene = wr.Scene(device="cpu").add_smpl(
        name="human",
        pose=np.zeros(72, dtype=np.float32),
        shape=np.zeros(10, dtype=np.float32),
    )

    structure = scene.structures[0]
    assert isinstance(structure, wr.Structure)
    assert isinstance(structure.geometry, wc.SMPLBody)
    assert bool(structure.metadata["dynamic"]) is True


def test_update_structure_moves_box_geometry():
    import witwin.radar as wr

    scene = wr.Scene(device="cpu").add_structure(
        wr.Structure(
            geometry=wr.Box(position=(0.0, 0.0, -3.0), size=(1.0, 1.0, 1.0)),
            material=wr.Material(),
            name="target",
        )
    )
    scene.update_structure("target", position=(1.0, 0.0, -3.0))

    compiled = scene.compile_renderables()["target"]
    center = compiled.vertices.mean(dim=0)
    np.testing.assert_allclose(center.detach().cpu().numpy(), np.array([1.0, 0.0, -3.0]), atol=1e-6)


def test_core_scene_to_mitsuba_builds_and_updates_scene():
    mi = pytest.importorskip("mitsuba")
    mi.set_variant("scalar_rgb")
    from mitsuba.scalar_rgb import Transform4f as T

    from witwin.core.scene_to_mitsuba import build_mitsuba_scene, update_mitsuba_scene_vertices

    vertices, faces = _make_box_arrays()
    state = build_mitsuba_scene(
        sensor={
            "type": "perspective",
            "to_world": T.look_at(origin=(0, 0, 0), target=(0, 0, -5), up=(0, 1, 0)),
            "fov": 60,
            "film": {
                "type": "hdrfilm",
                "width": 8,
                "height": 8,
                "pixel_format": "luminance",
                "component_format": "float32",
            },
            "sampler": {"type": "independent", "sample_count": 1},
        },
        renderables={
            "target": SimpleNamespace(vertices=vertices, faces=faces, bsdf=None),
        },
        variant="scalar_rgb",
    )

    assert state.scene is not None
    assert "target.vertex_positions" in state.params

    moved = vertices.detach().clone()
    moved[:, 0] += 0.05
    update_mitsuba_scene_vertices(
        state.params,
        {"target": SimpleNamespace(vertices=moved, faces=faces, bsdf=None)},
        variant="scalar_rgb",
    )


def test_geometry_vertices_are_preserved_when_loaded_into_mitsuba():
    from witwin.core import Box

    geometry = Box(
        position=(0.5, 0.25, -3.0),
        size=(2.0, 1.0, 0.5),
        rotation=(0.0, np.pi / 2.0, 0.0),
    )
    expected_vertices, _ = geometry.to_mesh()
    _, state = _build_geometry_state(geometry)

    actual_vertices = _param_vertices(state.params, "target.vertex_positions")
    np.testing.assert_allclose(
        actual_vertices,
        expected_vertices.detach().cpu().numpy(),
        atol=1e-6,
    )


def test_box_rotation_about_y_axis_hits_expected_front_face_depth():
    from witwin.core import Box

    mi, state = _build_geometry_state(
        Box(
            position=(0.0, 0.0, -3.0),
            size=(2.0, 1.0, 0.5),
            rotation=(0.0, np.pi / 2.0, 0.0),
        )
    )

    ray = mi.Ray3f(o=mi.Point3f(0.0, 0.0, 0.0), d=mi.Vector3f(0.0, 0.0, -1.0))
    si = state.scene.ray_intersect(ray)

    assert si.is_valid()
    assert abs(float(si.p.z) - (-2.0)) < 1e-4
    assert abs(float(si.p.x)) < 1e-4
    assert abs(float(si.p.y)) < 1e-4


def test_cylinder_axis_mapping_matches_scene_coordinate_convention():
    from witwin.core import Cylinder

    mi, state_z = _build_geometry_state(
        Cylinder(position=(0.0, 0.0, -3.0), radius=0.25, height=2.0, axis="z")
    )
    _, state_x = _build_geometry_state(
        Cylinder(position=(0.0, 0.0, -3.0), radius=0.25, height=2.0, axis="x")
    )

    ray = mi.Ray3f(o=mi.Point3f(0.0, 0.0, 0.0), d=mi.Vector3f(0.0, 0.0, -1.0))
    si_z = state_z.scene.ray_intersect(ray)
    si_x = state_x.scene.ray_intersect(ray)

    assert si_z.is_valid()
    assert si_x.is_valid()
    assert abs(float(si_z.p.z) - (-2.0)) < 1e-4
    assert abs(float(si_x.p.z) - (-2.75)) < 1e-4
