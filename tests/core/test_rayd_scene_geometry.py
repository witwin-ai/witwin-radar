"""Coverage for radar scenes consumed by the RayD tracer."""

from __future__ import annotations

import numpy as np
import pytest
import torch


def _config() -> dict:
    return {
        "num_tx": 1,
        "num_rx": 1,
        "fc": 77e9,
        "slope": 60.012,
        "adc_samples": 8,
        "adc_start_time": 0,
        "sample_rate": 4400,
        "idle_time": 7,
        "ramp_end_time": 58,
        "chirp_per_frame": 2,
        "frame_per_second": 10,
        "num_doppler_bins": 2,
        "num_range_bins": 8,
        "num_angle_bins": 8,
        "power": 12,
        "tx_loc": [[0, 0, 0]],
        "rx_loc": [[0, 0, 0]],
    }


def _make_box_arrays():
    import witwin.radar as wr

    return wr.Box(position=(0.0, -0.1, -3.0), size=(0.8, 1.6, 0.4)).to_mesh()


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
        faces=(
            faces.detach().clone().to(dtype=torch.int64)
            if isinstance(faces, torch.Tensor)
            else torch.tensor(faces, dtype=torch.int64)
        ),
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


@pytest.mark.gpu
def test_rayd_tracer_preserves_rotated_box_front_face_depth():
    import witwin.radar as wr

    geometry = wr.Box(
        position=(0.0, 0.0, -3.0),
        size=(2.0, 1.0, 0.5),
        rotation=(0.0, np.pi / 2.0, 0.0),
    )
    scene = wr.Scene(device="cuda").add_structure(
        wr.Structure(name="target", geometry=geometry, material=wr.Material(eps_r=3.0))
    )
    radar = wr.Radar(_config(), device="cuda", target=(0, 0, -5), fov=60)

    trace = wr.Tracer(scene, radar, resolution=1, sampling="pixel").trace()

    assert trace.points.shape[0] == 1
    assert abs(float(trace.points[0, 2]) - (-2.0)) < 1e-4
    assert abs(float(trace.points[0, 0])) < 1e-4
    assert abs(float(trace.points[0, 1])) < 1e-4


@pytest.mark.gpu
def test_rayd_tracer_matches_cylinder_axis_coordinate_convention():
    import witwin.radar as wr

    radar = wr.Radar(_config(), device="cuda", target=(0, 0, -5), fov=60)
    scene_z = wr.Scene(device="cuda").add_structure(
        wr.Structure(
            name="target",
            geometry=wr.Cylinder(position=(0.0, 0.0, -3.0), radius=0.25, height=2.0, axis="z"),
            material=wr.Material(eps_r=3.0),
        )
    )
    scene_x = wr.Scene(device="cuda").add_structure(
        wr.Structure(
            name="target",
            geometry=wr.Cylinder(position=(0.0, 0.0, -3.0), radius=0.25, height=2.0, axis="x"),
            material=wr.Material(eps_r=3.0),
        )
    )

    trace_z = wr.Tracer(scene_z, radar, resolution=1, sampling="pixel").trace()
    trace_x = wr.Tracer(scene_x, radar, resolution=1, sampling="pixel").trace()

    assert trace_z.points.shape[0] == 1
    assert trace_x.points.shape[0] == 1
    assert abs(float(trace_z.points[0, 2]) - (-2.0)) < 1e-4
    assert abs(float(trace_x.points[0, 2]) - (-2.75)) < 1e-4


@pytest.mark.gpu
def test_rayd_tracer_reuses_instance_after_material_update():
    import witwin.radar as wr

    scene = wr.Scene(device="cuda").add_structure(
        wr.Structure(
            name="target",
            geometry=wr.Box(position=(0.0, 0.0, -3.0), size=(1.0, 1.0, 1.0)),
            material=wr.Material(eps_r=2.0),
        )
    )
    radar = wr.Radar(_config(), device="cuda", target=(0, 0, -5), fov=60)
    tracer = wr.Tracer(scene, radar, resolution=1, sampling="pixel")

    first = tracer.trace()
    scene.update_structure("target", material=wr.Material(eps_r=9.0))
    second = tracer.trace()

    assert first.points.shape == second.points.shape == (1, 3)
    assert not torch.allclose(first.intensities, second.intensities)
