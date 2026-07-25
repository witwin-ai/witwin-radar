"""Radar scene geometry: promotion, placement, and world coordinates.

Three tracer tests were deleted with the Dr.Jit tracer they drove (rotated-box
front-face depth, the cylinder axis convention, and instance reuse after a
material update). They asserted properties of the tracer's intersection
result, not of the scene, and the route that replaces it - Channel's compiled
scene and RayD's own traversal - has its own coverage upstream. Restating them
here against a different intersector would be a new test wearing an old name.

What remains is the part Radar still owns: turning authored geometry into
compiled world-space renderables.
"""

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

    scene = wr.Scene(device="cpu").add_smpl(
        name="human",
        pose=np.zeros(72, dtype=np.float32),
        shape=np.zeros(10, dtype=np.float32),
    )

    structure = scene.structures[0]
    assert isinstance(structure, wr.Structure)
    assert isinstance(structure.geometry, wr.SMPLBody)
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
