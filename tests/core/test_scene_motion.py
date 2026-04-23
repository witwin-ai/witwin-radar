from __future__ import annotations

import math

import pytest
import torch

from witwin.core import Material, Mesh, Structure
from witwin.radar import (
    Radar,
    RadarConfig,
    TransformMotion,
)
from witwin.radar.material import fresnel
from witwin.radar.trace import TraceResult
from witwin.radar.scene import Scene


def _config(*, chirps: int = 4, adc_samples: int = 32) -> dict:
    return {
        "num_tx": 1,
        "num_rx": 1,
        "fc": 77e9,
        "slope": 60.012,
        "adc_samples": adc_samples,
        "adc_start_time": 0,
        "sample_rate": 4400,
        "idle_time": 7,
        "ramp_end_time": 58,
        "chirp_per_frame": chirps,
        "frame_per_second": 10,
        "num_doppler_bins": chirps,
        "num_range_bins": max(64, adc_samples),
        "num_angle_bins": 8,
        "power": 12,
        "tx_loc": [[0, 0, 0]],
        "rx_loc": [[0, 0, 0]],
    }


def _rotation_matrix(axis: tuple[float, float, float], angle: float) -> torch.Tensor:
    axis_t = torch.tensor(axis, dtype=torch.float32)
    axis_t = axis_t / torch.linalg.norm(axis_t)
    x, y, z = axis_t
    c = math.cos(angle)
    s = math.sin(angle)
    one_minus_c = 1.0 - c
    return torch.tensor(
        [
            [c + x * x * one_minus_c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s],
            [y * x * one_minus_c + z * s, c + y * y * one_minus_c, y * z * one_minus_c - x * s],
            [z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, c + z * z * one_minus_c],
        ],
        dtype=torch.float32,
    )


def _rotate_points(points: torch.Tensor, *, origin: tuple[float, float, float], axis: tuple[float, float, float], angle: float) -> torch.Tensor:
    rotation = _rotation_matrix(axis, angle)
    origin_t = torch.tensor(origin, dtype=torch.float32, device=points.device)
    return (points - origin_t) @ rotation.transpose(0, 1) + origin_t


def _rotating_scene(*, device: str) -> Scene:
    triangle = Mesh(
        vertices=torch.tensor(
            [
                [0.20, 0.00, 0.00],
                [0.30, -0.05, 0.00],
                [0.30, 0.05, 0.00],
            ],
            dtype=torch.float32,
        ),
        faces=torch.tensor([[0, 1, 2]], dtype=torch.int64),
        position=(0.0, 0.0, -2.0),
        recenter=False,
        device=device,
    )
    scene = Scene(device=device).add_structure(
        Structure(
            name="rotor",
            geometry=triangle,
            material=Material(eps_r=3.0),
        )
    )
    scene.add_structure_motion(
        "rotor",
        TransformMotion(
            axis=(0.0, 1.0, 0.0),
            angular_velocity=800.0,
            origin=(0.0, 0.0, 0.0),
            space="local",
        ),
    )
    return scene


def _centroid_trace(scene: Scene, *, time: float) -> TraceResult:
    compiled = scene.compile_renderables(time=time)["rotor"]
    centroid = compiled.vertices.mean(dim=0, keepdim=True)
    intensities = torch.ones(1, dtype=torch.float32, device=compiled.vertices.device)
    return TraceResult(centroid, intensities)


def _triangle_trace(scene: Scene, radar: Radar, *, time: float) -> TraceResult:
    compiled = scene.compile_renderables(time=time)["rotor"]
    vertices = compiled.vertices
    v0, v1, v2 = vertices[0], vertices[1], vertices[2]
    centroid = vertices.mean(dim=0, keepdim=True)
    cross = torch.cross(v1 - v0, v2 - v0, dim=0)
    area = 0.5 * torch.linalg.norm(cross)
    normal = cross / torch.clamp(torch.linalg.norm(cross), min=1e-10)
    origin = radar.position.to(dtype=vertices.dtype, device=vertices.device)
    view_dir = origin - centroid[0]
    view_dir = view_dir / torch.linalg.norm(view_dir)
    if torch.dot(view_dir, normal) <= 0:
        return TraceResult(
            torch.empty((0, 3), dtype=vertices.dtype, device=vertices.device),
            torch.empty((0,), dtype=torch.float32, device=vertices.device),
        )
    cos_i = torch.abs(torch.dot(view_dir, normal))
    intensity = (area * fresnel(cos_i, compiled.eps_r)).reshape(1).to(dtype=torch.float32)
    return TraceResult(centroid, intensity)


def test_scene_compile_renderables_applies_local_rotation_over_time():
    scene = _rotating_scene(device="cpu")

    renderable0 = scene.compile_renderables(time=0.0)["rotor"]
    renderable1 = scene.compile_renderables(time=0.001)["rotor"]

    expected = _rotate_points(
        renderable0.vertices,
        origin=(0.0, 0.0, -2.0),
        axis=(0.0, 1.0, 0.0),
        angle=0.8,
    )
    assert torch.allclose(renderable1.vertices, expected, atol=1e-6, rtol=1e-6)


def test_scene_parent_motion_carries_child_geometry():
    parent = Mesh(
        vertices=torch.tensor([[0.00, 0.00, 0.00], [0.10, 0.00, 0.00], [0.00, 0.10, 0.00]], dtype=torch.float32),
        faces=torch.tensor([[0, 1, 2]], dtype=torch.int64),
        position=(0.0, 0.0, -2.0),
        recenter=False,
        device="cpu",
    )
    child = Mesh(
        vertices=torch.tensor([[0.00, 0.00, 0.00], [0.05, 0.00, 0.00], [0.00, 0.05, 0.00]], dtype=torch.float32),
        faces=torch.tensor([[0, 1, 2]], dtype=torch.int64),
        position=(0.5, 0.0, -2.0),
        recenter=False,
        device="cpu",
    )
    scene = Scene(device="cpu")
    scene.add_structure(Structure(name="parent", geometry=parent, material=Material(eps_r=3.0)))
    scene.add_structure(Structure(name="child", geometry=child, material=Material(eps_r=3.0)))
    scene.add_structure_motion(
        "parent",
        TransformMotion(
            axis=(0.0, 0.0, 1.0),
            angular_velocity=math.pi / 2.0,
            origin=(0.0, 0.0, 0.0),
            space="world",
        ),
    )
    scene.add_structure_motion("child", TransformMotion(parent="parent"))

    child0 = scene.compile_renderables(time=0.0)["child"].vertices
    child1 = scene.compile_renderables(time=1.0)["child"].vertices
    expected = _rotate_points(
        child0,
        origin=(0.0, 0.0, 0.0),
        axis=(0.0, 0.0, 1.0),
        angle=math.pi / 2.0,
    )
    assert torch.allclose(child1, expected, atol=1e-6, rtol=1e-6)


def test_radar_motion_sampling_chirp_matches_manual_interpolator(monkeypatch):
    scene = _rotating_scene(device="cpu")
    config = RadarConfig.from_dict(_config(chirps=3, adc_samples=16))
    observed_times: list[float | None] = []

    class FakeRenderer:
        def __init__(
            self,
            scene,
            radar,
            resolution=128,
            epsilon_r=5.0,
            sampling="triangle",
            multipath=False,
            max_reflections=0,
            ray_batch_size=65536,
        ):
            self.scene = scene
            self.radar = radar

        def trace(self, *, time=None):
            observed_times.append(time)
            return _centroid_trace(self.scene, time=0.0 if time is None else float(time))

    monkeypatch.setattr("witwin.radar.trace.Renderer", FakeRenderer)

    radar = Radar(config, backend="pytorch", device="cpu")
    result = radar.simulate(
        scene,
        motion_sampling="per_chirp",
    )

    def interpolator(t):
        return _centroid_trace(scene, time=float(t))

    expected = radar.mimo(interpolator, 0.0)
    assert torch.allclose(result, expected, atol=1e-6, rtol=1e-6)

    chirp_period = (radar.idle_time + radar.ramp_end_time) * 1e-6 * radar.num_tx
    assert observed_times == pytest.approx([0.0, 0.0, chirp_period, 2.0 * chirp_period], rel=0.0, abs=1e-12)


def test_radar_motion_sampling_frame_uses_single_trace(monkeypatch):
    scene = _rotating_scene(device="cpu")
    observed_times: list[float | None] = []

    class FakeRenderer:
        def __init__(
            self,
            scene,
            radar,
            resolution=128,
            epsilon_r=5.0,
            sampling="triangle",
            multipath=False,
            max_reflections=0,
            ray_batch_size=65536,
        ):
            self.scene = scene
            self.radar = radar

        def trace(self, *, time=None):
            observed_times.append(time)
            return _centroid_trace(self.scene, time=0.0 if time is None else float(time))

    monkeypatch.setattr("witwin.radar.trace.Renderer", FakeRenderer)

    radar = Radar(RadarConfig.from_dict(_config(chirps=4, adc_samples=16)), backend="pytorch", device="cpu")
    radar.simulate(
        scene,
        motion_sampling="per_frame",
    )

    assert observed_times == [0.0]


def test_mimo_group_with_motion_matches_individual_runs(monkeypatch):
    scene = _rotating_scene(device="cpu")

    class FakeRenderer:
        def __init__(
            self,
            scene,
            radar,
            resolution=128,
            epsilon_r=5.0,
            sampling="triangle",
            multipath=False,
            max_reflections=0,
            ray_batch_size=65536,
        ):
            self.scene = scene
            self.radar = radar

        def trace(self, *, time=None):
            return _centroid_trace(self.scene, time=0.0 if time is None else float(time))

    monkeypatch.setattr("witwin.radar.trace.Renderer", FakeRenderer)

    config = RadarConfig.from_dict(_config(chirps=3, adc_samples=16))
    front = Radar(config, name="front", backend="pytorch", device="cpu")
    side = Radar(
        config,
        name="side",
        backend="pytorch",
        device="cpu",
        position=(1.0, 0.0, 0.0),
        target=(1.0, 0.0, -1.0),
    )

    group = Radar.simulate_group(
        scene,
        radars=[front, side],
        motion_sampling="per_chirp",
    )

    front_single = front.simulate(
        scene,
        motion_sampling="per_chirp",
    )
    side_single = side.simulate(
        scene,
        motion_sampling="per_chirp",
    )

    assert torch.allclose(group["front"], front_single, atol=1e-6, rtol=1e-6)
    assert torch.allclose(group["side"], side_single, atol=1e-6, rtol=1e-6)


@pytest.mark.gpu
def test_triangle_renderer_rotation_motion_matches_manual_signal_gpu():
    scene = _rotating_scene(device="cuda")
    config = RadarConfig.from_dict(_config(chirps=4, adc_samples=32))
    radar = Radar(config, backend="pytorch", device="cuda")

    result = radar.simulate(
        scene,
        sampling="triangle",
        motion_sampling="per_chirp",
        resolution=32,
    )

    def interpolator(t):
        return _triangle_trace(scene, radar, time=float(t))

    expected = radar.mimo(interpolator, 0.0)
    expected_trace = _triangle_trace(scene, radar, time=0.0)

    assert torch.allclose(radar.last_trace.points, expected_trace.points, atol=5e-4, rtol=5e-4)
    assert torch.allclose(radar.last_trace.intensities, expected_trace.intensities, atol=5e-4, rtol=5e-4)
    assert torch.allclose(result, expected, atol=5e-4, rtol=5e-4)
