"""Scene motion, at the level Radar still owns it: compiled renderables.

This file used to have eight tests. Six of them drove ``Radar.simulate`` or
``Radar.simulate_group`` through a monkeypatched Dr.Jit ``Tracer`` and covered
motion SAMPLING - per-chirp retracing, single-trace frames, linear
correspondence between two traces, and the multi-radar group entry. All four of
those behaviours belonged to the tracer-driven scene entry point, which no
longer exists; there is nothing left for them to assert about, and rewriting
them against a route that samples motion differently would be inventing
coverage rather than preserving it. They are recorded as coverage debt for the
scene-driven entry that replaces ``simulate``.

The two that remain never touched the tracer. ``Scene.compile_renderables`` is
Radar's own motion evaluation, and it is exactly as testable as it was.
"""

from __future__ import annotations

import math

import torch

from witwin.core import Mesh, PhysicalMaterial as Material, Structure
from witwin.radar import TransformMotion
from witwin.radar.scene import Scene


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
