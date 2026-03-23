"""Gradient checks for core geometry.to_mesh()."""

from __future__ import annotations

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))


def test_box_to_mesh_is_differentiable():
    from witwin.core import Box

    center = torch.tensor([0.2, -0.1, -3.0], dtype=torch.float32, requires_grad=True)
    size = torch.tensor([0.8, 1.6, 0.4], dtype=torch.float32, requires_grad=True)
    box = Box(position=center, size=size)

    vertices, faces = box.to_mesh()
    assert isinstance(vertices, torch.Tensor)
    assert faces.shape == (12, 3)

    loss = vertices.square().sum()
    loss.backward()

    assert center.grad is not None and center.grad.abs().sum() > 0
    assert size.grad is not None and size.grad.abs().sum() > 0


def test_mesh_to_mesh_is_differentiable_for_vertices_and_transform():
    from witwin.core import Mesh

    base_vertices = torch.tensor(
        [
            [-0.5, -0.5, 0.0],
            [0.5, -0.5, 0.0],
            [0.5, 0.5, 0.0],
            [-0.5, 0.5, 0.0],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int64)
    center = torch.tensor([0.1, 0.2, -2.5], dtype=torch.float32, requires_grad=True)
    scale = torch.tensor([1.0, 2.0, 1.0], dtype=torch.float32, requires_grad=True)
    mesh = Mesh(base_vertices, faces, position=center, scale=scale, recenter=False)

    vertices, _ = mesh.to_mesh()
    loss = vertices.square().sum()
    loss.backward()

    assert base_vertices.grad is not None and base_vertices.grad.abs().sum() > 0
    assert center.grad is not None and center.grad.abs().sum() > 0
    assert scale.grad is not None and scale.grad.abs().sum() > 0
