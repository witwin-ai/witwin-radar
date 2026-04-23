"""Radar sensor definitions."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .utils.vector import cross3, norm3, sub3, vec3_tuple


@dataclass(frozen=True)
class Sensor:
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    target: tuple[float, float, float] = (0.0, 0.0, -1.0)
    up: tuple[float, float, float] = (0.0, 1.0, 0.0)
    fov: float = 60.0

    def __post_init__(self):
        object.__setattr__(self, "origin", vec3_tuple(self.origin, name="Sensor.origin"))
        object.__setattr__(self, "target", vec3_tuple(self.target, name="Sensor.target"))
        object.__setattr__(self, "up", vec3_tuple(self.up, name="Sensor.up"))
        object.__setattr__(self, "fov", float(self.fov))
        forward = sub3(self.target, self.origin)
        if norm3(forward) <= 1e-12:
            raise ValueError("Sensor.target must differ from Sensor.origin.")
        if norm3(self.up) <= 1e-12:
            raise ValueError("Sensor.up must be non-zero.")
        if norm3(cross3(forward, self.up)) <= 1e-12:
            raise ValueError("Sensor.up must not be collinear with the viewing direction.")

    @classmethod
    def identity(cls, *, fov: float = 60.0) -> "Sensor":
        return cls(origin=(0.0, 0.0, 0.0), target=(0.0, 0.0, -1.0), up=(0.0, 1.0, 0.0), fov=fov)

    def updated(self, *, origin=None, target=None, up=None, fov=None) -> "Sensor":
        return Sensor(
            origin=self.origin if origin is None else origin,
            target=self.target if target is None else target,
            up=self.up if up is None else up,
            fov=self.fov if fov is None else fov,
        )

    def _world_from_local_matrix(self, *, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        origin = torch.tensor(self.origin, device=device, dtype=dtype)
        target = torch.tensor(self.target, device=device, dtype=dtype)
        up = torch.tensor(self.up, device=device, dtype=dtype)

        forward = target - origin
        forward = forward / torch.linalg.norm(forward)
        right = torch.cross(forward, up, dim=0)
        right = right / torch.linalg.norm(right)
        true_up = torch.cross(right, forward, dim=0)
        true_up = true_up / torch.linalg.norm(true_up)
        back = -forward
        world_from_local = torch.stack((right, true_up, back), dim=1)
        return origin, world_from_local

    def world_from_local_points(self, points: torch.Tensor) -> torch.Tensor:
        origin, world_from_local = self._world_from_local_matrix(device=points.device, dtype=points.dtype)
        return points @ world_from_local.transpose(0, 1) + origin

    def world_from_local_vectors(self, vectors: torch.Tensor) -> torch.Tensor:
        _, world_from_local = self._world_from_local_matrix(device=vectors.device, dtype=vectors.dtype)
        return vectors @ world_from_local.transpose(0, 1)

    def local_from_world_points(self, points: torch.Tensor) -> torch.Tensor:
        origin, world_from_local = self._world_from_local_matrix(device=points.device, dtype=points.dtype)
        return (points - origin) @ world_from_local

    def local_from_world_vectors(self, vectors: torch.Tensor) -> torch.Tensor:
        _, world_from_local = self._world_from_local_matrix(device=vectors.device, dtype=vectors.dtype)
        return vectors @ world_from_local
