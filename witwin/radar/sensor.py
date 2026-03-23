"""Radar sensor definitions."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


def _vec3(value, *, field_name: str) -> tuple[float, float, float]:
    values = tuple(float(component) for component in value)
    if len(values) != 3:
        raise ValueError(f"{field_name} must contain exactly three values.")
    return values


def _sub3(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _norm3(value: tuple[float, float, float]) -> float:
    return math.sqrt(value[0] * value[0] + value[1] * value[1] + value[2] * value[2])


def _cross3(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


@dataclass(frozen=True)
class Sensor:
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    target: tuple[float, float, float] = (0.0, 0.0, -1.0)
    up: tuple[float, float, float] = (0.0, 1.0, 0.0)
    fov: float = 60.0

    def __post_init__(self):
        object.__setattr__(self, "origin", _vec3(self.origin, field_name="Sensor.origin"))
        object.__setattr__(self, "target", _vec3(self.target, field_name="Sensor.target"))
        object.__setattr__(self, "up", _vec3(self.up, field_name="Sensor.up"))
        object.__setattr__(self, "fov", float(self.fov))
        forward = _sub3(self.target, self.origin)
        if _norm3(forward) <= 1e-12:
            raise ValueError("Sensor.target must differ from Sensor.origin.")
        if _norm3(self.up) <= 1e-12:
            raise ValueError("Sensor.up must be non-zero.")
        if _norm3(_cross3(forward, self.up)) <= 1e-12:
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
