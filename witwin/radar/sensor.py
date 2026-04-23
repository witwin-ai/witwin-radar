"""Radar sensor definitions."""

from __future__ import annotations

import torch

from .utils.vector import vec3_tensor


class Sensor:
    """Radar pose in world coordinates with tensor-valued fields."""

    def __init__(
        self,
        origin=(0.0, 0.0, 0.0),
        target=(0.0, 0.0, -1.0),
        up=(0.0, 1.0, 0.0),
        fov: float = 60.0,
    ) -> None:
        origin_t = vec3_tensor(origin, name="Sensor.origin")
        target_t = vec3_tensor(target, name="Sensor.target")
        up_t = vec3_tensor(up, name="Sensor.up")
        forward = target_t - origin_t
        if torch.linalg.norm(forward) <= 1e-12:
            raise ValueError("Sensor.target must differ from Sensor.origin.")
        if torch.linalg.norm(up_t) <= 1e-12:
            raise ValueError("Sensor.up must be non-zero.")
        if torch.linalg.norm(torch.cross(forward, up_t, dim=0)) <= 1e-12:
            raise ValueError("Sensor.up must not be collinear with the viewing direction.")
        self.origin: torch.Tensor = origin_t
        self.target: torch.Tensor = target_t
        self.up: torch.Tensor = up_t
        self.fov: float = float(fov)

    def __repr__(self) -> str:
        return (
            f"Sensor(origin={self.origin.tolist()}, target={self.target.tolist()}, "
            f"up={self.up.tolist()}, fov={self.fov})"
        )

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
        origin = self.origin.to(device=device, dtype=dtype)
        target = self.target.to(device=device, dtype=dtype)
        up = self.up.to(device=device, dtype=dtype)

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
