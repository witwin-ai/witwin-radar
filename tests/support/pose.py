"""The radar pose applied to test geometry, from ``Radar._world_from_local_matrix``.

A test authors targets in the radar's LOCAL frame - ``[0, 0, -d]`` is ``d``
metres straight ahead - and places them with these three transforms, so the
production pose frame is on the path rather than mirrored.
"""

from __future__ import annotations

import torch


def world_from_local_points(radar, points: torch.Tensor) -> torch.Tensor:
    position, world_from_local = radar._world_from_local_matrix(device=points.device, dtype=points.dtype)
    return points @ world_from_local.transpose(0, 1) + position


def world_from_local_vectors(radar, vectors: torch.Tensor) -> torch.Tensor:
    _, world_from_local = radar._world_from_local_matrix(device=vectors.device, dtype=vectors.dtype)
    return vectors @ world_from_local.transpose(0, 1)


def local_from_world_vectors(radar, vectors: torch.Tensor) -> torch.Tensor:
    _, world_from_local = radar._world_from_local_matrix(device=vectors.device, dtype=vectors.dtype)
    return vectors @ world_from_local


__all__ = ["local_from_world_vectors", "world_from_local_points", "world_from_local_vectors"]
