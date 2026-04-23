"""Vector math helpers — both pure-Python tuple form and torch form."""

from __future__ import annotations

import math

import torch


def vec3_tuple(value, *, name: str = "vector") -> tuple[float, float, float]:
    values = tuple(float(component) for component in value)
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly three values.")
    return values


def sub3(a, b) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def norm3(value) -> float:
    return math.sqrt(value[0] * value[0] + value[1] * value[1] + value[2] * value[2])


def cross3(a, b) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def coerce_vec3(value, *, name: str):
    """Coerce to shape-(3,) torch tensor or 3-tuple of floats."""
    if isinstance(value, torch.Tensor):
        tensor = value.to(dtype=torch.float32)
        if tensor.shape != (3,):
            raise ValueError(f"{name} must have shape (3,).")
        return tensor
    values = tuple(float(component) for component in value)
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly three values.")
    return values


def coerce_optional_vec3(value, *, name: str):
    if value is None:
        return None
    return coerce_vec3(value, name=name)


def coerce_scalar(value, *, name: str):
    if isinstance(value, torch.Tensor):
        return value.to(dtype=torch.float32).reshape(())
    return float(value)


def vector_norm(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(torch.linalg.norm(value.detach().cpu()).item())
    return sum(float(component) * float(component) for component in value) ** 0.5


def normalize_rows(vectors: torch.Tensor) -> torch.Tensor:
    return vectors / torch.clamp(torch.linalg.norm(vectors, dim=-1, keepdim=True), min=1e-12)
