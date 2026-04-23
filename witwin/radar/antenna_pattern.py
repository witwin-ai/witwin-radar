"""Antenna pattern configuration and runtime evaluation.

Validation lives in :mod:`witwin.radar.validation`. This module defines the
dataclass, interpolators, and the runtime that evaluates patterns on GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch


_DEFAULT_DIPOLE_ANGLES_DEG = tuple(float(angle) for angle in range(-90, 91))


def _half_wave_dipole_power_cut(angle_deg: float) -> float:
    """Normalized half-wave dipole power gain versus off-boresight angle."""
    angle_rad = math.radians(angle_deg)
    cos_angle = math.cos(angle_rad)
    if abs(cos_angle) < 1e-8:
        return 0.0

    field = math.cos(0.5 * math.pi * math.sin(angle_rad)) / cos_angle
    return max(field * field, 0.0)


_DEFAULT_DIPOLE_VALUES = tuple(_half_wave_dipole_power_cut(angle) for angle in _DEFAULT_DIPOLE_ANGLES_DEG)


@dataclass(frozen=True)
class AntennaPatternConfig:
    """Per-element antenna gain pattern.

    Use :func:`witwin.radar.validation.default_dipole_antenna_pattern` to build
    a default half-wave dipole pattern, or
    :func:`witwin.radar.validation.validate_antenna_pattern_config` to build
    from a dict.
    """

    kind: str
    x_angles_deg: tuple[float, ...]
    y_angles_deg: tuple[float, ...]
    x_values: tuple[float, ...] | None = None
    y_values: tuple[float, ...] | None = None
    values: tuple[tuple[float, ...], ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "kind": self.kind,
            "x_angles_deg": list(self.x_angles_deg),
            "y_angles_deg": list(self.y_angles_deg),
        }
        if self.kind == "separable":
            data["x_values"] = list(self.x_values or ())
            data["y_values"] = list(self.y_values or ())
        else:
            data["values"] = [list(row) for row in (self.values or ())]
        return data


def _interp1d_zero_outside(axis: torch.Tensor, values: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    if query.numel() == 0:
        return torch.empty_like(query, dtype=values.dtype)

    flat_query = query.reshape(-1)
    index_upper = torch.bucketize(flat_query.detach(), axis)
    index_left = torch.clamp(index_upper - 1, 0, axis.numel() - 1)
    index_right = torch.clamp(index_upper, 0, axis.numel() - 1)

    x0 = axis[index_left]
    x1 = axis[index_right]
    y0 = values[index_left]
    y1 = values[index_right]
    denom = torch.clamp(x1 - x0, min=1e-12)
    weight = torch.where(index_left == index_right, torch.zeros_like(flat_query), (flat_query - x0) / denom)
    interpolated = y0 + weight * (y1 - y0)
    inside = (flat_query >= axis[0]) & (flat_query <= axis[-1])
    return torch.where(inside, interpolated, torch.zeros_like(interpolated)).reshape(query.shape)


def _interp2d_zero_outside(
    x_axis: torch.Tensor,
    y_axis: torch.Tensor,
    values: torch.Tensor,
    x_query: torch.Tensor,
    y_query: torch.Tensor,
) -> torch.Tensor:
    flat_x = x_query.reshape(-1)
    flat_y = y_query.reshape(-1)

    x_upper = torch.bucketize(flat_x.detach(), x_axis)
    y_upper = torch.bucketize(flat_y.detach(), y_axis)
    x_left = torch.clamp(x_upper - 1, 0, x_axis.numel() - 1)
    x_right = torch.clamp(x_upper, 0, x_axis.numel() - 1)
    y_low = torch.clamp(y_upper - 1, 0, y_axis.numel() - 1)
    y_high = torch.clamp(y_upper, 0, y_axis.numel() - 1)

    x0 = x_axis[x_left]
    x1 = x_axis[x_right]
    y0 = y_axis[y_low]
    y1 = y_axis[y_high]
    tx = torch.where(x_left == x_right, torch.zeros_like(flat_x), (flat_x - x0) / torch.clamp(x1 - x0, min=1e-12))
    ty = torch.where(y_low == y_high, torch.zeros_like(flat_y), (flat_y - y0) / torch.clamp(y1 - y0, min=1e-12))

    v00 = values[y_low, x_left]
    v10 = values[y_low, x_right]
    v01 = values[y_high, x_left]
    v11 = values[y_high, x_right]

    interpolated = (
        (1.0 - tx) * (1.0 - ty) * v00
        + tx * (1.0 - ty) * v10
        + (1.0 - tx) * ty * v01
        + tx * ty * v11
    )
    inside = (
        (flat_x >= x_axis[0])
        & (flat_x <= x_axis[-1])
        & (flat_y >= y_axis[0])
        & (flat_y <= y_axis[-1])
    )
    return torch.where(inside, interpolated, torch.zeros_like(interpolated)).reshape(x_query.shape)


@dataclass(frozen=True)
class AntennaPatternRuntime:
    """Runtime tensor view of an antenna pattern config."""

    kind: str
    x_angles_deg: torch.Tensor
    y_angles_deg: torch.Tensor
    x_values: torch.Tensor | None = None
    y_values: torch.Tensor | None = None
    values: torch.Tensor | None = None

    @classmethod
    def from_config(cls, config: AntennaPatternConfig, *, device: str) -> "AntennaPatternRuntime":
        x_angles_deg = torch.tensor(config.x_angles_deg, dtype=torch.float32, device=device)
        y_angles_deg = torch.tensor(config.y_angles_deg, dtype=torch.float32, device=device)
        x_values = None
        y_values = None
        values = None

        if config.kind == "separable":
            x_values = torch.tensor(config.x_values, dtype=torch.float32, device=device)
            y_values = torch.tensor(config.y_values, dtype=torch.float32, device=device)
        else:
            values = torch.tensor(config.values, dtype=torch.float32, device=device)

        return cls(
            kind=config.kind,
            x_angles_deg=x_angles_deg,
            y_angles_deg=y_angles_deg,
            x_values=x_values,
            y_values=y_values,
            values=values,
        )

    def evaluate_xy(self, x_angles_deg: torch.Tensor, y_angles_deg: torch.Tensor) -> torch.Tensor:
        if self.kind == "separable":
            return _interp1d_zero_outside(self.x_angles_deg, self.x_values, x_angles_deg) * _interp1d_zero_outside(
                self.y_angles_deg,
                self.y_values,
                y_angles_deg,
            )
        return _interp2d_zero_outside(self.x_angles_deg, self.y_angles_deg, self.values, x_angles_deg, y_angles_deg)

    def evaluate_vectors(self, vectors: torch.Tensor) -> torch.Tensor:
        forward = -vectors[..., 2]
        x_angles_deg = torch.rad2deg(torch.atan2(vectors[..., 0], forward))
        y_angles_deg = torch.rad2deg(torch.atan2(vectors[..., 1], forward))
        return self.evaluate_xy(x_angles_deg, y_angles_deg)
