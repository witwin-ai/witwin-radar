"""Antenna pattern configuration and runtime evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from typing import Any

import torch


_DEFAULT_DIPOLE_ANGLES_DEG = tuple(float(angle) for angle in range(-90, 91))


def _finite_float(name: str, value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Antenna pattern field '{name}' must be a finite float.") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"Antenna pattern field '{name}' must be a finite float.")
    return parsed


def _validate_axis(name: str, value: Any) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"Antenna pattern field '{name}' must be a sequence of angles in degrees.")
    if len(value) < 2:
        raise ValueError(f"Antenna pattern field '{name}' must contain at least 2 samples.")

    axis = tuple(_finite_float(f"{name}[{index}]", angle) for index, angle in enumerate(value))
    for index in range(1, len(axis)):
        if axis[index] <= axis[index - 1]:
            raise ValueError(f"Antenna pattern field '{name}' must be strictly increasing.")
    return axis


def _validate_values_1d(name: str, value: Any, expected_count: int) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"Antenna pattern field '{name}' must be a sequence of gain values.")
    if len(value) != expected_count:
        raise ValueError(
            f"Antenna pattern field '{name}' must contain exactly {expected_count} entries; got {len(value)}."
        )

    gains = []
    for index, item in enumerate(value):
        gain = _finite_float(f"{name}[{index}]", item)
        if gain < 0.0:
            raise ValueError(f"Antenna pattern field '{name}[{index}]' must be non-negative.")
        gains.append(gain)
    return tuple(gains)


def _validate_values_2d(name: str, value: Any, expected_rows: int, expected_cols: int) -> tuple[tuple[float, ...], ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"Antenna pattern field '{name}' must be a 2D sequence of gain values.")
    if len(value) != expected_rows:
        raise ValueError(
            f"Antenna pattern field '{name}' must contain exactly {expected_rows} rows; got {len(value)}."
        )

    rows = []
    for row_index, row in enumerate(value):
        if not isinstance(row, (list, tuple)):
            raise ValueError(f"Antenna pattern field '{name}[{row_index}]' must be a sequence of gain values.")
        if len(row) != expected_cols:
            raise ValueError(
                f"Antenna pattern field '{name}[{row_index}]' must contain exactly {expected_cols} entries; "
                f"got {len(row)}."
            )

        parsed_row = []
        for col_index, item in enumerate(row):
            gain = _finite_float(f"{name}[{row_index}][{col_index}]", item)
            if gain < 0.0:
                raise ValueError(f"Antenna pattern field '{name}[{row_index}][{col_index}]' must be non-negative.")
            parsed_row.append(gain)
        rows.append(tuple(parsed_row))
    return tuple(rows)


def _detect_kind(data: dict[str, Any]) -> str:
    raw_kind = data.get("kind")
    if raw_kind is None:
        if "values" in data:
            return "map"
        if "x_values" in data or "y_values" in data:
            return "separable"
        raise ValueError("Antenna pattern config must define 'kind' or provide fields for a known pattern type.")

    kind = str(raw_kind)
    if kind not in {"separable", "map"}:
        raise ValueError("Antenna pattern field 'kind' must be 'separable' or 'map'.")
    return kind


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
    """Validated per-element antenna gain pattern."""

    kind: str
    x_angles_deg: tuple[float, ...]
    y_angles_deg: tuple[float, ...]
    x_values: tuple[float, ...] | None = None
    y_values: tuple[float, ...] | None = None
    values: tuple[tuple[float, ...], ...] | None = None

    @classmethod
    def default_dipole(cls) -> "AntennaPatternConfig":
        return cls(
            kind="separable",
            x_angles_deg=_DEFAULT_DIPOLE_ANGLES_DEG,
            y_angles_deg=_DEFAULT_DIPOLE_ANGLES_DEG,
            x_values=_DEFAULT_DIPOLE_VALUES,
            y_values=_DEFAULT_DIPOLE_VALUES,
        )

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "AntennaPatternConfig":
        if not isinstance(config, dict):
            raise TypeError("Antenna pattern config must be a dict.")

        kind = _detect_kind(config)
        x_angles_deg = _validate_axis("x_angles_deg", config.get("x_angles_deg"))
        y_angles_deg = _validate_axis("y_angles_deg", config.get("y_angles_deg"))

        if kind == "separable":
            x_values = _validate_values_1d("x_values", config.get("x_values"), len(x_angles_deg))
            y_values = _validate_values_1d("y_values", config.get("y_values"), len(y_angles_deg))
            return cls(
                kind=kind,
                x_angles_deg=x_angles_deg,
                y_angles_deg=y_angles_deg,
                x_values=x_values,
                y_values=y_values,
            )

        values = _validate_values_2d("values", config.get("values"), len(y_angles_deg), len(x_angles_deg))
        return cls(
            kind=kind,
            x_angles_deg=x_angles_deg,
            y_angles_deg=y_angles_deg,
            values=values,
        )

    @classmethod
    def from_json(cls, path: str | os.PathLike[str]) -> "AntennaPatternConfig":
        import json

        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls.from_dict(data)

    @classmethod
    def coerce(cls, config: "AntennaPatternConfig | dict[str, Any] | str | os.PathLike[str]") -> "AntennaPatternConfig":
        if isinstance(config, cls):
            return config
        if isinstance(config, (str, os.PathLike)):
            return cls.from_json(config)
        if isinstance(config, dict):
            return cls.from_dict(config)
        raise TypeError("Antenna pattern config must be an AntennaPatternConfig, dict, or JSON path.")

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
    if x_query.shape != y_query.shape:
        raise ValueError("2D antenna pattern queries must have matching shapes.")
    if x_query.numel() == 0:
        return torch.empty_like(x_query, dtype=values.dtype)

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
        if vectors.ndim < 2 or vectors.shape[-1] != 3:
            raise ValueError("Antenna pattern vectors must have shape (..., 3).")

        forward = -vectors[..., 2]
        x_angles_deg = torch.rad2deg(torch.atan2(vectors[..., 0], forward))
        y_angles_deg = torch.rad2deg(torch.atan2(vectors[..., 1], forward))
        return self.evaluate_xy(x_angles_deg, y_angles_deg)
