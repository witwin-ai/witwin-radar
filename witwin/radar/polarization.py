"""Simplified polarization configuration and runtime helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch

from .sensor import Sensor


_ALIASES = {
    "horizontal": (1.0, 0.0, 0.0),
    "h": (1.0, 0.0, 0.0),
    "vertical": (0.0, 1.0, 0.0),
    "v": (0.0, 1.0, 0.0),
}


def _finite_float(name: str, value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Polarization field '{name}' must be a finite float.") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"Polarization field '{name}' must be a finite float.")
    return parsed


def _vector3(name: str, value: Any) -> tuple[float, float, float]:
    if isinstance(value, str):
        alias = _ALIASES.get(value.lower())
        if alias is None:
            raise ValueError(
                f"Polarization field '{name}' must be 'horizontal', 'vertical', or a 3-element vector."
            )
        return alias
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"Polarization field '{name}' must be a 3-element vector or alias string.")
    if len(value) != 3:
        raise ValueError(f"Polarization field '{name}' must contain exactly three values.")
    vector = tuple(_finite_float(f"{name}[{index}]", component) for index, component in enumerate(value))
    norm = math.sqrt(sum(component * component for component in vector))
    if norm <= 1e-12:
        raise ValueError(f"Polarization field '{name}' must be non-zero.")
    return vector


def _is_single_vector_spec(value: Any) -> bool:
    if isinstance(value, str):
        return True
    if not isinstance(value, (list, tuple)):
        return False
    if len(value) != 3:
        return False
    return all(not isinstance(component, (list, tuple)) for component in value)


def _vector_bank(name: str, value: Any, expected_count: int) -> tuple[tuple[float, float, float], ...]:
    if _is_single_vector_spec(value):
        vector = _vector3(name, value)
        return tuple(vector for _ in range(expected_count))
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"Polarization field '{name}' must be a vector or a sequence of {expected_count} vectors.")
    if len(value) != expected_count:
        raise ValueError(
            f"Polarization field '{name}' must contain exactly {expected_count} entries; got {len(value)}."
        )
    return tuple(_vector3(f"{name}[{index}]", entry) for index, entry in enumerate(value))


def _normalize_rows(vectors: torch.Tensor) -> torch.Tensor:
    return vectors / torch.clamp(torch.linalg.norm(vectors, dim=-1, keepdim=True), min=1e-12)


@dataclass(frozen=True)
class PolarizationConfig:
    """Validated simplified polarization settings."""

    tx: tuple[tuple[float, float, float], ...]
    rx: tuple[tuple[float, float, float], ...]
    reflection_flip: bool = True

    @classmethod
    def from_dict(cls, config: dict[str, Any], *, num_tx: int, num_rx: int) -> "PolarizationConfig":
        if not isinstance(config, dict):
            raise TypeError("Polarization config must be a dict.")
        allowed = {"tx", "rx", "reflection_flip"}
        unknown = sorted(set(config) - allowed)
        if unknown:
            raise TypeError(f"Unsupported polarization config keys: {', '.join(unknown)}")

        tx_value = config.get("tx")
        rx_value = config.get("rx")
        if tx_value is None and rx_value is None:
            raise ValueError("Polarization config must define at least one of 'tx' or 'rx'.")
        if tx_value is None:
            tx_value = rx_value
        if rx_value is None:
            rx_value = tx_value

        return cls(
            tx=_vector_bank("tx", tx_value, num_tx),
            rx=_vector_bank("rx", rx_value, num_rx),
            reflection_flip=bool(config.get("reflection_flip", True)),
        )

    @classmethod
    def coerce(
        cls,
        config: "PolarizationConfig | dict[str, Any]",
        *,
        num_tx: int,
        num_rx: int,
    ) -> "PolarizationConfig":
        if isinstance(config, cls):
            return config
        if isinstance(config, dict):
            return cls.from_dict(config, num_tx=num_tx, num_rx=num_rx)
        raise TypeError("Polarization config must be a PolarizationConfig or dict.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "tx": [list(vector) for vector in self.tx],
            "rx": [list(vector) for vector in self.rx],
            "reflection_flip": self.reflection_flip,
        }


@dataclass(frozen=True)
class PolarizationRuntime:
    """World-space normalized polarization vectors for a concrete radar pose."""

    tx_world: torch.Tensor
    rx_world: torch.Tensor
    reflection_flip: bool = True

    @classmethod
    def from_config(
        cls,
        config: PolarizationConfig,
        *,
        device: str,
        sensor: Sensor,
    ) -> "PolarizationRuntime":
        tx_local = _normalize_rows(torch.tensor(config.tx, dtype=torch.float32, device=device))
        rx_local = _normalize_rows(torch.tensor(config.rx, dtype=torch.float32, device=device))
        tx_world = _normalize_rows(sensor.world_from_local_vectors(tx_local))
        rx_world = _normalize_rows(sensor.world_from_local_vectors(rx_local))
        return cls(
            tx_world=tx_world.contiguous(),
            rx_world=rx_world.contiguous(),
            reflection_flip=bool(config.reflection_flip),
        )
