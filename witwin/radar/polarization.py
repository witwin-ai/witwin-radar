"""Simplified polarization configuration and runtime helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .sensor import Sensor
from .utils.vector import normalize_rows, parse_vector3

_PREFIX = "Polarization field"
_ALIASES = {
    "horizontal": (1.0, 0.0, 0.0),
    "h": (1.0, 0.0, 0.0),
    "vertical": (0.0, 1.0, 0.0),
    "v": (0.0, 1.0, 0.0),
}


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
        vector = parse_vector3(name, value, prefix=_PREFIX, aliases=_ALIASES)
        return tuple(vector for _ in range(expected_count))
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{_PREFIX} '{name}' must be a vector or a sequence of {expected_count} vectors.")
    if len(value) != expected_count:
        raise ValueError(
            f"{_PREFIX} '{name}' must contain exactly {expected_count} entries; got {len(value)}."
        )
    return tuple(parse_vector3(f"{name}[{index}]", entry, prefix=_PREFIX, aliases=_ALIASES) for index, entry in enumerate(value))


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
        tx_local = normalize_rows(torch.tensor(config.tx, dtype=torch.float32, device=device))
        rx_local = normalize_rows(torch.tensor(config.rx, dtype=torch.float32, device=device))
        tx_world = normalize_rows(sensor.world_from_local_vectors(tx_local))
        rx_world = normalize_rows(sensor.world_from_local_vectors(rx_local))
        return cls(
            tx_world=tx_world.contiguous(),
            rx_world=rx_world.contiguous(),
            reflection_flip=bool(config.reflection_flip),
        )
