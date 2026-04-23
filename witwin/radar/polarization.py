"""Simplified polarization configuration and runtime helpers.

Validation lives in :mod:`witwin.radar.validation`. This module defines the
dataclass and the runtime tensors used by the solvers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class PolarizationConfig:
    """Simplified polarization settings."""

    tx: tuple[tuple[float, float, float], ...]
    rx: tuple[tuple[float, float, float], ...]
    reflection_flip: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "tx": [list(vector) for vector in self.tx],
            "rx": [list(vector) for vector in self.rx],
            "reflection_flip": self.reflection_flip,
        }


def _normalize_rows(vectors: torch.Tensor) -> torch.Tensor:
    return vectors / torch.clamp(torch.linalg.norm(vectors, dim=-1, keepdim=True), min=1e-12)


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
        device: str | torch.device,
        radar,
    ) -> "PolarizationRuntime":
        tx_local = _normalize_rows(torch.tensor(config.tx, dtype=torch.float32, device=device))
        rx_local = _normalize_rows(torch.tensor(config.rx, dtype=torch.float32, device=device))
        tx_world = _normalize_rows(radar.world_from_local_vectors(tx_local))
        rx_world = _normalize_rows(radar.world_from_local_vectors(rx_local))
        return cls(
            tx_world=tx_world.contiguous(),
            rx_world=rx_world.contiguous(),
            reflection_flip=bool(config.reflection_flip),
        )
