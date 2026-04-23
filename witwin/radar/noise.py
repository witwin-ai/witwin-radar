"""Radar noise model configuration and runtime application.

Validation lives in :mod:`witwin.radar.validation`. This module defines the
dataclasses and the runtime that injects noise into complex signal tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class ThermalNoiseConfig:
    std: float

    def to_dict(self) -> dict[str, Any]:
        return {"std": self.std}


@dataclass(frozen=True)
class QuantizationNoiseConfig:
    bits: int
    full_scale: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {"bits": self.bits, "full_scale": self.full_scale}


@dataclass(frozen=True)
class PhaseNoiseConfig:
    std: float

    def to_dict(self) -> dict[str, Any]:
        return {"std": self.std}


@dataclass(frozen=True)
class NoiseModelConfig:
    thermal: ThermalNoiseConfig | None = None
    quantization: QuantizationNoiseConfig | None = None
    phase: PhaseNoiseConfig | None = None
    seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        if self.thermal is not None:
            data["thermal"] = self.thermal.to_dict()
        if self.quantization is not None:
            data["quantization"] = self.quantization.to_dict()
        if self.phase is not None:
            data["phase"] = self.phase.to_dict()
        if self.seed is not None:
            data["seed"] = self.seed
        return data


def _component_dtype(signal: torch.Tensor) -> torch.dtype:
    return torch.float64 if signal.dtype == torch.complex128 else torch.float32


def _randn(shape, *, device, dtype, generator: torch.Generator | None) -> torch.Tensor:
    if generator is None:
        return torch.randn(shape, device=device, dtype=dtype)
    return torch.randn(shape, device=device, dtype=dtype, generator=generator)


def quantize_complex_signal(signal: torch.Tensor, *, bits: int, full_scale: float) -> torch.Tensor:
    levels = 2 ** bits
    step = (2.0 * full_scale) / (levels - 1)

    def _quantize(component: torch.Tensor) -> torch.Tensor:
        clipped = torch.clamp(component, min=-full_scale, max=full_scale)
        code = torch.round((clipped + full_scale) / step)
        return code * step - full_scale

    real = _quantize(signal.real)
    imag = _quantize(signal.imag)
    return torch.complex(real, imag).to(dtype=signal.dtype)


@dataclass(frozen=True)
class NoiseModelRuntime:
    config: NoiseModelConfig
    device: str

    @classmethod
    def from_config(cls, config: NoiseModelConfig, *, device: str) -> "NoiseModelRuntime":
        return cls(config=config, device=device)

    def apply(self, signal: torch.Tensor, *, generator: torch.Generator | None = None) -> torch.Tensor:
        noisy = signal
        if self.config.phase is not None and self.config.phase.std > 0.0:
            noisy = self._apply_phase_noise(noisy, std=self.config.phase.std, generator=generator)
        if self.config.thermal is not None and self.config.thermal.std > 0.0:
            noisy = self._apply_thermal_noise(noisy, std=self.config.thermal.std, generator=generator)
        if self.config.quantization is not None:
            noisy = quantize_complex_signal(
                noisy,
                bits=self.config.quantization.bits,
                full_scale=self.config.quantization.full_scale,
            )
        return noisy

    def _apply_phase_noise(
        self,
        signal: torch.Tensor,
        *,
        std: float,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        real = _component_dtype(signal)
        if signal.ndim == 4:
            phase_shape = signal.shape[-2:]
            broadcast_shape = (1, 1, *phase_shape)
        elif signal.ndim in (1, 2):
            phase_shape = signal.shape
            broadcast_shape = phase_shape
        else:
            raise ValueError("Phase noise currently supports chirp (T,), frame (F, T), or mimo (TX, RX, F, T) tensors.")

        innovations = _randn(phase_shape, device=signal.device, dtype=real, generator=generator) * std
        phase = torch.cumsum(innovations.reshape(-1), dim=0).reshape(phase_shape).reshape(broadcast_shape)
        phase_factor = torch.polar(torch.ones_like(phase, dtype=real), phase)
        return signal * phase_factor.to(dtype=signal.dtype)

    def _apply_thermal_noise(
        self,
        signal: torch.Tensor,
        *,
        std: float,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        real = _component_dtype(signal)
        real_part = _randn(signal.shape, device=signal.device, dtype=real, generator=generator) * std
        imag_part = _randn(signal.shape, device=signal.device, dtype=real, generator=generator) * std
        noise = torch.complex(real_part, imag_part).to(dtype=signal.dtype)
        return signal + noise
