"""Radar noise model configuration and runtime application."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from typing import Any

import torch


def _finite_float(name: str, value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Noise model field '{name}' must be a finite float.") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"Noise model field '{name}' must be a finite float.")
    return parsed


def _non_negative_float(name: str, value: Any) -> float:
    parsed = _finite_float(name, value)
    if parsed < 0.0:
        raise ValueError(f"Noise model field '{name}' must be non-negative.")
    return parsed


def _positive_float(name: str, value: Any) -> float:
    parsed = _finite_float(name, value)
    if parsed <= 0.0:
        raise ValueError(f"Noise model field '{name}' must be positive.")
    return parsed


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Noise model field '{name}' must be a positive int.")
    return value


def _optional_seed(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("Noise model field 'seed' must be a non-negative int.")
    return value


@dataclass(frozen=True)
class ThermalNoiseConfig:
    std: float

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "ThermalNoiseConfig":
        if not isinstance(config, dict):
            raise TypeError("Thermal noise config must be a dict.")
        return cls(std=_non_negative_float("thermal.std", config.get("std")))

    def to_dict(self) -> dict[str, Any]:
        return {"std": self.std}


@dataclass(frozen=True)
class QuantizationNoiseConfig:
    bits: int
    full_scale: float = 1.0

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "QuantizationNoiseConfig":
        if not isinstance(config, dict):
            raise TypeError("Quantization noise config must be a dict.")
        return cls(
            bits=_positive_int("quantization.bits", config.get("bits")),
            full_scale=_positive_float("quantization.full_scale", config.get("full_scale", 1.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"bits": self.bits, "full_scale": self.full_scale}


@dataclass(frozen=True)
class PhaseNoiseConfig:
    std: float

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "PhaseNoiseConfig":
        if not isinstance(config, dict):
            raise TypeError("Phase noise config must be a dict.")
        return cls(std=_non_negative_float("phase.std", config.get("std")))

    def to_dict(self) -> dict[str, Any]:
        return {"std": self.std}


@dataclass(frozen=True)
class NoiseModelConfig:
    thermal: ThermalNoiseConfig | None = None
    quantization: QuantizationNoiseConfig | None = None
    phase: PhaseNoiseConfig | None = None
    seed: int | None = None

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "NoiseModelConfig":
        if not isinstance(config, dict):
            raise TypeError("Noise model config must be a dict.")

        thermal = (
            ThermalNoiseConfig.from_dict(config["thermal"])
            if config.get("thermal") is not None
            else None
        )
        quantization = (
            QuantizationNoiseConfig.from_dict(config["quantization"])
            if config.get("quantization") is not None
            else None
        )
        phase = (
            PhaseNoiseConfig.from_dict(config["phase"])
            if config.get("phase") is not None
            else None
        )
        seed = _optional_seed(config.get("seed"))

        if thermal is None and quantization is None and phase is None:
            raise ValueError(
                "Noise model config must enable at least one of 'thermal', 'quantization', or 'phase'."
            )

        return cls(
            thermal=thermal,
            quantization=quantization,
            phase=phase,
            seed=seed,
        )

    @classmethod
    def from_json(cls, path: str | os.PathLike[str]) -> "NoiseModelConfig":
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls.from_dict(data)

    @classmethod
    def coerce(cls, config: "NoiseModelConfig | dict[str, Any] | str | os.PathLike[str]") -> "NoiseModelConfig":
        if isinstance(config, cls):
            return config
        if isinstance(config, (str, os.PathLike)):
            return cls.from_json(config)
        if isinstance(config, dict):
            return cls.from_dict(config)
        raise TypeError("Noise model config must be a NoiseModelConfig, dict, or JSON path.")

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
        if not torch.is_complex(signal):
            raise TypeError("NoiseModelRuntime.apply expects a complex-valued signal tensor.")

        noisy = signal
        if self.config.phase is not None and self.config.phase.std > 0.0:
            noisy = self._apply_phase_noise(noisy, std=self.config.phase.std, generator=generator)
        if self.config.thermal is not None and self.config.thermal.std > 0.0:
            noisy = self._apply_thermal_noise(noisy, std=self.config.thermal.std, generator=generator)
        if self.config.quantization is not None:
            noisy = self._apply_quantization(
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
        real_dtype = _component_dtype(signal)
        if signal.ndim == 4:
            phase_shape = signal.shape[-2:]
            broadcast_shape = (1, 1, *phase_shape)
        elif signal.ndim == 2:
            phase_shape = signal.shape
            broadcast_shape = phase_shape
        elif signal.ndim == 1:
            phase_shape = signal.shape
            broadcast_shape = phase_shape
        else:
            raise ValueError("Phase noise currently supports chirp (T,), frame (F, T), or mimo (TX, RX, F, T) tensors.")

        innovations = _randn(phase_shape, device=signal.device, dtype=real_dtype, generator=generator) * std
        phase = torch.cumsum(innovations.reshape(-1), dim=0).reshape(phase_shape)
        phase = phase.reshape(broadcast_shape)
        phase_factor = torch.polar(torch.ones_like(phase, dtype=real_dtype), phase)
        return signal * phase_factor.to(dtype=signal.dtype)

    def _apply_thermal_noise(
        self,
        signal: torch.Tensor,
        *,
        std: float,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        real_dtype = _component_dtype(signal)
        real = _randn(signal.shape, device=signal.device, dtype=real_dtype, generator=generator) * std
        imag = _randn(signal.shape, device=signal.device, dtype=real_dtype, generator=generator) * std
        noise = torch.complex(real, imag).to(dtype=signal.dtype)
        return signal + noise

    def _apply_quantization(self, signal: torch.Tensor, *, bits: int, full_scale: float) -> torch.Tensor:
        return quantize_complex_signal(signal, bits=bits, full_scale=full_scale)
