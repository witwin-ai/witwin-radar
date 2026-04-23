"""Optional receiver-chain configuration for absolute-scale radar outputs.

Validation lives in :mod:`witwin.radar.validation`. This module defines the
dataclasses and the runtime that applies LNA / AGC / ADC to complex signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .noise import QuantizationNoiseConfig, quantize_complex_signal


@dataclass(frozen=True)
class LNAConfig:
    gain_db: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"gain_db": self.gain_db}


@dataclass(frozen=True)
class AGCConfig:
    target_rms: float
    max_gain_db: float = 60.0
    min_gain_db: float = -60.0
    mode: str = "per_rx"

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_rms": self.target_rms,
            "max_gain_db": self.max_gain_db,
            "min_gain_db": self.min_gain_db,
            "mode": self.mode,
        }


@dataclass(frozen=True)
class ReceiverChainConfig:
    reference_impedance_ohm: float = 50.0
    lna: LNAConfig | None = None
    agc: AGCConfig | None = None
    adc: QuantizationNoiseConfig | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {"reference_impedance_ohm": self.reference_impedance_ohm}
        if self.lna is not None:
            data["lna"] = self.lna.to_dict()
        if self.agc is not None:
            data["agc"] = self.agc.to_dict()
        if self.adc is not None:
            data["adc"] = self.adc.to_dict()
        return data


def db_to_voltage_gain(gain_db: float) -> float:
    return 10.0 ** (float(gain_db) / 20.0)


@dataclass(frozen=True)
class ReceiverChainRuntime:
    config: ReceiverChainConfig
    device: str

    @classmethod
    def from_config(cls, config: ReceiverChainConfig, *, device: str) -> "ReceiverChainRuntime":
        return cls(config=config, device=device)

    def apply(self, signal: torch.Tensor) -> torch.Tensor:
        processed = signal
        if self.config.lna is not None:
            processed = processed * db_to_voltage_gain(self.config.lna.gain_db)
        if self.config.agc is not None:
            processed = self._apply_agc(processed, self.config.agc)
        if self.config.adc is not None:
            processed = quantize_complex_signal(
                processed,
                bits=self.config.adc.bits,
                full_scale=self.config.adc.full_scale,
            )
        return processed

    def _apply_agc(self, signal: torch.Tensor, config: AGCConfig) -> torch.Tensor:
        real_dtype = signal.real.dtype
        magnitude_sq = signal.real.square() + signal.imag.square()

        if signal.ndim == 4 and config.mode == "per_rx":
            rms = torch.sqrt(torch.clamp(magnitude_sq.mean(dim=(0, 2, 3), keepdim=True), min=1e-24))
            target = torch.tensor(config.target_rms, dtype=real_dtype, device=signal.device).view(1, 1, 1, 1)
        else:
            rms = torch.sqrt(torch.clamp(magnitude_sq.mean(), min=1e-24))
            target = torch.tensor(config.target_rms, dtype=real_dtype, device=signal.device)

        gain = target / rms
        min_gain = db_to_voltage_gain(config.min_gain_db)
        max_gain = db_to_voltage_gain(config.max_gain_db)
        gain = torch.clamp(gain, min=min_gain, max=max_gain)
        return signal * gain.to(dtype=signal.dtype)
