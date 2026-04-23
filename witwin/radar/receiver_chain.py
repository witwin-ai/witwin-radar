"""Optional receiver-chain configuration for absolute-scale radar outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .noise import QuantizationNoiseConfig, quantize_complex_signal
from .utils.validators import finite_float, positive_float

_PREFIX = "Receiver chain field"


@dataclass(frozen=True)
class LNAConfig:
    gain_db: float = 0.0

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "LNAConfig":
        if not isinstance(config, dict):
            raise TypeError("LNA config must be a dict.")
        return cls(gain_db=finite_float("lna.gain_db", config.get("gain_db", 0.0), prefix=_PREFIX))

    def to_dict(self) -> dict[str, Any]:
        return {"gain_db": self.gain_db}


@dataclass(frozen=True)
class AGCConfig:
    target_rms: float
    max_gain_db: float = 60.0
    min_gain_db: float = -60.0
    mode: str = "per_rx"

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "AGCConfig":
        if not isinstance(config, dict):
            raise TypeError("AGC config must be a dict.")
        mode = str(config.get("mode", "per_rx")).lower()
        if mode not in {"global", "per_rx"}:
            raise ValueError("Receiver chain field 'agc.mode' must be 'global' or 'per_rx'.")
        max_gain_db = finite_float("agc.max_gain_db", config.get("max_gain_db", 60.0), prefix=_PREFIX)
        min_gain_db = finite_float("agc.min_gain_db", config.get("min_gain_db", -60.0), prefix=_PREFIX)
        if min_gain_db > max_gain_db:
            raise ValueError("Receiver chain AGC requires min_gain_db <= max_gain_db.")
        return cls(
            target_rms=positive_float("agc.target_rms", config.get("target_rms"), prefix=_PREFIX),
            max_gain_db=max_gain_db,
            min_gain_db=min_gain_db,
            mode=mode,
        )

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

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "ReceiverChainConfig":
        if not isinstance(config, dict):
            raise TypeError("Receiver chain config must be a dict.")
        lna = LNAConfig.from_dict(config["lna"]) if config.get("lna") is not None else None
        agc = AGCConfig.from_dict(config["agc"]) if config.get("agc") is not None else None
        adc = QuantizationNoiseConfig.from_dict(config["adc"]) if config.get("adc") is not None else None
        if lna is None and agc is None and adc is None:
            raise ValueError("Receiver chain config must enable at least one of 'lna', 'agc', or 'adc'.")
        return cls(
            reference_impedance_ohm=positive_float(
                "receiver_chain.reference_impedance_ohm",
                config.get("reference_impedance_ohm", 50.0),
                prefix=_PREFIX,
            ),
            lna=lna,
            agc=agc,
            adc=adc,
        )

    @classmethod
    def coerce(cls, config: "ReceiverChainConfig | dict[str, Any]") -> "ReceiverChainConfig":
        if isinstance(config, cls):
            return config
        if isinstance(config, dict):
            return cls.from_dict(config)
        raise TypeError("Receiver chain config must be a ReceiverChainConfig or dict.")

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
        if not torch.is_complex(signal):
            raise TypeError("ReceiverChainRuntime.apply expects a complex-valued signal tensor.")

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
