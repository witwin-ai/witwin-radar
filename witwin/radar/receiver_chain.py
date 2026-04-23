"""Optional receiver-chain runtime for absolute-scale radar outputs."""

from __future__ import annotations

from dataclasses import dataclass

import torch

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


def db_to_voltage_gain(gain_db: float) -> float:
    return 10.0 ** (float(gain_db) / 20.0)


@dataclass(frozen=True)
class ReceiverChainRuntime:
    config: dict
    device: str | torch.device

    @classmethod
    def from_config(cls, config: dict, *, device: str | torch.device) -> "ReceiverChainRuntime":
        return cls(config=config, device=device)

    def apply(self, signal: torch.Tensor) -> torch.Tensor:
        processed = signal
        lna = self.config.get("lna")
        agc = self.config.get("agc")
        adc = self.config.get("adc")
        if lna is not None:
            processed = processed * db_to_voltage_gain(lna["gain_db"])
        if agc is not None:
            processed = self._apply_agc(processed, agc)
        if adc is not None:
            processed = quantize_complex_signal(
                processed,
                bits=adc["bits"],
                full_scale=adc["full_scale"],
            )
        return processed

    def _apply_agc(self, signal: torch.Tensor, config: dict) -> torch.Tensor:
        real_dtype = signal.real.dtype
        magnitude_sq = signal.real.square() + signal.imag.square()

        if signal.ndim == 4 and config["mode"] == "per_rx":
            rms = torch.sqrt(torch.clamp(magnitude_sq.mean(dim=(0, 2, 3), keepdim=True), min=1e-24))
            target = torch.tensor(config["target_rms"], dtype=real_dtype, device=signal.device).view(1, 1, 1, 1)
        else:
            rms = torch.sqrt(torch.clamp(magnitude_sq.mean(), min=1e-24))
            target = torch.tensor(config["target_rms"], dtype=real_dtype, device=signal.device)

        gain = target / rms
        min_gain = db_to_voltage_gain(config["min_gain_db"])
        max_gain = db_to_voltage_gain(config["max_gain_db"])
        gain = torch.clamp(gain, min=min_gain, max=max_gain)
        return signal * gain.to(dtype=signal.dtype)
