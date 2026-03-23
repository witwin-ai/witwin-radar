"""Validated radar configuration schema."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from typing import Any

from .antenna_pattern import AntennaPatternConfig
from .noise import NoiseModelConfig
from .polarization import PolarizationConfig
from .receiver_chain import ReceiverChainConfig


_REQUIRED_KEYS = (
    "num_tx",
    "num_rx",
    "fc",
    "slope",
    "adc_samples",
    "adc_start_time",
    "sample_rate",
    "idle_time",
    "ramp_end_time",
    "chirp_per_frame",
    "frame_per_second",
    "num_doppler_bins",
    "num_range_bins",
    "num_angle_bins",
    "power",
    "tx_loc",
    "rx_loc",
)


def _require_keys(config: dict[str, Any]) -> None:
    missing = [key for key in _REQUIRED_KEYS if key not in config]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Radar config is missing required keys: {joined}")


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Radar config field '{name}' must be a positive int.")
    return value


def _finite_float(name: str, value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Radar config field '{name}' must be a finite float.") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"Radar config field '{name}' must be a finite float.")
    return parsed


def _validate_locations(name: str, value: Any, expected_count: int) -> tuple[tuple[float, float, float], ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"Radar config field '{name}' must be a sequence of 3D coordinates.")
    if len(value) != expected_count:
        raise ValueError(
            f"Radar config field '{name}' must contain exactly {expected_count} entries; got {len(value)}."
        )

    coords: list[tuple[float, float, float]] = []
    for index, coord in enumerate(value):
        if not isinstance(coord, (list, tuple)) or len(coord) != 3:
            raise ValueError(f"Radar config field '{name}[{index}]' must be a 3-element coordinate.")
        coords.append(
            (
                _finite_float(f"{name}[{index}][0]", coord[0]),
                _finite_float(f"{name}[{index}][1]", coord[1]),
                _finite_float(f"{name}[{index}][2]", coord[2]),
            )
        )
    return tuple(coords)


@dataclass(frozen=True)
class RadarConfig:
    num_tx: int
    num_rx: int
    fc: float
    slope: float
    adc_samples: int
    adc_start_time: float
    sample_rate: float
    idle_time: float
    ramp_end_time: float
    chirp_per_frame: int
    frame_per_second: float
    num_doppler_bins: int
    num_range_bins: int
    num_angle_bins: int
    power: float
    tx_loc: tuple[tuple[float, float, float], ...]
    rx_loc: tuple[tuple[float, float, float], ...]
    antenna_pattern: AntennaPatternConfig | None = None
    noise_model: NoiseModelConfig | None = None
    polarization: PolarizationConfig | None = None
    receiver_chain: ReceiverChainConfig | None = None

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "RadarConfig":
        if not isinstance(config, dict):
            raise TypeError("Radar config must be a dict.")
        _require_keys(config)

        num_tx = _positive_int("num_tx", config["num_tx"])
        num_rx = _positive_int("num_rx", config["num_rx"])

        return cls(
            num_tx=num_tx,
            num_rx=num_rx,
            fc=_finite_float("fc", config["fc"]),
            slope=_finite_float("slope", config["slope"]),
            adc_samples=_positive_int("adc_samples", config["adc_samples"]),
            adc_start_time=_finite_float("adc_start_time", config["adc_start_time"]),
            sample_rate=_finite_float("sample_rate", config["sample_rate"]),
            idle_time=_finite_float("idle_time", config["idle_time"]),
            ramp_end_time=_finite_float("ramp_end_time", config["ramp_end_time"]),
            chirp_per_frame=_positive_int("chirp_per_frame", config["chirp_per_frame"]),
            frame_per_second=_finite_float("frame_per_second", config["frame_per_second"]),
            num_doppler_bins=_positive_int("num_doppler_bins", config["num_doppler_bins"]),
            num_range_bins=_positive_int("num_range_bins", config["num_range_bins"]),
            num_angle_bins=_positive_int("num_angle_bins", config["num_angle_bins"]),
            power=_finite_float("power", config["power"]),
            tx_loc=_validate_locations("tx_loc", config["tx_loc"], num_tx),
            rx_loc=_validate_locations("rx_loc", config["rx_loc"], num_rx),
            antenna_pattern=(
                AntennaPatternConfig.coerce(config["antenna_pattern"])
                if config.get("antenna_pattern") is not None
                else None
            ),
            noise_model=(
                NoiseModelConfig.coerce(config["noise_model"])
                if config.get("noise_model") is not None
                else None
            ),
            polarization=(
                PolarizationConfig.coerce(config["polarization"], num_tx=num_tx, num_rx=num_rx)
                if config.get("polarization") is not None
                else None
            ),
            receiver_chain=(
                ReceiverChainConfig.coerce(config["receiver_chain"])
                if config.get("receiver_chain") is not None
                else None
            ),
        )

    @classmethod
    def from_json(cls, path: str | os.PathLike[str]) -> "RadarConfig":
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls.from_dict(data)

    @classmethod
    def coerce(cls, config: "RadarConfig | dict[str, Any] | str | os.PathLike[str]") -> "RadarConfig":
        if isinstance(config, cls):
            return config
        if isinstance(config, (str, os.PathLike)):
            return cls.from_json(config)
        if isinstance(config, dict):
            return cls.from_dict(config)
        raise TypeError("Radar config must be a RadarConfig, dict, or JSON path.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_tx": self.num_tx,
            "num_rx": self.num_rx,
            "fc": self.fc,
            "slope": self.slope,
            "adc_samples": self.adc_samples,
            "adc_start_time": self.adc_start_time,
            "sample_rate": self.sample_rate,
            "idle_time": self.idle_time,
            "ramp_end_time": self.ramp_end_time,
            "chirp_per_frame": self.chirp_per_frame,
            "frame_per_second": self.frame_per_second,
            "num_doppler_bins": self.num_doppler_bins,
            "num_range_bins": self.num_range_bins,
            "num_angle_bins": self.num_angle_bins,
            "power": self.power,
            "tx_loc": [list(coord) for coord in self.tx_loc],
            "rx_loc": [list(coord) for coord in self.rx_loc],
            **(
                {"antenna_pattern": self.antenna_pattern.to_dict()}
                if self.antenna_pattern is not None
                else {}
            ),
            **(
                {"noise_model": self.noise_model.to_dict()}
                if self.noise_model is not None
                else {}
            ),
            **(
                {"polarization": self.polarization.to_dict()}
                if self.polarization is not None
                else {}
            ),
            **(
                {"receiver_chain": self.receiver_chain.to_dict()}
                if self.receiver_chain is not None
                else {}
            ),
        }
