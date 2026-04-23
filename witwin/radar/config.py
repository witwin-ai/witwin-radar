"""Validated radar configuration schema."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any

from .antenna_pattern import AntennaPatternConfig
from .noise import NoiseModelConfig
from .polarization import PolarizationConfig
from .receiver_chain import ReceiverChainConfig
from .utils.validators import finite_float, positive_int, require_keys

_PREFIX = "Radar config field"

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


def _validate_locations(name: str, value: Any, expected_count: int) -> tuple[tuple[float, float, float], ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{_PREFIX} '{name}' must be a sequence of 3D coordinates.")
    if len(value) != expected_count:
        raise ValueError(
            f"{_PREFIX} '{name}' must contain exactly {expected_count} entries; got {len(value)}."
        )

    coords: list[tuple[float, float, float]] = []
    for index, coord in enumerate(value):
        if not isinstance(coord, (list, tuple)) or len(coord) != 3:
            raise ValueError(f"{_PREFIX} '{name}[{index}]' must be a 3-element coordinate.")
        coords.append(
            (
                finite_float(f"{name}[{index}][0]", coord[0], prefix=_PREFIX),
                finite_float(f"{name}[{index}][1]", coord[1], prefix=_PREFIX),
                finite_float(f"{name}[{index}][2]", coord[2], prefix=_PREFIX),
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
        require_keys(config, _REQUIRED_KEYS, label="Radar config")

        num_tx = positive_int("num_tx", config["num_tx"], prefix=_PREFIX)
        num_rx = positive_int("num_rx", config["num_rx"], prefix=_PREFIX)

        def _ff(name: str) -> float:
            return finite_float(name, config[name], prefix=_PREFIX)

        def _pi(name: str) -> int:
            return positive_int(name, config[name], prefix=_PREFIX)

        return cls(
            num_tx=num_tx,
            num_rx=num_rx,
            fc=_ff("fc"),
            slope=_ff("slope"),
            adc_samples=_pi("adc_samples"),
            adc_start_time=_ff("adc_start_time"),
            sample_rate=_ff("sample_rate"),
            idle_time=_ff("idle_time"),
            ramp_end_time=_ff("ramp_end_time"),
            chirp_per_frame=_pi("chirp_per_frame"),
            frame_per_second=_ff("frame_per_second"),
            num_doppler_bins=_pi("num_doppler_bins"),
            num_range_bins=_pi("num_range_bins"),
            num_angle_bins=_pi("num_angle_bins"),
            power=_ff("power"),
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
        data: dict[str, Any] = {
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
        }
        for name, value in (
            ("antenna_pattern", self.antenna_pattern),
            ("noise_model", self.noise_model),
            ("polarization", self.polarization),
            ("receiver_chain", self.receiver_chain),
        ):
            if value is not None:
                data[name] = value.to_dict()
        return data
