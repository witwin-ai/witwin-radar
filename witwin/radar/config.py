"""Radar configuration dataclass.

Validation lives in :mod:`witwin.radar.validation`. This module defines the
pure data container only.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any


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
    antenna_pattern: dict[str, Any] | None = None
    noise_model: dict[str, Any] | None = None
    polarization: dict[str, Any] | None = None
    receiver_chain: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "RadarConfig":
        from .validation import validate_radar_config

        return validate_radar_config(config)

    @classmethod
    def from_json(cls, path: str | os.PathLike[str]) -> "RadarConfig":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

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
                data[name] = value
        return data
