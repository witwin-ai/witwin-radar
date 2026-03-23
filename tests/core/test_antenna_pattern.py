from __future__ import annotations

import math

import pytest
import torch

from witwin.radar import Radar


def _base_config() -> dict:
    return {
        "num_tx": 1,
        "num_rx": 1,
        "fc": 77e9,
        "slope": 60.012,
        "adc_samples": 128,
        "adc_start_time": 0,
        "sample_rate": 4400,
        "idle_time": 7,
        "ramp_end_time": 58,
        "chirp_per_frame": 1,
        "frame_per_second": 10,
        "num_doppler_bins": 1,
        "num_range_bins": 128,
        "num_angle_bins": 16,
        "power": 12,
        "tx_loc": [[0, 0, 0]],
        "rx_loc": [[0, 0, 0]],
    }


def _make_radar(*, antenna_pattern=None) -> Radar:
    config = _base_config()
    if antenna_pattern is not None:
        config["antenna_pattern"] = antenna_pattern
    return Radar(config, backend="pytorch", device="cpu")


def _target_position(x_deg: float, y_deg: float, radius: float = 2.0) -> torch.Tensor:
    direction = torch.tensor(
        [
            math.tan(math.radians(x_deg)),
            math.tan(math.radians(y_deg)),
            -1.0,
        ],
        dtype=torch.float32,
    )
    direction = direction / torch.linalg.norm(direction)
    return direction * radius


def _signal_peak(radar: Radar, *, x_deg: float, y_deg: float, radius: float = 2.0) -> torch.Tensor:
    position = _target_position(x_deg, y_deg, radius).to(device=radar.device)

    def interp(_t):
        return (
            torch.tensor([1.0], dtype=torch.float32, device=radar.device),
            position.unsqueeze(0),
        )

    return radar.mimo(interp).abs().max()


def _half_wave_dipole_power(angle_deg: float) -> float:
    angle_rad = math.radians(angle_deg)
    cos_angle = math.cos(angle_rad)
    if abs(cos_angle) < 1e-8:
        return 0.0
    field = math.cos(0.5 * math.pi * math.sin(angle_rad)) / cos_angle
    return field * field


def _bilinear_value(
    *,
    x_deg: float,
    y_deg: float,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    v00: float,
    v10: float,
    v01: float,
    v11: float,
) -> float:
    tx = (x_deg - x0) / (x1 - x0)
    ty = (y_deg - y0) / (y1 - y0)
    return (
        (1.0 - tx) * (1.0 - ty) * v00
        + tx * (1.0 - ty) * v10
        + (1.0 - tx) * ty * v01
        + tx * ty * v11
    )


def test_missing_antenna_pattern_uses_default_dipole_runtime():
    radar = _make_radar()

    assert radar.config.antenna_pattern is None
    assert radar.antenna_pattern_config.kind == "separable"

    center_gain = radar.antenna_pattern.evaluate_xy(
        torch.tensor([0.0], dtype=torch.float32),
        torch.tensor([0.0], dtype=torch.float32),
    )
    edge_gain = radar.antenna_pattern.evaluate_xy(
        torch.tensor([85.0], dtype=torch.float32),
        torch.tensor([0.0], dtype=torch.float32),
    )

    assert torch.allclose(center_gain, torch.tensor([1.0], dtype=torch.float32), atol=1e-6, rtol=1e-6)
    assert edge_gain.item() < 0.05


@pytest.mark.parametrize("angle_deg", [0.0, 30.0, 60.0])
def test_default_dipole_signal_matches_expected_gain(angle_deg: float):
    radar = _make_radar()
    center_peak = _signal_peak(radar, x_deg=0.0, y_deg=0.0)
    off_axis_peak = _signal_peak(radar, x_deg=angle_deg, y_deg=0.0)
    measured_ratio = (off_axis_peak / center_peak).item()

    assert measured_ratio == pytest.approx(_half_wave_dipole_power(angle_deg), rel=5e-3, abs=5e-3)


def test_flat_custom_pattern_keeps_signal_constant():
    radar = _make_radar(
        antenna_pattern={
            "kind": "separable",
            "x_angles_deg": [-90, 0, 90],
            "x_values": [1.0, 1.0, 1.0],
            "y_angles_deg": [-90, 0, 90],
            "y_values": [1.0, 1.0, 1.0],
        }
    )

    center_peak = _signal_peak(radar, x_deg=0.0, y_deg=0.0)
    for angle_deg in (15.0, 45.0, 70.0):
        off_axis_peak = _signal_peak(radar, x_deg=angle_deg, y_deg=0.0)
        assert (off_axis_peak / center_peak).item() == pytest.approx(1.0, rel=5e-3, abs=5e-3)


def test_2d_map_signal_matches_bilinear_gain():
    radar = _make_radar(
        antenna_pattern={
            "kind": "map",
            "x_angles_deg": [0, 40],
            "y_angles_deg": [0, 20],
            "values": [
                [1.0, 0.8],
                [0.6, 0.2],
            ],
        }
    )

    x_deg = 20.0
    y_deg = 10.0
    center_peak = _signal_peak(radar, x_deg=0.0, y_deg=0.0)
    query_peak = _signal_peak(radar, x_deg=x_deg, y_deg=y_deg)
    measured_ratio = (query_peak / center_peak).item()

    expected = _bilinear_value(
        x_deg=x_deg,
        y_deg=y_deg,
        x0=0.0,
        x1=40.0,
        y0=0.0,
        y1=20.0,
        v00=1.0,
        v10=0.8,
        v01=0.6,
        v11=0.2,
    )
    assert measured_ratio == pytest.approx(expected, rel=5e-3, abs=5e-3)
