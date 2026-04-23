from __future__ import annotations

import math

import pytest
import torch

from witwin.radar import Radar, RadarConfig, Sensor


def _config() -> dict:
    return {
        "num_tx": 1,
        "num_rx": 1,
        "fc": 77e9,
        "slope": 60.012,
        "adc_samples": 64,
        "adc_start_time": 0,
        "sample_rate": 4400,
        "idle_time": 7,
        "ramp_end_time": 58,
        "chirp_per_frame": 1,
        "frame_per_second": 10,
        "num_doppler_bins": 1,
        "num_range_bins": 64,
        "num_angle_bins": 8,
        "power": 12,
        "tx_loc": [[0, 0, 0]],
        "rx_loc": [[0, 0, 0]],
    }


def _signal_peak(radar: Radar, point_world: torch.Tensor) -> float:
    point_world = point_world.to(dtype=torch.float32, device=radar.device)

    def interp(_t):
        return (
            torch.tensor([1.0], dtype=torch.float32, device=radar.device),
            point_world.unsqueeze(0),
        )

    return float(radar.mimo(interp).abs().max().item())


def _local_target(x_deg: float, y_deg: float, radius: float = 2.0) -> torch.Tensor:
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


def _half_wave_dipole_power(angle_deg: float) -> float:
    angle_rad = math.radians(angle_deg)
    cos_angle = math.cos(angle_rad)
    if abs(cos_angle) < 1e-8:
        return 0.0
    field = math.cos(0.5 * math.pi * math.sin(angle_rad)) / cos_angle
    return field * field


def test_sensor_transforms_local_points_and_vectors():
    sensor = Sensor(origin=(1.0, 2.0, 3.0), target=(2.0, 2.0, 3.0), up=(0.0, 1.0, 0.0))
    local_points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -2.0],
        ],
        dtype=torch.float32,
    )

    world_points = sensor.world_from_local_points(local_points)
    expected_points = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 4.0],
            [1.0, 3.0, 3.0],
            [3.0, 2.0, 3.0],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(world_points, expected_points)

    world_forward = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    local_forward = sensor.local_from_world_vectors(world_forward)
    assert torch.allclose(local_forward, torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32), atol=1e-6, rtol=1e-6)


def test_radar_world_positions_follow_sensor_pose():
    sensor = Sensor(origin=(1.0, 0.0, 0.0), target=(2.0, 0.0, 0.0), up=(0.0, 1.0, 0.0))
    radar = Radar(
        RadarConfig.from_dict({
            **_config(),
            "num_tx": 2,
            "tx_loc": [[0, 0, 0], [2, 0, 0]],
        }),
        backend="pytorch",
        device="cpu",
        sensor=sensor,
    )
    spacing = radar._lambda / 2.0

    expected = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 2.0 * spacing],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(radar.tx_pos.cpu(), expected, atol=1e-6, rtol=1e-6)


def test_rotated_and_translated_radar_matches_same_local_geometry_signal():
    identity = Sensor.identity()
    moved = Sensor(origin=(1.5, -0.25, 0.5), target=(2.5, -0.25, 0.5), up=(0.0, 1.0, 0.0))

    radar_identity = Radar(RadarConfig.from_dict(_config()), backend="pytorch", device="cpu", sensor=identity)
    radar_moved = Radar(RadarConfig.from_dict(_config()), backend="pytorch", device="cpu", sensor=moved)

    target_local = torch.tensor([[0.0, 0.0, -2.0]], dtype=torch.float32)
    target_identity = identity.world_from_local_points(target_local).squeeze(0)
    target_moved = moved.world_from_local_points(target_local).squeeze(0)

    peak_identity = _signal_peak(radar_identity, target_identity)
    peak_moved = _signal_peak(radar_moved, target_moved)

    assert peak_moved == pytest.approx(peak_identity, rel=1e-6, abs=1e-6)


def test_rotated_radar_pattern_is_evaluated_in_local_frame():
    sensor = Sensor(origin=(0.0, 0.0, 0.0), target=(1.0, 0.0, 0.0), up=(0.0, 1.0, 0.0))
    radar = Radar(RadarConfig.from_dict(_config()), backend="pytorch", device="cpu", sensor=sensor)

    center_world = sensor.world_from_local_points(_local_target(0.0, 0.0).unsqueeze(0)).squeeze(0)
    off_axis_world = sensor.world_from_local_points(_local_target(45.0, 0.0).unsqueeze(0)).squeeze(0)

    center_peak = _signal_peak(radar, center_world)
    off_axis_peak = _signal_peak(radar, off_axis_world)

    assert off_axis_peak / center_peak == pytest.approx(_half_wave_dipole_power(45.0), rel=5e-3, abs=5e-3)
