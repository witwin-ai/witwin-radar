from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from witwin.radar import Radar, RadarConfig
from witwin.radar.solvers._runtime import (
    compute_path_amplitudes,
    compute_total_path_lengths,
    normalize_interpolated_sample,
)


def _tiny_config_dict() -> dict:
    return {
        "num_tx": 1,
        "num_rx": 1,
        "fc": 77e9,
        "slope": 60.012,
        "adc_samples": 8,
        "adc_start_time": 0,
        "sample_rate": 4400,
        "idle_time": 7,
        "ramp_end_time": 58,
        "chirp_per_frame": 1,
        "frame_per_second": 10,
        "num_doppler_bins": 1,
        "num_range_bins": 8,
        "num_angle_bins": 8,
        "power": 12,
        "tx_loc": [[0, 0, 0]],
        "rx_loc": [[0, 0, 0]],
    }


def _radar() -> Radar:
    return Radar(RadarConfig.from_dict(_tiny_config_dict()), backend="pytorch", device="cpu")


def _radar_with_pattern(pattern) -> Radar:
    config = _tiny_config_dict()
    config["antenna_pattern"] = pattern
    return Radar(RadarConfig.from_dict(config), backend="pytorch", device="cpu")


def test_normalize_interpolated_sample_accepts_legacy_tuple():
    intensities = torch.tensor([0.5, 0.25], dtype=torch.float32)
    points = torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -2.0]], dtype=torch.float32)

    sample = normalize_interpolated_sample((intensities, points), device="cpu")

    assert torch.equal(sample.points, points)
    assert torch.equal(sample.entry_points, points)
    assert torch.equal(sample.fixed_path_lengths, torch.zeros(2, dtype=torch.float32))
    assert torch.equal(sample.depths, torch.zeros(2, dtype=torch.int32))


def test_normalize_interpolated_sample_preserves_rich_fields():
    sample = normalize_interpolated_sample(
        SimpleNamespace(
            intensities=torch.tensor([0.5], dtype=torch.float32),
            points=torch.tensor([[0.0, 0.0, -3.0]], dtype=torch.float32),
            entry_points=torch.tensor([[0.0, 0.0, -2.0]], dtype=torch.float32),
            fixed_path_lengths=torch.tensor([1.25], dtype=torch.float32),
            depths=torch.tensor([1], dtype=torch.int32),
            normals=torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32),
        ),
        device="cpu",
    )

    assert torch.equal(sample.entry_points, torch.tensor([[0.0, 0.0, -2.0]], dtype=torch.float32))
    assert torch.equal(sample.fixed_path_lengths, torch.tensor([1.25], dtype=torch.float32))
    assert torch.equal(sample.depths, torch.tensor([1], dtype=torch.int32))
    assert torch.equal(sample.normals, torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32))


def test_total_path_length_uses_entry_fixed_and_exit_segments():
    tx_pos = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    rx_pos = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    sample = normalize_interpolated_sample(
        SimpleNamespace(
            intensities=torch.tensor([1.0], dtype=torch.float32),
            points=torch.tensor([[0.0, 0.0, -3.0]], dtype=torch.float32),
            entry_points=torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32),
            fixed_path_lengths=torch.tensor([0.5], dtype=torch.float32),
            depths=torch.tensor([1], dtype=torch.int32),
        ),
        device="cpu",
    )

    total = compute_total_path_lengths(sample, tx_pos, rx_pos)
    assert total.shape == (1, 1, 1)
    assert torch.allclose(total.squeeze(), torch.tensor(4.5, dtype=torch.float32))


def test_path_amplitudes_include_fspl():
    radar = _radar()
    sample = normalize_interpolated_sample(
        SimpleNamespace(
            intensities=torch.tensor([0.25], dtype=torch.float32),
            points=torch.tensor([[0.0, 0.0, -2.0]], dtype=torch.float32),
            entry_points=torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32),
            fixed_path_lengths=torch.tensor([1.0], dtype=torch.float32),
            depths=torch.tensor([1], dtype=torch.int32),
            normals=torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32),
        ),
        device="cpu",
    )
    total = torch.tensor([[[4.0]]], dtype=torch.float32)

    amplitudes = compute_path_amplitudes(radar, sample, total)
    expected = radar.gain * torch.sqrt(torch.tensor(0.25)) * (radar._lambda / (4.0 * torch.pi * 4.0))

    assert amplitudes.shape == (1, 1, 1)
    assert torch.allclose(amplitudes.squeeze(), expected.to(dtype=torch.float32))


def test_path_amplitudes_apply_separable_antenna_pattern():
    radar = _radar_with_pattern(
        {
            "kind": "separable",
            "x_angles_deg": [0, 45, 90],
            "x_values": [1.0, 0.5, 0.0],
            "y_angles_deg": [0, 90],
            "y_values": [1.0, 1.0],
        }
    )
    sample = normalize_interpolated_sample(
        SimpleNamespace(
            intensities=torch.tensor([1.0, 1.0], dtype=torch.float32),
            points=torch.tensor([[0.0, 0.0, -1.0], [1.0, 0.0, -1.0]], dtype=torch.float32),
            entry_points=torch.tensor([[0.0, 0.0, -1.0], [1.0, 0.0, -1.0]], dtype=torch.float32),
            fixed_path_lengths=torch.zeros(2, dtype=torch.float32),
            depths=torch.zeros(2, dtype=torch.int32),
            normals=torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=torch.float32),
        ),
        device="cpu",
    )
    total = torch.tensor([[[2.0, 2.0]]], dtype=torch.float32)

    amplitudes = compute_path_amplitudes(radar, sample, total).reshape(-1)
    base = torch.tensor(radar.gain * (radar._lambda / (4.0 * torch.pi * 2.0)), dtype=torch.float32)

    assert torch.allclose(amplitudes[0], base)
    assert torch.allclose(amplitudes[1], 0.5 * base, atol=1e-7, rtol=1e-6)


def test_path_amplitudes_apply_2d_antenna_pattern_map():
    radar = _radar_with_pattern(
        {
            "kind": "map",
            "x_angles_deg": [0, 45],
            "y_angles_deg": [0, 30],
            "values": [
                [1.0, 0.6],
                [0.8, 0.4],
            ],
        }
    )
    sample = normalize_interpolated_sample(
        SimpleNamespace(
            intensities=torch.tensor([1.0], dtype=torch.float32),
            points=torch.tensor(
                [[math.tan(math.radians(22.5)), math.tan(math.radians(15.0)), -1.0]],
                dtype=torch.float32,
            ),
            entry_points=torch.tensor(
                [[math.tan(math.radians(22.5)), math.tan(math.radians(15.0)), -1.0]],
                dtype=torch.float32,
            ),
            fixed_path_lengths=torch.zeros(1, dtype=torch.float32),
            depths=torch.zeros(1, dtype=torch.int32),
            normals=torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32),
        ),
        device="cpu",
    )
    total = torch.tensor([[[2.0]]], dtype=torch.float32)

    amplitudes = compute_path_amplitudes(radar, sample, total)
    base = torch.tensor(radar.gain * (radar._lambda / (4.0 * torch.pi * 2.0)), dtype=torch.float32)

    assert torch.allclose(amplitudes.squeeze(), 0.7 * base, atol=1e-7, rtol=1e-6)


def test_path_amplitudes_zero_outside_pattern_support():
    radar = _radar_with_pattern(
        {
            "kind": "separable",
            "x_angles_deg": [-45, 45],
            "x_values": [1.0, 1.0],
            "y_angles_deg": [-45, 45],
            "y_values": [1.0, 1.0],
        }
    )
    sample = normalize_interpolated_sample(
        SimpleNamespace(
            intensities=torch.tensor([1.0], dtype=torch.float32),
            points=torch.tensor([[2.0, 0.0, -1.0]], dtype=torch.float32),
            entry_points=torch.tensor([[2.0, 0.0, -1.0]], dtype=torch.float32),
            fixed_path_lengths=torch.zeros(1, dtype=torch.float32),
            depths=torch.zeros(1, dtype=torch.int32),
            normals=torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32),
        ),
        device="cpu",
    )
    total = torch.tensor([[[2.0]]], dtype=torch.float32)

    amplitudes = compute_path_amplitudes(radar, sample, total)
    assert torch.equal(amplitudes, torch.zeros_like(amplitudes))


def test_pytorch_mimo_respects_antenna_pattern_gain():
    radar = _radar_with_pattern(
        {
            "kind": "separable",
            "x_angles_deg": [0, 45, 90],
            "x_values": [1.0, 0.5, 0.0],
            "y_angles_deg": [0, 90],
            "y_values": [1.0, 1.0],
        }
    )

    def interp_center(_t):
        return (
            torch.tensor([1.0], dtype=torch.float32, device=radar.device),
            torch.tensor([[0.0, 0.0, -2.0]], dtype=torch.float32, device=radar.device),
        )

    off_axis = 2.0 / math.sqrt(2.0)

    def interp_off_axis(_t):
        return (
            torch.tensor([1.0], dtype=torch.float32, device=radar.device),
            torch.tensor([[off_axis, 0.0, -off_axis]], dtype=torch.float32, device=radar.device),
        )

    center_peak = radar.mimo(interp_center).abs().max()
    off_axis_peak = radar.mimo(interp_off_axis).abs().max()

    assert torch.allclose(off_axis_peak / center_peak, torch.tensor(0.5, dtype=center_peak.dtype), atol=1e-3, rtol=1e-3)


def test_path_amplitudes_apply_simplified_polarization_projection():
    radar = Radar(
        RadarConfig.from_dict({
            **_tiny_config_dict(),
            "polarization": {
                "tx": "horizontal",
                "rx": "vertical",
            },
        }),
        backend="pytorch",
        device="cpu",
    )
    sample = normalize_interpolated_sample(
        SimpleNamespace(
            intensities=torch.tensor([1.0], dtype=torch.float32),
            points=torch.tensor([[0.0, 0.0, -2.0]], dtype=torch.float32),
            entry_points=torch.tensor([[0.0, 0.0, -2.0]], dtype=torch.float32),
            fixed_path_lengths=torch.zeros(1, dtype=torch.float32),
            depths=torch.zeros(1, dtype=torch.int32),
            normals=torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
        ),
        device="cpu",
    )
    total = torch.tensor([[[4.0]]], dtype=torch.float32)

    amplitude = compute_path_amplitudes(radar, sample, total).squeeze()
    base = radar.gain * (radar._lambda / (4.0 * torch.pi * 4.0))

    assert torch.allclose(amplitude, torch.tensor(-2.0 / 3.0 * base, dtype=torch.float32), atol=1e-6, rtol=1e-6)


def test_pytorch_mimo_respects_simplified_polarization_gain():
    config = _tiny_config_dict()
    config["polarization"] = {"tx": "horizontal", "rx": "horizontal"}
    radar_hh = Radar(RadarConfig.from_dict(config), backend="pytorch", device="cpu")

    config_cross = dict(config)
    config_cross["polarization"] = {"tx": "horizontal", "rx": "vertical"}
    radar_hv = Radar(RadarConfig.from_dict(config_cross), backend="pytorch", device="cpu")

    position = torch.tensor([[0.0, 0.0, -2.0]], dtype=torch.float32)
    intensity = torch.tensor([1.0], dtype=torch.float32)
    normals = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32)

    def interp_hh(_t):
        return SimpleNamespace(intensities=intensity, points=position, normals=normals)

    def interp_hv(_t):
        return SimpleNamespace(intensities=intensity, points=position, normals=normals)

    hh_peak = radar_hh.mimo(interp_hh).abs().max()
    hv_peak = radar_hv.mimo(interp_hv).abs().max()

    assert torch.allclose(hv_peak / hh_peak, torch.tensor(2.0, dtype=hh_peak.dtype), atol=1e-3, rtol=1e-3)


def test_polarization_requires_surface_normals():
    radar = Radar(
        RadarConfig.from_dict({
            **_tiny_config_dict(),
            "polarization": {
                "tx": "horizontal",
                "rx": "vertical",
            },
        }),
        backend="pytorch",
        device="cpu",
    )

    def interp(_t):
        return (
            torch.tensor([1.0], dtype=torch.float32),
            torch.tensor([[0.0, 0.0, -2.0]], dtype=torch.float32),
        )

    with pytest.raises(ValueError, match="surface normals"):
        radar.mimo(interp)


def test_pytorch_frame_supports_pattern_with_multi_tx_config():
    radar = Radar(
        RadarConfig.from_dict({
            "num_tx": 3,
            "num_rx": 4,
            "fc": 77e9,
            "slope": 60.012,
            "adc_samples": 16,
            "adc_start_time": 0,
            "sample_rate": 4400,
            "idle_time": 7,
            "ramp_end_time": 58,
            "chirp_per_frame": 2,
            "frame_per_second": 10,
            "num_doppler_bins": 2,
            "num_range_bins": 16,
            "num_angle_bins": 8,
            "power": 12,
            "tx_loc": [[0, 0, 0], [2, 0, 0], [0, 1, 0]],
            "rx_loc": [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]],
            "antenna_pattern": {
                "kind": "separable",
                "x_angles_deg": [-60, 0, 60],
                "x_values": [0.2, 1.0, 0.2],
                "y_angles_deg": [-30, 0, 30],
                "y_values": [0.5, 1.0, 0.5],
            },
        }),
        backend="pytorch",
        device="cpu",
    )

    def interp(_t):
        return (
            torch.tensor([1.0], dtype=torch.float32, device=radar.device),
            torch.tensor([[0.0, 0.0, -2.0]], dtype=torch.float32, device=radar.device),
        )

    frame = radar.frame(interp)
    assert frame.shape == (2, 16)
    assert frame.abs().max() > 0
