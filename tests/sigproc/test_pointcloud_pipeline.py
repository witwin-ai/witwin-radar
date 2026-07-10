from __future__ import annotations

import numpy as np
import pytest
import torch

from conftest import STANDARD_CONFIG
from witwin.radar import Radar, RadarConfig
from witwin.radar.sigproc import (
    PointCloudProcessConfig,
    clutter_removal,
    doppler_fft,
    frame2pointcloud,
    process_pc,
    process_pc_tensor,
    process_rd,
    process_rd_tensor,
    range_fft,
)
from witwin.radar.sigproc.cfar import ca_cfar_2d_fast


CPU_SIGPROC_CONFIG = {
    **STANDARD_CONFIG,
    "adc_start_time": 0,
    "adc_samples": 64,
    "chirp_per_frame": 16,
    "num_doppler_bins": 16,
    "num_range_bins": 64,
    "num_angle_bins": 32,
}


def _make_cpu_radar() -> Radar:
    return Radar(RadarConfig.from_dict(CPU_SIGPROC_CONFIG), device="cpu")


def _static_interpolator(radar: Radar, position=(0.0, 0.0, -3.0), intensity=1.0):
    position_tensor = torch.tensor([position], dtype=torch.float32, device=radar.device)
    intensity_tensor = torch.tensor([intensity], dtype=torch.float32, device=radar.device)

    def interp(_t):
        return intensity_tensor, position_tensor

    return interp


def _synthetic_frame(radar: Radar) -> torch.Tensor:
    adc = radar.config.adc_samples
    chirps = radar.config.chirp_per_frame
    samples = torch.arange(adc, dtype=torch.float32, device=radar.device)
    slow_time = torch.arange(chirps, dtype=torch.float32, device=radar.device)
    range_tone = torch.exp(2j * torch.pi * 5.0 * samples / adc)
    doppler_tone = torch.exp(2j * torch.pi * 2.0 * slow_time / chirps)
    frame = doppler_tone[:, None] * range_tone[None, :]
    return frame.to(torch.complex64).reshape(1, 1, chirps, adc).repeat(
        radar.config.num_tx,
        radar.config.num_rx,
        1,
        1,
    )


def test_frame2pointcloud_requires_radar():
    radar = _make_cpu_radar()
    frame = _synthetic_frame(radar)
    cfg = PointCloudProcessConfig(radar)

    with pytest.raises(ValueError, match="requires a radar instance"):
        frame2pointcloud(frame, cfg)


def test_range_fft_peak_at_known_bin():
    """Sanity: a torch frame round-trips through range_fft with expected peak."""
    adc = 16
    fc = type("FC", (), {"numADCSamples": adc})()
    target_bin = 5
    t = torch.arange(adc, dtype=torch.float64) / adc
    signal = torch.exp(2j * torch.pi * target_bin * t)
    frame = signal.reshape(1, 1, 1, adc)

    result = range_fft(frame, fc)
    peak = int(torch.argmax(torch.abs(result[0, 0, 0])).item())
    assert abs(peak - target_bin) <= 1


def test_doppler_fft_zero_doppler_near_center():
    chirps, adc = 8, 16
    fc = type("FC", (), {"numLoopsPerFrame": chirps})()
    data = torch.ones((1, 1, chirps, adc), dtype=torch.complex64)
    result = doppler_fft(data, fc)
    peak = int(torch.argmax(torch.abs(result[0, 0, :, 0])).item())
    assert abs(peak - chirps // 2) <= 1


def test_clutter_removal_kills_dc():
    data = torch.randn(8, 16, dtype=torch.float64) + 3.0
    cleaned = clutter_removal(data, axis=0)
    assert float(torch.abs(cleaned.mean(dim=0)).max()) < 1e-10


def test_ca_cfar_fast_detects_injected_peak():
    g = torch.Generator().manual_seed(17)
    rd_map = torch.abs(torch.randn(32, 32, generator=g, dtype=torch.float32))
    rd_map[12, 19] = 50.0

    det, _ = ca_cfar_2d_fast(rd_map, guard_cells=(2, 3), training_cells=(4, 6), pfa=1e-3)
    assert bool(det[12, 19].item())


@pytest.mark.parametrize("static_clutter_removal", [False, True])
def test_process_rd_returns_expected_shapes(static_clutter_removal):
    radar = _make_cpu_radar()
    frame = _synthetic_frame(radar)

    rd_mag, rd_map, ranges, velocities = process_rd(
        radar,
        frame,
        static_clutter_removal=static_clutter_removal,
    )
    assert rd_mag.shape == rd_map.shape
    assert rd_mag.ndim == 2
    assert ranges.ndim == 1
    assert velocities.ndim == 1


@pytest.mark.parametrize("detector", ["cfar", "topk"])
def test_process_pc_returns_numpy(detector):
    radar = _make_cpu_radar()
    frame = _synthetic_frame(radar)

    pc = process_pc(
        radar,
        frame,
        detector=detector,
        positive_velocity_only=False,
        static_clutter_removal=False,
    )
    assert isinstance(pc, np.ndarray)
    assert pc.ndim == 2 and pc.shape[1] == 6


@pytest.mark.parametrize("detector", ["cfar", "topk"])
def test_tensor_processing_apis_preserve_device_and_match_numpy(detector):
    radar = _make_cpu_radar()
    frame = _synthetic_frame(radar)

    pc_tensor = process_pc_tensor(
        radar,
        frame,
        detector=detector,
        positive_velocity_only=False,
        static_clutter_removal=False,
    )
    pc_numpy = process_pc(
        radar,
        frame,
        detector=detector,
        positive_velocity_only=False,
        static_clutter_removal=False,
    )
    rd_tensors = process_rd_tensor(radar, frame)
    rd_numpy = process_rd(radar, frame)

    assert pc_tensor.device == frame.device
    np.testing.assert_allclose(pc_tensor.detach().numpy(), pc_numpy)
    for tensor_value, numpy_value in zip(rd_tensors, rd_numpy):
        assert tensor_value.device == frame.device
        np.testing.assert_allclose(tensor_value.detach().numpy(), numpy_value)
