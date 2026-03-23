from __future__ import annotations

import numpy as np
import pytest
import torch

from conftest import STANDARD_CONFIG
from witwin.radar import Radar
from witwin.radar.sigproc import (
    PointCloudProcessConfig,
    clutter_removal,
    doppler_fft,
    frame2pointcloud,
    process_pc,
    process_rd,
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
    return Radar(CPU_SIGPROC_CONFIG, backend="pytorch", device="cpu")


def _static_interpolator(radar: Radar, position=(0.0, 0.0, -3.0), intensity=1.0):
    position_tensor = torch.tensor([position], dtype=torch.float32, device=radar.device)
    intensity_tensor = torch.tensor([intensity], dtype=torch.float32, device=radar.device)

    def interp(_t):
        return intensity_tensor, position_tensor

    return interp


def _sort_pc(pc: np.ndarray) -> np.ndarray:
    if pc.shape[0] == 0:
        return pc
    order = np.lexsort((np.round(pc[:, 3], 6), np.round(pc[:, 5], 6)))
    return pc[order]


def test_frame2pointcloud_requires_radar():
    radar = _make_cpu_radar()
    frame = radar.mimo(_static_interpolator(radar))
    cfg = PointCloudProcessConfig(radar)

    with pytest.raises(ValueError, match="requires a radar instance"):
        frame2pointcloud(frame, cfg)


def test_range_fft_torch_matches_numpy():
    rng = np.random.RandomState(7)
    frame_np = (rng.randn(3, 4, 8, 16) + 1j * rng.randn(3, 4, 8, 16)).astype(np.complex64)
    frame_torch = torch.from_numpy(frame_np)
    frame_config = type("FC", (), {"numADCSamples": 16})()

    observed = range_fft(frame_torch, frame_config)
    expected = range_fft(frame_np, frame_config)

    np.testing.assert_allclose(observed.numpy(), expected, atol=1e-6, rtol=1e-6)


def test_doppler_fft_torch_matches_numpy():
    rng = np.random.RandomState(11)
    frame_np = (rng.randn(3, 4, 8, 16) + 1j * rng.randn(3, 4, 8, 16)).astype(np.complex64)
    frame_torch = torch.from_numpy(frame_np)
    frame_config = type("FC", (), {"numLoopsPerFrame": 8})()

    observed = doppler_fft(frame_torch, frame_config)
    expected = doppler_fft(frame_np, frame_config)

    np.testing.assert_allclose(observed.numpy(), expected, atol=1e-6, rtol=1e-6)


def test_clutter_removal_torch_matches_numpy():
    rng = np.random.RandomState(13)
    frame_np = (rng.randn(3, 4, 8, 16) + 1j * rng.randn(3, 4, 8, 16)).astype(np.complex64)
    frame_torch = torch.from_numpy(frame_np)

    observed = clutter_removal(frame_torch, axis=2)
    expected = clutter_removal(frame_np, axis=2)

    np.testing.assert_allclose(observed.numpy(), expected, atol=1e-6, rtol=1e-6)


def test_ca_cfar_fast_torch_matches_numpy():
    rng = np.random.RandomState(17)
    rd_map = np.abs(rng.randn(32, 32).astype(np.float32))
    rd_map[12, 19] = 50.0

    det_np, thr_np = ca_cfar_2d_fast(rd_map, guard_cells=(2, 3), training_cells=(4, 6), pfa=1e-3)
    det_t, thr_t = ca_cfar_2d_fast(torch.from_numpy(rd_map), guard_cells=(2, 3), training_cells=(4, 6), pfa=1e-3)

    np.testing.assert_array_equal(det_t.numpy(), det_np)
    np.testing.assert_allclose(thr_t.numpy(), thr_np, atol=2e-5, rtol=2e-6)


@pytest.mark.parametrize("static_clutter_removal", [False, True])
def test_process_rd_torch_frame_matches_numpy_frame(static_clutter_removal):
    radar = _make_cpu_radar()
    frame_torch = radar.mimo(_static_interpolator(radar))
    frame_numpy = frame_torch.detach().cpu().numpy()

    rd_mag_t, rd_map_t, ranges_t, velocities_t = process_rd(
        radar,
        frame_torch,
        static_clutter_removal=static_clutter_removal,
    )
    rd_mag_n, rd_map_n, ranges_n, velocities_n = process_rd(
        radar,
        frame_numpy,
        static_clutter_removal=static_clutter_removal,
    )

    np.testing.assert_allclose(rd_mag_t, rd_mag_n, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(rd_map_t, rd_map_n, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(ranges_t, ranges_n, atol=0, rtol=0)
    np.testing.assert_allclose(velocities_t, velocities_n, atol=0, rtol=0)


@pytest.mark.parametrize("detector", ["cfar", "topk"])
def test_process_pc_torch_frame_matches_numpy_frame(detector):
    radar = _make_cpu_radar()
    frame_torch = radar.mimo(_static_interpolator(radar))
    frame_numpy = frame_torch.detach().cpu().numpy()

    pc_t = process_pc(
        radar,
        frame_torch,
        detector=detector,
        positive_velocity_only=False,
        static_clutter_removal=False,
    )
    pc_n = process_pc(
        radar,
        frame_numpy,
        detector=detector,
        positive_velocity_only=False,
        static_clutter_removal=False,
    )

    np.testing.assert_allclose(_sort_pc(pc_t), _sort_pc(pc_n), atol=1e-5, rtol=1e-5)
