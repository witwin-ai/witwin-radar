"""Chirp-level validation for the native Dirichlet solver."""

import numpy as np
import pytest
import torch

from conftest import complex_correlation, mag_correlation, make_radar_or_skip, peak_ratio
from witwin.radar.solvers.common import pytorch_chirp_reference

pytestmark = pytest.mark.gpu


VERIFY_CHIRP_CONFIG = {
    "num_tx": 1,
    "num_rx": 1,
    "fc": 77e9,
    "slope": 100.0,
    "adc_samples": 400,
    "adc_start_time": 0,
    "sample_rate": 10000,
    "idle_time": 0,
    "ramp_end_time": 40,
    "chirp_per_frame": 1,
    "frame_per_second": 1,
    "num_doppler_bins": 1,
    "num_range_bins": 400,
    "num_angle_bins": 1,
    "power": 1,
    "tx_loc": [[0, 0, 0]],
    "rx_loc": [[0, 0, 0]],
}


def _make_verify_targets(n_targets=1024, seed=42):
    rng = np.random.RandomState(seed)
    distances = torch.tensor(rng.uniform(0.5, 5.0, n_targets), dtype=torch.float32, device="cuda")
    amplitudes = torch.tensor(rng.uniform(0.5, 1.0, n_targets), dtype=torch.float32, device="cuda")
    return distances, amplitudes


def _compute_reference_fft(radar, distances, amplitudes, n_fft):
    signal = pytorch_chirp_reference(radar, distances.to(torch.float64), amplitudes.to(torch.float64))
    return torch.fft.fft(signal, n=n_fft)[: n_fft // 2]


class TestChirpCrossValidation:
    def test_dirichlet_matches_pytorch_fft_reference(self):
        distances, amplitudes = _make_verify_targets()
        radar = make_radar_or_skip(VERIFY_CHIRP_CONFIG, backend="dirichlet")

        reference = _compute_reference_fft(radar, distances, amplitudes, radar.solver.N_fft)
        actual = radar.chirp(distances, amplitudes)

        reference_np = reference.detach().cpu().numpy()
        actual_np = actual.detach().cpu().numpy()
        mag_corr = mag_correlation(reference_np, actual_np)
        cx_corr = complex_correlation(reference_np, actual_np)
        ratio = peak_ratio(reference_np, actual_np)
        assert mag_corr > 0.999
        assert cx_corr > 0.999
        assert 0.98 < ratio < 1.02


class TestChirpPeakLocation:
    @pytest.mark.parametrize("distance", [1.0, 2.0, 3.0, 4.0, 5.0])
    def test_peak_bin_for_single_target(self, distance):
        radar = make_radar_or_skip(VERIFY_CHIRP_CONFIG, backend="dirichlet")
        distances = torch.tensor([distance], dtype=torch.float32, device="cuda")
        amplitudes = torch.tensor([1.0], dtype=torch.float32, device="cuda")

        spectrum = radar.chirp(distances, amplitudes)
        magnitude = torch.abs(spectrum).cpu().numpy()

        fs = radar.config.sample_rate * 1e3
        slope = radar.config.slope * 1e12
        beat_freq = slope * 2 * distance / radar.c0
        expected_bin = beat_freq / (fs / radar.solver.N_fft)

        peak_bin = np.argmax(magnitude)
        assert abs(peak_bin - expected_bin) <= 1

    def test_multiple_targets_produce_multiple_peaks(self):
        radar = make_radar_or_skip(VERIFY_CHIRP_CONFIG, backend="dirichlet")
        distances = torch.tensor([1.5, 4.0], dtype=torch.float32, device="cuda")
        amplitudes = torch.tensor([1.0, 1.0], dtype=torch.float32, device="cuda")

        spectrum = radar.chirp(distances, amplitudes)
        magnitude = torch.abs(spectrum).cpu().numpy()

        fs = radar.config.sample_rate * 1e3
        slope = radar.config.slope * 1e12
        bin1 = int(slope * 2 * 1.5 / radar.c0 / (fs / radar.solver.N_fft))
        bin2 = int(slope * 2 * 4.0 / radar.c0 / (fs / radar.solver.N_fft))

        region1 = magnitude[max(0, bin1 - 5): bin1 + 6]
        region2 = magnitude[max(0, bin2 - 5): bin2 + 6]
        assert region1.max() > magnitude.mean() * 5
        assert region2.max() > magnitude.mean() * 5
