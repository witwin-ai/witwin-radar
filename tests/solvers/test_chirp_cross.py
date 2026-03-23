"""
Chirp-level cross-validation aligned with tests/verify.py.

The reference comparison is:
- PyTorch time-domain beat signal -> FFT
- Slang time-domain beat signal -> FFT
- Dirichlet frequency-domain spectrum
"""

import numpy as np
import pytest
import torch

from conftest import complex_correlation, mag_correlation, peak_ratio, make_radar_or_skip
from witwin.radar.solvers.solver_slang import chirp_slang, chirp_slang_per_target

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
    """Random targets following verify.py."""
    rng = np.random.RandomState(seed)
    distances = torch.tensor(
        rng.uniform(0.5, 5.0, n_targets), dtype=torch.float64, device="cuda"
    )
    amplitudes = torch.tensor(
        rng.uniform(0.5, 1.0, n_targets), dtype=torch.float64, device="cuda"
    )
    return distances, amplitudes


def _compute_pytorch_fft(radar, distances, amplitudes, n_fft):
    signal = radar.chirp(distances, amplitudes)
    return torch.fft.fft(signal, n=n_fft)[: n_fft // 2]


def _compute_slang_fft(radar, distances, amplitudes, n_fft):
    signal = radar.chirp(distances.to(torch.float32), amplitudes.to(torch.float32))
    return torch.fft.fft(signal, n=n_fft)[: n_fft // 2]


class TestChirpCrossValidation:

    def test_verify_style_chirp_agreement(self):
        """Match verify.py: compare FFT(PyTorch), FFT(Slang), and Dirichlet directly."""
        distances, amplitudes = _make_verify_targets()

        radar_pt = make_radar_or_skip(VERIFY_CHIRP_CONFIG, backend="pytorch")
        radar_sl = make_radar_or_skip(VERIFY_CHIRP_CONFIG, backend="slang")
        radar_di = make_radar_or_skip(VERIFY_CHIRP_CONFIG, backend="dirichlet")

        fft_pt = _compute_pytorch_fft(radar_pt, distances, amplitudes, radar_di.solver.N_fft)
        fft_sl = _compute_slang_fft(radar_sl, distances, amplitudes, radar_di.solver.N_fft)
        spec_di = radar_di.chirp(distances.to(torch.float32), amplitudes.to(torch.float32))

        fft_pt_np = fft_pt.detach().cpu().numpy()
        fft_sl_np = fft_sl.detach().cpu().numpy()
        spec_di_np = spec_di.detach().cpu().numpy()

        for lhs_name, lhs, rhs_name, rhs in [
            ("pytorch_fft", fft_pt_np, "slang_fft", fft_sl_np),
            ("pytorch_fft", fft_pt_np, "dirichlet", spec_di_np),
            ("slang_fft", fft_sl_np, "dirichlet", spec_di_np),
        ]:
            mag_corr = mag_correlation(lhs, rhs)
            cx_corr = complex_correlation(lhs, rhs)
            ratio = peak_ratio(lhs, rhs)
            assert mag_corr > 0.999, f"{lhs_name} vs {rhs_name} magnitude correlation = {mag_corr:.6f}"
            assert cx_corr > 0.999, f"{lhs_name} vs {rhs_name} complex correlation = {cx_corr:.6f}"
            assert 0.98 < ratio < 1.02, f"{lhs_name} vs {rhs_name} peak ratio = {ratio:.6f}"

    def test_per_target_chirp_matches_chunked_kernel(self):
        """The non-default per-target Slang kernel should stay phase-aligned with the default chunked path."""
        distances, amplitudes = _make_verify_targets(n_targets=257, seed=7)
        radar_sl = make_radar_or_skip(VERIFY_CHIRP_CONFIG, backend="slang")

        chunked = chirp_slang(
            radar_sl.solver,
            distances.to(torch.float32),
            amplitudes.to(torch.float32),
        )
        per_target = chirp_slang_per_target(
            radar_sl.solver,
            distances.to(torch.float32),
            amplitudes.to(torch.float32),
        )

        chunked_np = chunked.detach().cpu().numpy()
        per_target_np = per_target.detach().cpu().numpy()

        cx_corr = complex_correlation(chunked_np, per_target_np)
        mag_corr = mag_correlation(chunked_np, per_target_np)
        rel_l2 = float(np.linalg.norm(per_target_np - chunked_np) / np.linalg.norm(chunked_np))

        assert cx_corr > 0.999999, f"chunked vs per_target complex correlation = {cx_corr:.9f}"
        assert mag_corr > 0.999999, f"chunked vs per_target magnitude correlation = {mag_corr:.9f}"
        assert rel_l2 < 5e-4, f"chunked vs per_target relative L2 error = {rel_l2:.3e}"


class TestChirpPeakLocation:
    """Verify chirp FFT peak appears at the correct frequency bin."""

    @pytest.mark.parametrize("distance", [1.0, 2.0, 3.0, 4.0, 5.0])
    def test_peak_bin_for_single_target(self, distance):
        """Single target at known distance -> peak at expected FFT bin."""
        radar = make_radar_or_skip(VERIFY_CHIRP_CONFIG, backend="pytorch")
        distances = torch.tensor([distance], dtype=torch.float64, device="cuda")
        amplitudes = torch.tensor([1.0], dtype=torch.float64, device="cuda")

        beat = radar.chirp(distances, amplitudes)
        spectrum = torch.fft.fft(beat)
        magnitude = torch.abs(spectrum).cpu().numpy()

        fs = radar.sample_rate * 1e3
        slope = radar.slope * 1e12
        beat_freq = slope * 2 * distance / radar.c0
        expected_bin = beat_freq / (fs / radar.adc_samples)

        peak_bin = np.argmax(magnitude[: radar.adc_samples // 2])
        assert abs(peak_bin - expected_bin) <= 1, (
            f"distance={distance}m: peak at bin {peak_bin}, expected ~{expected_bin:.1f}"
        )

    def test_multiple_targets_produce_multiple_peaks(self):
        """Two well-separated targets should produce two distinct peaks."""
        radar = make_radar_or_skip(VERIFY_CHIRP_CONFIG, backend="pytorch")
        distances = torch.tensor([1.5, 4.0], dtype=torch.float64, device="cuda")
        amplitudes = torch.tensor([1.0, 1.0], dtype=torch.float64, device="cuda")

        beat = radar.chirp(distances, amplitudes)
        n_fft = 4096
        spectrum = torch.fft.fft(beat, n=n_fft)
        magnitude = torch.abs(spectrum[: n_fft // 2]).cpu().numpy()

        fs = radar.sample_rate * 1e3
        slope = radar.slope * 1e12
        bin1 = int(slope * 2 * 1.5 / radar.c0 / (fs / n_fft))
        bin2 = int(slope * 2 * 4.0 / radar.c0 / (fs / n_fft))

        region1 = magnitude[max(0, bin1 - 5): bin1 + 6]
        region2 = magnitude[max(0, bin2 - 5): bin2 + 6]
        assert region1.max() > magnitude.mean() * 5, "First target peak not found"
        assert region2.max() > magnitude.mean() * 5, "Second target peak not found"
