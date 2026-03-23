"""
Solver edge-case and regression tests.
"""

import torch
import numpy as np
import pytest

from conftest import make_radar_or_skip, make_static_interpolator

pytestmark = pytest.mark.gpu


class TestSlangLargeADC:
    """Regression: Slang MIMO kernel must handle adc_samples > 256 (bug #1)."""

    def test_slang_mimo_512_samples(self):
        """With adc_samples=512, the second half should NOT be all zeros."""
        cfg = {
            "num_tx": 1, "num_rx": 1,
            "fc": 77e9, "slope": 60.012,
            "adc_samples": 512, "adc_start_time": 0,
            "sample_rate": 4400, "idle_time": 7, "ramp_end_time": 58,
            "chirp_per_frame": 1, "frame_per_second": 10,
            "num_doppler_bins": 1, "num_range_bins": 512,
            "num_angle_bins": 64, "power": 12,
            "tx_loc": [[0, 0, 0]],
            "rx_loc": [[0, 0, 0]],
        }
        r = make_radar_or_skip(cfg, backend="slang")
        interp = make_static_interpolator([0, 0, -3])
        frame = r.mimo(interp)

        # The old bug: samples 256+ were all zeros
        second_half = frame[0, 0, 0, 256:].abs().cpu()
        assert second_half.max().item() > 0, (
            "Slang kernel produced zeros for ADC samples > 256"
        )

    def test_slang_mimo_640_samples(self):
        """High-resolution config: 640 ADC samples."""
        cfg = {
            "num_tx": 1, "num_rx": 1,
            "fc": 77e9, "slope": 128.0,
            "adc_samples": 640, "adc_start_time": 0,
            "sample_rate": 12500, "idle_time": 7, "ramp_end_time": 58,
            "chirp_per_frame": 1, "frame_per_second": 10,
            "num_doppler_bins": 1, "num_range_bins": 640,
            "num_angle_bins": 64, "power": 12,
            "tx_loc": [[0, 0, 0]],
            "rx_loc": [[0, 0, 0]],
        }
        r = make_radar_or_skip(cfg, backend="slang")
        interp = make_static_interpolator([0, 0, -3])
        frame = r.mimo(interp)

        last_quarter = frame[0, 0, 0, 480:].abs().cpu()
        assert last_quarter.max().item() > 0, (
            "Slang kernel produced zeros for ADC samples > 256 (640 config)"
        )

    def test_slang_512_produces_valid_spectrum(self):
        """512-sample Slang output should have a valid range FFT peak."""
        cfg = {
            "num_tx": 1, "num_rx": 1,
            "fc": 77e9, "slope": 128.0,
            "adc_samples": 512, "adc_start_time": 0,
            "sample_rate": 12500, "idle_time": 7, "ramp_end_time": 58,
            "chirp_per_frame": 1, "frame_per_second": 10,
            "num_doppler_bins": 1, "num_range_bins": 512,
            "num_angle_bins": 64, "power": 12,
            "tx_loc": [[0, 0, 0]],
            "rx_loc": [[0, 0, 0]],
        }
        interp = make_static_interpolator([0, 0, -2])
        r = make_radar_or_skip(cfg, backend="slang")
        frame = r.mimo(interp)
        # Range FFT should show a peak (not all noise)
        import torch
        spectrum = torch.fft.fft(frame[0, 0, 0, :])
        mag = spectrum.abs().cpu().numpy()
        assert mag[:256].max() > mag[:256].mean() * 5, "No clear peak in range FFT"


class TestChirpEdgeCases:

    def test_very_close_target(self):
        """Target at 0.1m should still produce valid output."""
        cfg = {
            "num_tx": 1, "num_rx": 1,
            "fc": 77e9, "slope": 60.012,
            "adc_samples": 256, "adc_start_time": 0,
            "sample_rate": 4400, "idle_time": 7, "ramp_end_time": 58,
            "chirp_per_frame": 1, "frame_per_second": 10,
            "num_doppler_bins": 1, "num_range_bins": 256,
            "num_angle_bins": 64, "power": 12,
            "tx_loc": [[0, 0, 0]],
            "rx_loc": [[0, 0, 0]],
        }
        r = make_radar_or_skip(cfg, backend="pytorch")
        d = torch.tensor([0.1], dtype=torch.float64, device="cuda")
        a = torch.tensor([1.0], dtype=torch.float64, device="cuda")
        beat = r.chirp(d, a)
        assert not torch.isnan(beat).any()
        assert beat.abs().max().item() > 0

    def test_large_target_count(self):
        """4096 targets should not error or produce NaN."""
        cfg = {
            "num_tx": 1, "num_rx": 1,
            "fc": 77e9, "slope": 60.012,
            "adc_samples": 256, "adc_start_time": 0,
            "sample_rate": 4400, "idle_time": 7, "ramp_end_time": 58,
            "chirp_per_frame": 1, "frame_per_second": 10,
            "num_doppler_bins": 1, "num_range_bins": 256,
            "num_angle_bins": 64, "power": 12,
            "tx_loc": [[0, 0, 0]],
            "rx_loc": [[0, 0, 0]],
        }
        rng = np.random.RandomState(0)
        d = torch.tensor(rng.uniform(0.5, 8, 4096), dtype=torch.float64, device="cuda")
        a = torch.tensor(rng.uniform(0.01, 1, 4096), dtype=torch.float64, device="cuda")

        for backend in ("slang", "dirichlet"):
            r = make_radar_or_skip(cfg, backend=backend)
            result = r.chirp(d, a)
            assert not torch.isnan(result).any(), f"{backend}: NaN in output"
