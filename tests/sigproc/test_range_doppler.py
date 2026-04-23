"""
Tests for Range-Doppler DSP functions: range FFT, Doppler FFT, clutter removal.

All DSP helpers are torch-only.
"""

import numpy as np
import torch

from witwin.radar.sigproc.pointcloud import (
    frame_reshape, range_fft, doppler_fft, clutter_removal,
)


class _FC:
    """Minimal FrameConfig-like namespace for DSP function tests."""

    def __init__(self, num_tx=3, num_rx=4, chirps=128, adc=256):
        self.numTxAntennas = num_tx
        self.numRxAntennas = num_rx
        self.numLoopsPerFrame = chirps
        self.numADCSamples = adc
        self.numChirpsPerFrame = num_tx * chirps
        self.numRangeBins = adc
        self.numDopplerBins = chirps


def _randc(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g, dtype=torch.float64) + 1j * torch.randn(*shape, generator=g, dtype=torch.float64)


class TestFrameReshape:

    def test_output_shape(self):
        fc = _FC(num_tx=3, num_rx=4, chirps=128, adc=256)
        flat = _randc(fc.numLoopsPerFrame * fc.numTxAntennas * fc.numRxAntennas * fc.numADCSamples)
        reshaped = frame_reshape(flat, fc)
        assert reshaped.shape == (3, 4, 128, 256)

    def test_transpose_is_tx_rx_chirp_adc(self):
        fc = _FC(num_tx=2, num_rx=3, chirps=4, adc=8)
        flat = torch.arange(2 * 3 * 4 * 8, dtype=torch.int64).to(torch.complex64)
        reshaped = frame_reshape(flat, fc)
        assert reshaped.shape == (2, 3, 4, 8)


class TestRangeFFT:

    def test_output_shape_matches_input(self):
        fc = _FC(adc=256)
        frame = _randc(3, 4, 128, 256)
        result = range_fft(frame, fc)
        assert result.shape == frame.shape

    def test_peak_at_known_beat_frequency(self):
        """A pure sinusoid at bin k should produce FFT peak at bin k."""
        adc = 256
        fc = _FC(num_tx=1, num_rx=1, chirps=1, adc=adc)
        target_bin = 40
        t = torch.arange(adc, dtype=torch.float64) / adc
        signal = torch.exp(2j * torch.pi * target_bin * t)
        frame = signal.reshape(1, 1, 1, adc)

        result = range_fft(frame, fc)
        mag = torch.abs(result[0, 0, 0, :])
        peak = int(torch.argmax(mag).item())
        assert abs(peak - target_bin) <= 1, f"Expected bin ~{target_bin}, got {peak}"

    def test_hamming_window_applied(self):
        """Result should differ from raw FFT (Hamming window effect)."""
        adc = 256
        fc = _FC(num_tx=1, num_rx=1, chirps=1, adc=adc)
        signal = _randc(1, 1, 1, adc)

        windowed_result = range_fft(signal, fc)
        raw_fft = torch.fft.fft(signal)
        assert not torch.allclose(windowed_result, raw_fft)


class TestDopplerFFT:

    def test_output_shape(self):
        fc = _FC(chirps=128, adc=256)
        data = _randc(3, 4, 128, 256)
        result = doppler_fft(data, fc)
        assert result.shape == data.shape

    def test_fftshift_applied(self):
        """Zero-Doppler should be near the center of the Doppler axis."""
        chirps = 32
        adc = 64
        fc = _FC(num_tx=1, num_rx=1, chirps=chirps, adc=adc)
        data = torch.ones((1, 1, chirps, adc), dtype=torch.complex64)
        result = doppler_fft(data, fc)
        mag = torch.abs(result[0, 0, :, 0])
        peak_bin = int(torch.argmax(mag).item())
        assert abs(peak_bin - chirps // 2) <= 1


class TestClutterRemoval:

    def test_removes_dc_component(self):
        """Mean along the clutter axis should be ~0 after removal."""
        data = torch.randn(128, 256, dtype=torch.float64) + 5.0
        cleaned = clutter_removal(data, axis=0)
        mean_after = torch.abs(cleaned.mean(dim=0))
        assert float(mean_after.max()) < 1e-10

    def test_preserves_shape(self):
        data = _randc(3, 4, 128, 256)
        result = clutter_removal(data, axis=2)
        assert result.shape == data.shape

    def test_removes_static_preserves_motion(self):
        """A chirp-varying component should survive clutter removal."""
        chirps, adc = 128, 256
        static = torch.ones((chirps, adc), dtype=torch.float64) * 10.0
        motion = torch.sin(2 * torch.pi * torch.arange(chirps, dtype=torch.float64).unsqueeze(1) / chirps * 3)
        data = static + motion

        cleaned = clutter_removal(data, axis=0)
        assert float(torch.abs(cleaned.mean(dim=0)).max()) < 1e-10
        assert float(torch.abs(cleaned).max()) > 0.5
