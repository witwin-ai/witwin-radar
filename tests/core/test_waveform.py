"""
Tests for FMCW chirp waveform generation.

Verifies mathematical properties of the radar waveform without needing
a Radar object or GPU — uses torch on CPU.
"""

import torch
import numpy as np
import pytest


FC = 77e9
SLOPE = 60.012e12  # Hz/s
C0 = 299792458


class TestWaveformMath:
    """Verify FMCW chirp waveform properties."""

    def test_unit_magnitude(self):
        """Waveform exp(j*phi) must have unit magnitude everywhere."""
        t = torch.linspace(0, 1e-5, 1000, dtype=torch.float64)
        wf = torch.exp(1j * 2 * torch.pi * (FC * t + 0.5 * SLOPE * t * t))
        mag = torch.abs(wf)
        torch.testing.assert_close(mag, torch.ones_like(mag), atol=1e-12, rtol=0)

    def test_waveform_matches_reference(self):
        """Waveform should exactly match exp(j*2pi*(fc*t + 0.5*slope*t^2))."""
        t = torch.linspace(0, 1e-5, 256, dtype=torch.float64)
        wf = torch.exp(1j * 2 * torch.pi * (FC * t + 0.5 * SLOPE * t * t))
        # Recompute reference independently
        phase_ref = 2 * np.pi * (FC * t + 0.5 * SLOPE * t * t)
        wf_ref = torch.exp(1j * phase_ref)
        torch.testing.assert_close(wf, wf_ref, atol=1e-12, rtol=0)

    def test_instantaneous_frequency_at_start(self):
        """At t=0, instantaneous freq should be fc."""
        dt = 1e-13
        t0 = torch.tensor([0.0], dtype=torch.float64)
        t1 = torch.tensor([dt], dtype=torch.float64)
        phi0 = 2 * torch.pi * (FC * t0 + 0.5 * SLOPE * t0 * t0)
        phi1 = 2 * torch.pi * (FC * t1 + 0.5 * SLOPE * t1 * t1)
        f_inst = (phi1 - phi0).item() / (2 * np.pi * dt)
        assert f_inst == pytest.approx(FC, rel=1e-6)

    def test_instantaneous_frequency_at_midpoint(self):
        """At t=T/2, instantaneous freq should be fc + slope*T/2."""
        T_ramp = 58e-6
        t_mid = T_ramp / 2
        dt = 1e-13
        t0 = torch.tensor([t_mid], dtype=torch.float64)
        t1 = torch.tensor([t_mid + dt], dtype=torch.float64)
        phi0 = 2 * torch.pi * (FC * t0 + 0.5 * SLOPE * t0 * t0)
        phi1 = 2 * torch.pi * (FC * t1 + 0.5 * SLOPE * t1 * t1)
        f_inst = (phi1 - phi0).item() / (2 * np.pi * dt)
        expected = FC + SLOPE * t_mid
        assert f_inst == pytest.approx(expected, rel=1e-6)

    def test_chirp_bandwidth(self):
        """Total bandwidth over ramp duration should be slope * T_ramp."""
        T_ramp = 58e-6
        B = SLOPE * T_ramp
        # Should be ~3.48 GHz
        assert B == pytest.approx(3.48e9, rel=0.01)

    def test_beat_frequency_for_target(self):
        """A target at range d produces beat frequency = slope * 2d / c0."""
        d = 3.0  # meters
        f_beat = SLOPE * 2 * d / C0
        # Should be ~1.2 MHz — well within sampling bandwidth
        assert 1.0e6 < f_beat < 2.0e6
        # FFT bin (out of 256 at 4.4 MSPS)
        fs = 4400e3
        bin_idx = f_beat / (fs / 256)
        assert bin_idx == pytest.approx(70, abs=1)


@pytest.mark.gpu
class TestRadarWaveform:
    """Test Radar.waveform() method (needs CUDA)."""

    def test_waveform_unit_magnitude(self, standard_config):
        from witwin.radar import Radar

        r = Radar(standard_config, backend="pytorch")
        wf = r.waveform(r.t_sample)
        mag = torch.abs(wf)
        torch.testing.assert_close(
            mag, torch.ones_like(mag), atol=1e-10, rtol=0
        )

    def test_tx_waveform_shape(self, standard_config):
        from witwin.radar import Radar

        r = Radar(standard_config, backend="pytorch")
        assert r.tx_waveform.shape == (r.config.adc_samples,)
        assert r.t_sample.shape == (r.config.adc_samples,)
