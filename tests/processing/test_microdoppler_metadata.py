"""Micro-Doppler metadata: the absolute window centres and the reconciled Doppler sign.

``SlowTimeSignal`` carries the phasor its producer published, and the
spectrogram reconciles it, so one physical motion written in the beat
convention ``conj(exp(-j*k*d))`` and the same motion written in the Channel
convention ``exp(-j*k*d)`` - complex conjugates of each other as slow-time
sequences - must land on the SAME signed bin of the returned frequency axis.
That is why the synthetic tone below flips sign with the phasor while the
asserted peak does not, and why the window centres are absolute instants read
off the sample labels rather than offsets from zero: a caller that stitches two
frames reads a time, not a position in a buffer.
"""

import math

import pytest
import torch

from witwin.radar.processing import SlowTimeSignal, microdoppler_spectrogram
from witwin.radar.synthesis.assembly import BEAT_PHASOR, CHANNEL_PHASOR


@pytest.mark.parametrize("phasor,sign", [(BEAT_PHASOR, 1), (CHANNEL_PHASOR, -1)])
@pytest.mark.parametrize("velocity_sign", [-1, 1])
def test_microdoppler_sign_and_absolute_window_centres(phasor, sign, velocity_sign):
    times = tuple(3.0 + n * 0.001 for n in range(128))
    samples = torch.exp(sign * velocity_sign * 2j * math.pi * 125 * torch.arange(128).double() * 0.001)
    product = microdoppler_spectrogram(
        SlowTimeSignal(samples, times, phasor), window_slots=64, hop_slots=32, window="rectangular"
    )
    torch.testing.assert_close(product.times_s, torch.tensor([3.0315, 3.0635, 3.0955], dtype=torch.float64))
    peak = product.frequency_hz[product.data.abs().argmax(-1)]
    torch.testing.assert_close(peak, torch.full_like(peak, -velocity_sign * 125.0), rtol=1e-6, atol=0.0)
    assert product.velocity_mps is None


@pytest.mark.parametrize("window", ["rectangular", "hann"])
def test_the_spectrogram_is_amplitude_normalised_like_the_range_doppler_map(window):
    """F-7: a unit on-bin tone peaks at the window's coherent gain, as a Range-Doppler bin does."""

    times = tuple(n * 1e-3 for n in range(64))
    samples = torch.exp(2j * math.pi * 125 * torch.arange(64).double() * 1e-3)
    product = microdoppler_spectrogram(
        SlowTimeSignal(samples, times, CHANNEL_PHASOR), window_slots=32, hop_slots=16, window=window
    )
    peak = product.data.abs().amax(dim=-1)
    expected = 1.0 if window == "rectangular" else 0.5
    assert product.window_coherent_gain == expected
    torch.testing.assert_close(peak, torch.full_like(peak, expected), rtol=1e-12, atol=1e-12)


def test_a_declared_wavelength_publishes_the_velocity_axis_on_the_range_doppler_scale():
    """``v = lambda f / 2``, positive closing, the conversion ``ProcessingAxes.velocity_mps`` uses."""

    wavelength = 3e8 / 77e9
    times = tuple(n * 1e-4 for n in range(64))
    closing = 2.0
    # Channel phasor: a closing target advances as exp(+j 2 pi (2 v / lambda) t).
    samples = torch.exp(2j * math.pi * (2 * closing / wavelength) * torch.arange(64).double() * 1e-4)
    product = microdoppler_spectrogram(
        SlowTimeSignal(samples, times, CHANNEL_PHASOR, wavelength_m=wavelength), window_slots=64, hop_slots=64
    )
    torch.testing.assert_close(product.velocity_mps, product.frequency_hz * wavelength / 2.0)
    peak = float(product.velocity_mps[product.data[0].abs().argmax()])
    assert abs(peak - closing) <= wavelength / (2 * 64 * 1e-4) / 2


def test_microdoppler_refuses_frame_gaps_and_unlabelled_samples():
    with pytest.raises(ValueError, match="uniform"):
        SlowTimeSignal(torch.ones(4, dtype=torch.complex64), (0.0, 0.001, 0.1, 0.101), BEAT_PHASOR)
    with pytest.raises(TypeError, match="SlowTimeSignal"):
        microdoppler_spectrogram(torch.ones(4), window_slots=4, hop_slots=1)
