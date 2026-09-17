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
    t, f, spectrum = microdoppler_spectrogram(
        SlowTimeSignal(samples, times, phasor), window_slots=64, hop_slots=32, window="rectangular"
    )
    torch.testing.assert_close(t, torch.tensor([3.0315, 3.0635, 3.0955], dtype=torch.float64))
    assert torch.all(f[spectrum.abs().argmax(-1)] == -velocity_sign * 125)


def test_microdoppler_refuses_frame_gaps_and_unlabelled_samples():
    with pytest.raises(ValueError, match="uniform"):
        SlowTimeSignal(torch.ones(4, dtype=torch.complex64), (0.0, 0.001, 0.1, 0.101), BEAT_PHASOR)
    with pytest.raises(TypeError, match="SlowTimeSignal"):
        microdoppler_spectrogram(torch.ones(4), window_slots=4, hop_slots=1)
