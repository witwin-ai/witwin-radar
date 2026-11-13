"""M1-5: every result states the observation schedule its synthesis actually used.

``Result.observation_schedule`` is what the echo ran, which can be coarser than
the requested ``Motion``: an unmodulated OFDM CFR has no fast-time grid of its
own, so an ADC request observes it once per symbol, while a modulated OFDM
stream and a pulsed train are observed per sample like FMCW.
"""

from __future__ import annotations

import pytest
import torch
from support import oracle_scenes as sc

from witwin.radar import Motion, Ofdm, PointTargets, Pulsed

pytestmark = pytest.mark.gpu


def _ofdm(**fields):
    return Ofdm(
        subcarrier_spacing=1e6,
        num_subcarriers=16,
        cyclic_prefix=0.3e-6,
        num_symbols=2,
        max_expected_delay=0.2e-6,
        **fields,
    )


def _pulsed():
    return Pulsed(
        pulse_kind="rect",
        pulse_width=1e-6,
        bandwidth=1e6,
        pri=5e-6,
        num_pulses=2,
        sample_rate=10e6,
        num_samples=20,
        range_gate_start=0.0,
        max_expected_delay_rate=1e-6,
    )


@pytest.mark.parametrize(
    ("waveform", "motion", "expected"),
    [
        (sc.beat_fmcw(chirps=2), None, "frame"),
        (sc.beat_fmcw(chirps=2), Motion.chirp(), "slow_time"),
        (sc.beat_fmcw(chirps=2), Motion.adc(), "fast_time"),
        (_ofdm(), Motion.adc(), "slow_time"),
        (_ofdm(modulation="qpsk"), Motion.adc(), "fast_time"),
        (_pulsed(), Motion.chirp(), "slow_time"),
        (_pulsed(), Motion.adc(), "fast_time"),
    ],
    ids=["fmcw-still", "fmcw-chirp", "fmcw-adc", "ofdm-cfr-adc", "ofdm-qpsk-adc", "pulsed-chirp", "pulsed-adc"],
)
def test_the_result_publishes_the_schedule_that_ran(waveform, motion, expected):
    radar = sc.radar(waveform)
    origin = torch.tensor([[2.0, 0.0, 0.0]], device=radar.device)
    if motion is None:
        targets = PointTargets(positions=origin, rcs=1.0)
    else:
        targets = PointTargets(positions=origin, rcs=1.0, trajectory=lambda t: origin + 5.0 * t)
    result = radar.simulate(sc.empty_scene(), targets, times=(0.0,), los=True, reflections=0, motion=motion)
    assert result.observation_schedule == expected
    assert result.frame(0).synthesis.observation_schedule == expected
