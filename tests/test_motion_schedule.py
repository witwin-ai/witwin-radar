"""What a moving session publishes about its observation schedule, and the slow-time physics it carries.

M1-5 / D-6: a dynamic OFDM or pulsed run evaluates the world once per symbol or
pulse, so it publishes ``"symbol"`` / ``"pulse"`` and is not exhaustive.
M2-10 / D-12: the slow-time carrier phase those schedules carry, checked against
an independent float64 geometric delay.
"""

from __future__ import annotations

import math

import pytest
import torch
from support import oracle_scenes as sc

from witwin.radar import Motion, Ofdm, PointTargets, Pulsed

pytestmark = pytest.mark.gpu

C = 299_792_458.0
#: Radial speed of the mover, m/s. Small enough that the echo stays in one
#: fast-time cell over the frame, large enough that the slow-time phase steps
#: are tenths of a radian.
SPEED = 2.0
START = 6.0


def _ofdm():
    return Ofdm(
        subcarrier_spacing=1e6, num_subcarriers=64, cyclic_prefix=0.3e-6, num_symbols=16, max_expected_delay=0.25e-6
    )


def _pulsed():
    return Pulsed(
        pulse_kind="lfm",
        pulse_width=2e-6,
        bandwidth=20e6,
        pri=20e-6,
        num_pulses=16,
        sample_rate=50e6,
        num_samples=256,
        range_gate_start=0.0,
    )


def _mover(device):
    def trajectory(time_s):
        return torch.tensor([[START + SPEED * float(time_s), 0.0, 0.0]], dtype=torch.float32, device=device)

    return PointTargets(positions=trajectory(0.0), rcs=1.0, trajectory=trajectory)


def _run(waveform, motion=None):
    radar = sc.radar(waveform)
    return radar, radar.simulate(
        sc.empty_scene(), _mover(radar.device), times=(0.0,), los=True, reflections=0, motion=motion
    )


@pytest.mark.parametrize(
    ("waveform", "motion", "kind", "schedule"),
    [
        (_ofdm, None, "symbol", "slow_time"),
        (_ofdm, Motion.adc(), "symbol", "slow_time"),
        (_ofdm, Motion.chirp(), "symbol", "slow_time"),
        (_pulsed, Motion.chirp(), "pulse", "slow_time"),
        (_pulsed, Motion.adc(), "adc", "fast_time"),
        (_pulsed, None, "adc", "fast_time"),
    ],
)
def test_moving_ofdm_and_pulsed_publish_the_schedule_that_ran(waveform, motion, kind, schedule):
    """M1-5: the label is derived from the observation schedule, and only a fast-time one is exhaustive.

    A frequency-domain OFDM symbol has no fast-time grid, so every request
    observes it once per symbol; a pulsed train refreshes at every fast-time
    sample under an ADC request and holds each pulse under stop and hop.
    """

    _, result = _run(waveform(), motion)
    assert result.observation_schedule == schedule
    assert result.motion_sampling == kind
    assert result.motion_sampling_exhaustive is (schedule == "fast_time")


def test_fmcw_chirp_schedule_is_not_exhaustive_and_adc_is():
    radar = sc.radar(sc.beat_fmcw(samples=8, chirps=2))
    targets = _mover(radar.device)
    held = radar.simulate(sc.empty_scene(), targets, times=(0.0,), reflections=0, motion=Motion.chirp())
    every = radar.simulate(sc.empty_scene(), targets, times=(0.0,), reflections=0, motion=Motion.adc())
    assert (held.motion_sampling, held.motion_sampling_exhaustive) == ("chirp", False)
    assert (every.motion_sampling, every.motion_sampling_exhaustive) == ("adc", True)


def _slow_time_delay(result) -> torch.Tensor:
    """The float64 round-trip delay at each observation instant, from the trajectory itself."""

    times = torch.as_tensor(result.sample_times_s[0], dtype=torch.float64)
    return 2.0 * (START + SPEED * times) / C


def test_moving_ofdm_carries_the_geometric_slow_time_phase():
    """M2-10: symbol-to-symbol CFR phase is ``-2 pi f_k d tau`` and the level follows ``R^-4``.

    Subcarrier ``k`` of symbol ``m`` is ``H_m(k) = A_m exp(-j 2 pi f_k tau_m)``
    in Channel's ``exp(-jkd)`` convention, with ``f_k`` the subcarrier's RF
    frequency. The ratio of consecutive symbols removes every constant factor.
    """

    radar, result = _run(_ofdm(), Motion.chirp())
    cube = result.cube[0, 0, 0].to(torch.complex128)  # [symbol, subcarrier]
    tau = _slow_time_delay(result)
    spec = radar.waveform_spec()
    measured = torch.angle(cube[1:] * cube[:-1].conj())
    frequency = torch.tensor(
        [spec.subcarrier_frequency_hz(k) for k in range(cube.shape[1])], dtype=torch.float64, device=cube.device
    )
    step = (tau[1:] - tau[:-1]).to(cube.device)
    expected = torch.angle(torch.exp(-2j * math.pi * step[:, None] * frequency[None, :]))
    torch.testing.assert_close(measured, expected, atol=2e-3, rtol=0)
    ranges = (C * tau / 2).to(cube.device)
    level = cube.abs().mean(dim=1)
    torch.testing.assert_close(level / level[0], (ranges[0] / ranges) ** 2, atol=0, rtol=1e-4)


def test_moving_pulsed_carries_the_geometric_slow_time_phase():
    """M2-10: pulse-to-pulse phase inside the received pulse is ``-2 pi f_c d tau``.

    The fast-time sample sits mid-pulse (the 2 us pulse spans 100 samples and
    the echo arrives after 40 ns), where the LFM phase moves by ``2 pi K t
    d tau``, five orders below the carrier term over this frame.
    """

    _, result = _run(_pulsed(), Motion.chirp())
    echo = result.cube[0, 0, 0].to(torch.complex128)  # [pulse, fast time]
    sample = 50
    assert bool((echo[:, sample].abs() > 0.5 * echo[:, sample].abs().max()).all())
    tau = _slow_time_delay(result).to(echo.device)
    measured = torch.angle(echo[1:, sample] * echo[:-1, sample].conj())
    expected = torch.angle(torch.exp(-2j * math.pi * sc.CARRIER_HZ * (tau[1:] - tau[:-1])))
    torch.testing.assert_close(measured, expected, atol=5e-3, rtol=0)
