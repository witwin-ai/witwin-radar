"""What a moving session publishes about its observation schedule, and the slow-time physics it carries.

M1-5 / D-6: a dynamic OFDM or pulsed run evaluates the world once per symbol or
pulse, so it publishes ``"symbol"`` / ``"pulse"`` and is not exhaustive.
M2-10 / D-12: the slow-time carrier phase those schedules carry, checked against
an independent float64 geometric delay.
"""

from __future__ import annotations

import math

import numpy as np
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

    times = torch.as_tensor(result.sample_times[0], dtype=torch.float64)
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


# --------------------------------------------------------------------------
# M2-11: the default for a moving FMCW scene
# --------------------------------------------------------------------------


def _fmcw_mover(*, samples=8, chirps=4, **fields):
    radar = sc.radar(sc.beat_fmcw(samples=samples, chirps=chirps), **fields)
    return radar, _mover(radar.device)


def test_auto_resolves_to_adaptive_for_a_moving_fmcw_scene():
    radar, targets = _fmcw_mover()
    default = radar.simulate(sc.empty_scene(), targets, times=(0.0,), reflections=0)
    exact = radar.simulate(sc.empty_scene(), targets, times=(0.0,), reflections=0, motion=Motion.adc())
    assert default.motion_sampling == "adaptive" and default.motion_sampling_exhaustive is False
    assert exact.motion_sampling == "adc" and exact.motion_sampling_exhaustive is True
    error = float((default.cube - exact.cube).abs().norm() / exact.cube.abs().norm())
    assert error < 0.012, error
    # A declared cadence is the exhaustive schedule's knob, so it keeps that schedule.
    held = radar.simulate(
        sc.empty_scene(), targets, times=(0.0,), reflections=0, motion=Motion.auto(rediscover_every_chirps=1)
    )
    assert held.motion_sampling == "adc"


def test_the_adaptive_route_carries_a_declared_leakage():
    """The adaptive echo adds the board coupling the exhaustive route adds per observation."""

    from witwin.radar import Leakage

    radar, targets = _fmcw_mover(leakage=Leakage(isolation=30.0, delay=1e-10))
    kwargs = {"times": (0.0,), "reflections": 0}
    adaptive = radar.simulate(sc.empty_scene(), targets, motion=Motion.adaptive(), **kwargs)
    exact = radar.simulate(sc.empty_scene(), targets, motion=Motion.adc(), **kwargs)
    bare = radar.replace(leakage=None).simulate(sc.empty_scene(), targets, motion=Motion.adc(), **kwargs)
    coupling = float((exact.cube - bare.cube).abs().norm())
    assert coupling > 10 * float(bare.cube.abs().norm())
    assert float((adaptive.cube - exact.cube).abs().norm()) < 1e-3 * coupling


# --------------------------------------------------------------------------
# M2-5: discovery cadences between every observation and every N frames
# --------------------------------------------------------------------------


def test_a_per_chirp_cadence_discovers_once_per_chirp_and_replays_every_sample():
    radar, targets = _fmcw_mover(samples=8, chirps=4)
    kwargs = {"times": (0.0,), "reflections": 0}
    every = radar.simulate(sc.empty_scene(), targets, motion=Motion.adc(), **kwargs)
    per_chirp = radar.simulate(sc.empty_scene(), targets, motion=Motion.adc(rediscover_every_chirps=1), **kwargs)
    assert every.discovery_count == 4 * 8
    assert per_chirp.discovery_count == every.discovery_count // 8
    assert every.path_set_complete and not per_chirp.path_set_complete
    torch.testing.assert_close(per_chirp.cube, every.cube, rtol=1e-5, atol=0)
    # The lag bound is the discovery spacing actually incurred: one chirp.
    starts = every.sample_times[0].reshape(4, 8)[:, 0]
    assert per_chirp.birth_lag_bound[0] == pytest.approx(float(np.diff(starts).max()), rel=1e-9)
    assert every.birth_lag_bound[0] == pytest.approx(float(np.diff(every.sample_times[0]).max()), rel=1e-9)


def _expected_discoveries(times: np.ndarray, due) -> list[int]:
    """The observation indices a cadence rediscovers at, walked in time order."""

    found, last = [0], 0
    for index in range(1, len(times)):
        if due(last, index):
            found.append(index)
            last = index
    return found


def test_a_time_cadence_discovers_on_world_time():
    radar, targets = _fmcw_mover(samples=8, chirps=4)
    period = 2.5 * radar.waveform_spec().chirp_period_s
    result = radar.simulate(
        sc.empty_scene(), targets, times=(0.0,), reflections=0, motion=Motion.adc(rediscover_every_seconds=period)
    )
    times = result.sample_times[0]
    found = _expected_discoveries(times, lambda last, index: times[index] - times[last] >= period * (1 - 1e-9))
    assert result.discovery_count == len(found)
    gaps = np.diff(times[found]).tolist() + [times[-1] - times[found[-1]]]
    assert result.birth_lag_bound[0] == pytest.approx(max(gaps), rel=1e-9)
    assert not result.path_set_complete


def test_a_displacement_cadence_discovers_on_motion():
    radar, targets = _fmcw_mover(samples=8, chirps=4)
    step = SPEED * radar.waveform_spec().chirp_period_s * 1.5
    result = radar.simulate(
        sc.empty_scene(), targets, times=(0.0,), reflections=0, motion=Motion.adc(rediscover_on_displacement=step)
    )
    times = result.sample_times[0]
    position = START + SPEED * times
    found = _expected_discoveries(times, lambda last, index: abs(position[index] - position[last]) >= step)
    assert 1 < result.discovery_count == len(found) < len(times)
    assert not result.path_set_complete


@pytest.mark.parametrize(
    "build",
    [
        lambda: Motion.adc(rediscover_every_frames=1, rediscover_every_chirps=1),
        lambda: Motion.adc(rediscover_every_seconds=0.0),
        lambda: Motion.chirp(rediscover_on_displacement=-1.0),
    ],
)
def test_cadences_refuse_invalid_declarations(build):
    with pytest.raises(ValueError):
        build()


def test_adaptive_refuses_a_declared_cadence():
    radar, targets = _fmcw_mover()
    with pytest.raises(ValueError, match="owns its topology discovery cadence"):
        radar.simulate(
            sc.empty_scene(),
            targets,
            times=(0.0,),
            reflections=0,
            motion=Motion(kind="adaptive", rediscover_every_chirps=1),
        )
