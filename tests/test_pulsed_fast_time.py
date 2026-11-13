"""M2-9 / D-5: fast-time refreshed observations for a pulsed train.

Under ``Motion.adc()`` a pulsed train now observes the world at every range-gate
sample, the FMCW observation pattern, instead of once per pulse. Three analytic
oracles on a point target receding at constant radial speed ``v``:

* intra-pulse Doppler. Inside a rectangular echo, consecutive samples rotate
  by ``-2 pi f_c tau_dot T_s`` (Channel convention, ``tau_dot = 2 v / c``); a
  per-pulse observation holds the phase constant across the pulse.
* the sample-level closed form. The refreshed train divided by the per-pulse
  train is ``exp(-j 2 pi f_c tau_dot (t_g + m T_s))`` on every in-pulse sample
  (the delay change since the pulse start), to float32 phase precision.
* LFM range-Doppler coupling. An up-chirp of slope ``k = B / T_p`` carrying a
  Doppler tone ``f_D = -f_c tau_dot`` compresses at ``tau + f_c tau_dot / k``:
  3.85 samples late here. A per-pulse observation compresses at ``tau``.
"""

from __future__ import annotations

import math

import pytest
import torch
from support import oracle_scenes as sc
from support import pulsed_grid

from witwin.radar import Motion, PointTargets, Pulsed
from witwin.radar.policy import SPEED_OF_LIGHT_M_PER_S as C0

pytestmark = pytest.mark.gpu

#: Near and fast on purpose. A float32 position at 1.5 m has a 1.2e-7 m ulp, and
#: one sample of motion at 300 m/s is 1.2e-5 m, so the per-sample delay change is
#: a hundred ulps rather than one (R-ADR-022's D-1 phase staircase is measured
#: elsewhere and is not what this file tests).
RANGE_M = 1.5
SPEED_MPS = 300.0
TAU_DOT = 2.0 * SPEED_MPS / C0


def _waveform(kind: str) -> Pulsed:
    return Pulsed(
        pulse_kind=kind,
        pulse_width=10e-6,
        bandwidth=10e6 if kind == "lfm" else 1.0 / 10e-6,
        pri=20e-6,
        num_pulses=2,
        sample_rate=25e6,
        num_samples=288,
        range_gate_start=0.0,
        max_expected_delay_rate=2.5e-6,
    )


def _run(kind: str, motion: Motion):
    radar = sc.radar(_waveform(kind))
    origin = torch.tensor([[RANGE_M, 0.0, 0.0]], device=radar.device)
    direction = torch.tensor([[1.0, 0.0, 0.0]], device=radar.device)
    targets = PointTargets(positions=origin, rcs=1.0, trajectory=lambda t: origin + SPEED_MPS * t * direction)
    result = radar.simulate(sc.empty_scene(), targets, times=(0.0,), los=True, reflections=0, motion=motion)
    return radar, result


@pytest.fixture(scope="module")
def rect_runs():
    return {name: _run("rect", motion) for name, motion in (("fast", Motion.adc()), ("slow", Motion.chirp()))}


def test_the_published_schedule_is_what_ran(rect_runs):
    radar, fast = rect_runs["fast"]
    _, slow = rect_runs["slow"]
    assert fast.observation_schedule == "fast_time"
    assert slow.observation_schedule == "slow_time"
    spec = radar.waveform_spec()
    # One observation per pulse and gate sample, at its own absolute instant.
    times = fast.sample_times[0]
    assert len(times) == spec.num_pulses * spec.num_samples
    assert times[spec.num_samples + 3] == pytest.approx(spec.pri_s + 3 * spec.sample_period_s, abs=1e-15)


def _in_pulse(spec, pulse: int, margin: int = 2) -> slice:
    tau = 2.0 * (RANGE_M + SPEED_MPS * pulse * spec.pri_s) / C0
    first = math.ceil(tau / spec.sample_period_s) + margin
    last = math.floor((tau + spec.pulse_width_s) / spec.sample_period_s) - margin
    return slice(first, last)


def test_a_refreshed_pulse_rotates_at_the_intra_pulse_doppler(rect_runs):
    radar, fast = rect_runs["fast"]
    _, slow = rect_runs["slow"]
    spec = radar.waveform_spec()
    window = _in_pulse(spec, 0)
    refreshed = fast.cube[0, 0, 0, 0, window].to(torch.complex128).cpu()
    held = slow.cube[0, 0, 0, 0, window].to(torch.complex128).cpu()
    step = torch.angle(refreshed[1:] * refreshed[:-1].conj())
    expected = -2.0 * math.pi * radar.reference_frequency_hz * TAU_DOT * spec.sample_period_s
    assert float(step.mean()) == pytest.approx(expected, rel=1e-3)
    # The residual jitter is the float32 position rounding of each instant.
    assert float(step.std()) < 2e-2 * abs(expected)
    # Stop and hop: the per-pulse observation is one phasor across the pulse.
    assert float(torch.angle(held[1:] * held[:-1].conj()).abs().max()) < 1e-5


def test_the_refreshed_train_is_the_per_pulse_train_times_the_delay_change(rect_runs):
    radar, fast = rect_runs["fast"]
    _, slow = rect_runs["slow"]
    spec = radar.waveform_spec()
    for pulse in range(spec.num_pulses):
        window = _in_pulse(spec, pulse)
        ratio = (fast.cube[0, 0, 0, pulse, window] / slow.cube[0, 0, 0, pulse, window]).to(torch.complex128).cpu()
        elapsed = torch.arange(window.start, window.stop, dtype=torch.float64) * spec.sample_period_s
        start = RANGE_M + SPEED_MPS * pulse * spec.pri_s
        # Two one-way spreadings: the amplitude goes as 1 / r^2.
        expected = (start / (start + SPEED_MPS * elapsed)) ** 2 * torch.exp(
            -2j * math.pi * radar.reference_frequency_hz * TAU_DOT * elapsed
        )
        # Float32 positions: half an ulp at 1.5 m is 4e-4 rad of round-trip phase.
        assert float((ratio / expected).angle().abs().max()) < 2e-3
        assert float((ratio.abs() / expected.abs() - 1.0).abs().max()) < 1e-4


def test_lfm_range_doppler_coupling_appears_only_with_fast_time_observations():
    radar, fast = _run("lfm", Motion.adc())
    _, slow = _run("lfm", Motion.chirp())
    spec = radar.waveform_spec()
    k = spec.bandwidth_hz / spec.pulse_width_s
    coupling = radar.reference_frequency_hz * TAU_DOT / k
    assert coupling / spec.sample_period_s == pytest.approx(3.85, abs=0.01)
    for pulse in range(spec.num_pulses):
        start = pulse * spec.pri_s
        tau_start = 2.0 * (RANGE_M + SPEED_MPS * start) / C0
        tau_mid = tau_start * (1.0 + TAU_DOT) + TAU_DOT * spec.pulse_width_s / 2.0
        held, _, _ = pulsed_grid.peak_estimate(slow.cube[0, 0, 0, pulse].to(torch.complex128), spec)
        refreshed, _, _ = pulsed_grid.peak_estimate(fast.cube[0, 0, 0, pulse].to(torch.complex128), spec)
        # The parabolic fit's own floor is about 1e-10 s on this grid.
        assert held - tau_start == pytest.approx(0.0, abs=3e-10)
        assert refreshed - tau_mid == pytest.approx(coupling, rel=0.02)
