"""D-1, D-2: carrier-phase precision of a moving target against a float64 trajectory.

The trajectory is a closed form, ``p(t) = p0 + v t + a t^2 / 2``, evaluated
twice: in float64 by the reference (``tests/reference/radar_oracles.py``), and
in float64 then rounded to float32 by the callable production consumes,
because float32 is what a site position is. The reference therefore does not
share the float32 trajectory, which is the gap D-2 names in the older motion
tests.

Radar: one co-located element at the origin, a beat-domain FMCW ramp of 64
samples at 10 MS/s, two chirps 1 ms apart (so the acceleration case moves the
phase by ~0.65 rad between chirps), ``Motion.adc()`` so every sample is its own
world observation. The reference is the quasi-static beat
``A exp(j 2 pi (f_c tau + K tau (t_local - tau / 2)))`` with
``tau = 2 |p(t)| / c`` at each sample instant and ``A`` from the radar
equation; the light-time correction is outside the model on both sides.

Two figures of merit, both relative to the reference echo:

* integrated error, ``10 log10(sum |x - r|^2 / sum |r|^2)``;
* fast-time spur floor, the largest bin of ``|FFT(x - r)|^2`` over the largest
  bin of ``|FFT(r)|^2``, per chirp, worst chirp.

The expectation is the plan's M2-2 target, both below -80 dBc, at 2, 30 and
100 m. Today the float32 site position and float32 Channel range make the
carrier phase a staircase (D-1); the marked cases fail strictly until that is
fixed. A passing envelope test pins that the reference is the right model, so
the strict failures measure precision and not a model mismatch.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from reference import radar_oracles as ref
from support import oracle_scenes as sc

pytestmark = pytest.mark.gpu

SAMPLES = 64
CHIRPS = 2
SAMPLE_RATE_HZ = 10e6
SLOPE_HZ_PER_S = 30e12
IDLE_S = 1e-3
POWER_DBM = 10.0
RANGES_M = (2.0, 30.0, 100.0)
#: name -> (velocity m/s, acceleration m/s^2), both along the line of sight.
TRAJECTORIES = {"constant-velocity": (10.0, 0.0), "constant-acceleration": (1.0, 400.0)}
#: The M2-2 acceptance level for both figures of merit, dBc.
TARGET_DBC = -80.0

_D1 = (
    "D-1: site positions and Channel ranges are float32, so a moving target's carrier phase is a staircase "
    "of 4 pi ulp(R) / lambda; the M2-2 target is a spur floor and integrated error below -80 dBc"
)

#: Measured on Channel 0.5.0 (R-ADR-032 has the table): every case misses
#: -80 dBc today. Integrated error -73 / -55 / -43 dBc and spur floor
#: -76 / -57 / -47 dBc at 2 / 30 / 100 m, for both trajectories.
_FAILS_TODAY = {(r, name) for r in RANGES_M for name in TRAJECTORIES}

_CACHE: dict = {}


def _run(range_m: float, trajectory_name: str):
    """Production beat and the float64 reference, ``complex128[chirp, sample]``."""

    key = (range_m, trajectory_name)
    if key in _CACHE:
        return _CACHE[key]

    from witwin.radar import Fmcw, Motion, PointTargets

    velocity, acceleration = TRAJECTORIES[trajectory_name]
    origin, v, a = (range_m, 0.0, 0.0), (velocity, 0.0, 0.0), (acceleration, 0.0, 0.0)
    waveform = Fmcw(
        slope=SLOPE_HZ_PER_S,
        sample_rate=SAMPLE_RATE_HZ,
        samples_per_chirp=SAMPLES,
        chirps_per_frame=CHIRPS,
        adc_start=0.0,
        idle=IDLE_S,
        ramp_end=SAMPLES / SAMPLE_RATE_HZ + 1e-6,
        output="beat",
    )
    radar = sc.radar(waveform, power_dbm=POWER_DBM)

    def trajectory(time_s):
        position = ref.ballistic_position(np.asarray(float(time_s)), origin=origin, velocity=v, acceleration=a)
        return torch.tensor(position[None], dtype=torch.float32, device=radar.device)

    result = radar.simulate(
        sc.empty_scene(),
        PointTargets(positions=trajectory(0.0), rcs=1.0, trajectory=trajectory),
        times=(0.0,),
        los=True,
        reflections=0,
        motion=Motion.adc(),
    )
    assert result.motion_sampling == "adc"
    produced = result.cube[0, 0, 0].to(torch.complex128).cpu().numpy()

    # The sample schedule from the waveform's own definition, not from the
    # result: chirp c, sample m is at c * T_chirp + t_start + m / f_s.
    chirp_period = IDLE_S + waveform.ramp_end
    local = np.arange(SAMPLES) / SAMPLE_RATE_HZ
    instants = np.arange(CHIRPS)[:, None] * chirp_period + local[None, :]
    positions = ref.ballistic_position(instants, origin=origin, velocity=v, acceleration=a)
    beat = ref.monostatic_beat(
        positions_m=positions,
        transmitter=(0.0, 0.0, 0.0),
        receiver=(0.0, 0.0, 0.0),
        local_time_s=local[None, :],
        carrier_hz=sc.CARRIER_HZ,
        slope_hz_per_s=SLOPE_HZ_PER_S,
    )
    distance = np.linalg.norm(positions, axis=-1)
    wavelength = ref.wavelength_m(sc.CARRIER_HZ)
    amplitude = np.sqrt(
        ref.bistatic_received_power_w(
            transmit_power_w=ref.watts_from_dbm(POWER_DBM),
            gain_tx=1.0,
            gain_rx=1.0,
            wavelength=wavelength,
            rcs_m2=1.0,
            r_in=1.0,
            r_out=1.0,
        )
    ) / (distance * distance)
    _CACHE[key] = (produced, amplitude * beat)
    return _CACHE[key]


def figures_of_merit(produced: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    error = produced - reference
    integrated = ref.db(float(np.sum(np.abs(error) ** 2) / np.sum(np.abs(reference) ** 2)))
    error_spectrum = np.abs(np.fft.fft(error, axis=-1)) ** 2
    echo_spectrum = np.abs(np.fft.fft(reference, axis=-1)) ** 2
    spur = ref.db(float(np.max(error_spectrum.max(axis=-1) / echo_spectrum.max(axis=-1))))
    phase = np.angle(produced * np.conj(reference))
    return {
        "integrated_dbc": integrated,
        "spur_floor_dbc": spur,
        "phase_rms_rad": float(np.sqrt(np.mean(phase**2))),
        "phase_peak_rad": float(np.max(np.abs(phase))),
        "amplitude_ratio_max_dev": float(np.max(np.abs(np.abs(produced / reference) - 1.0))),
    }


def _ulp32(value: float) -> float:
    return float(np.spacing(np.float32(value)))


@pytest.mark.parametrize("trajectory_name", list(TRAJECTORIES))
@pytest.mark.parametrize("range_m", RANGES_M)
def test_production_stays_inside_the_float32_envelope_of_the_float64_reference(range_m, trajectory_name):
    """The reference is the production model up to float32 rounding.

    Amplitude to 1e-5, and phase to ``4 k ulp32(R)``: rounding the position
    and each leg's float32 range moves the round-trip length by at most a few
    ulps. A model mismatch (a wrong range term, a missed acceleration, a
    sign) exceeds this by orders of magnitude.
    """

    merit = figures_of_merit(*_run(range_m, trajectory_name))
    envelope = 4.0 * (2.0 * math.pi / ref.wavelength_m(sc.CARRIER_HZ)) * _ulp32(range_m)
    assert merit["amplitude_ratio_max_dev"] < 1e-5, merit
    assert merit["phase_peak_rad"] < envelope, (merit, envelope)


def _precision_cases():
    for range_m in RANGES_M:
        for name in TRAJECTORIES:
            marks = [pytest.mark.xfail(strict=True, reason=_D1)] if (range_m, name) in _FAILS_TODAY else []
            yield pytest.param(range_m, name, marks=marks, id=f"{range_m:g}m-{name}")


@pytest.mark.parametrize(("range_m", "trajectory_name"), list(_precision_cases()))
def test_moving_target_phase_error_and_spur_floor_meet_the_m2_2_target(range_m, trajectory_name):
    merit = figures_of_merit(*_run(range_m, trajectory_name))
    assert merit["spur_floor_dbc"] < TARGET_DBC and merit["integrated_dbc"] < TARGET_DBC, merit
