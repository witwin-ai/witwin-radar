"""FMCW mutual interference and bistatic returns against closed forms (R-ADR-051).

Two radars in free space. The interferer A transmits; the victim B receives.
The direct line of sight from A's transmitter to B's receiver is isolated as
the difference of B's cube with and without ``direct=True`` (every other row
is the same in both), and a bistatic target return as the difference with and
without the link.

The oracle for every sample is built from the signals themselves, in long
double with the full carrier phase: A radiates
``exp(j 2 pi [(f_c + d_A) T + S_A w^2 / 2])`` on its ramps (``T`` world time,
``w`` time into A's ramp), B mixes its LO ``exp(j 2 pi [(f_c + d_B) T + S_B v^2 / 2])``
against the conjugate of the received field, whose free-space amplitude is
Friis' ``sqrt(P_t) lambda / (4 pi D)``, and B's IF band passes
``|f_beat| <= f_s / 2``. On top of the sample oracle, each test checks the
classic signature in closed form:

* same slope, start-time offset ``dT``: a ghost tone at ``S (tau + dT)``,
  i.e. a target at ``c (tau + dT) / 2``, at zero Doppler;
* slope mismatch: a burst exactly where ``|(S_B - S_A) v + S_A (tau + dT)| <= f_s / 2``,
  ``2 (f_s / 2) / |S_A - S_B|`` long, spread over the whole range spectrum;
* a start-frequency offset ``d_A``: the ghost moves by ``-d_A`` in beat
  frequency and rotates by ``-2 pi d_A T_c`` from chirp to chirp;
* synchronized radars and a point target: the bistatic radar equation and a
  beat at ``S (R_1 + R_2) / c``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("witwin.channel")

from support import oracle_scenes as sc

pytestmark = pytest.mark.gpu

C = 299_792_458.0
F_C = sc.CARRIER_HZ
WAVELENGTH = C / F_C
POWER_DBM = 10.0
POWER_W = 10 ** (POWER_DBM / 10) * 1e-3
SAMPLE_RATE = 10e6
SAMPLES = 128
ADC_START = 2.5e-6
RAMP_END = 20e-6
IDLE = 10e-6
CHIRPS = 4
SEPARATION = 10.0


def _waveform(slope, start_offset=0.0):
    from witwin.radar import Chirp, Fmcw

    return Fmcw(
        slope=slope,
        sample_rate=SAMPLE_RATE,
        samples_per_chirp=SAMPLES,
        chirps_per_frame=CHIRPS,
        adc_start=ADC_START,
        idle=IDLE,
        ramp_end=RAMP_END,
        output="beat",
        sequence=(Chirp(start_offset=start_offset),) if start_offset else (),
    )


def _facing(slope_a, slope_b, start_offset_a=0.0):
    """A at the origin looking +x, B at (D, 0, 0) looking back; A's transmitter and B's receiver D apart."""

    from witwin.radar import Radar

    a = Radar(
        reference_frequency_hz=F_C,
        waveform=_waveform(slope_a, start_offset_a),
        tx=((0.01, 0.0, 0.0),),
        rx=((-0.01, 0.0, 0.0),),
        power=POWER_DBM,
        look_at=(1.0, 0.0, 0.0),
        up=sc.UP,
    )
    b = Radar(
        reference_frequency_hz=F_C,
        waveform=_waveform(slope_b),
        tx=((0.01, 0.0, 0.0),),
        rx=((-0.01, 0.0, 0.0),),
        power=POWER_DBM,
        position=(SEPARATION, 0.0, 0.0),
        look_at=(0.0, 0.0, 0.0),
        up=sc.UP,
    )
    return a, b


def _far_target():
    from witwin.radar import PointTargets

    return PointTargets(positions=[[5.0, 0.0, 30.0]], rcs=1e-4)


def _direct_interference(a, b, *, offset_s, frame_s=1e-3):
    """B's cube with A's line of sight minus B's cube without it, ``[chirp, sample]``."""

    from witwin.radar.network import Network

    network = Network((a, b), links=((0, 1),), clock_offsets=(offset_s, 0.0))
    with_direct = network.simulate(sc.empty_scene(), _far_target(), times=(frame_s,), direct=True)[1]
    without = network.simulate(sc.empty_scene(), _far_target(), times=(frame_s,), direct=False)[1]
    return (with_direct.cube - without.cube)[0, 0, 0].cpu().numpy().astype(np.complex128)


def _signal_oracle(*, slope_a, slope_b, offset_s, tau, amplitude, start_offset_a=0.0, frame_s=1e-3):
    """B's samples from the two signals in long double, ``[chirp, sample]``."""

    f_c = np.longdouble(F_C)
    d_a, d_b = np.longdouble(start_offset_a), np.longdouble(0.0)
    s_a, s_b = np.longdouble(slope_a), np.longdouble(slope_b)
    period = np.longdouble(RAMP_END + IDLE)
    frame_b = np.longdouble(frame_s)
    frame_a = frame_b + np.longdouble(offset_s)
    tau = np.longdouble(tau)
    out = np.zeros((CHIRPS, SAMPLES), dtype=np.complex128)
    for chirp in range(CHIRPS):
        for n in range(SAMPLES):
            v = np.longdouble(ADC_START) + np.longdouble(n) / np.longdouble(SAMPLE_RATE)
            absolute = frame_b + chirp * period + v
            emitted = absolute - tau - frame_a
            slot = np.floor(emitted / period)
            w = emitted - slot * period
            if slot < 0 or slot >= CHIRPS or w >= RAMP_END:
                continue
            if abs(d_b - d_a + s_b * v - s_a * w) > SAMPLE_RATE / 2:
                continue
            lo = (f_c + d_b) * absolute + s_b * v * v / 2
            tx = (f_c + d_a) * (absolute - tau) + s_a * w * w / 2
            # The received field is the delayed transmission at the Friis level: LO * conj(tx(T - tau)).
            cycles = lo - tx
            out[chirp, n] = amplitude * np.exp(2j * np.pi * float(cycles - np.floor(cycles)))
    return out


def _friis():
    tau = SEPARATION / C
    return tau, math.sqrt(POWER_W) * WAVELENGTH / (4 * math.pi * SEPARATION)


def _tolerance(amplitude):
    return 2e-3 * amplitude


def test_a_same_slope_interferer_offset_in_time_is_a_ghost_target():
    slope, offset = 30e12, 0.1e-6
    tau, amplitude = _friis()
    observed = _direct_interference(*_facing(slope, slope), offset_s=offset)
    expected = _signal_oracle(slope_a=slope, slope_b=slope, offset_s=offset, tau=tau, amplitude=amplitude)
    np.testing.assert_allclose(observed, expected, rtol=0, atol=_tolerance(amplitude))
    # Every sample hears the ghost at the Friis level.
    np.testing.assert_allclose(np.abs(observed), amplitude, rtol=1e-3)
    # A tone at S (tau + dT): a target at c (tau + dT) / 2, with zero Doppler.
    beat = slope * (tau + offset)
    measured = np.angle(observed[:, 1:] * np.conj(observed[:, :-1])).mean() * SAMPLE_RATE / (2 * math.pi)
    assert measured == pytest.approx(beat, rel=1e-5)
    spectrum = np.abs(np.fft.fft(observed, axis=1)).sum(axis=0)
    assert int(np.argmax(spectrum)) == round(beat * SAMPLES / SAMPLE_RATE)
    np.testing.assert_allclose(np.angle(observed[1:, 0] * np.conj(observed[:-1, 0])), 0.0, atol=1e-4)


def test_a_slope_mismatch_crossing_is_a_broadband_burst():
    slope_a, slope_b = 40e12, 30e12
    tau, amplitude = _friis()
    # Off the sample grid, so no sample sits exactly on the IF band edge.
    crossing = 8.53e-6
    # Choose dT so the beat (S_B - S_A) v + S_A (tau + dT) crosses zero mid-window.
    offset = (slope_a - slope_b) * crossing / slope_a - tau
    observed = _direct_interference(*_facing(slope_a, slope_b), offset_s=offset)
    expected = _signal_oracle(slope_a=slope_a, slope_b=slope_b, offset_s=offset, tau=tau, amplitude=amplitude)
    np.testing.assert_allclose(observed, expected, rtol=0, atol=_tolerance(amplitude))
    v = ADC_START + np.arange(SAMPLES) / SAMPLE_RATE
    inside = np.abs((slope_b - slope_a) * (v - crossing)) <= SAMPLE_RATE / 2
    for chirp in range(CHIRPS):
        assert np.array_equal(np.abs(observed[chirp]) > 0.5 * amplitude, inside)
        assert np.all(observed[chirp][~inside] == 0)
    # The burst lasts 2 (f_s / 2) / |S_A - S_B| and spreads over the spectrum.
    assert int(inside.sum()) == pytest.approx(SAMPLE_RATE / abs(slope_a - slope_b) * SAMPLE_RATE, abs=1)
    power = np.abs(np.fft.fft(observed[0])) ** 2
    assert power.max() < 0.2 * power.sum()
    assert np.count_nonzero(power > 0.1 * power.max()) > SAMPLES // 4


def test_a_start_frequency_offset_shifts_and_rotates_the_ghost():
    slope, offset, start_offset = 30e12, 0.1e-6, 2e3
    tau, amplitude = _friis()
    observed = _direct_interference(*_facing(slope, slope, start_offset_a=start_offset), offset_s=offset)
    expected = _signal_oracle(
        slope_a=slope, slope_b=slope, offset_s=offset, tau=tau, amplitude=amplitude, start_offset_a=start_offset
    )
    np.testing.assert_allclose(observed, expected, rtol=0, atol=_tolerance(amplitude))
    step = np.angle(observed[1:, 0] * np.conj(observed[:-1, 0]))
    np.testing.assert_allclose(step, -2 * math.pi * start_offset * (RAMP_END + IDLE), atol=1e-4)
    measured = np.angle(observed[:, 1:] * np.conj(observed[:, :-1])).mean() * SAMPLE_RATE / (2 * math.pi)
    assert measured == pytest.approx(slope * (tau + offset) - start_offset, rel=1e-5)


def test_synchronized_radars_hear_a_target_by_the_bistatic_radar_equation():
    from witwin.radar import PointTargets, Radar
    from witwin.radar.network import Network

    slope = 30e12
    radars = tuple(
        Radar(
            reference_frequency_hz=F_C,
            waveform=_waveform(slope),
            tx=((0.0, 0.0, 0.0),),
            rx=((0.0, 0.0, 0.0),),
            power=POWER_DBM,
            position=position,
            look_at=(1.0, 0.0, position[2]),
            up=sc.UP,
        )
        for position in ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    )
    target = np.array([5.0, 0.0, 0.4])
    targets = PointTargets(positions=[target.tolist()], rcs=1.0)
    linked = Network(radars, links=((0, 1),)).simulate(sc.empty_scene(), targets, times=(0.0,), direct=False)[1]
    alone = Network(radars).simulate(sc.empty_scene(), targets, times=(0.0,))[1]
    observed = (linked.cube - alone.cube)[0, 0, 0].cpu().numpy().astype(np.complex128)
    first = float(np.linalg.norm(target))
    second = float(np.linalg.norm(target - np.array([0.0, 0.0, 1.0])))
    amplitude = math.sqrt(POWER_W * WAVELENGTH**2 / ((4 * math.pi) ** 3 * first**2 * second**2))
    np.testing.assert_allclose(np.abs(observed), amplitude, rtol=1e-3)
    tau = (first + second) / C
    v = ADC_START + np.arange(SAMPLES) / SAMPLE_RATE
    ratio = observed * np.conj(observed[:, :1])
    np.testing.assert_allclose(np.angle(ratio * np.exp(-2j * math.pi * slope * tau * (v - v[0]))), 0.0, atol=2e-3)
    # Synchronized clocks and one carrier: no Doppler on a still target.
    np.testing.assert_allclose(np.angle(observed[1:, 0] * np.conj(observed[:-1, 0])), 0.0, atol=1e-4)
