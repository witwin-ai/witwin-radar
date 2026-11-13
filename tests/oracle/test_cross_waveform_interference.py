"""OFDM, pulsed and emitter interference against closed forms (R-ADR-051).

Two radars face each other ``D`` apart in free space (A's transmitter and B's
receiver exactly ``D`` apart, co-polarized), or one radar hears a declared
emitter ``D`` away. The heard signal is isolated as a difference of runs that
differ only in it (``direct=True`` minus ``direct=False``, or with and without
the emitter), and compared with the closed form at the Friis level
``sqrt(P_t) lambda / (4 pi D)``:

* an OFDM receiver demodulating another OFDM radar's identical symbols,
  offset ``dT`` inside the cyclic prefix: ``H_n = C exp(-j 2 pi n df (tau + dT))``
  on every subcarrier and symbol, ``C`` the Friis coefficient with phase
  ``-2 pi f_c tau``;
* the same with a carrier offset of ``eps`` subcarrier spacings: the
  demodulated signal-to-ICI ratio is ``|I0|^2 / (1 - |I0|^2)``,
  ``I0 = sin(pi eps) / (N sin(pi eps / N))``;
* a pulsed receiver sampling another pulsed radar's LFM pulse, offset ``dT``:
  ``y(t) = C s(t - tau - dT)`` sample by sample;
* an FMCW receiver hearing a CW emitter at ``f_c + d_E``: a burst exactly
  where ``|S v - d_E| <= f_s / 2``, at the Friis level;
* a pulsed receiver hearing a sampled tone ``f_0`` from an emitter at
  ``f_c + d_E``: a tone at ``f_0 + d_E`` at the Friis level.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("witwin.channel")

from support import oracle_scenes as sc

pytestmark = pytest.mark.gpu

C0 = 299_792_458.0
F_C = sc.CARRIER_HZ
WAVELENGTH = C0 / F_C
POWER_DBM = 10.0
POWER_W = 10 ** (POWER_DBM / 10) * 1e-3
SEPARATION = 10.0
TAU = SEPARATION / C0
AMPLITUDE = math.sqrt(POWER_W) * WAVELENGTH / (4 * math.pi * SEPARATION)


def _friis():
    """The Friis coefficient in the Channel convention, ``|C| exp(-j 2 pi f_c tau)``."""

    cycles = math.fmod(F_C * SEPARATION / C0, 1.0)
    return AMPLITUDE * np.exp(-2j * math.pi * cycles)


def _facing(waveform, other=None):
    from witwin.radar import Radar

    a = Radar(
        reference_frequency_hz=F_C,
        waveform=waveform,
        tx=((0.01, 0.0, 0.0),),
        rx=((-0.01, 0.0, 0.0),),
        power=POWER_DBM,
        look_at=(1.0, 0.0, 0.0),
        up=sc.UP,
    )
    b = Radar(
        reference_frequency_hz=F_C,
        waveform=waveform if other is None else other,
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


def _heard(network, *, frame_s=1e-3):
    """B's cube with A's line of sight minus without it, ``[slow, fast]``."""

    with_direct = network.simulate(sc.empty_scene(), _far_target(), times=(frame_s,), direct=True)[1]
    without = network.simulate(sc.empty_scene(), _far_target(), times=(frame_s,), direct=False)[1]
    return (with_direct.cube - without.cube)[0, 0, 0].cpu().numpy().astype(np.complex128)


def _ofdm(**fields):
    from witwin.radar import Ofdm

    values = {
        "subcarrier_spacing": 1e6,
        "num_subcarriers": 64,
        "cyclic_prefix": 0.5e-6,
        "num_symbols": 16,
        "max_expected_delay": 0.4e-6,
        "modulation": "qpsk",
    }
    values.update(fields)
    return Ofdm(**values)


def test_an_ofdm_receiver_demodulates_an_offset_ofdm_radar_inside_its_prefix():
    from witwin.radar.network import Network

    offset = 0.1e-6
    network = Network(_facing(_ofdm()), links=((0, 1),), clock_offsets=(offset, 0.0))
    observed = _heard(network)
    n = np.arange(64)
    expected = _friis() * np.exp(-2j * math.pi * n * 1e6 * (TAU + offset))
    np.testing.assert_allclose(observed, np.broadcast_to(expected, observed.shape), rtol=0, atol=2e-3 * AMPLITUDE)


def test_a_carrier_offset_between_ofdm_radars_is_inter_carrier_interference():
    from witwin.radar.network import Network

    eps, count = 0.2, 64
    network = Network(_facing(_ofdm(num_symbols=32)), links=((0, 1),), carrier_offsets=(eps * 1e6, 0.0))
    observed = _heard(network)
    aligned = observed * np.exp(2j * math.pi * np.arange(count) * 1e6 * TAU)
    # The common phase rotates from symbol to symbol; each symbol's mean is its signal term.
    signal = aligned.mean(axis=1, keepdims=True)
    sir = np.mean(np.abs(signal) ** 2) / np.mean(np.abs(aligned - signal) ** 2)
    i0 = abs(math.sin(math.pi * eps) / (count * math.sin(math.pi * eps / count))) ** 2
    assert 10 * math.log10(sir) == pytest.approx(10 * math.log10(i0 / (1 - i0)), abs=0.5)
    assert np.mean(np.abs(signal)) == pytest.approx(AMPLITUDE * math.sqrt(i0), rel=2e-2)


def test_a_pulsed_receiver_samples_an_offset_lfm_pulse():
    from witwin.radar import Pulsed
    from witwin.radar.network import Network

    width, bandwidth, rate, offset = 2e-6, 20e6, 50e6, 0.1e-6
    pulsed = Pulsed(
        pulse_kind="lfm",
        pulse_width=width,
        bandwidth=bandwidth,
        pri=20e-6,
        num_pulses=4,
        sample_rate=rate,
        num_samples=200,
        range_gate_start=0.0,
    )
    network = Network(_facing(pulsed), links=((0, 1),), clock_offsets=(offset, 0.0))
    observed = _heard(network)
    t = np.arange(200) / rate
    u = t - TAU - offset
    inside = (u >= 0) & (u < width)
    expected = np.where(inside, _friis() * np.exp(1j * math.pi * bandwidth * u**2 / width), 0.0)
    np.testing.assert_allclose(observed, np.broadcast_to(expected, observed.shape), rtol=0, atol=2e-3 * AMPLITUDE)


def _emitter(samples, rate, carrier_offset):
    from witwin.radar import Emitter

    # One metre-scale path: the receiver of the radar at the origin is at (0, 0, -0.01).
    return Emitter(
        position=(SEPARATION, 0.0, -0.01), samples=samples, rate=rate, power=POWER_DBM, carrier_offset=carrier_offset
    )


def test_an_fmcw_receiver_hears_a_cw_emitter_as_a_burst_where_its_ramp_crosses_it():
    from witwin.radar import Fmcw
    from witwin.radar.network import Network

    slope, rate, start = 10e12, 10e6, 2.5e-6
    crossing = 8.53e-6
    waveform = Fmcw(
        slope=slope,
        sample_rate=rate,
        samples_per_chirp=128,
        chirps_per_frame=2,
        adc_start=start,
        idle=10e-6,
        ramp_end=20e-6,
        output="beat",
    )
    radar = _facing(waveform)[0]
    emitter = _emitter((1.0,), 1e6, slope * crossing)
    heard = Network((radar,), emitters=(emitter,)).simulate(sc.empty_scene(), _far_target(), times=(1e-3,))[0]
    alone = radar.simulate(sc.empty_scene(), _far_target(), times=(1e-3,))
    observed = (heard.cube - alone.cube)[0, 0, 0].cpu().numpy()
    v = start + np.arange(128) / rate
    inside = np.abs(slope * v - slope * crossing) <= rate / 2
    assert int(inside.sum()) == 10
    for chirp in range(2):
        assert np.array_equal(np.abs(observed[chirp]) > 0, inside)
        np.testing.assert_allclose(np.abs(observed[chirp][inside]), AMPLITUDE, rtol=1e-3)


def test_a_pulsed_receiver_hears_an_emitter_tone_at_its_frequency():
    from witwin.radar import Pulsed
    from witwin.radar.network import Network

    rate, tone, offset = 50e6, 1.0e6, 0.5e6
    pulsed = Pulsed(
        pulse_kind="rect",
        pulse_width=1e-6,
        bandwidth=1e6,
        pri=20e-6,
        num_pulses=2,
        sample_rate=rate,
        num_samples=400,
        range_gate_start=0.0,
    )
    radar = _facing(pulsed)[0]
    table = np.exp(2j * math.pi * tone * np.arange(40) / 10e6)
    emitter = _emitter(table, 10e6, offset)
    heard = Network((radar,), emitters=(emitter,)).simulate(sc.empty_scene(), _far_target(), times=(1e-3,))[0]
    alone = radar.simulate(sc.empty_scene(), _far_target(), times=(1e-3,))
    observed = (heard.cube - alone.cube)[0, 0, 0].cpu().numpy().astype(np.complex128)
    np.testing.assert_allclose(np.abs(observed), AMPLITUDE, rtol=1e-3)
    measured = np.angle(observed[:, 1:] * np.conj(observed[:, :-1])).mean() * rate / (2 * math.pi)
    assert measured == pytest.approx(tone + offset, rel=1e-5)
