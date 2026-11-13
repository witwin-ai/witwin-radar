"""FE-1: post-integration SNR per waveform against the radar equation and kTB.

One co-located element, one isotropic ``1 m^2`` target at 15 m on boresight,
a receiver at ``T_sys = 290 K`` (0 dB noise figure, 290 K antenna). The
physical expectation is the matched-filter SNR

    SNR = E_r / N_0 = P_r T_obs / (k T_sys),
    P_r = P_t lambda^2 sigma / ((4 pi)^3 R^4),

with ``T_obs`` the time the echo is actually observed (FMCW: chirps times
samples over the sample rate; pulsed: pulses times pulse width, the pulse
lying wholly inside the range gate; OFDM: symbols times the useful symbol time
``1 / delta_f``, the cyclic prefix excluded). A coded pulse observes its
envelope energy ``int |s|^2`` per pulse, a continuous code its whole gate. An
in-phase-only receiver keeps the matched SNR (signal and noise both halve), and
DDM and BPM, firing every transmitter every chirp, multiply it by the
transmitter count.

The measured SNR is normalisation-free: run the radar twice with the same
seed, once with the noise source at 0 K (the same receive chain, no noise),
once at 290 K. The first cube is the signal ``s``, the difference is the noise
``n``, and the matched filter's output SNR on white noise is
``sum |s|^2 / E|n|^2``. Any unitary transform of the cube (the FMCW range
spectrum) leaves it unchanged, which is why both FMCW output domains are held
to the same number.
"""

from __future__ import annotations

import math

import pytest
import torch
from reference import radar_oracles as ref
from support import oracle_scenes as sc

pytestmark = pytest.mark.gpu

RANGE_M = 15.0
POWER_DBM = 10.0
#: The noise variance is estimated from >= 2560 complex samples, ~2% standard
#: error (0.09 dB); 0.5 dB is five standard errors and far below any
#: normalisation slip worth catching (a factor of two is 3 dB).
TOLERANCE_DB = 0.5


#: Two co-located transmitters: under DDM and BPM both fire every chirp, so the
#: coherent SNR is the transmitter count times one transmitter's.
TWO_TX = {"tx": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))}


def _waveforms():
    from witwin.radar import Fmcw, Ofdm, Pulsed

    def fmcw(output, **fields):
        return Fmcw(
            slope=30e12,
            sample_rate=10e6,
            samples_per_chirp=256,
            chirps_per_frame=16,
            adc_start=0.0,
            idle=5e-6,
            ramp_end=26.6e-6,
            output=output,
            **fields,
        )

    arbitrary = Pulsed.arbitrary(
        [0.8 * complex(math.cos(0.37 * k * k), math.sin(0.37 * k * k)) for k in range(250)],
        rate=50e6,
        pri=50e-6,
        num_pulses=4,
        sample_rate=50e6,
        num_samples=640,
    )

    return {
        "fmcw-beat": (fmcw("beat"), 16 * 256 / 10e6),
        "fmcw-spectrum": (fmcw("spectrum"), 16 * 256 / 10e6),
        "ofdm": (
            Ofdm(
                subcarrier_spacing=480e3,
                num_subcarriers=256,
                cyclic_prefix=0.5e-6,
                num_symbols=16,
                max_expected_delay=0.2e-6,
            ),
            16 / 480e3,
        ),
        "pulsed": (
            Pulsed(
                pulse_kind="lfm",
                pulse_width=10e-6,
                bandwidth=20e6,
                pri=50e-6,
                num_pulses=4,
                sample_rate=50e6,
                num_samples=640,
                range_gate_start=0.0,
            ),
            4 * 10e-6,
        ),
        # R-ADR-046/047/048: every new waveform keeps the noise contract.
        "fmcw-real-iq": (fmcw("beat", sampling="real"), 16 * 256 / 10e6),
        "fmcw-ddm": (fmcw("beat", mimo="ddm"), 16 * 256 / 10e6, TWO_TX, 2.0),
        "fmcw-bpm": (fmcw("beat", mimo="bpm"), 16 * 256 / 10e6, TWO_TX, 2.0),
        "ofdm-qpsk": (
            Ofdm(
                subcarrier_spacing=480e3,
                num_subcarriers=256,
                cyclic_prefix=0.5e-6,
                num_symbols=16,
                max_expected_delay=0.2e-6,
                modulation="qpsk",
                pilot_spacing=8,
                symbol_seed=3,
            ),
            16 / 480e3,
        ),
        "pmcw": (Pulsed.pmcw(Pulsed.m_sequence(7), chip_rate=50e6, repetitions=8, sample_rate=50e6), 8 * 127 / 50e6),
        "fsk": (
            Pulsed.fsk(
                [2.5e6 * value for value in Pulsed.costas(7)],
                chip_width=1e-6,
                pri=50e-6,
                num_pulses=4,
                sample_rate=50e6,
                num_samples=640,
            ),
            4 * 6e-6,
        ),
        "stepped": (
            Pulsed.stepped_frequency(
                step=1e6, num_steps=8, pulse_width=2e-6, pri=20e-6, sample_rate=50e6, num_samples=200
            ),
            8 * 2e-6,
        ),
        "arbitrary": (arbitrary, 4 * arbitrary.to_spec(reference_frequency_hz=sc.CARRIER_HZ).envelope_energy_s),
    }


def _cases():
    for name in _waveforms():
        yield pytest.param(name, id=name)


def measured_and_expected_snr(name: str) -> tuple[float, float]:
    from witwin.radar import Noise, PointTargets

    waveform, observed_time_s, *extra = _waveforms()[name]
    fields, gain = (extra + [{}, 1.0])[:2] if extra else ({}, 1.0)
    cubes = {}
    for key, noise in (("signal", Noise(antenna_temperature=0.0)), ("noisy", Noise())):
        radar = sc.radar(waveform, power_dbm=POWER_DBM, noise=noise, seed=7, **fields)
        result = radar.simulate(
            sc.empty_scene(),
            PointTargets(positions=[[RANGE_M, 0.0, 0.0]], rcs=1.0),
            times=(0.0,),
            los=True,
            reflections=0,
        )
        cubes[key] = result.cube.to(torch.complex128)
    signal = cubes["signal"]
    noise = cubes["noisy"] - signal
    measured = float(signal.abs().pow(2).sum()) / float(noise.abs().pow(2).mean())

    received = ref.bistatic_received_power_w(
        transmit_power_w=ref.watts_from_dbm(POWER_DBM),
        gain_tx=1.0,
        gain_rx=1.0,
        wavelength=ref.wavelength_m(sc.CARRIER_HZ),
        rcs_m2=1.0,
        r_in=RANGE_M,
        r_out=RANGE_M,
    )
    expected = gain * ref.coherent_integration_snr(
        received_power_w=received,
        observed_time_s=observed_time_s,
        system_temperature_k=ref.system_noise_temperature_k(antenna_temperature_k=290.0, noise_figure_db=0.0),
    )
    return measured, expected


@pytest.mark.parametrize("name", list(_cases()))
def test_post_integration_snr_is_the_radar_equation_over_ktb(name):
    measured, expected = measured_and_expected_snr(name)
    assert abs(ref.db(measured / expected)) <= TOLERANCE_DB, (
        f"{name}: measured {ref.db(measured):.2f} dB, expected {ref.db(expected):.2f} dB"
    )
