"""FE-1: post-integration SNR per waveform against the radar equation and kTB.

One co-located element, one isotropic ``1 m^2`` target at 15 m on boresight,
a receiver at ``T_sys = 290 K`` (0 dB noise figure, 290 K antenna). The
physical expectation is the matched-filter SNR

    SNR = E_r / N_0 = P_r T_obs / (k T_sys),
    P_r = P_t lambda^2 sigma / ((4 pi)^3 R^4),

with ``T_obs`` the time the echo is actually observed (FMCW: chirps times
samples over the sample rate; pulsed: pulses times pulse width, the pulse
lying wholly inside the range gate; OFDM: symbols times the useful symbol time
``1 / delta_f``, the cyclic prefix excluded).

The measured SNR is normalisation-free: run the radar twice with the same
seed, once with the noise source at 0 K (the same receive chain, no noise),
once at 290 K. The first cube is the signal ``s``, the difference is the noise
``n``, and the matched filter's output SNR on white noise is
``sum |s|^2 / E|n|^2``. Any unitary transform of the cube (the FMCW range
spectrum) leaves it unchanged, which is why both FMCW output domains are held
to the same number.
"""

from __future__ import annotations

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

_FE1 = (
    "FE-1: pulsed samples are C/sqrt(T_p) (unit-energy pulse) but thermal noise is kT*B_chirp on a cube treated as "
    "sqrt(W), so the pulsed SNR is optimistic by f_s/(B*T_p) (+54 dB at 20 MHz, 10 us, 50 MS/s)"
)


def _waveforms():
    from witwin.radar import Fmcw, Ofdm, Pulsed

    def fmcw(output):
        return Fmcw(
            slope=30e12,
            sample_rate=10e6,
            samples_per_chirp=256,
            chirps_per_frame=16,
            adc_start=0.0,
            idle=5e-6,
            ramp_end=26.6e-6,
            output=output,
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
    }


def _cases():
    for name in ("fmcw-beat", "fmcw-spectrum", "ofdm", "pulsed"):
        marks = [pytest.mark.xfail(strict=True, reason=_FE1)] if name == "pulsed" else []
        yield pytest.param(name, marks=marks, id=name)


def measured_and_expected_snr(name: str) -> tuple[float, float]:
    from witwin.radar import Noise, PointTargets

    waveform, observed_time_s = _waveforms()[name]
    cubes = {}
    for key, noise in (("signal", Noise(antenna_temperature=0.0)), ("noisy", Noise())):
        radar = sc.radar(waveform, power_dbm=POWER_DBM, noise=noise, seed=7)
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
    expected = ref.coherent_integration_snr(
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
