"""End-to-end SNR oracle, one per waveform: radar equation, kTB, processing gain.

The noise contract is one rule for all three waveforms: every published sample
is a PHYSICAL amplitude in sqrt(W) at the receive port, ``|x|^2`` is the power
the antenna delivers while the echo is present, and the thermal stage adds
``CN(0, k T_sys f_s)`` per complex sample, the noise bandwidth of a complex
sample stream at rate ``f_s``. Under that contract the coherent SNR of any
matched statistic over ``K`` echo-bearing samples is

    SNR = P_r * (K / f_s) / (k T_sys)

with ``P_r = P_t lambda^2 sigma / ((4 pi)^3 R^4)``: received power times the
coherent observation time over the noise density. ``K / f_s`` is ``N T_s``
per chirp for FMCW, ``T_u`` per symbol for OFDM and ``T_p`` per pulse for a
pulsed radar.

Everything on the oracle side is float64 host arithmetic written here: the
radar equation, ``k T_sys``, and the coherent observation time. The simulator
contributes two cubes, one ideal and one with thermal noise at the same seed,
so the noise realisation is their difference exactly (the chain is additive and
linear without an AGC or an ADC).
"""

from __future__ import annotations

import math

import pytest
import torch

pytest.importorskip("witwin.channel")

from witwin.radar import Fmcw, Noise, Ofdm, PointTargets, Pulsed, Radar

pytestmark = pytest.mark.gpu

BOLTZMANN = 1.380649e-23
T0 = 290.0
CARRIER_HZ = 77e9
POWER_DBM = 10.0
NOISE_FIGURE_DB = 6.0
RCS_M2 = 1.0
IMPEDANCE = 50.0
TOLERANCE_DB = 0.5


def _waveforms():
    return {
        "fmcw": (
            Fmcw(
                slope=30e12,
                sample_rate=10e6,
                samples_per_chirp=256,
                chirps_per_frame=64,
                adc_start=4e-6,
                idle=10e-6,
                ramp_end=40e-6,
                output="beat",
            ),
            6.0,
        ),
        "ofdm": (
            Ofdm(
                subcarrier_spacing=1e6,
                num_subcarriers=256,
                cyclic_prefix=0.3e-6,
                num_symbols=64,
                max_expected_delay=0.25e-6,
            ),
            30.0,
        ),
        "pulsed": (
            Pulsed(
                pulse_kind="lfm",
                pulse_width=10e-6,
                bandwidth=20e6,
                pri=40e-6,
                num_pulses=32,
                sample_rate=50e6,
                num_samples=1000,
                range_gate_start=0.0,
            ),
            150.0,
        ),
    }


def _radar(waveform, *, noise):
    return Radar(
        reference_frequency_hz=CARRIER_HZ,
        waveform=waveform,
        tx=((0.0, 0.0, 0.0),),
        rx=((0.0, 0.0, 0.0),),
        power=POWER_DBM,
        noise=noise,
        impedance=IMPEDANCE,
        seed=7,
        position=(0.0, 0.0, 0.0),
        look_at=(1.0, 0.0, 0.0),
        up=(0.0, 1.0, 0.0),
    )


def _cube(radar, range_m):
    from witwin.core import Scene

    targets = PointTargets(positions=torch.tensor([[range_m, 0.0, 0.0]], device=radar.device), rcs=RCS_M2)
    result = radar.simulate(Scene(structures=(), endpoints=[]), targets, times=(0.0,), los=True, reflections=0)
    return result.cube.to(torch.complex128)


def measure_snr(kind: str) -> dict[str, float]:
    """Measured and predicted coherent SNR for one waveform, in dB."""

    waveform, range_m = _waveforms()[kind]
    noise = Noise(figure=NOISE_FIGURE_DB)
    clean = _cube(_radar(waveform, noise=None), range_m)
    noisy = _cube(_radar(waveform, noise=noise), range_m) / math.sqrt(IMPEDANCE)
    thermal = noisy - clean

    wavelength = 299_792_458.0 / CARRIER_HZ
    received_w = 1e-3 * 10 ** (POWER_DBM / 10) * wavelength**2 * RCS_M2 / ((4 * math.pi) ** 3 * range_m**4)
    t_sys = T0 + T0 * (10 ** (NOISE_FIGURE_DB / 10) - 1.0)
    sample_rate = float(waveform.sample_rate)

    magnitude = clean.abs()
    support = magnitude > 1e-3 * float(magnitude.max())
    count = int(support.sum())
    # The matched statistic: phase-aligned coherent sum over the echo-bearing
    # samples. Its signal is sum |s| and its noise variance is K * var(n).
    signal = float(magnitude[support].sum()) ** 2
    noise_variance = float((thermal.abs() ** 2).mean())
    measured = signal / (count * noise_variance)
    predicted = received_w * (count / sample_rate) / (BOLTZMANN * t_sys)
    return {
        "received_db": 10 * math.log10(float((magnitude[support] ** 2).mean()) / received_w),
        "noise_db": 10 * math.log10(noise_variance / (BOLTZMANN * t_sys * sample_rate)),
        "measured_db": 10 * math.log10(measured),
        "predicted_db": 10 * math.log10(predicted),
    }


@pytest.mark.parametrize("kind", ["fmcw", "ofdm", "pulsed"])
def test_coherent_snr_matches_the_radar_equation_through_ktb(kind):
    """M1-2 / FE-1: the processed SNR is the radar equation over k T_sys, within 0.5 dB."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    row = measure_snr(kind)
    # The echo power is the radar equation's, and the per-sample noise is
    # k T_sys f_s; each is checked on its own so a compensating pair of errors
    # cannot pass as one correct SNR.
    assert abs(row["received_db"]) < TOLERANCE_DB, row
    assert abs(row["noise_db"]) < TOLERANCE_DB, row
    assert abs(row["measured_db"] - row["predicted_db"]) < TOLERANCE_DB, row


if __name__ == "__main__":  # pragma: no cover - the before/after table
    for name in ("fmcw", "ofdm", "pulsed"):
        values = measure_snr(name)
        print(name, {key: round(value, 2) for key, value in values.items()})
