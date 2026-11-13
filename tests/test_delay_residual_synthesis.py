"""The synthesis kernels read the join's float64 delay pair (R-ADR-040).

The join publishes the round-trip delay as ``total_delay_s`` (float32) plus a
non-differentiable float32 residual. Every waveform kernel forms its phase from
``tau_hi + tau_lo`` in float64, so a 4 GHz sweep at 100 m is not quantized to
the float32 ulp of the delay (5.7e-14 s there, 1.4 mrad of bandwidth phase).

One oracle per family - FMCW beat, FMCW range spectrum, fused adaptive FMCW,
OFDM CFR, OFDM time samples and the pulsed train - each against an independent
float64 statement of that family's equation at the float64 delay, each with a
float32 falsifier (the same rows without the residual miss the oracle), and
each with the empty-residual rule: an all-zero residual and no residual
produce the same bits.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from witwin.radar.synthesis.assembly import PULSE_KIND_LFM, FmcwSpec, OfdmSpec, PulsedSpec
from witwin.radar.synthesis.fmcw import synthesize_adaptive_fmcw, synthesize_fmcw_rows
from witwin.radar.synthesis.ofdm import synthesize_cfr_rows, synthesize_ofdm_time_rows
from witwin.radar.synthesis.pulsed import synthesize_echo_rows

pytestmark = pytest.mark.gpu

C0 = 299792458.0
FC = 77.0e9
BANDWIDTH_HZ = 4.0e9
#: 100 m and a little beyond, chosen so float32 drops 0.4 ulp of the delay.
RANGES_M = (100.0, 100.0037)
WEIGHTS = (0.7 - 0.2j, -0.3 + 0.55j)
#: The oracle's phase budget, and the float32 delay's phase error it must beat.
TOLERANCE = 1.0e-5
FLOAT32_ERROR = 1.0e-4


def _pairs():
    tau64 = np.asarray([2.0 * r / C0 for r in RANGES_M])
    hi = tau64.astype(np.float32)
    lo = (tau64 - hi.astype(np.float64)).astype(np.float32)
    exact = hi.astype(np.float64) + lo.astype(np.float64)
    # The float32 delay misses the float64 one by a sizeable fraction of its ulp.
    assert np.all(np.abs(lo) > 0.1 * np.spacing(hi).astype(np.float64))

    def cuda(values, dtype=torch.float32):
        return torch.tensor(values, dtype=dtype, device="cuda")

    return exact, cuda(hi), cuda(lo), cuda(list(WEIGHTS), torch.complex64)


def _relative_error(produced: torch.Tensor, reference: np.ndarray) -> float:
    reference = torch.as_tensor(reference, dtype=torch.complex128)
    produced = produced.detach().cpu().to(torch.complex128).reshape(reference.shape)
    return float((produced - reference).abs().max() / reference.abs().max())


def _check(run, reference: np.ndarray, hi: torch.Tensor, lo: torch.Tensor, *, zero_bits: bool = True) -> None:
    """The oracle, the float32 falsifier and the empty-residual rule for one family.

    ``zero_bits``: a zero residual reads ``tau_hi`` through the same float64
    arithmetic as no residual, so the two agree bit for bit.
    """

    assert _relative_error(run(lo), reference) < TOLERANCE
    assert _relative_error(run(None), reference) > FLOAT32_ERROR
    if zero_bits:
        assert torch.equal(run(None), run(torch.zeros_like(hi)))


def _fmcw_spec(domain: str) -> FmcwSpec:
    samples, period = 64, 0.5e-6
    return FmcwSpec(
        num_samples=samples,
        num_chirps=2,
        sample_period_s=period,
        chirp_period_s=40.0e-6,
        slope_hz_per_s=BANDWIDTH_HZ / (samples * period),
        t_start_s=2.0e-6,
        reference_frequency_hz=FC,
        carrier_hz=0.0,
        carrier_rate_hz=FC,
        num_tx=1,
        num_rx=1,
        output_domain=domain,
    )


def _beat_cycles(spec: FmcwSpec, tau: np.ndarray) -> np.ndarray:
    """``[paths, samples]``: slope tau (u - tau / 2), the weight owning the carrier at tau."""

    u = spec.t_start_s + np.arange(spec.num_samples) * spec.sample_period_s
    return spec.slope_hz_per_s * tau[:, None] * (u[None, :] - 0.5 * tau[:, None])


def test_the_fmcw_beat_reads_the_float64_delay():
    exact, hi, lo, weight = _pairs()
    spec = _fmcw_spec("beat")
    offsets = torch.tensor([0, 2], dtype=torch.int64, device="cuda")
    beat = np.asarray(WEIGHTS).conj()[:, None] * np.exp(2j * math.pi * _beat_cycles(spec, exact))
    reference = np.broadcast_to(beat.sum(0), (spec.num_chirps, 1, spec.num_samples))

    def run(residual):
        return synthesize_fmcw_rows(hi, None, weight.conj(), offsets, spec, total_delay_residual_s=residual)

    _check(run, reference, hi, lo)


def test_the_fmcw_range_spectrum_reads_the_float64_delay():
    exact, hi, lo, weight = _pairs()
    spec = _fmcw_spec("spectrum")
    offsets = torch.tensor([0, 2], dtype=torch.int64, device="cuda")
    n = spec.num_samples
    dft = np.exp(-2j * math.pi * np.outer(np.arange(n), np.arange(n)) / n) / n
    reference = np.broadcast_to(beat_reference(spec, exact) @ dft, (spec.num_chirps, 1, n))

    def run(residual):
        return synthesize_fmcw_rows(hi, None, weight.conj(), offsets, spec, total_delay_residual_s=residual)

    # A residual also moves the closed form's Dirichlet argument to float64,
    # whose float32 rounding (N ulp32 of the argument) is the size of the
    # float32 delay's own error; at the float32 delay that is the only change.
    _check(run, reference, hi, lo, zero_bits=False)
    at_hi = beat_reference(spec, hi.double().cpu().numpy()) @ dft
    assert _relative_error(run(torch.zeros_like(hi)), np.broadcast_to(at_hi, reference.shape)) < TOLERANCE


def beat_reference(spec: FmcwSpec, tau: np.ndarray) -> np.ndarray:
    """One chirp's summed beat ``[samples]`` at the delays ``tau``."""

    return (np.asarray(WEIGHTS).conj()[:, None] * np.exp(2j * math.pi * _beat_cycles(spec, tau))).sum(0)


def test_the_fused_adaptive_fmcw_reads_the_float64_delay():
    """Two identical probes per path; every ADC instant interpolates between them."""

    exact, hi, lo, weight = _pairs()
    spec = _fmcw_spec("beat")
    rows = len(RANGES_M)
    observations = spec.num_samples
    coefficient = torch.tensor(WEIGHTS, dtype=torch.complex128)
    clock = spec.t_start_s + np.arange(observations) * spec.sample_period_s

    def samples(residual):
        delay = hi.double() if residual is None else hi.double() + residual.double()
        probe = torch.stack((delay.cpu(), coefficient.real, coefficient.imag), dim=1)
        return torch.cat((probe, probe)).to("cuda").contiguous()

    metadata = (
        torch.ones(2 * rows, dtype=torch.int32, device="cuda"),
        torch.tensor([[0, rows]] * observations, dtype=torch.int64, device="cuda"),
        torch.tensor([[0.5, 0.5]] * observations, dtype=torch.float64, device="cuda"),
        torch.tensor([[0, rows]], dtype=torch.int64, device="cuda"),
        torch.zeros(observations, dtype=torch.int64, device="cuda"),
        torch.tensor(clock, dtype=torch.float64, device="cuda"),
    )
    cycles = spec.slope_hz_per_s * exact[:, None] * (clock[None, :] - 0.5 * exact[:, None])
    reference = (np.asarray(WEIGHTS).conj()[:, None] * np.exp(2j * math.pi * cycles)).sum(0)

    def run(residual):
        return synthesize_adaptive_fmcw(
            samples(residual), *metadata, spec, FC, 0, observations, precise_delay=residual is not None
        )

    _check(run, reference, hi, lo)


def _ofdm_spec(**overrides) -> OfdmSpec:
    fields = {
        "num_subcarriers": 64,
        "num_symbols": 1,
        "subcarrier_spacing_hz": BANDWIDTH_HZ / 64,
        "cyclic_prefix_s": 1.0e-6,
        "reference_frequency_hz": FC,
        "max_expected_delay_s": 0.9e-6,
        "carrier_hz": 0.0,
        "carrier_rate_hz": FC,
    }
    fields.update(overrides)
    return OfdmSpec(**fields)


def test_the_ofdm_cfr_reads_the_float64_delay():
    exact, hi, lo, weight = _pairs()
    spec = _ofdm_spec()
    offsets = torch.tensor([0, 2], dtype=torch.int64, device="cuda")
    n = np.arange(spec.num_subcarriers)
    response = np.asarray(WEIGHTS)[:, None] * np.exp(-2j * math.pi * spec.subcarrier_spacing_hz * np.outer(exact, n))
    reference = response.sum(0)[None, None, :]

    def run(residual):
        return synthesize_cfr_rows(hi, None, weight, offsets, spec, total_delay_residual_s=residual)

    _check(run, reference, hi, lo)


def test_the_ofdm_time_samples_read_the_float64_delay():
    """Every echo is inside the prefix, so sample ``i`` reads its own symbol at ``i T_s - tau``."""

    exact, hi, lo, weight = _pairs()
    spec = _ofdm_spec(modulation="qpsk", symbol_seed=5, output_domain="time")
    offsets = torch.tensor([0, 2], dtype=torch.int64, device="cuda")
    table = spec.symbol_table()[0]
    n = np.arange(spec.num_subcarriers)
    t = np.arange(spec.samples_per_symbol)[None, :] * spec.waveform_sample_period_s - exact[:, None]
    x = (table[None, None, :] * np.exp(2j * math.pi * spec.subcarrier_spacing_hz * t[:, :, None] * n)).sum(-1)
    reference = (np.asarray(WEIGHTS)[:, None] * x / math.sqrt(spec.num_subcarriers)).sum(0)[None, None, :]

    def run(residual):
        return synthesize_ofdm_time_rows(hi, None, weight, offsets, spec, total_delay_residual_s=residual)

    _check(run, reference, hi, lo)


def test_the_pulsed_train_reads_the_float64_delay():
    """A 4 GHz LFM pulse: the envelope phase B u^2 / (2 T_p) moves with the delay at up to B."""

    exact, hi, lo, weight = _pairs()
    width = 1.0e-6
    spec = PulsedSpec(
        num_pulses=1,
        num_samples=96,
        sample_period_s=10.0e-9,
        pri_s=10.0e-6,
        range_gate_start_s=0.6e-6,
        pulse_kind=PULSE_KIND_LFM,
        pulse_width_s=width,
        bandwidth_hz=BANDWIDTH_HZ,
        reference_frequency_hz=FC,
        max_expected_delay_rate=0.0,
        carrier_hz=0.0,
        carrier_rate_hz=FC,
    )
    offsets = torch.tensor([0, 2], dtype=torch.int64, device="cuda")
    u = spec.range_gate_start_s + np.arange(spec.num_samples)[None, :] * spec.sample_period_s - exact[:, None]
    inside = (u >= 0.0) & (u < width)
    # Every sample sits well inside or outside the support at both delays.
    assert np.all((np.abs(u) > 1e-9) & (np.abs(u - width) > 1e-9))
    envelope = np.where(inside, np.exp(1j * math.pi * BANDWIDTH_HZ * u**2 / width), 0.0)
    reference = (np.asarray(WEIGHTS)[:, None] * envelope).sum(0)[None, None, :]

    def run(residual):
        return synthesize_echo_rows(hi, None, weight, offsets, spec, total_delay_residual_s=residual)

    _check(run, reference, hi, lo)
