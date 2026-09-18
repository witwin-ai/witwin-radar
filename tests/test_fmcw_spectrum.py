"""Direct FMCW Dirichlet spectrum is the default and equals FFT(beat).

The equivalence is driven with a delay that does not walk, because that is
the delay the closed form exists for. A walking delay reaches its spectrum
through the beat family plus the range transform instead, which is the same
N log N route this file's FFT stands in for, and the spectrum family refuses
a rate by name rather than ignoring it."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
import torch.autograd.forward_ad as forward_ad

from witwin.radar.synthesis import FmcwSpec
from witwin.radar.synthesis.fmcw import synthesize_fmcw_rows


def _spec() -> FmcwSpec:
    return FmcwSpec(
        num_samples=32,
        num_chirps=3,
        sample_period_s=1.0 / 4.4e6,
        chirp_period_s=65.0e-6,
        slope_hz_per_s=60.012e12,
        t_start_s=6.0e-6,
        reference_frequency_hz=77.0e9,
        carrier_hz=0.0,
        carrier_rate_hz=77.0e9,
        num_tx=2,
        num_rx=1,
    )


def _inputs(*, requires_grad: bool = False):
    tau = torch.tensor([1.7e-8, 2.4e-8, 3.1e-8], device="cuda")
    rate = torch.tensor([1.2e-9, -0.7e-9, 0.4e-9], device="cuda")
    weight = torch.tensor([0.6 - 0.3j, -0.2 + 0.45j, 0.15 + 0.8j], dtype=torch.complex64, device="cuda")
    if requires_grad:
        tau.requires_grad_(True)
        rate.requires_grad_(True)
        weight.requires_grad_(True)
    offsets = torch.tensor([0, 2, 3], dtype=torch.int64, device="cuda")
    tx = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    return tau, rate, weight, offsets, tx


def _run(spec, values, *, walking: bool = False):
    tau, rate, weight, offsets, tx = values
    return synthesize_fmcw_rows(tau, rate if walking else None, weight, offsets, spec, segment_tx_index=tx)


def test_spectrum_is_the_default_domain():
    assert _spec().output_domain == "spectrum"


@pytest.mark.gpu
def test_direct_spectrum_equals_normalized_fft_of_explicit_beat():
    values = _inputs()
    direct = _run(_spec(), values)
    beat = _run(replace(_spec(), output_domain="beat"), values)
    expected = torch.fft.fft(beat, dim=-1, norm="forward")
    torch.testing.assert_close(direct, expected, rtol=3e-4, atol=3e-5)


@pytest.mark.gpu
def test_the_spectrum_refuses_a_walking_delay_and_names_the_cheaper_route():
    """The refusal is the whole reason the rate left this family's signature.

    Silently dropping the rate would return a stationary spectrum for a moving
    target, which looks like a plausible cube. Silently summing term by term
    would cost N per bin where the beat route costs N log N for the whole axis.
    So the caller is told which route to take.
    """

    values = _inputs()
    with pytest.raises(ValueError, match=r'output_domain="beat"'):
        _run(_spec(), values, walking=True)


@pytest.mark.gpu
def test_spectrum_vjp_equals_fft_of_beat_vjp():
    torch.manual_seed(20260728)
    cotangent = torch.randn((3, 2, 32), device="cuda", dtype=torch.complex64)
    direct_values = _inputs(requires_grad=True)
    direct = _run(_spec(), direct_values)
    direct_loss = torch.real((direct.conj() * cotangent).sum())
    # The rate is not an input of the spectrum family, so only tau and the weight
    # carry a cotangent here; the beat side is compared on the same two.
    direct_grads = torch.autograd.grad(direct_loss, (direct_values[0], direct_values[2]))

    beat_values = _inputs(requires_grad=True)
    beat = _run(replace(_spec(), output_domain="beat"), beat_values)
    transformed = torch.fft.fft(beat, dim=-1, norm="forward")
    beat_loss = torch.real((transformed.conj() * cotangent).sum())
    beat_grads = torch.autograd.grad(beat_loss, (beat_values[0], beat_values[2]))
    for measured, expected in zip(direct_grads, beat_grads, strict=True):
        torch.testing.assert_close(measured, expected, rtol=8e-4, atol=8e-4)


@pytest.mark.gpu
def test_spectrum_jvp_equals_fft_of_beat_jvp():
    values = _inputs()
    tangents = (
        torch.tensor([0.4e-10, -0.3e-10, 0.2e-10], device="cuda"),
        torch.tensor([0.1 + 0.2j, -0.3 + 0.05j, 0.2 - 0.1j], dtype=torch.complex64, device="cuda"),
    )

    def tangent(spec):
        with forward_ad.dual_level():
            tau = forward_ad.make_dual(values[0], tangents[0])
            weight = forward_ad.make_dual(values[2], tangents[1])
            output = _run(spec, (tau, values[1], weight, *values[3:]))
            return forward_ad.unpack_dual(output).tangent

    direct = tangent(_spec())
    beat = tangent(replace(_spec(), output_domain="beat"))
    torch.testing.assert_close(direct, torch.fft.fft(beat, dim=-1, norm="forward"), rtol=8e-4, atol=8e-4)
