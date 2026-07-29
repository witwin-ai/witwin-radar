"""Direct FMCW Dirichlet spectrum is the default and equals FFT(beat)."""

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
    weight = torch.tensor(
        [0.6 - 0.3j, -0.2 + 0.45j, 0.15 + 0.8j],
        dtype=torch.complex64,
        device="cuda",
    )
    if requires_grad:
        tau.requires_grad_(True)
        rate.requires_grad_(True)
        weight.requires_grad_(True)
    offsets = torch.tensor([0, 2, 3], dtype=torch.int64, device="cuda")
    tx = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    return tau, rate, weight, offsets, tx


def _run(spec, values):
    tau, rate, weight, offsets, tx = values
    return synthesize_fmcw_rows(
        tau, rate, weight, offsets, spec, segment_tx_index=tx
    )


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
def test_spectrum_vjp_equals_fft_of_beat_vjp():
    torch.manual_seed(20260728)
    cotangent = torch.randn((3, 2, 32), device="cuda", dtype=torch.complex64)
    direct_values = _inputs(requires_grad=True)
    direct = _run(_spec(), direct_values)
    direct_loss = torch.real((direct.conj() * cotangent).sum())
    direct_grads = torch.autograd.grad(direct_loss, direct_values[:3])

    beat_values = _inputs(requires_grad=True)
    beat = _run(replace(_spec(), output_domain="beat"), beat_values)
    transformed = torch.fft.fft(beat, dim=-1, norm="forward")
    beat_loss = torch.real((transformed.conj() * cotangent).sum())
    beat_grads = torch.autograd.grad(beat_loss, beat_values[:3])
    for measured, expected in zip(direct_grads, beat_grads, strict=True):
        torch.testing.assert_close(measured, expected, rtol=8e-4, atol=8e-4)


@pytest.mark.gpu
def test_spectrum_jvp_equals_fft_of_beat_jvp():
    values = _inputs()
    tangents = (
        torch.tensor([0.4e-10, -0.3e-10, 0.2e-10], device="cuda"),
        torch.tensor([0.7e-10, 0.2e-10, -0.5e-10], device="cuda"),
        torch.tensor(
            [0.1 + 0.2j, -0.3 + 0.05j, 0.2 - 0.1j],
            dtype=torch.complex64,
            device="cuda",
        ),
    )

    def tangent(spec):
        with forward_ad.dual_level():
            duals = tuple(
                forward_ad.make_dual(primal, tan)
                for primal, tan in zip(values[:3], tangents, strict=True)
            )
            output = _run(spec, (*duals, *values[3:]))
            return forward_ad.unpack_dual(output).tangent

    direct = tangent(_spec())
    beat = tangent(replace(_spec(), output_domain="beat"))
    torch.testing.assert_close(
        direct, torch.fft.fft(beat, dim=-1, norm="forward"),
        rtol=8e-4, atol=8e-4,
    )