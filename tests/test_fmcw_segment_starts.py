"""Per-segment ADC window starts on the native FMCW beat family.

``synthesize_fmcw_rows(..., segment_t_start_s=...)`` gives every sensor-pair
segment its own window start, which is what lets a refreshed ADC frame be one
launch. Two oracles:

* separate launches, one per segment, each with that start as the uniform
  ``t_start_s``: the forward value, the VJP and the JVP must agree BIT FOR BIT,
  because the per-segment start replaces exactly one scalar of the phase
  expression and the path order of each segment's sum is unchanged. The uniform
  family's own AD is checked against finite differences in
  ``tests/test_fmcw_beat_ad.py``;
* an independent float64 evaluation of the linear-delay beat model.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
import torch.autograd.forward_ad as forward_ad

from witwin.radar.synthesis.assembly import FmcwSpec
from witwin.radar.synthesis.fmcw import synthesize_fmcw_rows

pytestmark = pytest.mark.gpu

F_REF_HZ = 77.0e9

SPEC = FmcwSpec(
    num_samples=8,
    num_chirps=2,
    sample_period_s=1.0 / 4.4e6,
    chirp_period_s=65.0e-6,
    slope_hz_per_s=60.012e12,
    t_start_s=6.0e-6,
    reference_frequency_hz=F_REF_HZ,
    carrier_hz=0.0,
    carrier_rate_hz=F_REF_HZ,
    output_domain="beat",
)

DELAYS = (2.1e-8, 2.4e-8, 3.3e-8, 1.7e-8, 4.0e-8)
RATES = (2.385e-8, -1.1e-8, 0.5e-8, 0.0, -3.0e-8)
WEIGHTS = (0.6 - 0.3j, -0.2 + 0.45j, 0.1 + 0.9j, -0.7 - 0.1j, 0.33 + 0.2j)
OFFSETS = (0, 2, 2, 5)
STARTS = (6.0e-6, 6.0e-6 + 3 * SPEC.sample_period_s, 6.0e-6 + 7 * SPEC.sample_period_s)


def _inputs(*, requires_grad: bool = False):
    tau = torch.tensor(DELAYS, dtype=torch.float32, device="cuda", requires_grad=requires_grad)
    rate = torch.tensor(RATES, dtype=torch.float32, device="cuda", requires_grad=requires_grad)
    weight = torch.tensor(WEIGHTS, dtype=torch.complex64, device="cuda", requires_grad=requires_grad)
    return tau, rate, weight


def _offsets() -> torch.Tensor:
    return torch.tensor(OFFSETS, dtype=torch.int64, device="cuda")


def _starts() -> torch.Tensor:
    return torch.tensor(STARTS, dtype=torch.float64, device="cuda")


def _per_segment(tau, rate, weight):
    """One uniform-start launch per segment, stacked on the segment axis."""

    cubes = []
    for segment, start in enumerate(STARTS):
        begin, end = OFFSETS[segment], OFFSETS[segment + 1]
        offsets = torch.tensor((0, end - begin), dtype=torch.int64, device="cuda")
        cubes.append(
            synthesize_fmcw_rows(
                tau[begin:end], rate[begin:end], weight[begin:end], offsets, replace(SPEC, t_start_s=start)
            )
        )
    return torch.cat(cubes, dim=1)


def test_one_launch_equals_one_launch_per_start_bit_for_bit():
    tau, rate, weight = _inputs()
    timed = synthesize_fmcw_rows(tau, rate, weight, _offsets(), SPEC, segment_t_start_s=_starts())
    assert torch.equal(timed, _per_segment(tau, rate, weight))


def test_the_reverse_gradient_equals_the_per_start_launches_bit_for_bit():
    torch.manual_seed(20261112)
    cotangent = torch.randn((SPEC.num_chirps, len(STARTS), SPEC.num_samples), dtype=torch.complex64, device="cuda")

    def gradients(build):
        tau, rate, weight = _inputs(requires_grad=True)
        cube = build(tau, rate, weight)
        (cube * cotangent.conj()).real.sum().backward()
        return tau.grad, rate.grad, weight.grad

    timed = gradients(
        lambda tau, rate, weight: synthesize_fmcw_rows(tau, rate, weight, _offsets(), SPEC, segment_t_start_s=_starts())
    )
    separate = gradients(_per_segment)
    for got, expected in zip(timed, separate, strict=True):
        assert torch.equal(got, expected)


def test_the_forward_tangent_equals_the_per_start_launches_bit_for_bit():
    torch.manual_seed(20261113)
    tangents = (
        torch.randn(len(DELAYS), device="cuda") * 1e-10,
        torch.randn(len(DELAYS), device="cuda") * 1e-9,
        torch.randn(len(DELAYS), dtype=torch.complex64, device="cuda"),
    )

    def tangent(build):
        with forward_ad.dual_level():
            duals = [
                forward_ad.make_dual(value, direction) for value, direction in zip(_inputs(), tangents, strict=True)
            ]
            return forward_ad.unpack_dual(build(*duals)).tangent

    timed = tangent(
        lambda tau, rate, weight: synthesize_fmcw_rows(tau, rate, weight, _offsets(), SPEC, segment_t_start_s=_starts())
    )
    assert torch.equal(timed, tangent(_per_segment))


def test_each_segment_follows_the_float64_linear_delay_model():
    """``tau = tau0 + rate * (t_slot + u)`` and ``u = start + m * Ts``, independently."""

    tau, rate, weight = _inputs()
    timed = synthesize_fmcw_rows(tau, rate, weight, _offsets(), SPEC, segment_t_start_s=_starts()).cpu().numpy()
    delays, rates, weights = np.asarray(DELAYS), np.asarray(RATES), np.asarray(WEIGHTS)
    for segment, start in enumerate(STARTS):
        rows = slice(OFFSETS[segment], OFFSETS[segment + 1])
        for chirp in range(SPEC.num_chirps):
            u = start + np.arange(SPEC.num_samples) * SPEC.sample_period_s
            slot = chirp * SPEC.chirp_period_s
            drift = rates[rows, None] * (slot + u[None, :])
            delay = delays[rows, None] + drift
            cycles = SPEC.carrier_rate_hz * drift + SPEC.slope_hz_per_s * delay * (u[None, :] - 0.5 * delay)
            expected = (weights[rows, None] * np.exp(2j * np.pi * cycles)).sum(axis=0)
            np.testing.assert_allclose(timed[chirp, segment], expected, rtol=0, atol=2e-5)


def test_segment_starts_are_refused_where_they_mean_nothing():
    tau, rate, weight = _inputs()
    with pytest.raises(ValueError, match="beat domain"):
        synthesize_fmcw_rows(
            tau, None, weight, _offsets(), replace(SPEC, output_domain="spectrum"), segment_t_start_s=_starts()
        )
    with pytest.raises(ValueError, match="segment_t_start_s must be float64"):
        synthesize_fmcw_rows(tau, rate, weight, _offsets(), SPEC, segment_t_start_s=_starts()[:2])
    with pytest.raises(RuntimeError, match="segment window starts"):
        synthesize_fmcw_rows(tau, rate, weight, _offsets(), SPEC, segment_t_start_s=_starts().requires_grad_())
