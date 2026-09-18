"""Independent arbitrary-time FMCW CSR oracle, including first-order AD."""

import math

import pytest
import torch
import torch.autograd.forward_ad as ad

from witwin.radar.synthesis.assembly import FmcwSpec
from witwin.radar.synthesis.fmcw import synthesize_fmcw_observations

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("carrier", [0.0, 77e9])
def test_ragged_observations_match_complex_oracle_and_ad(carrier):
    spec = FmcwSpec(1, 1, 1e-6, 1e-3, 12e12, 0, 77e9, carrier_hz=carrier, output_domain="beat")
    delay = torch.tensor([23e-9, 31e-9, 90e-9, 150e-9], device="cuda", requires_grad=True)
    weight = torch.tensor([1 + 0.2j, 0.3 - 0.2j, -0.1j, 0.2 + 0.3j], device="cuda", requires_grad=True)
    clock = torch.tensor([1e-6, 1e-6, 7e-6, 14e-6], device="cuda", dtype=torch.float64)
    offsets = torch.tensor([0, 2, 2, 3, 4], device="cuda")
    actual = synthesize_fmcw_observations(delay, weight, offsets, clock, spec)
    tau = delay.double()
    phase = 2 * math.pi * (carrier * tau + spec.slope_hz_per_s * tau * (clock - tau / 2))
    rows = weight * torch.exp(1j * phase)
    expected = torch.stack([rows[:2].sum(), rows[:0].sum(), rows[2], rows[3]])
    torch.testing.assert_close(actual, expected.to(torch.complex64), rtol=2e-6, atol=2e-6)
    cotangent = torch.tensor([0.2 + 0.4j, 1j, 0.7, -0.6j], device="cuda")
    gradient = torch.autograd.grad(actual, (delay, weight), cotangent)
    reference = torch.autograd.grad(expected, (delay, weight), cotangent.to(torch.complex128))
    for value, oracle in zip(gradient, reference, strict=True):
        torch.testing.assert_close(value, oracle, rtol=3e-6, atol=1e-6)
    td, tw = torch.ones_like(delay) * 1e-10, torch.ones_like(weight) * (0.3 - 0.2j)
    with ad.dual_level():
        _, tangent = ad.unpack_dual(
            synthesize_fmcw_observations(
                ad.make_dual(delay.detach(), td), ad.make_dual(weight.detach(), tw), offsets, clock, spec
            )
        )
    lhs = (tangent.conj() * cotangent).real.sum()
    rhs = (gradient[0] * td).sum() + (gradient[1].conj() * tw).real.sum()
    torch.testing.assert_close(lhs, rhs, rtol=3e-6, atol=1e-6)


def test_empty_observations_and_time_derivative_refusal():
    spec = FmcwSpec(1, 1, 1e-6, 1e-3, 1e12, 0, 77e9, carrier_rate_hz=77e9, output_domain="beat")
    delay = torch.empty(0, device="cuda")
    weight = torch.empty(0, device="cuda", dtype=torch.complex64)
    offsets = torch.zeros(4, device="cuda", dtype=torch.int64)
    result = synthesize_fmcw_observations(delay, weight, offsets, delay, spec)
    assert torch.equal(result, torch.zeros_like(result))
    with pytest.raises(RuntimeError, match="ADC times"):
        synthesize_fmcw_observations(delay, weight, offsets, delay.clone().requires_grad_(), spec)
