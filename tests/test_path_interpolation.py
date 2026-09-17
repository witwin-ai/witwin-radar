"""Independent complex-arithmetic oracle for native carrier interpolation."""

import math

import pytest
import torch

from witwin.radar.paths import interpolate_path_rows

pytestmark = pytest.mark.gpu


def test_interpolation_preserves_carrier_wraps_and_native_derivatives():
    fc = 77e9
    d0 = torch.tensor([20e-9, 30e-9], device="cuda", requires_grad=True)
    d1 = torch.tensor([21e-9, 29e-9], device="cuda", requires_grad=True)
    w0 = torch.tensor([1 + 2j, 0.3 - 0.7j], device="cuda", requires_grad=True)
    w1 = torch.tensor([2 - 1j, 0.4 + 0.2j], device="cuda", requires_grad=True)
    alpha = torch.tensor([0.17, 0.71], device="cuda", dtype=torch.float64)

    def oracle(a, b, x, y):
        delta = b.double() - a.double()
        return ((1 - alpha) * a.double() + alpha * b.double()).float(), (
            (1 - alpha) * x.cdouble() * torch.exp(-2j * math.pi * fc * alpha * delta)
            + alpha * y.cdouble() * torch.exp(2j * math.pi * fc * (1 - alpha) * delta)
        ).cfloat()

    leaves = (d0, d1, w0, w1)
    actual = interpolate_path_rows(*leaves, alpha, fc)
    expected = oracle(*leaves)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=2e-6, atol=1e-7)

    def loss(values):
        return values[0].sum() * 1e8 + values[1].real.sum() + 0.3 * values[1].imag.sum()

    actual_grad = torch.autograd.grad(loss(actual), leaves)
    expected_grad = torch.autograd.grad(loss(expected), leaves)
    for a, b in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(a, b, rtol=3e-6, atol=1e-6)
    tangents = (
        torch.full_like(d0, 1e-12),
        torch.full_like(d1, -2e-12),
        torch.full_like(w0, 0.1j),
        torch.full_like(w1, 0.2),
    )
    with torch.autograd.forward_ad.dual_level():
        duals = [torch.autograd.forward_ad.make_dual(a.detach(), t) for a, t in zip(leaves, tangents, strict=True)]
        result = interpolate_path_rows(*duals, alpha, fc)
        reference = oracle(*duals)
        for a, b in zip(result, reference, strict=True):
            torch.testing.assert_close(
                torch.autograd.forward_ad.unpack_dual(a).tangent,
                torch.autograd.forward_ad.unpack_dual(b).tangent,
                rtol=3e-6,
                atol=1e-6,
            )


def test_single_physical_path_has_no_interpolation_fade():
    fc = 77e9
    d0 = torch.tensor([20e-9], device="cuda")
    d1 = torch.tensor([22e-9], device="cuda")
    w0 = torch.exp(-2j * math.pi * fc * d0.double()).cfloat()
    w1 = torch.exp(-2j * math.pi * fc * d1.double()).cfloat()
    alpha = torch.tensor([0.37], device="cuda", dtype=torch.float64)
    _, weight = interpolate_path_rows(d0, d1, w0, w1, alpha, fc)
    expected = torch.exp(-2j * math.pi * fc * ((1 - alpha) * d0.double() + alpha * d1.double())).cfloat()
    torch.testing.assert_close(weight, expected, rtol=1e-6, atol=1e-6)
