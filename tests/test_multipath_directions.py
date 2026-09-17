"""Scattering departure bearings and derivatives, including outbound reflection."""

import pytest
import torch
from support import multi_endpoint_driver as drv
from test_phase9_aspect_direction_ad import _response

from witwin.radar.scattering import _ScatterDirection

pytestmark = pytest.mark.gpu


def test_native_scatter_direction_primal_vjp_and_jvp():
    a = torch.tensor([[0.2, -0.3, 0.1], [1.0, 0.5, -0.2]], device="cuda", requires_grad=True)
    b = torch.tensor([[2.0, 0.7, 0.2], [1.4, -0.2, 0.8]], device="cuda", requires_grad=True)
    v = torch.tensor([[0.3, 0.2, -0.4], [-0.1, 0.5, 0.2]], device="cuda")
    u = _ScatterDirection.apply(a, b)
    oracle = torch.nn.functional.normalize(b - a, dim=-1)
    torch.testing.assert_close(u, oracle, atol=1e-7, rtol=1e-6)
    expected = torch.autograd.grad((oracle * v).sum(), (a, b))
    actual = torch.autograd.grad((u * v).sum(), (a, b))
    for x, y in zip(actual, expected, strict=True):
        torch.testing.assert_close(x, y, atol=1e-7, rtol=1e-6)
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(b.detach(), v)
        tangent = torch.autograd.forward_ad.unpack_dual(_ScatterDirection.apply(a.detach(), dual)).tangent
    torch.testing.assert_close(tangent, expected[1], atol=1e-7, rtol=1e-6)


def test_outbound_reflection_aspect_uses_departure_and_carries_gradients():
    spike = drv.MultiEndpointSpike()
    response = _response()
    positions = spike.site_tensor(requires_grad=True)
    inbound, outbound = spike.legs(positions, ad_mode="vjp")
    composer = spike.composer
    valid = torch.ones(composer.path_count, dtype=torch.int32, device="cuda")
    real, imaginary = response.evaluate_rows(composer, inbound, outbound, valid)
    # Analytic mirrored receiver across x=4: the site must look towards the
    # reflection, not along the reflected segment arriving at the receiver.
    sinks = spike.receiver_tensor()[outbound.sink_index.long()]
    mirrored = sinks.clone()
    mirrored[:, 0] = 8.0 - sinks[:, 0]
    targets = torch.where((outbound.depth > 0)[:, None], mirrored, sinks)
    directions = torch.nn.functional.normalize(targets - positions[outbound.source_index.long()], dim=-1)
    axis = response.axis[composer.response_slot]
    ci = (-(inbound.field_direction[composer.inbound_row] * axis).sum(-1)).clamp_min(0)
    co = ((directions[composer.outbound_row] * axis).sum(-1)).clamp_min(0)
    oracle = response.amplitude[composer.response_slot] * ci**response.exponent * co**response.exponent
    torch.testing.assert_close(torch.complex(real, imaginary).abs(), oracle, atol=1e-3, rtol=2e-6)
    assert bool((outbound.depth > 0).any())
    gradient = torch.autograd.grad((real + 0.7 * imaginary).sum(), positions)[0]
    direction = torch.tensor([[0.1, 0.3, -0.2], [-0.2, 0.1, 0.3]], device="cuda")

    def loss(points):
        leg_in, leg_out = spike.legs(points, ad_mode="none")
        re, im = response.evaluate_rows(composer, leg_in, leg_out, valid)
        return (re + 0.7 * im).double().sum()

    step = 1e-3
    finite_difference = (loss(positions.detach() + step * direction) - loss(positions.detach() - step * direction)) / (
        2 * step
    )
    torch.testing.assert_close((gradient * direction).double().sum(), finite_difference, atol=3.0, rtol=3e-3)
