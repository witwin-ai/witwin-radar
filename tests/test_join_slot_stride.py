"""The native join's slot stride: one join's tables across a slot-major replay.

``TwoWayComposer.compose_slots`` hands the native join ONE slot's routing and
CSR tables plus a slot count, and the kernels stride them across the batch.
The oracle is the replicated route, ``for_slots(...).compose(...)``, whose
tables are copied per slot: the forward value, the VJP into every leg and the
response, and the JVP must agree BIT FOR BIT, because the stride reproduces
exactly the indices and the CSR summation order the copies hold.
"""

from __future__ import annotations

import pytest
import torch
import torch.autograd.forward_ad as forward_ad

pytest.importorskip("witwin.channel")

from support import multi_endpoint_driver as drv

pytestmark = pytest.mark.gpu

SLOTS = 3


@pytest.fixture(scope="module")
def spike():
    return drv.MultiEndpointSpike()


def _sites(spike, *, requires_grad: bool = False) -> torch.Tensor:
    base = spike.site_tensor()
    steps = torch.arange(SLOTS, dtype=torch.float32, device=base.device)[:, None, None] * 1.0e-3
    stack = (base[None] + steps).reshape(-1, 3).contiguous()
    return stack.requires_grad_(requires_grad)


def _both(spike, sites, *, ad_mode: str = "none"):
    inbound, outbound = spike.slot_legs(sites, slot_count=SLOTS, ad_mode=ad_mode)
    response = drv.make_response()
    strided = spike.composer.compose_slots(inbound, outbound, response, SLOTS)
    copied = spike.composer.for_slots(SLOTS).compose(inbound, outbound, response)
    return strided, copied


def test_the_strided_join_equals_the_replicated_join_bit_for_bit(spike):
    strided, copied = _both(spike, _sites(spike))
    assert strided.rows_per_slot * SLOTS == copied.path_count
    assert torch.equal(strided.total_delay_s, copied.total_delay_s)
    assert torch.equal(strided.complex_transfer_ref, copied.complex_transfer_ref)
    assert torch.equal(strided.row_valid, copied.row_valid)


def test_the_strided_vjp_equals_the_replicated_vjp_bit_for_bit(spike):
    torch.manual_seed(20261114)

    def gradient(pick):
        sites = _sites(spike, requires_grad=True)
        inbound, outbound = spike.slot_legs(sites, slot_count=SLOTS, ad_mode="vjp")
        response = drv.make_response()
        if pick == "strided":
            rows = spike.composer.compose_slots(inbound, outbound, response, SLOTS)
        else:
            rows = spike.composer.for_slots(SLOTS).compose(inbound, outbound, response)
        weight = torch.linspace(0.5, 1.5, rows.total_delay_s.numel(), device=sites.device)
        loss = (rows.complex_transfer_ref.abs().square() * weight).sum() + (rows.total_delay_s * weight).sum() * 1e9
        loss.backward()
        return sites.grad

    assert torch.equal(gradient("strided"), gradient("copied"))


def test_the_strided_jvp_equals_the_replicated_jvp_bit_for_bit(spike):
    def tangent(pick):
        with forward_ad.dual_level():
            primal = _sites(spike)
            direction = torch.zeros_like(primal)
            direction[:, 0] = 1.0
            sites = forward_ad.make_dual(primal, direction)
            inbound, outbound = spike.slot_legs(sites, slot_count=SLOTS, ad_mode="jvp")
            response = drv.make_response()
            if pick == "strided":
                rows = spike.composer.compose_slots(inbound, outbound, response, SLOTS)
            else:
                rows = spike.composer.for_slots(SLOTS).compose(inbound, outbound, response)
            return (
                forward_ad.unpack_dual(rows.total_delay_s).tangent,
                forward_ad.unpack_dual(rows.complex_transfer_ref).tangent,
            )

    for got, expected in zip(tangent("strided"), tangent("copied"), strict=True):
        assert torch.equal(got, expected)


def test_a_mismatched_slot_count_is_refused_by_name(spike):
    inbound, outbound = spike.slot_legs(_sites(spike), slot_count=SLOTS)
    with pytest.raises(ValueError, match="slots of this join need"):
        spike.composer.compose_slots(inbound, outbound, drv.make_response(), SLOTS + 1)
