"""The antenna-pattern stage's slot stride: one stage's tables across a slot-major replay.

``RoundTripPatternStage.apply_slots`` hands the native sensor-weight operator
ONE slot's routing tables and a slot count, and the kernel strides them across
the batch (transmitter of batch row ``k`` = table row ``k % rows_per_slot``
offset by its slot). The oracle is the replicated route,
``pattern.for_slots(...).apply(...)`` after ``composer.for_slots(...)``, whose
tables are copied per slot: forward, VJP and JVP must agree BIT FOR BIT,
because the stride reproduces exactly the indices and the ascending-row
antenna reduction the copies hold.
"""

from __future__ import annotations

import pytest
import torch
import torch.autograd.forward_ad as forward_ad

pytest.importorskip("witwin.channel")

from support import multi_endpoint_driver as drv
from support import oracle_scenes as sc

pytestmark = pytest.mark.gpu

SLOTS = 3


@pytest.fixture(scope="module")
def spike():
    return drv.MultiEndpointSpike()


@pytest.fixture(scope="module")
def stage(spike):
    from witwin.radar import Pattern
    from witwin.radar.sensors import RoundTripPatternStage

    radar = sc.radar(tx=((0.0, 0.0, 0.0), (0.0, 0.002, 0.0)), rx=((0.0, 0.0, 0.0), (0.0, 0.004, 0.0)))
    return RoundTripPatternStage.freeze(
        radar, spike.composer, site_ids=spike.site_ids, pattern=Pattern.dipole(), rx_pattern=Pattern.isotropic(gain=3)
    )


def _sites(spike, *, requires_grad: bool = False) -> torch.Tensor:
    base = spike.site_tensor()
    steps = torch.arange(SLOTS, dtype=torch.float32, device=base.device)[:, None, None] * 1.0e-3
    return (base[None] + steps).reshape(-1, 3).contiguous().requires_grad_(requires_grad)


def _routes(spike):
    composer = spike.composer
    slot = torch.arange(SLOTS, device=composer.inbound_row.device)[:, None]
    return (
        (composer.inbound_row[None] + slot * composer.inbound_row_count).reshape(-1),
        (composer.outbound_row[None] + slot * composer.outbound_row_count).reshape(-1),
    )


def _weights(spike, stage, sites, pick: str, *, ad_mode: str = "none"):
    inbound, outbound = spike.slot_legs(sites, slot_count=SLOTS, ad_mode=ad_mode)
    tx = spike.stacked([position for _, position in spike.transmitters], SLOTS)
    rx = spike.stacked([position for _, position in spike.receivers], SLOTS)
    response = drv.make_response()
    if pick == "strided":
        rows = spike.composer.compose_slots(inbound, outbound, response, SLOTS)
        inbound_rows, outbound_rows = _routes(spike)
        weighted = stage.apply_slots(
            rows,
            tx_pos=tx,
            rx_pos=rx,
            tx_targets_m=inbound.departure_target_m.index_select(0, inbound_rows),
            rx_targets_m=outbound.departure_target_m.index_select(0, outbound_rows),
        )
    else:
        composer = spike.composer.for_slots(SLOTS)
        rows = composer.compose(inbound, outbound, response)
        weighted = stage.for_slots(SLOTS).apply(
            rows,
            tx_pos=tx,
            rx_pos=rx,
            tx_targets_m=inbound.departure_target_m.index_select(0, rows.topology.inbound_row),
            rx_targets_m=outbound.departure_target_m.index_select(0, rows.topology.outbound_row),
        )
    assert weighted.weight_includes_antenna_pattern
    return weighted.complex_transfer_ref


def test_the_strided_stage_equals_the_replicated_stage_bit_for_bit(spike, stage):
    strided = _weights(spike, stage, _sites(spike), "strided")
    copied = _weights(spike, stage, _sites(spike), "copied")
    assert float(strided.abs().max()) > 0.0
    assert torch.equal(strided, copied)


def test_the_strided_vjp_equals_the_replicated_vjp_bit_for_bit(spike, stage):
    def gradient(pick):
        sites = _sites(spike, requires_grad=True)
        weight = _weights(spike, stage, sites, pick, ad_mode="vjp")
        scale = torch.linspace(0.5, 1.5, weight.numel(), device=sites.device)
        (weight.abs().square() * scale).sum().backward()
        return sites.grad

    assert torch.equal(gradient("strided"), gradient("copied"))


def test_the_strided_jvp_equals_the_replicated_jvp_bit_for_bit(spike, stage):
    def tangent(pick):
        with forward_ad.dual_level():
            primal = _sites(spike)
            direction = torch.zeros_like(primal)
            direction[:, 1] = 1.0
            weight = _weights(spike, stage, forward_ad.make_dual(primal, direction), pick, ad_mode="jvp")
            return forward_ad.unpack_dual(weight).tangent

    assert torch.equal(tangent("strided"), tangent("copied"))
