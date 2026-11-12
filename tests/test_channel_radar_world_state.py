"""Channel's time-varying CIR and a Radar frame use the same world state.

Driven from ONE ``DynamicScene`` at ONE ``times_s`` vector. Same kernel, same
inputs, so the assertion is bitwise equality of the delays and an exact match on
``CompiledScene.time_s`` - a tolerance here would be an admission that they are
two computations.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("witwin.channel")

from support import multi_endpoint_driver as drv
from support import multi_endpoint_geometry as geo
from support import multi_endpoint_world as world

SLOT_COUNT = 6
SLOT_PERIOD_S = 1.0e-4
SNAPSHOT_TIME_S = 0.25


def _times() -> torch.Tensor:
    index = torch.arange(SLOT_COUNT, dtype=torch.float64)
    return SNAPSHOT_TIME_S + index * SLOT_PERIOD_S


@pytest.mark.gpu
def test_channel_cir_and_radar_frames_use_the_same_world_state():
    """One ``DynamicScene``, one ``times_s``, two consumers, one answer.

    Channel's ``evaluate_time_varying`` and Radar's slot-batched replay are
    given the SAME compiled scene, the SAME frozen topology and the SAME
    slot-major endpoint stacks. They must therefore agree BITWISE on the delay
    of every row at every instant: they are the same native call reached
    through two facades, and anything less than ``torch.equal`` would mean one
    of the two had inserted an operation of its own.

    ``CompiledScene.time_s`` carries the snapshot instant, so the timestamp the
    Channel result is labelled with and the world the Radar frame was built
    from are checkable against each other rather than assumed to match.
    """

    from witwin.channel.propagation import consumer

    from witwin.radar.channel import _endpoint_batch

    times = _times()
    dynamic = world.make_dynamic_scene()
    snapshot = dynamic.at(SNAPSHOT_TIME_S)
    compiled = world.compile_snapshot(snapshot)

    # The world state both consumers were built from, labelled and checkable.
    assert compiled.time_s == pytest.approx(SNAPSHOT_TIME_S)
    assert compiled.time_s == pytest.approx(float(times[0]))
    assert snapshot.time_s == pytest.approx(SNAPSHOT_TIME_S)

    spike = drv.MultiEndpointSpike(compiled=compiled)
    base = spike.site_tensor()
    stack = drv.slot_site_stack(base, geo.SITE_P_VELOCITY_M_PER_S, (times - times[0]).tolist())

    inbound, _ = spike.slot_legs(stack, slot_count=SLOT_COUNT)
    assert inbound.slot_count == SLOT_COUNT

    # The Channel side, from the same handle and the same stacks the adapter
    # built. Reaching for the adapter's own batch helper is deliberate: a
    # second, test-written endpoint batch would be a second world state and the
    # bitwise claim would be about the test rather than about the two consumers.
    transmitters = spike._stacked_ids(
        spike.stacked([position for _, position in spike.transmitters], SLOT_COUNT),
        spike.transmitter_ids,
        geo.TX_POWER_W,
    )
    sites = spike._stacked_ids(stack, spike.site_ids, None)
    evaluation = consumer.evaluate_time_varying(
        compiled,
        consumer.TimeVaryingRequest(
            sources=_endpoint_batch(transmitters, "source"),
            sinks=_endpoint_batch(sites, "sink"),
            reference_frequency_hz=geo.REFERENCE_FREQUENCY_HZ,
            topology=spike.inbound.prepared,
            times_s=times,
            response="scalar_transport",
            ad_mode="none",
        ),
    )

    assert evaluation.slot_count == SLOT_COUNT
    assert evaluation.row_count == inbound.rows_per_slot
    assert torch.equal(evaluation.times_s, times)

    moved = False
    for slot in range(SLOT_COUNT):
        radar_slot = inbound.slot(slot)
        assert torch.equal(evaluation.delay_s[slot], radar_slot.delay_s), slot
        assert torch.equal(evaluation.transport.coefficient[slot], radar_slot.coefficient), slot
        if slot and not torch.equal(evaluation.delay_s[slot], evaluation.delay_s[0]):
            moved = True
    # Non-vacuity: a world that did not move would make every slot trivially
    # equal to every other and the bitwise claim would prove nothing.
    assert moved

    # And the two consumers publish the same per-slot pair segmentation, which
    # is what makes "the same rows at the same instants" a checkable statement
    # rather than a coincidence of lengths.
    assert evaluation.pair_count == inbound.pairs_per_slot
    assert torch.equal(evaluation.pair_offsets, inbound.slot(0).pair_offsets)
