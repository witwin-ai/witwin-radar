"""One batched replay per frame, and what it must cost (plan items 2 and 3).

A radar frame is many slow-time slots. Before Phase 7 the adapter could replay
a frozen topology at ONE instant, so a frame was either a Python loop over
per-slot consumer calls - which multiplies the per-frame host-observation
budget by the slot count - or a stacked batch whose source-by-sink outer
product costs the SQUARE of the slot count in pair segments.

``reevaluate_slots`` is the third option: one call, block-diagonal pairing,
linear growth. Everything here either pins that it produces exactly what the
loop produces, or pins the cost that makes it worth having.
"""

from __future__ import annotations

import inspect

import pytest
import torch

pytest.importorskip("witwin.channel")

from support import multi_endpoint_driver as drv
from support import refreshed_slow_time as refreshed
from support.synthesis_batch import to_synthesis

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def spike():
    return drv.MultiEndpointSpike()


@pytest.fixture(scope="module")
def spec():
    return drv.make_spec()


def _slot_times(spec):
    """``t[slot] = slot * chirp_period_s``: the world time of every TDM slot."""

    from witwin.radar.synthesis.assembly import tdm_slot_count

    slots = tdm_slot_count(num_chirps=spec.num_chirps, num_tx=spec.num_tx)
    return torch.arange(slots, device="cuda", dtype=torch.float64) * spec.chirp_period_s


def _stack(spike, velocity, times):
    return drv.slot_site_stack(spike.site_tensor(), velocity, times)


def test_batched_slots_equal_a_per_slot_loop(spike):
    """Exact equality, slot by slot. Not a tolerance.

    The batched call and the loop hand the same numbers to the same kernel in
    the same order; the only difference is how many launches carry them. If
    that were true only to a tolerance, something in the batching would be
    reordering a reduction, and a tolerance would hide it.
    """

    slots = 8
    times = [index * 1.0e-4 for index in range(slots)]
    stack = _stack(spike, (0.0, 12.0, 0.0), times)
    batched_in, batched_out = spike.slot_legs(stack, slot_count=slots)

    assert batched_in.slot_count == slots
    assert batched_in.leg_count == slots * spike.inbound.row_count
    assert batched_out.leg_count == slots * spike.outbound.row_count

    sites = len(spike.site_ids)
    for slot in range(slots):
        one_in, one_out = spike.legs(stack[slot * sites : (slot + 1) * sites])
        for batched, single, leg in ((batched_in, one_in, "inbound"), (batched_out, one_out, "outbound")):
            view = batched.slot(slot)
            assert torch.equal(view.delay_s, single.delay_s), (leg, slot)
            assert torch.equal(view.coefficient, single.coefficient), (leg, slot)
            assert torch.equal(view.pair_index, single.pair_index), (leg, slot)
            assert torch.equal(view.pair_offsets, single.pair_offsets), (leg, slot)
            assert torch.equal(view.source_id, single.source_id), (leg, slot)
            assert torch.equal(view.sink_id, single.sink_id), (leg, slot)
            if single.row_valid is None:
                assert view.row_valid is None
            else:
                assert torch.equal(view.row_valid, single.row_valid), (leg, slot)


def test_pair_count_grows_linearly_not_quadratically(spike):
    """The whole reason ``slot_count`` exists, stated as a growth law.

    Both ends of the inbound leg are replicated here - the transmitters AND the
    sites - which is the case a plain stacked batch handles worst: its pair set
    is the full ``(T * S) x (T * K)`` outer product, so it grows as ``T^2`` and
    is ``T`` times larger than anything a caller wanted.
    """

    single, _ = spike.slot_legs(_stack(spike, (0.0, 0.0, 0.0), [0.0]), slot_count=1)
    base_pairs = single.pair_count
    base_rows = single.leg_count
    assert base_pairs == len(spike.transmitter_ids) * len(spike.site_ids)

    for slots in (8, 64):
        times = [index * 1.0e-5 for index in range(slots)]
        batched, _ = spike.slot_legs(_stack(spike, (0.0, 1.0, 0.0), times), slot_count=slots)
        assert batched.pair_count == slots * base_pairs
        assert batched.leg_count == slots * base_rows
        assert batched.pair_count != slots * slots * base_pairs
        assert batched.pairs_per_slot == base_pairs
        assert batched.rows_per_slot == base_rows


def test_the_batched_replay_is_exactly_one_consumer_call_per_leg(spike, monkeypatch):
    """The pin that forbids a Python per-slot loop.

    A loop would publish identical numbers, so nothing downstream could tell
    the difference; the only observable is the call count, and this is where it
    is observed.
    """

    from witwin.channel.propagation import consumer

    slots = 64
    times = [index * 1.0e-5 for index in range(slots)]
    stack = _stack(spike, (0.0, 1.0, 0.0), times)
    spike.slot_legs(stack, slot_count=slots)  # warm the replication cache

    calls = {"count": 0}
    original = consumer.reevaluate

    def counting(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(consumer, "reevaluate", counting)
    spike.slot_legs(stack, slot_count=slots)
    assert calls["count"] == 2, calls


def test_a_slot_view_aliases_the_batched_storage(spike):
    """A slot is a window on the batch, not a copy of it.

    The payload contract says the leg tensors alias the consumer's; slicing a
    slot has to keep that, or a caller that composes per slot would silently be
    working on copies and a gradient would stop flowing at the slice.
    """

    slots = 4
    times = [index * 1.0e-4 for index in range(slots)]
    batched, _ = spike.slot_legs(_stack(spike, (0.0, 3.0, 0.0), times), slot_count=slots)
    for slot in range(slots):
        view = batched.slot(slot)
        assert view.delay_s.data_ptr() == (
            batched.delay_s.data_ptr() + slot * view.leg_count * batched.delay_s.element_size()
        )
        assert view.coefficient.data_ptr() == (
            batched.coefficient.data_ptr() + slot * view.leg_count * batched.coefficient.element_size()
        )
        assert view.slot_count == 1


def test_a_ragged_stack_is_refused_before_any_native_work(spike):
    """Three endpoints cannot be two slots of anything."""

    stack = _stack(spike, (0.0, 0.0, 0.0), [0.0, 1.0e-4])
    transmitters = spike._stacked_ids(
        spike.stacked([position for _, position in spike.transmitters], 2), spike.transmitter_ids, 1.0
    )
    sinks = spike._stacked_ids(stack, spike.site_ids, None)
    with pytest.raises(ValueError, match="not divisible by slot_count"):
        spike.adapter.reevaluate_slots(spike.inbound, transmitters, sinks, slot_count=3, ad_mode="none")
    with pytest.raises(ValueError, match="slot_count must be a positive int"):
        spike.adapter.reevaluate_slots(spike.inbound, transmitters, sinks, slot_count=0, ad_mode="none")


def test_tdm_slot_indices_come_from_the_phase6_owner(spec):
    """The slot table is the beat kernel's, to the integer.

    ``fmcw_beat.cu`` computes ``slot(c, p) = c * num_tx + segment_tx_index[p]``
    with ``segment_tx_index`` from :func:`pair_tx_index`. These are the same
    integers, so the comparison is exact, and the source-level assertion is
    what stops a second slot table being introduced next to the first one.
    """

    from witwin.radar.synthesis.assembly import pair_slot_index, pair_tx_index, tdm_slot_count

    num_tx = spec.num_tx
    num_rx = spec.num_rx
    pairs = num_tx * num_rx
    chirps = spec.num_chirps
    transmitter = pair_tx_index(num_tx=num_tx, num_rx=num_rx, sensor_pair_count=pairs, device="cuda")
    table = pair_slot_index(num_chirps=chirps, num_tx=num_tx, num_rx=num_rx, sensor_pair_count=pairs, device="cuda")
    expected = torch.stack([chirp * num_tx + transmitter.to(torch.int64) for chirp in range(chirps)])
    assert torch.equal(table, expected)
    assert tdm_slot_count(num_chirps=chirps, num_tx=num_tx) == chirps * num_tx

    # No second owner: the slot index is BUILT from the Phase-6 tx table
    # rather than rederiving ``pair % num_tx`` next to it.
    source = inspect.getsource(pair_slot_index)
    assert "pair_tx_index(" in source
    body = source.split('"""')[-1]
    assert "%" not in body
    assert "remainder" not in body


def test_a_static_frame_is_bit_identical_in_both_modes(spike, spec):
    """Zero velocity removes every difference between the two models."""

    times = _slot_times(spec)
    slots = int(times.shape[0])
    response = drv.make_response()

    from witwin.radar.synthesis.fmcw import synthesize_fmcw

    composed, _, _ = spike.frame(spike.site_tensor(), response, ad_mode="none")
    frozen_cube = synthesize_fmcw(to_synthesis(composed), spec)

    stack = drv.slot_site_stack(spike.site_tensor(), (0.0, 0.0, 0.0), times)
    batched_in, batched_out = spike.slot_legs(stack, slot_count=slots)
    frames = spike.slot_frames(batched_in, batched_out, response)
    refreshed_cube = refreshed.refreshed_cube(frames, spec, num_chirps=spec.num_chirps)
    assert torch.equal(refreshed_cube, frozen_cube)
