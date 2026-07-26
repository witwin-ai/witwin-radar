"""One batched replay per frame, and what it must cost (plan items 2 and 3).

A radar frame is many slow-time slots. Before Phase 7 the adapter could replay
a frozen topology at ONE instant, so a frame was either a Python loop over
per-slot consumer calls - which multiplies the per-frame host-observation
budget by the slot count - or a stacked batch whose source-by-sink outer
product costs the SQUARE of the slot count in pair segments.

``reevaluate_slots`` is the third option: one call, block-diagonal pairing,
linear growth. Everything here either pins that it produces exactly what the
loop produces, or pins the cost that makes it worth having.

The refreshed-weight cube is built here for the first time. It is an oracle,
not a production mode: see ``test_frozen_and_refreshed_modes_agree`` and
``test_transverse_motion_defeats_the_frozen_first_order_model``.
"""

from __future__ import annotations

import inspect
import math

import pytest
import torch
import torch.autograd.forward_ad as forward_ad

pytest.importorskip("witwin.channel")

from support import multi_endpoint_driver as drv  # noqa: E402
from support import refreshed_slow_time as refreshed  # noqa: E402
from support.synthesis_batch import to_synthesis  # noqa: E402


pytestmark = pytest.mark.gpu

SPEED_OF_LIGHT_M_PER_S = 299792458.0

#: Float32 relative resolution. A round-trip delay is float32 all the way from
#: the Channel geometry kernel, so two models of the same delay cannot agree
#: more closely than a few of these no matter how exact the physics is. The
#: plan records the same floor as Phase-6 deviation 3.
FLOAT32_EPS = 2.0 ** -23


@pytest.fixture(scope="module")
def spike():
    return drv.MultiEndpointSpike()


@pytest.fixture(scope="module")
def spec():
    return drv.make_spec()


def _slot_times(spec):
    from witwin.radar.synthesis import tdm_slot_times_s

    return tdm_slot_times_s(
        num_chirps=spec.num_chirps,
        num_tx=spec.num_tx,
        chirp_period_s=spec.chirp_period_s,
        device="cuda",
    )


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
        for batched, single, leg in (
            (batched_in, one_in, "inbound"),
            (batched_out, one_out, "outbound"),
        ):
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
        batched, _ = spike.slot_legs(
            _stack(spike, (0.0, 1.0, 0.0), times), slot_count=slots
        )
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
            batched.delay_s.data_ptr()
            + slot * view.leg_count * batched.delay_s.element_size()
        )
        assert view.coefficient.data_ptr() == (
            batched.coefficient.data_ptr()
            + slot * view.leg_count * batched.coefficient.element_size()
        )
        assert view.slot_count == 1


def test_a_ragged_stack_is_refused_before_any_native_work(spike):
    """Three endpoints cannot be two slots of anything."""

    stack = _stack(spike, (0.0, 0.0, 0.0), [0.0, 1.0e-4])
    transmitters = spike._stacked_ids(
        spike.stacked([position for _, position in spike.transmitters], 2),
        spike.transmitter_ids,
        1.0,
    )
    sinks = spike._stacked_ids(stack, spike.site_ids, None)
    with pytest.raises(ValueError, match="not divisible by slot_count"):
        spike.adapter.reevaluate_slots(
            spike.inbound, transmitters, sinks, slot_count=3, ad_mode="none"
        )
    with pytest.raises(ValueError, match="slot_count must be a positive int"):
        spike.adapter.reevaluate_slots(
            spike.inbound, transmitters, sinks, slot_count=0, ad_mode="none"
        )


def test_forward_duals_survive_the_slot_stack(spike, spec):
    """A velocity dual has to reach every slot, and a dead tangent is silent.

    The stack is built with a differentiable expression over the base
    positions, never rebuilt from Python values, so the tangent survives. This
    test carries a RADIAL component on purpose: a lateral-only fixture cannot
    tell a dead tangent from a correct zero, because both publish
    ``delay_rate = 0``.
    """

    from support import multi_endpoint_geometry as geo

    slots = 4
    times = _slot_times(spec)[:slots]
    base = spike.site_tensor()
    speed = 6.0
    # Site P recedes from TX_A straight along the line of sight; site Q moves
    # perpendicular to its own line of sight. Both discriminants in one call:
    # a dead tangent shows up as a wrong P and a correct-looking Q.
    radial = base[0] / base[0].norm()
    lateral = torch.tensor(
        [-float(base[1][1]), float(base[1][0]), 0.0], device=base.device
    )
    lateral = lateral / lateral.norm()
    velocity = torch.stack([speed * radial, speed * lateral])

    with forward_ad.dual_level():
        dual = forward_ad.make_dual(base, velocity)
        stack = drv.slot_site_stack(dual, velocity, times)
        inbound, _ = spike.slot_legs(stack, slot_count=slots, ad_mode="jvp")
        rate = inbound.delay_rate

    assert rate is not None
    reference = float(spec.reference_frequency_hz)
    site_p, site_q = spike.site_ids
    transmitter = spike.transmitter_ids[0]
    los = inbound.component_id == geo.LOS_COMPONENT_ID
    rows_p = los & (inbound.sink_id == site_p) & (inbound.source_id == transmitter)
    assert int(rows_p.sum()) == slots
    # tau_rate = +v / c for a receding sink, and the Doppler shift is negative
    # because Channel's phasor is exp(-j k d).
    for value in rate[rows_p].tolist():
        assert value == pytest.approx(speed / SPEED_OF_LIGHT_M_PER_S, rel=2.0e-3)
        assert -reference * value < 0.0

    rows_q = los & (inbound.sink_id == site_q) & (inbound.source_id == transmitter)
    assert int(rows_q.sum()) == slots
    # A perpendicular site has a zero first-order rate at the frame origin.
    assert float(rate[rows_q][0].abs()) < 1.0e-12


def test_tdm_slot_indices_come_from_the_phase6_owner(spec):
    """The slot table is the beat kernel's, to the integer.

    ``fmcw_beat.cu`` computes ``slot(c, p) = c * num_tx + segment_tx_index[p]``
    with ``segment_tx_index`` from :func:`pair_tx_index`. These are the same
    integers, so the comparison is exact, and the source-level assertion is
    what stops a second slot table being introduced next to the first one.
    """

    from witwin.radar.synthesis import (
        pair_slot_index,
        pair_tx_index,
        tdm_slot_count,
        tdm_slot_times_s,
    )

    num_tx = spec.num_tx
    num_rx = spec.num_rx
    pairs = num_tx * num_rx
    chirps = spec.num_chirps
    transmitter = pair_tx_index(
        num_tx=num_tx, num_rx=num_rx, sensor_pair_count=pairs, device="cuda"
    )
    table = pair_slot_index(
        num_chirps=chirps,
        num_tx=num_tx,
        num_rx=num_rx,
        sensor_pair_count=pairs,
        device="cuda",
    )
    expected = torch.stack(
        [
            chirp * num_tx + transmitter.to(torch.int64)
            for chirp in range(chirps)
        ]
    )
    assert torch.equal(table, expected)
    assert tdm_slot_count(num_chirps=chirps, num_tx=num_tx) == chirps * num_tx

    times = tdm_slot_times_s(
        num_chirps=chirps,
        num_tx=num_tx,
        chirp_period_s=spec.chirp_period_s,
        device="cuda",
    )
    assert times.shape == (chirps * num_tx,)
    for slot in range(chirps * num_tx):
        assert float(times[slot]) == slot * spec.chirp_period_s

    # No second owner: the slot index is BUILT from the Phase-6 tx table
    # rather than rederiving ``pair % num_tx`` next to it.
    source = inspect.getsource(pair_slot_index)
    assert "pair_tx_index(" in source
    body = source.split('"""')[-1]
    assert "%" not in body
    assert "remainder" not in body


def test_frozen_and_refreshed_modes_agree(spike, spec):
    """Two independent models of the same frame, compared where they can be.

    The frozen mode extrapolates the delay from the frame origin with the
    first-order rate the propagation JVP published; the refreshed mode
    reevaluates the propagation at every slot's world state. With a near-radial
    target the second-order term is negligible and the two must agree to the
    FLOAT32 delay resolution and no better, which is what the derived bound
    below says. Asserting a tighter tolerance would be asserting that the
    fixture's float32 round-trip delay carries more information than it does.
    """

    velocity = (-1.0, 0.0, 0.0)
    times = _slot_times(spec)
    slots = int(times.shape[0])
    base = spike.site_tensor()
    response = drv.make_response()

    from witwin.radar.synthesis import synthesize_fmcw_beat

    with forward_ad.dual_level():
        dual = forward_ad.make_dual(
            base,
            torch.tensor([list(velocity)] * base.shape[0], device=base.device),
        )
        composed, _, _ = spike.frame(dual, response, ad_mode="jvp")
        frozen_cube = synthesize_fmcw_beat(to_synthesis(composed), spec)
        origin_delay = composed.total_delay_s.detach().double()
        rate = composed.delay_rate.detach().double()

    stack = drv.slot_site_stack(base, velocity, times)
    batched_in, batched_out = spike.slot_legs(stack, slot_count=slots)
    frames = spike.slot_frames(
        batched_in, batched_out, response, include_delay_rate=False
    )
    exact = torch.stack([frame.total_delay_s.double() for frame in frames])
    predicted = origin_delay.reshape(1, -1) + rate.reshape(1, -1) * times.double().reshape(
        -1, 1
    )

    # The delay statement: the frozen extrapolation reproduces the exact
    # geometry to a few units in the last place of a float32 delay.
    ulp = float(exact.max()) * FLOAT32_EPS
    gap = float((exact - predicted).abs().max())
    assert gap <= 4.0 * ulp, (gap, ulp)

    # The cube statement: normalised to the frame PEAK, not per cell. A per-cell
    # relative comparison is meaningless where eleven multipath rows nearly
    # cancel, and a null is not evidence about a slow-time model.
    refreshed_cube = refreshed.refreshed_cube(
        frames, spec, num_chirps=spec.num_chirps
    )
    assert refreshed_cube.shape == frozen_cube.shape
    peak = float(frozen_cube.abs().max())
    error = float((refreshed_cube - frozen_cube).abs().max())
    assert error <= 1.0e-3 * peak, (error, peak)


def test_a_static_frame_is_bit_identical_in_both_modes(spike, spec):
    """Zero velocity removes every difference between the two models."""

    times = _slot_times(spec)
    slots = int(times.shape[0])
    response = drv.make_response()

    from witwin.radar.synthesis import synthesize_fmcw_beat

    composed, _, _ = spike.frame(spike.site_tensor(), response, ad_mode="none")
    frozen_cube = synthesize_fmcw_beat(to_synthesis(composed), spec)

    stack = drv.slot_site_stack(spike.site_tensor(), (0.0, 0.0, 0.0), times)
    batched_in, batched_out = spike.slot_legs(stack, slot_count=slots)
    frames = spike.slot_frames(
        batched_in, batched_out, response, include_delay_rate=False
    )
    refreshed_cube = refreshed.refreshed_cube(
        frames, spec, num_chirps=spec.num_chirps
    )
    assert torch.equal(refreshed_cube, frozen_cube)


def test_transverse_motion_defeats_the_frozen_first_order_model(spike, spec):
    """The named scenario the frozen mode cannot serve (deliverable 6).

    The frozen mode models the delay as ``tau_rt + tau_rate * t``. For a target
    with a large TRANSVERSE velocity the delay is quadratic in slow time, and
    nothing in the FMCW contract bounds that: the aliasing limit constrains the
    RADIAL speed, which stays well inside it here, and the coherent-interval
    walk guard exists only on the pulsed spec. The frozen cube therefore
    accumulates a real phase error over the frame while every declared limit is
    satisfied.

    This is what the refreshed producer is FOR, and it is recorded rather than
    promoted: the production inner loop stays frozen, and a caller in this
    regime has to either shorten the coherent interval or drive the refreshed
    oracle.
    """

    velocity = (0.0, 12.0, 0.0)
    times = _slot_times(spec)
    slots = int(times.shape[0])
    base = spike.site_tensor()
    response = drv.make_response()

    with forward_ad.dual_level():
        dual = forward_ad.make_dual(
            base,
            torch.tensor([list(velocity)] * base.shape[0], device=base.device),
        )
        composed, _, _ = spike.frame(dual, response, ad_mode="jvp")
        origin_delay = composed.total_delay_s.detach().double()
        rate = composed.delay_rate.detach().double()

    stack = drv.slot_site_stack(base, velocity, times)
    batched_in, batched_out = spike.slot_legs(stack, slot_count=slots)
    frames = spike.slot_frames(
        batched_in, batched_out, response, include_delay_rate=False
    )
    exact = torch.stack([frame.total_delay_s.double() for frame in frames])
    predicted = origin_delay.reshape(1, -1) + rate.reshape(1, -1) * times.double().reshape(
        -1, 1
    )
    # Site P only. Site Q sits far enough off the boresight that this velocity
    # gives it a radial component OUTSIDE the array's unambiguous speed, and a
    # scenario that is already refused by the aliasing limit would not make the
    # point this test is about.
    site_p = spike.site_ids[0]
    rows = frames[0].topology.site_id == site_p
    assert bool(rows.any())
    gap = (exact - predicted).abs()[:, rows]
    rate = rate[rows]

    reference = float(spec.reference_frequency_hz)
    terminal_rad = 2.0 * math.pi * reference * float(gap[slots - 1].max())
    # A real, not a numerical, disagreement: several hundredths of a radian.
    assert terminal_rad > 5.0e-2, terminal_rad

    # And it is SECOND order in slow time: doubling the elapsed time quadruples
    # the gap. That is what identifies it as the missing term of the frozen
    # model rather than float32 noise.
    half = float(gap[slots // 2].max())
    full = float(gap[slots - 1].max())
    span = float(times[slots - 1] / times[slots // 2])
    assert full / half == pytest.approx(span * span, rel=0.15), (full, half, span)

    # Nothing in the declared limits catches it. The radial speed implied by
    # the published rate is inside the aliasing bound.
    radial = float(rate.abs().max()) * SPEED_OF_LIGHT_M_PER_S / 2.0
    assert radial < spec.max_unambiguous_speed_mps, (
        radial,
        spec.max_unambiguous_speed_mps,
    )


def test_double_doppler_is_still_refused(spike, spec):
    """A refreshed weight that also publishes a rate applies Doppler twice."""

    from witwin.radar.synthesis import (
        SlowTimeMode,
        SynthesisPathBatch,
        require_compatible,
    )

    base = spike.site_tensor()
    response = drv.make_response()
    with forward_ad.dual_level():
        dual = forward_ad.make_dual(
            base, torch.tensor([[0.0, 4.0, 0.0]] * base.shape[0], device=base.device)
        )
        composed, _, _ = spike.frame(dual, response, ad_mode="jvp")

    assert composed.delay_rate is not None
    with pytest.raises(ValueError, match="double-counted Doppler"):
        require_compatible(
            SynthesisPathBatch.from_radar_paths(
                composed, slow_time_mode=SlowTimeMode.REFRESHED_WEIGHT_NO_RATE
            ),
            spec,
        )
