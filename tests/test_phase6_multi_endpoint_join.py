"""The two-way join guarantees, re-run against REAL multi-pair Channel legs.

Every multi-pair statement the join makes - join by identity rather than by
array position, a canonical composed order, a CSR pair partition that spans the
declared front end including empty segments, a gradient reduction over several
sites - was previously pinned only against fabricated ``SimpleNamespace`` legs.
Those tests stay: they are the only way to permute leg rows ARBITRARILY, and
they remain the stronger statement about order invariance.

What they could not say is whether the real producer ever exercises any of it.
This file answers that with the multi-endpoint fixture: 2 TX x 2 sites x 2 RX
through the production ``ChannelPropagationAdapter``, three inbound rows, seven
outbound rows, eleven composed round trips over four sensor pairs, two of which
are genuinely empty.

The row-order divergence in particular is real rather than staged. Channel's
frozen leg row order is ascending ``(pair_index, component)`` with
``pair_index = sink_row * source_count + source_row`` over the caller's ENDPOINT
BATCH ROW POSITIONS; the join's canonical order is built from stable IDENTITY.
Declaring the site batch as ``[Q, P]`` - positions and stable IDs swapped
together, so it is the same physical world - makes the two disagree, and the
composed frame must come back elementwise identical anyway.

TWO GUARANTEES THIS FILE CANNOT REACH, measured rather than assumed. A mutation
audit over the join found that ten of twelve deliberate defects fail here; the
two survivors are both structural properties of a real single-wall scene, not
gaps in the assertions:

* Ordering a ``(sensor pair, site)`` cell by leg ROW POSITION instead of by the
  identity key is indistinguishable here. Channel emits line of sight before
  reflection within an endpoint pair, so ascending row index and ascending
  identity key coincide inside every cell this fixture can build.
* Dropping ``primitive_sequence`` from the identity key is indistinguishable
  here. A planar wall has exactly one specular point per endpoint pair, so no
  endpoint pair ever carries two reflection rows for the primitive column to
  separate; the differing ``primitive_sequence == [1]`` row lives in a
  DIFFERENT pair, where ``component`` already separates it.

Both are reachable only by publishing leg rows in an order, or with an identity
key, that this producer does not produce, so both are covered by fabricated
legs - but by DIFFERENT fabricated legs, and saying "the Phase-4/5 tests reach
them" would have been wrong about the second one:

* The row-position ordering is reached by ``test_phase5_join_identity.py``,
  which permutes fabricated leg rows on purpose, and again by
  ``test_phase6_identity_key_columns.py``.
* The identity-key columns are reached ONLY by
  ``test_phase6_identity_key_columns.py``. Every OTHER fabricated leg is built
  from the short row form of ``support.join_fixture.frozen_leg``, which aliases
  depth, primitive and material to the component - so ``component`` alone
  disambiguates there exactly as it does here, and a mutation audit confirmed
  that collapsing the key to the component alone passed the whole suite before
  that file existed.
"""

from __future__ import annotations

import pytest
import torch
import torch.autograd.forward_ad as forward_ad

pytest.importorskip("witwin.channel")

from reference.two_way_torch import PerSiteResponse  # noqa: E402
from support import multi_endpoint_driver as drv  # noqa: E402
from support import multi_endpoint_geometry as geo  # noqa: E402

pytestmark = pytest.mark.gpu

DELAY_RTOL = 1.0e-6

# tau_rt is nanosecond scale, so a delay loss enters through this scaling to put
# its gradient at order one. The same 1e8 the Phase-5 AD tests use.
DELAY_SCALE = 1.0e8

# Two clearly different per-site values, indexed by SITE RANK - the rank in the
# SORTED site ID list, which is the join's own site slot. ``ScalarRcsResponse``
# broadcasts one number over every site and therefore cannot tell a correct
# per-site slot assignment from a mixed-up one; the swap tests below need a
# response that can.
SITE_RESPONSE_VALUES = ((3.0e4 + 1.0e4j), (-7.0e4 + 5.0e4j))


def _site_response(*, requires_grad: bool = False) -> PerSiteResponse:
    value = torch.tensor(SITE_RESPONSE_VALUES, dtype=torch.complex64, device="cuda")
    if requires_grad:
        value = value.clone().requires_grad_(True)
    return PerSiteResponse(value)


@pytest.fixture(scope="module")
def spike():
    return drv.MultiEndpointSpike()


# --------------------------------------------------------------------------
# The composed partition
# --------------------------------------------------------------------------


def test_the_composed_rows_are_the_ones_the_geometry_predicts(spike):
    """Eleven round trips, in the canonical order, with the analytic delays.

    The oracle in ``support.multi_endpoint_geometry`` builds this list from the
    image-source table plus the two ordering rules the join declares (sink-major
    sensor pair rank over the DECLARED endpoint lists, then the rank in the
    SORTED site ID list, then the leg identity keys). It never imports
    ``witwin.radar.paths``, so agreement here is a cross-check rather
    than a mirror.
    """

    composed, _, _ = spike.frame()
    predicted = spike.predicted_combined_rows()

    assert composed.path_count == 11
    assert len(predicted) == 11
    assert drv.composed_keys(spike, composed) == [row.key for row in predicted]
    for index, row in enumerate(predicted):
        assert float(composed.total_delay_s[index]) == pytest.approx(row.total_delay_s, rel=DELAY_RTOL), row.key
    assert bool(composed.row_valid.all())
    # Not vacuous: the two mixed round trips through site P and RX_A are 20 ps
    # apart, so identity - not a sorted delay list - is what named them.
    keys = drv.composed_keys(spike, composed)
    mixed = [
        float(composed.total_delay_s[index])
        for index, key in enumerate(keys)
        if key[:3] == (10, 20, 30) and key[3] != key[4]
    ]
    assert len(mixed) == 2
    assert 1.0e-11 < abs(mixed[0] - mixed[1]) < 1.0e-10


def test_rows_are_sorted_into_a_valid_pair_partition(spike):
    composed, _, _ = spike.frame()

    assert composed.sensor_pair_count == 4
    offsets = composed.pair_offsets.tolist()
    assert offsets == [0, 5, 5, 11, 11]
    assert offsets == geo.combined_pair_offsets(spike.predicted_combined_rows(), sensor_pair_count=4)
    assert offsets[0] == 0
    assert offsets[-1] == composed.path_count
    assert offsets == sorted(offsets)
    assert len(offsets) == composed.sensor_pair_count + 1
    ranks = composed.sensor_pair_index.tolist()
    assert ranks == sorted(ranks)
    # SINK-MAJOR over the DECLARED endpoint lists, mirroring Channel's own pair
    # index. The two occupied ranks are 0 = (TX_A, RX_A) and 2 = (TX_A, RX_B).
    assert set(ranks) == {0, 2}
    pairs = list(zip(composed.topology.radar_source_id.tolist(), composed.topology.radar_sink_id.tolist(), strict=True))
    assert set(pairs) == {(10, 30), (10, 31)}


def test_the_pair_partition_spans_the_front_end_not_the_surviving_rows(spike):
    """Two REAL empty sensor-pair segments, all the way through synthesis.

    ``TX_B`` discovers nothing, so sensor pairs 1 = (TX_B, RX_A) and
    3 = (TX_B, RX_B) own zero-length segments. Deriving the pair set from
    surviving composed rows would renumber the IQ cube from four pairs to two,
    which is a silent wrong answer rather than a missing one, and every consumer
    that indexes a virtual array by pair would read the wrong channel.
    """

    from witwin.radar.synthesis.fmcw import synthesize_fmcw

    composed, _, _ = spike.frame()
    offsets = composed.pair_offsets.tolist()
    empty = [rank for rank in range(composed.sensor_pair_count) if offsets[rank] == offsets[rank + 1]]
    assert empty == [1, 3]

    spec = drv.make_spec(num_chirps=2)
    iq = synthesize_fmcw(drv.to_synthesis(composed), spec)
    assert tuple(iq.shape) == (2, 4, spec.num_samples)
    occupied = [int(torch.count_nonzero(iq[:, rank, :])) for rank in range(4)]
    assert occupied == [2 * spec.num_samples, 0, 2 * spec.num_samples, 0]


# --------------------------------------------------------------------------
# Identity, not array position
# --------------------------------------------------------------------------


def _reversed_spike():
    return drv.MultiEndpointSpike(sites=geo.SITES_REVERSED)


def test_channel_really_does_publish_a_different_row_order_for_the_swap():
    """The non-vacuity gate for everything below.

    Case A declares the site batch ``[P, Q]``, where the array position happens
    to agree with the stable-ID order, and Channel's leg order coincides with
    the join's canonical order. Case B declares ``[Q, P]`` - the same physical
    world, positions and IDs swapped together - and they diverge: the composed
    rows reference inbound leg rows ``[1, 1, 2, 2, 0, ...]`` instead of
    ``[0, 0, 1, 1, 2, ...]``, so a higher leg row is emitted first.

    If this ever stops diverging, the two tests after it become tautologies and
    must fail here rather than pass quietly.
    """

    straight = drv.MultiEndpointSpike()
    swapped = _reversed_spike()

    assert straight.site_ids == (geo.SITE_P_STABLE_ID, geo.SITE_Q_STABLE_ID)
    assert swapped.site_ids == (geo.SITE_Q_STABLE_ID, geo.SITE_P_STABLE_ID)
    # The legs themselves are published in a different order.
    assert straight.inbound.sink_id.tolist() == [20, 20, 21]
    assert swapped.inbound.sink_id.tolist() == [21, 20, 20]
    assert straight.composer.topology.inbound_row.tolist() != swapped.composer.topology.inbound_row.tolist()
    assert straight.composer.topology.inbound_row.tolist() == [0, 0, 1, 1, 2, 0, 0, 1, 1, 2, 2]
    assert swapped.composer.topology.inbound_row.tolist() == [1, 1, 2, 2, 0, 1, 1, 2, 2, 0, 0]


def test_the_swapped_batch_order_composes_to_an_elementwise_identical_frame():
    """Not an equal SET - elementwise, in the same sequence.

    The join's membership was already by identity; its composed ORDER is the
    part that used to be a function of the joined rows' positions. A set
    comparison would pass against a permuted sequence, and every consumer that
    reads a composed row by index would still be wrong.
    """

    straight = drv.MultiEndpointSpike()
    swapped = _reversed_spike()
    # A response with a DIFFERENT value per site, so a site slot assigned from
    # the endpoint batch position instead of the sorted stable-ID rank changes
    # the composed transfer instead of cancelling out.
    response = _site_response()
    first, _, _ = straight.frame(response=response)
    second, _, _ = swapped.frame(response=response)

    assert first.path_count == second.path_count == 11
    assert first.sensor_pair_count == second.sensor_pair_count
    assert first.pair_offsets.tolist() == second.pair_offsets.tolist()
    assert first.sensor_pair_index.tolist() == second.sensor_pair_index.tolist()
    assert drv.composed_keys(straight, first) == drv.composed_keys(swapped, second)
    for name in ("total_delay_s", "complex_transfer_ref", "row_valid"):
        assert torch.equal(getattr(first, name), getattr(second, name)), name
    # And the raw leg row indices differ, so the join genuinely had to reorder.
    assert first.topology.inbound_row.tolist() != second.topology.inbound_row.tolist()


def test_the_swapped_batch_order_produces_bit_identical_gradients():
    """Reverse mode through the swap, compared bit for bit.

    Bit identity is a legitimate assertion rather than a lucky one: the join's
    VJP reduces over the FROZEN CSR segments, one thread per gradient slot with
    no atomics, so if the composed order is genuinely canonical the segments are
    the same segments in the same order and the arithmetic is identical. The
    per-site response gradient is the one that would move first, because it is a
    reduction over every round trip through that site.

    The site position gradients live in endpoint batch order, so they are
    compared through the swap rather than positionally.
    """

    weights = torch.arange(1, 12, dtype=torch.float32, device="cuda")

    def gradients(sites):
        run = drv.MultiEndpointSpike(sites=sites)
        positions = run.site_tensor(requires_grad=True)
        response = _site_response(requires_grad=True)
        composed, _, _ = run.frame(positions, response, ad_mode="vjp")
        assert composed.path_count == 11
        loss = (weights * composed.total_delay_s * 1.0e8).sum() + (weights * composed.complex_transfer_ref.real).sum()
        loss.backward()
        return positions.grad, response.value.grad

    site_grad, response_grad = gradients(geo.SITES)
    swapped_site, swapped_response = gradients(geo.SITES_REVERSED)

    # The per-site response gradient lives in SITE RANK order, which the swap
    # does not touch: it is the one that would move first if the join walked
    # the CSR segments in a different order.
    assert torch.equal(response_grad, swapped_response)
    # The site position gradients live in ENDPOINT BATCH order, so they are
    # compared through the swap rather than positionally.
    assert torch.equal(site_grad.flip(0), swapped_site)

    # Not vacuous, and structured: both sites carry a nonzero in-plane gradient
    # and an exactly zero out-of-plane one, because every path lies in z = 0.
    assert float(site_grad[:, :2].abs().min()) > 1.0e-6
    assert site_grad[:, 2].tolist() == [0.0, 0.0]
    assert float(response_grad.abs().min()) > 0.0
    # The two sites see different geometry and carry different responses, so
    # neither gradient may be a broadcast of the other.
    assert not torch.equal(site_grad[0], site_grad[1])
    assert complex(response_grad[0]) != complex(response_grad[1])


def test_the_multi_site_delay_gradient_is_the_closed_form_one():
    """Reverse mode checked by VALUE, not against a second run of itself.

    The swap comparison above is bit identity between two runs, so a defect that
    scales every gradient the same way - and in particular one that only appears
    once the batch carries more than one site - cancels out of it. The join's own
    VJP is value-checked at two sites in ``test_phase5_two_way_join_ad.py``
    against a finite-difference-validated float64 oracle; what that leaves is the
    CHANNEL leg gradient at multi-site width, which is value-checked only at one
    site (``test_phase5_reflection_ad.py``).

    A delay-only loss closes that: ``d(tau)/d(site position)`` is the unit vector
    from the (mirrored) fixed endpoint to the site over ``c``, summed over both
    legs of every row that reaches the site, which is the transpose of the rate
    the Doppler test above already validates in forward mode. The transfer term
    is deliberately left out - its geometry dependence has no comparably cheap
    closed form, and a loss it dominated would hide the delay gradient.
    """

    run = drv.MultiEndpointSpike()
    weights = torch.arange(1, 12, dtype=torch.float32, device="cuda")
    positions = run.site_tensor(requires_grad=True)
    composed, _, _ = run.frame(positions, ad_mode="vjp")

    assert composed.path_count == 11
    assert bool(composed.row_valid.all())
    (weights * composed.total_delay_s * DELAY_SCALE).sum().backward()

    expected = geo.combined_delay_gradient_s_per_m(run.predicted_combined_rows(), weights.tolist())
    measured = positions.grad.tolist()
    assert len(measured) == 2
    for row, stable_id in enumerate(run.site_ids):
        for axis in range(3):
            assert measured[row][axis] == pytest.approx(
                DELAY_SCALE * expected[stable_id][axis], rel=1.0e-5, abs=1.0e-6
            ), (stable_id, axis)

    # Not vacuous, and not a broadcast: both sites carry an in-plane gradient of
    # order ten, they differ, and the out-of-plane one is exactly zero because
    # every path lies in z = 0.
    for row in range(2):
        assert abs(measured[row][0]) > 1.0 and abs(measured[row][1]) > 1.0
        assert measured[row][2] == 0.0
    assert measured[0] != measured[1]


def test_each_site_response_reaches_that_sites_rows_and_no_others(spike):
    """The response slot is the site, asserted absolutely rather than by a swap.

    The two swap tests above compare two runs against each other, so a response
    slot that was wrong the SAME way in both - every row reading site 0, say -
    cancels out of the comparison and survives them. Re-composing the same frame
    with a per-site response that changed by a different factor per site pins it
    absolutely: the composed transfer of a site-Q row must move by site Q's
    factor and by nothing else.

    Only the frozen response slots change between the two compositions, so the
    ratio is exact arithmetic on the same inputs rather than a re-derivation.
    """

    unit = PerSiteResponse(torch.tensor([1.0 + 0.0j, 1.0 + 0.0j], dtype=torch.complex64, device="cuda"))
    scaled = PerSiteResponse(torch.tensor([2.0 + 0.0j, -3.0 + 0.0j], dtype=torch.complex64, device="cuda"))
    base, _, _ = spike.frame(response=unit)
    moved, _, _ = spike.frame(response=scaled)

    # Site slot order is the SORTED stable-ID order, so slot 0 is P (20).
    factor = {geo.SITE_P_STABLE_ID: 2.0 + 0.0j, geo.SITE_Q_STABLE_ID: -3.0 + 0.0j}
    keys = drv.composed_keys(spike, base)
    assert {key[1] for key in keys} == set(factor)
    for index, key in enumerate(keys):
        reference = complex(base.complex_transfer_ref[index])
        assert reference != 0j, key
        ratio = complex(moved.complex_transfer_ref[index]) / reference
        assert ratio.real == pytest.approx(factor[key[1]].real, rel=1e-5), key
        assert abs(ratio.imag) < 1.0e-5 * abs(ratio.real), key


# --------------------------------------------------------------------------
# Agreement with the trusted single-pair chain
# --------------------------------------------------------------------------


def test_every_composed_row_agrees_with_a_single_pair_single_site_run(spike):
    """The batched frame against eleven 1 x 1 x 1 runs of the Phase-4/5 shape.

    Same compiled scene, so any difference is batching and nothing else.
    Bit-identical rather than close: batching endpoints changes the launch
    shape, not the arithmetic, and a tolerance here would hide a real
    cross-pair contamination at these magnitudes.

    Only ``TX_A`` appears: ``TX_B`` publishes no leg row at all, so a
    ``(TX_B, site, RX)`` composer refuses to freeze rather than producing a
    comparison row. That refusal is asserted in
    ``test_a_site_the_real_legs_never_reach_is_refused``.
    """

    composed, _, _ = spike.frame()
    keys = drv.composed_keys(spike, composed)
    reference = {
        key: (float(composed.total_delay_s[index]), complex(composed.complex_transfer_ref[index]))
        for index, key in enumerate(keys)
    }
    assert len(reference) == composed.path_count == 11

    seen = 0
    for site in geo.SITES:
        for receiver in geo.RECEIVERS:
            single = spike.single_pair(geo.TRANSMITTERS[0], site, receiver)
            one, _, _ = single.frame()
            for index, key in enumerate(drv.composed_keys(single, one)):
                assert key in reference, key
                delay, transfer = reference[key]
                assert float(one.total_delay_s[index]) == delay, key
                assert complex(one.complex_transfer_ref[index]) == transfer, key
                seen += 1
    assert seen == composed.path_count


# --------------------------------------------------------------------------
# Dying rows across several pairs
# --------------------------------------------------------------------------


def _reflection_survives(source, sink) -> bool:
    point = geo.specular_point_m(source, sink)
    return point is not None and geo.face_containing(point) is not None


def test_rows_that_die_under_motion_are_data_in_a_multi_pair_frame(spike):
    """Site P moves; some of its rows die, and site Q must not notice.

    At ``(2, 2.0, 0)`` the inbound specular point for P sits at ``y = 1.333``
    and the ``P -> RX_A`` one at ``y = 1.316``, both past the facet edge at
    1.2, while ``P -> RX_B`` survives at ``y = 0.291``. ADR-037 publishes that
    through ``row_valid`` as a complete answer, not an error.

    This is strictly richer than the single-pair dying-row test: the survivors
    and the casualties are spread across BOTH occupied sensor pairs, so a
    validity mask that was accidentally computed per pair, or broadcast from
    the first row of a segment, fails here and cannot fail there.
    """

    moved = ((geo.SITE_P_STABLE_ID, geo.SITE_P_MOVED_POSITION_M), geo.SITES[1])
    positions = dict(moved)
    positions.update(dict(geo.TRANSMITTERS))
    positions.update(dict(geo.RECEIVERS))

    static, _, _ = spike.frame()
    static_keys = drv.composed_keys(spike, static)
    composed, inbound, outbound = spike.frame(spike.site_tensor([position for _, position in moved]))

    def expected(rows):
        return [
            True if row.component == "los" else _reflection_survives(positions[row.source_id], positions[row.sink_id])
            for row in rows
        ]

    assert inbound.row_valid.tolist() == expected(spike.predicted_inbound_rows())
    assert outbound.row_valid.tolist() == expected(spike.predicted_outbound_rows())
    assert inbound.row_valid.tolist() == [True, False, True]
    assert outbound.row_valid.tolist() == [True, False, True, True, True, True, True]

    validity = composed.row_valid.tolist()
    assert validity == [True, False, False, False, True, True, True, False, False, True, True]
    assert sum(validity) == 6

    # Dead rows carry exact zeros, not a partial composition.
    for index, alive in enumerate(validity):
        if not alive:
            assert float(composed.total_delay_s[index]) == 0.0
            assert complex(composed.complex_transfer_ref[index]) == 0j
        else:
            assert float(composed.total_delay_s[index]) > 0.0

    # Survivors in BOTH occupied pairs, so the frame is not one dead segment.
    offsets = composed.pair_offsets.tolist()
    for rank in (0, 2):
        segment = validity[offsets[rank] : offsets[rank + 1]]
        assert any(segment) and not all(segment), rank

    # Site Q never moved, so its rows must be bit-unchanged. Only site P's rows
    # may differ; asserting that a row DID change is what keeps this honest.
    changed = 0
    for index, key in enumerate(static_keys):
        if key[1] == geo.SITE_Q_STABLE_ID:
            assert float(composed.total_delay_s[index]) == float(static.total_delay_s[index]), key
            assert complex(composed.complex_transfer_ref[index]) == complex(static.complex_transfer_ref[index]), key
        elif validity[index] and float(composed.total_delay_s[index]) != float(static.total_delay_s[index]):
            changed += 1
    assert changed > 0


# --------------------------------------------------------------------------
# Multi-site Doppler
# --------------------------------------------------------------------------


def test_two_sites_moving_apart_give_eleven_analytic_doppler_shifts(spike):
    """One forward-only dual carrying a different velocity per site.

    Both signs appear, because P recedes and Q approaches along the paths that
    reach it, and no two rows share a shift: the fixture resolves eleven paths
    rather than one path counted eleven times.
    """

    velocities = {geo.SITE_P_STABLE_ID: geo.SITE_P_VELOCITY_M_PER_S, geo.SITE_Q_STABLE_ID: geo.SITE_Q_VELOCITY_M_PER_S}
    tangent = torch.tensor([velocities[stable_id] for stable_id in spike.site_ids], dtype=torch.float32, device="cuda")
    positions = spike.site_tensor()
    with forward_ad.dual_level():
        composed, inbound, outbound = spike.frame(forward_ad.make_dual(positions, tangent), ad_mode="jvp")
        rate = composed.delay_rate.clone()
        assert inbound.delay_rate is not None
        assert outbound.delay_rate is not None

    expected = geo.combined_doppler_hz(spike.predicted_combined_rows(), velocities)
    measured = [-geo.REFERENCE_FREQUENCY_HZ * value for value in rate.tolist()]
    assert len(measured) == 11
    for index, (value, reference) in enumerate(zip(measured, expected, strict=True)):
        assert value == pytest.approx(reference, rel=1e-5), index

    assert any(value < 0.0 for value in measured)
    assert any(value > 0.0 for value in measured)
    shifts = sorted(measured)
    assert min(b - a for a, b in zip(shifts[:-1], shifts[1:], strict=True)) > 50.0


# --------------------------------------------------------------------------
# What the join costs, and what it refuses
# --------------------------------------------------------------------------


def test_freeze_host_reads_are_counted_at_multi_pair_width(spike):
    """Fourteen ``tolist`` calls: six identity columns per leg, the sites, and
    the composed pair index the freeze-time layout gate reads.

    The same fourteen the one-site Phase-4 join costs, at three and seven leg
    rows over four endpoint pairs. Freeze-time host traffic is a property of the
    number of COLUMNS read, not of the rows in them, and asserting it against a
    real wide leg is what makes that a measurement. The fourteenth is Phase 7
    wiring ``synthesis.assembly.validate_pair_ordering`` into production; it is
    one-time, and the per-frame budget below is still exactly zero.
    """

    from witwin.radar.paths import TwoWayComposer

    calls = {"n": 0}
    original = torch.Tensor.tolist

    def counting(self):
        calls["n"] += 1
        return original(self)

    torch.Tensor.tolist = counting
    try:
        TwoWayComposer.freeze(
            spike.inbound,
            spike.outbound,
            torch.tensor(sorted(spike.site_ids), dtype=torch.int64, device="cuda"),
            radar_source_ids=list(spike.transmitter_ids),
            radar_sink_ids=list(spike.receiver_ids),
            reference_frequency_hz=geo.REFERENCE_FREQUENCY_HZ,
        )
    finally:
        torch.Tensor.tolist = original
    assert calls["n"] == 14, calls["n"]


def test_compose_performs_no_host_observation_at_eleven_rows(spike, monkeypatch):
    """Per frame, over four pairs and two empty segments, nothing crosses back."""

    response = drv.make_response()
    inbound, outbound = spike.legs()
    spike.composer.compose(inbound, outbound, response)  # warm the operator table

    observed: list[str] = []
    for name in ("cpu", "item", "numpy", "tolist"):
        original = getattr(torch.Tensor, name)

        def observing(self, *args, _name=name, _original=original, **kwargs):
            observed.append(_name)
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, name, observing)

    composed = spike.composer.compose(inbound, outbound, response)
    assert composed.path_count == 11
    assert observed == []


def test_a_site_the_real_legs_never_reach_is_refused(spike):
    """A declared site with no real leg row is a wrong stable ID, not an empty pair.

    The empty-segment rule covers a sensor PAIR that discovered nothing - which
    this fixture has two of, and which compose without complaint. A declared
    SITE that appears in neither leg is the opposite problem, and dropping it
    silently would publish a plausible frame that is missing a target the caller
    asked about.
    """

    from witwin.radar.paths import TwoWayComposer

    with pytest.raises(ValueError, match="site 22 has no inbound leg row"):
        TwoWayComposer.freeze(
            spike.inbound,
            spike.outbound,
            torch.tensor([20, 21, 22], dtype=torch.int64, device="cuda"),
            radar_source_ids=list(spike.transmitter_ids),
            radar_sink_ids=list(spike.receiver_ids),
            reference_frequency_hz=geo.REFERENCE_FREQUENCY_HZ,
        )


def test_a_real_leg_endpoint_outside_the_declared_front_end_is_refused(spike):
    """Both halves, reached with real rows rather than fabricated ones.

    The outbound leg genuinely carries ``RX_B`` rows and the inbound leg
    genuinely carries ``TX_A`` rows, so under-declaring either front-end list
    leaves real rows that would simply never be visited.
    """

    from witwin.radar.paths import TwoWayComposer

    def freeze(sources, sinks):
        return TwoWayComposer.freeze(
            spike.inbound,
            spike.outbound,
            torch.tensor(sorted(spike.site_ids), dtype=torch.int64, device="cuda"),
            radar_source_ids=sources,
            radar_sink_ids=sinks,
            reference_frequency_hz=geo.REFERENCE_FREQUENCY_HZ,
        )

    with pytest.raises(ValueError, match=r"\[31\] that are not in radar_sink_ids"):
        freeze(list(spike.transmitter_ids), [geo.RX_A_STABLE_ID])
    with pytest.raises(ValueError, match=r"\[10\] that are not in radar_source_ids"):
        freeze([geo.TX_B_STABLE_ID], list(spike.receiver_ids))


def test_the_other_legs_frame_is_refused_rather_than_gathered(spike):
    """A seven-row batch where a three-row one belongs, with REAL legs.

    The composer's index tables address the frozen leg rows through raw
    pointers, so a batch of the wrong length is a different topology rather than
    a smaller frame. This fixture is the first place where two REAL leg batches
    of different lengths exist at once, so the mismatch can be built by handing
    the composer a genuine batch rather than a synthetic one.
    """

    inbound, outbound = spike.legs()
    assert inbound.leg_count == 3
    assert outbound.leg_count == 7
    with pytest.raises(ValueError) as caught:
        spike.composer.compose(outbound, outbound, drv.make_response())
    message = str(caught.value)
    assert "inbound" in message
    assert "7" in message and "3" in message
    assert "does not belong to this frozen topology" in message

    with pytest.raises(ValueError, match="outbound"):
        spike.composer.compose(inbound, inbound, drv.make_response())
