"""Which identity-key columns the join reads, pinned one column at a time.

The composed order and the collision guard are both built from the leg row key
``(component, depth, primitive sequence, material sequence)``. Three of those
four columns were pinned by nothing until this file existed, and that was
measured rather than suspected: a mutation audit dropped ``primitive_sequence``
from the key, then collapsed the whole key to ``(component, 0, (), ())``, and
the entire suite still passed.

Both producers hid it for their own reason. Every fabricated leg used the short
row form of ``support.join_fixture.frozen_leg``, which ALIASES depth, primitive
and material to the component. The real multi-endpoint fixture has one planar
wall, so an endpoint pair carries at most one reflection row and ``component``
alone separates everything it can build. In both cases a key of ``component``
alone induces the same order as the full key, so no assertion could tell them
apart.

These legs are fabricated for the same reason the Phase-5 permutation legs are.
The statement under test is about a producer that emits two rows of ONE endpoint
pair differing only in how deep they went, or only in which primitive they
struck - a two-bounce scene with a shared first bounce, or a facet pair a ray
can reach twice - and neither is something this fixture's geometry can be asked
for on demand. What can be asserted here is that when such rows arrive, the join
orders them by that column and refuses them when it cannot tell them apart.
"""

from __future__ import annotations

import pytest
import torch
from reference.two_way_torch import PerSiteResponse  # noqa: E402
from support import join_fixture as fx  # noqa: E402

from witwin.radar.paths import TwoWayComposer

pytestmark = pytest.mark.gpu

SOURCES = [10]
SINKS = [30]
SITES = [20]
REFERENCE_FREQUENCY_HZ = 77.0e9

# One endpoint pair, five rows. Each row after the first differs from BASE_ROW
# in EXACTLY ONE identity-key column, so a key that ignores that column makes it
# a duplicate of BASE_ROW and the join must refuse to freeze.
BASE_ROW = (10, 20, 1, 1, (5,), (2,))
INBOUND_ROWS = [
    BASE_ROW,  # 0
    (10, 20, 1, 2, (5,), (2,)),  # 1  deeper
    (10, 20, 1, 1, (7,), (2,)),  # 2  other primitive
    (10, 20, 1, 1, (5,), (3,)),  # 3  other material
    (10, 20, 2, 1, (5,), (2,)),  # 4  other component
]

# Ascending identity key over those five rows. The component outranks
# everything, then depth, then the primitive sequence, then the material
# sequence - so the material twin sorts BEFORE the primitive twin and both sort
# before the deeper row. Any key that drops a column collides instead of
# producing this.
CANONICAL_ORDER = [0, 3, 2, 1, 4]

# One outbound row, with key columns of its own that share nothing with the
# inbound ones: the composed order below is then the inbound key order alone.
OUTBOUND_ROWS = [(20, 30, 0, 0, (9,), (4,))]

# A row differing from BASE_ROW in exactly the named column.
COLUMN_TWINS = {
    "component": (10, 20, 2, 1, (5,), (2,)),
    "depth": (10, 20, 1, 2, (5,), (2,)),
    "primitive": (10, 20, 1, 1, (7,), (2,)),
    "material": (10, 20, 1, 1, (5,), (3,)),
}


def _freeze(inbound_rows, outbound_rows=OUTBOUND_ROWS) -> TwoWayComposer:
    return TwoWayComposer.freeze(
        fx.frozen_leg(inbound_rows),
        fx.frozen_leg(outbound_rows),
        torch.tensor(SITES, dtype=torch.int64, device="cuda"),
        radar_source_ids=SOURCES,
        radar_sink_ids=SINKS,
        reference_frequency_hz=REFERENCE_FREQUENCY_HZ,
    )


def _run(order):
    """Freeze and compose with the inbound rows published in ``order``."""

    composer = _freeze([INBOUND_ROWS[row] for row in order])
    tau_in, rate_in, c_in = fx.payload(len(INBOUND_ROWS), seed=61)
    tau_out, rate_out, c_out = fx.payload(len(OUTBOUND_ROWS), seed=62)
    _, _, site_value = fx.payload(len(SITES), seed=63)
    delays = tau_in.to(torch.float32) + tau_out.to(torch.float32)[0]

    def take(values):
        index = torch.tensor(order, dtype=torch.int64, device=values.device)
        return values.index_select(0, index).contiguous()

    inbound = fx.leg_batch(
        take(tau_in).to(torch.float32), take(c_in).to(torch.complex64), rate=take(rate_in).to(torch.float32)
    )
    outbound = fx.leg_batch(tau_out.to(torch.float32), c_out.to(torch.complex64), rate=rate_out.to(torch.float32))
    composed = composer.compose(inbound, outbound, PerSiteResponse(site_value.to(torch.complex64)))
    return composer, composed, delays


def _published_to_original(composer, order):
    """Composed rows named by ORIGINAL inbound row number.

    ``inbound_row`` indexes whatever order the leg was published in, so the two
    runs legitimately differ there; mapped back through the permutation they
    must agree, which is the statement that the join found the same rows.
    """

    return [order[row] for row in composer.topology.inbound_row.tolist()]


def test_the_composed_order_is_the_full_key_not_the_component_alone():
    """Five rows of one endpoint pair, ordered by every column of the key.

    Rows 1, 2 and 3 share row 0's component and are separated only by depth, by
    the primitive sequence and by the material sequence respectively. A key of
    ``component`` alone - or any key missing one of those three columns - cannot
    produce this order: it makes two of these rows the same row and the join
    refuses them in ``group_rows`` instead.
    """

    order = list(range(len(INBOUND_ROWS)))
    composer, composed, delays = _run(order)

    assert composed.path_count == len(INBOUND_ROWS)
    assert composer.topology.inbound_row.tolist() == CANONICAL_ORDER
    # Not merely an ordering claim: each composed row carries ITS OWN row's
    # payload, so the right sequence built from the wrong rows fails here. The
    # five delays are distinct, so this can tell them apart.
    expected = delays.index_select(0, torch.tensor(CANONICAL_ORDER, dtype=torch.int64, device="cuda"))
    assert len(set(delays.tolist())) == len(INBOUND_ROWS)
    assert torch.equal(composed.total_delay_s, expected)


def test_a_permuted_publication_order_composes_to_an_identical_frame():
    """The same five rows, published in a different sequence.

    This is the Phase-5 permutation statement re-run over rows whose ONLY
    difference is a tie-break column. Ordering a cell by leg row position
    survives the Phase-6 real-fixture tests; it does not survive this one.
    """

    straight_order = list(range(len(INBOUND_ROWS)))
    shuffled_order = [4, 1, 3, 0, 2]
    assert sorted(shuffled_order) == straight_order

    straight_composer, straight, _ = _run(straight_order)
    shuffled_composer, shuffled, _ = _run(shuffled_order)

    assert _published_to_original(straight_composer, straight_order) == _published_to_original(
        shuffled_composer, shuffled_order
    )
    assert straight_composer.topology.inbound_row.tolist() != shuffled_composer.topology.inbound_row.tolist()
    for name in ("total_delay_s", "delay_rate", "complex_transfer_ref"):
        assert torch.equal(getattr(straight, name), getattr(shuffled, name)), name


@pytest.mark.parametrize("column", sorted(COLUMN_TWINS))
def test_one_key_column_alone_separates_two_rows_of_one_endpoint_pair(column):
    """Two rows, identical but for ``column``, are two rows to the join.

    Freezing at all is the assertion. Drop this column from the key and these
    two rows collide inside their endpoint pair, which
    ``test_two_rows_that_share_every_key_column_are_refused`` shows is refused
    rather than tie-broken on row position.
    """

    composer = _freeze([BASE_ROW, COLUMN_TWINS[column]])

    assert composer.path_count == 2
    # BASE_ROW sorts first for every column: its component, depth, primitive and
    # material are all the lower value.
    assert composer.topology.inbound_row.tolist() == [0, 1]
    assert composer.topology.site_id.tolist() == [SITES[0], SITES[0]]


def test_two_rows_that_share_every_key_column_are_refused():
    """The guard the four tests above lean on, asserted directly.

    A collision is refused rather than tie-broken on row position, because a
    tie-break on position is exactly the positional dependence the identity key
    exists to remove, and it would turn every permutation test vacuous.
    """

    with pytest.raises(ValueError, match="share the identity key"):
        _freeze([BASE_ROW, BASE_ROW])
    with pytest.raises(ValueError, match="share the identity key"):
        _freeze([BASE_ROW], [OUTBOUND_ROWS[0], OUTBOUND_ROWS[0]])


def test_the_same_key_in_a_different_endpoint_pair_is_not_a_collision():
    """Identity is unique WITHIN a pair, not across the leg.

    Two transmitters that both see the site through the same primitive publish
    the same key in different endpoint pairs, and refusing that would refuse
    every real multi-pair leg.
    """

    composer = TwoWayComposer.freeze(
        fx.frozen_leg([BASE_ROW, (11, 20, 1, 1, (5,), (2,))]),
        fx.frozen_leg(OUTBOUND_ROWS),
        torch.tensor(SITES, dtype=torch.int64, device="cuda"),
        radar_source_ids=[10, 11],
        radar_sink_ids=SINKS,
        reference_frequency_hz=REFERENCE_FREQUENCY_HZ,
    )

    assert composer.path_count == 2
    assert composer.topology.radar_source_id.tolist() == [10, 11]
    assert composer.sensor_pair_count == 2
    assert composer.pair_offsets.tolist() == [0, 1, 2]
