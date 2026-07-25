"""Reflection legs through the Channel consumer, and the four combined paths.

The Phase-4 spike froze ``components={los}, max_depth=0`` and got one row per
leg, which cannot distinguish a correct join from a positional one and cannot
exercise the multipath contract at all. This turns the SAME fixture world into
the multipath case: two rows per leg, four combined round trips.

No adapter surgery was needed. ``ChannelPropagationAdapter`` already validates
the requested components against ``capabilities.fixed_topology_components``,
which is ``{los, reflection}``, and already routes discovery through
``prepare_fixed_topology``. The fixture wall at ``x = 4`` was authored in
Phase 4 for exactly this.

Every analytic expectation comes from ``support.phase4_geometry`` in float64,
computed by image source. A combined row is identified by its
``inbound_row``/``outbound_row``, never by its rank in a sorted delay list: the
two mixed paths differ by 20.15 ps and are not separable in a range bin, so a
sorted-set comparison could not tell them apart.
"""

from __future__ import annotations

import pytest
import torch
import torch.autograd.forward_ad as forward_ad

pytest.importorskip("witwin.channel")

from support import phase4_geometry as geo  # noqa: E402
from support import phase4_world as world  # noqa: E402
from support import spike_driver as drv  # noqa: E402


pytestmark = pytest.mark.gpu

MULTIPATH_COMPONENTS = frozenset({"los", "reflection"})
# Relative tolerance for a float32 delay compared against a float64 closed
# form. A nanosecond-scale delay held in float32 carries ~1e-7 relative
# precision, so this is tight, not generous.
DELAY_RTOL = 1.0e-6


@pytest.fixture(scope="module")
def multipath():
    return drv.Phase4Spike(components=MULTIPATH_COMPONENTS, max_depth=1)


def _component_order(frozen) -> list[str]:
    """Map each frozen row to its component name, from the published IDs."""

    names = {
        geo.LOS_COMPONENT_ID: "los",
        geo.REFLECTION_COMPONENT_ID: "reflection",
    }
    return [names[int(value)] for value in frozen.component_id.tolist()]


# --------------------------------------------------------------------------
# The legs themselves
# --------------------------------------------------------------------------


def test_each_leg_freezes_one_line_of_sight_and_one_reflection_row(multipath):
    for name, frozen in (
        ("inbound", multipath.inbound),
        ("outbound", multipath.outbound),
    ):
        assert frozen.row_count == 2, name
        assert frozen.components == ("los", "reflection"), name
        assert _component_order(frozen) == ["los", "reflection"], name
        assert frozen.depth.tolist() == [0, 1], name
        # A line-of-sight row interacts with nothing and carries the sentinel
        # -1; the reflection row names the wall primitive it bounced off.
        assert frozen.primitive_sequence.tolist() == [[-1], [0]], name
        assert frozen.material_sequence.tolist() == [[-1], [0]], name


def test_leg_delays_match_the_image_source_closed_form(multipath):
    tx, site, rx = drv.positions()
    inbound, outbound = _legs(multipath, tx, site, rx)
    expected = geo.leg_delays_s()
    for leg_name, legs in (("inbound", inbound), ("outbound", outbound)):
        components = _component_order(legs)
        for row, component in enumerate(components):
            assert float(legs.delay_s[row]) == pytest.approx(
                expected[(leg_name, component)], rel=DELAY_RTOL
            ), (leg_name, component)


def test_the_specular_points_are_where_the_image_source_puts_them():
    """A closed-form cross-check of the fixture, independent of the solver.

    If this drifts, every delay below drifts with it, and the failure would
    otherwise look like a Channel bug rather than a fixture one.
    """

    inbound = geo.specular_point_m(geo.TX_POSITION_M, geo.SITE_POSITION_M)
    outbound = geo.specular_point_m(geo.RX_POSITION_M, geo.SITE_POSITION_M)
    assert inbound == pytest.approx((4.0, 0.4, 0.0), abs=1e-12)
    assert outbound == pytest.approx((4.0, 0.394871794871795, 0.0), abs=1e-12)
    # And both land inside the authored facet, which is why they exist at all.
    for point in (inbound, outbound):
        assert -3.0 < point[1] < 3.0
        assert -3.0 < point[2] < 3.0


def test_the_frozen_row_identity_is_the_same_storage_on_every_frame(multipath):
    """The join keys on identity, so identity must not be rebuilt per frame.

    The consumer publishes the PREPARED topology's own tensors on reevaluate.
    Asserting storage identity rather than equality is what makes "the join is
    frozen" a measured property instead of a hope: an equal-but-new tensor
    would mean per-frame allocation on the hot path.
    """

    tx, site, rx = drv.positions()
    first_in, first_out = _legs(multipath, tx, site, rx)
    second_in, second_out = _legs(multipath, tx, site, rx)
    for a, b in ((first_in, second_in), (first_out, second_out)):
        for name in (
            "source_id",
            "sink_id",
            "component_id",
            "primitive_sequence",
            "material_sequence",
            "interaction_type",
        ):
            assert (
                getattr(a, name).data_ptr() == getattr(b, name).data_ptr()
            ), name


# --------------------------------------------------------------------------
# The four combined paths
# --------------------------------------------------------------------------


def _legs(spike, tx, site, rx, *, ad_mode: str = "none"):
    inbound = spike.adapter.reevaluate(
        spike.inbound,
        spike._source(tx, geo.TX_STABLE_ID),
        spike._sink(site, geo.SITE_STABLE_ID),
        ad_mode=ad_mode,
    )
    outbound = spike.adapter.reevaluate(
        spike.outbound,
        spike._source(site, geo.SITE_STABLE_ID),
        spike._sink(rx, geo.RX_STABLE_ID),
        ad_mode=ad_mode,
    )
    return inbound, outbound


def _combined_key(multipath, composed, row: int) -> tuple[str, str]:
    """Name a composed row by the components it joined.

    Via ``inbound_row``/``outbound_row``, which is the only honest way: the two
    mixed paths are 20.15 ps apart and a sorted delay comparison would confuse
    them for each other.
    """

    inbound_names = _component_order(multipath.inbound)
    outbound_names = _component_order(multipath.outbound)
    return (
        inbound_names[int(composed.topology.inbound_row[row])],
        outbound_names[int(composed.topology.outbound_row[row])],
    )


def test_four_combined_paths_carry_the_analytic_round_trip_delays(multipath):
    tx, site, rx = drv.positions()
    composed, _, _ = multipath.paths(tx, site, rx, drv.make_response())

    assert composed.path_count == 4
    assert composed.sensor_pair_count == 1
    assert composed.pair_offsets.tolist() == [0, 4]
    assert composed.row_valid is not None and bool(composed.row_valid.all())

    expected = geo.combined_delays_s()
    seen = {}
    for row in range(composed.path_count):
        key = _combined_key(multipath, composed, row)
        assert key not in seen, f"{key} appeared twice"
        seen[key] = float(composed.total_delay_s[row])
        assert seen[key] == pytest.approx(expected[key], rel=DELAY_RTOL), key
    assert set(seen) == set(expected)

    # The trap, asserted so it can never be quietly designed away: the two
    # mixed paths are distinct but only 20 ps apart, so they are NOT separable
    # by delay alone and only the topology columns tell them apart.
    gap = abs(seen[("los", "reflection")] - seen[("reflection", "los")])
    assert 1.0e-11 < gap < 1.0e-10


def test_the_combined_doppler_shifts_match_the_analytic_projections(multipath):
    """Four combined paths, four distinct Doppler shifts, all closed form."""

    velocity = torch.tensor(
        [geo.SITE_VELOCITY_M_PER_S], dtype=torch.float32, device="cuda"
    )
    tx, site, rx = drv.positions()
    with forward_ad.dual_level():
        composed, inbound, outbound = multipath.paths(
            tx,
            forward_ad.make_dual(site, velocity),
            rx,
            drv.make_response(),
            ad_mode="jvp",
        )
        rate = composed.delay_rate.clone()
        keys = [
            _combined_key(multipath, composed, row)
            for row in range(composed.path_count)
        ]
        assert inbound.delay_rate is not None
        assert outbound.delay_rate is not None

    expected = geo.combined_doppler_hz()
    measured = {}
    for row, key in enumerate(keys):
        measured[key] = -geo.REFERENCE_FREQUENCY_HZ * float(rate[row])
        assert measured[key] == pytest.approx(expected[key], rel=1e-5), key
    # Every combined path recedes, and no two share a shift: the fixture is
    # actually resolving four paths, not one path counted four times.
    assert all(value < 0.0 for value in measured.values())
    shifts = sorted(measured.values())
    assert min(b - a for a, b in zip(shifts[:-1], shifts[1:], strict=True)) > 50.0


def test_a_dying_reflection_row_is_data_and_flows_into_the_join():
    """A reflection row that stops existing publishes zeros, not an error.

    At ``site_y = 5.0`` the inbound specular point leaves the authored facet
    (it sits at ``y = 2 * y_site / 3``, above the facet edge at ``y = 3``), so
    the frozen inbound reflection row dies. ADR-037 publishes that through
    ``row_valid`` as a complete answer. The composed rows that joined it must
    inherit that verdict and carry exactly zero, and the surviving rows must be
    unaffected.
    """

    spike = drv.Phase4Spike(components=MULTIPATH_COMPONENTS, max_depth=1)
    tx, _, rx = drv.positions()
    far = torch.tensor([(2.0, 5.0, 0.0)], dtype=torch.float32, device="cuda")
    composed, inbound, outbound = spike.paths(tx, far, rx, drv.make_response())

    assert inbound.row_valid.tolist() == [True, False]
    assert outbound.row_valid is not None
    dead = [
        row
        for row in range(composed.path_count)
        if not bool(composed.row_valid[row])
    ]
    # Both specular points leave the facet at this site - the outbound one at
    # y = 3.29, the inbound at y = 3.33, against a facet edge at y = 3 - so
    # three of the four combined paths die and only line-of-sight survives.
    assert len(dead) == 3
    for row in dead:
        assert "reflection" in _combined_key(spike, composed, row)
        assert float(composed.total_delay_s[row]) == 0.0
        assert complex(composed.complex_transfer_ref[row]) == 0j
    alive = [row for row in range(composed.path_count) if row not in dead]
    assert [_combined_key(spike, composed, row) for row in alive] == [
        ("los", "los")
    ]
    assert float(composed.total_delay_s[alive[0]]) > 0.0


def test_a_site_behind_the_wall_kills_every_row_and_the_frame_is_still_valid():
    spike = drv.Phase4Spike(components=MULTIPATH_COMPONENTS, max_depth=1)
    tx, _, rx = drv.positions()
    behind = torch.tensor([(6.0, 0.0, 0.0)], dtype=torch.float32, device="cuda")
    composed, _, _ = spike.paths(tx, behind, rx, drv.make_response())

    assert composed.path_count == 4
    assert not bool(composed.row_valid.any())
    assert float(composed.total_delay_s.abs().sum()) == 0.0
    assert float(composed.complex_transfer_ref.abs().sum()) == 0.0

    from witwin.radar.synthesis.fmcw_beat import synthesize_fmcw_beat

    iq = synthesize_fmcw_beat(composed, drv.make_spec(num_chirps=2))
    assert torch.count_nonzero(iq) == 0


# --------------------------------------------------------------------------
# Refusals that must not be swallowed
# --------------------------------------------------------------------------


def test_a_rough_scene_refuses_reflection_reevaluation_rather_than_smoothing_it():
    """The consumer's rough-scene refusal must reach the caller.

    Coherent rough-surface attenuation is owned by Channel's discovery field
    loop. Reevaluating a frozen reflection topology on a rough scene would
    silently disagree with ``evaluate``, so the consumer refuses. Catching that
    here and returning the smooth answer would be precisely the fallback the
    architecture forbids, so the adapter lets it propagate.
    """

    compiled = world.compile_fixture_scene(rough=True)
    with pytest.raises(NotImplementedError, match="smooth scene"):
        spike = drv.Phase4Spike(
            components=MULTIPATH_COMPONENTS, max_depth=1, compiled=compiled
        )
        tx, site, rx = drv.positions()
        spike.paths(tx, site, rx, drv.make_response())


def test_polarimetric_multipath_discovery_is_still_refused_by_the_consumer():
    """A deferral, pinned so it cannot be mistaken for support.

    The consumer allows polarimetric transport for line of sight only during
    DISCOVERY. Reevaluating an already-prepared reflection topology under
    ``polarimetric_transport`` does work, so supporting polarimetric multipath
    means a two-response split in the adapter (scalar to discover, polarimetric
    to reevaluate, and a polarization basis on both leg batches). That is
    deferred; this asserts the boundary rather than leaving it undocumented.
    """

    from witwin.channel.propagation import consumer

    capabilities = consumer.capabilities()
    allowed = capabilities.components_for("polarimetric_transport")
    assert "los" in allowed
    assert "reflection" not in allowed
    # Both halves of the deferral: discovery refuses it, fixed-topology
    # reevaluation would accept it.
    assert "reflection" in capabilities.fixed_topology_components
    assert "polarimetric_transport" in capabilities.fixed_topology_responses
