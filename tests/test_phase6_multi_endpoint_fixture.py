"""The real multi-endpoint fixture: what Channel actually publishes for it.

Phase 4 and Phase 5 exercise the Channel consumer at exactly one TX, one site
and one RX. At that shape every pair has the same row count, no pair is empty,
and the endpoint batch order trivially agrees with the stable-ID order, so the
multi-pair half of the join could only ever be tested against fabricated legs.

This file pins the REAL legs the multi-endpoint geometry produces, against the
float64 image-source closed forms in ``support.multi_endpoint_geometry``. Every
shape the join tests in ``test_phase6_multi_endpoint_join`` depend on -
differing per-pair row counts, an empty pair segment, a reflection row on the
second triangle - is asserted here first, so that a fixture drift fails as a
fixture failure rather than surfacing as an apparent Channel bug one file over.

The oracle is independent: it derives which rows exist from facet containment
and image-source geometry, and it never reads anything Channel published.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("witwin.channel")

from support import multi_endpoint_driver as drv  # noqa: E402
from support import multi_endpoint_geometry as geo  # noqa: E402
from support import multi_endpoint_world as world  # noqa: E402

pytestmark = pytest.mark.gpu

# A nanosecond-scale delay held in float32 carries about 1e-7 relative
# precision, so this is tight rather than generous.
DELAY_RTOL = 1.0e-6


@pytest.fixture(scope="module")
def spike():
    return drv.MultiEndpointSpike()


def _columns(frozen):
    return {
        "source_id": frozen.source_id.tolist(),
        "sink_id": frozen.sink_id.tolist(),
        "component_id": frozen.component_id.tolist(),
        "depth": frozen.depth.tolist(),
        "primitive_sequence": frozen.primitive_sequence.tolist(),
        "material_sequence": frozen.material_sequence.tolist(),
    }


def _predicted_columns(rows):
    return {
        "source_id": [row.source_id for row in rows],
        "sink_id": [row.sink_id for row in rows],
        "component_id": [row.component_id for row in rows],
        "depth": [row.depth for row in rows],
        "primitive_sequence": [[row.primitive] for row in rows],
        "material_sequence": [[row.material] for row in rows],
    }


# --------------------------------------------------------------------------
# The fixture world itself
# --------------------------------------------------------------------------


def test_the_authored_wall_survives_mesh_construction():
    """``Mesh`` recentres by default and would silently move the design knob.

    The plane at ``x = 4`` sets every delay; the half-width ``y = 1.2`` is what
    decides which reflection rows exist at all. A recentred mesh keeps neither
    and raises nothing, so both are re-asserted rather than trusted.
    """

    _, mesh = world.make_scene()
    world.assert_world_coordinates_survived(mesh)
    # The vertices are float32, so 1.2 is held as 1.20000004768; the tolerance
    # is a float32 representation bound, not a slack budget. A recentred mesh
    # would move the wall by metres.
    vertices = mesh.vertices.detach().to(torch.float64).cpu()
    assert float(vertices[:, 0].min()) == pytest.approx(geo.WALL_PLANE_X_M, abs=1e-6)
    assert float(vertices[:, 0].max()) == pytest.approx(geo.WALL_PLANE_X_M, abs=1e-6)
    assert float(vertices[:, 1].max()) == pytest.approx(geo.WALL_HALF_Y_M, abs=1e-6)
    assert float(vertices[:, 1].min()) == pytest.approx(-geo.WALL_HALF_Y_M, abs=1e-6)


def test_the_specular_points_are_where_the_image_source_puts_them():
    """The eight-pair facet-containment table, solver-independent.

    This is the fixture's design argument written as an assertion: WHICH pairs
    have a reflection row, and by how much the ones that do not miss the facet.
    Widening the wall would make the fixture run and quietly stop testing
    anything, and this is where that fails.
    """

    expected = {
        # (source, sink): (specular y, is on the facet)
        (geo.TX_A_STABLE_ID, geo.SITE_P_STABLE_ID): (0.4, True),
        (geo.TX_A_STABLE_ID, geo.SITE_Q_STABLE_ID): (1.6, False),
        (geo.SITE_P_STABLE_ID, geo.RX_A_STABLE_ID): (0.394871794871795, True),
        (geo.SITE_Q_STABLE_ID, geo.RX_A_STABLE_ID): (1.5794871794871796, False),
        (geo.SITE_P_STABLE_ID, geo.RX_B_STABLE_ID): (-0.6307692307692307, True),
        (geo.SITE_Q_STABLE_ID, geo.RX_B_STABLE_ID): (0.5538461538461539, True),
    }
    positions = dict((*geo.TRANSMITTERS, *geo.SITES, *geo.RECEIVERS))
    margins = []
    for (source, sink), (y, on_facet) in expected.items():
        point = geo.specular_point_m(positions[source], positions[sink])
        assert point is not None, (source, sink)
        assert point[0] == pytest.approx(geo.WALL_PLANE_X_M, abs=1e-12)
        assert point[1] == pytest.approx(y, abs=1e-12)
        assert point[2] == pytest.approx(0.0, abs=1e-12)
        assert (geo.face_containing(point) is not None) is on_facet, (source, sink)
        margins.append(abs(abs(y) - geo.WALL_HALF_Y_M))

    # ``TX_B``'s image shares the plane x = 2 with both sites, so there is no
    # specular point to test for containment: the path does not exist at any
    # facet size. That is a different absence from a facet miss and the fixture
    # relies on both.
    for site, _ in geo.SITES:
        assert geo.specular_point_m(geo.TX_B_POSITION_M, positions[site]) is None

    # No float32 knife edges: the closest call is 0.379 m from the facet edge,
    # nine orders of magnitude above a float32 ULP at these coordinates.
    assert min(margins) > 0.37


def test_the_wall_blocks_exactly_the_two_transmitter_b_lines_of_sight():
    """Over the LEG pairs, which are the only ones the solver is asked about.

    Both of ``TX_B``'s lines to the sites cross ``x = 4`` inside the facet, at
    ``y = -0.2`` and ``y = +0.7``. Every other leg pair has both endpoints on
    the same side of the plane and is never blocked.
    """

    legs = [(source, sink) for source in geo.TRANSMITTERS for sink in geo.SITES] + [
        (source, sink) for source in geo.SITES for sink in geo.RECEIVERS
    ]
    blocked = {(source[0], sink[0]) for source, sink in legs if geo.line_of_sight_is_blocked(source[1], sink[1])}
    assert blocked == {(geo.TX_B_STABLE_ID, geo.SITE_P_STABLE_ID), (geo.TX_B_STABLE_ID, geo.SITE_Q_STABLE_ID)}


# --------------------------------------------------------------------------
# The legs Channel publishes
# --------------------------------------------------------------------------


def test_the_legs_carry_the_row_counts_and_identity_the_geometry_predicts(spike):
    """Three inbound rows and seven outbound rows, column for column.

    The inbound leg is the interesting one: pairs 0 and 2 carry two and one row
    respectively and pairs 1 and 3 carry NONE, because ``TX_B`` is occluded from
    both sites and has no image-source path to either. Differing per-pair counts
    and a genuinely empty pair segment are what the one-site fixture cannot
    produce.
    """

    assert spike.inbound.row_count == 3
    assert spike.outbound.row_count == 7
    assert spike.inbound.components == ("los", "reflection")
    assert spike.outbound.components == ("los", "reflection")
    assert _columns(spike.inbound) == _predicted_columns(spike.predicted_inbound_rows())
    assert _columns(spike.outbound) == _predicted_columns(spike.predicted_outbound_rows())
    # Spelled out, because these exact lists are what the join tests build on.
    assert spike.inbound.source_id.tolist() == [10, 10, 10]
    assert spike.inbound.sink_id.tolist() == [20, 20, 21]
    assert spike.outbound.sink_id.tolist() == [30, 30, 30, 31, 31, 31, 31]
    # ``TX_B`` (stable ID 11) reaches nothing at all.
    assert geo.TX_B_STABLE_ID not in spike.inbound.source_id.tolist()


def test_one_reflection_row_lands_on_the_second_triangle(spike):
    """The identity key exercises more than ``component_id``.

    ``P -> RX_B`` reflects at ``y = -0.63``, on the far side of the two
    triangles' shared diagonal, so its ADR-037 frozen ``primitive_sequence`` is
    ``[1]`` while every other reflection row in the fixture carries ``[0]``. The
    join keys on ``(component, depth, primitive, material)``; before this
    fixture nothing but ``component`` ever varied, so three quarters of the key
    were untested against real rows.
    """

    primitives = spike.outbound.primitive_sequence.tolist()
    components = spike.outbound.component_id.tolist()
    reflections = [
        primitives[row][0] for row in range(spike.outbound.row_count) if components[row] == geo.REFLECTION_COMPONENT_ID
    ]
    assert sorted(reflections) == [0, 0, 1]
    assert len(set(reflections)) == 2


def test_the_channel_leg_publishes_an_empty_pair_segment_of_its_own(spike):
    """Before the join is involved at all.

    ``pair_offsets`` comes straight from Channel's own sink-major partition over
    ``source_count * sink_count``, and it has zero-length segments at pairs 1
    and 3. The join's empty-segment rule is downstream of this; both are real.
    """

    inbound, outbound = spike.legs()
    assert inbound.pair_count == 4
    assert inbound.pair_offsets.tolist() == [0, 2, 2, 3, 3]
    assert inbound.pair_index.tolist() == [0, 0, 2]
    assert outbound.pair_count == 4
    assert outbound.pair_offsets.tolist() == [0, 2, 3, 5, 7]
    assert outbound.pair_index.tolist() == [0, 0, 1, 2, 2, 3, 3]
    assert inbound.pair_offsets.tolist() == geo.pair_offsets(spike.predicted_inbound_rows(), 4)
    assert outbound.pair_offsets.tolist() == geo.pair_offsets(spike.predicted_outbound_rows(), 4)


def test_leg_delays_match_the_image_source_closed_form(spike):
    inbound, outbound = spike.legs()
    for name, batch, rows in (
        ("inbound", inbound, spike.predicted_inbound_rows()),
        ("outbound", outbound, spike.predicted_outbound_rows()),
    ):
        assert batch.leg_count == len(rows), name
        assert bool(batch.row_valid.all()), name
        for index, row in enumerate(rows):
            assert float(batch.delay_s[index]) == pytest.approx(row.delay_s, rel=DELAY_RTOL), (
                name,
                index,
                row.component,
            )
        # Not vacuous: every row carries a nonzero coefficient.
        assert float(batch.coefficient.abs().min()) > 0.0, name


def test_the_occlusion_is_load_bearing_rather_than_a_broken_endpoint():
    """Move ``TX_B`` past the facet edge and its rows come back.

    The empty pair segments are the whole point of this fixture, so "``TX_B``
    publishes nothing" must be the geometry talking and not an endpoint that
    silently failed for an unrelated reason. At ``(6, 4, 0)`` its lines to both
    sites cross ``x = 4`` outside the facet and two line-of-sight rows appear.
    """

    moved = ((geo.TX_A_STABLE_ID, geo.TX_A_POSITION_M), (geo.TX_B_STABLE_ID, geo.TX_B_UNOCCLUDED_POSITION_M))
    unoccluded = drv.MultiEndpointSpike(transmitters=moved)
    assert unoccluded.inbound.row_count == 5
    assert unoccluded.inbound.source_id.tolist() == [10, 10, 11, 10, 11]
    assert unoccluded.inbound.sink_id.tolist() == [20, 20, 20, 21, 21]
    assert _columns(unoccluded.inbound) == _predicted_columns(unoccluded.predicted_inbound_rows())
    # And no pair is empty any more, which is the control for the empty-segment
    # assertions above.
    _, _ = unoccluded.legs()
    inbound, _ = unoccluded.legs()
    assert inbound.pair_offsets.tolist() == [0, 2, 3, 4, 5]


# --------------------------------------------------------------------------
# The frozen handles and what they cost
# --------------------------------------------------------------------------


def test_the_frozen_row_identity_is_the_same_storage_on_every_frame(spike):
    """Identity must not be rebuilt per frame, at three and seven rows either.

    Asserting storage identity rather than equality is what makes "the join is
    frozen" a measured property: an equal-but-new tensor would mean per-frame
    allocation on the hot path, and with a wider leg it would be a larger one.
    """

    first_in, first_out = spike.legs()
    second_in, second_out = spike.legs()
    for a, b in ((first_in, second_in), (first_out, second_out)):
        for name in (
            "source_id",
            "sink_id",
            "component_id",
            "primitive_sequence",
            "material_sequence",
            "interaction_type",
        ):
            assert getattr(a, name).data_ptr() == getattr(b, name).data_ptr(), name


def test_freezing_a_multi_endpoint_leg_costs_what_a_one_pair_leg_costs(spike):
    """Four copies, thirty-three bytes, four synchronizations - per FREEZE.

    Phase 5 measured the same three numbers on a two-row leg. These legs carry
    three and seven rows across four endpoint pairs and report the identical
    budget, so the preparation cost is a property of the frozen topology rather
    than of its cardinality. Asserting it as a comparison against a real 1x1x1
    leg on the same scene is what makes that a measurement instead of two
    numbers that happen to match.
    """

    single = spike.single_pair(geo.TRANSMITTERS[0], geo.SITES[0], geo.RECEIVERS[0])
    for frozen in (spike.inbound, spike.outbound, single.inbound, single.outbound):
        assert frozen.prepare_d2h_copies == 4
        assert frozen.prepare_d2h_bytes == 33
        assert frozen.prepare_synchronizations == 4
    assert spike.outbound.row_count == 7
    assert single.outbound.row_count == 2


def test_a_multi_endpoint_frame_costs_exactly_two_host_observations(spike, monkeypatch):
    """Eleven composed rows over four sensor pairs, and nothing crosses back.

    One validation copy per leg, exactly as at one row per leg. The empty pair
    segments in particular are resolved at freeze time, so they cost no per-frame
    observation to skip.
    """

    response = drv.make_response()
    spike.frame(response=response)  # resolve the operator table first

    counts = dict.fromkeys(("item", "cpu", "tolist", "numpy", "synchronize"), 0)
    for name in ("item", "cpu", "tolist", "numpy"):
        original = getattr(torch.Tensor, name)

        def observing(self, *args, _name=name, _original=original, **kwargs):
            counts[_name] += 1
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, name, observing)

    original_sync = torch.cuda.synchronize

    def counting_sync(*args, **kwargs):
        counts["synchronize"] += 1
        return original_sync(*args, **kwargs)

    monkeypatch.setattr(torch.cuda, "synchronize", counting_sync)

    composed, _, _ = spike.frame(response=response)
    assert composed.path_count == 11
    assert composed.sensor_pair_count == 4
    assert counts == {"item": 2, "cpu": 0, "tolist": 0, "numpy": 0, "synchronize": 0}, counts
