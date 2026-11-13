"""Which reflection rows exist near a wall, at discovery and at replay.

Channel's consumer decides whether a specular reflection exists; Radar only
composes the rows it is given. Two of its validity rules decide rows a radar
scene reaches easily (Channel ADR-059 and ADR-068):

* a path segment shorter than ``1e-5`` of the scene diagonal is not a path,
  EXCEPT a segment that ends at an endpoint (Channel 0.6.0, C-12): a scatter
  site on or next to a wall keeps its reflection off that wall, and discovery
  and fixed-topology replay apply the same rule (C-16);
* the specular point must lie on the face to within a few float roundings of
  its coordinates, so a point just past the edge of a wall is not a reflection.

Geometry: the fixture wall is the plane ``x = 4`` for ``|y| <= 1.2`` and
``|z| <= 3`` m, the transmitter is at the origin. Its image is ``(8, 0, 0)``,
so a site ``(2, y, 0)`` reflects at ``(4, 2y/3, 0)`` and the specular point is
past the ``y = 1.2`` edge exactly when ``y > 1.8``. A site at ``(4 - d, 0.6, 0)``
reflects at about ``(4, 0.6, 0)`` and its last segment is ``d * 1.011`` m long.
The wall's bounds are ``0 x 2.4 x 6`` m, a diagonal of 6.46 m, so the minimum
segment is 64.6 um.
"""

from __future__ import annotations

import math

import pytest

pytest.importorskip("witwin.channel")

from support import multi_endpoint_driver as drv
from support import multi_endpoint_geometry as geo

pytestmark = pytest.mark.gpu

TRANSMITTER = ((geo.TX_A_STABLE_ID, geo.TX_A_POSITION_M),)
RECEIVER = ((geo.RX_A_STABLE_ID, geo.RX_A_POSITION_M),)
REFLECTION = (geo.TX_A_STABLE_ID, geo.SITE_P_STABLE_ID, geo.REFLECTION_COMPONENT_ID)
LINE_OF_SIGHT = (geo.TX_A_STABLE_ID, geo.SITE_P_STABLE_ID, geo.LOS_COMPONENT_ID)

#: Channel's minimum segment, a fixed fraction of the compiled geometry's diagonal.
MIN_SEGMENT_M = 1.0e-5 * math.sqrt((2 * geo.WALL_HALF_Y_M) ** 2 + (2 * geo.WALL_HALF_Z_M) ** 2)

#: Sites whose specular point sits ``offset`` metres past (positive) or inside
#: (negative) the wall's ``y = 1.2`` edge.
EDGE_SITE_Y_PER_M = 1.5


def _near_wall(distance_m: float) -> tuple[float, float, float]:
    return (geo.WALL_PLANE_X_M - distance_m, 0.6, 0.0)


def _past_edge(offset_m: float) -> tuple[float, float, float]:
    return (2.0, EDGE_SITE_Y_PER_M * (geo.WALL_HALF_Y_M + offset_m), 0.0)


def _spike(site):
    return drv.MultiEndpointSpike(transmitters=TRANSMITTER, sites=((geo.SITE_P_STABLE_ID, site),), receivers=RECEIVER)


def _inbound_rows(spike, site) -> dict[tuple[int, int, int], bool]:
    """Row identity -> ``row_valid`` of the inbound leg replayed at ``site``."""

    inbound, _ = spike.legs(spike.site_tensor([site]))
    identities = zip(inbound.source_id.tolist(), inbound.sink_id.tolist(), inbound.component_id.tolist(), strict=True)
    return dict(zip(identities, inbound.row_valid.tolist(), strict=True))


def _discovered(site) -> dict[tuple[int, int, int], bool]:
    return _inbound_rows(_spike(site), site)


#: Site-to-wall distances on both sides of the minimum segment, down to the wall itself.
NEAR_WALL_DISTANCES_M = (1.0e-2, 1.0e-4, 5.0e-5, 1.0e-5, 0.0)


def test_c12_a_site_closer_to_a_wall_than_the_minimum_segment_keeps_its_reflection():
    assert 1.011 * 5.0e-5 < MIN_SEGMENT_M < 1.0e-4

    for distance_m in NEAR_WALL_DISTANCES_M:
        assert _discovered(_near_wall(distance_m)) == {LINE_OF_SIGHT: True, REFLECTION: True}, distance_m


def test_c12_a_specular_point_just_past_the_wall_edge_is_not_a_reflection():
    assert _discovered(_past_edge(-1.0e-4)) == {LINE_OF_SIGHT: True, REFLECTION: True}
    for offset_m in (1.0e-5, 1.0e-4, 1.0e-3):
        assert _discovered(_past_edge(offset_m)) == {LINE_OF_SIGHT: True}, offset_m


def test_c16_replay_and_discovery_agree_at_the_wall_edge():
    """A row frozen inside the edge is invalid on replay exactly where discovery has no row."""

    spike = _spike(_past_edge(-1.0e-3))
    for offset_m in (-1.0e-3, -1.0e-4):
        assert _inbound_rows(spike, _past_edge(offset_m)) == {LINE_OF_SIGHT: True, REFLECTION: True}, offset_m
    for offset_m in (1.0e-5, 1.0e-4, 1.0e-3):
        assert _inbound_rows(spike, _past_edge(offset_m)) == {LINE_OF_SIGHT: True, REFLECTION: False}, offset_m


def test_c16_replay_and_discovery_agree_for_a_site_near_a_wall():
    """A reflection frozen 1 cm from the wall replays valid wherever discovery finds it."""

    spike = _spike(_near_wall(1.0e-2))
    for distance_m in NEAR_WALL_DISTANCES_M:
        assert _inbound_rows(spike, _near_wall(distance_m)) == _discovered(_near_wall(distance_m)), distance_m
