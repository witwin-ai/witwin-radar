"""The target-plus-clutter declaration over the multi-endpoint world.

The multi-endpoint fixture already contains both classes in ONE frozen
topology - line-of-sight rows through the scatter sites and single-bounce rows
off the wall - with real per-pair row-count divergence, two genuinely empty
pair segments, and rows that die on demand. What this module adds is the
DECLARATION that turns those rows into named components.

The wall is declared clutter by its COMPILED MATERIAL SLOT, which is what a
frozen leg row carries; the two sites are declared targets by their stable
world IDs.
"""

from __future__ import annotations

from . import multi_endpoint_geometry as geo


def declaration():
    """Sites P and Q are targets; the wall's material slot is clutter."""

    from witwin.radar.paths import ComponentDeclaration

    return ComponentDeclaration(
        target_site_ids={geo.SITE_P_STABLE_ID, geo.SITE_Q_STABLE_ID},
        clutter_material_slots={geo.REFLECTION_MATERIAL_SLOT},
    )


def component_index(spike, decl=None):
    """The sidecar index for a two-way spike, built once from its frozen legs."""

    from witwin.radar.paths import RadarComponentIndex

    return RadarComponentIndex.from_two_way(
        spike.composer, spike.inbound, spike.outbound, declaration() if decl is None else decl
    )


__all__ = ["component_index", "declaration"]
