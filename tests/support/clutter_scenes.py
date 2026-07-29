"""Target-plus-clutter fixtures, built on the existing multi-endpoint world.

Nothing new is authored here. The Phase-6 multi-endpoint fixture already
contains both classes in ONE frozen topology - line-of-sight rows through the
scatter sites and single-bounce rows off the wall - with real per-pair row-count
divergence, two genuinely empty pair segments, and rows that die on demand. What
this module adds is the DECLARATION that turns those rows into named
components, and the three scene mobilities the plan asks for:

* :func:`static_clutter_scene` - a wall with no trajectory, so the epoch loop
  never recompiles and clutter is free;
* :func:`replayed_clutter_scene` - the wall translating along its own normal,
  driven under ``world_motion="fixed_winner_replay"``;
* :func:`path_gaining_clutter_scene` - the wall parked out of the way and
  arriving, so a clutter row COMES INTO EXISTENCE. That is the one invalidation
  class a replay cannot report, and the fixture exists so the limitation can be
  demonstrated rather than asserted away.

The wall is declared clutter by its COMPILED MATERIAL SLOT, which is what a
frozen leg row carries; the two sites are declared targets by their stable
world IDs.
"""

from __future__ import annotations

from . import multi_endpoint_driver as drv
from . import multi_endpoint_geometry as geo
from . import multi_endpoint_world as world


#: The wall arriving from where it was parked. Same value as the Phase-7
#: invalidation fixture, which is the point: the scene that births a row is the
#: same scene, seen through the component taxonomy.
ARRIVING_WALL_VELOCITY = (0.0, -2.0, 0.0)


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
        spike.composer,
        spike.inbound,
        spike.outbound,
        declaration() if decl is None else decl,
    )


def direct_route(spike, decl=None):
    """The transmitter-to-receiver route of the same world, frozen and indexed.

    Four rows on this fixture: three lines of sight and ONE
    transmitter-to-wall-to-receiver reflection. That fourth row is the reason
    ``direct_leakage`` is "no site and no declared clutter interaction" rather
    than "no site": it has no scatter site and it is unmistakably the
    environment's return.

    Returns ``(composer, leg_handle, index)``.
    """

    from witwin.radar.paths import DirectComposer, RadarComponentIndex

    leg = spike.adapter.freeze(
        world.endpoint_batch(
            [position for _, position in spike.transmitters],
            spike.transmitter_ids,
            power_w=geo.TX_POWER_W,
            device=spike.device,
        ),
        world.endpoint_batch(
            [position for _, position in spike.receivers],
            spike.receiver_ids,
            device=spike.device,
        ),
    )
    composer = DirectComposer.freeze(
        leg,
        radar_source_ids=list(spike.transmitter_ids),
        radar_sink_ids=list(spike.receiver_ids),
        reference_frequency_hz=geo.REFERENCE_FREQUENCY_HZ,
    )
    index = RadarComponentIndex.from_direct(
        composer, leg, declaration() if decl is None else decl
    )
    return composer, leg, index


def static_clutter_scene():
    """A wall that does not move: no trajectory, no deformation."""

    from witwin.core.dynamics import DynamicScene

    scene, mesh = world.make_scene()
    world.assert_world_coordinates_survived(mesh)
    return DynamicScene(scene)


def replayed_clutter_scene():
    """A wall translating along its own normal at 4 m/s."""

    return world.make_dynamic_scene(wall_velocity=geo.WALL_VELOCITY_M_PER_S)


def path_gaining_clutter_scene():
    """A wall parked clear of the geometry, arriving over one second.

    At ``t = 0`` the facet spans ``y in [0.8, 3.2]`` and the ``TX_A -> SITE_P``
    specular point falls outside it, so that reflection does not exist. By
    ``t = 1`` the wall is back at the origin and it does - a BORN clutter row,
    which a fixed-winner replay cannot report at all.
    """

    return world.make_dynamic_scene(
        wall_origin=geo.WALL_PARKED_OFFSET_M, wall_velocity=ARRIVING_WALL_VELOCITY
    )


class ClutterEpochState:
    """The caller's per-epoch state, rebuilt by the loop's ``bind``."""

    def __init__(self, decl=None):
        self.decl = declaration() if decl is None else decl
        self.spike = None
        self.index = None
        self.binds = 0


def clutter_epoch_loop(dynamic, policy, *, compile_scene=None, decl=None):
    """A :class:`SceneEpochLoop` configured by a component declaration.

    ``policy`` is the :class:`~witwin.radar.propagation.EpochPolicy` the
    caller resolved from its :class:`ClutterComponentSpec` set. Passing the
    resolved policy rather than the specs keeps the loop unaware of the
    component vocabulary, which is the layering the production modules declare.
    """

    from witwin.radar.propagation import FrozenEpoch, SceneEpochLoop

    state = ClutterEpochState(decl)

    def bind(compiled, snapshot, previous):
        del snapshot
        state.binds += 1
        state.spike = drv.MultiEndpointSpike(
            compiled=compiled,
            adapter=None if previous is None else previous.adapter,
        )
        state.index = component_index(state.spike, state.decl)
        return FrozenEpoch(
            adapter=state.spike.adapter,
            handles=(state.spike.inbound, state.spike.outbound),
            payload=state.spike,
        )

    loop = SceneEpochLoop(
        dynamic,
        reference_frequency_hz=geo.REFERENCE_FREQUENCY_HZ,
        bind=bind,
        compile_scene=compile_scene or world.compile_snapshot,
        motion_event_period_frames=policy.motion_event_period_frames,
        world_motion=policy.world_motion,
    )
    return loop, state


__all__ = [
    "ARRIVING_WALL_VELOCITY",
    "ClutterEpochState",
    "clutter_epoch_loop",
    "component_index",
    "declaration",
    "direct_route",
    "path_gaining_clutter_scene",
    "replayed_clutter_scene",
    "static_clutter_scene",
]
