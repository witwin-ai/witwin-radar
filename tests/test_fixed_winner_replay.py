"""``Motion(world="fixed_winner_replay")`` on moving reflectors (M2-10 / D-11, D-12).

Fixed-winner replay holds the discovered interaction sequences while the
geometry moves, so it recompiles at every observation and discovers only on
its declared cadence. Two consequences, both pinned here: on a flat wall the
held winner stays the true winner and the replay matches per-observation
rediscovery; on a TESSELLATED CURVED reflector the specular point slides off
its frozen facet, the row dies, and the neighbouring facet's reflection is not
born until the next discovery - a hole in the return that the event log and
the birth lag bound make visible rather than hiding.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from support import oracle_scenes as sc

from witwin.radar import Fmcw, Motion, PointTargets

pytestmark = pytest.mark.gpu

CHIRPS = 32


def _radar():
    waveform = Fmcw(
        slope=30e12,
        sample_rate=5e6,
        samples_per_chirp=8,
        chirps_per_frame=CHIRPS,
        adc_start=0.0,
        idle=1e-3,
        ramp_end=8 / 5e6 + 1e-6,
        output="beat",
    )
    return sc.radar(waveform)


def _moving(scene, velocity):
    from witwin.core.dynamics import DynamicScene, LinearTrajectory

    trajectory = LinearTrajectory(origin=torch.zeros(3), velocity=torch.tensor(velocity))
    return DynamicScene(scene, structure_trajectories={1: trajectory})


def _wall():
    from witwin.core import PhysicalMaterial

    corners = ((6.0, -3.0, -3.0), (6.0, 3.0, -3.0), (6.0, 3.0, 3.0), (6.0, -3.0, 3.0))
    return sc.plane_scene(corners=corners, material=PhysicalMaterial.perfect_conductor())


def _cylinder(facets: int = 12, radius: float = 1.0, centre=(6.0, 0.0, 0.0)):
    """A vertical cylinder of ``facets`` flat panels: a curved reflector, tessellated."""

    from witwin.core import PhysicalMaterial

    angles = 2.0 * math.pi * (np.arange(facets) + 0.5) / facets
    ring = [(centre[0] + radius * math.cos(a), centre[1] + radius * math.sin(a)) for a in angles]
    corners = [(x, y, -2.0) for x, y in ring] + [(x, y, 2.0) for x, y in ring]
    faces = []
    for index in range(facets):
        a, b = index, (index + 1) % facets
        faces += [(a, b, b + facets), (a, b + facets, a + facets)]
    return sc.plane_scene(corners=corners, material=PhysicalMaterial.perfect_conductor(), faces=tuple(faces))


def _simulate(scene, world: str, *, cadence: int | None):
    radar = _radar()
    motion = Motion.chirp(world=world, rediscover_every_frames=cadence)
    return radar.simulate(
        scene,
        PointTargets(positions=[[3.0, 0.2, 0.0]], rcs=1.0, ids=(5,)),
        times=(0.0,),
        los=True,
        reflections=1,
        motion=motion,
    )


def test_fixed_winner_replay_tracks_a_translating_flat_wall():
    """The held winners stay the winners, so replay matches rediscovery at a fraction of the discoveries."""

    held = _simulate(_moving(_wall(), (2.0, 0.0, 0.0)), "fixed_winner_replay", cadence=1)
    fresh = _simulate(_moving(_wall(), (2.0, 0.0, 0.0)), "frozen_world", cadence=None)
    assert held.compile_count == fresh.compile_count == CHIRPS
    assert held.discovery_count == 1 < fresh.discovery_count == CHIRPS
    assert len(held.events[0]) == len(fresh.events[0]) == 0
    torch.testing.assert_close(held.cube, fresh.cube, rtol=1e-4, atol=0)


def test_fixed_winner_replay_drops_a_curved_reflector_between_discoveries():
    """D-11, documented: the frozen facet's reflection dies and its successor waits for the next discovery.

    Translating the faceted cylinder along ``y`` walks the specular point
    across its facets. Rediscovering at every observation hands the
    reflection from facet to facet (each death is followed by a birth inside
    the frame); fixed-winner replay with one discovery per frame logs the
    deaths and no birth, and its lag bound is the whole frame.
    """

    scene = _moving(_cylinder(), (0.0, 20.0, 0.0))
    held = _simulate(scene, "fixed_winner_replay", cadence=1)
    fresh = _simulate(_moving(_cylinder(), (0.0, 20.0, 0.0)), "frozen_world", cadence=None)
    held_events, fresh_events = held.events[0], fresh.events[0]
    assert (~held_events.birth).sum() >= 1 and held_events.birth.sum() == 0
    assert fresh_events.birth.sum() >= 1
    assert (held_events.depth > 0).all()
    span = float(held.sample_times_s[0][-1] - held.sample_times_s[0][0])
    assert held.birth_lag_bound_s[0] == pytest.approx(span, rel=1e-9)
    assert fresh.birth_lag_bound_s[0] < span / 10
