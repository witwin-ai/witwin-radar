"""Invalidation acceptance: a frozen topology either answers or refuses.

Plan work item 7, from the consuming side. The rule the plan states is short and
absolute - an invalidated fixed topology must fail loudly or be explicitly
rediscovered, and must never return a stale answer, an errored primal, or a
detached gradient - and it has three distinct failure classes, which are the
three groups below.

1. **The world moved and the caller did not notice.** Refused before any native
   work by Channel's world provenance, and refused by the adapter's own epoch
   for the narrower class the version domains cannot see.
2. **A row stopped existing.** Not a failure: ``row_valid=False`` with an
   exactly zero payload is the complete answer, and under ``ad_mode="jvp"`` its
   tangent is exactly zero too. What must never happen is a row that is valid
   and silently tangent-free.
3. **A row started existing.** The one class replay cannot report at all. It is
   pinned here as a measured limitation and its mitigation - the epoch loop's
   declared motion-event cadence - is pinned recovering the born row.

The cadence itself is priced in the last group: compile, discover and prepare
are motion-event work, one batched replay is frame work, and the difference
between them is the whole reason the fixed-topology capability exists.
"""

from __future__ import annotations

import time

import pytest
import torch

pytest.importorskip("witwin.channel")

from support import multi_endpoint_driver as drv  # noqa: E402
from support import multi_endpoint_geometry as geo  # noqa: E402
from support import multi_endpoint_world as world  # noqa: E402
from witwin.radar.propagation import kinematics as kin  # noqa: E402
from witwin.radar.propagation.epochs import (  # noqa: E402
    FIRST_FRAME,
    MOTION_EVENT_CADENCE,
    STRUCTURE_MOTION,
    FrozenEpoch,
    SceneEpochLoop,
)


pytestmark = pytest.mark.gpu

#: The wall parked out of the way at ``t = 0`` and back at the origin at
#: ``t = 1``: two lines of sight die, one reflection dies and one is BORN.
ARRIVING_WALL_VELOCITY = (0.0, -2.0, 0.0)

RADIAL = geo.SITE_P_RADIAL_VELOCITY_M_PER_S


def _arriving_wall():
    return world.make_dynamic_scene(
        wall_origin=geo.WALL_PARKED_OFFSET_M,
        wall_velocity=ARRIVING_WALL_VELOCITY,
    )


def _transmitter_spec(spike):
    return spike._stacked_ids(
        spike.stacked([p for _, p in spike.transmitters], 1),
        spike.transmitter_ids,
        geo.TX_POWER_W,
    )


def _site_spec(spike, sites=None):
    return spike._stacked_ids(
        spike.site_tensor() if sites is None else sites, spike.site_ids, None
    )


def _inbound(spike, sites=None, *, ad_mode="none"):
    return spike.adapter.reevaluate(
        spike.inbound,
        _transmitter_spec(spike),
        _site_spec(spike, sites),
        ad_mode=ad_mode,
    )


def _row_identities(leg) -> list[tuple[int, int, int]]:
    return list(
        zip(
            leg.source_id.tolist(),
            leg.sink_id.tolist(),
            leg.component_id.tolist(),
            strict=True,
        )
    )


class _HostObservations:
    """Count every route by which a device value could reach the host."""

    def __init__(self, monkeypatch):
        self.counts = dict.fromkeys(
            ("item", "cpu", "tolist", "numpy", "synchronize"), 0
        )
        for name in ("item", "cpu", "tolist", "numpy"):
            original = getattr(torch.Tensor, name)

            def observing(tensor, *args, _name=name, _original=original, **kwargs):
                self.counts[_name] += 1
                return _original(tensor, *args, **kwargs)

            monkeypatch.setattr(torch.Tensor, name, observing)
        original_sync = torch.cuda.synchronize

        def counting_sync(*args, **kwargs):
            self.counts["synchronize"] += 1
            return original_sync(*args, **kwargs)

        monkeypatch.setattr(torch.cuda, "synchronize", counting_sync)


# --------------------------------------------------------------------------
# 1. The world moved
# --------------------------------------------------------------------------


def test_a_stale_compiled_scene_never_answers(monkeypatch):
    """A topology frozen on an old wall is refused, not replayed.

    Two adapters, because there are two distinct refusals and only one of them
    belongs to Channel.

    The first is Channel's, reached by handing a FRESH adapter on the new scene
    a handle frozen on the old one. Both are at epoch 0, so the adapter's own
    check passes and the world-provenance check is the one that speaks. It
    names ``geometry_version`` and it names the remedy.

    The second is the adapter's, after a default ``refreeze``. It also names
    ``geometry_version``, because it asks Channel which domain moved before
    complaining.

    Both must raise BEFORE any native work, which is asserted by measurement
    rather than by reading the source: a real replay of this leg performs
    exactly one host observation (the validation copy), so a refused call that
    performs ZERO of them cannot have launched.
    """

    from witwin.radar.propagation.channel_consumer import ChannelPropagationAdapter

    dynamic = world.make_dynamic_scene(wall_velocity=geo.WALL_VELOCITY_M_PER_S)
    early = world.compile_snapshot(dynamic.at(0.0))
    late = world.compile_snapshot(dynamic.at(1.0))
    assert early.topology_version == late.topology_version
    assert early.geometry_version != late.geometry_version

    spike = drv.MultiEndpointSpike(compiled=early)
    baseline = _HostObservations(monkeypatch)
    _inbound(spike)
    assert baseline.counts["item"] == 1, baseline.counts

    on_late = ChannelPropagationAdapter(
        late,
        reference_frequency_hz=geo.REFERENCE_FREQUENCY_HZ,
        components=drv.MULTIPATH_COMPONENTS,
        max_depth=1,
    )
    assert on_late.epoch == spike.inbound.epoch
    counter = _HostObservations(monkeypatch)
    with pytest.raises(ValueError, match="geometry_version"):
        on_late.reevaluate(
            spike.inbound,
            _transmitter_spec(spike),
            _site_spec(spike),
            ad_mode="none",
        )
    assert counter.counts == dict.fromkeys(counter.counts, 0), counter.counts

    spike.adapter.refreeze(late)
    assert spike.adapter.epoch == 1
    counter = _HostObservations(monkeypatch)
    with pytest.raises(ValueError, match="geometry_version"):
        _inbound(spike)
    assert counter.counts == dict.fromkeys(counter.counts, 0), counter.counts


def test_an_unsupported_world_motion_declaration_is_refused():
    """The declaration is a closed vocabulary, checked against Channel's own."""

    spike = drv.MultiEndpointSpike()
    with pytest.raises(ValueError, match="unsupported world_motion"):
        spike.adapter.refreeze(spike.compiled, world_motion="whatever_moves")
    assert spike.adapter.world_motion == "frozen_world"
    assert spike.adapter.epoch == 0


def test_a_declared_fixed_winner_replay_keeps_its_handles():
    """The declaration is what separates a replay from a rediscovery.

    Under the default the handles are retired and the caller must rediscover.
    Under ``fixed_winner_replay`` the SAME handles keep answering against the
    moved geometry, which is the whole capability - and the answer really is
    different, because the wall really did move.
    """

    dynamic = world.make_dynamic_scene(wall_velocity=geo.WALL_VELOCITY_M_PER_S)
    spike = drv.MultiEndpointSpike(compiled=world.compile_snapshot(dynamic.at(0.0)))
    before = _inbound(spike).delay_s.clone()

    spike.adapter.refreeze(
        world.compile_snapshot(dynamic.at(1.0)),
        world_motion="fixed_winner_replay",
    )
    assert spike.adapter.epoch == 0
    after = _inbound(spike)
    rows = spike.predicted_inbound_rows()
    los = torch.tensor(
        [row.component == "los" for row in rows], device=before.device
    )
    assert torch.equal(before[los], after.delay_s[los])
    assert not torch.equal(before[~los], after.delay_s[~los])


# --------------------------------------------------------------------------
# 2. A row stopped existing
# --------------------------------------------------------------------------


def test_invalidation_never_produces_a_detached_gradient():
    """Every published row is either live-and-differentiable or inert.

    The forbidden middle is a row that is ``row_valid=True`` and carries no
    derivative: that is a silent zero in an optimisation, and it looks exactly
    like a stationary target. The site velocity here is RADIAL, so every
    surviving row must carry a strictly non-zero rate and a near-zero one would
    fail - a transverse fixture could not tell the two apart.

    A row that stopped existing must be inert on BOTH channels: an exactly zero
    payload and an exactly zero tangent. ``row_valid`` is the sole authority and
    nothing here post-masks anything.
    """

    dynamic = _arriving_wall()
    spike = drv.MultiEndpointSpike(compiled=world.compile_snapshot(dynamic.at(0.0)))
    spike.adapter.refreeze(
        world.compile_snapshot(dynamic.at(1.0)),
        world_motion="fixed_winner_replay",
    )

    sites = kin.Kinematics(
        positions_m=spike.site_tensor(),
        velocities_m_per_s=torch.tensor(
            [RADIAL, RADIAL], dtype=torch.float32, device="cuda"
        ),
    )
    with kin.two_way_duals(sites=sites) as duals:
        leg = _inbound(spike, duals.sites, ad_mode="jvp")
        rate = leg.delay_rate.detach().clone()
        coefficient = leg.coefficient.detach().clone()
        delay = leg.delay_s.detach().clone()
        valid = leg.row_valid.clone()

    assert bool(valid.any()) and not bool(valid.all())
    dead = ~valid
    assert torch.equal(rate[dead], torch.zeros_like(rate[dead]))
    assert torch.equal(delay[dead], torch.zeros_like(delay[dead]))
    assert torch.equal(
        coefficient[dead], torch.zeros_like(coefficient[dead])
    )
    assert float(rate[valid].abs().min()) > 1.0e-12
    assert float(coefficient[valid].abs().min()) > 0.0


# --------------------------------------------------------------------------
# 3. A row started existing
# --------------------------------------------------------------------------


def test_a_born_row_forces_an_explicit_rediscovery():
    """The birth gap, measured, and the cadence that closes it.

    A wall arrives. Two ``TX_B`` lines of sight are occluded and one reflection
    stops existing, and all three are published correctly as ``row_valid=False``
    with exact zeros. But the ``TX_A -> SITE_P`` reflection that the arriving
    wall CREATES is simply absent from the replay, and no device signal reports
    it: replay is subtractive by construction.

    So the assertion is a strict subset relation on the VALID rows, plus the
    identity of the row that is missing, plus the epoch loop's motion-event
    cadence recovering it. Under-reporting with a named boundary and a
    caller-owned mitigation is the accepted behaviour; a silent wrong answer
    would not be.
    """

    dynamic = _arriving_wall()
    early = world.compile_snapshot(dynamic.at(0.0))
    late = world.compile_snapshot(dynamic.at(1.0))

    parked = drv.MultiEndpointSpike(compiled=early)
    parked.adapter.refreeze(late, world_motion="fixed_winner_replay")
    replay = _inbound(parked)

    fresh = drv.MultiEndpointSpike(compiled=late)
    fresh_rows = set(_row_identities(_inbound(fresh)))

    identities = _row_identities(replay)
    valid = replay.row_valid.tolist()
    replayed = {row for row, alive in zip(identities, valid, strict=True) if alive}
    dead = [row for row, alive in zip(identities, valid, strict=True) if not alive]

    assert replayed < fresh_rows, (sorted(replayed), sorted(fresh_rows))
    born = fresh_rows - replayed
    assert born == {
        (geo.TX_A_STABLE_ID, geo.SITE_P_STABLE_ID, geo.REFLECTION_COMPONENT_ID)
    }, sorted(born)
    # Three rows died, and the two lines of sight among them are TX_B's.
    assert len(dead) == 3, dead
    assert (geo.TX_B_STABLE_ID, geo.SITE_P_STABLE_ID, geo.LOS_COMPONENT_ID) in dead
    assert (geo.TX_B_STABLE_ID, geo.SITE_Q_STABLE_ID, geo.LOS_COMPONENT_ID) in dead
    # Nothing on the replay says a row is missing. That is the gap.
    assert born & set(identities) == set()

    # The mitigation, in production: a declared motion-event cadence.
    loop, state = _epoch_loop(dynamic, period=1, world_motion="fixed_winner_replay")
    loop.frame(0.0)
    assert born & set(_row_identities(_inbound(state.spike))) == set()
    recovered = loop.frame(1.0)
    assert recovered.rediscovered and recovered.reason == MOTION_EVENT_CADENCE
    assert set(_row_identities(_inbound(state.spike))) == fresh_rows


class _EpochState:
    """The caller's own per-epoch state, rebuilt by ``bind``."""

    def __init__(self):
        self.spike = None
        self.binds = 0


def _epoch_loop(dynamic, *, period, world_motion="frozen_world", compile_scene=None):
    state = _EpochState()

    def bind(compiled, snapshot, previous):
        del snapshot
        state.binds += 1
        state.spike = drv.MultiEndpointSpike(
            compiled=compiled,
            adapter=None if previous is None else previous.adapter,
        )
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
        motion_event_period_frames=period,
        world_motion=world_motion,
    )
    return loop, state


# --------------------------------------------------------------------------
# The cadence, and what it costs
# --------------------------------------------------------------------------


def test_endpoint_only_motion_does_not_recompile():
    """The Core C1 routing pin.

    ``witwin.core`` folds ``time_s`` and the endpoint states into
    ``geometry_version`` for any snapshot that came from a ``DynamicScene``, so
    a world whose wall never moves still reports a fresh geometry version at
    every instant. A loop that trusted that signal would rebuild the RayD scene
    and its BVH once per frame for nothing.

    Two assertions, and the second is what makes the first meaningful: the
    production loop compiles exactly ONCE over eight frames, and the snapshots
    it declined to recompile really would have produced different geometry
    versions. Core is recorded, not patched.
    """

    from witwin.core.dynamics import LinearTrajectory

    dynamic = world.make_dynamic_scene(
        endpoint_trajectories={
            77101: LinearTrajectory(
                origin=torch.zeros(3), velocity=torch.tensor([0.0, 1.0, 0.0])
            )
        }
    )
    assert not dynamic.structure_trajectories
    assert not dynamic.structure_deformations

    compiles = []

    def counting_compile(snapshot, *, reference_frequency_hz):
        compiles.append(float(snapshot.time_s))
        return world.compile_snapshot(
            snapshot, reference_frequency_hz=reference_frequency_hz
        )

    loop, state = _epoch_loop(dynamic, period=None, compile_scene=counting_compile)
    assert loop.structures_move is False
    frames = [loop.frame(index * 1.0e-3) for index in range(8)]

    assert loop.compile_count == 1, compiles
    assert loop.discovery_count == 1
    assert state.binds == 1
    assert frames[0].reason == FIRST_FRAME
    assert all(not frame.rediscovered for frame in frames[1:])
    assert all(not frame.recompiled for frame in frames[1:])
    assert all(frame.compiled is frames[0].compiled for frame in frames)
    assert loop.poll_count == 2 * (len(frames) - 1)

    # The Core gap this routes around, as a minimal repro: the same endpoint
    # motion over a completely static wall reports a different geometry version
    # at every instant, so a version-driven loop would recompile per frame.
    versions = {
        world.compile_snapshot(dynamic.at(index * 1.0e-3)).geometry_version
        for index in range(4)
    }
    assert len(versions) == 4


def test_a_moving_structure_recompiles_and_rediscovers_once_per_frame():
    """The other half of the same rule: declared structure motion does pay."""

    dynamic = world.make_dynamic_scene(wall_velocity=geo.WALL_VELOCITY_M_PER_S)
    loop, state = _epoch_loop(dynamic, period=None)
    assert loop.structures_move is True
    frames = [loop.frame(index * 1.0e-3) for index in range(4)]
    assert loop.compile_count == 4
    assert loop.discovery_count == 4
    assert state.binds == 4
    assert frames[0].reason == FIRST_FRAME
    assert all(frame.reason == STRUCTURE_MOTION for frame in frames[1:])
    assert loop.epoch == 3
    # Each epoch's handles belong to the adapter that froze them, and that
    # adapter is the SAME object throughout - a new one per epoch would make
    # the epoch number meaningless.
    assert all(frame.frozen.adapter is frames[0].frozen.adapter for frame in frames)


def test_the_motion_event_cadence_costs_what_it_says(monkeypatch):
    """The three tiers, measured, with the replay proved discovery-free.

    The absolute milliseconds are reported rather than asserted - they are a
    property of this machine - but three things ARE asserted, because they are
    properties of the contract:

    * a batched replay performs no discovery at all
      (``discovery_launch_count == 0``);
    * a whole frame costs two host observations and zero synchronizations,
      independently of the slot count, which is the S2 budget unchanged;
    * discovery is the expensive tier, so a motion event must cost strictly
      more than a frame or the whole cadence is pointless.
    """

    dynamic = world.make_dynamic_scene(wall_velocity=geo.WALL_VELOCITY_M_PER_S)
    loop, state = _epoch_loop(dynamic, period=None)
    loop.frame(0.0)  # warm every operator table
    spike = state.spike
    slots = 8
    stack = spike.stacked(spike.site_tensor(), slots)
    spike.slot_legs(stack, slot_count=slots)

    def timed(callable_, *args, **kwargs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = callable_(*args, **kwargs)
        torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1.0e3, result

    event_ms, frame = timed(loop.frame, 1.0e-3)
    assert frame.rediscovered
    replay_ms, legs = timed(
        state.spike.slot_legs, stack, slot_count=slots
    )
    inbound, outbound = legs
    for leg in (inbound, outbound):
        assert leg.diagnostics.discovery_launch_count == 0
        assert leg.diagnostics.validation_d2h_copies == 1
        assert leg.diagnostics.validation_sync_count == 1
        assert leg.diagnostics.compact_count_d2h_copies == 0
    assert event_ms > replay_ms, (event_ms, replay_ms)
    print(
        f"\nmotion event (compile + refreeze + freeze x2 + composer): "
        f"{event_ms:.3f} ms\nbatched replay, {slots} slots, two legs: "
        f"{replay_ms:.3f} ms ({replay_ms / slots:.4f} ms/slot)"
    )

    counter = _HostObservations(monkeypatch)
    state.spike.slot_legs(stack, slot_count=slots)
    assert counter.counts["item"] == 2, counter.counts
    assert counter.counts["cpu"] == 0, counter.counts
    assert counter.counts["tolist"] == 0, counter.counts
    assert counter.counts["numpy"] == 0, counter.counts
    assert counter.counts["synchronize"] == 0, counter.counts


def test_the_epoch_loop_refuses_a_bind_that_freezes_nothing():
    """A loop with no handles would poll nothing and replay forever."""

    dynamic = world.make_dynamic_scene(wall_velocity=geo.WALL_VELOCITY_M_PER_S)
    loop = SceneEpochLoop(
        dynamic,
        reference_frequency_hz=geo.REFERENCE_FREQUENCY_HZ,
        bind=lambda compiled, snapshot, previous: FrozenEpoch(
            adapter=None, handles=()
        ),
        compile_scene=world.compile_snapshot,
    )
    with pytest.raises(ValueError, match="no handles"):
        loop.frame(0.0)


def test_the_epoch_loop_refuses_a_non_dynamic_scene():
    with pytest.raises(TypeError, match="DynamicScene"):
        SceneEpochLoop(
            object(),
            reference_frequency_hz=geo.REFERENCE_FREQUENCY_HZ,
            bind=lambda *args: None,
            compile_scene=world.compile_snapshot,
        )
