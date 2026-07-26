"""The scene-driven epoch loop: when a moving world costs what.

A Core ``DynamicScene`` can be sampled at any instant, and every one of those
snapshots is a legitimate world. What it does NOT say is how much of the
propagation pipeline has to be rebuilt for each of them, and the difference
between the cheapest and the most expensive answer is two orders of magnitude:

===========================  ==========================  =====================
tier                         work                        measured, this fixture
===========================  ==========================  =====================
0  session freeze            compile + evaluate +        2.55 + 9.10 + 0.74 ms
                             prepare_fixed_topology
1  motion event              compile + refreeze +        the same, minus the
                             freeze                      compile when only
                                                         endpoints moved
2  inner loop                one batched reevaluate      2.30 ms for the whole
                             per leg, all slots          frame
===========================  ==========================  =====================

This module owns tier 0 and tier 1 - *when* the expensive things happen. The
caller owns tier 2, which is exactly one batched
:meth:`~witwin.radar.propagation.channel_consumer.ChannelPropagationAdapter.reevaluate_slots`
per leg per frame. Keeping the two separate is the point: a loop that also
owned the replay would have to know about waveforms, slots and composition, and
the cadence question would then be answered differently in every solver.

Three rules, and each of them is a decision rather than an implementation
detail.

**Endpoint motion never recompiles.** ``witwin.core`` folds ``time_s`` and the
endpoint states into ``geometry_version`` whenever a snapshot comes from a
``DynamicScene``, so a world whose wall never moves still reports a fresh
geometry version at every instant, and a loop that recompiled on that signal
would rebuild the RayD scene and its BVH once per frame for nothing - measured
at 2.41 ms per frame of pure waste on this fixture. This loop therefore decides
from the DECLARED descriptors instead: if the ``DynamicScene`` carries no
structure trajectory and no structure deformation, the compiled scene is built
exactly once no matter how many frames run. Endpoint and target motion belongs
in the endpoint tensors, where it costs nothing.

**A moved world is a declaration, not a guess.** When structures do move, the
loop compiles the new snapshot and rebinds the adapter with the caller's
declared ``world_motion``. Under the default ``"frozen_world"`` that retires
every frozen handle and forces a rediscovery. Under
``"fixed_winner_replay"`` the frozen rows stay live and are replayed against
the moved geometry, which is correct and is what makes a per-frame moving
environment affordable - at the price the next rule names.

**Replay can lose rows but can never gain them.** A frozen topology re-tested
at new geometry publishes a row that stopped existing as ``row_valid=False``
with an exactly zero payload, which is a complete answer. A row that STARTED
existing is simply absent, and nothing on the device can report its absence
without a full discovery. So a caller whose world can gain paths must
rediscover on a cadence it declares, and ``motion_event_period_frames`` is that
declaration. Polling
:meth:`~witwin.radar.propagation.channel_consumer.ChannelPropagationAdapter.rediscovery_required`
every frame is free and catches everything else, but it cannot catch this one:
a born row leaves no trace in any version domain that the frozen topology can
be compared against.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True, slots=True, eq=False)
class FrozenEpoch:
    """What a caller freezes once per topology epoch.

    ``adapter`` is the :class:`ChannelPropagationAdapter` this epoch's rows were
    frozen against; the loop rebinds it in place rather than replacing it, so
    the caller's own references stay valid.

    ``handles`` are the frozen leg topologies the loop polls every frame. They
    are declared rather than discovered because an epoch may freeze any number
    of legs - a two-way radar freezes two - and the poll has to cover all of
    them.

    ``payload`` is everything else the caller froze at the same moment: a
    ``TwoWayComposer``, index tables, endpoint specs. The loop never inspects
    it. It exists so that a caller does not have to keep a second structure
    alive next to the epoch and risk the two disagreeing about which epoch they
    belong to.
    """

    adapter: object
    handles: tuple
    payload: object = None


@dataclass(frozen=True, slots=True, eq=False)
class EpochFrame:
    """One frame's world state, and what it cost to get there.

    ``reason`` names why this frame rediscovered, or is ``None`` when it did
    not. It is a string a caller can log or assert on rather than a boolean,
    because "the wall moved" and "the declared cadence came round" are
    different events with different budgets even though both cost a discovery.
    """

    frame_index: int
    time_s: float
    snapshot: object
    compiled: object
    epoch: int
    frozen: FrozenEpoch
    recompiled: bool
    rediscovered: bool
    reason: str | None


#: Why a frame paid for a rediscovery. Frozen strings so a caller can assert.
FIRST_FRAME = "first_frame"
STRUCTURE_MOTION = "structure_motion"
MOTION_EVENT_CADENCE = "motion_event_cadence"


class SceneEpochLoop:
    """Drive one compiled-scene lifecycle from a Core ``DynamicScene``.

    ``bind`` is called whenever the loop needs a fresh topology epoch, as
    ``bind(compiled_scene, snapshot, previous)``. On the first frame
    ``previous`` is ``None`` and the callback must build the adapter and freeze
    every leg. On a later rediscovery ``previous`` is the retiring
    :class:`FrozenEpoch`, its adapter has ALREADY been rebound onto
    ``compiled_scene`` by this loop, and the callback must freeze again through
    ``previous.adapter``. Returning a new adapter there is allowed but is
    almost always a mistake: the frozen handles carry an epoch number that only
    the adapter they came from can validate.

    ``motion_event_period_frames`` is the birth-gap cadence in frames. ``None``
    means never - correct only for a world that provably cannot gain a path,
    which in practice means a world with no structure motion and endpoints that
    never cross an occluder. ``1`` means rediscover every frame, which is
    honest and costs the full 9-40 ms.

    ``world_motion`` is forwarded verbatim to
    :meth:`ChannelPropagationAdapter.refreeze`; read its docstring, because
    ``"fixed_winner_replay"`` is an assertion about the world and not a
    performance switch.

    ``compile_scene`` is required and is normally
    ``witwin.channel.scene.compile``, called as
    ``compile_scene(snapshot, reference_frequency_hz=...)``. It is an argument
    rather than an import because Radar's import boundary allows exactly one
    module to name ``witwin.channel`` - the adapter - and scene compilation is
    a Channel lifecycle operation with a Channel-owned cache, not something
    this loop should hide. Passing it in also makes the compile count
    observable, which is the only way to prove the endpoint-motion rule above
    actually holds.
    """

    def __init__(
        self,
        dynamic_scene: object,
        *,
        reference_frequency_hz: float,
        bind: Callable[[object, object, FrozenEpoch | None], FrozenEpoch],
        compile_scene: Callable[..., object],
        motion_event_period_frames: int | None = None,
        world_motion: str = "frozen_world",
    ) -> None:
        for name in ("at", "structure_trajectories", "structure_deformations"):
            if not hasattr(dynamic_scene, name):
                raise TypeError(
                    f"dynamic_scene must expose {name!r}; pass a "
                    "witwin.core.dynamics.DynamicScene"
                )
        if not callable(compile_scene):
            raise TypeError(
                "compile_scene must be callable; pass "
                "witwin.channel.scene.compile"
            )
        if motion_event_period_frames is not None and (
            type(motion_event_period_frames) is not int
            or motion_event_period_frames < 1
        ):
            raise ValueError(
                "motion_event_period_frames must be a positive int or None, "
                f"got {motion_event_period_frames!r}"
            )
        self._dynamic = dynamic_scene
        self._reference_frequency_hz = float(reference_frequency_hz)
        self._bind = bind
        self._period = motion_event_period_frames
        self._world_motion = str(world_motion)
        self._compile_scene = compile_scene
        self._structures_move = bool(
            dynamic_scene.structure_trajectories
            or dynamic_scene.structure_deformations
        )
        self._frozen: FrozenEpoch | None = None
        self._compiled: object | None = None
        self._frame_index = -1
        self._epoch = -1
        self._last_discovery_frame = -1
        self.compile_count = 0
        self.discovery_count = 0
        self.poll_count = 0

    # -- what the caller reads ---------------------------------------------

    @property
    def structures_move(self) -> bool:
        """Whether any structure carries a trajectory or a deformation.

        The compile decision, and deliberately a property of the DECLARED
        descriptors rather than of any snapshot. A snapshot cannot answer it:
        its ``geometry_version`` moves with time whether or not any structure
        does.
        """

        return self._structures_move

    @property
    def world_motion(self) -> str:
        return self._world_motion

    @property
    def compiled(self) -> object:
        return self._compiled

    @property
    def frozen(self) -> FrozenEpoch | None:
        return self._frozen

    @property
    def epoch(self) -> int:
        """How many topology epochs have been frozen, minus one."""

        return self._epoch

    @property
    def frame_count(self) -> int:
        return self._frame_index + 1

    # -- one frame ----------------------------------------------------------

    def frame(self, time_s: float) -> EpochFrame:
        """Advance the world to ``time_s`` and return this frame's epoch state.

        Everything expensive that this frame needs has happened by the time
        this returns. The caller's remaining work is one batched
        ``reevaluate_slots`` per leg and one composition, and neither of them
        discovers, prepares or compiles.
        """

        self._frame_index += 1
        snapshot = self._dynamic.at(time_s)
        recompiled = self._recompile(snapshot)
        reason = self._rediscovery_reason(recompiled)
        if reason is not None:
            self._rediscover(snapshot, reason)
        return EpochFrame(
            frame_index=self._frame_index,
            time_s=float(time_s),
            snapshot=snapshot,
            compiled=self._compiled,
            epoch=self._epoch,
            frozen=self._frozen,
            recompiled=recompiled,
            rediscovered=reason is not None,
            reason=reason,
        )

    # -- the three decisions ------------------------------------------------

    def _recompile(self, snapshot: object) -> bool:
        """Compile this snapshot, or keep the one already built.

        The first frame always compiles because nothing exists yet. After that
        only declared structure motion compiles, which is what keeps Core's
        time-folded ``geometry_version`` out of the budget.
        """

        if self._compiled is not None and not self._structures_move:
            return False
        self._compiled = self._compile(snapshot)
        self.compile_count += 1
        if self._frozen is not None:
            self._frozen.adapter.refreeze(
                self._compiled, world_motion=self._world_motion
            )
        return True

    def _rediscovery_reason(self, recompiled: bool) -> str | None:
        """Name why this frame must rediscover, or ``None`` to replay.

        Order matters and is by cost, not by importance: the first frame has no
        alternative, a retired handle has no alternative, and only then is the
        free per-frame poll consulted.
        """

        if self._frozen is None:
            return FIRST_FRAME
        if recompiled and self._world_motion == "frozen_world":
            # refreeze() retired every handle; there is nothing left to replay.
            return STRUCTURE_MOTION
        if self._period is not None and (
            self._frame_index - self._last_discovery_frame >= self._period
        ):
            return MOTION_EVENT_CADENCE
        return self._poll()

    def _poll(self) -> str | None:
        """The free per-frame check: four host integers per frozen handle.

        ``geometry_version`` is skipped under ``"fixed_winner_replay"`` because
        that declaration is precisely "I know the geometry moved and I am
        holding the winners fixed". Every other domain respecifies the labels
        the frozen rows carry and is never replayable, so it fires under either
        declaration.
        """

        adapter = self._frozen.adapter
        ignore_geometry = self._world_motion == "fixed_winner_replay"
        for handle in self._frozen.handles:
            self.poll_count += 1
            moved = adapter.rediscovery_required(handle)
            if moved is None:
                continue
            if moved == "geometry_version" and ignore_geometry:
                continue
            return moved
        return None

    def _rediscover(self, snapshot: object, reason: str) -> None:
        previous = self._frozen
        frozen = self._bind(self._compiled, snapshot, previous)
        if not isinstance(frozen, FrozenEpoch):
            raise TypeError(
                f"bind must return a FrozenEpoch, got {type(frozen).__name__}"
            )
        if not frozen.handles:
            raise ValueError(
                "bind returned a FrozenEpoch with no handles; the per-frame "
                "rediscovery poll would then never fire and a moved world "
                "would replay silently"
            )
        self._frozen = frozen
        self._epoch += 1
        self._last_discovery_frame = self._frame_index
        self.discovery_count += 1
        del reason

    def _compile(self, snapshot: object) -> object:
        return self._compile_scene(
            snapshot, reference_frequency_hz=self._reference_frequency_hz
        )


__all__ = [
    "FIRST_FRAME",
    "MOTION_EVENT_CADENCE",
    "STRUCTURE_MOTION",
    "EpochFrame",
    "FrozenEpoch",
    "SceneEpochLoop",
]
