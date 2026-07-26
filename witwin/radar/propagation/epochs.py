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
every frame is free, but it compares the frozen rows against the versions the
COMPILED SCENE recorded, so it catches exactly the drift the compiled scene
already knows about. Two classes escape it and both are handled above rather
than by the poll: a born row leaves no trace in any version domain, and a world
mutated in place after compilation leaves the compiled scene and its own rows
agreeing with each other. The second is why the motion-event tick rehashes the
live world - see :data:`SOURCE_MUTATION`.

Those three rules are stated in the loop's own vocabulary - ``world_motion``
and ``motion_event_period_frames`` - and a scene is not usually described that
way. A caller describes it by naming its parts: this wall is static, that
foliage moves and can gain paths, this vehicle moves and cannot.
:class:`ClutterComponentSpec` is that description and :func:`epoch_policy`
resolves a set of them into the two arguments this loop takes. The resolution
is one-way on purpose: the loop never reads a component declaration, so
"which parts does this scene have" and "when does the pipeline pay" stay
separable questions with separate owners.
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


#: How a declared scene component moves, and therefore what the loop must pay
#: for it. These are CALLER DECLARATIONS about the world, never inferences:
#: whether a moving structure can GAIN a path is not detectable from inside a
#: replay, so nothing on the device can answer this question.
#:
#: * ``"static"`` - the component's geometry does not move. No structure
#:   trajectory, no recompile, no rediscovery. The zero-cost case by
#:   construction.
#: * ``"replay"`` - the geometry moves and the caller asserts the discrete
#:   winner set does not, so the frozen rows are replayed against it under
#:   ``world_motion="fixed_winner_replay"``. Replay is SUBTRACTIVE: a row that
#:   stops existing publishes ``row_valid=False`` with an exactly zero payload,
#:   and a row that starts existing is simply absent.
#: * ``"rediscover"`` - the component can gain paths, so the caller declares a
#:   cadence in frames on which every frozen handle is retired and discovery
#:   runs again.
MOBILITIES: tuple[str, ...] = ("static", "replay", "rediscover")

STATIC = "static"
REPLAY = "replay"
REDISCOVER = "rediscover"


@dataclass(frozen=True, slots=True)
class ClutterComponentSpec:
    """One named scene component and how it moves.

    ``components`` is the set of Channel propagation components this class
    needs - ``{"los", "reflection"}`` for a wall, ``{"diffraction"}`` for an
    edge. It is checked against the consumer's own
    ``fixed_topology_components`` by :func:`epoch_policy`, because a component
    that cannot be FROZEN cannot ride the fixed-topology inner loop at all and
    must rediscover every time it is wanted.

    ``rediscovery_period_frames`` belongs to ``mobility="rediscover"`` and to
    nothing else. A period on a static or replayed component would be a
    declaration that does nothing, which is the kind of dead configuration that
    later reads as a promise.
    """

    name: str
    mobility: str
    components: frozenset[str] = frozenset({"los", "reflection"})
    rediscovery_period_frames: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty str")
        if self.mobility not in MOBILITIES:
            raise ValueError(
                f"mobility must be one of {list(MOBILITIES)}, got "
                f"{self.mobility!r}"
            )
        object.__setattr__(
            self, "components", frozenset(str(value) for value in self.components)
        )
        if not self.components:
            raise ValueError(
                f"component {self.name!r} declares no propagation components; "
                "a component set is what decides whether it can be frozen"
            )
        if self.mobility == REDISCOVER:
            period = self.rediscovery_period_frames
            if type(period) is not int or period < 1:
                raise ValueError(
                    f"component {self.name!r} declares mobility 'rediscover' "
                    "and must name rediscovery_period_frames as a positive int, "
                    f"got {period!r}"
                )
        elif self.rediscovery_period_frames is not None:
            raise ValueError(
                f"component {self.name!r} declares mobility "
                f"{self.mobility!r} and a rediscovery period of "
                f"{self.rediscovery_period_frames!r}; only a rediscovering "
                "component has a cadence, and a period here would never fire"
            )


@dataclass(frozen=True, slots=True)
class EpochPolicy:
    """The two :class:`SceneEpochLoop` arguments a declaration resolves to.

    A value record with value equality, deliberately: comparing two resolutions
    is the natural way to ask whether adding a component changed what the loop
    will do, and identity semantics would answer that question wrongly.
    """

    world_motion: str
    motion_event_period_frames: int | None


def epoch_policy(specs, *, fixed_topology_components) -> EpochPolicy:
    """Resolve a set of component declarations into ONE loop configuration.

    There is one compiled scene and one epoch loop per session, so a mixed
    declaration has to resolve to a single ``world_motion`` and a single
    cadence. The resolution, stated rather than implied:

    * every component static - nothing moves, so the declaration is
      ``"frozen_world"`` with no cadence and the loop never recompiles;
    * static plus replay - ``"fixed_winner_replay"`` with no cadence: the
      frozen rows are replayed against the moved geometry and no path can be
      born;
    * static plus rediscover - ``"frozen_world"`` on the shortest declared
      cadence, so every frozen handle is retired and discovery runs again;
    * replay AND rediscover together - ``"fixed_winner_replay"`` on the
      shortest declared cadence. Frames between the ticks replay, which is what
      the replayed component asked for, and the tick pays the discovery the
      rediscovering component asked for. Neither declaration is downgraded.

    ``fixed_topology_components`` is the Channel capability record's own set,
    passed in rather than imported: this module names no ``witwin`` package, and
    quoting the live record is what keeps a Channel that widens or narrows its
    freezable set from leaving a stale constant behind here.
    """

    declared = tuple(specs)
    if not declared:
        raise ValueError(
            "epoch_policy needs at least one ClutterComponentSpec; an empty "
            "declaration says nothing about how the world moves"
        )
    freezable = frozenset(str(value) for value in fixed_topology_components)
    names: set[str] = set()
    for spec in declared:
        if not isinstance(spec, ClutterComponentSpec):
            raise TypeError(
                "epoch_policy needs ClutterComponentSpec values, got "
                f"{type(spec).__name__}"
            )
        if spec.name in names:
            raise ValueError(f"component {spec.name!r} is declared twice")
        names.add(spec.name)
        if spec.mobility == REDISCOVER:
            continue
        outside = spec.components - freezable
        if outside:
            raise NotImplementedError(
                f"component {spec.name!r} declares mobility {spec.mobility!r} "
                f"over propagation components {sorted(spec.components)}, but "
                f"the consumer can freeze only {sorted(freezable)}; "
                f"{sorted(outside)} cannot be replayed from a frozen topology "
                "and must declare mobility 'rediscover' with a cadence"
            )

    mobilities = {spec.mobility for spec in declared}
    periods = [
        spec.rediscovery_period_frames
        for spec in declared
        if spec.mobility == REDISCOVER
    ]
    period = min(periods) if periods else None
    replays = REPLAY in mobilities
    return EpochPolicy(
        world_motion="fixed_winner_replay" if replays else "frozen_world",
        motion_event_period_frames=period,
    )


#: Why a frame paid for a rediscovery. Frozen strings so a caller can assert.
FIRST_FRAME = "first_frame"
STRUCTURE_MOTION = "structure_motion"
MOTION_EVENT_CADENCE = "motion_event_cadence"

#: The authored world was mutated in place behind the compiled scene.
#:
#: The four version domains are content hashes, so a compiled scene and the
#: rows discovered on it always agree with each other no matter what happened
#: to the world afterwards - the free per-frame poll compares the frozen rows
#: against what the compiled scene RECORDED and therefore cannot see this. The
#: only thing that can is rehashing the live world the compiled scene was built
#: from, which is ``O(scene)`` host work and belongs exactly where this loop
#: puts it: on the motion-event tick, which already pays a full discovery.
#:
#: When it fires the compiled scene itself is stale, so this is the one reason
#: that forces a RECOMPILE as well as a rediscovery. Rediscovering against the
#: stale compiled scene would reproduce the stale answer at full strength.
SOURCE_MUTATION = "source_mutation"


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

    ``motion_event_period_frames`` is the birth-gap cadence in frames, and it
    is also the cadence on which the live world is rehashed
    (:data:`SOURCE_MUTATION`). ``None`` means never, and it is therefore two
    declarations at once: that no path can be born, and that the authored world
    is never mutated outside the ``DynamicScene`` API. Both are true of a world
    with no structure motion and endpoints that never cross an occluder; a
    caller that edits mesh vertices in place must declare a period instead.
    ``1`` means rediscover every frame, which is honest and costs the full
    9-40 ms.

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
        self.revalidation_count = 0

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
        mutated = self._revalidate_source()
        recompiled = self._recompile(snapshot, force=mutated)
        reason = self._rediscovery_reason(recompiled, mutated)
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

    def _revalidate_source(self) -> bool:
        """Has the authored world been mutated behind the compiled scene?

        Only on the motion-event tick, and only when a period was declared:
        this rehashes the live world, which is ``O(scene)`` host work that the
        Channel consumer forbids in a frame loop. The tick already pays a full
        discovery, so the hash is invisible there.

        The source signal is isolated rather than trusted wholesale.
        ``rediscovery_required(revalidate_source=True)`` reports the recorded
        provenance drift FIRST and only falls through to the live world, so a
        non-``None`` answer on its own would confuse "the caller rebound onto a
        moved scene" (which the declared rules above already handle, and which
        ``fixed_winner_replay`` deliberately tolerates) with "the world moved
        under the compiled scene" (which nothing else can see). Only the second
        one recompiles.
        """

        if self._frozen is None or not self._motion_event_due():
            return False
        adapter = self._frozen.adapter
        for handle in self._frozen.handles:
            self.revalidation_count += 1
            if adapter.rediscovery_required(handle) is not None:
                continue
            if adapter.rediscovery_required(handle, revalidate_source=True):
                return True
        return False

    def _motion_event_due(self) -> bool:
        return self._period is not None and (
            self._frame_index - self._last_discovery_frame >= self._period
        )

    def _recompile(self, snapshot: object, *, force: bool = False) -> bool:
        """Compile this snapshot, or keep the one already built.

        The first frame always compiles because nothing exists yet. After that
        only declared structure motion compiles, which is what keeps Core's
        time-folded ``geometry_version`` out of the budget - unless ``force``
        says the world was mutated in place, in which case the compiled scene
        is stale no matter what the descriptors declare.
        """

        if self._compiled is not None and not self._structures_move and not force:
            return False
        self._compiled = self._compile(snapshot)
        self.compile_count += 1
        if self._frozen is not None:
            self._frozen.adapter.refreeze(
                self._compiled, world_motion=self._world_motion
            )
        return True

    def _rediscovery_reason(self, recompiled: bool, mutated: bool) -> str | None:
        """Name why this frame must rediscover, or ``None`` to replay.

        Order matters and is by cost, not by importance: the first frame has no
        alternative, a retired handle has no alternative, and only then is the
        free per-frame poll consulted. ``SOURCE_MUTATION`` outranks the cadence
        that discovered it because the two cost the same discovery but say
        different things, and "the world changed behind your back" is the one a
        caller has to act on.
        """

        if self._frozen is None:
            return FIRST_FRAME
        if mutated:
            return SOURCE_MUTATION
        if recompiled and self._world_motion == "frozen_world":
            # refreeze() retired every handle; there is nothing left to replay.
            return STRUCTURE_MOTION
        if self._motion_event_due():
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
    "MOBILITIES",
    "MOTION_EVENT_CADENCE",
    "REDISCOVER",
    "REPLAY",
    "SOURCE_MUTATION",
    "STATIC",
    "STRUCTURE_MOTION",
    "ClutterComponentSpec",
    "EpochFrame",
    "EpochPolicy",
    "FrozenEpoch",
    "SceneEpochLoop",
    "epoch_policy",
]
