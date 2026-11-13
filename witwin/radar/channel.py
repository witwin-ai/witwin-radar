"""Radar's adapter onto the stable Channel propagation consumer.

This is the ONLY Radar module that imports ``witwin.channel``, and it names
exactly two things from it: the solver-neutral consumer facade
``witwin.channel.propagation.consumer`` and the scene compile facade
``witwin.channel.scene.compile``. The second one arrived with the production
scene-driven entry (R-ADR-020): compiling a Core world is a Channel lifecycle
operation, so it belongs to the module that already owns this boundary rather
than to a second allowlisted crossing. It is a MODULE function
(:func:`compile_scene`), not an adapter method, because it produces the scene an
adapter is constructed WITH - there is no adapter yet when it is called - and it
is imported inside that function so that importing this module still loads
nothing beyond the consumer facade's own closure.

Nothing here touches a Channel solver, the enumerated engine, the internal
propagation contracts, or the native extension. R-ADR-001 records why: Radar is
a consumer of the published contract, not a second enumerated exception to
ADR-008.

The adapter owns four things and nothing else:

* endpoint batching  -  turning Radar endpoint specs into consumer batches;
* the freeze/reevaluate split  -  discovery happens once per frozen topology,
  and every later frame replays it with exactly one consumer call, whether that
  frame is one instant or a whole slot-major TDM/symbol/pulse block;
* the rediscovery cadence  -  ``rediscovery_required`` says when the world moved
  out from under a frozen topology, and ``refreeze`` is the supported way to
  rebind onto the new compiled scene, declaring either that every stale handle
  is retired (``world_motion="frozen_world"``) or that the discrete winner set
  is deliberately held fixed while the geometry moves
  (``world_motion="fixed_winner_replay"``);
* publishing Radar-shaped leg results  -  delay and coefficient.

It owns no physics. Every number it publishes was produced by a native Channel
kernel and is passed through by reference.

Cardinality discipline (R-ADR-006): the adapter adds ZERO host observations
beyond the ones the consumer already performs. ``K`` is read from
``PropagationPathBatch.path_count``, which the ADR-032 compact boundary already
published as a host int, so nothing here calls ``.item()``, ``.cpu()``,
``.tolist()``, or iterates a device tensor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from witwin.channel.propagation import consumer

from .propagation import RadarEndpointSpec, RadarLegBatch, require_endpoint_role

#: Components whose path is the straight source-to-sink segment. Free-space
#: propagation along it is undefined for coincident endpoints, so a leg that
#: requests one of them runs Channel's endpoint-separation check first.
STRAIGHT_SEGMENT_COMPONENTS = frozenset({"los", "transmission"})


@dataclass(frozen=True, slots=True, eq=False)
class FrozenLegTopology:
    """A discovered leg topology, the rows discovery evaluated, and its replay handle.

    ``topology`` is the consumer's discovery topology, stamped with the world
    provenance it was discovered against; the per-frame rediscovery poll reads
    it. ``prepared`` - the partition Channel replays - is built from it on
    first use and cached, so a run that never replays a handle never pays for
    preparing one. A primal run that rediscovers at every observation reads its
    rows from ``discovered`` instead and calls Channel once per leg rather than
    three times (discover, prepare, replay at the same instant).

    The recorded host-observation counters describe the ONE-TIME preparation,
    not a later :meth:`ChannelPropagationAdapter.reevaluate_slots` call.
    Preparing the handle once per frozen topology is the contract, and
    preparing it per frame gives up the whole point of the capability.

    ``source_id`` and ``sink_id`` are the frozen rows' stable world IDs. They
    are what a downstream composer joins on. Joining by array position instead
    would be silently wrong the moment a leg publishes its rows in a different
    order.

    ``component_id``, ``depth``, ``primitive_sequence`` and
    ``material_sequence`` complete that identity. With more than one multipath
    component per endpoint pair, the endpoint IDs alone no longer distinguish
    two rows of the same leg, and a composer that fell back on row position for
    the tie would reintroduce exactly the positional dependence the identity
    join exists to remove.

    ``complete`` is Channel's own statement that these rows are provably every
    path the request admits (``PropagationEvaluation.complete``). It is
    ``False`` when reflection candidates came from ray discovery, which can
    miss paths; a session ANDs it into ``path_set_complete``.
    ``transmission_dropped_rows`` is the discovery's count of source-sink pairs
    left without a transmission path because their straight segment crosses
    more walls than ``max_depth``.

    ``replayable`` is ``False`` when the request named a component Channel
    cannot freeze (``transmission``, ``diffraction``). Such a handle answers
    only through ``discovered``, and preparing it is refused by name.

    ``epoch`` names the compiled scene this handle was frozen against. An
    adapter that is rebound to a new compiled scene by
    :meth:`ChannelPropagationAdapter.refreeze` advances its epoch, and every
    handle from the previous epoch is retired. That is a host int comparison
    and it is deliberately BROADER than the Channel-side world-provenance
    check: provenance can only see version domains that actually moved, so a
    caller that rebound to a differently-built compiled scene of an
    identical-looking world would otherwise be answered rather than refused.

    ``slot_topologies`` caches the block-diagonal replications of ``prepared``
    keyed by ``(slot_count, source_count, sink_count)``. Replication is pure
    index arithmetic, but it allocates, and doing it per frame would put an
    avoidable allocation in the inner loop for a table that is a function of
    the frozen topology alone.
    """

    topology: object
    source_id: torch.Tensor
    sink_id: torch.Tensor
    component_id: torch.Tensor
    depth: torch.Tensor
    primitive_sequence: torch.Tensor
    material_sequence: torch.Tensor
    row_count: int
    #: The diffracting edge of each row (``-1`` on a row that diffracts at
    #: none); part of the identity of a leg that requested ``diffraction``.
    edge_id: torch.Tensor | None = None
    complete: bool = True
    transmission_dropped_rows: int = 0
    replayable: bool = True
    requested: frozenset[str] = frozenset()
    discovered: RadarLegBatch | None = field(default=None, repr=False)
    epoch: int = 0
    _prepared: list = field(default_factory=list, repr=False)
    slot_topologies: dict = field(default_factory=dict, repr=False)

    @property
    def prepared(self) -> object:
        """Channel's replay partition of these rows, prepared on first use.

        ``prepare_fixed_topology`` synchronizes, so it runs at most once per
        handle. A handle whose request named a component Channel cannot freeze
        is refused here, before Channel sees it: its rows exist only as the
        discovery evaluated them.
        """

        if not self._prepared:
            if not self.replayable:
                unfrozen = sorted(self.requested - consumer.capabilities().fixed_topology_components)
                raise NotImplementedError(
                    f"this leg was discovered with components {unfrozen}, which Channel cannot freeze for "
                    "reevaluation; their rows exist only at the discovery instant, so a route that replays "
                    "them (grad='vjp' or 'jvp', a discovery cadence longer than one observation, a wideband "
                    "band) is refused. Run grad='none' with discovery at every observation, or drop them"
                )
            prepared = consumer.prepare_fixed_topology(self.topology)
            unsupported = (
                frozenset(bucket.component for bucket in prepared.buckets)
                - consumer.capabilities().fixed_topology_components
            )
            if unsupported:
                raise NotImplementedError(
                    f"the discovered topology carries components {sorted(unsupported)} that cannot be reevaluated"
                )
            self._prepared.append(prepared)
        return self._prepared[0]

    @property
    def components(self) -> tuple[str, ...]:
        """The components the discovered rows carry, in Channel's bucket order."""

        return tuple(bucket.component for bucket in self.prepared.buckets)

    @property
    def prepare_d2h_copies(self) -> int:
        return int(self.prepared.prepare_d2h_copies)

    @property
    def prepare_d2h_bytes(self) -> int:
        return int(self.prepared.prepare_d2h_bytes)

    @property
    def prepare_synchronizations(self) -> int:
        return int(self.prepared.prepare_synchronizations)


def _endpoint_batch(spec: RadarEndpointSpec, role: str) -> consumer.EndpointBatch:
    """Build a consumer endpoint batch, preserving gradient state exactly."""

    require_endpoint_role(spec, role)
    return consumer.EndpointBatch(
        stable_ids=spec.stable_ids,
        positions_m=spec.positions_m,
        polarizations=spec.polarizations,
        powers_w=spec.powers_w,
    )


def _detached(spec: RadarEndpointSpec) -> RadarEndpointSpec:
    """Strip AD state for discovery.

    Freezing a topology is a discrete question - which paths exist - and the
    answer carries no derivative. Detaching here makes that explicit instead of
    relying on a downstream call happening to ignore the tape.
    """

    return RadarEndpointSpec(
        stable_ids=spec.stable_ids,
        positions_m=spec.positions_m.detach(),
        polarizations=spec.polarizations.detach(),
        powers_w=None if spec.powers_w is None else spec.powers_w.detach(),
    )


def compile_scene(scene_or_snapshot: object, *, reference_frequency_hz: float) -> object:
    """Compile one Core ``Scene`` or ``SceneSnapshot`` at one frequency.

    A thin wrapper by design: Channel owns compilation, the compiled
    resources, and the compile cache, and a Radar-side reimplementation of any
    of that would be a second owner of the same lifecycle.

    Called exactly as :class:`~witwin.radar.propagation.SceneEpochLoop`
    calls its ``compile_scene`` argument, so this function IS that argument for
    the production driver. The loop keeps taking it as a parameter rather than
    importing it: the compile count stays observable, and the loop stays free of
    the Channel import edge.

    ``reference_frequency_hz`` is checked against the scene that comes back
    before the scene is handed to anybody. Channel refuses a mismatch at
    ``evaluate``/``reevaluate`` time anyway, but that is after a request has been
    built; checking here means a scene that came out of the compile cache at
    some other frequency is refused at the compile call that asked for it, which
    is where a caller can still read the two numbers side by side. There is no
    implicit recompile: a mismatch is an error, never a second compile.
    """

    from witwin.channel.scene import compile as channel_compile

    compiled = channel_compile(scene_or_snapshot, reference_frequency_hz=reference_frequency_hz)
    require_reference_frequency(compiled, reference_frequency_hz)
    return compiled


def require_reference_frequency(compiled_scene: object, reference_frequency_hz: float) -> None:
    """Refuse a compiled scene that was not compiled at this frequency.

    Host-only: it compares two numbers and launches nothing. A driver that
    receives a compiled scene from somewhere other than :func:`compile_scene`
    calls this BEFORE building an adapter, so the mismatch is reported where the
    binding is made rather than inside the first replay.

    The refusal is Channel's own, quoted rather than re-derived, because the
    exactness rule (a hex comparison, not a tolerance) belongs to the side that
    owns the compiled constant.
    """

    check = getattr(compiled_scene, "require_reference_frequency", None)
    if check is None:
        raise TypeError("compiled_scene must expose require_reference_frequency; pass a witwin.channel CompiledScene")
    check(reference_frequency_hz)


class ChannelPropagationAdapter:
    """Radar-facing view of one compiled Channel scene at one frequency.

    Every refusal below quotes the Channel capability record rather than a local
    copy of it, so a Channel that widens or narrows what it can freeze moves
    this adapter with it instead of leaving a stale constant behind.
    """

    def __init__(
        self,
        compiled_scene: object,
        *,
        reference_frequency_hz: float,
        components: frozenset[str],
        max_depth: int,
        discovery: str = "auto",
        max_paths: int | None = None,
        replay: bool = True,
    ) -> None:
        """Bind one compiled scene and one propagation request.

        ``discovery`` is Channel's reflection candidate source: ``"auto"`` (the
        complete image tree, ray discovery past its cost budget), ``"exhaustive"``
        or ``"rays"``. ``max_paths`` bounds the rows one discovery may publish;
        Channel raises on overflow and never truncates.

        ``replay=False`` declares that this adapter's handles answer only
        through their discovered rows, which is what admits the components
        Channel cannot freeze (``transmission``, ``diffraction``). With the
        default ``replay=True`` such a component is refused here, before any
        discovery, because every handle must be replayable.
        """

        capabilities = consumer.capabilities()
        unsupported = frozenset(components) - capabilities.components
        if unsupported:
            raise NotImplementedError(f"the propagation consumer does not support components {sorted(unsupported)}")
        not_freezable = frozenset(components) - capabilities.fixed_topology_components
        if not_freezable and replay:
            raise NotImplementedError(
                f"components {sorted(not_freezable)} cannot be frozen for "
                f"reevaluation; supported components are "
                f"{sorted(capabilities.fixed_topology_components)}"
            )
        if discovery not in capabilities.discovery_modes:
            raise ValueError(f"discovery must be one of {sorted(capabilities.discovery_modes)}, got {discovery!r}")
        if max_paths is not None and (type(max_paths) is not int or max_paths < 1):
            raise ValueError(f"max_paths must be a positive int or None, got {max_paths!r}")
        self._compiled = compiled_scene
        self._reference_frequency_hz = float(reference_frequency_hz)
        self._components = frozenset(components)
        self._max_depth = int(max_depth)
        self._discovery = str(discovery)
        self._max_paths = max_paths
        self._capabilities = capabilities
        self._epoch = 0
        self._world_motion = "frozen_world"

    @property
    def components(self) -> frozenset[str]:
        """The propagation components every discovery of this adapter requests."""

        return self._components

    @property
    def reference_frequency_hz(self) -> float:
        return self._reference_frequency_hz

    @property
    def capabilities(self) -> object:
        return self._capabilities

    @property
    def compiled_scene(self) -> object:
        """The compiled scene every replay is currently evaluated against."""

        return self._compiled

    @property
    def epoch(self) -> int:
        """How many times this adapter has been rebound to a new scene."""

        return self._epoch

    @property
    def world_motion(self) -> str:
        """What the last rebind declared happened to the world.

        ``"frozen_world"`` until a caller says otherwise. It is the value every
        replay forwards to Channel, so it is readable rather than implicit.
        """

        return self._world_motion

    def refreeze(self, compiled_scene: object, *, world_motion: str = "frozen_world") -> None:
        """Rebind to a new compiled scene, declaring what moved.

        A moving structure, a deformed mesh or a new ``DynamicScene`` snapshot
        produces a NEW ``CompiledScene``. Replaying frozen rows against the
        scene this adapter was constructed with would answer out of geometry
        that has moved on, silently and at full strength; this call is the
        only way the adapter learns that the world moved.

        ``world_motion`` is the caller's declaration, and it is forwarded
        verbatim to every later Channel replay:

        ``"frozen_world"`` (the default) says the caller intends to rediscover.
        Every handle frozen before this call is RETIRED and refused by name
        afterwards, and Channel additionally refuses any moved world version
        domain. The caller must call :meth:`freeze` again, because which paths
        exist is exactly the question a moved world reopens.

        ``"fixed_winner_replay"`` says something specific and it is the whole
        content of the declaration: *the discrete winner set is held fixed
        while the geometry moves*. The frozen rows stay live and are replayed
        against the new geometry, so every published row is the SAME
        interaction sequence re-evaluated at the new vertex positions. Two
        consequences the caller is asserting it accepts:

        * a row that stops existing is published with ``row_valid=False`` and
          an exactly zero payload - a complete answer, not an error;
        * a row that STARTS existing is not published at all. Replay is
          subtractive by construction, so a caller whose world can gain paths
          must rediscover on a motion-event cadence
          (:class:`witwin.radar.propagation.SceneEpochLoop` is that
          cadence).

        It never accepts a moved topology, material or assignment version:
        those respecify the labels the frozen rows carry, and Channel refuses
        them under either declaration.

        :meth:`rediscovery_required` is the cheap half that tells a caller when
        to make this call in the first place.
        """

        if compiled_scene is None:
            raise ValueError("refreeze requires a compiled scene")
        # The vocabulary comes from the capability record rather than from a
        # module constant: the consumer facade exports AD_MODES, RESPONSES and
        # TOPOLOGY_MODES but not WORLD_MOTIONS, and capabilities.world_motions
        # is the published route to the same frozen set.
        supported = self._capabilities.world_motions
        if world_motion not in supported:
            raise ValueError(f"unsupported world_motion {world_motion!r}; supported values are {sorted(supported)}")
        self._compiled = compiled_scene
        self._world_motion = world_motion
        if world_motion == "frozen_world":
            self._epoch += 1

    def rediscovery_required(self, frozen: FrozenLegTopology, *, revalidate_source: bool = False) -> str | None:
        """Name the world version domain that moved, or ``None``.

        A host-only integer comparison against the versions the compiled scene
        recorded: no device work, no allocation, no synchronization, so a
        caller can poll it every frame for free. ``revalidate_source=True``
        additionally rehashes the live world, which is ``O(scene)`` host work
        and belongs on a motion-event cadence, never in a frame loop.

        A handle from a retired epoch is reported against the scene this
        adapter holds NOW, which is what a caller deciding whether to
        rediscover actually wants to know.
        """

        return consumer.rediscovery_required(self._compiled, frozen.topology, revalidate_source=revalidate_source)

    def freeze(self, sources: RadarEndpointSpec, sinks: RadarEndpointSpec) -> FrozenLegTopology:
        """Discover one leg's topology and partition it for replay, once.

        Call this OUTSIDE the per-frame loop. It runs full discovery and
        prepares the replay partition, and both synchronize; calling it per
        frame would reintroduce exactly the host observation the fixed-topology
        capability exists to avoid.
        """

        frozen = self.discover(sources, sinks, rows=False)
        _ = frozen.prepared
        return frozen

    def discover(
        self,
        sources: RadarEndpointSpec,
        sinks: RadarEndpointSpec,
        *,
        rows: bool = True,
        components: frozenset[str] | None = None,
    ) -> FrozenLegTopology:
        """Discover one leg's topology and, with ``rows``, keep the rows discovery evaluated.

        ``components`` narrows this one leg's request to a subset of the
        adapter's components (a radar's site-less leg asks for its own set);
        ``None`` requests them all.

        One Channel ``evaluate``. The replay partition is NOT prepared here: a
        handle prepares it on first replay, so a primal run that rediscovers at
        every observation and reads ``discovered`` never prepares at all.

        Discovery evaluates detached endpoints in ``ad_mode="none"``, so
        ``discovered`` carries no derivative. Its delay, coefficient and
        geometry are the same native evaluation a replay at the same endpoints
        returns for ``los`` and ``reflection`` rows; its ``row_valid`` is all
        true, because a discovered row exists by construction.

        A request with a straight-segment component (``los``, ``transmission``)
        first runs Channel's endpoint-separation check over every source-sink
        pair: free-space transport is singular at coincident endpoints, and
        Channel publishes a finite, unbounded coefficient there rather than
        refusing. The check is one one-byte device read per discovery, and it
        refuses by name the first coincident pair.
        """

        requested = self._components if components is None else frozenset(components)
        if not requested or not requested <= self._components:
            raise ValueError(
                f"a leg may request a non-empty subset of the adapter's components {sorted(self._components)}, "
                f"got {sorted(requested)}"
            )
        detached_sources = _detached(sources)
        detached_sinks = _detached(sinks)
        source_batch = _endpoint_batch(detached_sources, "source")
        sink_batch = _endpoint_batch(detached_sinks, "sink")
        if requested & STRAIGHT_SEGMENT_COMPONENTS:
            consumer.check_endpoint_separation(
                source_batch, sink_batch, reference_frequency_hz=self._reference_frequency_hz
            )
        evaluation = consumer.evaluate(
            self._compiled,
            consumer.PropagationRequest(
                sources=source_batch,
                sinks=sink_batch,
                reference_frequency_hz=self._reference_frequency_hz,
                components=requested,
                max_depth=self._max_depth,
                response="scalar_transport",
                topology_mode="discover",
                ad_mode="none",
                max_paths=self._max_paths,
                discovery=self._discovery,
            ),
        )
        paths = evaluation.paths
        topology = paths.topology
        diagnostics = evaluation.diagnostics
        discovered = None
        if rows:
            discovered = _leg_batch(
                paths,
                detached_sources,
                detached_sinks,
                row_valid=torch.ones(paths.path_count, dtype=torch.bool, device=paths.pair_index.device),
                diagnostics=diagnostics,
                slot_count=1,
            )
        return FrozenLegTopology(
            topology=topology,
            source_id=topology.source_id,
            sink_id=topology.sink_id,
            component_id=topology.component_id,
            depth=topology.depth,
            primitive_sequence=topology.primitive_sequence,
            material_sequence=topology.material_sequence,
            row_count=paths.path_count,
            edge_id=topology.edge_id,
            complete=bool(evaluation.complete),
            transmission_dropped_rows=int(getattr(diagnostics, "transmission_dropped_rows", 0)),
            replayable=requested <= self._capabilities.fixed_topology_components,
            requested=requested,
            discovered=discovered,
            epoch=self._epoch,
        )

    def reevaluate_slots(
        self,
        frozen: FrozenLegTopology,
        sources: RadarEndpointSpec,
        sinks: RadarEndpointSpec,
        *,
        slot_count: int,
        ad_mode: str,
        frequency_offsets_hz: tuple[float, ...] | None = None,
    ) -> RadarLegBatch:
        """Replay a frozen leg over a whole frame, pulse train or symbol block.

        ``sources`` and ``sinks`` are the SLOT-MAJOR STACKS: slot ``t`` owns
        rows ``[t * n, (t + 1) * n)`` of each, where ``n`` is that end's
        per-slot count. Every slot must repeat the same stable IDs in the same
        order, because the frozen rows name endpoints by identity and a slot
        that renamed them would be a different topology, not a later instant.

        This is ONE consumer call for the whole set: one launch per bucket, one
        validation copy, one synchronization, however many slots there are. A
        Python loop over per-slot replays produces the same numbers and
        multiplies the per-frame host-observation budget by the slot count,
        which is exactly what the batched contract exists to forbid.

        The pair partition is block diagonal, so the pair count grows LINEARLY
        in ``slot_count``. Stacking both ends into a plain reevaluation instead
        would take the full source-by-sink outer product across slots and cost
        the square of it.

        ``slot_count=1`` is one instant: exactly one consumer call, no
        discovery, no preparation, no second cardinality observation.

        ``frequency_offsets_hz`` asks Channel for the same rows at
        ``reference_frequency_hz + offset`` as well, in the same call; the band
        is published as ``coefficient_band`` ``[rows, F]`` beside the
        reference ``coefficient``. Geometry and ``row_valid`` stay the
        reference evaluation's, because they do not depend on frequency.
        """

        if ad_mode not in consumer.AD_MODES:
            raise NotImplementedError(
                f"unsupported ad_mode {ad_mode!r}; supported values are {sorted(consumer.AD_MODES)}"
            )
        if type(slot_count) is not int or slot_count < 1:
            raise ValueError(f"slot_count must be a positive int, got {slot_count!r}")
        self._require_current_epoch(frozen)
        topology = self._slot_topology(frozen, sources, sinks, slot_count)
        request: dict[str, object] = {}
        if frequency_offsets_hz is not None:
            request["frequency_offsets_hz"] = tuple(float(value) for value in frequency_offsets_hz)
        result = consumer.reevaluate(
            self._compiled,
            consumer.FixedTopologyRequest(
                sources=_endpoint_batch(sources, "source"),
                sinks=_endpoint_batch(sinks, "sink"),
                reference_frequency_hz=self._reference_frequency_hz,
                topology=topology,
                response="scalar_transport",
                ad_mode=ad_mode,
                slot_count=slot_count,
                world_motion=self._world_motion,
                **request,
            ),
        )
        return _leg_batch(
            result.paths,
            sources,
            sinks,
            row_valid=result.row_valid,
            diagnostics=result.diagnostics,
            slot_count=slot_count,
        )

    def _require_current_epoch(self, frozen: FrozenLegTopology) -> None:
        """Refuse a handle frozen against a compiled scene this adapter dropped.

        The message names the world version domain that moved when Channel can
        see one, so a caller reads ``geometry_version`` for a moved wall rather
        than a bare identity complaint. When no domain moved the handle is
        still refused: the adapter was rebound on purpose, and answering out of
        a scene the caller replaced is the stale answer this cadence exists to
        prevent.
        """

        if frozen.epoch == self._epoch:
            return
        moved = consumer.rediscovery_required(self._compiled, frozen.topology)
        cause = f"{moved} changed" if moved is not None else "no world version domain moved, but the scene object did"
        raise ValueError(
            f"this frozen leg topology was frozen at adapter epoch "
            f"{frozen.epoch} and the adapter is now at epoch {self._epoch} "
            f"({cause}); refreeze() retires every frozen handle, so rediscover "
            "with freeze() before replaying"
        )

    def _slot_topology(
        self, frozen: FrozenLegTopology, sources: RadarEndpointSpec, sinks: RadarEndpointSpec, slot_count: int
    ) -> object:
        """The block-diagonal replication of a frozen handle, built once.

        The per-slot endpoint counts are passed explicitly because they cannot
        be inferred: an endpoint that discovered no row never appears in the
        frozen topology at all, so the largest index a topology carries is not
        the endpoint count, and guessing it would shift every slot after the
        first onto the wrong endpoints and still publish a full answer.
        """

        if slot_count == 1:
            return frozen.prepared
        source_count = _per_slot_count("sources", sources.count, slot_count)
        sink_count = _per_slot_count("sinks", sinks.count, slot_count)
        key = (slot_count, source_count, sink_count)
        cached = frozen.slot_topologies.get(key)
        if cached is None:
            cached = consumer.replicate_over_slots(
                frozen.prepared, slot_count, source_count=source_count, sink_count=sink_count
            )
            frozen.slot_topologies[key] = cached
        return cached


def _leg_batch(
    paths: object,
    sources: RadarEndpointSpec,
    sinks: RadarEndpointSpec,
    *,
    row_valid: torch.Tensor | None,
    diagnostics: object,
    slot_count: int,
) -> RadarLegBatch:
    """One consumer path batch in Radar vocabulary, discovered or replayed.

    The payload members ALIAS the consumer's tensors. The three segment
    endpoints are gathered from the endpoint specs and the interaction table:
    the first segment runs from the source to the first interaction (the sink
    on a line-of-sight row), the last from the last interaction (the source on
    a line-of-sight row) to the sink.
    """

    p: Any = paths
    geometry = p.geometry
    transport = p.transport
    source_positions = sources.positions_m.index_select(0, p.topology.source_index.long())
    sink_positions = sinks.positions_m.index_select(0, p.topology.sink_index.long())
    interactions = geometry.interaction_positions_m
    if interactions.shape[1]:
        reflected = (p.topology.depth > 0)[:, None]
        first = torch.where(reflected, interactions[:, 0], sink_positions)
        row = torch.arange(p.path_count, device=interactions.device)
        last = interactions[row, (p.topology.depth.long() - 1).clamp_min(0)]
        last = torch.where(reflected, last, source_positions)
    else:
        first, last = sink_positions, source_positions
    band = getattr(transport, "coefficient_offsets", None)
    return RadarLegBatch(
        leg_count=p.path_count,
        pair_count=p.pair_count,
        pair_index=p.pair_index,
        pair_offsets=p.pair_offsets,
        source_index=p.topology.source_index,
        sink_index=p.topology.sink_index,
        depth=p.topology.depth,
        component_id=p.topology.component_id,
        source_id=p.topology.source_id,
        sink_id=p.topology.sink_id,
        primitive_sequence=p.topology.primitive_sequence,
        material_sequence=p.topology.material_sequence,
        interaction_type=p.topology.interaction_type,
        delay_s=geometry.delay_s,
        coefficient=transport.coefficient,
        # Aliased, like every other payload member: an aspect-dependent
        # scatter response reads this straight out of the consumer's geometry,
        # so copying it would break the zero-copy discipline for the one
        # consumer that needs a gradient through a direction.
        field_direction=geometry.field_direction,
        row_valid=row_valid,
        diagnostics=diagnostics,
        slot_count=slot_count,
        departure_origin_m=source_positions,
        departure_target_m=first,
        arrival_origin_m=last,
        coefficient_band=band,
        frequency_offsets_hz=None if band is None else tuple(transport.frequency_offsets_hz),
    )


def _per_slot_count(name: str, stacked: int, slot_count: int) -> int:
    """How many endpoints one slot owns, refusing a ragged stack.

    A stack whose length is not a whole number of slots is not a short frame -
    it is a caller that built its slots and its endpoints from two different
    counts, and letting it through would silently reassign endpoints to slots
    from the point of the mismatch onward.
    """

    if stacked % slot_count:
        raise ValueError(
            f"{name} carries {stacked} endpoints, which is not divisible by "
            f"slot_count {slot_count}; a slot-major stack repeats the same "
            "endpoint set once per slot"
        )
    return stacked // slot_count


__all__ = ["ChannelPropagationAdapter", "FrozenLegTopology", "compile_scene", "require_reference_frequency"]
