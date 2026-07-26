"""Radar's adapter onto the stable Channel propagation consumer.

This is the ONLY Radar module that imports ``witwin.channel``, and it imports
exactly one thing from it: the solver-neutral consumer facade
``witwin.channel.propagation.consumer``. Scene compilation goes through
``witwin.channel.scene.compile``, which the spike calls from ``tests/support``
rather than from any Radar module. Nothing here touches a Channel solver, the
enumerated engine, the internal propagation contracts, or the native extension.
R-ADR-001 records why: Radar is a consumer of the published contract, not a
second enumerated exception to ADR-008.

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
* publishing Radar-shaped leg results  -  delay, coefficient, delay rate.

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

import torch
import torch.autograd.forward_ad as forward_ad

from witwin.channel.propagation import consumer

from .contracts import RadarEndpointSpec, RadarLegBatch, require_endpoint_role


@dataclass(frozen=True, slots=True, eq=False)
class FrozenLegTopology:
    """A discovered leg topology, partitioned once for repeated reevaluation.

    The recorded host-observation counters describe the ONE-TIME freeze, not a
    later :meth:`ChannelPropagationAdapter.reevaluate` call. They are reported
    separately for exactly that reason: preparing the handle once per frozen
    topology is the contract, and preparing it per frame gives up the whole
    point of the capability.

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

    prepared: object
    source_id: torch.Tensor
    sink_id: torch.Tensor
    component_id: torch.Tensor
    depth: torch.Tensor
    primitive_sequence: torch.Tensor
    material_sequence: torch.Tensor
    components: tuple[str, ...]
    row_count: int
    prepare_d2h_copies: int
    prepare_d2h_bytes: int
    prepare_synchronizations: int
    epoch: int = 0
    slot_topologies: dict = field(default_factory=dict, repr=False)


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


class ChannelPropagationAdapter:
    """Radar-facing view of one compiled Channel scene at one frequency."""

    def __init__(
        self,
        compiled_scene: object,
        *,
        reference_frequency_hz: float,
        components: frozenset[str],
        max_depth: int,
    ) -> None:
        capabilities = consumer.capabilities()
        unsupported = frozenset(components) - capabilities.components
        if unsupported:
            raise NotImplementedError(
                f"the propagation consumer does not support components "
                f"{sorted(unsupported)}"
            )
        not_freezable = frozenset(components) - capabilities.fixed_topology_components
        if not_freezable:
            raise NotImplementedError(
                f"components {sorted(not_freezable)} cannot be frozen for "
                f"reevaluation; supported components are "
                f"{sorted(capabilities.fixed_topology_components)}"
            )
        self._compiled = compiled_scene
        self._reference_frequency_hz = float(reference_frequency_hz)
        self._components = frozenset(components)
        self._max_depth = int(max_depth)
        self._capabilities = capabilities
        self._epoch = 0
        self._world_motion = "frozen_world"

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

    def refreeze(
        self, compiled_scene: object, *, world_motion: str = "frozen_world"
    ) -> None:
        """Rebind to a new compiled scene, declaring what moved.

        A moving structure, a deformed mesh or a new ``DynamicScene`` snapshot
        produces a NEW ``CompiledScene``, and until this existed the adapter
        held the scene it was constructed with for its whole lifetime - so a
        moving-structure world replayed frozen rows against geometry that had
        moved on, silently and at full strength.

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
          (:class:`witwin.radar.propagation.epochs.SceneEpochLoop` is that
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
            raise ValueError(
                f"unsupported world_motion {world_motion!r}; supported values "
                f"are {sorted(supported)}"
            )
        self._compiled = compiled_scene
        self._world_motion = world_motion
        if world_motion == "frozen_world":
            self._epoch += 1

    def rediscovery_required(
        self, frozen: FrozenLegTopology, *, revalidate_source: bool = False
    ) -> str | None:
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

        return consumer.rediscovery_required(
            self._compiled, frozen.prepared, revalidate_source=revalidate_source
        )

    def freeze(
        self, sources: RadarEndpointSpec, sinks: RadarEndpointSpec
    ) -> FrozenLegTopology:
        """Discover one leg's topology and partition it once.

        Call this OUTSIDE the per-frame loop. It runs full discovery and it
        synchronizes; calling it per frame would reintroduce exactly the host
        observation the fixed-topology capability exists to avoid.
        """

        evaluation = consumer.evaluate(
            self._compiled,
            consumer.PropagationRequest(
                sources=_endpoint_batch(_detached(sources), "source"),
                sinks=_endpoint_batch(_detached(sinks), "sink"),
                reference_frequency_hz=self._reference_frequency_hz,
                components=self._components,
                max_depth=self._max_depth,
                response="scalar_transport",
                topology_mode="discover",
                ad_mode="none",
            ),
        )
        topology = evaluation.paths.topology
        prepared = consumer.prepare_fixed_topology(topology)
        components = tuple(bucket.component for bucket in prepared.buckets)
        unsupported = frozenset(components) - self._capabilities.fixed_topology_components
        if unsupported:
            raise NotImplementedError(
                f"the discovered topology carries components {sorted(unsupported)} "
                f"that cannot be reevaluated"
            )
        return FrozenLegTopology(
            prepared=prepared,
            source_id=topology.source_id,
            sink_id=topology.sink_id,
            component_id=topology.component_id,
            depth=topology.depth,
            primitive_sequence=topology.primitive_sequence,
            material_sequence=topology.material_sequence,
            components=components,
            row_count=evaluation.paths.path_count,
            prepare_d2h_copies=prepared.prepare_d2h_copies,
            prepare_d2h_bytes=prepared.prepare_d2h_bytes,
            prepare_synchronizations=prepared.prepare_synchronizations,
            epoch=self._epoch,
        )

    def reevaluate(
        self,
        frozen: FrozenLegTopology,
        sources: RadarEndpointSpec,
        sinks: RadarEndpointSpec,
        *,
        ad_mode: str,
    ) -> RadarLegBatch:
        """Replay a frozen leg at new endpoint positions.

        Exactly one consumer call per invocation. No discovery, no preparation,
        no second cardinality observation.
        """

        return self._replay(frozen, sources, sinks, ad_mode=ad_mode, slot_count=1)

    def reevaluate_slots(
        self,
        frozen: FrozenLegTopology,
        sources: RadarEndpointSpec,
        sinks: RadarEndpointSpec,
        *,
        slot_count: int,
        ad_mode: str,
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
        """

        return self._replay(
            frozen, sources, sinks, ad_mode=ad_mode, slot_count=slot_count
        )

    def _replay(
        self,
        frozen: FrozenLegTopology,
        sources: RadarEndpointSpec,
        sinks: RadarEndpointSpec,
        *,
        ad_mode: str,
        slot_count: int,
    ) -> RadarLegBatch:
        if ad_mode not in consumer.AD_MODES:
            raise NotImplementedError(
                f"unsupported ad_mode {ad_mode!r}; supported values are "
                f"{sorted(consumer.AD_MODES)}"
            )
        if type(slot_count) is not int or slot_count < 1:
            raise ValueError(
                f"slot_count must be a positive int, got {slot_count!r}"
            )
        self._require_current_epoch(frozen)
        topology = self._slot_topology(frozen, sources, sinks, slot_count)
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
            ),
        )
        paths = result.paths
        geometry = paths.geometry
        return RadarLegBatch(
            leg_count=paths.path_count,
            pair_count=paths.pair_count,
            pair_index=paths.pair_index,
            pair_offsets=paths.pair_offsets,
            source_index=paths.topology.source_index,
            sink_index=paths.topology.sink_index,
            depth=paths.topology.depth,
            component_id=paths.topology.component_id,
            source_id=paths.topology.source_id,
            sink_id=paths.topology.sink_id,
            primitive_sequence=paths.topology.primitive_sequence,
            material_sequence=paths.topology.material_sequence,
            interaction_type=paths.topology.interaction_type,
            delay_s=geometry.delay_s,
            coefficient=paths.transport.coefficient,
            delay_rate=_delay_rate(geometry.delay_s, ad_mode),
            row_valid=result.row_valid,
            diagnostics=result.diagnostics,
            slot_count=slot_count,
            # Aliased, like every other payload member: an aspect-dependent
            # scatter response reads this straight out of the consumer's
            # geometry, so copying it would break the zero-copy discipline for
            # the one consumer that needs a gradient through a direction.
            field_direction=geometry.field_direction,
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
        moved = consumer.rediscovery_required(self._compiled, frozen.prepared)
        cause = (
            f"{moved} changed"
            if moved is not None
            else "no world version domain moved, but the scene object did"
        )
        raise ValueError(
            f"this frozen leg topology was frozen at adapter epoch "
            f"{frozen.epoch} and the adapter is now at epoch {self._epoch} "
            f"({cause}); refreeze() retires every frozen handle, so rediscover "
            "with freeze() before replaying"
        )

    def _slot_topology(
        self,
        frozen: FrozenLegTopology,
        sources: RadarEndpointSpec,
        sinks: RadarEndpointSpec,
        slot_count: int,
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
                frozen.prepared,
                slot_count,
                source_count=source_count,
                sink_count=sink_count,
            )
            frozen.slot_topologies[key] = cached
        return cached


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


def _delay_rate(delay_s: torch.Tensor, ad_mode: str) -> torch.Tensor | None:
    """Unpack the forward-only delay tangent that carries Doppler.

    The tangent storage belongs to the enclosing dual level and is not
    guaranteed valid once that level exits, so it is cloned here, inside the
    level. That clone is O(K) metadata, not physics.

    A missing tangent is a hard error. Publishing ``delay_rate = 0`` as a
    stand-in would look exactly like a stationary scene and would be
    indistinguishable from a correct answer.
    """

    if ad_mode != "jvp":
        return None
    tangent = forward_ad.unpack_dual(delay_s).tangent
    if tangent is None:
        raise RuntimeError(
            "ad_mode='jvp' produced no delay_s tangent; the caller must build "
            "its endpoint positions as forward-AD duals inside an active "
            "torch.autograd.forward_ad.dual_level()"
        )
    return tangent.clone()


__all__ = ["ChannelPropagationAdapter", "FrozenLegTopology"]
