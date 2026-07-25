"""Radar's adapter onto the stable Channel propagation consumer.

This is the ONLY Radar module that imports ``witwin.channel``, and it imports
exactly one thing from it: the solver-neutral consumer facade
``witwin.channel.propagation.consumer``. Scene compilation goes through
``witwin.channel.scene.compile``, which the spike calls from ``tests/support``
rather than from any Radar module. Nothing here touches a Channel solver, the
enumerated engine, the internal propagation contracts, or the native extension.
R-ADR-001 records why: Radar is a consumer of the published contract, not a
second enumerated exception to ADR-008.

The adapter owns three things and nothing else:

* endpoint batching  -  turning Radar endpoint specs into consumer batches;
* the freeze/reevaluate split  -  discovery happens once per frozen topology,
  and every later frame replays it with exactly one consumer call;
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

from dataclasses import dataclass

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

    @property
    def reference_frequency_hz(self) -> float:
        return self._reference_frequency_hz

    @property
    def capabilities(self) -> object:
        return self._capabilities

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

        if ad_mode not in consumer.AD_MODES:
            raise NotImplementedError(
                f"unsupported ad_mode {ad_mode!r}; supported values are "
                f"{sorted(consumer.AD_MODES)}"
            )
        result = consumer.reevaluate(
            self._compiled,
            consumer.FixedTopologyRequest(
                sources=_endpoint_batch(sources, "source"),
                sinks=_endpoint_batch(sinks, "sink"),
                reference_frequency_hz=self._reference_frequency_hz,
                topology=frozen.prepared,
                response="scalar_transport",
                ad_mode=ad_mode,
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
        )


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
