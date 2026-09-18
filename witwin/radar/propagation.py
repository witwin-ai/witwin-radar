"""Radar-shaped propagation contracts.

These types are the Radar side of the Channel consumer boundary. They are
deliberately free of any ``witwin.channel`` import so that
``witwin.radar.propagation`` can be imported on a machine that has no
``witwin-channel`` installed; only :mod:`witwin.radar.channel`
reaches across the boundary.

A leg is one source-to-sink propagation segment. A radar round trip is composed
from two legs by :mod:`witwin.radar.paths`; this module has no opinion
about that composition.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from witwin.core.dynamics import DynamicScene

EndpointRole = Literal["source", "sink"]


def require_tensor(
    name: str, value: object, *, dtype: torch.dtype, ndim: int | None = None, shape: tuple[int, ...] | None = None
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}")
    if value.dtype != dtype:
        raise TypeError(f"{name} must use {dtype}, got {value.dtype}")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}, got {value.ndim}")
    if shape is not None and tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    return value


@dataclass(frozen=True, slots=True, eq=False)
class RadarEndpointSpec:
    """One batch of radar endpoints in world coordinates.

    ``positions_m`` is the only differentiable member. It may carry
    ``requires_grad`` for reverse mode or a forward-AD tangent for the
    ADR-038 forward-only dual; the remaining members are primal-only because
    the native field companions treat them as constants.

    Structural validation runs here and is device-agnostic on purpose: the
    CUDA requirement belongs to the Channel endpoint contract, so a caller
    gets the shape or dtype complaint it actually made rather than a device
    complaint that hides it.
    """

    stable_ids: torch.Tensor
    positions_m: torch.Tensor
    polarizations: torch.Tensor
    powers_w: torch.Tensor | None = None

    def __post_init__(self) -> None:
        positions = require_tensor("positions_m", self.positions_m, dtype=torch.float32, ndim=2)
        if positions.shape[1] != 3:
            raise ValueError(f"positions_m must have shape (N, 3), got {tuple(positions.shape)}")
        rows = int(positions.shape[0])
        require_tensor("stable_ids", self.stable_ids, dtype=torch.int64, shape=(rows,))
        require_tensor("polarizations", self.polarizations, dtype=torch.float32, shape=(rows, 3))
        if self.powers_w is not None:
            require_tensor("powers_w", self.powers_w, dtype=torch.float32, shape=(rows,))
        device = positions.device
        for name, value in (
            ("stable_ids", self.stable_ids),
            ("polarizations", self.polarizations),
            ("powers_w", self.powers_w),
        ):
            if value is not None and value.device != device:
                raise ValueError(f"{name} must share the positions_m device {device}, got {value.device}")

    @property
    def count(self) -> int:
        return int(self.positions_m.shape[0])

    @property
    def device(self) -> torch.device:
        return self.positions_m.device


def require_endpoint_role(spec: RadarEndpointSpec, role: EndpointRole) -> None:
    """Enforce the Channel source/sink power contract before any native work.

    A source radiates and therefore carries ``powers_w``; a sink receives and
    must not. Getting this wrong is rejected by the consumer anyway, but the
    Radar-side message names the leg endpoint the caller actually passed.
    """

    if role not in ("source", "sink"):
        raise ValueError(f"role must be 'source' or 'sink', got {role!r}")
    if role == "source" and spec.powers_w is None:
        raise ValueError("a source endpoint requires powers_w")
    if role == "sink" and spec.powers_w is not None:
        raise ValueError("a sink endpoint must not carry powers_w")


@dataclass(frozen=True, slots=True, eq=False)
class RadarLegBatch:
    """One reevaluated propagation leg in Radar vocabulary.

    ``delay_s`` and ``coefficient`` ALIAS the consumer tensors: same storage,
    same stride, same gradient state. Copying them would silently break the
    zero-copy discipline the compact contract exists to provide, so a change
    here has to preserve object identity.

    ``row_valid`` is the sole authority on whether a row's payload means
    anything. A dead row is a complete answer that this frozen path does not
    exist at these endpoint positions, contributing exactly zero; it is never
    an error and validity is never inferred from a zero payload.

    ``source_id``, ``sink_id``, ``primitive_sequence``, ``material_sequence``
    and ``interaction_type`` are the row's stable IDENTITY. They come straight
    off the frozen topology, so they are the same tensor objects on every frame
    of a frozen sequence and cost nothing to publish. A two-way composer joins
    on them; the sequences in particular are ADR-037 frozen labels rather than
    re-validated hits, which is exactly what makes them a stable key.

    ``field_direction`` is the row's PROPAGATION direction, a unit vector in
    world coordinates, aliased from the consumer's ``PropagationGeometry``. It
    is the direction of the row's FINAL segment, so it is the direction the
    field arrives at the sink travelling in - which for a line-of-sight row is
    also the direction it left the source in, and for a higher-order row is
    not. An aspect-dependent scatter response consumes it and is responsible
    for saying which of the two meanings it needs; see
    :mod:`witwin.radar.scattering`, which refuses an outbound leg whose
    rows are not line of sight rather than reading a departure direction off a
    row that does not carry one.

    ``slot_count`` states how many time slots - TDM slots, OFDM symbols or
    pulses - this batch carries. The default ``1`` is one instant and is what
    every single-shot reevaluation publishes. A batch with ``slot_count > 1``
    is SLOT MAJOR and FROZEN-ROW MINOR: row ``t * rows_per_slot + r`` is frozen
    row ``r`` at slot ``t``, and the pair partition is block diagonal, so pair
    ``t * pairs_per_slot + p`` is slot ``t``'s pair ``p``. That is the Channel
    consumer's ``slot_pair_layout``, restated here rather than reinvented; the
    whole point of the layout is that ``pair_count`` grows LINEARLY in the slot
    count instead of quadratically. :meth:`slot` is the only supported way to
    address one slot, so no consumer has to rederive the arithmetic.
    """

    leg_count: int
    pair_count: int
    pair_index: torch.Tensor
    pair_offsets: torch.Tensor
    source_index: torch.Tensor
    sink_index: torch.Tensor
    depth: torch.Tensor
    component_id: torch.Tensor
    source_id: torch.Tensor
    sink_id: torch.Tensor
    primitive_sequence: torch.Tensor
    material_sequence: torch.Tensor
    interaction_type: torch.Tensor
    delay_s: torch.Tensor
    coefficient: torch.Tensor
    field_direction: torch.Tensor
    row_valid: torch.Tensor | None
    diagnostics: object
    slot_count: int = 1
    departure_origin_m: torch.Tensor | None = None
    departure_target_m: torch.Tensor | None = None
    arrival_origin_m: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if type(self.leg_count) is not int or self.leg_count < 0:
            raise ValueError("leg_count must be a non-negative int")
        if type(self.pair_count) is not int or self.pair_count < 0:
            raise ValueError("pair_count must be a non-negative int")
        if type(self.slot_count) is not int or self.slot_count < 1:
            raise ValueError("slot_count must be a positive int")
        for name, total in (("leg_count", self.leg_count), ("pair_count", self.pair_count)):
            if total % self.slot_count:
                raise ValueError(
                    f"{name} {total} is not divisible by slot_count "
                    f"{self.slot_count}; a slot-major batch carries the same "
                    "frozen rows and the same pair partition in every slot"
                )
        rows = (self.leg_count,)
        require_tensor("pair_index", self.pair_index, dtype=torch.int64, shape=rows)
        require_tensor("pair_offsets", self.pair_offsets, dtype=torch.int64, shape=(self.pair_count + 1,))
        for name in ("source_index", "sink_index", "depth", "component_id"):
            require_tensor(name, getattr(self, name), dtype=torch.int32, shape=rows)
        for name in ("source_id", "sink_id"):
            require_tensor(name, getattr(self, name), dtype=torch.int64, shape=rows)
        width = (
            int(self.primitive_sequence.shape[1])
            if (isinstance(self.primitive_sequence, torch.Tensor) and self.primitive_sequence.ndim == 2)
            else -1
        )
        if width < 0:
            raise ValueError("primitive_sequence must have shape (rows, width)")
        for name in ("primitive_sequence", "material_sequence", "interaction_type"):
            require_tensor(name, getattr(self, name), dtype=torch.int32, shape=(self.leg_count, width))
        require_tensor("delay_s", self.delay_s, dtype=torch.float32, shape=rows)
        require_tensor("coefficient", self.coefficient, dtype=torch.complex64, shape=rows)
        require_tensor("field_direction", self.field_direction, dtype=torch.float32, shape=(self.leg_count, 3))
        if self.row_valid is not None:
            require_tensor("row_valid", self.row_valid, dtype=torch.bool, shape=rows)
        for name in ("departure_origin_m", "departure_target_m", "arrival_origin_m"):
            if getattr(self, name) is not None:
                require_tensor(name, getattr(self, name), dtype=torch.float32, shape=(self.leg_count, 3))

    @property
    def device(self) -> torch.device:
        return self.delay_s.device

    @property
    def rows_per_slot(self) -> int:
        return self.leg_count // self.slot_count

    @property
    def pairs_per_slot(self) -> int:
        return self.pair_count // self.slot_count

    def slot(self, index: int) -> RadarLegBatch:
        """One slot of a slot-major batch, as a single-slot batch.

        The payload members are NARROWED, so ``delay_s``, ``coefficient``,
        ``field_direction`` and ``row_valid`` still alias the batched storage
        and a gradient flows straight back through them. Only the two partition
        tables are rebased, because a slot's pair ranks have to start at zero
        for the slice to be a partition of that slot's rows; rebasing them is
        int64 metadata arithmetic and reads no payload value.

        This exists so that a consumer written against the single-slot contract
        - the two-way join, in particular - can be driven per slot WITHOUT a
        second statement of the block-diagonal layout living in the caller.
        """

        if type(index) is not int or not 0 <= index < self.slot_count:
            raise ValueError(f"slot index must be an int in [0, {self.slot_count}), got {index!r}")
        rows = self.rows_per_slot
        pairs = self.pairs_per_slot
        start = index * rows
        stop = start + rows
        base = index * pairs

        def narrow(value):
            return None if value is None else value[start:stop]

        return RadarLegBatch(
            leg_count=rows,
            pair_count=pairs,
            pair_index=self.pair_index[start:stop] - base,
            pair_offsets=(self.pair_offsets[base : base + pairs + 1] - self.pair_offsets[base]),
            source_index=narrow(self.source_index),
            sink_index=narrow(self.sink_index),
            depth=narrow(self.depth),
            component_id=narrow(self.component_id),
            source_id=narrow(self.source_id),
            sink_id=narrow(self.sink_id),
            primitive_sequence=narrow(self.primitive_sequence),
            material_sequence=narrow(self.material_sequence),
            interaction_type=narrow(self.interaction_type),
            delay_s=narrow(self.delay_s),
            coefficient=narrow(self.coefficient),
            field_direction=narrow(self.field_direction),
            row_valid=narrow(self.row_valid),
            diagnostics=self.diagnostics,
            slot_count=1,
            departure_origin_m=narrow(self.departure_origin_m),
            departure_target_m=narrow(self.departure_target_m),
            arrival_origin_m=narrow(self.arrival_origin_m),
        )


@dataclass(frozen=True, slots=True, eq=False)
class RadarPropagationLegs:
    """The two legs of one radar round trip, as ONE typed value.

    A radar frame evaluates two legs - transmitter to scatter site, scatter
    site to receiver - and every consumer of the pair has to know which is
    which. A tuple says it by position and a dict says it by a string key, and
    both let a caller hand the outbound leg to something expecting the inbound
    one; the two legs have the same type and the same shape family, so nothing
    downstream would notice. This type is what makes the pairing checkable, and
    it is why ``Result.last_propagation`` is not a tuple.

    It is a VIEW, not a copy: both members are the batches the adapter
    published, so every payload tensor still aliases the consumer's storage and
    keeps its gradient state. Nothing here reads a tensor value, so
    constructing it costs no launch, no allocation and no transfer.

    The two legs of one frame are evaluated at one world instant on one device,
    and a pair that disagrees about either is not a round trip. Both are checked
    on the host from members the batches already publish.
    """

    inbound: RadarLegBatch
    outbound: RadarLegBatch

    def __post_init__(self) -> None:
        for name in ("inbound", "outbound"):
            value = getattr(self, name)
            if not isinstance(value, RadarLegBatch):
                raise TypeError(f"{name} must be a RadarLegBatch, got {type(value).__name__}")
        if self.inbound.slot_count != self.outbound.slot_count:
            raise ValueError(
                f"the inbound leg carries {self.inbound.slot_count} slots and "
                f"the outbound leg {self.outbound.slot_count}; the two legs of "
                "one frame are the same slow-time axis"
            )
        if self.inbound.device != self.outbound.device:
            raise ValueError(
                f"the inbound leg is on {self.inbound.device} and the outbound "
                f"leg on {self.outbound.device}; one round trip is one device"
            )

    @property
    def slot_count(self) -> int:
        return self.inbound.slot_count

    @property
    def device(self) -> torch.device:
        return self.inbound.device


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

    ``topology_complete`` is the caller's standing certification, as HONOURED
    for this frame: the candidate family is complete for ALL time, so no path
    can be born between two observations however far apart they are. That is
    strictly stronger than ``rediscovered``, which enumerates the family at one
    instant and says nothing about the gap to the next one. It is published
    because a consumer that skipped observations cannot otherwise distinguish
    "the topology was proven complete and only motion was interpolated" from "a
    short-lived path may have been missed". ``False`` is the absence of a
    proof, never a proof of incompleteness.
    """

    time_s: float
    snapshot: object
    compiled: object
    epoch: int
    frozen: FrozenEpoch
    rediscovered: bool
    reason: str | None
    topology_complete: bool = False


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
        if not isinstance(dynamic_scene, DynamicScene):
            raise TypeError(
                f"dynamic_scene must be a witwin.core.dynamics.DynamicScene, got {type(dynamic_scene).__name__}"
            )
        if not callable(compile_scene):
            raise TypeError("compile_scene must be callable; pass witwin.channel.scene.compile")
        if motion_event_period_frames is not None and (
            type(motion_event_period_frames) is not int or motion_event_period_frames < 1
        ):
            raise ValueError(
                f"motion_event_period_frames must be a positive int or None, got {motion_event_period_frames!r}"
            )
        self._dynamic = dynamic_scene
        self._reference_frequency_hz = float(reference_frequency_hz)
        self._bind = bind
        self._period = motion_event_period_frames
        self._world_motion = str(world_motion)
        self._compile_scene = compile_scene
        self._structures_move = bool(dynamic_scene.structure_trajectories or dynamic_scene.structure_deformations)
        self._frozen: FrozenEpoch | None = None
        self._compiled: object | None = None
        self._frame_index = -1
        self._epoch = -1
        self._last_discovery_frame = -1
        self.compile_count = 0
        self.discovery_count = 0
        self.poll_count = 0
        self.revalidation_count = 0

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

    def frame(self, time_s: float, *, topology_complete: bool = False) -> EpochFrame:
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
        # A caller may certify that all possible rows already exist (e.g. the
        # Cartesian LOS family in an empty world). This suppresses only the
        # motion cadence, never source mutation, retirement or version checks.
        reason = self._rediscovery_reason(recompiled, mutated, topology_complete)
        if reason is not None:
            self._rediscover(snapshot)
        return EpochFrame(
            time_s=float(time_s),
            snapshot=snapshot,
            compiled=self._compiled,
            epoch=self._epoch,
            frozen=self._frozen,
            rediscovered=reason is not None,
            reason=reason,
            topology_complete=bool(topology_complete),
        )

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
        return self._period is not None and (self._frame_index - self._last_discovery_frame >= self._period)

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
            self._frozen.adapter.refreeze(self._compiled, world_motion=self._world_motion)
        return True

    def _rediscovery_reason(self, recompiled: bool, mutated: bool, topology_complete: bool = False) -> str | None:
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
        if self._motion_event_due() and not topology_complete:
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

    def _rediscover(self, snapshot: object) -> None:
        previous = self._frozen
        frozen = self._bind(self._compiled, snapshot, previous)
        if not isinstance(frozen, FrozenEpoch):
            raise TypeError(f"bind must return a FrozenEpoch, got {type(frozen).__name__}")
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

    def _compile(self, snapshot: object) -> object:
        return self._compile_scene(snapshot, reference_frequency_hz=self._reference_frequency_hz)


def _require_positions(name: str, value: object) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}")
    if value.ndim != 2 or int(value.shape[1]) != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {tuple(value.shape)}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} must use torch.float32, got {value.dtype}")
    return value


def _vector3(name: str, value: object, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """One world-frame 3-vector, or zeros when the caller declared nothing.

    ``None`` means "this quantity is not part of the motion", which is a
    statement about the world and not a missing argument, so it resolves to a
    real zero vector rather than raising. An endpoint with no ``RigidMotion``
    has no translation.
    """

    if value is None:
        return torch.zeros(3, dtype=dtype, device=device)
    tensor = value if isinstance(value, torch.Tensor) else torch.tensor(value, dtype=dtype, device=device)
    if tuple(tensor.shape) != (3,):
        raise ValueError(f"{name} must have shape (3,), got {tuple(tensor.shape)}")
    return tensor.to(device=device, dtype=dtype)


@dataclass(frozen=True, slots=True, eq=False)
class Kinematics:
    """One ordered endpoint set's world positions at one instant.

    ``positions_m`` is ``(N, 3)`` float32 and contiguous, which is the Channel
    endpoint contract restated at the point where the tensor is BUILT rather
    than at the point where it is rejected. There is no velocity member: the
    delay rate comes from the propagation solve at each observation instant,
    never from a site velocity.
    """

    positions_m: torch.Tensor

    def __post_init__(self) -> None:
        positions = _require_positions("positions_m", self.positions_m)
        if not positions.is_contiguous():
            raise ValueError("Kinematics.positions_m must be contiguous")

    @property
    def count(self) -> int:
        return int(self.positions_m.shape[0])

    @property
    def device(self) -> torch.device:
        return self.positions_m.device


def endpoint_kinematics(
    snapshot_or_states, antenna_ids: Sequence[int] | None = None, *, device: str | torch.device = "cuda"
) -> Kinematics:
    """The positions of an ordered set of Core endpoint states.

    ``snapshot_or_states`` is a ``SceneSnapshot`` or any sequence of
    ``EndpointState``. ``antenna_ids`` declares the ENDPOINT BATCH ORDER: it is
    the order the positions are built in, and it is the order the Channel leg
    rows will name. Omitting it keeps the snapshot's own declaration order,
    which is fine for a single caller and wrong the moment two callers
    disagree, so a batch that will be joined by identity should always declare
    it.

    Position resolution follows Core's own composition: the authored antenna
    position plus the snapshot's additional world-frame ``translation``. An
    endpoint's ``rotation`` is orientation and does not move its phase centre.
    """

    states = getattr(snapshot_or_states, "endpoints", snapshot_or_states)
    ordered = list(states)
    if antenna_ids is not None:
        by_id = {int(state.antenna.antenna_id): state for state in ordered}
        missing = [stable_id for stable_id in antenna_ids if int(stable_id) not in by_id]
        if missing:
            raise KeyError(f"the snapshot declares no endpoint for antenna IDs {missing}; it carries {sorted(by_id)}")
        ordered = [by_id[int(stable_id)] for stable_id in antenna_ids]
    if not ordered:
        raise ValueError("endpoint_kinematics requires at least one endpoint")

    resolved = torch.device(device)
    positions = []
    for state in ordered:
        motion = getattr(state, "rigid_motion", None)
        position = state.antenna.position.to(device=resolved, dtype=torch.float32)
        if tuple(position.shape) != (3,):
            raise ValueError(f"antenna position must have shape (3,), got {tuple(position.shape)}")
        translation = _vector3("translation", None if motion is None else motion.translation, resolved, torch.float32)
        positions.append(position + translation)
    return Kinematics(positions_m=torch.stack(positions).contiguous())


__all__ = [
    "EndpointRole",
    "EpochFrame",
    "FrozenEpoch",
    "Kinematics",
    "RadarEndpointSpec",
    "RadarLegBatch",
    "RadarPropagationLegs",
    "SceneEpochLoop",
    "endpoint_kinematics",
    "require_endpoint_role",
]
