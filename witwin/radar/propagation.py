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

import contextlib
from dataclasses import dataclass
from typing import Callable, Iterator, Literal, Protocol, Sequence, runtime_checkable

import torch
import torch.autograd.forward_ad as forward_ad

EndpointRole = Literal["source", "sink"]


def _require_tensor(
    name: str,
    value: object,
    *,
    dtype: torch.dtype,
    ndim: int | None = None,
    shape: tuple[int, ...] | None = None,
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


def require_wideband_pair(
    frequency_response: object,
    frequency_offsets_hz: object,
    row_count: int,
) -> int:
    """Validate a ``[rows, F]`` response against its ``[F]`` grid, host only.

    One statement in two tensors: a response without its grid names no
    frequencies, and a grid without a response promises a band that is not
    there. Both are refused rather than defaulted, because either default would
    be a guess about which frequencies a caller meant.

    This is the SAME rule ``SynthesisPathBatch`` already enforces on its own
    pair, hoisted here so that the leg batch, the composed batch and the
    synthesis batch cannot drift apart. It reads no tensor VALUE - only shapes,
    dtypes, contiguity and device - so it costs no transfer and no
    synchronization anywhere on the frame path.

    Returns the band count, ``0`` when the pair is absent.
    """

    if (frequency_response is None) != (frequency_offsets_hz is None):
        raise ValueError(
            "frequency_response and frequency_offsets_hz are one statement and "
            "must be supplied together; a response without its frequency grid "
            "says nothing, and a grid without a response promises a band that "
            "was never evaluated"
        )
    if frequency_response is None:
        return 0
    if not isinstance(frequency_offsets_hz, torch.Tensor):
        raise TypeError(
            "frequency_offsets_hz must be a torch.Tensor, got "
            f"{type(frequency_offsets_hz).__name__}"
        )
    if frequency_offsets_hz.ndim != 1:
        raise ValueError("frequency_offsets_hz must have shape (F,)")
    bands = int(frequency_offsets_hz.shape[0])
    if bands < 1:
        raise ValueError(
            "frequency_offsets_hz must declare at least one column; an empty "
            "grid is a narrowband batch, which is spelled with both members None"
        )
    _require_tensor(
        "frequency_response",
        frequency_response,
        dtype=torch.complex64,
        shape=(row_count, bands),
    )
    _require_tensor(
        "frequency_offsets_hz",
        frequency_offsets_hz,
        dtype=torch.float32,
        shape=(bands,),
    )
    if frequency_response.device != frequency_offsets_hz.device:
        raise ValueError(
            f"frequency_response is on {frequency_response.device} but its grid "
            f"is on {frequency_offsets_hz.device}; a band is single-device"
        )
    return bands


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
        positions = _require_tensor(
            "positions_m", self.positions_m, dtype=torch.float32, ndim=2
        )
        if positions.shape[1] != 3:
            raise ValueError(
                f"positions_m must have shape (N, 3), got {tuple(positions.shape)}"
            )
        rows = int(positions.shape[0])
        _require_tensor(
            "stable_ids", self.stable_ids, dtype=torch.int64, shape=(rows,)
        )
        _require_tensor(
            "polarizations", self.polarizations, dtype=torch.float32, shape=(rows, 3)
        )
        if self.powers_w is not None:
            _require_tensor(
                "powers_w", self.powers_w, dtype=torch.float32, shape=(rows,)
            )
        device = positions.device
        for name, value in (
            ("stable_ids", self.stable_ids),
            ("polarizations", self.polarizations),
            ("powers_w", self.powers_w),
        ):
            if value is not None and value.device != device:
                raise ValueError(
                    f"{name} must share the positions_m device {device}, "
                    f"got {value.device}"
                )

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

    ``delay_rate`` is ``d(delay_s)/dt`` in seconds per second, unpacked from a
    forward-only dual and published as a PRIMAL value. It is the Doppler
    primitive; consuming it as a primal deliberately severs the second-order
    ``d(delay_rate)/dx`` term, which this contract does not claim.

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

    It is optional only because a fabricated leg row - a test that builds a
    batch by hand to reach a validation path - has no geometry behind it. Every
    batch the adapter publishes carries it, and a consumer that needs it
    refuses a batch without it by name rather than inventing one.

    ``frequency_response`` and ``frequency_offsets_hz`` are ONE statement and
    are present or absent together. The response is ``[leg_count, F]``
    complex64: column ``j`` is this row's transport evaluated at
    ``reference_frequency_hz + frequency_offsets_hz[j]``, published by the
    Channel consumer's ADR-042 wideband route. It is not a narrowband
    coefficient shifted by the offset law - the material response, the
    ``lambda/(4*pi*d)`` spreading and the layer-stack fringes are all evaluated
    natively at the column's own frequency. Column ``j`` with
    ``frequency_offsets_hz[j] == 0`` is BIT-IDENTICAL to ``coefficient``, which
    is what makes a wideband batch a strict superset of a narrowband one.

    The grid is a ``[F]`` float32 tensor on the batch device, aliased from the
    adapter, which builds it once per declared grid rather than per frame. Row
    validity is deliberately NOT widened: ``row_valid`` stays ``[leg_count]``
    and is broadcast over the band, because whether a stationary point exists
    is a geometric fact about the endpoints and cannot depend on frequency.

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
    delay_rate: torch.Tensor | None
    row_valid: torch.Tensor | None
    diagnostics: object
    slot_count: int = 1
    field_direction: torch.Tensor | None = None
    frequency_response: torch.Tensor | None = None
    frequency_offsets_hz: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if type(self.leg_count) is not int or self.leg_count < 0:
            raise ValueError("leg_count must be a non-negative int")
        if type(self.pair_count) is not int or self.pair_count < 0:
            raise ValueError("pair_count must be a non-negative int")
        if type(self.slot_count) is not int or self.slot_count < 1:
            raise ValueError("slot_count must be a positive int")
        for name, total in (
            ("leg_count", self.leg_count),
            ("pair_count", self.pair_count),
        ):
            if total % self.slot_count:
                raise ValueError(
                    f"{name} {total} is not divisible by slot_count "
                    f"{self.slot_count}; a slot-major batch carries the same "
                    "frozen rows and the same pair partition in every slot"
                )
        rows = (self.leg_count,)
        _require_tensor("pair_index", self.pair_index, dtype=torch.int64, shape=rows)
        _require_tensor(
            "pair_offsets",
            self.pair_offsets,
            dtype=torch.int64,
            shape=(self.pair_count + 1,),
        )
        for name in ("source_index", "sink_index", "depth", "component_id"):
            _require_tensor(
                name, getattr(self, name), dtype=torch.int32, shape=rows
            )
        for name in ("source_id", "sink_id"):
            _require_tensor(
                name, getattr(self, name), dtype=torch.int64, shape=rows
            )
        width = int(self.primitive_sequence.shape[1]) if (
            isinstance(self.primitive_sequence, torch.Tensor)
            and self.primitive_sequence.ndim == 2
        ) else -1
        if width < 0:
            raise ValueError("primitive_sequence must have shape (rows, width)")
        for name in (
            "primitive_sequence",
            "material_sequence",
            "interaction_type",
        ):
            _require_tensor(
                name,
                getattr(self, name),
                dtype=torch.int32,
                shape=(self.leg_count, width),
            )
        _require_tensor("delay_s", self.delay_s, dtype=torch.float32, shape=rows)
        _require_tensor(
            "coefficient", self.coefficient, dtype=torch.complex64, shape=rows
        )
        if self.delay_rate is not None:
            _require_tensor(
                "delay_rate", self.delay_rate, dtype=torch.float32, shape=rows
            )
        if self.row_valid is not None:
            _require_tensor(
                "row_valid", self.row_valid, dtype=torch.bool, shape=rows
            )
        if self.field_direction is not None:
            _require_tensor(
                "field_direction",
                self.field_direction,
                dtype=torch.float32,
                shape=(self.leg_count, 3),
            )
        require_wideband_pair(
            self.frequency_response, self.frequency_offsets_hz, self.leg_count
        )

    @property
    def device(self) -> torch.device:
        return self.delay_s.device

    @property
    def band_count(self) -> int:
        """How many frequency columns this batch carries, ``0`` when narrowband.

        A host int, so a consumer can size a loop over the band without
        touching the device. ``0`` and ``1`` are deliberately different: one
        column at ``df = 0`` is a declared single-frequency band, while ``0``
        says no band was requested at all.
        """

        if self.frequency_offsets_hz is None:
            return 0
        return int(self.frequency_offsets_hz.shape[0])

    @property
    def rows_per_slot(self) -> int:
        return self.leg_count // self.slot_count

    @property
    def pairs_per_slot(self) -> int:
        return self.pair_count // self.slot_count

    def slot(self, index: int) -> "RadarLegBatch":
        """One slot of a slot-major batch, as a single-slot batch.

        The payload members are NARROWED, so ``delay_s``, ``coefficient``,
        ``delay_rate`` and ``row_valid`` still alias the batched storage and a
        gradient flows straight back through them. Only the two partition
        tables are rebased, because a slot's pair ranks have to start at zero
        for the slice to be a partition of that slot's rows; rebasing them is
        int64 metadata arithmetic and reads no payload value.

        This exists so that a consumer written against the single-slot contract
        - the two-way join, in particular - can be driven per slot WITHOUT a
        second statement of the block-diagonal layout living in the caller.
        """

        if type(index) is not int or not 0 <= index < self.slot_count:
            raise ValueError(
                f"slot index must be an int in [0, {self.slot_count}), "
                f"got {index!r}"
            )
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
            pair_offsets=(
                self.pair_offsets[base : base + pairs + 1]
                - self.pair_offsets[base]
            ),
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
            delay_rate=narrow(self.delay_rate),
            row_valid=narrow(self.row_valid),
            diagnostics=self.diagnostics,
            slot_count=1,
            field_direction=narrow(self.field_direction),
            # Rows narrow, the band does not: the grid is a declaration shared
            # by every slot, so it is aliased rather than sliced.
            frequency_response=narrow(self.frequency_response),
            frequency_offsets_hz=self.frequency_offsets_hz,
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
    it is why ``Radar.last_propagation`` is not a tuple.

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
                raise TypeError(
                    f"{name} must be a RadarLegBatch, got {type(value).__name__}"
                )
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

@runtime_checkable
class DeformationVelocity(Protocol):
    """An analytic ``d(vertices)/dt`` for a deforming structure.

    Core's ``DeformationState`` states WHERE the vertices are and never how fast
    they are moving, so a deforming mesh has no time derivative anywhere in
    Core. Production finite differences are forbidden, which leaves exactly one
    supported route: the descriptor that produced the deformation also states
    its rate in closed form.

    ``velocity_at`` returns a ``(V, 3)`` world-frame velocity in metres per
    second, one row per authored vertex, in authored vertex order. A caller that
    tracks a subset of vertices as scatter sites index-selects from it; see
    :func:`deformation_kinematics`.

    This is a Radar-side contract because the gap it fills is a Core gap that
    Radar is not permitted to patch. If Core later grows a velocity descriptor
    on ``DeformationState``, an adapter implementing this protocol over it is
    the whole migration.
    """

    def velocity_at(self, time_s: float) -> torch.Tensor:
        ...


def _require_positions(name: str, value: object) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}")
    if value.ndim != 2 or int(value.shape[1]) != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {tuple(value.shape)}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} must use torch.float32, got {value.dtype}")
    return value


def _require_velocity_is_not_a_leaf(name: str, owner: str, value: torch.Tensor):
    """Refuse a velocity that carries a derivative, before anything is built.

    Under ADR-038 a velocity in this module is a forward-AD TANGENT DIRECTION,
    never a leaf. ``make_dual(position, velocity)`` puts it in the tangent slot,
    and a tangent is consumed by the forward pass and never differentiated:
    there is no ``d(loss)/d(velocity)`` for autograd to return, in either mode.
    Accepting a marked velocity would therefore hand the caller ``grad = None``
    - or, worse, a plausible-looking zero - for a quantity the whole module is
    named after.

    Two shapes reach here:

    * a velocity the caller marked directly, which is the request this refuses;
    * a velocity DERIVED from a grad-carrying position, which
      :func:`rigid_site_velocities` produces because ``omega x (p - c)`` is a
      differentiable expression of ``p``. That one is the same defect one step
      removed: the derived tangent would carry a graph back to the position
      leaf, autograd would never traverse it, and the position gradient the
      caller actually wanted would arrive short by the Doppler term with nothing
      to say so. The fix is to derive the velocity from a detached copy of the
      positions and to dual the live positions with it, which keeps the two
      roles separate and is what the supported workflow does.

    A forward tangent on a velocity is a second-order forward request and is
    refused for the reason in ``supports_higher_order_ad``: nothing in either
    package ships a second derivative.
    """

    tangent = forward_ad.unpack_dual(value).tangent
    if not value.requires_grad and tangent is None:
        return value
    carrier = "requires_grad" if value.requires_grad else "a forward tangent"
    raise RuntimeError(
        f"{owner}.{name} carries {carrier}, and a velocity here is a forward-AD "
        "tangent DIRECTION rather than a leaf (ADR-038). It is consumed by "
        "make_dual as the tangent of the position primal, so d(loss)/d(velocity) "
        "is structurally unavailable in both AD modes and no gradient would ever "
        "come back. Its use as a tangent direction IS supported and is what "
        "publishes delay_rate. If this tensor inherited its graph from a "
        "grad-carrying position, derive the velocity from a detached copy of "
        "those positions and dual the live positions with the result; that keeps "
        "the position leaf and the tangent direction separate instead of "
        "silently merging them."
    )


def _vector3(
    name: str,
    value: object,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """One world-frame 3-vector, or zeros when the caller declared nothing.

    ``None`` means "this quantity is not part of the motion", which is a
    statement about the world and not a missing argument, so it resolves to a
    real zero vector rather than raising. A Core ``RigidMotion`` with no
    ``angular_velocity`` describes a purely translating body.
    """

    if value is None:
        return torch.zeros(3, dtype=dtype, device=device)
    tensor = (
        value
        if isinstance(value, torch.Tensor)
        else torch.tensor(value, dtype=dtype, device=device)
    )
    if tuple(tensor.shape) != (3,):
        raise ValueError(f"{name} must have shape (3,), got {tuple(tensor.shape)}")
    return tensor.to(device=device, dtype=dtype)


@dataclass(frozen=True, slots=True, eq=False)
class Kinematics:
    """One ordered endpoint set's positions and velocities.

    ``positions_m`` is the tensor that becomes a forward-AD PRIMAL and
    ``velocities_m_per_s`` is the tensor that becomes its TANGENT. They are held
    together in one object because handing them to ``make_dual`` in the wrong
    order, or pairing a position tensor with a velocity tensor built for a
    different endpoint order, produces a completely plausible Doppler spectrum
    for a world that does not exist.

    Both are ``float32``, both are contiguous, and both live on the same device.
    That is the Channel endpoint contract restated at the point where the
    tensors are BUILT rather than at the point where they are rejected.
    """

    positions_m: torch.Tensor
    velocities_m_per_s: torch.Tensor

    def __post_init__(self) -> None:
        positions = _require_positions("positions_m", self.positions_m)
        velocities = _require_velocity_is_not_a_leaf(
            "velocities_m_per_s",
            "Kinematics",
            _require_positions("velocities_m_per_s", self.velocities_m_per_s),
        )
        if tuple(positions.shape) != tuple(velocities.shape):
            raise ValueError(
                f"positions_m has shape {tuple(positions.shape)} and "
                f"velocities_m_per_s has shape {tuple(velocities.shape)}; a "
                "tangent must name the same endpoints in the same order as the "
                "primal it rides"
            )
        if positions.device != velocities.device:
            raise ValueError(
                f"positions_m is on {positions.device} and velocities_m_per_s "
                f"is on {velocities.device}"
            )
        if not positions.is_contiguous() or not velocities.is_contiguous():
            raise ValueError("Kinematics tensors must be contiguous")

    @property
    def count(self) -> int:
        return int(self.positions_m.shape[0])

    @property
    def device(self) -> torch.device:
        return self.positions_m.device


def rigid_site_velocities(
    positions_m: torch.Tensor,
    *,
    velocity=None,
    angular_velocity=None,
    centre_m=None,
) -> torch.Tensor:
    """``v(p) = v_cm + omega x (p - c)`` for world points riding a rigid body.

    This is the whole of rigid-body micro-Doppler. Two points of the same body
    at opposite ends of a rotor get equal and opposite projected velocities, and
    the resulting Doppler pair spread is the blade-flash signature; a model that
    only carried ``v_cm`` would give both points the same shift and could not
    produce the signature at all.

    ``centre_m`` is the instantaneous rotation centre and it is explicit on
    purpose - see :func:`rotation_centre_m` for the value that matches the way
    Channel composes a snapshot's rigid motion. Guessing it wrong shows up as a
    velocity offset that is uniform over the body, which looks exactly like a
    platform velocity.
    """

    positions = _require_positions("positions_m", positions_m)
    device, dtype = positions.device, positions.dtype
    linear = _vector3("velocity", velocity, device, dtype)
    omega = _vector3("angular_velocity", angular_velocity, device, dtype)
    centre = _vector3("centre_m", centre_m, device, dtype)
    offset = positions - centre
    spin = torch.linalg.cross(omega.expand(offset.shape), offset)
    return (linear + spin).contiguous()


def rotation_centre_m(rigid_motion, *, device=None, dtype=torch.float32):
    """The point Channel's snapshot composition actually rotates a structure about.

    Channel builds a moved structure as ``vertices @ R.T + t``
    (``scene/compiler.py``): the authored WORLD vertices are rotated about the
    world origin and the translation is applied afterwards. Differentiating that
    gives ``dp/dt = omega x (p - t) + t_dot``, so the instantaneous rotation
    centre is the CURRENT TRANSLATION, not the authored pose position.

    That distinction is worth a named function because the authored pose centre
    is the intuitive answer and it is wrong: using it puts a spurious
    ``omega x (t - pose)`` offset on every site of the body, uniform over the
    body and therefore indistinguishable from a platform velocity.
    """

    translation = None if rigid_motion is None else rigid_motion.translation
    resolved = (
        torch.device(device)
        if device is not None
        else (
            translation.device
            if isinstance(translation, torch.Tensor)
            else torch.device("cpu")
        )
    )
    return _vector3("translation", translation, resolved, dtype)


def structure_site_kinematics(
    state, positions_m: torch.Tensor
) -> Kinematics:
    """Rigid-body kinematics of world points riding one Core ``StructureState``.

    ``positions_m`` are the world positions of the tracked points - scatter
    sites, in the Radar architecture, where a site is an ENDPOINT of the two
    legs rather than a piece of geometry. The structure supplies the motion; the
    caller supplies which points of it it cares about.

    A structure with no ``rigid_motion`` is stationary and gets exact zeros,
    which is a complete answer rather than a missing one.
    """

    positions = _require_positions("positions_m", positions_m)
    motion = getattr(state, "rigid_motion", None)
    return Kinematics(
        positions_m=positions.contiguous(),
        velocities_m_per_s=rigid_site_velocities(
            positions,
            velocity=None if motion is None else motion.velocity,
            angular_velocity=None if motion is None else motion.angular_velocity,
            centre_m=rotation_centre_m(motion, device=positions.device),
        ),
    )


def deformation_kinematics(
    positions_m: torch.Tensor,
    descriptor: DeformationVelocity,
    time_s: float,
    *,
    vertex_index: torch.Tensor | None = None,
) -> Kinematics:
    """Kinematics of tracked vertices of a deforming structure.

    ``vertex_index`` selects which authored vertices the tracked points are; it
    is an ``index_select`` rather than a slice because a limb's sites are not
    contiguous in authored vertex order. Omitting it means every vertex is
    tracked, in authored order.

    The descriptor's rate is ANALYTIC. This function will not difference two
    snapshots to obtain one, in production or anywhere else: a finite difference
    would silently answer with a truncation error that grows with the step and
    would put a forbidden approximation inside a production hot path.
    """

    positions = _require_positions("positions_m", positions_m)
    velocities = descriptor.velocity_at(time_s)
    if not isinstance(velocities, torch.Tensor):
        raise TypeError(
            f"{type(descriptor).__name__}.velocity_at must return a torch."
            f"Tensor, got {type(velocities).__name__}"
        )
    # Named here rather than left to ``Kinematics`` below, because a custom
    # ``DeformationVelocity`` is a third place a caller authors a velocity and
    # the message should blame the descriptor that produced it.
    _require_velocity_is_not_a_leaf(
        "velocity_at", type(descriptor).__name__, velocities
    )
    if vertex_index is not None:
        if vertex_index.dtype != torch.int64:
            raise TypeError(
                f"vertex_index must use torch.int64, got {vertex_index.dtype}"
            )
        velocities = velocities.index_select(
            0, vertex_index.to(device=velocities.device)
        )
    velocities = velocities.to(
        device=positions.device, dtype=torch.float32
    ).contiguous()
    return Kinematics(
        positions_m=positions.contiguous(), velocities_m_per_s=velocities
    )


@dataclass(frozen=True, slots=True, eq=False)
class LinearDeformation:
    """Per-vertex constant velocity: the analytic hinge, rotor and limb.

    ``vertices(t) = vertices_m + velocities_m_per_s * (t - reference_time_s)``,
    which is exact rather than a linearisation, so ``velocity_at`` is the true
    derivative at every instant and not just at the reference time.

    Despite the name this is not restricted to a translation of the whole body.
    The velocity is declared PER VERTEX, so a hinge is a velocity that grows
    linearly along the limb, a rotor is a velocity that grows linearly along the
    blade, and a rigidly translating body is the special case where every row is
    equal. What the constant-velocity form cannot express is a velocity that
    itself changes with time - a rotation over a large angle, for instance, is
    only this over a short interval. For a rotation, use
    :func:`rigid_site_velocities`, which carries the true ``omega x r`` at every
    instant.

    It satisfies two protocols at once and that is the whole point of it. As a
    ``witwin.core.dynamics.Deformation`` (``at(t) -> DeformationState``) it
    drives the geometry a ``DynamicScene`` compiles; as a
    :class:`DeformationVelocity` (``velocity_at(t)``) it drives the forward-AD
    tangent of the sites riding on it. One descriptor answering both is what
    keeps the moving mesh and the Doppler it produces from being two
    independent statements that can silently disagree.
    """

    vertices_m: torch.Tensor
    velocities_m_per_s: torch.Tensor
    reference_time_s: float = 0.0

    def __post_init__(self) -> None:
        vertices = _require_positions("vertices_m", self.vertices_m)
        velocities = _require_velocity_is_not_a_leaf(
            "velocities_m_per_s",
            "LinearDeformation",
            _require_positions("velocities_m_per_s", self.velocities_m_per_s),
        )
        if tuple(vertices.shape) != tuple(velocities.shape):
            raise ValueError(
                f"vertices_m has shape {tuple(vertices.shape)} and "
                f"velocities_m_per_s has shape {tuple(velocities.shape)}; a "
                "deformation states one velocity per authored vertex"
            )
        if vertices.device != velocities.device:
            raise ValueError(
                f"vertices_m is on {vertices.device} and velocities_m_per_s "
                f"is on {velocities.device}"
            )

    def vertices_at(self, time_s: float) -> torch.Tensor:
        elapsed = float(time_s) - float(self.reference_time_s)
        return self.vertices_m + self.velocities_m_per_s * elapsed

    def at(self, time_s: float):
        """The ``witwin.core`` deformation descriptor at ``time_s``.

        Absolute vertices rather than offsets, because this descriptor already
        owns the authored positions and reconstructing an offset from them
        would only invite the two to be authored against different meshes.
        """

        from witwin.core.dynamics import DeformationState

        return DeformationState(vertices=self.vertices_at(time_s))

    def velocity_at(self, time_s: float) -> torch.Tensor:
        del time_s  # constant by construction; see the class docstring
        return self.velocities_m_per_s


def endpoint_kinematics(
    snapshot_or_states,
    antenna_ids: Sequence[int] | None = None,
    *,
    device: str | torch.device = "cuda",
) -> Kinematics:
    """``(positions, velocities)`` for an ordered set of Core endpoint states.

    ``snapshot_or_states`` is a ``SceneSnapshot`` or any sequence of
    ``EndpointState``. ``antenna_ids`` declares the ENDPOINT BATCH ORDER: it is
    the order the positions and the velocities are both built in, and it is the
    order the Channel leg rows will name. Omitting it keeps the snapshot's own
    declaration order, which is fine for a single caller and wrong the moment
    two callers disagree, so a batch that will be joined by identity should
    always declare it.

    Position resolution follows Core's own composition: the authored antenna
    position plus the snapshot's additional world-frame ``translation``. An
    endpoint's ``rotation`` is orientation and does not move its phase centre;
    an array element that ORBITS a rotating platform is a rigid-body site and
    belongs in :func:`rigid_site_velocities` with the platform centre, not here.

    Velocity is ``rigid_motion.velocity`` verbatim - the first consumer that
    field has ever had. An endpoint with no motion contributes exact zeros.
    """

    states = getattr(snapshot_or_states, "endpoints", snapshot_or_states)
    ordered = list(states)
    if antenna_ids is not None:
        by_id = {int(state.antenna.antenna_id): state for state in ordered}
        missing = [
            stable_id for stable_id in antenna_ids if int(stable_id) not in by_id
        ]
        if missing:
            raise KeyError(
                f"the snapshot declares no endpoint for antenna IDs {missing}; "
                f"it carries {sorted(by_id)}"
            )
        ordered = [by_id[int(stable_id)] for stable_id in antenna_ids]
    if not ordered:
        raise ValueError("endpoint_kinematics requires at least one endpoint")

    resolved = torch.device(device)
    positions = []
    velocities = []
    for state in ordered:
        motion = getattr(state, "rigid_motion", None)
        position = state.antenna.position.to(
            device=resolved, dtype=torch.float32
        )
        if tuple(position.shape) != (3,):
            raise ValueError(
                f"antenna position must have shape (3,), got "
                f"{tuple(position.shape)}"
            )
        translation = _vector3(
            "translation",
            None if motion is None else motion.translation,
            resolved,
            torch.float32,
        )
        positions.append(position + translation)
        velocities.append(
            _vector3(
                "velocity",
                None if motion is None else motion.velocity,
                resolved,
                torch.float32,
            )
        )
    return Kinematics(
        positions_m=torch.stack(positions).contiguous(),
        velocities_m_per_s=torch.stack(velocities).contiguous(),
    )


def replicate_slots(positions: torch.Tensor, slot_count: int) -> torch.Tensor:
    """Repeat one endpoint set once per slot, SLOT MAJOR, on a live tensor.

    ``index_select`` rather than a rebuild, and that is the whole point: a
    forward-AD dual survives a differentiable op and dies the moment its values
    are read back into Python. The slot-major layout - slot ``t`` owning rows
    ``[t * n, (t + 1) * n)`` - is the Channel consumer's ``slot_pair_layout``,
    which is why the arithmetic is stated here once instead of in every caller.

    A slot stack of a MOVING endpoint set is a different expression: build the
    per-slot displacement as a differentiable function of the base positions and
    keep the tangent flowing through it. This function is the STATIC case, where
    every slot sees the same positions and the frozen-mode kernel owns the
    slow-time carrier.
    """

    if type(slot_count) is not int or slot_count < 1:
        raise ValueError(f"slot_count must be a positive int, got {slot_count!r}")
    if slot_count == 1:
        return positions
    rows = int(positions.shape[0])
    index = torch.arange(rows, device=positions.device).repeat(slot_count)
    return positions.index_select(0, index)


@dataclass(frozen=True, slots=True, eq=False)
class TwoWayDuals:
    """The three position tensors of a radar round trip, dualised together.

    Valid only inside the :func:`two_way_duals` block that produced them. A
    forward tangent belongs to its level; reading one after the level exits is
    undefined, which is why the adapter clones the delay tangent inside the
    level and why this object is yielded rather than returned.
    """

    transmitters: torch.Tensor | None
    sites: torch.Tensor
    receivers: torch.Tensor | None
    slot_count: int


@contextlib.contextmanager
def two_way_duals(
    *,
    sites: Kinematics,
    transmitters: Kinematics | None = None,
    receivers: Kinematics | None = None,
    slot_count: int = 1,
) -> Iterator[TwoWayDuals]:
    """Dualise transmitter, site and receiver positions in ONE level.

    One level, not three. The inbound leg's delay rate is
    ``d|p_site - p_tx|/dt`` and needs both tangents live at once; the outbound
    leg's needs the site and the receiver. Nesting a level per tensor would make
    each leg see one moving end and one frozen end and would publish a round
    trip whose two halves describe different worlds.

    ``sites`` is required because a radar round trip without a target is not a
    round trip. The endpoints are optional: a static front end is the common
    case and passing ``None`` says so, rather than passing a zero tangent that
    a reader has to check.

    ``slot_count`` replicates each tensor slot major INSIDE the level, so the
    batched replay of a whole TDM frame carries the same live tangent in every
    slot.
    """

    if not isinstance(sites, Kinematics):
        raise TypeError(
            f"sites must be a Kinematics, got {type(sites).__name__}"
        )
    for name, value in (("transmitters", transmitters), ("receivers", receivers)):
        if value is not None and not isinstance(value, Kinematics):
            raise TypeError(
                f"{name} must be a Kinematics or None, got {type(value).__name__}"
            )
    if type(slot_count) is not int or slot_count < 1:
        raise ValueError(f"slot_count must be a positive int, got {slot_count!r}")

    def dual(value: Kinematics | None) -> torch.Tensor | None:
        if value is None:
            return None
        return replicate_slots(
            forward_ad.make_dual(value.positions_m, value.velocities_m_per_s),
            slot_count,
        )

    with forward_ad.dual_level():
        yield TwoWayDuals(
            transmitters=dual(transmitters),
            sites=dual(sites),
            receivers=dual(receivers),
            slot_count=slot_count,
        )

__all__ = [
    "MOBILITIES",
    "ClutterComponentSpec",
    "DeformationVelocity",
    "EndpointRole",
    "EpochFrame",
    "EpochPolicy",
    "FrozenEpoch",
    "Kinematics",
    "LinearDeformation",
    "RadarEndpointSpec",
    "RadarLegBatch",
    "RadarPropagationLegs",
    "SceneEpochLoop",
    "TwoWayDuals",
    "deformation_kinematics",
    "endpoint_kinematics",
    "epoch_policy",
    "replicate_slots",
    "require_endpoint_role",
    "rigid_site_velocities",
    "rotation_centre_m",
    "structure_site_kinematics",
    "two_way_duals",
]
