"""Scene-session orchestration: the world half, the instrument half, and their fusion.

Core owns geometry and authored motion; Channel evaluates each one-way path;
Radar composes the round trip and synthesizes the waveform. Dynamic
observations refresh the complex transport, so no parameter JVP is interpreted
as velocity. Topology is rediscovered at every observation by default: replay
alone cannot identify new paths. An explicitly longer discovery cadence is
reported as an incomplete path set. No velocity is inferred by subtracting
adjacent path rows.

All four public verbs run one :class:`Session`, opened by
:func:`open_session`: :meth:`~witwin.radar.Radar.trace` drains its frame traces
into a :class:`Paths`, and :meth:`~witwin.radar.Radar.simulate`,
:meth:`~witwin.radar.Radar.stream` and :meth:`~witwin.radar.Radar.echo` hand
each one to :func:`echo_frame`. :class:`WorldHalf` owns the epoch loop and every
Channel call; :func:`echo_frame` drives every synthesis launch outside the
adaptive route, through the radar's own waveform dispatch, and
:mod:`witwin.radar.adaptive` drives that route's.
"""

from __future__ import annotations

import hashlib
import math
from collections import OrderedDict
from collections.abc import Callable, Hashable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from functools import partial, reduce
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import numpy as np
import torch
from witwin.core.dynamics import DynamicScene

from .adaptive import AdaptiveTable, PlaneGroups, composed_plane_identity, echo_adaptive_frame, trace_adaptive_frame
from .binding import (
    SITE_SOURCE_EXPLICIT,
    RadarWorldBinding,
    ScatterSitePolicy,
    SensorEndpointIds,
    bind_radar_world,
    with_site_positions,
)
from .events import EventRecorder, Observed, PathEvents, Pool, batch_pool
from .motion import Motion
from .paths import RadarPathBatch, SlotRows
from .policy import _carries_derivative as _carries_value_derivative
from .policy import refuse_derivative
from .propagation import (
    DiscoveryReport,
    EpochFrame,
    FrozenEpoch,
    RadarEndpointSpec,
    RadarLegBatch,
    RadarPropagationLegs,
    SceneEpochLoop,
    band_as_slots,
    stack_slots,
)
from .results import Result, SceneFrame, assemble

if TYPE_CHECKING:
    from .paths import ComponentDeclaration, RadarComponentIndex, TwoWayComposer
    from .radar import PropagationConfig, Radar
    from .scattering import ScatterResponse
    from .sensors import Pattern, RoundTripPatternStage, SensorArraySpec
    from .synthesis.assembly import FmcwSpec, OfdmSpec, PulsedSpec, SlowTimeMode, SynthesisResult

    WaveformSpec = FmcwSpec | OfdmSpec | PulsedSpec


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, eq=False)
class FrozenRoundTrip:
    """What one topology epoch freezes besides its two leg handles.

    ``composer`` is the identity join, ``binding`` the endpoint specs the
    freeze used (a frame that just froze replays against them rather than
    building a second, identical binding), and ``pattern`` the antenna-pattern
    stage's constant tables for that join.
    """

    composer: TwoWayComposer
    binding: RadarWorldBinding
    pattern: RoundTripPatternStage
    #: The legs' packed row identities (inbound, outbound and, when composed,
    #: the site-less direct leg). Two epochs with equal identities join to the
    #: same composed rows, so a rediscovery that found the same topology reuses
    #: the join and pattern tables instead of rebuilding them.
    identity: tuple[bytes, ...]
    #: What the adaptive controller compares across probes: the composed rows
    #: keyed by plane group rather than triangle where the world's faces do
    #: not move, else :attr:`identity` itself.
    adaptive_identity: Hashable = None


def _round_trip(epoch: FrozenEpoch) -> FrozenRoundTrip:
    payload = epoch.payload
    if not isinstance(payload, FrozenRoundTrip):
        raise TypeError(f"a scene session epoch carries a FrozenRoundTrip, got {type(payload).__name__}")
    return payload


@dataclass(frozen=True, slots=True, eq=False)
class SlotBatch:
    """The slot-major rows one grouped replay composed, shared by its instants.

    ``rows`` holds ``slots`` copies of one join's composed rows, slot major:
    row ``g * rows_per_slot + r`` is join row ``r`` at slot ``g``. ``offsets`` is ONE slot's pair partition on the
    host, which is what lets the instrument half select sensor-pair segments
    without reading the device.
    """

    rows: SlotRows
    offsets: np.ndarray
    slots: int


@dataclass(frozen=True, slots=True, eq=False)
class EvaluatedObservation:
    """One evaluated instant, as the frame loop and the adaptive controller read it.

    ``legs`` is a CALLABLE returning that instant's legs: a batched group
    never reads them and the adaptive controller never reads them either, so
    only an observation that gets published pays for narrowing its slot out of
    the batch. ``identity`` is what the adaptive controller compares; it is
    ``()`` on every other route.
    """

    time_s: float
    epoch_frame: EpochFrame
    legs: Callable[[], RadarPropagationLegs]
    composed: RadarPathBatch
    identity: Hashable
    #: The grouped replay this instant is slot ``slot`` of.
    batch: SlotBatch
    slot: int


@dataclass(frozen=True, slots=True, eq=False)
class _PendingObservation:
    """An instant whose world has advanced and whose replay is not yet run."""

    time_s: float
    epoch_frame: EpochFrame
    binding: RadarWorldBinding
    identity: Hashable
    #: The radar's body rotation at this instant, ``float32[3, 3]``
    #: local-to-world, or ``None`` when the radar does not follow a pose.
    rotation: torch.Tensor | None = None


@dataclass(frozen=True, slots=True, eq=False)
class Observation:
    """One published instant: the world it saw and the rows it composed.

    ``legs`` is narrowed out of the grouped replay only when read, which on
    the fused route is once a frame.
    """

    time_s: float
    epoch_frame: EpochFrame
    legs_of: Callable[[], RadarPropagationLegs]
    composed: RadarPathBatch
    batch: SlotBatch | None = None
    slot: int = 0

    @property
    def legs(self) -> RadarPropagationLegs:
        return self.legs_of()


@dataclass(frozen=True, slots=True, eq=False)
class FrameTrace:
    """One frame's composed rows, before any instrument stage has run.

    ``observations`` is an ITERATOR on the fused route and is drained exactly
    once, in order, so the route holds at most one replay group of
    ``Motion.batch_observations`` instants; :func:`trace_scene` materialises
    it into a tuple instead.

    An adaptive frame yields a single observation, the one that closed it,
    because its rows live in ``adaptive`` rather than one batch per instant.
    """

    time_s: float
    sample_times_s: np.ndarray
    observations: Iterator[Observation] | tuple[Observation, ...]
    path_set_complete: bool
    motion_sampling_exhaustive: bool
    #: The frame's path births and deaths. Filled as ``observations`` drains,
    #: so it is complete once the frame has been traced or echoed.
    events: PathEvents = field(default_factory=PathEvents)
    adaptive: AdaptiveTable | None = None
    stats: dict[str, Any] | None = None
    opened: EpochFrame | None = None
    #: What Channel said about the discoveries this frame answered from. Read
    #: only after ``observations`` is drained, because the fused route
    #: discovers while it is drained.
    discovery: Callable[[], DiscoveryReport] = DiscoveryReport


@dataclass(frozen=True, slots=True)
class InstrumentSpecs:
    """The three waveform specs one radar's instrument half works from.

    ``declared`` is the radar's own spec and the output domain the result is
    published in; ``working`` is what synthesis runs, which is the beat domain
    whenever a receive chain must see ADC-time samples; ``single`` is the
    one-observation spec a sampled session synthesizes each observation with.
    """

    declared: WaveformSpec
    working: WaveformSpec
    single: WaveformSpec | None


class _HeldSynthesis:
    """One unsampled frame's synthesis, reused while its rows are the same object.

    A still world publishes the same composed batch every frame (see
    :class:`WorldHalf`), and synthesis is a deterministic function of the rows
    and the spec, so the frame's waveform is synthesized once and only the
    receive chain runs per frame. A batch that carries a derivative is never
    held, for the same reason the world half does not hold one.
    """

    __slots__ = ("composed", "spec", "synthesis")

    def __init__(self) -> None:
        self.composed: RadarPathBatch | None = None
        self.spec: WaveformSpec | None = None
        self.synthesis: SynthesisResult | None = None

    def synthesize(
        self, radar: Radar, composed: RadarPathBatch, mode: SlowTimeMode, spec: WaveformSpec
    ) -> SynthesisResult:
        if self.synthesis is not None and self.composed is composed and self.spec == spec:
            return self.synthesis
        synthesis = radar._synthesize(composed, slow_time_mode=mode, spec=spec)
        if _carries_derivative(composed):
            self.composed = self.spec = self.synthesis = None
        else:
            self.composed, self.spec, self.synthesis = composed, spec, synthesis
        return synthesis


@dataclass(frozen=True, slots=True, eq=False)
class Session:
    """What the instrument half needs that does not change between frames.

    Built once by :func:`open_session` from the radar and the resolved motion,
    and read by :func:`echo_frame`. Nothing here describes the world beyond
    the epoch loop's counters, which is the property that lets one traced
    session be echoed against a different receive chain.
    """

    radar: Radar
    array: SensorArraySpec
    loop: SceneEpochLoop
    mode: SlowTimeMode
    specs: InstrumentSpecs
    #: World-time offsets of this session's observations within a frame,
    #: ``float64[observations]``. An ADC-refreshed frame has one per chirp,
    #: transmitter and sample, which is why it is an array.
    offsets: np.ndarray
    #: ``int64[slow, pairs]``: which observation each output slot reads.
    pair_samples: torch.Tensor | None
    sampled: bool
    adc_sampled: bool
    samples_per_slot: int
    path_phase_noise: bool
    reference_frequency_hz: float
    #: The published sampling kind, after ``Motion.auto()`` resolved.
    kind: str
    components: frozenset[str]
    max_depth: int
    grad: str
    #: The sampling record as requested; its adaptive fields are the route's
    #: tolerances.
    motion: Motion
    times: tuple[float, ...]
    #: Whether the site-less leg carried the environment's returns and the
    #: elements' straight-segment coupling.
    environment: bool = False
    coupling: bool = False
    #: The last unsampled frame's waveform result, keyed by the rows it was
    #: synthesized from; see :class:`_HeldSynthesis`.
    held: _HeldSynthesis = field(default_factory=lambda: _HeldSynthesis())
    #: Device routing tables of the slot-batch synthesis, keyed by the host
    #: tables they were uploaded from. A frame's batches repeat the same
    #: routing every frame, so each table crosses to the device once.
    routing: _Bounded[bytes, torch.Tensor] = field(default_factory=lambda: _Bounded(ROUTING_CACHE))

    def specs_for(self, radar: Radar) -> InstrumentSpecs:
        """The spec triple ``radar`` would work from on this session's schedule.

        ONE owner: the session setup and :func:`echo_paths` against a
        different receive chain both derive the specs through this, so a
        receiver that flips the FMCW output domain cannot end up paired with a
        slot schedule derived from the other domain. The observation offsets
        are deliberately not re-derived: they are the traced schedule, and a
        spec that would change them is refused by name rather than silently
        re-scheduled.
        """

        from .synthesis.assembly import fast_time_single, waveform_sampling

        declared = radar.system_config.waveform_spec()
        working = _working_spec(radar, declared)
        if not self.sampled:
            return InstrumentSpecs(declared=declared, working=working, single=None)
        _, _, single = waveform_sampling(
            working, num_tx=self.array.num_tx, num_rx=self.array.num_rx, device=radar.device
        )
        if self.adc_sampled:
            single = fast_time_single(single)
        return InstrumentSpecs(declared=declared, working=working, single=single)

    def echoing(self, radar: Radar) -> Session:
        """This session with ``radar``'s instrument half in place of its own."""

        return replace(
            self,
            radar=radar,
            specs=self.specs_for(radar),
            path_phase_noise=radar.frontend is not None and radar.frontend.has_phase_noise,
            held=_HeldSynthesis(),
        )


# ---------------------------------------------------------------------------
# The world half
# ---------------------------------------------------------------------------


def _dynamic_scene(scene: object) -> DynamicScene:
    """A ``DynamicScene`` for whichever of the two Core worlds was passed.

    A static ``Scene`` is wrapped rather than refused: it IS a dynamic scene
    with no declared motion, the loop then reports ``structures_move = False``,
    and the compiled scene is built exactly once for the whole run.
    """

    if isinstance(scene, DynamicScene):
        return scene
    return DynamicScene(scene)


def _times(times: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in times)
    if not values:
        raise ValueError(
            "times must name at least one frame instant; an empty sequence asks for a simulation of nothing"
        )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("frame times must be finite")
    if any(later <= earlier for earlier, later in zip(values, values[1:], strict=False)):
        raise ValueError("frame times must be strictly increasing")
    return values


def _stacked(specs: Sequence[RadarEndpointSpec]) -> RadarEndpointSpec:
    """One slot-major endpoint stack, or the single spec itself."""

    if len(specs) == 1:
        return specs[0]
    first = specs[0]
    return RadarEndpointSpec(
        stable_ids=torch.cat([spec.stable_ids for spec in specs], dim=0),
        positions_m=torch.cat([spec.positions_m for spec in specs], dim=0),
        polarizations=torch.cat([spec.polarizations for spec in specs], dim=0),
        powers_w=None if first.powers_w is None else torch.cat([_powers(spec) for spec in specs], dim=0),
        precise_positions_m=(
            None if first.precise_positions_m is None else torch.cat([_precise(spec) for spec in specs], dim=0)
        ),
    )


#: How many slot-batched join and pattern table sets a session keeps. Each is
#: ``slots x rows`` int64 tables, so the cache is bounded by entries rather
#: than left to grow with every group size and epoch a run produces.
SLOT_TABLE_CACHE = 4

#: How many slot-batch routing tables a session keeps on the device.
ROUTING_CACHE = 64

K = TypeVar("K")
V = TypeVar("V")


class _Bounded(Generic[K, V]):
    """A least-recently-used map with a fixed number of entries."""

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._entries: OrderedDict[K, V] = OrderedDict()

    def get(self, key: K) -> V | None:
        value = self._entries.get(key)
        if value is not None:
            self._entries.move_to_end(key)
        return value

    def put(self, key: K, value: V) -> None:
        self._entries[key] = value
        self._entries.move_to_end(key)
        while len(self._entries) > self._capacity:
            self._entries.popitem(last=False)


def _tensor_carries_derivative(tensor: torch.Tensor) -> bool:
    """Whether a tensor is on a reverse tape or carries a forward tangent."""

    from torch.autograd.forward_ad import unpack_dual

    return tensor.requires_grad or unpack_dual(tensor).tangent is not None


def _carries_derivative(paths: RadarPathBatch) -> bool:
    """Whether a batch's payload is on a reverse tape or carries a forward tangent."""

    return _tensor_carries_derivative(paths.total_delay_s) or _tensor_carries_derivative(paths.complex_transfer_ref)


def _present(value: torch.Tensor | None, name: str) -> torch.Tensor:
    """A member every replayed batch publishes, refused by name if absent."""

    if value is None:
        raise ValueError(f"a replayed batch always carries {name}")
    return value


def _batch_pool(paths: object) -> Pool:
    assert not isinstance(paths, AdaptiveTable)
    return batch_pool(paths)


def _table_pool(table: object) -> Pool:
    assert isinstance(table, AdaptiveTable)
    return table.validity, torch.view_as_real(table.transfers.detach())


def _constant_legs(legs: RadarPropagationLegs) -> RadarPropagationLegs:
    return legs


def _constant_report(report: DiscoveryReport) -> DiscoveryReport:
    return report


def _slot_legs(replay: RadarPropagationLegs, index: int) -> RadarPropagationLegs:
    """One slot of a slot-major replay, narrowed only when a caller reads it."""

    return RadarPropagationLegs(
        inbound=replay.inbound.slot(index),
        outbound=replay.outbound.slot(index),
        direct=None if replay.direct is None else replay.direct.slot(index),
    )


def _present_leg(value: RadarLegBatch | None) -> RadarLegBatch:
    if value is None:
        raise ValueError("a primal epoch keeps the rows its discoveries evaluated, for every leg")
    return value


def _discovered_legs(epoch: FrozenEpoch) -> RadarPropagationLegs:
    """The rows one epoch's discoveries evaluated, as the legs of that instant."""

    batches = [_present_leg(getattr(handle, "discovered", None)) for handle in epoch.handles]
    return RadarPropagationLegs(
        inbound=batches[0], outbound=batches[1], direct=batches[2] if len(batches) > 2 else None
    )


@dataclass(frozen=True, slots=True)
class LegRequest:
    """What a session asks of every leg discovery beyond the component set.

    ``discovery`` and ``max_paths`` are forwarded to Channel.
    ``max_round_trips`` bounds the composed rows of one sensor pair and is
    checked on the host before the join. ``direct_components`` is the
    site-less TX->RX leg's request, empty when the run composes none.
    ``replays`` is ``False`` when a requested component cannot be frozen, so
    every handle answers only at its discovery instant.
    ``frequency_offsets_hz`` is the wideband grid a replay asks for, ``None``
    for a narrowband run.
    """

    discovery: str = "auto"
    max_paths: int | None = None
    max_round_trips: int | None = None
    direct_components: frozenset[str] = frozenset()
    replays: bool = True
    frequency_offsets_hz: tuple[float, ...] | None = None


def _precise(spec: RadarEndpointSpec) -> torch.Tensor:
    if spec.precise_positions_m is None:
        raise ValueError("every slot of a stack authored in float64 carries precise_positions_m")
    return spec.precise_positions_m


def _bound(response: ScatterResponse, binding: RadarWorldBinding, time_s: float) -> ScatterResponse:
    """The response at one observation: bound to its facets and its fluctuation draw, if it has either.

    A response with ``bind_observation`` depends on the observation - a mesh
    target's facet frame, a Swerling draw - and is bound here; every other
    response is the session's own. A slot batch binds its first observation,
    which is exact because only a world whose structures do not move is slot
    batched and a response that varies per observation is never batched.
    """

    bind = getattr(response, "bind_observation", None)
    if bind is None:
        if binding.facets is not None:
            raise TypeError("a mesh target's sites need a response that binds to their facets")
        return response
    bound: ScatterResponse = bind(binding, time_s)
    return bound


def _powers(spec: RadarEndpointSpec) -> torch.Tensor:
    if spec.powers_w is None:
        raise ValueError("every slot of a source stack carries powers_w")
    return spec.powers_w


class WorldHalf:
    """One session's world: the epoch loop, the binding, and every Channel call.

    Built once per session. The frozen join and pattern tables live in the
    loop's epochs; the slot-batched replicas of them live here, keyed by the
    leg identity they were built for and the group size.
    """

    def __init__(
        self,
        radar: Radar,
        dynamic: DynamicScene,
        *,
        sites: ScatterSitePolicy,
        polarization: Sequence[float],
        sensor_endpoints: SensorEndpointIds | None,
        response: ScatterResponse,
        antenna_pattern: Pattern,
        propagation: PropagationConfig,
        array: SensorArraySpec,
        motion_sampling: str,
        ad_mode: str,
        cadence: int | None,
        world_motion: str,
        period_s: float | None = None,
        displacement_m: float | None = None,
        pose: Callable[[float], Any] | None = None,
        still_world: bool = False,
        batch_observations: int = 1,
        request: LegRequest | None = None,
        rx_polarization: Sequence[float] | None = None,
    ) -> None:
        from .channel import compile_scene

        self.radar = radar
        self.dynamic = dynamic
        self.sites = sites
        self.polarization = polarization
        self.rx_polarization = polarization if rx_polarization is None else rx_polarization
        self.sensor_endpoints = sensor_endpoints
        self.response = response
        self.antenna_pattern = antenna_pattern
        self.propagation = propagation
        self.array = array
        self.motion_sampling = motion_sampling
        self.ad_mode = ad_mode
        self.reference_frequency_hz = propagation.reference_frequency_hz
        self.loop = SceneEpochLoop(
            dynamic,
            reference_frequency_hz=self.reference_frequency_hz,
            bind=self.bind,
            compile_scene=compile_scene,
            motion_event_period_frames=cadence,
            motion_event_period_s=period_s,
            world_motion=world_motion,
        )
        #: A displacement cadence: rediscover once any site or endpoint has
        #: moved this far, m, since the last discovery. ``_reference`` is the
        #: host copy of the positions at that discovery.
        self.displacement_m = displacement_m
        self._reference: np.ndarray | None = None
        #: The radar's ego-motion, ``pose(t) -> (rotation, translation)``, and
        #: the instant the loop is advancing to, which a discovery binds at.
        self.pose = pose
        self._clock = 0.0
        from .targets import TimedAspectResponse

        #: A response evaluated per instant (a rotating aspect lobe), resolved
        #: once per replay group; ``None`` for a time-invariant response.
        self.timed_response = response if isinstance(response, TimedAspectResponse) else None
        #: Coplanar face groups for the adaptive identity, in a world whose
        #: compiled faces never move; ``None`` elsewhere.
        self.planes = PlaneGroups() if motion_sampling == "adaptive" and not self.loop.structures_move else None
        #: A world that nothing moves and nothing rediscovers replays the same
        #: endpoints against the same frozen topology every frame, and Channel's
        #: replay is deterministic, so its rows are evaluated once and every
        #: later frame publishes them again. Only a still world that carries no
        #: derivative holds them: a shared graph would sum every frame's
        #: cotangent before the replay's backward rather than after it.
        self.still_world = still_world
        self.batch_observations = batch_observations
        self.request = LegRequest() if request is None else request
        #: A primal run reads each rediscovered instant's rows straight from the
        #: discovery that found them; Channel's replay at the same instant
        #: returns the same native evaluation for every replayable row, so the
        #: replay would be a second call for the same numbers. A differentiable
        #: run replays, because only the replay carries a derivative.
        self.discovered_rows = ad_mode == "none" and self.request.frequency_offsets_hz is None
        self._report = DiscoveryReport()
        #: Path birth and death detection, for a world in which a path can be
        #: born; a still world publishes an empty log.
        self.recorder = None if still_world else EventRecorder()
        self._still: Observation | None = None
        self._slot_tables: _Bounded[tuple[Hashable, int], tuple[TwoWayComposer, RoundTripPatternStage]] = _Bounded(
            SLOT_TABLE_CACHE
        )

    def _bind_world(self, snapshot: object, time_s: float | None = None) -> RadarWorldBinding:
        """Bind the radar to ``snapshot``; ``time_s`` is the instant a pose is read at.

        ``None`` is the instant the loop is advancing to, which is what a
        discovery inside :meth:`SceneEpochLoop.frame` binds at.
        """

        binding = bind_radar_world(
            self.radar,
            snapshot,
            sites=self.sites,
            polarization=self.polarization,
            rx_polarization=self.rx_polarization,
            sensor_endpoints=self.sensor_endpoints,
        )
        if self.pose is None:
            return binding
        return self._posed(binding, self._clock if time_s is None else time_s)[0]

    def _rotation_at(self, time_s: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """The pose at ``time_s``: a ``float32[3, 3]`` rotation and ``float32[3]`` translation."""

        assert self.pose is not None
        rotation, translation = self.pose(float(time_s))
        refuse_derivative(
            "the radar pose trajectory",
            "the rotation turns the polarization every leg is launched with, and Channel's replay takes an "
            "endpoint polarization as a primal-only constant, so the rotation's derivative cannot be "
            "delivered whole. The translation is a leaf; keep the rotation a host value.",
            rotation=rotation,
        )
        carrier = _carries_value_derivative(translation)
        if carrier is not None and self.ad_mode != ("vjp" if carrier == "a gradient" else "jvp"):
            raise RuntimeError(
                f"the radar pose trajectory's translation carries {carrier}, which moves every element "
                "position, and the legs carry that derivative only in the matching AD mode: pass "
                f"grad={'vjp' if carrier == 'a gradient' else 'jvp'!r} (got grad={self.ad_mode!r})."
            )
        rotation = torch.as_tensor(rotation, dtype=torch.float32, device=device)
        translation = torch.as_tensor(translation, dtype=torch.float32, device=device)
        if tuple(rotation.shape) != (3, 3) or tuple(translation.shape) != (3,):
            raise ValueError(
                "pose(t) must return a (3, 3) local-to-world rotation and a (3,) world translation, got "
                f"{tuple(rotation.shape)} and {tuple(translation.shape)}"
            )
        return rotation, translation

    def _posed(self, binding: RadarWorldBinding, time_s: float) -> tuple[RadarWorldBinding, torch.Tensor]:
        """The binding with the radar's elements carried by its pose at ``time_s``.

        The rotation turns the array about the radar's authored position and
        the translation then moves it: ``p(t) = c + T(t) + R(t) (p - c)``. The
        declared polarization turns with it on every endpoint. Returns the rotation too, which
        the pattern stage reads to see each direction in the body frame.
        """

        device = binding.device
        rotation, translation = self._rotation_at(time_s, device)
        centre = torch.as_tensor(self.radar.position, dtype=torch.float32, device=device)

        def turned(spec: RadarEndpointSpec) -> torch.Tensor:
            # Row vectors: ``v @ R^T`` is ``R v``.
            return (spec.polarizations @ rotation.T).contiguous()

        def carried(spec: RadarEndpointSpec) -> RadarEndpointSpec:
            moved = ((spec.positions_m - centre) @ rotation.T + centre + translation).contiguous()
            return replace(spec, positions_m=moved, polarizations=turned(spec))

        # Every endpoint of a binding carries the radar's declared polarization,
        # so all of them turn with the pose; only the radar's own elements move.
        # The receivers' two roles keep sharing one moved position tensor.
        receivers = carried(binding.receivers)
        posed = replace(
            binding,
            transmitters=carried(binding.transmitters),
            receivers=receivers,
            receiver_sources=replace(
                binding.receiver_sources, positions_m=receivers.positions_m, polarizations=receivers.polarizations
            ),
            sites=replace(binding.sites, polarizations=turned(binding.sites)),
        )
        return posed, rotation

    def bind(self, compiled: object, snapshot: object, previous: FrozenEpoch | None) -> FrozenEpoch:
        """Freeze one topology epoch: both legs, the join and the pattern tables.

        The loop's ``bind`` callback, run once per topology epoch. The pattern
        tables are a property of the frozen join - which pair each row belongs
        to and which site it visits - so they are built here, and the frame
        loop only gathers positions and launches.
        """

        from .channel import ChannelPropagationAdapter
        from .paths import TwoWayComposer, read_leg_identity
        from .sensors import RoundTripPatternStage

        binding = self._bind_world(snapshot)
        request = self.request
        adapter: Any = (
            ChannelPropagationAdapter(
                compiled,
                reference_frequency_hz=self.reference_frequency_hz,
                components=self.propagation.components | request.direct_components,
                max_depth=self.propagation.max_depth,
                discovery=request.discovery,
                max_paths=request.max_paths,
                replay=request.replays,
            )
            if previous is None
            else previous.adapter
        )
        rows = self.discovered_rows
        site_components = self.propagation.components
        # Both site legs run from a radar element toward the sites; the outbound
        # one is the reciprocal of site-to-receiver transport (R-ADR-039), so
        # the receivers are its 1 W sources here and the direct leg's sinks below.
        inbound = adapter.discover(binding.transmitters, binding.sites, rows=rows, components=site_components)
        outbound = adapter.discover(binding.receiver_sources, binding.sites, rows=rows, components=site_components)
        direct = None
        if request.direct_components:
            direct = adapter.discover(
                binding.transmitters,
                binding.receivers,
                rows=rows,
                components=request.direct_components,
                element_sinks=True,
            )
        handles = (inbound, outbound) if direct is None else (inbound, outbound, direct)
        self._report = self._report.including(handles)
        arriving = read_leg_identity(inbound, "inbound")
        leaving = read_leg_identity(outbound, "outbound")
        through = None if direct is None else read_leg_identity(direct, "direct")
        identity = (arriving.packed, leaving.packed) + (() if through is None else (through.packed,))
        retiring = None if previous is None else _round_trip(previous)
        if retiring is not None and retiring.identity == identity and retiring.binding.site_ids == binding.site_ids:
            # The same rows under the same declared IDs: the join and the
            # pattern tables are functions of exactly these, so the retiring
            # epoch's are this epoch's.
            composer, pattern = retiring.composer, retiring.pattern
        else:
            composer = TwoWayComposer.freeze(
                inbound,
                outbound,
                binding.site_ids,
                radar_source_ids=binding.transmitter_ids,
                radar_sink_ids=binding.receiver_ids,
                reference_frequency_hz=self.reference_frequency_hz,
                inbound_identity=arriving,
                outbound_identity=leaving,
                direct=direct,
                direct_identity=through,
                max_round_trips=request.max_round_trips,
                # Both legs were discovered for exactly these site endpoints,
                # so a site with no row is occluded at this instant rather
                # than undeclared, and contributes zero.
                leg_site_ids=binding.site_ids,
                outbound_line_of_sight_only=binding.facets is not None,
            )
            pattern = RoundTripPatternStage.freeze(
                self.radar,
                composer,
                site_ids=binding.site_ids,
                pattern=self.antenna_pattern,
                rx_pattern=self.radar.system_config.sensors.rx_pattern,
            )
        adaptive_identity: Hashable = identity
        if self.planes is not None:
            adaptive_identity = composed_plane_identity(
                composer,
                self.planes.leg_labels(inbound, arriving.primitive, arriving.depth),
                self.planes.leg_labels(outbound, leaving.primitive, leaving.depth),
            )
        return FrozenEpoch(
            adapter=adapter,
            handles=handles,
            payload=FrozenRoundTrip(
                composer=composer,
                binding=binding,
                pattern=pattern,
                identity=identity,
                adaptive_identity=adaptive_identity,
            ),
        )

    def begin_report(self) -> None:
        """Start a frame's discovery report from the epoch it opens on.

        A frame that rediscovers nothing still answers out of the current
        epoch's handles, so their completeness is part of its report.
        """

        frozen = self.loop.frozen
        self._report = DiscoveryReport().including(() if frozen is None else frozen.handles)

    def report(self) -> DiscoveryReport:
        """What Channel said about every discovery this frame answered from."""

        return self._report

    def _complete_los(self) -> bool:
        """Whether the adaptive route may certify this world's family for all time.

        Only an empty authored world proves the absence of occluders and
        reflection/diffraction births. Every endpoint pair must be present
        too: a degenerate initial LOS discovery cannot certify it.
        """

        frozen = self.loop.frozen
        authored_world = getattr(self.dynamic, "scene", None)
        if not (
            self.motion_sampling == "adaptive"
            and authored_world is not None
            and not authored_world.structures
            and self.propagation.components == frozenset({"los"})
            and frozen is not None
        ):
            return False
        site_count = _round_trip(frozen).binding.site_count
        return (
            frozen.handles[0].row_count == self.array.num_tx * site_count
            and frozen.handles[1].row_count == self.array.num_rx * site_count
        )

    def _binding_for(self, frame: EpochFrame) -> RadarWorldBinding:
        """The endpoint specs at ``frame``'s instant.

        A frame that just froze replays against the binding it froze with. An
        explicit site policy without moving sensor endpoints re-resolves only
        the site positions; anything else rebinds the radar to the snapshot.
        """

        template = _round_trip(frame.frozen).binding
        if frame.rediscovered:
            return template
        if self.sites.source == SITE_SOURCE_EXPLICIT and self.sensor_endpoints is None and self.pose is None:
            positions = self.sites.resolve(frame.snapshot, device=template.device)
            if positions.shape != template.site_positions_m.shape:
                raise ValueError("a site trajectory must preserve the declared site count")
            return replace(template, sites=with_site_positions(template.sites, positions))
        return self._bind_world(frame.snapshot, frame.time_s)

    def _identity(self, frame: EpochFrame) -> Hashable:
        """The leg identity the adaptive controller groups and compares by.

        Every other route groups nothing and compares nothing, so it is ``()``
        there.
        """

        if self.motion_sampling != "adaptive":
            return ()
        return _round_trip(frame.frozen).adaptive_identity

    def evaluate(self, query_times: Sequence[float]) -> list[EvaluatedObservation]:
        """Advance the world to each instant, then batch-replay identical handles.

        One record per query instant, in order. Moving compiled geometry is
        replayed before the next refreeze retires its handles. Static geometry
        permits one slot batch per full leg identity.
        """

        results: dict[float, EvaluatedObservation] = {}
        groups: dict[Hashable, list[_PendingObservation]] = {}
        positions = self._positions(query_times)
        for time_s in query_times:
            self._clock = time_s
            frame = self.loop.frame(
                time_s, topology_complete=self._complete_los(), displaced=self._moved(positions, time_s)
            )
            if positions is not None and frame.rediscovered:
                self._reference = positions[time_s]
            pending = self._pending(time_s, frame)
            if self.loop.structures_move or getattr(self.response, "varies_per_observation", False):
                self._replay([pending], results)
            else:
                groups.setdefault(_round_trip(frame.frozen).identity, []).append(pending)
        for group in groups.values():
            self._replay(group, results)
        return [results[time_s] for time_s in query_times]

    def _positions(self, query_times: Sequence[float]) -> dict[float, np.ndarray] | None:
        """Every site and endpoint position at every instant, for a displacement cadence.

        The group is bound up front and read to the host in ONE copy. Without
        a displacement cadence this is ``None`` and costs nothing.
        """

        if self.displacement_m is None:
            return None
        bindings = [self._bind_world(self.dynamic.at(time_s), time_s) for time_s in query_times]
        stacked = torch.stack(
            [
                torch.cat(
                    (binding.transmitters.positions_m, binding.receivers.positions_m, binding.site_positions_m)
                ).detach()
                for binding in bindings
            ]
        )
        return dict(zip(query_times, stacked.double().cpu().numpy(), strict=True))

    def _moved(self, positions: dict[float, np.ndarray] | None, time_s: float) -> bool:
        """Whether anything moved ``displacement_m`` since the last discovery."""

        if positions is None or self._reference is None:
            return False
        assert self.displacement_m is not None
        return float(np.linalg.norm(positions[time_s] - self._reference, axis=1).max()) >= self.displacement_m

    def _pending(self, time_s: float, frame: EpochFrame) -> _PendingObservation:
        binding = self._binding_for(frame)
        rotation = None if self.pose is None else self._rotation_at(time_s, binding.device)[0]
        return _PendingObservation(
            time_s=time_s, epoch_frame=frame, binding=binding, identity=self._identity(frame), rotation=rotation
        )

    def _response_at(self, group: list[_PendingObservation]) -> ScatterResponse:
        """The scatter response for one replay group, one site set per instant when it varies in time."""

        response = (
            self.response if self.timed_response is None else self.timed_response.at([entry.time_s for entry in group])
        )
        return _bound(response, group[0].binding, group[0].time_s)

    def _body_frame(
        self, group: list[_PendingObservation], pattern: RoundTripPatternStage, *points: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """World points seen in each instant's radar body frame, for the pattern stage.

        The stage reads only DIRECTIONS (differences of these points) and maps
        them through the authored pattern frame. Turning every point of slot
        ``g`` by ``R_g^T`` turns each of those directions into the body frame,
        so the authored frame then gives the pattern orientation the pose
        carried. One batched matrix product per point set; nothing per row.
        """

        if self.pose is None or pattern.plan.is_identity:
            return points
        rotations = torch.stack([entry.rotation for entry in group if entry.rotation is not None])
        slots = len(group)
        return tuple(
            torch.matmul(value.reshape(slots, -1, 3), rotations).reshape(-1, 3).contiguous() for value in points
        )

    def _replay(self, group: list[_PendingObservation], results: dict[float, EvaluatedObservation]) -> None:
        """Replay one group of topology-identical instants as one slot batch."""

        bindings = [entry.binding for entry in group]
        transmitters = _stacked([binding.transmitters for binding in bindings])
        receivers = _stacked([binding.receivers for binding in bindings])
        replay, discovered = self._group_legs(group, transmitters, receivers)
        if len(group) == 1:
            entry = group[0]
            round_trip = _round_trip(entry.epoch_frame.frozen)
            composer = round_trip.composer
            response = self._response_at(group)
            paths = composer.compose(replay.inbound, replay.outbound, response, replay.direct)
            tx_pos, rx_pos, tx_targets, rx_targets = self._body_frame(
                group,
                round_trip.pattern,
                entry.binding.transmitters.positions_m,
                entry.binding.receivers.positions_m,
                *composer.segment_targets(replay.inbound, replay.outbound, replay.direct),
            )
            paths = round_trip.pattern.apply(
                paths, tx_pos=tx_pos, rx_pos=rx_pos, tx_targets_m=tx_targets, rx_targets_m=rx_targets
            )
            band = self._band(
                round_trip,
                replay,
                1,
                None if self._strided(round_trip) else ((composer, round_trip.pattern), (round_trip.identity, 1)),
                transmitters,
                receivers,
                response,
            )
            if band is not None:
                paths = replace(
                    paths, complex_transfer_band=band, frequency_offsets_hz=self.request.frequency_offsets_hz
                )
            identity = entry.identity
            if self.motion_sampling == "adaptive":
                identity = (identity, _present(paths.row_valid, "row_valid").cpu().numpy().tobytes())
            results[entry.time_s] = EvaluatedObservation(
                time_s=entry.time_s,
                epoch_frame=entry.epoch_frame,
                legs=lambda: replay,
                composed=paths,
                identity=identity,
                batch=SlotBatch(
                    rows=SlotRows(
                        total_delay_s=paths.total_delay_s,
                        complex_transfer_ref=paths.complex_transfer_ref,
                        row_valid=paths.row_valid,
                        slots=1,
                        rows_per_slot=paths.path_count,
                        weight_includes_antenna_pattern=paths.weight_includes_antenna_pattern,
                        complex_transfer_band=paths.complex_transfer_band,
                        frequency_offsets_hz=paths.frequency_offsets_hz,
                        total_delay_residual_s=paths.total_delay_residual_s,
                    ),
                    offsets=round_trip.composer.pair_offsets_host,
                    slots=1,
                ),
                slot=0,
            )
            return
        batched = self._compose_slots(group, replay, transmitters, receivers)
        slot_legs = partial(_slot_legs, replay) if discovered is None else discovered.__getitem__
        shared = SlotBatch(
            rows=batched, offsets=_round_trip(group[0].epoch_frame.frozen).composer.pair_offsets_host, slots=len(group)
        )
        batched_valid = _present(batched.row_valid, "row_valid")
        valid_rows = batched_valid.reshape(len(group), -1).cpu().numpy() if self.motion_sampling == "adaptive" else None
        for slot, entry in enumerate(group):
            composer = _round_trip(entry.epoch_frame.frozen).composer
            selection = slice(slot * composer.path_count, (slot + 1) * composer.path_count)
            paths = RadarPathBatch(
                sensor_pair_count=composer.sensor_pair_count,
                path_count=composer.path_count,
                sensor_pair_index=composer.sensor_pair_index,
                pair_offsets=composer.pair_offsets,
                total_delay_s=batched.total_delay_s[selection],
                delay_rate=None,
                complex_transfer_ref=batched.complex_transfer_ref[selection],
                reference_frequency_hz=composer.reference_frequency_hz,
                row_valid=batched_valid[selection],
                topology=composer.topology,
                weight_includes_antenna_pattern=batched.weight_includes_antenna_pattern,
                complex_transfer_band=None
                if batched.complex_transfer_band is None
                else batched.complex_transfer_band[selection],
                frequency_offsets_hz=batched.frequency_offsets_hz,
                total_delay_residual_s=(
                    None if batched.total_delay_residual_s is None else batched.total_delay_residual_s[selection]
                ),
            )
            identity = entry.identity
            if valid_rows is not None:
                identity = (identity, valid_rows[slot].tobytes())
            # The per-slot narrowing is DEFERRED: a batched group composes once
            # and reads nothing from its individual slots, so narrowing eagerly
            # would build two revalidated single-slot batches per observation
            # for a record whose legs only a published observation reads.
            results[entry.time_s] = EvaluatedObservation(
                time_s=entry.time_s,
                epoch_frame=entry.epoch_frame,
                legs=partial(slot_legs, slot),
                composed=paths,
                identity=identity,
                batch=shared,
                slot=slot,
            )

    def _group_legs(
        self, group: list[_PendingObservation], transmitters: RadarEndpointSpec, receivers: RadarEndpointSpec
    ) -> tuple[RadarPropagationLegs, list[RadarPropagationLegs] | None]:
        """One group's legs, slot major, and each instant's own legs when it has them.

        A primal group whose every instant rediscovered answers from the rows
        those discoveries evaluated: no replay call at all, and the slot-major
        stack is one concatenation per member. Any other group is replayed in
        one slot-batched Channel call per leg, against the first instant's
        handles, which every instant of the group shares by identity.
        """

        if self.discovered_rows and all(entry.epoch_frame.rediscovered for entry in group):
            own = [_discovered_legs(entry.epoch_frame.frozen) for entry in group]
            if len(own) == 1:
                return own[0], None
            direct = own[0].direct
            return (
                RadarPropagationLegs(
                    inbound=stack_slots([legs.inbound for legs in own]),
                    outbound=stack_slots([legs.outbound for legs in own]),
                    direct=None if direct is None else stack_slots([_present_leg(legs.direct) for legs in own]),
                ),
                own,
            )
        first = group[0].epoch_frame.frozen
        bindings = [entry.binding for entry in group]
        slots = len(group)
        offsets = self.request.frequency_offsets_hz

        def replayed(
            handle: Any, sources: RadarEndpointSpec, sinks: RadarEndpointSpec, element_sinks: bool = False
        ) -> RadarLegBatch:
            batch: RadarLegBatch = first.adapter.reevaluate_slots(
                handle,
                sources,
                sinks,
                slot_count=slots,
                ad_mode=self.ad_mode,
                frequency_offsets_hz=offsets,
                element_sinks=element_sinks,
            )
            return batch

        sites = _stacked([b.sites for b in bindings])
        return (
            RadarPropagationLegs(
                inbound=replayed(first.handles[0], transmitters, sites),
                outbound=replayed(first.handles[1], _stacked([b.receiver_sources for b in bindings]), sites),
                direct=None
                if len(first.handles) < 3
                else replayed(first.handles[2], transmitters, receivers, element_sinks=True),
            ),
            None,
        )

    def _compose_slots(
        self,
        group: list[_PendingObservation],
        replay: RadarPropagationLegs,
        transmitters: RadarEndpointSpec,
        receivers: RadarEndpointSpec,
    ) -> SlotRows:
        """Compose and pattern-weight a slot-major replay.

        A per-site response is composed by the join striding its single-slot
        tables across the slots, and the pattern stage strides its own the same
        way (the identity under a unit-gain pattern). A geometry-dependent
        response still needs the join's per-row tables for every slot, which
        are built once per identity and group size and kept in a bounded cache.
        """

        round_trip = _round_trip(group[0].epoch_frame.frozen)
        slots = len(group)
        if self._strided(round_trip):
            composer = round_trip.composer
            response = self._response_at(group)
            rows = composer.compose_slots(replay.inbound, replay.outbound, response, slots, replay.direct)
            band = self._band(round_trip, replay, slots, None, transmitters, receivers, response)
            if round_trip.pattern.plan.is_identity:
                # The stage's identity: isotropic elements at 0 dBi publish the
                # weight unchanged and record that the pattern was applied.
                rows = replace(rows, weight_includes_antenna_pattern=True)
            else:
                tx_targets, rx_targets = composer.segment_targets(
                    replay.inbound, replay.outbound, replay.direct, slots=slots
                )
                rows = round_trip.pattern.apply_slots(
                    rows,
                    tx_pos=transmitters.positions_m,
                    rx_pos=receivers.positions_m,
                    tx_targets_m=tx_targets,
                    rx_targets_m=rx_targets,
                )
            return replace(
                rows,
                complex_transfer_band=band,
                frequency_offsets_hz=None if band is None else self.request.frequency_offsets_hz,
            )
        key = (round_trip.identity, slots)
        tables = self._slot_tables.get(key)
        if tables is None:
            tables = (
                round_trip.composer.for_slots(slots, per_slot_response=self.timed_response is not None),
                round_trip.pattern.for_slots(slots),
            )
            self._slot_tables.put(key, tables)
        composer, pattern = tables
        response = self._response_at(group)
        batched = composer.compose(replay.inbound, replay.outbound, response, replay.direct)
        tx_pos, rx_pos, tx_targets, rx_targets = self._body_frame(
            group,
            pattern,
            transmitters.positions_m,
            receivers.positions_m,
            *composer.segment_targets(replay.inbound, replay.outbound, replay.direct),
        )
        weighted = pattern.apply(
            batched, tx_pos=tx_pos, rx_pos=rx_pos, tx_targets_m=tx_targets, rx_targets_m=rx_targets
        )
        return SlotRows(
            total_delay_s=weighted.total_delay_s,
            complex_transfer_ref=weighted.complex_transfer_ref,
            row_valid=weighted.row_valid,
            slots=slots,
            rows_per_slot=round_trip.composer.path_count,
            weight_includes_antenna_pattern=weighted.weight_includes_antenna_pattern,
            complex_transfer_band=self._band(
                round_trip, replay, slots, (tables, key), transmitters, receivers, response
            ),
            frequency_offsets_hz=self.request.frequency_offsets_hz,
            total_delay_residual_s=weighted.total_delay_residual_s,
        )

    def _strided(self, round_trip: FrozenRoundTrip) -> bool:
        """Whether this group composes by striding single-slot tables.

        The join strides them for every response - a per-site one shared by
        the slots, a per-row one evaluated by its native family over the same
        single-slot tables - and the pattern stage for any pattern seen in the
        world frame; a radar pose turns each slot's pattern frame, which the
        per-row route carries.
        """

        return round_trip.pattern.plan.is_identity or self.pose is None

    def _band(
        self,
        round_trip: FrozenRoundTrip,
        replay: RadarPropagationLegs,
        slots: int,
        tables: tuple[tuple[TwoWayComposer, RoundTripPatternStage], tuple[Hashable, int]] | None,
        transmitters: RadarEndpointSpec,
        receivers: RadarEndpointSpec,
        response: ScatterResponse,
    ) -> torch.Tensor | None:
        """The wideband columns of a replay, ``[slots * rows, F]``, or ``None`` on a narrowband run.

        The frequency columns become outer slots (:func:`band_as_slots`), so the
        same join and pattern equations compose all of them: a strided join
        composes ``F * slots`` slots of its single-slot tables in one launch,
        and a per-row join replicates its group tables once per column count,
        cached with the slot tables. Antenna patterns are frequency
        independent, which is why one pattern stage serves every column.
        ``tables`` is the per-row route's group tables and their cache key,
        ``None`` on the strided route.
        """

        offsets = self.request.frequency_offsets_hz
        if offsets is None:
            return None
        if self.timed_response is not None or self.pose is not None:
            raise NotImplementedError(
                "wideband=True replicates one response and one pattern orientation across the frequency columns; "
                "a time-varying Aspect.axis or a radar pose is evaluated per instant and is not combined with it"
            )
        columns = len(offsets)
        inbound, outbound = band_as_slots(replay.inbound), band_as_slots(replay.outbound)
        direct = None if replay.direct is None else band_as_slots(replay.direct)
        if tables is None:
            composer = round_trip.composer
            rows = composer.compose_slots(
                inbound, outbound, response, slots * columns, direct, frequency_offsets_hz=offsets
            )
            if not round_trip.pattern.plan.is_identity:
                tx_targets, rx_targets = composer.segment_targets(inbound, outbound, direct, slots=slots * columns)
                rows = round_trip.pattern.apply_slots(
                    rows,
                    tx_pos=transmitters.positions_m.repeat(columns, 1),
                    rx_pos=receivers.positions_m.repeat(columns, 1),
                    tx_targets_m=tx_targets,
                    rx_targets_m=rx_targets,
                )
            weight = rows.complex_transfer_ref
        else:
            (group_composer, group_pattern), key = tables
            band_key = (key, columns)
            band_tables = self._slot_tables.get(band_key)
            if band_tables is None:
                band_tables = (group_composer.for_slots(columns), group_pattern.for_slots(columns))
                self._slot_tables.put(band_key, band_tables)
            composer, pattern = band_tables
            composed = composer.compose(inbound, outbound, response, direct, frequency_offsets_hz=offsets)
            tx_targets, rx_targets = composer.segment_targets(inbound, outbound, direct)
            weight = pattern.apply(
                composed,
                tx_pos=transmitters.positions_m.repeat(columns, 1),
                rx_pos=receivers.positions_m.repeat(columns, 1),
                tx_targets_m=tx_targets,
                rx_targets_m=rx_targets,
            ).complex_transfer_ref
        return weight.reshape(columns, -1).transpose(0, 1).contiguous()

    def observations(self, start: float, offsets: np.ndarray, events: PathEvents) -> Iterator[Observation]:
        """One frame's evaluated observations, in schedule order.

        A generator rather than a list: the fused route drops each replay group
        once it has been synthesized, so its retention is bounded by
        ``Motion.batch_observations`` rather than by the
        ``chirps * transmitters * samples`` observations an ADC-refreshed frame
        has. :func:`trace_scene` materialises the same generator and pays that
        cost deliberately.

        Both consumers drain a frame completely before asking for the next.
        These observations share the epoch loop's state, so interleaving two
        frames would replay one frame's legs against another frame's frozen
        topology.
        """

        self.begin_report()
        if self.still_world:
            yield from self._still_observations(start, offsets)
            return
        # A frame's instants are evaluated in groups of ``batch_observations``:
        # topology-identical instants of a group share one slot-major replay
        # and one composition, and the group is the largest unit of rows the
        # route holds at once.
        for begin in range(0, len(offsets), self.batch_observations):
            times = [float(start + offset) for offset in offsets[begin : begin + self.batch_observations]]
            records = self.evaluate(times)
            if self.recorder is not None:
                self.recorder.record(
                    events,
                    [
                        Observed(
                            time_s=record.time_s,
                            observation=begin + position,
                            pool=record.batch.rows,
                            offset=record.slot * _round_trip(record.epoch_frame.frozen).composer.path_count,
                            composer=_round_trip(record.epoch_frame.frozen).composer,
                            discovered=record.epoch_frame.rediscovered or record.epoch_frame.topology_complete,
                        )
                        for position, record in enumerate(records)
                    ],
                    _batch_pool,
                )
            for record in records:
                yield Observation(
                    time_s=record.time_s,
                    epoch_frame=record.epoch_frame,
                    legs_of=record.legs,
                    composed=record.composed,
                    batch=record.batch,
                    slot=record.slot,
                )
        if self.recorder is not None:
            self.recorder.close_frame(events)

    def _still_observations(self, start: float, offsets: np.ndarray) -> Iterator[Observation]:
        for offset in offsets:
            time_s = float(start + offset)
            held = self._still
            if held is not None:
                self._clock = time_s
                frame = self.loop.frame(time_s)
                if not frame.rediscovered and frame.frozen is held.epoch_frame.frozen:
                    yield replace(held, time_s=time_s, epoch_frame=frame)
                    continue
                record = self._replay_frame(time_s, frame)
            else:
                record = self.evaluate([time_s])[0]
            observation = Observation(
                time_s=time_s,
                epoch_frame=record.epoch_frame,
                legs_of=partial(_constant_legs, record.legs()),
                composed=record.composed,
                batch=record.batch,
                slot=record.slot,
            )
            if not _carries_derivative(observation.composed):
                self._still = observation
            yield observation

    def _replay_frame(self, time_s: float, frame: EpochFrame) -> EvaluatedObservation:
        """Replay one instant whose world the loop has already advanced."""

        results: dict[float, EvaluatedObservation] = {}
        self._replay([self._pending(time_s, frame)], results)
        return results[time_s]


# ---------------------------------------------------------------------------
# Opening a session
# ---------------------------------------------------------------------------


def _working_spec(radar: Radar, declared: WaveformSpec) -> WaveformSpec:
    """The spec synthesis runs: the beat domain whenever the frame needs ADC-time samples.

    Receiver stages and oscillator noise act on ADC-time samples, and so do a
    chirp sequence, a ramp phase error and a real ADC (the closed-form spectrum
    describes none of them). The direct-spectrum fast path is kept only for an
    ideal receiver on a uniform linear ramp.
    """

    from .synthesis.assembly import FmcwSpec, OfdmSpec

    if isinstance(declared, FmcwSpec) and (
        radar.frontend is not None or not (declared.admits_direct_spectrum and declared.uniform)
    ):
        return replace(declared, output_domain="beat")
    if isinstance(declared, OfdmSpec) and declared.time_domain:
        # A modulated stream is received as time samples; its CFR is the
        # demodulated estimate, formed after the receive chain.
        return replace(declared, output_domain="time")
    return declared


def _resolve_sampling(requested: Motion, *, sampled: bool, fmcw: bool) -> str:
    """The route a request runs, once it is known whether anything moves.

    ``auto`` resolves to the adaptive route for a world that moves under an
    FMCW waveform, because the exhaustive ADC route rediscovers at every ADC
    instant and is the reference, not the default. Adaptive interpolation is
    FMCW only and owns its own discovery cadence, so a non-FMCW waveform or a
    declared discovery cadence resolves ``auto`` to the exhaustive
    route instead.

    ``static`` is refused rather than silently upgraded: it asks for one
    observation a frame, and a world that moves inside a frame would publish
    no Doppler at all. A still world evaluates the same single observation
    per frame whatever the kind, which the rest of the session spells ``adc``.
    """

    motion_sampling = requested.kind
    if requested.kind == "auto":
        adaptive = sampled and fmcw and requested.cadence is None
        motion_sampling = "adaptive" if adaptive else ("adc" if sampled else "static")
    if motion_sampling == "static" and sampled:
        raise ValueError(
            "Motion.static() asks for one observation per frame, but this session moves within a frame; "
            "use Motion.adc() for the exhaustive reference, Motion.chirp() for stop-and-hop, or "
            "Motion.adaptive() for error-controlled interpolation"
        )
    return "adc" if motion_sampling == "static" else motion_sampling


def _schedule(route: str, *, sampled: bool, adc_sampled: bool) -> str:
    """The observation schedule a session runs, as the synthesis owner names it.

    One of :data:`~witwin.radar.synthesis.assembly.OBSERVATION_SCHEDULES`:
    ``frame`` for a still world, ``adaptive`` for the adaptive route,
    ``fast_time`` when every fast-time sample (an FMCW ADC instant, a pulsed
    fast-time sample, a time-domain OFDM sample) is its own observation, and
    ``slow_time`` when the world is held over each chirp, symbol or pulse.
    """

    from .synthesis.assembly import (
        OBSERVATION_ADAPTIVE,
        OBSERVATION_FAST_TIME,
        OBSERVATION_FRAME,
        OBSERVATION_SLOW_TIME,
    )

    if not sampled:
        return OBSERVATION_FRAME
    if route == "adaptive":
        return OBSERVATION_ADAPTIVE
    return OBSERVATION_FAST_TIME if adc_sampled else OBSERVATION_SLOW_TIME


def _published_sampling(schedule: str, working: object) -> tuple[str, bool]:
    """``(motion_sampling, exhaustive)`` derived from the observation schedule that ran.

    The kind names the schedule, not the request: ``fast_time`` is ``"adc"``,
    ``adaptive`` is ``"adaptive"``, a still world is ``"static"``, and a
    ``slow_time`` schedule is named by what it holds the world over, ``"chirp"``
    (FMCW), ``"symbol"`` (OFDM) or ``"pulse"`` (pulsed). ``exhaustive`` says the
    world was evaluated at EVERY instant the waveform puts a sample at, which
    is exactly the ``frame`` and ``fast_time`` schedules; an adaptive frame is
    decided per frame by its controller.
    """

    from .synthesis.assembly import OBSERVATION_ADAPTIVE, OBSERVATION_FAST_TIME, OBSERVATION_FRAME, OfdmSpec, PulsedSpec

    if schedule == OBSERVATION_FRAME:
        return "static", True
    if schedule == OBSERVATION_ADAPTIVE:
        return "adaptive", False
    if schedule == OBSERVATION_FAST_TIME:
        return "adc", True
    if isinstance(working, OfdmSpec):
        return "symbol", False
    if isinstance(working, PulsedSpec):
        return "pulse", False
    return "chirp", False


def open_session(
    radar: Radar,
    scene: object,
    *,
    times: Sequence[float],
    response: ScatterResponse,
    sites: ScatterSitePolicy,
    components: frozenset[str],
    max_depth: int,
    ad_mode: str,
    polarization: Sequence[float],
    antenna_pattern: Pattern,
    sensor_endpoints: SensorEndpointIds | None,
    motion: Motion,
    environment: bool = False,
    coupling: bool = False,
    discovery: str = "auto",
    max_paths: int | None = None,
    max_round_trips: int | None = None,
    wideband: bool = False,
    pose: Callable[[float], Any] | None = None,
    rx_polarization: Sequence[float] | None = None,
) -> tuple[Session, Iterator[FrameTrace]]:
    """Open one scene session: the world half, and what the instrument half needs.

    The single owner of the session; every argument arrives resolved by
    :meth:`~witwin.radar.Radar._session`. Returns the session record and a
    generator of per-frame traces.

    ``sites`` is the internal :class:`~witwin.radar.binding.ScatterSitePolicy`
    that :mod:`witwin.radar.targets` builds. A structure with no rigid motion
    has no Core-owned world anchor and is refused by name (R-ADR-020).

    ``components`` and ``max_depth`` carry the ``los`` / ``reflections``
    request for THIS call through
    :meth:`~witwin.radar.radar.RadarSystemConfig.with_propagation`, which
    returns a new configuration rather than mutating the radar's stored one.

    ``motion`` is a :class:`~witwin.radar.motion.Motion`. ``Motion.auto()``
    resolves here, once it is known whether anything moves within a frame and
    whether the receiver carries oscillator phase noise. Adaptive FMCW batches
    topology-identical probes, interpolates carrier-transported coefficients,
    and refines at observed path identity and validity changes; its sampled
    error tests cannot exclude arbitrarily brief unseen events between probes,
    so ``path_set_complete`` is true only when every observation was evaluated
    or the candidate family was certified complete for all time, and
    ``motion_sampling_exhaustive`` separately records whether any observation
    was interpolated. ``Motion.rediscover_every_frames`` converts from frames to
    observation count and marks path completeness false unless structure
    motion already forces discovery; ``Motion.world`` still controls whether
    compiled handles may be replayed.

    ``ad_mode`` is forwarded to every replay. ``"none"`` builds no graph;
    ``"vjp"`` makes the published cube differentiable with respect to the
    endpoint and site positions the binding passed through by identity.

    ``antenna_pattern`` is a :class:`~witwin.radar.sensors.Pattern`. A radar
    always supplies one; the isotropic default is a unit gain, which the stage
    publishes without a launch.

    ``environment`` and ``coupling`` add the third, site-less leg, transmitter
    straight to receiver: ``environment`` asks it for the session's reflection
    and diffraction components (the environment's own returns, such as a wall
    facing the radar), ``coupling`` for the straight segment (line of sight,
    and transmission when requested). Its rows join each sensor pair's segment
    with ``site_id == -1``. ``discovery`` and ``max_paths`` are forwarded to
    every Channel discovery, and ``max_round_trips`` bounds one sensor pair's
    composed rows, checked before the join.

    ``wideband`` evaluates every OFDM subcarrier's transport rather than
    shifting the reference coefficient: each replay asks Channel for the same
    frozen rows at ``f_ref + n * df`` in one call, the columns are composed
    with the reference column's equations, and the CFR kernel reads one weight
    per subcarrier. Spreading tilt and the material and layer response across
    the band are then exact. Only OFDM consumes a column per frequency.

    ``transmission`` and ``diffraction`` in ``components`` cannot be frozen by
    Channel, so they are admitted only where no row is replayed: ``grad="none"``
    with a discovery at every evaluated instant, which a primal run answers
    straight from the discovery's own rows.
    """

    from .synthesis.assembly import (
        FmcwSpec,
        OfdmSpec,
        PulsedSpec,
        SlowTimeMode,
        fast_time_grid,
        fast_time_offsets,
        fast_time_single,
        waveform_sampling,
    )

    if not isinstance(motion, Motion):
        raise TypeError(f"motion must be a Motion, got {type(motion).__name__}")
    instants = _times(times)
    if not isinstance(sites, ScatterSitePolicy):
        raise TypeError(
            f"sites must be a ScatterSitePolicy, got {type(sites).__name__}; "
            "where the scatter sites come from is a declaration, not a search"
        )
    direct_components = _direct_components(frozenset(components), environment=environment, coupling=coupling)
    solve_config = radar.system_config.with_propagation(components=components, max_depth=max_depth)
    propagation = solve_config.propagation
    array = solve_config.sensors.array
    reference_frequency_hz = propagation.reference_frequency_hz

    dynamic = _dynamic_scene(scene)
    if dynamic.endpoint_trajectories and sensor_endpoints is None:
        raise ValueError("moving Core endpoints require explicit sensor_endpoints in array order")
    from .targets import TimedAspectResponse

    if pose is not None and not callable(pose):
        raise TypeError("pose must be callable as pose(t) -> (rotation (3, 3), translation (3,))")
    sampled = bool(
        dynamic.structure_trajectories
        or dynamic.structure_deformations
        or dynamic.endpoint_trajectories
        or sites.trajectory is not None
        or pose is not None
        or isinstance(response, TimedAspectResponse)
    )
    declared = solve_config.waveform_spec()
    path_phase_noise = radar.frontend is not None and radar.frontend.has_phase_noise
    if path_phase_noise:
        if not isinstance(declared, FmcwSpec):
            raise NotImplementedError("scene-driven common-oscillator phase noise currently requires FMCW")
        # Checked against the REQUEST, because the kind is not resolved yet:
        # this branch is one of the things that decides it.
        if motion.kind not in ("auto", "adc", "adaptive"):
            raise ValueError("common-oscillator phase noise requires ADC-time observations")
        sampled = True
    working = _working_spec(radar, declared)
    # Adaptive interpolation covers one uniform TDM ramp, so only that is
    # what ``auto`` may hand it.
    interpolable = (
        isinstance(working, FmcwSpec)
        and working.uniform
        and not working.phase_error_cycles
        and not working.simultaneous
    )
    motion_sampling = _resolve_sampling(motion, sampled=sampled, fmcw=interpolable)
    if motion_sampling == "adaptive" and getattr(response, "per_observation", False):
        raise ValueError(
            "a pulse-to-pulse fluctuation (Swerling II or IV) redraws at every observation, and adaptive "
            "sampling interpolates between the instants it probes, which would blend independent draws; "
            "use Motion.adc() or Motion.chirp(), or a scan-to-scan model (Swerling I or III)"
        )
    if motion_sampling == "adaptive" and not isinstance(working, FmcwSpec):
        raise NotImplementedError("adaptive motion currently requires FMCW")
    if motion_sampling == "adaptive" and isinstance(working, FmcwSpec):
        if not working.uniform or working.phase_error_cycles or working.simultaneous:
            raise NotImplementedError(
                "adaptive motion interpolates one uniform TDM ramp; a chirp sequence, a start offset, a ramp phase "
                "error, DDM and BPM run with Motion.adc() or Motion.chirp()"
            )
    if isinstance(working, FmcwSpec) and working.simultaneous and radar.coupling is not None:
        raise NotImplementedError(
            "array mutual coupling acts on separately recorded transmit channels, and DDM and BPM record every "
            "transmitter on one; run TDM to model coupling"
        )
    if motion_sampling == "adaptive" and motion.cadence is not None:
        raise ValueError("adaptive motion owns its topology discovery cadence")
    # FMCW and pulsed trains have a fast-time grid of their own, so an ADC
    # request observes the world at every sample of it.
    fast_grid = isinstance(working, (FmcwSpec, PulsedSpec)) or (isinstance(working, OfdmSpec) and working.time_domain)
    adc_sampled = sampled and motion_sampling in ("adc", "adaptive") and fast_grid
    if sampled and fast_grid and not isinstance(working, FmcwSpec) and radar.leakage is not None:
        raise NotImplementedError(
            "a refreshed pulsed or time-domain OFDM frame is synthesized one slot batch at a time, and a declared "
            "leakage keeps one launch per observation; the two are not combined"
        )
    samples_per_slot = fast_time_grid(working)[2] if adc_sampled else 1

    mode = SlowTimeMode.FROZEN_WEIGHT_WITH_CARRIER_RATE
    cadence: int | None
    period_s = motion.rediscover_every_seconds
    displacement_m = motion.rediscover_on_displacement
    pair_samples: torch.Tensor | None = None
    single: WaveformSpec | None = None
    if sampled:
        slot_offsets, pair_samples, single = waveform_sampling(
            working, num_tx=array.num_tx, num_rx=array.num_rx, device=radar.device
        )
        offsets = np.asarray(slot_offsets, dtype=np.float64)
        if adc_sampled:
            first, period, count = fast_time_grid(working)
            if isinstance(working, FmcwSpec):
                slot_period = working.chirp_period_s
            elif isinstance(working, PulsedSpec):
                slot_period = working.pri_s
            else:
                slot_period = working.symbol_period_s
            # A chirp sequence validates each profile's own window instead.
            uniform = not isinstance(working, FmcwSpec) or not working.sequence
            if uniform and first + (count - 1) * period >= slot_period:
                raise ValueError("ADC observations must fit inside each slow-time slot")
            # An ADC frame has slots * samples observations, so the schedule is
            # an array rather than Python floats: the same additions in the
            # same order, paid once by numpy instead of once per observation.
            offsets = fast_time_offsets(working, offsets)
            single = fast_time_single(single)
        mode = SlowTimeMode.REFRESHED_WEIGHT_NO_RATE
        # A version poll cannot detect endpoint-induced path births. Complete
        # dynamic sampling therefore rediscovers at each observed instant;
        # a declared cadence converts to observations here, a time or
        # displacement cadence is decided per observation by the world half.
        cadence = 1 if motion.cadence is None else None
        if motion.rediscover_every_frames is not None:
            cadence = motion.rediscover_every_frames * len(offsets)
        elif motion.rediscover_every_chirps is not None:
            cadence = motion.rediscover_every_chirps * samples_per_slot
    else:
        offsets = np.zeros(1)
        # One observation a frame: a chirp cadence is a frame cadence, and
        # nothing in a still world can move by a displacement.
        cadence = motion.rediscover_every_frames
        if motion.rediscover_every_chirps is not None:
            cadence = 1
        displacement_m = None
    if sampled and any(b <= a + offsets[-1] for a, b in zip(instants, instants[1:], strict=False)):
        raise ValueError("dynamic frames must not overlap in observation time")
    still_world = (
        not sampled and cadence is None and period_s is None and not getattr(response, "varies_per_observation", False)
    )
    band_offsets = _subcarrier_offsets(working) if wideband else None
    unfrozen = (frozenset(components) | direct_components) & UNFREEZABLE_COMPONENTS
    if unfrozen:
        if band_offsets is not None:
            raise NotImplementedError(
                f"components {sorted(unfrozen)} cannot be frozen, and a wideband band is a replay of frozen "
                "rows at every subcarrier; drop wideband=True or those components"
            )
        _require_discovery_only(sorted(unfrozen), ad_mode=ad_mode, still_world=still_world, cadence=cadence)
    request = LegRequest(
        discovery=discovery,
        max_paths=max_paths,
        max_round_trips=max_round_trips,
        direct_components=direct_components,
        replays=not unfrozen,
        frequency_offsets_hz=band_offsets,
    )

    kind, exhaustive = _published_sampling(
        _schedule(motion_sampling, sampled=sampled, adc_sampled=adc_sampled), working
    )
    world = WorldHalf(
        radar,
        dynamic,
        sites=sites,
        polarization=polarization,
        rx_polarization=polarization if rx_polarization is None else rx_polarization,
        sensor_endpoints=sensor_endpoints,
        response=response,
        antenna_pattern=antenna_pattern,
        propagation=propagation,
        array=array,
        motion_sampling=motion_sampling,
        ad_mode=ad_mode,
        cadence=cadence,
        period_s=period_s,
        displacement_m=displacement_m,
        world_motion=motion.world,
        pose=pose,
        still_world=still_world,
        batch_observations=int(motion.batch_observations),
        request=request,
    )
    session = Session(
        radar=radar,
        array=array,
        loop=world.loop,
        mode=mode,
        specs=InstrumentSpecs(declared=declared, working=working, single=single),
        offsets=offsets,
        pair_samples=pair_samples,
        sampled=sampled,
        adc_sampled=adc_sampled,
        samples_per_slot=samples_per_slot,
        path_phase_noise=path_phase_noise,
        reference_frequency_hz=reference_frequency_hz,
        kind=kind,
        components=frozenset(propagation.components),
        max_depth=int(propagation.max_depth),
        grad=str(ad_mode),
        motion=motion,
        times=instants,
        environment=environment,
        coupling=coupling,
    )
    # Whether a non-adaptive route can have missed a path birth is a property
    # of the declared cadence and world motion, not of any one frame, so it is
    # decided once here and repeated on every frame the run publishes.
    complete = (
        not sampled
        or (cadence == 1 and period_s is None and displacement_m is None)
        or (motion.world == "frozen_world" and world.loop.structures_move)
    )
    if sampled and motion_sampling == "adaptive":
        return session, _adaptive_traces(session, world)
    return session, _frame_traces(session, world, path_set_complete=bool(complete), exhaustive=exhaustive)


#: Components Channel publishes at discovery and cannot freeze for replay.
UNFREEZABLE_COMPONENTS = frozenset({"transmission", "diffraction"})


def _direct_components(components: frozenset[str], *, environment: bool, coupling: bool) -> frozenset[str]:
    """The site-less TX->RX leg's components, empty when the run composes none.

    ``environment`` takes the session's interacting components - the paths that
    leave the transmitter, meet the environment and come back - and
    ``coupling`` the straight segment between the elements: the line of sight,
    and transmission through whatever lies on it when transmission is
    requested. Neither borrows a component the session did not ask for, except
    that coupling IS the line of sight.
    """

    for name, value in (("environment", environment), ("coupling", coupling)):
        if type(value) is not bool:
            raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
    direct: set[str] = set()
    if environment:
        interacting = components & {"reflection", "diffraction"}
        if not interacting:
            raise ValueError(
                "environment=True asks for the environment's own returns, which are reflections or diffraction; "
                "set reflections>0 or diffraction=True"
            )
        direct |= interacting
    if coupling:
        direct |= {"los"} | (components & {"transmission"})
    return frozenset(direct)


def _subcarrier_offsets(spec: object) -> tuple[float, ...]:
    """The OFDM subcarriers as offsets from the reference frequency, ``n * df``, for the spec synthesis runs."""

    from .synthesis.assembly import OfdmSpec

    if not isinstance(spec, OfdmSpec):
        raise NotImplementedError(
            f"wideband=True evaluates one transport column per frequency and only OFDM consumes one (a column "
            f"per subcarrier); a {type(spec).__name__} run uses the reference-frequency coefficient and the "
            "exact delay phase, see R-ADR-015"
        )
    if spec.time_domain:
        raise NotImplementedError(
            "wideband=True needs the OFDM CFR family, which reads one weight per subcarrier; a modulated or "
            "time-output OFDM stream is synthesized by the time-domain family, one tap per row, see R-ADR-035"
        )
    return tuple(float(n) * spec.subcarrier_spacing_hz for n in range(spec.num_subcarriers))


def _require_discovery_only(unfrozen: list[str], *, ad_mode: str, still_world: bool, cadence: int | None) -> None:
    """Refuse a run that would replay rows Channel cannot freeze, by name."""

    if ad_mode != "none":
        raise NotImplementedError(
            f"components {unfrozen} exist only at their discovery instant and carry no derivative: Channel cannot "
            f"freeze them for the replay a grad={ad_mode!r} run differentiates through. Use grad='none'"
        )
    if not still_world and cadence != 1:
        raise NotImplementedError(
            f"components {unfrozen} cannot be frozen, so every evaluated instant must rediscover; "
            f"a discovery cadence of {cadence!r} observations replays frozen rows between discoveries. "
            "Drop Motion.rediscover_every_frames"
        )


def _frame_traces(
    session: Session, world: WorldHalf, *, path_set_complete: bool, exhaustive: bool
) -> Iterator[FrameTrace]:
    for start in session.times:
        events = PathEvents()
        yield FrameTrace(
            time_s=float(start),
            sample_times_s=start + session.offsets,
            observations=world.observations(start, session.offsets, events),
            path_set_complete=path_set_complete,
            motion_sampling_exhaustive=exhaustive,
            discovery=world.report,
            events=events,
        )


def _adaptive_traces(session: Session, world: WorldHalf) -> Iterator[FrameTrace]:
    from .synthesis.assembly import FmcwSpec

    spec = session.specs.working
    if not isinstance(spec, FmcwSpec):
        raise NotImplementedError("adaptive motion currently requires FMCW")
    for start in session.times:
        actual_times = start + session.offsets
        world.begin_report()
        frame = trace_adaptive_frame(actual_times, world.evaluate, spec, session.motion, session.reference_frequency_hz)
        report = world.report()
        events = PathEvents()
        if world.recorder is not None:
            probes = [(index, frame.probes[index]) for index in sorted(frame.probes)]
            world.recorder.record(
                events,
                [
                    Observed(
                        time_s=float(actual_times[index]),
                        observation=index,
                        pool=frame.table,
                        offset=int(frame.table.starts[index]),
                        composer=_round_trip(record.epoch_frame.frozen).composer,
                        discovered=record.epoch_frame.rediscovered or record.epoch_frame.topology_complete,
                    )
                    for index, record in probes
                ],
                _table_pool,
            )
            world.recorder.close_frame(events)
        last = frame.last
        if not isinstance(last, EvaluatedObservation):
            raise TypeError("the adaptive controller returns the session's evaluated observations")
        closing = Observation(
            time_s=float(actual_times[-1]), epoch_frame=last.epoch_frame, legs_of=last.legs, composed=last.composed
        )
        yield FrameTrace(
            time_s=float(start),
            sample_times_s=actual_times,
            observations=iter((closing,)),
            adaptive=frame.table,
            stats=frame.stats,
            opened=frame.first.epoch_frame,
            path_set_complete=bool(frame.stats["exhaustive"] or frame.stats["topology_proved_complete"]),
            motion_sampling_exhaustive=bool(frame.stats["exhaustive"]),
            discovery=partial(_constant_report, report),
            events=events,
        )


# ---------------------------------------------------------------------------
# The instrument half
# ---------------------------------------------------------------------------


def _uploaded(session: Session, table: np.ndarray, device: torch.device) -> torch.Tensor:
    """``table`` as an int64 device tensor, uploaded once per distinct content."""

    host = np.ascontiguousarray(table, dtype=np.int64)
    key = hashlib.sha256(str(device).encode() + host.tobytes()).digest()
    cached = session.routing.get(key)
    if cached is None:
        cached = torch.from_numpy(host).to(device)
        session.routing.put(key, cached)
    return cached


def _batches_slots(session: Session) -> bool:
    """Whether a sampled frame is synthesized one slot batch at a time.

    Every refreshed FMCW or pulsed frame is, unless something acts on the rows
    between replay and synthesis (oscillator phase noise) or adds rows of its
    own (a declared coupling); those keep one launch per observation.
    """

    from .synthesis.assembly import FmcwSpec, OfdmSpec, PulsedSpec

    radar = session.radar
    working = session.specs.working
    gridded = isinstance(working, (FmcwSpec, PulsedSpec)) or (isinstance(working, OfdmSpec) and working.time_domain)
    return session.sampled and gridded and not session.path_phase_noise and radar.leakage is None


def _echo_slot_batches(
    session: Session, working: WaveformSpec, observations: Iterator[Observation] | tuple[Observation, ...]
) -> tuple[torch.Tensor, EpochFrame, Observation]:
    """Synthesize a sampled frame from its slot batches, active pairs only.

    An FMCW observation sits in one TDM slot and is heard by one transmitter's
    ``num_rx`` pairs; a pulsed observation is heard by every pair. The frame
    cube reads nothing else from it. Each slot batch is therefore synthesized
    as ONE launch over just those pair segments, each carrying its own
    schedule entry (the synthesis owner's :func:`refreshed_schedule`), and the
    frame cube gathers the results. Every segment sums the same rows in the
    same order as a per-observation launch would, so the cube is the same to
    the bit. Returns the ``[slow, pair, fast]`` cube, the opening epoch frame and
    the closing observation.
    """

    from .synthesis import SynthesisPathBatch
    from .synthesis.assembly import FmcwSpec, OfdmSpec, PulsedSpec, fast_count, refreshed_schedule, require_compatible
    from .synthesis.fmcw import channel_phasor_to_beat_weight, synthesize_fmcw_rows
    from .synthesis.ofdm import symbol_tensor, synthesize_ofdm_time_rows
    from .synthesis.pulsed import synthesize_echo_rows

    single = session.specs.single
    assert single is not None
    fmcw = isinstance(working, FmcwSpec)
    array = session.array
    num_tx, num_rx = array.num_tx, array.num_rx
    pairs = num_tx * num_rx
    # Under TDM a slot fires one transmitter, heard on its num_rx pairs; a
    # pulsed train, DDM and BPM illuminate every pair at once.
    tdm = isinstance(working, FmcwSpec) and not working.simultaneous
    heard = num_rx if tdm else pairs
    slots_per_slow = num_tx if tdm else 1
    samples = session.samples_per_slot
    width = fast_count(single)
    values: list[torch.Tensor] = []
    targets: list[np.ndarray] = []
    opened: EpochFrame | None = None
    closing: Observation | None = None
    pending: list[tuple[int, Observation]] = []

    def flush() -> None:
        batch = pending[0][1].batch
        assert batch is not None
        composed = batch.rows
        # The provenance of every slot is the first instant's; the waveform
        # specs satisfy the protocol at run time, though their fields are
        # frozen, which a protocol of plain attributes cannot express.
        checked: Any = single
        first = pending[0][1].composed
        require_compatible(SynthesisPathBatch.from_radar_paths(first, slow_time_mode=session.mode), checked)
        rows_per_slot = composed.rows_per_slot
        index = np.asarray([entry[0] for entry in pending], dtype=np.int64)
        slot_in_batch = np.asarray([entry[1].slot for entry in pending], dtype=np.int64)
        sample = index % samples
        frame_slot = index // samples
        if tdm:
            # Sink-major ranks: the slot's transmitter ``t`` is heard on pairs
            # ``t + num_tx * r``, one per receiver ``r``.
            pair = ((frame_slot % num_tx)[:, None] + num_tx * np.arange(num_rx)).ravel()
        else:
            pair = np.tile(np.arange(pairs), len(index))
        owner = np.repeat(slot_in_batch, heard)
        begin = owner * rows_per_slot + batch.offsets[pair]
        counts = batch.offsets[pair + 1] - batch.offsets[pair]
        prefix = np.cumsum(counts) - counts
        rows = np.repeat(begin - prefix, counts) + np.arange(int(counts.sum()))
        segment_offsets = np.concatenate(([0], np.cumsum(counts)))
        tables = [rows, segment_offsets, pair % num_tx]
        schedule = refreshed_schedule(
            working, np.repeat(frame_slot, heard), np.repeat(sample, heard) if session.adc_sampled else None
        )
        scheduled = session.adc_sampled or not isinstance(working, FmcwSpec) or not working.uniform
        if scheduled:
            # Carried bit for bit through the int64 upload.
            tables.append(np.ascontiguousarray(schedule).view(np.int64).ravel())
        routing = _uploaded(session, np.concatenate(tables), composed.total_delay_s.device)
        parts = torch.split(routing, [len(table) for table in tables])
        row_index, offsets, transmitter = parts[0], parts[1], parts[2]
        transfer = composed.complex_transfer_ref.index_select(0, row_index)
        if fmcw:
            transfer = channel_phasor_to_beat_weight(transfer)
        if composed.row_valid is not None:
            valid = composed.row_valid.index_select(0, row_index)
            transfer = torch.where(valid, transfer, torch.zeros_like(transfer))
        delay = composed.total_delay_s.index_select(0, row_index)
        residual = (
            None
            if composed.total_delay_residual_s is None
            else composed.total_delay_residual_s.index_select(0, row_index)
        )
        if fmcw:
            assert isinstance(single, FmcwSpec)
            out = synthesize_fmcw_rows(
                delay,
                None,
                transfer,
                offsets,
                single,
                segment_tx_index=transmitter.to(torch.int32),
                segment_schedule=parts[3].view(torch.float64).reshape(-1, 3) if scheduled else None,
                total_delay_residual_s=residual,
            )
        elif isinstance(working, OfdmSpec):
            assert isinstance(single, OfdmSpec)
            out = synthesize_ofdm_time_rows(
                delay,
                None,
                transfer,
                offsets,
                single,
                symbols=symbol_tensor(working, composed.total_delay_s.device),
                segment_schedule=parts[3].view(torch.float64).reshape(-1, 2),
                total_delay_residual_s=residual,
            )
        else:
            assert isinstance(single, PulsedSpec)
            out = synthesize_echo_rows(
                delay,
                None,
                transfer,
                offsets,
                single,
                segment_schedule=parts[3].view(torch.float64).reshape(-1, 2),
                total_delay_residual_s=residual,
            )
        values.append(out.reshape(len(pair), width))
        slow = np.repeat(frame_slot // slots_per_slow, heard)
        targets.append(((slow * pairs + pair) * samples + np.repeat(sample, heard)) * width)
        pending.clear()

    for position, observation in enumerate(observations):
        if opened is None:
            opened = observation.epoch_frame
        closing = observation
        if pending and observation.batch is not pending[0][1].batch:
            flush()
        pending.append((position, observation))
    if opened is None or closing is None:
        raise ValueError("a frame trace must carry at least one observation")
    flush()
    destination = np.concatenate(targets)
    order = np.argsort(destination, kind="stable")
    if not np.array_equal(destination[order], np.arange(len(destination)) * width):
        raise ValueError("a sampled frame's observations must cover every slot, pair and sample exactly once")
    gather = _uploaded(session, order, values[0].device)
    cube = torch.cat(values).index_select(0, gather)
    if isinstance(working, FmcwSpec):
        slow_count = working.num_chirps
    elif isinstance(working, OfdmSpec):
        slow_count = working.num_symbols
    else:
        slow_count = working.num_pulses
    return cube.reshape(slow_count, pairs, -1), opened, closing


def _finish_frame(
    session: Session, synthesis: SynthesisResult, frame_time_s: float
) -> tuple[torch.Tensor, SynthesisResult]:
    """Apply the receive chain, then land the frame in its declared domain.

    ``frame_time_s`` keys the frame's receiver noise realisation, so two frames
    never share one.
    """

    from .synthesis.assembly import FmcwSpec, OfdmSpec, SynthesisResult, assemble_frame_cube

    radar = session.radar
    declared, working = session.specs.declared, session.specs.working
    # DDM and BPM record every transmitter on each receiver: one transmit channel.
    channels = working.transmit_channels if isinstance(working, FmcwSpec) else session.array.num_tx
    frame_cube = assemble_frame_cube(synthesis.cube, num_tx=channels, num_rx=session.array.num_rx)
    if radar.frontend is not None:
        frame_cube = radar.frontend.apply(
            frame_cube, phase_in_signal=session.path_phase_noise, frame_time_s=frame_time_s
        ).signal
    if isinstance(working, FmcwSpec) and working.real_adc:
        # An in-phase-only receiver digitises Re of the complex baseband, after
        # every receive stage; its noise per real sample is half the complex
        # sample's, which is the same k T_sys f_s / 2 a real ADC at f_s sees.
        frame_cube = torch.complex(frame_cube.real, torch.zeros_like(frame_cube.real))
    if isinstance(declared, OfdmSpec) and declared.output_domain != synthesis.output_domain:
        from .processing.range_doppler import ofdm_cfr_estimate
        from .synthesis.ofdm import symbol_tensor

        assert isinstance(working, OfdmSpec)
        table = symbol_tensor(working, frame_cube.device)
        frame_cube = ofdm_cfr_estimate(frame_cube, torch.complex(table[..., 0], table[..., 1]))
        synthesis = replace(
            SynthesisResult.from_ofdm(
                frame_cube.permute(2, 1, 0, 3).reshape(working.num_symbols, -1, working.num_subcarriers), declared
            ),
            observation_schedule=synthesis.observation_schedule,
        )
    if isinstance(declared, FmcwSpec) and declared.output_domain != synthesis.output_domain:
        from .processing.range_doppler import fmcw_range_fft

        assert isinstance(working, FmcwSpec)
        frame_cube = fmcw_range_fft(frame_cube)
        synthesis = replace(
            SynthesisResult.from_fmcw(
                frame_cube.permute(2, 1, 0, 3).reshape(working.num_chirps, -1, working.num_samples), declared
            ),
            observation_schedule=synthesis.observation_schedule,
        )
    return frame_cube, synthesis


def observation_schedule(session: Session, trace: FrameTrace) -> str:
    """When this frame's synthesis observed the world, as the synthesis owner names it.

    One of :data:`~witwin.radar.synthesis.assembly.OBSERVATION_SCHEDULES`. It
    is what ran, not what was requested: an OFDM symbol has no fast-time grid,
    so an ADC request on one observes it once per symbol.
    """

    from .synthesis.assembly import OBSERVATION_ADAPTIVE

    if trace.adaptive is not None:
        return OBSERVATION_ADAPTIVE
    return _schedule("adc", sampled=session.sampled, adc_sampled=session.adc_sampled)


def echo_frame(session: Session, trace: FrameTrace) -> SceneFrame:
    """Synthesize one traced frame. The whole instrument half lives here.

    Everything this does is a property of the radar and the waveform, never of
    the world: the per-observation synthesis, the common-oscillator phase the
    receive chain applies per path before coherent summation, the slot gather,
    the receive chain itself and the output-domain transform. That is what lets
    a caller re-run it against a different receiver without re-tracing.

    ``trace.observations`` is drained exactly once, in order.
    """

    from .synthesis.assembly import FmcwSpec, OfdmSpec, PulsedSpec, SynthesisResult

    radar = session.radar
    working = session.specs.working
    opened: EpochFrame | None = None
    closing: Observation | None = None
    published: RadarPathBatch | None = None
    synthesis: SynthesisResult | None = None
    if trace.adaptive is not None:
        assert isinstance(working, FmcwSpec)
        for observation in trace.observations:
            opened = trace.opened
            closing = observation
            published = observation.composed
        cube, batches = echo_adaptive_frame(
            trace.adaptive,
            working,
            int(session.motion.batch_observations),
            session.reference_frequency_hz,
            radar.frontend,
        )
        # Written by the half that did the batching, so a Paths that has never
        # been echoed does not claim a synthesis count it could not have.
        assert trace.stats is not None
        trace.stats["synthesis_batches"] = batches
        beat = replace(working, output_domain="beat")
        # The declared coupling is a static row per sensor pair, so it is the
        # same full-frame beat cube whatever the world's sampling; the other
        # sampled routes add it per observation through ``Radar._synthesize``.
        cube = cube + radar._leakage_cube(beat, session.array.num_tx * session.array.num_rx, cube)
        synthesis = SynthesisResult.from_fmcw(cube, beat)
    elif _batches_slots(session):
        cube, opened, closing = _echo_slot_batches(session, working, trace.observations)
        published = closing.composed
        if isinstance(working, OfdmSpec):
            synthesis = SynthesisResult.from_ofdm(cube, working)
        elif not isinstance(working, FmcwSpec):
            synthesis = SynthesisResult.from_pulsed(cube, working)
        elif session.adc_sampled:
            if working.output_domain == "spectrum":
                from .processing.range_doppler import fmcw_range_fft

                cube = fmcw_range_fft(cube)
            synthesis = SynthesisResult.from_fmcw(cube, working)
        else:
            assert isinstance(session.specs.single, FmcwSpec)
            synthesis = SynthesisResult.from_fmcw(cube, session.specs.single)
    else:
        slot_cubes: list[torch.Tensor] = []
        single = session.specs.single
        for observation in trace.observations:
            if opened is None:
                opened = observation.epoch_frame
            closing = observation
            composed = observation.composed
            if session.path_phase_noise:
                assert radar.frontend is not None
                composed = radar.frontend.apply_path_phase(composed, observation.time_s)
            published = composed
            observation_spec = working
            if session.sampled:
                assert single is not None
                observation_spec = single
                if isinstance(working, PulsedSpec) and working.frequency_step_hz != 0.0:
                    raise NotImplementedError(
                        "a stepped-frequency train on a moving scene is synthesized one slot batch at a time, and a "
                        "declared leakage keeps one launch per observation; the two are not combined"
                    )
                if isinstance(working, FmcwSpec) and not working.uniform:
                    # A sequence's slot carries its own slope, offset and window.
                    profile = working.slot_spec(len(slot_cubes) // session.samples_per_slot)
                    assert isinstance(single, FmcwSpec)
                    observation_spec = replace(
                        single,
                        slope_hz_per_s=profile.slope_hz_per_s,
                        start_offset_hz=profile.start_offset_hz,
                        t_start_s=profile.t_start_s,
                    )
                if session.adc_sampled:
                    sample = len(slot_cubes) % session.samples_per_slot
                    if isinstance(working, FmcwSpec):
                        assert isinstance(observation_spec, FmcwSpec)
                        observation_spec = replace(
                            observation_spec, t_start_s=observation_spec.t_start_s + sample * working.sample_period_s
                        )
                    else:
                        assert isinstance(working, PulsedSpec) and isinstance(single, PulsedSpec)
                        observation_spec = replace(
                            single, range_gate_start_s=working.range_gate_start_s + sample * working.sample_period_s
                        )
            if session.sampled:
                synthesis = radar._synthesize(composed, slow_time_mode=session.mode, spec=observation_spec)
            else:
                synthesis = session.held.synthesize(radar, composed, session.mode, observation_spec)
            slot_cubes.append(synthesis.cube)
        assert synthesis is not None
        if session.sampled:
            stacked = torch.cat(slot_cubes, dim=0)
            pairs = torch.arange(session.array.num_tx * session.array.num_rx, device=stacked.device)
            if session.adc_sampled:
                stacked = stacked.reshape(-1, session.samples_per_slot, len(pairs)).transpose(1, 2)
                cube = stacked[session.pair_samples, pairs]
                if isinstance(working, FmcwSpec):
                    if working.output_domain == "spectrum":
                        from .processing.range_doppler import fmcw_range_fft

                        cube = fmcw_range_fft(cube)
                    synthesis = SynthesisResult.from_fmcw(cube, working)
                else:
                    assert isinstance(working, PulsedSpec)
                    synthesis = SynthesisResult.from_pulsed(cube, working)
            else:
                synthesis = replace(synthesis, cube=stacked[session.pair_samples, pairs])
    if opened is None or closing is None or published is None or synthesis is None:
        raise ValueError("a frame trace must carry at least one observation")
    if isinstance(working, FmcwSpec) and working.simultaneous:
        from .synthesis.fmcw import multiplex_transmitters

        synthesis = replace(synthesis, cube=multiplex_transmitters(synthesis.cube, working))
    synthesis = replace(synthesis, observation_schedule=observation_schedule(session, trace))
    discovery = trace.discovery()
    frame_cube, synthesis = _finish_frame(session, synthesis, trace.time_s)
    loop = session.loop
    return SceneFrame(
        cube=frame_cube,
        synthesis=synthesis,
        time_s=trace.time_s,
        sample_times_s=trace.sample_times_s,
        epoch=int(opened.epoch),
        reason=opened.reason,
        path_set_complete=trace.path_set_complete and discovery.complete,
        motion_sampling_exhaustive=trace.motion_sampling_exhaustive,
        discovery=discovery,
        compile_count=int(loop.compile_count),
        discovery_count=int(loop.discovery_count),
        epoch_frame=closing.epoch_frame,
        legs=closing.legs,
        composed=published,
        diagnostics=trace.stats,
        motion_sampling=session.kind,
        output_spec=session.specs.declared,
        array=session.array,
        events=trace.events,
    )


def _scene_frames(radar: Radar, scene: object, **arguments: Any) -> Iterator[SceneFrame]:
    """Trace one frame, echo it, drop its rows, repeat.

    The fused route. It is a generator so that argument validation still
    happens when iteration starts rather than when the call returns, which is
    what :func:`stream_scene` promises.
    """

    session, traces = open_session(radar, scene, **arguments)
    for trace in traces:
        yield echo_frame(session, trace)


# ---------------------------------------------------------------------------
# The world half's record, and the verbs
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, eq=False)
class Paths:
    """The round trips one :meth:`~witwin.radar.Radar.trace` composed.

    The world half of the pipeline, stopped before any instrument stage has
    run: the rows here carry propagation, scattering and the array's pattern
    gain, and they do not carry the waveform, the receive chain or the output
    domain. :meth:`~witwin.radar.Radar.echo` adds those, which is what lets a
    caller sweep receivers, seeds or the FMCW output domain without tracing the
    world again.

    RETENTION, stated because this record is a real tensor lifetime. It holds
    every evaluated observation's composed rows, so holding it holds that much
    device memory - roughly twenty bytes per live row, summed over every
    evaluated observation of every frame. An ADC-refreshed frame evaluates
    ``chirps * transmitters * samples`` instants, so a modest scene costs tens
    of megabytes per frame and a rich one costs gigabytes.
    :meth:`~witwin.radar.Radar.simulate` never pays that: it drops each
    replay group as it synthesizes. Trace a frame at a time if the sequence is
    long.

    ``path_set_complete`` and ``motion_sampling_exhaustive`` are TWO
    statements, and both belong here rather than to the echo: the first says no
    path birth can have been missed, the second says the world was evaluated
    at every instant the waveform samples. An adaptive trace of a certifiable world
    reports the first without the second, which is exactly the trade it exists
    to make.
    """

    #: ``"static"``, ``"chirp"``, ``"adc"``, ``"adaptive"``, ``"symbol"`` or
    #: ``"pulse"``: the observation schedule that resampled the world inside a
    #: frame, named by what ran rather than by what was requested.
    kind: str
    #: The reference frequency these rows were composed at, Hz.
    carrier: float
    #: The waveform that scheduled the observations. ``echo`` refuses a radar
    #: whose waveform differs, because the schedule is that waveform's.
    waveform: object
    #: Frame instants, s.
    times: tuple[float, ...]
    #: Every observation instant of every frame, ``float64[observations]``
    #: per frame.
    sample_times: tuple[np.ndarray, ...]
    #: The propagation request these rows answer. ``path_set_complete`` is a
    #: statement relative to it, not an absolute one.
    los: bool
    reflections: int
    transmission: bool
    diffraction: bool
    #: Whether site-less TX->RX rows were composed: the environment's own
    #: returns, and the elements' straight-segment coupling.
    environment: bool
    coupling: bool
    #: The differentiation mode the replay ran under.
    grad: str
    #: The sampling record as requested; ``kind`` is what it resolved to.
    motion: Motion
    epochs: tuple[int, ...]
    rediscovery_reasons: tuple[str | None, ...]
    compile_count: int
    discovery_count: int
    path_set_complete: bool
    motion_sampling_exhaustive: bool
    adaptive_diagnostics: tuple[dict[str, Any], ...]
    #: Channel's completeness and dropped-transmission count; see
    #: :class:`~witwin.radar.Result`. ``path_set_complete`` is ANDed with it.
    discovery_complete: bool
    transmission_dropped_rows: int
    #: Each frame's path births and deaths (:class:`~witwin.radar.events.PathEvents`).
    events: tuple[PathEvents, ...]
    #: The closing observation's typed state, as the result publishes it.
    last_snapshot: object
    last_compiled_scene: object
    last_propagation: RadarPropagationLegs
    last_radar_paths: RadarPathBatch
    _session: Session
    _frames: tuple[FrameTrace, ...]

    @property
    def frame_count(self) -> int:
        """Frames these rows cover."""

        return len(self._frames)

    @property
    def birth_lag_bound_s(self) -> tuple[float, ...]:
        """Per frame, how late a path birth can have been seen, s."""

        return tuple(frame.birth_lag_bound_s for frame in self.events)

    @property
    def observation_count(self) -> int:
        """Instants the world was evaluated at, summed over every frame.

        For an adaptive trace this counts the schedule, not the probes; the
        probes are in ``adaptive_diagnostics``, and the gap between the two is
        the work that route saved.
        """

        total = 0
        for frame in self._frames:
            total += frame.adaptive.observation_count if frame.adaptive is not None else len(_retained(frame))
        return total

    @property
    def row_count(self) -> int:
        """Composed rows retained, summed over every evaluated observation."""

        total = 0
        for frame in self._frames:
            if frame.adaptive is not None:
                total += frame.adaptive.row_count
            else:
                total += sum(int(observation.composed.path_count) for observation in _retained(frame))
        return total

    def frame(self, index: int) -> Paths:
        """One frame's rows, as a Paths of its own.

        The session travels with it, so the slice can be echoed. The compile
        and discovery counts are carried unchanged rather than recomputed,
        because slicing recompiled nothing.
        """

        frame_trace = self._frames[index]
        closing = _retained(frame_trace)[-1]
        return replace(
            self,
            times=(self.times[index],),
            sample_times=(self.sample_times[index],),
            epochs=(self.epochs[index],),
            rediscovery_reasons=(self.rediscovery_reasons[index],),
            path_set_complete=bool(frame_trace.path_set_complete),
            motion_sampling_exhaustive=bool(frame_trace.motion_sampling_exhaustive),
            adaptive_diagnostics=() if frame_trace.stats is None else (frame_trace.stats,),
            discovery_complete=frame_trace.discovery().complete,
            transmission_dropped_rows=frame_trace.discovery().transmission_dropped_rows,
            events=(frame_trace.events,),
            last_snapshot=closing.epoch_frame.snapshot,
            last_compiled_scene=closing.epoch_frame.compiled,
            last_propagation=closing.legs,
            last_radar_paths=closing.composed,
            _frames=(frame_trace,),
        )

    def rows(self, frame: int = 0, observation: int = -1) -> RadarPathBatch:
        """One evaluated observation's composed rows, as a typed batch.

        An adaptive frame keeps its rows in one interpolation table rather than
        one batch per instant, so it publishes only the observation that closed
        it; asking for another is refused rather than answered with that one.
        """

        frame_trace = self._frames[frame]
        if frame_trace.adaptive is not None and observation not in (0, -1):
            raise IndexError(
                "an adaptive frame stores its rows as one interpolation table, not one batch per instant, "
                "so only the observation that closed it is published as a batch"
            )
        return _retained(frame_trace)[observation].composed

    def component_index(
        self, frame: int = 0, observation: int = -1, declaration: ComponentDeclaration | None = None
    ) -> RadarComponentIndex:
        """Which component class owns each row of one observation's composed batch.

        ``declaration`` defaults to :func:`default_component_declaration` of
        that observation's epoch: every declared site a target and every
        interacted material clutter, so a site row bouncing off a wall is
        environment clutter, a site-less row through a wall is environment
        clutter and the site-less line of sight is direct leakage.
        """

        observation_record = _retained(self._frames[frame])[observation]
        return _component_index(observation_record.epoch_frame.frozen, declaration)

    def select(self, component: str, declaration: ComponentDeclaration | None = None) -> Paths:
        """These rows with every row outside one component class made inert.

        ``radar.echo(paths.select("environment_clutter"))`` synthesizes the
        clutter alone, through the same kernels and the same pair partition,
        and the per-class echoes sum to the full echo because synthesis is
        linear in the rows. A row outside the class is marked invalid, which
        every synthesis kernel treats as an exact zero.

        An adaptive frame interpolates its rows from a table, not per batch, so
        it is refused rather than half-masked.
        """

        frames = []
        masked: dict[int, SlotBatch] = {}
        for frame_trace in self._frames:
            if frame_trace.adaptive is not None:
                raise NotImplementedError(
                    "an adaptive frame interpolates its rows from one table; select a component on a "
                    "trace that keeps one batch per instant (Motion.adc(), Motion.chirp() or a static scene)"
                )
            observations = []
            for record in _retained(frame_trace):
                index = _component_index(record.epoch_frame.frozen, declaration)
                keep = index.mask(component)
                batch = record.batch
                if batch is not None and id(batch) not in masked:
                    rows = batch.rows
                    slot_keep = keep.repeat(batch.slots)
                    valid = slot_keep if rows.row_valid is None else rows.row_valid & slot_keep
                    masked[id(batch)] = replace(batch, rows=replace(rows, row_valid=valid))
                observations.append(
                    replace(
                        record,
                        composed=_masked(record.composed, keep),
                        batch=None if batch is None else masked[id(batch)],
                    )
                )
            frames.append(replace(frame_trace, observations=tuple(observations)))
        closing = _retained(frames[-1])[-1]
        return replace(self, _frames=tuple(frames), last_radar_paths=closing.composed)


def default_component_declaration(epoch: FrozenEpoch) -> ComponentDeclaration:
    """Every declared site a target, every interacted compiled material clutter.

    Read from the epoch's frozen legs on the host, at most once per epoch and
    never in a frame loop.
    """

    from .paths import ComponentDeclaration

    materials: set[int] = set()
    for handle in epoch.handles:
        values = handle.material_sequence.reshape(-1).tolist()
        materials.update(int(value) for value in values if int(value) >= 0)
    return ComponentDeclaration(
        target_site_ids=frozenset(_round_trip(epoch).binding.site_ids), clutter_material_slots=frozenset(materials)
    )


def _component_index(epoch: FrozenEpoch, declaration: ComponentDeclaration | None) -> RadarComponentIndex:
    from .paths import RadarComponentIndex

    handles = epoch.handles
    return RadarComponentIndex.from_two_way(
        _round_trip(epoch).composer,
        handles[0],
        handles[1],
        default_component_declaration(epoch) if declaration is None else declaration,
        direct=handles[2] if len(handles) > 2 else None,
    )


def _masked(paths: RadarPathBatch, keep: torch.Tensor) -> RadarPathBatch:
    valid = keep if paths.row_valid is None else paths.row_valid & keep
    return replace(paths, row_valid=valid)


def _retained(frame: FrameTrace) -> tuple[Observation, ...]:
    observations = frame.observations
    if not isinstance(observations, tuple):
        raise TypeError("a traced frame retains its observations as a tuple")
    return observations


def trace_scene(radar: Radar, scene: object, **arguments: Any) -> Paths:
    """Run the world half of a session and keep every composed row.

    The same generator :func:`simulate_scene` consumes, drained into a record
    instead of synthesized. Read :class:`Paths` on what that costs.
    """

    session, traces = open_session(radar, scene, **arguments)
    frames: list[FrameTrace] = []
    epochs: list[int] = []
    reasons: list[str | None] = []
    for trace in traces:
        observations = tuple(trace.observations)
        opened = trace.opened if trace.opened is not None else observations[0].epoch_frame
        report = trace.discovery()
        frames.append(
            replace(
                trace,
                observations=observations,
                opened=opened,
                discovery=partial(_constant_report, report),
                path_set_complete=trace.path_set_complete and report.complete,
            )
        )
        epochs.append(int(opened.epoch))
        reasons.append(opened.reason)
    closing = _retained(frames[-1])[-1]
    discovery = reduce(DiscoveryReport.merged, (frame.discovery() for frame in frames))
    return Paths(
        kind=session.kind,
        carrier=float(session.reference_frequency_hz),
        waveform=session.radar.waveform,
        times=tuple(session.times),
        sample_times=tuple(frame.sample_times_s for frame in frames),
        los="los" in session.components,
        reflections=int(session.max_depth) if "reflection" in session.components else 0,
        transmission="transmission" in session.components,
        diffraction="diffraction" in session.components,
        environment=session.environment,
        coupling=session.coupling,
        grad=session.grad,
        motion=session.motion,
        epochs=tuple(epochs),
        rediscovery_reasons=tuple(reasons),
        compile_count=int(session.loop.compile_count),
        discovery_count=int(session.loop.discovery_count),
        path_set_complete=all(frame.path_set_complete for frame in frames),
        motion_sampling_exhaustive=all(frame.motion_sampling_exhaustive for frame in frames),
        adaptive_diagnostics=tuple(frame.stats for frame in frames if frame.stats is not None),
        discovery_complete=discovery.complete,
        transmission_dropped_rows=discovery.transmission_dropped_rows,
        events=tuple(frame.events for frame in frames),
        last_snapshot=closing.epoch_frame.snapshot,
        last_compiled_scene=closing.epoch_frame.compiled,
        last_propagation=closing.legs,
        last_radar_paths=closing.composed,
        _session=session,
        _frames=tuple(frames),
    )


def echo_paths(radar: Radar, paths: Paths) -> Result:
    """Run the instrument half over traced rows and publish the frame cubes.

    ``radar`` supplies the receive chain and the output domain; the rows supply
    everything about the world. Every difference that would have changed the
    observation schedule is refused by name rather than replayed against a
    schedule that no longer describes it.
    """

    if not isinstance(paths, Paths):
        raise TypeError(f"echo takes the Paths that trace returned, got {type(paths).__name__}")
    _require_same_instrument(radar, paths, paths._session)
    session = paths._session.echoing(radar)
    return assemble([echo_frame(session, trace) for trace in paths._frames])


def _require_same_instrument(radar: Radar, paths: Paths, session: Session) -> None:
    """Refuse a radar these rows do not describe, by name.

    What may differ is the receive chain and the FMCW output domain. What may
    not is anything the observation schedule was derived from, because the rows
    were evaluated at the instants that schedule chose.
    """

    from .synthesis.assembly import FmcwSpec

    if float(radar.carrier) != float(paths.carrier):
        raise ValueError(
            f"these paths were composed at {paths.carrier} Hz and this radar is at {radar.carrier} Hz; "
            "the carrier is the reference frequency of the transport, not a synthesis choice"
        )
    array = session.array
    if radar.num_tx != array.num_tx or radar.num_rx != array.num_rx:
        raise ValueError(
            f"these paths carry {array.num_tx} x {array.num_rx} sensor pairs and this radar has "
            f"{radar.num_tx} x {radar.num_rx}; the pair partition is frozen into the composed rows"
        )
    if paths.kind != "static" and radar.waveform != paths.waveform:
        raise ValueError(
            f"these paths were scheduled by {type(paths.waveform).__name__} observations and this radar "
            f"declares a different waveform; a {paths.kind!r} trace evaluated the world at that waveform's "
            "instants, so another one has no rows to read. Re-trace, or change only the receive chain."
        )
    phase_noise = radar.frontend is not None and radar.frontend.has_phase_noise
    if phase_noise and paths.kind not in ("adc", "adaptive"):
        raise ValueError(
            f"a receiver with oscillator phase noise places its delayed phase difference at absolute ADC "
            f"time, and these paths were traced with {paths.kind!r} sampling, which has no ADC instants. "
            "Re-trace with Motion.adc() or Motion.adaptive()."
        )
    if phase_noise and not isinstance(session.specs.declared, FmcwSpec):
        raise NotImplementedError("scene-driven common-oscillator phase noise currently requires FMCW")


def simulate_scene(radar: Radar, scene: object, **arguments: Any) -> Result:
    """Run one scene session to completion and stack every frame.

    The whole sequence stays in device memory, so peak allocation scales with the
    frame count: one cube per frame plus the closing frame's state. Use
    :func:`stream_scene` for a sequence long enough that this is the binding
    constraint.
    """

    session, traces = open_session(radar, scene, **arguments)
    frames: list[SceneFrame] = []
    stacked: torch.Tensor | None = None
    for trace in traces:
        frame = echo_frame(session, trace)
        index = len(frames)
        # A cube with no derivative is written straight into the stacked
        # result, so a run holds one cube per frame rather than a list of
        # frame cubes and then their stack. A differentiable cube is stacked
        # by ``torch.stack`` instead, which owns its backward.
        if index == 0 and not _tensor_carries_derivative(frame.cube):
            stacked = frame.cube.new_empty((len(session.times), *frame.cube.shape))
        if stacked is not None:
            stacked[index].copy_(frame.cube)
            frame = replace(frame, cube=stacked[index])
        if frames:
            frames[-1] = frames[-1].released()
        frames.append(frame)
    return assemble(frames, stacked=stacked)


def stream_scene(radar: Radar, scene: object, **arguments: Any) -> Iterator[Result]:
    """Yield each frame of a scene session as its own one-frame result.

    Same physics, same session state and the same per-frame cubes as
    :func:`simulate_scene`; the difference is that nothing here retains a frame
    the caller has released, so a sequence of any length costs one frame of
    device memory plus whatever the caller keeps.

    A yielded result ALIASES that frame's device tensors through its four
    ``last_*`` members, exactly as the stacked result does. Holding every
    yielded result therefore costs MORE than calling :func:`simulate_scene`,
    not less: the point of this entry is that the caller consumes and drops.
    """

    for frame in _scene_frames(radar, scene, **arguments):
        yield assemble([frame])


__all__: list[str] = []
