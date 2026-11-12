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
from functools import partial
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import numpy as np
import torch
from witwin.core.dynamics import DynamicScene

from .adaptive import AdaptiveTable, echo_adaptive_frame, trace_adaptive_frame
from .binding import SITE_SOURCE_EXPLICIT, RadarWorldBinding, ScatterSitePolicy, SensorEndpointIds, bind_radar_world
from .motion import Motion
from .propagation import EpochFrame, FrozenEpoch, RadarEndpointSpec, RadarPropagationLegs, SceneEpochLoop
from .results import Result, SceneFrame, assemble

if TYPE_CHECKING:
    from .paths import RadarPathBatch, TwoWayComposer
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
    #: The two legs' packed row identities. Two epochs with equal identities
    #: join to the same composed rows, so a rediscovery that found the same
    #: topology reuses the join and pattern tables instead of rebuilding them.
    identity: tuple[bytes, bytes]


def _round_trip(epoch: FrozenEpoch) -> FrozenRoundTrip:
    payload = epoch.payload
    if not isinstance(payload, FrozenRoundTrip):
        raise TypeError(f"a scene session epoch carries a FrozenRoundTrip, got {type(payload).__name__}")
    return payload


@dataclass(frozen=True, slots=True, eq=False)
class SlotBatch:
    """The slot-major rows one grouped replay composed, shared by its instants.

    ``composed`` holds ``slots`` copies of one join's rows, slot major: row
    ``g * rows + r`` is join row ``r`` at slot ``g`` and pair ``g * pairs + p``
    is slot ``g``'s pair ``p``. ``offsets`` is ONE slot's pair partition on the
    host, which is what lets the instrument half select sensor-pair segments
    without reading the device.
    """

    composed: RadarPathBatch
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
    adaptive: AdaptiveTable | None = None
    stats: dict[str, Any] | None = None
    opened: EpochFrame | None = None


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

        from .synthesis.assembly import FmcwSpec, waveform_sampling

        declared = radar.system_config.waveform_spec()
        working = declared
        if isinstance(declared, FmcwSpec) and radar.frontend is not None:
            working = replace(declared, output_domain="beat")
        if not self.sampled:
            return InstrumentSpecs(declared=declared, working=working, single=None)
        _, _, single = waveform_sampling(
            working, num_tx=self.array.num_tx, num_rx=self.array.num_rx, device=radar.device
        )
        if self.adc_sampled:
            single = replace(single, num_samples=1, output_domain="beat")
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


def _constant_legs(legs: RadarPropagationLegs) -> RadarPropagationLegs:
    return legs


def _slot_legs(replay: RadarPropagationLegs, index: int) -> RadarPropagationLegs:
    """One slot of a slot-major replay, narrowed only when a caller reads it."""

    return RadarPropagationLegs(inbound=replay.inbound.slot(index), outbound=replay.outbound.slot(index))


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
        still_world: bool = False,
        batch_observations: int = 1,
    ) -> None:
        from .channel import compile_scene

        self.radar = radar
        self.dynamic = dynamic
        self.sites = sites
        self.polarization = polarization
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
            world_motion=world_motion,
        )
        #: A world that nothing moves and nothing rediscovers replays the same
        #: endpoints against the same frozen topology every frame, and Channel's
        #: replay is deterministic, so its rows are evaluated once and every
        #: later frame publishes them again. Only a still world that carries no
        #: derivative holds them: a shared graph would sum every frame's
        #: cotangent before the replay's backward rather than after it.
        self.still_world = still_world
        self.batch_observations = batch_observations
        self._still: Observation | None = None
        self._slot_tables: _Bounded[tuple[Hashable, int], tuple[TwoWayComposer, RoundTripPatternStage]] = _Bounded(
            SLOT_TABLE_CACHE
        )

    def _bind_world(self, snapshot: object) -> RadarWorldBinding:
        return bind_radar_world(
            self.radar,
            snapshot,
            sites=self.sites,
            polarization=self.polarization,
            sensor_endpoints=self.sensor_endpoints,
        )

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
        adapter = (
            ChannelPropagationAdapter(
                compiled,
                reference_frequency_hz=self.reference_frequency_hz,
                components=self.propagation.components,
                max_depth=self.propagation.max_depth,
            )
            if previous is None
            else previous.adapter
        )
        inbound = adapter.freeze(binding.transmitters, binding.site_sinks)
        outbound = adapter.freeze(binding.site_sources, binding.receivers)
        arriving = read_leg_identity(inbound, "inbound")
        leaving = read_leg_identity(outbound, "outbound")
        identity = (arriving.packed, leaving.packed)
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
            )
            pattern = RoundTripPatternStage.freeze(
                self.radar,
                composer,
                site_ids=binding.site_ids,
                pattern=self.antenna_pattern,
                rx_pattern=self.radar.system_config.sensors.rx_pattern,
            )
        return FrozenEpoch(
            adapter=adapter,
            handles=(inbound, outbound),
            payload=FrozenRoundTrip(composer=composer, binding=binding, pattern=pattern, identity=identity),
        )

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
        if self.sites.source == SITE_SOURCE_EXPLICIT and self.sensor_endpoints is None:
            positions = self.sites.resolve(frame.snapshot, device=template.device)
            if positions.shape != template.site_positions_m.shape:
                raise ValueError("a site trajectory must preserve the declared site count")
            return replace(
                template,
                site_sources=replace(template.site_sources, positions_m=positions),
                site_sinks=replace(template.site_sinks, positions_m=positions),
            )
        return self._bind_world(frame.snapshot)

    def _identity(self, frame: EpochFrame) -> Hashable:
        """The leg identity the adaptive controller groups and compares by.

        Every other route groups nothing and compares nothing, so it is ``()``
        there.
        """

        if self.motion_sampling != "adaptive":
            return ()
        return _round_trip(frame.frozen).identity

    def evaluate(self, query_times: Sequence[float]) -> list[EvaluatedObservation]:
        """Advance the world to each instant, then batch-replay identical handles.

        One record per query instant, in order. Moving compiled geometry is
        replayed before the next refreeze retires its handles. Static geometry
        permits one slot batch per full leg identity.
        """

        results: dict[float, EvaluatedObservation] = {}
        groups: dict[Hashable, list[_PendingObservation]] = {}
        for time_s in query_times:
            frame = self.loop.frame(time_s, topology_complete=self._complete_los())
            pending = _PendingObservation(
                time_s=time_s, epoch_frame=frame, binding=self._binding_for(frame), identity=self._identity(frame)
            )
            if self.loop.structures_move:
                self._replay([pending], results)
            else:
                groups.setdefault(_round_trip(frame.frozen).identity, []).append(pending)
        for group in groups.values():
            self._replay(group, results)
        return [results[time_s] for time_s in query_times]

    def _replay(self, group: list[_PendingObservation], results: dict[float, EvaluatedObservation]) -> None:
        """Replay one group of topology-identical instants as one slot batch."""

        first = group[0].epoch_frame.frozen
        bindings = [entry.binding for entry in group]
        transmitters = _stacked([binding.transmitters for binding in bindings])
        receivers = _stacked([binding.receivers for binding in bindings])
        replay = RadarPropagationLegs(
            inbound=first.adapter.reevaluate_slots(
                first.handles[0],
                transmitters,
                _stacked([binding.site_sinks for binding in bindings]),
                slot_count=len(group),
                ad_mode=self.ad_mode,
            ),
            outbound=first.adapter.reevaluate_slots(
                first.handles[1],
                _stacked([binding.site_sources for binding in bindings]),
                receivers,
                slot_count=len(group),
                ad_mode=self.ad_mode,
            ),
        )
        if len(group) == 1:
            entry = group[0]
            round_trip = _round_trip(entry.epoch_frame.frozen)
            paths = round_trip.composer.compose(replay.inbound, replay.outbound, self.response)
            paths = round_trip.pattern.apply(
                paths,
                tx_pos=entry.binding.transmitters.positions_m,
                rx_pos=entry.binding.receivers.positions_m,
                tx_targets_m=_present(replay.inbound.departure_target_m, "departure_target_m").index_select(
                    0, paths.topology.inbound_row
                ),
                rx_targets_m=_present(replay.outbound.arrival_origin_m, "arrival_origin_m").index_select(
                    0, paths.topology.outbound_row
                ),
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
                batch=SlotBatch(composed=paths, offsets=round_trip.composer.pair_offsets_host, slots=1),
                slot=0,
            )
            return
        batched = self._compose_slots(group, replay, transmitters, receivers)
        shared = SlotBatch(
            composed=batched,
            offsets=_round_trip(group[0].epoch_frame.frozen).composer.pair_offsets_host,
            slots=len(group),
        )
        batched_valid = _present(batched.row_valid, "row_valid")
        valid_rows = batched_valid.reshape(len(group), -1).cpu().numpy() if self.motion_sampling == "adaptive" else None
        for slot, entry in enumerate(group):
            composer = _round_trip(entry.epoch_frame.frozen).composer
            selection = slice(slot * composer.path_count, (slot + 1) * composer.path_count)
            paths = replace(
                batched,
                sensor_pair_count=composer.sensor_pair_count,
                path_count=composer.path_count,
                sensor_pair_index=composer.sensor_pair_index,
                pair_offsets=composer.pair_offsets,
                topology=composer.topology,
                total_delay_s=batched.total_delay_s[selection],
                complex_transfer_ref=batched.complex_transfer_ref[selection],
                row_valid=batched_valid[selection],
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
                legs=partial(_slot_legs, replay, slot),
                composed=paths,
                identity=identity,
                batch=shared,
                slot=slot,
            )

    def _compose_slots(
        self,
        group: list[_PendingObservation],
        replay: RadarPropagationLegs,
        transmitters: RadarEndpointSpec,
        receivers: RadarEndpointSpec,
    ) -> RadarPathBatch:
        """Compose and pattern-weight a slot-major replay with the slot tables."""

        round_trip = _round_trip(group[0].epoch_frame.frozen)
        key = (round_trip.identity, len(group))
        tables = self._slot_tables.get(key)
        if tables is None:
            tables = (round_trip.composer.for_slots(len(group)), round_trip.pattern.for_slots(len(group)))
            self._slot_tables.put(key, tables)
        composer, pattern = tables
        batched = composer.compose(replay.inbound, replay.outbound, self.response)
        return pattern.apply(
            batched,
            tx_pos=transmitters.positions_m,
            rx_pos=receivers.positions_m,
            tx_targets_m=_present(replay.inbound.departure_target_m, "departure_target_m").index_select(
                0, batched.topology.inbound_row
            ),
            rx_targets_m=_present(replay.outbound.arrival_origin_m, "arrival_origin_m").index_select(
                0, batched.topology.outbound_row
            ),
        )

    def observations(self, start: float, offsets: np.ndarray) -> Iterator[Observation]:
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

        if self.still_world:
            yield from self._still_observations(start, offsets)
            return
        # A frame's instants are evaluated in groups of ``batch_observations``:
        # topology-identical instants of a group share one slot-major replay
        # and one composition, and the group is the largest unit of rows the
        # route holds at once.
        for begin in range(0, len(offsets), self.batch_observations):
            times = [float(start + offset) for offset in offsets[begin : begin + self.batch_observations]]
            for record in self.evaluate(times):
                yield Observation(
                    time_s=record.time_s,
                    epoch_frame=record.epoch_frame,
                    legs_of=record.legs,
                    composed=record.composed,
                    batch=record.batch,
                    slot=record.slot,
                )

    def _still_observations(self, start: float, offsets: np.ndarray) -> Iterator[Observation]:
        for offset in offsets:
            time_s = float(start + offset)
            held = self._still
            if held is not None:
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
        pending = _PendingObservation(
            time_s=time_s, epoch_frame=frame, binding=self._binding_for(frame), identity=self._identity(frame)
        )
        self._replay([pending], results)
        return results[time_s]


# ---------------------------------------------------------------------------
# Opening a session
# ---------------------------------------------------------------------------


def _resolve_sampling(requested: Motion, *, sampled: bool) -> str:
    """The route a request runs, once it is known whether anything moves.

    ``static`` is refused rather than silently upgraded: it asks for one
    observation a frame, and a world that moves inside a frame would publish
    no Doppler at all. A still world evaluates the same single observation
    per frame whatever the kind, which the rest of the session spells ``adc``.
    """

    motion_sampling = ("adc" if sampled else "static") if requested.kind == "auto" else requested.kind
    if motion_sampling == "static" and sampled:
        raise ValueError(
            "Motion.static() asks for one observation per frame, but this session moves within a frame; "
            "use Motion.adc() for the exhaustive reference, Motion.chirp() for stop-and-hop, or "
            "Motion.adaptive() for error-controlled interpolation"
        )
    return "adc" if motion_sampling == "static" else motion_sampling


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
    """

    from .synthesis.assembly import FmcwSpec, SlowTimeMode, waveform_sampling

    if not isinstance(motion, Motion):
        raise TypeError(f"motion must be a Motion, got {type(motion).__name__}")
    instants = _times(times)
    if not isinstance(sites, ScatterSitePolicy):
        raise TypeError(
            f"sites must be a ScatterSitePolicy, got {type(sites).__name__}; "
            "where the scatter sites come from is a declaration, not a search"
        )
    solve_config = radar.system_config.with_propagation(components=components, max_depth=max_depth)
    propagation = solve_config.propagation
    array = solve_config.sensors.array
    reference_frequency_hz = propagation.reference_frequency_hz

    dynamic = _dynamic_scene(scene)
    if dynamic.endpoint_trajectories and sensor_endpoints is None:
        raise ValueError("moving Core endpoints require explicit sensor_endpoints in array order")
    sampled = bool(
        dynamic.structure_trajectories
        or dynamic.structure_deformations
        or dynamic.endpoint_trajectories
        or sites.trajectory is not None
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
    motion_sampling = _resolve_sampling(motion, sampled=sampled)

    # Receiver nonlinearity and oscillator noise act on ADC-time samples.
    # Preserve the direct-spectrum fast path only for an ideal receiver.
    working = declared
    if isinstance(declared, FmcwSpec) and radar.frontend is not None:
        working = replace(declared, output_domain="beat")
    if motion_sampling == "adaptive" and not isinstance(working, FmcwSpec):
        raise NotImplementedError("adaptive motion currently requires FMCW")
    if motion_sampling == "adaptive" and motion.rediscover_every_frames is not None:
        raise ValueError("adaptive motion owns its topology discovery cadence")
    adc_sampled = sampled and motion_sampling in ("adc", "adaptive") and isinstance(working, FmcwSpec)
    samples_per_slot = working.num_samples if adc_sampled and isinstance(working, FmcwSpec) else 1

    mode = SlowTimeMode.FROZEN_WEIGHT_WITH_CARRIER_RATE
    cadence: int | None
    pair_samples: torch.Tensor | None = None
    single: WaveformSpec | None = None
    if sampled:
        slot_offsets, pair_samples, single = waveform_sampling(
            working, num_tx=array.num_tx, num_rx=array.num_rx, device=radar.device
        )
        offsets = np.asarray(slot_offsets, dtype=np.float64)
        if adc_sampled and isinstance(working, FmcwSpec):
            if working.t_start_s + (working.num_samples - 1) * working.sample_period_s >= working.chirp_period_s:
                raise ValueError("ADC observations must fit inside each chirp period")
            # An ADC frame has chirps * transmitters * samples observations, so
            # the schedule is an array rather than Python floats: the same
            # additions in the same order, paid once by numpy instead of once
            # per observation by the interpreter.
            offsets = (
                offsets[:, None] + working.t_start_s + np.arange(working.num_samples) * working.sample_period_s
            ).ravel()
            single = replace(single, num_samples=1, output_domain="beat")
        mode = SlowTimeMode.REFRESHED_WEIGHT_NO_RATE
        # A version poll cannot detect endpoint-induced path births. Complete
        # dynamic sampling therefore rediscovers at each observed instant.
        cadence = 1 if motion.rediscover_every_frames is None else motion.rediscover_every_frames * len(offsets)
    else:
        offsets = np.zeros(1)
        cadence = motion.rediscover_every_frames
    if sampled and any(b <= a + offsets[-1] for a, b in zip(instants, instants[1:], strict=False)):
        raise ValueError("dynamic frames must not overlap in observation time")

    world = WorldHalf(
        radar,
        dynamic,
        sites=sites,
        polarization=polarization,
        sensor_endpoints=sensor_endpoints,
        response=response,
        antenna_pattern=antenna_pattern,
        propagation=propagation,
        array=array,
        motion_sampling=motion_sampling,
        ad_mode=ad_mode,
        cadence=cadence,
        world_motion=motion.world,
        still_world=not sampled and cadence is None,
        batch_observations=int(motion.batch_observations),
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
        kind="static" if not sampled else motion_sampling,
        components=frozenset(propagation.components),
        max_depth=int(propagation.max_depth),
        grad=str(ad_mode),
        motion=motion,
        times=instants,
    )
    # Whether a non-adaptive route can have missed a path birth is a property
    # of the declared cadence and world motion, not of any one frame, so it is
    # decided once here and repeated on every frame the run publishes.
    complete = not sampled or cadence == 1 or (motion.world == "frozen_world" and world.loop.structures_move)
    if sampled and motion_sampling == "adaptive":
        return session, _adaptive_traces(session, world)
    return session, _frame_traces(session, world, path_set_complete=bool(complete))


def _frame_traces(session: Session, world: WorldHalf, *, path_set_complete: bool) -> Iterator[FrameTrace]:
    for start in session.times:
        yield FrameTrace(
            time_s=float(start),
            sample_times_s=start + session.offsets,
            observations=world.observations(start, session.offsets),
            path_set_complete=path_set_complete,
            motion_sampling_exhaustive=True,
        )


def _adaptive_traces(session: Session, world: WorldHalf) -> Iterator[FrameTrace]:
    from .synthesis.assembly import FmcwSpec

    spec = session.specs.working
    if not isinstance(spec, FmcwSpec):
        raise NotImplementedError("adaptive motion currently requires FMCW")
    for start in session.times:
        actual_times = start + session.offsets
        frame = trace_adaptive_frame(actual_times, world.evaluate, spec, session.motion, session.reference_frequency_hz)
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

    Every refreshed FMCW frame is, unless something acts on the rows between
    replay and synthesis (oscillator phase noise) or adds rows of its own (a
    declared coupling); those keep one launch per observation.
    """

    from .synthesis.assembly import FmcwSpec

    radar = session.radar
    return (
        session.sampled
        and isinstance(session.specs.working, FmcwSpec)
        and not session.path_phase_noise
        and radar.leakage is None
    )


def _echo_slot_batches(
    session: Session, working: FmcwSpec, observations: Iterator[Observation] | tuple[Observation, ...]
) -> tuple[torch.Tensor, EpochFrame, Observation]:
    """Synthesize a sampled FMCW frame from its slot batches, active pairs only.

    An observation sits in one TDM slot and is heard by one transmitter's
    ``num_rx`` pairs; the frame cube reads nothing else from it. Each slot
    batch is therefore synthesized as ONE launch over just those pair
    segments - per ADC sample index on the ADC route, because the sample
    instant is a launch parameter - and the frame cube gathers the results.
    Every segment sums the same rows in the same order as a per-observation
    launch would, so the cube is the same to the bit. Returns the
    ``[chirp, pair, sample]`` cube, the opening epoch frame and the closing
    observation.
    """

    from .synthesis import SynthesisPathBatch
    from .synthesis.assembly import FmcwSpec, require_compatible
    from .synthesis.fmcw import channel_phasor_to_beat_weight, synthesize_fmcw_rows

    single = session.specs.single
    assert isinstance(single, FmcwSpec)
    array = session.array
    num_tx, num_rx = array.num_tx, array.num_rx
    pairs = num_tx * num_rx
    samples = session.samples_per_slot
    values: list[torch.Tensor] = []
    targets: list[np.ndarray] = []
    opened: EpochFrame | None = None
    closing: Observation | None = None
    pending: list[tuple[int, Observation]] = []

    def flush() -> None:
        batch = pending[0][1].batch
        assert batch is not None
        composed = batch.composed
        # The waveform specs satisfy the protocol at run time; their fields are
        # frozen, which a protocol of plain attributes cannot express.
        checked: Any = single
        require_compatible(SynthesisPathBatch.from_radar_paths(composed, slow_time_mode=session.mode), checked)
        rows_per_slot = composed.path_count // batch.slots
        index = np.asarray([entry[0] for entry in pending], dtype=np.int64)
        slot_in_batch = np.asarray([entry[1].slot for entry in pending], dtype=np.int64)
        sample = index % samples
        for value in np.unique(sample):
            chosen = sample == value
            frame_slot = index[chosen] // samples
            # Sink-major ranks: the slot's transmitter ``t`` is heard on pairs
            # ``t + num_tx * r``, one per receiver ``r``.
            pair = ((frame_slot % num_tx)[:, None] + num_tx * np.arange(num_rx)).ravel()
            owner = np.repeat(slot_in_batch[chosen], num_rx)
            begin = owner * rows_per_slot + batch.offsets[pair]
            counts = batch.offsets[pair + 1] - batch.offsets[pair]
            prefix = np.cumsum(counts) - counts
            rows = np.repeat(begin - prefix, counts) + np.arange(int(counts.sum()))
            segment_offsets = np.concatenate(([0], np.cumsum(counts)))
            routing = _uploaded(session, np.concatenate((rows, segment_offsets, pair % num_tx)), composed.device)
            row_index, offsets, transmitter = torch.split(routing, [len(rows), len(segment_offsets), len(pair)])
            transfer = channel_phasor_to_beat_weight(composed.complex_transfer_ref.index_select(0, row_index))
            if composed.row_valid is not None:
                valid = composed.row_valid.index_select(0, row_index)
                transfer = torch.where(valid, transfer, torch.zeros_like(transfer))
            spec = single
            if session.adc_sampled:
                spec = replace(single, t_start_s=working.t_start_s + int(value) * working.sample_period_s)
            out = synthesize_fmcw_rows(
                composed.total_delay_s.index_select(0, row_index),
                None,
                transfer,
                offsets,
                spec,
                segment_tx_index=transmitter.to(torch.int32),
            )
            values.append(out.reshape(len(pair), spec.num_samples))
            chirp = np.repeat(frame_slot // num_tx, num_rx)
            targets.append(((chirp * pairs + pair) * samples + value) * spec.num_samples)
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
    if not np.array_equal(destination[order], np.arange(len(destination)) * single.num_samples):
        raise ValueError("a sampled frame's observations must cover every chirp, pair and sample exactly once")
    gather = _uploaded(session, order, values[0].device)
    cube = torch.cat(values).index_select(0, gather)
    return cube.reshape(working.num_chirps, pairs, working.num_samples), opened, closing


def _finish_frame(
    session: Session, synthesis: SynthesisResult, frame_time_s: float
) -> tuple[torch.Tensor, SynthesisResult]:
    """Apply the receive chain, then land the frame in its declared domain.

    ``frame_time_s`` keys the frame's receiver noise realisation, so two frames
    never share one.
    """

    from .synthesis.assembly import FmcwSpec, SynthesisResult, assemble_frame_cube

    radar = session.radar
    frame_cube = assemble_frame_cube(synthesis.cube, num_tx=session.array.num_tx, num_rx=session.array.num_rx)
    if radar.frontend is not None:
        frame_cube = radar.frontend.apply(
            frame_cube, phase_in_signal=session.path_phase_noise, frame_time_s=frame_time_s
        ).signal
    declared, working = session.specs.declared, session.specs.working
    if isinstance(declared, FmcwSpec) and declared.output_domain != synthesis.output_domain:
        from .processing.range_doppler import fmcw_range_fft

        assert isinstance(working, FmcwSpec)
        frame_cube = fmcw_range_fft(frame_cube)
        synthesis = SynthesisResult.from_fmcw(
            frame_cube.permute(2, 1, 0, 3).reshape(working.num_chirps, -1, working.num_samples), declared
        )
    return frame_cube, synthesis


def echo_frame(session: Session, trace: FrameTrace) -> SceneFrame:
    """Synthesize one traced frame. The whole instrument half lives here.

    Everything this does is a property of the radar and the waveform, never of
    the world: the per-observation synthesis, the common-oscillator phase the
    receive chain applies per path before coherent summation, the slot gather,
    the receive chain itself and the output-domain transform. That is what lets
    a caller re-run it against a different receiver without re-tracing.

    ``trace.observations`` is drained exactly once, in order.
    """

    from .synthesis.assembly import FmcwSpec, SynthesisResult

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
        synthesis = SynthesisResult.from_fmcw(cube, replace(working, output_domain="beat"))
    elif _batches_slots(session):
        assert isinstance(working, FmcwSpec)
        cube, opened, closing = _echo_slot_batches(session, working, trace.observations)
        published = closing.composed
        if session.adc_sampled:
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
                if session.adc_sampled:
                    assert isinstance(working, FmcwSpec) and isinstance(single, FmcwSpec)
                    observation_spec = replace(
                        single,
                        t_start_s=working.t_start_s
                        + (len(slot_cubes) % session.samples_per_slot) * working.sample_period_s,
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
                assert isinstance(working, FmcwSpec)
                stacked = stacked.reshape(-1, session.samples_per_slot, len(pairs)).transpose(1, 2)
                cube = stacked[session.pair_samples, pairs]
                if working.output_domain == "spectrum":
                    from .processing.range_doppler import fmcw_range_fft

                    cube = fmcw_range_fft(cube)
                synthesis = SynthesisResult.from_fmcw(cube, working)
            else:
                synthesis = replace(synthesis, cube=stacked[session.pair_samples, pairs])
    if opened is None or closing is None or published is None or synthesis is None:
        raise ValueError("a frame trace must carry at least one observation")
    frame_cube, synthesis = _finish_frame(session, synthesis, trace.time_s)
    loop = session.loop
    return SceneFrame(
        cube=frame_cube,
        synthesis=synthesis,
        time_s=trace.time_s,
        sample_times_s=trace.sample_times_s,
        epoch=int(opened.epoch),
        reason=opened.reason,
        path_set_complete=trace.path_set_complete,
        motion_sampling_exhaustive=trace.motion_sampling_exhaustive,
        compile_count=int(loop.compile_count),
        discovery_count=int(loop.discovery_count),
        epoch_frame=closing.epoch_frame,
        legs=closing.legs,
        composed=published,
        diagnostics=trace.stats,
        motion_sampling=session.kind,
        output_spec=session.specs.declared,
        array=session.array,
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
    path birth can have been missed, the second says no observation's transport
    was interpolated between probes. An adaptive trace of a certifiable world
    reports the first without the second, which is exactly the trade it exists
    to make.
    """

    #: ``"static"``, ``"chirp"``, ``"adc"`` or ``"adaptive"``: how often the
    #: world was resampled inside a frame.
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
        frames.append(replace(trace, observations=observations, opened=opened))
        epochs.append(int(opened.epoch))
        reasons.append(opened.reason)
    closing = _retained(frames[-1])[-1]
    return Paths(
        kind=session.kind,
        carrier=float(session.reference_frequency_hz),
        waveform=session.radar.waveform,
        times=tuple(session.times),
        sample_times=tuple(frame.sample_times_s for frame in frames),
        los="los" in session.components,
        reflections=int(session.max_depth) if "reflection" in session.components else 0,
        grad=session.grad,
        motion=session.motion,
        epochs=tuple(epochs),
        rediscovery_reasons=tuple(reasons),
        compile_count=int(session.loop.compile_count),
        discovery_count=int(session.loop.discovery_count),
        path_set_complete=all(frame.path_set_complete for frame in frames),
        motion_sampling_exhaustive=all(frame.motion_sampling_exhaustive for frame in frames),
        adaptive_diagnostics=tuple(frame.stats for frame in frames if frame.stats is not None),
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
