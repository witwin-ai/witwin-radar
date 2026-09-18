"""Scene-driven radar assembly with motion evaluated at waveform observation times.

Core owns geometry and authored motion; Channel evaluates each one-way path;
Radar composes the round trip and synthesizes the waveform. Dynamic observations
refresh the complex transport, so no parameter JVP is interpreted as velocity.
Topology is rediscovered at every observation by default: replay alone cannot
identify new paths. An explicitly longer discovery cadence is reported as an
incomplete path set. No velocity is inferred by subtracting adjacent path rows.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, replace

import torch

from .propagation import RadarEndpointSpec

#: The three default ID bases, chosen far above Core's own counters.
#:
#: ``witwin.core.identity`` allocates structure, material, assignment and
#: antenna IDs from zero-based process counters, and a radar endpoint ID that
#: collided with one of those would be two different things wearing one name in
#: the same world. Starting each block at a distinct million keeps the three
#: radar blocks apart from each other AND clear of any plausible Core counter,
#: while staying small enough to read in a failure message.
DEFAULT_TRANSMITTER_ID_BASE = 1_000_000
DEFAULT_RECEIVER_ID_BASE = 2_000_000
DEFAULT_SITE_ID_BASE = 3_000_000

#: The world-frame endpoint polarization used when a caller declares none.
#:
#: Channel's endpoint polarization is a WORLD-frame vector. Channel owns the
#: material-field projection exactly once; the radar sensor stage receives the
#: resulting complex transfer and therefore has no polarization input.
DEFAULT_POLARIZATION = (0.0, 0.0, 1.0)


@dataclass(frozen=True, slots=True)
class AdaptiveMotionSpec:
    """Sampled error control, not a proof of absent events between probes.

    The phase tolerance is radians per path, before coherent summation; the
    amplitude tolerance is relative per path. Near coherent nulls a relative IQ
    bound cannot be inferred from these path bounds. Exhausting the discovery
    budget raises.

    ``max_interval_s`` is the MAXIMUM PROBE SPACING in seconds and it always
    applies: no accepted interval spans more than this, so the probe grid is
    never coarser than ``max_interval_s / (2 * (interpolation_nodes - 1))``.
    LOWERING it refines the grid. Every tolerance here is checked by sampling,
    so it bounds only what it samples: motion periodic at the grid's step, or
    at a divisor of it, sits at a zero of every probe and is invisible to all
    of these tests. This bound is what sets that step, and it is therefore an
    accuracy control as much as a topology one. It is NOT relaxed for a
    topologically certified family: the certification says no path can be born,
    not that no path moves fast. Within the bound the run starts from the
    coarsest partition it allows rather than bisecting down to it. Set it from
    the motion's bandwidth; the default 2 ms resolves roughly 250 Hz at two
    nodes.

    ``interpolation_nodes`` is how many sampled instants carry one accepted
    interval, and therefore the polynomial order of the delay it interpolates:
    the default 2 is the linear rule and 5 is a quartic. Each interval probes a
    grid of ``2 * (nodes - 1) + 1`` instants, spends the even ones as
    interpolation nodes and tests the error at the odd ones.

    Raise it when the PHASE TEST is what shortens your intervals, which is
    micro-Doppler at a high carrier: on a 4.096 ms frame of an 80 Hz rotor,
    2/3/5 nodes measured 38/21/25 probes and 36.0/22.2/21.7 ms per frame. Leave
    it at 2 when the bound above decides the length instead, because then the
    order cannot reduce the interval count and only multiplies each interval's
    grid: the same measurement on a 24.96 ms MIMO frame gave 27/53/105 probes
    and 95.5/166.7/173.5 ms for one unchanged 13-interval partition. Raising it
    never makes a coarse grid safe - the grid still has to resolve the motion.
    """

    phase_error_rad: float = 0.02
    relative_amplitude_error: float = 0.02
    max_interval_s: float = 0.002
    interpolation_nodes: int = 2
    max_evaluations: int = 8192
    batch_observations: int = 256

    def __post_init__(self):
        for name in ("phase_error_rad", "relative_amplitude_error", "max_interval_s"):
            value = getattr(self, name)
            if isinstance(value, torch.Tensor) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive host value")
        for name in ("max_evaluations", "batch_observations"):
            _positive_int(getattr(self, name), name=name)
        if _positive_int(self.interpolation_nodes, name="interpolation_nodes") < 2:
            raise ValueError("interpolation_nodes must be at least 2; one node cannot interpolate")


MOTION_KINDS = ("auto", "static", "chirp", "adc", "adaptive")


@dataclass(frozen=True, slots=True)
class Motion:
    """How often the world is resampled inside one frame.

    Build one with :meth:`auto`, :meth:`static`, :meth:`chirp`, :meth:`adc` or
    :meth:`adaptive`. The four knobs this replaces had to agree with each other
    - an adaptive tolerance without adaptive sampling was a refusal, and a
    discovery cadence with it was another - so the combinations that used to
    raise are now unwritable.

    The tolerance fields belong to :meth:`adaptive` and are ignored by every
    other kind. They are fields of this one record rather than a second object
    because a sampling choice and the error budget that justifies it are one
    statement.
    """

    kind: str = "auto"
    #: Adaptive: the largest tested round-trip phase error an accepted interval
    #: may carry, rad.
    phase_error: float = 0.02
    #: Adaptive: the largest tested relative amplitude error, dimensionless.
    relative_amplitude_error: float = 0.02
    #: Adaptive: the maximum probe spacing, s. Enforced unconditionally,
    #: including for a family certified complete for all time. Every tolerance
    #: here is checked by sampling and therefore cannot see motion periodic at
    #: the probe grid's step; this bound is what sets that step.
    max_interval: float = 0.002
    #: Adaptive: how many sampled instants one accepted interval interpolates
    #: through. Two is the linear rule and five is a quartic.
    nodes: int = 2
    #: Adaptive: the discovery budget. Exhausting it raises rather than
    #: returning an unchecked cube.
    max_evaluations: int = 8192
    #: Adaptive: observations per batched replay.
    batch_observations: int = 256
    #: Topology rediscovery cadence in frames. ``None`` rediscovers at every
    #: observation, which is the only cadence that cannot miss a path birth.
    rediscover_every_frames: int | None = None
    #: Channel's replay vocabulary for compiled geometry.
    world: str = "frozen_world"

    def __post_init__(self) -> None:
        if self.kind not in MOTION_KINDS:
            raise ValueError(f"Motion.kind must be one of {list(MOTION_KINDS)}, got {self.kind!r}")

    @classmethod
    def auto(cls, *, rediscover_every_frames: int | None = None, world: str = "frozen_world") -> Motion:
        """Resample at every ADC instant when anything moves, else once a frame.

        "Anything moves" is a property of the session, not of this record:
        structure trajectories or deformations, endpoint trajectories, a target
        trajectory, or a receiver with oscillator phase noise, which needs
        ADC-time observations to place its delayed phase difference.
        """

        return cls(kind="auto", rediscover_every_frames=rediscover_every_frames, world=world)

    @classmethod
    def static(cls, *, rediscover_every_frames: int | None = None, world: str = "frozen_world") -> Motion:
        """One observation per frame. Refused for a world that moves."""

        return cls(kind="static", rediscover_every_frames=rediscover_every_frames, world=world)

    @classmethod
    def chirp(cls, *, rediscover_every_frames: int | None = None, world: str = "frozen_world") -> Motion:
        """Stop and hop: geometry frozen within each chirp, symbol or pulse."""

        return cls(kind="chirp", rediscover_every_frames=rediscover_every_frames, world=world)

    @classmethod
    def adc(cls, *, rediscover_every_frames: int | None = None, world: str = "frozen_world") -> Motion:
        """Resample at every ADC instant. The exhaustive reference."""

        return cls(kind="adc", rediscover_every_frames=rediscover_every_frames, world=world)

    @classmethod
    def adaptive(
        cls,
        *,
        phase_error: float = 0.02,
        relative_amplitude_error: float = 0.02,
        max_interval: float = 0.002,
        nodes: int = 2,
        max_evaluations: int = 8192,
        batch_observations: int = 256,
        rediscover_every_frames: int | None = None,
        world: str = "frozen_world",
    ) -> Motion:
        """Interpolate between error-controlled probes. FMCW only.

        Faster than :meth:`adc` by 3 to 64 times on the measured scenes, and it
        certifies nothing between its probes: an arbitrarily brief path birth
        or oscillation is invisible to a sampled test. The run publishes
        ``path_set_complete`` and ``motion_sampling_exhaustive`` separately so
        that limit is readable rather than implied.
        """

        return cls(
            kind="adaptive",
            phase_error=phase_error,
            relative_amplitude_error=relative_amplitude_error,
            max_interval=max_interval,
            nodes=nodes,
            max_evaluations=max_evaluations,
            batch_observations=batch_observations,
            rediscover_every_frames=rediscover_every_frames,
            world=world,
        )

    def _adaptive_spec(self) -> AdaptiveMotionSpec:
        return AdaptiveMotionSpec(
            phase_error_rad=float(self.phase_error),
            relative_amplitude_error=float(self.relative_amplitude_error),
            max_interval_s=float(self.max_interval),
            interpolation_nodes=int(self.nodes),
            max_evaluations=int(self.max_evaluations),
            batch_observations=int(self.batch_observations),
        )


#: A scatter site is excited at exactly one watt.
#:
#: The site is a re-radiator, not a second transmitter: the whole target
#: strength lives in the two-way join's ``S = sqrt(4 pi sigma) / lambda``
#: factor. A site excitation of anything but unit power multiplies that factor
#: again, and with a transmit power of 1 W the extra ``sqrt(P)`` is numerically
#: invisible - which is exactly how a squared transmit power ships.
SITE_EXCITATION_POWER_W = 1.0

#: Where scatter sites may come from. Both are declarations by the caller or by
#: Core; neither derives a site from geometry.
SITE_SOURCE_EXPLICIT = "explicit"
SITE_SOURCE_STRUCTURE_ANCHOR = "structure_anchor"
SITE_SOURCES = (SITE_SOURCE_EXPLICIT, SITE_SOURCE_STRUCTURE_ANCHOR)

_MESH_SITE_DEFERRAL = (
    "deriving scatter sites by sampling a structure's MESH is a named Phase-11 "
    "deferral (R-ADR-020). A sampling rule is a geometry algorithm, and "
    "geometry on a production path belongs to Channel's native geometry owner, "
    "not to a Torch expression in Radar. Declare the sites instead - "
    "ScatterSitePolicy.explicit(positions) - or give the structure a rigid "
    "motion so that Core publishes a world anchor for it"
)


def _positive_int(value: object, *, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive int, got {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class StableIdAllocator:
    """Deterministic stable world IDs for one radar's endpoints and sites.

    Three contiguous blocks, one per endpoint role, each starting at a declared
    base. An ID is therefore a pure function of ``(role, array index)`` and of
    nothing else: not of construction order, not of the process, not of how many
    frames have been simulated. That is the property a frozen leg topology
    depends on, because it names its rows by ``source_id`` and ``sink_id`` and a
    later frame must be able to say it is talking about the same endpoints.

    The blocks are checked for overlap when they are allocated rather than when
    they are declared, because whether two bases collide depends on the counts.
    An overlap is refused: two endpoints sharing a stable ID is not a smaller
    answer, it is a leg that joins the wrong rows and still publishes a full
    result.
    """

    transmitter_base: int = DEFAULT_TRANSMITTER_ID_BASE
    receiver_base: int = DEFAULT_RECEIVER_ID_BASE
    site_base: int = DEFAULT_SITE_ID_BASE

    def __post_init__(self) -> None:
        for name in ("transmitter_base", "receiver_base", "site_base"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative int, got {value!r}")

    def allocate(
        self, *, transmitter_count: int, receiver_count: int, site_count: int
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        """The three ID tuples, in array order, with the blocks proved disjoint."""

        counts = (
            _positive_int(transmitter_count, name="transmitter_count"),
            _positive_int(receiver_count, name="receiver_count"),
            _positive_int(site_count, name="site_count"),
        )
        bases = (self.transmitter_base, self.receiver_base, self.site_base)
        names = ("transmitter", "receiver", "site")
        blocks = tuple(tuple(range(base, base + count)) for base, count in zip(bases, counts, strict=True))
        for first in range(len(blocks)):
            for second in range(first + 1, len(blocks)):
                low, high = bases[first], bases[second]
                if low < high + counts[second] and high < low + counts[first]:
                    raise ValueError(
                        f"the {names[first]} ID block "
                        f"[{bases[first]}, {bases[first] + counts[first]}) "
                        f"overlaps the {names[second]} block "
                        f"[{bases[second]}, {bases[second] + counts[second]}); "
                        "two endpoints cannot share one stable world ID"
                    )
        return blocks


@dataclass(frozen=True, slots=True, eq=False)
class ScatterSitePolicy:
    """Where the scatter sites of one binding come from, declared explicitly.

    Two sources, and the restriction is the design rather than an unfinished
    edge:

    ``explicit``
        The caller hands over an ``(S, 3)`` tensor (or a sequence of triples) of
        world positions. A live tensor is passed through UNTOUCHED, so a
        ``requires_grad`` leaf or a forward-AD dual keeps its tape all the way
        into both legs.

    ``structure_anchor``
        One site per selected structure, at the world translation the snapshot's
        rigid motion publishes for it. This is a Core-owned quantity read as it
        stands; Radar computes nothing from the mesh. A structure with no rigid
        motion has no Core-owned anchor and is refused by name.

    What is deliberately absent is any rule that *derives* a site from geometry -
    a surface sample, a centroid, a bounding-box centre, a visibility-weighted
    scatterer set. Every one of those is a geometry algorithm, and a geometry
    algorithm written in Torch on the production path is the thing this
    architecture exists to keep out. R-ADR-020 records the deferral and names
    what closing it would need.

    ``power_w`` is the site excitation and defaults to
    :data:`SITE_EXCITATION_POWER_W`. Changing it is almost always wrong; read
    that constant's note first.
    """

    source: str
    positions_m: object | None = None
    structure_ids: tuple[int, ...] | None = None
    stable_ids: tuple[int, ...] | None = None
    power_w: float = SITE_EXCITATION_POWER_W
    trajectory: object | None = None

    def __post_init__(self) -> None:
        if self.trajectory is not None and not callable(getattr(self.trajectory, "at", None)):
            raise TypeError("a site trajectory must expose at(time_s) returning Kinematics")
        if self.trajectory is not None and self.source != SITE_SOURCE_EXPLICIT:
            raise ValueError("a site trajectory requires explicit material-point positions")
        if self.source not in SITE_SOURCES:
            raise ValueError(f"source must be one of {list(SITE_SOURCES)}, got {self.source!r}")
        if self.source == SITE_SOURCE_EXPLICIT:
            if self.positions_m is None:
                raise ValueError(
                    "an explicit site policy requires positions_m; ScatterSitePolicy.explicit(positions) builds one"
                )
            if self.structure_ids is not None:
                raise ValueError(
                    "structure_ids belongs to the structure_anchor policy; an "
                    "explicit policy already names its sites by position"
                )
        else:
            if self.positions_m is not None:
                raise ValueError(
                    "positions_m belongs to the explicit policy; a "
                    "structure_anchor policy reads its positions from the "
                    "snapshot"
                )
        if not float(self.power_w) > 0.0:
            raise ValueError("power_w must be positive")

    @classmethod
    def explicit(
        cls,
        positions_m: object,
        *,
        stable_ids: tuple[int, ...] | None = None,
        power_w: float = SITE_EXCITATION_POWER_W,
        trajectory: object | None = None,
    ) -> ScatterSitePolicy:
        return cls(
            source=SITE_SOURCE_EXPLICIT,
            positions_m=positions_m,
            stable_ids=None if stable_ids is None else tuple(int(v) for v in stable_ids),
            power_w=power_w,
            trajectory=trajectory,
        )

    @classmethod
    def structure_anchor(
        cls,
        *,
        structure_ids: tuple[int, ...] | None = None,
        stable_ids: tuple[int, ...] | None = None,
        power_w: float = SITE_EXCITATION_POWER_W,
    ) -> ScatterSitePolicy:
        """Sites at the world anchors Core publishes for moving structures.

        ``structure_ids`` selects a subset; ``None`` takes every structure the
        snapshot carries. Selection and ordering are both by ascending structure
        ID rather than by the snapshot's tuple order, so the site array order is
        a function of world identity and survives a reordered scene.
        """

        return cls(
            source=SITE_SOURCE_STRUCTURE_ANCHOR,
            structure_ids=(None if structure_ids is None else tuple(int(value) for value in structure_ids)),
            stable_ids=None if stable_ids is None else tuple(int(v) for v in stable_ids),
            power_w=power_w,
        )

    def resolve(self, snapshot: object, *, device: torch.device) -> torch.Tensor:
        """The ``(S, 3)`` float32 site positions this policy names."""

        if self.source == SITE_SOURCE_EXPLICIT:
            if self.trajectory is not None:
                from .propagation import Kinematics

                sample = self.trajectory.at(snapshot.time_s)
                if not isinstance(sample, Kinematics):
                    raise TypeError("site trajectory.at(time_s) must return Kinematics")
                positions = _site_positions(sample.positions_m, device=device)
                if positions.shape != _site_positions(self.positions_m, device=device).shape:
                    raise ValueError("a site trajectory must preserve material-point count and ordering")
                return positions
            return _site_positions(self.positions_m, device=device)
        return _structure_anchor_positions(snapshot, self.structure_ids, device=device)


def _site_positions(positions: object, *, device: torch.device) -> torch.Tensor:
    """Normalise declared site positions without disturbing a live tensor.

    A tensor is validated and returned as it stands. It is deliberately NOT
    moved, cast, or made contiguous here: every one of those is a new node that
    would leave the caller holding a tensor that is no longer the one the legs
    differentiate through, and a device or dtype mismatch is a caller error
    worth a message rather than a silent copy.
    """

    if isinstance(positions, torch.Tensor):
        if positions.dtype != torch.float32:
            raise TypeError(
                f"site positions must use torch.float32, got {positions.dtype}; "
                "casting here would detach the tensor a caller expects to "
                "differentiate through"
            )
        if positions.device != device:
            raise ValueError(
                f"site positions are on {positions.device} but this binding is "
                f"on {device}; move them before declaring the policy so the "
                "moved tensor is the one you hold"
            )
        if positions.ndim != 2 or int(positions.shape[1]) != 3:
            raise ValueError(f"site positions must have shape (S, 3), got {tuple(positions.shape)}")
        if not positions.is_contiguous():
            raise ValueError("site positions must be contiguous")
        return positions
    return torch.tensor([tuple(float(value) for value in row) for row in positions], dtype=torch.float32, device=device)


def _structure_anchor_positions(
    snapshot: object, structure_ids: tuple[int, ...] | None, *, device: torch.device
) -> torch.Tensor:
    """One world anchor per selected structure, read out of the snapshot.

    ``StructureState.rigid_motion.translation`` is a world-frame vector Core
    already owns, and ``torch.stack`` preserves whatever tape it carries, so a
    site that rides a ``LinearTrajectory`` reaches the legs differentiably
    without this module ever forming a position of its own.
    """

    states = getattr(snapshot, "structures", None)
    if states is None:
        raise TypeError("snapshot must expose structures; pass a witwin.core SceneSnapshot")
    by_id: dict[int, object] = {}
    for state in states:
        key = int(state.structure_id)
        if key in by_id:
            raise ValueError(
                f"structure_id {key} appears twice in the snapshot; a site anchor must name exactly one structure"
            )
        by_id[key] = state
    if structure_ids is None:
        selected = sorted(by_id)
    else:
        selected = sorted(structure_ids)
        missing = [key for key in selected if key not in by_id]
        if missing:
            raise ValueError(f"structure_ids {missing} are not in this snapshot, which carries {sorted(by_id)}")
        if len(set(selected)) != len(selected):
            raise ValueError("structure_ids must not repeat a structure")
    anchors = []
    for key in selected:
        motion = getattr(by_id[key], "rigid_motion", None)
        translation = None if motion is None else motion.translation
        if translation is None:
            raise NotImplementedError(
                f"structure {key} carries no rigid-motion world anchor, so this "
                f"snapshot publishes no Core-owned site position for it: "
                f"{_MESH_SITE_DEFERRAL}"
            )
        anchors.append(translation.reshape(3))
    return torch.stack(anchors).to(device=device, dtype=torch.float32).contiguous()


@dataclass(frozen=True, slots=True, eq=False)
class RadarWorldBinding:
    """One radar and one snapshot, as the endpoint specs the legs consume.

    Four specs and three ID tuples. The two site specs are the same sites in
    their two roles - sink of the inbound leg, source of the outbound leg - and
    they SHARE one ``positions_m`` object, which ``__post_init__`` asserts
    rather than assumes. That aliasing is the whole reason this is one type
    instead of four loose arguments.

    The ID tuples are host tuples, not tensors, because the composer's declared
    identity lists are host lists and because reading them back out of a device
    tensor would be a host observation on a path that has none.
    """

    transmitters: RadarEndpointSpec
    receivers: RadarEndpointSpec
    site_sources: RadarEndpointSpec
    site_sinks: RadarEndpointSpec
    transmitter_ids: tuple[int, ...]
    receiver_ids: tuple[int, ...]
    site_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.site_sources.positions_m is not self.site_sinks.positions_m:
            raise ValueError(
                "the site source and sink specs must share one positions_m "
                "tensor; rebuilding it for the second role drops half of a "
                "reverse gradient and all of a forward tangent"
            )
        if self.transmitters.powers_w is None:
            raise ValueError("the transmitter spec must carry powers_w")
        if self.site_sources.powers_w is None:
            raise ValueError("the site source spec must carry powers_w")
        if self.receivers.powers_w is not None:
            raise ValueError("the receiver spec must not carry powers_w")
        if self.site_sinks.powers_w is not None:
            raise ValueError("the site sink spec must not carry powers_w")

    @property
    def site_positions_m(self) -> torch.Tensor:
        """The one site tensor both legs differentiate through."""

        return self.site_sources.positions_m

    @property
    def site_count(self) -> int:
        return self.site_sources.count

    @property
    def device(self) -> torch.device:
        return self.transmitters.device


def _endpoint_spec(
    positions_m: torch.Tensor, stable_ids: tuple[int, ...], *, polarization: torch.Tensor, power_w: float | None
) -> RadarEndpointSpec:
    rows = int(positions_m.shape[0])
    if rows != len(stable_ids):
        raise ValueError(
            f"{rows} positions were given {len(stable_ids)} stable IDs; the "
            "array order IS the ID order and the two are permuted together"
        )
    device = positions_m.device
    return RadarEndpointSpec(
        stable_ids=torch.tensor(stable_ids, dtype=torch.int64, device=device),
        positions_m=positions_m,
        polarizations=polarization.expand(rows, 3).contiguous(),
        powers_w=(None if power_w is None else torch.full((rows,), float(power_w), dtype=torch.float32, device=device)),
    )


def _polarization_tensor(polarization: object, *, device: torch.device) -> torch.Tensor:
    """Validate the declared polarization on the HOST, then build it once.

    The non-zero check is made on the three declared floats rather than on the
    tensor. ``bool(torch.any(...))`` would read a device tensor back, which is a
    host observation this module has no budget for even at setup, and it would
    be a strictly worse message besides.
    """

    values = tuple(float(value) for value in polarization)
    if len(values) != 3:
        raise ValueError(f"polarization must be a 3-vector, got {values!r}")
    if not any(values):
        raise ValueError("polarization must be non-zero")
    return torch.tensor(values, dtype=torch.float32, device=device)


def _array_positions(radar: object, name: str) -> torch.Tensor:
    positions = getattr(radar, name, None)
    if not isinstance(positions, torch.Tensor):
        raise TypeError(f"radar.{name} must be a torch.Tensor of world element positions; pass a witwin.radar.Radar")
    if positions.dtype != torch.float32:
        raise TypeError(f"radar.{name} must use torch.float32, got {positions.dtype}")
    if positions.ndim != 2 or int(positions.shape[1]) != 3:
        raise ValueError(f"radar.{name} must have shape (N, 3), got {tuple(positions.shape)}")
    if not positions.is_contiguous():
        raise ValueError(f"radar.{name} must be contiguous")
    return positions


@dataclass(frozen=True, slots=True)
class SensorEndpointIds:
    """Core phase-centre IDs in the radar's TX and RX array order.

    Endpoint trajectories move phase centres. The antenna orientation remains
    the Radar pose; rotating elements must author their pattern pose explicitly.
    """

    transmitters: tuple[int, ...]
    receivers: tuple[int, ...]

    def __post_init__(self):
        for name in ("transmitters", "receivers"):
            values = getattr(self, name)
            if not values or len(set(values)) != len(values):
                raise ValueError(f"{name} must contain distinct endpoint IDs")


def bind_radar_world(
    radar: object,
    snapshot: object,
    *,
    sites: ScatterSitePolicy,
    ids: StableIdAllocator | None = None,
    polarization: object = DEFAULT_POLARIZATION,
    sensor_endpoints: SensorEndpointIds | None = None,
) -> RadarWorldBinding:
    """Turn one ``Radar`` plus one ``SceneSnapshot`` into endpoint specs.

    The transmit elements become SOURCES carrying the array's transmit power in
    watts; the receive elements become SINKS with no power at all. Both come
    from ``radar.tx_pos`` / ``radar.rx_pos``, which are the pose-transformed
    world positions the radar already maintains - they are used as they stand
    rather than rebuilt, so a radar whose pose is a differentiable quantity
    keeps that property here.

    ``snapshot`` is required even for an explicit site policy. A binding is
    against a world at an instant, and letting it be optional would invite a
    caller to bind once and replay against a world that has moved on.

    This runs once per topology epoch, not per frame. It allocates IDs, builds
    three small constant tensors, and copies nothing back to the host.
    """

    transmitter_positions = _array_positions(radar, "tx_pos")
    receiver_positions = _array_positions(radar, "rx_pos")
    device = transmitter_positions.device
    if sensor_endpoints is not None:
        from .propagation import endpoint_kinematics

        if not isinstance(sensor_endpoints, SensorEndpointIds):
            raise TypeError("sensor_endpoints must be SensorEndpointIds")
        tx = endpoint_kinematics(snapshot, sensor_endpoints.transmitters, device=device)
        rx = endpoint_kinematics(snapshot, sensor_endpoints.receivers, device=device)
        if tx.positions_m.shape != transmitter_positions.shape or rx.positions_m.shape != receiver_positions.shape:
            raise ValueError("sensor endpoint counts must match the radar array")
        transmitter_positions, receiver_positions = tx.positions_m, rx.positions_m
    if receiver_positions.device != device:
        raise ValueError(
            f"radar.tx_pos is on {device} but radar.rx_pos is on {receiver_positions.device}; one radar is one device"
        )
    site_positions = sites.resolve(snapshot, device=device)

    allocator = StableIdAllocator() if ids is None else ids
    transmitter_ids, receiver_ids, allocated_site_ids = allocator.allocate(
        transmitter_count=int(transmitter_positions.shape[0]),
        receiver_count=int(receiver_positions.shape[0]),
        site_count=int(site_positions.shape[0]),
    )
    site_ids = allocated_site_ids if sites.stable_ids is None else sites.stable_ids
    if len(site_ids) != int(site_positions.shape[0]):
        raise ValueError(
            f"the site policy declared {len(site_ids)} stable IDs for {int(site_positions.shape[0])} site positions"
        )
    overlap = (set(site_ids) & set(transmitter_ids)) | (set(site_ids) & set(receiver_ids))
    if overlap:
        raise ValueError(
            f"site stable IDs {sorted(overlap)} collide with the transmitter or "
            "receiver blocks; two endpoints cannot share one stable world ID"
        )

    polarization_vector = _polarization_tensor(polarization, device=device)
    transmit_power_w = float(radar.transmit_power_watts)
    return RadarWorldBinding(
        transmitters=_endpoint_spec(
            transmitter_positions, transmitter_ids, polarization=polarization_vector, power_w=transmit_power_w
        ),
        receivers=_endpoint_spec(receiver_positions, receiver_ids, polarization=polarization_vector, power_w=None),
        site_sources=_endpoint_spec(site_positions, site_ids, polarization=polarization_vector, power_w=sites.power_w),
        site_sinks=_endpoint_spec(site_positions, site_ids, polarization=polarization_vector, power_w=None),
        transmitter_ids=transmitter_ids,
        receiver_ids=receiver_ids,
        site_ids=tuple(site_ids),
    )


#: The axis names of the published multi-frame cube. The last two are the
#: waveform's own slow and fast axes and are filled in from the synthesis
#: result, so an OFDM run publishes ``("frame", "tx", "rx", "symbol",
#: "subcarrier")`` without this module knowing what a symbol is.
SIMULATION_CUBE_LEADING_AXES = ("frame", "tx", "rx")


@dataclass(frozen=True, slots=True, eq=False)
class RadarSimulationResult:
    """What one :meth:`witwin.radar.Radar.simulate` call produced.

    Data only. The driver builds it through :meth:`from_frames`, which is this
    repository's standing shape for a result: the producer knows what it made,
    and a consumer never has to infer a phasor convention from a method name.

    ``cube`` is ``[frame, TX, RX, slow, fast]`` and is the product. It is one
    stacked tensor rather than a list because a frame sequence is what every
    downstream consumer - a range-Doppler map, a tracker, a loss - indexes, and
    the stack is a single differentiable op outside the frame loop.

    The four ``last_*`` members are the LAST frame's typed state, and this
    record is their only home: a radar holds no run state, so there is no
    second copy to disagree with these and no way for a call that raised part
    way through to leave one behind describing a world it never simulated.
    They describe one frame, not the sequence: a compiled scene and a leg pair
    are per-epoch and per-frame objects, and stacking them would either
    misrepresent the epochs or retain every frame's device memory for the life
    of the result. Keeping the last one is the diagnostic the plan asked for and
    the smallest retention that answers it.

    ``path_set_complete`` and ``motion_sampling_exhaustive`` are TWO
    statements, and a consumer that needs to know how much to trust a run
    needs both. The first says no path birth can have been missed - because
    every observation was evaluated, or because the candidate family was
    certified complete for all time. The second says no observation's transport
    was interpolated between probes. Adaptive sampling routinely gives the
    first and not the second, which is exactly the trade it exists to make;
    collapsing them into one flag would report a proven-complete adaptive run
    as if it might have lost a path.

    RETENTION, stated because these members are a real tensor lifetime. The
    ``last_*`` members alias the frame's own batches, so holding this result
    holds that frame's device tensors - and, when ``ad_mode`` asked for a graph,
    that frame's autograd graph. None of them holds a tape: an autograd context
    or a ``saved_tensors`` tuple in any of these fields would be a data record
    turned into a handle on somebody else's memory, and
    ``tests/test_phase9_tape_non_leak.py`` walks all four to keep it that way.
    """

    cube: torch.Tensor
    times_s: tuple[float, ...]
    kind: str
    axes: tuple[str, ...]
    phasor: str
    time_dependence: str
    reference_frequency_hz: float
    epochs: tuple[int, ...]
    rediscovery_reasons: tuple[str | None, ...]
    compile_count: int
    discovery_count: int
    last_snapshot: object
    last_compiled_scene: object
    last_propagation: object
    last_radar_paths: object
    sample_times_s: tuple[tuple[float, ...], ...] = ()
    path_set_complete: bool = True
    motion_sampling_exhaustive: bool = True
    motion_sampling: str = "static"
    output_domain: str = "beat"
    adaptive_diagnostics: tuple[dict, ...] = ()

    def __post_init__(self) -> None:
        if self.cube.dim() != len(self.axes):
            raise ValueError(
                f"a {self.kind} simulation cube has {len(self.axes)} axes "
                f"{self.axes}, got shape {tuple(self.cube.shape)}"
            )
        frames = int(self.cube.shape[0])
        for name in ("times_s", "epochs", "rediscovery_reasons"):
            values = getattr(self, name)
            if len(values) != frames:
                raise ValueError(
                    f"{name} carries {len(values)} entries for {frames} frames; "
                    "the per-frame records and the cube's frame axis are the "
                    "same sequence"
                )

    @property
    def frame_count(self) -> int:
        return int(self.cube.shape[0])

    def frame_synthesis(self, frame_index: int = 0):
        """Expose a simulated frame and its recorded axes without resynthesis."""
        from .synthesis.assembly import SynthesisResult

        frame = self.cube[frame_index]
        tx, rx, slow, fast = frame.shape
        packed = frame.permute(2, 1, 0, 3).reshape(slow, rx * tx, fast)
        return SynthesisResult(
            cube=packed,
            kind=self.kind,
            axes=(self.axes[-2], "sensor_pair", self.axes[-1]),
            phasor=self.phasor,
            time_dependence=self.time_dependence,
            reference_frequency_hz=self.reference_frequency_hz,
            output_domain=self.output_domain,
        )

    @classmethod
    def from_frames(
        cls,
        cubes,
        *,
        times_s,
        synthesis,
        epochs,
        rediscovery_reasons,
        compile_count: int,
        discovery_count: int,
        last_snapshot: object,
        last_compiled_scene: object,
        last_propagation: object,
        last_radar_paths: object,
        sample_times_s=(),
        path_set_complete: bool = True,
        motion_sampling_exhaustive: bool = True,
        motion_sampling: str = "static",
        adaptive_diagnostics=(),
    ) -> RadarSimulationResult:
        """Stack the per-frame cubes and carry the waveform's conventions.

        ``synthesis`` is the LAST frame's
        :class:`~witwin.radar.synthesis.assembly.SynthesisResult`. Its
        conventions are properties of the waveform spec, which is the radar's
        stored configuration and therefore the same for every frame; taking them
        from one frame rather than re-deriving them is what keeps this result
        from becoming a second owner of the phasor convention.
        """

        stacked = torch.stack(tuple(cubes), dim=0)
        return cls(
            cube=stacked,
            times_s=tuple(float(value) for value in times_s),
            kind=synthesis.kind,
            axes=(SIMULATION_CUBE_LEADING_AXES + (synthesis.axes[0], synthesis.axes[2])),
            phasor=synthesis.phasor,
            time_dependence=synthesis.time_dependence,
            reference_frequency_hz=float(synthesis.reference_frequency_hz),
            epochs=tuple(int(value) for value in epochs),
            rediscovery_reasons=tuple(rediscovery_reasons),
            compile_count=int(compile_count),
            discovery_count=int(discovery_count),
            last_snapshot=last_snapshot,
            last_compiled_scene=last_compiled_scene,
            last_propagation=last_propagation,
            last_radar_paths=last_radar_paths,
            sample_times_s=tuple(tuple(float(t) for t in frame) for frame in sample_times_s),
            path_set_complete=path_set_complete,
            motion_sampling_exhaustive=motion_sampling_exhaustive,
            motion_sampling=motion_sampling,
            output_domain=synthesis.output_domain,
            adaptive_diagnostics=tuple(adaptive_diagnostics),
        )


@dataclass(frozen=True, slots=True, eq=False)
class _SceneFrame:
    """One produced frame and the session state as of that frame.

    The unit a scene session yields, so that stacking the sequence and
    streaming it read the same record. Everything a one-frame
    :class:`RadarSimulationResult` needs is here; nothing that only makes sense
    for a whole run is. ``diagnostics`` is ``None`` for a route that does not
    adapt, which is what distinguishes "this route publishes no probe record"
    from "the probes found nothing".

    The tensor members ALIAS the frame's own device storage. This record is a
    hand-off, not a retention point.
    """

    cube: torch.Tensor
    synthesis: object
    time_s: float
    sample_times_s: tuple[float, ...]
    epoch: int
    reason: str | None
    path_set_complete: bool
    motion_sampling_exhaustive: bool
    compile_count: int
    discovery_count: int
    epoch_frame: object
    legs: object
    composed: object
    diagnostics: dict | None
    motion_sampling: str


def _assemble(frames: list[_SceneFrame]) -> RadarSimulationResult:
    """Stack one or more session frames into the published result.

    The run-level completeness statements are conjunctions: one frame that
    could have missed a path birth makes the sequence one that could have
    missed a path birth.
    """

    last = frames[-1]
    return RadarSimulationResult.from_frames(
        [frame.cube for frame in frames],
        times_s=[frame.time_s for frame in frames],
        synthesis=last.synthesis,
        epochs=[frame.epoch for frame in frames],
        rediscovery_reasons=[frame.reason for frame in frames],
        compile_count=last.compile_count,
        discovery_count=last.discovery_count,
        last_snapshot=last.epoch_frame.snapshot,
        last_compiled_scene=last.epoch_frame.compiled,
        last_propagation=last.legs,
        last_radar_paths=last.composed,
        sample_times_s=[frame.sample_times_s for frame in frames],
        path_set_complete=all(frame.path_set_complete for frame in frames),
        motion_sampling_exhaustive=all(frame.motion_sampling_exhaustive for frame in frames),
        motion_sampling=last.motion_sampling,
        adaptive_diagnostics=[frame.diagnostics for frame in frames if frame.diagnostics is not None],
    )


def _dynamic_scene(scene: object) -> object:
    """A ``DynamicScene`` for whichever of the two Core worlds was passed.

    A static ``Scene`` is wrapped rather than refused: it IS a dynamic scene
    with no declared motion, the loop then reports ``structures_move = False``,
    and the compiled scene is built exactly once for the whole run. Refusing it
    would force every caller with a still world to write the wrapper themselves.
    """

    if all(hasattr(scene, name) for name in ("at", "structure_trajectories", "structure_deformations")):
        return scene
    from witwin.core.dynamics import DynamicScene

    return DynamicScene(scene)


def _times(times: object) -> tuple[float, ...]:
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


@dataclass(frozen=True, slots=True, eq=False)
class _AdaptiveTable:
    """Every row the adaptive refinement evaluated, and how to read them.

    One row table for the whole frame, plus the host-authored integer maps that
    say which evaluated observations each observation interpolates through.
    ``node_index[i]`` names ``node_count`` evaluated observations and
    ``basis[i]`` their weights, which sum to one; an observation that was
    evaluated itself appears as K copies of itself with the first weight one,
    the same exact answer on a rectangular table without a degenerate basis.

    The tensors ALIAS the frame's own device storage. This is a hand-off, not a
    retention point, except when :func:`trace_scene` keeps it on purpose.
    """

    delays: torch.Tensor
    transfers: torch.Tensor
    validity: torch.Tensor
    starts: object
    counts: object
    node_index: object
    basis: object
    clock: object
    device: object
    node_count: int
    observation_count: int
    pairs: int

    @property
    def row_count(self) -> int:
        return int(self.delays.shape[0])


@dataclass(frozen=True, slots=True, eq=False)
class _Observation:
    """One evaluated instant: the world it saw and the rows it composed."""

    time_s: float
    epoch_frame: object
    legs: object
    composed: object


@dataclass(frozen=True, slots=True, eq=False)
class _FrameTrace:
    """One frame's composed rows, before any instrument stage has run.

    ``observations`` is an ITERATOR and is drained exactly once, in order. The
    fused route drains it as it synthesizes, so it never holds more than one
    observation; :func:`trace_scene` materialises it instead.

    An adaptive frame yields a single observation, the one that closed it,
    because its rows live in ``adaptive`` rather than one batch per instant.
    """

    time_s: float
    sample_times_s: tuple[float, ...]
    observations: object
    path_set_complete: bool
    motion_sampling_exhaustive: bool
    adaptive: _AdaptiveTable | None = None
    stats: dict | None = None
    opened: object = None


@dataclass(frozen=True, slots=True, eq=False)
class _Session:
    """What the instrument half needs that does not change between frames.

    Built once by :func:`_open_session` from the radar and the resolved motion,
    and read by :func:`_echo_frame`. Nothing here describes the world, which is
    the property that lets one traced session be echoed against a different
    receive chain.
    """

    radar: object
    array: object
    loop: object
    mode: object
    full_spec: object
    output_spec: object
    single_spec: object
    offsets: tuple[float, ...]
    pair_samples: object
    sampled: bool
    adc_sampled: bool
    samples_per_slot: int
    path_phase_noise: bool
    adaptive: AdaptiveMotionSpec
    reference_frequency_hz: float
    #: Derives the declared, working and single-observation specs for a radar.
    #: A callable rather than three stored specs because :func:`echo_paths` has
    #: to re-derive them for whatever receive chain it was handed.
    instrument: object
    kind: str
    waveform: object
    carrier: float
    components: frozenset
    max_depth: int
    grad: str
    motion: Motion
    times: tuple[float, ...]


def _lagrange_weights(query_s, node_s):
    """The Lagrange basis at ``query_s`` for each row's node instants.

    ``query_s`` is ``[rows]`` and ``node_s`` is ``[rows, K]``, both absolute
    times in seconds; the return is ``[rows, K]``, dimensionless. Row ``i``
    holds the unique degree-``K-1`` polynomial basis through that row's own
    nodes, so ``sum_j w_ij = 1`` identically and a query landing on a node
    gives that node weight one. Rows are independent: the adaptive partition
    gives different observations different intervals.

    Validity: the nodes of a row must be DISTINCT, which the caller guarantees
    by enumerating any interval too short to carry ``K`` separate instants. A
    repeated node divides by zero here rather than silently producing a
    plausible weight. Node spacing need not be uniform, and is not: ADC
    instants cluster inside a chirp and jump across the idle gap.

    Oracle: ``tests/test_path_interpolation.py`` checks the basis and the
    interpolant it drives against an independent numpy/complex formulation.
    """

    import numpy as np

    weights = np.ones(node_s.shape)
    for near in range(node_s.shape[1]):
        for far in range(node_s.shape[1]):
            if far != near:
                weights[:, near] *= (query_s - node_s[:, far]) / (node_s[:, near] - node_s[:, far])
    return weights


def _adaptive_trace(times, evaluate_many, spec, options, carrier_hz):
    """Control topology probes on the host and publish the accepted partition.

    The world half of the adaptive route. The grid's interior tests bound
    OBSERVED errors only: arbitrarily brief path births or adversarial
    oscillations between probes require the ADC reference, and the two
    completeness statements this returns are what say so. Host copies here are
    explicit adaptive decisions, not a differentiable physics implementation.
    AD follows the accepted, fixed partition.

    :func:`_adaptive_echo` consumes what this returns and is where every kernel
    launch happens.
    """
    import numpy as np

    # The refinement tests its probes against the same interpolant the echo
    # half will use on the accepted partition. That is deliberate: a tolerance
    # measured with one rule and spent by another bounds nothing.
    from .paths import interpolate_path_rows

    cache, partitions, pair_tables = {}, {}, {}
    stats = {
        "evaluations": 0,
        "topology_refinements": 0,
        "accepted_intervals": 0,
        "max_tested_phase_error_rad": 0.0,
        "max_tested_relative_amplitude_error": 0.0,
        "max_interval_s": options.max_interval_s,
        "exhaustive": False,
        "topology_proved_complete": False,
    }

    def ensure(indices):
        missing = sorted(set(indices) - cache.keys())
        if len(cache) + len(missing) > options.max_evaluations:
            raise RuntimeError("adaptive motion discovery budget exhausted before satisfying error/topology tests")
        for begin in range(0, len(missing), options.batch_observations):
            batch = missing[begin : begin + options.batch_observations]
            records = evaluate_many([times[index] for index in batch])
            for index, record in zip(batch, records, strict=True):
                cache[index] = record
                pair_tables[index] = record[2].pair_offsets.tolist()

    # The maximum interval is the MAXIMUM probe spacing, and it is
    # unconditional. Lowering it refines the grid.
    # Every error test here is a sampled test: it can only see motion at the
    # instants it probes. A sinusoid whose period divides the grid spacing sits
    # at a zero of every node and every test point, so the controller measures
    # no error at all while the interpolant misses the whole oscillation. This
    # bound is the only thing that sets how fine that grid is, which is why it
    # is not skipped for a topologically certified family: the certification
    # says no path can be BORN, and says nothing about how fast one moves.
    def within_interval_bound(left, right):
        return times[right] - times[left] <= options.max_interval_s

    # Start from the coarsest partition the bound allows rather than bisecting
    # down to it: the intervening levels are guaranteed to fail that bound, and
    # their probes answer a question already decided here.
    span = times[-1] - times[0]
    pieces = max(1, math.ceil(span / options.max_interval_s))
    edges = sorted({round(index * (len(times) - 1) / pieces) for index in range(pieces + 1)})
    pending = list(zip(edges, edges[1:], strict=False))

    # A higher polynomial order buys a LONGER accepted interval, so it pays only
    # where the phase test, not the bound above, is what shortens one. Where the
    # bound decides the length the order cannot reduce the interval count and
    # multiplies the per-interval grid instead: on the three-wall fixture nodes
    # 2/3/5/9 all accept nine intervals for the same 6e-4 error. The caller owns
    # that trade because only the caller knows which regime the motion is in.
    node_count = options.interpolation_nodes
    # One candidate interval spends `node_count` instants on the interpolation
    # and tests the error at the instants between them, so its probe grid is
    # this wide and the even positions are the nodes.
    grid_width = 2 * (node_count - 1) + 1
    stats["interpolation_nodes"] = node_count

    while pending:
        probes, enumerated = {}, []
        for left, right in pending:
            if right - left < grid_width:
                # Too short to carry distinct nodes. Enumerating it costs no
                # more probes than testing it would, and an enumerated
                # observation needs no interpolation at all.
                enumerated.append((left, right))
            else:
                probes[left, right] = [
                    left + round((right - left) * step / (grid_width - 1)) for step in range(grid_width)
                ]
        ensure(
            [index for group in probes.values() for index in group]
            + [index for left, right in enumerated for index in range(left, right + 1)]
        )
        stats["accepted_intervals"] += len(enumerated)
        tested = {
            (left, right): indices
            for (left, right), indices in probes.items()
            if within_interval_bound(left, right) and all(cache[index][3] == cache[left][3] for index in indices)
        }
        queries = [
            (left, right, index, tuple(indices[::2]))
            for (left, right), indices in tested.items()
            for index in indices[1::2]
        ]
        observations = {}
        if queries:
            actual = [cache[index][2] for _, _, index, _ in queries]
            counts = [len(value.total_delay_s) for value in actual]
            node_paths = [[cache[group[node]][2] for *_, group in queries] for node in range(len(queries[0][3]))]
            basis = _lagrange_weights(
                np.asarray([times[index] for _, _, index, _ in queries]),
                np.asarray([[times[index] for index in group] for *_, group in queries]),
            )
            device = node_paths[0][0].total_delay_s.device
            delay, transfer = interpolate_path_rows(
                [torch.cat([value.total_delay_s for value in column]) for column in node_paths],
                [torch.cat([value.complex_transfer_ref for value in column]) for column in node_paths],
                torch.as_tensor(np.repeat(basis, counts, axis=0), device=device),
                carrier_hz,
            )
            observed = torch.cat([value.complex_transfer_ref for value in actual])
            # One synchronization per refinement round. These detached columns
            # drive host decisions only; production rows retain their AD tape.
            columns = (
                torch.stack(
                    [
                        delay.double(),
                        torch.cat([value.total_delay_s for value in actual]).double(),
                        transfer.real.double(),
                        transfer.imag.double(),
                        observed.real.double(),
                        observed.imag.double(),
                        torch.cat([value.row_valid for value in actual]).double(),
                    ],
                    dim=1,
                )
                .detach()
                .cpu()
                .numpy()
            )
            begin = 0
            for query, count in zip(queries, counts, strict=True):
                observations[query] = columns[begin : begin + count].T
                begin += count
        next_pending = []
        for (left, right), indices in probes.items():
            identities = [cache[index][3] for index in indices]
            topology_ok = all(key == identities[0] for key in identities)
            phase_error = amplitude_error = 0.0
            accepted = False
            if topology_ok and within_interval_bound(left, right):
                for index in indices[1::2]:
                    predicted_delay, observed_delay, pr, pi, ar, ai, flags = observations[
                        left, right, index, tuple(indices[::2])
                    ]
                    d = predicted_delay - observed_delay
                    predicted = pr + 1j * pi
                    observed = ar + 1j * ai
                    live = flags.astype(bool)
                    if not np.any(live):
                        continue
                    if not all(np.isfinite(value[live]).all() for value in (d, predicted, observed)):
                        raise RuntimeError("nonfinite adaptive path probe; no error bound can be accepted")
                    magnitude = np.maximum(np.abs(predicted), np.abs(observed))
                    floor = max(float(magnitude.max()) * 1e-6, 1e-30)
                    phase_live = live & (magnitude > floor)
                    waveform_frequency = abs(spec.slope_hz_per_s) * (
                        abs(spec.t_start_s)
                        + (spec.num_samples - 1) * spec.sample_period_s
                        + float(
                            np.max(np.maximum(np.abs(predicted_delay[live]), np.abs(observed_delay[live])), initial=0)
                        )
                    )
                    phase_error = max(
                        phase_error,
                        float(np.max(2 * np.pi * carrier_hz * np.abs(d[live]), initial=0)),
                        float(
                            np.max(
                                np.abs(np.angle(predicted[phase_live] * observed[phase_live].conj()))
                                + 2 * np.pi * waveform_frequency * np.abs(d[phase_live]),
                                initial=0,
                            )
                        ),
                    )
                    amplitude_error = max(
                        amplitude_error,
                        float(
                            np.max(
                                np.abs(np.abs(predicted[live]) - np.abs(observed[live]))
                                / np.maximum(magnitude[live], floor),
                                initial=0,
                            )
                        ),
                    )
                accepted = (
                    phase_error <= options.phase_error_rad and amplitude_error <= options.relative_amplitude_error
                )
            if accepted:
                stats["accepted_intervals"] += 1
                stats["max_tested_phase_error_rad"] = max(stats["max_tested_phase_error_rad"], phase_error)
                stats["max_tested_relative_amplitude_error"] = max(
                    stats["max_tested_relative_amplitude_error"], amplitude_error
                )
                for index in range(left, right + 1):
                    partitions[index] = tuple(indices[::2])
            else:
                stats["topology_refinements"] += int(not topology_ok)
                middle = (left + right) // 2
                next_pending.extend([(left, middle), (middle, right)])
        pending = next_pending

    # Store each discovered row once. ADC scheduling gathers these differentiable
    # columns with host-authored integer maps; it never allocates one tensor per
    # observation. Ragged row counts and empty sensor pairs remain explicit.
    ordered = sorted(cache)
    records = [cache[index][2] for index in ordered]
    device = records[0].total_delay_s.device
    delays = torch.cat([record.total_delay_s for record in records])
    transfers = torch.cat([record.complex_transfer_ref for record in records])
    validity = torch.cat([record.row_valid for record in records])
    starts = np.zeros(len(times), dtype=np.int64)
    starts[ordered] = np.cumsum([0] + [len(record.total_delay_s) for record in records[:-1]])
    pairs = spec.num_tx * spec.num_rx
    counts = np.zeros((len(times), pairs), dtype=np.int64)
    counts[ordered] = np.diff(np.asarray([pair_tables[index] for index in ordered]), axis=1)
    # An observation reads either its own evaluated row or its interval's
    # nodes. An evaluated observation is written as K copies of itself with the
    # first weight one, which is the same exact answer and keeps the table
    # rectangular without a degenerate basis.
    node_index = np.zeros((len(times), node_count), dtype=np.int64)
    for index, group in partitions.items():
        node_index[index] = group
    node_index[ordered] = np.asarray(ordered)[:, None]
    clock = np.asarray(times)
    basis = np.zeros((len(times), node_count))
    basis[:, 0] = 1.0
    interpolated = np.ones(len(times), dtype=bool)
    interpolated[ordered] = False
    if interpolated.any():
        basis[interpolated] = _lagrange_weights(clock[interpolated], clock[node_index[interpolated]])

    stats["evaluations"] = len(cache)
    stats["exhaustive"] = len(cache) == len(times)
    # Two independent reasons no path birth can have been missed: every
    # observation was evaluated, or the family was certified complete for all
    # time. Neither implies the other.
    #
    # The certification is a property of the world and the propagation
    # configuration, not of an instant, so one probe carrying it proves the
    # family cannot change. The first probe of a run cannot carry it - nothing
    # is frozen yet - but it rediscovers, which enumerates the family at its
    # own instant. Those two together cover the frame. A probe that neither
    # certifies nor rediscovers leaves a gap and is refused. A source mutation
    # strictly between two probes remains outside every sampled test, as
    # AdaptiveMotionSpec states.
    frames = [cache[index][0] for index in cache]
    stats["topology_proved_complete"] = any(frame.topology_complete for frame in frames) and all(
        frame.topology_complete or frame.rediscovered for frame in frames
    )
    stats["observation_count"] = len(times)
    table = _AdaptiveTable(
        delays=delays,
        transfers=transfers,
        validity=validity,
        starts=starts,
        counts=counts,
        node_index=node_index,
        basis=basis,
        clock=clock,
        device=device,
        node_count=node_count,
        observation_count=len(times),
        pairs=pairs,
    )
    return table, stats, cache[0], cache[len(times) - 1]


def _adaptive_echo(table, spec, options, carrier_hz, frontend):
    """Interpolate the accepted partition and synthesize it, in bounded batches.

    The instrument half of the adaptive route: every kernel launch of that
    route is here and nothing here decides anything about the world. The
    partition is fixed by the time this runs, which is what makes the native
    interpolant's VJP and JVP well defined.

    Returns the frame's beat cube and how many batches it took, which is an
    echo quantity: a traced frame that has never been echoed has no batch count
    to report.
    """

    import numpy as np

    from .paths import interpolate_path_rows
    from .synthesis.fmcw import _synthesize_fmcw_observations, channel_phasor_to_beat_weight

    delays, transfers, validity = table.delays, table.transfers, table.validity
    starts, counts, node_index, basis, clock = table.starts, table.counts, table.node_index, table.basis, table.clock
    node_count, pairs, device = table.node_count, table.pairs, table.device
    observations_total = table.observation_count

    def upload(value):
        return torch.as_tensor(value, device=device)

    values = []
    left = node_index[:, 0]
    row_counts = counts[left].sum(axis=1)
    # The row bound caps the temporary node table, whose width is 4 columns per
    # node. Stated against the two-node width so that raising the polynomial
    # order does not silently raise peak allocation along with it.
    row_budget = 262144 * 8 // (4 * node_count)
    cumulative = np.concatenate(([0], np.cumsum(row_counts)))
    begin = 0
    while begin < observations_total:
        # Bound temporary expanded payloads by both observations and path rows.
        # One unusually large observation is indivisible and is still supported.
        stop = min(observations_total, begin + options.batch_observations * spec.num_samples)
        stop = min(stop, max(begin + 1, int(np.searchsorted(cumulative, cumulative[begin] + row_budget)) - 1))
        batch = np.arange(begin, stop)
        rows = row_counts[batch]
        offsets = np.concatenate(([0], np.cumsum(counts[left[batch]].ravel())))
        observation = np.repeat(batch, rows)
        local_row = np.arange(offsets[-1]) - np.repeat(np.cumsum(rows) - rows, rows)
        node_rows = [upload(starts[node_index[observation, node]] + local_row) for node in range(node_count)]
        delay, transfer = interpolate_path_rows(
            [delays[rows] for rows in node_rows],
            [transfers[rows] for rows in node_rows],
            upload(basis[observation]),
            carrier_hz,
        )
        transfer = torch.where(validity[node_rows[0]], transfer, torch.zeros_like(transfer))
        if frontend is not None:
            transfer = frontend._apply_path_phase_rows(delay, transfer, upload(clock[observation]))
        adc_time = spec.t_start_s + (observation % spec.num_samples) * spec.sample_period_s
        values.append(
            _synthesize_fmcw_observations(
                delay, channel_phasor_to_beat_weight(transfer), upload(offsets), upload(adc_time), spec
            ).reshape(len(batch), pairs)
        )
        begin = stop
    all_slots = torch.cat(values).reshape(-1, spec.num_samples, pairs).transpose(1, 2)
    pair = torch.arange(pairs, device=all_slots.device)
    slot = torch.arange(spec.num_chirps, device=all_slots.device)[:, None] * spec.num_tx + pair[None, :] % spec.num_tx
    return all_slots[slot, pair], len(values)


def _open_session(
    radar: object,
    scene: object,
    *,
    times,
    response: object,
    sites: object = None,
    components: frozenset[str] | None = None,
    max_depth: int | None = None,
    ad_mode: str = "none",
    ids: object = None,
    polarization: object = None,
    antenna_pattern: object = None,
    sensor_endpoints: SensorEndpointIds | None = None,
    motion: Motion | None = None,
) -> Iterator[_SceneFrame]:
    """Run ``radar`` over ``scene`` at ``times``, yielding one frame at a time.

    The single owner of the scene session. :func:`simulate_scene` and
    :func:`stream_scene` differ only in how much of this they retain, so the
    session setup, the epoch loop and the synthesis route are not written twice.
    Argument validation happens on the FIRST iteration, as it does for any
    generator; ``simulate_scene`` consumes immediately and therefore still
    raises from its own call.

    This is the whole of :meth:`witwin.radar.Radar.simulate`; the method is a
    delegation so that the assembly lives next to the contracts it assembles
    rather than inside the radar's own configuration and pose module.

    ``sites`` is a :class:`ScatterSitePolicy` and
    defaults to ``ScatterSitePolicy.structure_anchor()`` - one site at every
    Core-owned structure world anchor. A structure with no rigid motion has no
    such anchor and the policy refuses it by name; that refusal is the design
    (R-ADR-020), because the alternative is a mesh-sampling rule, which is a
    geometry algorithm and does not belong in a Torch expression here.

    ``components`` and ``max_depth`` override the radar's propagation block for
    THIS call through
    :meth:`~witwin.radar.radar.RadarSystemConfig.with_propagation`, which
    returns a new configuration rather than mutating the radar's stored one.

    Dynamic scenes refresh each ADC observation by default; ``motion_sampling``
    can explicitly select ``adaptive`` with :class:`AdaptiveMotionSpec`, or a
    chirp-frozen approximation. Adaptive FMCW batches topology-identical probes,
    interpolates carrier-transported coefficients, and refines at observed path
    identity/validity changes. Its sampled error tests cannot exclude arbitrarily
    brief unseen events between probes, so ``path_set_complete`` is true only
    when every observation was evaluated or the candidate family was certified
    complete for all time; ``motion_sampling_exhaustive`` separately records
    whether any observation was interpolated. Complete discovery is
    the default at every observation. A longer ``motion_event_period_frames``
    is converted from frames to observation count and marks path completeness
    false unless structure motion already forces discovery. The selected
    ``world_motion`` still controls whether compiled handles may be replayed.

    ``ad_mode`` is forwarded to every replay. ``"none"`` is the default and
    builds no graph; ``"vjp"`` makes the published cube differentiable with
    respect to the endpoint and site positions the binding passed through by
    identity.

    ``antenna_pattern`` is an
    :class:`~witwin.radar.sensors.AntennaPatternSpec` and defaults to
    ``None``, which applies no pattern and launches no extra kernel. It does NOT
    default to ``radar.system_config.sensors``: that spec falls back to
    a half-wave dipole, so adopting it here would attenuate every result by a
    number nobody chose. Pass that spec to use it,
    :data:`~witwin.radar.sensors.ISOTROPIC_PATTERN` to run the stage
    as a proven no-op, or leave it ``None``.
    """

    from .channel import ChannelPropagationAdapter, compile_scene
    from .paths import TwoWayComposer, validate_pair_ordering
    from .propagation import FrozenEpoch, RadarPropagationLegs, SceneEpochLoop
    from .sensors import RoundTripPatternStage

    requested = Motion.auto() if motion is None else motion
    if not isinstance(requested, Motion):
        raise TypeError(f"motion must be a Motion, got {type(requested).__name__}")
    adaptive = requested._adaptive_spec()
    world_motion = requested.world
    motion_event_period_frames = requested.rediscover_every_frames
    instants = _times(times)
    policy = ScatterSitePolicy.structure_anchor() if sites is None else sites
    if not isinstance(policy, ScatterSitePolicy):
        raise TypeError(
            f"sites must be a ScatterSitePolicy, got {type(policy).__name__}; "
            "where the scatter sites come from is a declaration, not a search"
        )
    orientation = DEFAULT_POLARIZATION if polarization is None else polarization

    solve_config = radar.system_config.with_propagation(components=components, max_depth=max_depth)
    propagation = solve_config.propagation
    array = solve_config.sensors.array
    reference_frequency_hz = propagation.reference_frequency_hz

    def bind(compiled, snapshot, previous):
        binding = bind_radar_world(
            radar, snapshot, sites=policy, ids=ids, polarization=orientation, sensor_endpoints=sensor_endpoints
        )
        adapter = (
            ChannelPropagationAdapter(
                compiled,
                reference_frequency_hz=reference_frequency_hz,
                components=propagation.components,
                max_depth=propagation.max_depth,
            )
            if previous is None
            else previous.adapter
        )
        inbound = adapter.freeze(binding.transmitters, binding.site_sinks)
        outbound = adapter.freeze(binding.site_sources, binding.receivers)
        composer = TwoWayComposer.freeze(
            inbound,
            outbound,
            torch.tensor(binding.site_ids, dtype=torch.int64, device=binding.device),
            radar_source_ids=list(binding.transmitter_ids),
            radar_sink_ids=list(binding.receiver_ids),
            reference_frequency_hz=reference_frequency_hz,
        )
        # Once per topology epoch, where the topology is decided and the host
        # read is free. It must never move into the frame loop: the same read
        # there is a per-frame device-to-host transfer.
        validate_pair_ordering(
            composer.sensor_pair_index,
            num_tx=array.num_tx,
            num_rx=array.num_rx,
            sensor_pair_count=composer.sensor_pair_count,
        )
        # The pattern tables are a property of the frozen join - which pair each
        # row belongs to and which site it visits - so they are built here, once
        # per epoch, and the frame loop only gathers positions and launches.
        stage = (
            None
            if antenna_pattern is None
            else RoundTripPatternStage.freeze(radar, composer, site_ids=binding.site_ids, pattern=antenna_pattern)
        )
        # The binding travels with the epoch so the frame that just froze does
        # not build a second one from the same snapshot. It is deterministic, so
        # the two would agree - which is exactly why building both is waste.
        return FrozenEpoch(adapter=adapter, handles=(inbound, outbound), payload=(composer, binding, stage))

    dynamic = _dynamic_scene(scene)
    if dynamic.endpoint_trajectories and sensor_endpoints is None:
        raise ValueError("moving Core endpoints require explicit sensor_endpoints in array order")
    sampled = bool(
        dynamic.structure_trajectories
        or dynamic.structure_deformations
        or dynamic.endpoint_trajectories
        or policy.trajectory is not None
        or callable(getattr(response, "at", None))
    )
    from .synthesis.assembly import FmcwSpec, SlowTimeMode, waveform_sampling

    mode = SlowTimeMode.FROZEN_WEIGHT_WITH_CARRIER_RATE
    full_spec = solve_config.waveform_spec()
    path_phase_noise = radar.frontend is not None and radar.frontend.has_phase_noise
    if path_phase_noise:
        if not isinstance(full_spec, FmcwSpec):
            raise NotImplementedError("scene-driven common-oscillator phase noise currently requires FMCW")
        # Checked against the REQUEST, because the kind is not resolved yet:
        # this branch is one of the things that decides it.
        if requested.kind not in ("auto", "adc", "adaptive"):
            raise ValueError("common-oscillator phase noise requires ADC-time observations")
        sampled = True
    # ``auto`` resolves HERE rather than at the top, because whether anything
    # moves is a property of the session and oscillator phase noise has just
    # made a still world a sampled one. ``static`` is refused rather than
    # silently upgraded: it asks for one observation a frame, and a world that
    # moves inside a frame would publish no Doppler at all.
    if requested.kind == "auto":
        motion_sampling = "adc" if sampled else "static"
    else:
        motion_sampling = requested.kind
    if motion_sampling == "static" and sampled:
        raise ValueError(
            "Motion.static() asks for one observation per frame, but this session moves within a frame; "
            "use Motion.adc() for the exhaustive reference, Motion.chirp() for stop-and-hop, or "
            "Motion.adaptive() for error-controlled interpolation"
        )
    if motion_sampling == "static":
        # Nothing moves, so every sampling kind evaluates the same single
        # observation per frame; ``adc`` is the name the rest of this function
        # already spells that with.
        motion_sampling = "adc"

    output_spec = full_spec
    # Receiver nonlinearity and oscillator noise act on ADC-time samples.
    # Preserve the direct-spectrum fast path only for an ideal receiver.
    if isinstance(full_spec, FmcwSpec) and radar.frontend is not None:
        full_spec = replace(full_spec, output_domain="beat")
    if motion_sampling == "adaptive" and not isinstance(full_spec, FmcwSpec):
        raise NotImplementedError("adaptive motion currently requires FMCW")
    if motion_sampling == "adaptive" and motion_event_period_frames is not None:
        raise ValueError("adaptive motion owns its topology discovery cadence")
    adc_sampled = sampled and motion_sampling in ("adc", "adaptive") and isinstance(full_spec, FmcwSpec)
    samples_per_slot = full_spec.num_samples if adc_sampled else 1

    if sampled:
        offsets, pair_samples, single_spec = waveform_sampling(
            full_spec, num_tx=array.num_tx, num_rx=array.num_rx, device=radar.device
        )
        if adc_sampled:
            if (
                full_spec.t_start_s + (full_spec.num_samples - 1) * full_spec.sample_period_s
                >= full_spec.chirp_period_s
            ):
                raise ValueError("ADC observations must fit inside each chirp period")
            offsets = tuple(
                slot + full_spec.t_start_s + m * full_spec.sample_period_s
                for slot in offsets
                for m in range(full_spec.num_samples)
            )
            single_spec = replace(single_spec, num_samples=1, output_domain="beat")
        mode = SlowTimeMode.REFRESHED_WEIGHT_NO_RATE
        # A version poll cannot detect endpoint-induced path births. Complete
        # dynamic sampling therefore rediscovers at each observed instant.
        cadence = 1 if motion_event_period_frames is None else motion_event_period_frames * len(offsets)
    else:
        offsets, pair_samples, single_spec = (0.0,), None, None
        cadence = motion_event_period_frames
    if sampled and any(b <= a + offsets[-1] for a, b in zip(instants, instants[1:], strict=False)):
        raise ValueError("dynamic frames must not overlap in observation time")
    loop = SceneEpochLoop(
        dynamic,
        reference_frequency_hz=reference_frequency_hz,
        bind=bind,
        compile_scene=compile_scene,
        motion_event_period_frames=cadence,
        world_motion=world_motion,
    )

    identity_cache = {}
    slot_composers = {}
    slot_patterns = {}

    def evaluate_many(query_times):
        """Discover probes, then batch replay only identical live scene handles.

        Moving compiled geometry is replayed before the next refreeze retires
        its handles. Static geometry permits one slot batch per full leg key.
        Host identity observations belong to adaptive control, not a replay.
        """
        from .paths import leg_identity

        results = {}
        groups = {}

        def finish(group):
            first = group[0][1].frozen
            bindings = [entry[2] for entry in group]

            def endpoints(name):
                specs = [getattr(binding, name) for binding in bindings]
                if len(specs) == 1:
                    return specs[0]
                return RadarEndpointSpec(
                    **{
                        field: None
                        if getattr(specs[0], field) is None
                        else torch.cat([getattr(spec, field) for spec in specs], dim=0)
                        for field in ("stable_ids", "positions_m", "polarizations", "powers_w")
                    }
                )

            replay = RadarPropagationLegs(
                inbound=first.adapter.reevaluate_slots(
                    first.handles[0],
                    endpoints("transmitters"),
                    endpoints("site_sinks"),
                    slot_count=len(group),
                    ad_mode=ad_mode,
                ),
                outbound=first.adapter.reevaluate_slots(
                    first.handles[1],
                    endpoints("site_sources"),
                    endpoints("receivers"),
                    slot_count=len(group),
                    ad_mode=ad_mode,
                ),
            )
            composer, _, pattern_stage = first.payload
            batched_paths = None
            if len(group) > 1 and not callable(getattr(response, "at", None)):
                key = (group[0][3], len(group))
                if key not in slot_composers:
                    slot_composers[key] = composer._for_slots(len(group))
                batched_paths = slot_composers[key].compose(
                    replay.inbound, replay.outbound, response, include_delay_rate=False
                )
                if pattern_stage is not None:
                    if key not in slot_patterns:
                        slot_patterns[key] = pattern_stage._for_slots(len(group))
                    batched_paths = slot_patterns[key].apply(
                        batched_paths,
                        tx_pos=endpoints("transmitters").positions_m,
                        rx_pos=endpoints("receivers").positions_m,
                        tx_targets_m=replay.inbound.departure_target_m.index_select(
                            0, batched_paths.topology.inbound_row
                        ),
                        rx_targets_m=replay.outbound.arrival_origin_m.index_select(
                            0, batched_paths.topology.outbound_row
                        ),
                    )
                valid_rows = batched_paths.row_valid.reshape(len(group), -1).tolist()
            for slot, (t, frame, binding, identity) in enumerate(group):
                legs = (
                    replay
                    if len(group) == 1
                    else RadarPropagationLegs(inbound=replay.inbound.slot(slot), outbound=replay.outbound.slot(slot))
                )
                composer, _, pattern_stage = frame.frozen.payload
                current_response = response.at(t) if callable(getattr(response, "at", None)) else response
                if batched_paths is None:
                    paths = composer.compose(legs.inbound, legs.outbound, current_response, include_delay_rate=False)
                else:
                    selection = slice(slot * composer.path_count, (slot + 1) * composer.path_count)
                    paths = replace(
                        batched_paths,
                        sensor_pair_count=composer.sensor_pair_count,
                        path_count=composer.path_count,
                        sensor_pair_index=composer.sensor_pair_index,
                        pair_offsets=composer.pair_offsets,
                        topology=composer.topology,
                        total_delay_s=batched_paths.total_delay_s[selection],
                        complex_transfer_ref=batched_paths.complex_transfer_ref[selection],
                        row_valid=batched_paths.row_valid[selection],
                        frequency_response=None
                        if batched_paths.frequency_response is None
                        else batched_paths.frequency_response[selection],
                    )
                if pattern_stage is not None and batched_paths is None:
                    paths = pattern_stage.apply(
                        paths,
                        tx_pos=binding.transmitters.positions_m,
                        rx_pos=binding.receivers.positions_m,
                        tx_targets_m=legs.inbound.departure_target_m.index_select(0, paths.topology.inbound_row),
                        rx_targets_m=legs.outbound.arrival_origin_m.index_select(0, paths.topology.outbound_row),
                    )
                if motion_sampling == "adaptive":
                    identity = (
                        identity,
                        tuple(paths.row_valid.tolist() if batched_paths is None else valid_rows[slot]),
                    )
                results[t] = (frame, legs, paths, identity)

        for t in query_times:
            frozen = loop.frozen
            authored_world = getattr(dynamic, "scene", None)
            # Only an empty authored world proves the absence of occluders and
            # reflection/diffraction births. Require every endpoint pair to be
            # present too: a degenerate initial LOS discovery cannot certify it.
            complete_los = (
                motion_sampling == "adaptive"
                and authored_world is not None
                and not authored_world.structures
                and propagation.components == frozenset({"los"})
                and frozen is not None
                and frozen.handles[0].row_count == array.num_tx * frozen.payload[1].site_count
                and frozen.handles[1].row_count == array.num_rx * frozen.payload[1].site_count
            )
            frame = loop.frame(t, topology_complete=complete_los)
            template = frame.frozen.payload[1]
            if frame.rediscovered:
                binding = template
            elif policy.source == SITE_SOURCE_EXPLICIT and sensor_endpoints is None:
                positions = policy.resolve(frame.snapshot, device=template.device)
                if positions.shape != template.site_positions_m.shape:
                    raise ValueError("a site trajectory must preserve the declared site count")
                binding = replace(
                    template,
                    site_sources=replace(template.site_sources, positions_m=positions),
                    site_sinks=replace(template.site_sinks, positions_m=positions),
                )
            else:
                binding = bind_radar_world(
                    radar,
                    frame.snapshot,
                    sites=policy,
                    ids=ids,
                    polarization=orientation,
                    sensor_endpoints=sensor_endpoints,
                )
            identity = ()
            if motion_sampling == "adaptive":
                if frame.epoch not in identity_cache:
                    identity_cache[frame.epoch] = tuple(
                        tuple(zip(*leg_identity(handle, "adaptive"), strict=True)) for handle in frame.frozen.handles
                    )
                identity = identity_cache[frame.epoch]
            entry = (t, frame, binding, identity)
            if loop.structures_move:
                finish([entry])
            else:
                groups.setdefault(identity, []).append(entry)
        for group in groups.values():
            finish(group)
        return [results[t] for t in query_times]

    # Whether a non-adaptive route can have missed a path birth is a property
    # of the declared cadence and world motion, not of any one frame, so it is
    # decided once here and repeated on every frame the run publishes.
    sampled_completeness = not sampled or cadence == 1 or world_motion == "frozen_world" and loop.structures_move
    published_sampling = "static" if not sampled else motion_sampling

    def instrument(other):
        """The spec pair this radar's instrument half works from.

        ONE owner: the setup here and :func:`echo_paths` against a different
        receive chain both derive the specs through this, so a receiver that
        flips the FMCW output domain cannot end up paired with a slot schedule
        derived from the other domain. The observation offsets are deliberately
        not re-derived: they are the traced schedule, and a spec that would
        change them is refused by name rather than silently re-scheduled.
        """

        declared = other.system_config.waveform_spec()
        working = declared
        if isinstance(declared, FmcwSpec) and other.frontend is not None:
            working = replace(declared, output_domain="beat")
        if not sampled:
            return declared, working, None
        _, _, single = waveform_sampling(working, num_tx=array.num_tx, num_rx=array.num_rx, device=other.device)
        if adc_sampled:
            single = replace(single, num_samples=1, output_domain="beat")
        return declared, working, single

    session = _Session(
        radar=radar,
        array=array,
        loop=loop,
        mode=mode,
        full_spec=full_spec,
        output_spec=output_spec,
        single_spec=single_spec,
        offsets=offsets,
        pair_samples=pair_samples,
        sampled=sampled,
        adc_sampled=adc_sampled,
        samples_per_slot=samples_per_slot,
        path_phase_noise=path_phase_noise,
        adaptive=adaptive,
        reference_frequency_hz=reference_frequency_hz,
        instrument=instrument,
        kind=published_sampling,
        waveform=radar.waveform,
        carrier=float(reference_frequency_hz),
        components=frozenset(propagation.components),
        max_depth=int(propagation.max_depth),
        grad=str(ad_mode),
        motion=requested,
        times=instants,
    )

    def observations(start):
        """One frame's evaluated observations, in schedule order.

        A generator rather than a list, and that is the whole reason the fused
        route's peak allocation did not change: it drops each observation once
        it has been synthesized, so nothing scales with the
        ``chirps * transmitters * samples`` observations an ADC-refreshed frame
        has. :func:`trace_scene` materialises the same generator and pays that
        cost deliberately, which is the trade its docstring states.

        Both consumers drain a frame completely before asking for the next.
        That is not politeness: these observations share the epoch loop's
        state, so interleaving two frames would replay one frame's legs against
        another frame's frozen topology.
        """

        for offset in offsets:
            time_s = start + offset
            epoch_frame, legs, composed, _ = evaluate_many([time_s])[0]
            yield _Observation(time_s=float(time_s), epoch_frame=epoch_frame, legs=legs, composed=composed)

    def traces():
        if sampled and motion_sampling == "adaptive":
            for start in instants:
                actual_times = tuple(start + offset for offset in offsets)
                table, stats, first, last = _adaptive_trace(
                    actual_times, evaluate_many, full_spec, adaptive, reference_frequency_hz
                )
                epoch_frame, legs, composed, _ = last
                closing = _Observation(
                    time_s=float(actual_times[-1]), epoch_frame=epoch_frame, legs=legs, composed=composed
                )
                yield _FrameTrace(
                    time_s=float(start),
                    sample_times_s=actual_times,
                    observations=iter((closing,)),
                    adaptive=table,
                    stats=stats,
                    opened=first[0],
                    path_set_complete=bool(stats["exhaustive"] or stats["topology_proved_complete"]),
                    motion_sampling_exhaustive=bool(stats["exhaustive"]),
                )
            return
        for start in instants:
            yield _FrameTrace(
                time_s=float(start),
                sample_times_s=tuple(start + offset for offset in offsets),
                observations=observations(start),
                path_set_complete=bool(sampled_completeness),
                motion_sampling_exhaustive=True,
            )

    return session, traces()


def _finish_frame(session: _Session, synthesis):
    """Apply the receive chain, then land the frame in its declared domain."""

    from .synthesis.assembly import FmcwSpec, SynthesisResult, assemble_frame_cube

    radar = session.radar
    frame_cube = radar._apply_signal_models(
        assemble_frame_cube(synthesis.cube, num_tx=session.array.num_tx, num_rx=session.array.num_rx),
        phase_in_signal=session.path_phase_noise,
    )
    output_spec = session.output_spec
    if isinstance(output_spec, FmcwSpec) and output_spec.output_domain != synthesis.output_domain:
        from .processing.range_doppler import fmcw_range_fft

        frame_cube = fmcw_range_fft(frame_cube)
        synthesis = SynthesisResult.from_fmcw(
            frame_cube.permute(2, 1, 0, 3).reshape(session.full_spec.num_chirps, -1, session.full_spec.num_samples),
            output_spec,
        )
    return frame_cube, synthesis


def _echo_frame(session: _Session, trace: _FrameTrace) -> _SceneFrame:
    """Synthesize one traced frame. The whole instrument half lives here.

    Everything this does is a property of the radar and the waveform, never of
    the world: the per-observation synthesis, the common-oscillator phase the
    receive chain applies per path before coherent summation, the slot gather,
    the receive chain itself and the output-domain transform. That is what lets
    a caller re-run it against a different receiver without re-tracing.

    ``trace.observations`` is drained exactly once, in order.
    """

    from .synthesis.assembly import SynthesisResult

    radar = session.radar
    full_spec = session.full_spec
    opened = None
    closing = None
    if trace.adaptive is not None:
        for observation in trace.observations:
            opened = trace.opened
            closing = observation
        cube, batches = _adaptive_echo(
            trace.adaptive, full_spec, session.adaptive, session.reference_frequency_hz, radar.frontend
        )
        # Written by the half that did the batching, so a Paths that has never
        # been echoed does not claim a synthesis count it could not have.
        trace.stats["synthesis_batches"] = batches
        synthesis = SynthesisResult.from_fmcw(cube, replace(full_spec, output_domain="beat"))
        composed = closing.composed
    else:
        slot_cubes = []
        synthesis = None
        composed = None
        for observation in trace.observations:
            if opened is None:
                opened = observation.epoch_frame
            closing = observation
            composed = observation.composed
            if session.path_phase_noise:
                composed = radar.frontend.apply_path_phase(composed, observation.time_s)
            observation_spec = (
                replace(
                    session.single_spec,
                    t_start_s=full_spec.t_start_s
                    + (len(slot_cubes) % session.samples_per_slot) * full_spec.sample_period_s,
                )
                if session.adc_sampled
                else session.single_spec
            )
            synthesis = radar._synthesize(
                composed, slow_time_mode=session.mode, spec=observation_spec if session.sampled else full_spec
            )
            slot_cubes.append(synthesis.cube)
        if session.sampled:
            stacked = torch.cat(slot_cubes, dim=0)
            pairs = torch.arange(session.array.num_tx * session.array.num_rx, device=stacked.device)
            if session.adc_sampled:
                stacked = stacked.reshape(-1, session.samples_per_slot, len(pairs)).transpose(1, 2)
                cube = stacked[session.pair_samples, pairs]
                if full_spec.output_domain == "spectrum":
                    from .processing.range_doppler import fmcw_range_fft

                    cube = fmcw_range_fft(cube)
                synthesis = SynthesisResult.from_fmcw(cube, full_spec)
            else:
                synthesis = replace(synthesis, cube=stacked[session.pair_samples, pairs])

    frame_cube, synthesis = _finish_frame(session, synthesis)
    loop = session.loop
    return _SceneFrame(
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
        composed=composed,
        diagnostics=trace.stats,
        motion_sampling=session.kind,
    )


def _scene_frames(*args, **kwargs) -> Iterator[_SceneFrame]:
    """Trace one frame, echo it, drop its rows, repeat.

    The fused route. It is a generator so that argument validation still
    happens when iteration starts rather than when the call returns, which is
    what :func:`stream_scene` promises.
    """

    session, traces = _open_session(*args, **kwargs)
    for trace in traces:
        yield _echo_frame(session, trace)


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
    observation as it synthesizes. Trace a frame at a time if the sequence is
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
    #: Every observation instant of every frame, s.
    sample_times: tuple[tuple[float, ...], ...]
    #: The propagation request these rows answer. ``path_set_complete`` is a
    #: statement relative to it, not an absolute one.
    los: bool
    reflections: int
    #: The differentiation mode the replay ran under.
    grad: str
    #: The resolved sampling record, after ``Motion.auto()`` chose.
    motion: Motion
    epochs: tuple[int, ...]
    rediscovery_reasons: tuple[str | None, ...]
    compile_count: int
    discovery_count: int
    path_set_complete: bool
    motion_sampling_exhaustive: bool
    adaptive_diagnostics: tuple[dict, ...]
    #: The closing observation's typed state, as the result publishes it.
    last_snapshot: object
    last_compiled_scene: object
    last_propagation: object
    last_radar_paths: object
    _session: object
    _frames: tuple

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
            total += frame.adaptive.observation_count if frame.adaptive is not None else len(frame.observations)
        return total

    @property
    def row_count(self) -> int:
        """Composed rows retained, summed over every evaluated observation."""

        total = 0
        for frame in self._frames:
            if frame.adaptive is not None:
                total += frame.adaptive.row_count
            else:
                total += sum(int(observation.composed.path_count) for observation in frame.observations)
        return total

    def frame(self, index: int) -> Paths:
        """One frame's rows, as a Paths of its own.

        The session travels with it, so the slice can be echoed. The compile
        and discovery counts are carried unchanged rather than recomputed,
        because slicing recompiled nothing.
        """

        frame_trace = self._frames[index]
        closing = frame_trace.observations[-1]
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

    def rows(self, frame: int = 0, observation: int = -1):
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
        return frame_trace.observations[observation].composed


def trace_scene(*args, **kwargs) -> Paths:
    """Run the world half of a session and keep every composed row.

    The same generator :func:`simulate_scene` consumes, drained into a record
    instead of synthesized. Read :class:`Paths` on what that costs.
    """

    session, traces = _open_session(*args, **kwargs)
    frames: list[_FrameTrace] = []
    epochs: list[int] = []
    reasons: list[str | None] = []
    for trace in traces:
        observations = tuple(trace.observations)
        opened = trace.opened if trace.opened is not None else observations[0].epoch_frame
        frames.append(replace(trace, observations=observations, opened=opened))
        epochs.append(int(opened.epoch))
        reasons.append(opened.reason)
    closing = frames[-1].observations[-1]
    return Paths(
        kind=session.kind,
        carrier=session.carrier,
        waveform=session.waveform,
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


def echo_paths(radar: object, paths: Paths) -> RadarSimulationResult:
    """Run the instrument half over traced rows and publish the frame cubes.

    ``radar`` supplies the receive chain and the output domain; the rows supply
    everything about the world. Every difference that would have changed the
    observation schedule is refused by name rather than replayed against a
    schedule that no longer describes it.
    """

    if not isinstance(paths, Paths):
        raise TypeError(f"echo takes the Paths that trace returned, got {type(paths).__name__}")
    session = paths._session
    _require_same_instrument(radar, paths, session)
    session = replace(session, radar=radar, **_instrument_specs(session, radar))
    return _assemble([_echo_frame(session, trace) for trace in paths._frames])


def _instrument_specs(session: _Session, radar: object) -> dict:
    """The session fields that belong to whichever radar is echoing."""

    declared, working, single = session.instrument(radar)
    return {
        "full_spec": working,
        "output_spec": declared,
        "single_spec": single,
        "path_phase_noise": radar.frontend is not None and radar.frontend.has_phase_noise,
    }


def _require_same_instrument(radar: object, paths: Paths, session: _Session) -> None:
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
    if phase_noise and not isinstance(session.output_spec, FmcwSpec):
        raise NotImplementedError("scene-driven common-oscillator phase noise currently requires FMCW")


def simulate_scene(*args, **kwargs) -> RadarSimulationResult:
    """Run one scene session to completion and stack every frame.

    The whole sequence stays in device memory: the frame cubes accumulate and
    :meth:`RadarSimulationResult.from_frames` stacks them, so peak allocation
    scales with the frame count - measured near three times the published cube
    at 128 frames, because the list and the stack are both live. Use
    :func:`stream_scene` for a sequence long enough that this is the binding
    constraint.
    """

    return _assemble(list(_scene_frames(*args, **kwargs)))


def stream_scene(*args, **kwargs) -> Iterator[RadarSimulationResult]:
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

    for frame in _scene_frames(*args, **kwargs):
        yield _assemble([frame])


#: ``Motion``, ``Paths`` and ``RadarSimulationResult`` are declared at the package
#: root instead: one public name per type, and the root is where the happy
#: path lives. This module's own public contribution is the identity record
#: a radar mounted on a moving structure needs.
__all__ = ["SensorEndpointIds"]
