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
    amplitude tolerance is relative per path. Max interval [s] bounds the
    topology probe spacing. Near coherent nulls a relative IQ bound cannot be
    inferred from these path bounds. Exhausting the discovery budget raises.
    """

    phase_error_rad: float = 0.02
    relative_amplitude_error: float = 0.02
    max_interval_s: float = 0.002
    max_evaluations: int = 8192
    batch_observations: int = 256

    def __post_init__(self):
        for name in ("phase_error_rad", "relative_amplitude_error", "max_interval_s"):
            value = getattr(self, name)
            if isinstance(value, torch.Tensor) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive host value")
        for name in ("max_evaluations", "batch_observations"):
            _positive_int(getattr(self, name), name=name)


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
    transmit_power_w = float(radar.system_config.sensors.tx_power.transmit_power_watts)
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

    The four ``last_*`` members are the LAST frame's typed state, and they are
    what :attr:`witwin.radar.Radar.last_snapshot` and its three siblings read.
    They describe one frame, not the sequence: a compiled scene and a leg pair
    are per-epoch and per-frame objects, and stacking them would either
    misrepresent the epochs or retain every frame's device memory for the life
    of the result. Keeping the last one is the diagnostic the plan asked for and
    the smallest retention that answers it.

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
            motion_sampling=motion_sampling,
            output_domain=synthesis.output_domain,
            adaptive_diagnostics=tuple(adaptive_diagnostics),
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


def _adaptive_fmcw(times, evaluate_many, spec, options, carrier_hz, frontend):
    """Control topology probes on the host; interpolate and synthesize on CUDA.

    Quarter/midpoint tests bound observed errors only. Arbitrarily brief path
    births or adversarial oscillations between probes require the ADC reference.
    Host copies below are explicit adaptive decisions, not a differentiable
    physics implementation. AD follows the accepted, fixed partition.
    """
    import numpy as np

    from .paths import interpolate_path_rows
    from .synthesis.fmcw import channel_phasor_to_beat_weight, synthesize_fmcw_rows

    cache, partitions, pair_tables = {}, {}, {}
    stats = {
        "evaluations": 0,
        "topology_refinements": 0,
        "accepted_intervals": 0,
        "max_tested_phase_error_rad": 0.0,
        "max_tested_relative_amplitude_error": 0.0,
        "max_interval_s": options.max_interval_s,
        "exhaustive": False,
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

    def prediction(left, right, index):
        a, b = cache[left][2], cache[right][2]
        alpha = torch.full_like(
            a.total_delay_s, (times[index] - times[left]) / (times[right] - times[left]), dtype=torch.float64
        )
        return interpolate_path_rows(
            a.total_delay_s, b.total_delay_s, a.complex_transfer_ref, b.complex_transfer_ref, alpha, carrier_hz
        )

    pending = [(0, len(times) - 1)]
    while pending:
        probes = {}
        for left, right in pending:
            probes[left, right] = sorted(
                {left, right, (3 * left + right) // 4, (left + right) // 2, (left + 3 * right) // 4}
            )
        ensure(index for group in probes.values() for index in group)
        next_pending = []
        for (left, right), indices in probes.items():
            identities = [cache[index][3] for index in indices]
            topology_ok = all(key == identities[0] for key in identities)
            phase_error = amplitude_error = 0.0
            accepted = right - left <= 1
            if not accepted and topology_ok and times[right] - times[left] <= options.max_interval_s:
                for index in indices[1:-1]:
                    delay, transfer = prediction(left, right, index)
                    actual = cache[index][2]
                    # These detached reads drive refinement only. They do not
                    # replace the native, differentiable production interpolant.
                    predicted_delay = delay.detach().double().cpu().numpy()
                    observed_delay = actual.total_delay_s.detach().double().cpu().numpy()
                    d = predicted_delay - observed_delay
                    predicted = transfer.detach().cpu().numpy().astype(np.complex128)
                    observed = actual.complex_transfer_ref.detach().cpu().numpy().astype(np.complex128)
                    live = actual.row_valid.cpu().numpy()
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
                    partitions[index] = (left, right)
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
    bounds = np.asarray([partitions[index] for index in range(len(times))], dtype=np.int64)
    bounds[ordered] = np.asarray(ordered)[:, None]
    left, right = bounds.T
    clock = np.asarray(times)
    span = clock[right] - clock[left]
    fraction = np.divide(clock - clock[left], span, out=np.zeros_like(clock), where=span != 0)

    def upload(value):
        return torch.as_tensor(value, device=device)

    columns = []
    # Group equal ADC offsets so the existing FMCW phase owner evaluates all
    # slow-time observations in one segment batch; never synthesize an N*N grid.
    for sample in range(spec.num_samples):
        indices = list(range(sample, len(times), spec.num_samples))
        values = []
        for begin in range(0, len(indices), options.batch_observations):
            batch = indices[begin : begin + options.batch_observations]
            batch = np.asarray(batch)
            row_counts = counts[left[batch]].sum(axis=1)
            offsets = np.concatenate(([0], np.cumsum(counts[left[batch]].ravel())))
            observation = np.repeat(np.arange(len(batch)), row_counts)
            local_row = np.arange(offsets[-1]) - np.repeat(np.cumsum(row_counts) - row_counts, row_counts)
            first_row = upload(starts[left[batch]][observation] + local_row)
            last_row = upload(starts[right[batch]][observation] + local_row)
            delay, transfer = interpolate_path_rows(
                delays[first_row],
                delays[last_row],
                transfers[first_row],
                transfers[last_row],
                upload(fraction[batch][observation]),
                carrier_hz,
            )
            transfer = torch.where(validity[first_row], transfer, torch.zeros_like(transfer))
            if frontend is not None:
                transfer = frontend._apply_path_phase_rows(delay, transfer, upload(clock[batch][observation]))
            batch_spec = replace(
                spec,
                num_chirps=1,
                num_samples=1,
                num_tx=1,
                num_rx=len(batch) * pairs,
                output_domain="beat",
                t_start_s=spec.t_start_s + sample * spec.sample_period_s,
            )
            cube = synthesize_fmcw_rows(
                delay, None, channel_phasor_to_beat_weight(transfer), upload(offsets), batch_spec
            )
            values.append(cube.reshape(len(batch), pairs))
        columns.append(torch.cat(values, dim=0))
    all_slots = torch.stack(columns, dim=-1)
    pair = torch.arange(pairs, device=all_slots.device)
    slot = torch.arange(spec.num_chirps, device=all_slots.device)[:, None] * spec.num_tx + pair[None, :] % spec.num_tx
    stats["evaluations"] = len(cache)
    stats["exhaustive"] = len(cache) == len(times)
    stats["observation_count"] = len(times)
    return all_slots[slot, pair], stats, cache[0], cache[len(times) - 1]


def simulate_scene(
    radar: object,
    scene: object,
    *,
    times,
    response: object,
    sites: object = None,
    components: frozenset[str] | None = None,
    max_depth: int | None = None,
    ad_mode: str = "none",
    world_motion: str = "frozen_world",
    motion_event_period_frames: int | None = None,
    ids: object = None,
    polarization: object = None,
    antenna_pattern: object = None,
    sensor_endpoints: SensorEndpointIds | None = None,
    motion_sampling: str = "adc",
    adaptive_motion: AdaptiveMotionSpec | None = None,
) -> RadarSimulationResult:
    """Run ``radar`` over ``scene`` at ``times`` and publish the frame cubes.

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
    brief unseen events; ``path_set_complete`` records this boundary. Complete discovery is
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
    from .synthesis.assembly import assemble_frame_cube

    if motion_sampling not in ("adc", "chirp", "adaptive"):
        raise ValueError("motion_sampling must be adc, chirp, or adaptive")
    if adaptive_motion is not None and motion_sampling != "adaptive":
        raise ValueError("adaptive_motion requires motion_sampling='adaptive'")
    adaptive = AdaptiveMotionSpec() if adaptive_motion is None else adaptive_motion
    if not isinstance(adaptive, AdaptiveMotionSpec):
        raise TypeError("adaptive_motion must be AdaptiveMotionSpec")
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
    from .synthesis.assembly import FmcwSpec, SlowTimeMode, SynthesisResult, waveform_sampling

    mode = SlowTimeMode.FROZEN_WEIGHT_WITH_CARRIER_RATE
    full_spec = solve_config.waveform_spec()
    path_phase_noise = radar.frontend is not None and radar.frontend.has_phase_noise
    if path_phase_noise:
        if not isinstance(full_spec, FmcwSpec):
            raise NotImplementedError("scene-driven common-oscillator phase noise currently requires FMCW")
        if motion_sampling not in ("adc", "adaptive"):
            raise ValueError("common-oscillator phase noise requires ADC-time observations")
        sampled = True
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
            for slot, (t, frame, binding, identity) in enumerate(group):
                legs = (
                    replay
                    if len(group) == 1
                    else RadarPropagationLegs(inbound=replay.inbound.slot(slot), outbound=replay.outbound.slot(slot))
                )
                composer, _, pattern_stage = frame.frozen.payload
                current_response = response.at(t) if callable(getattr(response, "at", None)) else response
                paths = composer.compose(legs.inbound, legs.outbound, current_response, include_delay_rate=False)
                if pattern_stage is not None:
                    paths = pattern_stage.apply(
                        paths,
                        tx_pos=binding.transmitters.positions_m,
                        rx_pos=binding.receivers.positions_m,
                        tx_targets_m=legs.inbound.departure_target_m.index_select(0, paths.topology.inbound_row),
                        rx_targets_m=legs.outbound.arrival_origin_m.index_select(0, paths.topology.outbound_row),
                    )
                if motion_sampling == "adaptive":
                    identity = (identity, tuple(paths.row_valid.tolist()))
                results[t] = (frame, legs, paths, identity)

        for t in query_times:
            frame = loop.frame(t)
            binding = (
                frame.frozen.payload[1]
                if frame.rediscovered
                else bind_radar_world(
                    radar,
                    frame.snapshot,
                    sites=policy,
                    ids=ids,
                    polarization=orientation,
                    sensor_endpoints=sensor_endpoints,
                )
            )
            identity = (
                tuple(tuple(zip(*leg_identity(handle, "adaptive"), strict=True)) for handle in frame.frozen.handles)
                if motion_sampling == "adaptive"
                else ()
            )
            entry = (t, frame, binding, identity)
            if loop.structures_move:
                finish([entry])
            else:
                groups.setdefault(identity, []).append(entry)
        for group in groups.values():
            finish(group)
        return [results[t] for t in query_times]

    def finish_frame(synthesis):
        frame_cube = radar._apply_signal_models(
            assemble_frame_cube(synthesis.cube, num_tx=array.num_tx, num_rx=array.num_rx),
            phase_in_signal=path_phase_noise,
        )
        if isinstance(output_spec, FmcwSpec) and output_spec.output_domain != synthesis.output_domain:
            from .processing.range_doppler import fmcw_range_fft

            frame_cube = fmcw_range_fft(frame_cube)
            synthesis = SynthesisResult.from_fmcw(
                frame_cube.permute(2, 1, 0, 3).reshape(full_spec.num_chirps, -1, full_spec.num_samples), output_spec
            )
        return frame_cube, synthesis

    cubes: list[torch.Tensor] = []
    epochs: list[int] = []
    reasons: list[str | None] = []
    synthesis = None
    legs = None
    composed = None
    epoch_frame = None
    sample_times = []
    slot_cubes = []
    diagnostics = []
    adaptive_active = sampled and motion_sampling == "adaptive"
    if adaptive_active:
        for start in instants:
            actual_times = tuple(start + offset for offset in offsets)
            cube, stats, first, last = _adaptive_fmcw(
                actual_times, evaluate_many, full_spec, adaptive, reference_frequency_hz, radar.frontend
            )
            diagnostics.append(stats)
            epochs.append(first[0].epoch)
            reasons.append(first[0].reason)
            sample_times.append(actual_times)
            epoch_frame, legs, composed, _ = last
            synthesis = SynthesisResult.from_fmcw(cube, replace(full_spec, output_domain="beat"))
            frame_cube, synthesis = finish_frame(synthesis)
            cubes.append(frame_cube)
    for frame_index, time_s in (
        (frame_index, start + offset)
        for frame_index, start in enumerate(() if adaptive_active else instants)
        for offset in offsets
    ):
        epoch_frame, legs, composed, _ = evaluate_many([time_s])[0]
        if not slot_cubes:
            epochs.append(epoch_frame.epoch)
            reasons.append(epoch_frame.reason)
            sample_times.append(tuple(instants[frame_index] + offset for offset in offsets))
        if path_phase_noise:
            composed = radar.frontend.apply_path_phase(composed, time_s)
        observation_spec = (
            replace(
                single_spec,
                t_start_s=full_spec.t_start_s + (len(slot_cubes) % samples_per_slot) * full_spec.sample_period_s,
            )
            if adc_sampled
            else single_spec
        )
        synthesis = (
            radar._synthesize(composed, slow_time_mode=mode, spec=observation_spec)
            if sampled
            else radar._synthesize(composed, slow_time_mode=mode, spec=full_spec)
        )
        slot_cubes.append(synthesis.cube)
        if len(slot_cubes) < len(offsets):
            continue
        if sampled:
            stacked = torch.cat(slot_cubes, dim=0)
            pairs = torch.arange(array.num_tx * array.num_rx, device=stacked.device)
            if adc_sampled:
                stacked = stacked.reshape(-1, samples_per_slot, len(pairs)).transpose(1, 2)
                cube = stacked[pair_samples, pairs]
                if full_spec.output_domain == "spectrum":
                    from .processing.range_doppler import fmcw_range_fft

                    cube = fmcw_range_fft(cube)
                synthesis = SynthesisResult.from_fmcw(cube, full_spec)
            else:
                synthesis = replace(synthesis, cube=stacked[pair_samples, pairs])
        frame_cube, synthesis = finish_frame(synthesis)
        cubes.append(frame_cube)
        slot_cubes = []

    return RadarSimulationResult.from_frames(
        cubes,
        times_s=instants,
        synthesis=synthesis,
        epochs=epochs,
        rediscovery_reasons=reasons,
        compile_count=loop.compile_count,
        discovery_count=loop.discovery_count,
        last_snapshot=epoch_frame.snapshot,
        last_compiled_scene=epoch_frame.compiled,
        last_propagation=legs,
        last_radar_paths=composed,
        sample_times_s=sample_times,
        path_set_complete=all(item["exhaustive"] for item in diagnostics)
        if adaptive_active
        else (not sampled or cadence == 1 or world_motion == "frozen_world" and loop.structures_move),
        motion_sampling="static" if not sampled else motion_sampling,
        adaptive_diagnostics=diagnostics,
    )


__all__ = ["AdaptiveMotionSpec", "RadarSimulationResult", "ScatterSitePolicy", "SensorEndpointIds", "StableIdAllocator"]
