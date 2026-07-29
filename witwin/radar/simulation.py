"""The scene-driven entry point: one Core world in, one frame cube out.

Phase 11 work item 1. Until this module existed ``Radar.simulate`` was a
refusal, and the only thing that assembled the pipeline end to end was fixture
orchestration under ``tests/support``. Everything that orchestration called was
already a production owner - the compile facade, the epoch loop, the propagation
adapter, the two-way join, the waveform kernels, the frame assembly - so this
module invents no physics and no geometry. It is the ASSEMBLY, and the assembly
is the thing that was missing.

The chain, once per frame, in the order it runs:

    SceneEpochLoop.frame(t)          which world, and what does it cost
      -> bind_radar_world             endpoints, sites and their stable IDs
      -> reevaluate_slots x 2         one consumer call per leg, no discovery
      -> TwoWayComposer.compose       the round trip, on the device
      -> RoundTripPatternStage.apply  the array pattern, when one was declared
      -> Radar.synthesize             the waveform this radar declares
      -> assemble_frame_cube          [chirp, pair, sample] -> [TX, RX, ...]
      -> Radar._apply_signal_models    the receive chain, if one is configured

Four decisions are written here rather than left to a reader:

**The binding is rebuilt every frame, the topology is not.** ``bind_radar_world``
reads the CURRENT snapshot, so a site riding a Core rigid motion moves between
frames; the frozen leg topologies and the join are built once per topology epoch
and replayed. Rebuilding the binding is what a moving world costs, and it is the
same three small constant tensors per endpoint set that the fixture already
built per frame - it adds no discovery, no preparation and no host observation.

**The frozen slow-time mode is the only mode this driver can honestly declare.**
It composes ONCE per frame, so the weight does not walk across chirps and the
waveform kernel owns the slow-time carrier. Declaring
``REFRESHED_WEIGHT_NO_RATE`` here would tell the kernel that a weight which
never walked has already walked, which drops the intra-frame Doppler while still
producing a plausible cube. It is refused by name.

**A scatter response has no default.** ``response`` is required. The two-way
join multiplies the round trip by the target's complex response, and every
possible default - unit amplitude, unit RCS, zero phase - is a statement about
how strongly the target scatters. Guessing it would put a number nobody chose
into every result.

**Intra-frame Doppler needs a velocity dual and this entry does not open one.**
``delay_rate`` is unpacked from a forward-AD tangent, and the tangent has to be
authored from Core kinematics - :func:`witwin.radar.propagation.two_way_duals`
is that owner. Frame-to-frame motion is fully modelled here, because every frame
re-resolves the world at its own instant; the slow-time walk WITHIN one frame is
zero unless a caller drives the kinematics seam itself. That is a named Phase-11
scope boundary, not an approximation hidden in a default.
"""

from __future__ import annotations

from dataclasses import dataclass

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

    def __post_init__(self) -> None:
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
        cls, positions_m: object, *, stable_ids: tuple[int, ...] | None = None, power_w: float = SITE_EXCITATION_POWER_W
    ) -> ScatterSitePolicy:
        return cls(
            source=SITE_SOURCE_EXPLICIT,
            positions_m=positions_m,
            stable_ids=None if stable_ids is None else tuple(int(v) for v in stable_ids),
            power_w=power_w,
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


def bind_radar_world(
    radar: object,
    snapshot: object,
    *,
    sites: ScatterSitePolicy,
    ids: StableIdAllocator | None = None,
    polarization: object = DEFAULT_POLARIZATION,
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


#: What this driver declares to the waveform kernel about the composed weight.
#:
#: Named rather than inlined so the refusal below and the value it enforces are
#: the same statement.
DRIVER_SLOW_TIME_MODE = "frozen_weight_with_carrier_rate"

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


def _slow_time_mode(declared: object):
    from .synthesis import SlowTimeMode

    if declared is None:
        return SlowTimeMode.FROZEN_WEIGHT_WITH_CARRIER_RATE
    mode = SlowTimeMode(declared)
    if mode is not SlowTimeMode.FROZEN_WEIGHT_WITH_CARRIER_RATE:
        raise ValueError(
            f"this entry point composes ONCE per frame, so its weight is "
            f"{DRIVER_SLOW_TIME_MODE!r} and cannot be declared {mode.value!r}. "
            "A refreshed weight is one that already walked across slow time; "
            "declaring it for a weight frozen at the frame's tau_rt drops the "
            "kernel's carrier-rate term and understates intra-frame Doppler "
            "while still producing a plausible cube. Drive the refreshed mode "
            "from a slot-batched replay instead"
        )
    return mode


def _times(times: object) -> tuple[float, ...]:
    values = tuple(float(value) for value in times)
    if not values:
        raise ValueError(
            "times must name at least one frame instant; an empty sequence asks for a simulation of nothing"
        )
    return values


def simulate_scene(
    radar: object,
    scene: object,
    *,
    times,
    response: object,
    sites: object = None,
    components: frozenset[str] | None = None,
    max_depth: int | None = None,
    slow_time_mode: object = None,
    ad_mode: str = "none",
    world_motion: str = "frozen_world",
    motion_event_period_frames: int | None = None,
    ids: object = None,
    polarization: object = None,
    antenna_pattern: object = None,
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

    ``world_motion`` and ``motion_event_period_frames`` are
    :class:`~witwin.radar.propagation.SceneEpochLoop`'s own two
    arguments, forwarded verbatim. A caller that would rather describe its scene
    by its parts resolves
    :func:`~witwin.radar.propagation.epoch_policy` first and passes the
    two fields it produces; the loop never reads a component declaration, and
    keeping that one-way is what stops "which parts does this scene have" and
    "when does the pipeline pay" from becoming one question.

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

    instants = _times(times)
    mode = _slow_time_mode(slow_time_mode)
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
        binding = bind_radar_world(radar, snapshot, sites=policy, ids=ids, polarization=orientation)
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

    loop = SceneEpochLoop(
        _dynamic_scene(scene),
        reference_frequency_hz=reference_frequency_hz,
        bind=bind,
        compile_scene=compile_scene,
        motion_event_period_frames=motion_event_period_frames,
        world_motion=world_motion,
    )

    cubes: list[torch.Tensor] = []
    epochs: list[int] = []
    reasons: list[str | None] = []
    synthesis = None
    legs = None
    composed = None
    epoch_frame = None
    for time_s in instants:
        epoch_frame = loop.frame(time_s)
        frozen = epoch_frame.frozen
        inbound_handle, outbound_handle = frozen.handles
        composer, epoch_binding, pattern_stage = frozen.payload
        # Rebound at every frame that did not just freeze, because the site
        # positions and the radar pose are read from the CURRENT world: a site
        # riding a Core rigid motion moves between frames while the frozen
        # topology and the join do not.
        binding = (
            epoch_binding
            if epoch_frame.rediscovered
            else bind_radar_world(radar, epoch_frame.snapshot, sites=policy, ids=ids, polarization=orientation)
        )
        legs = RadarPropagationLegs(
            inbound=frozen.adapter.reevaluate_slots(
                inbound_handle, binding.transmitters, binding.site_sinks, slot_count=1, ad_mode=ad_mode
            ),
            outbound=frozen.adapter.reevaluate_slots(
                outbound_handle, binding.site_sources, binding.receivers, slot_count=1, ad_mode=ad_mode
            ),
        )
        composed = composer.compose(legs.inbound, legs.outbound, response)
        if pattern_stage is not None:
            composed = pattern_stage.apply(
                composed,
                tx_pos=binding.transmitters.positions_m,
                rx_pos=binding.receivers.positions_m,
                site_positions_m=binding.site_positions_m,
            )
        synthesis = radar._synthesize(composed, slow_time_mode=mode)
        cubes.append(
            radar._apply_signal_models(assemble_frame_cube(synthesis.cube, num_tx=array.num_tx, num_rx=array.num_rx))
        )
        epochs.append(epoch_frame.epoch)
        reasons.append(epoch_frame.reason)

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
    )


__all__ = ["RadarSimulationResult", "ScatterSitePolicy", "StableIdAllocator"]
