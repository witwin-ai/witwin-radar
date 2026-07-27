"""Bind a Core world and a ``Radar`` to the typed inputs the pipeline consumes.

The scene-driven entry point needs one answer to three questions before any
propagation can happen: which endpoints exist, what they are called, and where
the scatter sites are. Until Phase 11 all three answers lived in
``tests/support/multi_endpoint_world.py``, which is fixture orchestration and
says so. This module is the production owner of the same three answers. It
invents no physics and no geometry: every number it publishes is either a
Radar-owned pose, a Core-owned world quantity, or a caller declaration.

Three rules shape the whole module, and each of them has already cost real
debugging time somewhere in this repository:

* **Stable IDs are a REPRODUCIBLE scheme, never a process counter.**
  ``witwin.core.identity.new_*_id()`` draws from a process-global counter, so
  the same world built in a different order or in a different process gets
  different IDs, and a frozen leg topology names its rows by identity. The
  allocator here derives every ID from a declared base plus an array index, so
  two bindings of the same world in two processes agree. Allocation happens once
  per binding, at setup; nothing here runs per frame.
* **A site tensor is ONE object playing two roles.** The sites are the sinks of
  the inbound leg and the sources of the outbound leg, and
  :class:`RadarWorldBinding` hands the SAME ``positions_m`` tensor to both. That
  identity is what lets a reverse gradient accumulate over both legs and what
  lets one forward-AD dual carry a site's velocity into both; rebuilding the
  tensor for the second role silently halves the first and zeroes the second.
* **Site derivation is a declaration, not a search.** See
  :class:`ScatterSitePolicy`.

The endpoint contract itself is Channel's and is quoted rather than restated:
``positions_m`` is float32, contiguous, ``(N, 3)`` and CUDA-resident by the time
it reaches the consumer; ``stable_ids`` is int64; a SOURCE carries ``powers_w``
and a SINK must not.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .propagation.contracts import RadarEndpointSpec


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
#: Channel's endpoint polarization is a WORLD-frame vector that its native
#: material evaluation projects the field onto. It is deliberately not derived
#: from :class:`~witwin.radar.sensors.contracts.PolarizationSpec`: that spec
#: describes the LEGACY real-amplitude transmit/receive projection, and Channel
#: has already projected, so reusing it here would be the second projection its
#: own docstring warns about.
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
                raise ValueError(
                    f"{name} must be a non-negative int, got {value!r}"
                )

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
        blocks = tuple(
            tuple(range(base, base + count))
            for base, count in zip(bases, counts, strict=True)
        )
        for first in range(len(blocks)):
            for second in range(first + 1, len(blocks)):
                low, high = bases[first], bases[second]
                if (
                    low < high + counts[second]
                    and high < low + counts[first]
                ):
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
            raise ValueError(
                f"source must be one of {list(SITE_SOURCES)}, got {self.source!r}"
            )
        if self.source == SITE_SOURCE_EXPLICIT:
            if self.positions_m is None:
                raise ValueError(
                    "an explicit site policy requires positions_m; "
                    "ScatterSitePolicy.explicit(positions) builds one"
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
    ) -> "ScatterSitePolicy":
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
    ) -> "ScatterSitePolicy":
        """Sites at the world anchors Core publishes for moving structures.

        ``structure_ids`` selects a subset; ``None`` takes every structure the
        snapshot carries. Selection and ordering are both by ascending structure
        ID rather than by the snapshot's tuple order, so the site array order is
        a function of world identity and survives a reordered scene.
        """

        return cls(
            source=SITE_SOURCE_STRUCTURE_ANCHOR,
            structure_ids=(
                None
                if structure_ids is None
                else tuple(int(value) for value in structure_ids)
            ),
            stable_ids=None if stable_ids is None else tuple(int(v) for v in stable_ids),
            power_w=power_w,
        )

    def resolve(self, snapshot: object, *, device: torch.device) -> torch.Tensor:
        """The ``(S, 3)`` float32 site positions this policy names."""

        if self.source == SITE_SOURCE_EXPLICIT:
            return _site_positions(self.positions_m, device=device)
        return _structure_anchor_positions(
            snapshot, self.structure_ids, device=device
        )


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
            raise ValueError(
                "site positions must have shape (S, 3), got "
                f"{tuple(positions.shape)}"
            )
        if not positions.is_contiguous():
            raise ValueError("site positions must be contiguous")
        return positions
    return torch.tensor(
        [tuple(float(value) for value in row) for row in positions],
        dtype=torch.float32,
        device=device,
    )


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
        raise TypeError(
            "snapshot must expose structures; pass a witwin.core SceneSnapshot"
        )
    by_id: dict[int, object] = {}
    for state in states:
        key = int(state.structure_id)
        if key in by_id:
            raise ValueError(
                f"structure_id {key} appears twice in the snapshot; a site "
                "anchor must name exactly one structure"
            )
        by_id[key] = state
    if structure_ids is None:
        selected = sorted(by_id)
    else:
        selected = sorted(structure_ids)
        missing = [key for key in selected if key not in by_id]
        if missing:
            raise ValueError(
                f"structure_ids {missing} are not in this snapshot, which "
                f"carries {sorted(by_id)}"
            )
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
    positions_m: torch.Tensor,
    stable_ids: tuple[int, ...],
    *,
    polarization: torch.Tensor,
    power_w: float | None,
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
        powers_w=(
            None
            if power_w is None
            else torch.full(
                (rows,), float(power_w), dtype=torch.float32, device=device
            )
        ),
    )


def _polarization_tensor(
    polarization: object, *, device: torch.device
) -> torch.Tensor:
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
        raise TypeError(
            f"radar.{name} must be a torch.Tensor of world element positions; "
            "pass a witwin.radar.Radar"
        )
    if positions.dtype != torch.float32:
        raise TypeError(f"radar.{name} must use torch.float32, got {positions.dtype}")
    if positions.ndim != 2 or int(positions.shape[1]) != 3:
        raise ValueError(
            f"radar.{name} must have shape (N, 3), got {tuple(positions.shape)}"
        )
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
            f"radar.tx_pos is on {device} but radar.rx_pos is on "
            f"{receiver_positions.device}; one radar is one device"
        )
    site_positions = sites.resolve(snapshot, device=device)

    allocator = StableIdAllocator() if ids is None else ids
    transmitter_ids, receiver_ids, allocated_site_ids = allocator.allocate(
        transmitter_count=int(transmitter_positions.shape[0]),
        receiver_count=int(receiver_positions.shape[0]),
        site_count=int(site_positions.shape[0]),
    )
    site_ids = (
        allocated_site_ids if sites.stable_ids is None else sites.stable_ids
    )
    if len(site_ids) != int(site_positions.shape[0]):
        raise ValueError(
            f"the site policy declared {len(site_ids)} stable IDs for "
            f"{int(site_positions.shape[0])} site positions"
        )
    overlap = (set(site_ids) & set(transmitter_ids)) | (
        set(site_ids) & set(receiver_ids)
    )
    if overlap:
        raise ValueError(
            f"site stable IDs {sorted(overlap)} collide with the transmitter or "
            "receiver blocks; two endpoints cannot share one stable world ID"
        )

    polarization_vector = _polarization_tensor(polarization, device=device)
    transmit_power_w = float(
        radar.system_config.sensors.tx_power.transmit_power_watts
    )
    return RadarWorldBinding(
        transmitters=_endpoint_spec(
            transmitter_positions,
            transmitter_ids,
            polarization=polarization_vector,
            power_w=transmit_power_w,
        ),
        receivers=_endpoint_spec(
            receiver_positions,
            receiver_ids,
            polarization=polarization_vector,
            power_w=None,
        ),
        site_sources=_endpoint_spec(
            site_positions,
            site_ids,
            polarization=polarization_vector,
            power_w=sites.power_w,
        ),
        site_sinks=_endpoint_spec(
            site_positions,
            site_ids,
            polarization=polarization_vector,
            power_w=None,
        ),
        transmitter_ids=transmitter_ids,
        receiver_ids=receiver_ids,
        site_ids=tuple(site_ids),
    )


__all__ = [
    "DEFAULT_POLARIZATION",
    "DEFAULT_RECEIVER_ID_BASE",
    "DEFAULT_SITE_ID_BASE",
    "DEFAULT_TRANSMITTER_ID_BASE",
    "SITE_EXCITATION_POWER_W",
    "SITE_SOURCES",
    "SITE_SOURCE_EXPLICIT",
    "SITE_SOURCE_STRUCTURE_ANCHOR",
    "RadarWorldBinding",
    "ScatterSitePolicy",
    "StableIdAllocator",
    "bind_radar_world",
]
