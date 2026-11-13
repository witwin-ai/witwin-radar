"""Binding one radar to one world snapshot: stable IDs, scatter sites, endpoint specs.

A binding is a pure function of the radar, the snapshot and the declared site
policy. It allocates the deterministic stable IDs a frozen leg topology names
its rows by, resolves the scatter sites, and builds the four endpoint specs the
two legs consume. It reads no device value and computes no geometry: sites are
declared by the caller, published by Core, or - for a mesh target - placed by
the native facet operator that :mod:`witwin.radar.facets` owns (R-ADR-042).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Protocol

import torch

from .policy import require_leg_ad_mode
from .propagation import Kinematics, RadarEndpointSpec, endpoint_kinematics

if TYPE_CHECKING:
    from .radar import Radar

#: Declared site positions: a live ``(S, 3)`` tensor or host triples.
SitePositions = torch.Tensor | Sequence[Sequence[float]]


class SiteTrajectory(Protocol):
    """A site motion: the material points' world positions at one instant."""

    def at(self, time_s: float) -> Kinematics: ...


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

#: The outbound leg is excited at exactly one watt.
#:
#: The outbound leg is evaluated from each receive element toward the sites,
#: with the receive element as a source of this power; by Lorentz reciprocity
#: that transport equals the site re-radiating at the same power toward the
#: receiver (R-ADR-039). The site is a re-radiator, not a second transmitter:
#: the whole target strength lives in the two-way join's
#: ``S = sqrt(4 pi sigma) / lambda`` factor. An excitation of anything but unit
#: power multiplies that factor again, and with a transmit power of 1 W the
#: extra ``sqrt(P)`` is numerically invisible - which is exactly how a squared
#: transmit power ships.
SITE_EXCITATION_POWER_W = 1.0

#: Where scatter sites may come from: declared by the caller, published by
#: Core, or the facets of a mesh target placed by the native facet operator.
SITE_SOURCE_EXPLICIT = "explicit"
SITE_SOURCE_STRUCTURE_ANCHOR = "structure_anchor"
SITE_SOURCE_FACETS = "facets"
SITE_SOURCES = (SITE_SOURCE_EXPLICIT, SITE_SOURCE_STRUCTURE_ANCHOR, SITE_SOURCE_FACETS)

_MESH_SITE_DEFERRAL = (
    "deriving scatter sites by sampling a structure's MESH is deferred by "
    "R-ADR-020. A sampling rule is a geometry algorithm, and "
    "geometry on a production path belongs to Channel's native geometry owner, "
    "not to a Torch expression in Radar. Declare the sites instead - "
    "PointTargets(positions=...) - give the structure a rigid "
    "motion so that Core publishes a world anchor for it, or scatter from its "
    "facets with MeshTargets"
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
        transmitters, receivers, sites = (
            tuple(range(base, base + count)) for base, count in zip(bases, counts, strict=True)
        )
        blocks = (transmitters, receivers, sites)
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

    ``power_w`` is the outbound-leg excitation and defaults to
    :data:`SITE_EXCITATION_POWER_W`. Changing it is almost always wrong; read
    that constant's note first.

    ``ad_mode`` is the AD mode the legs are replayed in. A site position moves
    an endpoint of both legs, so one carrying a derivative that mode does not
    carry is refused (:func:`witwin.radar.policy.require_leg_ad_mode`): a
    declared tensor when the policy is built, before any work; a trajectory's
    positions, a structure anchor and a mesh target's world vertices each time
    they are resolved, before the frame launches anything that reads them.
    ``None`` is a policy no leg replays (ground-truth labels, a network's
    second radar binding positions already admitted), which checks nothing.
    """

    source: str
    positions_m: SitePositions | None = None
    structure_ids: tuple[int, ...] | None = None
    stable_ids: tuple[int, ...] | None = None
    power_w: float = SITE_EXCITATION_POWER_W
    trajectory: SiteTrajectory | None = None
    #: The facet source of a mesh target (:class:`witwin.radar.facets.FacetSites`).
    facets: Any = None
    #: The AD mode the legs replay these sites in; ``None`` checks nothing.
    ad_mode: str | None = None

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
                    "an explicit site policy requires positions_m; PointTargets(positions=...) supplies them"
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
                    "structure_anchor or facets policy reads its positions from the "
                    "snapshot"
                )
        if (self.source == SITE_SOURCE_FACETS) != (self.facets is not None):
            raise ValueError("a facets site policy, and only that policy, carries a facet source")
        if not float(self.power_w) > 0.0:
            raise ValueError("power_w must be positive")
        self._admit("PointTargets.positions", self.positions_m)

    def _admit(self, leaf: str, positions: object) -> None:
        """Refuse site positions carrying a derivative the legs will not carry in this policy's AD mode."""

        if self.ad_mode is not None:
            require_leg_ad_mode(leaf, positions, self.ad_mode, moves="moves a scatter site, an endpoint of both legs")

    @classmethod
    def explicit(
        cls,
        positions_m: SitePositions,
        *,
        stable_ids: tuple[int, ...] | None = None,
        power_w: float = SITE_EXCITATION_POWER_W,
        trajectory: SiteTrajectory | None = None,
        ad_mode: str | None = None,
    ) -> ScatterSitePolicy:
        return cls(
            source=SITE_SOURCE_EXPLICIT,
            positions_m=positions_m,
            stable_ids=None if stable_ids is None else tuple(int(v) for v in stable_ids),
            power_w=power_w,
            trajectory=trajectory,
            ad_mode=ad_mode,
        )

    @classmethod
    def structure_anchor(
        cls,
        *,
        structure_ids: tuple[int, ...] | None = None,
        stable_ids: tuple[int, ...] | None = None,
        power_w: float = SITE_EXCITATION_POWER_W,
        ad_mode: str | None = None,
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
            ad_mode=ad_mode,
        )

    @classmethod
    def from_facets(
        cls, facets: Any, *, power_w: float = SITE_EXCITATION_POWER_W, ad_mode: str | None = None
    ) -> ScatterSitePolicy:
        """Sites at a mesh target's facets, placed per snapshot by the native facet operator."""

        return cls(source=SITE_SOURCE_FACETS, facets=facets, power_w=power_w, ad_mode=ad_mode)

    def resolve_frame(self, snapshot: Any, *, device: torch.device) -> tuple[torch.Tensor, Any]:
        """The site positions, and the facet frame they came from when this is a mesh target."""

        if self.source == SITE_SOURCE_FACETS:
            # The world vertices are admitted inside, before the facet operator reads them.
            frame = self.facets.resolve(snapshot, device=device, admit=self._admit)
            return frame.sites, frame
        return self.resolve(snapshot, device=device), None

    def resolve(self, snapshot: Any, *, device: torch.device) -> torch.Tensor:
        """The ``(S, 3)`` site positions this policy names, float32 or float64 as authored."""

        if self.source == SITE_SOURCE_FACETS:
            return self.resolve_frame(snapshot, device=device)[0]

        if self.source == SITE_SOURCE_EXPLICIT:
            if self.trajectory is not None:
                sample = self.trajectory.at(snapshot.time_s)
                if not isinstance(sample, Kinematics):
                    raise TypeError("site trajectory.at(time_s) must return Kinematics")
                positions = _site_positions(sample.positions_m, device=device)
                self._admit("the output of PointTargets.trajectory", positions)
                if positions.shape != _site_positions(self.positions_m, device=device).shape:
                    raise ValueError("a site trajectory must preserve material-point count and ordering")
                return positions
            return _site_positions(self.positions_m, device=device)
        anchors = _structure_anchor_positions(snapshot, self.structure_ids, device=device)
        self._admit("a StructureTargets anchor (a structure's rigid-motion translation)", anchors)
        return anchors


def _site_positions(positions: SitePositions | None, *, device: torch.device) -> torch.Tensor:
    """Normalise declared site positions without disturbing a live tensor.

    A tensor is validated and returned as it stands. It is deliberately NOT
    moved, cast, or made contiguous here: every one of those is a new node that
    would leave the caller holding a tensor that is no longer the one the legs
    differentiate through, and a device or dtype mismatch is a caller error
    worth a message rather than a silent copy. A float64 tensor is the precise
    position the join computes path lengths from (R-ADR-040); the binding
    rounds it to float32 for Channel.
    """

    if isinstance(positions, torch.Tensor):
        if positions.dtype not in (torch.float32, torch.float64):
            raise TypeError(
                f"site positions must use torch.float32 or torch.float64, got {positions.dtype}; "
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
    if positions is None:
        raise ValueError("an explicit site policy requires positions_m")
    return torch.tensor([tuple(float(value) for value in row) for row in positions], dtype=torch.float32, device=device)


def _structure_anchor_positions(
    snapshot: Any, structure_ids: tuple[int, ...] | None, *, device: torch.device
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
    by_id: dict[int, Any] = {}
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

    Four specs and three ID tuples. Both site legs are evaluated FROM a radar
    element TOWARD the sites: the inbound leg from the transmitters, the
    outbound leg from ``receiver_sources`` (the receivers as 1 W sources),
    and every site is a sink. ``receivers`` is the same elements as a sink,
    for the site-less transmitter-to-receiver leg; the two share one
    ``positions_m`` object. The outbound transport is the reciprocal of the site-to-receiver one
    (R-ADR-039), which is what lets one transported field vector per row carry
    a site's full scattering matrix. ``sites`` is therefore ONE spec shared by
    both legs, and its ``positions_m`` is the one site tensor both legs
    differentiate through.

    The ID tuples are host tuples, not tensors, because the composer's declared
    identity lists are host lists and because reading them back out of a device
    tensor would be a host observation on a path that has none.
    """

    transmitters: RadarEndpointSpec
    receivers: RadarEndpointSpec
    receiver_sources: RadarEndpointSpec
    sites: RadarEndpointSpec
    transmitter_ids: tuple[int, ...]
    receiver_ids: tuple[int, ...]
    site_ids: tuple[int, ...]
    #: A mesh target's facet frame (:class:`witwin.radar.facets.FacetFrame`) at
    #: this snapshot, whose ``sites`` are the site positions; ``None`` otherwise.
    facets: Any = None

    def __post_init__(self) -> None:
        if self.transmitters.powers_w is None:
            raise ValueError("the transmitter spec must carry powers_w")
        if self.receivers.powers_w is not None:
            raise ValueError("the receiver spec must not carry powers_w; it is the direct leg's sink")
        if self.receiver_sources.powers_w is None:
            raise ValueError("the receiver source spec must carry powers_w; it is the source of the outbound leg")
        if self.receiver_sources.positions_m is not self.receivers.positions_m:
            raise ValueError("the receivers' two roles must share one positions_m tensor")
        if self.sites.powers_w is not None:
            raise ValueError("the site spec must not carry powers_w; a site is the sink of both legs")

    @property
    def site_positions_m(self) -> torch.Tensor:
        """The one site tensor both legs differentiate through."""

        return self.sites.positions_m

    @property
    def site_count(self) -> int:
        return self.sites.count

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


def site_spec(
    positions_m: torch.Tensor, stable_ids: tuple[int, ...], *, polarization: torch.Tensor
) -> RadarEndpointSpec:
    """The sites as the sink of both legs.

    Float64 positions are kept as the spec's ``precise_positions_m`` and
    rounded to float32 for Channel through a differentiable cast, so a leaf in
    either precision receives its gradient through both (R-ADR-040).
    """

    if positions_m.dtype == torch.float64:
        spec = _endpoint_spec(positions_m.float(), stable_ids, polarization=polarization, power_w=None)
        return replace(spec, precise_positions_m=positions_m.contiguous())
    return _endpoint_spec(positions_m, stable_ids, polarization=polarization, power_w=None)


def with_site_positions(spec: RadarEndpointSpec, positions_m: torch.Tensor) -> RadarEndpointSpec:
    """``spec`` at new site positions, in the precision they were authored in."""

    if positions_m.dtype == torch.float64:
        return replace(spec, positions_m=positions_m.float(), precise_positions_m=positions_m.contiguous())
    return replace(spec, positions_m=positions_m, precise_positions_m=None)


def _polarization_tensor(polarization: Sequence[float], *, device: torch.device) -> torch.Tensor:
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


def _array_positions(radar: Radar, name: str) -> torch.Tensor:
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

    def __post_init__(self) -> None:
        for name in ("transmitters", "receivers"):
            values = getattr(self, name)
            if not values or len(set(values)) != len(values):
                raise ValueError(f"{name} must contain distinct endpoint IDs")


def bind_radar_world(
    radar: Radar,
    snapshot: Any,
    *,
    sites: ScatterSitePolicy,
    polarization: Sequence[float] = DEFAULT_POLARIZATION,
    sensor_endpoints: SensorEndpointIds | None = None,
    rx_polarization: Sequence[float] | None = None,
    ad_mode: str | None = None,
) -> RadarWorldBinding:
    """Turn one ``Radar`` plus one ``SceneSnapshot`` into endpoint specs.

    The transmit elements become SOURCES carrying the array's transmit power in
    watts; the receive elements become SOURCES of the outbound leg carrying the
    outbound excitation (R-ADR-039); the sites are the SINK of both legs. Both
    element sets come from ``radar.tx_pos`` / ``radar.rx_pos``, which are the
    pose-transformed world positions the radar already maintains - they are used as they stand
    rather than rebuilt, so a caller who supplied moving Core phase centres
    through ``sensor_endpoints`` keeps their tape all the way into both legs,
    and so does a radar whose own pose is a leaf (R-ADR-052).

    ``snapshot`` is required even for an explicit site policy. A binding is
    against a world at an instant, and letting it be optional would invite a
    caller to bind once and replay against a world that has moved on.

    It runs once per topology epoch, and once per observation when the sites or
    the sensor endpoints move with the snapshot. It allocates IDs, builds three
    small constant tensors, and copies nothing back to the host.

    ``ad_mode`` is the verb's. Core sensor endpoints become the element
    positions both legs start or end at, so one carrying a derivative that mode
    does not carry is refused here, each time they are read
    (:func:`witwin.radar.policy.require_leg_ad_mode`). ``None`` is a binding no
    leg replays, which checks nothing.
    """

    transmitter_positions = _array_positions(radar, "tx_pos")
    receiver_positions = _array_positions(radar, "rx_pos")
    device = transmitter_positions.device
    if sensor_endpoints is not None:
        if not isinstance(sensor_endpoints, SensorEndpointIds):
            raise TypeError("sensor_endpoints must be SensorEndpointIds")
        tx = endpoint_kinematics(snapshot, sensor_endpoints.transmitters, device=device)
        rx = endpoint_kinematics(snapshot, sensor_endpoints.receivers, device=device)
        if tx.positions_m.shape != transmitter_positions.shape or rx.positions_m.shape != receiver_positions.shape:
            raise ValueError("sensor endpoint counts must match the radar array")
        if ad_mode is not None:
            for role, kinematics in (("transmitter", tx), ("receiver", rx)):
                require_leg_ad_mode(
                    f"the Core {role} endpoint position tensor (SensorEndpointIds)",
                    kinematics.positions_m,
                    ad_mode,
                    moves=f"moves every {role} element, an endpoint of the legs",
                )
        transmitter_positions, receiver_positions = tx.positions_m, rx.positions_m
    if receiver_positions.device != device:
        raise ValueError(
            f"radar.tx_pos is on {device} but radar.rx_pos is on {receiver_positions.device}; one radar is one device"
        )
    site_positions, facets = sites.resolve_frame(snapshot, device=device)

    transmitter_ids, receiver_ids, allocated_site_ids = StableIdAllocator().allocate(
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
    receive_vector = (
        polarization_vector if rx_polarization is None else _polarization_tensor(rx_polarization, device=device)
    )
    transmit_power_w = float(radar.power_w)
    receivers = _endpoint_spec(receiver_positions, receiver_ids, polarization=receive_vector, power_w=None)
    return RadarWorldBinding(
        transmitters=_endpoint_spec(
            transmitter_positions, transmitter_ids, polarization=polarization_vector, power_w=transmit_power_w
        ),
        receivers=receivers,
        # The same receivers as 1 W sources (R-ADR-039): one ID, position and
        # polarization table shared by both roles, so the second role uploads
        # nothing.
        receiver_sources=replace(
            receivers,
            powers_w=torch.full(
                (len(receiver_ids),), float(sites.power_w), dtype=torch.float32, device=receiver_positions.device
            ),
        ),
        sites=site_spec(site_positions, tuple(site_ids), polarization=polarization_vector),
        transmitter_ids=transmitter_ids,
        receiver_ids=receiver_ids,
        site_ids=tuple(site_ids),
        facets=facets,
    )


__all__ = ["SensorEndpointIds"]
