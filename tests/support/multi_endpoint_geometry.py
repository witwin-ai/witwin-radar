"""The frozen multi-endpoint fixture geometry and its float64 closed forms.

The Phase-4/5 fixture is one TX, one site, one RX. That shape cannot produce a
pair with a different row count from its neighbour, cannot produce an empty pair
segment, and cannot make Channel's leg row order disagree with the join's
canonical order. Every multi-pair guarantee in the join was therefore pinned
only against fabricated legs. This module is the geometry that closes that gap
with REAL Channel rows.

It is deliberately NOT an extension of ``phase4_geometry``. The Phase-4 wall
spans ``y in [-3, 3]``, which is wide enough that every specular point in this
endpoint set lands on it; the design knob here is the narrower half-width
``WALL_HALF_Y_M = 1.2``, and sharing constants with a fixture whose numbers must
not move would couple the two.

The whole fixture is one design argument, stated once so the tests can assert
it rather than restate it:

* Mirror a point through ``x = 4``; the single-bounce path from ``a`` to ``b``
  exists exactly when the segment from ``image(a)`` to ``b`` meets the plane
  INSIDE the authored facet.
* ``TX_A -> Q`` misses by 0.4 m and ``Q -> RX_A`` misses by 0.379 m, so those
  two pairs publish a line-of-sight row and no reflection row. That is what
  makes the per-pair row counts differ.
* ``TX_B`` sits at ``x = 6``, behind the facet from both sites, so the wall
  blocks both of its lines of sight; and its image ``(2, -1, 0)`` shares the
  plane ``x = 2`` with both sites, so no specular path exists to find either.
  ``TX_B`` therefore publishes ZERO rows, which is where the empty pair
  segments come from.
* Every endpoint is at least 2 m from the wall plane, so the degenerate
  reflection observation recorded in the appendix of
  ``docs/dev/standards/radar-adr-001-multipath-service-boundary.md`` is not
  engaged by any row here.

Nothing is knife edged: the smallest margin between a specular point and the
facet edge is 0.379 m, which is nine orders of magnitude above a float32 ULP at
these coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass

C0_M_PER_S = 299792458.0

Point = tuple[float, float, float]

# --------------------------------------------------------------------------
# The world
# --------------------------------------------------------------------------

REFERENCE_FREQUENCY_HZ = 77.0e9

WALL_PLANE_X_M = 4.0
# The one design knob. Widening it removes the missing reflection rows; the
# fixture would still run and would silently stop testing anything.
WALL_HALF_Y_M = 1.2
WALL_HALF_Z_M = 3.0
WALL_VERTICES_M = (
    (WALL_PLANE_X_M, -WALL_HALF_Y_M, -WALL_HALF_Z_M),
    (WALL_PLANE_X_M, WALL_HALF_Y_M, -WALL_HALF_Z_M),
    (WALL_PLANE_X_M, WALL_HALF_Y_M, WALL_HALF_Z_M),
    (WALL_PLANE_X_M, -WALL_HALF_Y_M, WALL_HALF_Z_M),
)
WALL_FACES = ((0, 1, 2), (0, 2, 3))
WALL_EPS_R = 5.24
WALL_SIGMA_E = 0.0462

POLARIZATION = (0.0, 0.0, 1.0)

#: The transmitter's radiated power, in watts, and deliberately NOT 1.0.
#:
#: A Channel coefficient carries ``sqrt(tx_power)`` from the source endpoint's
#: ``powers_w``. The outbound leg's source is the SITE, so with a site power
#: equal to the transmitter power the composed ``C_rt`` carries
#: ``sqrt(P_tx) * sqrt(P_site)`` - and with both equal to 1.0 W that second
#: factor is numerically invisible, which is exactly how a squared transmit
#: power ships. Every absolute-level test in Phase 6 therefore runs at
#: ``P_tx != 1 W``.
TX_POWER_W = 0.01

#: The site endpoint's excitation power, in watts, and deliberately EXACTLY 1.0.
#:
#: The site is a re-radiator, not a second transmitter: the entire target
#: strength lives in the two-way join's ``S`` factor, normalised
#: ``S = sqrt(4 pi sigma) / lambda``, and a site excitation of anything but unit
#: power would multiply it again.
SITE_POWER_W = 1.0

TX_A_POSITION_M: Point = (0.0, 0.0, 0.0)
TX_B_POSITION_M: Point = (6.0, -1.0, 0.0)
SITE_P_POSITION_M: Point = (2.0, 0.6, 0.0)
SITE_Q_POSITION_M: Point = (2.0, 2.4, 0.0)
RX_A_POSITION_M: Point = (0.15, 0.0, 0.0)
RX_B_POSITION_M: Point = (0.15, -3.0, 0.0)

TX_A_STABLE_ID = 10
TX_B_STABLE_ID = 11
SITE_P_STABLE_ID = 20
SITE_Q_STABLE_ID = 21
RX_A_STABLE_ID = 30
RX_B_STABLE_ID = 31

# ``TX_B`` moved so that both of its lines to the sites cross ``x = 4`` OUTSIDE
# the facet. Used to prove the occlusion is load bearing rather than an endpoint
# that simply failed for some unrelated reason.
TX_B_UNOCCLUDED_POSITION_M: Point = (6.0, 4.0, 0.0)

# Site P moved along +y. At this position the inbound specular point sits at
# y = 1.3333 and the P -> RX_A one at y = 1.3162, both past the facet edge at
# 1.2, while P -> RX_B survives at y = 0.2906. That kills SOME rows of SOME
# pairs, which is the multi-pair dying-row case.
SITE_P_MOVED_POSITION_M: Point = (2.0, 2.0, 0.0)

# Two sites moving in opposite directions, so the composed Doppler shifts carry
# both signs and no two rows share one.
SITE_P_VELOCITY_M_PER_S: Point = (0.0, 12.0, 0.0)
SITE_Q_VELOCITY_M_PER_S: Point = (0.0, -5.0, 0.0)

TRANSMITTERS: tuple[tuple[int, Point], ...] = (
    (TX_A_STABLE_ID, TX_A_POSITION_M),
    (TX_B_STABLE_ID, TX_B_POSITION_M),
)
SITES: tuple[tuple[int, Point], ...] = (
    (SITE_P_STABLE_ID, SITE_P_POSITION_M),
    (SITE_Q_STABLE_ID, SITE_Q_POSITION_M),
)
RECEIVERS: tuple[tuple[int, Point], ...] = (
    (RX_A_STABLE_ID, RX_A_POSITION_M),
    (RX_B_STABLE_ID, RX_B_POSITION_M),
)

# The same physical sites, declared in the endpoint batch in DESCENDING stable
# ID order. Positions and IDs swap together, so this is the same world seen
# through a different array order - the only thing that makes Channel's leg row
# order disagree with the join's identity order.
SITES_REVERSED: tuple[tuple[int, Point], ...] = tuple(reversed(SITES))

# The frozen column values Channel publishes for this scene. They are observed
# constants of a single-material two-triangle wall, named so a test can say
# WHICH row it is asserting about.
LOS_COMPONENT_ID = 0
REFLECTION_COMPONENT_ID = 1
LOS_INTERACTION_TYPE = 0
REFLECTION_INTERACTION_TYPE = 1
# The compiled material slot of the only material in the scene. Not the
# authored ``material_id`` (which is 1); Channel publishes the compiled index.
REFLECTION_MATERIAL_SLOT = 0
# A row that interacts with nothing carries this sentinel in both sequences.
NO_INTERACTION_SENTINEL = -1

# A two-TX two-RX FMCW front end, so ``sensor_pair_count`` is 4 and the two
# empty segments are visible in the IQ cube's own shape.
FIXTURE_RADAR_CONFIG = {
    "num_tx": 2,
    "num_rx": 2,
    "fc": REFERENCE_FREQUENCY_HZ,
    "slope": 60.012,
    "adc_samples": 256,
    "adc_start_time": 6,
    "sample_rate": 4400,
    "idle_time": 7,
    "ramp_end_time": 58,
    "chirp_per_frame": 8,
    "frame_per_second": 10,
    "num_doppler_bins": 8,
    "num_range_bins": 256,
    "num_angle_bins": 64,
    "power": 12,
    "tx_loc": [[0, 0, 0], [1, 0, 0]],
    "rx_loc": [[0, 0, 0], [1, 0, 0]],
}


# --------------------------------------------------------------------------
# Closed forms
# --------------------------------------------------------------------------


def distance_m(a: Point, b: Point) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b, strict=True)) ** 0.5


def image_position_m(point: Point) -> Point:
    """Mirror a point through the wall plane ``x = WALL_PLANE_X_M``."""

    return (2.0 * WALL_PLANE_X_M - point[0], point[1], point[2])


def reflection_length_m(a: Point, b: Point) -> float:
    """Single-bounce path length via the wall, by image source not by search."""

    return distance_m(image_position_m(a), b)


def specular_point_m(a: Point, b: Point) -> Point | None:
    """Where the single-bounce path from ``a`` to ``b`` meets the wall plane.

    ``None`` when the image source lies in the same ``x`` plane as the target:
    the image-to-target segment is then parallel to the wall and never meets
    it, so no specular path exists at any facet size. That is exactly the
    ``TX_B`` case, and it is a geometric absence rather than a facet miss.
    """

    image = image_position_m(a)
    span = image[0] - b[0]
    if span == 0.0:
        return None
    t = (image[0] - WALL_PLANE_X_M) / span
    return tuple(image[axis] + t * (b[axis] - image[axis]) for axis in range(3))


def _cross_2d(origin, u, v) -> float:
    return (u[0] - origin[0]) * (v[1] - origin[1]) - (u[1] - origin[1]) * (
        v[0] - origin[0]
    )


def face_containing(point: Point, *, tolerance: float = 1.0e-9) -> int | None:
    """Index of the authored triangle containing ``point``, or ``None``.

    The wall is planar in ``x``, so containment is a 2D test in ``(y, z)`` over
    the AUTHORED faces rather than against a rectangle. That matters: the two
    triangles share a diagonal, and which side of it a specular point falls on
    is what makes one reflection row in this fixture carry
    ``primitive_sequence == [1]`` while every other carries ``[0]``. Hard-coding
    the rectangle would make the fixture unable to predict that column at all.

    Strict containment: a point on the shared diagonal belongs to neither
    triangle here rather than being silently assigned to the first. No point in
    this fixture is anywhere near it.
    """

    probe = (point[1], point[2])
    for index, face in enumerate(WALL_FACES):
        a, b, c = (
            (WALL_VERTICES_M[vertex][1], WALL_VERTICES_M[vertex][2])
            for vertex in face
        )
        area = _cross_2d(a, b, c)
        weights = (
            _cross_2d(b, c, probe) / area,
            _cross_2d(c, a, probe) / area,
            _cross_2d(a, b, probe) / area,
        )
        if all(weight > tolerance for weight in weights):
            return index
    return None


def line_of_sight_is_blocked(a: Point, b: Point) -> bool:
    """Does the authored facet sit between ``a`` and ``b``?

    Only a crossing STRICTLY between the two endpoints blocks. Endpoints on the
    same side of the plane never do, which is why ``TX_A`` (at ``x = 0``) sees
    both sites (at ``x = 2``) and ``TX_B`` (at ``x = 6``) sees neither.
    """

    span = b[0] - a[0]
    if span == 0.0:
        return False
    t = (WALL_PLANE_X_M - a[0]) / span
    if not 0.0 < t < 1.0:
        return False
    crossing = tuple(a[axis] + t * (b[axis] - a[axis]) for axis in range(3))
    return face_containing(crossing, tolerance=-1.0e-12) is not None


# --------------------------------------------------------------------------
# Predicted rows
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LegRow:
    """One frozen leg row the fixture geometry says Channel must publish."""

    pair_index: int
    source_row: int
    sink_row: int
    source_id: int
    sink_id: int
    component: str
    length_m: float
    primitive: int
    material: int

    @property
    def component_id(self) -> int:
        return LOS_COMPONENT_ID if self.component == "los" else REFLECTION_COMPONENT_ID

    @property
    def depth(self) -> int:
        return 0 if self.component == "los" else 1

    @property
    def interaction_type(self) -> int:
        return (
            LOS_INTERACTION_TYPE
            if self.component == "los"
            else REFLECTION_INTERACTION_TYPE
        )

    @property
    def delay_s(self) -> float:
        return self.length_m / C0_M_PER_S


def leg_rows(
    sources: tuple[tuple[int, Point], ...],
    sinks: tuple[tuple[int, Point], ...],
) -> list[LegRow]:
    """Every row of one leg, in the order Channel publishes them.

    ``sources`` and ``sinks`` are in ENDPOINT BATCH ROW ORDER, because that is
    what Channel's row order is a function of: ascending ``(pair_index,
    component)`` with ``pair_index = sink_row * source_count + source_row``.
    The join's canonical order is a function of stable IDENTITY instead, and the
    two disagree the moment the batch order disagrees with the ID order. Passing
    the batch order in here rather than a set of endpoints is what lets a test
    see that.
    """

    rows: list[LegRow] = []
    for sink_row, (sink_id, sink_position) in enumerate(sinks):
        for source_row, (source_id, source_position) in enumerate(sources):
            pair_index = sink_row * len(sources) + source_row
            if not line_of_sight_is_blocked(source_position, sink_position):
                rows.append(
                    LegRow(
                        pair_index=pair_index,
                        source_row=source_row,
                        sink_row=sink_row,
                        source_id=source_id,
                        sink_id=sink_id,
                        component="los",
                        length_m=distance_m(source_position, sink_position),
                        primitive=NO_INTERACTION_SENTINEL,
                        material=NO_INTERACTION_SENTINEL,
                    )
                )
            specular = specular_point_m(source_position, sink_position)
            face = None if specular is None else face_containing(specular)
            if face is not None:
                rows.append(
                    LegRow(
                        pair_index=pair_index,
                        source_row=source_row,
                        sink_row=sink_row,
                        source_id=source_id,
                        sink_id=sink_id,
                        component="reflection",
                        length_m=reflection_length_m(source_position, sink_position),
                        primitive=face,
                        material=REFLECTION_MATERIAL_SLOT,
                    )
                )
    return rows


def pair_offsets(rows: list[LegRow], pair_count: int) -> list[int]:
    """The CSR offsets the predicted rows imply, empty segments included."""

    counts = [0] * pair_count
    for row in rows:
        counts[row.pair_index] += 1
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + count)
    return offsets


@dataclass(frozen=True, slots=True)
class CombinedRow:
    """One composed round trip, in the join's canonical order."""

    sensor_pair_rank: int
    site_rank: int
    source_id: int
    site_id: int
    sink_id: int
    inbound: LegRow
    outbound: LegRow

    @property
    def total_delay_s(self) -> float:
        return self.inbound.delay_s + self.outbound.delay_s

    @property
    def key(self) -> tuple[int, int, int, str, str]:
        """A frame-invariant name for this composed row."""

        return (
            self.source_id,
            self.site_id,
            self.sink_id,
            self.inbound.component,
            self.outbound.component,
        )


def _leg_key(row: LegRow) -> tuple[int, int, tuple[int, ...], tuple[int, ...]]:
    return (row.component_id, row.depth, (row.primitive,), (row.material,))


def combined_rows(
    transmitters: tuple[tuple[int, Point], ...] = TRANSMITTERS,
    sites: tuple[tuple[int, Point], ...] = SITES,
    receivers: tuple[tuple[int, Point], ...] = RECEIVERS,
) -> list[CombinedRow]:
    """Predict the composed frame, independently of the join implementation.

    Built from the analytic image-source table plus the two ordering rules the
    join declares: the sensor pair rank is SINK-MAJOR over the declared source
    and sink list order, and the site rank is the rank in the SORTED site ID
    list. Within a ``(pair, site)`` cell the order is by leg identity key.

    This is an oracle rather than a mirror: it never touches
    ``witwin.radar.paths._identity``, and it derives which rows exist from the
    facet geometry rather than from anything Channel published.
    """

    inbound_rows = leg_rows(transmitters, sites)
    outbound_rows = leg_rows(sites, receivers)
    source_ids = [stable_id for stable_id, _ in transmitters]
    sink_ids = [stable_id for stable_id, _ in receivers]
    site_ids = sorted(stable_id for stable_id, _ in sites)

    ranked: list[tuple[tuple, CombinedRow]] = []
    for site_rank, site_id in enumerate(site_ids):
        for source_rank, source_id in enumerate(source_ids):
            for sink_rank, sink_id in enumerate(sink_ids):
                pair_rank = sink_rank * len(source_ids) + source_rank
                for inbound in inbound_rows:
                    if inbound.source_id != source_id or inbound.sink_id != site_id:
                        continue
                    for outbound in outbound_rows:
                        if (
                            outbound.source_id != site_id
                            or outbound.sink_id != sink_id
                        ):
                            continue
                        ranked.append(
                            (
                                (
                                    pair_rank,
                                    site_rank,
                                    _leg_key(inbound),
                                    _leg_key(outbound),
                                ),
                                CombinedRow(
                                    sensor_pair_rank=pair_rank,
                                    site_rank=site_rank,
                                    source_id=source_id,
                                    site_id=site_id,
                                    sink_id=sink_id,
                                    inbound=inbound,
                                    outbound=outbound,
                                ),
                            )
                        )
    ranked.sort(key=lambda entry: entry[0])
    return [row for _, row in ranked]


def combined_pair_offsets(
    rows: list[CombinedRow], *, sensor_pair_count: int
) -> list[int]:
    counts = [0] * sensor_pair_count
    for row in rows:
        counts[row.sensor_pair_rank] += 1
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + count)
    return offsets


def _leg_delay_rate_s_per_s(
    site: Point, other: Point, component: str, velocity: Point
) -> float:
    """``d(delay)/dt`` of one leg for a moving SITE, the other endpoint static.

    One formula covers both components, which is the point of the image source:
    the rate is the projection of the site velocity onto the unit vector from
    the (possibly mirrored) fixed endpoint to the site. It is symmetric in which
    end of the leg the site is, because ``|image(a) - b| == |a - image(b)|``.
    """

    origin = other if component == "los" else image_position_m(other)
    offset = tuple(site[axis] - origin[axis] for axis in range(3))
    length = sum(value**2 for value in offset) ** 0.5
    projection = sum(offset[axis] * velocity[axis] for axis in range(3))
    return projection / length / C0_M_PER_S


def combined_doppler_hz(
    rows: list[CombinedRow],
    velocities: dict[int, Point],
    transmitters: tuple[tuple[int, Point], ...] = TRANSMITTERS,
    sites: tuple[tuple[int, Point], ...] = SITES,
    receivers: tuple[tuple[int, Point], ...] = RECEIVERS,
) -> list[float]:
    """Doppler shift of each composed row, in the same order as ``rows``.

    ``f_D = -f_c * d(tau_rt)/dt``: the sign follows Channel's ``exp(-j k d)``
    phasor, so a receding site gives a negative shift.
    """

    positions = {
        stable_id: position
        for stable_id, position in (*transmitters, *sites, *receivers)
    }
    shifts: list[float] = []
    for row in rows:
        site = positions[row.site_id]
        velocity = velocities[row.site_id]
        rate = _leg_delay_rate_s_per_s(
            site, positions[row.source_id], row.inbound.component, velocity
        ) + _leg_delay_rate_s_per_s(
            site, positions[row.sink_id], row.outbound.component, velocity
        )
        shifts.append(-REFERENCE_FREQUENCY_HZ * rate)
    return shifts


def combined_delay_gradient_s_per_m(
    rows: list[CombinedRow],
    weights: list[float],
    transmitters: tuple[tuple[int, Point], ...] = TRANSMITTERS,
    sites: tuple[tuple[int, Point], ...] = SITES,
    receivers: tuple[tuple[int, Point], ...] = RECEIVERS,
) -> dict[int, Point]:
    """``d/d(site position)`` of ``sum_r weights[r] * tau_rt[r]``, per site ID.

    The reverse-mode transpose of the rate in :func:`combined_doppler_hz`: each
    leg contributes the unit vector from its (possibly mirrored) fixed endpoint
    to the site, divided by ``c``, and a site accumulates over EVERY composed
    row that reaches it - both legs of each. Evaluating the same rate function
    along the three unit basis directions states that relationship rather than
    restating the projection, and keeps this closed form tied to the one the
    forward-mode Doppler test already validates against Channel.

    Nothing here re-derives a specular point: by Fermat the derivative of a
    single-bounce path length with respect to an ENDPOINT is the unit vector
    along that endpoint's segment, so the moving stationary point contributes
    no first-order term.
    """

    positions = {
        stable_id: position
        for stable_id, position in (*transmitters, *sites, *receivers)
    }
    gradient = {stable_id: [0.0, 0.0, 0.0] for stable_id, _ in sites}
    for row, weight in zip(rows, weights, strict=True):
        site = positions[row.site_id]
        for axis in range(3):
            direction = tuple(
                1.0 if index == axis else 0.0 for index in range(3)
            )
            gradient[row.site_id][axis] += weight * (
                _leg_delay_rate_s_per_s(
                    site,
                    positions[row.source_id],
                    row.inbound.component,
                    direction,
                )
                + _leg_delay_rate_s_per_s(
                    site,
                    positions[row.sink_id],
                    row.outbound.component,
                    direction,
                )
            )
    return {
        stable_id: tuple(value) for stable_id, value in gradient.items()
    }


__all__ = [
    "C0_M_PER_S",
    "CombinedRow",
    "FIXTURE_RADAR_CONFIG",
    "LOS_COMPONENT_ID",
    "LOS_INTERACTION_TYPE",
    "LegRow",
    "NO_INTERACTION_SENTINEL",
    "POLARIZATION",
    "RECEIVERS",
    "REFERENCE_FREQUENCY_HZ",
    "REFLECTION_COMPONENT_ID",
    "REFLECTION_INTERACTION_TYPE",
    "REFLECTION_MATERIAL_SLOT",
    "RX_A_POSITION_M",
    "RX_A_STABLE_ID",
    "RX_B_POSITION_M",
    "RX_B_STABLE_ID",
    "SITES",
    "SITES_REVERSED",
    "SITE_P_MOVED_POSITION_M",
    "SITE_P_POSITION_M",
    "SITE_P_STABLE_ID",
    "SITE_P_VELOCITY_M_PER_S",
    "SITE_Q_POSITION_M",
    "SITE_Q_STABLE_ID",
    "SITE_Q_VELOCITY_M_PER_S",
    "TRANSMITTERS",
    "TX_A_POSITION_M",
    "TX_A_STABLE_ID",
    "TX_B_POSITION_M",
    "TX_B_STABLE_ID",
    "TX_B_UNOCCLUDED_POSITION_M",
    "SITE_POWER_W",
    "TX_POWER_W",
    "WALL_EPS_R",
    "WALL_FACES",
    "WALL_HALF_Y_M",
    "WALL_HALF_Z_M",
    "WALL_PLANE_X_M",
    "WALL_SIGMA_E",
    "WALL_VERTICES_M",
    "combined_delay_gradient_s_per_m",
    "combined_doppler_hz",
    "combined_pair_offsets",
    "combined_rows",
    "distance_m",
    "face_containing",
    "image_position_m",
    "leg_rows",
    "line_of_sight_is_blocked",
    "pair_offsets",
    "reflection_length_m",
    "specular_point_m",
]
