"""Adaptive motion sampling: error-controlled probes and their interpolated echo.

The world half (:func:`trace_adaptive_frame`) decides, on the host, which
observation instants to evaluate and which accepted intervals the rest are
interpolated across. The instrument half (:func:`echo_adaptive_frame`) is
where every kernel launch of the route happens; it reads the accepted
partition and decides nothing about the world. The partition is fixed by the
time the echo runs, which is what makes the native interpolant's VJP and JVP
well defined.

The tolerances are the fields of :class:`~witwin.radar.motion.Motion`, read
where they are spent; that record's docstring states what a sampled test can
and cannot bound.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, TypedDict

import numpy as np
import torch

from .motion import Motion

if TYPE_CHECKING:
    from .frontend import FrontendChain
    from .paths import RadarPathBatch
    from .propagation import EpochFrame
    from .synthesis.assembly import FmcwSpec


class ProbeRecord(Protocol):
    """What the controller reads from one evaluated observation."""

    @property
    def epoch_frame(self) -> EpochFrame: ...

    @property
    def composed(self) -> RadarPathBatch: ...

    @property
    def identity(self) -> Hashable: ...


#: Evaluates a batch of instants, one record per instant, in order.
Evaluate = Callable[[list[float]], Sequence[ProbeRecord]]


@dataclass(frozen=True, slots=True, eq=False)
class AdaptiveTable:
    """Every row the adaptive refinement evaluated, and how to read them.

    One row table for the whole frame, plus the host-authored integer maps that
    say which evaluated observations each observation interpolates through.
    ``node_index[i]`` names ``node_count`` evaluated observations and
    ``basis[i]`` their weights, which sum to one; an observation that was
    evaluated itself appears as K copies of itself with the first weight one,
    the same exact answer on a rectangular table without a degenerate basis.

    ``delays`` is the float64 round-trip delay [s] (R-ADR-040), so the
    carrier transport between nodes is exact. ``starts[i]`` is where evaluated
    observation ``i``'s rows begin in ``delays``; ``bounds`` says where each sensor pair's block begins inside
    one such observation and holds ONE ROW PER EVALUATED OBSERVATION, reached
    through ``rank[i]``. Both are indexed by a node, never by the observation
    being interpolated, because only an evaluated observation owns rows.

    The tensors ALIAS the frame's own device storage. This is a hand-off, not a
    retention point, except when a trace keeps it on purpose.
    """

    delays: torch.Tensor
    transfers: torch.Tensor
    validity: torch.Tensor
    starts: np.ndarray
    bounds: np.ndarray
    rank: np.ndarray
    node_index: np.ndarray
    basis: np.ndarray
    clock: np.ndarray
    device: torch.device
    node_count: int
    observation_count: int
    pairs: int
    #: Whether every evaluated row carried the join's float64 delay pair, so
    #: ``delays`` resolves the delay beyond float32 and synthesis reads it so.
    float64_delays: bool = False

    @property
    def row_count(self) -> int:
        return int(self.delays.shape[0])


class AdaptiveDiagnostics(TypedDict):
    """One adaptive frame's probe record, as ``Result.adaptive_diagnostics`` publishes it.

    A plain ``dict`` at run time, so it saves as JSON unchanged. ``complete``
    is Channel's key with Channel's meaning: no path within the requested
    bounds can have been missed in this frame, because every observation was
    evaluated (``exhaustive``) or because the candidate family was certified
    complete for all time (``topology_proved_complete``). It is the frame's
    share of ``Result.path_set_complete``.
    """

    #: World evaluations (probes) the frame spent.
    evaluations: int
    #: Observation instants in the frame's schedule.
    observation_count: int
    #: Intervals refined because a probe saw a different path set.
    topology_refinements: int
    accepted_intervals: int
    #: The largest phase error any accepted interval was tested at, rad.
    max_tested_phase_error_rad: float
    #: The largest relative amplitude error any accepted interval was tested at.
    max_tested_relative_amplitude_error: float
    #: The maximum probe spacing in force, s.
    max_interval_s: float
    #: Instants one accepted interval interpolates through.
    interpolation_nodes: int
    #: Jittered off-grid checks made, and how many of them forced a refinement.
    verification_probes: int
    verification_refinements: int
    exhaustive: bool
    topology_proved_complete: bool
    complete: bool
    #: Replay groups the echo synthesized the frame in; ``0`` before an echo.
    synthesis_batches: int


@dataclass(frozen=True, slots=True, eq=False)
class AdaptiveFrame:
    """One adaptive frame's world half: the table, its statistics, and two probes.

    ``first`` and ``last`` are the records of the frame's first and last
    observation instants, which are always probes; the session publishes the
    first one's epoch and the last one's rows.
    """

    table: AdaptiveTable
    stats: AdaptiveDiagnostics
    first: ProbeRecord
    last: ProbeRecord
    #: Every evaluated probe, by its index in the frame's schedule. Its rows
    #: start at ``table.starts[index]`` of the table.
    probes: dict[int, ProbeRecord]


def lagrange_weights(query_s: np.ndarray, node_s: np.ndarray) -> np.ndarray:
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

    weights = np.ones(node_s.shape)
    for near in range(node_s.shape[1]):
        for far in range(node_s.shape[1]):
            if far != near:
                weights[:, near] *= (query_s - node_s[:, far]) / (node_s[:, near] - node_s[:, far])
    return weights


def _validity(paths: RadarPathBatch) -> torch.Tensor:
    """A probe's row validity. A replayed batch always carries one."""

    if paths.row_valid is None:
        raise ValueError("an adaptive probe must carry row_valid; a replayed batch always publishes it")
    return paths.row_valid


#: One tested instant inside a candidate interval: ``(left, right, index,
#: nodes)``, the interval's ends, the instant tested and its node instants.
_Query = tuple[int, int, int, tuple[int, ...]]

#: The golden ratio's fractional part. Successive multiples of an irrational
#: never land on a dyadic grid, which is what makes the verification instant
#: incommensurate with every level of the nested probe grid.
_GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0

#: Speed of light, m/s, for the acceleration bound on the probe spacing.
_C = 299_792_458.0


def probe_spacing_cap(motion: Motion, carrier_hz: float) -> float:
    """The largest accepted interval the declared motion bounds allow, s.

    ``max_interval`` always applies. A declared ``motion_bandwidth`` ``B``
    caps the grid step at ``1 / (4 B)``, four probes per period of the fastest
    declared component, which with ``2 (nodes - 1)`` steps per interval is
    ``(nodes - 1) / (2 B)``. A declared ``max_acceleration`` ``a`` bounds the
    round-trip delay's curvature by ``2 a / c``; the linear interpolant's
    error over an interval ``h`` is then at most ``a h^2 / (4 c)`` seconds of
    delay, ``pi f_c a h^2 / (2 c)`` radians of carrier phase, and the cap is
    the ``h`` that makes that ``phase_error``. The linear bound is used at
    every order because it is the conservative one.
    """

    cap = float(motion.max_interval)
    if motion.motion_bandwidth is not None:
        cap = min(cap, (int(motion.nodes) - 1) / (2.0 * float(motion.motion_bandwidth)))
    if motion.max_acceleration is not None:
        cap = min(
            cap,
            math.sqrt(2.0 * _C * float(motion.phase_error) / (math.pi * carrier_hz * float(motion.max_acceleration))),
        )
    return cap


#: Channel's ``InteractionType.REFLECTION`` flag in a row's ``interaction_type``.
REFLECTION_INTERACTION = 1

#: Two face planes are one plane group when their unit normals agree to this
#: (component-wise, dimensionless) and their offsets to ``OFFSET_TOLERANCE``
#: of the larger offset plus ``OFFSET_FLOOR_M``. A float32 interaction point
#: carries about ``1e-7`` relative error, so these are three decades above the
#: noise and far below any two distinct walls of one scene.
NORMAL_TOLERANCE = 1e-4
OFFSET_TOLERANCE = 1e-5
OFFSET_FLOOR_M = 1e-6


class PlaneGroups:
    """Which compiled faces lie in one plane, learned from discoveries.

    The adaptive route compares topologies across probes, and a frozen row
    names the TRIANGLE its specular point lies on. A point that slides across
    the shared edge of two coplanar triangles is one continuous reflection,
    but it changes that name, and a subdivided wall then forces refinement at
    every edge. This registry maps a reflecting face to its plane group, so the
    adaptive identity names the plane instead.

    A face's plane is read from the discovery that first reflects off it: the
    interaction normal and point Channel's evaluation returns per row. The
    map is valid only while the compiled geometry does not move, which is why
    the session builds one only for a world without structure motion; a moving
    or deforming structure keeps the triangle names. Non-reflection slots
    (diffraction edges) keep their raw IDs in a separate label space.
    """

    def __init__(self) -> None:
        self._face_group: dict[int, int] = {}
        self._planes = np.zeros((0, 4))

    @property
    def group_count(self) -> int:
        return int(self._planes.shape[0])

    def _group_of(self, plane: np.ndarray) -> int:
        planes = self._planes
        if len(planes):
            normal = np.abs(planes[:, :3] - plane[:3]).max(axis=1) <= NORMAL_TOLERANCE
            offset = np.abs(planes[:, 3] - plane[3]) <= OFFSET_TOLERANCE * max(1.0, abs(plane[3])) + OFFSET_FLOOR_M
            match = np.flatnonzero(normal & offset)
            if len(match):
                return int(match[0])
        self._planes = np.concatenate((planes, plane[None]))
        return len(self._planes) - 1

    def leg_labels(self, leg: Any, primitive: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """``int64[rows, width]``: a leg's primitive sequence with reflecting faces named by plane group.

        A reflecting face becomes ``-2 - group``; any other slot keeps its
        primitive ID (``>= 0``) or the no-interaction ``-1``. One device copy
        reads the leg's interaction types, normals and points.
        """

        labels = np.asarray(primitive, dtype=np.int64).copy()
        types, normals, points = leg.interaction_type, leg.interaction_normals, leg.interaction_positions_m
        rows, width = labels.shape
        if rows == 0 or width == 0 or types is None or normals is None or points is None:
            return labels
        reach = int(normals.shape[1])
        packed = (
            torch.cat((types.to(torch.float32).reshape(-1), normals.reshape(-1), points.reshape(-1)))
            .cpu()
            .double()
            .numpy()
        )
        kind = packed[: rows * width].reshape(rows, width).astype(np.int64)
        normal = packed[rows * width : rows * width + rows * reach * 3].reshape(rows, reach, 3)
        point = packed[rows * width + rows * reach * 3 :].reshape(rows, reach, 3)
        slots = min(width, reach)
        reflecting = np.zeros_like(labels, dtype=bool)
        reflecting[:, :slots] = (
            ((kind[:, :slots] & REFLECTION_INTERACTION) != 0)
            & (labels[:, :slots] >= 0)
            & (np.arange(slots)[None, :] < depth[:, None])
        )
        row, slot = np.nonzero(reflecting)
        faces, first, inverse = np.unique(labels[row, slot], return_index=True, return_inverse=True)
        # One lookup per distinct face, not per row: a face's plane is read
        # from its first reflecting row and never again.
        groups = np.empty(len(faces), dtype=np.int64)
        for index, face in enumerate(faces.tolist()):
            group = self._face_group.get(face)
            if group is None:
                n = normal[row[first[index]], slot[first[index]]]
                n = n / max(float(np.linalg.norm(n)), 1e-30)
                # One orientation per plane: the first clearly non-zero
                # component is positive, whichever side the ray came from.
                lead = np.flatnonzero(np.abs(n) > 1e-6)
                if len(lead) and n[lead[0]] < 0:
                    n = -n
                group = self._group_of(np.concatenate((n, [float(n @ point[row[first[index]], slot[first[index]]])])))
                self._face_group[face] = group
            groups[index] = group
        labels[row, slot] = -2 - groups[inverse.reshape(-1)]
        return labels


def composed_plane_identity(composer: Any, inbound_labels: np.ndarray, outbound_labels: np.ndarray) -> bytes:
    """A join's rows, in composed order, keyed by plane group instead of triangle.

    Equal bytes mean the same rows in the same order with the same planes, so
    row ``r`` of one probe is row ``r`` of the other and interpolating between
    them is interpolating one path.
    """

    def labelled(labels: np.ndarray, rows: np.ndarray, sequences: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # A row with no entry in this leg (``-1``: a site-less row's outbound
        # side, or its inbound side, which is its own leg) keeps the raw
        # primitive IDs the join recorded for it.
        width = sequences.shape[1] // 2
        primitive = sequences[:, :width].copy()
        live = rows >= 0
        gathered = labels[rows[live]]
        primitive[live, : gathered.shape[1]] = gathered
        return primitive, sequences[:, width:]

    in_primitive, in_material = labelled(inbound_labels, composer.inbound_row_host, composer.inbound_sequence_host)
    out_primitive, out_material = labelled(outbound_labels, composer.outbound_row_host, composer.outbound_sequence_host)
    key = np.column_stack(
        (composer.identity_host, in_primitive, in_material, out_primitive, out_material, composer.edge_host)
    ).astype(np.int64)
    return np.asarray(key.shape, dtype=np.int64).tobytes() + key.tobytes()


def verification_instant(left: int, right: int, grid: Sequence[int]) -> int:
    """One schedule index strictly inside ``(left, right)`` and off ``grid``.

    The position is the golden-ratio jitter of the interval's own ends, so it
    is deterministic for a given partition, differs between intervals, and is
    never a point of the nested dyadic probe grid at any level. The caller
    guarantees ``right - left >= len(grid)``, which leaves an interior index
    that is not a grid point.
    """

    taken = set(grid)
    span = right - left - 1
    fraction = (left * _GOLDEN + right * _GOLDEN * _GOLDEN) % 1.0
    start = left + 1 + min(span - 1, int(fraction * span))
    for step in range(span):
        for candidate in (start + step, start - step):
            if left < candidate < right and candidate not in taken:
                return candidate
    raise ValueError(f"interval [{left}, {right}] has no interior instant off its probe grid")


class _Controller:
    """The refinement state of one adaptive frame.

    Host copies here are explicit adaptive decisions, not a differentiable
    physics implementation: one ``tolist`` per probe for its pair partition and
    one column read per refinement round for the error tests. AD follows the
    accepted, fixed partition.
    """

    def __init__(self, times: np.ndarray, evaluate: Evaluate, spec: FmcwSpec, motion: Motion, carrier_hz: float):
        self.times = times
        self.evaluate = evaluate
        self.spec = spec
        self.carrier_hz = carrier_hz
        self.phase_tolerance = float(motion.phase_error)
        self.amplitude_tolerance = float(motion.relative_amplitude_error)
        self.max_interval_s = probe_spacing_cap(motion, carrier_hz)
        self.max_evaluations = int(motion.max_evaluations)
        self.batch_observations = int(motion.batch_observations)
        self.node_count = int(motion.nodes)
        #: Relative resolution of the delays the error test compares, set by
        #: each refinement round from the batches it reads.
        self.delay_resolution = 2.0**-23
        self.cache: dict[int, ProbeRecord] = {}
        self.pair_tables: dict[int, list[int]] = {}
        self.partitions: list[tuple[int, int, list[int]]] = []
        #: The verification instant of each candidate interval, off its grid.
        self.verify: dict[tuple[int, int], int] = {}
        self.stats: AdaptiveDiagnostics = {
            "evaluations": 0,
            "observation_count": len(times),
            "topology_refinements": 0,
            "accepted_intervals": 0,
            "max_tested_phase_error_rad": 0.0,
            "max_tested_relative_amplitude_error": 0.0,
            "max_interval_s": self.max_interval_s,
            "interpolation_nodes": self.node_count,
            "verification_probes": 0,
            "verification_refinements": 0,
            "exhaustive": False,
            "topology_proved_complete": False,
            "complete": False,
            "synthesis_batches": 0,
        }

    def ensure(self, indices: list[int]) -> None:
        """Evaluate every instant in ``indices`` not evaluated yet, in batches."""

        missing = sorted(set(indices) - self.cache.keys())
        if len(self.cache) + len(missing) > self.max_evaluations:
            raise RuntimeError("adaptive motion discovery budget exhausted before satisfying error/topology tests")
        for begin in range(0, len(missing), self.batch_observations):
            batch = missing[begin : begin + self.batch_observations]
            records = self.evaluate([float(self.times[index]) for index in batch])
            for index, record in zip(batch, records, strict=True):
                self.cache[index] = record
                self.pair_tables[index] = record.composed.pair_offsets.tolist()

    def within_interval_bound(self, left: int, right: int) -> bool:
        """The maximum probe spacing, enforced unconditionally.

        Every error test here is a sampled test: it can only see motion at the
        instants it probes. A sinusoid whose period divides the grid spacing
        sits at a zero of every node and every test point, so the controller
        measures no error at all while the interpolant misses the whole
        oscillation. This bound is the only thing that sets how fine that grid
        is, which is why it is not skipped for a topologically certified
        family: the certification says no path can be BORN, and says nothing
        about how fast one moves.
        """

        return self.times[right] - self.times[left] <= self.max_interval_s

    def run(self) -> None:
        times = self.times
        # Start from the coarsest partition the bound allows rather than
        # bisecting down to it: the intervening levels are guaranteed to fail
        # that bound, and their probes answer a question already decided here.
        span = times[-1] - times[0]
        pieces = max(1, math.ceil(span / self.max_interval_s))
        edges = sorted({round(index * (len(times) - 1) / pieces) for index in range(pieces + 1)})
        pending = list(zip(edges, edges[1:], strict=False))

        # A higher polynomial order buys a LONGER accepted interval, so it pays
        # only where the phase test, not the bound above, is what shortens one.
        # Where the bound decides the length the order cannot reduce the
        # interval count and multiplies the per-interval grid instead. The
        # caller owns that trade because only the caller knows which regime the
        # motion is in. One candidate interval spends ``node_count`` instants on
        # the interpolation and tests the error at the instants between them,
        # so its probe grid is this wide and the even positions are the nodes.
        grid_width = 2 * (self.node_count - 1) + 1

        while pending:
            probes: dict[tuple[int, int], list[int]] = {}
            enumerated: list[tuple[int, int]] = []
            for left, right in pending:
                if right - left < grid_width:
                    # Too short to carry distinct nodes. Enumerating it costs
                    # no more probes than testing it would, and an enumerated
                    # observation needs no interpolation at all.
                    enumerated.append((left, right))
                else:
                    grid = [left + round((right - left) * step / (grid_width - 1)) for step in range(grid_width)]
                    probes[left, right] = grid
                    # A sampled test only sees the instants it samples, and the
                    # grid's instants are commensurate at every level: motion
                    # periodic at the grid's step is zero at all of them. One
                    # jittered instant off the grid, interpolated from the
                    # same nodes, is what catches it.
                    self.verify[left, right] = verification_instant(left, right, grid)
            self.ensure(
                [index for group in probes.values() for index in group]
                + [self.verify[key] for key in probes]
                + [index for left, right in enumerated for index in range(left, right + 1)]
            )
            self.stats["verification_probes"] += len(probes)
            self.stats["accepted_intervals"] += len(enumerated)
            observations = self._observe(probes)
            pending = self._decide(probes, observations)

    def _tested(self, left: int, right: int, indices: list[int]) -> list[int]:
        """The instants an interval's error is measured at: the grid's odd points and the jittered one."""

        return [*indices[1::2], self.verify[left, right]]

    def _same_topology(self, indices: list[int], reference: int) -> bool:
        identity = self.cache[reference].identity
        return all(self.cache[index].identity == identity for index in indices)

    def _observe(self, probes: dict[tuple[int, int], list[int]]) -> dict[_Query, np.ndarray]:
        """Interpolate every tested instant and read the comparison columns once.

        The refinement tests its probes against the same interpolant the echo
        half will use on the accepted partition. That is deliberate: a
        tolerance measured with one rule and spent by another bounds nothing.
        """

        from .paths import float64_delay, interpolate_path_rows

        cache, times = self.cache, self.times
        tested = {
            (left, right): indices
            for (left, right), indices in probes.items()
            if self.within_interval_bound(left, right)
            and self._same_topology([*indices, self.verify[left, right]], left)
        }
        queries: list[_Query] = [
            (left, right, index, tuple(indices[::2]))
            for (left, right), indices in tested.items()
            for index in self._tested(left, right, indices)
        ]
        observations: dict[_Query, np.ndarray] = {}
        if not queries:
            return observations
        actual = [cache[index].composed for _, _, index, _ in queries]
        counts = [len(value.total_delay_s) for value in actual]
        node_paths = [[cache[group[node]].composed for *_, group in queries] for node in range(len(queries[0][3]))]
        basis = lagrange_weights(
            np.asarray([times[index] for _, _, index, _ in queries]),
            np.asarray([[times[index] for index in group] for *_, group in queries]),
        )
        device = node_paths[0][0].total_delay_s.device
        # Delays are compared in float64 (R-ADR-040): a float32 delay is
        # quantized to 2 pi f_c ulp32(tau) of carrier phase, which exceeds the
        # default tolerance beyond about 70 m and would make acceptance a
        # matter of rounding luck.
        delay, transfer = interpolate_path_rows(
            [torch.cat([float64_delay(value) for value in column]) for column in node_paths],
            [torch.cat([value.complex_transfer_ref for value in column]) for column in node_paths],
            torch.as_tensor(np.repeat(basis, counts, axis=0), device=device),
            self.carrier_hz,
            float64_delay=True,
        )
        observed = torch.cat([value.complex_transfer_ref for value in actual])
        # Relative resolution of the compared delays: the float32 pair of a
        # composed batch carries tau to ~2^-46 tau; a bare float32 delay to its
        # ulp, at most 2^-23 tau.
        self.delay_resolution = (
            2.0**-46 if all(value.total_delay_residual_s is not None for value in actual) else 2.0**-23
        )
        # One synchronization per refinement round. These detached columns
        # drive host decisions only; production rows retain their AD tape.
        columns = (
            torch.stack(
                [
                    delay,
                    torch.cat([float64_delay(value) for value in actual]),
                    transfer.real.double(),
                    transfer.imag.double(),
                    observed.real.double(),
                    observed.imag.double(),
                    torch.cat([_validity(value) for value in actual]).double(),
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
        return observations

    def _interval_errors(
        self, left: int, right: int, indices: list[int], observations: dict[_Query, np.ndarray], tested: list[int]
    ) -> tuple[float, float]:
        """The largest phase and relative amplitude errors at ``tested`` instants of one interval."""

        spec, carrier_hz = self.spec, self.carrier_hz
        phase_error = amplitude_error = 0.0
        for index in tested:
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
                + float(np.max(np.maximum(np.abs(predicted_delay[live]), np.abs(observed_delay[live])), initial=0))
            )
            # The quantization allowance: what the representation alone can
            # contribute. Two delays at the batch's delay resolution, and two
            # complex64 phases (each within 2^-23.5 rad of exact).
            delay_allowance = (
                2 * np.pi * carrier_hz * 2.0 * self.delay_resolution * float(np.max(np.abs(observed_delay[live])))
            )
            phase_allowance = 2.0**-22
            phase_error = max(
                phase_error,
                float(np.max(2 * np.pi * carrier_hz * np.abs(d[live]), initial=0)) - delay_allowance,
                float(
                    np.max(
                        np.abs(np.angle(predicted[phase_live] * observed[phase_live].conj()))
                        + 2 * np.pi * waveform_frequency * np.abs(d[phase_live]),
                        initial=0,
                    )
                )
                - phase_allowance,
            )
            amplitude_error = max(
                amplitude_error,
                float(
                    np.max(
                        np.abs(np.abs(predicted[live]) - np.abs(observed[live])) / np.maximum(magnitude[live], floor),
                        initial=0,
                    )
                ),
            )
        return phase_error, amplitude_error

    def _decide(
        self, probes: dict[tuple[int, int], list[int]], observations: dict[_Query, np.ndarray]
    ) -> list[tuple[int, int]]:
        """Accept or bisect every probed interval; return the next round."""

        stats = self.stats
        next_pending: list[tuple[int, int]] = []
        for (left, right), indices in probes.items():
            verify = self.verify[left, right]
            topology_ok = self._same_topology([*indices, verify], indices[0])
            phase_error = amplitude_error = 0.0
            accepted = False
            if topology_ok and self.within_interval_bound(left, right):
                phase_error, amplitude_error = self._interval_errors(left, right, indices, observations, indices[1::2])
                grid_ok = phase_error <= self.phase_tolerance and amplitude_error <= self.amplitude_tolerance
                verify_phase, verify_amplitude = self._interval_errors(left, right, indices, observations, [verify])
                phase_error, amplitude_error = max(phase_error, verify_phase), max(amplitude_error, verify_amplitude)
                accepted = phase_error <= self.phase_tolerance and amplitude_error <= self.amplitude_tolerance
                stats["verification_refinements"] += int(grid_ok and not accepted)
            if accepted:
                stats["accepted_intervals"] += 1
                stats["max_tested_phase_error_rad"] = max(stats["max_tested_phase_error_rad"], phase_error)
                stats["max_tested_relative_amplitude_error"] = max(
                    stats["max_tested_relative_amplitude_error"], amplitude_error
                )
                self.partitions.append((left, right, indices[::2]))
            else:
                stats["topology_refinements"] += int(not topology_ok)
                middle = (left + right) // 2
                next_pending.extend([(left, middle), (middle, right)])
        return next_pending

    def table(self) -> AdaptiveTable:
        """Store each discovered row once and author the integer routing maps.

        ADC scheduling gathers these differentiable columns with host-authored
        integer maps; it never allocates one tensor per observation. Ragged row
        counts and empty sensor pairs remain explicit.
        """

        from .paths import float64_delay

        times, cache, spec, node_count = self.times, self.cache, self.spec, self.node_count
        ordered = sorted(cache)
        records = [cache[index].composed for index in ordered]
        device = records[0].total_delay_s.device
        delays = torch.cat([float64_delay(record) for record in records])
        transfers = torch.cat([record.complex_transfer_ref for record in records])
        validity = torch.cat([_validity(record) for record in records])
        starts = np.zeros(len(times), dtype=np.int64)
        starts[ordered] = np.cumsum([0] + [len(record.total_delay_s) for record in records[:-1]])
        pairs = spec.num_tx * spec.num_rx
        # ``pair_offsets`` already IS the exclusive prefix of an observation's
        # rows over its sensor pairs, so pair ``p`` owns rows
        # ``bounds[p]:bounds[p + 1]`` of that observation's block and no
        # cumulative sum is owed here. One row per EVALUATED observation,
        # reached through ``rank``: an interpolated observation reads its
        # interval's nodes and never a row of its own.
        bounds = np.asarray([self.pair_tables[index] for index in ordered])
        rank = np.zeros(len(times), dtype=np.int64)
        rank[ordered] = np.arange(len(ordered))
        # Written one slice per accepted interval, not one row per observation:
        # the partition holds as many distinct answers as it has intervals. A
        # later interval overwrites an earlier one where they meet, which is
        # the accept order, and an interval endpoint is always itself a probe,
        # so the self-reference on the next line settles every shared row.
        node_index = np.zeros((len(times), node_count), dtype=np.int64)
        for left, right, group in self.partitions:
            node_index[left : right + 1] = group
        node_index[ordered] = np.asarray(ordered)[:, None]
        clock = np.asarray(times)
        basis = np.zeros((len(times), node_count))
        basis[:, 0] = 1.0
        interpolated = np.ones(len(times), dtype=bool)
        interpolated[ordered] = False
        if interpolated.any():
            basis[interpolated] = lagrange_weights(clock[interpolated], clock[node_index[interpolated]])
        return AdaptiveTable(
            delays=delays,
            transfers=transfers,
            validity=validity,
            starts=starts,
            bounds=bounds,
            rank=rank,
            node_index=node_index,
            basis=basis,
            clock=clock,
            device=device,
            node_count=node_count,
            observation_count=len(times),
            pairs=pairs,
            float64_delays=all(record.total_delay_residual_s is not None for record in records),
        )

    def close(self) -> None:
        """Publish the two completeness statements' evidence in ``stats``.

        Two independent reasons no path birth can have been missed: every
        observation was evaluated, or the family was certified complete for all
        time. Neither implies the other. The certification is a property of the
        world and the propagation configuration, not of an instant, so one
        probe carrying it proves the family cannot change. The first probe of a
        run cannot carry it - nothing is frozen yet - but it rediscovers, which
        enumerates the family at its own instant. A probe that neither
        certifies nor rediscovers leaves a gap and is refused. A source
        mutation strictly between two probes remains outside every sampled
        test.
        """

        stats, cache = self.stats, self.cache
        stats["evaluations"] = len(cache)
        stats["exhaustive"] = len(cache) == len(self.times)
        frames = [cache[index].epoch_frame for index in cache]
        stats["topology_proved_complete"] = any(frame.topology_complete for frame in frames) and all(
            frame.topology_complete or frame.rediscovered for frame in frames
        )
        stats["complete"] = stats["exhaustive"] or stats["topology_proved_complete"]


def trace_adaptive_frame(
    times: np.ndarray, evaluate: Evaluate, spec: FmcwSpec, motion: Motion, carrier_hz: float
) -> AdaptiveFrame:
    """Control topology probes on the host and publish the accepted partition.

    The world half of the adaptive route. The grid's interior tests bound
    OBSERVED errors only: arbitrarily brief path births or adversarial
    oscillations between probes require the ADC reference, and the two
    completeness statements the session derives from ``stats`` are what say
    so. :func:`echo_adaptive_frame` consumes the table and is where every
    kernel launch happens.
    """

    controller = _Controller(times, evaluate, spec, motion, carrier_hz)
    controller.run()
    table = controller.table()
    controller.close()
    return AdaptiveFrame(
        table=table,
        stats=controller.stats,
        first=controller.cache[0],
        last=controller.cache[len(times) - 1],
        probes=dict(controller.cache),
    )


def echo_adaptive_frame(
    table: AdaptiveTable, spec: FmcwSpec, batch_observations: int, carrier_hz: float, frontend: FrontendChain | None
) -> tuple[torch.Tensor, int]:
    """Interpolate the accepted partition and synthesize it, in bounded batches.

    The instrument half of the adaptive route: every kernel launch of that
    route is here and nothing here decides anything about the world.

    Returns the frame's beat cube and how many batches it took, which is an
    echo quantity: a traced frame that has never been echoed has no batch count
    to report.
    """

    from .paths import interpolate_indexed_rows
    from .synthesis.fmcw import channel_phasor_to_beat_weight, synthesize_adaptive_fmcw, synthesize_fmcw_observations

    delays, transfers, validity = table.delays, table.transfers, table.validity
    node_count, pairs, device = table.node_count, table.pairs, table.device
    observations_total = table.observation_count

    def upload(value: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(value, device=device)

    # An observation sits in one TDM slot and hears one transmitter. Under
    # PAIR_RANK_LAYOUT that transmitter's pairs are the ``num_rx`` ranks
    # congruent to it modulo ``num_tx``; the rest carry no signal this
    # observation can receive and the slot gather at the end discards them.
    # Synthesizing only the active pairs is the same cube for ``1/num_tx`` of
    # the rows, which is the frame's dominant cost.
    receivers = pairs // spec.num_tx
    # An accepted interval's nodes share a topology identity and therefore
    # share their pair layout, so one node's bounds describe every observation
    # the interval covers; only the base offsets in ``starts`` differ.
    node_rank = table.rank[table.node_index[:, 0]][:, None]
    # Upload only compact per-observation routing. This function must not
    # observe device values.
    bounds, basis = upload(table.bounds), upload(table.basis)
    node_starts, owners = upload(table.starts[table.node_index]), upload(node_rank[:, 0])
    samples = torch.stack((delays.double(), transfers.real.double(), transfers.imag.double()), dim=1)
    # The same float64 clock as the row expression, evaluated once per ADC
    # instant instead of once per contributing path.
    adc_clock = (
        spec.t_start_s
        + (torch.arange(observations_total, device=device) % spec.num_samples).double() * spec.sample_period_s
    )
    # Guarded on the phase noise rather than on the front end, because the row
    # phase stage returns its input untouched without it and the timestamp
    # tensor would be uploaded and gathered for nothing.
    phase_noise = frontend is not None and frontend.has_phase_noise
    values: list[torch.Tensor] = []
    if not phase_noise:
        native_validity = validity.to(torch.int32)
        begin = 0
        while begin < observations_total:
            # Bound each output allocation by observations. No effect
            # intervenes between interpolation and dechirping, so the native
            # operator reads compact probes and accumulates ADC values without
            # expanded path rows, including under VJP and JVP.
            stop = min(observations_total, begin + batch_observations * spec.num_samples)
            values.append(
                synthesize_adaptive_fmcw(
                    samples,
                    native_validity,
                    node_starts,
                    basis,
                    bounds,
                    owners,
                    adc_clock,
                    spec,
                    carrier_hz,
                    begin,
                    stop,
                    precise_delay=table.float64_delays,
                ).reshape(stop - begin, receivers)
            )
            begin = stop
    else:
        assert frontend is not None
        clock = upload(table.clock)
        slot_pairs = (np.arange(observations_total) // spec.num_samples % spec.num_tx)[
            :, None
        ] + spec.num_tx * np.arange(receivers)
        row_counts = (table.bounds[node_rank, slot_pairs + 1] - table.bounds[node_rank, slot_pairs]).sum(axis=1)
        # The intervening receiver effect needs expanded rows. Cap its int64
        # indices plus float64 basis at 32 MiB, excluding waveform buffers.
        row_budget = (32 * 1024 * 1024) // (16 * node_count)
        cumulative = np.concatenate(([0], np.cumsum(row_counts)))
        receiver_rank = torch.arange(receivers, device=device)
        begin = 0
        while begin < observations_total:
            stop = min(observations_total, begin + batch_observations * spec.num_samples)
            stop = min(stop, max(begin + 1, int(np.searchsorted(cumulative, cumulative[begin] + row_budget)) - 1))
            # Known from the host's own row totals, which is what lets every
            # ``repeat_interleave`` below declare its output size instead of
            # synchronizing to discover it.
            total = int(cumulative[stop] - cumulative[begin])
            batch = torch.arange(begin, stop, device=device)
            owner = owners[begin:stop, None]
            active = (batch // spec.num_samples % spec.num_tx)[:, None] + spec.num_tx * receiver_rank
            segment_prefix = bounds[owner, active].ravel()
            segment_counts = bounds[owner, active + 1].ravel() - segment_prefix
            offsets = torch.cat((segment_counts.new_zeros(1), torch.cumsum(segment_counts, 0)))
            segment = torch.repeat_interleave(
                torch.arange(segment_counts.numel(), device=device), segment_counts, output_size=total
            )
            local_row = segment_prefix[segment] + torch.arange(total, device=device) - offsets[segment]
            observation = begin + segment // receivers
            node_rows = node_starts[observation] + local_row[:, None]
            delay, transfer = interpolate_indexed_rows(
                samples, node_rows, basis[observation], carrier_hz, float64_delay=table.float64_delays
            )
            transfer = torch.where(validity[node_rows[:, 0]], transfer, torch.zeros_like(transfer))
            transfer = frontend._apply_path_phase_rows(delay, transfer, clock[observation])
            adc_time = adc_clock[observation]
            values.append(
                synthesize_fmcw_observations(
                    delay, channel_phasor_to_beat_weight(transfer), offsets, adc_time, spec, segment=segment
                ).reshape(stop - begin, receivers)
            )
            begin = stop
    # Each slot now carries its own transmitter's receivers, so the gather
    # reads a pair by its receiver rank ``pair // num_tx`` in the slot its
    # transmitter rank ``pair % num_tx`` selects.
    all_slots = torch.cat(values).reshape(-1, spec.num_samples, receivers).transpose(1, 2)
    pair = torch.arange(pairs, device=all_slots.device)
    slot = torch.arange(spec.num_chirps, device=all_slots.device)[:, None] * spec.num_tx + pair[None, :] % spec.num_tx
    return all_slots[slot, pair // spec.num_tx], len(values)


__all__ = [
    "AdaptiveDiagnostics",
    "AdaptiveFrame",
    "AdaptiveTable",
    "echo_adaptive_frame",
    "lagrange_weights",
    "trace_adaptive_frame",
]
