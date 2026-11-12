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
from typing import TYPE_CHECKING, Any, Protocol

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

    ``starts[i]`` is where evaluated observation ``i``'s rows begin in
    ``delays``; ``bounds`` says where each sensor pair's block begins inside
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

    @property
    def row_count(self) -> int:
        return int(self.delays.shape[0])


@dataclass(frozen=True, slots=True, eq=False)
class AdaptiveFrame:
    """One adaptive frame's world half: the table, its statistics, and two probes.

    ``first`` and ``last`` are the records of the frame's first and last
    observation instants, which are always probes; the session publishes the
    first one's epoch and the last one's rows.
    """

    table: AdaptiveTable
    stats: dict[str, Any]
    first: ProbeRecord
    last: ProbeRecord


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
        self.max_interval_s = float(motion.max_interval)
        self.max_evaluations = int(motion.max_evaluations)
        self.batch_observations = int(motion.batch_observations)
        self.node_count = int(motion.nodes)
        self.cache: dict[int, ProbeRecord] = {}
        self.pair_tables: dict[int, list[int]] = {}
        self.partitions: list[tuple[int, int, list[int]]] = []
        self.stats: dict[str, Any] = {
            "evaluations": 0,
            "topology_refinements": 0,
            "accepted_intervals": 0,
            "max_tested_phase_error_rad": 0.0,
            "max_tested_relative_amplitude_error": 0.0,
            "max_interval_s": self.max_interval_s,
            "exhaustive": False,
            "topology_proved_complete": False,
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
        self.stats["interpolation_nodes"] = self.node_count

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
                    probes[left, right] = [
                        left + round((right - left) * step / (grid_width - 1)) for step in range(grid_width)
                    ]
            self.ensure(
                [index for group in probes.values() for index in group]
                + [index for left, right in enumerated for index in range(left, right + 1)]
            )
            self.stats["accepted_intervals"] += len(enumerated)
            observations = self._observe(probes)
            pending = self._decide(probes, observations)

    def _same_topology(self, indices: list[int], reference: int) -> bool:
        identity = self.cache[reference].identity
        return all(self.cache[index].identity == identity for index in indices)

    def _observe(self, probes: dict[tuple[int, int], list[int]]) -> dict[_Query, np.ndarray]:
        """Interpolate every tested instant and read the comparison columns once.

        The refinement tests its probes against the same interpolant the echo
        half will use on the accepted partition. That is deliberate: a
        tolerance measured with one rule and spent by another bounds nothing.
        """

        from .paths import interpolate_path_rows

        cache, times = self.cache, self.times
        tested = {
            (left, right): indices
            for (left, right), indices in probes.items()
            if self.within_interval_bound(left, right) and self._same_topology(indices, left)
        }
        queries: list[_Query] = [
            (left, right, index, tuple(indices[::2]))
            for (left, right), indices in tested.items()
            for index in indices[1::2]
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
        delay, transfer = interpolate_path_rows(
            [torch.cat([value.total_delay_s for value in column]) for column in node_paths],
            [torch.cat([value.complex_transfer_ref for value in column]) for column in node_paths],
            torch.as_tensor(np.repeat(basis, counts, axis=0), device=device),
            self.carrier_hz,
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
        self, left: int, right: int, indices: list[int], observations: dict[_Query, np.ndarray]
    ) -> tuple[float, float]:
        """The largest tested phase and relative amplitude errors of one interval."""

        spec, carrier_hz = self.spec, self.carrier_hz
        phase_error = amplitude_error = 0.0
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
                + float(np.max(np.maximum(np.abs(predicted_delay[live]), np.abs(observed_delay[live])), initial=0))
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
            topology_ok = self._same_topology(indices, indices[0])
            phase_error = amplitude_error = 0.0
            accepted = False
            if topology_ok and self.within_interval_bound(left, right):
                phase_error, amplitude_error = self._interval_errors(left, right, indices, observations)
                accepted = phase_error <= self.phase_tolerance and amplitude_error <= self.amplitude_tolerance
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

        times, cache, spec, node_count = self.times, self.cache, self.spec, self.node_count
        ordered = sorted(cache)
        records = [cache[index].composed for index in ordered]
        device = records[0].total_delay_s.device
        delays = torch.cat([record.total_delay_s for record in records])
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
        stats["observation_count"] = len(self.times)


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
        table=table, stats=controller.stats, first=controller.cache[0], last=controller.cache[len(times) - 1]
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
            delay, transfer = interpolate_indexed_rows(samples, node_rows, basis[observation], carrier_hz)
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


__all__ = ["AdaptiveFrame", "AdaptiveTable", "echo_adaptive_frame", "lagrange_weights", "trace_adaptive_frame"]
