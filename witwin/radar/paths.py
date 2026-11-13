"""Radar round-trip path composition.

:class:`TwoWayComposer` joins an inbound and an outbound leg through a scatter
site and publishes the :class:`RadarPathBatch` contract.

:class:`RadarComponentIndex` is a second thing and it is a SIDECAR: it names
what each composed row is - target echo, environment clutter, direct leakage,
multi-interaction - without adding a column to :class:`RadarPathTopology`.
Every component export therefore shares the same topology OBJECT, which is what
makes "processing does not change propagation row identity" a checkable
statement rather than a claim.

The module exports the contracts and the composer. It does not import the
Channel adapter; the composer duck-types the frozen leg handles it is given, so
this package never crosses the Channel boundary either.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from .cuda import native_ops as _ops
from .policy import first_order_only, refuse_derivative
from .propagation import FrozenLeg, RadarLegBatch, require_tensor

if TYPE_CHECKING:
    from .scattering import ScatterResponse

LegKey = tuple[Any, ...]

#: The ``site_id`` of a composed row that visits no scatter site.
NO_SITE = -1


def _present_member(value: torch.Tensor | None, name: str) -> torch.Tensor:
    if value is None:
        raise ValueError(f"a leg batch composed with segment targets must carry {name}")
    return value


class _PathInterpolation(torch.autograd.Function):
    """First-order native interpolation at a fixed adaptive time partition."""

    @staticmethod
    def forward(values: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor, carrier: float) -> torch.Tensor:
        rows = len(indices) if indices.ndim == 2 else len(values)
        result = values.new_empty((rows, 3))
        _ops().path_interpolate_forward(values, indices, weights, values.new_empty(0), result, carrier)
        return result

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: torch.Tensor) -> None:
        values, indices, weights, ctx.carrier = inputs
        ctx.save_for_backward(values, indices, weights)
        ctx.save_for_forward(values, indices, weights)

    @staticmethod
    @first_order_only
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        values, indices, weights = ctx.saved_tensors
        indexed = indices.ndim == 2
        result = values.new_empty((len(indices), 4 * indices.shape[1])) if indexed else torch.empty_like(values)
        _ops().path_interpolate_backward(values, indices, weights, grad.contiguous(), result, ctx.carrier)
        if indexed:
            # Sum cotangents from every observation reading the same probe.
            # Repeated node indices are intentional (exactly sampled queries).
            result = torch.zeros_like(values).index_add(0, indices.reshape(-1), result.reshape(-1, 4)[:, :3])
        return result, None, None, None

    @staticmethod
    def jvp(
        ctx: Any, tangent: torch.Tensor, indices_tangent: Any, weights_tangent: Any, carrier_tangent: Any
    ) -> torch.Tensor:
        values, indices, weights = ctx.saved_tensors
        rows = len(indices) if indices.ndim == 2 else len(values)
        result = values.new_empty((rows, 3))
        _ops().path_interpolate_jvp(values, indices, weights, tangent.contiguous(), result, ctx.carrier)
        return result


def interpolate_path_rows(
    delays: Sequence[torch.Tensor], transfers: Sequence[torch.Tensor], weights: torch.Tensor, carrier_hz: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transport Channel phasors to the interpolated delay, then blend envelopes.

    ``delays`` and ``transfers`` are ``K`` sequences of per-row node samples of
    ONE topologically identical path; ``weights`` is ``[rows, K]``, the caller's
    interpolation basis at the query time. Two nodes with ``(1 - alpha, alpha)``
    is the linear rule; more nodes raise the polynomial order without changing
    the contract. The weights must sum to one so that a query landing exactly
    on a node reproduces it.

    Topology equality and the choice of nodes are the caller's obligations. The
    weights are a fixed, dimensionless consequence of a discrete time-grid
    decision and are not differentiated. Delays [s] and complex transfers
    retain native first-order derivatives.
    """

    refuse_derivative("adaptive time partition", "time-grid decisions are discrete", weights=weights)
    if len(delays) != len(transfers) or len(delays) < 2:
        raise ValueError(f"interpolation needs at least two aligned node samples, got {len(delays)}/{len(transfers)}")
    if weights.shape[1] != len(delays):
        raise ValueError(f"weights carry {weights.shape[1]} nodes for {len(delays)} node samples")
    columns = []
    for index, (delay, transfer) in enumerate(zip(delays, transfers, strict=True)):
        columns += [delay.double(), transfer.real.double(), transfer.imag.double(), weights[:, index].double()]
    return _interpolate_packed_rows(torch.stack(columns, dim=1).contiguous(), carrier_hz)


def _interpolate_packed_rows(values: torch.Tensor, carrier_hz: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Native interpolation over repeated [delay, real, imaginary, basis] columns.

    Used by the adaptive controller's small error probes. Full observation
    synthesis uses indexed compact samples, through the same native equation.
    """
    result = _PathInterpolation.apply(
        values, torch.empty(0, device=values.device, dtype=torch.int64), values.new_empty(0), carrier_hz
    )
    return result[:, 0].float(), torch.complex(result[:, 1].float(), result[:, 2].float())


def interpolate_indexed_rows(
    samples: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor, carrier_hz: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather compact probe triples inside the interpolation kernel.

    ``samples`` stores float64 (delay [s], real C, imaginary C) per probe path;
    ``indices`` and the fixed Lagrange ``weights`` are [observations*paths, K].
    The same native transport equation handles both indexed and packed rows.
    """
    refuse_derivative("adaptive time partition", "time-grid decisions are discrete", weights=weights)
    result = _PathInterpolation.apply(samples, indices, weights, carrier_hz)
    return result[:, 0].float(), torch.complex(result[:, 1].float(), result[:, 2].float())


def stable_ids(values: torch.Tensor | Sequence[int], name: str) -> list[int]:
    """Normalize a stable-ID sequence to a host list of distinct ints."""

    if isinstance(values, torch.Tensor):
        if values.ndim != 1:
            raise ValueError(f"{name} must be a 1-D sequence of stable IDs")
        listed = [int(value) for value in values.tolist()]
    else:
        listed = [int(value) for value in values]
    if not listed:
        raise ValueError(f"{name} must not be empty")
    if len(set(listed)) != len(listed):
        raise ValueError(f"{name} must not repeat a stable ID, got {listed}")
    return listed


@dataclass(frozen=True, slots=True, eq=False)
class LegIdentity:
    """One frozen leg's row identity on the host, read in one device copy.

    ``source`` and ``sink`` are the rows' stable endpoint IDs; ``component``,
    ``depth``, ``primitive`` (``[rows, width]``) and ``material`` complete the
    key that distinguishes two rows of the SAME leg. Every column is frame
    invariant, so the composed order it induces is frame invariant too.

    ``edge`` is the diffracting edge of each row, read only for a leg that
    requested ``diffraction``: a diffracted row interacts with an edge rather
    than a face, so its primitive sequence does not tell two edges apart. It
    is appended last, so it only breaks what would otherwise be a tie.

    ``packed`` is the whole identity as one bytes object - the row count and
    sequence width followed by every column - so two legs have the same
    identity exactly when their ``packed`` values are equal. It is what an
    epoch is compared and grouped by; nothing hashes or compares row tuples.
    """

    source: np.ndarray
    sink: np.ndarray
    component: np.ndarray
    depth: np.ndarray
    primitive: np.ndarray
    material: np.ndarray
    packed: bytes
    edge: np.ndarray | None = None

    @property
    def row_count(self) -> int:
        return int(self.source.shape[0])

    def key_columns(self) -> np.ndarray:
        """``int64[rows, 2 + 2 * width]``: the key as columns, compared left to right.

        Lexicographic order over these columns is the order of the key tuple
        ``(component, depth, primitive, material)``, because every row of one
        leg carries sequences of the same width.
        """

        columns = (self.component, self.depth, self.primitive, self.material)
        return np.column_stack(columns if self.edge is None else (*columns, self.edge))

    def key(self, row: int) -> LegKey:
        key: LegKey = (
            int(self.component[row]),
            int(self.depth[row]),
            tuple(int(value) for value in self.primitive[row]),
            tuple(int(value) for value in self.material[row]),
        )
        return key if self.edge is None else (*key, int(self.edge[row]))


def read_leg_identity(frozen: FrozenLeg, name: str) -> LegIdentity:
    """Read one frozen leg's row identity to the host with one device copy.

    The columns are concatenated on the device and copied once, rather than
    once each. Their shapes are host metadata and are checked before the copy.
    """

    rows = int(frozen.source_id.shape[0])
    width = int(frozen.primitive_sequence.shape[1]) if frozen.primitive_sequence.dim() == 2 else -1
    for label, column, shape in (
        ("sink_id", frozen.sink_id, (rows,)),
        ("component_id", frozen.component_id, (rows,)),
        ("depth", frozen.depth, (rows,)),
        ("primitive_sequence", frozen.primitive_sequence, (rows, width)),
        ("material_sequence", frozen.material_sequence, (rows, width)),
    ):
        if tuple(column.shape) != shape:
            raise ValueError(f"{name} leg {label} has shape {tuple(column.shape)}, expected {shape}")
    edge_column = getattr(frozen, "edge_id", None) if "diffraction" in getattr(frozen, "requested", ()) else None
    columns = [
        frozen.source_id.to(torch.int64),
        frozen.sink_id.to(torch.int64),
        frozen.component_id.to(torch.int64),
        frozen.depth.to(torch.int64),
        frozen.primitive_sequence.to(torch.int64).reshape(-1),
        frozen.material_sequence.to(torch.int64).reshape(-1),
    ]
    if edge_column is not None:
        columns.append(edge_column.to(torch.int64).reshape(rows))
    packed = torch.cat(columns).cpu().numpy()
    sequence = rows * width
    end = 4 * rows + 2 * sequence
    return LegIdentity(
        source=packed[:rows],
        sink=packed[rows : 2 * rows],
        component=packed[2 * rows : 3 * rows],
        depth=packed[3 * rows : 4 * rows],
        primitive=packed[4 * rows : 4 * rows + sequence].reshape(rows, width),
        material=packed[4 * rows + sequence : end].reshape(rows, width),
        packed=np.asarray((rows, width), dtype=np.int64).tobytes() + packed.tobytes(),
        edge=None if edge_column is None else packed[end:],
    )


def _dense_rank(columns: np.ndarray) -> np.ndarray:
    """``int64[rows]``: each row's rank in the lexicographic order of its columns, ties equal."""

    rows = columns.shape[0]
    if rows == 0:
        return np.zeros(0, dtype=np.int64)
    order = np.lexsort(columns.T[::-1])
    ordered = columns[order]
    step = np.ones(rows, dtype=np.int64)
    step[1:] = np.any(ordered[1:] != ordered[:-1], axis=1)
    step[0] = 0
    rank = np.empty(rows, dtype=np.int64)
    rank[order] = np.cumsum(step)
    return rank


def _require_unique_keys(leg: LegIdentity, key_rank: np.ndarray, name: str) -> None:
    """Refuse two rows of one endpoint pair that share an identity key.

    A collision would make the canonical composed order ambiguous and would
    silently turn the permutation test vacuous, so it is refused rather than
    tie-broken on row position - which is exactly the positional dependence
    this module exists to remove. The message names the earliest row that
    repeats a key and the row that first claimed it.
    """

    if leg.row_count < 2:
        return
    columns = np.column_stack((leg.source, leg.sink, key_rank, np.arange(leg.row_count)))
    order = np.lexsort(columns.T[::-1])
    ordered = columns[order]
    repeated = np.all(ordered[1:, :3] == ordered[:-1, :3], axis=1)
    if not repeated.any():
        return
    second = ordered[1:][repeated]
    row = int(second[:, 3].min())
    first = ordered[
        (ordered[:, 0] == leg.source[row]) & (ordered[:, 1] == leg.sink[row]) & (ordered[:, 2] == key_rank[row])
    ]
    claimed = int(first[:, 3].min())
    raise ValueError(
        f"{name} leg rows {claimed} and {row} share the identity key {leg.key(row)} within endpoint pair "
        f"{(int(leg.source[row]), int(leg.sink[row]))}; the composed order would be ambiguous"
    )


def _ranks(values: np.ndarray, declared: list[int]) -> np.ndarray:
    """Each value's position in ``declared``; every value must be declared."""

    ids = np.asarray(declared, dtype=np.int64)
    order = np.argsort(ids, kind="stable")
    return order[np.searchsorted(ids[order], values)]


def sink_major_rank(sources: list[int], sinks: list[int]) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """The sensor-pair index, mirroring the Channel consumer's convention.

    Channel computes ``sink_row_index * source_count + source_row_index`` and
    spans ``source_count * sink_count``. Using anything else here would put a
    second, silently different, virtual-array numbering on the same data. The
    returned rule maps arrays of declared source and sink IDs to pair ranks.
    """

    def rank(source: np.ndarray, sink: np.ndarray) -> np.ndarray:
        return _ranks(sink, sinks) * len(sources) + _ranks(source, sources)

    return rank


def require_round_trip_budget(
    inbound_source: np.ndarray,
    inbound_site: np.ndarray,
    outbound_sink: np.ndarray,
    outbound_site: np.ndarray,
    *,
    site_count: int,
    sources: list[int],
    sinks: list[int],
    direct: tuple[np.ndarray, np.ndarray] | None,
    budget: int,
) -> None:
    """Refuse a join whose composed rows for some sensor pair would exceed ``budget``.

    The composed row count of pair ``(t, r)`` is ``sum_s A[s, t] * B[s, r]``,
    where ``A[s, t]`` counts inbound rows from transmitter ``t`` to site ``s``
    and ``B[s, r]`` outbound rows from site ``s`` to receiver ``r`` - one small
    host matrix product over per-site counts, so the cross join it bounds is
    never materialised to be measured. Direct rows add their own per-pair
    count. Host-only; reads no device value.
    """

    if type(budget) is not int or budget < 1:
        raise ValueError(f"max_round_trips must be a positive int, got {budget!r}")
    num_tx, num_rx = len(sources), len(sinks)
    arriving = np.bincount(
        inbound_site * num_tx + _ranks(inbound_source, sources), minlength=site_count * num_tx
    ).reshape(site_count, num_tx)
    leaving = np.bincount(outbound_site * num_rx + _ranks(outbound_sink, sinks), minlength=site_count * num_rx).reshape(
        site_count, num_rx
    )
    # rows[r, t]: sink major, the same rank law as the composed partition.
    rows = (arriving.T.astype(np.int64) @ leaving.astype(np.int64)).T
    if direct is not None:
        pair = sink_major_rank(sources, sinks)(direct[0], direct[1])
        rows = rows + np.bincount(pair, minlength=num_tx * num_rx).reshape(num_rx, num_tx)
    worst = int(rows.argmax())
    count = int(rows.reshape(-1)[worst])
    if count > budget:
        receiver, transmitter = divmod(worst, num_tx)
        raise ValueError(
            f"max_round_trips={budget} is exceeded: sensor pair (transmitter {sources[transmitter]}, receiver "
            f"{sinks[receiver]}) would compose {count} round-trip rows "
            f"({int((rows > budget).sum())} of {num_tx * num_rx} pairs exceed the budget). Each composed row is "
            "one inbound row times one outbound row through the same site, so reduce reflections, the site "
            "count or the scene's facets, or raise max_round_trips"
        )


def pair_offsets(pair_of_row: np.ndarray | Sequence[int], pair_count: int) -> np.ndarray:
    """A half-open partition of composed rows by sensor pair.

    Empty segments are legal and expected: the partition spans the FRONT END's
    cross product, so a pair that discovered nothing still owns a segment and
    the IQ cube keeps its declared shape.

    The kernel CLAMPS a malformed offsets table rather than failing, because
    reading its values per frame would be exactly the D2H the fixed-topology
    capability exists to avoid, and clamping turns a malformed table into a
    plausible wrong answer. So the gate lives here, at freeze time, where the
    table is still on the host and checking it is free. What is actually
    checkable is the input: once every row's pair rank is in range, counting
    them can only produce a valid partition.
    """

    if pair_count < 1:
        raise ValueError(f"pair_count must be positive, got {pair_count}")
    pair_of_row = np.asarray(pair_of_row, dtype=np.int64)
    outside = np.flatnonzero((pair_of_row < 0) | (pair_of_row >= pair_count))
    if outside.size:
        row = int(outside[0])
        raise ValueError(
            f"composed row {row} claims sensor pair {int(pair_of_row[row])}, which is "
            f"outside the declared {pair_count} pairs; the offsets table "
            "would not partition all composed rows"
        )
    return np.concatenate(([0], np.cumsum(np.bincount(pair_of_row, minlength=pair_count)))).astype(np.int64)


def csr(owner_of_row: np.ndarray, owner_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Group composed rows by an owner index, as a CSR offsets/rows pair.

    The VJP needs this: one thread owns one gradient slot and loops its own
    segment, so the reduction needs no atomics and its summation order is fixed
    by the frozen join. Within one owner the rows stay in composed order, which
    is what makes a bit-identical gradient comparison across a leg permutation
    a legitimate assertion rather than a lucky one.
    """

    rows = np.argsort(owner_of_row, kind="stable").astype(np.int64)
    offsets = np.concatenate(([0], np.cumsum(np.bincount(owner_of_row, minlength=owner_count)))).astype(np.int64)
    return offsets, rows


@dataclass(frozen=True, slots=True, eq=False)
class RadarPathTopology:
    """The identity of each composed round-trip row.

    The tuple ``(radar_source_id, site_id, radar_sink_id)`` is the row's
    identity and is stable across a frozen sequence. ``inbound_row`` and
    ``outbound_row`` record which frozen leg rows were joined, so a composed
    result can always be traced back to the two legs that produced it.

    Identity is what the join uses. Joining by array position instead would be
    silently wrong the moment a leg publishes its rows in a different order,
    and the resulting error looks like a physics bug rather than a bookkeeping
    one.

    A SITE-LESS row - transmitter straight to receiver, through the third leg -
    carries ``site_id == NO_SITE`` and ``inbound_row == outbound_row == -1``,
    and ``direct_row`` names its row of that leg. ``direct_row`` is ``None``
    when the join composed no third leg and ``-1`` on every site row when it
    did.
    """

    radar_source_id: torch.Tensor
    site_id: torch.Tensor
    radar_sink_id: torch.Tensor
    inbound_row: torch.Tensor
    outbound_row: torch.Tensor
    direct_row: torch.Tensor | None = None

    def __post_init__(self) -> None:
        rows = (int(self.radar_source_id.shape[0]),)
        for name in ("radar_source_id", "site_id", "radar_sink_id", "inbound_row", "outbound_row"):
            require_tensor(name, getattr(self, name), dtype=torch.int64, shape=rows)
        if self.direct_row is not None:
            require_tensor("direct_row", self.direct_row, dtype=torch.int64, shape=rows)

    @property
    def row_count(self) -> int:
        return int(self.radar_source_id.shape[0])


@dataclass(frozen=True, slots=True, eq=False)
class RadarPathBatch:
    """Composed round-trip rows ready for waveform synthesis.

    ``complex_transfer_ref`` is published in the CHANNEL phasor convention,
    ``exp(-j * k * d)`` with ``exp(+j * 2 * pi * f * t)`` time dependence,
    at ``reference_frequency_hz``. It is NOT a beat weight. FMCW de-chirping
    conjugates the received phasor, and that conversion has exactly one call
    site, in the synthesis facade.

    ``delay_rate`` is ``d(total_delay_s)/dt`` in s/s, primal-valued, and
    ``None`` on every batch the composer publishes: the delay rate of a moving
    world comes from re-evaluating the propagation at each observation instant,
    not from a rate carried on the row. The member exists because the
    frozen-weight synthesis kernels take a rate input, and a caller that models
    slow time with one supplies it.

    ``row_valid`` is the sole authority on whether a row means anything. A
    dead row is a complete answer contributing exactly zero, never an error,
    and validity is never inferred from a zero payload.

    ``weight_includes_antenna_pattern`` is the fourth provenance boolean of the
    pipeline and the only one the composer does not already know. A composed
    weight is Channel-sourced, so it always carries the reference-frequency
    phase, the free-space spreading and the transmit power; it carries the
    ARRAY's transmit and receive pattern gain only after
    :meth:`witwin.radar.sensors.RoundTripPatternStage.apply` has run. The
    composer publishes ``False`` because it applies no pattern, and that stage
    is the one producer that publishes ``True``. It exists to be READ: the
    stage refuses a batch that already carries it, because applying an antenna
    pattern twice squares its gain and no magnitude plot shows the difference.
    """

    sensor_pair_count: int
    path_count: int
    sensor_pair_index: torch.Tensor
    pair_offsets: torch.Tensor
    total_delay_s: torch.Tensor
    delay_rate: torch.Tensor | None
    complex_transfer_ref: torch.Tensor
    reference_frequency_hz: float
    row_valid: torch.Tensor | None
    topology: RadarPathTopology
    weight_includes_antenna_pattern: bool = False
    #: The same rows at ``reference_frequency_hz + frequency_offsets_hz[j]``,
    #: ``complex64[path_count, F]``, composed column by column exactly as
    #: ``complex_transfer_ref`` is. Present only on a wideband run, where a
    #: consumer that evaluates per frequency (OFDM, one column per subcarrier)
    #: reads it instead of shifting the reference coefficient.
    complex_transfer_band: torch.Tensor | None = None
    frequency_offsets_hz: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if type(self.weight_includes_antenna_pattern) is not bool:
            raise TypeError("weight_includes_antenna_pattern must be a bool")
        if (self.complex_transfer_band is None) != (self.frequency_offsets_hz is None):
            raise ValueError("complex_transfer_band and frequency_offsets_hz are paired: publish both or neither")
        if self.complex_transfer_band is not None and self.frequency_offsets_hz is not None:
            band = (self.path_count, len(self.frequency_offsets_hz))
            require_tensor("complex_transfer_band", self.complex_transfer_band, dtype=torch.complex64, shape=band)
        if type(self.sensor_pair_count) is not int or self.sensor_pair_count < 1:
            raise ValueError("sensor_pair_count must be a positive int")
        if type(self.path_count) is not int or self.path_count < 0:
            raise ValueError("path_count must be a non-negative int")
        rows = (self.path_count,)
        require_tensor("sensor_pair_index", self.sensor_pair_index, dtype=torch.int64, shape=rows)
        require_tensor("pair_offsets", self.pair_offsets, dtype=torch.int64, shape=(self.sensor_pair_count + 1,))
        require_tensor("total_delay_s", self.total_delay_s, dtype=torch.float32, shape=rows)
        require_tensor("complex_transfer_ref", self.complex_transfer_ref, dtype=torch.complex64, shape=rows)
        if self.delay_rate is not None:
            require_tensor("delay_rate", self.delay_rate, dtype=torch.float32, shape=rows)
        if self.row_valid is not None:
            require_tensor("row_valid", self.row_valid, dtype=torch.bool, shape=rows)
        if self.topology.row_count != self.path_count:
            raise ValueError("topology must have exactly path_count rows")

    @property
    def device(self) -> torch.device:
        return self.total_delay_s.device


def validate_pair_ordering(
    sensor_pair_index: torch.Tensor | np.ndarray, *, num_tx: int, num_rx: int, sensor_pair_count: int
) -> None:
    """Validate a sink-major sensor-pair partition on the host, at freeze time.

    The join validates its ranks while they are still a host array; a device
    tensor is read back once, which is a freeze-time cost and never a frame's.
    """

    if isinstance(num_tx, bool) or not isinstance(num_tx, int) or num_tx < 1:
        raise ValueError(f"num_tx must be a positive int, got {num_tx!r}")
    if isinstance(num_rx, bool) or not isinstance(num_rx, int) or num_rx < 1:
        raise ValueError(f"num_rx must be a positive int, got {num_rx!r}")
    expected = num_tx * num_rx
    if sensor_pair_count != expected:
        raise ValueError(
            f"the frozen topology spans {sensor_pair_count} sensor pairs but the "
            f"array is {num_tx} x {num_rx} = {expected} pairs"
        )
    if isinstance(sensor_pair_index, torch.Tensor):
        if sensor_pair_index.dtype != torch.int64:
            raise TypeError(f"sensor_pair_index must be int64, got {sensor_pair_index.dtype}")
        if sensor_pair_index.dim() != 1:
            raise ValueError(f"sensor_pair_index must be 1-D, got shape {tuple(sensor_pair_index.shape)}")
        ranks = np.asarray(sensor_pair_index.tolist(), dtype=np.int64)
    else:
        ranks = np.asarray(sensor_pair_index, dtype=np.int64)
        if ranks.ndim != 1:
            raise ValueError(f"sensor_pair_index must be 1-D, got shape {ranks.shape}")
    outside = np.flatnonzero((ranks < 0) | (ranks >= expected))
    if outside.size:
        row = int(outside[0])
        raise ValueError(
            f"row {row} names sensor pair {int(ranks[row])}, which is outside the "
            f"{num_tx} x {num_rx} array's range [0, {expected})"
        )
    drops = np.flatnonzero(ranks[1:] < ranks[:-1])
    if drops.size:
        row = int(drops[0]) + 1
        raise ValueError(
            "sensor pair ranks must be non-decreasing so that pair_offsets "
            f"is a half-open partition; row {row} drops from {int(ranks[row - 1])} to {int(ranks[row])}"
        )


@dataclass(frozen=True, slots=True, eq=False)
class ResponseFamily:
    """How the response is indexed and who owns its gradient.

    ``offsets`` and ``rows`` are the CSR of ONE slot; ``owners`` is how many
    response values there are in total. A per-site response is shared by every
    slot (``slot_local`` false); a per-row response has one value per composed
    row of every slot. ``slots`` is the slot count of the batch the join is
    composing: its index and CSR tables describe one slot, and the native join
    strides them across the rest.
    """

    offsets: torch.Tensor
    rows: torch.Tensor
    owners: int
    slots: int = 1
    slot_local: bool = False


@dataclass(frozen=True, slots=True, eq=False)
class SlotRows:
    """``slots`` copies of one join's composed rows, slot major.

    Row ``g * rows_per_slot + r`` is join row ``r`` at slot ``g``. Only the
    payload is batched: the join's identity, pair partition and routing tables
    are the single-slot ones and are not copied per slot.
    """

    total_delay_s: torch.Tensor
    complex_transfer_ref: torch.Tensor
    row_valid: torch.Tensor | None
    slots: int
    rows_per_slot: int
    weight_includes_antenna_pattern: bool = False
    #: Every slot's per-frequency weights, ``[slots * rows_per_slot, F]``, on a
    #: wideband run (see :class:`RadarPathBatch`); ``None`` otherwise.
    complex_transfer_band: torch.Tensor | None = None
    frequency_offsets_hz: tuple[float, ...] | None = None


class _TwoWayJoin(torch.autograd.Function):
    """Autograd bridge for the three native join operators.

    Two structural contracts, each with a test, both inherited from the beat
    family for the same reasons:

    * The facade ALWAYS routes through ``Function.apply``. An ADR-038
      forward-only dual has ``requires_grad == False``, so a ``requires_grad``
      shortcut around autograd would silently swallow its tangent and return a
      plain tensor.
    * No complex tensor crosses the autograd boundary. The composer splits
      every complex value into real and imaginary parts with Torch's own
      autograd-aware accessors and recombines the output the same way, which
      makes the conjugate-Wirtinger convention question structurally
      impossible to get wrong.
    """

    @staticmethod
    def forward(
        tau_in: torch.Tensor,
        tau_out: torch.Tensor,
        c_in_re: torch.Tensor,
        c_in_im: torch.Tensor,
        c_out_re: torch.Tensor,
        c_out_im: torch.Tensor,
        s_re: torch.Tensor,
        s_im: torch.Tensor,
        row_valid: torch.Tensor,
        idx_in: torch.Tensor,
        idx_out: torch.Tensor,
        idx_s: torch.Tensor,
        join: TwoWayComposer,
        response_family: ResponseFamily,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rows = int(row_valid.shape[0])
        tau_rt = torch.empty(rows, dtype=torch.float32, device=tau_in.device)
        c_rt_re = torch.empty_like(tau_rt)
        c_rt_im = torch.empty_like(tau_rt)
        _ops().two_way_join_forward(
            tau_in,
            tau_out,
            c_in_re,
            c_in_im,
            c_out_re,
            c_out_im,
            s_re,
            s_im,
            row_valid,
            idx_in,
            idx_out,
            idx_s,
            tau_rt,
            c_rt_re,
            c_rt_im,
            rows,
            response_family.slots,
            int(response_family.slot_local),
        )
        return tau_rt, c_rt_re, c_rt_im

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: tuple[torch.Tensor, ...]) -> None:
        (
            _tau_in,
            _tau_out,
            c_in_re,
            c_in_im,
            c_out_re,
            c_out_im,
            s_re,
            s_im,
            row_valid,
            idx_in,
            idx_out,
            idx_s,
            join,
            response_family,
        ) = inputs
        ctx.join = join
        # The response's gradient owners: which CSR reduces it and how many
        # slots it has. A per-site response uses the frozen site family; a
        # per-row response uses the identity family. Everything else about the
        # backward is identical, which is the whole point of routing a row
        # response through the same kernel.
        ctx.response_family = response_family
        saved = (c_in_re, c_in_im, c_out_re, c_out_im, s_re, s_im, row_valid, idx_in, idx_out, idx_s)
        ctx.save_for_backward(*saved)
        ctx.save_for_forward(*saved)

    @staticmethod
    @first_order_only
    def backward(
        ctx: Any, grad_tau_rt: torch.Tensor, grad_c_rt_re: torch.Tensor, grad_c_rt_im: torch.Tensor
    ) -> tuple[torch.Tensor | None, ...]:
        (c_in_re, c_in_im, c_out_re, c_out_im, s_re, s_im, row_valid, idx_in, idx_out, idx_s) = ctx.saved_tensors
        join = ctx.join
        family = ctx.response_family
        grad_tau_in = torch.empty_like(c_in_re)
        grad_c_in_re = torch.empty_like(c_in_re)
        grad_c_in_im = torch.empty_like(c_in_re)
        grad_tau_out = torch.empty_like(c_out_re)
        grad_c_out_re = torch.empty_like(c_out_re)
        grad_c_out_im = torch.empty_like(c_out_re)
        grad_s_re = torch.empty_like(s_re)
        grad_s_im = torch.empty_like(s_re)
        _ops().two_way_join_backward(
            c_in_re,
            c_in_im,
            c_out_re,
            c_out_im,
            s_re,
            s_im,
            row_valid,
            idx_in,
            idx_out,
            idx_s,
            join.by_inbound_offsets,
            join.by_inbound_rows,
            join.by_outbound_offsets,
            join.by_outbound_rows,
            family.offsets,
            family.rows,
            grad_tau_rt.contiguous(),
            grad_c_rt_re.contiguous(),
            grad_c_rt_im.contiguous(),
            grad_tau_in,
            grad_tau_out,
            grad_c_in_re,
            grad_c_in_im,
            grad_c_out_re,
            grad_c_out_im,
            grad_s_re,
            grad_s_im,
            int(row_valid.shape[0]),
            int(c_in_re.shape[0]),
            int(c_out_re.shape[0]),
            family.owners,
            family.slots,
            int(family.slot_local),
        )
        return (
            grad_tau_in,
            grad_tau_out,
            grad_c_in_re,
            grad_c_in_im,
            grad_c_out_re,
            grad_c_out_im,
            grad_s_re,
            grad_s_im,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    @staticmethod
    def jvp(
        ctx: Any,
        tan_tau_in: torch.Tensor | None,
        tan_tau_out: torch.Tensor | None,
        tan_c_in_re: torch.Tensor | None,
        tan_c_in_im: torch.Tensor | None,
        tan_c_out_re: torch.Tensor | None,
        tan_c_out_im: torch.Tensor | None,
        tan_s_re: torch.Tensor | None,
        tan_s_im: torch.Tensor | None,
        tan_row_valid: Any,
        tan_idx_in: Any,
        tan_idx_out: Any,
        tan_idx_s: Any,
        tan_join: Any,
        tan_response_family: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        family = ctx.response_family
        (c_in_re, c_in_im, c_out_re, c_out_im, s_re, s_im, row_valid, idx_in, idx_out, idx_s) = ctx.saved_tensors

        def inbound(tangent: torch.Tensor | None) -> torch.Tensor:
            return torch.zeros_like(c_in_re) if tangent is None else tangent.contiguous()

        def outbound(tangent: torch.Tensor | None) -> torch.Tensor:
            return torch.zeros_like(c_out_re) if tangent is None else tangent.contiguous()

        def site(tangent: torch.Tensor | None) -> torch.Tensor:
            return torch.zeros_like(s_re) if tangent is None else tangent.contiguous()

        rows = int(row_valid.shape[0])
        tan_tau_rt = torch.empty(rows, dtype=torch.float32, device=c_in_re.device)
        tan_c_rt_re = torch.empty_like(tan_tau_rt)
        tan_c_rt_im = torch.empty_like(tan_tau_rt)
        _ops().two_way_join_jvp(
            c_in_re,
            c_in_im,
            c_out_re,
            c_out_im,
            s_re,
            s_im,
            row_valid,
            idx_in,
            idx_out,
            idx_s,
            inbound(tan_tau_in),
            outbound(tan_tau_out),
            inbound(tan_c_in_re),
            inbound(tan_c_in_im),
            outbound(tan_c_out_re),
            outbound(tan_c_out_im),
            site(tan_s_re),
            site(tan_s_im),
            tan_tau_rt,
            tan_c_rt_re,
            tan_c_rt_im,
            rows,
            family.slots,
            int(family.slot_local),
        )
        return tan_tau_rt, tan_c_rt_re, tan_c_rt_im


@dataclass(frozen=True, slots=True, eq=False)
class TwoWayComposer:
    """A frozen inbound/outbound join for one set of scatter sites."""

    inbound_row: torch.Tensor
    outbound_row: torch.Tensor
    response_slot: torch.Tensor
    topology: RadarPathTopology
    sensor_pair_index: torch.Tensor
    pair_offsets: torch.Tensor
    sensor_pair_count: int
    site_count: int
    inbound_row_count: int
    outbound_row_count: int
    by_inbound_offsets: torch.Tensor
    by_inbound_rows: torch.Tensor
    by_outbound_offsets: torch.Tensor
    by_outbound_rows: torch.Tensor
    by_response_offsets: torch.Tensor
    by_response_rows: torch.Tensor
    reference_frequency_hz: float
    # The identity site family, for a response that publishes one value per
    # COMPOSED ROW rather than one per site. The join kernel indexes the
    # response through ``idx_s`` and reduces its gradient through a CSR, so a
    # per-row response is expressible with no kernel change at all: hand it an
    # identity index and an identity CSR and the site family becomes the row
    # family. Built at freeze because freeze is where every other table is
    # built; three small int64 tensors, allocated once per topology rather than
    # once per frame, so a row response costs no extra launch to set up.
    row_slot: torch.Tensor
    by_row_offsets: torch.Tensor
    by_row_rows: torch.Tensor
    # The deepest outbound row this join composes, read from the frozen leg
    # identity on the host. An aspect-dependent response needs the DEPARTURE
    # direction at the site and a leg publishes its final segment's direction,
    # so it refuses anything but a line-of-sight outbound leg; the host int is
    # what lets it refuse without reading a device column.
    outbound_max_depth: int
    #: One slot's pair partition on the host, as built: ``pair_offsets`` before
    #: its upload. A slot-major replica keeps the single-slot table, which is
    #: what an instrument half selecting pair segments slot by slot reads.
    pair_offsets_host: np.ndarray
    #: Each composed row's identity on the host, ``int64[rows, 7]``: radar
    #: source, site and radar sink IDs, then the inbound and outbound component
    #: IDs and the inbound and outbound depths. Kept from the freeze so that a
    #: consumer naming a row (the path event log) never reads the device.
    identity_host: np.ndarray
    #: The two legs' interaction sequences per composed row, primitive then
    #: material: ``int64[rows, 2 * width_in]`` and ``int64[rows, 2 * width_out]``.
    #: With :attr:`identity_host` they are the row's whole key, which is what
    #: matches a row of this join to a row of the next epoch's.
    inbound_sequence_host: np.ndarray
    outbound_sequence_host: np.ndarray
    #: The diffracting edge of each composed row's inbound and outbound leg,
    #: ``int64[rows, 2]``, ``-1`` where a leg recorded none. Part of the key: a
    #: diffracted row's primitive sequence does not tell two edges apart.
    edge_host: np.ndarray
    #: ``inbound_row`` and ``outbound_row`` on the host, as built.
    inbound_row_host: np.ndarray
    outbound_row_host: np.ndarray
    #: How many rows of the third, site-less leg this join composes, ``0``
    #: when it composes none.
    direct_row_count: int = 0
    #: The composed order over ``[site rows | direct rows]``: composed row
    #: ``i`` is row ``merge[i]`` of that concatenation. ``None`` when there are
    #: no direct rows, in which case the native join's output IS the composed
    #: order and nothing is gathered. The native join, its CSR families and
    #: ``inbound_row`` / ``outbound_row`` / ``response_slot`` address the site
    #: rows only; ``sensor_pair_index``, ``pair_offsets`` and ``topology``
    #: address the composed rows.
    merge: torch.Tensor | None = None

    @classmethod
    def freeze(
        cls,
        inbound: FrozenLeg,
        outbound: FrozenLeg,
        site_ids: torch.Tensor | Sequence[int],
        *,
        radar_source_ids: Sequence[int],
        radar_sink_ids: Sequence[int],
        reference_frequency_hz: float,
        inbound_identity: LegIdentity | None = None,
        outbound_identity: LegIdentity | None = None,
        direct: FrozenLeg | None = None,
        direct_identity: LegIdentity | None = None,
        max_round_trips: int | None = None,
        leg_site_ids: Sequence[int] | None = None,
    ) -> TwoWayComposer:
        """Build the identity join from two frozen leg topologies.

        ``inbound`` and ``outbound`` are
        :class:`witwin.radar.channel.FrozenLegTopology`
        handles; they are duck-typed here so this module does not import the
        Channel adapter.

        ``radar_source_ids`` and ``radar_sink_ids`` are the FRONT END's stable
        endpoint IDs, not the surviving rows'. They define the sensor-pair
        partition, so a pair that discovered nothing still owns an empty
        segment and the IQ cube keeps its declared shape.

        ``site_ids`` may be a SUBSET of the sites the legs actually reach: a
        caller composing two of five discovered targets is doing something
        legitimate.

        ``leg_site_ids`` names the site endpoints both legs were DISCOVERED
        for: the inbound leg's sinks and the outbound leg's sources. A declared
        site in that set with no row in a leg is a site discovery found no path
        to at this instant (a target behind an occluder); it owns no composed
        row and contributes exactly zero. A declared site outside it is a
        wrong stable ID and is refused, because dropping it silently is how a
        join produces a plausible empty answer. Without ``leg_site_ids`` the
        rows are the only evidence of which sites the legs were asked about,
        so a declared site with no row in a leg is refused as unknown.

        ``direct`` is the optional third leg, transmitter straight to receiver.
        Each of its rows becomes one site-less composed row in its sensor
        pair's segment, after that pair's site rows, ordered by its own leg key.

        ``max_round_trips`` bounds the composed rows of any ONE sensor pair. It
        is checked on the host from per-site row counts BEFORE the cross join is
        materialised, because the join is the step whose size is the product of
        the two legs' row counts, and a refusal there names the pair rather
        than an allocation failure somewhere downstream.
        """

        device = inbound.sink_id.device
        sources = stable_ids(radar_source_ids, "radar_source_ids")
        sinks = stable_ids(radar_sink_ids, "radar_sink_ids")
        sites = stable_ids(site_ids, "site_ids")
        sites.sort()
        arriving = read_leg_identity(inbound, "inbound") if inbound_identity is None else inbound_identity
        leaving = read_leg_identity(outbound, "outbound") if outbound_identity is None else outbound_identity
        inbound_key = _dense_rank(arriving.key_columns())
        outbound_key = _dense_rank(leaving.key_columns())
        _require_unique_keys(arriving, inbound_key, "inbound")
        _require_unique_keys(leaving, outbound_key, "outbound")

        # Nothing may fall outside the declared front end. A leg row whose
        # radar endpoint is not in the declared lists would never be joined,
        # which is a silent drop rather than an empty segment.
        stray_sources = sorted(set(arriving.source.tolist()) - set(sources))
        if stray_sources:
            raise ValueError(
                f"inbound leg rows carry radar source IDs {stray_sources} that are not in radar_source_ids {sources}"
            )
        stray_sinks = sorted(set(leaving.sink.tolist()) - set(sinks))
        if stray_sinks:
            raise ValueError(
                f"outbound leg rows carry radar sink IDs {stray_sinks} that are not in radar_sink_ids {sinks}"
            )

        # A declared site the legs were never asked about is a caller error:
        # the site list is the declaration of what this join is about, and
        # silently dropping one would hide a wrong stable ID. A site the legs
        # WERE asked about and found no path to is not - that is discovery
        # reporting an occluded site, and it composes to no row at all, exactly
        # as a sensor pair that sees nothing composes to an empty segment.
        if leg_site_ids is None:
            reachable_in = set(arriving.sink.tolist())
            reachable_out = set(leaving.source.tolist())
            for site in sites:
                if site not in reachable_in:
                    raise ValueError(f"site {site} has no inbound leg row in the frozen topology")
                if site not in reachable_out:
                    raise ValueError(f"site {site} has no outbound leg row in the frozen topology")
        else:
            discovered = set(stable_ids(leg_site_ids, "leg_site_ids"))
            unknown = sorted(set(sites) - discovered)
            if unknown:
                raise ValueError(
                    f"sites {unknown} are not endpoints either leg was discovered for ({sorted(discovered)}); "
                    "a declared site must name a site endpoint of both legs"
                )

        # Every (inbound, outbound) row pair that meets at one declared site.
        # Rows that reach an undeclared site are not part of this join.
        declared = np.asarray(sites, dtype=np.int64)
        inbound_rows = np.flatnonzero(np.isin(arriving.sink, declared))
        outbound_rows = np.flatnonzero(np.isin(leaving.source, declared))
        inbound_site = _ranks(arriving.sink[inbound_rows], sites)
        through = (
            None
            if direct is None
            else (read_leg_identity(direct, "direct") if direct_identity is None else direct_identity)
        )
        if through is not None:
            stray = sorted(set(through.source.tolist()) - set(sources)) + sorted(
                set(through.sink.tolist()) - set(sinks)
            )
            if stray:
                raise ValueError(
                    f"direct leg rows carry endpoint IDs {stray} outside radar_source_ids {sources} and "
                    f"radar_sink_ids {sinks}"
                )
        if max_round_trips is not None:
            require_round_trip_budget(
                arriving.source[inbound_rows],
                inbound_site,
                leaving.sink[outbound_rows],
                _ranks(leaving.source[outbound_rows], sites),
                site_count=len(sites),
                sources=sources,
                sinks=sinks,
                direct=None if through is None else (through.source, through.sink),
                budget=max_round_trips,
            )
        outbound_order = outbound_rows[np.argsort(_ranks(leaving.source[outbound_rows], sites), kind="stable")]
        per_site = np.bincount(_ranks(leaving.source[outbound_rows], sites), minlength=len(sites))
        first_partner = np.concatenate(([0], np.cumsum(per_site)[:-1]))
        partners = per_site[inbound_site]
        total = int(partners.sum())
        joined_in = np.repeat(inbound_rows, partners)
        within = np.arange(total) - np.repeat(np.cumsum(partners) - partners, partners)
        joined_out = outbound_order[np.repeat(first_partner[inbound_site], partners) + within]
        site_rank = np.repeat(inbound_site, partners)
        source = arriving.source[joined_in]
        sink = leaving.sink[joined_out]
        pair = np.broadcast_to(np.asarray(sink_major_rank(sources, sinks)(source, sink), dtype=np.int64), (total,))

        # The canonical order: pair rank, site rank, inbound key, outbound key.
        # Every component is frame invariant and the four together are unique,
        # so two freezes of permuted leg rows produce the SAME composed
        # sequence, not merely the same composed set.
        order = np.lexsort((outbound_key[joined_out], inbound_key[joined_in], site_rank, pair))
        pair, site_rank = pair[order], site_rank[order]
        joined_in, joined_out = joined_in[order], joined_out[order]
        source, sink = source[order], sink[order]

        pair_count = len(sources) * len(sinks)
        inbound_count = arriving.row_count
        outbound_count = leaving.row_count
        by_inbound = csr(joined_in, inbound_count)
        by_outbound = csr(joined_out, outbound_count)
        by_response = csr(site_rank, len(sites))

        # The site-less rows join their sensor pair's segment after its site
        # rows (site rank ``len(sites)``), in the order of their own leg key.
        # The site rows are already in canonical order, so a stable merge
        # keeps them as the native join publishes them.
        merge = None
        direct_count = 0
        composed_source, composed_site, composed_sink = source, arriving.sink[joined_in], sink
        composed_in, composed_out, composed_direct = joined_in, joined_out, None
        if through is not None:
            direct_count = through.row_count
            direct_key = _dense_rank(through.key_columns())
            _require_unique_keys(through, direct_key, "direct")
            direct_pair = np.asarray(sink_major_rank(sources, sinks)(through.source, through.sink), dtype=np.int64)
            merge = np.lexsort(
                (
                    np.concatenate((np.zeros(total, dtype=np.int64), direct_key)),
                    np.concatenate((np.arange(total, dtype=np.int64), np.zeros(direct_count, dtype=np.int64))),
                    np.concatenate((site_rank, np.full(direct_count, len(sites), dtype=np.int64))),
                    np.concatenate((pair, direct_pair)),
                )
            )
            none = np.full(direct_count, -1, dtype=np.int64)
            pair = np.concatenate((pair, direct_pair))[merge]
            composed_source = np.concatenate((source, through.source))[merge]
            composed_site = np.concatenate((composed_site, np.full(direct_count, NO_SITE, dtype=np.int64)))[merge]
            composed_sink = np.concatenate((sink, through.sink))[merge]
            composed_in = np.concatenate((joined_in, none))[merge]
            composed_out = np.concatenate((joined_out, none))[merge]
            composed_direct = np.concatenate((np.full(total, -1, dtype=np.int64), np.arange(direct_count)))[merge]
        validate_pair_ordering(pair, num_tx=len(sources), num_rx=len(sinks), sensor_pair_count=pair_count)
        offsets = pair_offsets(pair, pair_count)

        # Each composed row's identity on the host, in composed order. A
        # site-less row carries its own leg in the inbound columns and nothing
        # (``-1``, depth 0) in the outbound ones.
        in_width = max(arriving.primitive.shape[1], 0 if through is None else through.primitive.shape[1])
        out_width = leaving.primitive.shape[1]

        def sequences(leg: LegIdentity, rows: np.ndarray, width: int) -> np.ndarray:
            pad = np.full((len(rows), width - leg.primitive.shape[1]), -1, dtype=np.int64)
            return np.column_stack((leg.primitive[rows], pad, leg.material[rows], pad)).astype(np.int64)

        identity_host = np.column_stack(
            (
                source,
                arriving.sink[joined_in],
                sink,
                arriving.component[joined_in],
                leaving.component[joined_out],
                arriving.depth[joined_in],
                leaving.depth[joined_out],
            )
        ).astype(np.int64)
        inbound_sequence_host = sequences(arriving, joined_in, in_width)
        outbound_sequence_host = sequences(leaving, joined_out, out_width)

        def edges(leg: LegIdentity, rows: np.ndarray) -> np.ndarray:
            return np.full(len(rows), -1, dtype=np.int64) if leg.edge is None else leg.edge[rows].astype(np.int64)

        edge_host = np.column_stack((edges(arriving, joined_in), edges(leaving, joined_out)))
        if through is not None and merge is not None:
            every = np.arange(through.row_count)
            direct_identity_host = np.column_stack(
                (
                    through.source,
                    np.full(through.row_count, NO_SITE),
                    through.sink,
                    through.component,
                    np.full(through.row_count, -1),
                    through.depth,
                    np.zeros(through.row_count),
                )
            ).astype(np.int64)
            identity_host = np.concatenate((identity_host, direct_identity_host))[merge]
            inbound_sequence_host = np.concatenate((inbound_sequence_host, sequences(through, every, in_width)))[merge]
            outbound_sequence_host = np.concatenate(
                (outbound_sequence_host, np.full((through.row_count, 2 * out_width), -1, dtype=np.int64))
            )[merge]
            direct_edges = np.column_stack((edges(through, every), np.full(through.row_count, -1, dtype=np.int64)))
            edge_host = np.concatenate((edge_host, direct_edges))[merge]

        # Every integer table of the join crosses to the device in ONE copy and
        # is published as views of that buffer.
        tables: tuple[np.ndarray, ...] = (
            pair,
            site_rank,
            composed_source,
            composed_site,
            composed_sink,
            joined_in,
            joined_out,
            offsets,
            *by_inbound,
            *by_outbound,
            *by_response,
        )
        if composed_direct is not None and merge is not None:
            tables = (*tables, composed_in, composed_out, composed_direct, merge)
        uploaded = torch.from_numpy(np.concatenate([np.asarray(table, dtype=np.int64) for table in tables])).to(device)
        views = torch.split(uploaded, [len(table) for table in tables])
        (
            sensor_pair_index,
            response_slot,
            radar_source_id,
            site_id,
            radar_sink_id,
            inbound_row,
            outbound_row,
            offsets_table,
            by_inbound_offsets,
            by_inbound_rows,
            by_outbound_offsets,
            by_outbound_rows,
            by_response_offsets,
            by_response_rows,
        ) = views[:14]
        merge_table = None
        topology_in, topology_out, direct_row = inbound_row, outbound_row, None
        if merge is not None:
            topology_in, topology_out, direct_row, merge_table = views[14:]
        topology = RadarPathTopology(
            radar_source_id=radar_source_id,
            site_id=site_id,
            radar_sink_id=radar_sink_id,
            inbound_row=topology_in,
            outbound_row=topology_out,
            direct_row=direct_row,
        )
        return cls(
            inbound_row=inbound_row,
            outbound_row=outbound_row,
            response_slot=response_slot,
            topology=topology,
            sensor_pair_index=sensor_pair_index,
            pair_offsets=offsets_table,
            sensor_pair_count=pair_count,
            site_count=len(sites),
            inbound_row_count=inbound_count,
            outbound_row_count=outbound_count,
            by_inbound_offsets=by_inbound_offsets,
            by_inbound_rows=by_inbound_rows,
            by_outbound_offsets=by_outbound_offsets,
            by_outbound_rows=by_outbound_rows,
            by_response_offsets=by_response_offsets,
            by_response_rows=by_response_rows,
            reference_frequency_hz=float(reference_frequency_hz),
            row_slot=torch.arange(total, dtype=torch.int64, device=device),
            by_row_offsets=torch.arange(total + 1, dtype=torch.int64, device=device),
            by_row_rows=torch.arange(total, dtype=torch.int64, device=device),
            outbound_max_depth=int(leaving.depth[joined_out].max()) if total else 0,
            pair_offsets_host=offsets,
            direct_row_count=direct_count,
            merge=merge_table,
            identity_host=identity_host,
            inbound_sequence_host=inbound_sequence_host,
            outbound_sequence_host=outbound_sequence_host,
            edge_host=edge_host,
            inbound_row_host=np.asarray(composed_in, dtype=np.int64),
            outbound_row_host=np.asarray(composed_out, dtype=np.int64),
        )

    @property
    def site_row_count(self) -> int:
        """Composed rows that visit a scatter site: the native join's rows."""

        return int(self.inbound_row.shape[0])

    @property
    def path_count(self) -> int:
        return self.site_row_count + self.direct_row_count

    def compose(
        self,
        inbound: RadarLegBatch,
        outbound: RadarLegBatch,
        response: ScatterResponse,
        direct: RadarLegBatch | None = None,
    ) -> RadarPathBatch:
        """Compose one frame's round-trip rows. Device work only.

        A dead row's payload is exactly zero, not a partial composition. The
        row is a complete answer that this round trip does not exist at these
        endpoint positions; publishing ``tau_in + 0`` for it would be a
        plausible number that no consumer should ever read.

        ``direct`` is the third leg's batch when this join was frozen with one.
        A site-less row is that leg's row as it stands - its delay and its
        coefficient, with no scatter factor - so placing it is a gather, not
        an equation: the native join composes the site rows and the two sets
        are interleaved by the frozen ``merge`` order in one ``index_select``.
        Its derivative is the gather's.
        """

        self._require_frame(inbound, outbound, direct)
        rows = self.site_row_count
        device = inbound.delay_s.device
        row_valid = self._row_validity(inbound, outbound, rows, device)
        flags = torch.ones(rows, dtype=torch.int32, device=device) if row_valid is None else row_valid.to(torch.int32)
        response_re, response_im, response_index, response_family = self._response(
            response, inbound, outbound, flags, device
        )

        # Torch-owned, autograd-aware accessors: the real pairs cross the
        # boundary, never the complex tensors.
        tau_rt, transfer_re, transfer_im = _TwoWayJoin.apply(
            inbound.delay_s.contiguous(),
            outbound.delay_s.contiguous(),
            inbound.coefficient.real.contiguous(),
            inbound.coefficient.imag.contiguous(),
            outbound.coefficient.real.contiguous(),
            outbound.coefficient.imag.contiguous(),
            response_re,
            response_im,
            flags,
            self.inbound_row,
            self.outbound_row,
            response_index,
            self,
            response_family,
        )
        transfer = torch.complex(transfer_re, transfer_im)
        if self.merge is not None:
            assert direct is not None
            tau_rt = torch.cat((tau_rt, direct.delay_s)).index_select(0, self.merge)
            transfer = torch.cat((transfer, direct.coefficient)).index_select(0, self.merge)
            if row_valid is not None or direct.row_valid is not None:
                site_valid = torch.ones(rows, dtype=torch.bool, device=device) if row_valid is None else row_valid
                direct_valid = (
                    torch.ones(direct.leg_count, dtype=torch.bool, device=device)
                    if direct.row_valid is None
                    else direct.row_valid
                )
                row_valid = torch.cat((site_valid, direct_valid)).index_select(0, self.merge)

        return RadarPathBatch(
            sensor_pair_count=self.sensor_pair_count,
            path_count=self.path_count,
            sensor_pair_index=self.sensor_pair_index,
            pair_offsets=self.pair_offsets,
            total_delay_s=tau_rt,
            delay_rate=None,
            complex_transfer_ref=transfer,
            reference_frequency_hz=self.reference_frequency_hz,
            row_valid=row_valid,
            topology=self.topology,
        )

    def compose_slots(
        self,
        inbound: RadarLegBatch,
        outbound: RadarLegBatch,
        response: ScatterResponse,
        slots: int,
        direct: RadarLegBatch | None = None,
    ) -> SlotRows:
        """Compose a slot-major replay of ``slots`` instants with this one join.

        The legs carry ``slots`` copies of the frozen rows; the native join
        strides this join's single-slot tables across them, so no per-slot
        copy of a routing table is built. The result is the bit for bit equal
        of composing with :meth:`for_slots` tables. A per-site response is
        shared by every slot; a geometry-dependent response is evaluated per
        composed row and takes :meth:`for_slots` instead.

        ``direct`` is the slot-major site-less leg when this join was frozen
        with one. Its rows are interleaved per slot by the single-slot
        ``merge`` order over the ``[slots, site rows | direct rows]`` view, so
        no slot-replicated merge table exists either.
        """

        if (direct is None) != (self.merge is None):
            raise ValueError("pass the site-less leg exactly when this join was frozen with one")
        if response.is_geometry_dependent:
            raise ValueError("a geometry-dependent response is evaluated per composed row; compose it with for_slots")
        for name, batch, expected in (
            ("inbound", inbound, self.inbound_row_count * slots),
            ("outbound", outbound, self.outbound_row_count * slots),
            ("direct", direct, self.direct_row_count * slots),
        ):
            if batch is not None and batch.leg_count != expected:
                raise ValueError(
                    f"{name} leg carries {batch.leg_count} rows but {slots} slots of this join need {expected}"
                )
        rows = self.site_row_count * slots
        device = inbound.delay_s.device
        row_valid = self._slot_validity(inbound, outbound, slots, device)
        flags = torch.ones(rows, dtype=torch.int32, device=device) if row_valid is None else row_valid.to(torch.int32)
        value = self._site_response(response, device)
        tau_rt, transfer_re, transfer_im = _TwoWayJoin.apply(
            inbound.delay_s.contiguous(),
            outbound.delay_s.contiguous(),
            inbound.coefficient.real.contiguous(),
            inbound.coefficient.imag.contiguous(),
            outbound.coefficient.real.contiguous(),
            outbound.coefficient.imag.contiguous(),
            value.real.contiguous(),
            value.imag.contiguous(),
            flags,
            self.inbound_row,
            self.outbound_row,
            self.response_slot,
            self,
            ResponseFamily(self.by_response_offsets, self.by_response_rows, self.site_count, slots=slots),
        )
        transfer = torch.complex(transfer_re, transfer_im)
        if direct is not None and self.merge is not None:
            merge = self.merge

            def interleaved(site: torch.Tensor, through: torch.Tensor) -> torch.Tensor:
                joined = torch.cat((site.view(slots, -1), through.view(slots, -1)), dim=1)
                return joined.index_select(1, merge).reshape(-1)

            tau_rt = interleaved(tau_rt, direct.delay_s)
            transfer = interleaved(transfer, direct.coefficient)
            if row_valid is not None or direct.row_valid is not None:
                site_valid = torch.ones(rows, dtype=torch.bool, device=device) if row_valid is None else row_valid
                direct_valid = (
                    torch.ones(direct.leg_count, dtype=torch.bool, device=device)
                    if direct.row_valid is None
                    else direct.row_valid
                )
                row_valid = interleaved(site_valid, direct_valid)
        return SlotRows(
            total_delay_s=tau_rt,
            complex_transfer_ref=transfer,
            row_valid=row_valid,
            slots=slots,
            rows_per_slot=self.path_count,
        )

    def _slot_validity(
        self, inbound: RadarLegBatch, outbound: RadarLegBatch, slots: int, device: torch.device
    ) -> torch.Tensor | None:
        """Row validity of every slot, gathered through the single-slot tables."""

        if inbound.row_valid is None and outbound.row_valid is None:
            return None
        rows = self.site_row_count * slots
        ones = torch.ones(rows, dtype=torch.bool, device=device)
        valid_in = (
            ones
            if inbound.row_valid is None
            else inbound.row_valid.view(slots, -1).index_select(1, self.inbound_row).reshape(-1)
        )
        valid_out = (
            ones
            if outbound.row_valid is None
            else outbound.row_valid.view(slots, -1).index_select(1, self.outbound_row).reshape(-1)
        )
        return valid_in & valid_out

    def segment_targets(
        self, inbound: RadarLegBatch, outbound: RadarLegBatch, direct: RadarLegBatch | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per composed row, the far end of its first and the near end of its last segment.

        The antenna-pattern stage reads the transmit pattern toward the first
        and the receive pattern from the last. A site row takes them from its
        inbound and outbound legs, a site-less row from its own leg. Gathers of
        Channel's live geometry, so a derivative through a reflection point
        survives.
        """

        first = _present_member(inbound.departure_target_m, "departure_target_m")
        last = _present_member(outbound.arrival_origin_m, "arrival_origin_m")
        if self.merge is None:
            return first.index_select(0, self.topology.inbound_row), last.index_select(0, self.topology.outbound_row)
        assert direct is not None
        return (
            torch.cat(
                (
                    first.index_select(0, self.inbound_row),
                    _present_member(direct.departure_target_m, "departure_target_m"),
                )
            ).index_select(0, self.merge),
            torch.cat(
                (last.index_select(0, self.outbound_row), _present_member(direct.arrival_origin_m, "arrival_origin_m"))
            ).index_select(0, self.merge),
        )

    def for_slots(self, count: int, *, per_slot_response: bool = False) -> TwoWayComposer:
        """Block-diagonal join tables for a fixed topology's slot-major replay.

        Sites remain shared across slots, including their gradient owners,
        unless ``per_slot_response``: then slot ``g``'s site ``s`` is response
        site ``g * site_count + s``, so a response evaluated once per instant
        (a time-varying aspect) gives every slot its own value and its own
        gradient owner. Only integer routing tables are replicated; the
        composition equation and its AD remain owned by the ordinary native
        join.
        """
        device = self.inbound_row.device
        slot = torch.arange(count, device=device, dtype=torch.int64)[:, None]
        rows = self.site_row_count
        composed = self.path_count

        def shifted(table: torch.Tensor, stride: int) -> torch.Tensor:
            return (table[None, :] + slot * stride).reshape(-1)

        def slot_csr(offsets: torch.Tensor, stride: int = rows) -> torch.Tensor:
            return torch.cat((shifted(offsets[:-1], stride), offsets[-1:] * count))

        def shifted_or_none(table: torch.Tensor, stride: int) -> torch.Tensor:
            # -1 (this row has no entry in that leg) stays -1 in every slot.
            row = table[None, :]
            return torch.where(row >= 0, row + slot * stride, row).reshape(-1)

        inbound = shifted(self.inbound_row, self.inbound_row_count)
        outbound = shifted(self.outbound_row, self.outbound_row_count)
        if per_slot_response:
            responses = shifted(self.response_slot, self.site_count)
            response_offsets = slot_csr(self.by_response_offsets)
            response_rows = shifted(self.by_response_rows, rows)
        else:
            responses = self.response_slot.repeat(count)
            response_offsets = self.by_response_offsets * count
            response_rows = torch.argsort(responses, stable=True)
        topology_in, topology_out, direct_row = inbound, outbound, None
        merge = None
        if self.merge is not None and self.topology.direct_row is not None:
            topology_in = shifted_or_none(self.topology.inbound_row, self.inbound_row_count)
            topology_out = shifted_or_none(self.topology.outbound_row, self.outbound_row_count)
            direct_row = shifted_or_none(self.topology.direct_row, self.direct_row_count)
            # The slot-major concatenation is [count * site rows | count * direct
            # rows], so a site entry moves by whole site blocks and a direct
            # entry by whole direct blocks after every site row.
            site = self.merge < rows
            merge = torch.where(
                site[None, :],
                self.merge[None, :] + slot * rows,
                (self.merge - rows)[None, :] + count * rows + slot * self.direct_row_count,
            ).reshape(-1)
        topology = RadarPathTopology(
            self.topology.radar_source_id.repeat(count),
            self.topology.site_id.repeat(count),
            self.topology.radar_sink_id.repeat(count),
            topology_in,
            topology_out,
            direct_row,
        )
        return replace(
            self,
            inbound_row=inbound,
            outbound_row=outbound,
            response_slot=responses,
            topology=topology,
            merge=merge,
            direct_row_count=self.direct_row_count * count,
            sensor_pair_index=shifted(self.sensor_pair_index, self.sensor_pair_count),
            pair_offsets=slot_csr(self.pair_offsets, composed),
            sensor_pair_count=self.sensor_pair_count * count,
            inbound_row_count=self.inbound_row_count * count,
            outbound_row_count=self.outbound_row_count * count,
            by_inbound_offsets=slot_csr(self.by_inbound_offsets),
            by_inbound_rows=shifted(self.by_inbound_rows, rows),
            by_outbound_offsets=slot_csr(self.by_outbound_offsets),
            by_outbound_rows=shifted(self.by_outbound_rows, rows),
            by_response_offsets=response_offsets,
            by_response_rows=response_rows,
            site_count=self.site_count * count if per_slot_response else self.site_count,
            row_slot=torch.arange(count * rows, device=device, dtype=torch.int64),
            by_row_offsets=torch.arange(count * rows + 1, device=device, dtype=torch.int64),
            by_row_rows=torch.arange(count * rows, device=device, dtype=torch.int64),
        )

    def _require_frame(
        self, inbound: RadarLegBatch, outbound: RadarLegBatch, direct: RadarLegBatch | None = None
    ) -> None:
        """Refuse a frame that is not the one this join was frozen against.

        The index tables address the FROZEN leg rows, so a batch of a different
        length is not a smaller frame - it is a different topology. This is the
        only place that can see the mismatch: the forward and JVP entries are
        never told the leg counts (the backward entry is), and their length
        checks only tie the inputs to each other, so the kernel would gather
        through raw pointers with no bound and publish a plausible round trip
        built from whatever sat past the end of the buffer. Both counts are
        already host ints, so this costs nothing and observes nothing.

        The gap is not covered by the ``row_valid`` path either. That path
        bounds-checks incidentally, through ``index_select``, and only when a
        leg actually carries a mask - which makes it an inconsistent guard
        rather than a guard.
        """

        if (direct is None) != (self.merge is None):
            raise ValueError(
                "this join was frozen "
                + ("with" if self.merge is not None else "without")
                + " a site-less direct leg and the frame "
                + ("does not carry one" if direct is None else "carries one")
            )
        for name, batch, expected in (
            ("inbound", inbound, self.inbound_row_count),
            ("outbound", outbound, self.outbound_row_count),
            ("direct", direct, self.direct_row_count),
        ):
            if batch is not None and batch.leg_count != expected:
                raise ValueError(
                    f"{name} leg carries {batch.leg_count} rows but this join "
                    f"was frozen against {expected}; the frame does not belong "
                    "to this frozen topology"
                )

    def _response(
        self,
        response: ScatterResponse,
        inbound: RadarLegBatch,
        outbound: RadarLegBatch,
        flags: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, ResponseFamily]:
        """The response as a real pair, its index, and its gradient family.

        Two shapes, one join. A per-SITE response is broadcast across the rows
        of its site through the frozen ``response_slot`` and its gradient is
        reduced through the frozen site CSR. A per-ROW response is indexed by
        the identity table and reduced through the identity CSR, which is the
        same kernel with ``num_sites = path_count``.

        The refusal narrows here and does not disappear. A geometry-dependent
        response is per-path physics, and composing one in Torch is precisely
        what ``NATIVE_ROW_RESPONSE_OWNERS`` is a whitelist against: the check is
        against the response's OWN declared fully qualified name, not against a
        protocol, because a protocol check can see a method's name and not what
        runs behind it.
        """

        from .scattering import NATIVE_ROW_RESPONSE_OWNERS

        if not response.is_geometry_dependent:
            value = self._site_response(response, device)
            return (
                value.real.contiguous(),
                value.imag.contiguous(),
                self.response_slot,
                ResponseFamily(self.by_response_offsets, self.by_response_rows, self.site_count),
            )
        if getattr(response, "native_row_owner", None) not in NATIVE_ROW_RESPONSE_OWNERS:
            raise NotImplementedError(
                "a geometry-dependent scatter response varies per path and must "
                "be evaluated in a native kernel, not composed here"
            )
        # The whitelist above is what licenses the row entry; the protocol
        # declares only the per-site one.
        row_response: Any = response
        rows_re, rows_im = row_response.evaluate_rows(self, inbound, outbound, flags)
        for name, value in (("real", rows_re), ("imaginary", rows_im)):
            if not isinstance(value, torch.Tensor) or value.numel() != self.site_row_count:
                raise ValueError(
                    f"a row-evaluated scatter response must publish one {name} "
                    f"value per composed site row; this join has {self.site_row_count}"
                )
        return (
            rows_re,
            rows_im,
            self.row_slot,
            ResponseFamily(self.by_row_offsets, self.by_row_rows, self.site_row_count),
        )

    def _site_response(self, response: ScatterResponse, device: torch.device) -> torch.Tensor:
        """The per-site response, checked against the frozen site count.

        ``ScatterResponse`` is an extension point, and ``evaluate`` returning
        the wrong length is the same unbounded gather as a mismatched leg: the
        forward kernel's only check on the response is against itself. The
        protocol says ``complex[row_count]``, so holding it to that here is
        enforcing the contract, not second-guessing the implementation.
        """

        value = response.evaluate(self.site_count, device)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"a scatter response must evaluate to a torch.Tensor, got {type(value).__name__}")
        if value.numel() != self.site_count:
            raise ValueError(
                f"the scatter response evaluated to {value.numel()} values but "
                f"this join was frozen against {self.site_count} sites"
            )
        return value

    def _row_validity(
        self, inbound: RadarLegBatch, outbound: RadarLegBatch, rows: int, device: torch.device
    ) -> torch.Tensor | None:
        if rows == 0:
            # A join with no row (every declared site occluded) still publishes
            # a validity column, so a consumer of replayed rows reads one
            # contract whatever the topology; an empty leg may carry none.
            return torch.zeros(0, dtype=torch.bool, device=device)
        if inbound.row_valid is None and outbound.row_valid is None:
            return None
        ones = torch.ones(rows, dtype=torch.bool, device=device)
        valid_in = ones if inbound.row_valid is None else inbound.row_valid.index_select(0, self.inbound_row)
        valid_out = ones if outbound.row_valid is None else outbound.row_valid.index_select(0, self.outbound_row)
        return valid_in & valid_out


TARGET = "target"

#: A row that scatters off declared clutter geometry, or off a site the caller
#: declared to be clutter. It is not a lesser target echo: it is the return the
#: environment makes on its own, and it is coherent with the target return
#: because it lives in the same pair segment of the same waveform kernel.
ENVIRONMENT_CLUTTER = "environment_clutter"

#: The direct transmitter-to-receiver route: no scatter site and no declared
#: clutter interaction. A site-less row that DOES touch declared clutter
#: geometry - transmitter to wall to receiver - is environment clutter, not
#: leakage, because it is the environment's return and not the antenna
#: coupling term.
DIRECT_LEAKAGE = "direct_leakage"

#: A round trip that interacted more than the declared depth on either leg.
#: Kept separate from clutter because a multibounce return through a target is
#: neither a clean target echo nor an environment-only one.
MULTI_INTERACTION = "multi_interaction"

#: The declared taxonomy, in the order the index publishes it. Every class is
#: always published, including an empty one: a caller that sums the per-class
#: cubes must be able to iterate a fixed list rather than discover which classes
#: this particular frame happened to produce.
COMPONENT_NAMES: tuple[str, ...] = (TARGET, ENVIRONMENT_CLUTTER, DIRECT_LEAKAGE, MULTI_INTERACTION)

#: The value both interaction sequences carry where a row interacted with
#: nothing. Channel publishes it on every line-of-sight row.
NO_INTERACTION = -1


def _int_set(values: Any, name: str) -> frozenset[int]:
    if values is None:
        return frozenset()
    if isinstance(values, torch.Tensor):
        raise TypeError(f"{name} is a host declaration about the scene, not a device tensor; pass a set of stable IDs")
    if isinstance(values, (int, str)):
        raise TypeError(f"{name} must be an iterable of ints, got {type(values).__name__}")
    return frozenset(int(value) for value in values)


@dataclass(frozen=True, slots=True, eq=False)
class ComponentDeclaration:
    """What the caller says this scene contains.

    ``target_site_ids`` and ``clutter_site_ids`` are stable world IDs of scatter
    sites. ``clutter_material_slots`` are COMPILED material slots, which is what
    a frozen leg row carries in ``material_sequence``; a round trip that
    interacts with one of them is environment clutter no matter which site it
    also reached, because an echo that bounced off the wall on its way to the
    target is not a clean target echo.

    ``multi_interaction_depth`` is the deepest leg still treated as a simple
    return. The default ``1`` keeps single-bounce reflections in the clutter or
    target classes and sends anything deeper to
    :data:`MULTI_INTERACTION`. The three-way distinction (target echo /
    environment clutter / multi-interaction echo) is a declaration over ONE
    topology, not a join mode.

    A site declared both target and clutter is refused here rather than
    resolved: the two exports would overlap and the coherent recombination law
    would double-count it.
    """

    target_site_ids: frozenset[int] = frozenset()
    clutter_site_ids: frozenset[int] = frozenset()
    clutter_material_slots: frozenset[int] = frozenset()
    multi_interaction_depth: int = 1

    def __post_init__(self) -> None:
        for name in ("target_site_ids", "clutter_site_ids", "clutter_material_slots"):
            object.__setattr__(self, name, _int_set(getattr(self, name), name))
        if type(self.multi_interaction_depth) is not int or (self.multi_interaction_depth < 0):
            raise ValueError(
                f"multi_interaction_depth must be a non-negative int, got {self.multi_interaction_depth!r}"
            )
        overlap = self.target_site_ids & self.clutter_site_ids
        if overlap:
            raise ValueError(
                f"sites {sorted(overlap)} are declared both target and clutter; "
                "the two component exports would overlap and their coherent sum "
                "would count those rows twice"
            )

    def classify(self, *, site_id: int, depth: int, material_slots: frozenset[int]) -> tuple[str, ...]:
        """Every class this row belongs to, evaluated independently.

        The four predicates are written separately rather than as an if/elif
        ladder on purpose. A ladder makes "exactly one class" true by
        construction and therefore unassertable; evaluating all four lets the
        index refuse a declaration that produces none or two.

        ``direct_leakage`` is the route with no scatter site AND no declared
        clutter interaction, which is a narrower predicate than "no site". The
        difference is a real scene: the transmitter-to-wall-to-receiver path
        also has no site, and calling it leakage would file the strongest
        environment return in the frame under the name of the antenna coupling
        term. A site-less row that touches declared clutter geometry is
        environment clutter, which is what it is.
        """

        direct = site_id == NO_SITE
        deep = depth > self.multi_interaction_depth
        clutter = bool(material_slots & self.clutter_material_slots) or (site_id in self.clutter_site_ids)
        matched: list[str] = []
        if not direct and not deep and not clutter and (site_id in self.target_site_ids):
            matched.append(TARGET)
        if not deep and clutter:
            matched.append(ENVIRONMENT_CLUTTER)
        if direct and not deep and not clutter:
            matched.append(DIRECT_LEAKAGE)
        if deep:
            matched.append(MULTI_INTERACTION)
        return tuple(matched)


def _leg_facts(leg: FrozenLeg, name: str) -> tuple[list[int], list[frozenset[int]]]:
    """One frozen leg's depth and interacted material slots, read once."""

    depth = [int(value) for value in leg.depth.tolist()]
    materials = [
        frozenset(int(value) for value in row if int(value) != NO_INTERACTION) for row in leg.material_sequence.tolist()
    ]
    if len(depth) != len(materials):
        raise ValueError(f"{name} leg publishes {len(depth)} depths and {len(materials)} material sequences")
    return depth, materials


@dataclass(frozen=True, slots=True, eq=False)
class RadarComponentIndex:
    """Which component class owns each composed row.

    ``topology`` is held by REFERENCE and is the object every export must
    share. ``class_id`` is ``[path_count]`` int32 on the batch device, indexing
    :attr:`names`. ``counts`` is the same partition counted on the host at build
    time, so a test or a report can say "this class has four rows" without
    reading the device.
    """

    topology: RadarPathTopology
    class_id: torch.Tensor
    names: tuple[str, ...]
    counts: tuple[int, ...]
    declaration: ComponentDeclaration

    @property
    def row_count(self) -> int:
        return int(self.class_id.shape[0])

    def index_of(self, name: str) -> int:
        if name not in self.names:
            raise KeyError(f"{name!r} is not a declared component; this index publishes {list(self.names)}")
        return self.names.index(name)

    def count(self, name: str) -> int:
        """How many rows this class owns. A host int, decided at build time."""

        return self.counts[self.index_of(name)]

    def mask(self, name: str) -> torch.Tensor:
        """``[path_count]`` bool selecting this class's rows.

        Derived from ``class_id`` rather than stored, so no caller can hold a
        reference to a mask and mutate the index behind another caller's back.
        """

        return self.class_id == self.index_of(name)

    @classmethod
    def from_two_way(
        cls,
        composer: TwoWayComposer,
        inbound: FrozenLeg,
        outbound: FrozenLeg,
        declaration: ComponentDeclaration,
        direct: FrozenLeg | None = None,
    ) -> RadarComponentIndex:
        """Classify a two-way join's rows from its frozen leg topologies.

        ``composer`` is a :class:`~witwin.radar.paths.TwoWayComposer`
        and ``inbound`` / ``outbound`` (and ``direct``, when the join composed
        site-less rows) are the frozen leg handles it was frozen against; all
        are duck typed so this module adds no import edge to the Channel
        adapter. A site-less row's depth and materials are its direct leg row's.
        """

        if not isinstance(declaration, ComponentDeclaration):
            raise TypeError(f"declaration must be a ComponentDeclaration, got {type(declaration).__name__}")
        topology = composer.topology
        if (topology.direct_row is None) != (direct is None):
            raise ValueError("pass the direct leg exactly when the join composed site-less rows")
        site = [int(value) for value in topology.site_id.tolist()]
        inbound_row = [int(value) for value in topology.inbound_row.tolist()]
        outbound_row = [int(value) for value in topology.outbound_row.tolist()]
        direct_row = [-1] * len(site) if topology.direct_row is None else topology.direct_row.tolist()
        in_depth, in_materials = _leg_facts(inbound, "inbound")
        out_depth, out_materials = _leg_facts(outbound, "outbound")
        direct_depth, direct_materials = ([], []) if direct is None else _leg_facts(direct, "direct")

        classes: list[int] = []
        for row, site_id in enumerate(site):
            if site_id == NO_SITE:
                depth = direct_depth[int(direct_row[row])]
                materials = direct_materials[int(direct_row[row])]
            else:
                depth = max(in_depth[inbound_row[row]], out_depth[outbound_row[row]])
                materials = in_materials[inbound_row[row]] | out_materials[outbound_row[row]]
            matched = declaration.classify(site_id=site_id, depth=depth, material_slots=materials)
            if len(matched) != 1:
                raise ValueError(
                    f"composed row {row} (site {site_id}, depth {depth}, "
                    f"material slots {sorted(materials)}) matches "
                    f"{list(matched)}; every row must belong to exactly one "
                    "component class, so a row matching none is an undeclared "
                    "site and a row matching two is a declaration that "
                    "contradicts itself"
                )
            classes.append(COMPONENT_NAMES.index(matched[0]))

        counts = tuple(sum(1 for value in classes if value == index) for index in range(len(COMPONENT_NAMES)))
        return cls(
            topology=topology,
            class_id=torch.tensor(classes, dtype=torch.int32, device=topology.site_id.device),
            names=COMPONENT_NAMES,
            counts=counts,
            declaration=declaration,
        )


__all__ = [
    "COMPONENT_NAMES",
    "NO_SITE",
    "DIRECT_LEAKAGE",
    "ENVIRONMENT_CLUTTER",
    "MULTI_INTERACTION",
    "TARGET",
    "ComponentDeclaration",
    "RadarComponentIndex",
    "RadarPathBatch",
    "RadarPathTopology",
    "TwoWayComposer",
]
