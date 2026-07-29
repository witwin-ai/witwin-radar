"""Radar round-trip path composition.

Two composers, one result contract. :class:`TwoWayComposer` joins an inbound
and an outbound leg through a scatter site; :class:`DirectComposer` publishes a
single source-to-sink leg with no site at all. The mode is chosen explicitly by
the caller and recorded on the batch, so nothing downstream has to infer it and
there is no path by which one silently becomes the other.

:class:`RadarComponentIndex` is a third thing and it is a SIDECAR: it names
what each composed row is - target echo, environment clutter, direct leakage,
multi-interaction - without adding a column to :class:`RadarPathTopology`.
Every component export therefore shares the same topology OBJECT, which is what
makes "processing does not change propagation row identity" a checkable
statement rather than a claim.

The module exports the contracts and the composers. It does not import
the Channel adapter; the composers duck-type the frozen leg handles they are
given, so this package never crosses the Channel boundary either.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.autograd.forward_ad as forward_ad

from .cuda import native_ops as _ops
from .policy import first_order_only
from .propagation import RadarLegBatch, require_wideband_pair

LegKey = tuple[int, int, tuple[int, ...], tuple[int, ...]]


def stable_ids(values, name: str) -> list[int]:
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


def leg_identity(frozen, name: str) -> tuple[list[int], list[int], list[LegKey]]:
    """Read one frozen leg's row identity to the host, once.

    The key is everything that distinguishes two rows of the SAME leg: which
    multipath component, how deep, and which primitives and materials it
    interacted with. It is frame invariant, so the composed order it induces is
    frame invariant too.
    """

    source = [int(value) for value in frozen.source_id.tolist()]
    sink = [int(value) for value in frozen.sink_id.tolist()]
    component = [int(value) for value in frozen.component_id.tolist()]
    depth = [int(value) for value in frozen.depth.tolist()]
    primitive = [tuple(int(value) for value in row) for row in frozen.primitive_sequence.tolist()]
    material = [tuple(int(value) for value in row) for row in frozen.material_sequence.tolist()]
    rows = len(source)
    for label, column in (
        ("sink_id", sink),
        ("component_id", component),
        ("depth", depth),
        ("primitive_sequence", primitive),
        ("material_sequence", material),
    ):
        if len(column) != rows:
            raise ValueError(f"{name} leg {label} has {len(column)} rows, expected {rows}")
    keys: list[LegKey] = [(component[row], depth[row], primitive[row], material[row]) for row in range(rows)]
    return source, sink, keys


def group_rows(source: list[int], sink: list[int], keys: list[LegKey], name: str) -> dict[tuple[int, int], list[int]]:
    """Index a leg's rows by its ``(source_id, sink_id)`` endpoint pair.

    Also enforces that the identity key is UNIQUE inside each endpoint pair. A
    collision would make the canonical composed order ambiguous and would
    silently turn the permutation test vacuous, so it is refused here rather
    than tie-broken on row position - which is exactly the positional
    dependence this module exists to remove.
    """

    groups: dict[tuple[int, int], list[int]] = {}
    seen: dict[tuple[int, int], dict[LegKey, int]] = {}
    for row, endpoints in enumerate(zip(source, sink, strict=True)):
        groups.setdefault(endpoints, []).append(row)
        claimed = seen.setdefault(endpoints, {})
        if keys[row] in claimed:
            raise ValueError(
                f"{name} leg rows {claimed[keys[row]]} and {row} share the "
                f"identity key {keys[row]} within endpoint pair {endpoints}; "
                "the composed order would be ambiguous"
            )
        claimed[keys[row]] = row
    return groups


def sink_major_rank(sources: list[int], sinks: list[int]):
    """The sensor-pair index, mirroring the Channel consumer's convention.

    Channel computes ``sink_row_index * source_count + source_row_index`` and
    spans ``source_count * sink_count``. Using anything else here would put a
    second, silently different, virtual-array numbering on the same data.
    """

    source_rank = {value: rank for rank, value in enumerate(sources)}
    sink_rank = {value: rank for rank, value in enumerate(sinks)}

    def rank(source: int, sink: int) -> int:
        return sink_rank[sink] * len(sources) + source_rank[source]

    return rank


def pair_offsets(pair_of_row: list[int], pair_count: int) -> list[int]:
    """A half-open partition of composed rows by sensor pair.

    Empty segments are legal and expected: the partition spans the FRONT END's
    cross product, so a pair that discovered nothing still owns a segment and
    the IQ cube keeps its declared shape.

    The kernel CLAMPS a malformed offsets table rather than failing, because
    reading its values per frame would be exactly the D2H the fixed-topology
    capability exists to avoid, and clamping turns a malformed table into a
    plausible wrong answer. So the gate lives here, at freeze time, where the
    table is still a Python list and checking it is free. What is actually
    checkable is the input: once every row's pair rank is in range, counting
    them can only produce a valid partition.
    """

    if pair_count < 1:
        raise ValueError(f"pair_count must be positive, got {pair_count}")
    counts = [0] * pair_count
    for row, pair in enumerate(pair_of_row):
        if not 0 <= pair < pair_count:
            raise ValueError(
                f"composed row {row} claims sensor pair {pair}, which is "
                f"outside the declared {pair_count} pairs; the offsets table "
                "would not partition all composed rows"
            )
        counts[pair] += 1
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + count)
    return offsets


def csr(owner_of_row: list[int], owner_count: int) -> tuple[list[int], list[int]]:
    """Group composed rows by an owner index, as a CSR offsets/rows pair.

    The VJP needs this: one thread owns one gradient slot and loops its own
    segment, so the reduction needs no atomics and its summation order is fixed
    by the frozen join. That is what makes a bit-identical gradient comparison
    across a leg permutation a legitimate assertion rather than a lucky one.
    """

    buckets: list[list[int]] = [[] for _ in range(owner_count)]
    for composed_row, owner in enumerate(owner_of_row):
        buckets[owner].append(composed_row)
    offsets = [0]
    rows: list[int] = []
    for bucket in buckets:
        rows.extend(bucket)
        offsets.append(len(rows))
    return offsets, rows


JoinMode = Literal["direct", "multipath"]

JOIN_MODES: frozenset[str] = frozenset({"direct", "multipath"})


def _require_tensor(name: str, value: object, *, dtype: torch.dtype, shape: tuple[int, ...]) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}")
    if value.dtype != dtype:
        raise TypeError(f"{name} must use {dtype}, got {value.dtype}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
    return value


@dataclass(frozen=True, slots=True, eq=False)
class RadarPathTopology:
    """The identity of each composed round-trip row.

    The tuple ``(radar_source_id, site_id, radar_sink_id)`` is the row's
    identity and is stable across a frozen sequence. ``inbound_row`` and
    ``outbound_row`` record which frozen leg rows were joined, so a composed
    result can always be traced back to the two legs that produced it.

    A DIRECT row - radar source straight to radar sink, with no scatter site -
    uses ``site_id = -1`` and ``outbound_row = -1``. Those are sentinels, not
    missing data: a direct path has exactly one leg, and giving it a fabricated
    second one would make it indistinguishable from a round trip through a
    target with unit response.

    Identity is what the join uses. Joining by array position instead would be
    silently wrong the moment a leg publishes its rows in a different order,
    and the resulting error looks like a physics bug rather than a bookkeeping
    one.
    """

    radar_source_id: torch.Tensor
    site_id: torch.Tensor
    radar_sink_id: torch.Tensor
    inbound_row: torch.Tensor
    outbound_row: torch.Tensor

    def __post_init__(self) -> None:
        rows = (int(self.radar_source_id.shape[0]),)
        for name in ("radar_source_id", "site_id", "radar_sink_id", "inbound_row", "outbound_row"):
            _require_tensor(name, getattr(self, name), dtype=torch.int64, shape=rows)

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

    ``delay_rate`` is ``d(total_delay_s)/dt`` and is primal-valued: it arrives
    as an unpacked forward tangent, so consuming it here deliberately severs
    the second-order ``d(delay_rate)/dx`` term.

    RETARDATION, stated so an absurd-velocity test cannot be misread as a bug.
    ``delay_rate`` is ``rate_in + rate_out`` with BOTH legs evaluated at the
    same world instant ``t``. The exact two-way rate evaluates the outbound leg
    at ``t + tau_in``, where the target has moved on, and carries a
    ``(1 - v_r/c)`` factor from the same retardation. The relative error of the
    same-instant form is therefore ``O(v/c)``: about ``4e-8`` at 12 m/s, which
    is five orders of magnitude below the float32 delay quantisation these rows
    are published at. It is an approximation, it is named here rather than left
    implicit, and it is not corrected because the correction is smaller than the
    representation. A test driven at a relativistic velocity measures this
    approximation; it has not found a defect.

    ``row_valid`` is the sole authority on whether a row means anything. A
    dead row is a complete answer contributing exactly zero, never an error,
    and validity is never inferred from a zero payload.

    ``frequency_response`` and ``frequency_offsets_hz`` are the composed band,
    present or absent together and validated by the same host-only rule the leg
    batch and the synthesis batch use. The response is ``[path_count, F]``
    complex64 and column ``j`` is the round trip composed ENTIRELY at
    ``reference_frequency_hz + frequency_offsets_hz[j]``: both legs' transport
    at that frequency, multiplied by the scatter response.

    What the band does NOT contain, stated so it is not assumed: the scatter
    response is evaluated once and reused across the band. A wideband TARGET
    response - an RCS that varies across the band - is a separate capability
    with its own owner, and its absence here means a composed column is
    ``H_in(f_j) * S(f_ref) * H_out(f_j)``. The propagation and material band
    shape is exact; the target's is frozen.

    ``join_mode`` records which composer produced these rows. It is stored
    rather than inferred so that "which paths am I looking at" is a checkable
    property of the result and never a guess from its shape. Both modes publish
    THIS contract, so a consumer downstream of it - synthesis, in particular -
    needs no branch; the choice is made once, by the caller, upstream.

    ``weight_includes_antenna_pattern`` is the fourth provenance boolean of the
    pipeline and the only one a composer does not already know. A composed
    weight is Channel-sourced, so it always carries the reference-frequency
    phase, the free-space spreading and the transmit power; it carries the
    ARRAY's transmit and receive pattern gain only after
    :meth:`witwin.radar.sensors.RoundTripPatternStage.apply` has run. Both
    composers publish ``False`` because neither applies a pattern, and that
    stage is the one producer that publishes ``True``. It exists to be READ: the
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
    join_mode: JoinMode
    frequency_response: torch.Tensor | None = None
    frequency_offsets_hz: torch.Tensor | None = None
    weight_includes_antenna_pattern: bool = False

    def __post_init__(self) -> None:
        if type(self.weight_includes_antenna_pattern) is not bool:
            raise TypeError("weight_includes_antenna_pattern must be a bool")
        if self.join_mode not in JOIN_MODES:
            raise ValueError(f"join_mode must be one of {sorted(JOIN_MODES)}, got {self.join_mode!r}")
        if type(self.sensor_pair_count) is not int or self.sensor_pair_count < 1:
            raise ValueError("sensor_pair_count must be a positive int")
        if type(self.path_count) is not int or self.path_count < 0:
            raise ValueError("path_count must be a non-negative int")
        rows = (self.path_count,)
        _require_tensor("sensor_pair_index", self.sensor_pair_index, dtype=torch.int64, shape=rows)
        _require_tensor("pair_offsets", self.pair_offsets, dtype=torch.int64, shape=(self.sensor_pair_count + 1,))
        _require_tensor("total_delay_s", self.total_delay_s, dtype=torch.float32, shape=rows)
        _require_tensor("complex_transfer_ref", self.complex_transfer_ref, dtype=torch.complex64, shape=rows)
        if self.delay_rate is not None:
            _require_tensor("delay_rate", self.delay_rate, dtype=torch.float32, shape=rows)
        if self.row_valid is not None:
            _require_tensor("row_valid", self.row_valid, dtype=torch.bool, shape=rows)
        if self.topology.row_count != self.path_count:
            raise ValueError("topology must have exactly path_count rows")
        require_wideband_pair(self.frequency_response, self.frequency_offsets_hz, self.path_count)

    @property
    def device(self) -> torch.device:
        return self.total_delay_s.device

    @property
    def band_count(self) -> int:
        """How many frequency columns this batch carries, ``0`` when narrowband."""

        if self.frequency_offsets_hz is None:
            return 0
        return int(self.frequency_offsets_hz.shape[0])


def validate_pair_ordering(sensor_pair_index, *, num_tx, num_rx, sensor_pair_count):
    """Validate the frozen sink-major sensor-pair partition once on the host."""

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
    if sensor_pair_index.dtype != torch.int64:
        raise TypeError(f"sensor_pair_index must be int64, got {sensor_pair_index.dtype}")
    if sensor_pair_index.dim() != 1:
        raise ValueError(f"sensor_pair_index must be 1-D, got shape {tuple(sensor_pair_index.shape)}")
    previous = -1
    for row, rank in enumerate(sensor_pair_index.tolist()):
        if rank < 0 or rank >= expected:
            raise ValueError(
                f"row {row} names sensor pair {rank}, which is outside the "
                f"{num_tx} x {num_rx} array's range [0, {expected})"
            )
        if rank < previous:
            raise ValueError(
                "sensor pair ranks must be non-decreasing so that pair_offsets "
                f"is a half-open partition; row {row} drops from {previous} to {rank}"
            )
        previous = rank


def _primal_rate(delay_rate: torch.Tensor | None, rows: int, device: torch.device, name: str) -> torch.Tensor:
    """The leg's delay rate as a strictly primal kernel input.

    ``delay_rate`` is ``d(delay_s)/dt`` unpacked from a forward-only dual and
    published as a PRIMAL value, which deliberately severs the second-order
    ``d(delay_rate)/dx`` term the contract does not claim. The join therefore
    returns ``None`` for its gradient and a zero tangent for the composed rate.

    "Returns None" and "silently dropped a gradient" look identical from the
    outside, so a rate that arrives carrying a tape is REFUSED here rather than
    quietly zeroed.
    """

    if delay_rate is None:
        return torch.zeros(rows, dtype=torch.float32, device=device)
    if delay_rate.requires_grad:
        raise ValueError(
            f"{name} delay_rate carries requires_grad; it is a primal Doppler "
            "rate by contract and the join would return None for its gradient"
        )
    if forward_ad.unpack_dual(delay_rate).tangent is not None:
        raise ValueError(
            f"{name} delay_rate carries a forward tangent; it is a primal "
            "Doppler rate by contract and the join publishes a zero tangent "
            "for the composed rate"
        )
    return delay_rate.contiguous()


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
        tau_in,
        tau_out,
        rate_in,
        rate_out,
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
    ):
        rows = int(idx_in.shape[0])
        empty = torch.empty(rows, dtype=torch.float32, device=tau_in.device)
        tau_rt = empty
        rate_rt = torch.empty_like(empty)
        c_rt_re = torch.empty_like(empty)
        c_rt_im = torch.empty_like(empty)
        _ops().two_way_join_forward(
            tau_in,
            tau_out,
            rate_in,
            rate_out,
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
            rate_rt,
            c_rt_re,
            c_rt_im,
            rows,
        )
        return tau_rt, rate_rt, c_rt_re, c_rt_im

    @staticmethod
    def setup_context(ctx, inputs, output):
        (
            _tau_in,
            _tau_out,
            _rate_in,
            _rate_out,
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
    def backward(ctx, grad_tau_rt, grad_rate_rt, grad_c_rt_re, grad_c_rt_im):
        (c_in_re, c_in_im, c_out_re, c_out_im, s_re, s_im, row_valid, idx_in, idx_out, idx_s) = ctx.saved_tensors
        join = ctx.join
        response_offsets, response_rows, response_slots = ctx.response_family
        # grad_rate_rt is discarded, and that is exact rather than lossy:
        # rate_rt depends only on the two rate inputs, both of which are primal
        # by contract, so every row of its Jacobian against a differentiable
        # input is structurally zero.
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
            response_offsets,
            response_rows,
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
            int(idx_in.shape[0]),
            join.inbound_row_count,
            join.outbound_row_count,
            response_slots,
        )
        return (
            grad_tau_in,
            grad_tau_out,
            None,
            None,
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
        ctx,
        tan_tau_in,
        tan_tau_out,
        tan_rate_in,
        tan_rate_out,
        tan_c_in_re,
        tan_c_in_im,
        tan_c_out_re,
        tan_c_out_im,
        tan_s_re,
        tan_s_im,
        tan_row_valid,
        tan_idx_in,
        tan_idx_out,
        tan_idx_s,
        tan_join,
        tan_response_family,
    ):
        (c_in_re, c_in_im, c_out_re, c_out_im, s_re, s_im, row_valid, idx_in, idx_out, idx_s) = ctx.saved_tensors
        # tan_rate_in / tan_rate_out are ignored, and the refusal that makes
        # that honest lives in _primal_rate, at the facade. Autograd hands this
        # callback a zero-filled tangent for an input that carries none, so a
        # check HERE could not tell "no tangent" from "a genuine zero" and
        # would only be a comment with a raise attached. The facade refuses a
        # rate input that is a dual at all, which is checkable.

        def inbound(tangent):
            return torch.zeros_like(c_in_re) if tangent is None else tangent.contiguous()

        def outbound(tangent):
            return torch.zeros_like(c_out_re) if tangent is None else tangent.contiguous()

        def site(tangent):
            return torch.zeros_like(s_re) if tangent is None else tangent.contiguous()

        rows = int(idx_in.shape[0])
        tan_tau_rt = torch.empty(rows, dtype=torch.float32, device=c_in_re.device)
        tan_rate_rt = torch.empty_like(tan_tau_rt)
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
            tan_rate_rt,
            tan_c_rt_re,
            tan_c_rt_im,
            rows,
        )
        return tan_tau_rt, tan_rate_rt, tan_c_rt_re, tan_c_rt_im


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

    @classmethod
    def freeze(
        cls, inbound, outbound, site_ids, *, radar_source_ids, radar_sink_ids, reference_frequency_hz: float
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
        legitimate. What is refused is the reverse - a declared site with no
        row at all in one of the legs - because that is a wrong stable ID, and
        dropping it silently is how a join produces a plausible empty answer.
        """

        device = inbound.sink_id.device
        sources = stable_ids(radar_source_ids, "radar_source_ids")
        sinks = stable_ids(radar_sink_ids, "radar_sink_ids")
        sites = stable_ids(site_ids, "site_ids")
        sites.sort()

        inbound_source, inbound_sink, inbound_keys = leg_identity(inbound, "inbound")
        outbound_source, outbound_sink, outbound_keys = leg_identity(outbound, "outbound")
        arriving = group_rows(inbound_source, inbound_sink, inbound_keys, "inbound")
        leaving = group_rows(outbound_source, outbound_sink, outbound_keys, "outbound")
        pair_rank = sink_major_rank(sources, sinks)

        # (pair_rank, site_rank, source, site, sink, inbound_row, outbound_row,
        #  inbound_key, outbound_key)
        # Nothing may fall outside the declared front end. A leg row whose
        # radar endpoint is not in the declared lists would simply never be
        # visited below, which is a silent drop rather than an empty segment.
        stray_sources = sorted(set(inbound_source) - set(sources))
        if stray_sources:
            raise ValueError(
                f"inbound leg rows carry radar source IDs {stray_sources} that are not in radar_source_ids {sources}"
            )
        stray_sinks = sorted(set(outbound_sink) - set(sinks))
        if stray_sinks:
            raise ValueError(
                f"outbound leg rows carry radar sink IDs {stray_sinks} that are not in radar_sink_ids {sinks}"
            )

        # A site absent from a leg ENTIRELY is a caller error: the site list is
        # the declaration of what this join is about, and silently dropping one
        # would hide a wrong stable ID. A site absent for ONE endpoint is not -
        # that is discovery reporting that this particular TX/RX pair sees
        # nothing there, and it is published as an empty pair segment.
        reachable_in = {endpoints[1] for endpoints in arriving}
        reachable_out = {endpoints[0] for endpoints in leaving}
        for site in sites:
            if site not in reachable_in:
                raise ValueError(f"site {site} has no inbound leg row in the frozen topology")
            if site not in reachable_out:
                raise ValueError(f"site {site} has no outbound leg row in the frozen topology")

        rows: list[tuple[int, int, int, int, int, int, int, LegKey, LegKey]] = []
        for site_rank, site in enumerate(sites):
            for source in sources:
                inbound_rows = arriving.get((source, site), ())
                for sink in sinks:
                    outbound_rows = leaving.get((site, sink), ())
                    for i in inbound_rows:
                        for o in outbound_rows:
                            rows.append(
                                (pair_rank(source, sink), site_rank, source, site, sink, i, o)
                                + (inbound_keys[i], outbound_keys[o])
                            )

        # The canonical order. Every component is frame invariant, so two
        # freezes of permuted leg rows produce the SAME composed sequence, not
        # merely the same composed set.
        rows.sort(key=lambda row: (row[0], row[1], row[7], row[8]))

        def column(index: int) -> torch.Tensor:
            return torch.tensor([row[index] for row in rows], dtype=torch.int64, device=device)

        pair_count = len(sources) * len(sinks)
        sensor_pair_index = column(0)
        validate_pair_ordering(sensor_pair_index, num_tx=len(sources), num_rx=len(sinks), sensor_pair_count=pair_count)
        offsets = pair_offsets([row[0] for row in rows], pair_count)

        inbound_count = len(inbound_source)
        outbound_count = len(outbound_source)
        by_inbound = csr([row[5] for row in rows], inbound_count)
        by_outbound = csr([row[6] for row in rows], outbound_count)
        by_response = csr([row[1] for row in rows], len(sites))

        def table(values: list[int]) -> torch.Tensor:
            return torch.tensor(values, dtype=torch.int64, device=device)

        return cls(
            inbound_row=column(5),
            outbound_row=column(6),
            response_slot=column(1),
            topology=RadarPathTopology(
                radar_source_id=column(2),
                site_id=column(3),
                radar_sink_id=column(4),
                inbound_row=column(5),
                outbound_row=column(6),
            ),
            sensor_pair_index=sensor_pair_index,
            pair_offsets=table(offsets),
            sensor_pair_count=pair_count,
            site_count=len(sites),
            inbound_row_count=inbound_count,
            outbound_row_count=outbound_count,
            by_inbound_offsets=table(by_inbound[0]),
            by_inbound_rows=table(by_inbound[1]),
            by_outbound_offsets=table(by_outbound[0]),
            by_outbound_rows=table(by_outbound[1]),
            by_response_offsets=table(by_response[0]),
            by_response_rows=table(by_response[1]),
            reference_frequency_hz=float(reference_frequency_hz),
            row_slot=torch.arange(len(rows), dtype=torch.int64, device=device),
            by_row_offsets=torch.arange(len(rows) + 1, dtype=torch.int64, device=device),
            by_row_rows=torch.arange(len(rows), dtype=torch.int64, device=device),
            outbound_max_depth=max((row[8][1] for row in rows), default=0),
        )

    @property
    def path_count(self) -> int:
        return int(self.inbound_row.shape[0])

    def compose(
        self, inbound: RadarLegBatch, outbound: RadarLegBatch, response, *, include_delay_rate: bool = True
    ) -> RadarPathBatch:
        """Compose one frame's round-trip rows. Device work only.

        ``include_delay_rate`` exists because a forward-AD dual carries exactly
        one meaning at a time. When the dual direction is a site VELOCITY, the
        unpacked delay tangent is a Doppler rate and belongs in the result. When
        the dual direction is a position PERTURBATION, the same tangent is a
        directional derivative and reusing it as a rate would silently mix two
        meanings. The caller states which it has.

        The composed rate is ``rate_in + rate_out`` and BOTH legs are evaluated
        at the same world instant. The exact two-way rate evaluates the outbound
        leg at ``t + tau_in`` and carries a ``(1 - v_r/c)`` factor; the
        same-instant form is wrong by ``O(v/c)``, about ``4e-8`` at 12 m/s and
        far below the float32 delay quantisation. Stated here because it is the
        one approximation in this composition that a velocity, rather than a
        geometry, can make visible: driving the join at a relativistic velocity
        measures it and has not found a defect. ``RadarPathBatch.delay_rate``
        carries the same statement for the row that leaves here.

        A dead row's payload is exactly zero, not a partial composition. The
        row is a complete answer that this round trip does not exist at these
        endpoint positions; publishing ``tau_in + 0`` for it would be a
        plausible number that no consumer should ever read.
        """

        self._require_frame(inbound, outbound)
        rows = self.path_count
        device = inbound.delay_s.device
        row_valid = self._row_validity(inbound, outbound, rows, device)
        flags = torch.ones(rows, dtype=torch.int32, device=device) if row_valid is None else row_valid.to(torch.int32)
        band = self._band(inbound, outbound)
        response_re, response_im, response_index, response_family = self._response(
            response, inbound, outbound, flags, device
        )

        # Torch-owned, autograd-aware accessors: the real pairs cross the
        # boundary, never the complex tensors.
        tau_rt, rate_rt, transfer_re, transfer_im = _TwoWayJoin.apply(
            inbound.delay_s.contiguous(),
            outbound.delay_s.contiguous(),
            _primal_rate(inbound.delay_rate, inbound.leg_count, device, "inbound"),
            _primal_rate(outbound.delay_rate, outbound.leg_count, device, "outbound"),
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

        frequency_response = self._compose_band(
            band, inbound, outbound, response_re, response_im, response_index, response_family, flags
        )

        publish_rate = include_delay_rate and inbound.delay_rate is not None and outbound.delay_rate is not None
        return RadarPathBatch(
            sensor_pair_count=self.sensor_pair_count,
            path_count=rows,
            sensor_pair_index=self.sensor_pair_index,
            pair_offsets=self.pair_offsets,
            total_delay_s=tau_rt,
            delay_rate=rate_rt if publish_rate else None,
            complex_transfer_ref=torch.complex(transfer_re, transfer_im),
            reference_frequency_hz=self.reference_frequency_hz,
            row_valid=row_valid,
            topology=self.topology,
            join_mode="multipath",
            frequency_response=frequency_response,
            frequency_offsets_hz=(None if band is None else inbound.frequency_offsets_hz),
        )

    def _band(self, inbound: RadarLegBatch, outbound: RadarLegBatch) -> int | None:
        """The two legs' agreed band width, or ``None`` when neither has one.

        Both legs or neither. A round trip composed from one banded leg and one
        narrowband leg would have to broadcast the narrowband leg's single
        coefficient across the band, which is the narrowband approximation
        reintroduced silently on exactly one half of the round trip - the
        failure mode this whole capability exists to remove.
        """

        counts = (inbound.band_count, outbound.band_count)
        if counts == (0, 0):
            return None
        if 0 in counts:
            raise ValueError(
                f"the inbound leg carries {counts[0]} frequency columns and the "
                f"outbound leg carries {counts[1]}; a round trip is composed at "
                "one frequency at a time, so both legs must be evaluated over "
                "the same band or neither"
            )
        if counts[0] != counts[1]:
            raise ValueError(
                f"the two legs carry {counts[0]} and {counts[1]} frequency "
                "columns; they must be evaluated over the same band"
            )
        if not torch.equal(inbound.frequency_offsets_hz, outbound.frequency_offsets_hz):
            raise ValueError(
                "the two legs were evaluated over different frequency grids; a "
                "composed column multiplies one leg's response at f by the "
                "other's at the SAME f, so the grids must agree"
            )
        return counts[0]

    def _compose_band(self, band, inbound, outbound, response_re, response_im, response_index, response_family, flags):
        """Compose ``H_in(f_j) * S * H_out(f_j)`` for every column of the band.

        The frequency axis is a PYTHON LOOP over the existing ``[K]`` join
        primitive, not a strided ``[K, F]`` kernel. That is a deliberate Phase-8
        boundary: widening ``two_way_join.cu`` means widening its primal, its
        JVP and its VJP together, and it needs a measured reason first. The loop
        costs one launch per column and reproduces the reference column exactly,
        so the measurement can be made against something that already works.

        ``tau_rt`` and ``rate_rt`` are recomputed by every column and discarded:
        they are functions of the two delays alone and are identical across the
        band. That redundancy is the price of not widening the kernel, and it is
        recorded rather than hidden.

        The scatter response is evaluated ONCE, above the loop, and the same
        real pair is handed to every column. A response that varied across the
        band would be a wideband TARGET model, which is a separate capability;
        reusing one value here is the honest statement that the target's
        response is frozen at the reference frequency while propagation is not.
        """

        if band is None:
            return None
        columns = []
        for index in range(band):
            _tau, _rate, column_re, column_im = _TwoWayJoin.apply(
                inbound.delay_s.contiguous(),
                outbound.delay_s.contiguous(),
                _primal_rate(inbound.delay_rate, inbound.leg_count, flags.device, "inbound"),
                _primal_rate(outbound.delay_rate, outbound.leg_count, flags.device, "outbound"),
                inbound.frequency_response[:, index].real.contiguous(),
                inbound.frequency_response[:, index].imag.contiguous(),
                outbound.frequency_response[:, index].real.contiguous(),
                outbound.frequency_response[:, index].imag.contiguous(),
                response_re,
                response_im,
                flags,
                self.inbound_row,
                self.outbound_row,
                response_index,
                self,
                response_family,
            )
            columns.append(torch.complex(column_re, column_im))
        return torch.stack(columns, dim=1)

    def _require_frame(self, inbound: RadarLegBatch, outbound: RadarLegBatch) -> None:
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

        for name, batch, expected in (
            ("inbound", inbound, self.inbound_row_count),
            ("outbound", outbound, self.outbound_row_count),
        ):
            if batch.leg_count != expected:
                raise ValueError(
                    f"{name} leg carries {batch.leg_count} rows but this join "
                    f"was frozen against {expected}; the frame does not belong "
                    "to this frozen topology"
                )

    def _response(self, response, inbound, outbound, flags, device):
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
                (self.by_response_offsets, self.by_response_rows, self.site_count),
            )
        if getattr(response, "native_row_owner", None) not in NATIVE_ROW_RESPONSE_OWNERS:
            raise NotImplementedError(
                "a geometry-dependent scatter response varies per path and must "
                "be evaluated in a native kernel, not composed here"
            )
        rows_re, rows_im = response.evaluate_rows(self, inbound, outbound, flags)
        for name, value in (("real", rows_re), ("imaginary", rows_im)):
            if not isinstance(value, torch.Tensor) or value.numel() != self.path_count:
                raise ValueError(
                    f"a row-evaluated scatter response must publish one {name} "
                    f"value per composed row; this join has {self.path_count}"
                )
        return (rows_re, rows_im, self.row_slot, (self.by_row_offsets, self.by_row_rows, self.path_count))

    def _site_response(self, response, device: torch.device) -> torch.Tensor:
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
        if inbound.row_valid is None and outbound.row_valid is None:
            return None
        ones = torch.ones(rows, dtype=torch.bool, device=device)
        valid_in = ones if inbound.row_valid is None else inbound.row_valid.index_select(0, self.inbound_row)
        valid_out = ones if outbound.row_valid is None else outbound.row_valid.index_select(0, self.outbound_row)
        return valid_in & valid_out


NO_SITE = -1
NO_OUTBOUND_ROW = -1


@dataclass(frozen=True, slots=True, eq=False)
class DirectComposer:
    """A frozen source-to-sink leg, published in canonical composed order."""

    row_index: torch.Tensor
    topology: RadarPathTopology
    sensor_pair_index: torch.Tensor
    pair_offsets: torch.Tensor
    sensor_pair_count: int
    reference_frequency_hz: float

    @classmethod
    def freeze(cls, leg, *, radar_source_ids, radar_sink_ids, reference_frequency_hz: float) -> DirectComposer:
        """Order one frozen leg's rows the way a composed batch is ordered.

        Same canonical key as the two-way join - sensor pair, then row identity
        - so a direct batch and a multipath batch of the same scene agree on
        what row order means.
        """

        device = leg.source_id.device
        sources = stable_ids(radar_source_ids, "radar_source_ids")
        sinks = stable_ids(radar_sink_ids, "radar_sink_ids")
        source, sink, keys = leg_identity(leg, "direct")
        group_rows(source, sink, keys, "direct")
        pair_rank = sink_major_rank(sources, sinks)

        stray_sources = sorted(set(source) - set(sources))
        if stray_sources:
            raise ValueError(
                f"leg rows carry radar source IDs {stray_sources} that are not in radar_source_ids {sources}"
            )
        stray_sinks = sorted(set(sink) - set(sinks))
        if stray_sinks:
            raise ValueError(f"leg rows carry radar sink IDs {stray_sinks} that are not in radar_sink_ids {sinks}")

        rows: list[tuple[int, int, int, int, LegKey]] = [
            (pair_rank(source[row], sink[row]), source[row], sink[row], row, keys[row]) for row in range(len(source))
        ]
        rows.sort(key=lambda row: (row[0], row[4]))

        def column(index: int) -> torch.Tensor:
            return torch.tensor([row[index] for row in rows], dtype=torch.int64, device=device)

        def constant(value: int) -> torch.Tensor:
            return torch.full((len(rows),), value, dtype=torch.int64, device=device)

        pair_count = len(sources) * len(sinks)
        sensor_pair_index = column(0)
        # Same freeze-time layout gate as the two-way join. A direct batch feeds
        # the same synthesis cube assembly, so it depends on the same sink-major
        # pair rank and has to be held to it in the same place.
        validate_pair_ordering(sensor_pair_index, num_tx=len(sources), num_rx=len(sinks), sensor_pair_count=pair_count)
        offsets = pair_offsets([row[0] for row in rows], pair_count)
        return cls(
            row_index=column(3),
            topology=RadarPathTopology(
                radar_source_id=column(1),
                site_id=constant(NO_SITE),
                radar_sink_id=column(2),
                inbound_row=column(3),
                outbound_row=constant(NO_OUTBOUND_ROW),
            ),
            sensor_pair_index=sensor_pair_index,
            pair_offsets=torch.tensor(offsets, dtype=torch.int64, device=device),
            sensor_pair_count=pair_count,
            reference_frequency_hz=float(reference_frequency_hz),
        )

    @property
    def path_count(self) -> int:
        return int(self.row_index.shape[0])

    def compose(self, leg: RadarLegBatch, *, include_delay_rate: bool = True) -> RadarPathBatch:
        """Publish one frame's direct rows. A gather, not a computation.

        Nothing is added, multiplied, or conjugated here: the leg's transport
        already IS the direct path's transfer at the reference frequency. There
        is therefore no kernel, and no arithmetic for one to own - only the
        reordering that puts the rows in canonical composed order.

        Dead rows need no masking for the same reason. The consumer publishes
        exact zeros for a row that stopped existing, and a gather preserves
        them; the two-way join masks only because it MULTIPLIES a dead row's
        payload into a product that would otherwise be a plausible number.
        """

        # freeze() orders every frozen row exactly once, so path_count IS the
        # frozen leg's row count. A batch of a different length is a different
        # topology: a longer one gathers in-range but wrong rows and publishes a
        # plausible frame, a shorter one trips a device-side assert several
        # launches later. Both are refused here, on host ints already in hand.
        if leg.leg_count != self.path_count:
            raise ValueError(
                f"the leg carries {leg.leg_count} rows but this composer was "
                f"frozen against {self.path_count}; the frame does not belong "
                "to this frozen topology"
            )
        rows = self.row_index
        row_valid = None if leg.row_valid is None else leg.row_valid.index_select(0, rows)
        # A band reorders exactly like the reference column: `index_select` on
        # dim 0 keeps the frequency axis intact, so a direct wideband batch
        # needs no arithmetic here either.
        frequency_response = None if leg.frequency_response is None else leg.frequency_response.index_select(0, rows)
        delay_rate = leg.delay_rate.index_select(0, rows) if include_delay_rate and leg.delay_rate is not None else None
        return RadarPathBatch(
            sensor_pair_count=self.sensor_pair_count,
            path_count=self.path_count,
            sensor_pair_index=self.sensor_pair_index,
            pair_offsets=self.pair_offsets,
            total_delay_s=leg.delay_s.index_select(0, rows),
            delay_rate=delay_rate,
            complex_transfer_ref=leg.coefficient.index_select(0, rows),
            reference_frequency_hz=self.reference_frequency_hz,
            row_valid=row_valid,
            topology=self.topology,
            join_mode="direct",
            frequency_response=frequency_response,
            frequency_offsets_hz=(None if frequency_response is None else leg.frequency_offsets_hz),
        )


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


def _int_set(values: object, name: str) -> frozenset[int]:
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
    :data:`MULTI_INTERACTION`; it is the Phase-5 ``hybrid`` distinction
    (target echo / environment clutter / multi-interaction echo) expressed as a
    declaration over ONE topology rather than as a third join mode.

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


def _leg_facts(leg, name: str) -> tuple[list[int], list[frozenset[int]]]:
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
    def from_two_way(cls, composer, inbound, outbound, declaration: ComponentDeclaration) -> RadarComponentIndex:
        """Classify a two-way join's rows from its two frozen leg topologies.

        ``composer`` is a :class:`~witwin.radar.paths.TwoWayComposer`
        and ``inbound`` / ``outbound`` are the frozen leg handles it was frozen
        against; all three are duck typed so this module adds no import edge to
        the Channel adapter.
        """

        return cls._build(composer.topology, inbound, outbound, declaration)

    @classmethod
    def from_direct(cls, composer, leg, declaration: ComponentDeclaration) -> RadarComponentIndex:
        """Classify a direct composer's rows. There is no second leg."""

        return cls._build(composer.topology, leg, None, declaration)

    @classmethod
    def _build(
        cls, topology: RadarPathTopology, inbound, outbound, declaration: ComponentDeclaration
    ) -> RadarComponentIndex:
        if not isinstance(declaration, ComponentDeclaration):
            raise TypeError(f"declaration must be a ComponentDeclaration, got {type(declaration).__name__}")
        site = [int(value) for value in topology.site_id.tolist()]
        inbound_row = [int(value) for value in topology.inbound_row.tolist()]
        outbound_row = [int(value) for value in topology.outbound_row.tolist()]
        in_depth, in_materials = _leg_facts(inbound, "inbound")
        if outbound is None:
            out_depth, out_materials = [], []
        else:
            out_depth, out_materials = _leg_facts(outbound, "outbound")

        classes: list[int] = []
        for row, site_id in enumerate(site):
            depth = in_depth[inbound_row[row]]
            materials = in_materials[inbound_row[row]]
            if outbound is not None and outbound_row[row] >= 0:
                depth = max(depth, out_depth[outbound_row[row]])
                materials = materials | out_materials[outbound_row[row]]
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
    "DIRECT_LEAKAGE",
    "ENVIRONMENT_CLUTTER",
    "JOIN_MODES",
    "MULTI_INTERACTION",
    "TARGET",
    "ComponentDeclaration",
    "DirectComposer",
    "JoinMode",
    "RadarComponentIndex",
    "RadarPathBatch",
    "RadarPathTopology",
    "TwoWayComposer",
]
