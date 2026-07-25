"""Compose two propagation legs into radar round trips.

The join is BY IDENTITY: an inbound row joins an outbound row when the inbound
row's sink is the scatter site and the outbound row's source is the same site.
Joining by array position instead would be silently wrong the moment a leg
publishes its rows in a different order, and the resulting error would look
like a physics bug rather than a bookkeeping one. A permutation test pins this.

Membership is not the whole story. The composed row ORDER used to be a function
of the joined legs' row POSITIONS, so a permuted leg order preserved the
composed set and permuted the composed sequence. That is enough to break an
elementwise comparison, and it makes "shuffled legs, identical output" an
untestable claim. The canonical order is therefore built from frame-invariant
row IDENTITY: the sensor pair, the site, and each leg row's
``(component, depth, primitive sequence, material sequence)`` key.

Those sequences are ADR-037 frozen LABELS, not re-validated hits. When a
reevaluated stationary point slides onto a coplanar twin triangle, Channel
keeps the original label precisely so that a downstream consumer has a stable
identity to key on. That is what makes them usable here; it is not a claim
about which triangle the ray struck this frame.

The sensor-pair partition spans the FRONT END's full source x sink cross
product, not just the pairs that survived discovery. A pair whose only site
failed discovery must still own an (empty) segment: ``synthesize_fmcw_beat``
shapes its output ``[chirps, sensor_pair_count, samples]``, so deriving the
pair set from surviving rows would silently renumber and reshape the IQ cube.
Channel's own consumer spans ``source_count * sink_count`` for the same reason,
and this module mirrors its sink-major pair index exactly.

The join is built ONCE, at freeze time, from the frozen topologies. Host
observation is permitted there because ``prepare_fixed_topology`` has already
synchronized; a per-frame join would reintroduce exactly the host traffic the
fixed-topology capability exists to avoid. Per frame, :meth:`compose` performs
device gathers and arithmetic only.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..propagation.contracts import RadarLegBatch
from .contracts import RadarPathBatch, RadarPathTopology


LegKey = tuple[int, int, tuple[int, ...], tuple[int, ...]]


def _stable_ids(values, name: str) -> list[int]:
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


def _leg_identity(frozen, name: str) -> tuple[list[int], list[int], list[LegKey]]:
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
    primitive = [
        tuple(int(value) for value in row)
        for row in frozen.primitive_sequence.tolist()
    ]
    material = [
        tuple(int(value) for value in row)
        for row in frozen.material_sequence.tolist()
    ]
    rows = len(source)
    for label, column in (
        ("sink_id", sink),
        ("component_id", component),
        ("depth", depth),
        ("primitive_sequence", primitive),
        ("material_sequence", material),
    ):
        if len(column) != rows:
            raise ValueError(
                f"{name} leg {label} has {len(column)} rows, expected {rows}"
            )
    keys: list[LegKey] = [
        (component[row], depth[row], primitive[row], material[row])
        for row in range(rows)
    ]
    return source, sink, keys


def _group_rows(
    source: list[int], sink: list[int], keys: list[LegKey], name: str
) -> dict[tuple[int, int], list[int]]:
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


def _csr(owner_of_row: list[int], owner_count: int) -> tuple[list[int], list[int]]:
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

    @classmethod
    def freeze(
        cls,
        inbound,
        outbound,
        site_ids,
        *,
        radar_source_ids,
        radar_sink_ids,
        reference_frequency_hz: float,
    ) -> "TwoWayComposer":
        """Build the identity join from two frozen leg topologies.

        ``inbound`` and ``outbound`` are
        :class:`witwin.radar.propagation.channel_consumer.FrozenLegTopology`
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
        sources = _stable_ids(radar_source_ids, "radar_source_ids")
        sinks = _stable_ids(radar_sink_ids, "radar_sink_ids")
        sites = _stable_ids(site_ids, "site_ids")
        sites.sort()

        source_rank = {value: rank for rank, value in enumerate(sources)}
        sink_rank = {value: rank for rank, value in enumerate(sinks)}

        inbound_source, inbound_sink, inbound_keys = _leg_identity(inbound, "inbound")
        outbound_source, outbound_sink, outbound_keys = _leg_identity(
            outbound, "outbound"
        )
        arriving = _group_rows(inbound_source, inbound_sink, inbound_keys, "inbound")
        leaving = _group_rows(outbound_source, outbound_sink, outbound_keys, "outbound")

        # Sink-major, mirroring the Channel consumer's own pair index so one
        # convention crosses the boundary rather than two.
        def pair_rank(source: int, sink: int) -> int:
            return sink_rank[sink] * len(sources) + source_rank[source]

        # (pair_rank, site_rank, source, site, sink, inbound_row, outbound_row,
        #  inbound_key, outbound_key)
        # Nothing may fall outside the declared front end. A leg row whose
        # radar endpoint is not in the declared lists would simply never be
        # visited below, which is a silent drop rather than an empty segment.
        stray_sources = sorted(set(inbound_source) - set(sources))
        if stray_sources:
            raise ValueError(
                f"inbound leg rows carry radar source IDs {stray_sources} that "
                f"are not in radar_source_ids {sources}"
            )
        stray_sinks = sorted(set(outbound_sink) - set(sinks))
        if stray_sinks:
            raise ValueError(
                f"outbound leg rows carry radar sink IDs {stray_sinks} that are "
                f"not in radar_sink_ids {sinks}"
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
                raise ValueError(
                    f"site {site} has no inbound leg row in the frozen topology"
                )
            if site not in reachable_out:
                raise ValueError(
                    f"site {site} has no outbound leg row in the frozen topology"
                )

        rows: list[tuple[int, int, int, int, int, int, int, LegKey, LegKey]] = []
        for site_rank, site in enumerate(sites):
            for source in sources:
                inbound_rows = arriving.get((source, site), ())
                for sink in sinks:
                    outbound_rows = leaving.get((site, sink), ())
                    for i in inbound_rows:
                        for o in outbound_rows:
                            rows.append(
                                (
                                    pair_rank(source, sink),
                                    site_rank,
                                    source,
                                    site,
                                    sink,
                                    i,
                                    o,
                                )
                                + (inbound_keys[i], outbound_keys[o])
                            )

        # The canonical order. Every component is frame invariant, so two
        # freezes of permuted leg rows produce the SAME composed sequence, not
        # merely the same composed set.
        rows.sort(key=lambda row: (row[0], row[1], row[7], row[8]))

        def column(index: int) -> torch.Tensor:
            return torch.tensor(
                [row[index] for row in rows], dtype=torch.int64, device=device
            )

        pair_count = len(sources) * len(sinks)
        counts = [0] * pair_count
        for row in rows:
            counts[row[0]] += 1
        offsets = [0]
        for count in counts:
            offsets.append(offsets[-1] + count)

        # The offsets table is the only piece of synthesis metadata whose VALUES
        # the native kernel cannot validate: reading them on the host per frame
        # would be exactly the transfer the fixed-topology capability exists to
        # avoid, so the kernel clamps instead of failing. Clamping turns a
        # malformed table into a plausible wrong answer rather than an error.
        #
        # Validating here costs nothing -- the table is still a Python list, at
        # freeze time, on the host -- and it is what makes the production route
        # provably unable to hand the kernel a table it would have to clamp.
        if offsets[0] != 0 or offsets[-1] != len(rows):
            raise ValueError(
                f"pair offsets must partition all {len(rows)} composed rows, "
                f"got {offsets}"
            )
        if any(b < a for a, b in zip(offsets[:-1], offsets[1:], strict=True)):
            raise ValueError(f"pair offsets must be non-decreasing, got {offsets}")

        inbound_count = len(inbound_source)
        outbound_count = len(outbound_source)
        by_inbound = _csr([row[5] for row in rows], inbound_count)
        by_outbound = _csr([row[6] for row in rows], outbound_count)
        by_response = _csr([row[1] for row in rows], len(sites))

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
            sensor_pair_index=column(0),
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
        )

    @property
    def path_count(self) -> int:
        return int(self.inbound_row.shape[0])

    def compose(
        self,
        inbound: RadarLegBatch,
        outbound: RadarLegBatch,
        response,
        *,
        include_delay_rate: bool = True,
    ) -> RadarPathBatch:
        """Compose one frame's round-trip rows. Device work only.

        ``include_delay_rate`` exists because a forward-AD dual carries exactly
        one meaning at a time. When the dual direction is a site VELOCITY, the
        unpacked delay tangent is a Doppler rate and belongs in the result. When
        the dual direction is a position PERTURBATION, the same tangent is a
        directional derivative and reusing it as a rate would silently mix two
        meanings. The caller states which it has.

        A dead row's payload is exactly zero, not a partial composition. The
        row is a complete answer that this round trip does not exist at these
        endpoint positions; publishing ``tau_in + 0`` for it would be a
        plausible number that no consumer should ever read.
        """

        if response.is_geometry_dependent:
            raise NotImplementedError(
                "a geometry-dependent scatter response varies per path and must "
                "be evaluated in a native kernel, not composed here"
            )
        rows = self.path_count
        idx_in = self.inbound_row
        idx_out = self.outbound_row
        device = inbound.delay_s.device
        row_valid = self._row_validity(inbound, outbound, rows, device)

        total_delay = inbound.delay_s.index_select(0, idx_in) + (
            outbound.delay_s.index_select(0, idx_out)
        )
        site_response = response.evaluate(self.site_count, device).contiguous()
        transfer = (
            outbound.coefficient.index_select(0, idx_out)
            * site_response.index_select(0, self.response_slot)
            * inbound.coefficient.index_select(0, idx_in)
        )

        delay_rate = None
        if (
            include_delay_rate
            and inbound.delay_rate is not None
            and outbound.delay_rate is not None
        ):
            delay_rate = inbound.delay_rate.index_select(0, idx_in) + (
                outbound.delay_rate.index_select(0, idx_out)
            )

        if row_valid is not None:
            total_delay = torch.where(
                row_valid, total_delay, torch.zeros_like(total_delay)
            )
            transfer = torch.where(row_valid, transfer, torch.zeros_like(transfer))
            if delay_rate is not None:
                delay_rate = torch.where(
                    row_valid, delay_rate, torch.zeros_like(delay_rate)
                )

        return RadarPathBatch(
            sensor_pair_count=self.sensor_pair_count,
            path_count=rows,
            sensor_pair_index=self.sensor_pair_index,
            pair_offsets=self.pair_offsets,
            total_delay_s=total_delay,
            delay_rate=delay_rate,
            complex_transfer_ref=transfer,
            reference_frequency_hz=self.reference_frequency_hz,
            row_valid=row_valid,
            topology=self.topology,
        )

    def _row_validity(
        self,
        inbound: RadarLegBatch,
        outbound: RadarLegBatch,
        rows: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if inbound.row_valid is None and outbound.row_valid is None:
            return None
        ones = torch.ones(rows, dtype=torch.bool, device=device)
        valid_in = (
            ones
            if inbound.row_valid is None
            else inbound.row_valid.index_select(0, self.inbound_row)
        )
        valid_out = (
            ones
            if outbound.row_valid is None
            else outbound.row_valid.index_select(0, self.outbound_row)
        )
        return valid_in & valid_out


__all__ = ["TwoWayComposer"]
