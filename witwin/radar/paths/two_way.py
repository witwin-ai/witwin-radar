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
fixed-topology capability exists to avoid.

Per frame, :meth:`compose` launches ONE native kernel. The arithmetic it
replaced was roughly 17-19 device-side aten ops measuring a flat 0.2-0.6 ms
from four composed rows to twenty-four thousand: launch bound, not bandwidth
bound. Everything left in Python around that launch is validation, row
selection, and result assembly, and it performs no host observation at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.autograd.forward_ad as forward_ad
from ..ad_contracts import first_order_only

from ..propagation.contracts import RadarLegBatch
from . import _identity
from ._identity import LegKey
from .contracts import RadarPathBatch, RadarPathTopology


_OPS = None


def _validate_pair_ordering(sensor_pair_index, *, num_tx, num_rx, sensor_pair_count):
    """Run the synthesis layout check at freeze time, where a host read is free.

    Until this call existed the check had no production caller at all (the
    plan's Phase-6 gap 5): the frame path DEPENDS on the sink-major pair rank
    that ``pair_tx_index`` and ``assemble_frame_cube`` assume, and nothing ever
    asserted that a composed batch actually carried it. A composer that put a
    second, silently different numbering on the same data would have produced a
    cube that looked entirely reasonable and steered every angle wrongly.

    The import is deferred because ``witwin.radar.synthesis.contracts`` imports
    ``witwin.radar.paths.contracts``; a module-level import here would close
    that loop. Freeze time runs once per topology, so the import lookup is not
    on any hot path.
    """

    from ..synthesis.assembly import validate_pair_ordering

    validate_pair_ordering(
        sensor_pair_index,
        num_tx=num_tx,
        num_rx=num_rx,
        sensor_pair_count=sensor_pair_count,
    )


def _ops():
    """The native operator table, resolved once per process.

    Held here as well as in the build module because this runs on every frame,
    forward and backward: a per-launch import plus function call is pure
    overhead on the hot path.
    """

    global _OPS
    if _OPS is None:
        from ..cuda import build

        _OPS = build.build_extension()
    return _OPS


def _primal_rate(
    delay_rate: torch.Tensor | None,
    rows: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
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
        saved = (
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
        )
        ctx.save_for_backward(*saved)
        ctx.save_for_forward(*saved)

    @staticmethod
    @first_order_only
    def backward(ctx, grad_tau_rt, grad_rate_rt, grad_c_rt_re, grad_c_rt_im):
        (
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
        ) = ctx.saved_tensors
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
        (
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
        ) = ctx.saved_tensors
        # tan_rate_in / tan_rate_out are ignored, and the refusal that makes
        # that honest lives in _primal_rate, at the facade. Autograd hands this
        # callback a zero-filled tangent for an input that carries none, so a
        # check HERE could not tell "no tangent" from "a genuine zero" and
        # would only be a comment with a raise attached. The facade refuses a
        # rate input that is a dual at all, which is checkable.

        def inbound(tangent):
            return torch.zeros_like(c_in_re) if tangent is None else tangent.contiguous()

        def outbound(tangent):
            return (
                torch.zeros_like(c_out_re) if tangent is None else tangent.contiguous()
            )

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
        sources = _identity.stable_ids(radar_source_ids, "radar_source_ids")
        sinks = _identity.stable_ids(radar_sink_ids, "radar_sink_ids")
        sites = _identity.stable_ids(site_ids, "site_ids")
        sites.sort()

        inbound_source, inbound_sink, inbound_keys = _identity.leg_identity(
            inbound, "inbound"
        )
        outbound_source, outbound_sink, outbound_keys = _identity.leg_identity(
            outbound, "outbound"
        )
        arriving = _identity.group_rows(
            inbound_source, inbound_sink, inbound_keys, "inbound"
        )
        leaving = _identity.group_rows(
            outbound_source, outbound_sink, outbound_keys, "outbound"
        )
        pair_rank = _identity.sink_major_rank(sources, sinks)

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
        sensor_pair_index = column(0)
        _validate_pair_ordering(
            sensor_pair_index,
            num_tx=len(sources),
            num_rx=len(sinks),
            sensor_pair_count=pair_count,
        )
        offsets = _identity.pair_offsets([row[0] for row in rows], pair_count)

        inbound_count = len(inbound_source)
        outbound_count = len(outbound_source)
        by_inbound = _identity.csr([row[5] for row in rows], inbound_count)
        by_outbound = _identity.csr([row[6] for row in rows], outbound_count)
        by_response = _identity.csr([row[1] for row in rows], len(sites))

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
            by_row_offsets=torch.arange(
                len(rows) + 1, dtype=torch.int64, device=device
            ),
            by_row_rows=torch.arange(len(rows), dtype=torch.int64, device=device),
            outbound_max_depth=max((row[8][1] for row in rows), default=0),
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
        flags = (
            torch.ones(rows, dtype=torch.int32, device=device)
            if row_valid is None
            else row_valid.to(torch.int32)
        )
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
            band,
            inbound,
            outbound,
            response_re,
            response_im,
            response_index,
            response_family,
            flags,
        )

        publish_rate = (
            include_delay_rate
            and inbound.delay_rate is not None
            and outbound.delay_rate is not None
        )
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
            frequency_offsets_hz=(
                None if band is None else inbound.frequency_offsets_hz
            ),
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
        if not torch.equal(
            inbound.frequency_offsets_hz, outbound.frequency_offsets_hz
        ):
            raise ValueError(
                "the two legs were evaluated over different frequency grids; a "
                "composed column multiplies one leg's response at f by the "
                "other's at the SAME f, so the grids must agree"
            )
        return counts[0]

    def _compose_band(
        self,
        band,
        inbound,
        outbound,
        response_re,
        response_im,
        response_index,
        response_family,
        flags,
    ):
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
                _primal_rate(
                    inbound.delay_rate, inbound.leg_count, flags.device, "inbound"
                ),
                _primal_rate(
                    outbound.delay_rate, outbound.leg_count, flags.device, "outbound"
                ),
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

    def _require_frame(
        self, inbound: RadarLegBatch, outbound: RadarLegBatch
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

        from ..scattering.base import NATIVE_ROW_RESPONSE_OWNERS

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
        return (
            rows_re,
            rows_im,
            self.row_slot,
            (self.by_row_offsets, self.by_row_rows, self.path_count),
        )

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
            raise TypeError(
                "a scatter response must evaluate to a torch.Tensor, got "
                f"{type(value).__name__}"
            )
        if value.numel() != self.site_count:
            raise ValueError(
                f"the scatter response evaluated to {value.numel()} values but "
                f"this join was frozen against {self.site_count} sites"
            )
        return value

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
