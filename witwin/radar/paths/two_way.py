"""Compose two propagation legs into radar round trips.

The join is BY IDENTITY: an inbound row joins an outbound row when the inbound
row's sink is the scatter site and the outbound row's source is the same site.
Joining by array position instead would be silently wrong the moment a leg
publishes its rows in a different order, and the resulting error would look
like a physics bug rather than a bookkeeping one. A permutation test pins this.

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


@dataclass(frozen=True, slots=True, eq=False)
class TwoWayComposer:
    """A frozen inbound/outbound join for one set of scatter sites."""

    inbound_row: torch.Tensor
    outbound_row: torch.Tensor
    topology: RadarPathTopology
    sensor_pair_index: torch.Tensor
    pair_offsets: torch.Tensor
    sensor_pair_count: int
    reference_frequency_hz: float

    @classmethod
    def freeze(
        cls,
        inbound,
        outbound,
        site_ids: torch.Tensor,
        *,
        reference_frequency_hz: float,
    ) -> "TwoWayComposer":
        """Build the identity join from two frozen leg topologies.

        ``inbound`` and ``outbound`` are
        :class:`witwin.radar.propagation.channel_consumer.FrozenLegTopology`
        handles; they are duck-typed here so this module does not import the
        Channel adapter.
        """

        device = inbound.sink_id.device
        inbound_sink = inbound.sink_id.tolist()
        inbound_source = inbound.source_id.tolist()
        outbound_source = outbound.source_id.tolist()
        outbound_sink = outbound.sink_id.tolist()
        sites = [int(value) for value in site_ids.tolist()]

        rows: list[tuple[int, int, int, int, int]] = []
        for site in sites:
            arriving = [i for i, v in enumerate(inbound_sink) if int(v) == site]
            leaving = [o for o, v in enumerate(outbound_source) if int(v) == site]
            if not arriving or not leaving:
                raise ValueError(
                    f"site {site} has no {'inbound' if not arriving else 'outbound'} "
                    "leg row in the frozen topology"
                )
            for i in arriving:
                for o in leaving:
                    rows.append(
                        (
                            int(inbound_source[i]),
                            int(outbound_sink[o]),
                            site,
                            i,
                            o,
                        )
                    )

        # Sensor pairs are the distinct (radar source, radar sink) endpoint
        # identities. Sorting by pair keeps composed rows in a stable
        # pair-major order and makes pair_offsets a valid half-open partition.
        pairs = sorted({(row[0], row[1]) for row in rows})
        pair_of = {pair: index for index, pair in enumerate(pairs)}
        rows.sort(key=lambda row: (pair_of[(row[0], row[1])], row[3], row[4]))

        def column(index: int) -> torch.Tensor:
            return torch.tensor(
                [row[index] for row in rows], dtype=torch.int64, device=device
            )

        pair_index = torch.tensor(
            [pair_of[(row[0], row[1])] for row in rows],
            dtype=torch.int64,
            device=device,
        )
        offsets = [0]
        for index in range(len(pairs)):
            offsets.append(
                offsets[-1] + sum(1 for row in rows if pair_of[(row[0], row[1])] == index)
            )

        return cls(
            inbound_row=column(3),
            outbound_row=column(4),
            topology=RadarPathTopology(
                radar_source_id=column(0),
                site_id=column(2),
                radar_sink_id=column(1),
                inbound_row=column(3),
                outbound_row=column(4),
            ),
            sensor_pair_index=pair_index,
            pair_offsets=torch.tensor(offsets, dtype=torch.int64, device=device),
            sensor_pair_count=len(pairs),
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
        """

        if response.is_geometry_dependent:
            raise NotImplementedError(
                "a geometry-dependent scatter response varies per path and must "
                "be evaluated in a native kernel, not composed here"
            )
        rows = self.path_count
        idx_in = self.inbound_row
        idx_out = self.outbound_row

        total_delay = inbound.delay_s.index_select(0, idx_in) + (
            outbound.delay_s.index_select(0, idx_out)
        )
        transfer = (
            outbound.coefficient.index_select(0, idx_out)
            * response.evaluate(rows, total_delay.device)
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

        row_valid = None
        if inbound.row_valid is not None or outbound.row_valid is not None:
            ones = torch.ones(rows, dtype=torch.bool, device=total_delay.device)
            valid_in = (
                ones
                if inbound.row_valid is None
                else inbound.row_valid.index_select(0, idx_in)
            )
            valid_out = (
                ones
                if outbound.row_valid is None
                else outbound.row_valid.index_select(0, idx_out)
            )
            row_valid = valid_in & valid_out

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


__all__ = ["TwoWayComposer"]
