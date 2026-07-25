"""The direct radar path: radar source straight to radar sink, no scatter site.

This is a separate frozen composer rather than a flag inside the two-way join.
A direct path has ONE leg, so there is no join to do: no site to key on, no
scatter response to insert, and no product to differentiate. Expressing it as a
two-way join with a fabricated second leg and a unit response would make it
indistinguishable from a real round trip through a target whose response
happens to be one, and would put a ``has_outbound`` branch in the middle of the
join kernel to pay for the disguise.

What the two share is the RESULT. Both publish :class:`RadarPathBatch` with the
same pair partition convention, so ``synthesize_fmcw_beat`` needs no branch and
the caller's choice of mode is made once, upstream, and recorded on the batch.

Scope, because the words collide: "direct mode" here means the direct TX-to-RX
path evaluated THROUGH THE CHANNEL CONSUMER, on the same frozen-topology
contract as every other leg. It is not a Radar-owned native direct-path
evaluator, which is separate future work and is not short-cut by this.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..propagation.contracts import RadarLegBatch
from . import _identity
from ._identity import LegKey
from .contracts import RadarPathBatch, RadarPathTopology


# A direct row has no scatter site and no second leg. These sentinels say so
# explicitly, so a consumer that reads the topology sees "there is none" rather
# than a plausible index into something.
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
    def freeze(
        cls,
        leg,
        *,
        radar_source_ids,
        radar_sink_ids,
        reference_frequency_hz: float,
    ) -> "DirectComposer":
        """Order one frozen leg's rows the way a composed batch is ordered.

        Same canonical key as the two-way join - sensor pair, then row identity
        - so a direct batch and a multipath batch of the same scene agree on
        what row order means.
        """

        device = leg.source_id.device
        sources = _identity.stable_ids(radar_source_ids, "radar_source_ids")
        sinks = _identity.stable_ids(radar_sink_ids, "radar_sink_ids")
        source, sink, keys = _identity.leg_identity(leg, "direct")
        _identity.group_rows(source, sink, keys, "direct")
        pair_rank = _identity.sink_major_rank(sources, sinks)

        stray_sources = sorted(set(source) - set(sources))
        if stray_sources:
            raise ValueError(
                f"leg rows carry radar source IDs {stray_sources} that are not "
                f"in radar_source_ids {sources}"
            )
        stray_sinks = sorted(set(sink) - set(sinks))
        if stray_sinks:
            raise ValueError(
                f"leg rows carry radar sink IDs {stray_sinks} that are not in "
                f"radar_sink_ids {sinks}"
            )

        rows: list[tuple[int, int, int, int, LegKey]] = [
            (pair_rank(source[row], sink[row]), source[row], sink[row], row, keys[row])
            for row in range(len(source))
        ]
        rows.sort(key=lambda row: (row[0], row[4]))

        def column(index: int) -> torch.Tensor:
            return torch.tensor(
                [row[index] for row in rows], dtype=torch.int64, device=device
            )

        def constant(value: int) -> torch.Tensor:
            return torch.full(
                (len(rows),), value, dtype=torch.int64, device=device
            )

        pair_count = len(sources) * len(sinks)
        offsets = _identity.pair_offsets([row[0] for row in rows], pair_count)
        return cls(
            row_index=column(3),
            topology=RadarPathTopology(
                radar_source_id=column(1),
                site_id=constant(NO_SITE),
                radar_sink_id=column(2),
                inbound_row=column(3),
                outbound_row=constant(NO_OUTBOUND_ROW),
            ),
            sensor_pair_index=column(0),
            pair_offsets=torch.tensor(offsets, dtype=torch.int64, device=device),
            sensor_pair_count=pair_count,
            reference_frequency_hz=float(reference_frequency_hz),
        )

    @property
    def path_count(self) -> int:
        return int(self.row_index.shape[0])

    def compose(
        self, leg: RadarLegBatch, *, include_delay_rate: bool = True
    ) -> RadarPathBatch:
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

        rows = self.row_index
        row_valid = (
            None if leg.row_valid is None else leg.row_valid.index_select(0, rows)
        )
        delay_rate = (
            leg.delay_rate.index_select(0, rows)
            if include_delay_rate and leg.delay_rate is not None
            else None
        )
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
        )


__all__ = ["NO_OUTBOUND_ROW", "NO_SITE", "DirectComposer"]
