"""Which site each composed row visits, as a rank in ascending stable-ID order.

``RoundTripPatternStage.apply`` takes the per-row site points from its caller;
production gathers them from Channel's live geometry. A direct test gathers
them from its own site tensor instead, and this is the row-to-site index it
gathers with. The rank order matches ``TwoWayComposer.freeze``, which sorts
the declared site IDs, so a site tensor laid out in ascending ID order lines up.
"""

from __future__ import annotations

import torch


def site_rank(paths) -> torch.Tensor:
    ids = paths.topology.site_id
    return torch.searchsorted(torch.unique(ids, sorted=True), ids)


__all__ = ["site_rank"]
