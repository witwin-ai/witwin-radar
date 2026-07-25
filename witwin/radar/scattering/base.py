"""The scatter-response contract.

A scatter response turns a set of composed round-trip rows into the complex
factor that sits between the inbound and outbound transports. Phase 4 ships one
implementation, a per-target broadcast scale; its aspect-dependent,
material-informed, and polarimetric successors evaluate per path and therefore
belong in a native kernel, not here.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class ScatterResponse(Protocol):
    """A complex response evaluated for a batch of composed rows.

    The returned factor is authored in the CHANNEL phasor convention,
    ``exp(-j k d)``, because it multiplies transports authored there. The
    conversion to the beat convention happens once, downstream, in the
    synthesis facade.
    """

    def evaluate(self, row_count: int, device: torch.device) -> torch.Tensor:
        """Return ``complex64[row_count]``."""
        ...

    @property
    def is_geometry_dependent(self) -> bool:
        """Whether the response varies per path rather than per target.

        A geometry-dependent response is per-path physics and must be
        evaluated in a native kernel. This flag exists so that a future
        implementation cannot quietly become Torch hot-path physics while
        still satisfying the protocol.
        """
        ...


__all__ = ["ScatterResponse"]
