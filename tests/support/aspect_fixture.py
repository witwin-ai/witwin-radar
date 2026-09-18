"""One ``AspectScatterResponse`` on the multi-endpoint fixture, shared.

The direction-AD test and the departure-bearing test both drive this response
so that a change to its lobe is one edit rather than one per file.
"""

from __future__ import annotations

import torch

from witwin.radar.scattering import AspectScatterResponse

#: 15 degrees off the exact back-scatter direction of each site's live arrival.
#: Exactly on it would put the lobe at its peak, where the derivative with
#: respect to a UNIT direction is zero by construction and a finite difference
#: would be comparing two zeros.
AXIS_RAW = ((0.9659258, 0.2588190, 0.0), (-0.4188792, -0.9080614, 0.0))

ASPECT_EXPONENT = 2.0
COHERENT_INTERVAL_S = 1.0e-3


def aspect_axis(device: str = "cuda") -> torch.Tensor:
    raw = torch.tensor(AXIS_RAW, dtype=torch.float64)
    unit = raw / torch.linalg.vector_norm(raw, dim=1, keepdim=True)
    return unit.to(dtype=torch.float32, device=device)


def aspect_response(device: str = "cuda") -> AspectScatterResponse:
    return AspectScatterResponse(
        axis=aspect_axis(device),
        amplitude=torch.tensor([1.0e5, 0.8e5], dtype=torch.float32, device=device),
        phase_rad=torch.tensor([0.7, -0.3], dtype=torch.float32, device=device),
        exponent=ASPECT_EXPONENT,
        coherent_interval_s=COHERENT_INTERVAL_S,
    )


__all__ = ["ASPECT_EXPONENT", "AXIS_RAW", "COHERENT_INTERVAL_S", "aspect_axis", "aspect_response"]
