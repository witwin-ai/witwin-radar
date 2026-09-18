"""The exact summed-area cell-averaging CFAR: the oracle for ``ca_cfar``.

``witwin.radar.processing.ca_cfar`` forms the ring average from two pooled
means. This is the same estimator from one integral image per batch element,
kept as a test oracle so the production detector is measured against an
independent formation of the same rectangular ring, up to float
re-association.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from witwin.radar.processing import Detections


def _rect_sum(
    integral: torch.Tensor, r0: torch.Tensor, c0: torch.Tensor, r1: torch.Tensor, c1: torch.Tensor
) -> torch.Tensor:
    return integral[..., r1 + 1, c1 + 1] - integral[..., r0, c1 + 1] - integral[..., r1 + 1, c0] + integral[..., r0, c0]


def reference_ca_cfar(
    rd_map: torch.Tensor,
    *,
    guard_cells: tuple[int, int] = (2, 3),
    training_cells: tuple[int, int] = (4, 6),
    pfa: float = 1e-3,
) -> Detections:
    """Cell-averaging CFAR over ``[..., D, R]``, by summed-area table.

    An exact rectangular ring average from one integral image. Edges are
    replicate padded, so a cell at the border sees a ring of the same size
    rather than a smaller and therefore noisier one. Same threshold law and
    same defaults as the production detector.
    """

    real_dtype = torch.float64 if rd_map.dtype in {torch.float64, torch.complex128} else torch.float32
    values = (torch.abs(rd_map) if torch.is_complex(rd_map) else rd_map).to(real_dtype)
    if values.dim() < 2:
        raise ValueError(f"the map must be [..., doppler, range]; got shape {tuple(values.shape)}")
    leading = tuple(values.shape[:-2])
    flat = values.reshape(-1, *values.shape[-2:])
    doppler, ranges = int(flat.shape[-2]), int(flat.shape[-1])
    gd, gr = int(guard_cells[0]), int(guard_cells[1])
    td, tr = int(training_cells[0]), int(training_cells[1])
    outer_d, outer_r = gd + td, gr + tr
    n_train = (2 * outer_d + 1) * (2 * outer_r + 1) - (2 * gd + 1) * (2 * gr + 1)
    if n_train < 1:
        raise ValueError(
            f"guard_cells={guard_cells} and training_cells={training_cells} leave "
            "no training cells to estimate the noise from"
        )
    alpha = n_train * (float(pfa) ** (-1.0 / n_train) - 1.0)

    padded = F.pad(flat.unsqueeze(1), (outer_r, outer_r, outer_d, outer_d), mode="replicate")
    integral = F.pad(padded, (1, 0, 1, 0), mode="constant", value=0).cumsum(dim=-2).cumsum(dim=-1)
    device = flat.device
    row = torch.arange(doppler, device=device, dtype=torch.int64).reshape(-1, 1)
    col = torch.arange(ranges, device=device, dtype=torch.int64).reshape(1, -1)
    pi = row + outer_d
    pj = col + outer_r
    outer_sum = _rect_sum(integral, pi - outer_d, pj - outer_r, pi + outer_d, pj + outer_r)
    guard_sum = _rect_sum(integral, pi - gd, pj - gr, pi + gd, pj + gr)
    noise = (outer_sum - guard_sum) / n_train
    threshold = (alpha * noise).squeeze(1).reshape(*leading, doppler, ranges)
    return Detections(mask=values > threshold, threshold=threshold)
