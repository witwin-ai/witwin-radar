"""Private path geometry and antenna-pattern expressions for the DSP oracles.

These are a deliberate, verbatim copy of the two production helpers that
``dsp_oracles`` used to import from ``witwin.radar.solvers.common``. They are
copied rather than imported because the oracle is the INDEPENDENT reference the
native sensor geometry criterion is checked against, and an oracle that
imports the module the migration rewrites checks that module against itself.

Verbatim is the whole point. The expression order, the clamps, the broadcast
shapes, and the order of the multiplications are reproduced exactly, so the
oracle's numbers are bit-identical to what it produced before the split. A
"cleaner" rewrite here would silently move the reference.

The radar-facade calls (``_lambda``, ``local_from_world_vectors``,
``evaluate_antenna_pattern_vectors``) are NOT copied: they belong to
``witwin/radar/radar.py``, not to the module Phase-6 work item 8 migrated, and
duplicating a pattern interpolator would make the oracle test a different
antenna than the production path does.

"""

from __future__ import annotations

import torch


def _pattern_gain_from_vectors(radar, vectors: torch.Tensor) -> torch.Tensor:
    forward = -vectors[..., 2]
    x_angles_deg = torch.rad2deg(torch.atan2(vectors[..., 0], forward))
    y_angles_deg = torch.rad2deg(torch.atan2(vectors[..., 1], forward))
    return radar._evaluate_antenna_pattern_xy(x_angles_deg, y_angles_deg)


def compute_total_path_lengths(sample, tx_pos: torch.Tensor, rx_pos: torch.Tensor) -> torch.Tensor:
    """Return total path lengths with shape (TX, RX, N)."""
    dist_tx = torch.cdist(sample.entry_points, tx_pos).transpose(0, 1).unsqueeze(1)
    dist_rx = torch.cdist(sample.points, rx_pos).transpose(0, 1).unsqueeze(0)
    return dist_tx + sample.fixed_path_lengths.view(1, 1, -1) + dist_rx


def compute_antenna_pattern_gains(
    radar,
    sample,
    tx_pos: torch.Tensor,
    rx_pos: torch.Tensor,
) -> torch.Tensor | None:
    """Return per-path power gains from the configured TX/RX antenna pattern."""
    tx_vectors = radar._local_from_world_vectors(sample.entry_points.unsqueeze(0) - tx_pos.unsqueeze(1))
    rx_vectors = radar._local_from_world_vectors(sample.points.unsqueeze(0) - rx_pos.unsqueeze(1))
    tx_gains = _pattern_gain_from_vectors(radar, tx_vectors).unsqueeze(1)
    rx_gains = _pattern_gain_from_vectors(radar, rx_vectors).unsqueeze(0)
    return tx_gains * rx_gains


def compute_total_path_length_rates(sample, velocities, *, tx_pos, rx_pos):
    """Total path length rate with shape (TX, RX, N), verbatim from the solver.

    Copied here for the same reason as the four expressions above: Phase 6
    migrated the production statement into the native ``sensor_weight`` kernel,
    and a kernel checked against nothing is checked against nothing.

    The SIGNS are the content. The inbound leg's rate dots ``entry - tx`` with
    the site velocity and the outbound leg's dots ``point - rx``, which is the
    NEGATIVE of the propagation direction, because the outbound leg shortens as
    the site moves toward the receiver. Using one sign for both agrees exactly
    on a stationary scene and is wrong by up to a factor of two on a moving one.
    """

    entry_vectors = sample.entry_points.unsqueeze(0) - tx_pos.unsqueeze(1)
    point_vectors = sample.points.unsqueeze(0) - rx_pos.unsqueeze(1)
    entry_dist = torch.clamp(torch.linalg.norm(entry_vectors, dim=-1), min=1e-6)
    point_dist = torch.clamp(torch.linalg.norm(point_vectors, dim=-1), min=1e-6)
    entry_rates = (entry_vectors * velocities.unsqueeze(0)).sum(dim=-1) / entry_dist
    point_rates = (point_vectors * velocities.unsqueeze(0)).sum(dim=-1) / point_dist
    return entry_rates.unsqueeze(1) + point_rates.unsqueeze(0)


__all__ = [
    "compute_antenna_pattern_gains",
    "compute_total_path_length_rates",
    "compute_total_path_lengths",
]
