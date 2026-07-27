"""Private path geometry and amplitude expressions for the DSP oracles.

These are a deliberate, verbatim copy of the two production helpers that
``dsp_oracles`` used to import from ``witwin.radar.solvers.common``. They are
copied rather than imported because the oracle is the INDEPENDENT reference the
Phase-6 real-compatibility criterion is checked against, and an oracle that
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

``polarization`` used to be a fourth facade call, ``radar.polarization``.
Phase 11 deleted that runtime - it was a second projection of a field Channel
has already projected onto each endpoint's declared polarization - so the two
world-space vector banks and the flip flag are ARGUMENTS here now. The
expression is unchanged, including the order of the normalisations and the sign
of the mirror; only where the vectors come from moved.

One field IS no longer read off the radar: ``radar.gain``. Phase 6 deleted it,
because it applied ``sqrt(P R)`` on top of a weight that already carries
``sqrt(P_tx)``. The oracle keeps the factor as an explicit ``gain`` argument
defaulting to ``1.0``, which is the value the deleted attribute had on every
radar this oracle is used with, so every number it produces is unchanged.
"""

from __future__ import annotations

import math

import torch


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
    tx_vectors = radar.local_from_world_vectors(sample.entry_points.unsqueeze(0) - tx_pos.unsqueeze(1))
    rx_vectors = radar.local_from_world_vectors(sample.points.unsqueeze(0) - rx_pos.unsqueeze(1))
    tx_gains = radar.evaluate_antenna_pattern_vectors(tx_vectors).unsqueeze(1)
    rx_gains = radar.evaluate_antenna_pattern_vectors(rx_vectors).unsqueeze(0)
    return tx_gains * rx_gains


def _normalize_vectors(vectors: torch.Tensor) -> torch.Tensor:
    return vectors / torch.clamp(torch.linalg.norm(vectors, dim=-1, keepdim=True), min=1e-12)


def compute_polarization_amplitudes(polarization, sample) -> torch.Tensor | None:
    """Return signed TX/RX polarization projection factors for each path.

    ``polarization`` is any object carrying ``tx_world``, ``rx_world`` and
    ``reflection_flip``, or ``None`` for a sensor that declares none.
    """
    if polarization is None:
        return None
    if sample.normals is None:
        raise ValueError("Radar polarization requires per-path surface normals in the interpolated sample.")

    normals = _normalize_vectors(sample.normals)
    tx_world = _normalize_vectors(polarization.tx_world.to(device=normals.device, dtype=normals.dtype))
    rx_world = _normalize_vectors(polarization.rx_world.to(device=normals.device, dtype=normals.dtype))

    reflected_tx = tx_world.unsqueeze(1)
    if polarization.reflection_flip:
        reflected_tx = reflected_tx - 2.0 * (reflected_tx * normals.unsqueeze(0)).sum(dim=-1, keepdim=True) * normals.unsqueeze(0)
    reflected_tx = _normalize_vectors(reflected_tx)
    return (reflected_tx.unsqueeze(1) * rx_world.view(1, rx_world.shape[0], 1, 3)).sum(dim=-1)


def compute_path_amplitudes(
    radar,
    sample,
    total_path_lengths: torch.Tensor,
    *,
    tx_pos: torch.Tensor | None = None,
    rx_pos: torch.Tensor | None = None,
    tx_index: int | None = None,
    gain: float = 1.0,
    polarization=None,
) -> torch.Tensor:
    """Convert power-domain material coefficients to amplitude-domain weights with FSPL.

    ``tx_index`` selects a single TX row of the polarization factors when
    ``tx_pos`` holds only that antenna (per-TDM-slot evaluation).
    """
    fspl_amp = radar._lambda / (4.0 * math.pi * torch.clamp(total_path_lengths, min=1e-6))
    scatter_power = torch.clamp(sample.intensities, min=0.0).view(1, 1, -1)
    if tx_pos is None:
        tx_pos = radar.tx_pos
    if rx_pos is None:
        rx_pos = radar.rx_pos
    pattern_gains = compute_antenna_pattern_gains(radar, sample, tx_pos, rx_pos)
    if pattern_gains is not None:
        scatter_power = scatter_power * torch.clamp(pattern_gains, min=0.0)
    amplitudes = gain * torch.sqrt(scatter_power) * fspl_amp
    polarization_factor = compute_polarization_amplitudes(polarization, sample)
    if polarization_factor is not None:
        if tx_index is not None:
            polarization_factor = polarization_factor[tx_index : tx_index + 1]
        amplitudes = amplitudes * polarization_factor
    return amplitudes

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
    "compute_path_amplitudes",
    "compute_polarization_amplitudes",
    "compute_total_path_length_rates",
    "compute_total_path_lengths",
]
