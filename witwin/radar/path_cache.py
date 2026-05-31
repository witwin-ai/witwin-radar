"""Cached per-path MIMO geometry for fixed-trace radar simulation."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MimoPathCache:
    """Precomputed path inputs for fast Dirichlet MIMO generation.

    Tensors use shape ``(num_tx, num_rx, num_paths)``. Distances are one-way
    ranges in meters; amplitudes are already amplitude-domain solver weights.
    Optional distance rates are one-way range rates in meters per second.
    These rates are a first-order kinematic approximation around the trace
    pose; non-range geometry terms such as antenna pattern and polarization are
    expected to remain fixed for the frame.
    """

    one_way_distances: torch.Tensor
    amplitudes: torch.Tensor
    one_way_distance_rates: torch.Tensor | None = None
