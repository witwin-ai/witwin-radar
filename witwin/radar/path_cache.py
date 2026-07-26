"""Cached per-path MIMO geometry for fixed-trace radar simulation."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MimoPathCache:
    """Precomputed path inputs for fast Dirichlet MIMO generation.

    Tensors use shape ``(num_tx, num_rx, num_paths)``.

    ``total_delay_s`` is the ROUND-TRIP delay in seconds and ``delay_rate`` is
    its time derivative, dimensionless. They were a one-way distance in metres
    and a one-way range rate; the whole Phase-6 propagation contract speaks
    round-trip delay, and a cache that stored half of one forced every consumer
    to double it again - which is exactly how a path becomes self-consistently
    2x wrong.

    ``amplitudes`` is COMPLEX. A Channel coefficient carries a phase and a
    legacy real amplitude is the special case with a zero imaginary part, whose
    SIGN is still physics: a reflection flip is the only phase a real amplitude
    can express and taking its magnitude is a silent 180-degree error.

    The rate is a first-order kinematic approximation around the trace pose;
    non-range geometry terms such as antenna pattern and polarization are
    expected to remain fixed for the frame.
    """

    total_delay_s: torch.Tensor
    amplitudes: torch.Tensor
    delay_rate: torch.Tensor | None = None
