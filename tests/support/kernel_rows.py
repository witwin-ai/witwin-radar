"""Row fixtures and phase estimators the per-kernel synthesis tests share.

The FMCW, OFDM and pulsed kernel tests each hand a few fabricated rows to their
kernel and read a phase slope back. The row builders and the estimators are
the same in every file; only the geometry constants differ, so those are
parameters here and each test binds its own.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from witwin.radar.paths import RadarPathTopology
from witwin.radar.synthesis import SlowTimeMode, SynthesisPathBatch


def channel_coefficient(reference_frequency_hz: float, tau_s: float, amplitude: complex = 1.0 + 0.0j) -> complex:
    """A Channel-sourced coefficient: ``amplitude * exp(-j 2 pi f_ref tau)``.

    ``C_rt`` carries the phase at the FROZEN per-frame delay. The OFDM and
    pulsed kernels take it unconjugated, in Channel's own convention.
    """

    phase = -2.0 * math.pi * reference_frequency_hz * tau_s
    return amplitude * complex(math.cos(phase), math.sin(phase))


def beat_weight(reference_frequency_hz: float, tau_s: float, amplitude: complex = 1.0 + 0.0j) -> complex:
    """The FMCW beat weight: the conjugate of :func:`channel_coefficient`.

    It is constant across chirps, which is exactly why ``carrier_rate_hz`` has
    to exist.
    """

    return channel_coefficient(reference_frequency_hz, tau_s, amplitude).conjugate()


def cuda_rows(delays, weights, rates, offsets=None):
    """``(tau, rate, transfer, offsets)`` device rows; one segment unless told otherwise."""

    tau = torch.tensor(delays, dtype=torch.float32, device="cuda")
    rate = torch.tensor(rates, dtype=torch.float32, device="cuda")
    transfer = torch.tensor(weights, dtype=torch.complex64, device="cuda")
    if offsets is None:
        offsets = [0, len(delays)]
    table = torch.tensor(offsets, dtype=torch.int64, device="cuda")
    return tau, rate, transfer, table


def synthesis_batch(
    *,
    reference_frequency_hz: float,
    tau_s: float,
    tau_rate: float,
    weight: complex,
    row_valid=None,
    path_count: int = 1,
    pair_count: int = 1,
) -> SynthesisPathBatch:
    """``path_count`` identical rows, all in pair 0, with ``pair_count`` segments."""

    zeros = torch.zeros(path_count, dtype=torch.int64, device="cuda")
    offsets = torch.tensor([0] + [path_count] * pair_count, dtype=torch.int64, device="cuda")
    return SynthesisPathBatch(
        sensor_pair_count=pair_count,
        path_count=path_count,
        sensor_pair_index=torch.zeros(path_count, dtype=torch.int64, device="cuda"),
        pair_offsets=offsets,
        total_delay_s=torch.full((path_count,), tau_s, dtype=torch.float32, device="cuda"),
        delay_rate=torch.full((path_count,), tau_rate, dtype=torch.float32, device="cuda"),
        complex_transfer_ref=torch.full((path_count,), weight, dtype=torch.complex64, device="cuda"),
        reference_frequency_hz=reference_frequency_hz,
        topology=RadarPathTopology(zeros, zeros, zeros, zeros, zeros),
        row_valid=row_valid,
        weight_includes_reference_phase=True,
        weight_includes_spreading=True,
        weight_includes_tx_power=True,
        slow_time_mode=SlowTimeMode.FROZEN_WEIGHT_WITH_CARRIER_RATE,
    )


def unwrapped(values: torch.Tensor) -> torch.Tensor:
    return torch.from_numpy(np.unwrap(torch.angle(values).numpy()))


def lsq_slope(phase: torch.Tensor) -> float:
    """Least-squares slope of an unwrapped phase sequence, per index step."""

    index = torch.arange(phase.numel(), dtype=torch.float64)
    index = index - index.mean()
    return float((index * (phase - phase.mean())).sum() / (index * index).sum())


def imaginary_central_difference(evaluate, value, index, step):
    """``d(loss)/d(Im w)`` by a central difference along the imaginary axis.

    ``support.fd.central_difference`` divides by the step, and a purely
    imaginary step would make the quotient complex; the directional derivative
    along ``i`` is the real quotient over the REAL step length.
    """

    plus = value.clone()
    minus = value.clone()
    plus[index] = plus[index] + 1j * step
    minus[index] = minus[index] - 1j * step
    return float((evaluate(plus) - evaluate(minus)) / (2.0 * step))


__all__ = [
    "beat_weight",
    "channel_coefficient",
    "cuda_rows",
    "imaginary_central_difference",
    "lsq_slope",
    "synthesis_batch",
    "unwrapped",
]
