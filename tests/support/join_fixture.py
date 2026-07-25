"""Synthetic frozen legs for the two-way join, built without a Channel scene.

The real Phase-5 fixture has one sensor pair and one site. That is enough to
check the physics against closed forms and nowhere near enough to check the
JOIN: a permuted leg order, an empty pair segment, a multi-site gradient
reduction, and a per-site response all need shapes the fixture cannot produce.
Channel owns the frozen leg row order, so these legs are fabricated - which is
also the only way to permute them on purpose.

Everything here is deterministic. A join test that fails should fail the same
way twice.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch


def frozen_leg(rows, *, device: str = "cuda"):
    """A duck-typed frozen leg topology from ``(source, sink, component)`` rows."""

    def column(index: int, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor(
            [row[index] for row in rows], dtype=dtype, device=device
        )

    sequence = torch.tensor(
        [[row[2]] for row in rows], dtype=torch.int32, device=device
    )
    return SimpleNamespace(
        source_id=column(0, torch.int64),
        sink_id=column(1, torch.int64),
        component_id=column(2, torch.int32),
        depth=column(2, torch.int32),
        primitive_sequence=sequence,
        material_sequence=sequence,
    )


def leg_rows(sources, sinks, components) -> list[tuple[int, int, int]]:
    return [
        (source, sink, component)
        for source in sources
        for sink in sinks
        for component in components
    ]


def leg_batch(
    delay: torch.Tensor,
    coefficient: torch.Tensor,
    *,
    rate: torch.Tensor | None = None,
    row_valid: torch.Tensor | None = None,
):
    """A ``RadarLegBatch`` around already-built payload tensors.

    The identity columns are zeros: the JOIN reads identity from the FROZEN
    topology, at freeze time, not from the per-frame batch. Filling them with
    anything meaningful here would suggest otherwise.
    """

    from witwin.radar.propagation import RadarLegBatch

    rows = int(delay.shape[0])
    device = delay.device

    def zeros(dtype, shape=None):
        return torch.zeros(shape or (rows,), dtype=dtype, device=device)

    return RadarLegBatch(
        leg_count=rows,
        pair_count=1,
        pair_index=zeros(torch.int64),
        pair_offsets=torch.tensor([0, rows], dtype=torch.int64, device=device),
        source_index=zeros(torch.int32),
        sink_index=zeros(torch.int32),
        depth=zeros(torch.int32),
        component_id=zeros(torch.int32),
        source_id=zeros(torch.int64),
        sink_id=zeros(torch.int64),
        primitive_sequence=zeros(torch.int32, (rows, 1)),
        material_sequence=zeros(torch.int32, (rows, 1)),
        interaction_type=zeros(torch.int32, (rows, 1)),
        delay_s=delay,
        coefficient=coefficient,
        delay_rate=rate,
        row_valid=row_valid,
        diagnostics=None,
    )


def payload(rows: int, *, seed: int, device: str = "cuda", scale: float = 1.0):
    """Deterministic pseudo-random delays, rates and complex coefficients.

    Delays are nanosecond scale and coefficients are order one, matching what
    the real chain produces closely enough that a tolerance chosen here means
    the same thing there.
    """

    generator = torch.Generator().manual_seed(seed)

    def sample(count: int) -> torch.Tensor:
        return torch.rand(count, generator=generator, dtype=torch.float64)

    delay = (1.0e-8 + 2.0e-8 * sample(rows)).to(device=device)
    rate = (1.0e-9 * (sample(rows) - 0.5)).to(device=device)
    real = scale * (sample(rows) - 0.5)
    imag = scale * (sample(rows) - 0.5)
    coefficient = torch.complex(real, imag).to(device=device)
    return delay, rate, coefficient


__all__ = ["frozen_leg", "leg_batch", "leg_rows", "payload"]
