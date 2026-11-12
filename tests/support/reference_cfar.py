"""An independent CFAR: the oracle for every detector in ``witwin.radar.processing``.

Formed from the definitions, cell by cell in spirit and offset by offset in
code, sharing nothing with the production path but the training-ring geometry
it is told to use:

* the ring is enumerated as an explicit list of ``(doppler, range)`` offsets,
  split into a lagging half (negative range offset, or zero range offset and
  negative Doppler offset) and its mirror;
* every offset is applied as a shift of the whole map - ``torch.roll`` along
  Doppler, which is the wrap, and a zero-filled shift with a validity mask
  along range, which is the truncation;
* the threshold constants are solved by ``scipy.optimize.brentq`` against the
  false-alarm laws written from their definitions: the closed forms for CA and
  OS, and for GO and SO a ``scipy.integrate.quad`` of ``E[exp(-a stat)]`` over
  the density of the max or min of two gamma-distributed half means.

A test oracle, so loops over offsets and over distinct ring sizes are fine
here and nowhere else.
"""

from __future__ import annotations

import functools
import math

import torch
from scipy import integrate, optimize, stats


def ring_offsets(guard_cells, training_cells) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    gd, gr = guard_cells
    td, tr = training_cells
    od, orr = gd + td, gr + tr
    lag, lead = [], []
    for d in range(-od, od + 1):
        for r in range(-orr, orr + 1):
            if abs(d) <= gd and abs(r) <= gr:
                continue
            if r < 0 or (r == 0 and d < 0):
                lag.append((d, r))
            else:
                lead.append((d, r))
    return lag, lead


def _shift(power: torch.Tensor, d: int, r: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``power[..., i + d (mod D), j + r]`` and whether ``j + r`` is on the map."""

    ranges = int(power.shape[-1])
    rolled = torch.roll(power, shifts=-d, dims=-2)
    shifted = torch.zeros_like(rolled)
    valid = torch.zeros(ranges, dtype=torch.bool)
    if r >= 0:
        shifted[..., : ranges - r] = rolled[..., r:]
        valid[: ranges - r] = True
    else:
        shifted[..., -r:] = rolled[..., : ranges + r]
        valid[-r:] = True
    return shifted, valid


def _law(kind: str, alpha: float, n1: int, n2: int, rank: int) -> float:
    if kind == "ca":
        n = n1 + n2
        return (1.0 + alpha / n) ** (-n)
    if kind == "os":
        n = n1 + n2
        return math.prod((n - i) / (n - i + alpha) for i in range(rank))
    if n1 == 0 or n2 == 0:
        n = max(n1, n2)
        return (1.0 + alpha / n) ** (-n)
    a = stats.gamma(n1, scale=1.0 / n1)
    b = stats.gamma(n2, scale=1.0 / n2)
    if kind == "go":
        density = lambda t: a.pdf(t) * b.cdf(t) + a.cdf(t) * b.pdf(t)  # noqa: E731
    else:
        density = lambda t: a.pdf(t) * b.sf(t) + a.sf(t) * b.pdf(t)  # noqa: E731
    value, _ = integrate.quad(
        lambda t: math.exp(-alpha * t) * density(t), 0.0, math.inf, epsabs=0.0, epsrel=1e-12, limit=400
    )
    return value


@functools.cache
def solve_alpha(kind: str, pfa: float, n1: int, n2: int, rank: int = 0) -> float:
    high = 1.0
    while _law(kind, high, n1, n2, rank) > pfa:
        high *= 2.0
    return optimize.brentq(lambda a: _law(kind, a, n1, n2, rank) - pfa, 0.0, high, xtol=1e-14, rtol=1e-14)


def reference_cfar(
    kind: str,
    power: torch.Tensor,
    *,
    guard_cells=(2, 3),
    training_cells=(4, 6),
    pfa: float = 1e-3,
    rank_fraction: float = 0.75,
) -> torch.Tensor:
    """The ``[..., D, R]`` threshold of detector ``kind`` on a real POWER map, in float64."""

    power = power.to(torch.float64)
    lag, lead = ring_offsets(guard_cells, training_cells)
    ranges = int(power.shape[-1])
    sums, counts, shifted_all, valid_all = [], [], [], []
    for half in (lag, lead):
        total = torch.zeros_like(power)
        count = torch.zeros(ranges, dtype=torch.int64)
        for d, r in half:
            shifted, valid = _shift(power, d, r)
            total = total + shifted
            count = count + valid.to(torch.int64)
            shifted_all.append(torch.where(valid, shifted, torch.full_like(shifted, math.inf)))
            valid_all.append(valid)
        sums.append(total)
        counts.append(count)
    alpha = torch.zeros(ranges, dtype=torch.float64)
    level = torch.zeros_like(power)
    if kind == "os":
        ordered = torch.stack(shifted_all, dim=0).sort(dim=0).values
    for column in range(ranges):
        n1, n2 = int(counts[0][column]), int(counts[1][column])
        n = n1 + n2
        if kind == "ca":
            alpha[column] = solve_alpha("ca", pfa, n1, n2)
            level[..., column] = (sums[0][..., column] + sums[1][..., column]) / n
        elif kind == "os":
            index = min(int(rank_fraction * n), n - 1)
            alpha[column] = solve_alpha("os", pfa, n1, n2, index + 1)
            level[..., column] = ordered[index][..., column]
        else:
            alpha[column] = solve_alpha(kind, pfa, n1, n2)
            mean1 = sums[0][..., column] / n1 if n1 else sums[1][..., column] / n2
            mean2 = sums[1][..., column] / n2 if n2 else mean1
            level[..., column] = torch.maximum(mean1, mean2) if kind == "go" else torch.minimum(mean1, mean2)
    return alpha * level
