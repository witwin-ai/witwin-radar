"""Constant-false-alarm-rate detection, batched.

The three ``sigproc`` detectors took a rank-2 ``(Nd, Nr)`` map, so detecting on a
``[B, D, R]`` beam cube meant a Python loop over beams - one launch per beam for
an operation that is a single convolution. Every entry here takes
``[..., D, R]`` with an arbitrary leading batch and returns
:class:`Detections`, so a beam cube, a per-pair map and a single slice are the
same call.

A range profile with no Doppler axis had no detector at all;
:func:`ca_cfar_1d` is the range-only form, over ``[..., R]``.

The threshold law is the standard cell-averaging one and is published rather
than buried: with ``N`` training cells and a design false-alarm probability
``P_fa``, the scale is

    ``alpha = N * (P_fa ** (-1 / N) - 1)``

which for an exponentially distributed power estimate gives
``P_fa = (1 + alpha / N) ** (-N)`` exactly. That identity is what the
false-alarm-rate test measures against, so the constant cannot be tuned to make
a test pass without the measured rate moving with it.

Note the units the identity holds in. It is a statement about POWER: the noise
estimate must be an average of ``|x| ** 2``. Fed a magnitude or a decibel map -
which is what every existing caller does - the detector still adapts to the
local level, but the nominal ``P_fa`` is a design parameter rather than a
prediction. That is stated here because it was not stated anywhere before.

The identity is equally a statement about a MEAN. :func:`os_cfar` scales an
ordered statistic with that same cell-averaging constant, so the rate it
achieves is NOT the ``pfa`` it is handed. Its own docstring gives the law its
rate does follow, the measured ratio, and why the constant is left alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True, slots=True, eq=False)
class Detections:
    """A boolean mask and the adaptive threshold that produced it.

    Both have the shape of the map they were computed on, leading batch
    included. The threshold travels with the mask because a detection without
    the level it beat is not reproducible, and because the point-cloud stage
    reports an energy relative to it.
    """

    mask: torch.Tensor
    threshold: torch.Tensor

    def __post_init__(self) -> None:
        if self.mask.dtype != torch.bool:
            raise TypeError(f"mask must be bool, got {self.mask.dtype}")
        if tuple(self.mask.shape) != tuple(self.threshold.shape):
            raise ValueError(
                f"mask is {tuple(self.mask.shape)} but threshold is "
                f"{tuple(self.threshold.shape)}; they describe the same map"
            )

    @property
    def count(self) -> torch.Tensor:
        """The number of detections, as a DEVICE scalar.

        Deliberately not an ``int``. Reading it to the host is a
        synchronization, and this package refuses to add one implicitly - a
        caller that needs the number asks for it and pays for it visibly.
        """

        return self.mask.sum()


def _alpha(n_train: int, pfa: float) -> float:
    return n_train * (float(pfa) ** (-1.0 / n_train) - 1.0)


def _real_values(data: torch.Tensor) -> torch.Tensor:
    real_dtype = (
        torch.float64
        if data.dtype in {torch.float64, torch.complex128}
        else torch.float32
    )
    values = torch.abs(data) if torch.is_complex(data) else data
    return values.to(real_dtype)


def _as_batch(values: torch.Tensor, rank: int) -> tuple[torch.Tensor, tuple[int, ...]]:
    if values.dim() < rank:
        raise ValueError(
            f"the map must be [..., {'doppler, range' if rank == 2 else 'range'}]; "
            f"got shape {tuple(values.shape)}"
        )
    leading = tuple(values.shape[: -rank])
    return values.reshape(-1, *values.shape[-rank:]), leading


def _replicate_pad_2d(data: torch.Tensor, pad_h: int, pad_w: int) -> torch.Tensor:
    return F.pad(data.unsqueeze(1), (pad_w, pad_w, pad_h, pad_h), mode="replicate")


def _rect_sum(
    integral: torch.Tensor,
    r0: torch.Tensor,
    c0: torch.Tensor,
    r1: torch.Tensor,
    c1: torch.Tensor,
) -> torch.Tensor:
    return (
        integral[..., r1 + 1, c1 + 1]
        - integral[..., r0, c1 + 1]
        - integral[..., r1 + 1, c0]
        + integral[..., r0, c0]
    )


def ca_cfar(
    rd_map: torch.Tensor,
    *,
    guard_cells: tuple[int, int] = (2, 3),
    training_cells: tuple[int, int] = (4, 6),
    pfa: float = 1e-3,
) -> Detections:
    """Cell-averaging CFAR over ``[..., D, R]``, by summed-area table.

    The reference implementation: an exact rectangular ring average, computed
    from one integral image per batch element. Edges are handled by replicate
    padding, so a cell at the border sees a ring of the same size rather than a
    smaller and therefore noisier one.
    """

    values = _real_values(rd_map)
    flat, leading = _as_batch(values, 2)
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
    alpha = _alpha(n_train, pfa)

    padded = _replicate_pad_2d(flat, outer_d, outer_r)
    integral = (
        F.pad(padded, (1, 0, 1, 0), mode="constant", value=0)
        .cumsum(dim=-2)
        .cumsum(dim=-1)
    )
    device = flat.device
    row = torch.arange(doppler, device=device, dtype=torch.int64).reshape(-1, 1)
    col = torch.arange(ranges, device=device, dtype=torch.int64).reshape(1, -1)
    pi = row + outer_d
    pj = col + outer_r
    outer_sum = _rect_sum(
        integral, pi - outer_d, pj - outer_r, pi + outer_d, pj + outer_r
    )
    guard_sum = _rect_sum(integral, pi - gd, pj - gr, pi + gd, pj + gr)
    noise = (outer_sum - guard_sum) / n_train
    threshold = (alpha * noise).squeeze(1).reshape(*leading, doppler, ranges)
    return Detections(mask=values > threshold, threshold=threshold)


def ca_cfar_fast(
    rd_map: torch.Tensor,
    *,
    guard_cells: tuple[int, int] = (2, 3),
    training_cells: tuple[int, int] = (4, 6),
    pfa: float = 1e-3,
) -> Detections:
    """The same estimator, from two pooled averages instead of one table.

    Mathematically identical to :func:`ca_cfar` up to float re-association: the
    ring sum is the outer mean times the outer count minus the guard mean times
    the guard count. It exists because ``avg_pool2d`` is a single fused kernel
    where the summed-area route is three passes plus two gathers.
    """

    values = _real_values(rd_map)
    flat, leading = _as_batch(values, 2)
    doppler, ranges = int(flat.shape[-2]), int(flat.shape[-1])
    gd, gr = int(guard_cells[0]), int(guard_cells[1])
    td, tr = int(training_cells[0]), int(training_cells[1])
    outer_d, outer_r = gd + td, gr + tr
    n_outer = (2 * outer_d + 1) * (2 * outer_r + 1)
    n_guard = (2 * gd + 1) * (2 * gr + 1)
    n_train = n_outer - n_guard
    if n_train < 1:
        raise ValueError(
            f"guard_cells={guard_cells} and training_cells={training_cells} leave "
            "no training cells to estimate the noise from"
        )
    alpha = _alpha(n_train, pfa)

    def pooled(size: tuple[int, int]) -> torch.Tensor:
        padded = _replicate_pad_2d(flat, size[0] // 2, size[1] // 2)
        return F.avg_pool2d(padded, kernel_size=size, stride=1).squeeze(1)

    outer_mean = pooled((2 * outer_d + 1, 2 * outer_r + 1))
    guard_mean = pooled((2 * gd + 1, 2 * gr + 1))
    noise = (outer_mean * n_outer - guard_mean * n_guard) / n_train
    threshold = (alpha * noise).reshape(*leading, doppler, ranges)
    return Detections(mask=values > threshold, threshold=threshold)


def os_cfar(
    rd_map: torch.Tensor,
    *,
    guard_cells: tuple[int, int] = (2, 3),
    training_cells: tuple[int, int] = (4, 6),
    rank_fraction: float = 0.75,
    pfa: float = 1e-3,
) -> Detections:
    """Ordered-statistic CFAR over ``[..., D, R]``.

    The noise estimate is the ``rank_fraction``-th ordered training sample
    rather than their mean, which is what makes it robust when a second target
    sits inside the training ring and would otherwise raise the threshold above
    the first.

    This is the memory outlier of the three: it materialises every cell's
    training patch, ``[batch, D * R, n_outer]``, and sorts it. The cost is
    recorded rather than hidden.

    ``pfa`` IS NOT THE ACHIEVED RATE HERE. The scale is the cell-averaging
    constant ``alpha = N (pfa ** (-1 / N) - 1)``, which inverts the false-alarm
    law of a MEAN of ``N`` exponential samples; this detector thresholds against
    the ``k``-th smallest of them instead, whose exact rate on exponential power
    is Rohling's

        ``P_fa = prod_{i=0}^{k-1} (N - i) / (N - i + alpha)``

    with ``k = min(int(rank_fraction * N), N - 1) + 1``. At the defaults that
    lands well BELOW the declared number - measured 1.88e-3 for ``pfa=1e-2`` and
    1.03e-4 for ``pfa=1e-3`` at ``N=40``, ``k=31``, ratios 0.19 and 0.10 - so
    the error is in the conservative direction and costs sensitivity rather than
    producing surprise false alarms. Read ``pfa`` on this function as a design
    constant in the cell-averaging parameterisation; when the achieved rate is
    what matters, invert the law above.

    The constant is deliberately left alone rather than re-solved for the
    ordered statistic: it is what the pre-cutover ``os_cfar_2d`` computed, the
    migration adapter is pinned bitwise to that behaviour, and moving it is a
    numerical change that owes its own decision and its own golden update.
    """

    values = _real_values(rd_map)
    flat, leading = _as_batch(values, 2)
    batch = int(flat.shape[0])
    doppler, ranges = int(flat.shape[-2]), int(flat.shape[-1])
    gd, gr = int(guard_cells[0]), int(guard_cells[1])
    td, tr = int(training_cells[0]), int(training_cells[1])
    outer_d, outer_r = gd + td, gr + tr
    n_outer = (2 * outer_d + 1) * (2 * outer_r + 1)
    n_train = n_outer - (2 * gd + 1) * (2 * gr + 1)
    if n_train < 1:
        raise ValueError(
            f"guard_cells={guard_cells} and training_cells={training_cells} leave "
            "no training cells to estimate the noise from"
        )
    alpha = _alpha(n_train, pfa)

    # A POSITION mask, not a value threshold: the guard band is where it is, not
    # wherever the values happen to be large.
    keep = torch.ones(
        (2 * outer_d + 1, 2 * outer_r + 1), dtype=torch.bool, device=flat.device
    )
    keep[td : td + 2 * gd + 1, tr : tr + 2 * gr + 1] = False

    padded = _replicate_pad_2d(flat, outer_d, outer_r)
    patches = F.unfold(
        padded, kernel_size=(2 * outer_d + 1, 2 * outer_r + 1), stride=1
    ).transpose(1, 2)
    training, _ = torch.sort(patches[:, :, keep.reshape(-1)], dim=-1)
    index = min(int(rank_fraction * n_train), int(training.shape[-1]) - 1)
    threshold = (alpha * training[:, :, index]).reshape(batch, doppler, ranges)
    threshold = threshold.reshape(*leading, doppler, ranges)
    return Detections(mask=values > threshold, threshold=threshold)


def ca_cfar_1d(
    profile: torch.Tensor,
    *,
    guard_cells: int = 2,
    training_cells: int = 8,
    pfa: float = 1e-3,
) -> Detections:
    """Range-only cell-averaging CFAR over ``[..., R]``.

    A range profile with no Doppler axis had no detector anywhere in this
    repository. Same law, one axis, and the same replicate-padded ring so a
    detection at the first range bin is not systematically favoured.
    """

    values = _real_values(profile)
    flat, leading = _as_batch(values, 1)
    ranges = int(flat.shape[-1])
    guard = int(guard_cells)
    train = int(training_cells)
    outer = guard + train
    n_train = 2 * train
    if n_train < 1:
        raise ValueError(
            f"training_cells={training_cells} leaves no training cells to "
            "estimate the noise from"
        )
    alpha = _alpha(n_train, pfa)

    padded = F.pad(flat.unsqueeze(1), (outer, outer), mode="replicate")
    integral = F.pad(padded, (1, 0), mode="constant", value=0).cumsum(dim=-1)
    device = flat.device
    centre = torch.arange(ranges, device=device, dtype=torch.int64) + outer
    outer_sum = integral[..., centre + outer + 1] - integral[..., centre - outer]
    guard_sum = integral[..., centre + guard + 1] - integral[..., centre - guard]
    noise = (outer_sum - guard_sum) / n_train
    threshold = (alpha * noise).squeeze(1).reshape(*leading, ranges)
    return Detections(mask=values > threshold, threshold=threshold)


__all__ = ["Detections", "ca_cfar", "ca_cfar_1d", "ca_cfar_fast", "os_cfar"]
