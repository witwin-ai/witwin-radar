"""Central-difference helpers with an explicit per-parameter step policy.

The step is never guessed. A position in metres and a dimensionless amplitude
have step sizes that differ by orders of magnitude, and using one step for both
is how a finite-difference check ends up reporting a confident zero.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


def central_difference(
    evaluate: Callable[[torch.Tensor], torch.Tensor],
    value: torch.Tensor,
    index: tuple[int, ...] | int,
    step: float,
) -> float:
    """d(evaluate)/d(value[index]) by central difference, in float64."""

    plus = value.clone()
    minus = value.clone()
    plus[index] = plus[index] + step
    minus[index] = minus[index] - step
    return float((evaluate(plus) - evaluate(minus)) / (2.0 * step))


def directional_derivative(
    evaluate: Callable[..., torch.Tensor],
    values: tuple[torch.Tensor, ...],
    directions: tuple[torch.Tensor, ...],
    step: float,
) -> float:
    """Directional derivative along ``directions``, by central difference."""

    plus = tuple(
        value + step * direction
        for value, direction in zip(values, directions, strict=True)
    )
    minus = tuple(
        value - step * direction
        for value, direction in zip(values, directions, strict=True)
    )
    return float((evaluate(*plus) - evaluate(*minus)) / (2.0 * step))


def fourth_order_difference(samples: dict[int, float], step: float) -> float:
    """d/dx from values at ``-2h, -h, +h, +2h``, fourth order in the step.

    The second-order stencil is enough against a float64 oracle. It is not
    enough against the float32 PRODUCTION chain: there the truncation error at
    a step large enough to clear the float32 noise floor is still several
    percent, so a second-order difference cannot distinguish a wrong derivative
    from its own truncation. Fourth order collapses that gap without shrinking
    the step.

    ``step`` is the REALIZED step, not the requested one. A float32 parameter
    rounds the perturbation on the way in, and at metre-scale coordinates that
    rounding is a fraction of a percent of a 1e-4 m step - the same order as
    everything else being measured here.
    """

    return (
        -samples[2] + 8.0 * samples[1] - 8.0 * samples[-1] + samples[-2]
    ) / (12.0 * step)


def relative_error(measured: float, reference: float, *, floor: float) -> float:
    """Relative error against ``reference``, with an absolute floor.

    ``floor`` is the magnitude below which a component is treated as
    structurally zero rather than as a small number with a meaningful ratio.
    Callers state it explicitly; there is no default.
    """

    scale = max(abs(reference), floor)
    return abs(measured - reference) / scale


__all__ = [
    "central_difference",
    "directional_derivative",
    "fourth_order_difference",
    "relative_error",
]
