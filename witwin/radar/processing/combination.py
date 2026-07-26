"""Combine per-component results, coherently or in power.

COHERENT combination needs no function. Components are row subsets of ONE
topology evaluated by the same waveform launches, so their cubes are complex
amplitudes on the same axes and plain addition IS the coherent law:

    ``sum_j cube(component_j) == cube(every row)``

up to float re-association of the partial sums. It is not bitwise, because the
kernel writes a literal ``0.0`` into a masked row's accumulation slot and
``(a + 0 + c) + (0 + b + 0)`` is not ``(a + b + c)`` in float32. The acceptance
test pins it with a tolerance derived from the row count and the largest
per-row contribution, and records the measured residual.

INCOHERENT combination is a different physical claim and therefore a different
function. It says the components have no fixed phase relationship, so their
POWERS add and their amplitudes do not. That is a post-synthesis statement
about an ensemble, and it belongs here rather than inside a waveform kernel: an
"incoherent" flag on a fused synthesis op would put a second summation
semantic inside a kernel whose whole contract is that it sums complex
amplitudes over a pair segment.

DEFERRED, with the reason. The physically honest incoherent model is not a
power sum at all - it is a per-realization random phase drawn into the scatter
response, so that an ensemble of frames averages to the power sum while each
individual frame remains a legitimate coherent field with speckle. That needs a
native RNG and a seed contract consistent with the frontend's, which is a
numerical change to a native response with its own decision record. Phase 8
ships the power-domain law and says so, rather than shipping a random phase
with an undeclared seed.
"""

from __future__ import annotations

import torch


def combine_incoherent(cubes) -> torch.Tensor:
    """``sum_j |cube_j|^2``: the power sum of independently exported components.

    Returns a REAL tensor. That is the point of the operation and it is not a
    convenience: the result has no phase, cannot be fed back into a coherent
    stage, and a caller that wanted an amplitude has to say which phase it
    meant.

    The magnitude is formed as ``re^2 + im^2`` rather than as ``abs()**2``
    because ``abs`` is not differentiable at the origin, and an exactly zero
    entry is the normal case here: every masked row of every component export
    contributes one.
    """

    listed = list(cubes)
    if not listed:
        raise ValueError(
            "combine_incoherent needs at least one cube; an empty sum is not a "
            "zero-power scene, it is a caller that forgot to export anything"
        )
    total = None
    for index, cube in enumerate(listed):
        if not isinstance(cube, torch.Tensor):
            raise TypeError(
                f"cube {index} must be a torch.Tensor, got {type(cube).__name__}"
            )
        if cube.shape != listed[0].shape:
            raise ValueError(
                f"cube {index} has shape {tuple(cube.shape)} but cube 0 has "
                f"{tuple(listed[0].shape)}; a power sum is elementwise and the "
                "components must share their axes"
            )
        if cube.device != listed[0].device:
            raise ValueError(
                f"cube {index} is on {cube.device} but cube 0 is on "
                f"{listed[0].device}"
            )
        power = (
            cube.real * cube.real + cube.imag * cube.imag
            if cube.is_complex()
            else cube * cube
        )
        total = power if total is None else total + power
    return total


__all__ = ["combine_incoherent"]
