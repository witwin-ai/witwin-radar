"""The beam / velocity / range cube.

Nothing in this repository formed a beam cube. The three existing angle routes
are ESTIMATORS - two FFT peak finders and a MUSIC spectrum - and an estimator
answers "which direction" for a detection that already exists. A cube answers
"how much energy in this direction, at this velocity, at this range" for a grid
of directions, which is what a detector runs on and what a display shows.

:func:`beam_cube` applies weights and knows no array geometry and no phasor. It
computes exactly ``y[b] = sum_p conj(w[p, b]) x[p]``, which is the definition of
a beamformer output and is what conventional, MVDR and any future weight family
all mean. The weight owners live in
:mod:`witwin.radar.processing.beamforming`, together with the array geometry and
the phase-sign reconciliation, so a weight family can be swapped here without
this module learning what an element is.

The pair axes of the map are flattened in the order the cube is published in,
which is TX major - ``[TX, RX, ...]`` out of ``assemble_frame_cube`` - and that
is the order :class:`~witwin.radar.processing.beamforming.ArrayGeometry` builds
its element table in.
"""

from __future__ import annotations

import torch

from .contracts import BeamCube, RangeDopplerMap


def beam_cube(
    rd: RangeDopplerMap, steering: torch.Tensor, *, directions: torch.Tensor
) -> BeamCube:
    """``[*pair, D, R]`` and ``[P, *beam]`` -> ``BeamCube[*beam, D, R]``.

    A ``[TX, RX, D, R]`` map and a ``[P, D, R]`` one give the same cube, because
    ``[TX, RX]`` IS ``[P]`` reshaped in the published TX-major order.

    ``directions`` is required and keyword only, which is a deliberate deviation
    from the design's two-argument sketch. A beam index means nothing without the
    grid it was steered over, a weight matrix does not carry one, and the
    alternative - defaulting it to something - would publish a cube whose angles
    are silently wrong rather than a call that does not compile.
    """

    if not isinstance(rd, RangeDopplerMap):
        raise TypeError(
            "beam_cube consumes a RangeDopplerMap, so that the range and "
            f"Doppler axes it publishes are already decided; got {type(rd).__name__}"
        )
    if not isinstance(steering, torch.Tensor) or not steering.is_complex():
        raise TypeError("steering must be a complex torch.Tensor")
    data = rd.data
    if data.dim() < 3:
        raise ValueError(
            "a Range-Doppler map to be beamformed is [*pair, doppler, range]; "
            f"got shape {tuple(data.shape)}"
        )
    pairs = 1
    for size in data.shape[:-2]:
        pairs *= int(size)
    if pairs != int(steering.shape[0]):
        raise ValueError(
            f"the map spans {pairs} sensor pairs but the steering matrix is "
            f"built for {int(steering.shape[0])}; they must be the same front end"
        )
    doppler = int(data.shape[-2])
    ranges = int(data.shape[-1])
    flat = data.reshape(pairs, doppler, ranges)
    weights = steering.reshape(pairs, -1).to(flat.dtype)
    formed = torch.tensordot(weights.conj(), flat, dims=([0], [0]))
    beam_shape = tuple(steering.shape[1:])
    if tuple(directions.shape) != (*beam_shape, 3):
        raise ValueError(
            f"the steering matrix spans beams {beam_shape} but directions has "
            f"shape {tuple(directions.shape)}; the two are one statement"
        )
    return BeamCube(
        data=formed.reshape(*beam_shape, doppler, ranges),
        axes=rd.axes,
        directions=directions,
    )


__all__ = ["beam_cube"]
