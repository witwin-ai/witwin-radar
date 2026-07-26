"""The beam / velocity / range cube, and the conventional steering it is tested with.

Nothing in this repository formed a beam cube. The three existing angle routes
are ESTIMATORS - two FFT peak finders and a MUSIC spectrum - and an estimator
answers "which direction" for a detection that already exists. A cube answers
"how much energy in this direction, at this velocity, at this range" for a grid
of directions, which is what a detector runs on and what a display shows.

Two entries, and the split is deliberate:

* :func:`beam_cube` applies weights and knows no array geometry and no phasor.
  It computes exactly ``y[b] = sum_p conj(w[p, b]) x[p]``, which is the
  definition of a beamformer output and is what MVDR, conventional and any
  future weight family all mean. Stage S4's weight owners plug into it
  unchanged.
* :func:`conventional_steering` builds the array manifold, and it is where the
  phasor convention lives.

**The conjugation trap appears a second time here, and it is closed the same
way.** An FMCW beat cube is the conjugate of Channel's ``exp(-j k d)`` product,
so its SPATIAL phase across the virtual array is conjugated too, exactly as its
slow-time phase is. Steering it with a Channel-convention manifold would point
every beam at the mirror-image direction. The manifold is therefore built in the
CUBE's own convention, driven by the same ``axes.doppler_sign`` that
:func:`~witwin.radar.processing.doppler.range_doppler` reads, so there is one
derived quantity behind both reconciliations rather than two independent sign
decisions that can drift apart.

The virtual element positions follow ``PAIR_RANK_LAYOUT``: the composed pair
rank is SINK MAJOR, ``pair = rx_rank * num_tx + tx_rank``, so the transmitter
index of pair ``p`` is ``p % num_tx`` and the receiver index is ``p // num_tx``.
Getting that backwards transposes the array and silently mis-steers every angle
whenever ``num_tx == num_rx``.
"""

from __future__ import annotations

import math

import torch

from .contracts import BeamCube, RangeDopplerMap


def virtual_element_offsets_m(axes) -> torch.Tensor:
    """``[P, 3]`` float64: the two-way phase centre of every sensor pair.

    A TDM-MIMO virtual element sits at the SUM of its transmitter and receiver
    offsets, because the round-trip phase is the sum of the two one-way phases.
    Built in float64 on the axes record's device from the half-wavelength
    offsets the array spec declares, times the element spacing that turns them
    into metres.
    """

    device = axes.device
    transmitters = torch.tensor(
        axes.tx_loc_half_wavelength, dtype=torch.float64, device=device
    )
    receivers = torch.tensor(
        axes.rx_loc_half_wavelength, dtype=torch.float64, device=device
    )
    rank = torch.arange(axes.sensor_pair_count, device=device)
    tx_index = torch.remainder(rank, axes.num_tx)
    rx_index = torch.div(rank, axes.num_tx, rounding_mode="floor")
    offsets = transmitters.index_select(0, tx_index) + receivers.index_select(
        0, rx_index
    )
    return (offsets * axes.element_spacing_m).contiguous()


def conventional_steering(
    axes,
    directions: torch.Tensor,
    *,
    normalize: bool = True,
    dtype: torch.dtype = torch.complex64,
) -> torch.Tensor:
    """``[P, *beam]`` phase-shift weights for a grid of LOCAL-frame directions.

    ``directions`` is ``[*beam, 3]``; the vectors are expected to be unit length
    and point FROM the array TOWARD the look direction. They are not normalised
    here, because silently normalising a caller's grid would hide the one bug
    this argument can have.

    With ``normalize=True`` (the default) the weights satisfy ``w^H a = 1`` for a
    wavefront matched to the beam, so a beam cube formed with them is in the
    same amplitude convention as the range and Doppler stages: a single path row
    peaks at its own coefficient magnitude. That is also the constraint an MVDR
    weight satisfies, so the two weight families are interchangeable at
    :func:`beam_cube` without a scale factor appearing between them.
    """

    if not isinstance(directions, torch.Tensor):
        raise TypeError(
            f"directions must be a torch.Tensor, got {type(directions).__name__}"
        )
    if directions.dim() < 1 or int(directions.shape[-1]) != 3:
        raise ValueError(
            f"directions must be [*beam, 3]; got shape {tuple(directions.shape)}"
        )
    offsets = virtual_element_offsets_m(axes)
    beam_shape = tuple(directions.shape[:-1])
    flat = directions.reshape(-1, 3).to(torch.float64).to(offsets.device)
    # [P, B]: the projection of every virtual element onto every look direction.
    projection = offsets @ flat.transpose(0, 1)
    wavenumber = 2.0 * math.pi / axes.wavelength_m
    # The cube's own convention: Channel's exp(-j k d) makes the array response
    # exp(+j k <r, u>); a conjugated beat cube reverses it, and doppler_sign is
    # the single derived quantity that says which.
    phase = projection * (-axes.doppler_sign * wavenumber)
    manifold = torch.polar(torch.ones_like(phase), phase).to(dtype)
    if normalize:
        manifold = manifold / axes.sensor_pair_count
    return manifold.reshape(axes.sensor_pair_count, *beam_shape).contiguous()


def beam_cube(
    rd: RangeDopplerMap, steering: torch.Tensor, *, directions: torch.Tensor
) -> BeamCube:
    """``[*pair, D, R]`` and ``[P, *beam]`` -> ``BeamCube[*beam, D, R]``.

    The pair axes of the map are flattened in their published order, which is
    the sink-major composed rank, and contracted against the weight's leading
    axis. A ``[TX, RX, D, R]`` map and a ``[P, D, R]`` one give the same cube,
    because ``[TX, RX]`` IS ``[P]`` reshaped.

    ``directions`` is required and keyword only, which is a deliberate
    deviation from the design's two-argument sketch. A beam index means nothing
    without the grid it was steered over, a weight matrix does not carry one,
    and the alternative - defaulting it to something - would publish a cube
    whose angles are silently wrong rather than a call that does not compile.
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


__all__ = ["beam_cube", "conventional_steering", "virtual_element_offsets_m"]
