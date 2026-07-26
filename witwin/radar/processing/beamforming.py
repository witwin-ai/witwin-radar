"""The array, and the two weight families every angular stage is built on.

Three things live here: the geometry record, conventional (phase-shift,
delay-and-sum) weights, and the minimum-variance weights. Everything angular in
this package - the beam cube, the FFT angle estimators, MUSIC, the point cloud -
reads :class:`ArrayGeometry` and nothing else, so there is exactly one statement
of where an element is and exactly one statement of which way its phase runs.

**No half-wavelength spacing is hard coded anywhere.** ``MUSICImager`` used a
literal ``spacing = 0.5`` that no configuration could change, so a quarter-wave
array reported the wrong angle with no symptom other than being wrong.
:attr:`ArrayGeometry.element_spacing_m` is what one unit of the declared
``tx_loc`` / ``rx_loc`` grid is worth in metres, and
:attr:`ArrayGeometry.spacing_wavelengths` is the same number divided by the
wavelength. Both are data.

**The pair rank is TX MAJOR here, and that is not the composed rank.** The
composed pair rank published by the path layer is SINK major,
``pair = rx_rank * num_tx + tx_rank`` (``PAIR_RANK_LAYOUT``). Every cube that
reaches this package has already been through
``synthesis/assembly.py::assemble_frame_cube``, which transposes it to
``[TX, RX, chirp, sample]`` precisely because a virtual-antenna index is TX
major, ``va = tx * num_rx + rx``. Flattening the array axes of that cube gives
the TX-major order, and so do ``tdm_compensate``'s transmitter slices and every
angle estimator in this package. The element positions are therefore built in
the TX-major order that the cube they will multiply is actually in. Getting this
backwards transposes the array and silently mis-steers every angle whenever
``num_tx == num_rx``.

**The conjugation trap, closed once.** An FMCW beat cube is the conjugate of
Channel's ``exp(-j k d)`` product in SPACE as well as in slow time, so its
spatial phase across the virtual array runs the other way. Steering it with a
Channel-convention manifold points every beam at the mirror-image direction.
:attr:`ArrayGeometry.phase_sign` carries that, derived from the same
``axes.doppler_sign`` the Doppler stage reads, so there is one derived quantity
behind both reconciliations rather than two sign decisions that can drift.

**Build it once per array.** The record holds a materialised ``[P, 3]`` tensor,
so a per-frame :func:`conventional_steering` costs no host-to-device copy of the
element table. Rebuilding the geometry inside every call was measured at
152 microseconds - the most expensive call in the whole processing chain - for a
quantity that does not change between frames.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True, eq=False)
class ArrayGeometry:
    """Where every virtual element is, in metres, and which way its phase runs.

    ``element_positions_m`` is ``[P, 3]`` float64 in the array's LOCAL frame,
    ordered TX major. A TDM-MIMO virtual element sits at the SUM of its
    transmitter and receiver offsets, because the round-trip phase is the sum of
    the two one-way phases.
    """

    element_positions_m: torch.Tensor
    num_tx: int
    num_rx: int
    element_spacing_m: float
    wavelength_m: float
    phase_sign: int
    tx_loc_half_wavelength: tuple[tuple[float, float, float], ...]
    rx_loc_half_wavelength: tuple[tuple[float, float, float], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.element_positions_m, torch.Tensor):
            raise TypeError(
                "element_positions_m must be a torch.Tensor, got "
                f"{type(self.element_positions_m).__name__}"
            )
        if self.element_positions_m.dtype != torch.float64:
            raise TypeError(
                "element_positions_m must be float64: it is a coordinate, and a "
                "float32 metre carries a phase error of its own at 77 GHz"
            )
        expected = (self.num_tx * self.num_rx, 3)
        if tuple(self.element_positions_m.shape) != expected:
            raise ValueError(
                f"element_positions_m must be {expected} for a "
                f"{self.num_tx} x {self.num_rx} array, got "
                f"{tuple(self.element_positions_m.shape)}"
            )
        if self.phase_sign not in (1, -1):
            raise ValueError(f"phase_sign must be +1 or -1, got {self.phase_sign!r}")
        if not self.wavelength_m > 0.0:
            raise ValueError(f"wavelength_m must be positive, got {self.wavelength_m}")
        if not self.element_spacing_m > 0.0:
            raise ValueError(
                f"element_spacing_m must be positive, got {self.element_spacing_m}"
            )

    @property
    def sensor_pair_count(self) -> int:
        return self.num_tx * self.num_rx

    @property
    def device(self) -> torch.device:
        return self.element_positions_m.device

    @property
    def spacing_wavelengths(self) -> float:
        """One unit of the declared element grid, in wavelengths.

        Exactly ``0.5`` for the conventional half-wavelength array, and read
        rather than assumed everywhere. This is the number ``MUSICImager`` had
        hard coded.
        """

        return self.element_spacing_m / self.wavelength_m

    @property
    def transmitter_index(self) -> torch.Tensor:
        """``[P]`` int64: which transmitter each virtual element belongs to.

        TX major, so it is ``pair // num_rx``. This is the index a TDM slot
        offset multiplies, and the one place the ordering is written as code.
        """

        rank = torch.arange(self.sensor_pair_count, device=self.device)
        return torch.div(rank, self.num_rx, rounding_mode="floor")

    @property
    def receiver_index(self) -> torch.Tensor:
        """``[P]`` int64: which receiver each virtual element belongs to."""

        rank = torch.arange(self.sensor_pair_count, device=self.device)
        return torch.remainder(rank, self.num_rx)

    @classmethod
    def from_axes(cls, axes) -> "ArrayGeometry":
        """Build it from the one metadata record every processing stage reads.

        The half-wavelength offsets and the spacing come off the record, which
        got them off the ``SensorArraySpec``; the phase sign comes off the same
        derived ``doppler_sign`` the Doppler stage uses.
        """

        return cls.from_offsets(
            axes.tx_loc_half_wavelength,
            axes.rx_loc_half_wavelength,
            element_spacing_m=float(axes.element_spacing_m),
            wavelength_m=float(axes.wavelength_m),
            phase_sign=int(axes.doppler_sign),
            device=axes.device,
        )

    @classmethod
    def from_offsets(
        cls,
        tx_loc,
        rx_loc,
        *,
        element_spacing_m: float,
        wavelength_m: float,
        phase_sign: int = -1,
        device: torch.device | str = "cpu",
    ) -> "ArrayGeometry":
        """Build it from two element-offset grids in units of the spacing.

        ``element_spacing_m`` is an explicit argument rather than
        ``wavelength_m / 2``, which is what makes a non-half-wavelength array
        expressible at all.
        """

        target = torch.device(device)
        transmitters = torch.tensor(
            [[float(value) for value in row] for row in tx_loc],
            dtype=torch.float64,
            device=target,
        )
        receivers = torch.tensor(
            [[float(value) for value in row] for row in rx_loc],
            dtype=torch.float64,
            device=target,
        )
        if transmitters.dim() != 2 or int(transmitters.shape[1]) != 3:
            raise ValueError(f"tx_loc must be [num_tx, 3], got {tuple(transmitters.shape)}")
        if receivers.dim() != 2 or int(receivers.shape[1]) != 3:
            raise ValueError(f"rx_loc must be [num_rx, 3], got {tuple(receivers.shape)}")
        num_tx = int(transmitters.shape[0])
        num_rx = int(receivers.shape[0])
        # TX major: element (tx, rx) sits at rank tx * num_rx + rx.
        positions = (
            transmitters.reshape(num_tx, 1, 3) + receivers.reshape(1, num_rx, 3)
        ).reshape(num_tx * num_rx, 3) * float(element_spacing_m)
        return cls(
            element_positions_m=positions.contiguous(),
            num_tx=num_tx,
            num_rx=num_rx,
            element_spacing_m=float(element_spacing_m),
            wavelength_m=float(wavelength_m),
            phase_sign=int(phase_sign),
            tx_loc_half_wavelength=tuple(
                tuple(float(value) for value in row) for row in tx_loc
            ),
            rx_loc_half_wavelength=tuple(
                tuple(float(value) for value in row) for row in rx_loc
            ),
        )


def conventional_steering(
    array: ArrayGeometry,
    directions: torch.Tensor,
    *,
    normalize: bool = True,
    dtype: torch.dtype = torch.complex64,
) -> torch.Tensor:
    """``[P, *beam]`` phase-shift weights for a grid of LOCAL-frame directions.

    ``directions`` is ``[*beam, 3]``; the vectors are expected to be unit length
    and to point FROM the array TOWARD the look direction. They are not
    normalised here, because silently normalising a caller's grid would hide the
    one bug this argument can have.

    With ``normalize=True`` (the default) the weights satisfy ``w^H a = 1`` for a
    wavefront matched to the beam, so a beam cube formed with them is in the
    same amplitude convention as the range and Doppler stages: a single path row
    peaks at its own coefficient magnitude. That is also the constraint
    :func:`mvdr_weights` satisfies, so the two families are interchangeable at
    :func:`~witwin.radar.processing.beam_cube.beam_cube` with no scale factor
    appearing between them.
    """

    if not isinstance(array, ArrayGeometry):
        raise TypeError(
            "conventional_steering takes an ArrayGeometry, so that the element "
            "spacing is read rather than assumed; got "
            f"{type(array).__name__}"
        )
    if not isinstance(directions, torch.Tensor):
        raise TypeError(
            f"directions must be a torch.Tensor, got {type(directions).__name__}"
        )
    if directions.dim() < 1 or int(directions.shape[-1]) != 3:
        raise ValueError(
            f"directions must be [*beam, 3]; got shape {tuple(directions.shape)}"
        )
    offsets = array.element_positions_m
    beam_shape = tuple(directions.shape[:-1])
    flat = directions.reshape(-1, 3).to(torch.float64).to(offsets.device)
    # [P, B]: the projection of every virtual element onto every look direction.
    projection = offsets @ flat.transpose(0, 1)
    wavenumber = 2.0 * math.pi / array.wavelength_m
    # Channel's exp(-j k d) makes the array response exp(+j k <r, u>); a
    # conjugated beat cube reverses it, and phase_sign is the single derived
    # quantity that says which.
    phase = projection * (-array.phase_sign * wavenumber)
    manifold = torch.polar(torch.ones_like(phase), phase).to(dtype)
    if normalize:
        manifold = manifold / array.sensor_pair_count
    return manifold.reshape(array.sensor_pair_count, *beam_shape).contiguous()


def mvdr_weights(
    covariance: torch.Tensor,
    steering: torch.Tensor,
    *,
    diagonal_loading: float,
) -> torch.Tensor:
    """Minimum-variance distortionless-response weights, ``[..., P, B]``.

    ``w = R^-1 a / (a^H R^-1 a)``, which minimises the output power subject to
    ``w^H a = 1`` - the same unit-response normalisation
    :func:`conventional_steering` publishes, so the two are interchangeable
    downstream.

    ``covariance`` is ``[..., P, P]`` Hermitian and ``steering`` is ``[P, *beam]``
    as :func:`conventional_steering` returns it. The result carries the
    covariance's leading batch.

    ``diagonal_loading`` is REQUIRED and has no default. A sample covariance
    estimated from fewer snapshots than elements is exactly singular, and the
    number of snapshots is the caller's fact, not this function's. It is a
    FRACTION of ``trace(R) / P``, so it is scale free: the same 1e-3 means the
    same thing on a covariance in volts squared and on one in watts. The loaded
    matrix is ``R + loading * (trace(R) / P) * I``, and the solve is
    ``torch.linalg.solve``, never an explicit inverse.
    """

    if not isinstance(covariance, torch.Tensor) or not covariance.is_complex():
        raise TypeError("covariance must be a complex torch.Tensor")
    if not isinstance(steering, torch.Tensor) or not steering.is_complex():
        raise TypeError("steering must be a complex torch.Tensor")
    if covariance.dim() < 2 or int(covariance.shape[-1]) != int(covariance.shape[-2]):
        raise ValueError(
            f"covariance must be [..., P, P]; got shape {tuple(covariance.shape)}"
        )
    pairs = int(covariance.shape[-1])
    if int(steering.shape[0]) != pairs:
        raise ValueError(
            f"the covariance spans {pairs} sensor pairs but the steering matrix "
            f"is built for {int(steering.shape[0])}; they must be the same front end"
        )
    loading = float(diagonal_loading)
    if not loading >= 0.0:
        raise ValueError(
            f"diagonal_loading must be non-negative, got {diagonal_loading!r}"
        )
    beam_shape = tuple(steering.shape[1:])
    flat = steering.reshape(pairs, -1).to(covariance.dtype)

    trace = covariance.diagonal(dim1=-2, dim2=-1).sum(dim=-1).real / pairs
    identity = torch.eye(pairs, dtype=covariance.dtype, device=covariance.device)
    loaded = covariance + (loading * trace).reshape(*trace.shape, 1, 1) * identity

    # [..., P, B]: one solve for the whole beam grid, no Python loop over beams.
    inverse_times_a = torch.linalg.solve(loaded, flat)
    normalisation = (flat.conj() * inverse_times_a).sum(dim=-2, keepdim=True)
    weights = inverse_times_a / normalisation
    return weights.reshape(*weights.shape[:-2], pairs, *beam_shape)


__all__ = ["ArrayGeometry", "conventional_steering", "mvdr_weights"]
