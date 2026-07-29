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


import math
from dataclasses import dataclass

import torch

from .signal import _require_complex


@dataclass(frozen=True, slots=True, eq=False)
class BeamCube:
    """``[*beam, D, R]`` complex: the beam / velocity / range cube.

    ``directions`` is the ``[*beam, 3]`` unit-vector grid the cube was formed
    over, in the array's LOCAL frame. It travels with the cube because a beam
    index means nothing without it, and because a detector that reports an angle
    has to read the same grid the former steered with.
    """

    data: torch.Tensor
    axes: object
    directions: torch.Tensor

    def __post_init__(self) -> None:
        _require_complex("data", self.data)
        if self.data.dim() < 3:
            raise ValueError(
                "a beam cube is [*beam, doppler, range]; got shape "
                f"{tuple(self.data.shape)}"
            )
        beam_shape = tuple(self.data.shape[:-2])
        if tuple(self.directions.shape) != (*beam_shape, 3):
            raise ValueError(
                f"the cube spans beams {beam_shape} but directions has shape "
                f"{tuple(self.directions.shape)}; the two are one statement"
            )

    @property
    def range_axis(self) -> torch.Tensor:
        return self.axes.range_m

    @property
    def doppler_axis(self) -> torch.Tensor:
        return self.axes.velocity_mps

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
    :func:`~witwin.radar.processing.angle.beam_cube` with no scale factor
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



"""Angle of arrival: TDM compensation, two FFT routes, and MUSIC.

Every entry here takes an
:class:`~witwin.radar.processing.angle.ArrayGeometry` and a
:class:`~witwin.radar.processing.signal.ProcessingAxes` instead of a legacy
``Radar``, so an estimator can be driven from a synthetic array with no radar
object in sight, and so no estimator can reach a raw ``radar.config`` field.

Three defects of the ``sigproc`` originals are fixed here rather than carried:

* ``_compensate_tdm_phase`` was a Python ``for tx_i in range(1, num_tx)`` loop
  with an in-place ``*=`` on a clone: ``num_tx - 1`` kernel launches for what is
  one broadcast multiply, and a velocity sign that was only correct for the
  unreconciled FMCW Doppler axis it happened to be fed.
  :func:`tdm_compensate` is one multiply and reads the canonical
  closing-positive velocity, with the phasor reconciliation carried by
  ``array.phase_sign``.
* ``MUSICImager`` built its steering vectors at a literal ``spacing = 0.5``. A
  quarter-wave array reported the wrong angle with no symptom other than being
  wrong. :func:`music_spectrum` reads ``array.spacing_wavelengths``.
* ``MUSICImager.music_spectrum`` built its forward-backward smoothing with an
  ``(L + 1) ** 2``-way ``torch.stack`` over a list comprehension - a Python loop
  over sixteen slices at the default smoothing factor. It is one
  :meth:`torch.Tensor.unfold` pair here, and the ``numpy`` angle grids are gone.

The virtual-antenna ordering is TX MAJOR, ``va = tx * num_rx + rx``, matching
the ``[TX, RX, ...]`` cube ``assemble_frame_cube`` publishes and the element
table :class:`ArrayGeometry` builds. Every slice below is expressed through
``array.transmitter_index`` rather than by arithmetic on a raw rank.

**This module sits on BOTH sides of the non-differentiability wall, and the
split is by function rather than by file.**

* :func:`tdm_compensate` is one multiply, :func:`upa_steering` is a manifold,
  and :func:`music_spectrum` is a smooth pseudo-spectrum of the covariance.
  All three stay differentiable, and the MUSIC one is measured rather than
  asserted: its gradient agrees with a central difference on the fixture its
  test pins.
* :func:`phase_comparison_aoa` and :func:`fft2_aoa` read an ``argmax`` BIN and
  publish a direction cosine derived from that index. The index is discrete,
  the cosine is a quantized function of it with a zero derivative inside every
  bin, and the phase read at the peak keeps the tape. Both refuse a derivative
  at their entry.

:func:`music_image` is on the differentiable side, and that is a statement
about this code rather than about MUSIC in general: it selects range bins the
CALLER supplies - it refuses to auto-detect a peak, deliberately, and says so -
and then calls :func:`music_spectrum`. There is no peak pick in it to guard.
"""


import math

import torch

from .signal import _require_complex

from ..policy import refuse_derivative


#: Why the two FFT routes have no derivative. Written once and quoted by both.
_PEAK_BIN_REASON = (
    "the direction cosine is read off an argmax BIN INDEX, which is discrete: "
    "the published cosine is a staircase of the input with a zero derivative "
    "inside every bin and an undefined one at each bin edge, and the phase "
    "sampled at the peak carries a tape describing a peak that is held fixed."
)


#: The direction-cosine rows :func:`phase_comparison_aoa` and :func:`fft2_aoa`
#: return, in order. Published as data so a consumer indexes by meaning.
DIRECTION_COSINE_ROWS = ("x", "y", "z")


def _require_array(array) -> ArrayGeometry:
    if not isinstance(array, ArrayGeometry):
        raise TypeError(
            "this estimator takes an ArrayGeometry so that the element spacing "
            f"is read rather than assumed; got {type(array).__name__}"
        )
    return array


def _require_virtual(virtual_ant: torch.Tensor, array: ArrayGeometry) -> None:
    if not isinstance(virtual_ant, torch.Tensor) or not virtual_ant.is_complex():
        raise TypeError("virtual_ant must be a complex torch.Tensor")
    if virtual_ant.dim() != 2:
        raise ValueError(
            "virtual_ant is [P, N]: one column per detection; got shape "
            f"{tuple(virtual_ant.shape)}"
        )
    if int(virtual_ant.shape[0]) != array.sensor_pair_count:
        raise ValueError(
            f"virtual_ant spans {int(virtual_ant.shape[0])} elements but the "
            f"array is {array.num_tx} x {array.num_rx} = "
            f"{array.sensor_pair_count}"
        )


def _real_dtype(tensor: torch.Tensor) -> torch.dtype:
    return torch.float64 if tensor.dtype == torch.complex128 else torch.float32


# ---------------------------------------------------------------------------
# TDM compensation
# ---------------------------------------------------------------------------


def tdm_compensate(
    aoa_input: torch.Tensor,
    velocities: torch.Tensor,
    array: ArrayGeometry,
    axes,
) -> torch.Tensor:
    """Remove the TDM slot phase a moving target writes across transmitters.

    ``aoa_input`` is ``[P, N]`` complex in the TX-major virtual-antenna order,
    ``velocities`` is ``[N]`` in the canonical CLOSING-POSITIVE convention that
    :data:`~witwin.radar.processing.signal.PROCESSING_DOPPLER_CONVENTION`
    publishes. The result has the same shape and the same dtype.

    The derivation, because the sign is the whole content of this function. A
    cube in the phasor the axes record declares carries
    ``exp(+j s 2 pi f_ref tau)`` with ``s = axes.doppler_sign``. Transmitter
    ``m`` is sampled ``m T_chirp`` later, so its delay is
    ``tau + tau_rate m T_chirp`` and it carries the extra factor
    ``exp(+j s 2 pi f_ref tau_rate m T_chirp)``. With
    ``tau_rate = -2 v_closing / c`` that is
    ``exp(-j s 4 pi v m T_chirp / lambda)``, and the compensation is its
    inverse.

    ``T_chirp`` is the RAW chirp period, ``axes.slow_time_period_s / num_tx``.
    The record publishes the TDM SLOT period, which is ``num_tx`` times larger;
    confusing the two costs a factor of ``num_tx`` in every compensated
    elevation.

    One broadcast multiply, no Python loop, no in-place write. Transmitter zero
    is multiplied by an exact ``1 + 0j`` rather than skipped, which is bitwise
    the identity and is one branch fewer.
    """

    array = _require_array(array)
    _require_virtual(aoa_input, array)
    if not isinstance(velocities, torch.Tensor):
        raise TypeError(
            f"velocities must be a torch.Tensor, got {type(velocities).__name__}"
        )
    if velocities.dim() != 1 or int(velocities.shape[0]) != int(aoa_input.shape[1]):
        raise ValueError(
            f"velocities must be [N] for the {int(aoa_input.shape[1])} detections "
            f"in aoa_input; got shape {tuple(velocities.shape)}"
        )
    chirp_period_s = float(axes.slow_time_period_s) / int(axes.num_tx)
    transmitter = array.transmitter_index.to(velocities.device)
    phase = (
        array.phase_sign
        * 4
        * torch.pi
        * velocities.reshape(1, -1)
        * transmitter.reshape(-1, 1)
        * chirp_period_s
        / array.wavelength_m
    )
    compensation = torch.exp(1j * phase)
    return (aoa_input * compensation).to(aoa_input.dtype)


# ---------------------------------------------------------------------------
# The two FFT routes
# ---------------------------------------------------------------------------


def _finish_direction_cosines(
    x_vector: torch.Tensor, z_vector: torch.Tensor, array: ArrayGeometry
) -> torch.Tensor:
    """``[3, N]``: reconcile the phasor, complete the unit vector, zero the rest.

    **The conjugation trap, for the third and last time.** A DFT peak measures
    the phase progression of the data it was given. Channel's array response is
    ``exp(+j k <r, u>)``, so a Channel-convention cube's peak sits at ``+u`` and
    a CONJUGATED beat cube's peak sits at ``-u``: the same target reads as its
    own mirror image. It is reconciled here, from the same derived
    ``phase_sign`` the Doppler stage and the steering weights read, so there is
    one quantity behind all three and not three sign decisions that can drift.

    ``y`` is the boresight component. A pair of direction cosines whose squares
    already exceed one describes no real direction, so its row is published as
    an exact zero triple rather than as a clamped guess a consumer would plot.
    """

    reconcile = -array.phase_sign
    x_vector = reconcile * x_vector
    z_vector = reconcile * z_vector
    possible = 1 - x_vector.square() - z_vector.square()
    valid = possible >= 0
    zero = torch.zeros_like(x_vector)
    x_vector = torch.where(valid, x_vector, zero)
    z_vector = torch.where(valid, z_vector, zero)
    y_vector = torch.sqrt(torch.clamp(possible, min=0.0))
    return torch.stack((x_vector, y_vector, z_vector), dim=0)


def phase_comparison_aoa(
    virtual_ant: torch.Tensor,
    array: ArrayGeometry,
    *,
    fft_size: int = 64,
) -> torch.Tensor:
    """``[P, N]`` -> ``[3, N]`` direction cosines, by two one-dimensional FFTs.

    Azimuth comes from the ``2 * num_rx`` sub-aperture at the head of the
    virtual array; elevation from the phase difference between that peak and the
    peak of the sub-aperture offset by ``2 * num_rx`` elements, corrected for the
    azimuth walk between the two transmitter rows.

    The exact-bin relation this publishes, and that its test pins:
    ``wx = (2 pi / fft_size) * signed_k`` and ``x = wx / pi``, so on a
    half-wavelength array ``x = sin(theta_az)`` and bin ``k`` is exact when
    ``sin(theta_az) = 2 k / fft_size``.

    Accuracy note for the ELEVATION row. The azimuth-walk correction is built
    from ``wx``, which is the azimuth FFT PEAK BIN and not the continuous
    azimuth, so a target between bins leaves a residual walk of
    ``el_tx_dx`` times the azimuth quantization in the elevation phase. On a
    noiseless off-bin scene at ``fft_size=64`` that measured as a cosine bias of
    about 0.009 at an elevation cosine of 0.08 - inside any half-bin criterion,
    and the reason this route is the coarse one. Precision elevation wants the
    two-dimensional estimator, :func:`fft2_aoa`, which reads both angles off one
    grid instead of correcting one with the other.
    """

    refuse_derivative(
        "witwin.radar.processing.angle.phase_comparison_aoa",
        _PEAK_BIN_REASON,
        virtual_ant=virtual_ant,
    )
    array = _require_array(array)
    _require_virtual(virtual_ant, array)
    if type(fft_size) is not int or fft_size < 2:
        raise ValueError(f"fft_size must be an int of at least 2, got {fft_size!r}")
    if array.num_tx < 3:
        raise ValueError(
            f"phase comparison needs at least three transmitter rows to measure "
            f"an elevation; this array has {array.num_tx}"
        )

    num_rx = array.num_rx
    detections = int(virtual_ant.shape[1])
    device = virtual_ant.device
    real_dtype = _real_dtype(virtual_ant)
    column = torch.arange(detections, device=device)

    n_az = min(2 * num_rx, fft_size)
    azimuth_padded = torch.zeros(
        (fft_size, detections), dtype=virtual_ant.dtype, device=device
    )
    azimuth_padded[:n_az, :] = virtual_ant[:n_az, :]
    azimuth_fft = torch.fft.fft(azimuth_padded, dim=0)
    k_max = torch.argmax(torch.abs(azimuth_fft), dim=0).to(torch.int64)
    peak_azimuth = azimuth_fft[k_max, column]
    signed_k = torch.where(k_max > (fft_size // 2) - 1, k_max - fft_size, k_max)
    wx = (2 * torch.pi / fft_size) * signed_k.to(real_dtype)
    x_vector = wx / torch.pi

    el_start = 2 * num_rx
    n_el = min(num_rx, int(virtual_ant.shape[0]) - el_start)
    if n_el < 1:
        raise ValueError(
            "the elevation sub-aperture is empty: this array has "
            f"{array.sensor_pair_count} virtual elements and the azimuth "
            f"sub-aperture already consumes {el_start}"
        )
    elevation_padded = torch.zeros(
        (fft_size, detections), dtype=virtual_ant.dtype, device=device
    )
    elevation_padded[:n_el, :] = virtual_ant[el_start : el_start + n_el, :]
    elevation_fft = torch.fft.fft(elevation_padded, dim=0)
    elevation_max = torch.argmax(torch.abs(elevation_fft), dim=0).to(torch.int64)
    peak_elevation = elevation_fft[elevation_max, column]

    # The two sub-apertures sit at different transmitters, so the elevation
    # phase difference also carries the azimuth walk between their x offsets.
    # That offset is READ off the declared array rather than assumed to be two.
    tx_offsets = array.tx_loc_half_wavelength
    el_tx_dx = float(tx_offsets[2][0] - tx_offsets[0][0])
    phase_adjust = torch.exp(
        1j * torch.tensor(el_tx_dx, dtype=real_dtype, device=device) * wx
    )
    # The ELEVATION aperture leads the azimuth one by the array's own z offset,
    # so the ratio is taken that way round. The deleted original took its
    # reciprocal and therefore published an elevation cosine that pointed the
    # opposite way to the array's z axis - which is why every legacy elevation
    # assertion in the tree was written on an absolute value. The adapter
    # negates this row back, once and by name.
    wz = torch.angle(
        peak_elevation * torch.conj(peak_azimuth) * torch.conj(phase_adjust)
    )
    return _finish_direction_cosines(x_vector, wz / torch.pi, array)


def fft2_aoa(
    virtual_ant: torch.Tensor,
    array: ArrayGeometry,
    *,
    fft_size: int = 64,
) -> torch.Tensor:
    """``[P, N]`` -> ``[3, N]`` direction cosines, by one zero-padded ``fft2``.

    For a virtual planar array: the odd and even transmitter rows are
    interleaved into a ``num_tx // 2`` by ``2 * num_rx`` grid, zero padded to
    ``fft_size`` on both axes, and the joint peak read directly. The same exact
    bin relation holds on both axes as in :func:`phase_comparison_aoa`.
    """

    refuse_derivative(
        "witwin.radar.processing.angle.fft2_aoa",
        _PEAK_BIN_REASON,
        virtual_ant=virtual_ant,
    )
    array = _require_array(array)
    _require_virtual(virtual_ant, array)
    if type(fft_size) is not int or fft_size < 2:
        raise ValueError(f"fft_size must be an int of at least 2, got {fft_size!r}")
    if array.num_tx < 4:
        raise ValueError(
            "the two-dimensional route needs at least four transmitter rows to "
            f"form a planar grid; this array has {array.num_tx}"
        )

    num_tx = array.num_tx
    num_rx = array.num_rx
    detections = int(virtual_ant.shape[1])
    device = virtual_ant.device
    real_dtype = _real_dtype(virtual_ant)

    reshaped = virtual_ant.reshape(num_tx, num_rx, detections)
    rows = num_tx // 2
    grid = torch.zeros(
        (rows, 2 * num_rx, detections), dtype=virtual_ant.dtype, device=device
    )
    grid[:, :num_rx, :] = reshaped[0::2][:rows]
    grid[:, num_rx:, :] = reshaped[1::2][:rows]

    padded = torch.zeros(
        (fft_size, fft_size, detections), dtype=virtual_ant.dtype, device=device
    )
    padded[:rows, : 2 * num_rx, :] = grid
    spectrum = torch.fft.fft2(padded, dim=(0, 1))
    peak = torch.argmax(torch.abs(spectrum).reshape(-1, detections), dim=0).to(
        torch.int64
    )
    k_el = peak // fft_size
    k_az = peak % fft_size
    k_az = torch.where(k_az > fft_size // 2 - 1, k_az - fft_size, k_az)
    k_el = torch.where(k_el > fft_size // 2 - 1, k_el - fft_size, k_el)
    x_vector = (2 * torch.pi / fft_size) * k_az.to(real_dtype) / torch.pi
    z_vector = (2 * torch.pi / fft_size) * k_el.to(real_dtype) / torch.pi
    return _finish_direction_cosines(x_vector, z_vector, array)


#: The two FFT routes, by name. Route selection is EXPLICIT - the caller picks -
#: because a dispatch on ``num_tx`` is how a change of front end silently
#: changes the estimator.
AOA_ROUTES = {
    "phase_comparison": phase_comparison_aoa,
    "fft2": fft2_aoa,
}


# ---------------------------------------------------------------------------
# MUSIC
# ---------------------------------------------------------------------------


def upa_steering(
    array: ArrayGeometry,
    *,
    rows: int,
    columns: int,
    elevation_rad: torch.Tensor,
    azimuth_rad: torch.Tensor,
    dtype: torch.dtype = torch.complex64,
) -> torch.Tensor:
    """``[T, P, rows * columns]``: the planar manifold MUSIC scans.

    ``rows`` and ``columns`` are the EFFECTIVE array dimensions after spatial
    smoothing, and the element spacing in wavelengths is
    ``array.spacing_wavelengths`` - read, never assumed to be a half.
    """

    array = _require_array(array)
    device = elevation_rad.device
    spacing = array.spacing_wavelengths
    real = elevation_rad.dtype
    row_index = torch.arange(rows, dtype=real, device=device)
    column_index = torch.arange(columns, dtype=real, device=device)
    turn = 2.0 * math.pi * spacing
    # [T, 1, rows] and [1, P, columns]: the two one-dimensional manifolds.
    elevation_phase = turn * torch.sin(elevation_rad).reshape(-1, 1, 1) * row_index
    azimuth_phase = turn * torch.sin(azimuth_rad).reshape(1, -1, 1) * column_index
    along_rows = torch.polar(torch.ones_like(elevation_phase), elevation_phase)
    along_columns = torch.polar(torch.ones_like(azimuth_phase), azimuth_phase)
    manifold = along_rows.unsqueeze(-1) * along_columns.unsqueeze(-2)
    return manifold.reshape(
        int(elevation_rad.shape[0]), int(azimuth_rad.shape[0]), rows * columns
    ).to(dtype)


def music_spectrum(
    angle_data: torch.Tensor,
    array: ArrayGeometry,
    *,
    elevation_rad: torch.Tensor,
    azimuth_rad: torch.Tensor,
    num_signals: int = 7,
    spatial_smooth: int = 3,
) -> torch.Tensor:
    """``[B, M, N, T]`` -> ``[B, len(elevation), len(azimuth)]`` pseudo-spectrum.

    ``B`` range bins, an ``M x N`` planar array, ``T`` snapshots. The covariance
    is forward spatially smoothed over the ``(L + 1) ** 2`` sub-apertures of an
    ``L``-cell smoothing window, which is what decorrelates coherent multipath.

    The smoothing is built with two :meth:`torch.Tensor.unfold` calls rather
    than with an ``(L + 1) ** 2``-way ``torch.stack`` over a list comprehension.
    The sub-aperture ORDER is preserved exactly - row shift major, then column -
    because the smoothed covariance is a sum over them and reordering a float
    sum changes its last bits.

    **This entry is DIFFERENTIABLE and is not guarded**, which is worth stating
    because it sits one function away from two that are. The ``topk`` here sorts
    EIGENVALUES to split the signal subspace from the noise subspace; it is a
    permutation, not a peak pick, and away from an eigenvalue crossing the
    published spectrum is a smooth function of the covariance. Measured on the
    fixture its test pins: the autograd gradient of ``sum |spectrum|`` with
    respect to one element of ``angle_data`` is 1.8279e-2 and a central
    difference at ``h = 1e-2`` gives 1.8311e-2, 0.2 percent apart in a float32
    pipeline. What is NOT differentiable is reading a peak off this spectrum,
    and this function does not do that - :func:`music_image` makes the caller
    supply the range bins for exactly that reason.
    """

    array = _require_array(array)
    if not isinstance(angle_data, torch.Tensor) or not angle_data.is_complex():
        raise TypeError("angle_data must be a complex torch.Tensor")
    if angle_data.dim() != 4:
        raise ValueError(
            "angle_data is [bins, rows, columns, snapshots]; got shape "
            f"{tuple(angle_data.shape)}"
        )
    bins, rows, columns, snapshots = (int(size) for size in angle_data.shape)
    smoothing = int(spatial_smooth)
    if smoothing < 0 or smoothing >= min(rows, columns):
        raise ValueError(
            f"spatial_smooth={spatial_smooth} must be in [0, min(rows, columns)) "
            f"for a {rows} x {columns} array"
        )
    effective_rows = rows - smoothing
    effective_columns = columns - smoothing
    elements = effective_rows * effective_columns
    if not 0 < int(num_signals) < elements:
        raise ValueError(
            f"num_signals={num_signals} must leave a non-empty noise subspace of "
            f"the {elements} effective elements"
        )

    # [B, jj, kk, rows_eff, cols_eff, T] in the same sub-aperture order the
    # (L + 1) ** 2 list comprehension produced: row shift major.
    windows = angle_data.unfold(1, effective_rows, 1).unfold(2, effective_columns, 1)
    sub_apertures = windows.permute(0, 1, 2, 4, 5, 3).reshape(
        bins, (smoothing + 1) ** 2, elements, snapshots
    )
    covariance = torch.einsum(
        "bijk,bilk->bjlk", sub_apertures, sub_apertures.conj()
    ).sum(dim=-1) / (snapshots * (smoothing + 1) ** 2)

    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    _, order = torch.topk(
        eigenvalues, k=eigenvalues.size(-1), largest=True, sorted=True
    )
    noise = torch.gather(
        eigenvectors,
        2,
        order[:, int(num_signals) :]
        .unsqueeze(1)
        .expand(-1, eigenvectors.size(1), -1),
    ).to(torch.complex64)

    steering = upa_steering(
        array,
        rows=effective_rows,
        columns=effective_columns,
        elevation_rad=elevation_rad,
        azimuth_rad=azimuth_rad,
    ).to(device=angle_data.device)
    # ``a^H P_n a``, with the conjugate on the LEFT. The deleted original put it
    # on the right, which evaluates the form at the CONJUGATE steering vector
    # and therefore peaks at the mirror image of the true angle. Its own angle
    # grids ran from ``+fov/2`` down to ``-fov/2``, which hid the mirror behind
    # a descending axis and made the published spectrum wrong by a reflection.
    # Corrected here, and the reflection is asserted against the pre-cutover
    # golden in tests/processing/test_adapters.py.
    projector = torch.matmul(noise, noise.transpose(-1, -2).conj())
    quadratic = torch.matmul(
        torch.einsum("ijk,akl->aijl", steering.conj(), projector),
        steering.transpose(-1, -2),
    )
    return torch.reciprocal(quadratic.diagonal(dim1=-2, dim2=-1)).reshape(
        bins, int(elevation_rad.shape[0]), int(azimuth_rad.shape[0])
    )


def music_image(
    profile,
    array: ArrayGeometry,
    *,
    elevation_rad: torch.Tensor,
    azimuth_rad: torch.Tensor,
    range_bins: torch.Tensor | None = None,
    num_signals: int = 7,
    spatial_smooth: int = 3,
    num_snapshots: int = 8,
) -> torch.Tensor:
    """A ``RangeProfile`` -> ``[elevation, azimuth, bins]`` MUSIC image.

    ``MUSICImager.radar_image`` did its own ``torch.fft.fft(sig, dim=3)`` - a
    third range transform, with no window and no axes record. This takes a
    :class:`~witwin.radar.processing.signal.RangeProfile` from the one range
    owner instead, so the image and the point cloud are formed on the same range
    grid by construction.
    """

    from .range_doppler import RangeProfile

    if not isinstance(profile, RangeProfile):
        raise TypeError(
            "music_image consumes a RangeProfile, so the range transform has "
            f"exactly one owner; got {type(profile).__name__}"
        )
    data = profile.data
    if data.dim() != 4:
        raise ValueError(
            "the profile must be [tx, rx, slow_time, range] for an image; got "
            f"shape {tuple(data.shape)}"
        )
    if range_bins is None:
        raise ValueError(
            "range_bins is required: an auto-detected peak is a modelling choice "
            "made silently, and the caller already has the range axis"
        )
    selected = range_bins.to(device=data.device, dtype=torch.int64)
    # [bins, tx, rx, snapshots]
    angle_data = data[:, :, :num_snapshots, :].index_select(-1, selected).permute(
        3, 0, 1, 2
    )
    image = music_spectrum(
        angle_data.contiguous(),
        array,
        elevation_rad=elevation_rad,
        azimuth_rad=azimuth_rad,
        num_signals=num_signals,
        spatial_smooth=spatial_smooth,
    )
    return image.permute(1, 2, 0)



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
:mod:`witwin.radar.processing.angle`, together with the array geometry and
the phase-sign reconciliation, so a weight family can be swapped here without
this module learning what an element is.

The pair axes of the map are flattened in the order the cube is published in,
which is TX major - ``[TX, RX, ...]`` out of ``assemble_frame_cube`` - and that
is the order :class:`~witwin.radar.processing.angle.ArrayGeometry` builds
its element table in.
"""


import torch

from .signal import _require_complex

from .range_doppler import RangeDopplerMap


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
