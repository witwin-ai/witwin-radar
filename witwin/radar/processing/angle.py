"""The array, the two weight families, the angle estimators and the beam cube.

Everything angular in this package - the beam cube, the FFT angle estimators,
MUSIC, the point cloud - reads :class:`ArrayGeometry` and nothing else, so
there is exactly one statement of where an element is and exactly one
statement of which way its phase runs.

**No half-wavelength spacing is hard coded anywhere.**
:attr:`ArrayGeometry.element_spacing_m` is what one unit of the declared
``tx_loc`` / ``rx_loc`` grid is worth in metres, and
:attr:`ArrayGeometry.spacing_wavelengths` is the same number divided by the
wavelength. Both are data, so a quarter-wave array reports the right angle.

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

**One frame for every direction: the PROCESSING frame.** ``x`` is the radar's
right, ``y`` its boresight and ``z`` its up. Element positions, steering
directions, beam grids, direction cosines and point clouds are all in it. The
radar's own LOCAL frame (``Radar.tx``/``rx`` and ``SensorArraySpec``) is
``(right, up, -boresight)``, so :meth:`ArrayGeometry.from_axes` maps an offset
``(x, y, z)`` declared there to ``(x, -z, y)`` here, and
:meth:`~witwin.radar.processing.detection.PointCloud.to_world` maps back
through the radar's pose.

**Build it once per array.** The record holds a materialised ``[P, 3]`` tensor,
so a per-frame :func:`conventional_steering` costs no host-to-device copy of the
element table. Rebuilding the geometry inside every call was measured at
152 microseconds - the most expensive call in the whole processing chain - for a
quantity that does not change between frames.

**This module sits on BOTH sides of the non-differentiability wall, and the
split is by function rather than by file.** :func:`tdm_compensate` is one
multiply, :func:`upa_steering` is a manifold, :func:`music_spectrum` is a
smooth pseudo-spectrum of the covariance and :func:`beam_cube` is a weighted
sum; :func:`range_angle_map` is a beam cube's power. All stay differentiable.
:func:`phase_comparison_aoa`, :func:`fft2_aoa` and :func:`scan_aoa` read an
``argmax`` BIN and publish a direction cosine derived from that index, so all
three refuse a derivative at their entry.
:func:`music_image` selects range bins the CALLER supplies - it refuses to
auto-detect a peak - and then calls :func:`music_spectrum`, so there is no
peak pick in it to guard.
"""

import math
from dataclasses import dataclass, field, replace

import torch

from ..display import compact_repr
from ..policy import refuse_derivative
from .range_doppler import RangeDopplerMap, RangeProfile
from .signal import ProcessingAxes, ProcessingCube, _require_complex, real_dtype_of


@dataclass(frozen=True, slots=True, eq=False)
class BeamCube:
    """``[*beam, D, R]`` complex: the beam / velocity / range cube.

    ``directions`` is the ``[*beam, 3]`` unit-vector grid the cube was formed
    over, in the array's LOCAL frame. It travels with the cube because a beam
    index means nothing without it, and because a detector that reports an angle
    has to read the same grid the former steered with.
    """

    __repr__ = compact_repr

    data: torch.Tensor
    axes: ProcessingAxes
    directions: torch.Tensor

    def __post_init__(self) -> None:
        _require_complex("data", self.data)
        if self.data.dim() < 3:
            raise ValueError(f"a beam cube is [*beam, doppler, range]; got shape {tuple(self.data.shape)}")
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

    ``element_positions_m`` is ``[P, 3]`` float64 in the PROCESSING frame
    (``x`` right, ``y`` boresight, ``z`` up), ordered TX major. A TDM-MIMO
    virtual element sits at the SUM of its transmitter and receiver offsets,
    because the round-trip phase is the sum of the two one-way phases. The two
    offset tuples are in the same frame, in units of ``element_spacing_m``.

    ``lattice`` is the array's description as a planar grid when it is one -
    see :class:`ArrayLattice` - and ``None`` otherwise. It is derived from the
    declared offsets on the host at construction, so no estimator reads an
    element position back from the device to decide what it may do.
    """

    __repr__ = compact_repr

    element_positions_m: torch.Tensor
    num_tx: int
    num_rx: int
    element_spacing_m: float
    wavelength_m: float
    phase_sign: int
    tx_loc_half_wavelength: tuple[tuple[float, float, float], ...]
    rx_loc_half_wavelength: tuple[tuple[float, float, float], ...]
    lattice: "ArrayLattice | None" = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "lattice", ArrayLattice.of(self._virtual_offsets()))
        if not isinstance(self.element_positions_m, torch.Tensor):
            raise TypeError(
                f"element_positions_m must be a torch.Tensor, got {type(self.element_positions_m).__name__}"
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
            raise ValueError(f"element_spacing_m must be positive, got {self.element_spacing_m}")

    @property
    def sensor_pair_count(self) -> int:
        return self.num_tx * self.num_rx

    @property
    def device(self) -> torch.device:
        return self.element_positions_m.device

    def _virtual_offsets(self) -> tuple[tuple[float, float, float], ...]:
        """The ``P`` virtual offsets in spacing units, TX major, as host floats."""

        return tuple(
            tuple(a + b for a, b in zip(tx, rx, strict=True))
            for tx in self.tx_loc_half_wavelength
            for rx in self.rx_loc_half_wavelength
        )

    @property
    def spacing_wavelengths(self) -> float:
        """One unit of the declared element grid, in wavelengths.

        Exactly ``0.5`` for the conventional half-wavelength array, and read
        rather than assumed everywhere.
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
    def from_axes(cls, axes: ProcessingAxes) -> "ArrayGeometry":
        """Build it from the one metadata record every processing stage reads.

        The half-wavelength offsets and the spacing come off the record, which
        got them off the ``SensorArraySpec`` in the radar's LOCAL frame
        ``(right, up, -boresight)``; they are mapped to the processing frame
        ``(right, boresight, up)`` here, once, as ``(x, y, z) -> (x, -z, y)``.
        The phase sign comes off the same derived ``doppler_sign`` the Doppler
        stage uses.
        """

        return cls.from_offsets(
            _radar_local_to_processing(axes.tx_loc_half_wavelength),
            _radar_local_to_processing(axes.rx_loc_half_wavelength),
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

        The offsets are in the PROCESSING frame: ``x`` right, ``y`` boresight,
        ``z`` up. An offset list written in the radar's local frame (the one
        ``Radar.tx`` uses) goes through :meth:`from_axes` instead.
        ``element_spacing_m`` is an explicit argument rather than
        ``wavelength_m / 2``, which is what makes a non-half-wavelength array
        expressible at all.
        """

        target = torch.device(device)
        transmitters = torch.tensor(
            [[float(value) for value in row] for row in tx_loc], dtype=torch.float64, device=target
        )
        receivers = torch.tensor(
            [[float(value) for value in row] for row in rx_loc], dtype=torch.float64, device=target
        )
        if transmitters.dim() != 2 or int(transmitters.shape[1]) != 3:
            raise ValueError(f"tx_loc must be [num_tx, 3], got {tuple(transmitters.shape)}")
        if receivers.dim() != 2 or int(receivers.shape[1]) != 3:
            raise ValueError(f"rx_loc must be [num_rx, 3], got {tuple(receivers.shape)}")
        num_tx = int(transmitters.shape[0])
        num_rx = int(receivers.shape[0])
        # TX major: element (tx, rx) sits at rank tx * num_rx + rx.
        positions = (transmitters.reshape(num_tx, 1, 3) + receivers.reshape(1, num_rx, 3)).reshape(
            num_tx * num_rx, 3
        ) * float(element_spacing_m)
        return cls(
            element_positions_m=positions.contiguous(),
            num_tx=num_tx,
            num_rx=num_rx,
            element_spacing_m=float(element_spacing_m),
            wavelength_m=float(wavelength_m),
            phase_sign=int(phase_sign),
            tx_loc_half_wavelength=tuple(tuple(float(value) for value in row) for row in tx_loc),
            rx_loc_half_wavelength=tuple(tuple(float(value) for value in row) for row in rx_loc),
        )


def _radar_local_to_processing(rows) -> tuple[tuple[float, float, float], ...]:
    """``(right, up, -boresight)`` offsets -> ``(right, boresight, up)``: ``(x, y, z) -> (x, -z, y)``."""

    return tuple((float(x), -float(z) + 0.0, float(y)) for x, y, z in rows)


#: How close a declared offset must sit to a lattice point, in spacing units.
_LATTICE_TOLERANCE = 1e-6
#: The largest lattice extent an FFT route accepts along one axis. An offset
#: list with an irrational spacing ratio has a float GCD near the tolerance and
#: an extent near ``1 / tolerance``; it is not a lattice in any useful sense.
_LATTICE_MAX_EXTENT = 4096


def _float_gcd(values) -> float:
    """The largest ``g`` with every value an integer multiple of it, to the lattice tolerance."""

    found = 0.0
    for value in values:
        a, b = max(found, value), min(found, value)
        while b > _LATTICE_TOLERANCE:
            a, b = b, math.fmod(a, b)
            if a - b < _LATTICE_TOLERANCE:
                b = 0.0
        found = a
    return found


@dataclass(frozen=True, slots=True)
class ArrayLattice:
    """A virtual array whose elements sit on a planar ``x``-``z`` grid.

    ``pitch_x``/``pitch_z`` are the grid steps in units of the element spacing,
    ``0.0`` along an axis the array does not extend in. ``index_x``/``index_z``
    give each TX-major virtual element its grid cell, and ``size_x``/``size_z``
    the grid extent. Several elements may share a cell (overlapping MIMO
    virtual elements); a cell may be empty (a hole). Every element shares one
    boresight (``y``) offset, which is what makes the grid planar.

    The grid steps fix the FFT routes' unambiguous field of view: along ``x``
    a direction cosine is recovered modulo ``1 / (pitch_x * spacing_wavelengths)``,
    so a half-wavelength grid sees all of ``[-1, 1)`` and a one-wavelength grid
    sees ``[-1/2, 1/2)``.
    """

    pitch_x: float
    pitch_z: float
    index_x: tuple[int, ...]
    index_z: tuple[int, ...]
    size_x: int
    size_z: int

    @classmethod
    def of(cls, offsets) -> "ArrayLattice | None":
        """The lattice the offsets sit on, or ``None`` when they sit on none."""

        boresight = [row[1] for row in offsets]
        if max(boresight) - min(boresight) > _LATTICE_TOLERANCE:
            return None
        pitches, indices, sizes = [], [], []
        for axis in (0, 2):
            values = [row[axis] - min(r[axis] for r in offsets) for row in offsets]
            pitch = _float_gcd(values)
            if pitch <= _LATTICE_TOLERANCE:
                pitches.append(0.0)
                indices.append(tuple(0 for _ in values))
                sizes.append(1)
                continue
            index = [round(value / pitch) for value in values]
            if any(abs(value / pitch - cell) > _LATTICE_TOLERANCE for value, cell in zip(values, index, strict=True)):
                return None
            if max(index) + 1 > _LATTICE_MAX_EXTENT:
                return None
            pitches.append(pitch)
            indices.append(tuple(index))
            sizes.append(max(index) + 1)
        return cls(
            pitch_x=pitches[0],
            pitch_z=pitches[1],
            index_x=indices[0],
            index_z=indices[1],
            size_x=sizes[0],
            size_z=sizes[1],
        )


def conventional_steering(
    array: ArrayGeometry, directions: torch.Tensor, *, normalize: bool = True, dtype: torch.dtype = torch.complex64
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
        raise TypeError(f"directions must be a torch.Tensor, got {type(directions).__name__}")
    if directions.dim() < 1 or int(directions.shape[-1]) != 3:
        raise ValueError(f"directions must be [*beam, 3]; got shape {tuple(directions.shape)}")
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


def mvdr_weights(covariance: torch.Tensor, steering: torch.Tensor, *, diagonal_loading: float) -> torch.Tensor:
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
        raise ValueError(f"covariance must be [..., P, P]; got shape {tuple(covariance.shape)}")
    pairs = int(covariance.shape[-1])
    if int(steering.shape[0]) != pairs:
        raise ValueError(
            f"the covariance spans {pairs} sensor pairs but the steering matrix "
            f"is built for {int(steering.shape[0])}; they must be the same front end"
        )
    loading = float(diagonal_loading)
    if not loading >= 0.0:
        raise ValueError(f"diagonal_loading must be non-negative, got {diagonal_loading!r}")
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


#: Why the two FFT routes have no derivative. Written once and quoted by both.
_PEAK_BIN_REASON = (
    "the direction cosine is read off an argmax BIN INDEX, which is discrete: "
    "the published cosine is a staircase of the input with a zero derivative "
    "inside every bin and an undefined one at each bin edge, and the phase "
    "sampled at the peak carries a tape describing a peak that is held fixed."
)


def _require_array(array) -> ArrayGeometry:
    if not isinstance(array, ArrayGeometry):
        raise TypeError(
            "this estimator takes an ArrayGeometry so that the element spacing "
            f"is read rather than assumed; got {type(array).__name__}"
        )
    return array


def require_separated(axes: ProcessingAxes, stage: str) -> None:
    """Refuse a DDM or BPM product whose transmitters still share one channel.

    Every angle route reads one element per (transmitter, receiver) pair. A
    DDM or BPM cube holds all transmitters in one transmit channel until
    :func:`demultiplex` separates them, which it can do only in slow time,
    before the Doppler transform; a map built from the physical cube has no
    virtual array to steer. ``Frame.points`` and ``Frame.range_angle``
    demultiplex on their own.
    """

    if axes.mimo != "tdm" and not axes.demultiplexed:
        raise ValueError(
            f"{stage} needs one channel per transmitter, but this {axes.mimo.upper()} product holds every transmitter "
            "in one channel: separate them with witwin.radar.processing.demultiplex(cube) before the range and "
            "Doppler transforms, or use the Frame facade, which does it"
        )


def _require_virtual(virtual_ant: torch.Tensor, array: ArrayGeometry) -> None:
    if not isinstance(virtual_ant, torch.Tensor) or not virtual_ant.is_complex():
        raise TypeError("virtual_ant must be a complex torch.Tensor")
    if virtual_ant.dim() != 2:
        raise ValueError(f"virtual_ant is [P, N]: one column per detection; got shape {tuple(virtual_ant.shape)}")
    if int(virtual_ant.shape[0]) != array.sensor_pair_count:
        raise ValueError(
            f"virtual_ant spans {int(virtual_ant.shape[0])} elements but the "
            f"array is {array.num_tx} x {array.num_rx} = "
            f"{array.sensor_pair_count}"
        )


def tdm_compensate(
    aoa_input: torch.Tensor, velocities: torch.Tensor, array: ArrayGeometry, axes: ProcessingAxes
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
    if getattr(axes, "mimo", "tdm") != "tdm":
        # DDM and BPM fire every transmitter every chirp: there is no slot phase.
        return aoa_input
    if not isinstance(velocities, torch.Tensor):
        raise TypeError(f"velocities must be a torch.Tensor, got {type(velocities).__name__}")
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


def demultiplex(cube: ProcessingCube) -> ProcessingCube:
    """Separate a DDM or BPM cube into its ``[TX, RX, slow, fast]`` virtual array (R-ADR-048).

    ``cube`` is ``[1, RX, C, S]`` as received: each receiver holds every
    transmitter, transmitter ``t`` carrying ``code[c, t]`` (the axes record's
    ``transmit_code``, in the cube's phasor convention). Both schemes decimate
    slow time by the code period ``P`` (``axes.code_period``), so the virtual
    cube has ``K = C / P`` slow-time samples at ``P`` chirp periods and the
    unambiguous velocity is ``1 / P`` of the physical cube's:

    * DDM. Multiplying by ``conj(code[:, t])`` moves transmitter ``t`` to zero
      Doppler and every other transmitter ``(t' - t) / P`` of the band away; the
      ``K`` slow-time bins around zero, transformed back (one forward and one
      ``K``-point inverse transform, scaled by ``K / C`` so a tone keeps its
      amplitude), are transmitter ``t`` alone. A target faster than the
      sub-band edge folds into a neighbour's band: that is DDM's ambiguity.
    * BPM. Block ``b`` of ``P`` chirps is inverted by the code: ``z_t[b] = (1/P)
      sum_i conj(code[bP + i, t]) y[bP + i]``, exact while the scene holds
      still across a block; a moving target leaves a residual between the
      transmitters that grows with its Doppler phase over the block.

    A TDM cube is already separated and is returned as it is.
    """

    if not isinstance(cube, ProcessingCube):
        raise TypeError(f"demultiplex takes a ProcessingCube, got {type(cube).__name__}")
    axes = cube.axes
    if axes.mimo == "tdm" or axes.demultiplexed:
        return cube
    if axes.transmit_code is None:
        raise ValueError(f"a {axes.mimo} axes record must carry its transmit code")
    data = cube.data[0]
    receivers, chirps, samples = (int(value) for value in data.shape)
    period = int(axes.code_period)
    if chirps % period:
        raise ValueError(f"{chirps} chirps do not split into code periods of {period}")
    blocks = chirps // period
    code = axes.transmit_code.to(device=data.device, dtype=data.dtype)
    if axes.mimo == "ddm":
        # [T, R, C, S]: each transmitter at baseband, then its own sub-band.
        baseband = data.unsqueeze(0) * code.conj().transpose(0, 1).reshape(-1, 1, chirps, 1)
        spectrum = torch.fft.fft(baseband, dim=2)
        half = blocks // 2
        band = torch.cat((spectrum[:, :, : blocks - half], spectrum[:, :, chirps - half :]), dim=2)
        virtual = torch.fft.ifft(band, dim=2) * (blocks / chirps)
    elif axes.mimo == "bpm":
        blocked = data.reshape(receivers, blocks, period, samples)
        weights = code.conj().reshape(blocks, period, -1)
        virtual = torch.einsum("rbps,bpt->trbs", blocked, weights) / period
    else:
        raise ValueError(f"no demultiplexer for mimo={axes.mimo!r}")
    period_s = axes.slow_time_period_s * period
    record = replace(
        axes,
        slow_time_period_s=period_s,
        max_unambiguous_speed_mps=axes.wavelength_m / (4.0 * period_s),
        demultiplexed=True,
    ).with_bin_counts(doppler_bins=blocks)
    return ProcessingCube(data=virtual.contiguous(), axes=record)


def _finish_direction_cosines(x_vector: torch.Tensor, z_vector: torch.Tensor) -> torch.Tensor:
    """``[3, N]``: complete reconciled ``x``/``z`` cosines into unit vectors, zero the rest.

    ``y`` is the boresight component, taken in front of the array. A pair of
    direction cosines whose squares already exceed one describes no real
    direction, so its row is published as an exact zero triple rather than as
    a clamped guess a consumer would plot.
    """

    possible = 1 - x_vector.square() - z_vector.square()
    valid = possible >= 0
    zero = torch.zeros_like(x_vector)
    x_vector = torch.where(valid, x_vector, zero)
    z_vector = torch.where(valid, z_vector, zero)
    y_vector = torch.sqrt(torch.clamp(possible, min=0.0))
    return torch.stack((x_vector, y_vector, z_vector), dim=0)


def _signed_bin(index: torch.Tensor, size: int) -> torch.Tensor:
    return torch.where(index > size // 2 - 1, index - size, index)


def _require_fft_size(fft_size) -> int:
    if type(fft_size) is not int or fft_size < 2:
        raise ValueError(f"fft_size must be an int of at least 2, got {fft_size!r}")
    return fft_size


def _refuse_layout(route: str, array: ArrayGeometry, why: str) -> ValueError:
    offsets = array._virtual_offsets()
    return ValueError(
        f"route {route!r} cannot estimate angles on this {array.num_tx} x {array.num_rx} array: {why}. Its virtual "
        f"elements sit at {offsets} (processing frame, spacing units). Use route='scan', which steers the declared "
        "geometry directly, or route='fft2' for an array on a planar lattice"
    )


def phase_comparison_aoa(virtual_ant: torch.Tensor, array: ArrayGeometry, *, fft_size: int = 64) -> torch.Tensor:
    """``[P, N]`` -> ``[3, N]`` direction cosines, by one FFT and one phase comparison.

    The layout this route is written for, checked rather than assumed: the
    first ``2 * num_rx`` virtual elements, in index order, form a uniform line
    along ``x`` (pitch ``p`` spacing units), and the next ``num_rx`` form a
    uniform line of the same pitch displaced by ``dz != 0`` along ``z`` and by a
    whole number ``m`` of pitches along ``x``, so that it sits directly above
    elements ``m .. m + num_rx - 1`` of the first line. The TI three-transmitter
    layout is this shape. Anything else is refused by name.

    Azimuth is the zero-padded FFT peak of the first line:
    ``u_x = s k / (fft_size * p * d / lambda)`` with ``s = -phase_sign`` and
    ``d`` the element spacing, so on a half-wavelength line bin ``k`` is exact
    when ``u_x = 2 k / fft_size``. Elevation is the phase of
    ``sum_i b_i conj(a_{m+i})`` over the vertically aligned pairs, which carries
    ``s 2 pi (d / lambda) dz u_z`` and nothing of the azimuth, so it needs no
    walk correction and is unbiased between azimuth bins.
    """

    refuse_derivative("witwin.radar.processing.angle.phase_comparison_aoa", _PEAK_BIN_REASON, virtual_ant=virtual_ant)
    array = _require_array(array)
    _require_virtual(virtual_ant, array)
    fft_size = _require_fft_size(fft_size)
    pitch, shift, dz = _phase_comparison_layout(array)
    num_rx = array.num_rx
    n_az = 2 * num_rx
    if fft_size < n_az:
        raise ValueError(f"fft_size={fft_size} is shorter than the {n_az}-element azimuth line")
    real_dtype = real_dtype_of(virtual_ant)
    reconcile = -array.phase_sign
    turn = array.spacing_wavelengths

    azimuth_fft = torch.fft.fft(virtual_ant[:n_az], n=fft_size, dim=0)
    k_max = torch.argmax(torch.abs(azimuth_fft), dim=0).to(torch.int64)
    x_vector = reconcile * _signed_bin(k_max, fft_size).to(real_dtype) / (fft_size * pitch * turn)

    upper = virtual_ant[n_az : n_az + num_rx]
    lower = virtual_ant[shift : shift + num_rx]
    wz = torch.angle((upper * lower.conj()).sum(dim=0))
    z_vector = reconcile * wz / (2.0 * math.pi * turn * dz)
    return _finish_direction_cosines(x_vector, z_vector)


def _phase_comparison_layout(array: ArrayGeometry) -> tuple[float, int, float]:
    """``(pitch, m, dz)`` of the phase-comparison layout, or a named refusal."""

    route = "phase_comparison"
    if array.num_tx < 3:
        raise _refuse_layout(
            route, array, f"it needs a third transmitter row for elevation, this array has {array.num_tx}"
        )
    offsets = array._virtual_offsets()
    num_rx = array.num_rx
    line = offsets[: 2 * num_rx]
    upper = offsets[2 * num_rx : 3 * num_rx]
    tolerance = _LATTICE_TOLERANCE

    def uniform(rows) -> float | None:
        if any(abs(r[1] - rows[0][1]) > tolerance or abs(r[2] - rows[0][2]) > tolerance for r in rows):
            return None
        if len(rows) < 2:
            return None
        step = rows[1][0] - rows[0][0]
        if step <= tolerance:
            return None
        if any(abs(b[0] - a[0] - step) > tolerance for a, b in zip(rows, rows[1:], strict=False)):
            return None
        return step

    pitch = uniform(line)
    if pitch is None:
        raise _refuse_layout(route, array, f"its first {2 * num_rx} virtual elements are not a uniform line along x")
    if num_rx > 1 and (uniform(upper) is None or abs(uniform(upper) - pitch) > tolerance):
        raise _refuse_layout(route, array, "the third transmitter's elements are not a line of the same pitch")
    dz = upper[0][2] - line[0][2]
    if abs(dz) <= tolerance or abs(upper[0][1] - line[0][1]) > tolerance:
        raise _refuse_layout(route, array, "the third transmitter is not displaced along z alone")
    shift_float = (upper[0][0] - line[0][0]) / pitch
    shift = round(shift_float)
    if abs(shift_float - shift) > tolerance or not 0 <= shift <= num_rx:
        raise _refuse_layout(
            route, array, "the third transmitter's elements do not sit above elements of the first line"
        )
    return pitch, int(shift), dz


def _lattice_grid(virtual_ant: torch.Tensor, array: ArrayGeometry, lattice: ArrayLattice) -> torch.Tensor:
    """``[P, N]`` -> ``[size_z, size_x, N]``: the snapshot placed on its lattice.

    Elements sharing a cell are averaged, which is exact for a plane wave (they
    carry the same phase) and keeps the grid in the snapshot's amplitude. An
    empty cell is a zero, so the transform of the grid is the steering scan of
    the declared elements evaluated on the FFT's direction grid.
    """

    cells = lattice.size_z * lattice.size_x
    flat_index = torch.tensor(
        [z * lattice.size_x + x for x, z in zip(lattice.index_x, lattice.index_z, strict=True)],
        dtype=torch.int64,
        device=virtual_ant.device,
    )
    detections = int(virtual_ant.shape[1])
    grid = torch.zeros((cells, detections), dtype=virtual_ant.dtype, device=virtual_ant.device)
    grid.index_add_(0, flat_index, virtual_ant)
    multiplicity = torch.zeros(cells, dtype=real_dtype_of(virtual_ant), device=virtual_ant.device)
    multiplicity.index_add_(0, flat_index, torch.ones_like(flat_index, dtype=multiplicity.dtype))
    grid = grid / multiplicity.clamp(min=1.0).reshape(-1, 1)
    return grid.reshape(lattice.size_z, lattice.size_x, detections)


def fft2_aoa(virtual_ant: torch.Tensor, array: ArrayGeometry, *, fft_size: int = 64) -> torch.Tensor:
    """``[P, N]`` -> ``[3, N]`` direction cosines, by one zero-padded FFT over the lattice.

    For any array on a planar lattice (:class:`ArrayLattice`): a uniform line
    (ULA) is the one-row case, a filled or holed rectangle the general one.
    The snapshot is placed on its grid, zero padded to ``fft_size`` along each
    axis the array extends in, and the joint peak read directly:
    ``u_x = s k_x / (fft_size * pitch_x * d / lambda)``, likewise for ``z``,
    with ``s = -phase_sign``. An axis the array does not extend in has no
    phase progression and its cosine is published as zero; a line array
    therefore reports the cone angle it measures, placed in the
    ``x``-``y`` (or ``z``-``y``) plane.

    The cosine is recovered within the lattice's unambiguous field of view,
    ``|u| < 1 / (2 pitch d / lambda)`` - all of ``[-1, 1)`` for a half
    wavelength pitch - and an array without a lattice is refused by name.
    """

    refuse_derivative("witwin.radar.processing.angle.fft2_aoa", _PEAK_BIN_REASON, virtual_ant=virtual_ant)
    array = _require_array(array)
    _require_virtual(virtual_ant, array)
    fft_size = _require_fft_size(fft_size)
    lattice = array.lattice
    if lattice is None:
        raise _refuse_layout("fft2", array, "its elements do not sit on a planar x-z lattice with one boresight offset")
    if max(lattice.size_x, lattice.size_z) > fft_size:
        raise ValueError(
            f"fft_size={fft_size} is shorter than the {lattice.size_z} x {lattice.size_x} lattice it transforms"
        )
    real_dtype = real_dtype_of(virtual_ant)
    detections = int(virtual_ant.shape[1])
    grid = _lattice_grid(virtual_ant, array, lattice)
    size_z = fft_size if lattice.size_z > 1 else 1
    size_x = fft_size if lattice.size_x > 1 else 1
    spectrum = torch.fft.fft2(grid, s=(size_z, size_x), dim=(0, 1))
    peak = torch.argmax(torch.abs(spectrum).reshape(-1, detections), dim=0).to(torch.int64)
    k_z = _signed_bin(peak // size_x, size_z)
    k_x = _signed_bin(peak % size_x, size_x)
    reconcile = -array.phase_sign
    turn = array.spacing_wavelengths
    zero = torch.zeros(detections, dtype=real_dtype, device=virtual_ant.device)
    x_vector = zero if size_x == 1 else reconcile * k_x.to(real_dtype) / (size_x * lattice.pitch_x * turn)
    z_vector = zero if size_z == 1 else reconcile * k_z.to(real_dtype) / (size_z * lattice.pitch_z * turn)
    return _finish_direction_cosines(x_vector, z_vector)


def scan_aoa(virtual_ant: torch.Tensor, array: ArrayGeometry, *, fft_size: int = 64) -> torch.Tensor:
    """``[P, N]`` -> ``[3, N]`` direction cosines, by a steering-vector scan of any layout.

    The peak of ``|w(u)^H a|^2`` over a grid of directions, where ``w`` is
    :func:`conventional_steering` of the declared element positions - any
    planar or non-planar layout, holes, overlaps and non-integer offsets
    included. The grid is ``fft_size`` direction cosines per axis the array
    extends in, ``u = -U + 2 U k / fft_size`` with ``U = 1`` or, on a lattice,
    the lattice's unambiguous half width when that is smaller, so a
    grating-lobe pair is never two equal peaks of one scan. On a half
    wavelength lattice this is the FFT route's own grid, and the two routes
    agree bin for bin.

    A line array measures one cosine; the axis it does not extend in is
    published as zero, as on :func:`fft2_aoa`. A line that is not along ``x`` or
    ``z`` mixes the two and is refused by name.
    """

    refuse_derivative("witwin.radar.processing.angle.scan_aoa", _PEAK_BIN_REASON, virtual_ant=virtual_ant)
    array = _require_array(array)
    _require_virtual(virtual_ant, array)
    fft_size = _require_fft_size(fft_size)
    offsets = array._virtual_offsets()
    extent = [max(r[axis] for r in offsets) - min(r[axis] for r in offsets) for axis in range(3)]
    spans_x = extent[0] > _LATTICE_TOLERANCE
    spans_z = extent[2] > _LATTICE_TOLERANCE
    if spans_x and spans_z and _collinear(offsets):
        raise _refuse_layout("scan", array, "it is a line that is not along x or z, so it measures a mixed cosine")
    lattice = array.lattice
    turn = array.spacing_wavelengths

    def half_width(pitch: float) -> float:
        return 1.0 if pitch <= 0.0 else min(1.0, 1.0 / (2.0 * pitch * turn))

    device = virtual_ant.device
    step = torch.arange(fft_size, dtype=torch.float64, device=device) / fft_size
    zero = torch.zeros(1, dtype=torch.float64, device=device)
    width_x = half_width(lattice.pitch_x if lattice is not None else 0.0)
    width_z = half_width(lattice.pitch_z if lattice is not None else 0.0)
    grid_x = width_x * (2.0 * step - 1.0) if spans_x else zero
    grid_z = width_z * (2.0 * step - 1.0) if spans_z else zero
    u_z, u_x = torch.meshgrid(grid_z, grid_x, indexing="ij")
    u_x, u_z = u_x.reshape(-1), u_z.reshape(-1)
    boresight_sq = 1.0 - u_x.square() - u_z.square()
    directions = torch.stack((u_x, torch.sqrt(boresight_sq.clamp(min=0.0)), u_z), dim=1)
    weights = conventional_steering(array, directions, normalize=False, dtype=virtual_ant.dtype)
    response = torch.abs(weights.transpose(0, 1).conj() @ virtual_ant)
    response = torch.where(boresight_sq.reshape(-1, 1) >= 0, response, torch.full_like(response, -1.0))
    best = torch.argmax(response, dim=0)
    real_dtype = real_dtype_of(virtual_ant)
    return _finish_direction_cosines(u_x[best].to(real_dtype), u_z[best].to(real_dtype))


def _collinear(offsets) -> bool:
    origin = offsets[0]
    vectors = [tuple(b - a for a, b in zip(origin, row, strict=True)) for row in offsets[1:]]
    vectors = [v for v in vectors if max(abs(c) for c in v) > _LATTICE_TOLERANCE]
    if not vectors:
        return True
    base = vectors[0]
    for v in vectors[1:]:
        cross = (base[1] * v[2] - base[2] * v[1], base[2] * v[0] - base[0] * v[2], base[0] * v[1] - base[1] * v[0])
        if max(abs(c) for c in cross) > _LATTICE_TOLERANCE:
            return False
    return True


#: The angle routes, by name. Route selection is EXPLICIT - the caller picks -
#: because a dispatch on the layout is how a change of front end silently
#: changes the estimator. Every route checks the layout it needs and refuses
#: one it cannot read rather than returning a wrong angle.
AOA_ROUTES = {"phase_comparison": phase_comparison_aoa, "fft2": fft2_aoa, "scan": scan_aoa}


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
    return manifold.reshape(int(elevation_rad.shape[0]), int(azimuth_rad.shape[0]), rows * columns).to(dtype)


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
        raise ValueError(f"angle_data is [bins, rows, columns, snapshots]; got shape {tuple(angle_data.shape)}")
    bins, rows, columns, snapshots = (int(size) for size in angle_data.shape)
    smoothing = int(spatial_smooth)
    if smoothing < 0 or smoothing >= min(rows, columns):
        raise ValueError(
            f"spatial_smooth={spatial_smooth} must be in [0, min(rows, columns)) for a {rows} x {columns} array"
        )
    effective_rows = rows - smoothing
    effective_columns = columns - smoothing
    elements = effective_rows * effective_columns
    if not 0 < int(num_signals) < elements:
        raise ValueError(
            f"num_signals={num_signals} must leave a non-empty noise subspace of the {elements} effective elements"
        )

    # [B, jj, kk, rows_eff, cols_eff, T] in the same sub-aperture order the
    # (L + 1) ** 2 list comprehension produced: row shift major.
    windows = angle_data.unfold(1, effective_rows, 1).unfold(2, effective_columns, 1)
    sub_apertures = windows.permute(0, 1, 2, 4, 5, 3).reshape(bins, (smoothing + 1) ** 2, elements, snapshots)
    covariance = torch.einsum("bijk,bilk->bjlk", sub_apertures, sub_apertures.conj()).sum(dim=-1) / (
        snapshots * (smoothing + 1) ** 2
    )

    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    _, order = torch.topk(eigenvalues, k=eigenvalues.size(-1), largest=True, sorted=True)
    noise = torch.gather(
        eigenvectors, 2, order[:, int(num_signals) :].unsqueeze(1).expand(-1, eigenvectors.size(1), -1)
    ).to(torch.complex64)

    steering = upa_steering(
        array, rows=effective_rows, columns=effective_columns, elevation_rad=elevation_rad, azimuth_rad=azimuth_rad
    ).to(device=angle_data.device)
    # ``a^H P_n a``, with the conjugate on the LEFT. Conjugating on the right
    # evaluates the form at the CONJUGATE steering vector and peaks at the
    # mirror image of the true angle.
    projector = torch.matmul(noise, noise.transpose(-1, -2).conj())
    quadratic = torch.matmul(torch.einsum("ijk,akl->aijl", steering.conj(), projector), steering.transpose(-1, -2))
    return torch.reciprocal(quadratic.diagonal(dim1=-2, dim2=-1)).reshape(
        bins, int(elevation_rad.shape[0]), int(azimuth_rad.shape[0])
    )


def music_image(
    profile: RangeProfile,
    array: ArrayGeometry,
    *,
    elevation_rad: torch.Tensor,
    azimuth_rad: torch.Tensor,
    range_bins: torch.Tensor,
    num_signals: int = 7,
    spatial_smooth: int = 3,
    num_snapshots: int = 8,
) -> torch.Tensor:
    """A ``RangeProfile`` -> ``[elevation, azimuth, bins]`` MUSIC image.

    Takes a :class:`~witwin.radar.processing.range_doppler.RangeProfile` from
    the one range owner rather than transforming the cube itself, so the image
    and the point cloud are formed on the same range grid by construction.

    ``range_bins`` is required: an auto-detected peak is a modelling choice
    made silently, and the caller already has the range axis.
    """

    if not isinstance(profile, RangeProfile):
        raise TypeError(
            "music_image consumes a RangeProfile, so the range transform has "
            f"exactly one owner; got {type(profile).__name__}"
        )
    data = profile.data
    if data.dim() != 4:
        raise ValueError(f"the profile must be [tx, rx, slow_time, range] for an image; got shape {tuple(data.shape)}")
    selected = range_bins.to(device=data.device, dtype=torch.int64)
    # [bins, tx, rx, snapshots]
    angle_data = data[:, :, :num_snapshots, :].index_select(-1, selected).permute(3, 0, 1, 2)
    image = music_spectrum(
        angle_data.contiguous(),
        array,
        elevation_rad=elevation_rad,
        azimuth_rad=azimuth_rad,
        num_signals=num_signals,
        spatial_smooth=spatial_smooth,
    )
    return image.permute(1, 2, 0)


def beam_cube(rd: RangeDopplerMap, steering: torch.Tensor, *, directions: torch.Tensor) -> BeamCube:
    """``[*pair, D, R]`` and ``[P, *beam]`` -> ``BeamCube[*beam, D, R]``.

    Computes exactly ``y[b] = sum_p conj(w[p, b]) x[p]``, which is the
    definition of a beamformer output and is what conventional, MVDR and any
    future weight family all mean. It applies weights and knows no array
    geometry and no phasor; the weight owners carry those, so a weight family
    can be swapped here without this function learning what an element is.

    The pair axes of the map are flattened in the order the cube is published
    in, which is TX major - ``[TX, RX, ...]`` out of ``assemble_frame_cube`` -
    and that is the order :class:`ArrayGeometry` builds its element table in.
    A ``[TX, RX, D, R]`` map and a ``[P, D, R]`` one therefore give the same
    cube.

    ``directions`` is required and keyword only. A beam index means nothing
    without the grid it was steered over, a weight matrix does not carry one,
    and defaulting it to something would publish a cube whose angles are
    silently wrong rather than a call that does not compile.
    """

    if not isinstance(rd, RangeDopplerMap):
        raise TypeError(
            "beam_cube consumes a RangeDopplerMap, so that the range and "
            f"Doppler axes it publishes are already decided; got {type(rd).__name__}"
        )
    if not isinstance(steering, torch.Tensor) or not steering.is_complex():
        raise TypeError("steering must be a complex torch.Tensor")
    require_separated(rd.axes, "beam_cube")
    data = rd.data
    if data.dim() < 3:
        raise ValueError(
            f"a Range-Doppler map to be beamformed is [*pair, doppler, range]; got shape {tuple(data.shape)}"
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
    return BeamCube(data=formed.reshape(*beam_shape, doppler, ranges), axes=rd.axes, directions=directions)


@dataclass(frozen=True, slots=True, eq=False)
class RangeAngleMap:
    """``[A, R]`` real POWER over a beam grid and the range axis.

    ``directions`` is ``[A, 3]`` in the processing frame, ``azimuth_rad`` the
    ``[A]`` float64 azimuth grid it was built from (positive toward ``+x``,
    the radar's right) and ``elevation_rad`` the one elevation of the cut. The
    range axis is :attr:`ProcessingAxes.range_m`.

    The power is ``sum_D |y[a, D, R]|^2`` over the Doppler bins of the beam
    cube, which by Parseval with the ``1 / D`` Doppler normalisation is the
    mean slow-time power of the beamformed range profile (exactly so for a
    rectangular slow-time window; a taper weights the slow-time samples by
    ``w^2 / mean(w^2)``). With unit-response beam weights, a return on a beam
    and on a range bin reads ``|C|^2`` times the squared coherent gains.
    Differentiable in the map.
    """

    __repr__ = compact_repr

    data: torch.Tensor
    axes: ProcessingAxes
    directions: torch.Tensor
    azimuth_rad: torch.Tensor
    elevation_rad: float

    def __post_init__(self) -> None:
        if self.data.is_complex() or self.data.dim() != 2:
            raise ValueError(
                f"a range-angle map is real [angle, range]; got {self.data.dtype} {tuple(self.data.shape)}"
            )
        if int(self.data.shape[1]) != int(self.axes.range_bin_count):
            raise ValueError(
                f"the map has {int(self.data.shape[1])} range bins but its axes record publishes "
                f"{int(self.axes.range_bin_count)}"
            )
        if tuple(self.azimuth_rad.shape) != (int(self.data.shape[0]),):
            raise ValueError(f"azimuth_rad must be [{int(self.data.shape[0])}], got {tuple(self.azimuth_rad.shape)}")

    @property
    def range_axis(self) -> torch.Tensor:
        return self.axes.range_m


def azimuth_directions(azimuth_rad: torch.Tensor, elevation_rad: float = 0.0) -> torch.Tensor:
    """``[A]`` azimuths at one elevation -> ``[A, 3]`` processing-frame unit vectors.

    ``(sin az cos el, cos az cos el, sin el)``: azimuth from boresight toward
    the radar's right, elevation from the ``x``-``y`` plane toward up.
    """

    if not isinstance(azimuth_rad, torch.Tensor) or azimuth_rad.dim() != 1:
        raise ValueError("azimuth_rad must be a one-dimensional tensor of angles in radians")
    azimuth = azimuth_rad.to(torch.float64)
    cos_el = math.cos(float(elevation_rad))
    elevation = torch.full_like(azimuth, math.sin(float(elevation_rad)))
    return torch.stack((torch.sin(azimuth) * cos_el, torch.cos(azimuth) * cos_el, elevation), dim=1)


def range_angle_map(
    rd: RangeDopplerMap, array: ArrayGeometry, *, azimuth_rad: torch.Tensor, elevation_rad: float = 0.0
) -> RangeAngleMap:
    """``RangeDopplerMap[*pair, D, R]`` -> ``RangeAngleMap[A, R]`` by conventional beamforming.

    The beams are :func:`conventional_steering` of ``array`` toward
    :func:`azimuth_directions`, formed by :func:`beam_cube`, and their power is
    summed over Doppler. One weight owner, one beamformer; this function only
    chooses the grid and the reduction.
    """

    array = _require_array(array)
    directions = azimuth_directions(azimuth_rad, elevation_rad).to(array.device)
    weights = conventional_steering(array, directions, dtype=rd.data.dtype)
    beams = beam_cube(rd, weights, directions=directions)
    power = (beams.data.real.square() + beams.data.imag.square()).sum(dim=-2)
    return RangeAngleMap(
        data=power,
        axes=rd.axes,
        directions=directions,
        azimuth_rad=azimuth_rad.to(device=directions.device, dtype=torch.float64),
        elevation_rad=float(elevation_rad),
    )
