"""Angle of arrival: TDM compensation, two FFT routes, and MUSIC.

Every entry here takes an
:class:`~witwin.radar.processing.beamforming.ArrayGeometry` and a
:class:`~witwin.radar.processing.axes.ProcessingAxes` instead of a legacy
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
"""

from __future__ import annotations

import math

import torch

from .beamforming import ArrayGeometry


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
    :data:`~witwin.radar.processing.contracts.PROCESSING_DOPPLER_CONVENTION`
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
    :class:`~witwin.radar.processing.contracts.RangeProfile` from the one range
    owner instead, so the image and the point cloud are formed on the same range
    grid by construction.
    """

    from .contracts import RangeProfile

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


__all__ = [
    "AOA_ROUTES",
    "DIRECTION_COSINE_ROWS",
    "fft2_aoa",
    "music_image",
    "music_spectrum",
    "phase_comparison_aoa",
    "tdm_compensate",
    "upa_steering",
]
