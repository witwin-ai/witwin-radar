"""Migration adapters: every public ``sigproc`` name, on the new facade.

``witwin.radar.sigproc`` keeps its whole public surface. What it no longer keeps
is a second implementation of any of it. Every name in ``sigproc/__init__.py``'s
``__all__`` is re-exported from here, each one a thin wrapper that builds a
:class:`~witwin.radar.processing.axes.ProcessingAxes` and an
:class:`~witwin.radar.processing.beamforming.ArrayGeometry` from a legacy
``Radar`` and calls the new owner. The wrappers live INSIDE the processing
package rather than in ``sigproc`` for one reason: after the cutover, every
production ``torch.fft`` and every detector, angle estimator and beamformer in
the radar tree lives under ``witwin/radar/processing/``, and a static scan can
say so.

**Behaviour is preserved, deliberately and including the parts that are wrong.**
These names have callers - the validation suite, ``Timeline.generate_rd``, user
scripts - and a migration adapter that quietly improves the answer is a
migration adapter that cannot be trusted. Three legacy conventions are therefore
reproduced exactly here and are NOT what a new caller should use:

* the legacy transforms window with a SYMMETRIC Hamming and are unnormalised,
  where the facade's stages use a periodic window and are amplitude normalised;
* the legacy Doppler axis has NO phasor reconciliation, so its positive
  velocities are RECEDING targets on an FMCW cube, the opposite of
  :data:`~witwin.radar.processing.contracts.PROCESSING_DOPPLER_CONVENTION`. The
  legacy TDM compensation was correct for exactly that convention, which is why
  it survives translation: the sign that reaches
  :func:`~witwin.radar.processing.aoa.tdm_compensate` is negated here, once, at
  the boundary between the two conventions;
* ``process_rd_tensor`` removes the fast-time mean unconditionally and
  undocumented. The facade makes that an explicit flag defaulting to off; the
  adapter keeps it on, because turning it off would change the map every
  existing caller plots.

One deliberate correction, recorded rather than smuggled: when the CFAR point
cloud thinned its detections to ``max_points``, the legacy code reordered the
peak list by energy while reading energies and angles in mask order, so the
range and velocity of a point no longer belonged to its own angle. The thinning
here happens on the MASK, before the row list exists, so every column of a point
comes from one cell.
"""

from __future__ import annotations

import math
import warnings

import torch

from ..synthesis.contracts import BEAT_PHASOR
from .aoa import fft2_aoa, music_spectrum, phase_comparison_aoa, tdm_compensate
from .axes import ProcessingAxes
from .beamforming import ArrayGeometry
from .cfar import ca_cfar, ca_cfar_fast, os_cfar
from .contracts import FAST_TIME_NAMES, SLOW_TIME_NAMES, DetectorType
from .microdoppler import (  # noqa: F401  (re-exported for the sigproc shim)
    dominant_frequencies_hz,
    doppler_frequencies_hz,
    microdoppler_spectrogram,
    slow_time_spectrum,
)
from .pointcloud import PointCloud
from .primitives import remove_mean, taper
from .tracking import DetectionFrame


#: The window every legacy ``sigproc`` transform applied:
#: ``torch.hamming_window(N, periodic=False)``. Named so that the one place the
#: legacy convention differs from the facade's is a constant and not a habit.
LEGACY_WINDOW = "hamming_symmetric"

#: The decibel floor the legacy point cloud used, and the value it wrote into a
#: range-gated cell.
LEGACY_ENERGY_FLOOR = 1e-6
LEGACY_GATE_FLOOR_DB = -100.0

#: The bin indices the legacy range gate was written in, and the 128 x 256
#: configuration they silently assumed. Kept only to convert them into METRES.
LEGACY_RANGE_CUT_BINS = (25, 125)

#: The legacy angle estimators never reconciled the beat conjugation in SPACE:
#: they read a DFT peak off a conjugated cube and published its position as a
#: direction cosine, so a target at ``+u`` was reported at ``-u``. The facade
#: reconciles it from ``ArrayGeometry.phase_sign``; the adapters hand the
#: estimator an array declared in the CHANNEL convention so the legacy answer -
#: mirror image and all - is reproduced exactly.
LEGACY_UNRECONCILED_PHASE_SIGN = -1


def _deprecated(name: str, replacement: str) -> None:
    warnings.warn(
        f"witwin.radar.sigproc.{name} is a migration adapter over "
        f"{replacement}. It preserves the legacy behaviour, including the "
        "unreconciled Doppler sign and the symmetric window, and will be "
        "removed once its callers move.",
        DeprecationWarning,
        stacklevel=3,
    )


# ---------------------------------------------------------------------------
# The one legacy transform pair
# ---------------------------------------------------------------------------


def legacy_range_transform(
    data: torch.Tensor, *, dim: int = -1, window: bool = True
) -> torch.Tensor:
    """The unnormalised fast-time transform every legacy entry used.

    There were THREE of these - inside ``range_fft``, inside
    ``process_rd_tensor``, and inside ``MUSICImager.radar_image`` - with three
    different windowing choices and no shared owner. This is the one owner; the
    windowing choice is the argument that used to be the difference between
    them.
    """

    if window:
        data = taper(data, LEGACY_WINDOW, dim=dim)
    return torch.fft.fft(data, dim=dim)


def legacy_doppler_transform(
    data: torch.Tensor, *, dim: int = -2, window: bool = True
) -> torch.Tensor:
    """The unnormalised, ``fftshift``ed slow-time transform, with NO sign fix.

    The facade's :func:`~witwin.radar.processing.doppler.range_doppler` reverses
    the frequency index of a conjugated cube so that a positive bin is a closing
    target in every waveform. This does not, because the legacy velocity axis
    and the legacy TDM compensation are both written in the unreconciled
    convention and changing one of the three would break the other two.
    """

    if window:
        data = taper(data, LEGACY_WINDOW, dim=dim)
    return torch.fft.fftshift(torch.fft.fft(data, dim=dim), dim=dim)


# ---------------------------------------------------------------------------
# Legacy Radar -> the two new records
# ---------------------------------------------------------------------------


def axes_from_radar(radar, *, doppler_bins: int | None = None) -> ProcessingAxes:
    """Build the metadata record from a legacy FMCW ``Radar``.

    Everything comes off ``radar.axes`` - the Phase-6 record that exists exactly
    so ``sigproc`` stops reading raw configuration fields - plus the two array
    counts. ``FrameConfig``'s seven raw ``radar.config.*`` reads are gone: the
    only configuration this touches is how many transmitters, receivers,
    chirps and samples there are, which are shapes, not waveform parameters.

    FMCW only, and it says so. The legacy ``Radar`` cannot be constructed for
    any other waveform - ``RadarSystemConfig.axes()`` raises, and
    ``from_radar_config`` hard-codes FMCW - so an adapter that pretended
    otherwise would be describing something that cannot exist.
    """

    legacy = radar.axes
    config = radar.config
    device = legacy.ranges.device
    num_tx = int(config.num_tx)
    num_rx = int(config.num_rx)
    range_count = int(config.adc_samples)
    doppler_count = int(config.chirp_per_frame if doppler_bins is None else doppler_bins)
    range_bin = float(legacy.range_resolution)
    velocity_bin = float(legacy.doppler_resolution)
    spacing = float(legacy.element_spacing_m)

    range_m = (
        torch.arange(range_count, dtype=torch.float64, device=device) * range_bin
    )
    # The legacy bin-to-velocity map, verbatim: bin k is
    # ``(k - D // 2) * doppler_resolution``. That is numerically the shifted
    # ``fftfreq`` grid the facade publishes; what differs is the SIGN
    # convention it is read in, and that difference lives in the callers below.
    velocity_mps = (
        torch.arange(doppler_count, dtype=torch.float64, device=device)
        - doppler_count // 2
    ) * velocity_bin

    return ProcessingAxes(
        waveform="fmcw",
        fast_time_name=FAST_TIME_NAMES["fmcw"],
        slow_time_name=SLOW_TIME_NAMES["fmcw"],
        slow_time_period_s=float(legacy.chirp_period_s) * num_tx,
        range_bin_count=range_count,
        doppler_bin_count=doppler_count,
        range_bin_m=range_bin,
        range_origin_m=0.0,
        max_unambiguous_range_m=float(legacy.max_range),
        velocity_bin_mps=velocity_bin,
        max_unambiguous_speed_mps=float(legacy.max_doppler),
        wavelength_m=float(legacy.wavelength_m),
        reference_frequency_hz=float(config.fc),
        phasor=BEAT_PHASOR,
        doppler_sign=1,
        num_tx=num_tx,
        num_rx=num_rx,
        element_spacing_m=spacing,
        tx_loc_half_wavelength=tuple(
            tuple(float(value) for value in row) for row in radar.tx_loc / spacing
        ),
        rx_loc_half_wavelength=tuple(
            tuple(float(value) for value in row) for row in radar.rx_loc / spacing
        ),
        range_m=range_m.contiguous(),
        velocity_mps=velocity_mps.contiguous(),
    )


def array_from_radar(radar) -> ArrayGeometry:
    """Build the array record from a legacy ``Radar``.

    ``radar.tx_loc`` and ``radar.rx_loc`` are already in METRES - the
    configuration's half-wavelength grid times the element spacing - so they are
    divided back out here and the spacing is carried as data. Nothing assumes a
    half wavelength.
    """

    return ArrayGeometry.from_axes(axes_from_radar(radar))


# ---------------------------------------------------------------------------
# FrameConfig and PointCloudProcessConfig
# ---------------------------------------------------------------------------


class FrameConfig:
    """Derived frame parameters, now read off the two records.

    The seven raw ``radar.config.*`` reads are gone. What remains are the four
    SHAPES a frame has - transmitters, receivers, chirps, samples - which the
    axes record and the array record already publish, plus the legacy attribute
    names every existing caller spells.
    """

    def __init__(self, radar):
        axes = axes_from_radar(radar)
        array = ArrayGeometry.from_axes(axes)
        self.axes = axes
        self.array = array

        self.numTxAntennas = axes.num_tx
        self.numRxAntennas = axes.num_rx
        self.numLoopsPerFrame = axes.doppler_bin_count
        self.numADCSamples = axes.range_bin_count

        self.numChirpsPerFrame = self.numTxAntennas * self.numLoopsPerFrame
        self.numRangeBins = axes.range_bin_count
        self.numDopplerBins = axes.doppler_bin_count

        self.range_resolution = axes.range_bin_m
        self.doppler_resolution = axes.velocity_bin_mps
        self.tx_loc_hw = torch.tensor(
            axes.tx_loc_half_wavelength, dtype=torch.float64
        )


class PointCloudProcessConfig:
    """Configuration for point cloud extraction from radar frames."""

    def __init__(
        self,
        radar,
        static_clutter_removal: bool = False,
        energy_top_k: int = 128,
        range_cut: bool = False,
        output_velocity: bool = True,
        output_snr: bool = True,
        output_range: bool = True,
        output_in_meter: bool = True,
    ):
        self.frame_config = FrameConfig(radar)
        self.enable_static_clutter_removal = static_clutter_removal
        self.use_energy_top_k = energy_top_k > 0
        self.energy_top_k = energy_top_k
        self.range_cut = range_cut
        self.output_velocity = output_velocity
        self.output_snr = output_snr
        self.output_range = output_range
        self.output_in_meter = output_in_meter

        dim = 3
        if self.output_velocity:
            self.velocity_dim = dim
            dim += 1
        if self.output_snr:
            self.snr_dim = dim
            dim += 1
        if self.output_range:
            self.range_dim = dim
            dim += 1


# ---------------------------------------------------------------------------
# The DSP entries
# ---------------------------------------------------------------------------


def range_fft(reshaped_frame: torch.Tensor, frame_config) -> torch.Tensor:
    """Apply a Hamming-windowed FFT along the fast-time axis."""

    _deprecated("range_fft", "witwin.radar.processing.range_profile")
    return legacy_range_transform(reshaped_frame, dim=-1)


def clutter_removal(input_val: torch.Tensor, axis: int = 0) -> torch.Tensor:
    """Static clutter removal by subtracting the mean along one axis."""

    _deprecated(
        "clutter_removal", "witwin.radar.processing.range_profile(remove_dc=True)"
    )
    return remove_mean(input_val, dim=axis)


def doppler_fft(range_result: torch.Tensor, frame_config) -> torch.Tensor:
    """Apply a Hamming-windowed FFT along slow time with fftshift."""

    _deprecated("doppler_fft", "witwin.radar.processing.range_doppler")
    return legacy_doppler_transform(range_result, dim=2)


def naive_xyz(virtual_ant, num_tx=3, num_rx=4, fft_size=64, tx_loc_hw=None):
    """Estimate direction cosines from virtual antenna data.

    The ``num_tx > 4`` dispatch to the two-dimensional route survives HERE and
    only here. In the facade the route is named by the caller, because a change
    of front end that silently changes the estimator is a change of answer with
    no change of call.
    """

    _deprecated("naive_xyz", "witwin.radar.processing.aoa")
    if num_tx <= 2:
        raise AssertionError("Need > 2 TX antennas for 3D AoA estimation")
    array = _synthetic_array(num_tx, num_rx, tx_loc_hw)
    route = fft2_aoa if num_tx > 4 else phase_comparison_aoa
    cosines = route(virtual_ant, array, fft_size=fft_size)
    return cosines[0], cosines[1], _legacy_elevation(cosines[2], route)


def _legacy_elevation(z_vector: torch.Tensor, route) -> torch.Tensor:
    """The legacy elevation cosine pointed against the array's z axis.

    The phase-comparison route took the azimuth-over-elevation phase ratio, the
    reciprocal of the one the array geometry implies, so its ``z`` came out
    negated. Every legacy elevation assertion in this tree is written on an
    absolute value, which is the symptom. The facade takes the ratio the right
    way round; this puts the legacy sign back for the legacy name only.
    """

    return -z_vector if route is phase_comparison_aoa else z_vector


def _synthetic_array(num_tx: int, num_rx: int, tx_loc_hw) -> ArrayGeometry:
    """An ``ArrayGeometry`` for the loose-int legacy AoA call.

    The FFT estimators read only ``num_tx``, ``num_rx`` and - for the elevation
    correction - the declared transmitter offsets, so a canonical linear
    receiver grid is enough and the wavelength cancels. Stated rather than left
    implicit, because the alternative is a caller believing this describes a
    real array.
    """

    if tx_loc_hw is None:
        # The legacy default was an elevation offset of exactly 2.0 between
        # the third and first transmitter; a unit-step row reproduces it.
        transmitters = [[float(index), 0.0, 0.0] for index in range(num_tx)]
    else:
        transmitters = [[float(value) for value in row] for row in tx_loc_hw]
    receivers = [[float(index), 0.0, 0.0] for index in range(num_rx)]
    return ArrayGeometry.from_offsets(
        transmitters,
        receivers,
        element_spacing_m=0.5,
        wavelength_m=1.0,
        phase_sign=LEGACY_UNRECONCILED_PHASE_SIGN,
    )


def ca_cfar_2d(rd_map, guard_cells=(2, 3), training_cells=(4, 6), pfa: float = 1e-3):
    """Cell-Averaging CFAR on a 2D Range-Doppler magnitude map."""

    _deprecated("ca_cfar_2d", "witwin.radar.processing.ca_cfar")
    result = ca_cfar(
        rd_map, guard_cells=guard_cells, training_cells=training_cells, pfa=pfa
    )
    return result.mask, result.threshold


def ca_cfar_2d_fast(
    rd_map, guard_cells=(2, 3), training_cells=(4, 6), pfa: float = 1e-3
):
    """Vectorized CA-CFAR using pooled averages."""

    _deprecated("ca_cfar_2d_fast", "witwin.radar.processing.ca_cfar_fast")
    result = ca_cfar_fast(
        rd_map, guard_cells=guard_cells, training_cells=training_cells, pfa=pfa
    )
    return result.mask, result.threshold


def os_cfar_2d(
    rd_map,
    guard_cells=(2, 3),
    training_cells=(4, 6),
    rank_fraction: float = 0.75,
    pfa: float = 1e-3,
):
    """Ordered-Statistic CFAR on a 2D Range-Doppler map."""

    _deprecated("os_cfar_2d", "witwin.radar.processing.os_cfar")
    result = os_cfar(
        rd_map,
        guard_cells=guard_cells,
        training_cells=training_cells,
        rank_fraction=rank_fraction,
        pfa=pfa,
    )
    return result.mask, result.threshold


# ---------------------------------------------------------------------------
# The point-cloud pipeline: ONE body, the detector is an argument
# ---------------------------------------------------------------------------


def _legacy_cubes(frame: torch.Tensor, *, static_clutter_removal: bool):
    """``[TX, RX, C, S]`` -> the legacy Doppler cube, through the one owner."""

    range_result = legacy_range_transform(frame, dim=-1)
    if static_clutter_removal:
        range_result = remove_mean(range_result, dim=2)
    return legacy_doppler_transform(range_result, dim=2)


def _legacy_range_gate_db(energy_db: torch.Tensor, axes) -> torch.Tensor:
    """Apply the legacy range gate, expressed in METRES rather than in bins.

    The literal ``[:, :25] = -100`` and ``[:, 125:] = -100`` were a 256-bin
    configuration written into the source. The same two edges are computed here
    from :attr:`ProcessingAxes.range_bin_m`, so the gate stays where it is in
    the SCENE when the bin count changes.
    """

    low_bin, high_bin = LEGACY_RANGE_CUT_BINS
    low_m = low_bin * axes.range_bin_m
    high_m = high_bin * axes.range_bin_m
    inside = (axes.range_m >= low_m) & (axes.range_m < high_m)
    return torch.where(
        inside.reshape(1, -1).to(energy_db.device),
        energy_db,
        torch.full_like(energy_db, LEGACY_GATE_FLOOR_DB),
    )


def _energy_topk_mask(values: torch.Tensor, top_k: int) -> torch.Tensor:
    total_bins = values.numel()
    top_k = min(top_k, total_bins)
    if top_k <= 0:
        return torch.zeros_like(values, dtype=torch.bool)
    if top_k >= total_bins:
        return torch.ones_like(values, dtype=torch.bool)
    threshold = torch.topk(values.reshape(-1), top_k).values.min()
    return values >= threshold


def _legacy_point_cloud(
    radar,
    frame: torch.Tensor,
    *,
    detector: DetectorType,
    static_clutter_removal: bool,
    range_cut: bool,
    energy_top_k: int,
    guard_cells,
    training_cells,
    pfa: float,
    max_points: int,
    output_in_meter: bool,
) -> PointCloud:
    """The single legacy point-cloud body. The detector is an argument.

    ``frame2pointcloud`` and ``_process_pc_cfar_tensor`` were two copies of this
    that differed only in how the mask was made. The angle estimation, the TDM
    compensation and the packing are the new owners' - there is no processing
    arithmetic here that is not either a detector choice or the legacy sign
    convention.
    """

    axes = axes_from_radar(radar, doppler_bins=int(frame.shape[-2]))
    array = ArrayGeometry.from_axes(axes)
    doppler_result = _legacy_cubes(
        frame, static_clutter_removal=static_clutter_removal
    )
    pairs = axes.sensor_pair_count
    flat = doppler_result.reshape(pairs, *doppler_result.shape[-2:])
    combined = flat.sum(dim=0)
    energy_db = 20 * torch.log10(combined.abs() + LEGACY_ENERGY_FLOOR)

    if detector == DetectorType.TOPK:
        if range_cut:
            energy_db = _legacy_range_gate_db(energy_db, axes)
        mask = (
            _energy_topk_mask(energy_db, energy_top_k)
            if energy_top_k > 0
            else torch.zeros_like(energy_db, dtype=torch.bool)
        )
    else:
        detected = ca_cfar_fast(
            combined.abs(),
            guard_cells=guard_cells,
            training_cells=training_cells,
            pfa=pfa,
        )
        mask = detected.mask
        if max_points is not None:
            from .pointcloud import _keep_strongest

            mask = _keep_strongest(mask, energy_db, int(max_points))

    cells = torch.argwhere(mask)
    if int(cells.shape[0]) == 0:
        return PointCloud.empty(device=frame.device)
    doppler_index = cells[:, 0]
    range_index = cells[:, 1]

    r = range_index.to(torch.float64)
    v = (doppler_index - axes.doppler_bin_count // 2).to(torch.float64)
    if output_in_meter:
        r = r * axes.range_bin_m
        v = v * axes.velocity_bin_mps
    energy = energy_db[doppler_index, range_index].to(torch.float64)

    aoa_input = flat[:, doppler_index, range_index]
    # The boundary between the two Doppler conventions, crossed once. The legacy
    # axis is receding-positive on this conjugated cube, and tdm_compensate
    # takes the canonical closing-positive velocity.
    aoa_input = tdm_compensate(aoa_input, -v, array, axes)
    route = fft2_aoa if axes.num_tx > 4 else phase_comparison_aoa
    unreconciled = ArrayGeometry.from_offsets(
        axes.tx_loc_half_wavelength,
        axes.rx_loc_half_wavelength,
        element_spacing_m=axes.element_spacing_m,
        wavelength_m=axes.wavelength_m,
        phase_sign=LEGACY_UNRECONCILED_PHASE_SIGN,
        device=aoa_input.device,
    )
    cosines = route(aoa_input, unreconciled, fft_size=64).to(torch.float64)
    cosines = torch.stack(
        (cosines[0], cosines[1], _legacy_elevation(cosines[2], route)), dim=0
    )

    cloud = PointCloud(
        xyz=(cosines * r.reshape(1, -1)).transpose(0, 1).contiguous(),
        velocity_mps=v,
        energy=energy,
        range_m=r,
    )
    return cloud.select(cosines[1] != 0)


def frame2pointcloud(frame: torch.Tensor, cfg, radar=None) -> torch.Tensor:
    """Convert a radar frame tensor to a point cloud ``(6, N)``."""

    _deprecated("frame2pointcloud", "witwin.radar.processing.point_cloud")
    if radar is None:
        raise ValueError(
            "frame2pointcloud requires a radar instance so TDM-MIMO compensation "
            "is always applied."
        )
    cloud = _legacy_point_cloud(
        radar,
        frame,
        detector=DetectorType.TOPK,
        static_clutter_removal=cfg.enable_static_clutter_removal,
        range_cut=cfg.range_cut,
        energy_top_k=cfg.energy_top_k if cfg.use_energy_top_k else 0,
        guard_cells=(2, 4),
        training_cells=(4, 8),
        pfa=1e-3,
        max_points=None,
        output_in_meter=cfg.output_in_meter,
    )
    return cloud.as_columns().transpose(0, 1)


def process_pc_tensor(
    radar,
    frame: torch.Tensor,
    static_clutter_removal=True,
    positive_velocity_only=True,
    detector: DetectorType = "cfar",
    guard_cells=(2, 4),
    training_cells=(4, 8),
    pfa=1e-3,
    max_points=512,
    energy_top_k=128,
) -> torch.Tensor:
    """Radar frame -> filtered point cloud ``(N, 6)`` on the input device."""

    _deprecated("process_pc_tensor", "witwin.radar.processing.point_cloud")
    kind = DetectorType(detector)
    cloud = _legacy_point_cloud(
        radar,
        frame,
        detector=kind,
        static_clutter_removal=static_clutter_removal,
        range_cut=False,
        energy_top_k=energy_top_k,
        guard_cells=guard_cells,
        training_cells=training_cells,
        pfa=pfa,
        max_points=max_points,
        output_in_meter=True,
    )
    if positive_velocity_only and len(cloud) > 0:
        cloud = cloud.select(cloud.velocity_mps > 0)
    return cloud.as_columns()


def process_pc(
    radar,
    frame: torch.Tensor,
    static_clutter_removal=True,
    positive_velocity_only=True,
    detector: DetectorType = "cfar",
    guard_cells=(2, 4),
    training_cells=(4, 8),
    pfa=1e-3,
    max_points=512,
    energy_top_k=128,
):
    """Radar frame -> filtered point cloud ``(N, 6)`` as numpy."""

    return (
        process_pc_tensor(
            radar,
            frame,
            static_clutter_removal=static_clutter_removal,
            positive_velocity_only=positive_velocity_only,
            detector=detector,
            guard_cells=guard_cells,
            training_cells=training_cells,
            pfa=pfa,
            max_points=max_points,
            energy_top_k=energy_top_k,
        )
        .detach()
        .cpu()
        .numpy()
    )


def process_rd_tensor(
    radar,
    frame: torch.Tensor,
    tx: int = 0,
    rx: int = 0,
    *,
    static_clutter_removal: bool = False,
):
    """Compute a Range-Doppler map and keep all outputs on the input device."""

    _deprecated("process_rd_tensor", "witwin.radar.processing.range_doppler")
    data = frame[tx, rx]
    # The unconditional fast-time mean removal, preserved. The facade makes this
    # an explicit ``remove_dc`` flag that defaults to OFF, because it is a
    # clutter operation and a component-export test that silently got the
    # clutter removed would be comparing two different quantities.
    data = remove_mean(data, dim=-1)
    if static_clutter_removal:
        data = remove_mean(data, dim=-2)
    # BOTH windows before EITHER transform, which is the order the deleted body
    # used. Applying the slow-time taper to the fast-time SPECTRUM instead is
    # the same operation mathematically and a different float.
    data = taper(taper(data, LEGACY_WINDOW, dim=-1), LEGACY_WINDOW, dim=-2)
    rd_map = legacy_doppler_transform(
        legacy_range_transform(data, dim=-1, window=False), dim=-2, window=False
    )
    rd_mag = 20 * torch.log10(torch.abs(rd_map) + LEGACY_ENERGY_FLOOR)
    return rd_mag, rd_map, radar.axes.ranges, radar.axes.velocities


def process_rd(
    radar,
    frame: torch.Tensor,
    tx: int = 0,
    rx: int = 0,
    *,
    static_clutter_removal: bool = False,
):
    """Compute a Range-Doppler map from a MIMO frame, as numpy arrays."""

    rd_mag, rd_map, ranges, velocities = process_rd_tensor(
        radar, frame, tx=tx, rx=rx, static_clutter_removal=static_clutter_removal
    )
    return (
        rd_mag.detach().cpu().numpy(),
        rd_map.detach().cpu().numpy(),
        ranges.detach().cpu().numpy(),
        velocities.detach().cpu().numpy(),
    )


def reg_data(data, pc_size):
    """Regularize a point cloud to a fixed size by sampling or duplication.

    The ``numpy`` / ``np.random`` body is gone. This builds a
    :class:`~witwin.radar.processing.tracking.DetectionFrame` and calls the
    fixed-size batching helper the detection contract publishes, which does the
    same three cases in torch, on the input device, and with an explicit
    generator instead of the global ``numpy`` random state.
    """

    _deprecated("reg_data", "witwin.radar.processing.DetectionFrame.as_fixed_size")
    tensor = torch.as_tensor(data)
    if tensor.dim() != 2 or int(tensor.shape[1]) < 6:
        raise ValueError(
            "reg_data takes the [N, 6] point-cloud columns; got shape "
            f"{tuple(tensor.shape)}"
        )
    if int(tensor.shape[0]) == 0:
        return torch.zeros(
            (int(pc_size), int(tensor.shape[1])), dtype=torch.float32
        ).numpy()
    frame = DetectionFrame(
        time_s=0.0,
        xyz=tensor[:, :3].to(torch.float64),
        velocity_mps=tensor[:, 3].to(torch.float64),
        energy=tensor[:, 4].to(torch.float64),
        frame_index=0,
    )
    return frame.as_fixed_size(int(pc_size)).to(torch.float32).numpy()


# ---------------------------------------------------------------------------
# MUSIC
# ---------------------------------------------------------------------------


class MUSICImager:
    """2D MUSIC-based radar imager for Uniform Planar Arrays (UPA).

    The hard-coded ``spacing = 0.5`` is gone. ``array`` is an optional
    :class:`~witwin.radar.processing.beamforming.ArrayGeometry`; when it is not
    given, a half-wavelength array is BUILT and carried as data, which preserves
    the legacy answer while making the assumption visible and replaceable. The
    ``numpy`` angle grids are gone with it.
    """

    def __init__(
        self,
        num_tx=20,
        num_rx=20,
        num_signals=7,
        spatial_smooth=3,
        num_pixels=128,
        fov=math.pi / 2,
        num_chirps=8,
        array: ArrayGeometry | None = None,
    ):
        _deprecated("MUSICImager", "witwin.radar.processing.music_spectrum")
        self.M = num_tx
        self.N = num_rx
        self.num_signals = num_signals
        self.spatial_smooth = spatial_smooth
        self.num_pixels = num_pixels
        self.num_chirps = num_chirps
        self.v_angle = torch.linspace(fov / 2, -fov / 2, num_pixels)
        self.h_angle = torch.linspace(fov / 2, -fov / 2, num_pixels)
        self.array = array if array is not None else _half_wavelength_upa(num_tx, num_rx)

    def music_spectrum(self, angle_data):
        """``(B, M, N, T)`` -> ``(B, num_pixels, num_pixels)`` pseudo-spectrum."""

        return music_spectrum(
            angle_data,
            self.array,
            elevation_rad=self.v_angle.to(angle_data.device),
            azimuth_rad=self.h_angle.to(angle_data.device),
            num_signals=self.num_signals,
            spatial_smooth=self.spatial_smooth,
        )

    def radar_image(self, sig, range_bins=None):
        """``(TX, RX, chirps, samples)`` -> ``(H, W, bins)``.

        The third range transform is gone: this calls the one legacy transform
        owner, unwindowed, exactly as the original did.
        """

        range_profile = legacy_range_transform(sig, dim=3, window=False)
        if range_bins is None:
            _, peak = torch.max(torch.abs(range_profile[0, 0, 0, :]), dim=0)
            range_bins = torch.arange(peak - 4, peak + 1, device=sig.device)
        elif not isinstance(range_bins, torch.Tensor):
            range_bins = torch.as_tensor(range_bins, device=sig.device)
        else:
            range_bins = range_bins.to(device=sig.device)
        angle_data = range_profile[:, :, : self.num_chirps, range_bins].permute(
            3, 0, 1, 2
        )
        return self.music_spectrum(angle_data).permute(1, 2, 0)


def _half_wavelength_upa(rows: int, columns: int) -> ArrayGeometry:
    """The half-wavelength planar array ``MUSICImager`` used to assume."""

    return ArrayGeometry.from_offsets(
        [[0.0, 0.0, float(index)] for index in range(rows)],
        [[float(index), 0.0, 0.0] for index in range(columns)],
        element_spacing_m=0.5,
        wavelength_m=1.0,
        phase_sign=-1,
    )


__all__ = [
    "LEGACY_ENERGY_FLOOR",
    "LEGACY_GATE_FLOOR_DB",
    "LEGACY_RANGE_CUT_BINS",
    "LEGACY_WINDOW",
    "FrameConfig",
    "MUSICImager",
    "PointCloudProcessConfig",
    "array_from_radar",
    "axes_from_radar",
    "ca_cfar_2d",
    "ca_cfar_2d_fast",
    "clutter_removal",
    "dominant_frequencies_hz",
    "doppler_fft",
    "doppler_frequencies_hz",
    "frame2pointcloud",
    "legacy_doppler_transform",
    "legacy_range_transform",
    "microdoppler_spectrogram",
    "naive_xyz",
    "os_cfar_2d",
    "process_pc",
    "process_pc_tensor",
    "process_rd",
    "process_rd_tensor",
    "range_fft",
    "reg_data",
    "slow_time_spectrum",
]
