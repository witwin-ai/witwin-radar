"""The cutover's safety net: every public ``sigproc`` name, against a golden.

``tests/goldens/legacy_sigproc.pt`` was captured from the PRE-cutover tree - the
commit before the old internal paths were deleted - by replaying the inputs in
``tests/support/legacy_golden.py``. This file replays the same inputs through
the migration adapters and compares. Nothing here builds an expected value from
a formula: the reference is what the code used to do.

Tolerances, and why each one is what it is:

* BITWISE where the computation is genuinely unchanged - the two transforms, the
  mean subtraction, the three detectors, the four micro-Doppler entries, and the
  point-cloud pipelines. These were moved, not rewritten, and moving code that
  performs the same operations in the same order does not change a float.
* ``rtol = 1e-6`` where an OWNER changed in a way that re-associates. The only
  two are the MUSIC steering manifold (``torch.polar`` where the original wrote
  ``torch.exp(1j * ...)``) and the spatial smoothing, which is now built by
  ``unfold`` rather than by a sixteen-way ``stack``. The sub-aperture ORDER is
  preserved, so the sum is over the same terms in the same sequence; what moves
  is the last bit of the manifold.
* ONE deliberate behaviour change, asserted as a change rather than hidden:
  ``matched_filter`` no longer upcasts to ``complex128``. The golden is
  reproduced exactly when the caller asks for that dtype, and the default now
  follows the input.
"""

from __future__ import annotations

import pathlib
import warnings

import numpy as np
import pytest
import torch

from conftest import MockRadar
from support import legacy_golden as g
from witwin.radar.processing import ArrayGeometry, tdm_compensate
from witwin.radar.processing.adapters import axes_from_radar
from witwin.radar.sigproc import (
    FrameConfig,
    MUSICImager,
    PointCloudProcessConfig,
    ca_cfar_2d,
    ca_cfar_2d_fast,
    clutter_removal,
    dominant_frequencies_hz,
    doppler_fft,
    doppler_frequencies_hz,
    frame2pointcloud,
    microdoppler_spectrogram,
    naive_xyz,
    os_cfar_2d,
    process_pc,
    process_pc_tensor,
    process_rd,
    process_rd_tensor,
    range_fft,
    reg_data,
    slow_time_spectrum,
)


GOLDEN_PATH = pathlib.Path(__file__).resolve().parents[1] / "goldens" / "legacy_sigproc.pt"


@pytest.fixture(scope="module")
def golden() -> dict[str, torch.Tensor]:
    return torch.load(GOLDEN_PATH, weights_only=True)


@pytest.fixture(autouse=True)
def _allow_deprecation():
    """Every adapter warns by design; this file is measuring what it returns."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        yield


@pytest.fixture(scope="module")
def radar() -> MockRadar:
    return MockRadar(g.GOLDEN_CONFIG)


@pytest.fixture(scope="module")
def radar_2d() -> MockRadar:
    return MockRadar(g.GOLDEN_CONFIG_2D)


def _same(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    assert actual.dtype == expected.dtype, (actual.dtype, expected.dtype)
    assert torch.equal(actual, expected)


# ---------------------------------------------------------------------------
# The configuration record
# ---------------------------------------------------------------------------


def test_the_frame_config_publishes_the_same_numbers_without_the_raw_reads(
    golden, radar
):
    """The seven ``radar.config.*`` reads are gone and nothing downstream moved."""

    fc = FrameConfig(radar)
    counts = torch.tensor(
        [
            fc.numTxAntennas,
            fc.numRxAntennas,
            fc.numLoopsPerFrame,
            fc.numADCSamples,
            fc.numChirpsPerFrame,
            fc.numRangeBins,
            fc.numDopplerBins,
        ],
        dtype=torch.int64,
    )
    _same(counts, golden["frame_config"])
    scalars = torch.tensor(
        [fc.range_resolution, fc.doppler_resolution], dtype=torch.float64
    )
    _same(scalars, golden["frame_config_scalars"])
    torch.testing.assert_close(
        fc.tx_loc_hw.to(torch.float64),
        golden["frame_config_tx_loc_hw"],
        rtol=1e-12,
        atol=1e-12,
    )
    # And the two new records are reachable from it, which is what makes the
    # adapter a migration rather than a translation layer that has to stay.
    assert isinstance(fc.axes, type(axes_from_radar(radar)))
    assert isinstance(fc.array, ArrayGeometry)


# ---------------------------------------------------------------------------
# The transforms
# ---------------------------------------------------------------------------


def test_the_two_legacy_transforms_are_bitwise_unchanged(golden, radar):
    fc = FrameConfig(radar)
    frame = g.frame()
    ranged = range_fft(frame, fc)
    _same(ranged, golden["range_fft"])
    _same(clutter_removal(ranged, axis=2), golden["clutter_removal"])
    _same(doppler_fft(ranged, fc), golden["doppler_fft"])


# ---------------------------------------------------------------------------
# Angle of arrival
# ---------------------------------------------------------------------------


def test_both_aoa_routes_are_bitwise_unchanged(golden, radar, radar_2d):
    """Design R4: the 3 x 4 front end cannot reach the 2-D route at all.

    ``naive_xyz`` dispatches to ``fft2`` only when ``num_tx > 4``, so covering
    both routes needs two array configurations. Both are here and both are
    golden.
    """

    fc = FrameConfig(radar)
    x, y, z = naive_xyz(
        g.virtual_antenna(num_tx=3, num_rx=4),
        num_tx=3,
        num_rx=4,
        fft_size=64,
        tx_loc_hw=fc.tx_loc_hw,
    )
    _same(torch.stack((x, y, z), dim=0), golden["naive_xyz_phase_comparison"])

    fc2d = FrameConfig(radar_2d)
    x, y, z = naive_xyz(
        g.virtual_antenna(num_tx=6, num_rx=4),
        num_tx=6,
        num_rx=4,
        fft_size=64,
        tx_loc_hw=fc2d.tx_loc_hw,
    )
    _same(torch.stack((x, y, z), dim=0), golden["naive_xyz_fft2"])


def test_the_vectorized_tdm_compensation_is_bitwise_equal_to_the_deleted_loop(
    golden, radar
):
    """One broadcast multiply where there was a Python loop, same last bit.

    The velocity handed to :func:`tdm_compensate` is NEGATED, because the legacy
    Doppler axis is receding-positive on a conjugated FMCW cube and the facade's
    convention is closing-positive. That negation is the whole of the
    translation, and this is what proves it is the whole of it.
    """

    axes = axes_from_radar(radar)
    array = ArrayGeometry.from_axes(axes)
    compensated = tdm_compensate(
        g.virtual_antenna(num_tx=3, num_rx=4), -g.velocities(), array, axes
    )
    _same(compensated, golden["tdm_compensate"])


# ---------------------------------------------------------------------------
# CFAR
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,entry",
    [
        ("ca_cfar_2d", ca_cfar_2d),
        ("ca_cfar_2d_fast", ca_cfar_2d_fast),
        ("os_cfar_2d", os_cfar_2d),
    ],
)
def test_every_detector_is_bitwise_unchanged(golden, name, entry):
    mask, threshold = entry(
        g.rd_magnitude(), guard_cells=(2, 3), training_cells=(4, 6), pfa=1e-3
    )
    _same(mask, golden[f"{name}_mask"])
    _same(threshold, golden[f"{name}_threshold"])


# ---------------------------------------------------------------------------
# The point-cloud pipelines
# ---------------------------------------------------------------------------


def test_the_point_cloud_pipelines_are_bitwise_unchanged(golden, radar):
    frame = g.frame()
    cfg = PointCloudProcessConfig(radar, static_clutter_removal=False, energy_top_k=32)
    _same(frame2pointcloud(frame, cfg, radar=radar), golden["frame2pointcloud"])

    gated = PointCloudProcessConfig(
        radar, static_clutter_removal=True, energy_top_k=32, range_cut=True
    )
    _same(
        frame2pointcloud(frame, gated, radar=radar), golden["frame2pointcloud_gated"]
    )


def test_the_magic_range_gate_is_now_a_distance_and_still_lands_on_the_same_cells(
    golden, radar
):
    """Cutover item 6, asserted as an equivalence rather than as a rewrite.

    ``[:, :25] = -100`` and ``[:, 125:] = -100`` are gone from the source. The
    same two edges are computed from ``ProcessingAxes.range_bin_m``, so the gate
    is stated in METRES and follows the scene when the bin count changes - and
    on the configuration the constants were written for, it selects exactly the
    same cells, which is what the golden above proves.
    """

    from witwin.radar.processing.adapters import (
        LEGACY_RANGE_CUT_BINS,
        _legacy_range_gate_db,
    )

    axes = axes_from_radar(radar)
    energy = torch.zeros((4, axes.range_bin_count), dtype=torch.float64)
    gated = _legacy_range_gate_db(energy, axes)
    low, high = LEGACY_RANGE_CUT_BINS
    floored = (gated[0] == -100.0).nonzero().reshape(-1).tolist()
    expected = list(range(0, min(low, axes.range_bin_count)))
    expected += list(range(min(high, axes.range_bin_count), axes.range_bin_count))
    assert floored == expected


@pytest.mark.parametrize("detector", ["cfar", "topk"])
def test_process_pc_and_its_tensor_form_are_bitwise_unchanged(golden, radar, detector):
    frame = g.frame()
    _same(
        process_pc_tensor(
            radar,
            frame,
            detector=detector,
            positive_velocity_only=False,
            static_clutter_removal=False,
        ),
        golden[f"process_pc_tensor_{detector}"],
    )
    _same(
        process_pc_tensor(
            radar,
            frame,
            detector=detector,
            positive_velocity_only=True,
            static_clutter_removal=True,
        ),
        golden[f"process_pc_tensor_{detector}_positive"],
    )
    numpy_form = process_pc(
        radar,
        frame,
        detector=detector,
        positive_velocity_only=False,
        static_clutter_removal=False,
    )
    assert isinstance(numpy_form, np.ndarray)
    _same(torch.as_tensor(numpy_form), golden[f"process_pc_{detector}"])


@pytest.mark.parametrize("static_clutter_removal", [False, True])
def test_the_range_doppler_entries_are_bitwise_unchanged(
    golden, radar, static_clutter_removal
):
    frame = g.frame()
    mag, rd_map, ranges, velocities = process_rd_tensor(
        radar, frame, static_clutter_removal=static_clutter_removal
    )
    _same(mag, golden[f"process_rd_tensor_mag_{static_clutter_removal}"])
    _same(rd_map, golden[f"process_rd_tensor_map_{static_clutter_removal}"])
    _same(ranges, golden[f"process_rd_tensor_ranges_{static_clutter_removal}"])
    _same(velocities, golden[f"process_rd_tensor_velocities_{static_clutter_removal}"])


def test_process_rd_still_returns_numpy(golden, radar):
    mag, rd_map, ranges, velocities = process_rd(radar, g.frame())
    for value in (mag, rd_map, ranges, velocities):
        assert isinstance(value, np.ndarray)
    _same(torch.as_tensor(mag), golden["process_rd_mag"])


# ---------------------------------------------------------------------------
# MUSIC
# ---------------------------------------------------------------------------


def test_the_music_spectrum_and_image_agree_to_the_manifold_rewrite(golden):
    """``rtol = 1e-6``: the sixteen-way stack and ``torch.exp`` both moved.

    The sub-aperture ORDER is preserved exactly, so the smoothed covariance is a
    sum over the same terms in the same sequence. What changed is how the
    manifold's unit phasors are formed, and that moves the last bit.
    """

    imager = MUSICImager(
        num_tx=6, num_rx=6, num_signals=2, spatial_smooth=2, num_pixels=8, num_chirps=4
    )
    torch.testing.assert_close(
        imager.music_spectrum(g.angle_data()),
        golden["music_spectrum"],
        rtol=1e-6,
        atol=1e-8,
    )
    torch.testing.assert_close(
        imager.radar_image(g.music_frame(), range_bins=g.music_range_bins()),
        golden["music_image"],
        rtol=1e-6,
        atol=1e-8,
    )


def test_the_music_imager_no_longer_hides_a_half_wavelength_spacing():
    """Cutover item 7. The spacing is data, and a different array is expressible."""

    default = MUSICImager(num_tx=4, num_rx=4, num_signals=1, spatial_smooth=1,
                          num_pixels=4, num_chirps=2)
    assert default.array.spacing_wavelengths == 0.5

    quarter = ArrayGeometry.from_offsets(
        [[0.0, 0.0, float(i)] for i in range(4)],
        [[float(i), 0.0, 0.0] for i in range(4)],
        element_spacing_m=0.25,
        wavelength_m=1.0,
    )
    tuned = MUSICImager(
        num_tx=4,
        num_rx=4,
        num_signals=1,
        spatial_smooth=1,
        num_pixels=4,
        num_chirps=2,
        array=quarter,
    )
    assert tuned.array.spacing_wavelengths == 0.25
    data = g.angle_data(rows=4, columns=4, bins=1, snapshots=3)
    assert not torch.allclose(default.music_spectrum(data), tuned.music_spectrum(data))


# ---------------------------------------------------------------------------
# Micro-Doppler
# ---------------------------------------------------------------------------


def test_the_four_microdoppler_entries_are_bitwise_unchanged(golden):
    samples = g.slow_time()
    _same(slow_time_spectrum(samples), golden["slow_time_spectrum"])
    _same(doppler_frequencies_hz(64, 1.3e-4), golden["doppler_frequencies_hz"])
    times, frequencies, spectrum = microdoppler_spectrogram(
        samples, slot_period_s=1.3e-4, window_slots=16, hop_slots=8
    )
    _same(times, golden["microdoppler_times"])
    _same(frequencies, golden["microdoppler_frequencies"])
    _same(spectrum, golden["microdoppler_spectrum"])
    _same(
        dominant_frequencies_hz(spectrum, frequencies),
        golden["dominant_frequencies_hz"],
    )


# ---------------------------------------------------------------------------
# reg_data and the matched filter
# ---------------------------------------------------------------------------


def test_reg_data_keeps_its_shape_contract_without_numpy_random(golden):
    """The three cases survive; the global ``numpy`` random state does not.

    ``reg_data`` drew from ``np.random`` and could not be seeded from the call
    site, so its output was never assertable value for value. What IS assertable
    is the contract: the shape, the dtype, and that every returned row is one of
    the input rows.
    """

    columns = g.point_columns().numpy()
    batch = reg_data(columns, 16)
    assert isinstance(batch, np.ndarray)
    assert batch.dtype == np.float32
    _same(torch.tensor(list(batch.shape), dtype=torch.int64), golden["reg_data_shape"])

    rows = torch.as_tensor(batch).to(torch.float64)
    source = torch.as_tensor(columns).to(torch.float64)
    for index in range(rows.shape[0]):
        distance = (source - rows[index]).abs().max(dim=1).values
        assert float(distance.min()) < 1e-4

    # Every input row appears at least once when there is room for all of them.
    for index in range(source.shape[0]):
        distance = (rows - source[index]).abs().max(dim=1).values
        assert float(distance.min()) < 1e-4

    assert reg_data(np.zeros((0, 6), dtype=np.float32), 8).shape == (8, 6)


def test_the_matched_filter_upcast_is_now_an_explicit_argument(golden):
    """Cutover item 10, asserted in both directions."""

    from witwin.radar.processing.matched_filter import matched_filter
    from witwin.radar.synthesis.contracts import PulsedEchoSpec

    spec = PulsedEchoSpec(
        num_pulses=4,
        num_samples=128,
        sample_period_s=2e-8,
        pri_s=5e-5,
        range_gate_start_s=0.0,
        pulse_kind="lfm",
        pulse_width_s=6e-7,
        bandwidth_hz=2e7,
        reference_frequency_hz=77e9,
        max_expected_delay_rate=1e-7,
    )
    signal = g._complex((2, 128), seed=g.SEED + 9)
    assert signal.dtype == torch.complex64

    asked = matched_filter(signal, spec, dtype=torch.complex128)
    _same(asked, golden["matched_filter_complex128"])

    # The default now follows the input, which is the change item 10 asks for.
    followed = matched_filter(signal, spec)
    assert followed.dtype == torch.complex64
    torch.testing.assert_close(
        followed.to(torch.complex128), asked, rtol=2e-6, atol=1e-9
    )
