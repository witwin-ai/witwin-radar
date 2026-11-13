"""Calibration against a known scene: channel gains, range bias, background.

The oracle is an array with injected errors: a per-channel complex gain, a
round-trip delay bias, a leakage tone and a static return, all written into
cubes from the waveform's closed form (the helpers of ``test_imaging``). The
calibrated array must read angles exactly as the ideal array reads them, the
delay bias must come back to a small fraction of a range bin, and the
background must cancel to rounding.
"""

from __future__ import annotations

import math

import pytest
import torch

from processing.test_imaging import PAD, _fmcw_cube, _frame, _profile, _radar
from witwin.radar.processing import (
    Aperture,
    ArrayCalibration,
    ArrayGeometry,
    ProcessingCube,
    RangeCalibration,
    average_background,
    backproject,
    estimate_array_calibration,
    estimate_range_calibration,
    range_doppler_map,
    scan_aoa,
    subtract_background,
)
from witwin.radar.synthesis.assembly import SPEED_OF_LIGHT_M_PER_S as C0

SCAN = 512


def _errors(num_tx, num_rx, seed=0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    magnitude = 0.6 + 0.8 * torch.rand((num_tx, num_rx), generator=generator, dtype=torch.float64)
    phase = (2 * torch.rand((num_tx, num_rx), generator=generator, dtype=torch.float64) - 1) * math.pi
    return torch.polar(magnitude, phase)


def _snapshot(profile, target_range_m: float) -> torch.Tensor:
    """``[V, 1]`` TX-major snapshot at the target's range bin, averaged over slow time."""

    k = round((target_range_m - profile.axes.range_origin_m) / profile.axes.range_bin_m)
    return profile.data[..., k].mean(dim=-1).reshape(-1, 1)


def _azimuth(profile, target_range_m: float) -> float:
    array = ArrayGeometry.from_axes(profile.axes)
    return float(scan_aoa(_snapshot(profile, target_range_m), array, fft_size=SCAN)[0, 0])


def test_an_array_with_channel_errors_is_calibrated_back_to_the_ideal_arrays_angles():
    radar = _radar(2, 4, chirps=8)
    right, forward = _frame(radar)
    errors = _errors(2, 4)
    reflector = 1.5 * forward + 0.2 * right
    aperture = Aperture.from_radar(radar)
    calibration = estimate_array_calibration(
        _profile(_fmcw_cube(radar, [(reflector, 1.0)], gains=errors)), aperture, reflector
    )
    assert calibration.gains.shape == (2, 4)
    assert float(calibration.gains.abs().mean()) == pytest.approx(1.0, abs=1e-12)
    assert abs(float(torch.angle(calibration.gains[0, 0]))) <= 1e-12
    ratio = calibration.gains / errors
    # Up to one common complex factor the estimate is the injected error; what is left is the
    # per-channel linear-interpolation loss of the reflector's fractional bin.
    ratio = ratio / ratio.mean()
    assert float(torch.angle(ratio).abs().max()) <= 5e-3
    assert float((ratio.abs() - 1).abs().max()) <= 0.02

    for u in (-0.45, -0.1, 0.27, 0.6):
        target = 2.2 * (math.sqrt(1 - u * u) * forward + u * right)
        ideal = _azimuth(_profile(_fmcw_cube(radar, [(target, 1.0)])), 2.2)
        broken = _profile(_fmcw_cube(radar, [(target, 1.0)], gains=errors))
        # The bin read off a padded profile carries a small element-dependent phase ramp; the
        # oracle is that calibration restores the ideal array's reading, not the scan grid.
        assert ideal == pytest.approx(u, abs=0.02)
        assert _azimuth(calibration.apply(broken), 2.2) == ideal
        assert _azimuth(broken, 2.2) != ideal
    # The same correction applies to a cube and to a Range-Doppler map.
    cube = _fmcw_cube(radar, [(reflector, 1.0)], gains=errors)
    fixed = calibration.apply(cube)
    assert isinstance(fixed, ProcessingCube)
    torch.testing.assert_close(fixed.data, cube.data / calibration.gains.reshape(2, 4, 1, 1))
    assert calibration.apply(range_doppler_map(_profile(cube))).data.shape[-4:-2] == (2, 4)


def test_a_round_trip_delay_bias_is_measured_and_moves_the_range_axis():
    radar = _radar(1, 4, chirps=8)
    right, forward = _frame(radar)
    reflector = 1.7 * forward - 0.1 * right
    bias_s = 0.25e-9  # 3.75 cm of range, close to one resolution cell
    aperture = Aperture.from_radar(radar)
    profile = _profile(_fmcw_cube(radar, [(reflector, 1.0)], delay_bias_s=bias_s))
    calibration = estimate_range_calibration(profile, aperture, reflector, search_m=0.15)
    native_bin = profile.axes.range_bin_m * PAD
    assert calibration.delay_s == pytest.approx(bias_s, abs=0.01 * 2 * native_bin / C0)
    assert calibration.range_bias_m == pytest.approx(C0 * bias_s / 2, rel=0.01)

    corrected = calibration.apply(profile)
    assert corrected.data is profile.data
    assert corrected.axes.range_origin_m == pytest.approx(-calibration.range_bias_m)
    torch.testing.assert_close(corrected.axes.range_m, profile.axes.range_m - calibration.range_bias_m)
    point = reflector.reshape(1, 3)
    before = float(backproject(profile, aperture, point).data.abs())
    after = float(backproject(corrected, aperture, point).data.abs())
    assert before < 0.5 and after >= 0.97, (before, after)
    with pytest.raises(RuntimeError, match="not differentiable"):
        estimate_range_calibration(
            type(profile)(
                data=profile.data.clone().requires_grad_(True),
                axes=profile.axes,
                window=profile.window,
                window_coherent_gain=profile.window_coherent_gain,
            ),
            aperture,
            reflector,
            search_m=0.15,
        )
    with pytest.raises(ValueError, match="runs off the range axis"):
        estimate_range_calibration(profile, aperture, reflector, search_m=100.0)
    assert RangeCalibration(delay_s=0.0).apply(profile).axes.range_origin_m == 0.0


def test_leakage_and_static_clutter_cancel_against_a_reference_frame():
    radar = _radar(1, 2, chirps=8)
    right, forward = _frame(radar)
    leakage = (0.05 * forward, 40.0)  # a strong near-range return that never left the board
    clutter = (2.0 * forward + 0.3 * right, 2.0)
    target = (1.2 * forward, 0.1)
    reference = _fmcw_cube(radar, [leakage, clutter])
    frame = _fmcw_cube(radar, [leakage, clutter, target])
    alone = _fmcw_cube(radar, [target])
    cleaned = subtract_background(frame, reference)
    torch.testing.assert_close(cleaned.data, alone.data, rtol=0, atol=1e-10)
    # The same on range profiles, with a leading batch of two frames, against an averaged reference.
    profiles = _profile(frame)
    batch = type(profiles)(
        data=torch.stack((profiles.data, profiles.data)),
        axes=profiles.axes,
        window=profiles.window,
        window_coherent_gain=profiles.window_coherent_gain,
    )
    background = average_background([_profile(reference), _profile(reference)])
    torch.testing.assert_close(subtract_background(batch, background).data[1], _profile(alone).data, atol=1e-10, rtol=0)
    with pytest.raises(ValueError, match="different grid"):
        subtract_background(_profile(frame), range_profile_unpadded(reference))
    with pytest.raises(TypeError, match="background is a"):
        subtract_background(frame, _profile(reference))


def range_profile_unpadded(cube):
    from witwin.radar.processing import range_profile

    return range_profile(cube)


def test_the_calibration_records_refuse_what_they_cannot_apply():
    with pytest.raises(ValueError, match="reference channel"):
        ArrayCalibration(gains=torch.ones((2, 4), dtype=torch.complex128), reference=(2, 0))
    calibration = ArrayCalibration(gains=torch.ones((2, 4), dtype=torch.complex128), reference=(0, 0))
    radar = _radar(1, 4, chirps=4)
    _, forward = _frame(radar)
    with pytest.raises(ValueError, match="channel axes"):
        calibration.apply(_fmcw_cube(radar, [(forward, 1.0)]))
    with pytest.raises(TypeError, match="ProcessingCube, RangeProfile or RangeDopplerMap"):
        calibration.apply(torch.ones(3))
    with pytest.raises(ValueError, match="finite"):
        RangeCalibration(delay_s=float("nan"))


# ---------------------------------------------------------------------------
# GPU, end to end: an array with port errors and transmit leakage, simulated.
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_a_simulated_array_with_port_errors_and_leakage_is_calibrated_to_the_ideal_angles():
    from conftest import PROCESSING_CONFIG, empty_world, scene_radar

    from witwin.radar import Coupling, Leakage, PointTargets
    from witwin.radar.processing import range_profile

    errors = _errors(2, 4, seed=3)
    # Separable port errors: the transmit and receive coupling matrices are diagonal.
    tx_gain = errors[:, 0] / errors[0, 0].abs()
    rx_gain = errors[0, :] / errors[0, 0]
    diagonal = (
        tuple(tuple(complex(tx_gain[i]) if i == j else 0j for j in range(2)) for i in range(2)),
        tuple(tuple(complex(rx_gain[i]) if i == j else 0j for j in range(4)) for i in range(4)),
    )
    config = {
        **PROCESSING_CONFIG,
        "num_tx": 2,
        "num_rx": 4,
        "tx_loc": [[0, 0, 0], [4, 0, 0]],
        "rx_loc": [[r, 0, 0] for r in range(4)],
        "adc_samples": 256,
        "chirp_per_frame": 8,
        "output_domain": "beat",
    }
    ideal = scene_radar(config)
    broken = ideal.replace(
        coupling=Coupling(tx=diagonal[0], rx=diagonal[1]), leakage=Leakage(isolation=25.0, delay=0.4e-9)
    )
    right, forward = _frame(ideal)
    reflector = 1.5 * forward + 0.25 * right
    u = -0.3
    target = 2.4 * (math.sqrt(1 - u * u) * forward + u * right)

    def frame(radar, positions):
        stacked = torch.stack(positions).to(torch.float32).cuda()
        result = radar.simulate(
            empty_world(), PointTargets(positions=stacked, rcs=1.0), times=(0.0,), los=True, reflections=0
        )
        return result.frame(0).processing_cube()

    def profile(cube):
        return range_profile(cube, window="hann", range_bins=PAD * 256)

    aperture = Aperture.from_radar(ideal)
    reference = frame(broken, [reflector])
    calibration = estimate_array_calibration(profile(reference), aperture, reflector)
    ratio = calibration.gains.cpu() / (tx_gain.reshape(2, 1) * rx_gain.reshape(1, 4))
    ratio = ratio / ratio.mean()
    assert float(torch.angle(ratio).abs().max()) <= 0.02, ratio
    cleaned = subtract_background(frame(broken, [reflector, target]), reference)
    # Leakage and the reflector are gone: what is left at the reflector's range is below -40 dB.
    residual = element_level(profile(cleaned), aperture, reflector)
    assert residual <= 1e-2 * element_level(profile(reference), aperture, reflector)
    calibrated = calibration.apply(profile(cleaned))
    expected = _azimuth_device(profile(frame(ideal, [target])), 2.4)
    assert expected == pytest.approx(u, abs=0.02)
    assert _azimuth_device(calibrated, 2.4) == expected
    assert _azimuth_device(profile(cleaned), 2.4) != expected


def element_level(profile, aperture, point) -> float:
    from witwin.radar.processing import element_responses

    return float(element_responses(profile, aperture, point).abs().mean())


def _azimuth_device(profile, target_range_m):
    snapshot = _snapshot(profile, target_range_m)
    return float(scan_aoa(snapshot, ArrayGeometry.from_axes(profile.axes), fft_size=SCAN)[0, 0])
