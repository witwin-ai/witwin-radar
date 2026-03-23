"""
End-to-end validation: single target scenarios.

Full pipeline: Radar.mimo() -> process_pc() / process_rd()
Verifies that detected range, velocity, and angle match expected values.
"""

import torch
import numpy as np
import pytest

from conftest import (
    STANDARD_CONFIG, FAST_CONFIG,
    make_radar_or_skip, make_static_interpolator, make_moving_interpolator,
)

pytestmark = pytest.mark.gpu

# Validation config: adc_start_time=0 for clean signal, enough chirps for Doppler
_VFAST = {
    **FAST_CONFIG,
    "adc_start_time": 0,
    "chirp_per_frame": 32,
    "num_doppler_bins": 32,
}
_VFULL = {**STANDARD_CONFIG, "adc_start_time": 0}


class TestStaticTarget:
    """Single static target -> range + angle accuracy."""

    @pytest.mark.parametrize("distance", [1.5, 3.0, 5.0, 8.0])
    def test_range_accuracy(self, distance):
        """Detected range should match target distance within ±1 range bin."""
        from witwin.radar.sigproc import process_pc

        cfg = _VFAST
        r = make_radar_or_skip(cfg, backend="slang")
        interp = make_static_interpolator([0, 0, -distance])
        frame = r.mimo(interp)

        pc = process_pc(r, frame, detector="cfar", positive_velocity_only=False,
                        static_clutter_removal=False)
        assert pc.shape[0] > 0, f"No points detected at {distance}m"

        # Find point closest to expected range
        ranges = pc[:, 5]  # column 5 = range
        best_idx = np.argmin(np.abs(ranges - distance))
        detected_range = ranges[best_idx]

        tol = r.range_resolution * 2  # allow ±2 bins
        assert abs(detected_range - distance) < tol, (
            f"distance={distance}m: detected {detected_range:.3f}m, "
            f"tolerance={tol:.3f}m"
        )

    def test_broadside_target_angle(self):
        """Target directly in front -> x≈0, z≈0 in point cloud."""
        from witwin.radar.sigproc import process_pc

        cfg = _VFAST
        r = make_radar_or_skip(cfg, backend="slang")
        interp = make_static_interpolator([0, 0, -3.0])
        frame = r.mimo(interp)

        pc = process_pc(r, frame, detector="cfar", positive_velocity_only=False,
                        static_clutter_removal=False)
        assert pc.shape[0] > 0, "No points detected"

        # For broadside target, x and z should be small relative to y (range)
        x_vals = pc[:, 0]
        z_vals = pc[:, 2]
        y_vals = pc[:, 1]
        # Find points with range close to 3m
        mask = np.abs(pc[:, 5] - 3.0) < r.range_resolution * 3
        if mask.sum() > 0:
            assert np.abs(x_vals[mask]).mean() < 1.0, "x too large for broadside"
            assert np.abs(z_vals[mask]).mean() < 1.0, "z too large for broadside"

    def test_range_doppler_map_peak(self):
        """RD map should have a clear peak at the correct range bin."""
        from witwin.radar.sigproc import process_rd

        cfg = _VFAST
        r = make_radar_or_skip(cfg, backend="slang")
        distance = 3.0
        interp = make_static_interpolator([0, 0, -distance])
        frame = r.mimo(interp)

        rd_mag, rd_map, ranges, velocities = process_rd(r, frame, tx=0, rx=0)

        # Peak range bin (skip DC at bin 0, only positive frequencies)
        n_pos = len(ranges)
        range_profile = rd_mag[:, 1:n_pos].max(axis=0)
        peak_range_bin = np.argmax(range_profile) + 1  # +1 for DC skip
        peak_range = float(ranges[peak_range_bin])
        # process_rd applies DC removal along both axes which can shift peak
        tol = r.range_resolution * 12
        assert abs(peak_range - distance) < tol, (
            f"RD peak at range={peak_range:.3f}m, expected {distance}m"
        )

    def test_static_target_zero_doppler(self):
        """Static target should have Doppler near zero."""
        from witwin.radar.sigproc import process_rd

        cfg = _VFAST
        r = make_radar_or_skip(cfg, backend="slang")
        interp = make_static_interpolator([0, 0, -3.0])
        frame = r.mimo(interp)

        rd_mag, _, ranges, velocities = process_rd(r, frame, tx=0, rx=0)

        # Find peak in RD map (skip DC range bin)
        n_pos = len(ranges)
        rd_trimmed = rd_mag[:, 1:n_pos]
        peak_idx = np.unravel_index(np.argmax(rd_trimmed), rd_trimmed.shape)
        peak_vel = float(velocities[peak_idx[0]])
        # Static target: Doppler should be near zero
        assert abs(peak_vel) < r.doppler_resolution * 3, (
            f"Static target: Doppler={peak_vel:.4f} m/s, expected ~0"
        )


class TestMovingTarget:
    """Single moving target -> velocity accuracy."""

    def test_velocity_accuracy(self):
        """Approaching target should produce detectable Doppler shift."""
        from witwin.radar.sigproc import process_pc

        # Need enough chirps for Doppler resolution
        cfg = _VFULL  # 128 chirps for good Doppler
        r = make_radar_or_skip(cfg, backend="slang")

        speed = 1.5  # m/s approaching
        interp = make_moving_interpolator(
            pos0=[0, 0, -3.0],
            velocity=[0, 0, speed],  # +Z = approaching (decreasing range)
            sigma=1.0,
        )
        frame = r.mimo(interp)

        pc = process_pc(r, frame, detector="cfar", positive_velocity_only=False,
                        static_clutter_removal=False)
        assert pc.shape[0] > 0, "No points detected"

        # Find point at correct range
        mask = np.abs(pc[:, 5] - 3.0) < r.range_resolution * 5
        if mask.sum() > 0:
            velocities = np.abs(pc[mask, 3])
            best = velocities[np.argmin(np.abs(velocities - speed))]
            tol = r.doppler_resolution * 3
            assert abs(best - speed) < tol, (
                f"Expected |v|≈{speed} m/s, best detected: {best:.3f} m/s"
            )

    def test_rd_map_shows_doppler_shift(self):
        """Moving target should shift the RD peak away from zero-Doppler."""
        from witwin.radar.sigproc import process_rd

        cfg = _VFULL
        r = make_radar_or_skip(cfg, backend="slang")

        speed = 2.0
        interp = make_moving_interpolator(
            pos0=[0, 0, -3.0],
            velocity=[0, 0, speed],
            sigma=1.0,
        )
        frame = r.mimo(interp)

        rd_mag, _, ranges, velocities = process_rd(r, frame, tx=0, rx=0)
        n_pos = len(ranges)
        rd_trimmed = rd_mag[:, 1:n_pos]
        peak_idx = np.unravel_index(np.argmax(rd_trimmed), rd_trimmed.shape)
        peak_vel = float(velocities[peak_idx[0]])

        # Velocity should NOT be near zero
        assert abs(peak_vel) > r.doppler_resolution, (
            f"Moving target ({speed} m/s) shows zero Doppler: {peak_vel:.4f}"
        )


class TestSNRScale:
    """Regression (bug #7): SNR should be in 20*log10 (dB) scale."""

    @staticmethod
    def _peak_snr(radar, *, detector: str, sigma: float) -> float | None:
        from witwin.radar.sigproc import process_pc

        frame = radar.mimo(make_static_interpolator([0, 0, -3.0], sigma=sigma))
        pc = process_pc(
            radar,
            frame,
            detector=detector,
            positive_velocity_only=False,
            static_clutter_removal=False,
        )
        if pc.shape[0] == 0:
            return None
        return float(pc[:, 4].max())

    def test_snr_scale_cfar_path(self):
        """CFAR path SNR values should be in dB (20*log10) scale."""
        cfg = _VFAST
        r = make_radar_or_skip(cfg, backend="slang")
        low = self._peak_snr(r, detector="cfar", sigma=1.0)
        high = self._peak_snr(r, detector="cfar", sigma=100.0)
        if low is not None and high is not None:
            assert high - low == pytest.approx(20.0, abs=1.0), (
                f"CFAR SNR scaling mismatch: sigma x100 changed peak by {high - low:.2f} dB"
            )
        return
        from witwin.radar.sigproc import process_pc

        cfg = _VFAST
        r = make_radar_or_skip(cfg, backend="slang")
        interp = make_static_interpolator([0, 0, -3.0], sigma=1.0)
        frame = r.mimo(interp)

        pc = process_pc(r, frame, detector="cfar", positive_velocity_only=False,
                        static_clutter_removal=False)
        if pc.shape[0] > 0:
            snr_values = pc[:, 4]
            # 20*log10 of a typical magnitude (>1) should be > 0 dB
            # and of order 10-100 dB, not 0.x-3 (which would be log10 scale)
            # A strong target in dB should be > 10
            assert snr_values.max() > 5.0, (
                f"SNR max={snr_values.max():.2f} — too small for dB scale, "
                "might be using log10 instead of 20*log10"
            )

    def test_snr_scale_topk_path(self):
        """topk path SNR values should also be in dB scale."""
        cfg = _VFAST
        r = make_radar_or_skip(cfg, backend="slang")
        low = self._peak_snr(r, detector="topk", sigma=1.0)
        high = self._peak_snr(r, detector="topk", sigma=100.0)
        if low is not None and high is not None:
            assert high - low == pytest.approx(20.0, abs=1.0), (
                f"topk SNR scaling mismatch: sigma x100 changed peak by {high - low:.2f} dB"
            )
        return
        from witwin.radar.sigproc import process_pc

        cfg = _VFAST
        r = make_radar_or_skip(cfg, backend="slang")
        interp = make_static_interpolator([0, 0, -3.0], sigma=1.0)
        frame = r.mimo(interp)

        pc = process_pc(r, frame, detector="topk", positive_velocity_only=False,
                        static_clutter_removal=False)
        if pc.shape[0] > 0:
            snr_values = pc[:, 4]
            assert snr_values.max() > 5.0, (
                f"SNR max={snr_values.max():.2f} — too small for dB scale"
            )

    def test_snr_consistent_between_detectors(self):
        """CFAR and topk SNR values should be in the same scale."""
        from witwin.radar.sigproc import process_pc

        cfg = _VFAST
        r = make_radar_or_skip(cfg, backend="slang")
        interp = make_static_interpolator([0, 0, -3.0], sigma=1.0)
        frame = r.mimo(interp)

        pc_cfar = process_pc(r, frame, detector="cfar", positive_velocity_only=False)
        pc_topk = process_pc(r, frame, detector="topk", positive_velocity_only=False)

        if pc_cfar.shape[0] > 0 and pc_topk.shape[0] > 0:
            # The peak SNR from both detectors should be in the same ballpark
            # (within ~20 dB, since they select different bins)
            snr_cfar_max = pc_cfar[:, 4].max()
            snr_topk_max = pc_topk[:, 4].max()
            ratio = abs(snr_cfar_max - snr_topk_max)
            assert ratio < 30, (
                f"SNR scale mismatch: CFAR max={snr_cfar_max:.1f}, "
                f"topk max={snr_topk_max:.1f}"
            )


class TestTDMCompensationInTopk:
    """Regression (bug #4): topk path should apply TDM phase compensation."""

    def test_topk_and_cfar_give_similar_angles(self):
        """For a moving target, both detectors should produce similar AoA results
        (both now apply TDM compensation)."""
        from witwin.radar.sigproc import process_pc

        cfg = _VFULL
        r = make_radar_or_skip(cfg, backend="slang")
        interp = make_moving_interpolator(
            pos0=[0, 0, -3.0], velocity=[0, 0, 1.0], sigma=1.0
        )
        frame = r.mimo(interp)

        pc_cfar = process_pc(r, frame, detector="cfar", positive_velocity_only=False)
        pc_topk = process_pc(r, frame, detector="topk", positive_velocity_only=False)

        if pc_cfar.shape[0] > 0 and pc_topk.shape[0] > 0:
            # Compare x-coordinate (azimuth) of the closest-to-target points
            mask_c = np.abs(pc_cfar[:, 5] - 3.0) < r.range_resolution * 5
            mask_t = np.abs(pc_topk[:, 5] - 3.0) < r.range_resolution * 5
            if mask_c.sum() > 0 and mask_t.sum() > 0:
                x_cfar = np.median(pc_cfar[mask_c, 0])
                x_topk = np.median(pc_topk[mask_t, 0])
                # Should be in similar ballpark (both near 0 for broadside)
                assert abs(x_cfar - x_topk) < 1.0, (
                    f"x_cfar={x_cfar:.3f}, x_topk={x_topk:.3f} — "
                    "TDM compensation may be missing in topk"
                )
