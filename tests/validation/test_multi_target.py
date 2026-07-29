"""End-to-end validation on the scene-driven route: multi-target scenarios.

The companion of ``test_single_target.py``; that module's docstring states the
route, the pose and the Doppler seam and is not repeated here.

One property of the route shapes every test below: the two-way join takes ONE
scatter response for the whole batch, so every site in a run has the SAME cross
section. The old interpolator handed a per-target sigma vector to the Dirichlet
solver; the new route has no per-row response, and rather than fabricating one
here the multi-target scenarios are all equal-strength. Where the old test used
a strength difference to make a target dominate, this one uses range or Doppler
separation instead, which is the property those tests were really about.
"""

import numpy as np
import pytest
import torch
from conftest import (
    FAST_CONFIG,
    STANDARD_CONFIG,
    make_processing_axes,
    make_scene_radar_or_skip,
    simulate_point_targets,
)

pytestmark = pytest.mark.gpu

_VFAST = {**FAST_CONFIG, "adc_start_time": 0, "chirp_per_frame": 32, "num_doppler_bins": 32}
_VFULL = {**STANDARD_CONFIG, "adc_start_time": 0}


def _local(distance, *, x=0.0, y=0.0):
    return (x, y, -float(distance))


def _range_profile(frame):
    """Peak-over-Doppler magnitude per range bin, as a numpy vector."""

    return frame.range_profile_db().cpu().numpy()


class TestTwoTargetsDifferentRanges:
    """Two targets at different ranges should both be detected."""

    def test_both_detected(self):
        radar = make_scene_radar_or_skip(_VFAST)
        d1, d2 = 2.0, 5.0
        frame = simulate_point_targets(radar, [_local(d1), _local(d2)])
        cloud = frame.point_cloud(positive_velocity_only=False)

        ranges = cloud.range_m.cpu().numpy()
        tol = frame.axes.range_bin_m * 3
        assert np.any(np.abs(ranges - d1) < tol), (d1, ranges)
        assert np.any(np.abs(ranges - d2) < tol), (d2, ranges)

    def test_rd_map_shows_two_range_peaks(self):
        radar = make_scene_radar_or_skip(_VFAST)
        d1, d2 = 2.0, 4.5
        frame = simulate_point_targets(radar, [_local(d1), _local(d2)])

        profile = _range_profile(frame)
        axis = frame.axes.range_m.cpu().numpy()
        floor = np.median(profile)
        for distance in (d1, d2):
            centre = int(np.argmin(np.abs(axis - distance)))
            region = profile[max(0, centre - 2) : centre + 3]
            assert region.max() > floor * 4.0, (distance, region.max(), floor)

    def test_the_two_range_peaks_are_the_two_declared_distances(self):
        """The falsifier: the two strongest bins ARE the two targets.

        The test above checks each expected bin against the noise floor, which a
        single smeared peak spanning both could satisfy. This asserts the pair
        directly, and it is only expressible because the two targets carry the
        same cross section on this route.
        """

        radar = make_scene_radar_or_skip(_VFAST)
        d1, d2 = 2.0, 5.0
        frame = simulate_point_targets(radar, [_local(d1), _local(d2)])

        profile = torch.as_tensor(_range_profile(frame))
        axis = frame.axes.range_m.cpu()
        # Two peaks, found by suppressing a window around the first.
        first = int(profile.argmax())
        window = max(1, int(round(0.5 / frame.axes.range_bin_m)))
        masked = profile.clone()
        masked[max(0, first - window) : first + window + 1] = -1.0
        second = int(masked.argmax())

        found = sorted((float(axis[first]), float(axis[second])))
        tol = frame.axes.range_bin_m * 2
        assert abs(found[0] - d1) < tol, found
        assert abs(found[1] - d2) < tol, found


class TestTwoTargetsDifferentVelocities:
    """Two targets at the same range but different closing speeds."""

    def test_rd_map_shows_two_doppler_peaks(self):
        """Each target lands in its own Doppler bin, both signed correctly.

        The old test asserted only that SOME Doppler peak existed at the range
        bin, which a single stationary target would have satisfied. What the
        route can now say is stronger and is what the scenario was for: the two
        bins are the two declared closing speeds, one positive and one negative.
        """

        radar = make_scene_radar_or_skip(_VFULL)
        distance = 3.0
        v1, v2 = 1.0, -1.5  # closing and receding
        frame = simulate_point_targets(
            radar, [(_local(distance), (0.0, 0.0, v1)), (_local(distance, x=0.1), (0.0, 0.0, v2))]
        )

        combined = frame.combined_map().abs()
        axis = frame.axes.range_m.cpu().numpy()
        range_bin = int(np.argmin(np.abs(axis - distance)))
        slice_db = combined[:, range_bin].cpu()

        first = int(slice_db.argmax())
        masked = slice_db.clone()
        window = max(1, int(round(0.5 / frame.axes.velocity_bin_mps)))
        masked[max(0, first - window) : first + window + 1] = -1.0
        second = int(masked.argmax())

        velocities = sorted((float(frame.axes.velocity_mps[first]), float(frame.axes.velocity_mps[second])))
        tol = frame.axes.velocity_bin_mps * 3
        assert abs(velocities[0] - v2) < tol, velocities
        assert abs(velocities[1] - v1) < tol, velocities


class TestRangeResolutionLimit:
    """Two targets closer than the range resolution merge into one."""

    def test_unresolvable_targets_merge(self):
        radar = make_scene_radar_or_skip(_VFAST)
        centre = 3.0
        resolution = make_processing_axes(_VFAST).range_bin_m
        delta = frame_delta = resolution * 0.3
        frame = simulate_point_targets(radar, [_local(centre - delta), _local(centre + frame_delta)])
        cloud = frame.point_cloud(positive_velocity_only=False)

        ranges = cloud.range_m.cpu().numpy()
        near = ranges[np.abs(ranges - centre) < frame.axes.range_bin_m * 3]
        assert near.size > 0, ranges
        spread = float(near.max() - near.min())
        assert spread < frame.axes.range_bin_m * 6, (
            f"unresolvable targets have range spread {spread:.4f} m (resolution {frame.axes.range_bin_m:.4f} m)"
        )

    def test_resolvable_targets_separate(self):
        """Two targets 5 bins apart leave a valley between their peaks."""

        radar = make_scene_radar_or_skip(_VFAST)
        d1 = 3.0
        resolution = make_processing_axes(_VFAST).range_bin_m
        frame = simulate_point_targets(radar, [_local(d1), _local(d1 + resolution * 5)])
        profile = _range_profile(frame)
        axis = frame.axes.range_m.cpu().numpy()

        bin1 = int(np.argmin(np.abs(axis - d1)))
        bin2 = int(np.argmin(np.abs(axis - (d1 + resolution * 5))))
        assert bin1 < bin2
        peak = max(profile[bin1], profile[bin2])
        valley = profile[bin1 : bin2 + 1].min()
        assert peak > valley, "no variation between resolvable targets"


class TestPointCloudOutputFormat:
    """The published point cloud is a typed record, not a bare column matrix.

    ``sigproc.process_pc`` returned an ``(N, 6)`` numpy array whose column
    meanings lived in a comment. ``processing.point_cloud`` publishes a
    ``PointCloud`` with four named tensors and ``POINT_CLOUD_COLUMNS`` as the
    single statement of the flat order, so the format test is about the record
    and the adapter's column order at once.
    """

    def test_the_record_and_its_column_order_agree(self):
        from witwin.radar.processing import POINT_CLOUD_COLUMNS

        radar = make_scene_radar_or_skip(_VFAST)
        frame = simulate_point_targets(radar, [_local(2.0), _local(4.0)])
        cloud = frame.point_cloud(positive_velocity_only=False)

        assert POINT_CLOUD_COLUMNS == ("x", "y", "z", "velocity_mps", "energy", "range_m")
        count = int(cloud.xyz.shape[0])
        assert count > 0
        assert tuple(cloud.xyz.shape) == (count, 3)
        for name in ("velocity_mps", "energy", "range_m"):
            assert tuple(getattr(cloud, name).shape) == (count,)
        assert bool((cloud.range_m >= 0).all())
        # The boresight is local -z, which the point cloud publishes as +y.
        assert bool((cloud.xyz[:, 1] >= 0).all())

    def test_a_world_with_no_target_is_refused_rather_than_answered(self):
        """The empty scene is a REFUSAL on this route, and that is the answer.

        ``sigproc``'s pipeline answered an empty interpolator with an empty
        ``(0, 6)`` array, because a solver that traces nothing produces nothing.
        A two-way join is not that: it is frozen against a declared set of
        scatter sites, and a round trip with no site is not a round trip with
        zero rows - it is a topology that cannot be frozen. The refusal is
        pinned so the deletion of the old behaviour is a recorded change of
        contract rather than a silent one.
        """

        from witwin.radar.simulation import ScatterSitePolicy

        radar = make_scene_radar_or_skip(_VFAST)
        with pytest.raises(ValueError, match="site_count must be a positive int"):
            simulate_point_targets(radar, [])

        # And the POLICY itself accepts the empty declaration: where the sites
        # come from is a statement about the caller, and the refusal belongs to
        # the join that has to be frozen against them. Pinning both halves is
        # what says the refusal is placed deliberately rather than wherever the
        # first shape check happened to be.
        empty = torch.zeros((0, 3), dtype=torch.float32, device=radar.device)
        assert int(ScatterSitePolicy.explicit(empty).positions_m.shape[0]) == 0
