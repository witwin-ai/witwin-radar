"""F-7 and M5-5 products on the host: padding, clutter, range-angle, clustering, tracking, axes from a radar.

Every claim is against a synthetic analytic signal built here, in SI units,
with the closing-positive velocity convention the Range-Doppler stage
publishes.
"""

from __future__ import annotations

import math
from dataclasses import fields

import pytest
import torch
from conftest import PROCESSING_CONFIG, make_processing_axes

from witwin.radar import Radar
from witwin.radar.processing import (
    ArrayGeometry,
    DetectionFrame,
    KalmanTracker,
    PointCloud,
    ProcessingAxes,
    ProcessingCube,
    RangeDopplerMap,
    cluster_points,
    greedy_assignment,
    range_angle_map,
    range_doppler_map,
    range_profile,
    remove_static_clutter,
)
from witwin.radar.synthesis.assembly import SynthesisResult

BEAT_CONFIG = {**PROCESSING_CONFIG, "output_domain": "beat"}


def _beat_cube(axes, *, range_bin: float, doppler_bin: float, amplitude: complex = 1.0) -> torch.Tensor:
    """``[TX, RX, C, S]`` beat cube: one tone at fractional range and Doppler bins, closing positive.

    The beat phasor is conjugated, so a closing target's slow-time tone sits at
    a NEGATIVE raw frequency and the Doppler stage reverses it.
    """

    samples, chirps = axes.range_bin_count, axes.doppler_bin_count
    fast = torch.arange(samples, dtype=torch.float64)
    slow = torch.arange(chirps, dtype=torch.float64)
    tone = torch.exp(2j * math.pi * range_bin * fast / samples)
    walk = torch.exp(-2j * math.pi * doppler_bin * slow / chirps)
    cube = amplitude * walk.reshape(-1, 1) * tone.reshape(1, -1)
    return cube.expand(axes.num_tx, axes.num_rx, chirps, samples).to(torch.complex128).contiguous()


# ---------------------------------------------------------------------------
# M5-5: axes and cubes from a radar
# ---------------------------------------------------------------------------


def test_m5_5_from_radar_equals_the_record_the_echo_builds():
    """Every field of ``from_radar`` equals ``from_synthesis`` on the cube the radar's waveform publishes."""

    radar = Radar.from_dict(dict(BEAT_CONFIG), device="cpu")
    spec = radar.waveform_spec()
    array = radar.system_config.sensors.array
    cube = torch.zeros((spec.num_chirps, array.sensor_pair_count, spec.num_samples), dtype=torch.complex64)
    expected = ProcessingAxes.from_synthesis(SynthesisResult.from_fmcw(cube, spec), spec, array)
    found = ProcessingAxes.from_radar(radar)
    for field in fields(ProcessingAxes):
        left, right = getattr(found, field.name), getattr(expected, field.name)
        if isinstance(left, torch.Tensor):
            assert torch.equal(left, right), field.name
        else:
            assert left == right, field.name


def test_from_radar_refuses_something_that_is_not_a_radar():
    with pytest.raises(TypeError, match="Radar"):
        ProcessingAxes.from_radar(make_processing_axes())


# ---------------------------------------------------------------------------
# Zero padding
# ---------------------------------------------------------------------------


def test_zero_padding_keeps_the_amplitude_and_refines_both_axes():
    """An off-bin tone read on a 4x finer grid: the peak lands on the true range and velocity."""

    radar = Radar.from_dict(dict(BEAT_CONFIG), device="cpu")
    axes = ProcessingAxes.from_radar(radar)
    cube = ProcessingCube(data=_beat_cube(axes, range_bin=12.25, doppler_bin=3.5), axes=axes)
    profile = range_profile(cube, range_bins=4 * axes.range_bin_count)
    rd = range_doppler_map(profile, doppler_bins=4 * axes.doppler_bin_count)
    assert rd.axes.range_bin_m == pytest.approx(axes.range_bin_m / 4)
    assert rd.axes.velocity_bin_mps == pytest.approx(axes.velocity_bin_mps / 4)
    magnitude = rd.data[0, 0].abs()
    peak = int(magnitude.argmax())
    doppler_index, range_index = divmod(peak, magnitude.shape[-1])
    assert float(rd.axes.range_m[range_index]) == pytest.approx(12.25 * axes.range_bin_m, rel=1e-12)
    assert float(rd.axes.velocity_mps[doppler_index]) == pytest.approx(3.5 * axes.velocity_bin_mps, rel=1e-12)
    # On the padded grid the tone is on a bin, so the amplitude normalisation reads the coefficient itself.
    assert float(magnitude.max()) == pytest.approx(1.0, rel=1e-9)


def test_zero_padding_is_refused_where_it_does_not_apply():
    spectrum = make_processing_axes(PROCESSING_CONFIG)
    spectrum = ProcessingAxes.from_radar(Radar.from_dict(dict(PROCESSING_CONFIG), device="cpu"))
    cube = ProcessingCube(
        data=torch.zeros((3, 4, spectrum.doppler_bin_count, spectrum.range_bin_count), dtype=torch.complex64),
        axes=spectrum,
    )
    with pytest.raises(ValueError, match="already transformed"):
        range_profile(cube, range_bins=128)
    beat = ProcessingAxes.from_radar(Radar.from_dict(dict(BEAT_CONFIG), device="cpu"))
    cube = ProcessingCube(data=torch.zeros((3, 4, beat.doppler_bin_count, 64), dtype=torch.complex64), axes=beat)
    with pytest.raises(ValueError, match="no smaller"):
        range_profile(cube, range_bins=32)


# ---------------------------------------------------------------------------
# Static clutter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["mean", "mti2", "mti3"])
def test_static_clutter_is_removed_and_a_mover_is_kept_on_its_velocity(method):
    radar = Radar.from_dict(dict(BEAT_CONFIG), device="cpu")
    axes = ProcessingAxes.from_radar(radar)
    clutter = _beat_cube(axes, range_bin=10, doppler_bin=0, amplitude=100.0)
    mover = _beat_cube(axes, range_bin=30, doppler_bin=4)
    profile = range_profile(ProcessingCube(data=clutter + mover, axes=axes))
    cleaned = remove_static_clutter(profile, method=method)
    order = {"mean": 0, "mti2": 1, "mti3": 2}[method]
    assert cleaned.data.shape[-2] == axes.doppler_bin_count - order
    assert cleaned.axes.doppler_bin_count == axes.doppler_bin_count - order
    rd = range_doppler_map(cleaned)
    assert float(rd.data[..., 10].abs().max()) < 1e-9
    magnitude = rd.data[0, 0, :, 30].abs()
    velocity = float(rd.axes.velocity_mps[int(magnitude.argmax())])
    # The canceller's output is shorter, so its bin is coarser: the peak is within half a bin.
    assert abs(velocity - 4 * axes.velocity_bin_mps) <= rd.axes.velocity_bin_mps / 2 + 1e-12
    assert velocity > 0.0


def test_the_two_pulse_canceller_has_its_stated_response():
    """``|H(f)| = 2 |sin(pi f T)|``, measured on an on-bin tone of the shortened sequence."""

    radar = Radar.from_dict(dict(BEAT_CONFIG), device="cpu")
    axes = ProcessingAxes.from_radar(radar)
    profile = range_profile(ProcessingCube(data=_beat_cube(axes, range_bin=8, doppler_bin=3), axes=axes))
    filtered = remove_static_clutter(profile, method="mti2").data[0, 0, :, 8]
    expected = 2.0 * abs(math.sin(math.pi * 3 / axes.doppler_bin_count))
    torch.testing.assert_close(filtered.abs(), torch.full_like(filtered.abs(), expected), rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# Range-angle map
# ---------------------------------------------------------------------------


def test_the_range_angle_map_peaks_on_the_target_azimuth_and_range():
    axes = make_processing_axes({**PROCESSING_CONFIG, "tx_loc": [[0, 0, 0], [4, 0, 0], [0, 1, 0]]})
    array = ArrayGeometry.from_axes(axes)
    azimuth = math.asin(0.25)
    direction = torch.tensor([[math.sin(azimuth), math.cos(azimuth), 0.0]], dtype=torch.float64)
    from witwin.radar.processing import conventional_steering

    manifold = conventional_steering(array, direction, normalize=False, dtype=torch.complex128).reshape(3, 4, 1, 1)
    cube = _beat_cube(axes, range_bin=17, doppler_bin=0) * manifold
    rd = range_doppler_map(range_profile(ProcessingCube(data=cube, axes=axes)))
    grid = torch.linspace(-1.2, 1.2, 241, dtype=torch.float64)
    image = range_angle_map(rd, array, azimuth_rad=grid)
    assert tuple(image.data.shape) == (241, axes.range_bin_count)
    peak = int(image.data.argmax())
    angle_index, range_index = divmod(peak, axes.range_bin_count)
    assert range_index == 17
    assert abs(float(grid[angle_index]) - azimuth) <= 0.01
    # Unit-response weights: the on-beam, on-bin power is |C|^2 = 1.
    assert float(image.data.max()) == pytest.approx(1.0, rel=1e-3)


def test_the_range_angle_map_is_differentiable_in_the_map():
    axes = make_processing_axes()
    array = ArrayGeometry.from_axes(axes)
    data = torch.randn((3, 4, axes.doppler_bin_count, axes.range_bin_count), dtype=torch.complex64)
    data.requires_grad_(True)
    rd = RangeDopplerMap(data=data, axes=axes, window="rectangular", window_coherent_gain=1.0)
    image = range_angle_map(rd, array, azimuth_rad=torch.linspace(-0.5, 0.5, 5))
    (gradient,) = torch.autograd.grad(image.data.sum(), data)
    assert float(gradient.abs().sum()) > 0.0


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def _cloud(points, velocities=None) -> PointCloud:
    xyz = torch.tensor(points, dtype=torch.float64)
    count = int(xyz.shape[0])
    velocity = torch.zeros(count, dtype=torch.float64) if velocities is None else torch.tensor(velocities)
    return PointCloud(
        xyz=xyz,
        velocity_mps=velocity.to(torch.float64),
        energy=torch.zeros(count, dtype=torch.float64),
        range_m=xyz.norm(dim=1),
    )


def reference_dbscan(xyz: torch.Tensor, eps: float, min_points: int) -> list[int]:
    """Textbook DBSCAN by breadth-first expansion, labels canonicalised to the smallest core index."""

    count = int(xyz.shape[0])
    neighbours = [[j for j in range(count) if float((xyz[i] - xyz[j]).norm()) <= eps] for i in range(count)]
    core = [len(n) >= min_points for n in neighbours]
    root = [-1] * count
    for start in range(count):
        if not core[start] or root[start] >= 0:
            continue
        queue = [start]
        root[start] = start
        while queue:
            point = queue.pop()
            for other in neighbours[point]:
                if core[other] and root[other] < 0:
                    root[other] = start
                    queue.append(other)
    for point in range(count):
        if not core[point]:
            roots = [root[j] for j in neighbours[point] if core[j]]
            root[point] = min(roots) if roots else -1
    return root


def test_dbscan_finds_two_clusters_and_noise():
    generator = torch.Generator().manual_seed(4)
    left = torch.randn((20, 3), generator=generator, dtype=torch.float64) * 0.1 + torch.tensor([2.0, 5.0, 0.0])
    right = torch.randn((15, 3), generator=generator, dtype=torch.float64) * 0.1 + torch.tensor([-1.0, 7.0, 0.5])
    noise = torch.tensor([[10.0, 10.0, 10.0], [-8.0, 3.0, 1.0]], dtype=torch.float64)
    xyz = torch.cat((left, noise[:1], right, noise[1:]))
    clusters = cluster_points(_cloud(xyz.tolist()), eps_m=0.4, min_points=4)
    assert clusters.count == 2
    assert clusters.labels[:20].unique().tolist() == [0]
    assert clusters.labels[21:36].unique().tolist() == [1]
    assert clusters.labels[20].item() == -1 and clusters.labels[36].item() == -1
    torch.testing.assert_close(clusters.centroid[0], left.mean(dim=0))
    assert clusters.size.tolist() == [20, 15]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_dbscan_matches_the_textbook_expansion(seed):
    generator = torch.Generator().manual_seed(seed)
    xyz = torch.rand((80, 3), generator=generator, dtype=torch.float64) * 4.0
    clusters = cluster_points(_cloud(xyz.tolist()), eps_m=0.55, min_points=3)
    roots = reference_dbscan(xyz, 0.55, 3)
    distinct = sorted({r for r in roots if r >= 0})
    expected = [distinct.index(r) if r >= 0 else -1 for r in roots]
    assert clusters.labels.tolist() == expected


def test_a_velocity_scale_separates_two_crossing_targets():
    """Two co-located groups moving at different radial speeds are one cluster in space, two with velocity."""

    points = [[0.0, 5.0, 0.0], [0.05, 5.0, 0.0], [0.0, 5.05, 0.0]] * 2
    velocities = [1.0, 1.0, 1.0, -1.0, -1.0, -1.0]
    assert cluster_points(_cloud(points, velocities), eps_m=0.2, min_points=2).count == 1
    assert cluster_points(_cloud(points, velocities), eps_m=0.2, min_points=2, velocity_scale_s=1.0).count == 2


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------


def _frame(points, time_s, index) -> DetectionFrame:
    xyz = torch.tensor(points, dtype=torch.float64).reshape(-1, 3)
    count = int(xyz.shape[0])
    zero = torch.zeros(count, dtype=torch.float64)
    return DetectionFrame(time_s=time_s, xyz=xyz, velocity_mps=zero, energy=zero, frame_index=index)


def test_the_kalman_tracker_follows_two_crossing_constant_velocity_targets():
    """Two targets on straight lines whose paths cross keep their ids, and the velocity converges."""

    tracker = KalmanTracker(process_noise_mps2=0.5, measurement_noise_m=0.05, gate=4.0)
    generator = torch.Generator().manual_seed(9)
    velocity_a, velocity_b = torch.tensor([1.0, 0.5, 0.0]), torch.tensor([-1.0, 0.5, 0.0])
    start_a, start_b = torch.tensor([-5.0, 5.0, 0.0]), torch.tensor([5.0, 5.0, 0.2])
    ids = []
    for step in range(30):
        time = 0.1 * step
        noise = torch.randn((2, 3), generator=generator, dtype=torch.float64) * 0.02
        points = torch.stack((start_a + velocity_a * time, start_b + velocity_b * time)).to(torch.float64) + noise
        ids.append(tracker.push(_frame(points.tolist(), time, step)).tolist())
    assert all(row == ids[0] for row in ids), ids
    state = tracker.tracks
    order = torch.argsort(state.ids)
    torch.testing.assert_close(state.velocity[order], torch.stack((velocity_a, velocity_b)).double(), atol=0.1, rtol=0)


def test_a_coasting_track_is_dropped_after_its_misses_and_a_new_one_is_born():
    tracker = KalmanTracker(max_misses=1)
    tracker.push(_frame([[0.0, 5.0, 0.0]], 0.0, 0))
    tracker.push(_frame([[0.0, 5.0, 0.0]], 0.1, 1))
    tracker.push(_frame(torch.zeros((0, 3)).tolist(), 0.2, 2))
    assert len(tracker.tracks) == 1
    tracker.push(_frame(torch.zeros((0, 3)).tolist(), 0.3, 3))
    assert len(tracker.tracks) == 0
    assert tracker.push(_frame([[0.0, 5.0, 0.0]], 0.4, 4)).tolist() == [1]


def test_the_tracker_refuses_an_unordered_stream():
    tracker = KalmanTracker()
    tracker.push(_frame([[0.0, 5.0, 0.0]], 1.0, 0))
    with pytest.raises(ValueError, match="not after"):
        tracker.push(_frame([[0.0, 5.0, 0.0]], 1.0, 1))


def test_greedy_assignment_is_the_pair_by_pair_greedy_matching():
    generator = torch.Generator().manual_seed(12)
    for _ in range(20):
        cost = torch.rand((7, 5), generator=generator, dtype=torch.float64)
        expected = [-1] * 7
        taken_rows, taken_columns = set(), set()
        for flat in torch.argsort(cost.reshape(-1)).tolist():
            row, column = divmod(flat, 5)
            if row in taken_rows or column in taken_columns or float(cost[row, column]) > 0.6:
                continue
            expected[row] = column
            taken_rows.add(row)
            taken_columns.add(column)
        assert greedy_assignment(cost, gate=0.6).tolist() == expected
