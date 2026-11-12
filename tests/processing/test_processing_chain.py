"""The processing facade runs end to end, from a synthesis layout to a track handoff."""

from __future__ import annotations

import torch


def test_the_whole_chain_runs_from_a_synthesis_layout_to_a_detection_frame():
    """synthesize -> cube -> range profile -> RD -> beam cube -> CFAR -> AoA ->
    point cloud -> handoff, in one call sequence with no legacy name in it."""

    import math

    from conftest import PROCESSING_CONFIG, make_processing_axes

    from witwin.radar.processing import (
        ArrayGeometry,
        DetectionFrame,
        ProcessingCube,
        TrackHandoff,
        beam_cube,
        ca_cfar,
        conventional_steering,
        point_cloud,
        range_doppler_map,
        range_profile,
    )
    from witwin.radar.synthesis.assembly import assemble_frame_cube

    config = {
        **PROCESSING_CONFIG,
        "tx_loc": [[0, 0, 0], [4, 0, 0], [0, 0, 1]],
        "rx_loc": [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]],
    }
    axes = make_processing_axes(config)
    array = ArrayGeometry.from_axes(axes)

    generator = torch.Generator().manual_seed(2026)
    rank3 = torch.complex(
        torch.randn((axes.doppler_bin_count, array.sensor_pair_count, axes.range_bin_count), generator=generator),
        torch.randn((axes.doppler_bin_count, array.sensor_pair_count, axes.range_bin_count), generator=generator),
    ).to(torch.complex64)

    cube = ProcessingCube(data=assemble_frame_cube(rank3, num_tx=axes.num_tx, num_rx=axes.num_rx), axes=axes)
    profile = range_profile(cube)
    rd = range_doppler_map(profile, window="hann")

    directions = torch.tensor(
        [[math.sin(angle), math.cos(angle), 0.0] for angle in (-0.4, -0.2, 0.0, 0.2, 0.4)], dtype=torch.float64
    )
    weights = conventional_steering(array, directions)
    beams = beam_cube(rd, weights, directions=directions)
    assert tuple(beams.data.shape) == (5, axes.doppler_bin_count, axes.range_bin_count)

    # The batched detector runs on the whole beam cube at once, which is the
    # capability that did not exist: three rank-2 detectors and a loop.
    detected = ca_cfar(beams.data.abs(), guard_cells=(1, 2), training_cells=(2, 3), pfa=1e-2)
    assert tuple(detected.mask.shape) == tuple(beams.data.shape)

    combined = rd.data.reshape(array.sensor_pair_count, *rd.data.shape[-2:]).sum(dim=0)
    cells = ca_cfar(combined.abs(), guard_cells=(1, 2), training_cells=(2, 3), pfa=1e-2)
    cloud = point_cloud(cells, rd, array, max_points=8)
    handoff = TrackHandoff()
    assignment = handoff.push(DetectionFrame.from_point_cloud(cloud, time_s=0.0, frame_index=0))
    assert int(assignment.shape[0]) == len(cloud)
    assert handoff.track_count == len(cloud)
