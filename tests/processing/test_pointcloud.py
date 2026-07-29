"""One target, placed exactly, recovered through the whole chain.

The point cloud is where every convention in this package has to agree at once:
the range axis in metres, the closing-positive Doppler sign, the TDM slot phase,
the array's phasor, and the virtual-element order. A single target placed on an
exact range bin, an exact Doppler bin and an exact angle bin is therefore the
one test that can catch any of them being wrong on its own.

The cube is built directly in the range / Doppler / angle domain rather than
propagated, because what is under test is the PROCESSING. Its three factors are
the three the chain inverts, and the amplitude convention makes the recovered
peak the coefficient itself.
"""

from __future__ import annotations

import math

import pytest
import torch

from conftest import PROCESSING_CONFIG, make_processing_axes
from witwin.radar.processing import (
    ArrayGeometry,
    PointCloud,
    ProcessingCube,
    ca_cfar_fast,
    conventional_steering,
    point_cloud,
    range_doppler_map,
    range_profile,
)
from witwin.radar.processing.detection import Detections, range_gate_mask


#: Transmitters at 0 and 4 half wavelengths make the first ``2 * num_rx``
#: virtual elements a uniform line, which is what the phase-comparison relation
#: is written against; the third is displaced in z only, so the elevation
#: estimate carries no azimuth-walk correction.
CONFIG = {
    **PROCESSING_CONFIG,
    "tx_loc": [[0, 0, 0], [4, 0, 0], [0, 0, 1]],
    "rx_loc": [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]],
}

RANGE_BIN = 19
DOPPLER_BIN = 3
AZIMUTH_COSINE = 0.25
COEFFICIENT = complex(0.7, -0.4)


def _records():
    axes = make_processing_axes(CONFIG)
    return axes, ArrayGeometry.from_axes(axes)


def _cube(axes, array, *, cosine: float = AZIMUTH_COSINE) -> torch.Tensor:
    """``[TX, RX, C, S]``: one target, on three exact bins, plus a noise floor.

    ``exp(+2j pi k_r s / S)`` puts the beat tone on range bin ``k_r``;
    ``exp(-2j pi k_d c / C)`` puts the slow-time tone where the CONJUGATED beat
    convention leaves a closing target, so the reconciled Doppler stage has to
    move it back to ``+k_d``; and the array manifold is built in the cube's own
    convention by the same owner the beamformer uses.

    The TDM slot phase is carried too, and it has to be: transmitter ``m`` is
    sampled ``m T_chirp`` into the slot, so a MOVING target writes
    ``exp(-j s 4 pi v m T_chirp / lambda)`` across the transmitter rows. Leaving
    it out would make the compensation the point-cloud stage applies a spurious
    phase rather than a removal, and the recovered azimuth would land a bin
    away - which is exactly what happened when this fixture first ran.
    """

    samples = int(axes.range_bin_count)
    chirps = int(axes.doppler_bin_count)
    direction = torch.tensor(
        [[cosine, math.sqrt(1.0 - cosine**2), 0.0]], dtype=torch.float64
    )
    manifold = conventional_steering(
        array, direction, normalize=False, dtype=torch.complex64
    ).reshape(array.num_tx, array.num_rx, 1, 1)

    chirp_period_s = axes.slow_time_period_s / axes.num_tx
    closing = DOPPLER_BIN * axes.velocity_bin_mps
    slot_phase = (
        -array.phase_sign
        * 4.0
        * math.pi
        * closing
        * torch.arange(array.num_tx, dtype=torch.float64)
        * chirp_period_s
        / array.wavelength_m
    )
    slot = torch.polar(torch.ones_like(slot_phase), slot_phase).to(
        torch.complex64
    ).reshape(array.num_tx, 1, 1, 1)

    fast = torch.arange(samples, dtype=torch.float64)
    slow = torch.arange(chirps, dtype=torch.float64)
    tone = torch.polar(
        torch.ones(samples, dtype=torch.float64),
        2.0 * math.pi * RANGE_BIN * fast / samples,
    ).to(torch.complex64)
    walk = torch.polar(
        torch.ones(chirps, dtype=torch.float64),
        -2.0 * math.pi * DOPPLER_BIN * slow / chirps,
    ).to(torch.complex64)

    generator = torch.Generator().manual_seed(606)
    floor = torch.complex(
        torch.randn((array.num_tx, array.num_rx, chirps, samples), generator=generator),
        torch.randn((array.num_tx, array.num_rx, chirps, samples), generator=generator),
    ).to(torch.complex64) * 1e-3
    signal = (
        manifold * slot * walk.reshape(1, 1, -1, 1) * tone.reshape(1, 1, 1, -1)
    )
    return signal * COEFFICIENT + floor


def _chain(array, cube: ProcessingCube):
    rd = range_doppler_map(range_profile(cube))
    combined = rd.data.reshape(array.sensor_pair_count, *rd.data.shape[-2:]).sum(dim=0)
    detected = ca_cfar_fast(
        combined.abs(), guard_cells=(1, 2), training_cells=(3, 4), pfa=1e-6
    )
    return rd, detected


# ---------------------------------------------------------------------------
# The chain
# ---------------------------------------------------------------------------


def test_one_target_lands_on_its_own_three_bins_and_becomes_one_point():
    """Range to half a bin, direction cosine to 1e-3, velocity to half a bin."""

    axes, array = _records()
    rd, detected = _chain(array, ProcessingCube(_cube(axes, array), axes))

    cloud = point_cloud(detected, rd, axes, array, route="phase_comparison")
    assert isinstance(cloud, PointCloud)
    assert len(cloud) == 1

    expected_range = float(axes.range_m[RANGE_BIN])
    assert abs(float(cloud.range_m[0]) - expected_range) <= 0.5 * axes.range_bin_m

    expected_velocity = DOPPLER_BIN * axes.velocity_bin_mps
    assert abs(float(cloud.velocity_mps[0]) - expected_velocity) <= (
        0.5 * axes.velocity_bin_mps
    )
    # Closing positive, and the sign is the thing: an unreconciled cube would
    # publish the same magnitude with the opposite sign.
    assert float(cloud.velocity_mps[0]) > 0.0

    direction = cloud.xyz[0] / float(cloud.range_m[0])
    assert float(direction[0]) == pytest.approx(AZIMUTH_COSINE, abs=1e-3)
    assert float(direction[2]) == pytest.approx(0.0, abs=1e-3)
    assert float(direction[1]) == pytest.approx(
        math.sqrt(1.0 - AZIMUTH_COSINE**2), abs=1e-3
    )
    # The xyz row is the direction times the range, which is the only place a
    # bin index becomes a position. The residual is the float32 estimator's:
    # its three cosines close to one only to single precision, and the range
    # itself is float64 and exact.
    torch.testing.assert_close(
        cloud.xyz[0].square().sum().sqrt(),
        cloud.range_m[0],
        rtol=1e-6,
        atol=1e-9,
    )


def test_the_published_columns_are_the_named_fields_in_the_published_order():
    axes, array = _records()
    rd, detected = _chain(array, ProcessingCube(_cube(axes, array), axes))
    cloud = point_cloud(detected, rd, axes, array)
    columns = cloud.as_columns()
    assert tuple(columns.shape) == (1, 6)
    assert torch.equal(columns[:, :3], cloud.xyz)
    assert torch.equal(columns[:, 3], cloud.velocity_mps)
    assert torch.equal(columns[:, 4], cloud.energy)
    assert torch.equal(columns[:, 5], cloud.range_m)


# ---------------------------------------------------------------------------
# The range gate, in metres
# ---------------------------------------------------------------------------


def test_the_range_gate_is_a_distance_and_not_a_bin_index():
    """The magic ``[:, :25]`` and ``[:, 125:]`` are gone. This is the contract.

    The gate is half open in METRES, so it says the same thing about the scene
    at any range-bin count - which the two literals it replaces could not,
    because they were a 128 x 256 configuration written into the source.
    """

    axes, array = _records()
    rd, detected = _chain(array, ProcessingCube(_cube(axes, array), axes))
    target = float(axes.range_m[RANGE_BIN])

    inside = point_cloud(
        detected,
        rd,
        axes,
        array,
        range_gate_m=(target - axes.range_bin_m, target + axes.range_bin_m),
    )
    assert len(inside) == 1

    outside = point_cloud(
        detected,
        rd,
        axes,
        array,
        range_gate_m=(target + 2.0 * axes.range_bin_m, target + 20.0 * axes.range_bin_m),
    )
    assert len(outside) == 0

    mask = range_gate_mask(axes, (target, target + axes.range_bin_m))
    assert int(mask.sum()) == 1
    assert bool(mask[RANGE_BIN])
    assert range_gate_mask(axes, None) is None
    with pytest.raises(ValueError, match="high > low"):
        range_gate_mask(axes, (5.0, 1.0))


def test_the_strongest_detections_survive_a_max_points_thinning():
    """Thinned on the MASK, before the row list exists.

    The deleted pipeline reordered the peak list by energy while reading
    energies and angles in mask order, so after thinning a point's range no
    longer belonged to its own angle. Here every column of a point comes from
    one cell by construction, which this asserts by thinning to one and getting
    the target.
    """

    axes, array = _records()
    cube = _cube(axes, array)
    # A second target six range bins away, in the SAME direction so that both
    # see the same array factor when the pairs are summed, and a fifth of the
    # amplitude so which one is stronger is decided by the amplitude alone.
    samples = int(axes.range_bin_count)
    fast = torch.arange(samples, dtype=torch.float64)
    shift = torch.polar(
        torch.ones(samples, dtype=torch.float64),
        2.0 * math.pi * 6.0 * fast / samples,
    ).to(torch.complex64)
    cube = cube + 0.2 * _cube(axes, array) * shift.reshape(1, 1, 1, -1)

    rd, detected = _chain(array, ProcessingCube(cube, axes))
    assert int(detected.mask.sum()) >= 2

    thinned = point_cloud(detected, rd, axes, array, max_points=1)
    assert len(thinned) == 1
    assert abs(
        float(thinned.range_m[0]) - float(axes.range_m[RANGE_BIN])
    ) <= 0.5 * axes.range_bin_m


def test_an_empty_detection_mask_gives_an_empty_cloud_and_not_a_crash():
    axes, array = _records()
    rd, _ = _chain(array, ProcessingCube(_cube(axes, array), axes))
    empty = Detections(
        mask=torch.zeros(
            (axes.doppler_bin_count, axes.range_bin_count), dtype=torch.bool
        ),
        threshold=torch.zeros(
            (axes.doppler_bin_count, axes.range_bin_count), dtype=torch.float32
        ),
    )
    cloud = point_cloud(empty, rd, axes, array)
    assert len(cloud) == 0
    assert tuple(cloud.xyz.shape) == (0, 3)
    assert tuple(cloud.as_columns().shape) == (0, 6)


def test_the_route_is_named_by_the_caller_and_an_unknown_one_is_refused():
    axes, array = _records()
    rd, detected = _chain(array, ProcessingCube(_cube(axes, array), axes))
    with pytest.raises(ValueError, match="route must be one of"):
        point_cloud(detected, rd, axes, array, route="magic")
    with pytest.raises(TypeError):
        point_cloud(detected.mask, rd, axes, array)
    with pytest.raises(ValueError, match=r"\[doppler, range\]"):
        point_cloud(
            Detections(
                mask=detected.mask.unsqueeze(0), threshold=detected.threshold.unsqueeze(0)
            ),
            rd,
            axes,
            array,
        )


def test_the_positive_velocity_filter_reads_the_reconciled_sign():
    axes, array = _records()
    rd, detected = _chain(array, ProcessingCube(_cube(axes, array), axes))
    kept = point_cloud(detected, rd, axes, array, positive_velocity_only=True)
    assert len(kept) == 1
    assert float(kept.velocity_mps[0]) > 0.0


# ---------------------------------------------------------------------------
# The whole chain, end to end
# ---------------------------------------------------------------------------


def test_the_cube_former_and_the_chain_agree_on_the_same_target():
    """Synthesis packing -> cube -> profile -> map -> detection -> point cloud.

    The cube is packed by ``ProcessingCube.from_synthesis`` from a rank-3
    synthesis result rather than assembled by hand, so the sink-major to
    tx-major transpose is inside the loop and the element table has to be in the
    order that transpose leaves. Feeding a sink-major element table to a
    tx-major cube transposes the array, which on a square front end changes no
    shape at all and mis-steers every angle.
    """

    from witwin.radar.synthesis.assembly import assemble_frame_cube

    axes, array = _records()
    cube = _cube(axes, array)
    # Back to the rank-3 (chirp, sensor_pair, sample) synthesis layout, in the
    # SINK-major composed pair rank that assemble_frame_cube transposes from.
    sink_major = (
        cube.permute(2, 0, 1, 3)
        .permute(0, 2, 1, 3)
        .reshape(
            axes.doppler_bin_count, array.sensor_pair_count, axes.range_bin_count
        )
        .contiguous()
    )
    packed = assemble_frame_cube(
        sink_major, num_tx=axes.num_tx, num_rx=axes.num_rx
    )
    assert torch.equal(packed, cube)

    rd, detected = _chain(array, ProcessingCube(data=packed, axes=axes))
    cloud = point_cloud(detected, rd, axes, array)
    assert len(cloud) == 1
    assert abs(
        float(cloud.range_m[0]) - float(axes.range_m[RANGE_BIN])
    ) <= 0.5 * axes.range_bin_m
    assert float(cloud.xyz[0, 0] / cloud.range_m[0]) == pytest.approx(
        AZIMUTH_COSINE, abs=1e-3
    )


def test_the_chain_runs_on_a_processing_cube_record_as_well_as_a_tensor():
    axes, array = _records()
    cube = ProcessingCube(data=_cube(axes, array), axes=axes)
    profile = range_profile(cube)
    rd = range_doppler_map(profile)
    assert tuple(rd.data.shape) == (
        axes.num_tx,
        axes.num_rx,
        axes.doppler_bin_count,
        axes.range_bin_count,
    )
