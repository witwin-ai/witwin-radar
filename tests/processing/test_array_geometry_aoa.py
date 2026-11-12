"""Angle estimation driven by the declared array, against analytic angles, through the pose.

The oracle is the geometry itself: a Radar on the host places its elements in
the world (``Radar.tx_pos`` / ``rx_pos``, the pose owner), a far target sits in
a known world direction, and the virtual-array snapshot is the exact round-trip
phase ``exp(-j k (|T - tx| + |T - rx|))`` conjugated into the beat phasor the
FMCW cube carries. Nothing of the processing package is used to build it, so a
frame mistake, a sign mistake or a spacing assumption shows up as a wrong
world direction.

Directions are placed exactly on an estimator bin (``u = 2 k / fft_size`` on a
half-wavelength lattice, ``k / fft_size`` on a one-wavelength one), so the
expected answer is exact rather than a tolerance band, and they are compared in
the WORLD frame after :meth:`PointCloud.to_world`.
"""

from __future__ import annotations

import math

import pytest
import torch
from conftest import PROCESSING_CONFIG

from witwin.radar import Radar
from witwin.radar.processing import (
    ArrayGeometry,
    PointCloud,
    ProcessingAxes,
    ProcessingCube,
    ca_cfar,
    combine_coherent,
    fft2_aoa,
    phase_comparison_aoa,
    point_cloud,
    range_doppler_map,
    range_profile,
    scan_aoa,
)

FFT_SIZE = 64
FAR_M = 2.0e4

#: Flat-config layouts in half wavelengths, in the radar's LOCAL frame
#: (right, up, -boresight). The TI layout puts its third transmitter one half
#: wavelength UP, which is what its elevation comes from.
LAYOUTS = {
    "1x8": ([[0, 0, 0]], [[i, 0, 0] for i in range(8)]),
    "2x4": ([[0, 0, 0], [4, 0, 0]], [[i, 0, 0] for i in range(4)]),
    "lambda": ([[0, 0, 0]], [[2 * i, 0, 0] for i in range(4)]),
    "ti3x4": ([[0, 0, 0], [4, 0, 0], [2, 1, 0]], [[-6, 0, 0], [-5, 0, 0], [-4, 0, 0], [-3, 0, 0]]),
}

#: A pose that is not the default one, so the world transform is exercised.
POSE = {"position": (1.0, -2.0, 0.5), "look_at": (11.0, 3.0, 2.0), "up": (0.0, 0.0, 1.0)}


def _radar(layout: str, **pose) -> Radar:
    transmitters, receivers = LAYOUTS[layout]
    config = {
        **PROCESSING_CONFIG,
        "num_tx": len(transmitters),
        "num_rx": len(receivers),
        "tx_loc": transmitters,
        "rx_loc": receivers,
        "output_domain": "beat",
    }
    return Radar.from_dict(config, device="cpu", **(pose or POSE))


def _world_direction(radar: Radar, u_x: float, u_z: float) -> torch.Tensor:
    """The world unit vector whose processing-frame cosines are ``(u_x, ., u_z)``."""

    _, world_from_local = radar._world_from_local_matrix(device="cpu", dtype=torch.float64)
    u_y = math.sqrt(1.0 - u_x * u_x - u_z * u_z)
    local = torch.tensor([u_x, u_z, -u_y], dtype=torch.float64)
    return world_from_local @ local


def _snapshot(radar: Radar, target: torch.Tensor) -> torch.Tensor:
    """``[P, 1]`` TX-major beat-phasor snapshot of a point at world position ``target``."""

    tx = radar.tx_pos.to(torch.float64)
    rx = radar.rx_pos.to(torch.float64)
    distance = (target - tx).norm(dim=1).reshape(-1, 1) + (target - rx).norm(dim=1).reshape(1, -1)
    wavenumber = 2.0 * math.pi / radar.wavelength
    # Channel's exp(-j k d), conjugated: the beat cube's phasor.
    return torch.polar(torch.ones_like(distance), wavenumber * distance).reshape(-1, 1).to(torch.complex128)


def _estimate(radar: Radar, route, u_x: float, u_z: float) -> torch.Tensor:
    axes = ProcessingAxes.from_radar(radar)
    array = ArrayGeometry.from_axes(axes)
    position = torch.tensor(radar.position, dtype=torch.float64)
    target = position + FAR_M * _world_direction(radar, u_x, u_z)
    cosines = route(_snapshot(radar, target), array, fft_size=FFT_SIZE)
    cloud = PointCloud(
        xyz=(cosines.to(torch.float64) * FAR_M).transpose(0, 1),
        velocity_mps=torch.zeros(1, dtype=torch.float64),
        energy=torch.zeros(1, dtype=torch.float64),
        range_m=torch.full((1,), FAR_M, dtype=torch.float64),
    )
    world = cloud.to_world(radar)
    assert world.frame == "world"
    return (world.xyz[0] - position) / FAR_M


@pytest.mark.parametrize("layout", ["1x8", "2x4", "ti3x4"])
@pytest.mark.parametrize("route", [fft2_aoa, scan_aoa], ids=["fft2", "scan"])
@pytest.mark.parametrize("k", [-9, 0, 5, 13])
def test_m5_4_azimuth_on_a_half_wavelength_layout_is_exact(layout, route, k):
    radar = _radar(layout)
    u_x = 2.0 * k / FFT_SIZE
    found = _estimate(radar, route, u_x, 0.0)
    torch.testing.assert_close(found, _world_direction(radar, u_x, 0.0), rtol=0.0, atol=1e-9)


@pytest.mark.parametrize("route", [fft2_aoa, scan_aoa], ids=["fft2", "scan"])
@pytest.mark.parametrize("k", [-11, 4, 11])
def test_m5_4_a_one_wavelength_layout_reads_its_own_spacing(route, k):
    """The audit's case: ``sin(az) = 0.172`` on a lambda-spaced line came back as 0.344."""

    radar = _radar("lambda")
    u_x = k / FFT_SIZE
    found = _estimate(radar, route, u_x, 0.0)
    torch.testing.assert_close(found, _world_direction(radar, u_x, 0.0), rtol=0.0, atol=1e-9)


@pytest.mark.parametrize("route", [phase_comparison_aoa, fft2_aoa, scan_aoa], ids=["phase", "fft2", "scan"])
@pytest.mark.parametrize(("k_x", "k_z"), [(6, 0), (-4, 16), (10, -8)])
def test_m5_4_the_ti_layout_recovers_azimuth_and_elevation(route, k_x, k_z):
    """Elevation comes from the third transmitter, one half wavelength UP in the radar's frame.

    The two-row lattice resolves elevation coarsely, so the FFT and scan
    routes are held to their bin; phase comparison reads elevation from the
    vertically aligned element pairs, continuously, so it carries the float32
    rounding of ``Radar.tx_pos`` (about 1e-7 m, 2e-4 rad at 77 GHz).
    """

    radar = _radar("ti3x4")
    u_x = 2.0 * k_x / FFT_SIZE
    u_z = 2.0 * k_z / FFT_SIZE
    found = _estimate(radar, route, u_x, u_z)
    expected = _world_direction(radar, u_x, u_z)
    tolerance = 5e-5 if route is phase_comparison_aoa else 1e-9
    torch.testing.assert_close(found, expected, rtol=0.0, atol=tolerance)


def test_m1_8_phase_comparison_refuses_the_layouts_it_cannot_read_by_name():
    """1x8, 2x4 and lambda-spaced lines used to come back with a wrong angle or a row-count error."""

    for layout in ("1x8", "2x4", "lambda"):
        array = ArrayGeometry.from_axes(ProcessingAxes.from_radar(_radar(layout)))
        virtual = torch.ones((array.sensor_pair_count, 1), dtype=torch.complex64)
        with pytest.raises(ValueError, match="route 'phase_comparison' cannot estimate angles"):
            phase_comparison_aoa(virtual, array)
    # The overlapping 3 x 4 test array: its first eight elements are not a line.
    overlap = ArrayGeometry.from_offsets(
        ((0, 0, 0), (2, 0, 0), (0, 0, 1)), tuple((i, 0, 0) for i in range(4)), element_spacing_m=1.0, wavelength_m=2.0
    )
    with pytest.raises(ValueError, match="not a uniform line"):
        phase_comparison_aoa(torch.ones((12, 1), dtype=torch.complex64), overlap)


def test_the_fft_route_refuses_an_array_with_no_lattice_and_the_scan_route_reads_it():
    """Non-commensurate offsets: no lattice, so only the steering scan applies."""

    array = ArrayGeometry.from_offsets(
        ((0.0, 0.0, 0.0),),
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0 + math.sqrt(2.0) / 10, 0.0, 0.0), (3.7, 0.0, 0.0)),
        element_spacing_m=0.5,
        wavelength_m=1.0,
    )
    assert array.lattice is None
    with pytest.raises(ValueError, match="route 'fft2' cannot estimate angles"):
        fft2_aoa(torch.ones((4, 1), dtype=torch.complex64), array)
    u_x = 2.0 * 7 / FFT_SIZE
    phase = (
        2.0
        * math.pi
        * array.spacing_wavelengths
        * torch.tensor([0.0, 1.0, 2.0 + math.sqrt(2.0) / 10, 3.7], dtype=torch.float64)
    )
    # Channel phasor (phase_sign -1): exp(+j k <r, u>).
    snapshot = torch.polar(torch.ones(4, dtype=torch.float64), phase * u_x).reshape(-1, 1)
    cosines = scan_aoa(snapshot.to(torch.complex128), array, fft_size=FFT_SIZE)
    assert float(cosines[0, 0]) == pytest.approx(u_x, abs=1e-12)
    assert float(cosines[2, 0]) == 0.0


def test_the_lattice_of_each_layout():
    lattices = {name: ArrayGeometry.from_axes(ProcessingAxes.from_radar(_radar(name))).lattice for name in LAYOUTS}
    assert (lattices["1x8"].size_x, lattices["1x8"].size_z, lattices["1x8"].pitch_x) == (8, 1, 1.0)
    assert (lattices["2x4"].size_x, lattices["2x4"].size_z) == (8, 1)
    assert (lattices["lambda"].size_x, lattices["lambda"].pitch_x) == (4, 2.0)
    assert (lattices["ti3x4"].size_x, lattices["ti3x4"].size_z, lattices["ti3x4"].pitch_z) == (8, 2, 1.0)


def test_from_axes_maps_the_radar_frame_to_the_processing_frame():
    """``(right, up, -boresight)`` -> ``(right, boresight, up)``: up offsets become ``z``."""

    array = ArrayGeometry.from_axes(ProcessingAxes.from_radar(_radar("ti3x4")))
    third = array.element_positions_m[8:] / array.element_spacing_m
    torch.testing.assert_close(third[:, 2], torch.ones(4, dtype=torch.float64))
    torch.testing.assert_close(third[:, 1], torch.zeros(4, dtype=torch.float64))


# ---------------------------------------------------------------------------
# From a recorded cube to a world point
# ---------------------------------------------------------------------------


def _recorded_cube(radar: Radar, target: torch.Tensor, range_bin: int) -> torch.Tensor:
    """``[slow, pair, fast]`` beat samples of a static target, pairs SINK major."""

    axes = ProcessingAxes.from_radar(radar)
    snapshot = _snapshot(radar, target).reshape(radar.num_tx, radar.num_rx)
    sink_major = snapshot.transpose(0, 1).reshape(-1)  # pair = rx * num_tx + tx
    fast = torch.arange(axes.range_bin_count, dtype=torch.float64)
    tone = torch.polar(torch.ones_like(fast), 2.0 * math.pi * range_bin * fast / axes.range_bin_count)
    slow = torch.ones(axes.doppler_bin_count, dtype=torch.complex128)
    cube = slow.reshape(-1, 1, 1) * sink_major.reshape(1, -1, 1) * tone.reshape(1, 1, -1)
    generator = torch.Generator().manual_seed(5)
    noise = torch.complex(
        torch.randn(cube.shape, generator=generator, dtype=torch.float64),
        torch.randn(cube.shape, generator=generator, dtype=torch.float64),
    )
    return (cube + 1e-2 * noise).to(torch.complex64)


@pytest.mark.parametrize(("layout", "route"), [("ti3x4", "phase_comparison"), ("2x4", "fft2"), ("lambda", "scan")])
def test_m5_5_a_recorded_cube_becomes_a_world_point(layout, route):
    """``ProcessingCube.from_tensor`` -> range/Doppler -> CFAR -> point cloud -> ``to_world``."""

    radar = _radar(layout)
    axes = ProcessingAxes.from_radar(radar)
    range_bin = 20
    distance = range_bin * axes.range_bin_m
    u_x = 2.0 * 6 / FFT_SIZE if layout != "lambda" else 6 / FFT_SIZE
    position = torch.tensor(radar.position, dtype=torch.float64)
    target = position + distance * _world_direction(radar, u_x, 0.0)
    cube = ProcessingCube.from_tensor(_recorded_cube(radar, target, range_bin), radar)
    rd = range_doppler_map(range_profile(cube))
    found = ca_cfar(combine_coherent(rd.data), guard_cells=(2, 4), training_cells=(4, 8), pfa=1e-6)
    cloud = point_cloud(found, rd, ArrayGeometry.from_axes(cube.axes), route=route).to_world(radar)
    assert len(cloud) == 1
    # A few metres out, the 16 mm aperture is not in the far field: the spherical wavefront bends the
    # element phases by up to k L^2 / (2 R) ~ 0.06 rad, which a plane-wave estimator reads as ~1e-3 of cosine.
    torch.testing.assert_close(cloud.xyz[0], target, rtol=0.0, atol=2e-3 * distance)


def test_from_tensor_takes_the_array_layout_as_well_and_refuses_other_shapes():
    radar = _radar("ti3x4")
    axes = ProcessingAxes.from_radar(radar)
    sink_major = torch.randn((axes.doppler_bin_count, 12, axes.range_bin_count), dtype=torch.complex64)
    packed = ProcessingCube.from_tensor(sink_major, radar)
    # [tx, rx]: pair (tx, rx) is at sink-major rank rx * num_tx + tx.
    assert torch.equal(packed.data[2, 1], sink_major[:, 1 * 3 + 2])
    assert torch.equal(ProcessingCube.from_tensor(packed.data, radar).data, packed.data)
    with pytest.raises(ValueError, match="sink major"):
        ProcessingCube.from_tensor(sink_major[:, :, :-1], radar)
    with pytest.raises(ValueError, match="slow, pair, fast"):
        ProcessingCube.from_tensor(sink_major[0], radar)


@pytest.mark.gpu
@pytest.mark.parametrize("layout", ["ti3x4", "2x4", "lambda"])
def test_m5_4_a_simulated_target_lands_at_its_world_position(layout):
    """``Radar.simulate`` -> ``Frame.points(route="scan")`` -> ``PointCloud.to_world`` on a posed radar.

    The radar looks along world ``+x`` with ``+y`` up (the scene-driven
    pose: on the default ``-z`` boresight the polarization nulls the
    transport). The target is off boresight in both angles, a few metres out.
    """

    from conftest import empty_world, scene_radar

    from witwin.radar import Motion, PointTargets

    transmitters, receivers = LAYOUTS[layout]
    config = {
        **PROCESSING_CONFIG,
        "adc_samples": 256,
        "chirp_per_frame": 32,
        "num_tx": len(transmitters),
        "num_rx": len(receivers),
        "tx_loc": transmitters,
        "rx_loc": receivers,
    }
    radar = scene_radar(config, position=(0.5, 0.2, -0.3))
    axes = ProcessingAxes.from_radar(radar)
    distance = 60 * axes.range_bin_m
    u_x = 0.25 if layout != "lambda" else 0.2
    u_z = 0.0 if layout != "ti3x4" else 0.1
    target = torch.tensor(radar.position, dtype=torch.float64) + distance * _world_direction(radar, u_x, u_z)
    result = radar.simulate(
        empty_world(),
        PointTargets(positions=target.to(torch.float32).reshape(1, 3).cuda(), rcs=1.0),
        times=(0.0,),
        los=True,
        reflections=0,
        motion=Motion.chirp(),
    )
    cloud = result.frame(0).points(route="scan", pfa=1e-6).to_world(radar)
    assert len(cloud) >= 1
    nearest = int((cloud.xyz.cpu() - target).norm(dim=1).argmin())
    error = float((cloud.xyz[nearest].cpu() - target).norm())
    # One range bin plus one scan-grid step of angle at this distance.
    assert error <= axes.range_bin_m + distance * 2.0 / 64, (error, cloud.xyz.cpu(), target)
