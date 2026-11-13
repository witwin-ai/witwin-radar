"""Ground-truth labels and the path summary (U-10, M5-8), against closed-form geometry."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.gpu

C0 = 299_792_458.0
CARRIER_HZ = 77e9


def _radar(**fields):
    from witwin.radar import Fmcw, Radar

    waveform = Fmcw(
        slope=30e12,
        sample_rate=10e6,
        samples_per_chirp=64,
        chirps_per_frame=16,
        adc_start=1e-6,
        idle=5e-6,
        ramp_end=10e-6,
    )
    return Radar(
        carrier=CARRIER_HZ,
        waveform=waveform,
        tx=((0.0, 0.0, 0.0),),
        rx=((0.0, 0.0, 0.0),),
        power=12.0,
        antenna_unit="half_wavelength",
        look_at=(0.0, 0.0, -1.0),
        **fields,
    )


def _empty():
    from witwin.core import Scene

    return Scene()


START = np.array([[2.0, 1.0, -8.0], [0.0, -0.5, -4.0]])
VELOCITY = np.array([[1.0, 0.0, 3.0], [0.0, 0.0, 0.0]])


def _trajectory(time_s: float) -> torch.Tensor:
    return torch.tensor(START + VELOCITY * time_s, dtype=torch.float32, device="cuda")


def _expected(position: np.ndarray, velocity: np.ndarray) -> dict[str, np.ndarray]:
    """Radar at the origin looking down -z, up +y: the radar frame is (x, -z, y)."""

    x, y, z = position[:, 0], -position[:, 2], position[:, 1]
    rng = np.linalg.norm(position, axis=1)
    return {
        "range_m": rng,
        "radial_velocity_mps": -(position * velocity).sum(axis=1) / rng,
        "azimuth_rad": np.arctan2(x, y),
        "elevation_rad": np.arctan2(z, np.hypot(x, y)),
    }


def test_point_target_labels_match_the_closed_form():
    from witwin.radar import Aspect, PointTargets
    from witwin.radar.labels import ground_truth

    axis = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    targets = PointTargets(
        positions=_trajectory(0.0),
        rcs=[1.0, 2.0],
        trajectory=_trajectory,
        aspect=Aspect(axis=axis.tolist(), exponent=2.0, coherent_interval=1.0),
        ids=(41, 17),
    )
    times = (0.0, 0.25)
    labels = ground_truth(_radar(), _empty(), targets, times=times)
    assert labels.kind == "point" and labels.target_id.tolist() == [41, 17]
    for frame, time_s in enumerate(times):
        position = START + VELOCITY * time_s
        for name, value in _expected(position, VELOCITY).items():
            # float32 positions are ~1e-6 m, so the central difference over 2 h = 2 ms is good to ~5e-4 m/s.
            tolerance = 1e-3 if name == "radial_velocity_mps" else 2e-5
            np.testing.assert_allclose(
                getattr(labels, name)[frame].numpy(), value, rtol=0, atol=tolerance, err_msg=name
            )
        # The aspect lobe max(u . axis, 0)^(2n) on the amplitude, squared on the cross section.
        toward = -position / np.linalg.norm(position, axis=1, keepdims=True)
        lobe = np.clip((toward * axis).sum(axis=1), 0.0, None) ** 4.0
        np.testing.assert_allclose(labels.rcs_m2[frame].numpy(), np.array([1.0, 2.0]) * lobe**2, rtol=1e-6)


def test_the_radar_s_own_motion_and_rotation_enter_the_labels():
    from witwin.radar import PointTargets
    from witwin.radar.labels import GroundTruth

    angle_rate = 0.5  # rad/s about +y

    def pose(time_s: float):
        angle = angle_rate * time_s
        rotation = torch.tensor(
            [[math.cos(angle), 0.0, math.sin(angle)], [0.0, 1.0, 0.0], [-math.sin(angle), 0.0, math.cos(angle)]]
        )
        return rotation, torch.tensor([0.0, 0.0, -2.0 * time_s])

    labeler = GroundTruth(_radar(), _empty(), PointTargets(positions=[(0.0, 0.0, -10.0)], rcs=1.0), pose=pose)
    labels = labeler.at(0.4)
    origin = np.array([0.0, 0.0, -0.8])
    # The pose is float32, so the radar's position is good to ~1e-7 m.
    assert labels.range_m[0, 0] == pytest.approx(10.0 - 0.8, abs=1e-6)
    assert labels.radial_velocity_mps[0, 0] == pytest.approx(2.0, abs=1e-3)
    # A rotation by +angle about +y turns the boresight -z toward -x, so the target slides to the right.
    assert labels.azimuth_rad[0, 0] == pytest.approx(angle_rate * 0.4, abs=1e-6)
    assert labels.world_position_m[0, 0].tolist() == pytest.approx([0.0, 0.0, -10.0])
    assert float(np.linalg.norm(labels.world_position_m[0, 0].numpy() - origin)) == pytest.approx(9.2, abs=1e-6)


def test_a_tabulated_cross_section_is_read_at_the_monostatic_aspect():
    from witwin.radar import PointTargets, RcsTable
    from witwin.radar.labels import ground_truth

    table = RcsTable(axis=[[1.0, 0.0, 0.0]], rcs_m2=[0.0, 2.0, 4.0])  # on theta = 0, pi/2, pi
    targets = PointTargets(positions=[(3.0, 0.0, -4.0)], rcs=table)
    labels = ground_truth(_radar(), _empty(), targets, times=(0.0,))
    # The target-to-radar direction (-3, 0, 4)/5 is acos(-0.6) off the axis; the response
    # interpolates the amplitude sqrt(sigma) there, and the cross section is its square.
    position = math.acos(-0.6) / math.pi * 2.0
    amplitude = math.sqrt(2.0) + (position - 1.0) * (2.0 - math.sqrt(2.0))
    assert labels.rcs_m2[0, 0] == pytest.approx(amplitude**2, rel=1e-6)


def test_structure_anchor_labels_follow_the_rigid_motion():
    from witwin.core import Mesh, PhysicalMaterial, Scene, Structure
    from witwin.core.dynamics import DynamicScene, LinearTrajectory

    from witwin.radar import StructureTargets
    from witwin.radar.labels import ground_truth

    mesh = Mesh(
        vertices=torch.tensor(((-0.1, -0.1, 0.0), (0.1, -0.1, 0.0), (0.1, 0.1, 0.0))),
        faces=torch.tensor(((0, 1, 2),), dtype=torch.int64),
        recenter=False,
        fill_mode="surface",
        topology_diagnostics=False,
    )
    material = PhysicalMaterial(name="metal", eps_r=1.0, sigma_e=1e7)
    plate = Structure(geometry=mesh, material=material, structure_id=7, material_id=1, assignment_id=1, surface_id=1)
    world = DynamicScene(
        Scene(structures=(plate,), endpoints=[]),
        structure_trajectories={
            7: LinearTrajectory(origin=torch.tensor([0.0, 0.0, -6.0]), velocity=torch.tensor([0.0, 0.0, 3.0]))
        },
    )
    labels = ground_truth(_radar(), world, StructureTargets(rcs=0.5), times=(0.0, 0.5))
    assert labels.kind == "structure"
    np.testing.assert_allclose(labels.range_m.numpy()[:, 0], [6.0, 4.5], atol=1e-6)
    np.testing.assert_allclose(labels.radial_velocity_mps.numpy()[:, 0], [3.0, 3.0], atol=1e-3)
    assert labels.rcs_m2.tolist() == [[0.5], [0.5]]


def test_mesh_labels_aggregate_the_facets_per_structure():
    from support import oracle_scenes as sc
    from witwin.core import PhysicalMaterial

    from witwin.radar import MeshTargets
    from witwin.radar.labels import ground_truth

    vertices, faces = sc.plate_mesh(3, side_m=0.3, range_m=6.0)
    scene = sc.mesh_scene(vertices, faces, material=PhysicalMaterial.perfect_conductor(), structure_id=4)
    labels = ground_truth(sc.radar(), scene, MeshTargets(), times=(0.0,))
    assert labels.kind == "mesh" and labels.target_id.tolist() == [4]
    assert labels.range_m[0, 0] == pytest.approx(6.0, abs=1e-6)
    assert labels.area_m2[0, 0] == pytest.approx(0.09, rel=1e-5)
    assert labels.radial_velocity_mps[0, 0] == pytest.approx(0.0, abs=1e-9)
    near, far = labels.range_extent_m[0, 0].tolist()
    assert 6.0 <= near < far < math.hypot(6.0, 0.15 * math.sqrt(2.0))
    assert math.isnan(float(labels.rcs_m2[0, 0]))
    assert set(labels.site_target.tolist()) == {4} and labels.site_id.shape[0] == 18


def test_the_path_summary_matches_the_labels():
    """A traced chirp-sampled frame: each row's delay is 2R/c and its Doppler 2 v_r f / c, closing positive."""

    from witwin.radar import Motion, PointTargets
    from witwin.radar.labels import ground_truth, path_summary

    radar = _radar()
    targets = PointTargets(positions=_trajectory(0.0), rcs=[1.0, 2.0], trajectory=_trajectory)
    labels = ground_truth(radar, _empty(), targets, times=(0.0,))
    paths = radar.trace(_empty(), targets, times=(0.0,), los=True, reflections=0, motion=Motion.chirp())
    summary = path_summary(paths)
    assert summary.row_count == 2 and summary.valid.all()
    order = [labels.site_id.tolist().index(site) for site in summary.site_id.tolist()]
    t_last = summary.time_s
    position = START + VELOCITY * t_last
    expected = _expected(position, VELOCITY)
    np.testing.assert_allclose(summary.delay_s.numpy() * C0 / 2.0, expected["range_m"][order], atol=1e-5)
    doppler = 2.0 * CARRIER_HZ / C0 * labels.radial_velocity_mps[0].numpy()[order]
    np.testing.assert_allclose(summary.doppler_hz.numpy(), doppler, rtol=2e-3, atol=1e-6)
    assert [summary.component_names[index] for index in summary.component.tolist()] == ["target", "target"]
    assert summary.depth.tolist() == [0, 0]
    assert (summary.amplitude > 0).all()

    still = radar.trace(
        _empty(), PointTargets(positions=START.tolist(), rcs=1.0), times=(0.0,), los=True, reflections=0
    )
    assert still.kind == "static" and path_summary(still).doppler_hz.tolist() == [0.0, 0.0]
