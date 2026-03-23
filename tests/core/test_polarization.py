from __future__ import annotations

import torch
import pytest

from witwin.core import Material
from witwin.radar import Scene, Simulation, Sensor


def _config(*, polarization) -> dict:
    return {
        "num_tx": 1,
        "num_rx": 1,
        "fc": 77e9,
        "slope": 60.012,
        "adc_samples": 32,
        "adc_start_time": 0,
        "sample_rate": 4400,
        "idle_time": 7,
        "ramp_end_time": 58,
        "chirp_per_frame": 4,
        "frame_per_second": 10,
        "num_doppler_bins": 4,
        "num_range_bins": 64,
        "num_angle_bins": 8,
        "power": 12,
        "tx_loc": [[0, 0, 0]],
        "rx_loc": [[0, 0, 0]],
        "polarization": polarization,
    }


def _slanted_plate_scene(*, device: str) -> Scene:
    normal = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=device)
    normal = normal / torch.linalg.norm(normal)
    reference = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=device)
    tangent_u = torch.cross(reference, normal, dim=0)
    tangent_u = tangent_u / torch.linalg.norm(tangent_u)
    tangent_v = torch.cross(normal, tangent_u, dim=0)
    center = torch.tensor([0.0, 0.0, -2.5], dtype=torch.float32, device=device)
    half_size = 0.9

    vertices = torch.stack(
        [
            center - half_size * tangent_u - half_size * tangent_v,
            center + half_size * tangent_u - half_size * tangent_v,
            center + half_size * tangent_u + half_size * tangent_v,
            center - half_size * tangent_u + half_size * tangent_v,
        ],
        dim=0,
    )
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int64, device=device)

    scene = Scene(
        sensor=Sensor(origin=(0.0, 0.0, 0.0), target=(0.0, 0.0, -1.0), up=(0.0, 1.0, 0.0)),
        device=device,
    )
    scene.add_mesh(
        name="plate",
        vertices=vertices,
        faces=faces,
        material=Material(eps_r=3.0),
        dynamic=True,
    )
    return scene


@pytest.mark.gpu
@pytest.mark.parametrize("sampling", ["triangle", "pixel"])
def test_end_to_end_polarization_changes_measured_signal(sampling):
    scene = _slanted_plate_scene(device="cuda")
    hh = Simulation.mimo(
        scene,
        config=_config(polarization={"tx": "horizontal", "rx": "horizontal"}),
        backend="pytorch",
        device="cuda",
        sampling=sampling,
        resolution=96 if sampling == "pixel" else 32,
    ).run()
    hv = Simulation.mimo(
        scene,
        config=_config(polarization={"tx": "horizontal", "rx": "vertical"}),
        backend="pytorch",
        device="cuda",
        sampling=sampling,
        resolution=96 if sampling == "pixel" else 32,
    ).run()

    hh_peak = hh.signal().abs().max()
    hv_peak = hv.signal().abs().max()
    ratio = hv_peak / hh_peak

    assert hh.trace_normals() is not None
    assert hh.trace_normals().shape == hh.trace_points().shape
    torch.testing.assert_close(ratio, torch.tensor(2.0, dtype=ratio.dtype, device=ratio.device), atol=0.15, rtol=0.1)


@pytest.mark.gpu
def test_process_rd_preserves_polarization_contrast():
    from witwin.radar.sigproc import process_rd

    scene = _slanted_plate_scene(device="cuda")
    hh = Simulation.mimo(
        scene,
        config=_config(polarization={"tx": "horizontal", "rx": "horizontal"}),
        backend="pytorch",
        device="cuda",
        sampling="triangle",
    ).run()
    hv = Simulation.mimo(
        scene,
        config=_config(polarization={"tx": "horizontal", "rx": "vertical"}),
        backend="pytorch",
        device="cuda",
        sampling="triangle",
    ).run()

    hh_rd, _, _, _ = process_rd(hh.radar, hh.signal(), tx=0, rx=0)
    hv_rd, _, _, _ = process_rd(hv.radar, hv.signal(), tx=0, rx=0)

    assert hv_rd.shape == hh_rd.shape
    assert float(hv_rd.max()) > float(hh_rd.max()) + 4.0
