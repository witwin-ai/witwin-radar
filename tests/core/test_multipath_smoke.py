from __future__ import annotations

import numpy as np
import pytest
import torch

from witwin.core import Box, Material, Structure
from witwin.radar import Renderer, Sensor, Simulation
from witwin.radar.scene import Scene


pytestmark = pytest.mark.gpu


CONFIG = {
    "num_tx": 1,
    "num_rx": 1,
    "fc": 77e9,
    "slope": 60.012,
    "adc_samples": 16,
    "adc_start_time": 0,
    "sample_rate": 4400,
    "idle_time": 7,
    "ramp_end_time": 58,
    "chirp_per_frame": 2,
    "frame_per_second": 10,
    "num_doppler_bins": 2,
    "num_range_bins": 16,
    "num_angle_bins": 8,
    "power": 12,
    "tx_loc": [[0, 0, 0]],
    "rx_loc": [[0, 0, 0]],
}


def _scene() -> Scene:
    return Scene(
        sensor=Sensor(origin=(0, 0, 0), target=(0, 0, -5), up=(0, 1, 0), fov=60),
    ).add_structure(
        Structure(
            name="target",
            geometry=Box(position=(0.0, 0.0, -4.0), size=(1.0, 1.0, 1.0)),
            material=Material(eps_r=3.0),
        )
    )


def test_renderer_multipath_smoke_returns_rich_trace():
    try:
        trace = Renderer(
            _scene(),
            resolution=16,
            sampling="pixel",
            multipath=True,
            max_reflections=1,
            ray_batch_size=64,
        ).trace()
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        pytest.skip(f"renderer unavailable: {exc}")

    assert trace.points.ndim == 2 and trace.points.shape[1] == 3
    assert trace.entry_points.shape == trace.points.shape
    assert trace.fixed_path_lengths.shape == trace.intensities.shape
    assert trace.depths.shape == trace.intensities.shape
    assert torch.all(trace.depths >= 0)


@pytest.mark.parametrize("backend", ["pytorch", "dirichlet", "slang"])
def test_simulation_multipath_smoke_runs_for_all_backends(backend):
    try:
        result = Simulation.mimo(
            _scene(),
            config=CONFIG,
            backend=backend,
            resolution=16,
            sampling="pixel",
            multipath=True,
            max_reflections=1,
            ray_batch_size=64,
        ).run()
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        pytest.skip(f"{backend} unavailable: {exc}")

    signal = result.signal()
    assert signal.shape == (1, 1, 2, 16)
    assert result.trace_entry_points().shape == result.trace_points().shape
    assert result.trace_fixed_path_lengths().shape == result.trace_intensities().shape


def test_process_rd_runs_on_multipath_signal():
    from witwin.radar.sigproc import process_rd

    try:
        result = Simulation.mimo(
            _scene(),
            config=CONFIG,
            backend="pytorch",
            resolution=16,
            sampling="pixel",
            multipath=True,
            max_reflections=1,
            ray_batch_size=64,
        ).run()
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        pytest.skip(f"multipath unavailable: {exc}")

    rd_mag, rd_map, ranges, velocities = process_rd(result.radar, result.signal(), tx=0, rx=0)

    assert rd_mag.shape == (CONFIG["chirp_per_frame"], CONFIG["adc_samples"])
    assert rd_map.shape == rd_mag.shape
    assert ranges.shape[0] == CONFIG["num_range_bins"] // 2
    assert velocities.shape[0] == CONFIG["num_doppler_bins"]
    assert np.isfinite(rd_mag).all()
