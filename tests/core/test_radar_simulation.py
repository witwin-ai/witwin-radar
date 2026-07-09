from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from witwin.core import Box, Material, Structure
from witwin.radar import Radar, RadarConfig, Tracer
from witwin.radar.scene import Scene


def _config() -> dict:
    return {
        "num_tx": 1,
        "num_rx": 1,
        "fc": 77e9,
        "slope": 60.012,
        "adc_samples": 8,
        "adc_start_time": 0,
        "sample_rate": 4400,
        "idle_time": 7,
        "ramp_end_time": 58,
        "chirp_per_frame": 2,
        "frame_per_second": 10,
        "num_doppler_bins": 2,
        "num_range_bins": 8,
        "num_angle_bins": 8,
        "power": 12,
        "tx_loc": [[0, 0, 0]],
        "rx_loc": [[0, 0, 0]],
    }


def _scene() -> Scene:
    return Scene(device="cpu").add_structure(
        Structure(
            name="target",
            geometry=Box(position=(0.0, 0.0, -3.0), size=(0.8, 0.8, 0.8)),
            material=Material(eps_r=3.0),
        )
    )


def _trace():
    return SimpleNamespace(
        points=torch.tensor([[0.0, 0.0, -3.0]], dtype=torch.float32),
        intensities=torch.tensor([0.75], dtype=torch.float32),
        entry_points=torch.tensor([[0.0, 0.0, -3.0]], dtype=torch.float32),
        fixed_path_lengths=torch.tensor([0.0], dtype=torch.float32),
        depths=torch.tensor([0], dtype=torch.int32),
        normals=None,
    )


def _fake_signal(radar: Radar, trace) -> torch.Tensor:
    shape = (
        radar.config.num_tx,
        radar.config.num_rx,
        radar.config.chirp_per_frame,
        radar.config.adc_samples,
    )
    value = float(trace.points.sum().item()) + float(radar.position.sum().item())
    return torch.full(shape, complex(value, 0.0), dtype=torch.complex64, device=radar.device)


def _attach_fake_solver(radar: Radar) -> Radar:
    radar.solver = SimpleNamespace(mimo_from_trace=lambda trace, t0=0, **options: _fake_signal(radar, trace))
    return radar


def test_radar_simulate_returns_signal_tensor_and_records_last_trace(monkeypatch):
    trace = _trace()

    class FakeRenderer:
        def __init__(
            self,
            scene,
            radar,
            resolution=128,
            epsilon_r=5.0,
            sampling="triangle",
            multipath=False,
            max_reflections=0,
            ray_batch_size=65536,
        ):
            self.scene = scene
            self.radar = radar
            self.resolution = resolution
            self.epsilon_r = epsilon_r
            self.sampling = sampling
            self.multipath = multipath
            self.max_reflections = max_reflections
            self.ray_batch_size = ray_batch_size

        def trace(self):
            return trace

    monkeypatch.setattr("witwin.radar.trace.Tracer", FakeRenderer)

    radar = Radar(
        RadarConfig.from_dict(_config()),
        device="cpu",
        position=(0.0, 0.0, 0.0),
        target=(0.0, 0.0, -5.0),
        fov=60.0,
    )
    _attach_fake_solver(radar)
    signal = radar.simulate(_scene(), resolution=32, sampling="pixel")

    assert signal.shape == (1, 1, 2, 8)
    assert isinstance(signal, torch.Tensor)
    assert radar.last_trace is trace
    assert radar.last_tracer is not None


def test_radar_simulate_rejects_unknown_sampling_mode():
    radar = Radar(RadarConfig.from_dict(_config()), device="cpu")
    with pytest.raises(ValueError, match="not a valid SamplingMode"):
        radar.simulate(_scene(), sampling="unknown")


def test_radar_simulate_rejects_multipath_for_triangle():
    radar = Radar(RadarConfig.from_dict(_config()), device="cpu")
    with pytest.raises(ValueError, match="multipath=True requires sampling='pixel'"):
        radar.simulate(_scene(), sampling="triangle", multipath=True)


def test_tracer_rejects_multipath_for_triangle():
    radar = Radar(RadarConfig.from_dict(_config()), device="cpu")
    with pytest.raises(ValueError, match="multipath=True requires sampling='pixel'"):
        Tracer(_scene(), radar, sampling="triangle", multipath=True)


def test_radar_simulate_group_returns_named_results(monkeypatch):
    trace = _trace()

    class FakeRenderer:
        def __init__(self, scene, radar, **_kwargs):
            self.scene = scene
            self.radar = radar

        def trace(self):
            return trace

    monkeypatch.setattr("witwin.radar.trace.Tracer", FakeRenderer)

    front = _attach_fake_solver(Radar(_config(), name="front", device="cpu"))
    side = _attach_fake_solver(Radar(
        _config(),
        name="side",
        device="cpu",
        position=(2.0, 0.0, 0.0),
        target=(2.0, 0.0, -1.0),
    ))

    result = Radar.simulate_group(_scene(), radars=[front, side])

    assert tuple(result) == ("front", "side")
    assert result["front"].shape == (1, 1, 2, 8)
    assert result["side"].shape == (1, 1, 2, 8)
    assert not torch.allclose(result["front"], result["side"])


def test_radar_simulate_group_requires_names_for_sequences():
    radar = Radar(_config(), device="cpu")
    with pytest.raises(ValueError, match="requires names"):
        Radar.simulate_group(_scene(), radars=[radar])


def test_radar_simulate_group_rejects_duplicate_names():
    radar_a = Radar(_config(), name="dup", device="cpu")
    radar_b = Radar(_config(), name="dup", device="cpu")
    with pytest.raises(ValueError, match="unique"):
        Radar.simulate_group(_scene(), radars=[radar_a, radar_b])
