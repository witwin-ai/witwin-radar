from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from witwin.radar import RadarConfig
from witwin.radar.renderer import Renderer
from witwin.radar.scene import Scene
from witwin.radar.sensor import Sensor
from witwin.radar.simulation import RadarSpec, Simulation
from witwin.core import Box, Material, Structure


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
    return Scene(
        sensor=Sensor(origin=(0, 0, 0), target=(0, 0, -5), up=(0, 1, 0), fov=60),
        device="cpu",
    ).add_structure(
        Structure(
            name="target",
            geometry=Box(position=(0.0, 0.0, -3.0), size=(0.8, 0.8, 0.8)),
            material=Material(eps_r=3.0),
        )
    )


def test_mimo_smoke_result_exposes_signal_and_trace_tensors(monkeypatch):
    trace = SimpleNamespace(
        points=torch.tensor([[0.0, 0.0, -3.0]], dtype=torch.float32),
        intensities=torch.tensor([0.75], dtype=torch.float32),
        entry_points=torch.tensor([[0.0, 0.0, -3.0]], dtype=torch.float32),
        fixed_path_lengths=torch.tensor([0.0], dtype=torch.float32),
        depths=torch.tensor([0], dtype=torch.int32),
    )
    signal = torch.ones((1, 1, 2, 8), dtype=torch.complex64)

    class FakeRadar:
        def __init__(self, config, backend="dirichlet", device="cuda", sensor=None):
            self.config = config
            self.backend = backend
            self.device = device
            self.sensor = sensor

        def mimo(self, interpolator, t0=0.0):
            sample = interpolator(t0)
            intensities, points = sample.intensities, sample.points
            assert torch.equal(intensities, trace.intensities)
            assert torch.equal(points, trace.points)
            return signal

    class FakeRenderer:
        def __init__(
            self,
            scene,
            resolution=128,
            epsilon_r=5.0,
            sampling="triangle",
            sensor=None,
            multipath=False,
            max_reflections=0,
            ray_batch_size=65536,
        ):
            self.scene = scene
            self.resolution = resolution
            self.epsilon_r = epsilon_r
            self.sampling = sampling
            self.sensor = sensor
            self.multipath = multipath
            self.max_reflections = max_reflections
            self.ray_batch_size = ray_batch_size

        def trace(self):
            return trace

    monkeypatch.setattr("witwin.radar.simulation.Radar", FakeRadar)
    monkeypatch.setattr("witwin.radar.simulation.Renderer", FakeRenderer)

    result = Simulation.mimo(
        _scene(),
        config=RadarConfig.from_dict(_config()),
        backend="dirichlet",
        resolution=32,
        sampling="pixel",
    )

    assert result.method == "mimo"
    assert torch.equal(result.signal(), signal)
    assert torch.equal(result.trace_points(), trace.points)
    assert torch.equal(result.trace_intensities(), trace.intensities)
    assert torch.equal(result.trace_entry_points(), trace.entry_points)
    assert torch.equal(result.trace_fixed_path_lengths(), trace.fixed_path_lengths)
    assert torch.equal(result.trace_depths(), trace.depths)
    assert torch.equal(result.tensor("signal"), signal)
    assert torch.equal(result.tensor("trace_points"), trace.points)
    assert torch.equal(result.tensor("trace_intensities"), trace.intensities)
    assert torch.equal(result.tensor("trace_entry_points"), trace.entry_points)
    assert torch.equal(result.tensor("trace_path_lengths"), trace.fixed_path_lengths)
    assert torch.equal(result.tensor("trace_depths"), trace.depths)


def test_simulation_rejects_unknown_sampling_mode():
    with pytest.raises(ValueError, match="not a valid SamplingMode"):
        Simulation.mimo(
            _scene(),
            config=RadarConfig.from_dict(_config()),
            sampling="unknown",
        )


def test_simulation_rejects_multipath_for_triangle():
    with pytest.raises(ValueError, match="multipath=True requires sampling='pixel'"):
        Simulation.mimo(
            _scene(),
            config=RadarConfig.from_dict(_config()),
            sampling="triangle",
            multipath=True,
        )


def test_renderer_rejects_multipath_for_triangle():
    with pytest.raises(ValueError, match="multipath=True requires sampling='pixel'"):
        Renderer(_scene(), sampling="triangle", multipath=True)


def test_simulation_uses_scene_sensor_by_default(monkeypatch):
    trace = SimpleNamespace(
        points=torch.tensor([[0.0, 0.0, -3.0]], dtype=torch.float32),
        intensities=torch.tensor([0.75], dtype=torch.float32),
        entry_points=torch.tensor([[0.0, 0.0, -3.0]], dtype=torch.float32),
        fixed_path_lengths=torch.tensor([0.0], dtype=torch.float32),
        depths=torch.tensor([0], dtype=torch.int32),
    )
    signal = torch.ones((1, 1, 2, 8), dtype=torch.complex64)
    observed = {}

    class FakeRadar:
        def __init__(self, config, backend="dirichlet", device="cuda", sensor=None):
            observed["radar_sensor"] = sensor

        def mimo(self, interpolator, t0=0.0):
            return signal

    class FakeRenderer:
        def __init__(self, scene, resolution=128, epsilon_r=5.0, sampling="triangle", sensor=None, multipath=False, max_reflections=0, ray_batch_size=65536):
            observed["renderer_sensor"] = sensor

        def trace(self):
            return trace

    monkeypatch.setattr("witwin.radar.simulation.Radar", FakeRadar)
    monkeypatch.setattr("witwin.radar.simulation.Renderer", FakeRenderer)

    scene = _scene()
    Simulation.mimo(scene, config=RadarConfig.from_dict(_config()), backend="pytorch", device="cpu")

    assert observed["radar_sensor"] == scene.sensor
    assert observed["renderer_sensor"] == scene.sensor


def test_simulation_explicit_sensor_overrides_scene_sensor(monkeypatch):
    trace = SimpleNamespace(
        points=torch.tensor([[0.0, 0.0, -3.0]], dtype=torch.float32),
        intensities=torch.tensor([0.75], dtype=torch.float32),
        entry_points=torch.tensor([[0.0, 0.0, -3.0]], dtype=torch.float32),
        fixed_path_lengths=torch.tensor([0.0], dtype=torch.float32),
        depths=torch.tensor([0], dtype=torch.int32),
    )
    signal = torch.ones((1, 1, 2, 8), dtype=torch.complex64)
    observed = {}
    override = Sensor(origin=(1.0, 0.0, 0.0), target=(1.0, 0.0, -1.0), up=(0.0, 1.0, 0.0), fov=45.0)

    class FakeRadar:
        def __init__(self, config, backend="dirichlet", device="cuda", sensor=None):
            observed["radar_sensor"] = sensor

        def mimo(self, interpolator, t0=0.0):
            return signal

    class FakeRenderer:
        def __init__(self, scene, resolution=128, epsilon_r=5.0, sampling="triangle", sensor=None, multipath=False, max_reflections=0, ray_batch_size=65536):
            observed["renderer_sensor"] = sensor

        def trace(self):
            return trace

    monkeypatch.setattr("witwin.radar.simulation.Radar", FakeRadar)
    monkeypatch.setattr("witwin.radar.simulation.Renderer", FakeRenderer)

    Simulation.mimo(_scene(), config=RadarConfig.from_dict(_config()), backend="pytorch", device="cpu", sensor=override)

    assert observed["radar_sensor"] == override
    assert observed["renderer_sensor"] == override


def test_multi_simulation_returns_named_results(monkeypatch):
    trace = SimpleNamespace(
        points=torch.tensor([[0.0, 0.0, -3.0]], dtype=torch.float32),
        intensities=torch.tensor([0.75], dtype=torch.float32),
        entry_points=torch.tensor([[0.0, 0.0, -3.0]], dtype=torch.float32),
        fixed_path_lengths=torch.tensor([0.0], dtype=torch.float32),
        depths=torch.tensor([0], dtype=torch.int32),
    )
    seen = []

    class FakeRadar:
        def __init__(self, config, backend="dirichlet", device="cuda", sensor=None):
            self.sensor = sensor
            seen.append(("radar", sensor.origin))

        def mimo(self, interpolator, t0=0.0):
            value = float(self.sensor.origin[0] + 1.0)
            return torch.full((1, 1, 2, 8), complex(value, 0.0), dtype=torch.complex64)

    class FakeRenderer:
        def __init__(self, scene, resolution=128, epsilon_r=5.0, sampling="triangle", sensor=None, multipath=False, max_reflections=0, ray_batch_size=65536):
            seen.append(("renderer", sensor.origin))

        def trace(self):
            return trace

    monkeypatch.setattr("witwin.radar.simulation.Radar", FakeRadar)
    monkeypatch.setattr("witwin.radar.simulation.Renderer", FakeRenderer)

    front = Sensor(origin=(0.0, 0.0, 0.0), target=(0.0, 0.0, -1.0), up=(0.0, 1.0, 0.0))
    side = Sensor(origin=(2.0, 0.0, 0.0), target=(2.0, 0.0, -1.0), up=(0.0, 1.0, 0.0))

    result = Simulation.mimo_group(
        _scene(),
        radars=[
            RadarSpec(name="front", config=_config(), sensor=front, backend="pytorch", device="cpu"),
            RadarSpec(name="side", config=_config(), sensor=side, backend="pytorch", device="cpu"),
        ],
    )

    assert result.names() == ("front", "side")
    assert torch.allclose(result.signal("front").real, torch.ones((1, 1, 2, 8)))
    assert torch.allclose(result.signal("side").real, torch.full((1, 1, 2, 8), 3.0))
    assert [label for label, _ in seen] == ["radar", "renderer", "radar", "renderer"]
    assert torch.equal(seen[0][1], front.origin)
    assert torch.equal(seen[1][1], front.origin)
    assert torch.equal(seen[2][1], side.origin)
    assert torch.equal(seen[3][1], side.origin)


def test_multi_simulation_rejects_duplicate_radar_names():
    sensor = Sensor.identity()
    with pytest.raises(ValueError, match="unique"):
        Simulation.mimo_group(
            _scene(),
            radars=[
                RadarSpec(name="dup", config=_config(), sensor=sensor, backend="pytorch", device="cpu"),
                RadarSpec(name="dup", config=_config(), sensor=sensor, backend="pytorch", device="cpu"),
            ],
        )
