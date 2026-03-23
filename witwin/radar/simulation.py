"""High-level radar simulation entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import RadarConfig
from .radar import Radar
from .renderer import Renderer
from .result import MultiResult, Result
from .scene import Scene, SceneModule
from .sensor import Sensor
from .types import (
    MotionSampling,
    SamplingMode,
    SolverBackend,
    normalize_motion_sampling,
    normalize_sampling_mode,
    normalize_solver_backend,
)


def _resolve_scene_input(scene_like):
    if isinstance(scene_like, Scene):
        return scene_like, None
    if isinstance(scene_like, SceneModule):
        return scene_like.to_scene(), scene_like
    raise TypeError("scene must be a radar.Scene or radar.SceneModule.")


@dataclass(frozen=True)
class RenderOptions:
    resolution: int = 128
    epsilon_r: float = 5.0
    sampling: SamplingMode = "triangle"
    multipath: bool = False
    max_reflections: int = 0
    ray_batch_size: int = 65536

    def __post_init__(self):
        object.__setattr__(self, "sampling", normalize_sampling_mode(self.sampling))
        if self.max_reflections < 0:
            raise ValueError("max_reflections must be >= 0.")
        if self.ray_batch_size <= 0:
            raise ValueError("ray_batch_size must be > 0.")
        if self.multipath and self.sampling != "pixel":
            raise ValueError("multipath=True requires sampling='pixel'.")


@dataclass(frozen=True)
class RadarSpec:
    name: str
    config: RadarConfig | dict | str
    sensor: Sensor | None = None
    backend: SolverBackend = "dirichlet"
    device: str = "cuda"
    t0: float = 0.0
    metadata: dict[str, Any] | None = None

    def __post_init__(self):
        name = str(self.name)
        if not name:
            raise ValueError("RadarSpec.name must be a non-empty string.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "config", RadarConfig.coerce(self.config))
        object.__setattr__(self, "backend", normalize_solver_backend(self.backend))
        object.__setattr__(self, "device", str(self.device))
        object.__setattr__(self, "t0", float(self.t0))
        if self.sensor is not None and not isinstance(self.sensor, Sensor):
            raise TypeError("RadarSpec.sensor must be a radar.Sensor.")
        object.__setattr__(self, "metadata", dict(self.metadata or {}))


class Simulation:
    """Scene -> renderer -> radar solver -> result."""

    def __init__(
        self,
        *,
        scene,
        method: str,
        config: RadarConfig | dict | str,
        backend: SolverBackend = "dirichlet",
        render: RenderOptions | None = None,
        device: str = "cuda",
        t0: float = 0.0,
        sensor: Sensor | None = None,
        motion_sampling: MotionSampling = "chirp",
        metadata: dict[str, Any] | None = None,
    ):
        resolved_scene, scene_module = _resolve_scene_input(scene)
        self.scene_input = scene
        self.scene = resolved_scene
        self.scene_module = scene_module
        self.method = str(method)
        self.config = RadarConfig.coerce(config)
        self.backend = normalize_solver_backend(backend)
        self.device = str(device)
        self.render = render or RenderOptions()
        self.t0 = float(t0)
        self.sensor = sensor
        self.motion_sampling = normalize_motion_sampling(motion_sampling)
        self.metadata = dict(metadata or {})

    @classmethod
    def mimo(
        cls,
        scene,
        *,
        config,
        backend: SolverBackend = "dirichlet",
        resolution: int = 128,
        epsilon_r: float = 5.0,
        sampling: SamplingMode = "triangle",
        multipath: bool = False,
        max_reflections: int = 0,
        ray_batch_size: int = 65536,
        device: str = "cuda",
        t0: float = 0.0,
        sensor: Sensor | None = None,
        motion_sampling: MotionSampling = "chirp",
        metadata: dict[str, Any] | None = None,
    ) -> "Simulation":
        return cls(
            scene=scene,
            method="mimo",
            config=config,
            backend=backend,
            render=RenderOptions(
                resolution=resolution,
                epsilon_r=epsilon_r,
                sampling=sampling,
                multipath=multipath,
                max_reflections=max_reflections,
                ray_batch_size=ray_batch_size,
            ),
            device=device,
            t0=t0,
            sensor=sensor,
            motion_sampling=motion_sampling,
            metadata=metadata,
        )

    @classmethod
    def mimo_group(
        cls,
        scene,
        *,
        radars: list[RadarSpec] | tuple[RadarSpec, ...],
        resolution: int = 128,
        epsilon_r: float = 5.0,
        sampling: SamplingMode = "triangle",
        multipath: bool = False,
        max_reflections: int = 0,
        ray_batch_size: int = 65536,
        motion_sampling: MotionSampling = "chirp",
        metadata: dict[str, Any] | None = None,
    ) -> "MultiSimulation":
        return MultiSimulation(
            scene=scene,
            method="mimo",
            radars=radars,
            render=RenderOptions(
                resolution=resolution,
                epsilon_r=epsilon_r,
                sampling=sampling,
                multipath=multipath,
                max_reflections=max_reflections,
                ray_batch_size=ray_batch_size,
            ),
            motion_sampling=motion_sampling,
            metadata=metadata,
        )

    def prepare(self):
        self._refresh_scene()
        sensor = self.sensor or self.scene.sensor
        radar = Radar(self.config, backend=self.backend, device=self.device, sensor=sensor)
        renderer = Renderer(
            self.scene,
            resolution=self.render.resolution,
            epsilon_r=self.render.epsilon_r,
            sampling=self.render.sampling,
            sensor=sensor,
            multipath=self.render.multipath,
            max_reflections=self.render.max_reflections,
            ray_batch_size=self.render.ray_batch_size,
        )
        return PreparedSimulation(self, radar=radar, renderer=renderer)

    def run(self) -> Result:
        prepared = self.prepare()
        trace = prepared.renderer.trace(time=self.t0) if self.scene.has_motion else prepared.renderer.trace()

        if self.scene.has_motion and self.motion_sampling == "chirp":
            def interpolator(t):
                return prepared.renderer.trace(time=t)
        else:
            def interpolator(t):
                del t
                return trace

        if self.method == "mimo":
            signal = prepared.radar.mimo(interpolator, self.t0)
        else:
            raise ValueError(f"Unsupported radar simulation method '{self.method}'.")

        return Result(
            method=self.method,
            scene=self.scene,
            signal=signal,
            trace=trace,
            radar=prepared.radar,
            renderer=prepared.renderer,
            metadata=self.metadata,
        )

    def _refresh_scene(self):
        if self.scene_module is not None:
            self.scene = self.scene_module.to_scene()
        elif isinstance(self.scene_input, Scene):
            self.scene = self.scene_input


class PreparedSimulation:
    def __init__(self, simulation: Simulation, *, radar: Radar, renderer: Renderer):
        self.simulation = simulation
        self.radar = radar
        self.renderer = renderer


class MultiSimulation:
    """Run the same scene against multiple radar poses/configurations."""

    def __init__(
        self,
        *,
        scene,
        method: str,
        radars: list[RadarSpec] | tuple[RadarSpec, ...],
        render: RenderOptions | None = None,
        motion_sampling: MotionSampling = "chirp",
        metadata: dict[str, Any] | None = None,
    ):
        _resolve_scene_input(scene)
        self.scene_input = scene
        self.method = str(method)
        self.radars = tuple(radars)
        if not self.radars:
            raise ValueError("MultiSimulation requires at least one RadarSpec.")
        names = [spec.name for spec in self.radars]
        if len(names) != len(set(names)):
            raise ValueError("MultiSimulation radar names must be unique.")
        self.render = render or RenderOptions()
        self.motion_sampling = normalize_motion_sampling(motion_sampling)
        self.metadata = dict(metadata or {})

    def run(self) -> MultiResult:
        results: dict[str, Result] = {}
        for spec in self.radars:
            simulation = Simulation(
                scene=self.scene_input,
                method=self.method,
                config=spec.config,
                backend=spec.backend,
                render=self.render,
                device=spec.device,
                t0=spec.t0,
                sensor=spec.sensor,
                motion_sampling=self.motion_sampling,
                metadata=spec.metadata,
            )
            results[spec.name] = simulation.run()
        return MultiResult(results=results, metadata=self.metadata)
