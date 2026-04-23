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


def _resolve_scene(scene_like) -> Scene:
    if isinstance(scene_like, Scene):
        return scene_like
    if isinstance(scene_like, SceneModule):
        return scene_like.to_scene()
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


def _execute_mimo(
    *,
    scene: Scene,
    config: RadarConfig | dict | str,
    backend: SolverBackend,
    render: RenderOptions,
    device: str,
    t0: float,
    sensor: Sensor | None,
    motion_sampling: MotionSampling,
    metadata: dict[str, Any] | None,
) -> Result:
    effective_sensor = sensor or scene.sensor
    radar = Radar(RadarConfig.coerce(config), backend=backend, device=device, sensor=effective_sensor)
    renderer = Renderer(
        scene,
        resolution=render.resolution,
        epsilon_r=render.epsilon_r,
        sampling=render.sampling,
        sensor=effective_sensor,
        multipath=render.multipath,
        max_reflections=render.max_reflections,
        ray_batch_size=render.ray_batch_size,
    )

    trace = renderer.trace(time=t0) if scene.has_motion else renderer.trace()

    if scene.has_motion and motion_sampling == "chirp":
        def interpolator(t):
            return renderer.trace(time=t)
    else:
        def interpolator(t):
            del t
            return trace

    signal = radar.mimo(interpolator, t0)

    return Result(
        method="mimo",
        scene=scene,
        signal=signal,
        trace=trace,
        radar=radar,
        renderer=renderer,
        metadata=dict(metadata or {}),
    )


class Simulation:
    """Scene -> renderer -> radar solver -> result."""

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
    ) -> Result:
        return _execute_mimo(
            scene=_resolve_scene(scene),
            config=config,
            backend=normalize_solver_backend(backend),
            render=RenderOptions(
                resolution=resolution,
                epsilon_r=epsilon_r,
                sampling=sampling,
                multipath=multipath,
                max_reflections=max_reflections,
                ray_batch_size=ray_batch_size,
            ),
            device=str(device),
            t0=float(t0),
            sensor=sensor,
            motion_sampling=normalize_motion_sampling(motion_sampling),
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
    ) -> MultiResult:
        specs = tuple(radars)
        if not specs:
            raise ValueError("mimo_group requires at least one RadarSpec.")
        names = [spec.name for spec in specs]
        if len(names) != len(set(names)):
            raise ValueError("mimo_group radar names must be unique.")

        render = RenderOptions(
            resolution=resolution,
            epsilon_r=epsilon_r,
            sampling=sampling,
            multipath=multipath,
            max_reflections=max_reflections,
            ray_batch_size=ray_batch_size,
        )
        resolved_sampling = normalize_motion_sampling(motion_sampling)

        results: dict[str, Result] = {}
        for spec in specs:
            results[spec.name] = _execute_mimo(
                scene=_resolve_scene(scene),
                config=spec.config,
                backend=spec.backend,
                render=render,
                device=spec.device,
                t0=spec.t0,
                sensor=spec.sensor,
                motion_sampling=resolved_sampling,
                metadata=spec.metadata,
            )
        return MultiResult(results=results, metadata=dict(metadata or {}))
