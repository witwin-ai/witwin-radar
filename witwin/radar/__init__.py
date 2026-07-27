"""Public radar API.

The Dr.Jit ray tracer is GONE, not deprecated in place. ``Tracer``,
``fresnel``, ``Radar.simulate`` and ``Radar.simulate_group`` were the only
production entry points that reached it, and every one of them now raises with
a pointer to its replacement. Nothing silently falls back: a caller who used
the tracer gets an error naming the route that replaces it, which is the only
honest outcome when the replacement is a different contract rather than a
drop-in.

Removing the import is what makes the process-global "no Dr.Jit anywhere"
assertion possible at all. It used to be unachievable for one reason: importing
any ``witwin.radar`` submodule initialized this file, which imported
``trace.py``, which imported ``drjit``.
"""

from .radar import Radar, RadarConfig, quantize_complex_signal
from .solvers import Solver
from .trace_result import TraceResult
from .path_cache import MimoPathCache
from .scene_binding import (
    RadarWorldBinding,
    ScatterSitePolicy,
    StableIdAllocator,
    bind_radar_world,
)
from .simulation import RadarSimulationResult
from .propagation.contracts import RadarPropagationLegs
from .types import DetectorType, MotionSampling, SamplingMode
from witwin.core import (
    Box,
    Cone,
    Cylinder,
    Ellipsoid,
    Geometry as CoreGeometry,
    GeometryBase,
    HollowBox,
    Mesh,
    Prism,
    Pyramid,
    Sphere,
    Structure,
    Torus,
    PhysicalMaterial as Material,
)
from .geometry import SMPLBody

Geometry = CoreGeometry | SMPLBody


# Names that existed, were removed, and must not come back as a bare
# ImportError. A module-level __getattr__ turns `from witwin.radar import
# Tracer` into a message that names the replacement instead of a traceback that
# names a missing symbol.
_REMOVED = {
    "Tracer": (
        "witwin.radar.Tracer has been removed with the Dr.Jit ray tracer. "
        "Propagation now goes through the Channel consumer: build a "
        "witwin.radar.propagation.ChannelPropagationAdapter, freeze the legs, "
        "compose them with witwin.radar.paths.TwoWayComposer or "
        "DirectComposer, and synthesize with "
        "witwin.radar.synthesis.synthesize_fmcw_beat."
    ),
    "fresnel": (
        "witwin.radar.fresnel has been removed with the Dr.Jit ray tracer. "
        "Reflection coefficients are owned by Channel's native material "
        "evaluation and reach Radar inside a leg's transport coefficient; "
        "there is no Radar-side Fresnel function to call."
    ),
    # The radar-owned logical world, deleted in Phase 11. These four names must
    # not resolve to their Core counterparts: `witwin.radar.Scene` was a SECOND
    # logical world with its own structure list, its own dirty/version domain,
    # its own motion graph and its own compiler, and a caller who asks for it by
    # the old name is asking for a different object than `witwin.core.Scene`.
    "Scene": (
        "witwin.radar.Scene has been removed. The one logical world is "
        "witwin.core.Scene; build it there and hand it to Radar.simulate, "
        "which compiles it through witwin.radar.scene_binding.bind_radar_world "
        "and the Channel compile facade."
    ),
    "SceneModule": (
        "witwin.radar.SceneModule has been removed with witwin.radar.Scene. A "
        "torch.nn.Module that materializes a world builds a witwin.core.Scene "
        "in its forward and passes it to Radar.simulate; there is no "
        "radar-owned Scene subclass to derive from."
    ),
    "Timeline": (
        "witwin.radar.Timeline has been removed. Frame sequencing is the "
        "`times` argument of Radar.simulate, and world motion belongs to Core: "
        "witwin.core.RigidMotion, LinearTrajectory, Trajectory and "
        "DynamicScene."
    ),
    "TransformMotion": (
        "witwin.radar.TransformMotion has been removed with "
        "witwin.radar.Timeline. Structure motion is declared with Core's "
        "witwin.core.RigidMotion / LinearTrajectory on a DynamicScene, which "
        "the propagation epoch loop already consumes."
    ),
}


# Deployment reporting, exported LAZILY. `build_info` loads the native
# extension and `runtime_diagnostics` imports torch; binding either eagerly
# would make `import witwin.radar` pay for a native load, and would let a
# broken or missing prebuilt fail a bare import of the package root. The
# measured property that `import witwin.radar` loads neither
# `witwin.radar.cuda.build` nor any `witwin.channel` module is what
# acceptance criterion A2 rests on, so these three names resolve on first
# ACCESS rather than on import.
_LAZY = {
    "build_info": ("deployment", "build_info"),
    "runtime_diagnostics": ("deployment", "runtime_diagnostics"),
    "require_supported_runtime": ("deployment", "require_supported_runtime"),
    "capabilities": ("capabilities", "capabilities"),
}


def __getattr__(name: str):
    if name in _REMOVED:
        raise AttributeError(_REMOVED[name])
    if name in _LAZY:
        from importlib import import_module

        module_name, attribute = _LAZY[name]
        return getattr(import_module(f".{module_name}", __name__), attribute)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted({*globals(), *_LAZY})


__all__ = [
    'build_info',
    'capabilities',
    'require_supported_runtime',
    'runtime_diagnostics',
    'Radar',
    'RadarConfig',
    'quantize_complex_signal',
    'Solver',
    'TraceResult',
    'DetectorType',
    'MotionSampling',
    'SamplingMode',
    'MimoPathCache',
    'RadarPropagationLegs',
    'RadarSimulationResult',
    'RadarWorldBinding',
    'ScatterSitePolicy',
    'StableIdAllocator',
    'bind_radar_world',
    'SMPLBody',
    'Material',
    'Structure',
    'GeometryBase',
    'Geometry',
    'Mesh',
    'Box',
    'Sphere',
    'Cylinder',
    'Cone',
    'Ellipsoid',
    'Pyramid',
    'Prism',
    'Torus',
    'HollowBox',
]
