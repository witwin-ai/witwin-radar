"""Public radar API.

The Dr.Jit ray tracer is GONE, not deprecated in place. ``Tracer`` and
``fresnel`` were the production entry points that reached it, and both now
raise with a pointer to their replacement. Nothing silently falls back: a
caller who used the tracer gets an error naming the route that replaces it,
which is the only honest outcome when the replacement is a different contract
rather than a drop-in.

Removing the import is what makes the process-global "no Dr.Jit anywhere"
assertion possible at all. It used to be unachievable for one reason: importing
any ``witwin.radar`` submodule initialized this file, which imported
``trace.py``, which imported ``drjit``.

Phase 11 applied the same rule to the Dirichlet route. ``Solver``,
``TraceResult``, ``MimoPathCache``, the radar-owned ``Scene`` and ``Timeline``,
and the ``Radar.mimo*`` / ``chirp`` / ``frame`` / ``waveform`` methods are all
gone. ``Radar.simulate`` is the entry point, and it is a different contract: a
Core world and a declared frame sequence rather than a callable that reports
scatterer positions at a time.
"""

from .radar import Radar, RadarConfig
from .scene_binding import (
    RadarWorldBinding,
    ScatterSitePolicy,
    StableIdAllocator,
    bind_radar_world,
)
from .simulation import RadarSimulationResult
from .propagation.contracts import RadarPropagationLegs
from .types import DetectorType
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
    # The Dirichlet route, deleted in Phase 11. Its replacement is not a
    # renamed function: `Radar.simulate` reads a Core world, so a caller who
    # held an interpolator has to say where their targets are instead.
    "Solver": (
        "witwin.radar.Solver has been removed with the Dirichlet route. There "
        "is no backend selector: Radar.simulate is the one production entry "
        "point, and it goes Core Scene -> Channel CompiledScene -> propagation "
        "-> two-way join -> synthesis."
    ),
    "TraceResult": (
        "witwin.radar.TraceResult has been removed with the Dirichlet route. "
        "Scatterers are declared as sites of a "
        "witwin.radar.ScatterSitePolicy over a witwin.core.Scene, not as a "
        "per-frame sample of points and intensities."
    ),
    "MimoPathCache": (
        "witwin.radar.MimoPathCache has been removed with the Dirichlet "
        "route. The scene-driven pipeline caches a frozen topology per epoch "
        "inside witwin.radar.propagation, and Radar.last_propagation is how "
        "you look at it."
    ),
    "SamplingMode": (
        "witwin.radar.SamplingMode has been removed with the Dirichlet "
        "route's interpolator contract. It had no consumer in the tree."
    ),
    "MotionSampling": (
        "witwin.radar.MotionSampling has been removed with the Dirichlet "
        "route's interpolator contract. Frame instants are the `times` "
        "argument of Radar.simulate and intra-frame motion is an ADR-038 "
        "forward dual over the site tensor."
    ),
    "quantize_complex_signal": (
        "witwin.radar.quantize_complex_signal has been removed with the "
        "legacy receive chain. The ADC is a stage of "
        "witwin.radar.frontend.FrontendChain, configured by the `frontend` "
        "block, and it runs in the native frontend_quantize operator."
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
    'DetectorType',
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
