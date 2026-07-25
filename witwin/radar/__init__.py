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
from .scene import Scene, SceneModule
from .timeline import Timeline, TransformMotion
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
}


def __getattr__(name: str):
    if name in _REMOVED:
        raise AttributeError(_REMOVED[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    'Radar',
    'RadarConfig',
    'quantize_complex_signal',
    'Solver',
    'TraceResult',
    'DetectorType',
    'MotionSampling',
    'SamplingMode',
    'MimoPathCache',
    'Scene',
    'SceneModule',
    'SMPLBody',
    'Timeline',
    'TransformMotion',
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
