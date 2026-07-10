"""Public radar API."""

from .radar import Radar, RadarConfig, quantize_complex_signal
from .solvers import Solver
from .trace import TraceResult, Tracer
from .material import fresnel
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
    Material,
    Mesh,
    Prism,
    Pyramid,
    Sphere,
    Structure,
    Torus,
)
from .geometry import SMPLBody

Geometry = CoreGeometry | SMPLBody

__all__ = [
    'Radar',
    'RadarConfig',
    'quantize_complex_signal',
    'Solver',
    'Tracer',
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
    'fresnel',
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
