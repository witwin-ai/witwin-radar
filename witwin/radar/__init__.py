"""Public radar API."""

from .antenna_pattern import AntennaPatternConfig
from .config import RadarConfig
from .noise import NoiseModelConfig, PhaseNoiseConfig, QuantizationNoiseConfig, ThermalNoiseConfig
from .polarization import PolarizationConfig
from .receiver_chain import AGCConfig, LNAConfig, ReceiverChainConfig
from .radar import Radar
from .result import MultiResult, Result
from .simulation import RadarSpec, RenderOptions, Simulation
from .solvers import Solver
from .renderer import Renderer, TraceResult
from .material import fresnel
from .motion import RotationMotion, StructureMotion, TranslationMotion
from .scene import Scene, SceneModule
from .sensor import Sensor
from .timeline import Timeline
from .types import DetectorType, MotionSampling, SamplingMode, SolverBackend
from witwin.core import (
    Box,
    Cone,
    Cylinder,
    Ellipsoid,
    Geometry,
    GeometryBase,
    HollowBox,
    Material,
    Mesh,
    Prism,
    Pyramid,
    SMPLBody,
    Sphere,
    Structure,
    Torus,
)

__all__ = [
    'Radar',
    'RadarConfig',
    'AntennaPatternConfig',
    'NoiseModelConfig',
    'ThermalNoiseConfig',
    'QuantizationNoiseConfig',
    'PhaseNoiseConfig',
    'PolarizationConfig',
    'LNAConfig',
    'AGCConfig',
    'ReceiverChainConfig',
    'Result',
    'MultiResult',
    'Simulation',
    'RadarSpec',
    'RenderOptions',
    'Solver',
    'Renderer',
    'TraceResult',
    'SolverBackend',
    'DetectorType',
    'MotionSampling',
    'SamplingMode',
    'Scene',
    'SceneModule',
    'Sensor',
    'SMPLBody',
    'Timeline',
    'TranslationMotion',
    'RotationMotion',
    'StructureMotion',
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
