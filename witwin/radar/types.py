"""Public string-literal types and validators for radar APIs."""

from __future__ import annotations

from typing import Literal, TypeAlias, cast


SolverBackend: TypeAlias = Literal["pytorch", "slang", "dirichlet"]
DetectorType: TypeAlias = Literal["cfar", "topk"]
SamplingMode: TypeAlias = Literal["pixel", "triangle"]
MotionSampling: TypeAlias = Literal["frame", "chirp"]


_SOLVER_BACKENDS = ("pytorch", "slang", "dirichlet")
_DETECTOR_TYPES = ("cfar", "topk")
_SAMPLING_MODES = ("pixel", "triangle")
_MOTION_SAMPLING_MODES = ("frame", "chirp")


def normalize_solver_backend(value: str) -> SolverBackend:
    backend = str(value).lower()
    if backend not in _SOLVER_BACKENDS:
        raise ValueError(f"Unknown backend: {value}")
    return cast(SolverBackend, backend)


def normalize_detector_type(value: str) -> DetectorType:
    detector = str(value).lower()
    if detector not in _DETECTOR_TYPES:
        raise ValueError(f"Unsupported detector '{value}'. Expected 'cfar' or 'topk'.")
    return cast(DetectorType, detector)


def normalize_sampling_mode(value: str) -> SamplingMode:
    sampling = str(value).lower()
    if sampling not in _SAMPLING_MODES:
        raise ValueError(f"Unsupported sampling mode '{value}'.")
    return cast(SamplingMode, sampling)


def normalize_motion_sampling(value: str) -> MotionSampling:
    sampling = str(value).lower()
    if sampling not in _MOTION_SAMPLING_MODES:
        raise ValueError(f"Unsupported motion sampling mode '{value}'.")
    return cast(MotionSampling, sampling)
