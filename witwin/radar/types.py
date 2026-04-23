"""Public string-literal types and validators for radar APIs."""

from __future__ import annotations

from typing import Literal, TypeAlias, cast


SolverBackend: TypeAlias = Literal["pytorch", "slang", "dirichlet"]
DetectorType: TypeAlias = Literal["cfar", "topk"]
SamplingMode: TypeAlias = Literal["pixel", "triangle"]
MotionSampling: TypeAlias = Literal["frame", "chirp"]


def _check_literal(value, *, allowed: tuple[str, ...], label: str) -> str:
    normalized = str(value).lower()
    if normalized not in allowed:
        raise ValueError(f"Unsupported {label} '{value}'. Expected one of: {', '.join(allowed)}.")
    return normalized


def normalize_solver_backend(value: str) -> SolverBackend:
    return cast(SolverBackend, _check_literal(value, allowed=("pytorch", "slang", "dirichlet"), label="backend"))


def normalize_detector_type(value: str) -> DetectorType:
    return cast(DetectorType, _check_literal(value, allowed=("cfar", "topk"), label="detector"))


def normalize_sampling_mode(value: str) -> SamplingMode:
    return cast(SamplingMode, _check_literal(value, allowed=("pixel", "triangle"), label="sampling mode"))


def normalize_motion_sampling(value: str) -> MotionSampling:
    return cast(MotionSampling, _check_literal(value, allowed=("frame", "chirp"), label="motion sampling mode"))
