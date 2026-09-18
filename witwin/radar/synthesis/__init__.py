"""Explicit radar waveform synthesis facade."""

from .assembly import (
    FmcwSpec,
    OfdmSpec,
    PulsedSpec,
    SlowTimeMode,
    SynthesisPathBatch,
    SynthesisResult,
    select_component,
)

__all__ = [
    "FmcwSpec",
    "OfdmSpec",
    "PulsedSpec",
    "SlowTimeMode",
    "SynthesisPathBatch",
    "SynthesisResult",
    "select_component",
]
