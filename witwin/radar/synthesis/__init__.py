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
from .fmcw import synthesize_fmcw
from .ofdm import synthesize_ofdm
from .pulsed import synthesize_pulsed

__all__ = [
    "FmcwSpec",
    "OfdmSpec",
    "PulsedSpec",
    "SlowTimeMode",
    "SynthesisPathBatch",
    "SynthesisResult",
    "select_component",
    "synthesize_fmcw",
    "synthesize_ofdm",
    "synthesize_pulsed",
]
