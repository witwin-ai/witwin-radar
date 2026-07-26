"""Radar waveform synthesis.

The waveform hot loops are native CUDA kernels; this package holds their typed
descriptions, the one input contract all of them consume, and their single
Python owners.

:class:`SynthesisPathBatch` and :func:`require_compatible` are the shared part.
A waveform kernel never asks where its weight came from; it asks the batch, and
the batch refuses any spec that would apply a factor the weight already
carries.
"""

from .contracts import (
    FmcwBeatSpec,
    SlowTimeMode,
    SynthesisPathBatch,
    WaveformSpecProtocol,
    require_compatible,
)
from .fmcw_beat import (
    channel_phasor_to_beat_weight,
    synthesize_beat_rows,
    synthesize_fmcw_beat,
)

__all__ = [
    "FmcwBeatSpec",
    "SlowTimeMode",
    "SynthesisPathBatch",
    "WaveformSpecProtocol",
    "channel_phasor_to_beat_weight",
    "require_compatible",
    "synthesize_beat_rows",
    "synthesize_fmcw_beat",
]
