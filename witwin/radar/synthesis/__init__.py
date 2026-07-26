"""Radar waveform synthesis.

The waveform hot loops are native CUDA kernels; this package holds their typed
descriptions, the one input contract all of them consume, and their single
Python owners.

:class:`SynthesisPathBatch` and :func:`require_compatible` are the shared part.
A waveform kernel never asks where its weight came from; it asks the batch, and
the batch refuses any spec that would apply a factor the weight already
carries.
"""

from .assembly import (
    FRAME_CUBE_AXES,
    PAIR_RANK_LAYOUT,
    assemble_frame_cube,
    pair_rx_index,
    pair_tx_index,
    validate_pair_ordering,
)
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
    "FRAME_CUBE_AXES",
    "PAIR_RANK_LAYOUT",
    "FmcwBeatSpec",
    "SlowTimeMode",
    "SynthesisPathBatch",
    "WaveformSpecProtocol",
    "assemble_frame_cube",
    "channel_phasor_to_beat_weight",
    "pair_rx_index",
    "pair_tx_index",
    "require_compatible",
    "synthesize_beat_rows",
    "synthesize_fmcw_beat",
]
