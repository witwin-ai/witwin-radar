"""Radar waveform synthesis.

The waveform hot loops are native CUDA kernels; this package holds their typed
descriptions, the one input contract all of them consume, and their single
Python owners.

:class:`SynthesisPathBatch` and :func:`require_compatible` are the shared part.
A waveform kernel never asks where its weight came from; it asks the batch, and
the batch refuses any spec that would apply a factor the weight already
carries.

The three waveform products are in DIFFERENT phasor conventions, and each
carries its own as data rather than by reputation. The FMCW beat cube is
conjugated, because de-chirping multiplies the echo by the conjugate of the
transmitted chirp. The OFDM CFR cube is not, because per-subcarrier equalisation
removes the transmitted symbol but not the carrier convention. The pulsed echo
train is not either, because a pulsed receiver de-chirps nothing.

What differs between the three is exactly one factor of the received phasor -
the chirp ramp, the subcarrier phase, or the pulse envelope. The slow-time
factor is identical in all three, which is what the shared input contract is
for.
"""

from .assembly import (
    FRAME_CUBE_AXES,
    PAIR_RANK_LAYOUT,
    assemble_frame_cube,
    pair_rx_index,
    pair_slot_index,
    pair_tx_index,
    segment_of_each_row,
    tdm_slot_count,
    tdm_slot_times_s,
    validate_pair_ordering,
)
from .contracts import (
    BEAT_PHASOR,
    CHANNEL_PHASOR,
    CHANNEL_TIME_DEPENDENCE,
    FMCW_AXES,
    OFDM_AXES,
    PULSED_AXES,
    PULSE_KIND_LFM,
    PULSE_KIND_RECT,
    PULSE_KINDS,
    PULSE_NORMALIZATION_UNIT_ENERGY,
    SPEED_OF_LIGHT_M_PER_S,
    SUBCARRIER_ORIGIN_F_REF_AT_N0,
    FmcwBeatSpec,
    OfdmCfrSpec,
    PulsedEchoSpec,
    SlowTimeMode,
    SynthesisPathBatch,
    SynthesisResult,
    WaveformSpecProtocol,
    require_compatible,
    require_ofdm_compatible,
    require_pulsed_compatible,
    require_single_carrier_home,
)
from .fmcw_beat import (
    channel_phasor_to_beat_weight,
    synthesize_beat_rows,
    synthesize_fmcw_beat,
)
from .ofdm_cfr import synthesize_cfr_rows, synthesize_ofdm_cfr
from .pulsed_echo import synthesize_echo_rows, synthesize_pulsed_echo
from .selection import select_component

__all__ = [
    "BEAT_PHASOR",
    "CHANNEL_PHASOR",
    "CHANNEL_TIME_DEPENDENCE",
    "FMCW_AXES",
    "FRAME_CUBE_AXES",
    "OFDM_AXES",
    "PULSED_AXES",
    "PAIR_RANK_LAYOUT",
    "PULSE_KINDS",
    "PULSE_KIND_LFM",
    "PULSE_KIND_RECT",
    "PULSE_NORMALIZATION_UNIT_ENERGY",
    "SPEED_OF_LIGHT_M_PER_S",
    "SUBCARRIER_ORIGIN_F_REF_AT_N0",
    "FmcwBeatSpec",
    "OfdmCfrSpec",
    "PulsedEchoSpec",
    "SlowTimeMode",
    "SynthesisPathBatch",
    "SynthesisResult",
    "WaveformSpecProtocol",
    "assemble_frame_cube",
    "channel_phasor_to_beat_weight",
    "pair_rx_index",
    "pair_slot_index",
    "pair_tx_index",
    "require_compatible",
    "require_ofdm_compatible",
    "require_pulsed_compatible",
    "require_single_carrier_home",
    "segment_of_each_row",
    "select_component",
    "tdm_slot_count",
    "tdm_slot_times_s",
    "validate_pair_ordering",
    "synthesize_beat_rows",
    "synthesize_cfr_rows",
    "synthesize_echo_rows",
    "synthesize_fmcw_beat",
    "synthesize_ofdm_cfr",
    "synthesize_pulsed_echo",
]
