"""Deprecated: micro-Doppler slow-time spectra and spectrograms.

Moved to :mod:`witwin.radar.processing.microdoppler` at the Phase-8 cutover,
unchanged in values and conventions. A slow-time spectrum IS processing, and
after the cutover every production ``torch.fft`` expression in the radar
processing chain lives under ``witwin/radar/processing/``.
"""

from __future__ import annotations

from ..processing.microdoppler import (
    WINDOWS,
    dominant_frequencies_hz,
    doppler_frequencies_hz,
    microdoppler_spectrogram,
    slow_time_spectrum,
)

__all__ = [
    "WINDOWS",
    "dominant_frequencies_hz",
    "doppler_frequencies_hz",
    "microdoppler_spectrogram",
    "slow_time_spectrum",
]
