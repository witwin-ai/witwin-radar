"""Radar waveform synthesis.

The FMCW beat hot loop is a native CUDA kernel; this package holds its typed
description and its single Python owner.
"""

from .contracts import FmcwBeatSpec

__all__ = ["FmcwBeatSpec"]
