"""Radar post-processing.

Everything in this package is PyTorch by owner directive: range profiles,
Range-Doppler maps, beam cubes, AoA, CFAR, point clouds and detection handoff
are post-processing, not simulation, and a native DSP kernel needs a measured
dispatch, layout, fusion or tape bottleneck plus its own decision record before
it can exist.

Processing CONSUMES synthesis results. It never mutates a path batch, never
changes composed row identity, and publishes no field that crosses back into
the Channel capability record, its public API, or its native binding manifest.

:class:`ProcessingAxes` is the one metadata / axes / units record every stage
reads. It is built from the waveform SPECS - never from the flat
engineering-unit ``RadarConfig``, which has exactly one documented conversion
site - and it is where the cross-waveform Doppler sign is fixed.
:class:`ProcessingCube` is where the chain attaches to ``SynthesisResult``. The
range / Doppler / beam stages land on top of them; this package also owns the
component combination laws that Phase-8 clutter export needs.
"""

from .axes import ProcessingAxes
from .combination import combine_incoherent
from .contracts import (
    FAST_TIME_NAMES,
    PROCESSING_AMPLITUDE_CONVENTION,
    PROCESSING_DOPPLER_CONVENTION,
    PROCESSING_UNITS,
    SLOW_TIME_NAMES,
    BeamCube,
    RangeDopplerMap,
    RangeProfile,
)
from .cube import ProcessingCube
from .primitives import WINDOWS

__all__ = [
    "FAST_TIME_NAMES",
    "PROCESSING_AMPLITUDE_CONVENTION",
    "PROCESSING_DOPPLER_CONVENTION",
    "PROCESSING_UNITS",
    "SLOW_TIME_NAMES",
    "WINDOWS",
    "BeamCube",
    "ProcessingAxes",
    "ProcessingCube",
    "RangeDopplerMap",
    "RangeProfile",
    "combine_incoherent",
]
