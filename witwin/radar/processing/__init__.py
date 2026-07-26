"""Radar post-processing.

Everything in this package is PyTorch by owner directive: range profiles,
Range-Doppler maps, beam cubes, AoA, CFAR, point clouds and detection handoff
are post-processing, not simulation, and a native DSP kernel needs a measured
dispatch, layout, fusion or tape bottleneck plus its own decision record before
it can exist.

Processing CONSUMES synthesis results. It never mutates a path batch, never
changes composed row identity, and publishes no field that crosses back into
the Channel capability record, its public API, or its native binding manifest.

The chain, in order:

    SynthesisResult -> ProcessingCube -> range_profile -> range_doppler
                    -> beam_cube -> (detection, point cloud, tracking: stage S4)

:class:`ProcessingAxes` is the one metadata / axes / units record all of them
read. It is built from the waveform SPECS - never from the flat engineering-unit
``RadarConfig``, which has exactly one documented conversion site - and it is
where the cross-waveform Doppler sign is fixed. This package also owns the
component combination laws that Phase-8 clutter export needs.
"""

from .axes import ProcessingAxes
from .beam_cube import beam_cube, conventional_steering, virtual_element_offsets_m
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
from .doppler import range_doppler
from .primitives import WINDOWS
from .range_profile import range_profile

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
    "beam_cube",
    "combine_incoherent",
    "conventional_steering",
    "range_doppler",
    "range_profile",
    "virtual_element_offsets_m",
]
