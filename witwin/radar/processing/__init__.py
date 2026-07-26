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
                    -> beam_cube -> ca_cfar -> point_cloud -> DetectionFrame

:class:`ProcessingAxes` is the one metadata / axes / units record all of them
read. It is built from the waveform SPECS - never from the flat engineering-unit
``RadarConfig``, which has exactly one documented conversion site - and it is
where the cross-waveform Doppler sign is fixed. :class:`ArrayGeometry` is the
matching statement about the array: where every virtual element is, in metres,
with no half-wavelength spacing assumed anywhere. This package also owns the
component combination laws that Phase-8 clutter export needs.

``witwin.radar.sigproc`` keeps its whole public surface as migration adapters
over this facade; the adapters are in :mod:`witwin.radar.processing.adapters`
and the old internal paths are deleted.
"""

from .aoa import (
    AOA_ROUTES,
    DIRECTION_COSINE_ROWS,
    fft2_aoa,
    music_image,
    music_spectrum,
    phase_comparison_aoa,
    tdm_compensate,
    upa_steering,
)
from .axes import ProcessingAxes
from .beam_cube import beam_cube
from .beamforming import ArrayGeometry, conventional_steering, mvdr_weights
from .cfar import Detections, ca_cfar, ca_cfar_1d, ca_cfar_fast, os_cfar
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
from .microdoppler import (
    dominant_frequencies_hz,
    doppler_frequencies_hz,
    microdoppler_spectrogram,
    slow_time_spectrum,
)
from .pointcloud import (
    POINT_CLOUD_COLUMNS,
    PointCloud,
    point_cloud,
    range_gate_mask,
)
from .primitives import WINDOWS
from .range_profile import range_profile
from .tracking import (
    Associator,
    DetectionFrame,
    TrackHandoff,
    nearest_neighbour_associator,
)

__all__ = [
    "AOA_ROUTES",
    "DIRECTION_COSINE_ROWS",
    "FAST_TIME_NAMES",
    "POINT_CLOUD_COLUMNS",
    "PROCESSING_AMPLITUDE_CONVENTION",
    "PROCESSING_DOPPLER_CONVENTION",
    "PROCESSING_UNITS",
    "SLOW_TIME_NAMES",
    "WINDOWS",
    "ArrayGeometry",
    "Associator",
    "BeamCube",
    "DetectionFrame",
    "Detections",
    "PointCloud",
    "ProcessingAxes",
    "ProcessingCube",
    "RangeDopplerMap",
    "RangeProfile",
    "TrackHandoff",
    "beam_cube",
    "ca_cfar",
    "ca_cfar_1d",
    "ca_cfar_fast",
    "combine_incoherent",
    "conventional_steering",
    "dominant_frequencies_hz",
    "doppler_frequencies_hz",
    "fft2_aoa",
    "microdoppler_spectrogram",
    "music_image",
    "music_spectrum",
    "mvdr_weights",
    "nearest_neighbour_associator",
    "os_cfar",
    "phase_comparison_aoa",
    "point_cloud",
    "range_doppler",
    "range_gate_mask",
    "range_profile",
    "slow_time_spectrum",
    "tdm_compensate",
    "upa_steering",
]
