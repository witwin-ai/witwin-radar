"""
Radar signal processing: the legacy public surface, on the Phase-8 facade.

Every name below is preserved and every one of them is now a thin migration
adapter over ``witwin.radar.processing``. The adapters live in
``witwin/radar/processing/adapters.py``; this package re-exports them and
computes nothing, so a static scan can say that after the Phase-8 cutover every
production ``torch.fft``, detector, angle estimator and beamformer in the radar
tree is inside the processing facade.

The replacements, name for name:

- ``range_fft`` / ``doppler_fft`` -> ``processing.range_profile`` and
  ``processing.range_doppler`` (windowed and amplitude normalised, with the
  cross-waveform Doppler sign reconciled);
- ``clutter_removal`` -> ``processing.range_profile(remove_dc=True)``;
- ``naive_xyz`` -> ``processing.aoa``;
- ``ca_cfar_2d`` / ``ca_cfar_2d_fast`` / ``os_cfar_2d`` -> ``processing.cfar``,
  batched over ``[..., D, R]``;
- ``frame2pointcloud`` / ``process_pc`` / ``process_pc_tensor`` ->
  ``processing.point_cloud``;
- ``process_rd`` / ``process_rd_tensor`` -> ``processing.range_doppler``;
- ``reg_data`` -> ``processing.DetectionFrame.as_fixed_size``;
- ``MUSICImager`` -> ``processing.music_spectrum``;
- the four micro-Doppler entries -> ``processing.microdoppler``.
"""

from .microdoppler import (
    dominant_frequencies_hz,
    doppler_frequencies_hz,
    microdoppler_spectrogram,
    slow_time_spectrum,
)
from .pointcloud import (
    FrameConfig,
    PointCloudProcessConfig,
    frame2pointcloud,
    process_pc,
    process_pc_tensor,
    process_rd,
    process_rd_tensor,
    reg_data,
    range_fft,
    doppler_fft,
    clutter_removal,
    naive_xyz,
)
from .cfar import ca_cfar_2d, ca_cfar_2d_fast, os_cfar_2d
from .music import MUSICImager

__all__ = [
    'FrameConfig',
    'PointCloudProcessConfig',
    'frame2pointcloud',
    'process_pc',
    'process_pc_tensor',
    'process_rd',
    'process_rd_tensor',
    'reg_data',
    'range_fft',
    'doppler_fft',
    'clutter_removal',
    'naive_xyz',
    'ca_cfar_2d',
    'ca_cfar_2d_fast',
    'os_cfar_2d',
    'MUSICImager',
    'dominant_frequencies_hz',
    'doppler_frequencies_hz',
    'microdoppler_spectrogram',
    'slow_time_spectrum',
]
