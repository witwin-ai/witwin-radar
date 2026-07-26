"""
Radar signal processing modules.

- pointcloud: Range-Doppler FFT processing and 3D point cloud extraction
- cfar: CFAR detectors (CA-CFAR, OS-CFAR) for Range-Doppler maps
- music: MUSIC-based 2D radar imaging for Uniform Planar Arrays
- microdoppler: slow-time spectra and spectrograms of a micro-Doppler signal
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
