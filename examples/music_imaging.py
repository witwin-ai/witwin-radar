"""
Smoke test: MUSIC-based 2D radar imaging with 20x20 UPA.

Usage:
    python -m radar.examples.music_imaging
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "witwin-radar"))

import numpy as np
import torch
from core import Radar
from sigproc import MUSICImager

torch.set_default_device('cuda')

c0 = 299792458
fc = 77e9
spacing = c0 / fc / 2

config = {
    "num_tx": 20, "num_rx": 20,
    "fc": fc,
    "slope": 60.012,
    "adc_samples": 256,
    "adc_start_time": 6,
    "sample_rate": 4400,
    "idle_time": 7,
    "ramp_end_time": 65,
    "chirp_per_frame": 8,
    "frame_per_second": 10,
    "num_doppler_bins": 8,
    "num_range_bins": 256,
    "num_angle_bins": 64,
    "power": 15,
    "tx_loc": [[i, 0, 0] for i in range(20)],
    "rx_loc": [[20, -i, 0] for i in range(20)],
}
radar = Radar(config)

points = np.array([
    [-0.5, 0, -3],
    [0.5,  0, -3],
], dtype=np.float32)
velocity = np.array([0, 0, 0.01])


def location_function(t):
    pos = torch.tensor(points + velocity * t, dtype=torch.float32, device='cuda')
    intensity = torch.ones(pos.shape[0], dtype=torch.float32, device='cuda')
    return intensity, pos


print("Generating radar frame (20x20 MIMO)...")
frame = radar.mimo(location_function, t0=0)
assert frame.shape == (20, 20, 8, 256), f"Unexpected frame shape: {frame.shape}"
print(f"  Frame shape: {frame.shape}  OK")

print("Running MUSIC algorithm...")
imager = MUSICImager(num_tx=20, num_rx=20, num_signals=7,
                     spatial_smooth=3, num_pixels=128, num_chirps=8)
image3D = imager.radar_image(frame)
assert image3D.shape[0] == 128 and image3D.shape[1] == 128, f"Unexpected image shape: {image3D.shape}"
print(f"  Image shape: {image3D.shape}  OK")

print("PASSED")
