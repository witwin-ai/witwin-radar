"""
Smoke test: SMPL human body ray tracing + radar frame.

Usage:
    python -m radar.examples.humanbody

Requires: smplpytorch, mitsuba (cuda_ad_rgb variant)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "witwin-radar"))

import numpy as np
import torch
from core import Radar, Renderer, Scene
from sigproc import process_pc, process_rd

torch.set_default_device('cuda')

config = {
    "num_tx": 3, "num_rx": 4,
    "fc": 77e9,
    "slope": 60.012,
    "adc_samples": 256,
    "adc_start_time": 6,
    "sample_rate": 4400,
    "idle_time": 7,
    "ramp_end_time": 65,
    "chirp_per_frame": 128,
    "frame_per_second": 10,
    "num_doppler_bins": 128,
    "num_range_bins": 256,
    "num_angle_bins": 64,
    "power": 15,
    "tx_loc": [[0, 0, 0], [4, 0, 0], [2, 1, 0]],
    "rx_loc": [[-6, 0, 0], [-5, 0, 0], [-4, 0, 0], [-3, 0, 0]],
}
radar = Radar(config)

scene = Scene(fov=60)
scene.set_sensor(origin=(0, 0, 0), target=(0, 0, -5))

print("Adding SMPL body...")
pose = np.zeros(72)
shape = np.zeros(10)
scene.add_smpl("human", pose, shape, position=[0, -1, -3], gender='male')

renderer = Renderer(scene, resolution=128)

print("Ray tracing...")
points, intensities = renderer.trace()
assert points.shape[0] > 0, "No reflection points"
print(f"  {points.shape[0]} reflection points")

velocity = torch.tensor([0, 0, 0.005], device='cuda')

def location_function(t):
    return intensities, points + velocity * t

print("Generating MIMO frame...")
frame = radar.mimo(location_function, t0=0)
assert frame.shape == (3, 4, 128, 256), f"Unexpected frame shape: {frame.shape}"
print(f"  Frame shape: {frame.shape}  OK")

pc = process_pc(radar, frame)
print(f"  Point cloud: {pc.shape[0]} points")

rd_mag, _, _, _ = process_rd(radar, frame, tx=0, rx=0)
assert rd_mag.shape == (128, 256), f"Unexpected RD shape: {rd_mag.shape}"
print(f"  RD map: {rd_mag.shape}  OK")

print("PASSED")
