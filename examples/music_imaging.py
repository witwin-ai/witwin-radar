"""
Smoke test: MUSIC-based 2D radar imaging with 20x20 UPA.

Usage:
    python -m examples.music_imaging
    python examples/music_imaging.py
"""

import pathlib
import sys

import numpy as np
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from witwin.radar import Radar
from witwin.radar.sigproc import MUSICImager

c0 = 299792458
fc = 77e9

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
points = np.array([
    [-0.5, 0, -3],
    [0.5,  0, -3],
], dtype=np.float32)
velocity = np.array([[0, 0, 0.01]], dtype=np.float32)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backend = "dirichlet" if device == "cuda" else "pytorch"
    radar = Radar(config, backend=backend, device=device)

    def location_function(t):
        pos = torch.tensor(points + velocity * t, dtype=torch.float32, device=radar.device)
        intensity = torch.ones(pos.shape[0], dtype=torch.float32, device=radar.device)
        return intensity, pos

    print(f"Using backend={backend} device={device}")
    print("Generating radar frame (20x20 MIMO)...")
    frame = radar.mimo(location_function, t0=0)
    assert frame.shape == (20, 20, 8, 256), f"Unexpected frame shape: {frame.shape}"
    print(f"  Frame shape: {frame.shape}  OK")

    print("Running MUSIC algorithm...")
    imager = MUSICImager(num_tx=20, num_rx=20, num_signals=7, spatial_smooth=3, num_pixels=128, num_chirps=8)
    image3d = imager.radar_image(frame)
    assert image3d.shape[0] == 128 and image3d.shape[1] == 128, f"Unexpected image shape: {image3d.shape}"
    print(f"  Image shape: {image3d.shape}  OK")
    print("PASSED")


if __name__ == "__main__":
    main()
