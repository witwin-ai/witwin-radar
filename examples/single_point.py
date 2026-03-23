"""
Smoke test: single-point radar simulation.

Usage:
    python -m examples.single_point
    python examples/single_point.py
"""

import pathlib
import sys

import numpy as np
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from witwin.radar import Radar
from witwin.radar.sigproc import process_pc, process_rd

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
device = "cuda" if torch.cuda.is_available() else "cpu"
backend = "dirichlet" if device == "cuda" else "pytorch"

point = np.array([[0, 0, -3]], dtype=np.float32)
velocity = np.array([[0, 0, 0.01]], dtype=np.float32)

def main():
    radar = Radar(config, backend=backend, device=device)

    def location_function(t):
        pos = torch.tensor(point + velocity * t, dtype=torch.float32, device=radar.device)
        intensity = torch.ones(pos.shape[0], dtype=torch.float32, device=radar.device)
        return intensity, pos

    print(f"Using backend={backend} device={device}")
    print("Generating MIMO frame...")
    frame = radar.mimo(location_function, t0=0)
    assert frame.shape == (3, 4, 128, 256), f"Unexpected frame shape: {frame.shape}"
    print(f"  Frame shape: {frame.shape}  OK")

    pc = process_pc(radar, frame)
    print(f"  Point cloud: {pc.shape[0]} points")

    rd_mag, _, _, _ = process_rd(radar, frame, tx=0, rx=0)
    assert rd_mag.shape == (128, 256), f"Unexpected RD shape: {rd_mag.shape}"
    print(f"  RD map shape: {rd_mag.shape}  OK")
    print("PASSED")


if __name__ == "__main__":
    main()
