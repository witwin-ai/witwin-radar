"""
Smoke test: SMPL human body ray tracing + radar frame.

Usage:
    python -m examples.humanbody
    python examples/humanbody.py

Requires: smplpytorch, mitsuba (cuda_ad_rgb variant)
"""

import pathlib
import sys

import numpy as np
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from witwin.radar import Material, Radar, RadarConfig, Scene, Tracer
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

MODEL_ROOT = REPO_ROOT / "models" / "smpl_models"


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires CUDA for Mitsuba rendering and radar simulation.")


def main():
    require_cuda()
    if not MODEL_ROOT.exists():
        raise FileNotFoundError(f"SMPL models not found: {MODEL_ROOT}")

    radar = Radar(RadarConfig.from_dict(config), backend="dirichlet", device="cuda", target=(0, 0, -5), fov=60)

    scene = Scene()

    print("Adding SMPL body...")
    pose = np.zeros(72, dtype=np.float32)
    shape = np.zeros(10, dtype=np.float32)
    scene.add_smpl(
        name="human",
        pose=pose,
        shape=shape,
        position=[0, -1, -3],
        gender="male",
        model_root=str(MODEL_ROOT),
        material=Material(eps_r=3.0),
    )

    tracer = Tracer(scene, radar, resolution=128)

    print("Ray tracing...")
    points, intensities = tracer.trace()
    assert points.shape[0] > 0, "No reflection points"
    print(f"  {points.shape[0]} reflection points")

    velocity = torch.tensor([0.0, 0.0, 0.005], dtype=torch.float32, device=radar.device)

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


if __name__ == "__main__":
    main()
