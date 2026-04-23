"""
Smoke test: mesh scene ray tracing + radar frame + Timeline multi-frame.

Usage:
    python -m examples.mesh_scene
    python examples/mesh_scene.py

Requires: mitsuba (cuda_ad_rgb variant)
"""

import pathlib
import sys

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from witwin.radar import Radar, RadarConfig, Renderer, Scene, Timeline
from witwin.radar.sigproc import process_pc, process_rd
from witwin.core import Box

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

def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires CUDA for Mitsuba rendering and radar simulation.")


def main():
    require_cuda()
    radar = Radar(RadarConfig.from_dict(config), backend="dirichlet", device="cuda")

    scene = Scene()
    scene.set_sensor(origin=(0, 0, 0), target=(0, 0, -5), fov=60)

    wall_v, wall_f = Box(position=(0, 0, -5), size=(6, 4, 0.01)).to_mesh()
    scene.add_mesh(name="wall", vertices=wall_v, faces=wall_f)
    box_v, box_f = Box(position=(0.5, 0, -3), size=(0.8, 0.8, 0.8)).to_mesh()
    scene.add_mesh(name="box_a", vertices=box_v, faces=box_f)
    box2_v, box2_f = Box(position=(-1.0, -0.5, -4), size=(0.6, 1.0, 0.6)).to_mesh()
    scene.add_mesh(name="box_b", vertices=box2_v, faces=box2_f)

    renderer = Renderer(scene, resolution=128)

    print("Ray tracing...")
    points, intensities = renderer.trace()
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

    print("Timeline multi-frame...")
    timeline = Timeline(frame_rate=30)
    for i in range(3):
        offset = torch.tensor([0.0, 0.0, -0.5], dtype=torch.float32, device=radar.device) * (i / 30)
        timeline.add_keyframe(points + offset, intensities)

    frames = timeline.generate(radar, progress=False)
    print(f"  Timeline frames: {frames.shape}")
    assert frames.ndim == 5, f"Expected 5D output, got {frames.ndim}D"
    assert frames.shape[1:] == (3, 4, 128, 256), f"Unexpected per-frame shape: {frames.shape[1:]}"

    rd_mags, _, _ = timeline.generate_rd(radar, progress=False)
    print(f"  RD maps: {rd_mags.shape}")
    print("PASSED")


if __name__ == "__main__":
    main()
