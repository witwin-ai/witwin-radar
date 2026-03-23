"""Manual verification for pixel and triangle radar tracing modes."""

import os

import numpy as np
import torch

from witwin.radar import Radar, Renderer, Scene
from witwin.radar.sigproc import process_rd

torch.set_default_device("cuda")


MODEL_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "smpl_models")

RADAR_CONFIG = {
    "num_tx": 3,
    "num_rx": 4,
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

TRANSLATION = (0.0, -0.1, -3.0)


def fix_pose(pose):
    fixed = pose.copy()
    fixed[:3] = 0.0
    return fixed


def make_scene(pose, shape, gender="male"):
    return (
        Scene()
        .set_sensor(origin=(0, 0, 0), target=(0, 0, -5), fov=80)
        .add_smpl(
            name="human",
            pose=fix_pose(pose),
            shape=shape,
            position=TRANSLATION,
            gender=gender,
            model_root=MODEL_ROOT,
        )
    )


def test_static_comparison():
    print("=" * 60)
    print("Test 1: Static body comparison (T-pose)")
    print("=" * 60)

    pose = np.zeros(72, dtype=np.float32)
    shape = np.zeros(10, dtype=np.float32)
    scene = make_scene(pose, shape)

    renderer_pixel = Renderer(scene, resolution=256, sampling="pixel")
    pts_pixel, int_pixel = renderer_pixel.trace()

    renderer_triangle = Renderer(scene, resolution=256, sampling="triangle")
    pts_triangle, int_triangle = renderer_triangle.trace()

    print(f"Pixel mode:    {pts_pixel.shape[0]} points, energy = {int_pixel.sum().item():.4f}")
    print(f"Triangle mode: {pts_triangle.shape[0]} points, energy = {int_triangle.sum().item():.6f}")

    for name, points in [("Pixel", pts_pixel), ("Triangle", pts_triangle)]:
        bounds_min = points.min(dim=0).values.cpu().numpy()
        bounds_max = points.max(dim=0).values.cpu().numpy()
        print(
            f"  {name} bounds: X=[{bounds_min[0]:.3f},{bounds_max[0]:.3f}] "
            f"Y=[{bounds_min[1]:.3f},{bounds_max[1]:.3f}] Z=[{bounds_min[2]:.3f},{bounds_max[2]:.3f}]"
        )

    print()
    return True


def test_velocity_tracking():
    print("=" * 60)
    print("Test 2: Moving body velocity tracking")
    print("=" * 60)

    shape = np.zeros(10, dtype=np.float32)
    pose0 = np.zeros(72, dtype=np.float32)
    pose1 = pose0.copy()
    pose1[18 * 3] = 1.5

    for mode in ["pixel", "triangle"]:
        scene = make_scene(pose0, shape)
        renderer = Renderer(scene, resolution=256, sampling=mode)

        trace0 = renderer.trace()
        scene.update_structure("human", pose=fix_pose(pose1), shape=shape, position=TRANSLATION)
        trace1 = renderer.trace()

        p0, p1, intensities = renderer.match(trace0, trace1)
        displacement = (p1 - p0).norm(dim=1)
        num_points = p0.shape[0]
        num_large = (displacement > 0.5).sum().item()

        print(f"{mode:8s}: {num_points} matched points")
        print(
            f"  Displacement: mean={displacement.mean().item():.4f}, "
            f"max={displacement.max().item():.4f}, std={displacement.std().item():.4f}"
        )
        print(f"  Points with displacement > 0.5m: {num_large} ({100 * num_large / max(num_points, 1):.1f}%)")

    print()
    return True


def test_end_to_end():
    print("=" * 60)
    print("Test 3: End-to-end radar pipeline")
    print("=" * 60)

    radar = Radar(RADAR_CONFIG)
    shape = np.zeros(10, dtype=np.float32)
    pose0 = np.zeros(72, dtype=np.float32)
    pose1 = pose0.copy()
    pose1[18 * 3] = 1.0

    chirp_period = (radar.idle_time + radar.ramp_end_time) * 1e-6
    frame_period = chirp_period * radar.num_tx * radar.chirp_per_frame

    for mode in ["pixel", "triangle"]:
        scene = make_scene(pose0, shape)
        renderer = Renderer(scene, resolution=256, sampling=mode)

        trace0 = renderer.trace()
        scene.update_structure("human", pose=fix_pose(pose1), shape=shape, position=TRANSLATION)
        trace1 = renderer.trace()

        p0, p1, intensities = renderer.match(trace0, trace1)

        def make_interp(start_points, end_points, weights, duration):
            def fn(t):
                frac = t / duration if duration > 0 else 0.0
                return weights, start_points + (end_points - start_points) * frac

            return fn

        frame = radar.mimo(make_interp(p0, p1, intensities, frame_period), t0=0)
        rd_mag, _, _, _ = process_rd(radar, frame, tx=0, rx=0)

        peak_db = rd_mag.max()
        num_active = (rd_mag > peak_db - 20).sum()
        print(
            f"  {mode:8s}: RD map shape={rd_mag.shape}, "
            f"peak={peak_db:.1f} dB, cells within 20dB={num_active}"
        )

    print()
    return True


if __name__ == "__main__":
    print("Triangle Sampling Mode Verification\n")
    ok = True
    ok &= test_static_comparison()
    ok &= test_velocity_tracking()
    ok &= test_end_to_end()
    print("All tests passed." if ok else "Some tests failed.")
