"""
AMASS BMLmovi: multi-frame radar point cloud from SMPL body motion capture.

Loads SMPL pose sequences from the BMLmovi subset of AMASS, drives a full
SMPL mesh through Mitsuba ray tracing, and runs the FMCW radar pipeline
to produce 30 consecutive radar point clouds with Range-Doppler maps.

Usage (from repo root):
    python -m examples.amass_pointcloud
    python examples/amass_pointcloud.py

Requires:
    - BMLmovi data in data/BMLmovi_full/ (extracted from BMLmovi.tar.bz2)
    - SMPL model files in models/smpl_models/
    - smplpytorch, mitsuba (cuda_ad_rgb variant)
"""

import pathlib
import sys

import numpy as np
import torch

repo_root = pathlib.Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from witwin.radar import Material, Radar, RadarConfig, Scene, Tracer
from witwin.radar.sigproc import process_pc

MODEL_ROOT = repo_root / "models" / "smpl_models"
DATA_DIR = repo_root / "data" / "BMLmovi_full" / "BMLmovi"


def fix_pose(pose):
    fixed = pose.copy()
    fixed[:3] = 0.0
    return fixed


# ---- Radar config ----
config = {
    "num_tx": 3, "num_rx": 4, "fc": 77e9, "slope": 60.012,
    "adc_samples": 256, "adc_start_time": 6, "sample_rate": 4400,
    "idle_time": 7, "ramp_end_time": 65, "chirp_per_frame": 128,
    "frame_per_second": 10, "num_doppler_bins": 128, "num_range_bins": 256,
    "num_angle_bins": 64, "power": 15,
    "tx_loc": [[0, 0, 0], [4, 0, 0], [2, 1, 0]],
    "rx_loc": [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]],
}


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires CUDA for Mitsuba rendering and radar simulation.")


def main():
    require_cuda()
    if not MODEL_ROOT.exists():
        raise FileNotFoundError(f"SMPL models not found: {MODEL_ROOT}")
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"AMASS BMLmovi data not found: {DATA_DIR}")

    subjects = sorted(DATA_DIR.iterdir())
    if not subjects:
        raise FileNotFoundError(f"No subjects found in {DATA_DIR}")

    subj_dir = subjects[0]
    pose_files = sorted(subj_dir.glob("*_poses.npz"))
    if not pose_files:
        raise FileNotFoundError(f"No pose files found in {subj_dir}")

    pose_data = np.load(str(pose_files[0]))
    shape_data = np.load(str(subj_dir / "shape.npz"))

    all_poses = pose_data["poses"][:, :72]
    betas = shape_data["betas"][:10]
    gender = str(shape_data["gender"])
    source_fps = float(pose_data["mocap_framerate"])
    print(f"Loaded {pose_files[0].name}: {all_poses.shape[0]} frames at {source_fps} fps, gender={gender}")

    radar = Radar(RadarConfig.from_dict(config), backend="dirichlet", device="cuda", target=(0, 0, -5), fov=80)

    step = int(source_fps / config["frame_per_second"])
    num_frames = 30
    start = 50
    indices = [start + i * step for i in range(num_frames + 1)]
    assert indices[-1] < all_poses.shape[0]

    t_chirp = (config["idle_time"] + config["ramp_end_time"]) * 1e-6
    t_frame = t_chirp * config["num_tx"] * config["chirp_per_frame"]
    vel_scale = t_frame / (step / source_fps)
    translation = [0, -0.1, -3]

    scene = Scene()
    scene.add_smpl(
        name="human",
        pose=fix_pose(all_poses[indices[0]]),
        shape=betas,
        position=translation,
        gender=gender,
        model_root=str(MODEL_ROOT),
        material=Material(eps_r=3.0),
    )
    tracer = Tracer(scene, radar, resolution=256)

    print(f"Generating {num_frames} radar frames...")
    all_pcs = []
    for fi in range(num_frames):
        src_cur, src_next = indices[fi], indices[fi + 1]

        scene.update_structure("human", pose=fix_pose(all_poses[src_cur]), shape=betas, position=translation)
        pts_cur, int_cur = tracer.trace()

        scene.update_structure("human", pose=fix_pose(all_poses[src_next]), shape=betas, position=translation)
        pts_next, _ = tracer.trace()

        n = min(pts_cur.shape[0], pts_next.shape[0])
        if n == 0:
            all_pcs.append(np.zeros((0, 6), dtype=np.float32))
            continue

        p0, p1, inten = pts_cur[:n], pts_next[:n], int_cur[:n]
        p1_scaled = p0 + (p1 - p0) * vel_scale

        def interpolator(t, _p0=p0, _p1=p1_scaled, _i=inten):
            return _i, _p0 + (_p1 - _p0) * (t / t_frame)

        frame = radar.mimo(interpolator, t0=0)
        pc = process_pc(radar, frame, static_clutter_removal=False, positive_velocity_only=False)
        all_pcs.append(pc)

        if fi % 10 == 0:
            print(f"  [{fi:2d}/{num_frames}] {pts_cur.shape[0]} ray pts -> {pc.shape[0]} PC pts")

    print(f"Done. Total: {sum(pc.shape[0] for pc in all_pcs)} points across {num_frames} frames")


if __name__ == "__main__":
    main()
