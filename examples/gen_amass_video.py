"""
Generate combined radar visualization videos from AMASS BMLmovi clips.

Each clip runs the full pipeline:
AMASS pose -> SMPL mesh -> Mitsuba ray tracing -> FMCW radar ->
CA-CFAR detection -> combined MP4 with reflections, RD map, and radar point cloud.

Usage:
    python -m examples.gen_amass_video
    python examples/gen_amass_video.py
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

matplotlib.use("Agg")

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from witwin.radar import Radar, Renderer, Scene
from witwin.radar.sigproc import process_pc, process_rd

MODEL_ROOT = REPO_ROOT / "models" / "smpl_models"
DATA_DIR = REPO_ROOT / "data" / "BMLmovi_full" / "BMLmovi"
OUT_DIR = REPO_ROOT / "output"

CLIPS = [
    ("Subject_6_F_MoSh", "Subject_6_F_7_poses"),
    ("Subject_31_F_MoSh", "Subject_31_F_10_poses"),
    ("Subject_52_F_MoSh", "Subject_52_F_17_poses"),
    ("Subject_32_F_MoSh", "Subject_32_F_8_poses"),
    ("Subject_56_F_MoSh", "Subject_56_F_9_poses"),
    ("Subject_62_F_MoSh", "Subject_62_F_2_poses"),
    ("Subject_75_F_MoSh", "Subject_75_F_15_poses"),
    ("Subject_8_F_MoSh", "Subject_8_F_16_poses"),
    ("Subject_74_F_MoSh", "Subject_74_F_6_poses"),
    ("Subject_83_F_MoSh", "Subject_83_F_3_poses"),
]

CONFIG = {
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

RADAR_FPS = CONFIG["frame_per_second"]
NUM_FRAMES = 30
CFAR_GUARD = (2, 4)
CFAR_TRAIN = (4, 8)
CFAR_PFA = 1e-2
CFAR_MAX_PTS = 512

SMPL_BONES = [
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 4),
    (2, 5),
    (3, 6),
    (4, 7),
    (5, 8),
    (6, 9),
    (7, 10),
    (8, 11),
    (9, 12),
    (9, 13),
    (9, 14),
    (12, 15),
    (13, 16),
    (14, 17),
    (16, 18),
    (17, 19),
    (18, 20),
    (19, 21),
]


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires CUDA for Mitsuba rendering and radar simulation.")


def ensure_inputs():
    if not MODEL_ROOT.exists():
        raise FileNotFoundError(f"SMPL models not found: {MODEL_ROOT}")
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"AMASS BMLmovi data not found: {DATA_DIR}")
    OUT_DIR.mkdir(exist_ok=True)


def fix_pose(pose: np.ndarray) -> np.ndarray:
    fixed = pose.copy()
    fixed[:3] = 0.0
    return fixed


def draw_skeleton(ax, joints, *, color="cyan", linewidth=1.5, alpha=0.7):
    jx, jy, jz = joints[:, 0], -joints[:, 2], joints[:, 1]
    ax.scatter(jx[:22], jy[:22], jz[:22], c=color, s=20, alpha=alpha, edgecolors="none", zorder=10)
    for i, j in SMPL_BONES:
        ax.plot([jx[i], jx[j]], [jy[i], jy[j]], [jz[i], jz[j]], c=color, linewidth=linewidth, alpha=alpha, zorder=10)


def generate_clip(radar: Radar, clip_idx: int, subj_name: str, clip_name: str):
    subj_dir = DATA_DIR / subj_name
    pose_file = subj_dir / f"{clip_name}.npz"
    shape_file = subj_dir / "shape.npz"
    if not pose_file.exists():
        raise FileNotFoundError(f"Missing pose file: {pose_file}")
    if not shape_file.exists():
        raise FileNotFoundError(f"Missing shape file: {shape_file}")

    pose_data = np.load(str(pose_file), allow_pickle=True)
    shape_data = np.load(str(shape_file), allow_pickle=True)

    all_poses = pose_data["poses"][:, :72]
    betas = shape_data["betas"][:10]
    gender = str(shape_data["gender"])
    source_fps = float(pose_data["mocap_framerate"])

    step = int(source_fps / RADAR_FPS)
    dt_source = step / source_fps
    t_chirp = (CONFIG["idle_time"] + CONFIG["ramp_end_time"]) * 1e-6
    t_frame = t_chirp * CONFIG["num_tx"] * CONFIG["chirp_per_frame"]
    vel_scale = t_frame / dt_source

    start = min(50, all_poses.shape[0] - (NUM_FRAMES + 1) * step - 1)
    start = max(0, start)
    indices = [start + i * step for i in range(NUM_FRAMES + 1)]

    scene = Scene()
    scene.set_sensor(origin=(0, 0, 0), target=(0, 0, -5), fov=80)
    translation = [0, -0.1, -3]
    scene.add_smpl(
        name="human",
        pose=fix_pose(all_poses[indices[0]]),
        shape=betas,
        position=translation,
        gender=gender,
        model_root=str(MODEL_ROOT),
    )

    renderer = Renderer(scene, resolution=256)
    all_pcs = []
    all_rd_mags = []
    all_ray_data = []
    all_joints = []

    for frame_idx in range(NUM_FRAMES):
        src_cur, src_next = indices[frame_idx], indices[frame_idx + 1]

        scene.update_structure("human", pose=fix_pose(all_poses[src_cur]), shape=betas, position=translation)
        trace_cur = renderer.trace()
        joints_cur = scene.get_joints("human").copy()

        scene.update_structure("human", pose=fix_pose(all_poses[src_next]), shape=betas, position=translation)
        trace_next = renderer.trace()

        pts_cur, int_cur = trace_cur
        pts_next, _ = trace_next

        n = min(pts_cur.shape[0], pts_next.shape[0])
        all_ray_data.append((pts_cur.cpu().numpy(), int_cur.cpu().numpy()))
        all_joints.append(joints_cur)

        if n == 0:
            all_pcs.append(np.zeros((0, 6), dtype=np.float32))
            all_rd_mags.append(np.zeros((128, 256), dtype=np.float32))
            continue

        p0, p1, inten = pts_cur[:n], pts_next[:n], int_cur[:n]
        p1_scaled = p0 + (p1 - p0) * vel_scale

        def interpolator(t, _p0=p0, _p1=p1_scaled, _i=inten):
            return _i, _p0 + (_p1 - _p0) * (t / t_frame)

        frame = radar.mimo(interpolator, t0=0)
        pc = process_pc(
            radar,
            frame,
            static_clutter_removal=False,
            positive_velocity_only=False,
            guard_cells=CFAR_GUARD,
            training_cells=CFAR_TRAIN,
            pfa=CFAR_PFA,
            max_points=CFAR_MAX_PTS,
        )
        rd_mag, _, _, _ = process_rd(radar, frame, tx=0, rx=0)

        all_pcs.append(pc)
        all_rd_mags.append(rd_mag)

        if frame_idx % 10 == 0:
            print(f"    frame {frame_idx}/{NUM_FRAMES}: {pts_cur.shape[0]} ray pts -> {pc.shape[0]} PC pts")

    total_ray = sum(ray.shape[0] for ray, _ in all_ray_data)
    total_pc = sum(pc.shape[0] for pc in all_pcs)
    print(f"  [{clip_idx}] {total_ray} ray pts, {total_pc} PC pts across {NUM_FRAMES} frames")

    label = clip_name.replace("_poses", "")
    out_path = OUT_DIR / f"cfar_{label}.mp4"
    ranges_np = radar.ranges.cpu().numpy()
    velocities_np = radar.velocities.cpu().numpy()

    fig = plt.figure(figsize=(18, 6))

    def update(frame_idx):
        fig.clear()

        ax_left = fig.add_subplot(1, 3, 1, projection="3d")
        ray_points, ray_intensities = all_ray_data[frame_idx]
        if ray_points.shape[0] > 0:
            ax_left.scatter(
                ray_points[:, 0],
                -ray_points[:, 2],
                ray_points[:, 1],
                c=ray_intensities,
                cmap="hot",
                s=3,
                alpha=0.8,
                vmin=0,
                vmax=1,
            )
        draw_skeleton(ax_left, all_joints[frame_idx])
        ax_left.set_xlabel("Lateral")
        ax_left.set_ylabel("Range")
        ax_left.set_zlabel("Height")
        ax_left.set_title(f"SMPL Reflections ({ray_points.shape[0]})")
        ax_left.set_xlim(-0.8, 0.8)
        ax_left.set_ylim(2.5, 3.5)
        ax_left.set_zlim(-1.0, 0.8)
        ax_left.set_box_aspect([1, 0.6, 1.2])
        ax_left.view_init(elev=20, azim=-60)

        ax_center = fig.add_subplot(1, 3, 2)
        ax_center.imshow(
            all_rd_mags[frame_idx][:, : len(ranges_np)],
            extent=[ranges_np[0], ranges_np[-1], velocities_np[0], velocities_np[-1]],
            aspect="auto",
            origin="lower",
            cmap="jet",
        )
        ax_center.set_xlabel("Range (m)")
        ax_center.set_ylabel("Velocity (m/s)")
        ax_center.set_title("Range-Doppler")

        ax_right = fig.add_subplot(1, 3, 3, projection="3d")
        pc = all_pcs[frame_idx]
        if pc.shape[0] > 0:
            ax_right.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=pc[:, 4], cmap="hot", s=10, alpha=0.8)
        draw_skeleton(ax_right, all_joints[frame_idx])
        ax_right.set_xlabel("Lateral")
        ax_right.set_ylabel("Range")
        ax_right.set_zlabel("Height")
        ax_right.set_title(f"Radar PC ({pc.shape[0]} pts)")
        ax_right.set_xlim(-0.8, 0.8)
        ax_right.set_ylim(2.5, 3.5)
        ax_right.set_zlim(-1.0, 0.8)
        ax_right.set_box_aspect([1, 0.6, 1.2])
        ax_right.view_init(elev=20, azim=-60)

        t_sec = frame_idx / RADAR_FPS
        fig.suptitle(f"{label} (CFAR) | Frame {frame_idx}/{NUM_FRAMES} ({t_sec:.1f}s)", fontsize=13)
        fig.tight_layout()
        return []

    ani = animation.FuncAnimation(fig, update, frames=NUM_FRAMES, interval=200, blit=False)
    ani.save(str(out_path), writer="ffmpeg", fps=5, dpi=120)
    plt.close(fig)
    print(f"  -> {out_path.name} ({out_path.stat().st_size / 1024:.0f} KB)")
    return out_path


def main():
    require_cuda()
    ensure_inputs()

    radar = Radar(CONFIG, backend="dirichlet", device="cuda")
    print(f"Generating {len(CLIPS)} clips x {NUM_FRAMES} frames each (CFAR pfa={CFAR_PFA})...")

    outputs = []
    for index, (subject, clip) in enumerate(CLIPS, start=1):
        print(f"[{index}/{len(CLIPS)}] {subject} / {clip}")
        outputs.append(generate_clip(radar, index, subject, clip))

    print(f"\nAll done. {len(outputs)} videos in {OUT_DIR}:")
    for path in outputs:
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
