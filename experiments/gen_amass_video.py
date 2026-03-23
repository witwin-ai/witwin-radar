"""
Generate combined radar visualization videos from 10 AMASS BMLmovi clips.

Uses CA-CFAR detection instead of top-K for better limb/extremity coverage.

Each clip: AMASS pose -> SMPL mesh -> Mitsuba ray tracing -> FMCW radar ->
CA-CFAR detection -> combined MP4 (SMPL reflections + RD map + radar PC).

Usage:
    cd RadarTwin/experiments && python gen_amass_video.py
"""
import sys, pathlib

repo = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo / 'witwin-radar'))
sys.path.insert(0, str(repo))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from core import Radar, Renderer, Scene
from sigproc import process_pc, process_rd

torch.set_default_device('cuda')

MODEL_ROOT = str(repo / 'models' / 'smpl_models')
OUT_DIR = repo / 'output'
OUT_DIR.mkdir(exist_ok=True)
DATA_DIR = repo / 'data' / 'BMLmovi_full' / 'BMLmovi'

# 10 clips with distinct motion IDs, ranked by body motion magnitude.
CLIPS = [
    ('Subject_6_F_MoSh', 'Subject_6_F_7_poses'),       # motion#7
    ('Subject_31_F_MoSh', 'Subject_31_F_10_poses'),     # motion#10
    ('Subject_52_F_MoSh', 'Subject_52_F_17_poses'),     # motion#17
    ('Subject_32_F_MoSh', 'Subject_32_F_8_poses'),      # motion#8
    ('Subject_56_F_MoSh', 'Subject_56_F_9_poses'),      # motion#9
    ('Subject_62_F_MoSh', 'Subject_62_F_2_poses'),      # motion#2
    ('Subject_75_F_MoSh', 'Subject_75_F_15_poses'),     # motion#15
    ('Subject_8_F_MoSh', 'Subject_8_F_16_poses'),       # motion#16
    ('Subject_74_F_MoSh', 'Subject_74_F_6_poses'),      # motion#6
    ('Subject_83_F_MoSh', 'Subject_83_F_3_poses'),      # motion#3
]

config = {
    "num_tx": 3, "num_rx": 4, "fc": 77e9, "slope": 60.012,
    "adc_samples": 256, "adc_start_time": 6, "sample_rate": 4400,
    "idle_time": 7, "ramp_end_time": 65, "chirp_per_frame": 128,
    "frame_per_second": 10, "num_doppler_bins": 128, "num_range_bins": 256,
    "num_angle_bins": 64, "power": 15,
    "tx_loc": [[0, 0, 0], [4, 0, 0], [2, 1, 0]],
    "rx_loc": [[-6, 0, 0], [-5, 0, 0], [-4, 0, 0], [-3, 0, 0]],
}
radar = Radar(config)
ranges_np = radar.ranges.cpu().numpy()
velocities_np = radar.velocities.cpu().numpy()

T_chirp = (config["idle_time"] + config["ramp_end_time"]) * 1e-6
T_frame = T_chirp * config["num_tx"] * config["chirp_per_frame"]
RADAR_FPS = config["frame_per_second"]
NUM_FRAMES = 30

# CFAR parameters
CFAR_GUARD = (2, 4)
CFAR_TRAIN = (4, 8)
CFAR_PFA = 1e-2
CFAR_MAX_PTS = 512


def fix_pose(pose):
    """Strip AMASS root orientation (Z-up convention), keep body articulation."""
    fixed = pose.copy()
    fixed[:3] = 0.0
    return fixed


# SMPL 24-joint skeleton connectivity (first 22 joints = body, last 2 = hands)
SMPL_BONES = [
    (0, 1), (0, 2), (0, 3),       # pelvis -> L hip, R hip, spine1
    (1, 4), (2, 5),               # hips -> knees
    (3, 6),                       # spine1 -> spine2
    (4, 7), (5, 8),               # knees -> ankles
    (6, 9),                       # spine2 -> spine3
    (7, 10), (8, 11),             # ankles -> feet
    (9, 12), (9, 13), (9, 14),    # spine3 -> neck, L collar, R collar
    (12, 15),                     # neck -> head
    (13, 16), (14, 17),           # collars -> shoulders
    (16, 18), (17, 19),           # shoulders -> elbows
    (18, 20), (19, 21),           # elbows -> wrists
]


def draw_skeleton(ax, joints, color='cyan', linewidth=1.5, alpha=0.7):
    """Draw SMPL skeleton on a 3D axis.

    joints: (24, 3) in scene coordinates (X lateral, Y height, Z depth).
    Plot mapping: (X, -Z, Y) = (Lateral, Range, Height).
    """
    jx, jy, jz = joints[:, 0], -joints[:, 2], joints[:, 1]
    ax.scatter(jx[:22], jy[:22], jz[:22], c=color, s=20, alpha=alpha,
               edgecolors='none', zorder=10)
    for i, j in SMPL_BONES:
        ax.plot([jx[i], jx[j]], [jy[i], jy[j]], [jz[i], jz[j]],
                c=color, linewidth=linewidth, alpha=alpha, zorder=10)


def generate_clip(clip_idx, subj_name, clip_name):
    subj_dir = DATA_DIR / subj_name
    pose_file = subj_dir / f'{clip_name}.npz'
    pose_data = np.load(str(pose_file), allow_pickle=True)
    shape_data = np.load(str(subj_dir / 'shape.npz'), allow_pickle=True)

    all_poses = pose_data['poses'][:, :72]
    betas = shape_data['betas'][:10]
    gender = str(shape_data['gender'])
    source_fps = float(pose_data['mocap_framerate'])

    step = int(source_fps / RADAR_FPS)
    dt_source = step / source_fps
    # Scale displacement so interpolator velocity matches real physical velocity.
    # The interpolator traverses (p1 - p0) over T_frame, but the source frames
    # are dt_source apart. Without scaling, velocity is inflated by dt_source/T_frame.
    vel_scale = T_frame / dt_source

    start = min(50, all_poses.shape[0] - (NUM_FRAMES + 1) * step - 1)
    start = max(0, start)
    indices = [start + i * step for i in range(NUM_FRAMES + 1)]

    scene = Scene(fov=80)
    scene.set_sensor(origin=(0, 0, 0), target=(0, 0, -5))
    translation = [0, -0.1, -3]
    scene.add_smpl('human', fix_pose(all_poses[indices[0]]), betas,
                   position=translation, gender=gender, model_root=MODEL_ROOT)

    all_pcs, all_rd_mags, all_ray_data, all_joints = [], [], [], []
    renderer = Renderer(scene, resolution=256)

    for fi in range(NUM_FRAMES):
        src_cur, src_next = indices[fi], indices[fi + 1]

        scene.update_smpl('human', fix_pose(all_poses[src_cur]), betas,
                          position=translation)
        pts_cur, int_cur = renderer.trace()
        joints_cur = scene.get_joints('human').copy()

        scene.update_smpl('human', fix_pose(all_poses[src_next]), betas,
                          position=translation)
        pts_next, int_next = renderer.trace()

        n = min(pts_cur.shape[0], pts_next.shape[0])
        all_ray_data.append((pts_cur.cpu().numpy(), int_cur.cpu().numpy()))
        all_joints.append(joints_cur)

        if n == 0:
            all_pcs.append(np.zeros((0, 6)))
            all_rd_mags.append(np.zeros((128, 256)))
            continue

        p0, p1, inten = pts_cur[:n], pts_next[:n], int_cur[:n]
        p1_scaled = p0 + (p1 - p0) * vel_scale

        def make_interp(p0, p1, inten, T):
            def fn(t):
                return inten, p0 + (p1 - p0) * (t / T)
            return fn

        frame = radar.mimo(make_interp(p0, p1_scaled, inten, T_frame), t0=0)

        pc = process_pc(
            radar, frame,
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

        if fi % 10 == 0:
            print(f"    frame {fi}/{NUM_FRAMES}: {pts_cur.shape[0]} ray pts "
                  f"-> {pc.shape[0]} PC pts")

    total_ray = sum(rd.shape[0] for rd, _ in all_ray_data)
    total_pc = sum(pc.shape[0] for pc in all_pcs)
    print(f"  [{clip_idx}] {total_ray} ray pts, {total_pc} PC pts "
          f"across {NUM_FRAMES} frames")

    # ---- Render combined video ----
    label = clip_name.replace('_poses', '')
    out_path = OUT_DIR / f'cfar_{label}.mp4'

    fig = plt.figure(figsize=(18, 6))

    def update(fi):
        fig.clear()

        ax_l = fig.add_subplot(1, 3, 1, projection='3d')
        rp, ri = all_ray_data[fi]
        if rp.shape[0] > 0:
            ax_l.scatter(rp[:, 0], -rp[:, 2], rp[:, 1],
                         c=ri, cmap='hot', s=3, alpha=0.8, vmin=0, vmax=1)
        draw_skeleton(ax_l, all_joints[fi], color='cyan', linewidth=1.5)
        ax_l.set_xlabel('Lateral')
        ax_l.set_ylabel('Range')
        ax_l.set_zlabel('Height')
        ax_l.set_title(f'SMPL Reflections ({rp.shape[0]})')
        ax_l.set_xlim(-0.8, 0.8)
        ax_l.set_ylim(2.5, 3.5)
        ax_l.set_zlim(-1.0, 0.8)
        ax_l.set_box_aspect([1, 0.6, 1.2])
        ax_l.view_init(elev=20, azim=-60)

        ax_c = fig.add_subplot(1, 3, 2)
        ax_c.imshow(all_rd_mags[fi][:, :len(ranges_np)],
                    extent=[ranges_np[0], ranges_np[-1],
                            velocities_np[0], velocities_np[-1]],
                    aspect='auto', origin='lower', cmap='jet')
        ax_c.set_xlabel('Range (m)')
        ax_c.set_ylabel('Velocity (m/s)')
        ax_c.set_title('Range-Doppler')

        ax_r = fig.add_subplot(1, 3, 3, projection='3d')
        pc = all_pcs[fi]
        if pc.shape[0] > 0:
            ax_r.scatter(pc[:, 0], pc[:, 1], pc[:, 2],
                         c=pc[:, 4], cmap='hot', s=10, alpha=0.8)
        draw_skeleton(ax_r, all_joints[fi], color='cyan', linewidth=1.5)
        ax_r.set_xlabel('Lateral')
        ax_r.set_ylabel('Range')
        ax_r.set_zlabel('Height')
        ax_r.set_title(f'Radar PC ({pc.shape[0]} pts)')
        ax_r.set_xlim(-0.8, 0.8)
        ax_r.set_ylim(2.5, 3.5)
        ax_r.set_zlim(-1.0, 0.8)
        ax_r.set_box_aspect([1, 0.6, 1.2])
        ax_r.view_init(elev=20, azim=-60)

        t_sec = fi / RADAR_FPS
        fig.suptitle(
            f'{label} (CFAR) | Frame {fi}/{NUM_FRAMES} ({t_sec:.1f}s)',
            fontsize=13)
        fig.tight_layout()
        return []

    ani = animation.FuncAnimation(fig, update, frames=NUM_FRAMES,
                                  interval=200, blit=False)
    ani.save(str(out_path), writer='ffmpeg', fps=5, dpi=120)
    plt.close(fig)
    print(f"  -> {out_path.name}  ({out_path.stat().st_size / 1024:.0f} KB)")
    return out_path


print(f"Generating {len(CLIPS)} clips x {NUM_FRAMES} frames each "
      f"(CFAR pfa={CFAR_PFA})...\n")
outputs = []
for i, (subj, clip) in enumerate(CLIPS):
    print(f"[{i+1}/{len(CLIPS)}] {subj} / {clip}")
    out = generate_clip(i, subj, clip)
    outputs.append(out)

print(f"\nAll done. {len(outputs)} videos in {OUT_DIR}/:")
for p in outputs:
    print(f"  {p.name}")
