"""
Batch radar point cloud generation from the AMASS dataset.

Processes all clips in an AMASS dataset directory, generating per-clip .npz
files containing radar point clouds and Range-Doppler maps.

Optimizations over the original gen_amass_video.py:
  - Incremental vertex update (no full Mitsuba scene rebuild per frame)
  - Trace caching (each source keyframe traced once, reused for adjacent frames)
  - Single Renderer instance per clip (reused across all frames)
  - No video rendering (separate post-processing step)

Usage:
    python batch_generate.py --data_dir data/BMLmovi_full/BMLmovi --out_dir output/npz
    python batch_generate.py --data_dir data/BMLmovi_full/BMLmovi --out_dir output/npz --max_clips 5
    python batch_generate.py --profile  # measure GPU memory per clip to estimate parallelism
"""

import sys
import pathlib
import argparse
import time
import json

repo = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(repo / 'witwin-radar'))

import numpy as np
import torch
torch.set_default_device('cuda')

from core import Radar, Renderer, Scene
from sigproc import process_pc, process_rd


# ── Default config ────────────────────────────────────────────────

DEFAULT_RADAR_CONFIG = {
    "num_tx": 3, "num_rx": 4, "fc": 77e9, "slope": 60.012,
    "adc_samples": 256, "adc_start_time": 6, "sample_rate": 4400,
    "idle_time": 7, "ramp_end_time": 65, "chirp_per_frame": 128,
    "frame_per_second": 10, "num_doppler_bins": 128, "num_range_bins": 256,
    "num_angle_bins": 64, "power": 15,
    "tx_loc": [[0, 0, 0], [4, 0, 0], [2, 1, 0]],
    "rx_loc": [[-6, 0, 0], [-5, 0, 0], [-4, 0, 0], [-3, 0, 0]],
}

DEFAULT_CFAR = {
    "guard_cells": (2, 4),
    "training_cells": (4, 8),
    "pfa": 1e-2,
    "max_points": 512,
}

RENDERER_RESOLUTION = 256
NUM_FRAMES = 30
TRANSLATION = [0, -0.1, -3]


def fix_pose(pose):
    """Strip AMASS root orientation, keep body articulation."""
    fixed = pose.copy()
    fixed[:3] = 0.0
    return fixed


def discover_clips(data_dir):
    """Scan AMASS directory for all subject/clip pairs.

    Returns list of (subject_dir_name, clip_stem) tuples.
    """
    data_dir = pathlib.Path(data_dir)
    clips = []
    for subj_dir in sorted(data_dir.iterdir()):
        if not subj_dir.is_dir():
            continue
        for npz_file in sorted(subj_dir.glob('*_poses.npz')):
            clips.append((subj_dir.name, npz_file.stem))
    return clips


def process_clip(clip_info, radar, model_root, data_dir, out_dir, cfar_cfg,
                 sampling='pixel'):
    """Process a single clip: trace -> radar -> point cloud -> save .npz.

    Uses trace caching and incremental vertex updates for speed.
    """
    subj_name, clip_stem = clip_info
    subj_dir = pathlib.Path(data_dir) / subj_name
    pose_file = subj_dir / f'{clip_stem}.npz'

    if not pose_file.exists():
        return None

    pose_data = np.load(str(pose_file), allow_pickle=True)
    shape_file = subj_dir / 'shape.npz'
    if not shape_file.exists():
        return None
    shape_data = np.load(str(shape_file), allow_pickle=True)

    all_poses = pose_data['poses'][:, :72]
    betas = shape_data['betas'][:10]
    gender = str(shape_data['gender'])
    source_fps = float(pose_data['mocap_framerate'])

    radar_fps = radar.frame_per_second
    step = max(1, int(source_fps / radar_fps))
    dt_source = step / source_fps

    T_chirp = (radar.idle_time + radar.ramp_end_time) * 1e-6
    T_frame = T_chirp * radar.num_tx * radar.chirp_per_frame
    vel_scale = T_frame / dt_source

    num_frames = NUM_FRAMES
    needed = (num_frames + 1) * step + 1
    if all_poses.shape[0] < needed:
        num_frames = min(num_frames, (all_poses.shape[0] - 1) // step - 1)
    if num_frames < 2:
        return None

    start = min(50, all_poses.shape[0] - (num_frames + 1) * step - 1)
    start = max(0, start)
    indices = [start + i * step for i in range(num_frames + 1)]

    # Build scene and renderer once
    scene = Scene(fov=80)
    scene.set_sensor(origin=(0, 0, 0), target=(0, 0, -5))
    scene.add_smpl('human', fix_pose(all_poses[indices[0]]), betas,
                   position=TRANSLATION, gender=gender, model_root=model_root)
    renderer = Renderer(scene, resolution=RENDERER_RESOLUTION, sampling=sampling)

    # Phase 1: trace all source keyframes (with caching)
    trace_cache = {}
    for idx in indices:
        if idx in trace_cache:
            continue
        scene.update_smpl('human', fix_pose(all_poses[idx]), betas,
                          position=TRANSLATION)
        trace_cache[idx] = renderer.trace()

    # Phase 2: generate radar frames from cached traces
    all_pcs = []
    all_rd_mags = []

    for fi in range(num_frames):
        src_cur, src_next = indices[fi], indices[fi + 1]
        p0, p1, inten = renderer.match(trace_cache[src_cur], trace_cache[src_next])

        if p0.shape[0] == 0:
            all_pcs.append(np.zeros((0, 6), dtype=np.float32))
            all_rd_mags.append(np.zeros((radar.num_doppler_bins, radar.num_range_bins),
                                        dtype=np.float32))
            continue

        p1_scaled = p0 + (p1 - p0) * vel_scale

        def make_interp(p0, p1s, inten, T):
            def fn(t):
                frac = t / T if T > 0 else 0.0
                return inten, p0 + (p1s - p0) * frac
            return fn

        frame = radar.mimo(make_interp(p0, p1_scaled, inten, T_frame), t0=0)

        pc = process_pc(
            radar, frame,
            static_clutter_removal=False,
            positive_velocity_only=False,
            **cfar_cfg,
        )
        rd_mag, _, _, _ = process_rd(radar, frame, tx=0, rx=0)

        all_pcs.append(pc.astype(np.float32))
        all_rd_mags.append(rd_mag.astype(np.float32))

    # Save .npz
    out_path = pathlib.Path(out_dir) / f'{clip_stem}.npz'
    max_pts = max((pc.shape[0] for pc in all_pcs), default=0)

    # Pad point clouds to uniform shape for stacking
    pc_padded = np.zeros((num_frames, max(max_pts, 1), 6), dtype=np.float32)
    pc_counts = np.zeros(num_frames, dtype=np.int32)
    for i, pc in enumerate(all_pcs):
        n = pc.shape[0]
        pc_counts[i] = n
        if n > 0:
            pc_padded[i, :n] = pc

    np.savez_compressed(
        str(out_path),
        point_clouds=pc_padded,
        point_counts=pc_counts,
        rd_maps=np.array(all_rd_mags),
        subject=subj_name,
        clip=clip_stem,
        gender=gender,
        source_fps=source_fps,
        radar_fps=radar_fps,
        num_frames=num_frames,
        radar_config=json.dumps(DEFAULT_RADAR_CONFIG),
    )

    total_pts = int(pc_counts.sum())
    return {
        'clip': clip_stem,
        'subject': subj_name,
        'num_frames': num_frames,
        'total_points': total_pts,
        'output': str(out_path),
    }


def profile_gpu_memory(model_root, data_dir, clips):
    """Measure GPU memory usage for a single clip to estimate parallelism."""
    print("=== GPU Memory Profiling ===\n")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()

    radar = Radar(DEFAULT_RADAR_CONFIG)
    mem_after_radar = torch.cuda.memory_allocated()

    subj_name, clip_stem = clips[0]
    result = process_clip(
        (subj_name, clip_stem), radar, model_root,
        data_dir, '/tmp', DEFAULT_CFAR, sampling='pixel',
    )

    mem_peak = torch.cuda.max_memory_allocated()
    mem_after = torch.cuda.memory_allocated()

    total_gpu = torch.cuda.get_device_properties(0).total_memory
    gpu_name = torch.cuda.get_device_properties(0).name

    print(f"GPU: {gpu_name}")
    print(f"Total VRAM: {total_gpu / 1024**3:.1f} GB")
    print(f"Memory before: {mem_before / 1024**2:.0f} MB")
    print(f"Memory after Radar init: {mem_after_radar / 1024**2:.0f} MB")
    print(f"Peak memory (1 clip): {mem_peak / 1024**2:.0f} MB")
    print(f"Memory after cleanup: {mem_after / 1024**2:.0f} MB")
    print()

    per_clip_mb = (mem_peak - mem_before) / 1024**2
    available_mb = total_gpu / 1024**2 * 0.85  # 85% safety margin
    max_parallel = max(1, int(available_mb / per_clip_mb))

    print(f"Per-clip peak usage: {per_clip_mb:.0f} MB")
    print(f"Available (85% of total): {available_mb:.0f} MB")
    print(f"Estimated max parallel clips on this GPU: {max_parallel}")
    print()

    if result:
        print(f"Test clip: {result['clip']}")
        print(f"  Frames: {result['num_frames']}, Points: {result['total_points']}")

    return max_parallel, per_clip_mb


def main():
    parser = argparse.ArgumentParser(description='Batch radar point cloud generation from AMASS')
    parser.add_argument('--data_dir', type=str,
                        default=str(repo / 'data' / 'BMLmovi_full' / 'BMLmovi'))
    parser.add_argument('--out_dir', type=str,
                        default=str(repo / 'output' / 'npz'))
    parser.add_argument('--model_root', type=str,
                        default=str(repo / 'models' / 'smpl_models'))
    parser.add_argument('--max_clips', type=int, default=0,
                        help='Max clips to process (0 = all)')
    parser.add_argument('--profile', action='store_true',
                        help='Profile GPU memory and exit')
    parser.add_argument('--num_frames', type=int, default=30,
                        help='Radar frames per clip')
    parser.add_argument('--sampling', type=str, default='pixel',
                        choices=['pixel', 'triangle'],
                        help='Surface sampling mode: pixel (ray grid) or triangle (one per face)')
    args = parser.parse_args()

    global NUM_FRAMES
    NUM_FRAMES = args.num_frames

    data_dir = pathlib.Path(args.data_dir)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {data_dir} ...")
    clips = discover_clips(data_dir)
    print(f"Found {len(clips)} clips\n")

    if len(clips) == 0:
        print("No clips found. Check --data_dir path.")
        return

    if args.profile:
        profile_gpu_memory(args.model_root, args.data_dir, clips)
        return

    if args.max_clips > 0:
        clips = clips[:args.max_clips]

    radar = Radar(DEFAULT_RADAR_CONFIG)

    # Skip already-processed clips
    todo = []
    for c in clips:
        out_path = out_dir / f'{c[1]}.npz'
        if out_path.exists():
            continue
        todo.append(c)

    print(f"Processing {len(todo)} clips ({len(clips) - len(todo)} already done)\n")

    t_start = time.time()
    results = []
    for i, clip_info in enumerate(todo):
        t0 = time.time()
        result = process_clip(clip_info, radar, args.model_root,
                              args.data_dir, str(out_dir), DEFAULT_CFAR,
                              sampling=args.sampling)
        dt = time.time() - t0

        if result:
            results.append(result)
            elapsed = time.time() - t_start
            eta = elapsed / (i + 1) * (len(todo) - i - 1)
            print(f"[{i+1}/{len(todo)}] {result['clip']}: "
                  f"{result['num_frames']}f, {result['total_points']}pts, "
                  f"{dt:.1f}s (ETA: {eta/60:.0f}min)")
        else:
            print(f"[{i+1}/{len(todo)}] {clip_info[1]}: skipped ({time.time()-t0:.1f}s)")

    total_time = time.time() - t_start
    print(f"\nDone. {len(results)}/{len(todo)} clips in {total_time/60:.1f} min")
    print(f"Output: {out_dir}/")

    # Save manifest
    manifest_path = out_dir / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump({
            'num_clips': len(results),
            'total_time_sec': total_time,
            'radar_config': DEFAULT_RADAR_CONFIG,
            'cfar_config': {k: list(v) if isinstance(v, tuple) else v
                           for k, v in DEFAULT_CFAR.items()},
            'num_frames_per_clip': NUM_FRAMES,
            'clips': results,
        }, f, indent=2)
    print(f"Manifest: {manifest_path}")


if __name__ == '__main__':
    main()
