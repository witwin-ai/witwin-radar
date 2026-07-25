"""Render depth images and Range-Doppler maps for a Unity OBJ sequence.

The exported scene is expected to contain:

* ``furniture/*.obj``: static background meshes
* ``Cat_frames/frame_*.obj``: one topology-consistent cat mesh per source frame

Example:
    python -m examples.unity_cat_depth_rd ^
        --scene-root E:/Code/data/unity_export_7_23 ^
        --output-dir output/unity_cat_depth_rd
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from witwin.core import Material
from witwin.radar import Radar, RadarConfig, Scene, Tracer
from witwin.radar.sigproc import process_rd


RADAR_CONFIG = {
    "num_tx": 1,
    "num_rx": 1,
    "fc": 77e9,
    "slope": 60.012,
    "adc_samples": 256,
    "adc_start_time": 6,
    "sample_rate": 10000,
    "idle_time": 7,
    "ramp_end_time": 43,
    "chirp_per_frame": 128,
    "frame_per_second": 30,
    "num_doppler_bins": 128,
    "num_range_bins": 256,
    "num_angle_bins": 64,
    "power": 15,
    "tx_loc": [[0, 0, 0]],
    "rx_loc": [[0, 0, 0]],
}


def load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    loaded = trimesh.load(path, force="mesh", process=False, maintain_order=True)
    if isinstance(loaded, trimesh.Scene):
        meshes = [geometry for geometry in loaded.geometry.values() if isinstance(geometry, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"No mesh geometry found in {path}.")
        loaded = trimesh.util.concatenate(meshes)
    vertices = np.asarray(loaded.vertices, dtype=np.float32)
    faces = np.asarray(loaded.faces, dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Expected a triangular OBJ mesh in {path}.")
    return np.ascontiguousarray(vertices), np.ascontiguousarray(faces)


def validate_cat_topology(paths: list[Path], vertex_count: int, faces: np.ndarray) -> None:
    for path in paths[1:]:
        vertices_i, faces_i = load_obj(path)
        if len(vertices_i) != vertex_count or not np.array_equal(faces_i, faces):
            raise ValueError(f"Cat topology differs in {path.name}; triangle correspondence would be invalid.")


def save_frame_figure(
    path: Path,
    depth: np.ndarray,
    rd_filtered_db: np.ndarray,
    rd_unfiltered_db: np.ndarray,
    ranges: np.ndarray,
    velocities: np.ndarray,
    *,
    frame_index: int,
    depth_max: float,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(17, 4.6), dpi=140)

    depth_image = np.ma.masked_less_equal(depth, 0.0)
    depth_plot = axes[0].imshow(depth_image, cmap="viridis_r", vmin=0.0, vmax=depth_max)
    axes[0].set_title(f"Depth — source frame {frame_index}")
    axes[0].set_xlabel("pixel x")
    axes[0].set_ylabel("pixel y")
    figure.colorbar(depth_plot, ax=axes[0], label="distance (m)", fraction=0.046)

    for axis, rd_db, title in (
        (axes[1], rd_filtered_db, "RD — static clutter removed"),
        (axes[2], rd_unfiltered_db, "RD — no static filtering"),
    ):
        rd_relative = rd_db - float(np.max(rd_db))
        rd_plot = axis.imshow(
            rd_relative,
            extent=[float(ranges[0]), float(ranges[-1]), float(velocities[0]), float(velocities[-1])],
            origin="lower",
            aspect="auto",
            cmap="turbo",
            vmin=-60.0,
            vmax=0.0,
        )
        axis.set_title(f"{title}\npeak={float(np.max(rd_db)):.1f} dB")
        axis.set_xlabel("range (m)")
        axis.set_ylabel("radial velocity (m/s)")
        figure.colorbar(rd_plot, ax=axis, label="relative magnitude (dB)", fraction=0.046)

    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def build_scene(
    scene_root: Path,
    cat_vertices: np.ndarray,
    cat_faces: np.ndarray,
) -> Scene:
    scene = Scene(device="cuda")
    furniture_material = Material(eps_r=5.0)
    for path in sorted((scene_root / "furniture").glob("*.obj")):
        vertices, faces = load_obj(path)
        # Triangle sampling visits meshes marked dynamic; unchanged furniture
        # then contributes zero-Doppler paths while keeping stable triangle IDs.
        scene.add_mesh(
            name=f"furniture_{path.stem}",
            vertices=vertices,
            faces=faces,
            material=furniture_material,
            dynamic=True,
        )
    scene.add_mesh(
        name="cat",
        vertices=cat_vertices,
        faces=cat_faces,
        material=Material(eps_r=3.0),
        dynamic=True,
    )
    return scene


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This simulation requires a CUDA-enabled PyTorch environment.")

    scene_root = args.scene_root.expanduser().resolve()
    furniture_dir = scene_root / "furniture"
    cat_dir = scene_root / "Cat_frames"
    if not furniture_dir.is_dir() or not cat_dir.is_dir():
        raise FileNotFoundError("Expected furniture/ and Cat_frames/ under --scene-root.")

    cat_paths = sorted(cat_dir.glob("frame_*.obj"))
    if len(cat_paths) < 2:
        raise FileNotFoundError(f"Need at least two cat frames in {cat_dir}.")

    start = args.start_frame
    available = len(cat_paths) - 1 - start
    frame_count = available if args.num_frames <= 0 else min(args.num_frames, available)
    if start < 0 or frame_count <= 0:
        raise ValueError("The requested frame range is empty.")
    selected_paths = cat_paths[start : start + frame_count + 1]

    output_dir = args.output_dir.expanduser().resolve()
    frame_dir = output_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    first_vertices, cat_faces = load_obj(selected_paths[0])
    if not args.skip_topology_check:
        validate_cat_topology(selected_paths, len(first_vertices), cat_faces)

    config_dict = dict(RADAR_CONFIG)
    config_dict["frame_per_second"] = args.source_fps
    radar = Radar(
        RadarConfig.from_dict(config_dict),
        device="cuda",
        position=args.camera_position,
        target=args.camera_target,
        up=(0.0, 1.0, 0.0),
        fov=args.fov,
        name="unity_cat",
    )
    scene = build_scene(scene_root, first_vertices, cat_faces)
    tracer = Tracer(scene, radar, resolution=args.resolution, sampling="triangle")

    chirp_period = (radar.config.idle_time + radar.config.ramp_end_time) * 1e-6
    radar_frame_time = chirp_period * radar.config.num_tx * radar.config.chirp_per_frame
    source_dt = 1.0 / args.source_fps
    velocity_scale = radar_frame_time / source_dt

    current_trace = tracer.trace()
    current_depth = tracer.render_depth().detach().cpu().numpy().astype(np.float32)
    depth_frames: list[np.ndarray] = []
    rd_filtered_frames: list[np.ndarray] = []
    rd_unfiltered_frames: list[np.ndarray] = []
    ranges = None
    velocities = None

    started = time.perf_counter()
    for local_index in range(frame_count):
        source_index = start + local_index
        next_vertices, _ = load_obj(selected_paths[local_index + 1])
        scene.update_structure("cat", vertices=next_vertices)
        next_trace = tracer.trace()

        p0, p1, intensities = tracer.match(current_trace, next_trace)
        p1_scaled = p0 + (p1 - p0) * velocity_scale

        def interpolator(t, _p0=p0, _p1=p1_scaled, _intensities=intensities):
            fraction = t / radar_frame_time if radar_frame_time > 0.0 else 0.0
            return _intensities, _p0 + (_p1 - _p0) * fraction

        radar_frame = radar.mimo(interpolator, t0=0.0)
        rd_filtered_db, _, ranges, velocities = process_rd(
            radar,
            radar_frame,
            tx=0,
            rx=0,
            static_clutter_removal=True,
        )
        rd_unfiltered_db, _, _, _ = process_rd(
            radar,
            radar_frame,
            tx=0,
            rx=0,
            static_clutter_removal=False,
        )
        rd_filtered_db = rd_filtered_db[:, : len(ranges)].astype(np.float32, copy=False)
        rd_unfiltered_db = rd_unfiltered_db[:, : len(ranges)].astype(np.float32, copy=False)

        depth_frames.append(current_depth)
        rd_filtered_frames.append(rd_filtered_db)
        rd_unfiltered_frames.append(rd_unfiltered_db)
        save_frame_figure(
            frame_dir / f"frame_{source_index:04d}.png",
            current_depth,
            rd_filtered_db,
            rd_unfiltered_db,
            ranges,
            velocities,
            frame_index=source_index,
            depth_max=args.depth_max,
        )

        current_trace = next_trace
        if local_index + 1 < frame_count:
            current_depth = tracer.render_depth().detach().cpu().numpy().astype(np.float32)

        elapsed = time.perf_counter() - started
        print(
            f"[{local_index + 1:03d}/{frame_count:03d}] source={source_index:04d} "
            f"paths={len(p0):6d} elapsed={elapsed:7.1f}s"
        )

    depth_stack = np.stack(depth_frames)
    rd_filtered_stack = np.stack(rd_filtered_frames)
    rd_unfiltered_stack = np.stack(rd_unfiltered_frames)
    np.save(output_dir / "depth_m.npy", depth_stack)
    np.save(output_dir / "rd_db.npy", rd_filtered_stack)
    np.save(output_dir / "rd_db_unfiltered.npy", rd_unfiltered_stack)
    np.savez(output_dir / "rd_axes.npz", ranges=ranges, velocities=velocities)

    metadata = {
        "scene_root": str(scene_root),
        "source_fps": args.source_fps,
        "start_frame": start,
        "num_frames": frame_count,
        "camera_position": list(args.camera_position),
        "camera_target": list(args.camera_target),
        "fov_degrees": args.fov,
        "depth_resolution": args.resolution,
        "radar_config": config_dict,
        "depth_shape": list(depth_stack.shape),
        "rd_static_removed_shape": list(rd_filtered_stack.shape),
        "rd_unfiltered_shape": list(rd_unfiltered_stack.shape),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved results to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-root", type=Path, default=Path("E:/Code/data/unity_export_7_23"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/unity_cat_depth_rd"))
    parser.add_argument("--source-fps", type=float, default=10.0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=0, help="0 processes every available frame pair.")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--depth-max", type=float, default=14.0)
    parser.add_argument("--camera-position", type=float, nargs=3, default=(-2.0, 2.4, -3.2))
    parser.add_argument("--camera-target", type=float, nargs=3, default=(2.0, 0.55, 2.0))
    parser.add_argument("--fov", type=float, default=60.0)
    parser.add_argument("--skip-topology-check", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
