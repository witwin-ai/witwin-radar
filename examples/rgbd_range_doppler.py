"""
Generate Range-Doppler maps from an RGBD/depth sequence.

This example treats the depth camera view as the radar view: the radar is at
the depth camera origin and looks along the depth camera forward axis. Depth
pixels are back-projected into radar scene coordinates (X lateral, Y height,
Z negative range), then passed to Radar.mimo() through the standard
interpolator interface.

Usage:
    python -m examples.rgbd_range_doppler --input path/to/depths.npy
    python -m examples.rgbd_range_doppler --input path/to/rgbd_sequence.npz
    python -m examples.rgbd_range_doppler --input path/to/recording.mkv

Input formats:
    .npy depth:      (T, H, W), depth in meters or millimeters
    .npy pointcloud: (T, N, 3) or (T, H, W, 3)
    .npz:            depth key such as "depths", optional pointcloud/mask keys
    .mkv:            Azure Kinect playback, requires pykinect_azure
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from dataclasses import dataclass
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_RADAR_CONFIG = {
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

DEPTH_KEYS = ("depths", "depth", "depth_frames", "depth_images")
POINTCLOUD_KEYS = ("pointclouds", "pointcloud", "points", "pc", "pcs")
MASK_KEYS = ("masks", "mask", "body_mask", "human_mask", "segmentation", "segments")


@dataclass
class RGBDSequence:
    depths: np.ndarray | None
    pointclouds: np.ndarray | None
    masks: np.ndarray | None
    fps: float


def _load_config(path: str | None) -> dict[str, Any]:
    if path is None:
        return dict(DEFAULT_RADAR_CONFIG)
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _pick_npz_key(data: np.lib.npyio.NpzFile, explicit: str | None, candidates: tuple[str, ...]) -> str | None:
    if explicit:
        if explicit not in data:
            raise KeyError(f"Key '{explicit}' not found in {data.files}.")
        return explicit
    for key in candidates:
        if key in data:
            return key
    return None


def _infer_npz_key_by_shape(data: np.lib.npyio.NpzFile, *, pointcloud: bool, mask: bool = False) -> str | None:
    for key in data.files:
        shape = data[key].shape
        if mask:
            if len(shape) in {2, 3, 4} and key not in DEPTH_KEYS and key not in POINTCLOUD_KEYS:
                return key
        elif pointcloud:
            if (len(shape) == 3 and shape[-1] == 3) or (len(shape) == 4 and shape[-1] == 3):
                return key
        elif len(shape) == 3 and shape[-1] != 3:
            return key
    return None


def _load_numpy_sequence(path: pathlib.Path, args: argparse.Namespace) -> RGBDSequence:
    if path.suffix.lower() == ".npy":
        array = np.load(path)
        fps = args.source_fps if args.source_fps is not None else 30.0
        if array.ndim == 3 and array.shape[-1] == 3:
            return RGBDSequence(depths=None, pointclouds=array, masks=None, fps=fps)
        if array.ndim == 4 and array.shape[-1] == 3:
            return RGBDSequence(depths=None, pointclouds=array, masks=None, fps=fps)
        if array.ndim == 3:
            return RGBDSequence(depths=array, pointclouds=None, masks=None, fps=fps)
        raise ValueError(
            ".npy input must have shape (T,H,W), (T,N,3), or (T,H,W,3); "
            f"got {array.shape}."
        )

    with np.load(path) as data:
        depth_key = _pick_npz_key(data, args.depth_key, DEPTH_KEYS)
        pc_key = _pick_npz_key(data, args.pointcloud_key, POINTCLOUD_KEYS)
        mask_key = _pick_npz_key(data, args.mask_key, MASK_KEYS)
        if depth_key is None and args.depth_key is None:
            depth_key = _infer_npz_key_by_shape(data, pointcloud=False)
        if pc_key is None and args.pointcloud_key is None:
            pc_key = _infer_npz_key_by_shape(data, pointcloud=True)
        if depth_key is None and pc_key is None:
            raise KeyError(
                "No depth or pointcloud array found in .npz. "
                f"Available keys: {data.files}. Use --depth-key or --pointcloud-key."
            )
        if args.source_fps is not None:
            fps = args.source_fps
        elif "fps" in data:
            fps = float(data["fps"])
        else:
            fps = 30.0
        depths = np.asarray(data[depth_key]) if depth_key is not None else None
        pointclouds = np.asarray(data[pc_key]) if pc_key is not None else None
        masks = np.asarray(data[mask_key]) if mask_key is not None else None
    return RGBDSequence(depths=depths, pointclouds=pointclouds, masks=masks, fps=float(fps))


def _load_mkv_sequence(path: pathlib.Path, args: argparse.Namespace) -> RGBDSequence:
    try:
        import pykinect_azure as pykinect
    except ImportError as exc:
        raise ImportError(
            "Reading .mkv requires pykinect_azure. For a dependency-free path, "
            "export depth/pointcloud frames to .npy or .npz first."
        ) from exc

    pykinect.initialize_libraries()
    playback = pykinect.start_playback(str(path))

    depths: list[np.ndarray] = []
    pointclouds: list[np.ndarray] = []
    keep_every = max(1, int(args.source_frame_stride))
    max_source = None if args.max_source_frames <= 0 else int(args.max_source_frames)
    source_idx = 0

    while max_source is None or len(depths) < max_source:
        try:
            capture = playback.update()
        except Exception:
            break
        if capture is None:
            break

        ret_depth, depth = capture.get_depth_image()
        if not ret_depth:
            source_idx += 1
            continue
        if source_idx % keep_every != 0:
            source_idx += 1
            continue

        pc = None
        if hasattr(capture, "get_pointcloud"):
            ret_pc, maybe_pc = capture.get_pointcloud()
            if ret_pc:
                pc = maybe_pc
        if pc is None and hasattr(capture, "camera_transform"):
            pc = capture.camera_transform.depth_image_to_point_cloud(depth)

        depths.append(np.asarray(depth))
        if pc is not None:
            pointclouds.append(np.asarray(pc))
        source_idx += 1

    if not depths:
        raise RuntimeError(f"No depth frames could be read from {path}.")

    fps = args.source_fps if args.source_fps is not None else 30.0
    fps = fps / keep_every
    pcs = np.asarray(pointclouds) if len(pointclouds) == len(depths) else None
    return RGBDSequence(depths=np.asarray(depths), pointclouds=pcs, masks=None, fps=float(fps))


def load_rgbd_sequence(path: pathlib.Path, args: argparse.Namespace) -> RGBDSequence:
    suffix = path.suffix.lower()
    if suffix in {".npy", ".npz"}:
        return _load_numpy_sequence(path, args)
    if suffix == ".mkv":
        return _load_mkv_sequence(path, args)
    raise ValueError(f"Unsupported input suffix '{path.suffix}'. Expected .npy, .npz, or .mkv.")


def load_mask(path: pathlib.Path, args: argparse.Namespace) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.asarray(np.load(path))
    if path.suffix.lower() == ".npz":
        with np.load(path) as data:
            key = _pick_npz_key(data, args.mask_key, MASK_KEYS)
            if key is None:
                key = _infer_npz_key_by_shape(data, pointcloud=False, mask=True)
            if key is None:
                raise KeyError(f"No mask-like array found in {path}. Available keys: {data.files}.")
            return np.asarray(data[key])
    raise ValueError(f"Unsupported mask suffix '{path.suffix}'. Expected .npy or .npz.")


def _auto_scale(values: np.ndarray, explicit: str | float) -> float:
    if str(explicit).lower() != "auto":
        return float(explicit)
    sample = np.asarray(values)
    if sample.size == 0:
        return 1.0
    sample = sample[np.isfinite(sample)]
    sample = sample[np.abs(sample) > 0]
    if sample.size == 0:
        return 1.0
    percentile = float(np.percentile(np.abs(sample[: min(sample.size, 1_000_000)]), 95))
    return 0.001 if percentile > 50.0 else 1.0


def _prepare_depths(depths: np.ndarray | None, args: argparse.Namespace) -> np.ndarray | None:
    if depths is None:
        return None
    if depths.ndim != 3:
        raise ValueError(f"Depth frames must have shape (T,H,W); got {depths.shape}.")
    scale = _auto_scale(depths, args.depth_scale)
    return np.asarray(depths, dtype=np.float32) * scale


def _prepare_pointclouds(pointclouds: np.ndarray | None, args: argparse.Namespace) -> np.ndarray | None:
    if pointclouds is None:
        return None
    if not (pointclouds.ndim == 3 and pointclouds.shape[-1] == 3) and not (
        pointclouds.ndim == 4 and pointclouds.shape[-1] == 3
    ):
        raise ValueError(f"Pointcloud frames must have shape (T,N,3) or (T,H,W,3); got {pointclouds.shape}.")
    scale = _auto_scale(pointclouds, args.pointcloud_scale)
    pcs = np.asarray(pointclouds, dtype=np.float32) * scale
    if args.pointcloud_convention == "camera":
        pcs = pcs.copy()
        pcs[..., 1] *= -1.0
        pcs[..., 2] *= -1.0
    return pcs


def _prepare_masks(masks: np.ndarray | None, args: argparse.Namespace) -> np.ndarray | None:
    if masks is None:
        return None
    masks = np.asarray(masks)
    if masks.dtype == np.bool_:
        return masks
    if masks.ndim == 4 and masks.shape[-1] in {3, 4}:
        if args.mask_mode == "foreground-nonzero":
            masks = np.any(masks[..., :3] > 0, axis=-1)
        else:
            masks = ~np.all(masks[..., :3] >= 250, axis=-1)
    elif masks.ndim in {1, 2, 3}:
        if args.mask_mode == "not-white":
            masks = masks < 250
        else:
            masks = masks > 0
    else:
        raise ValueError(f"Mask frames must have shape (T,H,W), (T,N), or (T,H,W,3); got {masks.shape}.")

    masks = np.asarray(masks, dtype=bool)
    return masks


def _buffer_sampled_masks(sampled_masks: torch.Tensor, buffer: int) -> torch.Tensor:
    if buffer <= 0 or sampled_masks.shape[0] <= 1:
        return sampled_masks
    buffered = sampled_masks.clone()
    for offset in range(1, buffer + 1):
        buffered[offset:] |= sampled_masks[:-offset]
        buffered[:-offset] |= sampled_masks[offset:]
    return buffered


def _intrinsics_from_args(width: int, height: int, args: argparse.Namespace) -> tuple[float, float, float, float]:
    cx = (width - 1) * 0.5 if args.cx is None else float(args.cx)
    cy = (height - 1) * 0.5 if args.cy is None else float(args.cy)
    if args.fx is not None and args.fy is not None:
        return float(args.fx), float(args.fy), cx, cy
    fov_rad = math.radians(float(args.fov_deg))
    fx = (width * 0.5) / math.tan(fov_rad * 0.5)
    fy = fx if args.fy is None else float(args.fy)
    return fx, fy, cx, cy


def _sample_indices_from_grid(height: int, width: int, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.mgrid[0:height:args.pixel_stride, 0:width:args.pixel_stride]
    ys = ys.reshape(-1)
    xs = xs.reshape(-1)
    if args.max_points > 0 and xs.size > args.max_points:
        pick = np.linspace(0, xs.size - 1, args.max_points, dtype=np.int64)
        ys = ys[pick]
        xs = xs[pick]
    return ys.astype(np.int64), xs.astype(np.int64)


def _sample_indices_from_count(count: int, args: argparse.Namespace) -> np.ndarray:
    if args.max_points > 0 and count > args.max_points:
        return np.linspace(0, count - 1, args.max_points, dtype=np.int64)
    return np.arange(count, dtype=np.int64)


def _depth_to_points(depth_samples: torch.Tensor, rays: torch.Tensor) -> torch.Tensor:
    return rays.unsqueeze(0) * depth_samples.unsqueeze(-1)


def build_interpolator(
    sequence: RGBDSequence,
    *,
    args: argparse.Namespace,
    device: str,
):
    depths_np = _prepare_depths(sequence.depths, args)
    pcs_np = _prepare_pointclouds(sequence.pointclouds, args)
    masks_np = _prepare_masks(sequence.masks, args)

    if depths_np is None and pcs_np is None:
        raise ValueError("At least one of depth or pointcloud frames is required.")

    if depths_np is not None:
        num_frames, height, width = depths_np.shape
    else:
        num_frames = int(pcs_np.shape[0])
        height = width = -1

    if pcs_np is not None and pcs_np.shape[0] != num_frames:
        raise ValueError("Depth and pointcloud sequences must have the same number of frames.")
    rays = None
    sampled_depths = None
    sampled_points = None
    sampled_masks = None

    if depths_np is not None:
        ys, xs = _sample_indices_from_grid(height, width, args)
        sampled_depths = torch.as_tensor(depths_np[:, ys, xs], dtype=torch.float32, device=device)
        if masks_np is not None:
            if masks_np.ndim == 2 and masks_np.shape == (height, width):
                sampled_mask_values = masks_np[ys, xs][None, :]
            elif masks_np.ndim == 3 and masks_np.shape[-2:] == (height, width):
                sampled_mask_values = masks_np[:, ys, xs]
            else:
                raise ValueError("Depth masks must have shape (T,H,W) matching depth frames.")
            sampled_masks = torch.as_tensor(sampled_mask_values, dtype=torch.bool, device=device)
        fx, fy, cx, cy = _intrinsics_from_args(width, height, args)
        x_ray = (xs.astype(np.float32) - cx) / fx
        y_ray = -(ys.astype(np.float32) - cy) / fy
        z_ray = -np.ones_like(x_ray, dtype=np.float32)
        rays = torch.as_tensor(np.stack([x_ray, y_ray, z_ray], axis=-1), dtype=torch.float32, device=device)

    if pcs_np is not None:
        if pcs_np.ndim == 4:
            _, pc_h, pc_w, _ = pcs_np.shape
            ys, xs = _sample_indices_from_grid(pc_h, pc_w, args)
            sampled = pcs_np[:, ys, xs, :]
            if sampled_masks is None and masks_np is not None:
                if masks_np.ndim == 2 and masks_np.shape == (pc_h, pc_w):
                    sampled_mask_values = masks_np[ys, xs][None, :]
                elif masks_np.ndim == 3 and masks_np.shape[-2:] == (pc_h, pc_w):
                    sampled_mask_values = masks_np[:, ys, xs]
                else:
                    raise ValueError("Pointcloud grid masks must have shape (T,H,W) matching pointcloud frames.")
                sampled_masks = torch.as_tensor(sampled_mask_values, dtype=torch.bool, device=device)
        else:
            idx = _sample_indices_from_count(pcs_np.shape[1], args)
            sampled = pcs_np[:, idx, :]
            if sampled_masks is None and masks_np is not None:
                if masks_np.ndim == 1 and masks_np.shape[0] == pcs_np.shape[1]:
                    sampled_mask_values = masks_np[idx][None, :]
                elif masks_np.ndim == 2 and masks_np.shape[1] == pcs_np.shape[1]:
                    sampled_mask_values = masks_np[:, idx]
                else:
                    raise ValueError("Flat pointcloud masks must have shape (T,N) matching pointcloud frames.")
                sampled_masks = torch.as_tensor(sampled_mask_values, dtype=torch.bool, device=device)
        sampled_points = torch.as_tensor(sampled, dtype=torch.float32, device=device)

    if sampled_masks is not None:
        if sampled_masks.shape[0] not in {1, num_frames}:
            raise ValueError("Mask sequence must have one frame or the same number of frames as the RGBD sequence.")
        sampled_masks = _buffer_sampled_masks(sampled_masks, int(args.mask_buffer))

    source_fps = float(sequence.fps)
    total_time = (num_frames - 1) / source_fps
    depth_min = float(args.depth_min)
    depth_max = float(args.depth_max)

    def interpolate_pair(time: float):
        clamped_time = min(max(float(time), 0.0), total_time)
        position = clamped_time * source_fps
        i0 = min(int(math.floor(position)), num_frames - 1)
        i1 = min(i0 + 1, num_frames - 1)
        alpha = float(position - i0)

        if sampled_points is not None:
            p0 = sampled_points[i0]
            p1 = sampled_points[i1]
        else:
            d0_for_points = sampled_depths[i0]
            d1_for_points = sampled_depths[i1]
            p0 = _depth_to_points(d0_for_points, rays).squeeze(0)
            p1 = _depth_to_points(d1_for_points, rays).squeeze(0)

        if sampled_depths is not None:
            d0 = sampled_depths[i0]
            d1 = sampled_depths[i1]
        else:
            d0 = torch.linalg.norm(p0, dim=-1)
            d1 = torch.linalg.norm(p1, dim=-1)

        valid0 = torch.isfinite(d0) & (d0 >= depth_min) & (d0 <= depth_max)
        valid1 = torch.isfinite(d1) & (d1 >= depth_min) & (d1 <= depth_max)
        if args.zero_fill:
            p0 = torch.where(valid0.unsqueeze(-1), p0, p1)
            p1 = torch.where(valid1.unsqueeze(-1), p1, p0)
            d0 = torch.where(valid0, d0, d1)
            d1 = torch.where(valid1, d1, d0)

        points = p0 * (1.0 - alpha) + p1 * alpha
        depths = d0 * (1.0 - alpha) + d1 * alpha
        valid = (
            torch.isfinite(depths)
            & (depths >= depth_min)
            & (depths <= depth_max)
            & torch.isfinite(points).all(dim=-1)
        )
        if sampled_masks is not None:
            mask0 = sampled_masks[0 if sampled_masks.shape[0] == 1 else i0]
            mask1 = sampled_masks[0 if sampled_masks.shape[0] == 1 else i1]
            valid = valid & (mask0 | mask1)
        points = points[valid].contiguous()
        intensities = torch.ones(points.shape[0], dtype=torch.float32, device=device)
        return intensities, points

    return interpolate_pair, total_time, num_frames, source_fps


def save_rd_png(
    path: pathlib.Path,
    rd_db: np.ndarray,
    ranges: np.ndarray,
    velocities: np.ndarray,
    *,
    title: str,
    vmin: float | None,
    vmax: float | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4), dpi=160)
    im = ax.imshow(
        rd_db,
        extent=[float(ranges[0]), float(ranges[-1]), float(velocities[0]), float(velocities[-1])],
        origin="lower",
        aspect="auto",
        cmap="jet",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Magnitude (dB)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def generate_range_doppler(args: argparse.Namespace) -> None:
    from witwin.radar import Radar, RadarConfig
    from witwin.radar.sigproc import process_rd

    input_path = pathlib.Path(args.input).expanduser().resolve()
    output_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.pixel_stride <= 0:
        raise ValueError("--pixel-stride must be positive.")
    if args.start_frame < 0:
        raise ValueError("--start-frame must be non-negative.")

    config = _load_config(args.config)
    device = args.device
    if args.backend == "dirichlet" and device != "cuda":
        raise ValueError("The default dirichlet backend requires --device cuda.")

    sequence = load_rgbd_sequence(input_path, args)
    if args.mask is not None:
        sequence.masks = load_mask(pathlib.Path(args.mask).expanduser().resolve(), args)
    radar = Radar(RadarConfig.from_dict(config), backend=args.backend, device=device)
    interpolator, total_time, source_frames, source_fps = build_interpolator(sequence, args=args, device=radar.device)

    chirp_period = (radar.idle_time + radar.ramp_end_time) * 1e-6
    radar_valid_time = chirp_period * radar.num_tx * max(0, radar.chirp_per_frame - 1)
    start_time = args.start_frame / source_fps
    remaining_time = total_time - start_time
    max_output_frames = max(0, int(math.floor((remaining_time - radar_valid_time) * radar.frame_per_second)) + 1)
    if max_output_frames <= 0:
        raise ValueError(
            "Input sequence is too short for one radar frame. "
            f"Need more than {radar_valid_time:.4f}s after --start-frame; got {remaining_time:.4f}s."
        )
    num_frames = min(args.num_frames, max_output_frames) if args.num_frames > 0 else max_output_frames

    print(f"Input: {input_path}")
    print(f"Source frames: {source_frames} at {source_fps:.3f} fps")
    print(f"Radar: backend={args.backend} device={radar.device} start_frame={args.start_frame} output_frames={num_frames}")

    rd_maps = []
    ranges = None
    velocities = None
    for frame_idx in range(num_frames):
        t0 = start_time + frame_idx / radar.frame_per_second
        frame = radar.mimo(interpolator, t0=t0)
        rd_db, _, ranges, velocities = process_rd(
            radar,
            frame,
            tx=args.tx,
            rx=args.rx,
            static_clutter_removal=args.static_clutter_removal,
        )
        rd_db = rd_db[:, : len(ranges)]
        rd_maps.append(rd_db.astype(np.float32, copy=False))

        png_path = output_dir / f"rd_frame_{frame_idx:04d}.png"
        save_rd_png(
            png_path,
            rd_db,
            ranges,
            velocities,
            title=f"RGBD Range-Doppler Frame {frame_idx}",
            vmin=args.db_min,
            vmax=args.db_max,
        )
        print(f"[{frame_idx + 1:03d}/{num_frames:03d}] saved {png_path}")

    rd_stack = np.stack(rd_maps, axis=0)
    np.save(output_dir / "rd_maps_db.npy", rd_stack)
    np.savez(
        output_dir / "rd_axes.npz",
        ranges=np.asarray(ranges),
        velocities=np.asarray(velocities),
        tx=np.asarray(args.tx),
        rx=np.asarray(args.rx),
    )
    print(f"Saved RD stack: {output_dir / 'rd_maps_db.npy'}  shape={rd_stack.shape}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to .npy, .npz, or Azure Kinect .mkv input.")
    parser.add_argument("--output-dir", default="output/rgbd_range_doppler", help="Directory for PNGs and .npy output.")
    parser.add_argument("--config", default=None, help="Optional radar config JSON. Defaults to TI1843-like config.")
    parser.add_argument("--backend", default="dirichlet", choices=("dirichlet", "slang", "pytorch"))
    parser.add_argument("--device", default="cuda", help="Torch device. Dirichlet/slang require cuda.")
    parser.add_argument("--num-frames", type=int, default=10, help="Number of radar frames to generate. Use 0 for all.")
    parser.add_argument("--start-frame", type=int, default=0, help="Source RGBD frame index to start from.")
    parser.add_argument("--source-fps", type=float, default=None, help="RGBD source frame rate; defaults to npz fps or 30.")
    parser.add_argument("--source-frame-stride", type=int, default=1, help="Keep every Nth frame when reading .mkv.")
    parser.add_argument("--max-source-frames", type=int, default=0, help="Limit .mkv frames loaded; 0 means no limit.")
    parser.add_argument("--depth-key", default=None, help="Depth array key for .npz inputs.")
    parser.add_argument("--pointcloud-key", default=None, help="Pointcloud array key for .npz inputs.")
    parser.add_argument("--mask", default=None, help="Optional .npy/.npz human/body mask path.")
    parser.add_argument("--mask-key", default=None, help="Mask array key for .npz inputs.")
    parser.add_argument(
        "--mask-mode",
        choices=("auto", "not-white", "foreground-nonzero"),
        default="auto",
        help="How to convert numeric/RGB masks to foreground.",
    )
    parser.add_argument("--mask-buffer", type=int, default=1, help="Temporal OR buffer in source frames for masks.")
    parser.add_argument("--depth-scale", default="auto", help="Scale depth to meters; use 0.001 for millimeters.")
    parser.add_argument("--pointcloud-scale", default="auto", help="Scale pointcloud coordinates to meters.")
    parser.add_argument("--pointcloud-convention", choices=("camera", "radar"), default="camera")
    parser.add_argument("--fov-deg", type=float, default=70.0, help="Horizontal FOV used when fx/fy are omitted.")
    parser.add_argument("--fx", type=float, default=None, help="Depth camera focal length in pixels.")
    parser.add_argument("--fy", type=float, default=None, help="Depth camera focal length in pixels.")
    parser.add_argument("--cx", type=float, default=None, help="Depth camera principal point x in pixels.")
    parser.add_argument("--cy", type=float, default=None, help="Depth camera principal point y in pixels.")
    parser.add_argument("--pixel-stride", type=int, default=2, help="Regular spatial downsampling stride.")
    parser.add_argument("--max-points", type=int, default=4096, help="Max RGBD points per chirp. Use 0 for all.")
    parser.add_argument("--depth-min", type=float, default=0.10, help="Minimum valid depth in meters.")
    parser.add_argument("--depth-max", type=float, default=20.0, help="Maximum valid depth in meters.")
    parser.add_argument("--zero-fill", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tx", type=int, default=0, help="TX index used for RD visualization.")
    parser.add_argument("--rx", type=int, default=0, help="RX index used for RD visualization.")
    parser.add_argument(
        "--static-clutter-removal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Subtract slow-time mean before RD FFT.",
    )
    parser.add_argument("--db-min", type=float, default=None, help="PNG color lower bound in dB.")
    parser.add_argument("--db-max", type=float, default=None, help="PNG color upper bound in dB.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    generate_range_doppler(args)


if __name__ == "__main__":
    main()
