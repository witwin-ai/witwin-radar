"""
Preprocess an RFGen RGBD recording into a single .npz file.

The expected source layout matches the user's RFGen_RD folder:
    - bin_depths_short_resized.npy : depth frames in meters, shape (T, H, W)
    - color.mp4                    : optional RGB preview video

The output .npz contains:
    - depths : float32, shape (T, H, W), meters
    - masks  : bool,    shape (T, H, W)
    - rgb    : uint8,   shape (T, H, W, 3), optional, resized to depth shape
    - fps    : float32
    - frame_indices : int32, original source frame ids after trimming
"""

from __future__ import annotations

import argparse
import pathlib

import cv2
import numpy as np


DEFAULT_INPUT_DIR = pathlib.Path(r"E:\Research2026\RFGen1.7\RFGen_RD")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=pathlib.Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--depth-npy", type=str, default="bin_depths_short_resized.npy")
    parser.add_argument("--color-video", type=str, default="color.mp4")
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1] / "output" / "rfgen_rgbd_sequence.npz",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means use all frames.")
    parser.add_argument(
        "--rgb-mode",
        choices=("resize-to-depth", "skip"),
        default="resize-to-depth",
        help="Store resized RGB preview frames or omit RGB entirely.",
    )
    return parser.parse_args()


def _load_depths(path: pathlib.Path) -> np.ndarray:
    depths = np.load(path).astype(np.float32, copy=False)
    if depths.ndim != 3:
        raise ValueError(f"Expected depth array shape (T,H,W); got {depths.shape}.")
    return depths


def _read_video_frames(path: pathlib.Path, *, limit: int | None, start_frame: int, size: tuple[int, int]) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")

    try:
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames: list[np.ndarray] = []
        while limit is None or len(frames) < limit:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if frame_rgb.shape[:2] != size:
                frame_rgb = cv2.resize(frame_rgb, (size[1], size[0]), interpolation=cv2.INTER_AREA)
            frames.append(frame_rgb)
    finally:
        cap.release()

    if not frames:
        raise RuntimeError(f"No frames read from {path}")
    return np.stack(frames, axis=0)


def main() -> None:
    args = _parse_args()

    input_dir = args.input_dir.resolve()
    depth_path = input_dir / args.depth_npy
    color_path = input_dir / args.color_video
    output_path = args.output.resolve()

    depths = _load_depths(depth_path)
    total_depth_frames, height, width = depths.shape

    start = max(0, int(args.start_frame))
    if start >= total_depth_frames:
        raise ValueError(f"start-frame {start} is outside depth sequence length {total_depth_frames}.")

    depth_limit = None if args.max_frames <= 0 else int(args.max_frames)
    depths = depths[start:] if depth_limit is None else depths[start : start + depth_limit]
    frame_indices = np.arange(start, start + len(depths), dtype=np.int32)
    masks = depths > 0.0

    arrays: dict[str, np.ndarray] = {
        "depths": depths,
        "masks": masks,
        "fps": np.asarray(float(args.fps), dtype=np.float32),
        "frame_indices": frame_indices,
    }

    if args.rgb_mode != "skip":
        rgb_limit = len(depths)
        rgb = _read_video_frames(color_path, limit=rgb_limit, start_frame=start, size=(height, width))
        common = min(len(depths), len(rgb))
        if common != len(depths):
            depths = depths[:common]
            masks = masks[:common]
            frame_indices = frame_indices[:common]
            arrays["depths"] = depths
            arrays["masks"] = masks
            arrays["frame_indices"] = frame_indices
        arrays["rgb"] = rgb[:common]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)

    print(f"saved: {output_path}")
    print(f"depths: {arrays['depths'].shape} {arrays['depths'].dtype}")
    print(f"masks:  {arrays['masks'].shape} {arrays['masks'].dtype}")
    if "rgb" in arrays:
        print(f"rgb:    {arrays['rgb'].shape} {arrays['rgb'].dtype}")
    print(f"fps:    {float(arrays['fps'])}")


if __name__ == "__main__":
    main()
