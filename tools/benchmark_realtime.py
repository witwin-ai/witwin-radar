"""Benchmark steady-state radar simulation FPS.

This script separates three costs:
- preprocessed reflection points -> MIMO frame generation
- optional Range-Doppler / point-cloud DSP
- optional ray tracing for a small box scene

Example:
    python tools/benchmark_realtime.py --targets 1024 4096 16384 --runs 20
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from witwin.radar import (  # noqa: E402
    Box,
    Material,
    Radar,
    RadarConfig,
    Scene,
    Structure,
    TraceResult,
    Tracer,
    TransformMotion,
)
from witwin.radar.sigproc import process_pc, process_pc_tensor, process_rd, process_rd_tensor  # noqa: E402


STANDARD_CONFIG = {
    "num_tx": 3,
    "num_rx": 4,
    "fc": 77e9,
    "slope": 60.012,
    "adc_samples": 256,
    "adc_start_time": 6,
    "sample_rate": 4400,
    "idle_time": 7,
    "ramp_end_time": 58,
    "chirp_per_frame": 128,
    "frame_per_second": 10,
    "num_doppler_bins": 128,
    "num_range_bins": 256,
    "num_angle_bins": 64,
    "power": 12,
    "tx_loc": [[0, 0, 0], [2, 0, 0], [0, 1, 0]],
    "rx_loc": [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=int, nargs="+", default=[256, 1024, 4096, 16384])
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--chirps", type=int, default=128)
    parser.add_argument("--adc-samples", type=int, default=256)
    parser.add_argument("--tx", type=int, default=3)
    parser.add_argument("--rx", type=int, default=4)
    parser.add_argument("--freq-domain", action="store_true", help="Keep Dirichlet MIMO output in frequency domain.")
    parser.add_argument("--with-rd", action="store_true", help="Also benchmark process_rd on the generated frame.")
    parser.add_argument("--with-pc", action="store_true", help="Also benchmark process_pc on the generated frame.")
    parser.add_argument("--tensor-dsp", action="store_true", help="Keep DSP benchmark outputs on the torch device.")
    parser.add_argument("--with-autograd", action="store_true", help="Benchmark static MIMO forward and backward.")
    parser.add_argument("--with-trace", action="store_true", help="Also benchmark a small pixel-ray traced scene.")
    parser.add_argument("--with-dynamic-trace", action="store_true", help="Benchmark full moving-scene trace and MIMO.")
    parser.add_argument("--dynamic-sampling", choices=("pixel", "triangle"), default="triangle")
    parser.add_argument("--dynamic-motion-sampling", choices=("per_chirp", "linear"), default="linear")
    parser.add_argument("--trace-resolution", type=int, default=128)
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> dict:
    cfg = dict(STANDARD_CONFIG)
    cfg["chirp_per_frame"] = args.chirps
    cfg["num_doppler_bins"] = args.chirps
    cfg["adc_samples"] = args.adc_samples
    cfg["num_range_bins"] = args.adc_samples
    cfg["num_tx"] = args.tx
    cfg["num_rx"] = args.rx
    cfg["tx_loc"] = make_tx_layout(args.tx, args.rx)
    cfg["rx_loc"] = [[i, 0, 0] for i in range(args.rx)]
    return cfg


def make_tx_layout(num_tx: int, num_rx: int) -> list[list[int]]:
    if num_tx == 1:
        return [[0, 0, 0]]
    if num_tx == 3:
        return [[0, 0, 0], [2, 0, 0], [0, 1, 0]]
    if num_tx == 4:
        return [[0, 0, 0], [num_rx, 0, 0], [0, 1, 0], [num_rx, 1, 0]]
    if num_tx % 2 == 0:
        return [[0 if i % 2 == 0 else num_rx, i // 2, 0] for i in range(num_tx)]
    raise ValueError("Only num_tx=1, 3, 4, or even TX counts are supported by this benchmark helper.")


def make_preprocessed_trace(num_targets: int, *, device: torch.device) -> TraceResult:
    side = math.ceil(math.sqrt(num_targets))
    xs = torch.linspace(-1.5, 1.5, side, dtype=torch.float32, device=device)
    ys = torch.linspace(-0.9, 0.9, side, dtype=torch.float32, device=device)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
    points = torch.stack(
        (
            grid_x.reshape(-1)[:num_targets],
            grid_y.reshape(-1)[:num_targets],
            torch.full((num_targets,), -3.0, dtype=torch.float32, device=device),
        ),
        dim=1,
    ).contiguous()
    intensities = torch.full((num_targets,), 0.2, dtype=torch.float32, device=device)
    normals = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32, device=device).repeat(num_targets, 1)
    return TraceResult(points, intensities, normals=normals)


def make_box_scene(*, device: str) -> Scene:
    scene = Scene(device=device)
    for name, position, size in (
        ("box_a", (0.0, 0.0, -3.0), (0.8, 1.6, 0.4)),
        ("box_b", (-0.9, -0.2, -4.0), (0.6, 0.8, 0.6)),
        ("wall", (0.0, 0.0, -5.5), (5.0, 3.0, 0.05)),
    ):
        scene.add_structure(
            Structure(
                name=name,
                geometry=Box(position=position, size=size),
                material=Material(eps_r=3.0),
            )
        )
    return scene


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(label: str, fn, *, warmup: int, runs: int, device: torch.device):
    for _ in range(warmup):
        result = fn()
        sync(device)

    times = []
    result = None
    for _ in range(runs):
        start = time.perf_counter()
        result = fn()
        sync(device)
        times.append((time.perf_counter() - start) * 1000.0)

    med = statistics.median(times)
    avg = statistics.mean(times)
    print(f"{label:<28} median={med:8.3f} ms  mean={avg:8.3f} ms  fps={1000.0 / med:8.2f}")
    return result, times


def mimo_autograd_step(radar: Radar, trace: TraceResult):
    points = trace.points.detach().requires_grad_(True)
    intensities = trace.intensities.detach().requires_grad_(True)
    differentiable_trace = TraceResult(points, intensities, normals=trace.normals)
    frame = radar.mimo_from_trace(differentiable_trace)
    return torch.autograd.grad(frame.abs().square().sum(), (points, intensities))


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for native Dirichlet simulation.")

    device = torch.device("cuda")
    radar = Radar(RadarConfig.from_dict(make_config(args)), device=device)
    cfg = radar.config

    print(f"device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else device}")
    print(
        "config: "
        f"tx={cfg.num_tx}, rx={cfg.num_rx}, "
        f"chirps={cfg.chirp_per_frame}, adc={cfg.adc_samples}, freq_domain={args.freq_domain}"
    )
    print()

    if args.with_trace:
        scene = make_box_scene(device=str(device))
        tracer = Tracer(scene, radar, resolution=args.trace_resolution, sampling="pixel")
        trace, _ = benchmark(
            f"trace pixel {args.trace_resolution}x{args.trace_resolution}",
            tracer.trace,
            warmup=args.warmup,
            runs=args.runs,
            device=device,
        )
        print(f"  traced reflection points: {trace.points.shape[0]}")
        print()

    if args.with_dynamic_trace:
        dynamic_scene = make_box_scene(device=str(device))
        dynamic_scene.add_structure_motion("box_a", TransformMotion(velocity=(0.2, 0.0, 0.0)))
        dynamic_frame, _ = benchmark(
            f"dynamic {args.dynamic_sampling}/{args.dynamic_motion_sampling}",
            lambda: radar.simulate(
                dynamic_scene,
                resolution=args.trace_resolution,
                sampling=args.dynamic_sampling,
                motion_sampling=args.dynamic_motion_sampling,
            ),
            warmup=args.warmup,
            runs=args.runs,
            device=device,
        )
        print(f"  dynamic frame shape: {tuple(dynamic_frame.shape)}")
        print()

    for num_targets in args.targets:
        trace = make_preprocessed_trace(num_targets, device=device)
        mimo_options = {"freq_domain": True} if args.freq_domain else {}

        print(f"targets: {num_targets}")
        frame, _ = benchmark(
            "cached trace -> mimo",
            lambda: radar.mimo_from_trace(trace, **mimo_options),
            warmup=args.warmup,
            runs=args.runs,
            device=device,
        )
        print(f"  frame shape: {tuple(frame.shape)}")

        if args.with_rd and not args.freq_domain:
            rd_fn = process_rd_tensor if args.tensor_dsp else process_rd
            benchmark(
                "process_rd tensor" if args.tensor_dsp else "process_rd numpy",
                lambda: rd_fn(radar, frame),
                warmup=args.warmup,
                runs=args.runs,
                device=device,
            )
        if args.with_pc and not args.freq_domain:
            pc_fn = process_pc_tensor if args.tensor_dsp else process_pc
            benchmark(
                "process_pc tensor" if args.tensor_dsp else "process_pc numpy",
                lambda: pc_fn(radar, frame),
                warmup=args.warmup,
                runs=args.runs,
                device=device,
            )
        if args.with_autograd and not args.freq_domain:
            benchmark(
                "mimo autograd e2e",
                lambda: mimo_autograd_step(radar, trace),
                warmup=args.warmup,
                runs=args.runs,
                device=device,
            )
        print()


if __name__ == "__main__":
    main()
