"""Benchmark the native Dirichlet CUDA kernels against PyTorch references.

Example:
    python tools/benchmark_dirichlet_cuda.py --targets 1024 16384 262144 --runs 20
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if str(REPO_ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tests"))

from witwin.radar import Radar, RadarConfig  # noqa: E402

# The Torch reference lives under tests/, where a CPU reference oracle belongs;
# this benchmark exists to compare the native kernel against it.
from reference.dsp_oracles import pytorch_chirp_reference  # noqa: E402


CONFIG = {
    "num_tx": 1,
    "num_rx": 1,
    "fc": 77e9,
    "slope": 100.0,
    "adc_samples": 400,
    "adc_start_time": 0,
    "sample_rate": 10000,
    "idle_time": 0,
    "ramp_end_time": 40,
    "chirp_per_frame": 1,
    "frame_per_second": 1,
    "num_doppler_bins": 1,
    "num_range_bins": 400,
    "num_angle_bins": 1,
    "power": 1,
    "tx_loc": [[0, 0, 0]],
    "rx_loc": [[0, 0, 0]],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=int, nargs="+", default=[1024, 16384, 262144, 1048576])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--max-reference-targets", type=int, default=262144)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON after the table.")
    return parser.parse_args()


def cuda_time(fn, *, warmup: int, runs: int) -> tuple[float, list[float]]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))
    return statistics.median(times), times


def make_targets(num_targets: int) -> tuple[torch.Tensor, torch.Tensor]:
    distances = torch.empty(num_targets, dtype=torch.float32, device="cuda").uniform_(0.5, 5.0)
    amplitudes = torch.empty(num_targets, dtype=torch.float32, device="cuda").uniform_(0.5, 1.0)
    return distances.contiguous(), amplitudes.contiguous()


def torch_reference_forward(radar: Radar, distances: torch.Tensor, amplitudes: torch.Tensor) -> torch.Tensor:
    signal = pytorch_chirp_reference(radar, distances.to(torch.float64), amplitudes.to(torch.float64))
    return torch.fft.fft(signal, n=radar.solver.N_fft)[: radar.solver.N_fft // 2]


def torch_reference_backward(
    radar: Radar,
    distances: torch.Tensor,
    amplitudes: torch.Tensor,
    grad_re: torch.Tensor,
    grad_im: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    d = distances.detach().to(torch.float64).requires_grad_(True)
    a = amplitudes.detach().to(torch.float64).requires_grad_(True)
    spectrum = torch_reference_forward(radar, d, a)
    loss = (spectrum.real * grad_re.to(torch.float64) + spectrum.imag * grad_im.to(torch.float64)).sum()
    loss.backward()
    return d.grad.to(torch.float32), a.grad.to(torch.float32)


def native_autograd_step(
    radar: Radar,
    distances: torch.Tensor,
    amplitudes: torch.Tensor,
    grad_re: torch.Tensor,
    grad_im: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    d = distances.detach().requires_grad_(True)
    a = amplitudes.detach().requires_grad_(True)
    spectrum = radar.chirp(d, a)
    loss = (spectrum.real * grad_re + spectrum.imag * grad_im).sum()
    return torch.autograd.grad(loss, (d, a))


def peak_memory_mb(fn) -> float:
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)


def maybe_speedup(reference_ms: float | None, native_ms: float) -> float | None:
    if reference_ms is None or native_ms <= 0:
        return None
    return reference_ms / native_ms


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the native Dirichlet benchmark.")

    radar = Radar(RadarConfig.from_dict(CONFIG), device="cuda")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"fft_bins: {radar.solver.N_fft // 2}, runs: {args.runs}, warmup: {args.warmup}")
    print()
    print(
        f"{'targets':>10}  {'native_fwd':>11}  {'torch_fwd':>11}  {'fwd_x':>7}  "
        f"{'native_bwd':>11}  {'native_e2e':>11}  {'torch_e2e':>11}  {'e2e_x':>7}"
    )
    print("-" * 98)

    results = []
    for num_targets in args.targets:
        distances, amplitudes = make_targets(num_targets)
        grad_re = torch.randn(radar.solver.N_fft // 2, dtype=torch.float32, device="cuda")
        grad_im = torch.randn(radar.solver.N_fft // 2, dtype=torch.float32, device="cuda")

        native_fwd_ms, _ = cuda_time(lambda: radar.chirp(distances, amplitudes), warmup=args.warmup, runs=args.runs)
        native_bwd_ms, _ = cuda_time(
            lambda: radar.solver.backward(distances, amplitudes, grad_re, grad_im),
            warmup=args.warmup,
            runs=args.runs,
        )
        native_e2e_ms, _ = cuda_time(
            lambda: native_autograd_step(radar, distances, amplitudes, grad_re, grad_im),
            warmup=args.warmup,
            runs=args.runs,
        )
        native_peak_mb = peak_memory_mb(
            lambda: native_autograd_step(radar, distances, amplitudes, grad_re, grad_im)
        )

        torch_fwd_ms = None
        torch_e2e_ms = None
        torch_peak_mb = None
        if num_targets <= args.max_reference_targets:
            torch_fwd_ms, _ = cuda_time(
                lambda: torch_reference_forward(radar, distances, amplitudes),
                warmup=args.warmup,
                runs=args.runs,
            )
            torch_e2e_ms, _ = cuda_time(
                lambda: torch_reference_backward(radar, distances, amplitudes, grad_re, grad_im),
                warmup=args.warmup,
                runs=args.runs,
            )
            torch_peak_mb = peak_memory_mb(
                lambda: torch_reference_backward(radar, distances, amplitudes, grad_re, grad_im)
            )

        fwd_x = maybe_speedup(torch_fwd_ms, native_fwd_ms)
        e2e_x = maybe_speedup(torch_e2e_ms, native_e2e_ms)
        results.append(
            {
                "targets": num_targets,
                "native_forward_ms": native_fwd_ms,
                "torch_forward_ms": torch_fwd_ms,
                "forward_speedup": fwd_x,
                "native_backward_ms": native_bwd_ms,
                "native_autograd_e2e_ms": native_e2e_ms,
                "torch_autograd_e2e_ms": torch_e2e_ms,
                "autograd_e2e_speedup": e2e_x,
                "native_autograd_peak_mb": native_peak_mb,
                "torch_autograd_peak_mb": torch_peak_mb,
            }
        )
        torch_fwd = f"{torch_fwd_ms:11.3f}" if torch_fwd_ms is not None else f"{'skip':>11}"
        torch_e2e = f"{torch_e2e_ms:11.3f}" if torch_e2e_ms is not None else f"{'skip':>11}"
        fwd_speed = f"{fwd_x:7.2f}" if fwd_x is not None else f"{'n/a':>7}"
        e2e_speed = f"{e2e_x:7.2f}" if e2e_x is not None else f"{'n/a':>7}"
        print(
            f"{num_targets:10d}  {native_fwd_ms:11.3f}  {torch_fwd}  {fwd_speed}  "
            f"{native_bwd_ms:11.3f}  {native_e2e_ms:11.3f}  {torch_e2e}  {e2e_speed}"
        )

    if args.json:
        print()
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
