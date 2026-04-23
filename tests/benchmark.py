"""
Benchmark: PyTorch vs Slang+FFT vs Dirichlet
Slang: chunked (default) and per-target strategies
Dirichlet: chunked (default, fastest)
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "witwin-radar"))

import time
import numpy as np
import torch

from core import Radar, RadarConfig
from core.solvers import solver_slang
from core.solvers import solver_dirichlet

_config = {
    "num_tx": 1, "num_rx": 1, "fc": 77, "slope": 100,
    "adc_samples": 400, "adc_start_time": 0, "sample_rate": 10000,
    "idle_time": 0, "ramp_end_time": 40, "chirp_per_frame": 1,
    "frame_per_second": 1, "num_doppler_bins": 1, "num_range_bins": 400,
    "num_angle_bins": 1, "power": 1,
    "tx_loc": [[0, 0, 0]], "rx_loc": [[0, 0, 0]],
}


# ============================================================================
# PyTorch (batched, external sum via tensor ops)
# ============================================================================
def compute_pytorch(radar_py, distances, amplitudes, pad_factor=16):
    N_samples = radar_py.adc_samples
    N_fft = N_samples * pad_factor
    num_targets = len(distances)

    dist_tensor = torch.tensor(distances, dtype=torch.float64, device='cuda') * 2
    dist_tensor = dist_tensor.view(1, 1, num_targets, 1)
    amp_tensor = torch.tensor(amplitudes, dtype=torch.float64, device='cuda')

    toa = dist_tensor / radar_py.c0
    t_shifted = radar_py.t_sample.view(1, 1, 1, -1) - toa

    fc = radar_py.fc * t_shifted + 0.5 * (radar_py.slope * 1e12) * t_shifted * t_shifted
    rx = torch.exp(1j * 2 * torch.pi * fc)

    sig = radar_py.tx_waveform * torch.conj(rx)
    sig_weighted = sig * amp_tensor.view(1, 1, num_targets, 1)
    sig_total = sig_weighted.sum(dim=2).squeeze()

    fft_result = torch.fft.fft(sig_total, N_fft)
    return fft_result[:N_fft // 2]


# ============================================================================
# Slang+FFT: chunked (default)
# ============================================================================
def compute_slang(radar_slang, distances, amplitudes, pad_factor=16):
    N_fft = radar_slang.adc_samples * pad_factor

    dist_tensor = torch.tensor(distances, dtype=torch.float32) * 2
    amp_tensor = torch.tensor(amplitudes, dtype=torch.float32)

    sig = solver_slang.chirp_slang(radar_slang.solver, dist_tensor, amp_tensor)
    fft_result = torch.fft.fft(sig, N_fft)
    return fft_result[:N_fft // 2]


# ============================================================================
# Slang+FFT: per-target (alternative for large N)
# ============================================================================
def compute_slang_per_target(radar_slang, distances, amplitudes, pad_factor=16):
    N_fft = radar_slang.adc_samples * pad_factor

    dist_tensor = torch.tensor(distances, dtype=torch.float32) * 2
    amp_tensor = torch.tensor(amplitudes, dtype=torch.float32)

    sig = solver_slang.chirp_slang_per_target(radar_slang.solver, dist_tensor, amp_tensor)
    fft_result = torch.fft.fft(sig, N_fft)
    return fft_result[:N_fft // 2]


# ============================================================================
# Dirichlet: chunked (default, fastest)
# ============================================================================
def compute_dirichlet(radar_dir, distances, amplitudes):
    d = torch.tensor(distances, dtype=torch.float32, device='cuda')
    a = torch.tensor(amplitudes, dtype=torch.float32, device='cuda')
    return solver_dirichlet.spectrum(radar_dir.solver, d, a)


def benchmark(num_targets_list, num_warmup=3, num_runs=10):
    cfg = RadarConfig.from_dict(_config)
    radar_py = Radar(cfg, backend="pytorch")
    radar_slang = Radar(cfg, backend="slang")
    radar_dir = Radar(cfg, backend="dirichlet")
    pad_factor = radar_dir.solver.pad_factor

    print(f"{'Targets':<10} {'PyTorch':<10} {'Slang':<10} {'Slang_pt':<10} {'Dirichlet':<10}")
    print("=" * 50)

    for num_targets in num_targets_list:
        distances = np.random.uniform(0.5, 5.0, num_targets).tolist()
        amplitudes = np.random.uniform(0.5, 1.0, num_targets).tolist()

        # Warmup
        for _ in range(num_warmup):
            if num_targets <= 262144:
                _ = compute_pytorch(radar_py, distances, amplitudes, pad_factor)
            _ = compute_slang(radar_slang, distances, amplitudes, pad_factor)
            _ = compute_slang_per_target(radar_slang, distances, amplitudes, pad_factor)
            _ = compute_dirichlet(radar_dir, distances, amplitudes)
        torch.cuda.synchronize()

        # Benchmark each method
        times = {}

        # Skip PyTorch for large N (memory limit: N x T x 16 bytes)
        if num_targets <= 262144:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(num_runs):
                _ = compute_pytorch(radar_py, distances, amplitudes, pad_factor)
            torch.cuda.synchronize()
            times['pytorch'] = (time.perf_counter() - t0) / num_runs * 1000
        else:
            times['pytorch'] = float('nan')

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(num_runs):
            _ = compute_slang(radar_slang, distances, amplitudes, pad_factor)
        torch.cuda.synchronize()
        times['slang'] = (time.perf_counter() - t0) / num_runs * 1000

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(num_runs):
            _ = compute_slang_per_target(radar_slang, distances, amplitudes, pad_factor)
        torch.cuda.synchronize()
        times['slang_pt'] = (time.perf_counter() - t0) / num_runs * 1000

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(num_runs):
            _ = compute_dirichlet(radar_dir, distances, amplitudes)
        torch.cuda.synchronize()
        times['dirichlet'] = (time.perf_counter() - t0) / num_runs * 1000

        pt_str = f"{times['pytorch']:<10.2f}" if not np.isnan(times['pytorch']) else "---       "
        print(f"{num_targets:<10} {pt_str} {times['slang']:<10.2f} {times['slang_pt']:<10.2f} {times['dirichlet']:<10.2f}")


if __name__ == '__main__':
    # 2^2, 4^2, 8^2, 16^2, 32^2, 64^2, 128^2, 256^2, 512^2, 1024^2
    num_targets_list = [4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576]
    benchmark(num_targets_list)
