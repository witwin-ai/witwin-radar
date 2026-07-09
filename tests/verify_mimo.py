"""Manual MIMO verification for native Dirichlet CUDA paths."""

from __future__ import annotations

import numpy as np
import torch

from witwin.radar import Radar, RadarConfig, TraceResult


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
    "chirp_per_frame": 2,
    "frame_per_second": 10,
    "num_doppler_bins": 2,
    "num_range_bins": 256,
    "num_angle_bins": 64,
    "power": 15,
    "tx_loc": [[0, 0, 0], [4, 0, 0], [2, 1, 0]],
    "rx_loc": [[-6, 0, 0], [-5, 0, 0], [-4, 0, 0], [-3, 0, 0]],
}


def main() -> None:
    radar = Radar(RadarConfig.from_dict(CONFIG), backend="dirichlet", device="cuda")

    rng = np.random.RandomState(42)
    num_targets = 50
    positions = rng.randn(num_targets, 3).astype(np.float32)
    positions[:, 2] -= 3
    intensities = rng.uniform(0.5, 1.5, num_targets).astype(np.float32)
    trace = TraceResult(
        torch.tensor(positions, dtype=torch.float32, device="cuda"),
        torch.tensor(intensities, dtype=torch.float32, device="cuda"),
    )

    def interp(_t):
        return trace

    print("Computing per-chirp Dirichlet MIMO...")
    legacy = radar.mimo(interp, t0=0, fast=False)
    print(f"  shape: {legacy.shape}")

    print("Computing cached-path Dirichlet MIMO...")
    fast = radar.mimo_from_trace(trace)
    print(f"  shape: {fast.shape}")

    legacy_flat = legacy.detach().cpu().numpy().ravel()
    fast_flat = fast.detach().cpu().numpy().ravel()
    mag_corr = np.corrcoef(np.abs(legacy_flat), np.abs(fast_flat))[0, 1]
    complex_corr = np.abs(np.vdot(legacy_flat, fast_flat)) / (
        np.linalg.norm(legacy_flat) * np.linalg.norm(fast_flat)
    )
    peak_ratio = np.abs(fast_flat).max() / np.abs(legacy_flat).max()

    print("\n=== Native MIMO Comparison ===")
    print(f"Magnitude correlation: {mag_corr:.10f}")
    print(f"Complex correlation:   {complex_corr:.10f}")
    print(f"Peak ratio:            {peak_ratio:.6f}")
    print(f"\n{'PASS' if mag_corr > 0.99 and complex_corr > 0.99 else 'FAIL'}")


if __name__ == "__main__":
    main()
