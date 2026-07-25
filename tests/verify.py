"""Manual chirp-spectrum verification for the native Dirichlet CUDA kernel."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from witwin.radar import Radar, RadarConfig
from reference.dsp_oracles import pytorch_chirp_reference


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


def compute_reference_fft(radar: Radar, distances: torch.Tensor, amplitudes: torch.Tensor) -> torch.Tensor:
    signal = pytorch_chirp_reference(radar, distances.to(torch.float64), amplitudes.to(torch.float64))
    return torch.fft.fft(signal, n=radar.solver.N_fft)[: radar.solver.N_fft // 2]


def main() -> None:
    num_targets = 1024
    rng = np.random.RandomState(42)

    radar = Radar(RadarConfig.from_dict(CONFIG), device="cuda")
    fs = radar.config.sample_rate * 1e3
    slope = radar.config.slope * 1e12
    freq_axis = np.fft.fftfreq(radar.solver.N_fft, 1 / fs)[: radar.solver.N_fft // 2]
    range_axis = freq_axis * radar.c0 / (2 * slope)

    distances = torch.tensor(rng.uniform(0.5, 5.0, num_targets), dtype=torch.float32, device="cuda")
    amplitudes = torch.tensor(rng.uniform(0.5, 1.0, num_targets), dtype=torch.float32, device="cuda")

    reference = compute_reference_fft(radar, distances, amplitudes).detach().cpu().numpy()
    native = radar.chirp(distances, amplitudes).detach().cpu().numpy()
    mag_corr = np.corrcoef(np.abs(reference), np.abs(native))[0, 1]
    complex_corr = np.abs(np.vdot(reference, native)) / (np.linalg.norm(reference) * np.linalg.norm(native))

    print("\n=== Native Dirichlet CUDA Verification ===")
    print(f"Targets: {num_targets}")
    print(f"Reference FFT peak: {np.abs(reference).max():.4f}")
    print(f"Native peak:        {np.abs(native).max():.4f}")
    print(f"Magnitude corr:     {mag_corr:.10f}")
    print(f"Complex corr:       {complex_corr:.10f}")

    reference_norm = reference / np.abs(reference).max()
    native_norm = native / np.abs(native).max()
    peak_idx = int(np.argmax(np.abs(reference)))
    zoom_width = 30
    zoom_start = max(0, peak_idx - zoom_width)
    zoom_end = min(len(reference), peak_idx + zoom_width)
    bins = np.arange(zoom_start, zoom_end)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(range_axis, np.abs(reference_norm), "b-", lw=1, label="PyTorch FFT reference")
    axes[0].plot(range_axis, np.abs(native_norm), "r-", lw=1, label="Native Dirichlet")
    axes[0].set_xlabel("Range (m)")
    axes[0].set_ylabel("Magnitude (normalized)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(bins, np.abs(reference_norm[zoom_start:zoom_end]), "b-", marker="o", markersize=3)
    axes[1].plot(bins, np.abs(native_norm[zoom_start:zoom_end]), "r-", marker="s", markersize=3)
    axes[1].set_xlabel("Bin index")
    axes[1].set_ylabel("Magnitude")
    axes[1].grid(True, alpha=0.3)

    error = np.abs(reference_norm[zoom_start:zoom_end]) - np.abs(native_norm[zoom_start:zoom_end])
    axes[2].plot(bins, error, "k-", marker="o", markersize=3)
    axes[2].axhline(y=0, color="gray", lw=0.5)
    axes[2].set_xlabel("Bin index")
    axes[2].set_ylabel("Magnitude error")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "verify.png"), dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
