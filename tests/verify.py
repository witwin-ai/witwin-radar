"""Manual chirp-spectrum verification across the three radar backends."""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from witwin.radar import Radar


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


def compute_fft_pytorch(radar, distances, amplitudes, n_fft):
    signal = radar.chirp(distances, amplitudes)
    return torch.fft.fft(signal, n_fft)[: n_fft // 2]


def compute_fft_slang(radar, distances, amplitudes, n_fft):
    signal = radar.chirp(distances.to(torch.float32), amplitudes.to(torch.float32))
    return torch.fft.fft(signal, n_fft)[: n_fft // 2]


def compute_dirichlet(radar, distances, amplitudes):
    return radar.chirp(distances.to(torch.float32), amplitudes.to(torch.float32))


def main():
    num_targets = 1024
    rng = np.random.RandomState(42)

    radar_pytorch = Radar(CONFIG, backend="pytorch")
    radar_slang = Radar(CONFIG, backend="slang")
    radar_dirichlet = Radar(CONFIG, backend="dirichlet")

    n_fft = radar_dirichlet.solver.N_fft
    fs = radar_dirichlet.sample_rate * 1e3
    slope = radar_dirichlet.slope * 1e12

    freq_axis = np.fft.fftfreq(n_fft, 1 / fs)[: n_fft // 2]
    range_axis = freq_axis * radar_dirichlet.c0 / (2 * slope)

    distances = torch.tensor(rng.uniform(0.5, 5.0, num_targets), dtype=torch.float64, device="cuda")
    amplitudes = torch.tensor(rng.uniform(0.5, 1.0, num_targets), dtype=torch.float64, device="cuda")

    print(f"Number of targets: {num_targets}")
    print("Computing...")

    fft_pytorch = compute_fft_pytorch(radar_pytorch, distances, amplitudes, n_fft).cpu().numpy()
    fft_slang = compute_fft_slang(radar_slang, distances, amplitudes, n_fft).cpu().numpy()
    dirichlet = compute_dirichlet(radar_dirichlet, distances, amplitudes).cpu().numpy()

    corr_pt_sl = np.corrcoef(np.abs(fft_pytorch), np.abs(fft_slang))[0, 1]
    corr_pt_di = np.corrcoef(np.abs(fft_pytorch), np.abs(dirichlet))[0, 1]
    corr_sl_di = np.corrcoef(np.abs(fft_slang), np.abs(dirichlet))[0, 1]

    print("\n=== Comparison ===")
    print(f"PyTorch FFT peak: {np.abs(fft_pytorch).max():.4f}")
    print(f"Slang FFT peak:   {np.abs(fft_slang).max():.4f}")
    print(f"Dirichlet peak:   {np.abs(dirichlet).max():.4f}")
    print("\nMagnitude correlations:")
    print(f"  PyTorch FFT vs Slang FFT: {corr_pt_sl:.10f}")
    print(f"  PyTorch FFT vs Dirichlet: {corr_pt_di:.10f}")
    print(f"  Slang FFT   vs Dirichlet: {corr_sl_di:.10f}")

    fft_pytorch_norm = fft_pytorch / np.abs(fft_pytorch).max()
    fft_slang_norm = fft_slang / np.abs(fft_slang).max()
    dirichlet_norm = dirichlet / np.abs(dirichlet).max()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    peak_idx = np.argmax(np.abs(fft_pytorch))
    zoom_width = 30
    zoom_start = max(0, peak_idx - zoom_width)
    zoom_end = min(len(fft_pytorch), peak_idx + zoom_width)
    bins = np.arange(zoom_start, zoom_end)

    ax = axes[0, 0]
    ax.plot(range_axis, np.abs(fft_pytorch_norm), "b-", lw=1, label="PyTorch FFT", alpha=0.8)
    ax.plot(range_axis, np.abs(dirichlet_norm), "r-", lw=1, label="Dirichlet", alpha=0.8)
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Magnitude (normalized)")
    ax.set_title(f"Magnitude ({num_targets} targets)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(bins, np.abs(fft_pytorch_norm[zoom_start:zoom_end]), "b-", lw=1.5, label="PyTorch FFT", marker="o", markersize=3)
    ax.plot(bins, np.abs(dirichlet_norm[zoom_start:zoom_end]), "r-", lw=1.5, label="Dirichlet", marker="s", markersize=3)
    for bin_id in bins:
        ax.axvline(x=bin_id, color="gray", lw=0.3, alpha=0.5)
    ax.set_xlabel("Bin index")
    ax.set_ylabel("Magnitude")
    ax.set_title(f"Zoomed Magnitude (bins {zoom_start}-{zoom_end})")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[0, 2]
    error_zoom = np.abs(fft_pytorch_norm[zoom_start:zoom_end]) - np.abs(dirichlet_norm[zoom_start:zoom_end])
    ax.plot(bins, error_zoom, "r-", lw=1.5, marker="o", markersize=3)
    ax.axhline(y=0, color="gray", lw=0.5)
    for bin_id in bins:
        ax.axvline(x=bin_id, color="gray", lw=0.3, alpha=0.5)
    ax.set_xlabel("Bin index")
    ax.set_ylabel("Error")
    ax.set_title(f"Magnitude Error (max: {np.abs(error_zoom).max():.4f})")
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1, 0]
    ax.plot(bins, fft_pytorch_norm[zoom_start:zoom_end].real, "b-", lw=1.5, label="PyTorch FFT", marker="o", markersize=3)
    ax.plot(bins, dirichlet_norm[zoom_start:zoom_end].real, "r-", lw=1.5, label="Dirichlet", marker="s", markersize=3)
    ax.axhline(y=0, color="gray", lw=0.5)
    for bin_id in bins:
        ax.axvline(x=bin_id, color="gray", lw=0.3, alpha=0.5)
    ax.set_xlabel("Bin index")
    ax.set_ylabel("Real")
    ax.set_title("Zoomed Real Part")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1, 1]
    ax.plot(bins, fft_pytorch_norm[zoom_start:zoom_end].imag, "b-", lw=1.5, label="PyTorch FFT", marker="o", markersize=3)
    ax.plot(bins, dirichlet_norm[zoom_start:zoom_end].imag, "r-", lw=1.5, label="Dirichlet", marker="s", markersize=3)
    ax.axhline(y=0, color="gray", lw=0.5)
    for bin_id in bins:
        ax.axvline(x=bin_id, color="gray", lw=0.3, alpha=0.5)
    ax.set_xlabel("Bin index")
    ax.set_ylabel("Imag")
    ax.set_title("Zoomed Imag Part")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1, 2]
    real_error = fft_pytorch_norm[zoom_start:zoom_end].real - dirichlet_norm[zoom_start:zoom_end].real
    imag_error = fft_pytorch_norm[zoom_start:zoom_end].imag - dirichlet_norm[zoom_start:zoom_end].imag
    ax.plot(bins, real_error, "b-", lw=1.5, label="Real Error", marker="o", markersize=3)
    ax.plot(bins, imag_error, color="orange", lw=1.5, label="Imag Error", marker="s", markersize=3)
    ax.axhline(y=0, color="gray", lw=0.5)
    for bin_id in bins:
        ax.axvline(x=bin_id, color="gray", lw=0.3, alpha=0.5)
    ax.set_xlabel("Bin index")
    ax.set_ylabel("Error")
    ax.set_title("Real & Imag Error")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "verify.png"), dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
