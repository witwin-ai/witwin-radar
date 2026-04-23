"""
Debug the bin offset and value offset between FFT and Dirichlet.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "witwin-radar"))

import numpy as np
import torch
import slangtorch

from core import Radar, RadarConfig

# Load Slang module for Dirichlet
_slang_module = None


def get_slang_module():
    global _slang_module
    if _slang_module is None:
        slang_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'core', 'solvers', 'dirichlet.slang')
        _slang_module = slangtorch.loadModule(slang_path)
    return _slang_module


def create_radar():
    config = {
        "num_tx": 1,
        "num_rx": 1,
        "fc": 77,
        "slope": 100,
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
    return Radar(RadarConfig.from_dict(config), backend="pytorch")


def compute_fft(radar, distance, pad_factor=16):
    """Compute FFT for a single target."""
    cfg = radar.config
    N_samples = cfg.adc_samples
    N_fft = N_samples * pad_factor

    dist_tensor = torch.tensor([distance], dtype=torch.float64, device='cuda')
    amp_tensor = torch.tensor([1.0], dtype=torch.float64, device='cuda')
    sig = radar.chirp(dist_tensor, amp_tensor)

    fft_result = torch.fft.fft(sig, N_fft)
    return fft_result[:N_fft // 2]


def compute_dirichlet(distance, amplitude, N_samples, N_fft, k0_per_meter, fc, slope):
    """Compute Dirichlet for a single target."""
    n = (N_samples - 1) / 2
    num_bins = N_fft // 2

    d = torch.tensor([distance], dtype=torch.float32, device='cuda').contiguous()
    a = torch.tensor([amplitude], dtype=torch.float32, device='cuda').contiguous()

    output_re = torch.zeros(num_bins, dtype=torch.float32, device='cuda')
    output_im = torch.zeros(num_bins, dtype=torch.float32, device='cuda')

    module = get_slang_module()
    module.forward(
        d=d, a=a, output_re=output_re, output_im=output_im,
        n=n, k0_per_meter=k0_per_meter,
        num_bins=num_bins, N_fft=N_fft, num_targets=1,
        fc=fc, slope=slope
    ).launchRaw(
        blockSize=(256, 1, 1),
        gridSize=((num_bins + 255) // 256, 1, 1)
    )

    return torch.complex(output_re, output_im)


def main():
    radar = create_radar()
    cfg = radar.config
    N_samples = cfg.adc_samples
    pad_factor = 16
    N_fft = N_samples * pad_factor
    fs = cfg.sample_rate * 1e3
    slope = cfg.slope * 1e12
    k0_per_meter = (slope * 2 / radar.c0) * N_fft / fs
    fc = cfg.fc

    print(f"N_samples: {N_samples}")
    print(f"N_fft: {N_fft}")
    print(f"pad_factor: {pad_factor}")
    print(f"k0_per_meter: {k0_per_meter:.10f}")
    print(f"n (Dirichlet order): {(N_samples - 1) / 2}")
    print(f"fc: {fc}")
    print(f"slope: {slope:.2e} Hz/s")
    print()

    # Test with a few single targets at different distances
    test_distances = [1.0, 2.0, 3.0, 4.0]

    print("=" * 80)
    print(f"{'Distance':<10} {'k0 (expected)':<15} {'FFT peak':<12} {'Dir peak':<12} {'Offset':<8} {'FFT mag':<12} {'Dir mag':<12} {'Mag ratio':<10}")
    print("=" * 80)

    for dist in test_distances:
        # Expected k0
        k0_expected = dist * k0_per_meter

        # Compute FFT
        fft_result = compute_fft(radar, dist, pad_factor).cpu().numpy()
        fft_peak_bin = np.argmax(np.abs(fft_result))
        fft_peak_mag = np.abs(fft_result[fft_peak_bin])

        # Compute Dirichlet
        dirichlet_result = compute_dirichlet(dist, 1.0, N_samples, N_fft, k0_per_meter, fc, slope).cpu().numpy()
        dir_peak_bin = np.argmax(np.abs(dirichlet_result))
        dir_peak_mag = np.abs(dirichlet_result[dir_peak_bin])

        offset = dir_peak_bin - fft_peak_bin
        mag_ratio = dir_peak_mag / fft_peak_mag if fft_peak_mag > 0 else 0

        print(f"{dist:<10.1f} {k0_expected:<15.4f} {fft_peak_bin:<12} {dir_peak_bin:<12} {offset:<8} {fft_peak_mag:<12.4f} {dir_peak_mag:<12.4f} {mag_ratio:<10.6f}")

    print()
    print("=" * 80)
    print("Detailed analysis for distance = 2.0m")
    print("=" * 80)

    dist = 2.0
    k0 = dist * k0_per_meter
    print(f"k0 = {k0:.10f}")
    print(f"k0 floor = {int(k0)}")
    print(f"k0 round = {round(k0)}")
    print(f"k0 fractional part = {k0 - int(k0):.10f}")
    print()

    fft_result = compute_fft(radar, dist, pad_factor).cpu().numpy()
    dirichlet_result = compute_dirichlet(dist, 1.0, N_samples, N_fft, k0_per_meter, fc, slope).cpu().numpy()

    fft_peak_bin = np.argmax(np.abs(fft_result))
    dir_peak_bin = np.argmax(np.abs(dirichlet_result))

    # Look at values around the peak
    window = 5
    start = max(0, fft_peak_bin - window)
    end = min(len(fft_result), fft_peak_bin + window + 1)

    print(f"{'Bin':<8} {'FFT mag':<12} {'Dir mag':<12} {'FFT real':<12} {'Dir real':<12} {'FFT imag':<12} {'Dir imag':<12}")
    print("-" * 80)
    for b in range(start, end):
        fft_mag = np.abs(fft_result[b])
        dir_mag = np.abs(dirichlet_result[b])
        fft_re = fft_result[b].real
        dir_re = dirichlet_result[b].real
        fft_im = fft_result[b].imag
        dir_im = dirichlet_result[b].imag

        marker = ""
        if b == fft_peak_bin:
            marker += " <-- FFT peak"
        if b == dir_peak_bin:
            marker += " <-- Dir peak"

        print(f"{b:<8} {fft_mag:<12.4f} {dir_mag:<12.4f} {fft_re:<12.4f} {dir_re:<12.4f} {fft_im:<12.4f} {dir_im:<12.4f}{marker}")

    print()
    print("=" * 80)
    print("Check if offset is exactly 0.5 bin")
    print("=" * 80)

    # The Dirichlet formula uses x = 2*pi*(bin - k0) / N_fft
    # Maybe there's a half-bin offset issue?

    # Try computing with k0 shifted by 0.5
    print("\nTrying k0 + 0.5 offset...")
    k0_shifted = k0 + 0.5

    # Manual Dirichlet at the FFT peak bin
    n = (N_samples - 1) / 2
    bin_fft = fft_peak_bin

    x_original = 2 * np.pi * (bin_fft - k0) / N_fft
    x_shifted = 2 * np.pi * (bin_fft - k0_shifted) / N_fft

    print(f"At FFT peak bin {bin_fft}:")
    print(f"  x with k0         = {x_original:.10f}")
    print(f"  x with k0 + 0.5   = {x_shifted:.10f}")

    # Check the time sample indexing
    print()
    print("=" * 80)
    print("Check time sample indexing")
    print("=" * 80)
    print(f"t_sample[0] = {radar.t_sample[0].item():.10e}")
    print(f"t_sample[1] = {radar.t_sample[1].item():.10e}")
    print(f"t_sample[-1] = {radar.t_sample[-1].item():.10e}")
    print(f"adc_start_time = {cfg.adc_start_time}")


def analyze_phase():
    """Analyze phase difference between FFT and Dirichlet."""
    radar = create_radar()
    cfg = radar.config
    N_samples = cfg.adc_samples
    pad_factor = 16
    N_fft = N_samples * pad_factor
    fs = cfg.sample_rate * 1e3
    slope = cfg.slope * 1e12
    k0_per_meter = (slope * 2 / radar.c0) * N_fft / fs
    fc = cfg.fc

    print("=" * 80)
    print("Phase analysis")
    print("=" * 80)

    dist = 2.0
    k0 = dist * k0_per_meter

    fft_result = compute_fft(radar, dist, pad_factor).cpu().numpy()
    dirichlet_result = compute_dirichlet(dist, 1.0, N_samples, N_fft, k0_per_meter, fc, slope).cpu().numpy()

    peak_bin = np.argmax(np.abs(fft_result))

    fft_val = fft_result[peak_bin]
    dir_val = dirichlet_result[peak_bin]

    fft_phase = np.angle(fft_val)
    dir_phase = np.angle(dir_val)

    print(f"At peak bin {peak_bin}:")
    print(f"  FFT: {fft_val.real:.6f} + {fft_val.imag:.6f}j  |  mag={np.abs(fft_val):.6f}  phase={np.degrees(fft_phase):.4f}°")
    print(f"  Dir: {dir_val.real:.6f} + {dir_val.imag:.6f}j  |  mag={np.abs(dir_val):.6f}  phase={np.degrees(dir_phase):.4f}°")
    print(f"  Phase difference: {np.degrees(fft_phase - dir_phase):.4f}°")

    # Theoretical phase calculation
    n = (N_samples - 1) / 2
    x = 2 * np.pi * (peak_bin - k0) / N_fft
    theoretical_phase = -n * x
    print(f"\n  x = 2π*(bin-k0)/N_fft = {x:.10f}")
    print(f"  Theoretical Dirichlet phase (-n*x) = {np.degrees(theoretical_phase):.4f}°")

    # Check FFT phase formula: exp(j*u*(N-1)/2) where u = 2π*(k0-k)/N_fft = -x
    u = -x
    theoretical_fft_phase = u * (N_samples - 1) / 2
    print(f"  Theoretical FFT phase (u*(N-1)/2) = {np.degrees(theoretical_fft_phase):.4f}°")


def analyze_multi_target():
    """Analyze with multiple targets (like verify.py)."""
    radar = create_radar()
    cfg = radar.config
    N_samples = cfg.adc_samples
    pad_factor = 16
    N_fft = N_samples * pad_factor
    fs = cfg.sample_rate * 1e3
    slope = cfg.slope * 1e12
    k0_per_meter = (slope * 2 / radar.c0) * N_fft / fs
    fc = cfg.fc

    print("\n" + "=" * 80)
    print("Multi-target analysis (1024 targets)")
    print("=" * 80)

    np.random.seed(42)
    num_targets = 1024
    distances = np.random.uniform(0.5, 5.0, num_targets).tolist()
    amplitudes = np.random.uniform(0.5, 1.0, num_targets).tolist()

    # Compute FFT using unified chirp API
    dist_tensor = torch.tensor(distances, dtype=torch.float64, device='cuda')
    amp_tensor = torch.tensor(amplitudes, dtype=torch.float64, device='cuda')
    sig_total = radar.chirp(dist_tensor, amp_tensor)

    fft_result = torch.fft.fft(sig_total, N_fft)[:N_fft // 2].cpu().numpy()

    # Compute Dirichlet
    n = (N_samples - 1) / 2
    num_bins = N_fft // 2
    d = torch.tensor(distances, dtype=torch.float32, device='cuda').contiguous()
    a = torch.tensor(amplitudes, dtype=torch.float32, device='cuda').contiguous()
    output_re = torch.zeros(num_bins, dtype=torch.float32, device='cuda')
    output_im = torch.zeros(num_bins, dtype=torch.float32, device='cuda')

    module = get_slang_module()
    module.forward(
        d=d, a=a, output_re=output_re, output_im=output_im,
        n=n, k0_per_meter=k0_per_meter,
        num_bins=num_bins, N_fft=N_fft, num_targets=num_targets,
        fc=fc, slope=slope
    ).launchRaw(
        blockSize=(256, 1, 1),
        gridSize=((num_bins + 255) // 256, 1, 1)
    )
    dirichlet_result = torch.complex(output_re, output_im).cpu().numpy()

    # Find peaks
    fft_peak_bin = np.argmax(np.abs(fft_result))
    dir_peak_bin = np.argmax(np.abs(dirichlet_result))

    print(f"FFT peak bin: {fft_peak_bin}")
    print(f"Dir peak bin: {dir_peak_bin}")
    print(f"Offset: {dir_peak_bin - fft_peak_bin}")

    # Normalize
    fft_norm = fft_result / np.abs(fft_result).max()
    dir_norm = dirichlet_result / np.abs(dirichlet_result).max()

    # Look at values around FFT peak
    window = 5
    start = max(0, fft_peak_bin - window)
    end = min(len(fft_result), fft_peak_bin + window + 1)

    print(f"\nNormalized values around peak:")
    print(f"{'Bin':<8} {'FFT mag':<12} {'Dir mag':<12} {'Mag diff':<12} {'FFT real':<12} {'Dir real':<12} {'Real diff':<12}")
    print("-" * 90)
    for b in range(start, end):
        fft_mag = np.abs(fft_norm[b])
        dir_mag = np.abs(dir_norm[b])
        fft_re = fft_norm[b].real
        dir_re = dir_norm[b].real

        marker = ""
        if b == fft_peak_bin:
            marker += " <-- FFT"
        if b == dir_peak_bin:
            marker += " <-- Dir"

        print(f"{b:<8} {fft_mag:<12.6f} {dir_mag:<12.6f} {dir_mag-fft_mag:<12.6f} {fft_re:<12.6f} {dir_re:<12.6f} {dir_re-fft_re:<12.6f}{marker}")


if __name__ == '__main__':
    main()
    analyze_phase()
    analyze_multi_target()
