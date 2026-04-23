"""Manual MIMO verification across radar solver backends."""

import numpy as np
import torch

from witwin.radar import Radar, RadarConfig


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


def main():
    cfg = RadarConfig.from_dict(CONFIG)
    print("Creating Slang radar...")
    radar_slang = Radar(cfg, backend="slang")
    print("Creating Dirichlet radar...")
    radar_dirichlet = Radar(cfg, backend="dirichlet")

    rng = np.random.RandomState(42)
    num_targets = 50
    positions = rng.randn(num_targets, 3).astype(np.float32)
    positions[:, 2] -= 3
    intensities = rng.uniform(0.5, 1.5, num_targets).astype(np.float32)

    pos_t = torch.tensor(positions, dtype=torch.float32, device="cuda")
    int_t = torch.tensor(intensities, dtype=torch.float32, device="cuda")

    def interp(t):
        return int_t, pos_t

    print("Computing Slang MIMO...")
    frame_slang = radar_slang.mimo(interp, t0=0)
    print(f"  shape: {frame_slang.shape}")

    print("Computing Dirichlet MIMO...")
    frame_dirichlet = radar_dirichlet.mimo(interp, t0=0)
    print(f"  shape: {frame_dirichlet.shape}")

    slang_flat = frame_slang.cpu().numpy().ravel()
    dirichlet_flat = frame_dirichlet.cpu().numpy().ravel()

    mag_corr = np.corrcoef(np.abs(slang_flat), np.abs(dirichlet_flat))[0, 1]
    complex_corr = np.abs(np.vdot(slang_flat, dirichlet_flat)) / (
        np.linalg.norm(slang_flat) * np.linalg.norm(dirichlet_flat)
    )
    peak_ratio = np.abs(dirichlet_flat).max() / np.abs(slang_flat).max()

    print("\n=== Overall MIMO Comparison ===")
    print(f"Magnitude correlation: {mag_corr:.10f}")
    print(f"Complex correlation:   {complex_corr:.10f}")
    print(f"Peak ratio (dir/slang): {peak_ratio:.6f}")

    print("\n=== Per-chirp (TX0, RX0) ===")
    for chirp_id in range(CONFIG["chirp_per_frame"]):
        slang_chirp = frame_slang[0, 0, chirp_id].cpu().numpy()
        dirichlet_chirp = frame_dirichlet[0, 0, chirp_id].cpu().numpy()
        chirp_mag_corr = np.corrcoef(np.abs(slang_chirp), np.abs(dirichlet_chirp))[0, 1]
        chirp_complex_corr = np.abs(np.vdot(slang_chirp, dirichlet_chirp)) / (
            np.linalg.norm(slang_chirp) * np.linalg.norm(dirichlet_chirp)
        )
        print(
            f"  Chirp {chirp_id}: mag_corr={chirp_mag_corr:.10f}  "
            f"complex_corr={chirp_complex_corr:.10f}"
        )

    passed = mag_corr > 0.99 and complex_corr > 0.99
    print(f"\n{'PASS' if passed else 'FAIL'}: mag_corr={mag_corr:.6f}, complex_corr={complex_corr:.6f}")


if __name__ == "__main__":
    main()
