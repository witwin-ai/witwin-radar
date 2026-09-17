"""Export matched path-level FMCW inputs; analyze actual Radar Toolbox output.

This compares waveform synthesis on declared paths, not MATLAB's scene solver
against Channel. The MATLAB half requires radarTransceiver and must run before
--analyze can succeed. No analytic Python result substitutes for MATLAB.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat, savemat

from witwin.radar.synthesis.assembly import FmcwSpec
from witwin.radar.synthesis.fmcw import synthesize_fmcw_rows


def export(directory):
    directory.mkdir(parents=True, exist_ok=True)
    fc, chirps = 77e9, 256
    period = 512 / 20e6
    slope = 10e6 / period
    scenarios = (
        ("static", [60.0], [0.0], [1.0]),
        ("radial", [60.0], [2.0], [1.0]),
        ("multipath", [60.0, 180.0, 300.0], [2.0, 1.0, -1.0], [1.0, 0.6, 0.3]),
    )
    cases = [(name, lengths, rates, gains, 20e6, 512) for name, lengths, rates, gains in scenarios]
    cases += [(name + "_os4", lengths, rates, gains, 80e6, 2048) for name, lengths, rates, gains in scenarios]
    # A dyadic sample interval gives an exactly representable float32 delay.
    # This isolates waveform/phasor agreement from fractional-delay filtering.
    cases += [("static_integer", [4 / 2**24 * 299792458], [0.0], [1.0], float(2**24), 512)]
    for name, lengths, rates, gains, fs, samples in cases:
        case_period = samples / fs
        delay = torch.tensor(np.array(lengths) / 299792458, dtype=torch.float32, device="cuda")
        rate = torch.tensor(np.array(rates) / 299792458, dtype=torch.float32, device="cuda")
        amplitude = torch.tensor(gains, dtype=torch.complex64, device="cuda")
        spec = FmcwSpec(samples, chirps, 1 / fs, case_period, slope, 0.0, fc, carrier_hz=fc, output_domain="beat")
        offsets = torch.tensor([0, len(lengths)], device="cuda", dtype=torch.int64)

        synthesize_fmcw_rows(delay, rate, amplitude, offsets, spec)
        torch.cuda.synchronize()
        start = time.perf_counter()
        cube = synthesize_fmcw_rows(delay, rate, amplitude, offsets, spec)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        savemat(
            directory / f"{name}-input.mat",
            {
                "fc": fc,
                "fs": fs,
                "samples": samples,
                "chirps": chirps,
                "period": case_period,
                "slope": slope,
                "lengths": delay.double().cpu().numpy() * 299792458,
                "rates": rate.double().cpu().numpy() * 299792458,
                "gains": np.array(gains),
                "witwin_iq": cube[:, 0].T.cpu().numpy(),
                "witwin_seconds": seconds,
                "scope": "declared round-trip paths; noiseless unity hardware; positive tx*conj(rx) beat",
            },
        )
    (directory / "status.json").write_text(
        json.dumps(
            {
                "status": "awaiting_actual_matlab_execution",
                "required": "Radar Toolbox radarTransceiver; Phased Array System Toolbox",
                "script": "tools/compare_matlab_radar.m",
            },
            indent=2,
        )
        + "\n"
    )


def independent_linear_delay_oracle(source, guard):
    """Diagnostic oracle for sampled chirps with linear fractional delay.

    SI units, positive tx*conj(rx) beat. Hold envelope delay at chirp start,
    apply carrier Doppler to the received waveform, then linearly interpolate
    adjacent transmitted samples. This diagnostic never substitutes for an
    actual MATLAB result and is not used in production synthesis.
    """
    fs, fc, slope = source["fs"], source["fc"], source["slope"]
    u = np.arange(guard, int(source["samples"]))[:, None] / fs
    slow = np.arange(1, int(source["chirps"]))[None, :] * source["period"]
    received = np.zeros((len(u), slow.shape[1]), dtype=np.complex128)
    for length, rate, gain in zip(
        np.atleast_1d(source["lengths"]), np.atleast_1d(source["rates"]), np.atleast_1d(source["gains"]), strict=True
    ):
        tau = (length + rate * slow) / 299792458
        delay_samples = tau * fs
        fraction = delay_samples - np.floor(delay_samples)
        t0 = u - np.floor(delay_samples) / fs
        t1 = t0 - 1 / fs
        fd = -fc * rate / 299792458
        a = np.exp(2j * np.pi * (0.5 * slope * t0**2 + fd * (tau + t0)))
        b = np.exp(2j * np.pi * (0.5 * slope * t1**2 + fd * (tau + t1)))
        received += gain * np.exp(-2j * np.pi * fc * tau) * ((1 - fraction) * a + fraction * b)
    return np.exp(1j * np.pi * slope * u**2) * received.conj()


def range_doppler_metrics(source, reference, actual):
    """Compare identically windowed power and path-local FFT peaks, no fit."""
    shape = (int(source["samples"]), int(source["chirps"]))
    window = np.hanning(reference.shape[0])[:, None] * np.hanning(reference.shape[1])[None, :]
    powers = [np.abs(np.fft.fftshift(np.fft.fft2(iq * window, s=shape), axes=1)) ** 2 for iq in (reference, actual)]
    ranges = np.arange(shape[0]) * source["fs"] / shape[0] * 299792458 / (2 * source["slope"])
    velocities = np.fft.fftshift(np.fft.fftfreq(shape[1], source["period"])) * 299792458 / (2 * source["fc"])
    peaks = []
    for length, rate in zip(np.atleast_1d(source["lengths"]), np.atleast_1d(source["rates"]), strict=True):
        ri = int(np.argmin(abs(ranges - length / 2)))
        vi = int(np.argmin(abs(velocities - rate / 2)))
        section = (slice(max(0, ri - 1), ri + 2), slice(max(0, vi - 1), vi + 2))
        indices = []
        for power in powers:
            peak = np.unravel_index(np.argmax(power[section]), power[section].shape)
            indices.append([int(peak[0] + section[0].start), int(peak[1] + section[1].start)])
        peaks.append(
            {
                "expected_equivalent_range_m": float(length / 2),
                "expected_receding_velocity_m_s": float(rate / 2),
                "witwin_peak_bins": indices[0],
                "matlab_peak_bins": indices[1],
                "same_peak_bins": indices[0] == indices[1],
            }
        )
    return {
        "power_relative_l2": float(np.linalg.norm(powers[0] - powers[1]) / np.linalg.norm(powers[0])),
        "paths": peaks,
    }


def startup_guard(source):
    """Exclude the largest delay plus a fixed 0.8 us filter guard in both cubes."""
    return int(np.ceil(max(np.atleast_1d(source["lengths"])) / 299792458 * source["fs"])) + int(
        np.ceil(0.8e-6 * source["fs"])
    )


def analyze(directory):
    rows = {}
    input_paths = sorted(directory.glob("*-input.mat"))
    if not input_paths:
        raise ValueError("No exported WiTwin inputs found")
    for input_path in input_paths:
        name = input_path.name.removesuffix("-input.mat")
        source = loadmat(directory / f"{name}-input.mat", squeeze_me=True)
        result = loadmat(directory / f"{name}-matlab.mat", squeeze_me=True)
        reference, actual = source["witwin_iq"], result["matlab_iq"]
        if reference.shape != actual.shape:
            raise ValueError(f"{name}: differing sample axes {reference.shape} vs {actual.shape}")
        # Exclude the declared propagation-filter startup region in BOTH cubes.
        guard = startup_guard(source)
        reference, actual = reference[guard:, 1:], actual[guard:, 1:]
        scale = np.vdot(actual, reference) / np.vdot(actual, actual)
        oracle = independent_linear_delay_oracle(source, guard)
        rows[name] = {
            "raw_iq_relative_l2": float(np.linalg.norm(reference - actual) / np.linalg.norm(reference)),
            "aligned_iq_relative_l2": float(np.linalg.norm(reference - scale * actual) / np.linalg.norm(reference)),
            "single_global_alignment_real": float(scale.real),
            "single_global_alignment_imag": float(scale.imag),
            "excluded_startup_samples": guard,
            "excluded_chirps": 1,
            "witwin_cuda_synthesis_seconds": float(source["witwin_seconds"]),
            "matlab_transceiver_seconds": float(result["matlab_seconds"]),
            "timing_scope_comparable": False,
            "matlab_vs_linear_delay_oracle_relative_l2": float(
                np.linalg.norm(actual - oracle) / np.linalg.norm(oracle)
            ),
            "range_doppler": range_doppler_metrics(source, reference, actual),
        }
    (directory / "comparison.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))
    # Declared engineering acceptance bounds for this noiseless path-level
    # suite, not a universal equivalence or commercial certification claim.
    failures = []
    for name, row in rows.items():
        limit = 1e-6 if name == "static_integer" else 0.02
        if not np.isfinite(row["raw_iq_relative_l2"]) or row["raw_iq_relative_l2"] > limit:
            failures.append(f"{name}: raw IQ exceeds {limit}")
        if not row["matlab_vs_linear_delay_oracle_relative_l2"] < 1e-8:
            failures.append(f"{name}: MATLAB differs from the declared sampled-delay diagnostic model")
        if not all(path["same_peak_bins"] for path in row["range_doppler"]["paths"]):
            failures.append(f"{name}: range-Doppler peaks differ")
    for base in ("static", "radial", "multipath"):
        if base in rows and base + "_os4" in rows:
            if rows[base + "_os4"]["raw_iq_relative_l2"] >= rows[base]["raw_iq_relative_l2"]:
                failures.append(f"{base}: oversampling did not reduce raw IQ error")
    (directory / "status.json").write_text(
        json.dumps(
            {
                "status": "failed" if failures else "passed_declared_path_level_bounds",
                "actual_matlab_cases": len(rows),
                "failures": failures,
                "limits": {"raw_iq": 0.02, "integer_delay_iq": 1e-6, "matlab_linear_delay_model": 1e-8},
                "scope": "ideal hardware and declared paths; excludes scene discovery and matched-scope performance",
            },
            indent=2,
        )
        + "\n"
    )
    if failures:
        raise AssertionError("; ".join(failures))


def plot_comparison(directory):
    """Save actual three-path RD power with common axes and normalization."""
    import matplotlib.pyplot as plt

    source = loadmat(directory / "multipath-input.mat", squeeze_me=True)
    result = loadmat(directory / "multipath-matlab.mat", squeeze_me=True)
    guard = startup_guard(source)
    shape = source["witwin_iq"].shape
    powers = []
    for iq in (source["witwin_iq"], result["matlab_iq"]):
        trimmed = iq[guard:, 1:]
        window = np.hanning(trimmed.shape[0])[:, None] * np.hanning(trimmed.shape[1])[None, :]
        powers.append(np.abs(np.fft.fftshift(np.fft.fft2(trimmed * window, s=shape), axes=1)) ** 2)
    scale = powers[0].max()
    ranges = np.arange(shape[0]) * source["fs"] / shape[0] * 299792458 / (2 * source["slope"])
    velocities = np.fft.fftshift(np.fft.fftfreq(shape[1], source["period"])) * 299792458 / (2 * source["fc"])
    select_r, select_v = ranges < 195, abs(velocities) < 2.5
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), constrained_layout=True, sharex=True, sharey=True)
    arrays = [powers[0], powers[1], abs(powers[0] - powers[1])]
    titles = ["WiTwin CUDA", "MATLAB Radar Toolbox R2025b U4", "Absolute power difference"]
    for axis, power, title in zip(axes, arrays, titles, strict=True):
        db = 10 * np.log10(np.maximum(power / scale, 1e-8))
        mesh = axis.pcolormesh(
            velocities[select_v], ranges[select_r], db[np.ix_(select_r, select_v)], vmin=-50, vmax=0, shading="auto"
        )
        axis.set(title=title, xlabel="Equivalent receding velocity (m/s)")
        axis.scatter(source["rates"] / 2, source["lengths"] / 2, marker="x", color="red", s=45)
    axes[0].set_ylabel("Equivalent range (m)")
    fig.colorbar(mesh, ax=axes, label="dB relative to WiTwin peak power")
    fig.suptitle("Three declared paths, ideal hardware; red crosses: expected path coordinates")
    fig.savefig(directory / "range-doppler-comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("output/doppler-repair/matlab"))
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--plot", action="store_true", help="plot actual three-path outputs after analysis")
    args = parser.parse_args()
    (analyze if args.analyze else export)(args.output)
    if args.plot:
        plot_comparison(args.output)
