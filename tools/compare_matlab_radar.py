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
    fc, fs, samples, chirps = 77e9, 20e6, 512, 256
    period = samples / fs
    slope = 10e6 / period
    for name, lengths, rates, gains in (
        ("static", [60.0], [0.0], [1.0]),
        ("radial", [60.0], [2.0], [1.0]),
        ("multipath", [60.0, 180.0, 300.0], [2.0, 1.0, -1.0], [1.0, 0.6, 0.3]),
    ):
        delay = torch.tensor(np.array(lengths) / 299792458, dtype=torch.float32, device="cuda")
        rate = torch.tensor(np.array(rates) / 299792458, dtype=torch.float32, device="cuda")
        amplitude = torch.tensor(gains, dtype=torch.complex64, device="cuda")
        spec = FmcwSpec(samples, chirps, 1 / fs, period, slope, 0.0, fc, carrier_hz=fc, output_domain="beat")
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
                "period": period,
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


def analyze(directory):
    rows = {}
    for name in ("static", "radial", "multipath"):
        source = loadmat(directory / f"{name}-input.mat", squeeze_me=True)
        result = loadmat(directory / f"{name}-matlab.mat", squeeze_me=True)
        reference, actual = source["witwin_iq"], result["matlab_iq"]
        if reference.shape != actual.shape:
            raise ValueError(f"{name}: differing sample axes {reference.shape} vs {actual.shape}")
        # Exclude the declared propagation-filter startup region in BOTH cubes.
        guard = int(np.ceil(max(np.atleast_1d(source["lengths"])) / 299792458 * source["fs"])) + 16
        reference, actual = reference[guard:, 1:], actual[guard:, 1:]
        scale = np.vdot(actual, reference) / np.vdot(actual, actual)
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
        }
    (directory / "comparison.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("output/doppler-repair/matlab"))
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    (analyze if args.analyze else export)(args.output)
