"""Retain measured exact/adaptive IQ errors and end-to-end timings in witwin2."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import torch
from witwin.core import Scene

from tools.validate_doppler_motion import Trajectory, make_radar
from tools.validate_heavy_multipath import LinearPoint, room
from witwin.radar import Motion, PointTargets
from witwin.radar.processing.range_doppler import fmcw_range_fft


def run(output, cases):
    output.mkdir(parents=True, exist_ok=True)
    rows = json.loads((output / "results.json").read_text()) if (output / "results.json").exists() else {}
    for kind in cases:
        radar = make_radar()
        # ``Radar`` is immutable, so a reshaped waveform makes a new radar. The
        # timings are SI: the fixture's 442 and 1942 us idles are seconds here.
        radar = radar.replace(
            waveform=replace(
                radar.waveform,
                samples_per_chirp=64 if kind == "heavy" else 16,
                chirps_per_frame=32,
                idle=442e-6 if kind == "heavy" else 1942e-6,
                output="beat",
            )
        )
        trajectory = LinearPoint(radar.device) if kind == "heavy" else Trajectory(kind, radar.device)
        scene = room() if kind == "heavy" else Scene(structures=(), endpoints=[])
        targets = PointTargets(positions=trajectory.positions(0), amplitude=1.0, trajectory=trajectory.positions)
        kwargs = {"times": (0.0,), "los": True, "reflections": 2 if kind == "heavy" else 0}
        # Warm native loading and scene compilation outside both measurements.
        radar.simulate(scene, targets, **kwargs, motion=Motion.adaptive())
        measured = {}
        for mode, motion in (("adc", Motion.adc()), ("adaptive", Motion.adaptive())):
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = radar.simulate(scene, targets, **kwargs, motion=motion)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            measured[mode] = (result, seconds)
            torch.save(result.cube.detach().cpu(), output / f"{kind}-{mode}.pt")
        exact, seconds_exact = measured["adc"]
        adaptive, seconds_adaptive = measured["adaptive"]
        iq_error = float((exact.cube - adaptive.cube).norm() / exact.cube.norm())

        def rd(value):
            return torch.fft.fftshift(torch.fft.fft(fmcw_range_fft(value), dim=-2), dim=-2)

        reference, predicted = rd(exact.cube), rd(adaptive.cube)
        power_error = float(
            (reference.abs().square() - predicted.abs().square()).norm() / reference.abs().square().norm()
        )
        rows[kind] = {
            "iq_relative_l2": iq_error,
            "rd_power_relative_l2": power_error,
            "exact_seconds": seconds_exact,
            "adaptive_seconds": seconds_adaptive,
            "speedup": seconds_exact / seconds_adaptive,
            "exact_discoveries": exact.discovery_count,
            "adaptive_discoveries": adaptive.discovery_count,
            "paths": exact.last_radar_paths.path_count,
            "diagnostics": adaptive.adaptive_diagnostics,
        }
        print(kind, json.dumps(rows[kind]), flush=True)
        (output / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
        assert iq_error < 0.015 and power_error < 0.025, rows[kind]
        assert seconds_adaptive < seconds_exact and adaptive.discovery_count < exact.discovery_count, rows[kind]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("output/doppler-repair/adaptive"))
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=("radial", "rotor", "limbs", "heavy"),
        default=("radial", "rotor", "limbs", "heavy"),
    )
    args = parser.parse_args()
    run(args.output, args.cases)
