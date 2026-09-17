"""How far the adaptive probe count sits above the partition it publishes.

A partition of A accepted intervals through K nodes cannot be built or tested
for less than `2*(K-1)*A + 1` evaluated instants: each interval needs its own
nodes and the instants between them, and neighbours share one endpoint. Every
probe beyond that went to an interval the controller later rejected.

That excess is the entire budget a cross-frame warm start could recover, since
a warm start still has to build and test whatever partition it ends up with.
Measuring it says whether seeding the partition from the previous frame is
worth its staleness risk, rather than assuming it is.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import torch
from witwin.core import Scene

from witwin.radar.propagation import Kinematics
from witwin.radar.scattering import ScalarRcsResponse
from witwin.radar.simulation import AdaptiveMotionSpec, ScatterSitePolicy


class Rotor:
    """80 Hz, 3 mm radius, 30 m: the micro-Doppler stress case, in metres."""

    def at(self, t):
        rate = 2 * math.pi * 80.0
        points = [[30.0 + 0.003 * math.cos(rate * t), 0.003 * math.sin(rate * t), 0.0]]
        speeds = [[-0.003 * rate * math.sin(rate * t), 0.003 * rate * math.cos(rate * t), 0.0]]
        return Kinematics(
            torch.tensor(points, dtype=torch.float32, device="cuda"),
            torch.tensor(speeds, dtype=torch.float32, device="cuda"),
        )


def _radar(waveform):
    from validate_doppler_motion import make_radar

    radar = make_radar()
    radar.system_config = replace(
        radar.system_config, waveform=replace(radar.system_config.waveform, output_domain="beat", **waveform)
    )
    return radar


def rotor(frames, tolerance):
    waveform = {
        "adc_samples": 128,
        "chirp_per_frame": 128,
        "sample_rate": 4000.0,
        "slope": 0.015625,
        "adc_start_time": 0.0,
        "ramp_end_time": 32.0,
        "idle_time": 0.0,
    }
    radar = _radar(waveform)
    spec = radar.system_config.waveform_spec()
    trajectory = Rotor()
    return radar.simulate(
        Scene(structures=(), endpoints=[]),
        times=tuple(index * spec.num_chirps * spec.chirp_period_s for index in range(frames)),
        response=ScalarRcsResponse.from_rcs(1.0, reference_frequency_hz=77e9, device="cuda"),
        sites=ScatterSitePolicy.explicit(trajectory.at(0).positions_m, trajectory=trajectory),
        components=frozenset({"los"}),
        max_depth=0,
        motion_sampling="adaptive",
        adaptive_motion=AdaptiveMotionSpec(phase_error_rad=tolerance, max_evaluations=40000),
    )


def heavy(frames, tolerance):
    from validate_heavy_multipath import LinearPoint, room

    radar = _radar({"adc_samples": 64, "chirp_per_frame": 32, "idle_time": 442})
    spec = radar.system_config.waveform_spec()
    trajectory = LinearPoint(radar.device)
    return radar.simulate(
        room(),
        times=tuple(index * spec.num_chirps * spec.chirp_period_s for index in range(frames)),
        response=ScalarRcsResponse.from_values(1.0, 0.0, device=radar.device),
        sites=ScatterSitePolicy.explicit(trajectory.at(0).positions_m, trajectory=trajectory),
        components=frozenset({"los", "reflection"}),
        max_depth=2,
        motion_sampling="adaptive",
        adaptive_motion=AdaptiveMotionSpec(phase_error_rad=tolerance),
    )


CASES = {"rotor": (rotor, 4, 0.02), "rotor_tight": (rotor, 2, 0.002), "heavy": (heavy, 3, 0.02)}


def run(name):
    build, frames, tolerance = CASES[name]
    result = build(frames, tolerance)
    rows = []
    for stats in result.adaptive_diagnostics:
        floor = 2 * (stats["interpolation_nodes"] - 1) * stats["accepted_intervals"] + 1
        rows.append(
            {
                "probes": stats["evaluations"],
                "accepted_intervals": stats["accepted_intervals"],
                "interpolation_nodes": stats["interpolation_nodes"],
                "partition_floor": floor,
                "rejected_interval_probes": max(0, stats["evaluations"] - floor),
            }
        )
    probes = sum(row["probes"] for row in rows)
    recoverable = sum(row["rejected_interval_probes"] for row in rows)
    return {
        "case": name,
        "frames": len(rows),
        "phase_error_rad": tolerance,
        "total_probes": probes,
        "warm_start_recoverable_probes": recoverable,
        "warm_start_recoverable_fraction": recoverable / probes if probes else 0.0,
        "per_frame": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=sorted(CASES), default=sorted(CASES))
    parser.add_argument("--output", type=Path, default=Path("output/adaptive-probe-efficiency"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for name in args.cases:
        results.append(run(name))
        print(json.dumps(results[-1]), flush=True)
    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
