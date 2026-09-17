"""Calibrate the adaptive phase tolerance against realized IQ error.

The adaptive controller bounds a PER-PATH phase error. What a consumer cares
about is the relative error of the coherently summed IQ cube. The two are not
the same number and no bound converts one into the other near a coherent null,
so this tool measures the relationship on named fixtures instead of assuming
it: for each tolerance it records the largest error the controller actually
tested, the realized error against the exhaustive per-ADC route, and the cost.

The per-ADC route is the reference, not an oracle: it is the same physics
evaluated at every observation. Latency is shared-desktop evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import torch
from witwin.core import Scene

from witwin.radar import Radar
from witwin.radar.processing.range_doppler import fmcw_range_fft
from witwin.radar.propagation import Kinematics
from witwin.radar.scattering import ScalarRcsResponse
from witwin.radar.sensors import ISOTROPIC_PATTERN
from witwin.radar.simulation import AdaptiveMotionSpec, ScatterSitePolicy

TOLERANCES = (0.002, 0.005, 0.02, 0.05, 0.2, 0.5)

BASE = {
    "fc": 77e9,
    "slope": 60.012,
    "adc_start_time": 0,
    "sample_rate": 4400,
    "idle_time": 7,
    "ramp_end_time": 58,
    "frame_per_second": 10,
    "num_angle_bins": 64,
    "power": 12,
    "antenna_pattern": {
        "kind": ISOTROPIC_PATTERN.kind,
        "x_angles_deg": list(ISOTROPIC_PATTERN.x_angles_deg),
        "y_angles_deg": list(ISOTROPIC_PATTERN.y_angles_deg),
        "x_values": list(ISOTROPIC_PATTERN.x_values),
        "y_values": list(ISOTROPIC_PATTERN.y_values),
    },
}

#: Each fixture is an array shape and a trajectory, sized so that the
#: exhaustive per-ADC reference is affordable for every tolerance.
FIXTURES = {
    "walker": {
        "array": {
            "num_tx": 3,
            "num_rx": 4,
            "tx_loc": [[0, 0, 0], [2, 0, 0], [0, 1, 0]],
            "rx_loc": [[i, 0, 0] for i in range(4)],
        },
        "waveform": {"adc_samples": 64, "chirp_per_frame": 16},
    },
    "rotor": {
        "array": {"num_tx": 1, "num_rx": 1, "tx_loc": [[0, 0, 0]], "rx_loc": [[0, 0, 0]]},
        "waveform": {"adc_samples": 64, "chirp_per_frame": 32},
    },
}


class Motion:
    """Metres and m/s. ``rotor`` is the micro-Doppler stress case."""

    def __init__(self, kind):
        self.kind = kind

    def at(self, t):
        if self.kind == "rotor":
            rate = 2 * math.pi * 80.0
            points = [[30.0 + 0.003 * math.cos(rate * t), 0.003 * math.sin(rate * t), 0.0]]
            speeds = [[-0.003 * rate * math.sin(rate * t), 0.003 * rate * math.cos(rate * t), 0.0]]
        else:
            rate = 2 * math.pi * 2.0
            points = [[6.0 + 1.2 * t, 0.2 * math.sin(rate * t), 0.0]]
            speeds = [[1.2, 0.2 * rate * math.cos(rate * t), 0.0]]
        return Kinematics(
            torch.tensor(points, dtype=torch.float32, device="cuda"),
            torch.tensor(speeds, dtype=torch.float32, device="cuda"),
        )


def build(name):
    fixture = FIXTURES[name]
    config = {**BASE, **fixture["array"], **fixture["waveform"]}
    config.update(
        num_range_bins=fixture["waveform"]["adc_samples"], num_doppler_bins=fixture["waveform"]["chirp_per_frame"]
    )
    radar = Radar(config, position=(0, 0, 0), target=(1, 0, 0))
    radar.system_config = replace(
        radar.system_config, waveform=replace(radar.system_config.waveform, output_domain="beat", **fixture["waveform"])
    )
    trajectory = Motion(name)
    return (
        radar,
        trajectory,
        {
            "times": (0.0,),
            "response": ScalarRcsResponse.from_rcs(1.0, reference_frequency_hz=77e9, device="cuda"),
            "sites": ScatterSitePolicy.explicit(trajectory.at(0).positions_m, trajectory=trajectory),
            "components": frozenset({"los"}),
            "max_depth": 0,
        },
    )


def timed(call):
    torch.cuda.synchronize()
    start = time.perf_counter()
    value = call()
    torch.cuda.synchronize()
    return value, time.perf_counter() - start


def relative(actual, reference):
    return float((actual - reference).norm() / reference.norm())


def run(name):
    radar, _, kwargs = build(name)
    radar.simulate(Scene(structures=(), endpoints=[]), **kwargs, motion_sampling="adaptive")  # warm
    scene = Scene(structures=(), endpoints=[])
    reference, reference_s = timed(lambda: radar.simulate(scene, **kwargs, motion_sampling="adc"))
    reference_rd = torch.fft.fftshift(torch.fft.fft(fmcw_range_fft(reference.cube), dim=-2), dim=-2)
    rows = []
    for tolerance in TOLERANCES:
        spec = AdaptiveMotionSpec(phase_error_rad=tolerance, relative_amplitude_error=tolerance)
        result, seconds = timed(
            lambda spec=spec: radar.simulate(scene, **kwargs, motion_sampling="adaptive", adaptive_motion=spec)
        )
        stats = result.adaptive_diagnostics[0]
        power = torch.fft.fftshift(torch.fft.fft(fmcw_range_fft(result.cube), dim=-2), dim=-2)
        rows.append(
            {
                "phase_tolerance_rad": tolerance,
                "max_tested_phase_rad": stats["max_tested_phase_error_rad"],
                "max_tested_relative_amplitude": stats["max_tested_relative_amplitude_error"],
                "tolerance_headroom": tolerance / max(stats["max_tested_phase_error_rad"], 1e-12),
                "iq_relative_l2": relative(result.cube, reference.cube),
                "rd_power_relative_l2": relative(power.abs().square(), reference_rd.abs().square()),
                "evaluations": stats["evaluations"],
                "accepted_intervals": stats["accepted_intervals"],
                "max_interval_enforced": stats["max_interval_enforced"],
                "seconds": seconds,
                "speedup_vs_adc": reference_s / seconds,
                "path_set_complete": result.path_set_complete,
            }
        )
        print(name, json.dumps(rows[-1]), flush=True)
        del result, power
    return {
        "fixture": name,
        "observations": stats["observation_count"],
        "adc_reference_seconds": reference_s,
        "tolerances": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", nargs="+", choices=sorted(FIXTURES), default=sorted(FIXTURES))
    parser.add_argument("--output", type=Path, default=Path("output/adaptive-tolerance"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = [run(name) for name in args.fixtures]
    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
