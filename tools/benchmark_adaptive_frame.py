"""Per-frame cost of the adaptive route, in the three regimes it has.

The regimes are what make one number insufficient. A frame is limited by the
probe-spacing bound when the motion is smooth and the observation count is
large, by the phase test when the motion is fast enough to shorten intervals
below that bound, and by topology discovery when the world has structures. The
first two are here; the third is ``tools/validate_heavy_multipath.py``.

Fixtures live in this file rather than in a report, because a latency table
whose fixture is described only as "1x1, 128 x 128" cannot be reproduced.

``--sequence`` answers the question a dataset author actually asks - how long
does a minute of data take - by streaming it rather than by extrapolating one
frame. It reports per-segment cost and probe count, because neither is
guaranteed to be flat: a target receding far enough that one float32 step of
its position exceeds the phase tolerance makes every interval uncertifiable,
and the controller bisects.

Latency on a shared desktop is evidence, not a portable threshold.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch
from witwin.core import Scene

from witwin.radar import Motion, Pattern, PointTargets, Radar

#: A TI IWR1843-shaped 77 GHz TDM-MIMO front end.
MIMO = {
    "num_tx": 3,
    "num_rx": 4,
    "fc": 77e9,
    "slope": 60.012,
    "adc_samples": 256,
    "adc_start_time": 0,
    "sample_rate": 4400,
    "idle_time": 7,
    "ramp_end_time": 58,
    "chirp_per_frame": 128,
    "power": 12,
    "tx_loc": [[0, 0, 0], [2, 0, 0], [0, 1, 0]],
    "rx_loc": [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]],
}
#: One transmitter and one receiver, shorter and faster, for the micro-Doppler row.
SISO = {
    **MIMO,
    "num_tx": 1,
    "num_rx": 1,
    "adc_samples": 128,
    "tx_loc": [[0, 0, 0]],
    "rx_loc": [[0, 0, 0]],
    "sample_rate": 4000,
    "idle_time": 2,
    "ramp_end_time": 30,
}
#: The same array on a quarter of the observations, to expose fixed overhead.
SMALL = {**MIMO, "adc_samples": 64, "chirp_per_frame": 32}


def walker(t):
    """A target closing at 1.2 m/s with a 2 Hz lateral sway, in metres.

    Smooth enough that the probe-spacing bound, not the phase test, is what
    sets the probe count.
    """

    return torch.tensor(
        [[6.0 + 1.2 * t, 0.2 * math.sin(2 * math.pi * 2.0 * t), 0.0]], dtype=torch.float32, device="cuda"
    )


def rotor(t):
    """A 5 cm blade tip at 80 Hz, 3 m out, in metres.

    Its 25 m/s tangential speed is what makes the phase test, not the bound,
    decide the interval length - the regime a higher interpolation order is for.
    """

    angle = 2 * math.pi * 80.0 * t
    return torch.tensor(
        [[3.0 + 0.05 * math.cos(angle), 0.05 * math.sin(angle), 0.0]], dtype=torch.float32, device="cuda"
    )


def pair(t):
    """Two scatterers at different rates, so the pair layout is not trivial."""

    angle = 2 * math.pi * 30.0 * t
    return torch.tensor(
        [[4.0 + 0.8 * t, 0.02 * math.sin(angle), 0.1], [5.5 - 0.3 * t, -0.2, -0.15]], dtype=torch.float32, device="cuda"
    )


def pacer(t):
    """A person pacing inside 3 to 8 m, in metres.

    The sequence fixture, and deliberately NOT a receding one: past about 50 m
    one float32 step of position is 0.02 rad of two-way phase at 77 GHz, which
    is the default tolerance, and the probe count then grows without bound.
    ``--recede`` selects that case on purpose.
    """

    return torch.tensor(
        [[5.5 + 2.5 * math.sin(2 * math.pi * 0.15 * t), 0.2 * math.sin(2 * math.pi * 2.0 * t), 0.0]],
        dtype=torch.float32,
        device="cuda",
    )


FIXTURES = {
    "walker": (MIMO, walker, "bound-limited MIMO, 98304 observations"),
    "rotor": (SISO, rotor, "phase-limited micro-Doppler, 16384 observations"),
    "pair": (SMALL, pair, "two scatterers, 6144 observations"),
}


def build(config, trajectory):
    radar = Radar.from_dict(
        config, pattern=Pattern.isotropic(), polarization=(0, 0, 1), position=(0, 0, 0), look_at=(1, 0, 0)
    )
    return radar, Scene(structures=(), endpoints=[])


def session(trajectory, times):
    return {
        "targets": PointTargets(positions=trajectory(times[0]), rcs=1.0, trajectory=trajectory),
        "times": times,
        "los": True,
        "reflections": 0,
        "motion": Motion.adaptive(),
    }


def one_frame(name, repeats):
    """One frame, timed on its own: no second frame averaged into it."""

    config, trajectory, label = FIXTURES[name]
    radar, scene = build(config, trajectory)
    kwargs = session(trajectory, (0.0,))
    result = radar.simulate(scene, **kwargs)
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        begin = time.perf_counter()
        radar.simulate(scene, **kwargs)
        torch.cuda.synchronize()
        samples.append(1e3 * (time.perf_counter() - begin))
    diagnostics = dict(result.adaptive_diagnostics[0])
    return {
        "fixture": name,
        "regime": label,
        "median_ms": round(statistics.median(samples), 2),
        "min_ms": round(min(samples), 2),
        "max_ms": round(max(samples), 2),
        "runs": repeats,
        "observations": diagnostics["observation_count"],
        "probes": diagnostics["evaluations"],
        "intervals": diagnostics["accepted_intervals"],
        "batches": diagnostics["synthesis_batches"],
        "discoveries": int(result.discovery_count),
    }


def sequence(frames, fps, recede):
    """A whole sequence, streamed, reported in segments rather than averaged."""

    trajectory = walker if recede else pacer
    radar, scene = build(MIMO, trajectory)
    step = max(1, frames // 12)
    list(radar.stream(scene, **session(trajectory, tuple(i / fps for i in range(4)))))
    torch.cuda.reset_peak_memory_stats()
    resident = torch.cuda.memory_allocated()

    segments, probes = [], []
    torch.cuda.synchronize()
    begin = last = time.perf_counter()
    for index, frame in enumerate(radar.stream(scene, **session(trajectory, tuple(i / fps for i in range(frames))))):
        probes.append(frame.adaptive_diagnostics[0]["evaluations"])
        del frame
        if (index + 1) % step == 0:
            torch.cuda.synchronize()
            now = time.perf_counter()
            segments.append(round(1e3 * (now - last) / step, 1))
            last = now
    torch.cuda.synchronize()
    total = time.perf_counter() - begin
    return {
        "frames": frames,
        "fps": fps,
        "trajectory": "receding walker" if recede else "pacing in 3-8 m",
        "scene_seconds": round(frames / fps, 2),
        "wall_seconds": round(total, 2),
        "ms_per_frame": round(1e3 * total / frames, 2),
        "segment_ms": segments,
        "segment_probes": [probes[i] for i in range(step - 1, frames, step)],
        "peak_mib": round((torch.cuda.max_memory_allocated() - resident) / 1048576, 1),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", nargs="+", default=sorted(FIXTURES), choices=sorted(FIXTURES))
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--sequence", type=int, default=0, help="also stream this many frames")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--recede", action="store_true", help="stream the receding walker instead of a pacer")
    parser.add_argument("--output", type=Path, default=Path("output/adaptive-frame"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows = [one_frame(name, args.runs) for name in args.fixtures]
    for row in rows:
        print(json.dumps(row), flush=True)
    record = {"frames": rows}
    if args.sequence:
        record["sequence"] = sequence(args.sequence, args.fps, args.recede)
        print(json.dumps(record["sequence"]), flush=True)
    (args.output / "results.json").write_text(json.dumps(record, indent=2) + "\n")
