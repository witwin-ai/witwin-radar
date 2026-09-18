"""Measured cost of a long frame sequence, stacked against streamed.

Runs the realistic MIMO walker fixture at several sequence lengths through both
public entries and records wall time, peak device allocation and the bit-exact
agreement of every frame. Latency on a shared desktop is evidence, not a
portable threshold.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from witwin.core import Scene

from witwin.radar import Motion, Pattern, PointTargets, Radar
from witwin.radar.propagation import Kinematics

CONFIG = {
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


class Walker:
    """A closing target with a 2 Hz lateral sway, in metres.

    ``positions`` is what ``PointTargets.trajectory`` takes.
    """

    def positions(self, t):
        rate = 2 * math.pi * 2.0
        points = [[6.0 + 1.2 * t, 0.2 * math.sin(rate * t), 0.0]]
        return torch.tensor(points, dtype=torch.float32, device="cuda")

    def at(self, t):
        return Kinematics(self.positions(t))


def session(radar, frames, fps):
    trajectory = Walker()
    return {
        "targets": PointTargets(positions=trajectory.positions(0), rcs=1.0, trajectory=trajectory.positions),
        "times": tuple(index / fps for index in range(frames)),
        "los": True,
        "reflections": 0,
        "motion": Motion.adaptive(),
    }


def timed(call):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    resident = torch.cuda.memory_allocated()
    start = time.perf_counter()
    value = call()
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    return value, seconds, torch.cuda.max_memory_allocated() - resident


def run(frames, fps, checksum):
    # Isotropic elements and a ``+z`` polarization: the walker moves in the
    # ``z = 0`` plane, so neither weighting varies over the sequence.
    radar = Radar.from_dict(
        CONFIG, pattern=Pattern.isotropic(), polarization=(0, 0, 1), position=(0, 0, 0), look_at=(1, 0, 0)
    )
    scene = Scene(structures=(), endpoints=[])
    kwargs = session(radar, frames, fps)
    list(radar.stream(scene, **kwargs))  # warm native loading and the allocator

    def consume():
        """Produce and release every frame, which is what a writer does."""

        count = 0
        for frame in radar.stream(scene, **kwargs):
            count += 1
            del frame
        return count

    produced, streamed_s, streamed_peak = timed(consume)
    assert produced == frames
    torch.cuda.empty_cache()
    record = {
        "frames": frames,
        "fps": fps,
        "streamed_seconds": streamed_s,
        "streamed_peak_bytes": streamed_peak,
        "streamed_per_frame_ms": 1e3 * streamed_s / frames,
    }
    if checksum:
        result, stacked_s, stacked_peak = timed(lambda: radar.simulate(scene, **kwargs))
        record.update(
            stacked_seconds=stacked_s,
            stacked_peak_bytes=stacked_peak,
            stacked_per_frame_ms=1e3 * stacked_s / frames,
            peak_ratio=stacked_peak / streamed_peak,
            cube_bytes=result.cube.numel() * result.cube.element_size(),
        )
        # Bit-exact per frame. A single reduction over the whole stacked cube
        # would use a different summation tree than 128 per-frame reductions
        # and would differ in the last float32 digits for no physical reason,
        # so the comparison is frame by frame on identical shapes.
        stacked_frames = [result.cube[index].clone() for index in range(frames)]
        del result
        torch.cuda.empty_cache()
        mismatched = 0
        for index, frame in enumerate(radar.stream(scene, **kwargs)):
            mismatched += int(not torch.equal(frame.cube[0], stacked_frames[index]))
            del frame
        record["mismatched_frames"] = mismatched
        assert mismatched == 0, record
    torch.cuda.empty_cache()
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, nargs="+", default=[8, 32, 128])
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--no-stacked", action="store_true", help="skip the stacked comparison for long runs")
    parser.add_argument("--output", type=Path, default=Path("output/frame-streaming"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for count in args.frames:
        rows.append(run(count, args.fps, not args.no_stacked))
        print(json.dumps(rows[-1]), flush=True)
    (args.output / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
