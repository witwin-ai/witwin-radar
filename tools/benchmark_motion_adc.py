"""Completed-CUDA motion-to-ADC measurements, without motion generation or DSP.

An optional frozen source directory provides an independent before-change
oracle (smpl.py, simulation.py, walk.py). Alternate its execution order each
round to reduce shared-desktop load bias. Never compare historical timings as
if they were paired measurements.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

import benchmark_smpl_walk as walk
import numpy as np
import torch

from witwin.radar import Motion, simulation


def load_snapshot(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--motion", type=Path, default=Path("tools/fixtures/smpl_walk_genesis_seed10.npz"))
    parser.add_argument("--model-root", type=Path, default=Path("output/smpl-walk/models"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--paired-stream", action="store_true")
    parser.add_argument("--sites", type=int, default=128)
    args = parser.parse_args()
    if args.paired_stream and not args.baseline:
        parser.error("--paired-stream requires --baseline")
    if args.rounds < 1 or args.frames < 0 or args.sites < 1:
        parser.error("rounds/sites must be positive and frames nonnegative")
    if (args.output / "environment.json").exists():
        parser.error("output already contains a measurement; choose a new directory to preserve its evidence")
    args.output.mkdir(parents=True, exist_ok=True)
    config = {**walk.WALK_RADAR, "output_domain": "beat"}
    body, setup_seconds = walk.completed(lambda: walk.WalkingBody(args.motion, args.model_root, args.sites, 10))
    bodies = {"current": body}
    echoes = {"current": simulation._adaptive_echo}
    if args.baseline:
        old_smpl = load_snapshot("witwin.radar._before_smpl", args.baseline / "smpl.py")
        old_simulation = load_snapshot("witwin.radar._before_simulation", args.baseline / "simulation.py")
        old_walk = load_snapshot("_before_walk", args.baseline / "walk.py")
        old_walk.SMPLBody = old_smpl.SMPLBody
        bodies["baseline"] = old_walk.WalkingBody(args.motion, args.model_root, args.sites, 10)
        echoes["baseline"] = old_simulation._adaptive_echo
    sessions = {name: walk.make_session(body, config) for name, body in bodies.items()}
    rows = []
    errors = []
    record = {
        "scope": "motion sequence -> live full SMPL -> LOS -> complete complex ADC; no diffusion or DSP",
        "radar_config": config,
        "motion_sha256": walk.digest(args.motion),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "shared_desktop_load": True,
        "setup_seconds_including_graph_capture": setup_seconds,
        "channel_build": walk.build_info(),
        "model_sha256": walk.digest(args.model_root / "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"),
        "script_sha256": walk.digest(__file__),
        "sources": {
            str(p): walk.digest(p)
            for p in (
                Path("witwin/radar/smpl.py"),
                Path("witwin/radar/simulation.py"),
                Path("witwin/radar/paths.py"),
                Path("tools/benchmark_smpl_walk.py"),
            )
        },
    }
    from witwin.radar.cuda import runtime

    record["radar_native_build"] = runtime.read_build_info(runtime.prebuilt_extension_path())
    if args.baseline:
        record["baseline_sources"] = {str(p): walk.digest(p) for p in args.baseline.glob("*.py")}
    (args.output / "environment.json").write_text(json.dumps(record, indent=2))

    def run(name, instant):
        with patch.object(simulation, "_adaptive_echo", echoes[name]):
            return walk.completed(lambda: walk.simulate(*sessions[name], instant))

    record["first_call_ms"] = {name: run(name, 0.0)[1] * 1000 for name in bodies}
    if "baseline" in bodies:
        retained = body.vertices_at(0.0)
        saved = retained.clone()
        vertex_errors = []
        for instant in np.linspace(0, 30, 31):
            a = bodies["baseline"].vertices_at(float(instant))
            b = body.vertices_at(float(instant))
            vertex_errors.append(float((a - b).abs().max()))
        assert torch.equal(retained, saved), "graph replay overwrote retained vertices"
        assert max(vertex_errors) == 0.0, vertex_errors
        record["vertices_max_absolute_error_m"] = max(vertex_errors)
        del a, b, retained, saved
    for repeat in range(args.rounds):
        for instant in (0.0, 2.3, 7.5, 15.0, 22.5):
            cubes = {}
            names = list(bodies) if repeat % 2 else list(reversed(bodies))
            for name in names:
                resident = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                result, seconds = run(name, instant)
                peak_increment = (torch.cuda.max_memory_allocated() - resident) / 1048576
                cubes[name] = result.cube.detach().clone()
                row = {
                    "variant": name,
                    "round": repeat,
                    "time_s": instant,
                    "ms": seconds * 1000,
                    "peak_increment_mib": peak_increment,
                    "diagnostics": dict(result.adaptive_diagnostics[0]),
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
                del result
            if "baseline" in cubes:
                a, b = cubes["baseline"], cubes["current"]
                error = float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(a))
                errors.append({"round": repeat, "time_s": instant, "relative_l2": error, "equal": torch.equal(a, b)})
                assert error < 0.005, errors[-1]
            del cubes
            (args.output / "paired.json").write_text(json.dumps({"measurements": rows, "errors": errors}, indent=2))

    body = bodies["current"]
    radar, scene, targets = sessions["current"]
    if args.frames:
        variants = list(bodies) if args.paired_stream else ["current"]
        iterators = {
            name: sessions[name][0].stream(
                sessions[name][1],
                sessions[name][2],
                times=tuple(i / 10 for i in range(args.frames)),
                los=True,
                reflections=0,
                motion=Motion.adaptive(),
            )
            for name in variants
        }
        resident = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        sequence = []
        peak_allocated = resident
        for index in range(args.frames):
            frames, timings = {}, {}
            for name in variants if index % 2 else list(reversed(variants)):
                with patch.object(simulation, "_adaptive_echo", echoes[name]):
                    frames[name], timings[name] = walk.completed(lambda name=name: next(iterators[name]))
            frame, seconds = frames["current"], timings["current"]
            assert bool(torch.isfinite(frame.cube).all()) and float(frame.cube.abs().max()) > 0
            row = {"frame": index, "ms": seconds * 1000, "diagnostics": dict(frame.adaptive_diagnostics[0])}
            if args.paired_stream:
                reference = frames["baseline"].cube
                error = float(torch.linalg.vector_norm(frame.cube - reference) / torch.linalg.vector_norm(reference))
                assert error < 1e-5, (index, error)
                assert row["diagnostics"]["evaluations"] == frames["baseline"].adaptive_diagnostics[0]["evaluations"]
                row.update(baseline_ms=timings["baseline"] * 1000, relative_l2=error)
                del reference
            sequence.append(row)
            with (args.output / "frames.jsonl").open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            if index in {0, args.frames // 2, args.frames - 1}:
                np.savez(args.output / f"adc-{index:04d}.npz", adc=frame.cube.cpu().numpy())
            peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated())
            del frame, frames
            if (index + 1) % 10 == 0:
                print(
                    json.dumps({"frame": index + 1, "recent_ms": np.mean([r["ms"] for r in sequence[-10:]])}),
                    flush=True,
                )
        for iterator in iterators.values():
            iterator.close()
        record["sequence"] = {
            "frames": args.frames,
            "scene_seconds": args.frames / 10,
            "simulation_seconds": sum(r["ms"] for r in sequence) / 1000,
            "timing": walk.summary([r["ms"] for r in sequence]),
            "peak_allocated_mib": peak_allocated / 1048576,
            "peak_increment_mib": (peak_allocated - resident) / 1048576,
        }
        if args.paired_stream:
            baseline_seconds = sum(r["baseline_ms"] for r in sequence) / 1000
            record["sequence"].update(
                baseline_seconds=baseline_seconds,
                baseline_timing=walk.summary([r["baseline_ms"] for r in sequence]),
                speedup=baseline_seconds / record["sequence"]["simulation_seconds"],
                max_relative_l2=max(r["relative_l2"] for r in sequence),
            )
    record["timings"] = {name: walk.summary([r["ms"] for r in rows if r["variant"] == name]) for name in bodies}
    record["errors"] = errors
    record["profile"] = walk.profile_frame(body, radar, scene, targets, args.output)
    record["validation"] = walk.validate(body, args.output, config)
    (args.output / "results.json").write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
