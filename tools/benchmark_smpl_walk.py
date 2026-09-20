"""Completed-CUDA SMPL walking benchmark, within 15 m of the radar.

Consumes a Genesis NPZ (smpl_pose, shape, root, fps). The full SMPL mesh is
posed at every adaptive probe. Fixed area-sampled material points feed the
public PointTargets route; this is a discrete scalar-RCS body, not a full-wave
skin solver. No mesh occlusion or room reflections are implied by the LOS case.
Motion generation, setup, simulation, DSP and diagnostic profiling are timed
separately. No precomputed vertex animation is substituted for live SMPL.

The generated joint motion is looped with a periodic cubic spline and a 0.5 s
closing bridge. Root heading/translation are authored on a smooth ellipse:
x=8-5 cos(wt), y=1.5 sin(wt), w=2 pi/30 s^-1. SMPL +Y maps to world +Z;
SMPL +Z follows the tangent. This gives continuous turns, not teleportation.
Analytic vertex velocity is a forward-AD diagnostic of that same expression;
the public dynamic radar route consumes positions, not this diagnostic.
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import math
import platform
import pstats
import subprocess
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.autograd.forward_ad as forward_ad
import trimesh
from benchmark_adaptive_frame import MIMO
from scipy.interpolate import CubicSpline
from witwin.channel import build_info
from witwin.core import Scene

from witwin.radar import Motion, Pattern, PointTargets, Radar, simulation
from witwin.radar import channel as channel_module
from witwin.radar.smpl import SMPLBody

# The legacy point fixture aliases beyond 10.99 m. Keep its array and schedule,
# but cover the user's 15 m working volume with a 16.49 m complex range window.
WALK_RADAR = {**MIMO, "slope": 40.0}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def completed(call):
    torch.cuda.synchronize()
    start = time.perf_counter()
    value = call()
    torch.cuda.synchronize()
    return value, time.perf_counter() - start


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
        "samples_ms": values.tolist(),
    }


class WalkingBody:
    """Authored world-space material points on the complete 6890-vertex body."""

    def __init__(self, motion_path, model_root, count, seed):
        with np.load(motion_path, allow_pickle=False) as data:
            poses = data["smpl_pose"].reshape(-1, 72).astype(np.float64)
            shape = data["shape"].reshape(-1)[:10]
            fps = float(data["fps"])
        if len(poses) < 3 or not np.isfinite(poses).all():
            raise ValueError("a finite motion with at least three frames is required")
        # The route owns root yaw; Genesis owns the 23 articulated joint poses.
        poses[:, :3] = 0
        self.duration = (len(poses) - 1) / fps + 0.5
        clock = np.r_[np.arange(len(poses)) / fps, self.duration]
        self.spline = CubicSpline(clock, np.concatenate((poses, poses[:1])), bc_type="periodic")
        self.body = SMPLBody(pose=poses[0], shape=shape, gender="neutral", model_root=str(model_root), device="cuda")
        self.calls = 0
        vertices, faces = self.body.to_mesh()
        self.faces = faces
        self.vertex_count = len(vertices)
        mesh = trimesh.Trimesh(vertices.cpu().numpy(), faces.cpu().numpy(), process=False)
        points, face_ids = trimesh.sample.sample_surface(mesh, count=count, seed=seed)
        barycentric = trimesh.triangles.points_to_barycentric(mesh.triangles[face_ids], points)
        self.triangles = faces[torch.as_tensor(face_ids, device="cuda")]
        self.barycentric = torch.as_tensor(barycentric, dtype=torch.float32, device="cuda")
        self.height = -float(vertices[:, 1].min())
        self.layout = {"face_indices": face_ids.tolist(), "barycentric": barycentric.tolist()}
        # A fixed motion asset owns one reusable inference graph. It still
        # evaluates all 6890 vertices at every requested instant; no vertex
        # animation or sampled-position cache substitutes for the SMPL model.
        # The pinned input has one allocation per call so an asynchronous copy
        # cannot race a subsequent host write. Torch records its copy lifetime.
        self._input = torch.empty(73, device="cuda")
        self._input[:72].copy_(self.body.pose)
        self._input[72].zero_()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.no_grad():
            for _ in range(3):
                self._world(self._input[:72], self._input[72])
        torch.cuda.current_stream().wait_stream(stream)
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph), torch.no_grad():
            self._vertices = self._world(self._input[:72], self._input[72])

    def _world(self, pose, clock):
        vertices, _ = self.body.updated(pose=pose).to_mesh()
        angle = clock * (2 * math.pi / 30)
        dx, dy = 5 * torch.sin(angle), 1.5 * torch.cos(angle)
        norm = torch.sqrt(dx * dx + dy * dy)
        fx, fy = dx / norm, dy / norm
        # A right-handed basis: local right=(fy,-fx,0), up=(0,0,1), forward=(fx,fy,0).
        x = fy * vertices[:, 0] + fx * vertices[:, 2] + 8 - 5 * torch.cos(angle)
        y = -fx * vertices[:, 0] + fy * vertices[:, 2] + 1.5 * torch.sin(angle)
        z = vertices[:, 1] + self.height
        return torch.stack((x, y, z), dim=1)

    def vertices_at(self, time_s):
        packed = torch.empty(73, dtype=torch.float32, pin_memory=True)
        host = packed.numpy()
        host[:72] = self.spline(time_s % self.duration)
        host[72] = time_s
        self._input.copy_(packed, non_blocking=True)
        self._graph.replay()
        # Callers may retain vertices across observations. Graph storage itself
        # is overwritten at the next replay, so return an independently owned
        # tensor. This adapter is serial, like the scene's trajectory callback.
        return self._vertices.clone()

    def velocity_at(self, time_s):
        phase = time_s % self.duration
        pose = torch.as_tensor(self.spline(phase), dtype=torch.float32, device="cuda")
        rate = torch.as_tensor(self.spline(phase, 1), dtype=torch.float32, device="cuda")
        clock = torch.tensor(time_s, dtype=torch.float32, device="cuda")
        with forward_ad.dual_level():
            vertices = self._world(
                forward_ad.make_dual(pose, rate), forward_ad.make_dual(clock, torch.ones_like(clock))
            )
            return forward_ad.unpack_dual(vertices).tangent.clone()

    def __call__(self, time_s):
        self.calls += 1
        vertices = self.vertices_at(time_s)
        return (vertices[self.triangles] * self.barycentric[:, :, None]).sum(dim=1)


def make_session(body, config=WALK_RADAR):
    radar = Radar.from_dict(config, pattern=Pattern.isotropic(), position=(0, 0, 1), look_at=(1, 0, 1))
    scene = Scene(structures=(), endpoints=[])
    targets = PointTargets(body(0.0), rcs=1.0 / len(body.triangles), trajectory=body)
    return radar, scene, targets


def preview(body, output):
    """Retain body poses and route so the workload can be visually inspected."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    figure = plt.figure(figsize=(14, 4))
    faces = body.faces.cpu().numpy()
    for index, instant in enumerate((1.0, 1.25, 1.5, 1.75)):
        vertices = body.vertices_at(instant).cpu().numpy()
        vertices[:, :2] -= vertices[:, :2].mean(axis=0)
        axis = figure.add_subplot(1, 5, index + 1, projection="3d")
        axis.add_collection3d(Poly3DCollection(vertices[faces], facecolor="#78a7cb", linewidth=0))
        axis.set(xlim=(-0.85, 0.85), ylim=(-0.85, 0.85), zlim=(0, 2), title=f"t={instant:.2f} s")
        axis.set_box_aspect((1, 1, 1.3))
        axis.view_init(elev=12, azim=-65)
        axis.set_axis_off()
    axis = figure.add_subplot(1, 5, 5)
    angle = np.linspace(0, 2 * math.pi, 200)
    axis.plot(8 - 5 * np.cos(angle), 1.5 * np.sin(angle))
    axis.scatter([0], [0], marker="^", label="Radar")
    axis.set(xlabel="x (m)", ylabel="y (m)", title="30 s smooth return loop", xlim=(-1, 15), ylim=(-4, 4))
    axis.set_aspect("equal")
    axis.legend()
    figure.suptitle("Genesis-generated SMPL walk; authored root route; 128 material surface sites")
    figure.tight_layout()
    figure.savefig(output / "workload.png", dpi=150)
    plt.close(figure)


def simulate(radar, scene, targets, time_s, motion=None):
    return radar.simulate(scene, targets, times=(time_s,), los=True, reflections=0, motion=motion or Motion.adaptive())


def validate(body, output, config=WALK_RADAR):
    """Small exhaustive ADC oracle plus loop/velocity checks, outside timing."""
    t = 0.31
    h = 0.001
    velocity = body.velocity_at(t)
    difference = (body.vertices_at(t + h) - body.vertices_at(t - h)) / (2 * h)
    relative = float(torch.linalg.vector_norm(velocity - difference) / torch.linalg.vector_norm(velocity))
    assert relative < 0.01, relative
    phase_jump = float(np.max(np.abs(body.spline(0) - body.spline(body.duration))))
    rate_jump = float(np.max(np.abs(body.spline(0, 1) - body.spline(body.duration, 1))))
    small = {
        **config,
        "num_tx": 1,
        "num_rx": 1,
        "tx_loc": [[0, 0, 0]],
        "rx_loc": [[0, 0, 0]],
        "chirp_per_frame": 4,
        "adc_samples": 16,
    }
    radar, scene, targets = make_session(body, small)
    reference = simulate(radar, scene, targets, t, Motion.adc())
    assert reference.axes.max_unambiguous_range_m > 15
    adaptive = simulate(radar, scene, targets, t)
    error = float(torch.linalg.vector_norm(adaptive.cube - reference.cube) / torch.linalg.vector_norm(reference.cube))
    assert error < 0.02, error
    assert bool(torch.isfinite(reference.cube).all()) and float(reference.cube.abs().max()) > 0
    bounds = torch.stack([body.vertices_at(float(t)) for t in np.linspace(0, 30, 61)])
    ranges = torch.linalg.vector_norm(bounds - bounds.new_tensor([0, 0, 1]), dim=-1)
    assert float(ranges.max()) < 15
    record = {
        "velocity_fd_relative_l2": relative,
        "loop_pose_jump_rad": phase_jump,
        "loop_pose_rate_jump_rad_s": rate_jump,
        "adc_iq_relative_l2": error,
        "adc_observations": 64,
        "max_unambiguous_range_m": reference.axes.max_unambiguous_range_m,
        "range_min_m": float(ranges.min()),
        "range_max_m": float(ranges.max()),
    }
    np.savez(output / "oracle.npz", reference=reference.cube.cpu().numpy(), adaptive=adaptive.cube.cpu().numpy())
    return record


def profile_frame(body, radar, scene, targets, output):
    """Instrumented inclusive times; never substituted for clean wall latency."""
    stages = {}

    def wrapper(name, original):
        def run(*args, **kwargs):
            value, elapsed = completed(lambda: original(*args, **kwargs))
            entry = stages.setdefault(name, {"calls": 0, "inclusive_ms": 0.0})
            entry["calls"] += 1
            entry["inclusive_ms"] += elapsed * 1000
            return value

        return run

    adapter = channel_module.ChannelPropagationAdapter
    with ExitStack() as stack:
        for owner, method, name in (
            (WalkingBody, "vertices_at", "smpl_vertices"),
            (channel_module, "compile_scene", "compile_scene"),
            (adapter, "freeze", "channel_discovery"),
            (adapter, "reevaluate_slots", "channel_replay"),
            (simulation, "_adaptive_trace", "adaptive_trace"),
            (simulation, "_adaptive_echo", "adaptive_echo"),
        ):
            stack.enter_context(patch.object(owner, method, wrapper(name, getattr(owner, method))))
        result, wall = completed(lambda: simulate(radar, scene, targets, 2.3))
    del result
    profile = cProfile.Profile()
    profile.enable()
    simulate(radar, scene, targets, 2.3)
    torch.cuda.synchronize()
    profile.disable()
    profile.dump_stats(str(output / "frame.prof"))
    with (output / "profile.txt").open("w") as handle:
        pstats.Stats(profile, stream=handle).strip_dirs().sort_stats("cumulative").print_stats(65)
    return {"instrumented_wall_ms": wall * 1000, "inclusive_stages": stages}


def benchmark(body, frames, fps, runs, output, do_profile):
    radar, scene, targets = make_session(body)
    cold, cold_s = completed(lambda: simulate(radar, scene, targets, 0.0))
    print(json.dumps({"stage": "first_frame", "sites": len(body.triangles), "seconds": cold_s}), flush=True)
    shape = list(cold.cube.shape)
    del cold
    sample_times = [0.0, 2.3, 7.5, 15.0, 22.5]
    singles = []
    for _ in range(runs):
        for t in sample_times:
            value, elapsed = completed(lambda t=t: simulate(radar, scene, targets, t))
            singles.append({"time_s": t, "ms": elapsed * 1000, "diagnostics": dict(value.adaptive_diagnostics[0])})
            print(
                json.dumps(
                    {
                        "stage": "single_frame",
                        "time_s": t,
                        "ms": elapsed * 1000,
                        "probes": value.adaptive_diagnostics[0]["evaluations"],
                    }
                ),
                flush=True,
            )
            del value
    (output / "single-frames.json").write_text(json.dumps(singles, indent=2) + "\n")
    body.calls = 0
    resident = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    per_frame, dsp, probes, batches, checks = [], [], [], [], []
    torch.cuda.synchronize()
    begin = time.perf_counter()
    iterator = radar.stream(
        scene, targets, times=tuple(i / fps for i in range(frames)), los=True, reflections=0, motion=Motion.adaptive()
    )
    discoveries = 0
    for index in range(frames):
        result, elapsed = completed(lambda: next(iterator))
        per_frame.append(elapsed * 1000)
        diag = result.adaptive_diagnostics[0]
        probes.append(int(diag["evaluations"]))
        batches.append(int(diag["synthesis_batches"]))
        with (output / "frames.jsonl").open("a") as handle:
            handle.write(
                json.dumps(
                    {"index": index, "time_s": index / fps, "simulation_ms": per_frame[-1], "diagnostics": dict(diag)}
                )
                + "\n"
            )
        discoveries = int(result.discovery_count)
        rd, elapsed = completed(lambda result=result: result.frame(0).range_doppler())
        dsp.append(elapsed * 1000)
        checks.append(torch.stack((torch.isfinite(result.cube).all(), result.cube.abs().amax() > 0)))
        if index in {0, frames // 2, frames - 1}:
            np.savez(output / f"frame-{index:04d}.npz", cube=result.cube.cpu().numpy(), rd=rd.data.cpu().numpy())
        del rd, result
        if (index + 1) % 10 == 0:
            print(
                json.dumps(
                    {
                        "sites": len(body.triangles),
                        "frame": index + 1,
                        "recent_ms": float(np.mean(per_frame[-10:])),
                        "probes": probes[-1],
                    }
                ),
                flush=True,
            )
            with (output / "progress.jsonl").open("a") as handle:
                handle.write(
                    json.dumps({"frame": index + 1, "simulation_seconds": sum(per_frame) / 1000, "probes": probes[-1]})
                    + "\n"
                )
    iterator.close()
    torch.cuda.synchronize()
    wall = time.perf_counter() - begin
    assert bool(torch.stack(checks).all()), "nonfinite or zero frame"
    peak = torch.cuda.max_memory_allocated()
    sequence = {
        "frames": frames,
        "fps": fps,
        "scene_seconds": frames / fps,
        "simulation_seconds": sum(per_frame) / 1000,
        "dsp_seconds": sum(dsp) / 1000,
        "wall_with_checks_and_sample_io_seconds": wall,
        "frame_timing": summary(per_frame),
        "dsp_timing": summary(dsp),
        "probes": probes,
        "synthesis_batches": batches,
        "discovery_count": discoveries,
        "smpl_evaluations": body.calls,
        "peak_allocated_mib": peak / 1048576,
        "peak_increment_mib": (peak - resident) / 1048576,
    }
    geometry = {}
    for name, call in (("positions", lambda: body(2.3)), ("analytic_vertex_velocity", lambda: body.velocity_at(2.3))):
        call()
        geometry[name] = summary([completed(call)[1] * 1000 for _ in range(10)])
    return {
        "cube_shape": shape,
        "first_call_ms": cold_s * 1000,
        "single_frames": singles,
        "single_frame_summary": summary([row["ms"] for row in singles]),
        "sequence": sequence,
        "geometry_microbench": geometry,
        "profile": profile_frame(body, radar, scene, targets, output) if do_profile else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--sites", nargs="+", type=int, default=[128])
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--fps", type=float, default=10)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("output/smpl-walk/benchmark"))
    args = parser.parse_args()
    if args.frames < 1 or args.runs < 1 or args.fps <= 0 or min(args.sites) < 1:
        parser.error("frames, runs, fps and sites must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    record = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "platform": platform.platform(),
        "channel_build": build_info(),
        "radar_config": WALK_RADAR,
        "radar_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "motion_path": str(args.motion.resolve()),
        "motion_sha256": digest(args.motion),
        "model_sha256": digest(args.model_root / "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"),
        "script_sha256": digest(__file__),
        "seed": args.seed,
        "scope": "live full SMPL, fixed surface point RCS, LOS, no self-occlusion or room multipath",
        "cases": {},
    }
    (args.output / "environment.json").write_text(json.dumps(record, indent=2) + "\n")
    for count in args.sites:
        output = args.output / str(count)
        output.mkdir(exist_ok=True)
        body, setup = completed(lambda count=count: WalkingBody(args.motion, args.model_root, count, args.seed))
        (output / "surface-layout.json").write_text(json.dumps(body.layout) + "\n")
        validation = validate(body, output) if args.validate else None
        (output / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
        if args.preview:
            preview(body, output)
        result = benchmark(body, args.frames, args.fps, args.runs, output, args.profile)
        record["cases"][str(count)] = {
            "setup_seconds": setup,
            "vertices": body.vertex_count,
            "triangles": len(body.faces),
            "loop_seconds": body.duration,
            "validation": validation,
            **result,
        }
        (args.output / "results.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps({"sites": count, "sequence_seconds": result["sequence"]["simulation_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
