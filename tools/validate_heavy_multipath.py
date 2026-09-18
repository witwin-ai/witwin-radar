"""Moving-point Range-Doppler experiment in a three-wall, depth-two room.

Run with the witwin2 Channel runtime, as for validate_doppler_motion.py.
Uses the public per-ADC simulator and processing metadata. Image-source
geometry below is an independent test oracle, not production propagation.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import replace
from pathlib import Path

import torch
from witwin.core import Mesh, PhysicalMaterial, Scene, Structure

from tools.validate_doppler_motion import C0, make_radar
from witwin.radar import Motion, PointTargets
from witwin.radar.processing import ProcessingAxes, ProcessingCube, range_doppler_map, range_profile
from witwin.radar.propagation import Kinematics


class LinearPoint:
    """One point on a straight line. ``positions`` is the simulator's view of it.

    ``velocity`` is the constant rate, which only the image oracle below reads;
    both are views of one closed form, so the oracle cannot drift from the
    motion the simulator was handed.
    """

    def __init__(self, device):
        self.origin = torch.tensor([[2.0, 0.35, 0.0]], device=device)
        self.velocity = torch.tensor([[0.8, 0.3, 0.0]], device=device)

    def positions(self, t):
        return self.origin + t * self.velocity

    def at(self, t):
        return Kinematics(self.positions(t))


def room():
    vertices = (
        [[6, -2, -2], [6, 2, -2], [6, 2, 2], [6, -2, 2]],
        [[-1, -2, -2], [6, -2, -2], [6, -2, 2], [-1, -2, 2]],
        [[-1, 2, -2], [-1, 2, 2], [6, 2, 2], [6, 2, -2]],
    )
    structures = []
    for index, points in enumerate(vertices, 1):
        mesh = Mesh(
            vertices=torch.tensor(points, dtype=torch.float32),
            faces=torch.tensor([[0, 1, 2], [0, 2, 3]]),
            recenter=False,
            fill_mode="surface",
            topology_diagnostics=False,
        )
        structures.append(
            Structure(
                geometry=mesh,
                material=PhysicalMaterial(name="reflective_wall", eps_r=16.0, sigma_e=0.05),
                structure_id=index,
                material_id=index,
                assignment_id=index,
                surface_id=index,
            )
        )
    return Scene(structures=tuple(structures), endpoints=[])


def images():
    # Static infinite-plane image candidates. Finite-facet visibility remains
    # Channel's responsibility: only candidates matched to valid legs are used.
    planes = ((0, 6.0), (1, -2.0), (1, 2.0))
    result = [((), torch.zeros(3, dtype=torch.float64))]
    for depth in (1, 2):
        for sequence in itertools.product(range(3), repeat=depth):
            if depth == 2 and sequence[0] == sequence[1]:
                continue
            point = torch.zeros(3, dtype=torch.float64)
            for wall in sequence:
                axis, coordinate = planes[wall]
                point[axis] = 2 * coordinate - point[axis]
            result.append((sequence, point))
    return result


def oracle_rows(result, trajectory, spec):
    paths, legs = result.last_radar_paths, result.last_propagation
    last_time = result.sample_times_s[0][-1]
    middle_time = sum((result.sample_times_s[0][0], last_time)) / 2
    point = trajectory.at(last_time).positions_m[0].double().cpu()
    mid = trajectory.at(middle_time).positions_m[0].double().cpu()
    velocity = trajectory.velocity[0].double().cpu()
    candidates = images()
    matched = []
    maximum_error = 0.0
    for leg in (legs.inbound, legs.outbound):
        indices = []
        for length, depth in zip((leg.delay_s.double() * C0).cpu(), leg.depth.cpu(), strict=True):
            options = [i for i, (sequence, _) in enumerate(candidates) if len(sequence) == int(depth)]
            errors = [abs(float((point - candidates[i][1]).norm() - length)) for i in options]
            best = min(range(len(options)), key=errors.__getitem__)
            maximum_error = max(maximum_error, errors[best])
            indices.append(options[best])
        matched.append(indices)
    assert maximum_error < 2e-5, maximum_error
    groups = {}
    for row, (incoming, outgoing) in enumerate(
        zip(paths.topology.inbound_row.cpu(), paths.topology.outbound_row.cpu(), strict=True)
    ):
        a, b = matched[0][int(incoming)], matched[1][int(outgoing)]
        key = tuple(sorted((a, b)))
        if key not in groups:
            delta = torch.stack((mid - candidates[a][1], mid - candidates[b][1]))
            lengths = delta.norm(dim=-1)
            rate = float(((delta / lengths[:, None]) @ velocity).sum())
            groups[key] = {
                "images": [list(candidates[i][0]) for i in key],
                "equivalent_range_m": float(lengths.sum() / 2),
                "closing_velocity_mps": -rate / 2,
                "physical_doppler_hz": -spec.reference_frequency_hz * rate / C0,
                "complex_weight": 0j,
                "rows": 0,
            }
        groups[key]["complex_weight"] += complex(paths.complex_transfer_ref[row])
        groups[key]["rows"] += 1
    for group in groups.values():
        group["coherent_amplitude"] = abs(group.pop("complex_weight"))
    return list(groups.values()), maximum_error


def run(args):
    radar = make_radar()
    # ``Radar`` is immutable, so a reshaped waveform makes a new radar. The
    # timings are SI: the 442 us idle of the original fixture is 442e-6 s.
    radar = radar.replace(
        waveform=replace(
            radar.waveform, samples_per_chirp=args.samples, chirps_per_frame=args.chirps, idle=442e-6, output="beat"
        )
    )
    spec = radar.waveform_spec()
    trajectory, scene = LinearPoint(radar.device), room()
    results = []
    for t in (0.0, 0.5):
        start = time.perf_counter()
        result = radar.simulate(
            scene,
            PointTargets(positions=trajectory.origin, amplitude=1.0, trajectory=trajectory.positions),
            times=(t,),
            los=True,
            reflections=2,
            motion=Motion.adc(),
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        synthesis = result.frame_synthesis()
        axes = ProcessingAxes.from_synthesis(synthesis, spec, radar.system_config.sensors.array)
        rd = range_doppler_map(range_profile(ProcessingCube.from_synthesis(synthesis, axes), window="hann"))
        power = rd.data.abs().square().squeeze().cpu()
        ranges, velocities = rd.range_axis.cpu(), rd.doppler_axis.cpu()
        groups, error = oracle_rows(result, trajectory, spec)
        for group in groups:
            r = int((ranges - group["equivalent_range_m"]).abs().argmin())
            d = int((velocities - group["closing_velocity_mps"]).abs().argmin())
            group["bin_power"] = float(power[d, r])
        record = {
            "frame_start_s": t,
            "seconds": elapsed,
            "observations": len(result.sample_times_s[0]),
            "discoveries": result.discovery_count,
            "path_set_complete": result.path_set_complete,
            "motion_sampling": result.motion_sampling,
            "round_trip_rows": result.last_radar_paths.path_count,
            "inbound_legs": result.last_propagation.inbound.leg_count,
            "outbound_legs": result.last_propagation.outbound.leg_count,
            "max_image_length_error_m": error,
            "groups": groups,
        }
        assert record["round_trip_rows"] >= 16, record["round_trip_rows"]
        results.append(record)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        torch.save({"power": power, "range_m": ranges, "closing_velocity_mps": velocities}, args.output / f"rd-{t}.pt")
        print(json.dumps({k: v for k, v in record.items() if k != "groups"}), flush=True)


def plot_saved(output):
    results = json.loads((output / "results.json").read_text(encoding="utf-8"))
    plots = []
    for record in results:
        data = torch.load(output / f"rd-{record['frame_start_s']}.pt", weights_only=True)
        plots.append((data["power"], data["range_m"], data["closing_velocity_mps"]))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reference = max(float(p.max()) for p, _, _ in plots)
    figure, axes_plot = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for row, ((power, ranges, velocities), record) in enumerate(zip(plots, results, strict=True)):
        db = 10 * torch.log10((power / reference).clamp_min(1e-7))
        image = axes_plot[row, 0].pcolormesh(ranges, velocities, db, shading="auto", vmin=-55, vmax=0)
        axes_plot[row, 0].set(
            title=f"t={record['frame_start_s']:.1f}s, {record['round_trip_rows']} round trips",
            xlabel="Equivalent range L/2 (m)",
            ylabel="Closing-positive velocity (m/s)",
            xlim=(0, 12),
        )
        groups = record["groups"]
        axes_plot[row, 0].scatter(
            [g["equivalent_range_m"] for g in groups],
            [g["closing_velocity_mps"] for g in groups],
            marker="x",
            color="white",
            s=24,
            label="Image-geometry prediction",
        )
        axes_plot[row, 0].legend(fontsize=8)
        profile = power.sum(0)
        axes_plot[row, 1].plot(ranges, 10 * torch.log10((profile / reference).clamp_min(1e-7)))
        axes_plot[row, 1].set(
            title="Doppler-integrated power; shared reference", xlabel="Equivalent range (m)", ylabel="dB", xlim=(0, 12)
        )
        axes_plot[row, 1].grid(alpha=0.25)
    figure.colorbar(image, ax=axes_plot[:, 0], label="Power (dB), shared maximum")
    figure.suptitle(
        "One moving target (0.8, 0.3, 0) m/s; three walls; up to two reflections per leg\n"
        "Hann range window; rectangular Doppler window (weak vertical streaks include sidelobes)"
    )
    figure.savefig(output / "range-doppler.png", dpi=160)
    plt.close(figure)


def analyze_saved(output):
    """Check resolved peaks against image geometry, retaining unmatched peaks too."""
    records = json.loads((output / "results.json").read_text(encoding="utf-8"))
    tensors = [torch.load(output / f"rd-{r['frame_start_s']}.pt", weights_only=True) for r in records]
    reference = max(float(t["power"].max()) for t in tensors)
    summary = []
    for record, tensors_frame in zip(records, tensors, strict=True):
        power = tensors_frame["power"]
        ranges, velocities = tensors_frame["range_m"], tensors_frame["closing_velocity_mps"]
        dr, dv = float(ranges[1] - ranges[0]), float(velocities[1] - velocities[0])
        residual = power.clone()
        peaks = []
        for _ in range(12):
            d, r = divmod(int(residual.argmax()), residual.shape[1])
            measured_range, measured_velocity = float(ranges[r]), float(velocities[d])
            group = min(
                record["groups"],
                key=lambda g: (
                    ((g["equivalent_range_m"] - measured_range) / dr) ** 2
                    + ((g["closing_velocity_mps"] - measured_velocity) / dv) ** 2
                ),
            )
            range_error = abs(group["equivalent_range_m"] - measured_range)
            velocity_error = abs(group["closing_velocity_mps"] - measured_velocity)
            peaks.append(
                {
                    "range_m": measured_range,
                    "closing_velocity_mps": measured_velocity,
                    "power_db_shared_reference": float(10 * torch.log10(power[d, r] / reference)),
                    "closest_image_pair": group["images"],
                    "range_error_m": range_error,
                    "velocity_error_mps": velocity_error,
                    "within_one_bin": range_error <= dr and velocity_error <= dv,
                }
            )
            residual[max(0, d - 1) : d + 2, max(0, r - 1) : r + 2] = 0
        # Strongest four independently resolved peaks must all agree with the
        # analytic image geometry; weak sidelobes are recorded, not relabelled.
        assert all(peak["within_one_bin"] for peak in peaks[:4]), peaks[:4]
        assert max(peak["range_m"] for peak in peaks[:4]) - min(peak["range_m"] for peak in peaks[:4]) > 2
        summary.append(
            {"time_s": record["frame_start_s"], "range_resolution_m": dr, "velocity_resolution_mps": dv, "peaks": peaks}
        )
    (output / "peak-validation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Strongest four peaks in each frame agree with image geometry within one range/velocity bin.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/doppler-repair/heavy-multipath"))
    parser.add_argument("--chirps", type=int, default=64)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--analyze-only", action="store_true", help="check existing saved maps without resimulation")
    arguments = parser.parse_args()
    if not arguments.analyze_only:
        run(arguments)
    plot_saved(arguments.output)
    analyze_saved(arguments.output)
