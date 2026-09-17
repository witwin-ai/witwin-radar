"""Actual MATLAB comparisons for material response, motion, and timed workloads.

All reference geometry and material equations in this tool are explicitly
independent experiment oracles. Production values come through Radar/Channel.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace
from functools import partial
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat, savemat
from scipy.signal import stft
from witwin.core import Mesh, PhysicalMaterial, Scene, Structure

from witwin.radar.propagation import Kinematics, RadarEndpointSpec
from witwin.radar.scattering import ScalarRcsResponse
from witwin.radar.sensors import TxPowerSpec
from witwin.radar.synthesis.assembly import FmcwSpec
from witwin.radar.synthesis.fmcw import synthesize_fmcw_rows

C0 = 299792458.0


def endpoint(position, polarization, source):
    return RadarEndpointSpec(
        torch.tensor([1 if source else 2], dtype=torch.int64, device="cuda"),
        torch.tensor([position], dtype=torch.float32, device="cuda"),
        torch.tensor([polarization], dtype=torch.float32, device="cuda"),
        torch.ones(1, device="cuda") if source else None,
    )


def adapter(scene, frequency, components):
    from witwin.radar.channel import ChannelPropagationAdapter, compile_scene

    return ChannelPropagationAdapter(
        compile_scene(scene, reference_frequency_hz=frequency),
        reference_frequency_hz=frequency,
        components=frozenset(components),
        max_depth=1,
    )


def material_export(directory):
    """Deembed actual reflected Channel fields with a matched image-space LOS."""
    directory.mkdir(parents=True, exist_ok=True)
    mesh = Mesh(
        vertices=torch.tensor([[-100, -100, 0], [100, -100, 0], [100, 100, 0], [-100, 100, 0]], dtype=torch.float32),
        faces=torch.tensor([[0, 1, 2], [0, 2, 3]]),
        recenter=False,
        fill_mode="surface",
        topology_diagnostics=False,
    )
    # Parameterized material surrogates, not measured or frequency-dispersive
    # databases for the named real-world materials. SI conductivity in S/m.
    materials = [
        ("low_loss_dielectric", 2.1, 0.001),
        ("glass_like", 6.31, 0.01),
        ("concrete_like", 5.24, 0.0462),
        ("wet_soil_like", 15.0, 1.0),
        ("copper_like", 1.0, 5.8e7),
    ]
    rows = []
    for frequency in (10e9, 24e9, 77e9):
        empty = adapter(Scene(structures=(), endpoints=[]), frequency, {"los"})
        for name, eps, sigma in materials:
            for thickness in (1000.0, 0.003):
                wall = Structure(
                    geometry=mesh,
                    material=PhysicalMaterial(name=name, eps_r=eps, sigma_e=sigma, thickness_m=thickness),
                    structure_id=1,
                    material_id=1,
                    assignment_id=1,
                    surface_id=1,
                )
                reflected = adapter(Scene(structures=(wall,), endpoints=[]), frequency, {"reflection"})
                for grazing in (5.0, 15.0, 30.0, 45.0, 60.0, 85.0):
                    angle = math.radians(grazing)
                    half_distance = np.float32(1 / math.tan(angle)).item()
                    actual_angle = math.atan2(1.0, half_distance)
                    for pol in ("H", "V"):
                        # Use p = s cross k on BOTH incident and reflected
                        # waves. Their vertical bases face opposite ways;
                        # a global z polarization is not MATLAB's signed TM basis.
                        source_pol = [0, 1, 0] if pol == "H" else [-math.sin(actual_angle), 0, -math.cos(actual_angle)]
                        sink_pol = [0, 1, 0] if pol == "H" else [math.sin(actual_angle), 0, -math.cos(actual_angle)]
                        tx = endpoint([-half_distance, 0.21, 1], source_pol, True)
                        rx = endpoint([half_distance, 0.21, 1], sink_pol, False)
                        image_tx = endpoint([-half_distance, 0.21, -1], sink_pol, True)
                        leg = reflected.reevaluate(reflected.freeze(tx, rx), tx, rx, ad_mode="none")
                        direct = empty.reevaluate(empty.freeze(image_tx, rx), image_tx, rx, ad_mode="none")
                        assert len(leg.delay_s) == len(direct.delay_s) == 1
                        assert bool(leg.row_valid.all()) and bool(direct.row_valid.all())
                        coefficient = complex((leg.coefficient / direct.coefficient).item())
                        rows.append(
                            {
                                "name": name,
                                "frequency": frequency,
                                "eps": eps,
                                "sigma": sigma,
                                "thickness": thickness,
                                "grazing": math.degrees(actual_angle),
                                "polarization": pol,
                                "witwin": coefficient,
                            }
                        )
                print(f"materials {frequency:g} {name} thickness={thickness}", flush=True)
    savemat(directory / "materials-input.mat", {key: np.array([row[key] for row in rows]) for key in rows[0]})


def timed(call, repeats=11, warmup=3):
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    durations = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        value = call()
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - start)
    return value, durations


def performance_export(directory):
    """Declared paths to beat IQ, with GPU resident and host-return timings."""
    directory.mkdir(parents=True, exist_ok=True)
    for samples, chirps, count in ((128, 64, 1), (512, 256, 1), (512, 256, 16), (512, 256, 64), (512, 256, 256)):
        name = f"n{samples}_c{chirps}_p{count}"
        fs, fc = float(2**24), 77e9
        period, slope = samples / fs, 10e6 / (samples / fs)
        tau_cpu = np.arange(1, count + 1, dtype=np.float32) / fs
        rate_cpu = np.zeros(count, dtype=np.float32)
        gains_cpu = np.full(count, 1 / math.sqrt(count), dtype=np.complex64)
        delay = torch.tensor(tau_cpu, device="cuda")
        rate = torch.tensor(rate_cpu, device="cuda")
        gains = torch.tensor(gains_cpu, device="cuda")
        offsets = torch.tensor([0, count], device="cuda")
        spec = FmcwSpec(samples, chirps, 1 / fs, period, slope, 0.0, fc, carrier_hz=fc, output_domain="beat")

        resident = partial(synthesize_fmcw_rows, delay, rate, gains, offsets, spec)

        def host_boundary(tau_cpu=tau_cpu, rate_cpu=rate_cpu, gains_cpu=gains_cpu, count=count, spec=spec):
            return synthesize_fmcw_rows(
                torch.tensor(tau_cpu, device="cuda"),
                torch.tensor(rate_cpu, device="cuda"),
                torch.tensor(gains_cpu, device="cuda"),
                torch.tensor([0, count], device="cuda"),
                spec,
            ).cpu()

        value, resident_seconds = timed(resident, repeats=31)
        _, host_seconds = timed(host_boundary, repeats=31)
        savemat(
            directory / f"{name}-performance-input.mat",
            {
                "fs": fs,
                "fc": fc,
                "samples": samples,
                "chirps": chirps,
                "period": period,
                "slope": slope,
                "lengths": tau_cpu.astype(float) * C0,
                "rates": rate_cpu.astype(float) * C0,
                "gains": gains_cpu.astype(complex),
                "witwin_iq": value[:, 0].T.cpu().numpy(),
                "resident_seconds": resident_seconds,
                "host_seconds": host_seconds,
            },
        )
        print(name, np.median(resident_seconds), np.median(host_seconds), flush=True)


class ExperimentMotion:
    def __init__(self, kind):
        self.kind = kind

    def at(self, t):
        w = 2 * math.pi * 80
        if self.kind == "static":
            points, speeds = [[30, 0, 0]], [[0, 0, 0]]
        elif self.kind == "acceleration":
            points, speeds = [[30 + 0.3 * t + 15 * t * t, 0, 0]], [[0.3 + 30 * t, 0, 0]]
        elif self.kind == "rotor":
            points = [[30 + 0.003 * math.cos(w * t), 0.003 * math.sin(w * t), 0]]
            speeds = [[-0.003 * w * math.sin(w * t), 0.003 * w * math.cos(w * t), 0]]
        else:
            points = [[30 + 0.002 * math.sin(w * t), 0.1, 0], [30.03 + 0.004 * math.sin(w * t / 2), -0.1, 0]]
            speeds = [[0.002 * w * math.cos(w * t), 0, 0], [0.002 * w * math.cos(w * t / 2), 0, 0]]
        return Kinematics(
            torch.tensor(points, dtype=torch.float32, device="cuda"),
            torch.tensor(speeds, dtype=torch.float32, device="cuda"),
        )


def motion_export(directory, cases, accuracy_only=False):
    from tools.validate_doppler_motion import make_radar
    from witwin.radar.simulation import AdaptiveMotionSpec, ScatterSitePolicy

    directory.mkdir(parents=True, exist_ok=True)
    for kind in cases:
        base_kind = kind.removesuffix("_os4")
        ground = kind.startswith("ground_")
        if ground:
            base_kind = "acceleration"
        factor = 4 if kind.endswith("_os4") else 1
        radar = make_radar()
        radar.system_config = replace(
            radar.system_config,
            waveform=replace(
                radar.system_config.waveform,
                adc_samples=512 if ground else 128 * factor,
                chirp_per_frame=512 if ground else 128 if base_kind == "static" else 1024,
                sample_rate=20000.0 if ground else 4000.0 * factor,
                slope=0.390625 if ground else 0.015625,
                adc_start_time=0.0,
                ramp_end_time=25.6 if ground else 32.0,
                idle_time=0.0,
                output_domain="beat",
            ),
            sensors=replace(radar.system_config.sensors, tx_power=TxPowerSpec(30.0)),
        )
        spec = radar.system_config.waveform_spec()
        trajectory = ExperimentMotion(base_kind)
        scene = Scene(structures=(), endpoints=[])
        if ground:
            eps, sigma = (5.24, 0.0462) if kind == "ground_concrete" else (1.0, 5.8e7)
            mesh = Mesh(
                vertices=torch.tensor(
                    [[-100, -30, -100], [-100, -30, 100], [100, -30, 100], [100, -30, -100]], dtype=torch.float32
                ),
                faces=torch.tensor([[0, 1, 2], [0, 2, 3]]),
                recenter=False,
                fill_mode="surface",
                topology_diagnostics=False,
            )
            wall = Structure(
                geometry=mesh,
                material=PhysicalMaterial(eps_r=eps, sigma_e=sigma, thickness_m=1000),
                structure_id=1,
                material_id=1,
                assignment_id=1,
                surface_id=1,
            )
            scene = Scene(structures=(wall,), endpoints=[])
        kwargs = {
            "times": (0.0,),
            "response": ScalarRcsResponse.from_rcs(1.0, reference_frequency_hz=77e9, device="cuda"),
            "sites": ScatterSitePolicy.explicit(
                trajectory.at(0).positions_m, trajectory=None if base_kind == "static" else trajectory
            ),
            "components": frozenset({"los", "reflection"}) if ground else frozenset({"los"}),
            "max_depth": 1 if ground else 0,
            "motion_sampling": "adaptive",
            "adaptive_motion": AdaptiveMotionSpec(phase_error_rad=0.02),
        }
        print("starting scene", kind, flush=True)
        result, seconds = timed(
            partial(radar.simulate, scene, **kwargs),
            repeats=1 if accuracy_only else 3,
            warmup=0 if accuracy_only else 1,
        )
        if ground:
            assert result.last_radar_paths.path_count == 4
        positions, velocities = [], []
        for chirp in range(spec.num_chirps):
            state = trajectory.at(chirp * spec.chirp_period_s)
            positions.append(state.positions_m.cpu().numpy())
            velocities.append(state.velocities_m_per_s.cpu().numpy())
        savemat(
            directory / f"{kind}-motion-input.mat",
            {
                "fs": 1 / spec.sample_period_s,
                "fc": spec.reference_frequency_hz,
                "samples": spec.num_samples,
                "chirps": spec.num_chirps,
                "period": spec.chirp_period_s,
                "slope": spec.slope_hz_per_s,
                "positions": positions,
                "velocities": velocities,
                "witwin_iq": result.cube[0, 0, 0].T.cpu().numpy(),
                "seconds": seconds,
                "accuracy_only": accuracy_only,
                "eps": eps if ground else 1.0,
                "sigma": sigma if ground else 0.0,
            },
        )
        (directory / f"{kind}-diagnostics.json").write_text(
            json.dumps(
                {
                    "seconds": seconds,
                    "discovery_count": result.discovery_count,
                    "path_count": result.last_radar_paths.path_count,
                    "adaptive": result.adaptive_diagnostics,
                },
                indent=2,
            )
            + "\n"
        )
        print("finished scene", kind, seconds, flush=True)


def timing_summary(values):
    values = np.atleast_1d(values)
    return {
        "repeats": len(values),
        "median_seconds": float(np.median(values)),
        "p95_seconds": float(np.percentile(values, 95)),
        "min_seconds": float(np.min(values)),
        "max_seconds": float(np.max(values)),
    }


def independent_free_space_oracle(source, kind):
    """Scalar far-field monostatic oracle: P=1 W, sigma=1 m², unit antenna gain.

    SI coordinates; positive dechirp phase; instantaneous round-trip delay.
    Independent NumPy geometry is sampled at every ADC time, with the same
    float32 authored position precision as the two scene interfaces.
    """
    u = np.arange(int(source["samples"]))[:, None] / source["fs"]
    t = u + np.arange(int(source["chirps"]))[None, :] * source["period"]
    w = 2 * np.pi * 80
    if kind == "static":
        points = [(np.full(t.shape, 30.0), np.zeros(t.shape))]
    elif kind == "acceleration":
        points = [(30 + 0.3 * t + 15 * t * t, np.zeros(t.shape))]
    elif kind == "rotor":
        points = [(30 + 0.003 * np.cos(w * t), 0.003 * np.sin(w * t))]
    else:
        points = [
            (30 + 0.002 * np.sin(w * t), np.full(t.shape, 0.1)),
            (30.03 + 0.004 * np.sin(w * t / 2), np.full(t.shape, -0.1)),
        ]
    value = np.zeros(t.shape, dtype=complex)
    for x, y in points:
        distance = np.sqrt(x.astype(np.float32).astype(float) ** 2 + y.astype(np.float32).astype(float) ** 2)
        tau = 2 * distance / C0
        amplitude = (C0 / source["fc"]) / ((4 * np.pi) ** 1.5 * distance**2)
        value += amplitude * np.exp(2j * np.pi * (source["fc"] * tau + source["slope"] * tau * (u - tau / 2)))
    return value


def independent_ground_oracle(source):
    """Four image-source paths, H Fresnel, 1 W/1 m²; exp(+j*w*t) convention.

    A smooth half-space lies 30 m below the monostatic radar and point target.
    Both mixed paths are retained coherently. This analytic oracle does not
    discover arbitrary mesh paths or model diffuse/extended-target scattering.
    """
    u = np.arange(int(source["samples"]))[:, None] / source["fs"]
    t = u + np.arange(int(source["chirps"]))[None, :] * source["period"]
    x = (30 + 0.3 * t + 15 * t * t).astype(np.float32).astype(float)
    distances = [x, np.sqrt(x**2 + 60**2)]
    sine = 60 / distances[1]
    eps0 = 8.8541878128e-12
    epsc = source["eps"] - 1j * source["sigma"] / (2 * np.pi * source["fc"] * eps0)
    root = np.sqrt(epsc - (1 - sine**2))
    rho = (sine - root) / (sine + root)
    value = np.zeros(t.shape, dtype=complex)
    amplitude = (C0 / source["fc"]) / (4 * np.pi) ** 1.5
    for a in (0, 1):
        for b in (0, 1):
            tau = (distances[a] + distances[b]) / C0
            weight = amplitude * rho.conj() ** (a + b) / (distances[a] * distances[b])
            value += weight * np.exp(2j * np.pi * (source["fc"] * tau + source["slope"] * tau * (u - tau / 2)))
    return value


def l2_norm(value, axis=None):
    # Explicit reduction avoids spawning a BLAS worker pool for strided IQ
    # views; metrics themselves must not contend with the timed GPU workload.
    return np.sqrt(np.sum(np.abs(value) ** 2, axis=axis))


def iq_metrics(reference, actual, guard):
    reference, actual = reference[guard:, 1:], actual[guard:, 1:]
    assert reference.shape == actual.shape
    scale = np.sum(actual.conj() * reference) / np.sum(abs(actual) ** 2)
    window = np.hanning(reference.shape[0])[:, None] * np.hanning(reference.shape[1])[None, :]
    powers = [np.abs(np.fft.fft2(iq * window)) ** 2 for iq in (reference, actual)]
    return {
        "raw_iq_relative_l2": float(l2_norm(reference - actual) / l2_norm(reference)),
        "aligned_iq_relative_l2": float(l2_norm(reference - scale * actual) / l2_norm(reference)),
        "global_alignment_real": float(scale.real),
        "global_alignment_imag": float(scale.imag),
        "rd_power_relative_l2": float(l2_norm(powers[0] - powers[1]) / l2_norm(powers[0])),
        "guard_samples": guard,
    }


def analyze(directory):
    report = {}
    source = loadmat(directory / "materials-input.mat", squeeze_me=True)
    result = loadmat(directory / "materials-matlab.mat", squeeze_me=True)
    difference = abs(source["witwin"] - result["matlab_coefficient"])
    material_rows = {}
    for name in np.unique(source["name"]):
        for thickness in (1000, 0.003):
            selected = (source["name"] == name) & (source["thickness"] == thickness)
            material_rows[f"{name.strip()}_{thickness:g}m"] = {
                "cases": int(selected.sum()),
                "max_complex_absolute_error": float(difference[selected].max()),
                "relative_l2": float(l2_norm(difference[selected]) / l2_norm(result["matlab_coefficient"][selected])),
                "reference": "actual Radar Toolbox Fresnel"
                if thickness == 1000
                else "Radar Toolbox interfaces plus independent Airy slab oracle",
            }
    report["materials"] = {
        "cases": len(difference),
        "maximum_absolute_error": float(difference.max()),
        "groups": material_rows,
    }
    report["path_to_iq_performance"] = {}
    for path in sorted(directory.glob("*-performance-input.mat")):
        source = loadmat(path, squeeze_me=True)
        result = loadmat(str(path).replace("-input.mat", "-matlab.mat"), squeeze_me=True)
        guard = int(np.ceil(max(np.atleast_1d(source["lengths"])) / C0 * source["fs"])) + 16
        metrics = iq_metrics(source["witwin_iq"], result["matlab_iq"], guard)
        metrics.update(
            gpu_resident=timing_summary(source["resident_seconds"]),
            gpu_with_host_transfers=timing_summary(source["host_seconds"]),
            matlab_cpu_double=timing_summary(result["seconds"]),
            matlab_output_dtype=str(result["output_class"]),
            witwin_output_dtype=str(source["witwin_iq"].dtype),
            matlab_over_gpu_host_ratio=float(np.median(result["seconds"]) / np.median(source["host_seconds"])),
        )
        report["path_to_iq_performance"][path.name.removesuffix("-performance-input.mat")] = metrics
    report["scene_motion"] = {}
    for path in sorted((directory / "motion").glob("*-motion-input.mat")):
        source = loadmat(path, squeeze_me=True)
        result = loadmat(str(path).replace("-input.mat", "-matlab.mat"), squeeze_me=True)
        kind = path.name.removesuffix("-motion-input.mat")
        guard = int(np.ceil((2 * l2_norm(source["positions"], axis=-1).max() / C0 + 0.8e-6) * source["fs"]))
        metrics = iq_metrics(source["witwin_iq"], result["matlab_iq"], guard)
        oracle = independent_free_space_oracle(source, kind.removesuffix("_os4"))[guard:, 1:]
        spectra = []
        for iq in (source["witwin_iq"], result["matlab_iq"]):
            slow = iq[guard:].mean(axis=0)
            _, _, spectrum = stft(
                slow,
                fs=1 / source["period"],
                nperseg=128,
                noverlap=96,
                nfft=512,
                return_onesided=False,
                boundary=None,
                padded=False,
            )
            spectra.append(abs(spectrum) ** 2)
        metrics.update(
            microdoppler_power_relative_l2=float(l2_norm(spectra[0] - spectra[1]) / l2_norm(spectra[0])),
            witwin_scene=timing_summary(source["seconds"]),
            matlab_scene=timing_summary(result["seconds"]),
            matlab_over_witwin_ratio=float(np.median(result["seconds"]) / np.median(source["seconds"])),
            witwin_vs_continuous_oracle_relative_l2=float(
                l2_norm(source["witwin_iq"][guard:, 1:] - oracle) / l2_norm(oracle)
            ),
            matlab_vs_continuous_oracle_relative_l2=float(
                l2_norm(result["matlab_iq"][guard:, 1:] - oracle) / l2_norm(oracle)
            ),
            accuracy_only=len(np.atleast_1d(source["seconds"])) < 3,
        )
        report["scene_motion"][kind] = metrics
    report["ground_motion"] = {}
    for path in sorted((directory / "ground").glob("*-motion-input.mat")):
        source = loadmat(path, squeeze_me=True)
        result = loadmat(str(path).replace("-input.mat", "-matlab.mat"), squeeze_me=True)
        metrics = iq_metrics(source["witwin_iq"], result["matlab_iq"], 25)
        oracle = independent_ground_oracle(source)[25:, 1:]
        metrics["witwin_vs_continuous_oracle_relative_l2"] = float(
            l2_norm(source["witwin_iq"][25:, 1:] - oracle) / l2_norm(oracle)
        )
        metrics["matlab_vs_continuous_oracle_relative_l2"] = float(
            l2_norm(result["matlab_iq"][25:, 1:] - oracle) / l2_norm(oracle)
        )
        metrics["matlab_internal_sample_rate_hz"] = float(result["internal_sample_rate"])
        metrics["scope"] = (
            "moving point + smooth half-space, four coherent round trips; MATLAB twoRayChannel composition"
        )
        metrics["accuracy_only"] = True
        report["ground_motion"][path.name.removesuffix("-motion-input.mat")] = metrics
    (directory / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def plot_results(directory):
    import matplotlib.pyplot as plt

    report = json.loads((directory / "comparison.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    keys = sorted(report["path_to_iq_performance"], key=lambda k: (int(k.split("_")[0][1:]), int(k.split("_p")[1])))
    x = np.arange(len(keys))
    for offset, field, title in (
        (-0.18, "gpu_with_host_transfers", "WiTwin CUDA complex64 + transfers"),
        (0.18, "matlab_cpu_double", "MATLAB CPU double"),
    ):
        values = [report["path_to_iq_performance"][k][field]["median_seconds"] * 1000 for k in keys]
        axes[0].bar(x + offset, values, width=0.36, label=title)
    axes[0].set_xticks(x, [k.replace("n", "N").replace("_c", "\nC").replace("_p", " / P") for k in keys])
    axes[0].set(yscale="log", ylabel="Median latency (ms, log scale)", title="Declared paths to beat IQ")
    axes[0].legend(fontsize=8)
    keys = ["static", "acceleration", "rotor", "limbs"]
    x = np.arange(len(keys))
    for offset, field, title in (
        (-0.18, "witwin_scene", "WiTwin public scene entry (CUDA)"),
        (0.18, "matlab_scene", "MATLAB point-target scene (CPU)"),
    ):
        values = [report["scene_motion"][k][field]["median_seconds"] for k in keys]
        axes[1].bar(x + offset, values, width=0.36, label=title)
    axes[1].set_xticks(x, keys)
    axes[1].set(yscale="log", ylabel="Median latency (s, log scale)", title="Trajectory + scene geometry to beat IQ")
    axes[1].legend(fontsize=8)
    fig.suptitle("RTX 5080 / Ryzen 7 9800X3D; shared desktop, different native precisions")
    fig.savefig(directory / "performance-comparison.png", dpi=180)
    plt.close(fig)

    path = directory / "motion"
    source = loadmat(path / "rotor-motion-input.mat", squeeze_me=True)
    actual = loadmat(path / "rotor-motion-matlab.mat", squeeze_me=True)
    control = loadmat(path / "rotor_os4-motion-matlab.mat", squeeze_me=True)
    inputs = [
        (source["witwin_iq"], 5, "WiTwin adaptive"),
        (actual["matlab_iq"], 5, "MATLAB baseline"),
        (control["matlab_iq"], 17, "MATLAB 4x sample rate"),
    ]
    spectra = []
    for iq, guard, _title in inputs:
        f, t, value = stft(
            iq[guard:].mean(axis=0),
            fs=1 / source["period"],
            nperseg=128,
            noverlap=96,
            nfft=512,
            return_onesided=False,
            boundary=None,
            padded=False,
        )
        spectra.append(np.fft.fftshift(abs(value) ** 2, axes=0))
    f = np.fft.fftshift(f)
    selected = abs(f) < 1500
    peak = spectra[0].max()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3), constrained_layout=True, sharex=True, sharey=True)
    for axis, power, (_, _, title) in zip(axes, spectra, inputs, strict=True):
        mesh = axis.pcolormesh(
            t * 1e3,
            f[selected],
            10 * np.log10(np.maximum(power[selected] / peak, 1e-8)),
            vmin=-40,
            vmax=0,
            shading="auto",
        )
        velocity = -0.003 * 2 * np.pi * 80 * np.sin(2 * np.pi * 80 * t)
        axis.plot(t * 1e3, 2 * source["fc"] * velocity / C0, color="white", linestyle="--", linewidth=1)
        axis.set(title=title, xlabel="Slow time (ms)")
    axes[0].set_ylabel("Beat Doppler (Hz; positive = receding)")
    fig.colorbar(mesh, ax=axes, label="dB relative to WiTwin peak")
    fig.suptitle("80 Hz rotating point, 3 mm radius; dashed: instantaneous theoretical Doppler")
    fig.savefig(directory / "microdoppler-comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("materials", "performance", "motion", "analyze", "plot"))
    parser.add_argument("--output", type=Path, default=Path("output/matlab-scenarios"))
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=(
            "static",
            "acceleration",
            "rotor",
            "limbs",
            "static_os4",
            "rotor_os4",
            "ground_concrete",
            "ground_metal",
        ),
        default=("static", "acceleration", "rotor", "limbs"),
    )
    parser.add_argument("--accuracy-only", action="store_true")
    args = parser.parse_args()
    if args.action == "plot":
        plot_results(args.output)
    elif args.action == "analyze":
        analyze(args.output)
    elif args.action == "motion":
        motion_export(args.output, args.cases, args.accuracy_only)
    else:
        (material_export if args.action == "materials" else performance_export)(args.output)
