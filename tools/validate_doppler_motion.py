"""Reproducible physical-motion experiments; all equations here are test oracles.

Run in witwin2 with the same Channel runtime as the integration suite:
    python tools/validate_doppler_motion.py --output output/doppler-repair/experiments
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
from witwin.radar.processing import SlowTimeSignal, microdoppler_spectrogram
from witwin.radar.propagation import Kinematics
from witwin.radar.synthesis.assembly import BEAT_PHASOR, FmcwSpec
from witwin.radar.synthesis.fmcw import synthesize_fmcw_rows

C0 = 299792458.0


def make_radar():
    # Isotropic elements and a world-frame ``+z`` polarization keep the oracles
    # below exact: every fixture moves in the ``z = 0`` plane, so the field
    # projection is one and the only amplitude law left is ``1 / R^2``.
    return Radar.from_dict(
        {
            "num_tx": 1,
            "num_rx": 1,
            "fc": 77e9,
            "slope": 60.0,
            "adc_samples": 4,
            "adc_start_time": 6,
            "sample_rate": 5000,
            "idle_time": 1942,
            "ramp_end_time": 58,
            "chirp_per_frame": 128,
            "power": 12,
            "tx_loc": [[0, 0, 0]],
            "rx_loc": [[0, 0, 0]],
        },
        pattern=Pattern.isotropic(),
        polarization=(0, 0, 1),
        position=(0, 0, 0),
        look_at=(1, 0, 0),
    )


class Trajectory:
    """Material-point motion for one fixture, in metres and m/s.

    ``positions`` is what ``PointTargets.trajectory`` takes; ``at`` additionally
    publishes the analytic velocity, which only the oracles below read. They are
    two views of one closed-form path, so the oracle can never drift from the
    motion the simulator was handed.
    """

    def __init__(self, kind, device):
        self.kind, self.device = kind, device

    def positions(self, t):
        return self.at(t).positions_m

    def at(self, t):
        w = 2 * math.pi * 8
        if self.kind == "radial":
            points, velocity = [[2 + 0.2 * t, 0, 0]], [[0.2, 0, 0]]
        elif self.kind == "rotor":
            radius = 0.003
            points = [[2 + radius * math.cos(w * t), radius * math.sin(w * t), 0]]
            velocity = [[-radius * w * math.sin(w * t), radius * w * math.cos(w * t), 0]]
        else:
            points = [[2 + 0.002 * math.sin(w * t), 0.1, 0], [2.3 + 0.004 * math.sin(w * t / 2), -0.1, 0]]
            velocity = [[0.002 * w * math.cos(w * t), 0, 0], [0.002 * w * math.cos(w * t / 2), 0, 0]]
        return Kinematics(torch.tensor(points, device=self.device), torch.tensor(velocity, device=self.device))


def scene_experiment(kind):
    radar = make_radar()
    trajectory = Trajectory(kind, radar.device)
    targets = PointTargets(positions=trajectory.positions(0), amplitude=1.0, trajectory=trajectory.positions)
    start = time.perf_counter()
    # Per-ADC sampling is named rather than left to ``Motion.auto()``: the
    # oracle below evaluates the closed form at every observation instant, so
    # the run it is compared against must be the exhaustive one.
    result = radar.simulate(
        Scene(structures=(), endpoints=[]), targets, times=(0.0,), los=True, reflections=0, motion=Motion.adc()
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    spec = radar.waveform_spec()
    beat = torch.fft.ifft(result.cube[0, 0, 0], dim=-1, norm="forward").to(torch.complex128)
    expected, theoretical = [], []
    # Independent free-space two-way oracle. All points lie in the polarization
    # transverse plane: field projection is one and amplitude is proportional to 1/R^2.
    for index, t in enumerate(result.sample_times_s[0]):
        state = trajectory.at(t)
        p, v = state.positions_m.double(), state.velocities_m_per_s.double()
        distance = p.norm(dim=-1)
        delay = 2 * distance / C0
        u = spec.t_start_s + (index % spec.num_samples) * spec.sample_period_s
        phase = 2 * math.pi * (spec.reference_frequency_hz * delay + spec.slope_hz_per_s * delay * (u - delay / 2))
        expected.append((torch.exp(1j * phase) / distance.square()).sum())
        theoretical.append(-2 * spec.reference_frequency_hz / C0 * (p * v).sum(-1) / distance)
    expected = torch.stack(expected).reshape_as(beat)
    # Remove only the constant response/power factor, calibrated at one observation.
    expected *= beat[0, 0] / expected[0, 0]
    relative_rms = float((beat - expected).abs().square().mean().sqrt() / expected.abs().square().mean().sqrt())
    assert relative_rms < 2e-3, (kind, relative_rms)
    times = tuple(result.sample_times_s[0][:: spec.num_samples])
    centres, frequencies, spectrum = microdoppler_spectrogram(
        SlowTimeSignal(beat[:, 0], times, BEAT_PHASOR), window_slots=32, hop_slots=8
    )
    _, _, reference = microdoppler_spectrogram(
        SlowTimeSignal(expected[:, 0], times, BEAT_PHASOR), window_slots=32, hop_slots=8
    )
    spectrum_error = float((spectrum - reference).abs().norm() / reference.abs().norm())
    assert spectrum_error < 2e-3
    metrics = {
        "relative_iq_rms_error": relative_rms,
        "relative_stft_error": spectrum_error,
        "seconds": elapsed,
        "observations": len(result.sample_times_s[0]),
        "discoveries": result.discovery_count,
        "motion_sampling": result.motion_sampling,
        "path_set_complete": result.path_set_complete,
    }
    if kind == "radial":
        peak = float(frequencies[spectrum.abs().mean(0).argmax()])
        metrics.update(peak_hz=peak, expected_hz=-2 * spec.reference_frequency_hz * 0.2 / C0)
        assert abs(peak - metrics["expected_hz"]) < 1 / (32 * spec.chirp_period_s)
    return metrics, (
        centres.cpu(),
        frequencies.cpu(),
        spectrum.abs().cpu(),
        times,
        torch.stack(theoretical)[:: spec.num_samples].cpu(),
    )


def continuous_chirp():
    spec = FmcwSpec(256, 1, 1 / 5e6, 60e-6, 6e13, 6e-6, 77e9, carrier_rate_hz=77e9, output_domain="beat")
    tau = torch.tensor([2 * 3.7 / C0], device="cuda")
    rate = torch.tensor([2 * 2.0 / C0], device="cuda")
    out = (
        synthesize_fmcw_rows(
            tau,
            rate,
            torch.ones(1, device="cuda", dtype=torch.complex64),
            torch.tensor([0, 1], device="cuda", dtype=torch.int64),
            spec,
        )[0, 0]
        .cpu()
        .to(torch.complex128)
    )
    u = spec.t_start_s + torch.arange(spec.num_samples).double() * spec.sample_period_s
    delay = float(tau[0]) + float(rate[0]) * u
    oracle = torch.exp(
        2j
        * math.pi
        * (spec.reference_frequency_hz * float(rate[0]) * u + spec.slope_hz_per_s * delay * (u - delay / 2))
    )

    def frequency(signal):
        return float(torch.angle(signal[1:] * signal[:-1].conj()).mean()) / (2 * math.pi * spec.sample_period_s)

    error = float((out - oracle).abs().max())
    assert error < 2e-6
    return {"measured_fast_time_hz": frequency(out), "oracle_fast_time_hz": frequency(oracle), "max_iq_error": error}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/doppler-repair/experiments"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = {"continuous_chirp": continuous_chirp()}
    plots = []
    for kind in ("radial", "rotor", "two_limb_proxy"):
        results[kind], data = scene_experiment(kind)
        plots.append((kind, data))
        print(kind, json.dumps(results[kind]), flush=True)
    (args.output / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 1, figsize=(10, 10), constrained_layout=True)
    for axis, (kind, (times, frequencies, magnitude, truth_times, truth)) in zip(axes, plots, strict=True):
        db = 20 * torch.log10((magnitude / magnitude.max()).clamp_min(1e-5))
        axis.pcolormesh(times.numpy(), frequencies.numpy(), db.T.numpy(), shading="auto", vmin=-60, vmax=0)
        axis.plot(truth_times, truth, color="white", linewidth=1, alpha=0.8)
        axis.set(title=kind, ylabel="Physical Doppler (Hz)", ylim=(-250, 250))
    axes[-1].set_xlabel("Time (s); white = analytic material-point Doppler")
    figure.savefig(args.output / "microdoppler.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
