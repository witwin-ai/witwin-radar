"""Three point targets, two of them moving, from a range-Doppler map to a point cloud.

    PointTargets(trajectory=...)  ->  Radar.simulate  ->  Result
                                  ->  Frame.range_doppler / Frame.points

The world is empty (no structures), so the only rows are the direct round trips
to the three targets:

* target A, on the boresight at 3 m, closing at 2 m/s;
* target B, 5 m out and 20 degrees to the right, receding at 1.5 m/s along its
  line of sight;
* target C, 4 m out and 15 degrees to the left, holding still.

A target moves only if it has a ``trajectory=``: a callable from time (s) to
positions ``[N, 3]`` (m, world frame) on the radar's device. Without one it is
static and has zero Doppler.

Because something in the scene moves and the waveform is FMCW, the default
``Motion.auto()`` resolves to ``Motion.adaptive()``: the world is evaluated at a
few probe instants per frame and the echo between them is interpolated within a
tested phase and amplitude tolerance (0.02 rad and 2 % by default).
``Result.adaptive_diagnostics`` records what the frame spent. The last part of
the example runs one short frame both ways, adaptively and with
``Motion.adc()`` - the exhaustive reference that evaluates the world at every
ADC sample - and checks that the two cubes agree within that tolerance.
``Motion.adc()`` costs one world evaluation per transmitter, chirp and sample,
which is why the comparison frame is kept to 4 chirps of 32 samples.

Velocities are positive for a CLOSING target, in the range-Doppler map and in
the point cloud alike. The point cloud is in the radar processing frame:
``x`` to the radar's right, ``y`` along the boresight, ``z`` up.

What is asserted: every target is detected within one range bin, one velocity
bin and 4 degrees of azimuth of the truth; the strongest range-Doppler cell of
A sits in A's bins; the scene resolves to adaptive sampling; the adaptive and
exhaustive cubes agree to within the adaptive tolerance and peak in the same
cell.

Usage:
    python -m examples.moving_target
    python examples/moving_target.py
"""

from __future__ import annotations

import dataclasses
import math
import pathlib
import sys

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from witwin.core import Scene  # noqa: E402

from witwin.radar import Fmcw, Motion, Noise, PointTargets, Radar  # noqa: E402

#: 77 GHz FMCW, 60 MHz/us, 256 complex samples at 4.4 MHz, 64 chirps per
#: transmitter. Times are in seconds.
CHIRP = Fmcw(
    slope=60e12,
    sample_rate=4.4e6,
    samples_per_chirp=256,
    chirps_per_frame=64,
    adc_start=6e-6,
    idle=7e-6,
    ramp_end=65e-6,
)

#: Two transmitters four half-wavelengths apart and four receivers one
#: half-wavelength apart: a uniform 8-element virtual array along the radar's
#: right axis, time-division multiplexed (one transmitter per chirp slot).
TX = ((0.0, 0.0, 0.0), (4.0, 0.0, 0.0))
RX = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0))


@dataclasses.dataclass(frozen=True)
class Target:
    """One point target: where it starts and how fast it closes.

    ``range_m`` and ``azimuth_deg`` place the target in the horizontal plane of
    the radar (azimuth positive to the radar's right, which is world ``+x``
    for the default pose looking along ``-z``). ``closing_mps`` is its speed
    along the line of sight, positive towards the radar.
    """

    name: str
    range_m: float
    azimuth_deg: float
    closing_mps: float
    rcs_m2: float

    def direction(self) -> tuple[float, float, float]:
        azimuth = math.radians(self.azimuth_deg)
        return (math.sin(azimuth), 0.0, -math.cos(azimuth))


TARGETS = (
    Target("A", range_m=3.0, azimuth_deg=0.0, closing_mps=2.0, rcs_m2=0.1),
    Target("B", range_m=5.0, azimuth_deg=20.0, closing_mps=-1.5, rcs_m2=0.1),
    Target("C", range_m=4.0, azimuth_deg=-15.0, closing_mps=0.0, rcs_m2=0.1),
)


def build_radar() -> Radar:
    """The front end at the origin, looking along ``-z``, with a thermal-noise receiver.

    Without receiver noise the map has no noise floor and the CFAR detector
    reports range and Doppler sidelobes as targets.
    """

    return Radar(
        reference_frequency_hz=77e9,
        waveform=CHIRP,
        tx=TX,
        rx=RX,
        power=15.0,
        antenna_unit="half_wavelength",
        noise=Noise(figure=10.0),
        seed=20261113,
    )


def build_targets(device: torch.device) -> PointTargets:
    """All three targets in one record, each on a straight line at constant speed."""

    unit = torch.tensor([target.direction() for target in TARGETS], dtype=torch.float32, device=device)
    start = unit * torch.tensor([[target.range_m] for target in TARGETS], dtype=torch.float32, device=device)
    # A closing target moves against its outward line-of-sight unit vector.
    velocity = -unit * torch.tensor([[target.closing_mps] for target in TARGETS], dtype=torch.float32, device=device)

    def trajectory(time_s: float) -> torch.Tensor:
        return start + velocity * time_s

    rcs = torch.tensor([target.rcs_m2 for target in TARGETS], dtype=torch.float32, device=device)
    return PointTargets(positions=start, rcs=rcs, trajectory=trajectory)


def check_point_cloud(cloud, axes) -> None:
    """Every target has a detection within one range bin, one velocity bin and 4 degrees."""

    xyz = cloud.xyz.cpu()
    ranges = cloud.range_m.cpu()
    velocities = cloud.velocity_mps.cpu()
    azimuths = torch.rad2deg(torch.atan2(xyz[:, 0], xyz[:, 1]))
    for target in TARGETS:
        hits = (
            ((ranges - target.range_m).abs() <= axes.range_bin_m)
            & ((velocities - target.closing_mps).abs() <= axes.velocity_bin_mps)
            & ((azimuths - target.azimuth_deg).abs() <= 4.0)
        )
        assert bool(hits.any()), (
            f"no detection of {target.name} ({target.range_m} m, {target.closing_mps:+} m/s, "
            f"{target.azimuth_deg:+} deg) in the cloud:\n"
            + "\n".join(
                f"  {float(r):.3f} m {float(v):+.3f} m/s {float(a):+.1f} deg"
                for r, v, a in zip(ranges, velocities, azimuths, strict=True)
            )
        )
        index = int(torch.nonzero(hits)[0])
        print(
            f"  {target.name}: {float(ranges[index]):.3f} m, {float(velocities[index]):+.3f} m/s, "
            f"{float(azimuths[index]):+.1f} deg  (truth {target.range_m:.3f} m, "
            f"{target.closing_mps:+.3f} m/s, {target.azimuth_deg:+.1f} deg)  OK"
        )


def compare_with_exhaustive(radar: Radar, targets: PointTargets) -> None:
    """One short, noiseless frame under ``Motion.auto()`` and ``Motion.adc()``."""

    short = radar.replace(waveform=dataclasses.replace(CHIRP, chirps_per_frame=4, samples_per_chirp=32), noise=None)
    adaptive = short.simulate(Scene(), targets, times=(0.0,), reflections=0)
    exhaustive = short.simulate(Scene(), targets, times=(0.0,), reflections=0, motion=Motion.adc())
    assert adaptive.motion_sampling == "adaptive" and exhaustive.motion_sampling == "adc"

    tolerance = Motion.adaptive()
    bound = max(tolerance.phase_error, tolerance.relative_amplitude_error)
    relative = float((adaptive.cube - exhaustive.cube).norm() / exhaustive.cube.norm())
    assert relative <= bound, f"adaptive and exhaustive cubes differ by {relative:.3e} (bound {bound})"

    peaks = [int(result.cube[0].abs().sum(dim=(0, 1, 2)).argmax()) for result in (adaptive, exhaustive)]
    assert peaks[0] == peaks[1], f"the adaptive cube peaks in range bin {peaks[0]}, the exhaustive one in {peaks[1]}"
    evaluations = adaptive.adaptive_diagnostics[0]["evaluations"]
    observations = adaptive.adaptive_diagnostics[0]["observation_count"]
    print(
        f"  {observations} observations: adaptive {evaluations} world evaluations, adc {observations}; "
        f"relative L2 difference {relative:.2e} (bound {bound}), same peak bin {peaks[0]}  OK"
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires CUDA: the simulation runs in native CUDA kernels only.")

    radar = build_radar()
    targets = build_targets(radar.device)

    print(f"Using device={radar.device}")
    print("Simulating one frame...")
    result = radar.simulate(Scene(), targets, times=(0.0,), reflections=0)
    assert result.motion_sampling == "adaptive", result.motion_sampling
    diagnostics = result.adaptive_diagnostics[0]
    print(
        f"  Motion.auto() -> {result.motion_sampling}: {diagnostics['evaluations']} world evaluations for "
        f"{diagnostics['observation_count']} observations, {diagnostics['accepted_intervals']} accepted intervals, "
        f"max tested phase error {diagnostics['max_tested_phase_error_rad']:.1e} rad"
    )

    frame = result.frame(0)
    axes = frame.axes
    print(f"  Range bin {axes.range_bin_m:.4f} m, velocity bin {axes.velocity_bin_mps:.4f} m/s")

    # Coherent sum over the eight virtual channels, then the strongest cell.
    rd = frame.range_doppler(window="hann")
    power = rd.data.sum(dim=(0, 1)).abs().square()
    doppler_bin, range_bin = divmod(int(power.argmax()), int(power.shape[1]))
    peak_range_m = float(axes.range_m[range_bin])
    peak_velocity_mps = float(axes.velocity_mps[doppler_bin])
    strongest = TARGETS[0]
    assert abs(peak_range_m - strongest.range_m) <= axes.range_bin_m, peak_range_m
    assert abs(peak_velocity_mps - strongest.closing_mps) <= axes.velocity_bin_mps, peak_velocity_mps
    print(f"  Strongest range-Doppler cell: {peak_range_m:.3f} m, {peak_velocity_mps:+.3f} m/s (target A)  OK")

    # The FMCW cube is already a range spectrum, so its range transform is
    # rectangular and a strong target has range sidelobes 20-45 dB down along
    # its Doppler row. At a false-alarm rate of 1e-6 the CA-CFAR threshold
    # stays above them; at 1e-4 the strongest target's sidelobes and a noise
    # cell or two join the cloud.
    cloud = frame.points(pfa=1e-6, max_points=32)
    print(f"Point cloud: {len(cloud)} points")
    check_point_cloud(cloud, axes)

    print("Adaptive against exhaustive sampling on a 4-chirp frame...")
    compare_with_exhaustive(radar, targets)
    print("PASSED")


if __name__ == "__main__":
    main()
