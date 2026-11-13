"""A walking person and its micro-Doppler spectrogram.

    articulated body (PointTargets with a trajectory)  ->  Radar.simulate
        ->  Frame.range_profile per frame  ->  SlowTimeSignal
        ->  processing.microdoppler_spectrogram

The body is an articulated stand-in that needs no model files: ten point
scatterers (torso, head, and per side a thigh, a shin, a foot and a forearm)
moved by a sinusoidal gait. The hips swing +/-20 degrees, the knee flexes most in
the swing phase, the arms swing against the legs and the torso bobs 2 cm at
twice the stride frequency, while the whole body walks straight at the radar at
1.2 m/s from 5 m. It is a kinematic caricature, not a biomechanical model, but
it has what a micro-Doppler signature is made of: a torso line at the walking
speed and limbs that swing faster and slower than it.

The world is Y-up with the floor at ``y = 0``. The radar stands 1 m above the
floor at the origin and looks along ``-z``; the person walks along ``+z``
towards it. Positive velocity means CLOSING.

Slow time is uniform across frames only if the frames are back to back, so the
frame instants are multiples of one frame's duration (128 chirps of 200 us).
The 200 us chirp period sets the unambiguous velocity to
``lambda / (4 T) = 4.9 m/s``, above the fastest foot. Every frame is moving FMCW,
so ``Motion.auto()`` resolves to adaptive sampling.

What is asserted: the strongest line of the spectrogram is the torso, within two
velocity bins of the walking speed; the limbs spread the signature at least
1 m/s above and 0.8 m/s below the torso at -20 dB; and no energy above -20 dB
lies outside the range of radial velocities the body actually had (from the
trajectory itself), widened by the window's main lobe.

SMPL body (optional, untested here). ``--smpl-model PATH`` replaces the stand-in
with an SMPL mesh posed by the same gait angles and scattering as a skin
dielectric through ``MeshTargets``. It needs the ``smpl`` extra
(``smplpytorch``) and the SMPL model files in ``PATH`` (the ``model_root``
``witwin.radar.smpl.SMPLBody`` passes to smplpytorch). Neither ships with this
repository, so that route has not been run in this repository's checks; the
assertions above are made only for the stand-in.

Usage:
    python -m examples.walking_human
    python examples/walking_human.py
    python examples/walking_human.py --plot spectrogram.png
    python examples/walking_human.py --smpl-model path/to/smpl/models
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from witwin.core import Scene  # noqa: E402

from witwin.radar import Fmcw, Noise, PointTargets, Radar  # noqa: E402
from witwin.radar.processing import SlowTimeSignal, microdoppler_spectrogram  # noqa: E402

#: 64 complex samples per chirp (0.17 m range bins, 11 m span) and a 200 us
#: chirp period: 65 us ramp plus 135 us idle.
CHIRP = Fmcw(
    slope=60e12,
    sample_rate=4.4e6,
    samples_per_chirp=64,
    chirps_per_frame=128,
    adc_start=6e-6,
    idle=135e-6,
    ramp_end=65e-6,
)
RADAR_HEIGHT_M = 1.0

WALK_SPEED_MPS = 1.2
#: One gait cycle (two steps) covers this distance, so the stride frequency
#: is ``WALK_SPEED_MPS / STRIDE_LENGTH_M`` = 0.92 Hz.
STRIDE_LENGTH_M = 1.3
START_DISTANCE_M = 5.0
#: One gait cycle; the run is this rounded up to whole frames.
DURATION_S = STRIDE_LENGTH_M / WALK_SPEED_MPS

#: The STFT: 128 slow-time samples (25.6 ms, 0.076 m/s bins) every 32.
WINDOW_SLOTS = 128
HOP_SLOTS = 32
#: The level, relative to the spectrogram's peak, that counts as signature.
ENVELOPE_DB = -20.0

#: Segment lengths, m, and the per-part radar cross sections, m^2.
THIGH_M, SHIN_M, UPPER_ARM_M, FOREARM_M = 0.45, 0.45, 0.30, 0.30
HIP_HEIGHT_M, SHOULDER_HEIGHT_M = 0.95, 1.45
LIMBS = (("thigh", 0.06), ("shin", 0.04), ("foot", 0.02), ("forearm", 0.02))
PARTS = (
    ("torso", 0.5),
    ("head", 0.05),
    *((f"{part} {side}", rcs) for side in ("left", "right") for part, rcs in LIMBS),
)


def gait(time_s: float) -> dict[str, float]:
    """Joint angles (rad) and the body's position along the walk.

    Hip and shoulder angles are measured from hanging straight down, positive
    when the limb swings forward (towards ``+z``); knee and elbow angles are
    flexions, the knee bending the shin back and the elbow the forearm forward.
    The left leg leads by half a cycle on the right one; each arm swings against
    the leg on its side. ``pelvis_z_m`` is the body's position along the walk
    (it starts ``START_DISTANCE_M`` down the boresight) and ``bob_m`` its
    vertical offset.
    """

    phase = 2.0 * math.pi * WALK_SPEED_MPS / STRIDE_LENGTH_M * time_s
    angles = {"pelvis_z_m": -START_DISTANCE_M + WALK_SPEED_MPS * time_s, "bob_m": 0.02 * math.sin(2.0 * phase)}
    for side, offset in (("left", 0.0), ("right", math.pi)):
        leg = phase + offset
        angles[f"hip {side}"] = 0.35 * math.sin(leg)
        # The knee is straightest at mid-stance and most bent at mid-swing.
        angles[f"knee {side}"] = 0.3 * (1.0 + math.cos(leg))
        angles[f"shoulder {side}"] = -0.3 * math.sin(leg)
        angles[f"elbow {side}"] = 0.3
    return angles


class ArticulatedWalker:
    """Ten point scatterers moved by :func:`gait`, as ``PointTargets.trajectory`` takes them."""

    def __init__(self, device: torch.device) -> None:
        self.device = device

    def positions(self, time_s: float) -> torch.Tensor:
        """``[10, 3]`` world positions (m) at ``time_s``, in :data:`PARTS` order."""

        angles = gait(time_s)
        z0, bob = angles["pelvis_z_m"], angles["bob_m"]
        points = [(0.0, 1.25 + bob, z0), (0.0, 1.65 + bob, z0)]
        # Facing +z with +y up, the person's left is world +x.
        for side, x in (("left", 0.1), ("right", -0.1)):
            hip, knee = angles[f"hip {side}"], angles[f"hip {side}"] - angles[f"knee {side}"]
            hip_y = HIP_HEIGHT_M + bob
            knee_y, knee_z = hip_y - THIGH_M * math.cos(hip), z0 + THIGH_M * math.sin(hip)
            ankle_y, ankle_z = knee_y - SHIN_M * math.cos(knee), knee_z + SHIN_M * math.sin(knee)
            shoulder, forearm = angles[f"shoulder {side}"], angles[f"shoulder {side}"] + angles[f"elbow {side}"]
            elbow_y = SHOULDER_HEIGHT_M + bob - UPPER_ARM_M * math.cos(shoulder)
            elbow_z = z0 + UPPER_ARM_M * math.sin(shoulder)
            wrist_y, wrist_z = elbow_y - FOREARM_M * math.cos(forearm), elbow_z + FOREARM_M * math.sin(forearm)
            points += [
                (x, 0.5 * (hip_y + knee_y), 0.5 * (z0 + knee_z)),
                (x, 0.5 * (knee_y + ankle_y), 0.5 * (knee_z + ankle_z)),
                (x, ankle_y, ankle_z + 0.05),
                (2.0 * x, 0.5 * (elbow_y + wrist_y), 0.5 * (elbow_z + wrist_z)),
            ]
        return torch.tensor(points, dtype=torch.float32, device=self.device)

    def targets(self) -> PointTargets:
        rcs = torch.tensor([rcs for _, rcs in PARTS], dtype=torch.float32, device=self.device)
        return PointTargets(positions=self.positions(0.0), rcs=rcs, trajectory=self.positions)

    def closing_speeds(self, times_s, *, step_s: float = 1e-4) -> torch.Tensor:
        """``[times, parts]`` radial speed towards the radar (m/s), by central difference."""

        radar = torch.tensor((0.0, RADAR_HEIGHT_M, 0.0), device=self.device)
        speeds = []
        for time_s in times_s:
            before = (self.positions(time_s - step_s) - radar).norm(dim=-1)
            after = (self.positions(time_s + step_s) - radar).norm(dim=-1)
            speeds.append((before - after) / (2.0 * step_s))
        return torch.stack(speeds)


def smpl_scene(model_root: pathlib.Path, device: torch.device, max_sites: int):
    """The SMPL route: ``(DynamicScene, MeshTargets)`` for a body posed by :func:`gait`.

    Not run against real SMPL files in this repository (neither the model files
    nor smplpytorch are available to its checks). SMPL's rest frame is Y-up and faces ``+z``, which is this world's walking
    direction, and its root joint is at the origin; the body is lifted so its
    feet touch ``y = 0`` and moved along ``z`` by the gait. Joint indices are
    SMPL's: 1/2 hips, 4/5 knees, 16/17 shoulders, 18/19 elbows.
    """

    from witwin.core import Mesh, PhysicalMaterial, Structure
    from witwin.core.dynamics import DeformationState, DynamicScene

    from witwin.radar import MeshTargets
    from witwin.radar.smpl import SMPLBody

    body = SMPLBody(
        pose=torch.zeros(72), shape=torch.zeros(10), gender="neutral", model_root=str(model_root), device=device
    )

    def pose(time_s: float) -> torch.Tensor:
        angles = gait(time_s)
        joints = torch.zeros(24, 3)
        for side, hip, knee, shoulder, elbow, down in (("left", 1, 4, 16, 18, -1.3), ("right", 2, 5, 17, 19, 1.3)):
            # A rotation about +x swings a hanging limb backwards, so a forward
            # swing is a negative angle; the arms are lowered from SMPL's T pose
            # about z. Summing the two axis-angle components is a small-angle
            # approximation of composing them.
            joints[hip, 0] = -angles[f"hip {side}"]
            joints[knee, 0] = angles[f"knee {side}"]
            joints[shoulder] = torch.tensor((-angles[f"shoulder {side}"], 0.0, down))
            joints[elbow, 1] = math.copysign(angles[f"elbow {side}"], down)
        return joints.reshape(72).to(device)

    rest, faces = body.updated(pose=pose(0.0)).to_mesh()
    lift = -float(rest[:, 1].min())

    class Deformation:
        """Absolute world vertices of the posed body; the rest mesh sits at the identity."""

        def vertices_at(self, time_s: float) -> torch.Tensor:
            angles = gait(time_s)
            vertices, _ = body.updated(pose=pose(time_s)).to_mesh()
            offset = torch.tensor((0.0, lift + angles["bob_m"], angles["pelvis_z_m"]), device=vertices.device)
            return vertices + offset

        def at(self, time_s: float) -> DeformationState:
            return DeformationState(vertices=self.vertices_at(float(time_s)))

    deformation = Deformation()
    # ``recenter=False``: a recentred mesh would be moved to its bounding-box
    # centre, and the deformed vertices with it, silently.
    mesh = Mesh(
        vertices=deformation.vertices_at(0.0),
        faces=faces.to(torch.int64),
        recenter=False,
        fill_mode="surface",
        topology_diagnostics=False,
    )
    # Wet skin near 77 GHz (relative permittivity 6.4, conductivity 42 S/m); a
    # 1 cm layer of it is opaque at the carrier.
    skin = PhysicalMaterial(name="skin", eps_r=6.4, sigma_e=42.0, thickness_m=0.01)
    structure = Structure(geometry=mesh, material=skin, structure_id=1, material_id=1, assignment_id=1, surface_id=1)
    scene = DynamicScene(Scene(structures=(structure,), endpoints=[]), structure_deformations={1: deformation})
    return scene, MeshTargets(structure_ids=(1,), max_sites=max_sites)


def build_radar() -> Radar:
    """One transmitter and one receiver 1 m above the floor, looking along ``-z``."""

    return Radar(
        reference_frequency_hz=77e9,
        waveform=CHIRP,
        tx=((0.0, 0.0, 0.0),),
        rx=((0.0, 0.0, 0.0),),
        power=15.0,
        noise=Noise(figure=10.0),
        seed=20261113,
        position=(0.0, RADAR_HEIGHT_M, 0.0),
        look_at=(0.0, RADAR_HEIGHT_M, -1.0),
    )


def slow_time_signal(result, gate: torch.Tensor) -> SlowTimeSignal:
    """The frames' range profiles in the gated bins, back to back along slow time.

    The slow-time instants are each chirp's first ADC instant, read off
    ``Result.sample_times``; ``SlowTimeSignal`` refuses them unless they are
    uniform, so a gap between frames cannot pass silently.
    """

    chirps = CHIRP.chirps_per_frame
    samples, times = [], []
    for index in range(result.frame_count):
        profile = result.frame(index).range_profile()  # [tx, rx, chirp, range]
        samples.append(profile.data[0, 0][:, gate])
        times.extend(result.sample_times[index].reshape(chirps, -1)[:, 0].tolist())
    return SlowTimeSignal(
        torch.cat(samples).T.contiguous(), tuple(times), result.phasor, wavelength_m=result.axes.wavelength_m
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plot", type=pathlib.Path, default=None, help="Save the spectrogram as a PNG here.")
    parser.add_argument(
        "--smpl-model", type=pathlib.Path, default=None, help="SMPL model directory; selects the SMPL body (untested)."
    )
    parser.add_argument("--smpl-sites", type=int, default=256, help="Scatter-site budget of the SMPL mesh.")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires CUDA: the simulation runs in native CUDA kernels only.")

    radar = build_radar()
    walker = ArticulatedWalker(radar.device)
    if args.smpl_model is None:
        scene, targets = Scene(), walker.targets()
        print(f"Body: articulated stand-in, {len(PARTS)} point scatterers")
    else:
        scene, targets = smpl_scene(args.smpl_model, radar.device, args.smpl_sites)
        print(f"Body: SMPL mesh from {args.smpl_model}, at most {args.smpl_sites} scatter sites")

    frame_s = CHIRP.chirps_per_frame * CHIRP.chirp_period * radar.num_tx
    frames = math.ceil(DURATION_S / frame_s)
    times = tuple(index * frame_s for index in range(frames))
    print(f"Simulating {frames} back-to-back frames ({frames * frame_s:.2f} s, one gait cycle)...")
    result = radar.simulate(scene, targets, times=times, reflections=0)
    evaluations = sum(record["evaluations"] for record in result.adaptive_diagnostics)
    print(f"  Motion.auto() -> {result.motion_sampling}: {evaluations} world evaluations over {frames} frames")
    assert result.motion_sampling == "adaptive", result.motion_sampling

    axes = result.axes
    # The range gate holds the body for the whole walk: from the closest it gets
    # to the farthest it starts, two bins either side.
    reach = (START_DISTANCE_M - WALK_SPEED_MPS * frames * frame_s - 0.5, START_DISTANCE_M + 0.5)
    gate = (axes.range_m >= reach[0] - 2 * axes.range_bin_m) & (axes.range_m <= reach[1] + 2 * axes.range_bin_m)
    signal = slow_time_signal(result, gate)
    spectrogram = microdoppler_spectrogram(signal, window_slots=WINDOW_SLOTS, hop_slots=HOP_SLOTS)
    power = spectrogram.data.abs().square().sum(dim=0)  # incoherent over the range gate: [windows, velocity]
    velocity = spectrogram.velocity_mps
    bin_mps = float(velocity[1] - velocity[0])
    print(
        f"  Spectrogram: {tuple(power.shape)} (windows, velocity bins) over {int(gate.sum())} range bins; "
        f"{bin_mps:.3f} m/s bins, +/-{float(velocity.abs().max()):.2f} m/s unambiguous"
    )

    torso_mps = float(velocity[power.mean(dim=0).argmax()])
    level_db = 10.0 * torch.log10(power / power.max())
    signature = velocity[(level_db > ENVELOPE_DB).any(dim=0)]
    low, high = float(signature.min()), float(signature.max())
    print(f"  Torso line: {torso_mps:+.3f} m/s (walking at {WALK_SPEED_MPS:+.3f} m/s)")
    print(f"  Signature above {ENVELOPE_DB:.0f} dB: {low:+.2f} to {high:+.2f} m/s")
    if args.smpl_model is None:
        truth = walker.closing_speeds([float(t) for t in spectrogram.times_s])
        true_low, true_high = float(truth.min()), float(truth.max())
        print(f"  Body radial speeds (from the trajectory): {true_low:+.2f} to {true_high:+.2f} m/s")
        assert abs(torso_mps - WALK_SPEED_MPS) <= 2 * bin_mps, f"torso line at {torso_mps:.3f} m/s"
        assert high >= WALK_SPEED_MPS + 1.0 and low <= WALK_SPEED_MPS - 0.8, (low, high)
        # Three bins: the Hann window's main lobe is two bins either side of a line.
        margin = 3 * bin_mps
        assert true_low - margin <= low and high <= true_high + margin, (low, high, true_low, true_high)
        print("  Torso line and limb micro-Doppler spread  OK")

    if args.plot is not None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        times_s = spectrogram.times_s.cpu().numpy()
        figure, ax = plt.subplots(figsize=(7, 3.5), dpi=120)
        image = ax.imshow(
            level_db.T.cpu().numpy(),
            extent=[times_s[0], times_s[-1], float(velocity[0]), float(velocity[-1])],
            origin="lower",
            aspect="auto",
            cmap="viridis",
            vmin=-40.0,
            vmax=0.0,
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Closing velocity (m/s)")
        ax.set_title("Walking person, micro-Doppler")
        figure.colorbar(image, ax=ax, label="dB re peak")
        figure.tight_layout()
        figure.savefig(args.plot)
        print(f"  Saved {args.plot}")
    print("PASSED")


if __name__ == "__main__":
    main()
