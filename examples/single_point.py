"""One point target in front of a wall, through the scene-driven pipeline.

This is the smallest complete use of the scene-driven entry point:

    witwin.core.Scene  ->  Radar.simulate  ->  Result
                       ->  witwin.radar.processing

The world is a concrete wall and a single scatter site 3 m in front of the
radar. Because the site is a declared point and the wall is real geometry, the
solve publishes BOTH the direct round trip and the two single-bounce round trips
that go via the wall, so the range profile has a target peak and a multipath
peak whose positions are known in closed form. The example checks them.

Two numbers are asserted rather than printed, because both are exactly
predictable and a wrong pipeline still produces a plausible picture:

* the strongest composed transport equals the free-space radar equation
  ``sqrt(P_tx) * (lambda / 4 pi R) * (sqrt(4 pi sigma) / lambda) *
  (lambda / 4 pi R)`` for this range and cross section;
* the range-profile peak sits within one range bin of the true 3 m.

Two conventions are worth reading before copying this file:

* the radar looks along ``-z`` (the camera convention its pose uses) and says
  nothing about polarization, because it does not have to. Channel projects the
  transmitted field onto a world-frame vector, and a vector parallel to the
  boresight radiates nothing; ``Radar.polarization`` therefore defaults to
  ``"up"``, the pose's own up axis, which is transverse whichever way the radar
  is pointed. A world vector is still accepted, and one parallel to the
  boresight is refused rather than publishing a cube of exact zeros.
* the receive chain is made of fields on the radar - here one ``Noise`` stage.
  ``Radar.from_dict`` reads the flat FMCW mapping and takes every other field as
  a keyword override, so the chain and the pose are attached in the same call.
  Without the noise the cube is noiseless and CFAR detects sidelobes rather than
  targets.

Usage:
    python -m examples.single_point
    python examples/single_point.py
"""

from __future__ import annotations

import math
import pathlib
import sys

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from witwin.core import AntennaState, Mesh, PhysicalMaterial, Scene, Structure  # noqa: E402
from witwin.core.identity import reserve_antenna_id  # noqa: E402

from witwin.radar import Noise, PointTargets, Radar  # noqa: E402

SPEED_OF_LIGHT_M_PER_S = 299792458.0

CONFIG = {
    "num_tx": 3,
    "num_rx": 4,
    "fc": 77e9,
    "slope": 60.012,
    "adc_samples": 256,
    "adc_start_time": 6,
    "sample_rate": 4400,
    "idle_time": 7,
    "ramp_end_time": 65,
    "chirp_per_frame": 128,
    "power": 15,
    "tx_loc": [[0, 0, 0], [4, 0, 0], [2, 1, 0]],
    "rx_loc": [[-6, 0, 0], [-5, 0, 0], [-4, 0, 0], [-3, 0, 0]],
}

#: The world. The radar sits at the origin looking along ``-z``; the wall is a
#: 4 m square in the plane ``z = -5`` and the target is on the boresight at 3 m.
TARGET_POSITION_M = (0.0, 0.0, -3.0)
TARGET_RCS_M2 = 1.0
WALL_PLANE_Z_M = -5.0
WALL_HALF_EXTENT_M = 2.0
FRAME_TIMES_S = (0.0, 0.1, 0.2)


def build_scene() -> Scene:
    """One concrete wall, in world coordinates.

    ``recenter=False`` is mandatory: ``Mesh`` otherwise subtracts the bounding
    box centre from the authored vertices, which would move the wall off the
    plane every closed form below is written against, and nothing would raise.
    """

    mesh = Mesh(
        vertices=torch.tensor(
            (
                (-WALL_HALF_EXTENT_M, -WALL_HALF_EXTENT_M, WALL_PLANE_Z_M),
                (WALL_HALF_EXTENT_M, -WALL_HALF_EXTENT_M, WALL_PLANE_Z_M),
                (WALL_HALF_EXTENT_M, WALL_HALF_EXTENT_M, WALL_PLANE_Z_M),
                (-WALL_HALF_EXTENT_M, WALL_HALF_EXTENT_M, WALL_PLANE_Z_M),
            ),
            dtype=torch.float32,
        ),
        faces=torch.tensor(((0, 1, 2), (0, 2, 3)), dtype=torch.int64),
        recenter=False,
        fill_mode="surface",
        topology_diagnostics=False,
    )
    wall = Structure(
        geometry=mesh,
        material=PhysicalMaterial(name="concrete", eps_r=5.24, sigma_e=0.0462),
        structure_id=1,
        material_id=1,
        assignment_id=1,
        surface_id=1,
    )
    return Scene(
        structures=(wall,),
        endpoints=[AntennaState(reserve_antenna_id(770101), "tx", torch.tensor((0.0, 0.0, 0.0), dtype=torch.float32))],
    )


def build_radar() -> Radar:
    """The front end, with a thermal-noise receive chain attached."""

    # The noise bandwidth is left unset: a thermal stage integrates over the
    # waveform's own sampling bandwidth, and the radar fills that in from the
    # ``sample_rate`` this mapping already declares.
    return Radar.from_dict(
        CONFIG, noise=Noise(figure=10.0), seed=20260727, position=(0.0, 0.0, 0.0), look_at=(0.0, 0.0, -1.0)
    )


def expected_transport(radar: Radar) -> float:
    """The free-space two-way coefficient this world must publish.

    Written out rather than read off the result: the point of the check is that
    the pipeline reproduces the radar equation, and comparing the pipeline
    against itself proves nothing.
    """

    wavelength_m = SPEED_OF_LIGHT_M_PER_S / radar.carrier
    transmit_power_w = radar.transmit_power_watts
    range_m = math.dist((0.0, 0.0, 0.0), TARGET_POSITION_M)
    spreading = wavelength_m / (4.0 * math.pi * range_m)
    strength = math.sqrt(4.0 * math.pi * TARGET_RCS_M2) / wavelength_m
    return math.sqrt(transmit_power_w) * spreading * strength * spreading


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This example requires CUDA: the propagation solve runs entirely in "
            "native CUDA kernels and has no CPU path."
        )

    radar = build_radar()
    scene = build_scene()
    targets = PointTargets(
        positions=torch.tensor([TARGET_POSITION_M], dtype=torch.float32, device=radar.device), rcs=TARGET_RCS_M2
    )

    print(f"Using device={radar.device}")
    print("Simulating the scene...")
    result = radar.simulate(scene, targets, times=FRAME_TIMES_S, los=True, reflections=1)

    assert result.cube.shape == (
        len(FRAME_TIMES_S),
        radar.num_tx,
        radar.num_rx,
        radar.waveform.chirps_per_frame,
        radar.waveform.samples_per_chirp,
    ), f"Unexpected cube shape: {tuple(result.cube.shape)}"
    print(f"  Cube: {tuple(result.cube.shape)} {result.axis_names}  OK")

    # The world does not move, so the pipeline compiles the scene once and
    # discovers the path topology once for the whole run.
    assert result.compile_count == 1 and result.discovery_count == 1, (
        f"a still world compiled {result.compile_count} times and discovered {result.discovery_count} topologies"
    )
    print(f"  Epochs: {result.epochs}  compiles={result.compile_count} discoveries={result.discovery_count}  OK")

    # The four typed diagnostics, all describing the LAST frame.
    print(
        "  Diagnostics: "
        f"{type(result.last_snapshot).__name__}, "
        f"{type(result.last_compiled_scene).__name__}, "
        f"{type(result.last_propagation).__name__}, "
        f"{type(result.last_radar_paths).__name__}"
    )

    paths = result.last_radar_paths
    print(f"  Composed rows: {paths.path_count} over {paths.sensor_pair_count} pairs")
    measured = float(paths.complex_transfer_ref.abs().max())
    predicted = expected_transport(radar)
    relative = abs(measured - predicted) / predicted
    assert relative < 1e-4, (
        f"the strongest transport is {measured:.6e} but the radar equation "
        f"predicts {predicted:.6e} ({relative:.3e} relative)"
    )
    print(f"  |C_rt| = {measured:.6e} vs radar equation {predicted:.6e}  OK")

    # The result carries the metadata every processing stage reads, so a
    # frame is a cube and its axes together. Assembling that pairing by hand
    # was how a cube could end up described by a different array than the one
    # it came from.
    frame = result.frame(0)
    axes = frame.axes
    geometry = frame.array()
    # No fast-time window here: this waveform's ``output`` is ``"spectrum"``, so
    # the cube arrives already transformed and a window applied after the
    # transform would weight bins rather than samples. The Doppler stage still
    # takes one, because slow time has not been transformed yet.
    rd = frame.range_doppler(window="hann")
    combined = rd.data.reshape(geometry.sensor_pair_count, *rd.data.shape[-2:]).sum(dim=0)
    range_response = combined.abs().amax(dim=0)

    peak_bin = int(torch.argmax(range_response))
    peak_range_m = float(axes.range_m[peak_bin])
    true_range_m = math.dist((0.0, 0.0, 0.0), TARGET_POSITION_M)
    assert abs(peak_range_m - true_range_m) <= axes.range_bin_m, (
        f"the range peak is at {peak_range_m:.4f} m but the target is at "
        f"{true_range_m:.4f} m ({axes.range_bin_m:.4f} m bins)"
    )
    print(f"  Range peak: {peak_range_m:.4f} m (target at {true_range_m:.4f} m)  OK")

    # The wall turns one target into three round trips: direct-direct at 3 m,
    # the two direct-reflected cross terms at 5 m, and reflected-reflected at
    # 7 m. The 5 m peak is the one the wall's image source predicts.
    multipath_range_m = 0.5 * (true_range_m + math.dist((0.0, 0.0, 2.0 * WALL_PLANE_Z_M), TARGET_POSITION_M))
    multipath_bin = int(round(multipath_range_m / axes.range_bin_m))
    window = range_response[multipath_bin - 2 : multipath_bin + 3]
    assert float(window.max()) > 0.1 * float(range_response[peak_bin]), (
        f"no multipath return near {multipath_range_m:.4f} m"
    )
    print(f"  Multipath peak near {multipath_range_m:.4f} m  OK")

    cloud = frame.points(pfa=1e-4, guard_cells=(2, 4), training_cells=(4, 8), max_points=64)
    print(f"  Point cloud: {len(cloud)} points")
    print("PASSED")


if __name__ == "__main__":
    main()
