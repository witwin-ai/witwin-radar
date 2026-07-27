"""One point target in front of a wall, through the scene-driven pipeline.

This is the smallest complete use of the Phase-11 entry point:

    witwin.core.Scene  ->  Radar.simulate  ->  RadarSimulationResult
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

* the radar looks along ``-z`` (the camera convention its pose uses) and the
  endpoint polarization is therefore declared along ``+y``. Channel projects the
  field onto that world-frame vector, so a polarization parallel to the
  propagation direction radiates nothing and every transport comes back exactly
  zero. The default ``(0, 0, 1)`` is transverse for a radar that looks along
  ``x``; it is NOT transverse for one that looks along ``z``.
* the receive chain is a ``FrontendSpec``. It is attached to the ``RadarConfig``
  after validation because the flat mapping accepted by ``RadarConfig.from_dict``
  does not carry a ``frontend`` block. Without it the cube is noiseless and CFAR
  detects sidelobes rather than targets.

Usage:
    python -m examples.single_point
    python examples/single_point.py
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

from witwin.core import AntennaState, Mesh, PhysicalMaterial, Scene, Structure  # noqa: E402
from witwin.core.identity import reserve_antenna_id  # noqa: E402
from witwin.radar import Radar, RadarConfig, ScatterSitePolicy  # noqa: E402
from witwin.radar.frontend import FrontendSpec, NoiseSpec, SeedSpec  # noqa: E402
from witwin.radar.processing import (  # noqa: E402
    ArrayGeometry,
    ProcessingAxes,
    ca_cfar_fast,
    point_cloud,
    range_doppler,
    range_profile,
)
from witwin.radar.scattering import ScalarRcsResponse  # noqa: E402
from witwin.radar.synthesis import SlowTimeMode  # noqa: E402

SPEED_OF_LIGHT_M_PER_S = 299792458.0

CONFIG = {
    "num_tx": 3, "num_rx": 4,
    "fc": 77e9,
    "slope": 60.012,
    "adc_samples": 256,
    "adc_start_time": 6,
    "sample_rate": 4400,
    "idle_time": 7,
    "ramp_end_time": 65,
    "chirp_per_frame": 128,
    "frame_per_second": 10,
    "num_doppler_bins": 128,
    "num_range_bins": 256,
    "num_angle_bins": 64,
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
POLARIZATION = (0.0, 1.0, 0.0)
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
        endpoints=[
            AntennaState(
                reserve_antenna_id(770101),
                "tx",
                torch.tensor((0.0, 0.0, 0.0), dtype=torch.float32),
            )
        ],
    )


def build_radar() -> Radar:
    """The front end, with a thermal-noise receive chain attached."""

    config = RadarConfig.from_dict(CONFIG)
    config = dataclasses.replace(
        config,
        frontend=FrontendSpec(
            noise=NoiseSpec(noise_figure_db=10.0, bandwidth_hz=CONFIG["sample_rate"] * 1e3),
            seed=SeedSpec(20260727),
        ),
    )
    return Radar(config, position=(0.0, 0.0, 0.0), target=(0.0, 0.0, -1.0))


def expected_transport(radar: Radar) -> float:
    """The free-space two-way coefficient this world must publish.

    Written out rather than read off the result: the point of the check is that
    the pipeline reproduces the radar equation, and comparing the pipeline
    against itself proves nothing.
    """

    wavelength_m = SPEED_OF_LIGHT_M_PER_S / radar.config.fc
    transmit_power_w = radar.system_config.sensors.tx_power.transmit_power_watts
    range_m = math.dist((0.0, 0.0, 0.0), TARGET_POSITION_M)
    spreading = wavelength_m / (4.0 * math.pi * range_m)
    strength = math.sqrt(4.0 * math.pi * TARGET_RCS_M2) / wavelength_m
    return math.sqrt(transmit_power_w) * spreading * strength * spreading


def processing_axes(radar: Radar) -> ProcessingAxes:
    """The metadata record every processing stage reads.

    ``ProcessingAxes`` is built from a ``SynthesisResult``, and the simulation
    result publishes the ASSEMBLED ``[frame, tx, rx, slow, fast]`` cube rather
    than the rank-3 synthesis product. Re-synthesizing the last frame's composed
    rows is the public route to one: the record carries shapes and conventions,
    both of which are properties of the waveform specification and are therefore
    the same for every frame.
    """

    synthesis = radar.synthesize(
        radar.last_radar_paths,
        slow_time_mode=SlowTimeMode.FROZEN_WEIGHT_WITH_CARRIER_RATE,
    )
    return ProcessingAxes.from_synthesis(
        synthesis,
        radar.system_config.waveform_spec(),
        radar.system_config.sensors.array,
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This example requires CUDA: the propagation solve runs entirely in "
            "native CUDA kernels and has no CPU path."
        )

    radar = build_radar()
    scene = build_scene()
    sites = ScatterSitePolicy.explicit(
        torch.tensor([TARGET_POSITION_M], dtype=torch.float32, device=radar.device)
    )
    response = ScalarRcsResponse.from_rcs(
        TARGET_RCS_M2,
        reference_frequency_hz=radar.config.fc,
        device=radar.device,
    )

    print(f"Using device={radar.device}")
    print("Simulating the scene...")
    result = radar.simulate(
        scene,
        times=FRAME_TIMES_S,
        response=response,
        sites=sites,
        polarization=POLARIZATION,
        components=frozenset({"los", "reflection"}),
        max_depth=1,
    )

    array = radar.system_config.sensors.array
    assert result.cube.shape == (
        len(FRAME_TIMES_S),
        array.num_tx,
        array.num_rx,
        radar.system_config.waveform.chirp_per_frame,
        radar.system_config.waveform.adc_samples,
    ), f"Unexpected cube shape: {tuple(result.cube.shape)}"
    print(f"  Cube: {tuple(result.cube.shape)} {result.axes}  OK")

    # The world does not move, so the pipeline compiles the scene once and
    # discovers the path topology once for the whole run.
    assert result.compile_count == 1 and result.discovery_count == 1, (
        f"a still world compiled {result.compile_count} times and discovered "
        f"{result.discovery_count} topologies"
    )
    print(
        f"  Epochs: {result.epochs}  compiles={result.compile_count} "
        f"discoveries={result.discovery_count}  OK"
    )

    # The four typed diagnostics, all describing the LAST frame.
    print(
        "  Diagnostics: "
        f"{type(radar.last_snapshot).__name__}, "
        f"{type(radar.last_compiled_scene).__name__}, "
        f"{type(radar.last_propagation).__name__}, "
        f"{type(radar.last_radar_paths).__name__}"
    )

    paths = radar.last_radar_paths
    print(f"  Composed rows: {paths.path_count} over {paths.sensor_pair_count} pairs")
    measured = float(paths.complex_transfer_ref.abs().max())
    predicted = expected_transport(radar)
    relative = abs(measured - predicted) / predicted
    assert relative < 1e-4, (
        f"the strongest transport is {measured:.6e} but the radar equation "
        f"predicts {predicted:.6e} ({relative:.3e} relative)"
    )
    print(f"  |C_rt| = {measured:.6e} vs radar equation {predicted:.6e}  OK")

    axes = processing_axes(radar)
    geometry = ArrayGeometry.from_axes(axes)
    profile = range_profile(result.cube[0], axes=axes, window="hann")
    rd = range_doppler(profile, window="hann")
    combined = rd.data.reshape(
        geometry.sensor_pair_count, *rd.data.shape[-2:]
    ).sum(dim=0)
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
    multipath_range_m = 0.5 * (
        true_range_m
        + math.dist((0.0, 0.0, 2.0 * WALL_PLANE_Z_M), TARGET_POSITION_M)
    )
    multipath_bin = int(round(multipath_range_m / axes.range_bin_m))
    window = range_response[multipath_bin - 2 : multipath_bin + 3]
    assert float(window.max()) > 0.1 * float(range_response[peak_bin]), (
        f"no multipath return near {multipath_range_m:.4f} m"
    )
    print(f"  Multipath peak near {multipath_range_m:.4f} m  OK")

    detections = ca_cfar_fast(
        combined.abs(), guard_cells=(2, 4), training_cells=(4, 8), pfa=1e-4
    )
    cloud = point_cloud(
        detections, rd, axes, geometry, route="phase_comparison", max_points=64
    )
    print(f"  Point cloud: {len(cloud)} points")
    print("PASSED")


if __name__ == "__main__":
    main()
