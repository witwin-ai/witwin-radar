"""MUSIC imaging of two point targets with a 20 x 20 UPA.

    witwin.core.Scene  ->  Radar.simulate  ->  RadarSimulationResult
                       ->  witwin.radar.processing.range_doppler
                       ->  witwin.radar.processing.music_image

The world is two point targets at the same range, half a metre either side of
the boresight, and no geometry at all: the ``Scene`` carries no structures, so
the only rows the solve publishes are the two direct round trips. That is
deliberate - this example is about angular resolution, and a wall would add
multipath rows at other ranges that the MUSIC range gate would then have to
exclude.

**MUSIC needs a noise subspace, so this example configures one.** The
pseudo-spectrum is formed from the eigenvectors of the array covariance that do
NOT span the signal, and a noiseless simulation of two coherent targets produces
a covariance of rank two with every other eigenvalue exactly zero. There is no
noise subspace to find, the eigendecomposition itself is ill conditioned, and
the answer is undefined rather than merely inaccurate. The receive chain here is
a real thermal-noise front end at a 10 dB noise figure. It is attached to the
``RadarConfig`` after validation because the flat mapping accepted by
``RadarConfig.from_dict`` does not carry a ``frontend`` block.

Note also that ``Radar.simulate`` composes the round trip ONCE per frame, so the
eight chirps of one frame are identical snapshots and the covariance is
decorrelated by the spatial smoothing rather than by slow time. Intra-frame
Doppler needs a forward-AD velocity dual and is a named Phase-11 deferral; see
``docs/pipeline_guide.md``.

Usage:
    python -m examples.music_imaging
    python examples/music_imaging.py
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

from witwin.core import AntennaState, Scene  # noqa: E402
from witwin.core.identity import reserve_antenna_id  # noqa: E402
from witwin.radar import Radar, RadarConfig  # noqa: E402
from witwin.radar.simulation import ScatterSitePolicy  # noqa: E402
from witwin.radar.frontend import FrontendSpec, NoiseSpec, SeedSpec  # noqa: E402
from witwin.radar.processing import (  # noqa: E402
    ArrayGeometry,
    ProcessingAxes,
    ProcessingCube,
    music_image,
    range_profile,
)
from witwin.radar.scattering import ScalarRcsResponse  # noqa: E402
from witwin.radar.synthesis import SlowTimeMode  # noqa: E402

ARRAY_SIZE = 20
FIELD_OF_VIEW_RAD = math.pi / 2
NUM_PIXELS = 128
NUM_SIGNALS = 7
SPATIAL_SMOOTH = 3

CONFIG = {
    "num_tx": ARRAY_SIZE, "num_rx": ARRAY_SIZE,
    "fc": 77e9,
    "slope": 60.012,
    "adc_samples": 256,
    "adc_start_time": 6,
    "sample_rate": 4400,
    "idle_time": 7,
    "ramp_end_time": 65,
    "chirp_per_frame": 8,
    "frame_per_second": 10,
    "num_doppler_bins": 8,
    "num_range_bins": 256,
    "num_angle_bins": 64,
    "power": 15,
    # The transmit row runs along the array's local x, the receive row along its
    # local y, so the virtual array is a planar 20 x 20 grid. MUSIC's first image
    # axis is the transmit row and its second is the receive row.
    "tx_loc": [[index, 0, 0] for index in range(ARRAY_SIZE)],
    "rx_loc": [[ARRAY_SIZE, -index, 0] for index in range(ARRAY_SIZE)],
}

#: The radar looks along ``-z``, so its local x axis is world ``+x`` and the two
#: targets - separated in world x - are separated along the image's first axis.
TARGET_RANGE_M = 3.0
TARGET_OFFSET_M = 0.5
TARGET_POSITIONS_M = (
    (-TARGET_OFFSET_M, 0.0, -TARGET_RANGE_M),
    (TARGET_OFFSET_M, 0.0, -TARGET_RANGE_M),
)
TARGET_RCS_M2 = 1.0
#: Transverse to the ``-z`` boresight. Channel projects the field onto this
#: world-frame vector; the package default ``(0, 0, 1)`` is parallel to this
#: boresight and would publish exactly zero transport.
POLARIZATION = (0.0, 1.0, 0.0)


def build_radar() -> Radar:
    config = RadarConfig.from_dict(CONFIG)
    config = dataclasses.replace(
        config,
        frontend=FrontendSpec(
            noise=NoiseSpec(noise_figure_db=10.0, bandwidth_hz=CONFIG["sample_rate"] * 1e3),
            seed=SeedSpec(20260727),
        ),
    )
    return Radar(config, position=(0.0, 0.0, 0.0), target=(0.0, 0.0, -1.0))


def build_scene() -> Scene:
    """An empty world: the targets are declared scatter sites, not geometry."""

    return Scene(
        structures=(),
        endpoints=[
            AntennaState(
                reserve_antenna_id(770201),
                "tx",
                torch.tensor((0.0, 0.0, 0.0), dtype=torch.float32),
            )
        ],
    )


def processing_axes(radar: Radar) -> ProcessingAxes:
    """The metadata record every processing stage reads.

    ``ProcessingAxes`` is built from a rank-3 ``SynthesisResult`` while the
    simulation result publishes the assembled ``[frame, tx, rx, slow, fast]``
    cube, so the last frame's composed rows are re-synthesized to obtain one.
    The record carries shapes and conventions, which are properties of the
    waveform specification and are the same for every frame.
    """

    synthesis = radar._synthesize(
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
    sites = ScatterSitePolicy.explicit(
        torch.tensor(TARGET_POSITIONS_M, dtype=torch.float32, device=radar.device)
    )
    response = ScalarRcsResponse.from_rcs(
        TARGET_RCS_M2,
        reference_frequency_hz=radar.config.fc,
        device=radar.device,
    )

    print(f"Using device={radar.device}")
    print(f"Simulating two targets with a {ARRAY_SIZE}x{ARRAY_SIZE} MIMO array...")
    result = radar.simulate(
        build_scene(),
        times=(0.0,),
        response=response,
        sites=sites,
        polarization=POLARIZATION,
        components=frozenset({"los"}),
        max_depth=0,
    )
    assert result.cube.shape == (
        1, ARRAY_SIZE, ARRAY_SIZE, CONFIG["chirp_per_frame"], CONFIG["adc_samples"]
    ), f"Unexpected cube shape: {tuple(result.cube.shape)}"
    print(f"  Cube: {tuple(result.cube.shape)} {result.axes}  OK")
    print(f"  Composed rows: {radar.last_radar_paths.path_count}")

    axes = processing_axes(radar)
    geometry = ArrayGeometry.from_axes(axes)
    profile = range_profile(ProcessingCube(result.cube[0], axes), window="hann")

    # The range gate is chosen here rather than inside the imager: reading a
    # peak off a spectrum is a modelling choice and ``music_image`` refuses to
    # make it silently. Both targets share one range, so one gate holds both.
    range_energy = profile.data.abs().sum(dim=(0, 1, 2))
    peak_bin = int(torch.argmax(range_energy))
    range_bins = torch.arange(peak_bin - 2, peak_bin + 3, device=radar.device)
    print(f"  Range gate: bins {peak_bin - 2}..{peak_bin + 2} "
          f"({float(axes.range_m[peak_bin]):.4f} m)")

    print("Running MUSIC...")
    grid = torch.linspace(
        -FIELD_OF_VIEW_RAD / 2, FIELD_OF_VIEW_RAD / 2, NUM_PIXELS, device=radar.device
    )
    image = music_image(
        profile,
        geometry,
        elevation_rad=grid,
        azimuth_rad=grid,
        range_bins=range_bins,
        num_signals=NUM_SIGNALS,
        spatial_smooth=SPATIAL_SMOOTH,
        num_snapshots=CONFIG["chirp_per_frame"],
    )
    assert tuple(image.shape) == (NUM_PIXELS, NUM_PIXELS, int(range_bins.numel())), (
        f"Unexpected image shape: {tuple(image.shape)}"
    )
    print(f"  Image: {tuple(image.shape)}  OK")

    # Both targets sit on the image's first axis, so collapse the other two and
    # read the two strongest lobes off the resulting profile.
    lobe_profile = image.abs().amax(dim=2).amax(dim=1)
    peaks = _two_strongest_lobes(lobe_profile)
    measured_deg = sorted(float(grid[index]) * 180.0 / math.pi for index in peaks)
    expected_deg = sorted(
        math.degrees(math.atan2(offset, TARGET_RANGE_M))
        for offset in (-TARGET_OFFSET_M, TARGET_OFFSET_M)
    )
    pixel_deg = float(grid[1] - grid[0]) * 180.0 / math.pi
    for measured, expected in zip(measured_deg, expected_deg):
        assert abs(measured - expected) <= 2.0 * pixel_deg, (
            f"MUSIC reports {measured:.2f} deg where the target is at "
            f"{expected:.2f} deg ({pixel_deg:.2f} deg pixels)"
        )
    print(
        f"  Resolved at {measured_deg[0]:.2f} and {measured_deg[1]:.2f} deg; "
        f"targets at {expected_deg[0]:.2f} and {expected_deg[1]:.2f} deg  OK"
    )
    print("PASSED")


def _two_strongest_lobes(profile: torch.Tensor) -> tuple[int, int]:
    """The indices of the two strongest LOCAL maxima of a 1-D profile.

    ``topk(2)`` would return two samples of the same lobe. A local maximum test
    is the smallest thing that separates two peaks without assuming how wide
    they are.
    """

    values = profile.detach().cpu()
    interior = values[1:-1]
    is_peak = (interior > values[:-2]) & (interior > values[2:])
    indices = torch.nonzero(is_peak, as_tuple=False).reshape(-1) + 1
    if int(indices.numel()) < 2:
        raise AssertionError(
            f"the MUSIC profile has {int(indices.numel())} lobes; two targets "
            "must produce at least two"
        )
    order = torch.argsort(values[indices], descending=True)
    return int(indices[order[0]]), int(indices[order[1]])


if __name__ == "__main__":
    main()
