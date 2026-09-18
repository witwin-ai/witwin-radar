"""
Pytest configuration and shared fixtures for the radar test suite.

Run:
    cd radar
    pytest tests/                         # CPU-only tests
    pytest tests/ --gpu                   # include GPU tests (needs CUDA)
    pytest tests/processing/ -v           # processing owners
"""

import os
import sys

import pytest

# Ensure witwin.radar is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# pytest plugins
# ---------------------------------------------------------------------------


def pytest_addoption(parser):
    parser.addoption(
        "--gpu",
        action="store_true",
        default=False,
        help="Run GPU-only tests (solver cross-validation, end-to-end validation)",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: test requires CUDA GPU")


def pytest_collection_modifyitems(config, items):
    import torch

    run_gpu = config.getoption("--gpu") and torch.cuda.is_available()
    if run_gpu:
        return

    skip_gpu = pytest.mark.skip(reason="needs --gpu flag and CUDA device")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)


# ---------------------------------------------------------------------------
# Standard radar configurations
# ---------------------------------------------------------------------------
#
# These are the flat FMCW file format ``Radar.from_dict`` reads, in the vendor
# units it converts: MHz/us of slope, kSPS of sample rate, microseconds of
# timing, dBm of power and half wavelengths of element offset.
#
# ``frame_per_second``, ``num_doppler_bins``, ``num_range_bins`` and
# ``num_angle_bins`` used to sit here and are now REFUSED by the loader. They
# described a processing grid nothing consumed: the bin counts come from the
# waveform spec and the frame rate is the caller's own scheduling number, so a
# test that needs one declares it as its own local constant.

STANDARD_CONFIG = {
    "num_tx": 3,
    "num_rx": 4,
    "fc": 77e9,
    "slope": 60.012,
    "adc_samples": 256,
    "adc_start_time": 6,
    "sample_rate": 4400,
    "idle_time": 7,
    "ramp_end_time": 58,
    "chirp_per_frame": 128,
    "power": 12,
    "tx_loc": [[0, 0, 0], [2, 0, 0], [0, 1, 0]],
    "rx_loc": [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]],
}

FAST_CONFIG = {**STANDARD_CONFIG, "chirp_per_frame": 32}

#: The validation suite's configs: ``adc_start_time=0`` for a clean signal,
#: and enough chirps for Doppler.
VALIDATION_FAST_CONFIG = {**FAST_CONFIG, "adc_start_time": 0}
VALIDATION_FULL_CONFIG = {**STANDARD_CONFIG, "adc_start_time": 0}

MINIMAL_CONFIG = {
    "num_tx": 1,
    "num_rx": 1,
    "fc": 77e9,
    "slope": 60.012,
    "adc_samples": 256,
    "adc_start_time": 0,
    "sample_rate": 4400,
    "idle_time": 7,
    "ramp_end_time": 58,
    "chirp_per_frame": 2,
    "power": 12,
    "tx_loc": [[0, 0, 0]],
    "rx_loc": [[0, 0, 0]],
}


PROCESSING_CONFIG = {
    **STANDARD_CONFIG,
    "num_tx": 3,
    "num_rx": 4,
    "adc_start_time": 0,
    "adc_samples": 64,
    "chirp_per_frame": 16,
    "tx_loc": [[0, 0, 0], [2, 0, 0], [0, 1, 0]],
    "rx_loc": [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]],
}


def make_processing_axes(config=None, *, doppler_bins: int | None = None):
    """Build processing metadata through the canonical synthesis contracts."""

    from dataclasses import replace

    import torch

    from witwin.radar import Radar
    from witwin.radar.processing import ProcessingAxes
    from witwin.radar.synthesis.assembly import SynthesisResult

    raw = PROCESSING_CONFIG if config is None else config
    # ``device="cpu"`` because this builds metadata out of an all-zero cube and
    # must not need a CUDA device to describe an axis.
    radar = raw if isinstance(raw, Radar) else Radar.from_dict(dict(raw), device="cpu")
    spec = replace(radar.waveform_spec(), output_domain="beat")
    if doppler_bins is not None:
        spec = replace(spec, num_chirps=int(doppler_bins))
    array = radar.system_config.sensors.array
    cube = torch.zeros((spec.num_chirps, array.sensor_pair_count, spec.num_samples), dtype=torch.complex64)
    result = SynthesisResult.from_fmcw(cube, spec)
    return ProcessingAxes.from_synthesis(result, spec, array)


# ---------------------------------------------------------------------------
# CPU-only Radar contract fixture
# ---------------------------------------------------------------------------


class RadarFixture:
    """A CPU radar and the processing axes its waveform describes.

    It holds a real ``Radar`` built on ``device="cpu"`` and adds only what a
    formula test reads: the axes record, and the element offsets in metres
    rather than in the half wavelengths the array stores.
    """

    def __init__(self, config=None):
        import torch

        from witwin.radar import Radar
        from witwin.radar.processing import ProcessingAxes
        from witwin.radar.synthesis import SynthesisResult

        raw = STANDARD_CONFIG if config is None else config
        self.radar = raw if isinstance(raw, Radar) else Radar.from_dict(dict(raw), device="cpu")

        self.wavelength_m = self.radar.wavelength
        self.system_config = self.radar.system_config
        # Through the array spec rather than by scaling ``radar.tx``: the array
        # always stores half wavelengths, but ``radar.tx`` is in whatever
        # ``antenna_unit`` says, so scaling it here would be silently wrong for
        # a metre-authored radar handed in directly.
        self.tx_loc, self.rx_loc = self.system_config.sensors.array.local_offsets_m(device="cpu")
        spec = self.radar.waveform_spec()
        array = self.system_config.sensors.array
        cube = torch.zeros(spec.num_chirps, array.sensor_pair_count, spec.num_samples, dtype=torch.complex64)
        result = SynthesisResult.from_fmcw(cube, spec)
        self.axes = ProcessingAxes.from_synthesis(result, spec, array)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
#
# The three config fixtures used to return a validated ``RadarConfig``. That
# record is gone and the flat mapping IS the configuration, so they hand back a
# copy of the mapping and validation happens where it now lives, in
# ``Radar.from_dict``. A copy rather than the module constant so that a test
# which edits what it is given cannot leak that edit into the next test.


@pytest.fixture
def standard_config():
    return dict(STANDARD_CONFIG)


@pytest.fixture
def fast_config():
    return dict(FAST_CONFIG)


@pytest.fixture
def minimal_config():
    return dict(MINIMAL_CONFIG)


@pytest.fixture
def radar_fixture():
    return RadarFixture(STANDARD_CONFIG)


# ---------------------------------------------------------------------------
# The scene-driven point-target fixtures
# ---------------------------------------------------------------------------
#
# The validation tests drive ``Radar.simulate``, which reads a Core world, so a
# moving target is a declaration of where it is and how fast it moves.
#
# Two conventions are fixed here once so that no validation test restates them:
#
# * **The radar looks along world +x** with the default up. Targets are still
#   authored in the radar's LOCAL frame - ``[0, 0, -d]`` is still "d metres
#   straight ahead" - and ``support.pose.world_from_local_points`` applies the
#   radar's own pose frame, so the production pose transform is on the path
#   rather than mirrored.
#
#   ``Radar.polarization`` defaults to ``"up"``, derived from the pose and
#   therefore transverse by construction, so no boresight can publish the
#   exactly ZERO transport a fixed world polarization vector would. The
#   convention stays because every expected number in ``tests/validation`` was
#   measured against it, and a suite-wide pose change is a physics change.
#
# * Moving targets use authored trajectories through the public scene entry.
#   These DSP tests explicitly select chirp-frozen motion to isolate slow-time
#   processing. Continuous ADC motion has separate independent phase oracles.

#: The boresight this suite poses its radars along, in world coordinates.
SCENE_DRIVEN_LOOK_AT_M = (1.0, 0.0, 0.0)

#: The world up vector that goes with it.
SCENE_DRIVEN_UP = (0.0, 1.0, 0.0)


def scene_radar(config, **pose):
    """A :class:`Radar` posed along :data:`SCENE_DRIVEN_LOOK_AT_M`.

    The pose is the whole reason this exists rather than a call to
    ``Radar.from_dict`` at each site - see the note above about the
    polarization null.
    """

    from witwin.radar import Radar

    options = {"position": (0.0, 0.0, 0.0), "look_at": SCENE_DRIVEN_LOOK_AT_M, "up": SCENE_DRIVEN_UP}
    options.update(pose)
    if isinstance(config, Radar):
        return config.replace(**options)
    return Radar.from_dict(dict(config), **options)


def empty_world():
    """A Core ``Scene`` with no structures at all.

    A point-target accuracy test wants exactly the free-space round trip and
    nothing else, so it declares a world with nothing in it and asks for
    ``los=True`` at ``reflections=0``. Putting a wall somewhere harmless
    instead would make every accuracy number depend on the claim that it really
    was harmless.
    """

    from witwin.core import Scene

    return Scene(structures=(), endpoints=[])


class PointTargetFrame:
    """One simulated frame plus the processing metadata that describes it.

    ``cube`` is ``[TX, RX, slow, fast]`` - the frame axis is already indexed off
    - and ``axes`` is the :class:`ProcessingAxes` record every processing stage
    reads. They are published together because a cube without its axes record
    cannot be turned into metres, and rebuilding the record per stage is how two
    stages end up describing different arrays.

    Metadata comes from ``Result.frame_synthesis()`` and the
    original waveform spec. No second synthesis can freeze a moving frame or
    replace its frontend output while constructing processing metadata.
    """

    def __init__(self, result, cube, axes, array, synthesis):
        self.result = result
        self.cube = cube
        self.axes = axes
        self.array = array
        self.synthesis = synthesis

    def processing_cube(self):
        from witwin.radar.processing import ProcessingCube

        return ProcessingCube(data=self.cube, axes=self.axes)

    def assert_axes_describe_the_cube(self):
        import torch

        from witwin.radar.processing import ProcessingCube

        packed = ProcessingCube.from_synthesis(self.synthesis, self.axes)
        assert torch.equal(packed.data, self.cube)

    def range_doppler(self, *, window="hann"):
        from witwin.radar.processing import range_doppler_map, range_profile

        range_window = "rectangular" if self.axes.output_domain == "spectrum" else window
        return range_doppler_map(range_profile(self.processing_cube(), window=range_window), window=window)

    def range_profile_db(self, *, window="hann"):
        """Peak-over-Doppler magnitude per range bin, summed over the array."""

        rd = self.range_doppler(window=window)
        return rd.data.abs().sum(dim=(0, 1)).max(dim=0).values

    def combined_map(self, *, window="hann"):
        """The coherently combined ``[doppler, range]`` map the detector reads."""

        rd = self.range_doppler(window=window)
        return rd.data.reshape(self.array.sensor_pair_count, *rd.data.shape[-2:]).sum(dim=0)

    def point_cloud(self, *, window="hann", pfa=1e-2, max_points=64, **options):
        from witwin.radar.processing import ca_cfar, point_cloud

        rd = self.range_doppler(window=window)
        combined = rd.data.reshape(self.array.sensor_pair_count, *rd.data.shape[-2:]).sum(dim=0)
        cells = ca_cfar(combined.abs(), guard_cells=(1, 2), training_cells=(2, 3), pfa=pfa)
        return point_cloud(cells, rd, self.array, max_points=max_points, **options)


def _target_tensors(radar, targets):
    """Split ``targets`` into local position and velocity tensors."""

    import torch

    positions = []
    velocities = []
    moving = False
    for entry in targets:
        if len(entry) == 2 and not isinstance(entry[0], (int, float)):
            position, velocity = entry
            moving = moving or any(float(value) != 0.0 for value in velocity)
        else:
            position, velocity = entry, (0.0, 0.0, 0.0)
        positions.append([float(value) for value in position])
        velocities.append([float(value) for value in velocity])
    # ``reshape(-1, 3)`` so that an EMPTY target list is a genuine (0, 3)
    # tensor rather than a rank-1 empty one: the empty case has to reach the
    # production refusal, not die in a shape error here.
    local_positions = torch.tensor(positions, dtype=torch.float32, device=radar.device).reshape(-1, 3)
    local_velocities = torch.tensor(velocities, dtype=torch.float32, device=radar.device).reshape(-1, 3)
    return local_positions, local_velocities, moving


def simulate_point_targets(radar, targets, *, sigma_m2=1.0):
    """Simulate one frame of free-space point targets through ``Radar.simulate``.

    ``targets`` is a sequence of either a local ``(x, y, z)`` position or a
    ``(position, velocity)`` pair, both in the radar's LOCAL frame and in metres
    (per second). Every target carries the same cross section, because the
    two-way join takes ONE scatter response for the whole batch - a per-target
    strength is a different capability and inventing it here would put a
    per-row response nothing in production publishes into the fixture.

    Returns a :class:`PointTargetFrame`.
    """

    from support.pose import world_from_local_points, world_from_local_vectors

    from witwin.radar import Motion, PointTargets
    from witwin.radar.processing import ArrayGeometry, ProcessingAxes

    local_positions, local_velocities, moving = _target_tensors(radar, targets)
    world_positions = world_from_local_points(radar, local_positions)
    world_velocity = world_from_local_vectors(radar, local_velocities)

    def linear_trajectory(time_s):
        """The same material points at ``time_s``, moving at a constant rate.

        A plain callable returning POSITIONS: the site velocity is never
        differenced into physics, so publishing one here would only invite a
        second owner of the delay rate.
        """

        return world_positions + time_s * world_velocity

    result = radar.simulate(
        empty_world(),
        PointTargets(positions=world_positions, rcs=sigma_m2, trajectory=linear_trajectory if moving else None),
        times=(0.0,),
        los=True,
        reflections=0,
        motion=Motion.chirp(),
    )
    cube = result.cube
    synthesis = result.frame_synthesis()

    axes = ProcessingAxes.from_synthesis(synthesis, radar.waveform_spec(), radar.system_config.sensors.array)
    return PointTargetFrame(
        result=result,
        cube=cube[0] if cube.dim() == 5 else cube,
        axes=axes,
        array=ArrayGeometry.from_axes(axes),
        synthesis=synthesis,
    )
