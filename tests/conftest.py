"""
Pytest configuration and shared fixtures for the radar test suite.

Run:
    cd radar
    pytest tests/                         # CPU-only tests
    pytest tests/ --gpu                   # include GPU tests (needs CUDA)
    pytest tests/sigproc/ -v              # single subfolder
"""

import sys
import os

import numpy as np
import pytest

# Ensure witwin.radar is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# pytest plugins
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    try:
        parser.addoption(
            "--gpu", action="store_true", default=False,
            help="Run GPU-only tests (solver cross-validation, end-to-end validation)",
        )
    except ValueError:
        pass


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

STANDARD_CONFIG = {
    "num_tx": 3, "num_rx": 4,
    "fc": 77e9, "slope": 60.012,
    "adc_samples": 256, "adc_start_time": 6,
    "sample_rate": 4400, "idle_time": 7, "ramp_end_time": 58,
    "chirp_per_frame": 128, "frame_per_second": 10,
    "num_doppler_bins": 128, "num_range_bins": 256,
    "num_angle_bins": 64, "power": 12,
    "tx_loc": [[0, 0, 0], [2, 0, 0], [0, 1, 0]],
    "rx_loc": [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]],
}

FAST_CONFIG = {
    **STANDARD_CONFIG,
    "chirp_per_frame": 32,
    "num_doppler_bins": 32,
}

MINIMAL_CONFIG = {
    "num_tx": 1, "num_rx": 1,
    "fc": 77e9, "slope": 60.012,
    "adc_samples": 256, "adc_start_time": 0,
    "sample_rate": 4400, "idle_time": 7, "ramp_end_time": 58,
    "chirp_per_frame": 2, "frame_per_second": 10,
    "num_doppler_bins": 2, "num_range_bins": 256,
    "num_angle_bins": 64, "power": 12,
    "tx_loc": [[0, 0, 0]],
    "rx_loc": [[0, 0, 0]],
}


# ---------------------------------------------------------------------------
# CPU-only mock for sigproc tests that need FrameConfig / PointCloudProcessConfig
# ---------------------------------------------------------------------------

class MockRadar:
    """Lightweight CPU-only mock providing the attributes needed by sigproc code."""

    def __init__(self, config=None):
        import torch
        from witwin.radar import RadarConfig

        self.c0 = 299792458
        if config is None:
            config = STANDARD_CONFIG
        if isinstance(config, RadarConfig):
            self.config = config
        else:
            self.config = RadarConfig.from_dict(dict(config))
        cfg = self.config

        from witwin.radar.config import RadarSystemConfig

        self._lambda = self.c0 / cfg.fc
        antenna_spacing = self._lambda / 2
        self.tx_loc = torch.tensor(cfg.tx_loc, dtype=torch.float32) * antenna_spacing
        self.rx_loc = torch.tensor(cfg.rx_loc, dtype=torch.float32) * antenna_spacing

        # The mock derives its axes from the SAME record the real radar does,
        # rather than repeating six formulas that then drift apart. `sigproc`
        # reads `radar.axes` and this is what makes the mock a real duck-type of
        # that read rather than a lookalike.
        self.system_config = RadarSystemConfig.from_radar_config(cfg)
        self.axes = self.system_config.axes(device="cpu")
        self.range_resolution = self.axes.range_resolution
        self.max_range = self.axes.max_range
        self.doppler_resolution = self.axes.doppler_resolution
        self.max_doppler = self.axes.max_doppler
        self.ranges = self.axes.ranges
        self.velocities = self.axes.velocities

        self.gain = 1.0

    # Convenience accessors for sigproc/test code that still needs flat fields
    @property
    def num_tx(self) -> int:
        return self.config.num_tx

    @property
    def num_rx(self) -> int:
        return self.config.num_rx

    @property
    def chirp_per_frame(self) -> int:
        return self.config.chirp_per_frame

    @property
    def adc_samples(self) -> int:
        return self.config.adc_samples

    @property
    def num_angle_bins(self) -> int:
        return self.config.num_angle_bins

    @property
    def idle_time(self) -> float:
        return self.config.idle_time

    @property
    def ramp_end_time(self) -> float:
        return self.config.ramp_end_time

    @property
    def num_doppler_bins(self) -> int:
        return self.config.num_doppler_bins

    @property
    def num_range_bins(self) -> int:
        return self.config.num_range_bins


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def standard_config():
    from witwin.radar import RadarConfig
    return RadarConfig.from_dict(STANDARD_CONFIG)


@pytest.fixture
def fast_config():
    from witwin.radar import RadarConfig
    return RadarConfig.from_dict(FAST_CONFIG)


@pytest.fixture
def minimal_config():
    from witwin.radar import RadarConfig
    return RadarConfig.from_dict(MINIMAL_CONFIG)


@pytest.fixture
def mock_radar():
    return MockRadar(STANDARD_CONFIG)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mag_correlation(a, b):
    """Pearson correlation of magnitudes (works for numpy and torch)."""
    a_m = np.abs(np.asarray(a).ravel()).astype(np.float64)
    b_m = np.abs(np.asarray(b).ravel()).astype(np.float64)
    a_c = a_m - a_m.mean()
    b_c = b_m - b_m.mean()
    denom = np.linalg.norm(a_c) * np.linalg.norm(b_c)
    if denom < 1e-30:
        return 1.0 if np.linalg.norm(a_c) < 1e-30 and np.linalg.norm(b_c) < 1e-30 else 0.0
    return float(np.dot(a_c, b_c) / denom)


def complex_correlation(a, b):
    """Normalized complex inner-product correlation."""
    a_c = np.asarray(a).ravel().astype(np.complex128)
    b_c = np.asarray(b).ravel().astype(np.complex128)
    denom = np.linalg.norm(a_c) * np.linalg.norm(b_c)
    if denom < 1e-30:
        return 1.0 if np.linalg.norm(a_c) < 1e-30 and np.linalg.norm(b_c) < 1e-30 else 0.0
    return float(np.abs(np.vdot(a_c, b_c)) / denom)


def peak_ratio(a, b):
    """Ratio of maximum magnitudes, matching verify.py style checks."""
    a_peak = float(np.abs(np.asarray(a)).max())
    b_peak = float(np.abs(np.asarray(b)).max())
    if a_peak < 1e-30 or b_peak < 1e-30:
        return 1.0 if max(a_peak, b_peak) < 1e-30 else 0.0
    return b_peak / a_peak


def make_static_interpolator(pos, sigma=1.0):
    """Create an interpolator for a static target (GPU tensors).

    LEGACY, SCHEDULED FOR DELETION. Nothing on the scene-driven route consumes
    an interpolator: a target is a ``ScatterSitePolicy`` site in a Core world,
    not a callable that reports positions at a time. The two remaining callers
    are ``tests/solvers/test_mimo_cross.py`` and ``tests/solvers/test_solver_edge.py``,
    both of which the Phase-11 deletion stage removes with the Dirichlet route;
    this helper goes with them in the same commit. It is kept here rather than
    deleted now only so that no intermediate commit fails collection.

    The replacement is :func:`simulate_point_targets`.
    """
    import torch
    pos_t = torch.tensor([pos], dtype=torch.float32, device="cuda")
    sigma_t = torch.tensor([sigma], dtype=torch.float32, device="cuda")

    def interp(t):
        return sigma_t, pos_t

    return interp


# ---------------------------------------------------------------------------
# The scene-driven point-target fixtures
# ---------------------------------------------------------------------------
#
# ``make_moving_interpolator`` used to live beside the helper above and had
# exactly one consumer, ``tests/validation/``. Those tests now drive
# ``Radar.simulate``, which reads a Core world rather than a callable, so the
# moving fixture is not "ported" - it is replaced by a declaration of where the
# targets are and how fast they move.
#
# Two conventions are fixed here once so that no validation test restates them:
#
# * **The radar looks along world +x** with the default up. Targets are still
#   authored in the radar's LOCAL frame - ``[0, 0, -d]`` is still "d metres
#   straight ahead" - and ``Radar.world_from_local_points`` does the transform,
#   so the test text did not have to change and the production pose transform is
#   on the path rather than mirrored.
#
#   The old default pose looked along ``-z``, which on this route publishes an
#   exactly ZERO transport: the default endpoint polarization is ``(0, 0, 1)``,
#   a field is transverse, and a look direction parallel to the polarization has
#   no transverse component to carry. Channel is right to publish zero and there
#   is no tolerance that recovers it - the fix is a pose whose boresight is not
#   the polarization axis.
#
# * **Intra-frame Doppler is opened by the caller.** ``Radar.simulate`` has no
#   ``velocities=`` keyword (a named Phase-11 scope boundary), so a moving target
#   is driven by dualising the site tensor with
#   ``witwin.radar.propagation.kinematics.two_way_duals`` and asking for
#   ``ad_mode="jvp"``. The site policy passes its tensor through by identity, so
#   the dual reaches the legs and the join publishes ``delay_rate``, which is
#   what the waveform kernel's slow-time carrier consumes. Nothing here computes
#   a Doppler shift.

#: The boresight this suite poses its radars along, in world coordinates.
SCENE_DRIVEN_LOOK_AT_M = (1.0, 0.0, 0.0)

#: The world up vector that goes with it.
SCENE_DRIVEN_UP = (0.0, 1.0, 0.0)


def make_scene_radar_or_skip(config, **pose):
    """A :class:`Radar` posed along :data:`SCENE_DRIVEN_LOOK_AT_M`.

    Same skip contract as :func:`make_radar_or_skip`; the only difference is the
    pose, and the pose is the whole reason this exists rather than a keyword on
    the other one - see the note above about the polarization null.
    """

    from witwin.radar import Radar, RadarConfig

    if not isinstance(config, RadarConfig):
        config = RadarConfig.from_dict(dict(config))
    options = {
        "position": (0.0, 0.0, 0.0),
        "target": SCENE_DRIVEN_LOOK_AT_M,
        "up": SCENE_DRIVEN_UP,
    }
    options.update(pose)
    try:
        return Radar(config, **options)
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        pytest.skip(f"radar runtime unavailable: {exc}")


def empty_world():
    """A Core ``Scene`` with no structures at all.

    A point-target accuracy test wants exactly the free-space round trip and
    nothing else, so it declares a world with nothing in it and asks for
    ``components={"los"}`` at ``max_depth=0``. Putting a wall somewhere harmless
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

    The axes record is derived from a SECOND synthesis of the same composed
    rows. ``RadarSimulationResult`` publishes the stacked cube and the waveform
    conventions but not the rank-3 ``SynthesisResult`` that
    ``ProcessingAxes.from_synthesis`` reads, so a consumer that wants
    ``processing/`` on a ``Radar.simulate`` product has to re-synthesize one
    frame to get it. That is a real gap in the entry point and it is recorded as
    one; :meth:`assert_axes_describe_the_cube` pins that the two agree BITWISE,
    so the workaround cannot quietly start describing something else.
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
        from witwin.radar.processing import range_doppler, range_profile

        return range_doppler(
            range_profile(self.processing_cube(), window=window), window=window
        )

    def range_profile_db(self, *, window="hann"):
        """Peak-over-Doppler magnitude per range bin, summed over the array."""

        rd = self.range_doppler(window=window)
        return rd.data.abs().sum(dim=(0, 1)).max(dim=0).values

    def combined_map(self, *, window="hann"):
        """The coherently combined ``[doppler, range]`` map the detector reads."""

        rd = self.range_doppler(window=window)
        return rd.data.reshape(
            self.array.sensor_pair_count, *rd.data.shape[-2:]
        ).sum(dim=0)

    def point_cloud(self, *, window="hann", pfa=1e-2, max_points=64, **options):
        from witwin.radar.processing import ca_cfar_fast, point_cloud

        rd = self.range_doppler(window=window)
        combined = rd.data.reshape(
            self.array.sensor_pair_count, *rd.data.shape[-2:]
        ).sum(dim=0)
        cells = ca_cfar_fast(
            combined.abs(), guard_cells=(1, 2), training_cells=(2, 3), pfa=pfa
        )
        return point_cloud(
            cells, rd, self.axes, self.array, max_points=max_points, **options
        )


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
    local_positions = torch.tensor(
        positions, dtype=torch.float32, device=radar.device
    ).reshape(-1, 3)
    local_velocities = torch.tensor(
        velocities, dtype=torch.float32, device=radar.device
    ).reshape(-1, 3)
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

    import torch

    from witwin.radar import ScatterSitePolicy
    from witwin.radar.processing import ArrayGeometry, ProcessingAxes
    from witwin.radar.propagation import kinematics as kin
    from witwin.radar.scattering import ScalarRcsResponse
    from witwin.radar.synthesis import SlowTimeMode

    local_positions, local_velocities, moving = _target_tensors(radar, targets)
    world_positions = radar.world_from_local_points(local_positions)
    response = ScalarRcsResponse.from_rcs(
        sigma_m2,
        reference_frequency_hz=radar.system_config.propagation.reference_frequency_hz,
        device=radar.device,
    )

    def solve(sites, ad_mode):
        return radar.simulate(
            empty_world(),
            times=(0.0,),
            response=response,
            sites=ScatterSitePolicy.explicit(sites),
            components=frozenset({"los"}),
            max_depth=0,
            ad_mode=ad_mode,
        )

    if moving:
        track = kin.Kinematics(
            positions_m=world_positions,
            velocities_m_per_s=radar.world_from_local_vectors(local_velocities),
        )
        with kin.two_way_duals(sites=track) as duals:
            result = solve(duals.sites, "jvp")
            # The published cube is a dual inside the level and a bare tensor
            # outside it. Unpacking the primal here is what makes the two cases
            # return the same type; the TANGENT of the cube is not the Doppler -
            # the Doppler is already inside the primal, carried by the join's
            # ``delay_rate`` and consumed by the waveform kernel's slow-time
            # carrier.
            cube = _primal(result.cube).detach().clone()
            synthesis = radar.synthesize(
                radar.last_radar_paths,
                slow_time_mode=SlowTimeMode.FROZEN_WEIGHT_WITH_CARRIER_RATE,
            )
            synthesis = _detached_synthesis(synthesis)
    else:
        result = solve(world_positions, "none")
        cube = result.cube
        synthesis = radar.synthesize(
            radar.last_radar_paths,
            slow_time_mode=SlowTimeMode.FROZEN_WEIGHT_WITH_CARRIER_RATE,
        )

    axes = ProcessingAxes.from_synthesis(
        synthesis,
        radar.system_config.waveform_spec(),
        radar.system_config.sensors.array,
    )
    return PointTargetFrame(
        result=result,
        cube=cube[0] if cube.dim() == 5 else cube,
        axes=axes,
        array=ArrayGeometry.from_axes(axes),
        synthesis=synthesis,
    )


def _primal(tensor):
    import torch.autograd.forward_ad as forward_ad

    return forward_ad.unpack_dual(tensor).primal


def _detached_synthesis(synthesis):
    """Lift one synthesis result out of a forward-AD level."""

    import dataclasses

    return dataclasses.replace(
        synthesis, cube=_primal(synthesis.cube).detach().clone()
    )


def make_radar_or_skip(config):
    """Construct a Radar or skip when the local runtime/toolchain is missing."""
    from witwin.radar import Radar, RadarConfig

    if not isinstance(config, RadarConfig):
        config = RadarConfig.from_dict(dict(config))
    try:
        return Radar(config)
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        pytest.skip(f"radar runtime unavailable: {exc}")
