"""The world half and the instrument half, and that splitting them moved nothing.

``Radar.trace`` composes the round trips; ``Radar.echo`` synthesizes them. The
strongest statement this file makes is the one that justifies the split at all:
``echo(trace(...))`` publishes the SAME BITS as ``simulate(...)`` on every
motion kind, in both FMCW output domains, with and without a receive chain, and
with the common-oscillator phase noise that acts on rows rather than on samples.
``torch.equal`` rather than a tolerance, because a cut that moved a number in
the last bit is a cut through an equation rather than through a seam.

The rest is what a numerical check cannot see:

* one trace echoed by two different receivers gives each receiver's own answer,
  which is the capability the split exists to provide;
* a radar the rows do not describe is refused by name rather than replayed
  against a schedule that no longer fits it;
* an adaptive trace retains its accepted partition, not its schedule, so its
  row count is the probes and not the observations.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

pytest.importorskip("witwin.channel")

from witwin.core import AntennaState, Mesh, PhysicalMaterial, Scene, Structure  # noqa: E402
from witwin.core.identity import reserve_antenna_id  # noqa: E402

from witwin.radar import Adc, Motion, Noise, Paths, PointTargets, Radar  # noqa: E402

pytestmark = pytest.mark.gpu

CONFIG = {
    "num_tx": 2,
    "num_rx": 2,
    "fc": 77e9,
    "slope": 60.0,
    "adc_samples": 16,
    "adc_start_time": 1,
    "sample_rate": 2000,
    "idle_time": 1,
    "ramp_end_time": 20,
    "chirp_per_frame": 4,
    "power": 12,
    "tx_loc": [[0, 0, 0], [2, 0, 0]],
    "rx_loc": [[-3, 0, 0], [-2, 0, 0]],
}

TARGET_M = (0.0, 0.0, -3.0)
WALL_PLANE_Z_M = -5.0
WALL_HALF_EXTENT_M = 2.0
SITE_SPEED_M_PER_S = 5.0


def _radar() -> Radar:
    return Radar.from_dict(CONFIG, position=(0.0, 0.0, 0.0), look_at=(0.0, 0.0, -1.0))


def _wall_scene() -> Scene:
    """A wall, so the trace carries multipath rows and not only the direct one."""

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
        endpoints=[AntennaState(reserve_antenna_id(990101), "tx", torch.tensor((0.0, 0.0, 0.0), dtype=torch.float32))],
    )


def _empty_scene() -> Scene:
    """No structures, which is what lets the adaptive route certify its family."""

    return Scene(
        structures=(),
        endpoints=[AntennaState(reserve_antenna_id(990201), "tx", torch.tensor((0.0, 0.0, 0.0), dtype=torch.float32))],
    )


def _moving(time_s: float) -> torch.Tensor:
    return torch.tensor(
        [[0.0, 0.0, TARGET_M[2] + SITE_SPEED_M_PER_S * float(time_s)]], dtype=torch.float32, device="cuda"
    )


def _still() -> PointTargets:
    return PointTargets(positions=[TARGET_M], rcs=1.0)


def _mover() -> PointTargets:
    return PointTargets(positions=[TARGET_M], rcs=1.0, trajectory=_moving)


#: Every route a frame can take from world to cube, and a session that exercises
#: it. ``still`` worlds go against the wall so the rows include multipath.
ROUTES = {
    "static_ideal": ("wall", "still", {"times": (0.0, 0.1), "reflections": 1}, {}),
    "static_beat": ("wall", "still", {"times": (0.0,), "reflections": 1}, {"output": "beat"}),
    "static_receiver": (
        "wall",
        "still",
        {"times": (0.0, 0.1), "reflections": 1},
        {"noise": Noise(figure=8.0), "adc": Adc(bits=10, full_scale=1e-6), "seed": 11},
    ),
    "chirp": ("empty", "mover", {"times": (0.0,), "reflections": 0, "motion": Motion.chirp()}, {}),
    "adc": ("empty", "mover", {"times": (0.0,), "reflections": 0, "motion": Motion.adc()}, {}),
    "adaptive": (
        "empty",
        "mover",
        {"times": (0.0,), "reflections": 0, "motion": Motion.adaptive(max_interval=1e-3)},
        {},
    ),
    "adc_oscillator_noise": (
        "empty",
        "mover",
        {"times": (0.0,), "reflections": 0, "motion": Motion.adc()},
        {"noise": Noise(phase_density=-80.0, phase_offset=1e5), "seed": 5},
    ),
}


def _build(route: str):
    scene_kind, target_kind, session, radar_fields = ROUTES[route]
    radar = _radar()
    output = radar_fields.pop("output", None)
    if output is not None:
        radar = radar.replace(waveform=dataclasses.replace(radar.waveform, output=output))
    if radar_fields:
        radar = radar.replace(**radar_fields)
    scene = _wall_scene() if scene_kind == "wall" else _empty_scene()
    targets = _still() if target_kind == "still" else _mover()
    return radar, scene, targets, session


# ---------------------------------------------------------------------------
# The statement that justifies the split
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("route", sorted(ROUTES))
def test_simulate_is_echo_of_trace_bit_for_bit(route: str) -> None:
    radar, scene, targets, session = _build(route)
    fused = radar.simulate(scene, targets, **session)
    split = radar.echo(radar.trace(scene, targets, **session))

    assert torch.equal(fused.cube, split.cube)
    assert fused.cube.dtype == split.cube.dtype
    for name in ("kind", "axis_names", "phasor", "time_dependence", "output_domain", "times_s", "motion_sampling"):
        assert getattr(fused, name) == getattr(split, name), name
    # The metadata record holds tensors, so it is compared by the scalars that
    # decide what a bin means rather than by equality on the whole record.
    for name in ("range_bin_m", "velocity_bin_mps", "doppler_sign", "reference_frequency_hz"):
        assert getattr(fused.axes, name) == getattr(split.axes, name), name
    assert fused.path_set_complete == split.path_set_complete
    assert fused.motion_sampling_exhaustive == split.motion_sampling_exhaustive
    assert fused.epochs == split.epochs
    assert fused.compile_count == split.compile_count
    assert fused.discovery_count == split.discovery_count


@pytest.mark.parametrize("route", sorted(ROUTES))
def test_a_trace_publishes_the_completeness_its_fused_run_does(route: str) -> None:
    """The two statements belong to the world half, so they are known before echo."""

    radar, scene, targets, session = _build(route)
    paths = radar.trace(scene, targets, **session)
    fused = radar.simulate(scene, targets, **session)
    assert paths.path_set_complete == fused.path_set_complete
    assert paths.motion_sampling_exhaustive == fused.motion_sampling_exhaustive
    assert paths.times == fused.times_s
    assert paths.kind == fused.motion_sampling


# ---------------------------------------------------------------------------
# What the split is for
# ---------------------------------------------------------------------------


def test_one_trace_serves_two_receivers() -> None:
    """The capability the split exists to provide.

    Each echo must equal what that radar's own fused run would have produced,
    which is the difference between reusing a trace and approximating one.
    """

    ideal = _radar()
    noisy = ideal.replace(noise=Noise(figure=8.0), adc=Adc(bits=10, full_scale=1e-6), seed=11)
    session = {"times": (0.0,), "reflections": 1}

    paths = ideal.trace(_wall_scene(), _still(), **session)
    quiet = ideal.echo(paths)
    loud = noisy.echo(paths)

    assert not torch.equal(quiet.cube, loud.cube)
    assert torch.equal(quiet.cube, ideal.simulate(_wall_scene(), _still(), **session).cube)
    assert torch.equal(loud.cube, noisy.simulate(_wall_scene(), _still(), **session).cube)


def test_a_frame_slice_echoes_to_that_frame() -> None:
    radar = _radar()
    session = {"times": (0.0, 0.1), "reflections": 1}
    paths = radar.trace(_wall_scene(), _still(), **session)
    whole = radar.echo(paths)

    assert paths.frame_count == 2
    for index in range(2):
        sliced = paths.frame(index)
        assert sliced.frame_count == 1
        assert sliced.times == (paths.times[index],)
        assert torch.equal(radar.echo(sliced).cube[0], whole.cube[index])


def test_echo_does_not_consume_the_paths() -> None:
    """A Paths is a record, not a stream: echoing it twice gives the same cube."""

    radar = _radar()
    paths = radar.trace(_wall_scene(), _still(), times=(0.0,), reflections=1)
    assert torch.equal(radar.echo(paths).cube, radar.echo(paths).cube)


# ---------------------------------------------------------------------------
# What a Paths knows about itself
# ---------------------------------------------------------------------------


def test_an_adaptive_trace_retains_its_probes_and_not_its_schedule() -> None:
    """The interpolation table is the whole point: far fewer rows than instants.

    The exhaustive route stores one composed batch per ADC instant; the
    adaptive one stores the evaluated probes and the weights that read them, so
    its retained row count is a fraction of the schedule it answers.
    """

    radar = _radar()
    session = {"times": (0.0,), "reflections": 0}
    exhaustive = radar.trace(_empty_scene(), _mover(), motion=Motion.adc(), **session)
    adaptive = radar.trace(_empty_scene(), _mover(), motion=Motion.adaptive(max_interval=1e-3), **session)

    assert adaptive.observation_count == exhaustive.observation_count
    assert adaptive.row_count < exhaustive.row_count
    assert adaptive.motion_sampling_exhaustive is False
    assert exhaustive.motion_sampling_exhaustive is True
    diagnostics = adaptive.adaptive_diagnostics[0]
    assert diagnostics["evaluations"] < diagnostics["observation_count"]
    # A trace has synthesized nothing, so it cannot report a batch count.
    assert "synthesis_batches" not in diagnostics


def test_a_trace_records_the_request_its_completeness_is_relative_to() -> None:
    radar = _radar()
    paths = radar.trace(_wall_scene(), _still(), times=(0.0,), los=True, reflections=1, grad="none")
    assert paths.los is True
    assert paths.reflections == 1
    assert paths.grad == "none"
    assert paths.carrier == radar.carrier
    assert paths.waveform == radar.waveform

    direct = radar.trace(_wall_scene(), _still(), times=(0.0,), los=True, reflections=0)
    assert direct.reflections == 0
    assert direct.row_count < paths.row_count


def test_rows_publishes_a_typed_batch_and_refuses_an_interpolated_instant() -> None:
    from witwin.radar.paths import RadarPathBatch

    radar = _radar()
    paths = radar.trace(_wall_scene(), _still(), times=(0.0,), reflections=1)
    assert isinstance(paths.rows(), RadarPathBatch)

    adaptive = radar.trace(
        _empty_scene(), _mover(), times=(0.0,), reflections=0, motion=Motion.adaptive(max_interval=1e-3)
    )
    with pytest.raises(IndexError, match="one interpolation table"):
        adaptive.rows(observation=3)


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_echo_refuses_a_radar_the_rows_do_not_describe() -> None:
    radar = _radar()
    paths = radar.trace(_wall_scene(), _still(), times=(0.0,), reflections=1)

    with pytest.raises(ValueError, match="the carrier is the reference frequency"):
        radar.replace(carrier=76e9).echo(paths)

    wider = Radar.from_dict({**CONFIG, "num_rx": 3, "rx_loc": [[-3, 0, 0], [-2, 0, 0], [-1, 0, 0]]})
    with pytest.raises(ValueError, match="pair partition is frozen"):
        wider.echo(paths)


def test_echo_refuses_oscillator_phase_noise_on_paths_without_adc_instants() -> None:
    """The phase is placed at absolute ADC time, so it needs ADC observations."""

    radar = _radar()
    chirp_paths = radar.trace(_empty_scene(), _mover(), times=(0.0,), reflections=0, motion=Motion.chirp())
    noisy = radar.replace(noise=Noise(phase_density=-80.0, phase_offset=1e5), seed=5)
    with pytest.raises(ValueError, match="has no ADC instants"):
        noisy.echo(chirp_paths)


def test_echo_refuses_something_that_is_not_a_paths() -> None:
    radar = _radar()
    result = radar.simulate(_wall_scene(), _still(), times=(0.0,), reflections=1)
    with pytest.raises(TypeError, match="the Paths that trace returned"):
        radar.echo(result)


def test_paths_is_a_frozen_record() -> None:
    radar = _radar()
    paths = radar.trace(_wall_scene(), _still(), times=(0.0,), reflections=1)
    assert isinstance(paths, Paths)
    with pytest.raises(dataclasses.FrozenInstanceError):
        paths.kind = "adc"
