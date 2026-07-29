"""The scene-driven entry point (Phase 11 work items 1 and 2).

``Radar.simulate`` was a refusal until this phase, so there is no pre-existing
numerical output to preserve and nothing here is a regression pin. What there IS
is a de-facto reference implementation - ``tests/support/multi_endpoint_driver``,
which assembled the same production owners by hand - and the strongest statement
this file makes is that the production entry reproduces it EXACTLY on the shared
half: identical composed row identity and bitwise identical round-trip delays,
with the transport differing only by the transmit power the two declare.

The rest is what a numerical check cannot see, because a wrong answer in any of
it still produces a plausible cube:

* the entry returns a typed record, not a bare tensor;
* the four diagnostics are typed, describe ONE frame, and are ``None`` before
  the first call and after a failed one;
* the pair partition is this array's TX x RX grid in the composer's own
  sink-major rank;
* a topology is discovered exactly ONCE per epoch, and the epoch cadence is the
  declared one rather than whatever the geometry version happens to do;
* ``simulate_group`` is gone rather than refusing, because a permanent refusal
  is itself a legacy shim.
"""

from __future__ import annotations

import dataclasses
import inspect

import pytest
import torch

pytest.importorskip("witwin.channel")

from support import multi_endpoint_driver as drv  # noqa: E402
from support import multi_endpoint_geometry as geo  # noqa: E402
from support import multi_endpoint_world as world  # noqa: E402

import witwin.radar as wr  # noqa: E402
from witwin.radar import Radar  # noqa: E402
from witwin.radar.paths import RadarPathBatch  # noqa: E402
from witwin.radar.propagation import RadarLegBatch, RadarPropagationLegs  # noqa: E402
from witwin.radar.scattering import ScalarRcsResponse  # noqa: E402
from witwin.radar.sensors import ISOTROPIC_PATTERN  # noqa: E402
from witwin.radar.simulation import (  # noqa: E402  # noqa: E402
    RadarSimulationResult,
    ScatterSitePolicy,
    StableIdAllocator,
)

pytestmark = pytest.mark.gpu

#: The radar looks along +x, so its two half-wavelength elements sit along the
#: world z axis. That is not the multi-endpoint fixture's transmitter geometry
#: and it does not have to be: the fixture wall and sites are what make the
#: reflection rows interesting, and the front end is this radar's own.
LOOK_AT_M = (1.0, 0.0, 0.0)

SITE_POSITIONS_M = (geo.SITE_P_POSITION_M, geo.SITE_Q_POSITION_M)


def _radar() -> Radar:
    config = dict(geo.FIXTURE_RADAR_CONFIG)
    config["antenna_pattern"] = {
        "kind": ISOTROPIC_PATTERN.kind,
        "x_angles_deg": list(ISOTROPIC_PATTERN.x_angles_deg),
        "y_angles_deg": list(ISOTROPIC_PATTERN.y_angles_deg),
        "x_values": list(ISOTROPIC_PATTERN.x_values),
        "y_values": list(ISOTROPIC_PATTERN.y_values),
    }
    return Radar(config, position=(0.0, 0.0, 0.0), target=LOOK_AT_M)


def _response(radar: Radar, *, requires_grad: bool = False) -> ScalarRcsResponse:
    return ScalarRcsResponse.from_values(
        drv.FIXTURE_AMPLITUDE, drv.FIXTURE_PHASE_RAD, device=radar.device, requires_grad=requires_grad
    )


def _sites(radar: Radar, *, requires_grad: bool = False) -> ScatterSitePolicy:
    positions = torch.tensor(SITE_POSITIONS_M, dtype=torch.float32, device=radar.device).requires_grad_(requires_grad)
    return ScatterSitePolicy.explicit(positions)


def _static_scene():
    scene, mesh = world.make_scene()
    world.assert_world_coordinates_survived(mesh)
    return scene


def _simulate(radar: Radar, scene, times, **options) -> RadarSimulationResult:
    return radar.simulate(scene, times=times, response=_response(radar), sites=_sites(radar), **options)


# ---------------------------------------------------------------------------
# The product
# ---------------------------------------------------------------------------


def test_simulate_runs_the_whole_pipeline_and_publishes_a_frame_cube():
    """Core Scene -> CompiledScene -> propagation -> two-way -> synthesis."""

    radar = _radar()
    result = _simulate(radar, _static_scene(), (0.0, 1.0e-3, 2.0e-3))

    array = radar.system_config.sensors.array
    waveform = radar.system_config.waveform
    assert result.cube.shape == (3, array.num_tx, array.num_rx, waveform.chirp_per_frame, waveform.adc_samples)
    assert result.cube.dtype == torch.complex64
    assert result.cube.device.type == radar.device.type
    assert result.axes == ("frame", "tx", "rx", "chirp", "range_bin")
    assert result.kind == "fmcw"
    assert result.times_s == (0.0, 1.0e-3, 2.0e-3)
    assert result.frame_count == 3
    assert result.reference_frequency_hz == geo.REFERENCE_FREQUENCY_HZ
    # The conventions come from the waveform owner rather than from this entry.
    from witwin.radar.synthesis.assembly import BEAT_PHASOR

    assert result.phasor == BEAT_PHASOR


def test_the_entry_returns_the_typed_record_rather_than_a_bare_tensor():
    """A tuple or a tensor would make every consumer re-derive the axes."""

    radar = _radar()
    result = _simulate(radar, _static_scene(), (0.0,))
    assert isinstance(result, RadarSimulationResult)
    assert dataclasses.is_dataclass(result)
    with pytest.raises(AttributeError):
        result.cube = torch.zeros(1)


def test_a_still_world_publishes_the_same_frame_at_every_instant():
    """Nothing moves, so nothing may change - not even by a float32 ULP.

    This is the calibration for every motion test below: a difference between
    two frames of a still world would mean the frame loop carries state it
    should not, and would make "the wall moved" unmeasurable.
    """

    radar = _radar()
    result = _simulate(radar, _static_scene(), (0.0, 5.0e-3))
    assert torch.equal(result.cube[0], result.cube[1])


# ---------------------------------------------------------------------------
# Agreement with the reference orchestration
# ---------------------------------------------------------------------------


def test_the_composed_rows_agree_with_the_reference_orchestration():
    """The production entry reproduces ``MultiEndpointSpike`` exactly.

    Same compiled world, same endpoint positions, same site excitation, same
    components and depth - so the geometry half must be BITWISE identical and
    the row identity must be the same sequence, not merely the same set. The
    transport differs by exactly one declared quantity: the spike excites its
    transmitters at ``geo.TX_POWER_W`` while the radar uses its own configured
    transmit power, and a Channel coefficient carries ``sqrt(P_tx)``. That
    ratio is checked rather than tolerated, because a constant ratio is a
    statement about which factor differs while a loose tolerance is not.
    """

    radar = _radar()
    allocator = StableIdAllocator()
    transmitter_ids, receiver_ids, site_ids = allocator.allocate(
        transmitter_count=int(radar.tx_pos.shape[0]),
        receiver_count=int(radar.rx_pos.shape[0]),
        site_count=len(SITE_POSITIONS_M),
    )
    _simulate(radar, _static_scene(), (0.0,))
    produced = radar.last_radar_paths

    spike = drv.MultiEndpointSpike(
        transmitters=tuple(zip(transmitter_ids, [tuple(row) for row in radar.tx_pos.tolist()], strict=True)),
        sites=tuple(zip(site_ids, SITE_POSITIONS_M, strict=True)),
        receivers=tuple(zip(receiver_ids, [tuple(row) for row in radar.rx_pos.tolist()], strict=True)),
    )
    reference, _, _ = spike.frame(response=_response(radar))

    assert produced.path_count == reference.path_count
    assert produced.sensor_pair_count == reference.sensor_pair_count
    for name in ("radar_source_id", "site_id", "radar_sink_id"):
        assert torch.equal(getattr(produced.topology, name), getattr(reference.topology, name)), name
    assert torch.equal(produced.total_delay_s, reference.total_delay_s)

    expected = (radar.system_config.sensors.tx_power.transmit_power_watts / geo.TX_POWER_W) ** 0.5
    ratio = produced.complex_transfer_ref.abs() / reference.complex_transfer_ref.abs()
    torch.testing.assert_close(ratio, torch.full_like(ratio, float(expected)), rtol=1e-5, atol=0.0)


# ---------------------------------------------------------------------------
# The four typed diagnostics (work item 2)
# ---------------------------------------------------------------------------


def test_the_diagnostics_are_none_before_the_first_simulate():
    """The pinned answer. Raising would make "has it run" a try/except."""

    radar = _radar()
    for name in ("last_snapshot", "last_compiled_scene", "last_propagation", "last_radar_paths", "last_result"):
        assert getattr(radar, name) is None, name


def test_the_four_diagnostics_are_typed_and_describe_the_last_frame():
    radar = _radar()
    result = _simulate(radar, _static_scene(), (0.0, 1.0e-3))

    from witwin.channel.scene import CompiledScene
    from witwin.core import SceneSnapshot

    assert isinstance(radar.last_snapshot, SceneSnapshot)
    assert isinstance(radar.last_compiled_scene, CompiledScene)
    assert isinstance(radar.last_propagation, RadarPropagationLegs)
    assert isinstance(radar.last_radar_paths, RadarPathBatch)

    # The LAST frame, named by its own time, not the first.
    assert float(radar.last_snapshot.time_s) == 1.0e-3
    assert radar.last_result is result
    assert radar.last_snapshot is result.last_snapshot
    assert radar.last_compiled_scene is result.last_compiled_scene
    assert radar.last_propagation is result.last_propagation
    assert radar.last_radar_paths is result.last_radar_paths


def test_the_leg_pair_is_typed_rather_than_a_tuple_or_a_dict():
    """``RadarPropagationLegs`` is what makes the pairing checkable."""

    radar = _radar()
    _simulate(radar, _static_scene(), (0.0,))
    legs = radar.last_propagation
    assert isinstance(legs.inbound, RadarLegBatch)
    assert isinstance(legs.outbound, RadarLegBatch)
    assert legs.slot_count == 1
    assert legs.device.type == radar.device.type
    assert not isinstance(legs, (tuple, dict))

    with pytest.raises(TypeError, match="outbound must be a RadarLegBatch"):
        RadarPropagationLegs(inbound=legs.inbound, outbound=object())


def test_a_failed_simulate_leaves_no_stale_diagnostics():
    """A stale world claiming to describe this radar is worse than nothing."""

    radar = _radar()
    _simulate(radar, _static_scene(), (0.0,))
    assert radar.last_radar_paths is not None
    with pytest.raises(ValueError):
        radar.simulate(_static_scene(), times=(), response=_response(radar), sites=_sites(radar))
    for name in ("last_snapshot", "last_compiled_scene", "last_propagation", "last_radar_paths", "last_result"):
        assert getattr(radar, name) is None, name


# ---------------------------------------------------------------------------
# Row and pair ordering
# ---------------------------------------------------------------------------


def test_the_pair_partition_is_this_arrays_tx_by_rx_grid():
    """The cube's pair axis is an ordered grid, not a scatter.

    ``validate_pair_ordering`` already refuses a partition that is not this
    array's, once per epoch. This asserts the property it guards on the row
    identity a caller can actually read, because the composer's rank is SINK
    major and a reader who assumed TX major would find every angle mis-steered
    and nothing raised.
    """

    radar = _radar()
    _simulate(radar, _static_scene(), (0.0,))
    paths = radar.last_radar_paths
    array = radar.system_config.sensors.array

    assert paths.sensor_pair_count == array.num_tx * array.num_rx
    ranks = paths.sensor_pair_index
    assert bool(torch.all(ranks[1:] >= ranks[:-1])), "pair ranks must not decrease"
    assert int(paths.pair_offsets[0]) == 0
    assert int(paths.pair_offsets[-1]) == paths.path_count

    # Sink major: pair = rx_rank * num_tx + tx_rank. Read off the identity
    # columns rather than restated, so a change of convention fails here.
    sources = sorted({int(value) for value in paths.topology.radar_source_id.tolist()})
    sinks = sorted({int(value) for value in paths.topology.radar_sink_id.tolist()})
    for row, rank in enumerate(ranks.tolist()):
        tx_rank = sources.index(int(paths.topology.radar_source_id[row]))
        rx_rank = sinks.index(int(paths.topology.radar_sink_id[row]))
        assert rank == rx_rank * array.num_tx + tx_rank


def test_the_composed_row_order_is_frame_invariant():
    """A frozen topology names its rows by identity, on every frame."""

    radar = _radar()
    _simulate(radar, _static_scene(), (0.0,))
    first = radar.last_radar_paths.topology
    _simulate(radar, _static_scene(), (0.0, 3.0e-3))
    second = radar.last_radar_paths.topology
    for name in ("radar_source_id", "site_id", "radar_sink_id"):
        assert torch.equal(getattr(first, name), getattr(second, name)), name


# ---------------------------------------------------------------------------
# One discovery per epoch (the cadence, measured)
# ---------------------------------------------------------------------------


def _count_freezes(monkeypatch) -> list:
    from witwin.radar.channel import ChannelPropagationAdapter

    calls: list = []
    original = ChannelPropagationAdapter.freeze

    def counting(self, sources, sinks):
        calls.append((id(self), len(calls)))
        return original(self, sources, sinks)

    monkeypatch.setattr(ChannelPropagationAdapter, "freeze", counting)
    return calls


def test_a_still_world_compiles_and_discovers_exactly_once(monkeypatch):
    """Six frames, one compile, one discovery, two freezes - one per leg.

    Core folds ``time_s`` into ``geometry_version`` for any snapshot from a
    ``DynamicScene``, so a loop that trusted that signal would rebuild the RayD
    scene once per frame for nothing. This is that pin at the production entry.
    """

    freezes = _count_freezes(monkeypatch)
    radar = _radar()
    result = _simulate(radar, _static_scene(), tuple(k * 1.0e-3 for k in range(6)))

    assert result.compile_count == 1
    assert result.discovery_count == 1
    assert result.epochs == (0,) * 6
    assert result.rediscovery_reasons == ("first_frame", None, None, None, None, None)
    assert len(freezes) == 2, freezes


def test_a_moving_world_discovers_exactly_once_per_epoch(monkeypatch):
    """The declared cadence, and the freeze count that proves it.

    Under the default ``frozen_world`` a moved structure retires every frozen
    handle, so each frame is its own epoch and each epoch freezes both legs
    exactly once. ``discovery_count`` and the distinct epoch numbers have to
    agree; a loop that rediscovered twice for one epoch would still produce a
    plausible cube.
    """

    freezes = _count_freezes(monkeypatch)
    radar = _radar()
    dynamic = world.make_dynamic_scene(wall_velocity=geo.WALL_VELOCITY_M_PER_S)
    result = _simulate(radar, dynamic, (0.0, 1.0e-3, 2.0e-3))

    assert result.epochs == (0, 1, 2)
    assert result.rediscovery_reasons == ("first_frame", "structure_motion", "structure_motion")
    assert result.compile_count == 3
    assert result.discovery_count == len(set(result.epochs)) == 3
    assert len(freezes) == 2 * result.discovery_count, freezes


def test_fixed_winner_replay_holds_one_epoch_across_a_moving_world(monkeypatch):
    """The declaration that makes a per-frame moving world affordable.

    ``fixed_winner_replay`` says the discrete winner set is held fixed while the
    geometry moves. The compiled scene is still rebuilt every frame - the wall
    really did move - but the frozen rows are replayed, so there is exactly one
    epoch and exactly one pair of freezes for the whole run.
    """

    freezes = _count_freezes(monkeypatch)
    radar = _radar()
    dynamic = world.make_dynamic_scene(wall_velocity=geo.WALL_VELOCITY_M_PER_S)
    result = _simulate(radar, dynamic, (0.0, 1.0e-3, 2.0e-3), world_motion="fixed_winner_replay")

    assert result.epochs == (0, 0, 0)
    assert result.discovery_count == 1
    assert result.compile_count == 3
    assert len(freezes) == 2, freezes
    # And the replay really tracked the moved wall.
    assert not torch.equal(result.cube[0], result.cube[2])


# ---------------------------------------------------------------------------
# Differentiability and the per-solve overrides
# ---------------------------------------------------------------------------


def test_the_published_cube_is_differentiable_through_the_site_positions():
    """``ad_mode='vjp'`` reaches the leaf the site policy passed through.

    The site tensor is the SINK of the inbound leg and the SOURCE of the
    outbound one, and the binding hands the same object to both, so a gradient
    that reached only one leg would still be finite and non-zero. What is
    asserted is that a gradient exists at all through the production entry -
    the two-leg accumulation itself is pinned in the Phase-9 AD suite against
    an analytic oracle.
    """

    radar = _radar()
    policy = _sites(radar, requires_grad=True)
    result = radar.simulate(_static_scene(), times=(0.0,), response=_response(radar), sites=policy, ad_mode="vjp")
    assert result.cube.requires_grad
    result.cube.abs().square().sum().backward()
    grad = policy.positions_m.grad
    assert grad is not None
    assert bool(torch.isfinite(grad).all())
    assert bool((grad != 0).any())


def test_components_and_max_depth_override_one_solve_and_not_the_radar():
    """A propagation request is a statement about ONE solve."""

    radar = _radar()
    full = _simulate(radar, _static_scene(), (0.0,))
    full_rows = radar.last_radar_paths.path_count
    assert full.cube.shape == full.cube.shape

    narrowed = _simulate(radar, _static_scene(), (0.0,), components=frozenset({"los"}), max_depth=0)
    assert narrowed.cube.shape == full.cube.shape
    assert radar.last_radar_paths.path_count < full_rows
    # The radar's stored configuration never moved.
    assert radar.system_config.propagation.components == frozenset({"los", "reflection"})
    assert radar.system_config.propagation.max_depth == 1


# ---------------------------------------------------------------------------
# The refusals
# ---------------------------------------------------------------------------


def test_simulate_group_is_deleted_rather_than_permanently_refusing():
    """A permanent NotImplementedError is itself a legacy shim."""

    assert not hasattr(Radar, "simulate_group")
    assert not hasattr(wr.Radar, "_SIMULATE_REPLACEMENT")


def test_slow_time_mode_is_not_a_public_simulation_choice():
    """The scene driver fixes its synthesis mode internally."""

    assert "slow_time_mode" not in inspect.signature(Radar.simulate).parameters


def test_an_empty_time_sequence_is_refused():
    radar = _radar()
    with pytest.raises(ValueError, match="at least one frame instant"):
        radar.simulate(_static_scene(), times=(), response=_response(radar), sites=_sites(radar))


def test_a_site_declaration_that_is_not_a_policy_is_refused():
    """Where the sites come from is a declaration, never a search."""

    radar = _radar()
    with pytest.raises(TypeError, match="must be a ScatterSitePolicy"):
        radar.simulate(
            _static_scene(),
            times=(0.0,),
            response=_response(radar),
            sites=torch.tensor(SITE_POSITIONS_M, device=radar.device),
        )


def test_a_static_world_has_no_core_owned_site_anchor():
    """The default site policy fails loudly rather than sampling a mesh.

    ``sites=None`` resolves to ``structure_anchor()``, and the fixture wall
    carries no rigid motion, so this world publishes no Core-owned site
    position for it. The message has to name the mesh-sampling deferral,
    because that is the thing a caller will otherwise reach for.
    """

    radar = _radar()
    with pytest.raises(NotImplementedError, match="named Phase-11 deferral"):
        radar.simulate(_static_scene(), times=(0.0,), response=_response(radar))
