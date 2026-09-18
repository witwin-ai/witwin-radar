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
* the four diagnostics are typed, describe ONE frame, and live on the RESULT -
  the radar has no retention site for them, so neither a fresh radar nor one
  whose last call raised can hand a caller a stale world;
* the pair partition is this array's TX x RX grid in the composer's own
  sink-major rank;
* a topology is discovered exactly ONCE per epoch, and the epoch cadence is the
  declared one rather than whatever the geometry version happens to do.
"""

from __future__ import annotations

import dataclasses
import inspect

import pytest
import torch

pytest.importorskip("witwin.channel")

from support import multi_endpoint_driver as drv
from support import multi_endpoint_geometry as geo
from support import multi_endpoint_world as world
from support.simulate_fixture import SITE_POSITIONS_M, fixture_radar, fixture_targets, static_scene

from witwin.radar import Motion, PointTargets, Radar, Result, StructureTargets
from witwin.radar.paths import RadarPathBatch
from witwin.radar.propagation import RadarLegBatch, RadarPropagationLegs
from witwin.radar.simulation import StableIdAllocator

pytestmark = pytest.mark.gpu


def _simulate(radar: Radar, scene, times, **options) -> Result:
    options.setdefault("motion", Motion.chirp())
    return radar.simulate(scene, fixture_targets(radar), times=times, **options)


# ---------------------------------------------------------------------------
# The product
# ---------------------------------------------------------------------------


def test_simulate_runs_the_whole_pipeline_and_publishes_a_frame_cube():
    """Core Scene -> CompiledScene -> propagation -> two-way -> synthesis."""

    radar = fixture_radar()
    result = _simulate(radar, static_scene(), (0.0, 1.0e-3, 2.0e-3))

    waveform = radar.waveform
    assert result.cube.shape == (3, radar.num_tx, radar.num_rx, waveform.chirps_per_frame, waveform.samples_per_chirp)
    assert result.cube.dtype == torch.complex64
    assert result.cube.device.type == radar.device.type
    assert result.axis_names == ("frame", "tx", "rx", "chirp", "range_bin")
    assert result.kind == "fmcw"
    assert result.times_s == (0.0, 1.0e-3, 2.0e-3)
    assert result.frame_count == 3
    assert result.reference_frequency_hz == geo.REFERENCE_FREQUENCY_HZ
    # The conventions come from the waveform owner rather than from this entry.
    from witwin.radar.synthesis.assembly import BEAT_PHASOR

    assert result.phasor == BEAT_PHASOR


def test_the_entry_returns_the_typed_record_rather_than_a_bare_tensor():
    """A tuple or a tensor would make every consumer re-derive the axes."""

    radar = fixture_radar()
    result = _simulate(radar, static_scene(), (0.0,))
    assert isinstance(result, Result)
    assert dataclasses.is_dataclass(result)
    with pytest.raises(AttributeError):
        result.cube = torch.zeros(1)


def test_a_still_world_publishes_the_same_frame_at_every_instant():
    """Nothing moves, so nothing may change - not even by a float32 ULP.

    This is the calibration for every motion test below: a difference between
    two frames of a still world would mean the frame loop carries state it
    should not, and would make "the wall moved" unmeasurable.
    """

    radar = fixture_radar()
    result = _simulate(radar, static_scene(), (0.0, 5.0e-3))
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

    radar = fixture_radar()
    allocator = StableIdAllocator()
    transmitter_ids, receiver_ids, site_ids = allocator.allocate(
        transmitter_count=int(radar.tx_pos.shape[0]),
        receiver_count=int(radar.rx_pos.shape[0]),
        site_count=len(SITE_POSITIONS_M),
    )
    produced = _simulate(radar, static_scene(), (0.0,)).last_radar_paths

    spike = drv.MultiEndpointSpike(
        transmitters=tuple(zip(transmitter_ids, [tuple(row) for row in radar.tx_pos.tolist()], strict=True)),
        sites=tuple(zip(site_ids, SITE_POSITIONS_M, strict=True)),
        receivers=tuple(zip(receiver_ids, [tuple(row) for row in radar.rx_pos.tolist()], strict=True)),
    )
    reference, _, _ = spike.frame(response=drv.make_response(device=radar.device))

    assert produced.path_count == reference.path_count
    assert produced.sensor_pair_count == reference.sensor_pair_count
    for name in ("radar_source_id", "site_id", "radar_sink_id"):
        assert torch.equal(getattr(produced.topology, name), getattr(reference.topology, name)), name
    assert torch.equal(produced.total_delay_s, reference.total_delay_s)

    expected = (radar.transmit_power_watts / geo.TX_POWER_W) ** 0.5
    ratio = produced.complex_transfer_ref.abs() / reference.complex_transfer_ref.abs()
    torch.testing.assert_close(ratio, torch.full_like(ratio, float(expected)), rtol=1e-5, atol=0.0)


# ---------------------------------------------------------------------------
# The four typed diagnostics (work item 2)
# ---------------------------------------------------------------------------


def test_the_four_diagnostics_are_typed_and_describe_the_last_frame():
    radar = fixture_radar()
    result = _simulate(radar, static_scene(), (0.0, 1.0e-3))

    from witwin.channel.scene import CompiledScene
    from witwin.core import SceneSnapshot

    assert isinstance(result.last_snapshot, SceneSnapshot)
    assert isinstance(result.last_compiled_scene, CompiledScene)
    assert isinstance(result.last_propagation, RadarPropagationLegs)
    assert isinstance(result.last_radar_paths, RadarPathBatch)

    # The LAST frame, named by its own time, not the first.
    assert float(result.last_snapshot.time_s) == 1.0e-3


def test_the_leg_pair_is_typed_rather_than_a_tuple_or_a_dict():
    """``RadarPropagationLegs`` is what makes the pairing checkable."""

    radar = fixture_radar()
    legs = _simulate(radar, static_scene(), (0.0,)).last_propagation
    assert isinstance(legs.inbound, RadarLegBatch)
    assert isinstance(legs.outbound, RadarLegBatch)
    assert legs.slot_count == 1
    assert legs.device.type == radar.device.type
    assert not isinstance(legs, (tuple, dict))

    with pytest.raises(TypeError, match="outbound must be a RadarLegBatch"):
        RadarPropagationLegs(inbound=legs.inbound, outbound=object())


def test_an_empty_frame_sequence_is_refused():
    radar = fixture_radar()
    with pytest.raises(ValueError, match="at least one frame instant"):
        radar.simulate(static_scene(), fixture_targets(radar), times=())


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

    radar = fixture_radar()
    paths = _simulate(radar, static_scene(), (0.0,)).last_radar_paths

    assert paths.sensor_pair_count == radar.num_tx * radar.num_rx
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
        assert rank == rx_rank * radar.num_tx + tx_rank


def test_the_composed_row_order_is_frame_invariant():
    """A frozen topology names its rows by identity, on every frame."""

    radar = fixture_radar()
    first = _simulate(radar, static_scene(), (0.0,)).last_radar_paths.topology
    second = _simulate(radar, static_scene(), (0.0, 3.0e-3)).last_radar_paths.topology
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
    radar = fixture_radar()
    result = _simulate(radar, static_scene(), tuple(k * 1.0e-3 for k in range(6)))

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
    radar = fixture_radar()
    dynamic = world.make_dynamic_scene(wall_velocity=geo.WALL_VELOCITY_M_PER_S)
    result = _simulate(radar, dynamic, (0.0, 1.0e-3, 2.0e-3))

    slots = len(result.sample_times_s[0])
    assert result.epochs == (0, slots, 2 * slots)
    assert result.rediscovery_reasons == ("first_frame", "structure_motion", "structure_motion")
    assert result.compile_count == 3 * slots
    assert result.discovery_count == 3 * slots
    assert result.path_set_complete
    assert len(freezes) == 2 * result.discovery_count, freezes


def test_fixed_winner_replay_holds_one_epoch_across_a_moving_world(monkeypatch):
    """The declaration that makes a per-frame moving world affordable.

    ``fixed_winner_replay`` says the discrete winner set is held fixed while the
    geometry moves. The compiled scene is still rebuilt every frame - the wall
    really did move - but the frozen rows are replayed, so there is exactly one
    epoch and exactly one pair of freezes for the whole run.
    """

    freezes = _count_freezes(monkeypatch)
    radar = fixture_radar()
    dynamic = world.make_dynamic_scene(wall_velocity=geo.WALL_VELOCITY_M_PER_S)
    result = _simulate(
        radar,
        dynamic,
        (0.0, 1.0e-3, 2.0e-3),
        motion=Motion.chirp(world="fixed_winner_replay", rediscover_every_frames=10),
    )

    assert result.epochs == (0, 0, 0)
    assert result.discovery_count == 1
    assert result.compile_count == 3 * len(result.sample_times_s[0])
    assert not result.path_set_complete
    assert len(freezes) == 2, freezes
    # And the replay really tracked the moved wall.
    assert not torch.equal(result.cube[0], result.cube[2])


# ---------------------------------------------------------------------------
# Differentiability and the per-solve overrides
# ---------------------------------------------------------------------------


def test_the_published_cube_is_differentiable_through_the_site_positions():
    """``grad='vjp'`` reaches the leaf the target record passed through.

    The site tensor is the SINK of the inbound leg and the SOURCE of the
    outbound one, and the binding hands the same object to both, so a gradient
    that reached only one leg would still be finite and non-zero. What is
    asserted is that a gradient exists at all through the production entry -
    the two-leg accumulation itself is pinned in the Phase-9 AD suite against
    an analytic oracle.
    """

    radar = fixture_radar()
    targets = fixture_targets(radar, requires_grad=True)
    result = radar.simulate(static_scene(), targets, times=(0.0,), grad="vjp")
    assert result.cube.requires_grad
    result.cube.abs().square().sum().backward()
    grad = targets.positions.grad
    assert grad is not None
    assert bool(torch.isfinite(grad).all())
    assert bool((grad != 0).any())


def test_parameter_jvp_does_not_change_the_primal_scene():
    import torch.autograd.forward_ad as forward_ad

    radar = fixture_radar()
    scene = static_scene()
    positions = torch.tensor(SITE_POSITIONS_M, dtype=torch.float32, device=radar.device)

    def targets_at(sites) -> PointTargets:
        return PointTargets(positions=sites, amplitude=drv.FIXTURE_AMPLITUDE, phase=drv.FIXTURE_PHASE_RAD)

    reference = radar.simulate(scene, targets_at(positions), times=(0.0,))
    for scale in (0.0, 1.0, -3.0):
        direction = torch.zeros_like(positions)
        direction[:, 0] = scale
        with forward_ad.dual_level():
            dual = forward_ad.make_dual(positions, direction)
            result = radar.simulate(scene, targets_at(dual), times=(0.0,), grad="jvp")
            primal, tangent = forward_ad.unpack_dual(result.cube)
            torch.testing.assert_close(primal, reference.cube, rtol=0, atol=0)
            assert tangent is not None
            assert bool(torch.isfinite(tangent).all())
            assert result.last_radar_paths.delay_rate is None


def test_los_and_reflections_override_one_solve_and_not_the_radar():
    """A propagation request is a statement about ONE solve."""

    radar = fixture_radar()
    full = _simulate(radar, static_scene(), (0.0,))
    full_rows = full.last_radar_paths.path_count

    narrowed = _simulate(radar, static_scene(), (0.0,), los=True, reflections=0)
    assert narrowed.cube.shape == full.cube.shape
    assert narrowed.last_radar_paths.path_count < full_rows
    # The radar's stored configuration never moved.
    assert radar.system_config.propagation.components == frozenset({"los", "reflection"})
    assert radar.system_config.propagation.max_depth == 1


# ---------------------------------------------------------------------------
# The refusals
# ---------------------------------------------------------------------------


def test_slow_time_mode_is_not_a_public_simulation_choice():
    """The scene driver fixes its synthesis mode internally."""

    assert "slow_time_mode" not in inspect.signature(Radar.simulate).parameters


def test_an_empty_time_sequence_is_refused():
    radar = fixture_radar()
    with pytest.raises(ValueError, match="at least one frame instant"):
        radar.simulate(static_scene(), fixture_targets(radar), times=())


def test_a_target_set_that_is_not_a_target_record_is_refused():
    """Where the scatterers are is a declaration, never a search."""

    radar = fixture_radar()
    with pytest.raises(TypeError, match="targets must be PointTargets or StructureTargets"):
        radar.simulate(static_scene(), torch.tensor(SITE_POSITIONS_M, device=radar.device), times=(0.0,))


def test_a_static_world_has_no_core_owned_site_anchor():
    """``StructureTargets`` fails loudly rather than sampling a mesh.

    A structure target puts one scatterer at the world anchor Core publishes for
    each structure, and the fixture wall carries no rigid motion, so this world
    publishes no such position for it. The message has to name the mesh-sampling
    deferral, because that is the thing a caller will otherwise reach for.
    """

    radar = fixture_radar()
    with pytest.raises(NotImplementedError, match="deferred by R-ADR-020"):
        radar.simulate(static_scene(), StructureTargets(rcs=1.0), times=(0.0,))
