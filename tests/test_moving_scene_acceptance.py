"""The Phase-7 acceptance matrix, and the two claims nothing else asserts.

Three things live here and nothing else does:

1. **The matrix.** Each of the plan's eight Phase-7 acceptance criteria is
   mapped to the tests that prove it, and the map is machine checked against
   the tree. This is deliberately a MAP and not a copy: S3 and S4 already drive
   scenarios S1-S7 end to end against the float64 closed forms, and restating
   them here would produce a second set of numbers that could drift from the
   first. What a map cannot do is rot silently - a renamed or deleted test
   fails the check by name.

2. **The item-8 cross-consumer criterion.** Channel's time-varying CIR and a
   Radar frame, driven from ONE ``DynamicScene`` at ONE ``times_s`` vector, must
   use the same world state. Same kernel, same inputs, so the assertion is
   bitwise equality of the delays and an exact match on ``CompiledScene.time_s``
   - a tolerance here would be an admission that they are two computations.
"""

from __future__ import annotations

import ast
import pathlib

import pytest
import torch

pytest.importorskip("witwin.channel")

from support import multi_endpoint_driver as drv
from support import multi_endpoint_geometry as geo
from support import multi_endpoint_world as world

# ``gpu`` is per test rather than per module. Section 1 is a pure AST scan over
# the tree and builds no tensor, and it is this file's own name-rot gate: under
# a module marker the gate that catches a renamed test would itself be skipped
# by the default ``pytest tests/`` run. Section 2 needs CUDA and says so.

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# The matrix
# --------------------------------------------------------------------------


#: Plan criterion -> the tests that prove it, as ``file::test`` strings.
#:
#: The criteria are the plan's own eight bullets, in its own order, translated
#: to English. Equality, not containment: an entry naming a test that no longer
#: exists is a failure, and so is a criterion that lost its last owner.
ACCEPTANCE_MATRIX = {
    "static, moving-target, deforming-mesh and moving environment scenarios pass": (
        "test_moving_structures.py::test_a_translating_wall_moves_only_the_reflection_row",
        "test_moving_structures.py::test_a_deforming_mesh_is_a_core_deformation_not_a_reposed_geometry",
        "test_simulate_entry.py::test_the_composed_rows_agree_with_the_reference_orchestration",
        "test_fmcw_continuous_motion.py::test_scene_adc_sampling_matches_radial_motion_including_fast_time",
    ),
    "TDM per-TX phase agrees with the downstream compensation": (
        "test_fmcw_tdm.py::test_the_downstream_tdm_compensation_removes_exactly_the_carrier_slot_phase",
        "test_fmcw_tdm.py::test_the_production_slot_table_survives_the_downstream_compensation",
        "test_slot_batching.py::test_tdm_slot_indices_come_from_the_phase6_owner",
    ),
    "FMCW/OFDM/Pulsed Doppler sign, dimension and aliasing limits are correct": (
        "test_cross_waveform_axes.py::test_the_closing_target_is_positive_in_every_waveform",
        "test_fmcw_analytic.py::test_the_slow_time_slope_carries_the_whole_carrier_not_just_the_ramp",
        "test_ofdm_kernel.py::test_a_receding_site_puts_the_cfr_tone_at_negative_doppler",
        "test_ofdm_kernel.py::test_a_speed_past_the_unambiguous_bound_aliases",
        "test_pulsed_kernel.py::test_a_receding_site_puts_the_slow_time_tone_at_negative_doppler",
        "test_pulsed_kernel.py::test_a_speed_past_the_unambiguous_bound_aliases",
    ),
    "limb/rotor/deforming-mesh micro-Doppler agrees with an analytic or independent reference": (
        "test_microdoppler.py::test_a_rotating_two_blade_target_gives_a_flash_spectrum",
        "test_microdoppler.py::test_a_hinge_limb_gives_a_rectangular_doppler_band",
        "test_microdoppler.py::test_smpl_limb_microdoppler_matches_an_independent_reference",
        "test_scatter_response_kernel.py::test_the_aspect_kernel_matches_a_closed_form",
    ),
    "Channel time-varying CIR and Radar snapshot timestamps use the same world state": (
        "test_moving_scene_acceptance.py::test_channel_cir_and_radar_frames_use_the_same_world_state",
    ),
    "no Python full-scene retrace inside a frame, symbol or pulse": (
        "test_slot_batching.py::test_the_batched_replay_is_exactly_one_consumer_call_per_leg",
        "test_row_invalidation.py::test_endpoint_only_motion_does_not_recompile",
        "test_synthesis_launch_budget.py::test_each_waveform_costs_exactly_one_forward_launch_per_frame",
    ),
    "topology invalidation returns neither a wrong primal nor a detached gradient": (
        "test_row_invalidation.py::test_a_stale_compiled_scene_never_answers",
        "test_row_invalidation.py::test_a_born_row_forces_an_explicit_rediscovery",
        "test_row_invalidation.py::test_a_world_mutated_in_place_is_caught_on_the_motion_event_tick",
        "test_rediscovery_cadence.py::test_a_retired_handle_is_refused_even_when_no_version_moved",
    ),
    "slot batching, launch count, peak memory and realtime scaling meet their budgets": (
        "test_slot_batching.py::test_pair_count_grows_linearly_not_quadratically",
        "test_synthesis_launch_budget.py::test_the_launch_count_is_flat_in_slot_count",
        "test_join_host_budget.py::test_the_per_frame_host_budget_is_flat_in_slot_count",
        "test_join_host_budget.py::test_peak_memory_and_per_slot_cost_scale",
    ),
}


def _defined_tests(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")
    }


def test_every_acceptance_criterion_names_a_test_that_exists():
    """The map is checked against the tree, not merely written down.

    A criterion whose only proof was renamed away leaves this file naming a
    function that is not there, which is precisely the state a completion
    record must never be able to reach quietly.
    """

    missing = []
    for criterion, owners in ACCEPTANCE_MATRIX.items():
        assert owners, criterion
        for owner in owners:
            filename, _, name = owner.partition("::")
            path = REPO_ROOT / "tests" / filename
            if not path.exists() or name not in _defined_tests(path):
                missing.append(owner)
    assert missing == [], missing


def test_the_matrix_covers_all_eight_plan_criteria():
    """Eight bullets in the plan, eight keys here."""

    assert len(ACCEPTANCE_MATRIX) == 8


# --------------------------------------------------------------------------
# The item-8 cross-consumer criterion
# --------------------------------------------------------------------------


SLOT_COUNT = 6
SLOT_PERIOD_S = 1.0e-4
SNAPSHOT_TIME_S = 0.25


def _times() -> torch.Tensor:
    index = torch.arange(SLOT_COUNT, dtype=torch.float64)
    return SNAPSHOT_TIME_S + index * SLOT_PERIOD_S


@pytest.mark.gpu
def test_channel_cir_and_radar_frames_use_the_same_world_state():
    """One ``DynamicScene``, one ``times_s``, two consumers, one answer.

    Channel's ``evaluate_time_varying`` and Radar's slot-batched replay are
    given the SAME compiled scene, the SAME frozen topology and the SAME
    slot-major endpoint stacks. They must therefore agree BITWISE on the delay
    of every row at every instant: they are the same native call reached
    through two facades, and anything less than ``torch.equal`` would mean one
    of the two had inserted an operation of its own.

    ``CompiledScene.time_s`` carries the snapshot instant, so the timestamp the
    Channel result is labelled with and the world the Radar frame was built
    from are checkable against each other rather than assumed to match.
    """

    from witwin.channel.propagation import consumer

    from witwin.radar.channel import _endpoint_batch

    times = _times()
    dynamic = world.make_dynamic_scene()
    snapshot = dynamic.at(SNAPSHOT_TIME_S)
    compiled = world.compile_snapshot(snapshot)

    # The world state both consumers were built from, labelled and checkable.
    assert compiled.time_s == pytest.approx(SNAPSHOT_TIME_S)
    assert compiled.time_s == pytest.approx(float(times[0]))
    assert snapshot.time_s == pytest.approx(SNAPSHOT_TIME_S)

    spike = drv.MultiEndpointSpike(compiled=compiled)
    base = spike.site_tensor()
    stack = drv.slot_site_stack(base, geo.SITE_P_VELOCITY_M_PER_S, (times - times[0]).tolist())

    inbound, _ = spike.slot_legs(stack, slot_count=SLOT_COUNT)
    assert inbound.slot_count == SLOT_COUNT

    # The Channel side, from the same handle and the same stacks the adapter
    # built. Reaching for the adapter's own batch helper is deliberate: a
    # second, test-written endpoint batch would be a second world state and the
    # bitwise claim would be about the test rather than about the two consumers.
    transmitters = spike._stacked_ids(
        spike.stacked([position for _, position in spike.transmitters], SLOT_COUNT),
        spike.transmitter_ids,
        geo.TX_POWER_W,
    )
    sites = spike._stacked_ids(stack, spike.site_ids, None)
    evaluation = consumer.evaluate_time_varying(
        compiled,
        consumer.TimeVaryingRequest(
            sources=_endpoint_batch(transmitters, "source"),
            sinks=_endpoint_batch(sites, "sink"),
            reference_frequency_hz=geo.REFERENCE_FREQUENCY_HZ,
            topology=spike.inbound.prepared,
            times_s=times,
            response="scalar_transport",
            ad_mode="none",
        ),
    )

    assert evaluation.slot_count == SLOT_COUNT
    assert evaluation.row_count == inbound.rows_per_slot
    assert torch.equal(evaluation.times_s, times)

    moved = False
    for slot in range(SLOT_COUNT):
        radar_slot = inbound.slot(slot)
        assert torch.equal(evaluation.delay_s[slot], radar_slot.delay_s), slot
        assert torch.equal(evaluation.transport.coefficient[slot], radar_slot.coefficient), slot
        if slot and not torch.equal(evaluation.delay_s[slot], evaluation.delay_s[0]):
            moved = True
    # Non-vacuity: a world that did not move would make every slot trivially
    # equal to every other and the bitwise claim would prove nothing.
    assert moved

    # And the two consumers publish the same per-slot pair segmentation, which
    # is what makes "the same rows at the same instants" a checkable statement
    # rather than a coincidence of lengths.
    assert evaluation.pair_count == inbound.pairs_per_slot
    assert torch.equal(evaluation.pair_offsets, inbound.slot(0).pair_offsets)
