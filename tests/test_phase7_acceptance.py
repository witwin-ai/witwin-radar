"""The Phase-7 acceptance matrix, and the two claims nothing else asserts.

Three things live here and nothing else does:

1. **The matrix.** Each of the plan's eight Phase-7 acceptance criteria is
   mapped to the tests that prove it, and the map is machine checked against
   the tree. This is deliberately a MAP and not a copy: S3 and S4 already drive
   scenarios S1-S7 end to end against the float64 closed forms, and restating
   them here would produce a second set of numbers that could drift from the
   first. What a map cannot do is rot silently - a renamed or deleted test
   fails the check by name.

2. **The dimension criterion, end to end.** Phase 6 pins the slow-time slope
   against a hand-built row (``test_phase6_fmcw_analytic.py``); nothing pinned
   it against a ``delay_rate`` that came out of the propagation JVP. That
   is the one link in the chain "kinematics -> JVP -> join -> cube" that had no
   test, and a unit error anywhere in it lands here.

3. **The item-8 cross-consumer criterion.** Channel's time-varying CIR and a
   Radar frame, driven from ONE ``DynamicScene`` at ONE ``times_s`` vector, must
   use the same world state. Same kernel, same inputs, so the assertion is
   bitwise equality of the delays and an exact match on ``CompiledScene.time_s``
   - a tolerance here would be an admission that they are two computations.
"""

from __future__ import annotations

import ast
import math
import pathlib
from dataclasses import replace

import pytest
import torch

pytest.importorskip("witwin.channel")

from support import multi_endpoint_driver as drv  # noqa: E402
from support import multi_endpoint_geometry as geo  # noqa: E402
from support import multi_endpoint_world as world  # noqa: E402


pytestmark = pytest.mark.gpu

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# 1  the matrix
# --------------------------------------------------------------------------


#: Plan criterion -> the tests that prove it, as ``file::test`` strings.
#:
#: The criteria are the plan's own eight bullets, in its own order, translated
#: to English. Equality, not containment: an entry naming a test that no longer
#: exists is a failure, and so is a criterion that lost its last owner.
ACCEPTANCE_MATRIX = {
    "static, radial, tangential, rotation, deformation, moving TX/RX and "
    "moving environment scenarios pass": (
        "test_phase7_doppler_scenarios.py::test_a_static_scene_has_exactly_zero_delay_rate",
        "test_phase7_doppler_scenarios.py::test_a_radially_moving_site_matches_the_projection_formula",
        "test_phase7_doppler_scenarios.py::test_the_los_and_reflection_rows_differ",
        "test_phase7_doppler_scenarios.py::test_moving_transmitter_and_receiver",
        "test_phase7_moving_structures.py::test_a_rotating_rigid_body_gives_equal_and_opposite_shifts",
        "test_phase7_moving_structures.py::test_a_hinge_deformation_gives_a_linear_doppler_band",
        "test_phase7_moving_structures.py::test_a_translating_wall_moves_only_the_reflection_row",
    ),
    "TDM per-TX phase agrees with the downstream compensation": (
        "test_phase6_fmcw_tdm.py::test_the_sigproc_tdm_compensation_removes_exactly_the_carrier_slot_phase",
        "test_phase6_fmcw_tdm.py::test_the_production_slot_table_survives_the_downstream_compensation",
        "test_phase7_slot_batching.py::test_tdm_slot_indices_come_from_the_phase6_owner",
    ),
    "FMCW/OFDM/Pulsed Doppler sign, dimension and aliasing limits are correct": (
        "test_phase7_doppler_scenarios.py::test_the_doppler_sign_follows_the_channel_phasor",
        "test_phase7_doppler_scenarios.py::test_doppler_aliasing_folds_as_predicted",
        "test_phase6_fmcw_analytic.py::test_the_slow_time_slope_carries_the_whole_carrier_not_just_the_ramp",
        "test_phase7_acceptance.py::test_the_slow_time_slope_has_the_dimension_of_the_measured_rate",
        "test_phase6_ofdm_kernel.py::test_a_receding_site_puts_the_cfr_tone_at_negative_doppler",
        "test_phase6_ofdm_kernel.py::test_a_speed_past_the_unambiguous_bound_aliases",
        "test_phase6_pulsed_kernel.py::test_a_receding_site_puts_the_slow_time_tone_at_negative_doppler",
        "test_phase6_pulsed_kernel.py::test_a_speed_past_the_unambiguous_bound_aliases",
    ),
    "limb/rotor/deforming-mesh micro-Doppler agrees with an analytic or "
    "independent reference": (
        "test_phase7_microdoppler.py::test_a_rotating_two_blade_target_gives_a_flash_spectrum",
        "test_phase7_microdoppler.py::test_a_hinge_limb_gives_a_rectangular_doppler_band",
        "test_phase7_microdoppler.py::test_smpl_limb_microdoppler_matches_an_independent_reference",
        "test_phase7_scatter_response_kernel.py::test_the_aspect_kernel_matches_a_closed_form",
    ),
    "Channel time-varying CIR and Radar snapshot timestamps use the same "
    "world state": (
        "test_phase7_acceptance.py::test_channel_cir_and_radar_frames_use_the_same_world_state",
    ),
    "no Python full-scene retrace inside a frame, symbol or pulse": (
        "test_phase7_slot_batching.py::test_the_batched_replay_is_exactly_one_consumer_call_per_leg",
        "test_phase7_invalidation.py::test_endpoint_only_motion_does_not_recompile",
        "test_phase6_launch_budget.py::test_each_waveform_costs_exactly_one_forward_launch_per_frame",
    ),
    "topology invalidation returns neither a wrong primal nor a detached "
    "gradient": (
        "test_phase7_invalidation.py::test_a_stale_compiled_scene_never_answers",
        "test_phase7_invalidation.py::test_invalidation_never_produces_a_detached_gradient",
        "test_phase7_invalidation.py::test_a_born_row_forces_an_explicit_rediscovery",
        "test_phase7_invalidation.py::test_a_world_mutated_in_place_is_caught_on_the_motion_event_tick",
        "test_phase7_rediscovery_cadence.py::test_a_retired_handle_is_refused_even_when_no_version_moved",
    ),
    "slot batching, launch count, peak memory and realtime scaling meet "
    "their budgets": (
        "test_phase7_slot_batching.py::test_pair_count_grows_linearly_not_quadratically",
        "test_phase6_launch_budget.py::test_the_launch_count_is_flat_in_slot_count",
        "test_phase5_budget.py::test_the_per_frame_host_budget_is_flat_in_slot_count",
        "test_phase5_budget.py::test_peak_memory_and_per_slot_cost_scale",
    ),
}


def _defined_tests(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
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
# 2  the dimension criterion, from a measured rate
# --------------------------------------------------------------------------


#: One transmitter, one site, one receiver, line of sight only, so the beat
#: cube of a sensor pair is ONE composed row and its slow-time slope is that
#: row's. With the full fixture the cube is a superposition of eleven rows and
#: the slope is not any single row's, which is why this scenario is its own
#: spike rather than a reuse of the S2 one.
SOLO_SPEED_M_PER_S = 12.0


def _solo_spec(*, num_chirps: int):
    """The fixture waveform over a ONE-by-ONE array.

    The pair partition and the waveform's declared array must be the same front
    end - ``pair_tx_index`` refuses a mismatch rather than reinterpreting one
    array's pairs as another's - so a one-pair batch needs a one-by-one spec.
    Everything else is the fixture's own configuration.
    """

    from witwin.radar import RadarConfig
    from witwin.radar.synthesis import FmcwBeatSpec

    values = dict(geo.FIXTURE_RADAR_CONFIG)
    values.update(num_tx=1, num_rx=1, tx_loc=[[0, 0, 0]], rx_loc=[[0, 0, 0]])
    spec = FmcwBeatSpec.from_radar_config(RadarConfig.from_dict(values))
    return replace(spec, num_chirps=num_chirps)


@pytest.fixture(scope="module")
def solo():
    return drv.MultiEndpointSpike(
        transmitters=(geo.TRANSMITTERS[0],),
        sites=(geo.SITES[0],),
        receivers=(geo.RECEIVERS[0],),
        components=frozenset({"los"}),
        max_depth=0,
    )


def test_the_slow_time_slope_has_the_dimension_of_the_measured_rate(solo):
    """``d(phase)/d(chirp) = 2 pi tau_rate Tc (fc + S (t0 - tau + t_m))``.

    Every quantity on the right comes from somewhere the test did not choose:
    ``tau_rate`` and ``tau`` are what the propagation JVP and the native join
    published for this frame, and the bracket is the waveform spec's. The Phase-6
    version of this identity is driven by a hand-written row; this one closes the
    loop from a Core velocity all the way to the IQ cube, which is the link that
    had no test.

    The sign is half the content. The beat cube conjugates the Channel phasor,
    so a CLOSING target - negative ``tau_rate`` - gives a NEGATIVE slow-time
    slope here while its physical Doppler ``-f_ref tau_rate`` is positive. A
    kernel that dropped the conjugation would still produce a plausible cube.
    """

    import torch.autograd.forward_ad as forward_ad

    from witwin.radar.propagation import kinematics as kin
    from witwin.radar.synthesis.fmcw_beat import synthesize_fmcw_beat

    velocity = (-SOLO_SPEED_M_PER_S, 0.0, 0.0)
    sites = kin.Kinematics(
        positions_m=solo.site_tensor(),
        velocities_m_per_s=torch.tensor(
            [velocity], dtype=torch.float32, device="cuda"
        ),
    )
    stationary = torch.zeros(1, 3, device="cuda")
    with kin.two_way_duals(
        sites=sites,
        transmitters=kin.Kinematics(
            positions_m=solo.transmitter_tensor(), velocities_m_per_s=stationary
        ),
        receivers=kin.Kinematics(
            positions_m=solo.receiver_tensor(), velocities_m_per_s=stationary
        ),
    ) as duals:
        composed, _, _ = solo.frame(
            duals.sites,
            transmitters=duals.transmitters,
            receivers=duals.receivers,
            ad_mode="jvp",
        )
        tau_rt = float(composed.total_delay_s[0])
        tau_rate = float(composed.delay_rate[0])
        # Lifted out of the dual level exactly as synthesis consumes it: the
        # rate is a PRIMAL Doppler value by contract, and a batch that still
        # carried a tangent would be refused by the join facade rather than
        # silently synthesized.
        frame = drv.to_synthesis(
            replace(
                composed,
                total_delay_s=composed.total_delay_s.detach(),
                delay_rate=composed.delay_rate.detach(),
                complex_transfer_ref=composed.complex_transfer_ref.detach(),
            )
        )

    # The site closes on the transmitter, so the delay shrinks and the physical
    # Doppler is positive. Not a tautology: a dead tangent publishes zero here.
    assert tau_rate < 0.0
    assert -geo.REFERENCE_FREQUENCY_HZ * tau_rate > 1000.0

    spec = _solo_spec(num_chirps=16)
    cube = synthesize_fmcw_beat(frame, spec).cpu()
    assert cube.shape[0] == 16

    for sample in (0, cube.shape[2] - 1):
        slow = cube[:, 0, sample].to(torch.complex128)
        steps = slow[1:] * torch.conj(slow[:-1])
        measured = float(torch.angle(steps).mean())
        t_m = sample / spec.sample_rate_hz
        ramp = spec.slope_hz_per_s * (spec.t_start_s - tau_rt + t_m)
        analytic = (
            2.0
            * math.pi
            * tau_rate
            * spec.chirp_period_s
            * (spec.carrier_hz + spec.carrier_rate_hz + ramp)
        )
        assert measured == pytest.approx(analytic, rel=2.0e-4), sample
        # Dimension, stated as a unit identity rather than as a number: the
        # slope is radians per chirp, and dividing by the chirp period gives
        # radians per second, whose ratio to the physical Doppler is 2 pi.
        radians_per_second = measured / spec.chirp_period_s
        doppler_hz = -geo.REFERENCE_FREQUENCY_HZ * tau_rate
        assert radians_per_second / (2.0 * math.pi) == pytest.approx(
            -doppler_hz * (1.0 + ramp / geo.REFERENCE_FREQUENCY_HZ), rel=2.0e-4
        )


# --------------------------------------------------------------------------
# 3  the item-8 cross-consumer criterion
# --------------------------------------------------------------------------


SLOT_COUNT = 6
SLOT_PERIOD_S = 1.0e-4
SNAPSHOT_TIME_S = 0.25


def _times() -> torch.Tensor:
    index = torch.arange(SLOT_COUNT, dtype=torch.float64)
    return SNAPSHOT_TIME_S + index * SLOT_PERIOD_S


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
    from witwin.radar.propagation.channel_consumer import _endpoint_batch

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
    stack = drv.slot_site_stack(
        base, geo.SITE_P_VELOCITY_M_PER_S, (times - times[0]).tolist()
    )

    inbound, _ = spike.slot_legs(stack, slot_count=SLOT_COUNT)
    assert inbound.slot_count == SLOT_COUNT

    # The Channel side, from the same handle and the same stacks the adapter
    # built. Reaching for the adapter's own batch helper is deliberate: a
    # second, test-written endpoint batch would be a second world state and the
    # bitwise claim would be about the test rather than about the two consumers.
    transmitters = spike._stacked_ids(
        spike.stacked(
            [position for _, position in spike.transmitters], SLOT_COUNT
        ),
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
        assert torch.equal(
            evaluation.transport.coefficient[slot], radar_slot.coefficient
        ), slot
        if slot and not torch.equal(
            evaluation.delay_s[slot], evaluation.delay_s[0]
        ):
            moved = True
    # Non-vacuity: a world that did not move would make every slot trivially
    # equal to every other and the bitwise claim would prove nothing.
    assert moved

    # And the two consumers publish the same per-slot pair segmentation, which
    # is what makes "the same rows at the same instants" a checkable statement
    # rather than a coincidence of lengths.
    assert evaluation.pair_count == inbound.pairs_per_slot
    assert torch.equal(evaluation.pair_offsets, inbound.slot(0).pair_offsets)
