"""Adaptive motion against the independently executed ADC discovery route."""

import math
from dataclasses import replace

import numpy as np
import pytest
import torch
from support.simulate_fixture import fixture_radar, point_targets, static_scene

from witwin.radar import Motion, Noise
from witwin.radar.simulation import AdaptiveMotionSpec

pytestmark = pytest.mark.gpu

#: Oscillator phase noise with the thermal stage held at exactly zero.
#:
#: A zero bandwidth used to say "no thermal noise" and is now refused. A zero
#: antenna temperature with the default zero noise figure says it in physical
#: units instead: ``T_sys`` is zero, so the thermal sigma is exactly zero
#: whatever bandwidth the waveform resolves.
CORRELATED_NOISE = Noise(antenna_temperature=0.0, phase_density=-60, phase_offset=1e6, phase_sample_rate=5e6)


def _beat_radar(radar, *, samples, chirps):
    return radar.replace(
        waveform=replace(radar.waveform, samples_per_chirp=samples, chirps_per_frame=chirps, output="beat")
    )


@pytest.mark.parametrize("noise", [False, True])
@pytest.mark.parametrize("curved", [False, True])
def test_adaptive_matches_adc_with_tdm_and_correlated_noise(noise, curved):
    radar = _beat_radar(fixture_radar(), samples=16, chirps=4)
    if noise:
        radar = radar.replace(noise=CORRELATED_NOISE, seed=7)
    origin = torch.tensor([[2.0, 0.2, 0.0]], device=radar.device)
    velocity = torch.tensor([[0.7, 0.0, 0.0]], device=radar.device)
    acceleration = 300 if curved else 0

    def trajectory(t):
        return origin + velocity * t + acceleration * t * t

    kwargs = {"times": (0.0,), "targets": point_targets(origin, trajectory=trajectory), "los": True, "reflections": 0}
    exact = radar.simulate(static_scene(), **kwargs, motion=Motion.adc())
    adaptive = radar.simulate(static_scene(), **kwargs, motion=Motion.adaptive())
    error = (adaptive.cube - exact.cube).abs().norm() / exact.cube.abs().norm()
    assert error < 0.012, float(error)
    assert adaptive.discovery_count < exact.discovery_count
    assert not adaptive.path_set_complete
    assert adaptive.motion_sampling == "adaptive"
    assert all(
        np.array_equal(left, right) for left, right in zip(adaptive.sample_times_s, exact.sample_times_s, strict=True)
    )
    assert adaptive.adaptive_diagnostics[0]["max_tested_phase_error_rad"] <= 0.02
    with pytest.raises(RuntimeError, match="budget exhausted"):
        radar.simulate(static_scene(), **kwargs, motion=Motion.adaptive(max_evaluations=2))


def test_completeness_and_exhaustiveness_are_reported_separately():
    """An empty LOS world proves the family without enumerating observations."""

    from witwin.core import Scene

    radar = _beat_radar(fixture_radar(), samples=16, chirps=4)
    origin = torch.tensor([[2.0, 0.2, 0.0]], device=radar.device)
    velocity = torch.tensor([[0.7, 0.0, 0.0]], device=radar.device)

    kwargs = {
        "times": (0.0,),
        "targets": point_targets(origin, trajectory=lambda t: origin + velocity * t),
        "los": True,
        "reflections": 0,
    }
    empty = radar.simulate(Scene(structures=(), endpoints=[]), **kwargs, motion=Motion.adaptive())
    assert empty.path_set_complete
    assert not empty.motion_sampling_exhaustive
    assert empty.adaptive_diagnostics[0]["topology_proved_complete"]
    assert not empty.adaptive_diagnostics[0]["exhaustive"]

    exact = radar.simulate(Scene(structures=(), endpoints=[]), **kwargs, motion=Motion.adc())
    assert exact.path_set_complete and exact.motion_sampling_exhaustive

    # A structured world cannot be certified, so neither statement holds.
    structured = radar.simulate(static_scene(), **kwargs, motion=Motion.adaptive())
    assert not structured.path_set_complete
    assert not structured.motion_sampling_exhaustive


def test_the_probe_grid_bound_is_enforced_even_for_a_certified_family():
    """Every tolerance here is sampled, so the grid spacing is the real control.

    A sinusoid whose period divides the probe grid's step sits at a zero of
    every node and every test instant. The controller then measures no error at
    all and accepts an interpolant that misses the whole oscillation. Only
    ``max_interval`` limits that step, which is why a proof that no path can be
    BORN must not relax it: the proof says nothing about how fast one moves.
    """

    from witwin.core import Scene

    radar = fixture_radar()
    radar = radar.replace(
        waveform=replace(
            radar.waveform,
            samples_per_chirp=64,
            chirps_per_frame=64,
            sample_rate=4.0e6,
            adc_start=0.0,
            ramp_end=16e-6,
            idle=0.0,
            output="beat",
        )
    )
    spec = radar.waveform_spec()
    span = (spec.num_chirps * spec.num_tx * spec.num_samples - 1) * spec.sample_period_s
    origin = torch.tensor([[3.0, 0.0, 0.0]], device=radar.device)

    # One whole-frame interval at the default order steps the grid by span/8.
    # This motion completes a full period in exactly that step.
    hidden_hz = 8.0 / span
    amplitude = 4.0e-4  # metres; 1.29 rad of round-trip carrier phase at 77 GHz

    def hidden(t):
        offset = amplitude * math.sin(2 * math.pi * hidden_hz * t)
        return origin + torch.tensor([[offset, 0.0, 0.0]], device=radar.device)

    kwargs = {"times": (0.0,), "targets": point_targets(origin, trajectory=hidden), "los": True, "reflections": 0}
    scene = Scene(structures=(), endpoints=[])
    exact = radar.simulate(scene, **kwargs, motion=Motion.adc())

    def adaptive(cap):
        result = radar.simulate(scene, **kwargs, motion=Motion.adaptive(max_interval=cap))
        error = float((result.cube - exact.cube).abs().norm() / exact.cube.abs().norm())
        return result.adaptive_diagnostics[0], error

    # A bound the caller sized from the motion resolves it.
    resolved, resolved_error = adaptive(1.0 / (16 * hidden_hz))
    assert resolved_error < 0.05, (resolved_error, resolved)

    # The world is certified complete, and that must not buy a coarser grid.
    assert resolved["topology_proved_complete"]
    assert resolved["accepted_intervals"] > 1

    # A bound wider than the frame is the aliasing case, and it is the caller's
    # declaration that produces it rather than something the controller does
    # behind their back. The sampled test still reports success, which is
    # exactly why the bound cannot be inferred.
    aliased, aliased_error = adaptive(4 * span)
    assert aliased["accepted_intervals"] == 1
    assert aliased["max_tested_phase_error_rad"] <= 0.02
    assert aliased_error > 0.5, (aliased_error, aliased)


def test_a_higher_interpolation_order_buys_interval_length_under_the_proof():
    """And is refused the chance to spend probes where it cannot buy any."""

    from witwin.core import Scene

    radar = _beat_radar(fixture_radar(), samples=32, chirps=16)
    # Fast, curved motion: the case where the phase test, not the interval
    # bound, is what shortens an interval.
    origin = torch.tensor([[2.0, 0.2, 0.0]], device=radar.device)
    swing = torch.tensor([[0.0, 0.004, 0.0]], device=radar.device)

    def trajectory(t):
        return origin + swing * math.sin(2 * math.pi * 300.0 * t)

    kwargs = {"times": (0.0,), "targets": point_targets(origin, trajectory=trajectory), "los": True, "reflections": 0}
    exact = radar.simulate(Scene(structures=(), endpoints=[]), **kwargs, motion=Motion.adc())

    def adaptive(nodes):
        result = radar.simulate(Scene(structures=(), endpoints=[]), **kwargs, motion=Motion.adaptive(nodes=nodes))
        stats = result.adaptive_diagnostics[0]
        error = float((result.cube - exact.cube).abs().norm() / exact.cube.abs().norm())
        return stats, error

    linear, linear_error = adaptive(2)
    quartic, quartic_error = adaptive(5)

    assert linear["interpolation_nodes"] == 2
    assert quartic["interpolation_nodes"] == 5
    assert quartic["accepted_intervals"] < linear["accepted_intervals"]
    assert quartic["evaluations"] < linear["evaluations"]
    assert quartic_error < linear_error
    for stats in (linear, quartic):
        assert stats["max_tested_phase_error_rad"] <= 0.02

    # The order is the caller's declaration everywhere, including a structured
    # world; what changes is whether it can buy anything.
    capped = radar.simulate(static_scene(), **kwargs, motion=Motion.adaptive(nodes=9))
    assert capped.adaptive_diagnostics[0]["interpolation_nodes"] == 9


@pytest.mark.parametrize("options", [{"phase_error": float("nan")}, {"batch_observations": 0}, {"nodes": 1}])
def test_adaptive_options_refuse_invalid_values(options):
    """The public knobs are validated where they are turned into the spec."""

    with pytest.raises(ValueError):
        Motion.adaptive(**options)._adaptive_spec()


@pytest.mark.parametrize("baseline", [0, 1])
def test_short_lived_topology_event_is_refined_without_blending_path_identities(baseline):
    from types import SimpleNamespace

    from witwin.radar.simulation import _adaptive_echo, _adaptive_trace
    from witwin.radar.synthesis.assembly import FmcwSpec

    # Long enough that the default order's nine-instant grid can accept whole
    # intervals away from the event, and the event placed on a top-level grid
    # instant so that it is SEEN - an event between probes is the documented
    # blind spot, not what this test is about.
    times = tuple(index * 1e-6 for index in range(129))
    spec = FmcwSpec(129, 1, 1e-6, 1e-3, 0.0, 0.0, 77e9, carrier_rate_hz=77e9, output_domain="beat")

    def evaluate(queries):
        records = []
        for time in queries:
            count = baseline + int(60e-6 <= time <= 68e-6)
            paths = SimpleNamespace(
                total_delay_s=torch.full((count,), 20e-9, device="cuda"),
                complex_transfer_ref=torch.ones(count, device="cuda", dtype=torch.complex64),
                row_valid=torch.ones(count, device="cuda", dtype=torch.bool),
                pair_offsets=torch.tensor([0, count], device="cuda", dtype=torch.int64),
            )
            records.append(
                (SimpleNamespace(topology_complete=False, rediscovered=True), None, paths, tuple(range(count)))
            )
        return records

    # The two halves composed: refinement decides the partition, synthesis
    # reads it. Calling them in sequence here is what the production route
    # does, and it keeps this test measuring the refinement rather than the
    # seam between them.
    options = AdaptiveMotionSpec()
    table, stats, _, _ = _adaptive_trace(times, evaluate, spec, options, 77e9)
    result, _ = _adaptive_echo(table, spec, options, 77e9, None)
    expected = torch.tensor([float(baseline + int(60e-6 <= t <= 68e-6)) for t in times], device="cuda")
    torch.testing.assert_close(result[0, 0].real, expected, rtol=0, atol=0)
    assert stats["topology_refinements"] > 0
    assert stats["evaluations"] < len(times)


def test_adaptive_recompiles_moving_reflectors_and_preserves_scene_adjoint():
    import torch.autograd.forward_ad as ad
    from support import multi_endpoint_world as world

    radar = _beat_radar(fixture_radar(), samples=4, chirps=4)
    origin = torch.tensor([[2.0, 0.6, 0.0]], device=radar.device)

    def solve(point, motion, mode):
        return radar.simulate(
            world.make_dynamic_scene(wall_velocity=(4.0, 0.0, 0.0)),
            point_targets(point),
            times=(0.0,),
            motion=motion,
            grad=mode,
        )

    reference = solve(origin, Motion.adc(), "none")
    result = solve(origin, Motion.adaptive(), "none")
    assert result.compile_count == result.discovery_count
    assert (result.cube - reference.cube).norm() / reference.cube.norm() < 0.012
    leaf = origin.clone().requires_grad_()
    gradient = torch.autograd.grad(solve(leaf, Motion.adaptive(), "vjp").cube.real.sum(), leaf)[0]
    direction = torch.tensor([[0.3, -0.4, 0.0]], device=radar.device)
    with ad.dual_level():
        dual = ad.make_dual(origin, direction)
        primal, tangent = ad.unpack_dual(solve(dual, Motion.adaptive(), "jvp").cube)
        torch.testing.assert_close(primal, result.cube, rtol=0, atol=0)
        torch.testing.assert_close(tangent.real.sum(), (gradient * direction).sum(), rtol=3e-4, atol=1e-7)


def test_batched_multi_site_reflections_keep_site_gradients_and_topology_probes():
    import torch.autograd.forward_ad as ad
    from support import multi_endpoint_world as world

    radar = _beat_radar(fixture_radar(), samples=4, chirps=3)
    origin = torch.tensor([[2.0, 0.6, 0.0], [2.2, -0.4, 0.1]], device=radar.device)
    scene = world.make_dynamic_scene(wall_velocity=(0.0, 0.0, 0.0)).scene

    def solve(point, motion, mode):
        return radar.simulate(
            scene, point_targets(point, trajectory=lambda t: point + t * 0.7), times=(0.0,), motion=motion, grad=mode
        )

    exact = solve(origin, Motion.adc(), "none")
    result = solve(origin, Motion.adaptive(), "none")
    assert result.discovery_count > 1  # Geometry prevents the complete-LOS certificate.
    assert result.last_radar_paths.path_count > 2
    assert (result.cube - exact.cube).norm() / exact.cube.norm() < 0.012
    leaf = origin.clone().requires_grad_()
    gradient = torch.autograd.grad(solve(leaf, Motion.adaptive(), "vjp").cube.real.sum(), leaf)[0]
    direction = torch.tensor([[0.3, -0.4, 0.0], [-0.2, 0.1, 0.5]], device=radar.device)
    with ad.dual_level():
        primal, tangent = ad.unpack_dual(solve(ad.make_dual(origin, direction), Motion.adaptive(), "jvp").cube)
        torch.testing.assert_close(primal, result.cube, rtol=0, atol=0)
        torch.testing.assert_close(tangent.real.sum(), (gradient * direction).sum(), rtol=4e-4, atol=1e-7)
