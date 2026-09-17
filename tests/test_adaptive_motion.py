"""Adaptive motion against the independently executed ADC discovery route."""

import math
from dataclasses import replace

import pytest
import torch
from test_phase11_simulate_entry import _radar, _response, _static_scene

from witwin.radar.frontend import FrontendChain, FrontendSpec, NoiseSpec, SeedSpec
from witwin.radar.propagation import Kinematics
from witwin.radar.simulation import AdaptiveMotionSpec, ScatterSitePolicy

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("noise", [False, True])
@pytest.mark.parametrize("curved", [False, True])
def test_adaptive_matches_adc_with_tdm_and_correlated_noise(noise, curved):
    radar = _radar()
    radar.system_config = replace(
        radar.system_config,
        waveform=replace(radar.system_config.waveform, adc_samples=16, chirp_per_frame=4, output_domain="beat"),
    )
    if noise:
        radar.frontend = FrontendChain(
            FrontendSpec(
                noise=NoiseSpec(
                    bandwidth_hz=0, phase_noise_dbc_per_hz=-60, phase_offset_hz=1e6, phase_sample_rate_hz=5e6
                ),
                seed=SeedSpec(7),
            )
        )
    origin = torch.tensor([[2.0, 0.2, 0.0]], device=radar.device)
    velocity = torch.tensor([[0.7, 0.0, 0.0]], device=radar.device)

    class Motion:
        def at(self, t):
            acceleration = 300 if curved else 0
            return Kinematics(origin + velocity * t + acceleration * t * t, velocity + 2 * acceleration * t)

    kwargs = {
        "times": (0.0,),
        "response": _response(radar),
        "sites": ScatterSitePolicy.explicit(origin, trajectory=Motion()),
        "components": frozenset({"los"}),
        "max_depth": 0,
    }
    exact = radar.simulate(_static_scene(), **kwargs, motion_sampling="adc")
    adaptive = radar.simulate(_static_scene(), **kwargs, motion_sampling="adaptive")
    error = (adaptive.cube - exact.cube).abs().norm() / exact.cube.abs().norm()
    assert error < 0.012, float(error)
    assert adaptive.discovery_count < exact.discovery_count
    assert not adaptive.path_set_complete
    assert adaptive.motion_sampling == "adaptive"
    assert adaptive.sample_times_s == exact.sample_times_s
    assert adaptive.adaptive_diagnostics[0]["max_tested_phase_error_rad"] <= 0.02
    with pytest.raises(RuntimeError, match="budget exhausted"):
        radar.simulate(
            _static_scene(), **kwargs, motion_sampling="adaptive", adaptive_motion=AdaptiveMotionSpec(max_evaluations=2)
        )


def test_completeness_and_exhaustiveness_are_reported_separately():
    """An empty LOS world proves the family without enumerating observations."""

    from witwin.core import Scene

    radar = _radar()
    radar.system_config = replace(
        radar.system_config,
        waveform=replace(radar.system_config.waveform, adc_samples=16, chirp_per_frame=4, output_domain="beat"),
    )
    origin = torch.tensor([[2.0, 0.2, 0.0]], device=radar.device)
    velocity = torch.tensor([[0.7, 0.0, 0.0]], device=radar.device)

    class Motion:
        def at(self, t):
            return Kinematics(origin + velocity * t, velocity)

    kwargs = {
        "times": (0.0,),
        "response": _response(radar),
        "sites": ScatterSitePolicy.explicit(origin, trajectory=Motion()),
        "components": frozenset({"los"}),
        "max_depth": 0,
    }
    empty = radar.simulate(Scene(structures=(), endpoints=[]), **kwargs, motion_sampling="adaptive")
    assert empty.path_set_complete
    assert not empty.motion_sampling_exhaustive
    assert empty.adaptive_diagnostics[0]["topology_proved_complete"]
    assert not empty.adaptive_diagnostics[0]["exhaustive"]

    exact = radar.simulate(Scene(structures=(), endpoints=[]), **kwargs, motion_sampling="adc")
    assert exact.path_set_complete and exact.motion_sampling_exhaustive

    # A structured world cannot be certified, so neither statement holds.
    structured = radar.simulate(_static_scene(), **kwargs, motion_sampling="adaptive")
    assert not structured.path_set_complete
    assert not structured.motion_sampling_exhaustive


def test_the_probe_grid_bound_is_enforced_even_for_a_certified_family():
    """Every tolerance here is sampled, so the grid spacing is the real control.

    A sinusoid whose period divides the probe grid's step sits at a zero of
    every node and every test instant. The controller then measures no error at
    all and accepts an interpolant that misses the whole oscillation. Only
    ``max_interval_s`` limits that step, which is why a proof that no path can
    be BORN must not relax it: the proof says nothing about how fast one moves.
    """

    import math

    from witwin.core import Scene

    radar = _radar()
    radar.system_config = replace(
        radar.system_config,
        waveform=replace(
            radar.system_config.waveform,
            adc_samples=64,
            chirp_per_frame=64,
            sample_rate=4000,
            adc_start_time=0,
            ramp_end_time=16,
            idle_time=0,
            output_domain="beat",
        ),
    )
    spec = radar.system_config.waveform_spec()
    span = (spec.num_chirps * spec.num_tx * spec.num_samples - 1) * spec.sample_period_s
    origin = torch.tensor([[3.0, 0.0, 0.0]], device=radar.device)

    # One whole-frame interval at the default order steps the grid by span/8.
    # This motion completes a full period in exactly that step.
    hidden_hz = 8.0 / span
    amplitude = 4.0e-4  # metres; 1.29 rad of round-trip carrier phase at 77 GHz

    class Hidden:
        def at(self, t):
            phase = 2 * math.pi * hidden_hz * t
            offset = torch.tensor([[amplitude * math.sin(phase), 0.0, 0.0]], device=radar.device)
            rate = amplitude * 2 * math.pi * hidden_hz * math.cos(phase)
            return Kinematics(origin + offset, torch.tensor([[rate, 0.0, 0.0]], device=radar.device))

    kwargs = {
        "times": (0.0,),
        "response": _response(radar),
        "sites": ScatterSitePolicy.explicit(origin, trajectory=Hidden()),
        "components": frozenset({"los"}),
        "max_depth": 0,
    }
    scene = Scene(structures=(), endpoints=[])
    exact = radar.simulate(scene, **kwargs, motion_sampling="adc")

    def adaptive(cap):
        result = radar.simulate(
            scene, **kwargs, motion_sampling="adaptive", adaptive_motion=AdaptiveMotionSpec(max_interval_s=cap)
        )
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

    radar = _radar()
    radar.system_config = replace(
        radar.system_config,
        waveform=replace(radar.system_config.waveform, adc_samples=32, chirp_per_frame=16, output_domain="beat"),
    )
    # Fast, curved motion: the case where the phase test, not the interval
    # bound, is what shortens an interval.
    origin = torch.tensor([[2.0, 0.2, 0.0]], device=radar.device)

    class Motion:
        def at(self, t):
            offset = torch.tensor([[0.0, 0.004, 0.0]], device=radar.device) * math.sin(2 * math.pi * 300.0 * t)
            rate = torch.tensor([[0.0, 0.004, 0.0]], device=radar.device) * math.cos(2 * math.pi * 300.0 * t)
            return Kinematics(origin + offset, rate * 2 * math.pi * 300.0)

    kwargs = {
        "times": (0.0,),
        "response": _response(radar),
        "sites": ScatterSitePolicy.explicit(origin, trajectory=Motion()),
        "components": frozenset({"los"}),
        "max_depth": 0,
    }
    exact = radar.simulate(Scene(structures=(), endpoints=[]), **kwargs, motion_sampling="adc")

    def adaptive(nodes):
        result = radar.simulate(
            Scene(structures=(), endpoints=[]),
            **kwargs,
            motion_sampling="adaptive",
            adaptive_motion=AdaptiveMotionSpec(interpolation_nodes=nodes),
        )
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
    capped = radar.simulate(
        _static_scene(), **kwargs, motion_sampling="adaptive", adaptive_motion=AdaptiveMotionSpec(interpolation_nodes=9)
    )
    assert capped.adaptive_diagnostics[0]["interpolation_nodes"] == 9


def test_adaptive_options_refuse_invalid_values():
    with pytest.raises(ValueError):
        AdaptiveMotionSpec(phase_error_rad=float("nan"))
    with pytest.raises(ValueError):
        AdaptiveMotionSpec(batch_observations=0)
    with pytest.raises(ValueError):
        AdaptiveMotionSpec(interpolation_nodes=1)


@pytest.mark.parametrize("baseline", [0, 1])
def test_short_lived_topology_event_is_refined_without_blending_path_identities(baseline):
    from types import SimpleNamespace

    from witwin.radar.simulation import _adaptive_fmcw
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

    result, stats, _, _ = _adaptive_fmcw(times, evaluate, spec, AdaptiveMotionSpec(), 77e9, None)
    expected = torch.tensor([float(baseline + int(60e-6 <= t <= 68e-6)) for t in times], device="cuda")
    torch.testing.assert_close(result[0, 0].real, expected, rtol=0, atol=0)
    assert stats["topology_refinements"] > 0
    assert stats["evaluations"] < len(times)


def test_adaptive_recompiles_moving_reflectors_and_preserves_scene_adjoint():
    import torch.autograd.forward_ad as ad
    from support import multi_endpoint_world as world

    radar = _radar()
    radar.system_config = replace(
        radar.system_config,
        waveform=replace(radar.system_config.waveform, adc_samples=4, chirp_per_frame=4, output_domain="beat"),
    )
    origin = torch.tensor([[2.0, 0.6, 0.0]], device=radar.device)

    def solve(point, sampling, mode):
        return radar.simulate(
            world.make_dynamic_scene(wall_velocity=(4.0, 0.0, 0.0)),
            times=(0.0,),
            sites=ScatterSitePolicy.explicit(point),
            response=_response(radar),
            motion_sampling=sampling,
            ad_mode=mode,
        )

    reference = solve(origin, "adc", "none")
    result = solve(origin, "adaptive", "none")
    assert result.compile_count == result.discovery_count
    assert (result.cube - reference.cube).norm() / reference.cube.norm() < 0.012
    leaf = origin.clone().requires_grad_()
    gradient = torch.autograd.grad(solve(leaf, "adaptive", "vjp").cube.real.sum(), leaf)[0]
    direction = torch.tensor([[0.3, -0.4, 0.0]], device=radar.device)
    with ad.dual_level():
        dual = ad.make_dual(origin, direction)
        primal, tangent = ad.unpack_dual(solve(dual, "adaptive", "jvp").cube)
        torch.testing.assert_close(primal, result.cube, rtol=0, atol=0)
        torch.testing.assert_close(tangent.real.sum(), (gradient * direction).sum(), rtol=3e-4, atol=1e-7)


def test_batched_multi_site_reflections_keep_site_gradients_and_topology_probes():
    import torch.autograd.forward_ad as ad
    from support import multi_endpoint_world as world

    radar = _radar()
    radar.system_config = replace(
        radar.system_config,
        waveform=replace(radar.system_config.waveform, adc_samples=4, chirp_per_frame=3, output_domain="beat"),
    )
    origin = torch.tensor([[2.0, 0.6, 0.0], [2.2, -0.4, 0.1]], device=radar.device)
    scene = world.make_dynamic_scene(wall_velocity=(0.0, 0.0, 0.0)).scene

    def solve(point, sampling, mode):
        class Motion:
            def at(self, t):
                return Kinematics(point + t * 0.7, torch.full_like(point, 0.7))

        return radar.simulate(
            scene,
            times=(0.0,),
            sites=ScatterSitePolicy.explicit(point, trajectory=Motion()),
            response=_response(radar),
            motion_sampling=sampling,
            ad_mode=mode,
        )

    exact = solve(origin, "adc", "none")
    result = solve(origin, "adaptive", "none")
    assert result.discovery_count > 1  # Geometry prevents the complete-LOS certificate.
    assert result.last_radar_paths.path_count > 2
    assert (result.cube - exact.cube).norm() / exact.cube.norm() < 0.012
    leaf = origin.clone().requires_grad_()
    gradient = torch.autograd.grad(solve(leaf, "adaptive", "vjp").cube.real.sum(), leaf)[0]
    direction = torch.tensor([[0.3, -0.4, 0.0], [-0.2, 0.1, 0.5]], device=radar.device)
    with ad.dual_level():
        primal, tangent = ad.unpack_dual(solve(ad.make_dual(origin, direction), "adaptive", "jvp").cube)
        torch.testing.assert_close(primal, result.cube, rtol=0, atol=0)
        torch.testing.assert_close(tangent.real.sum(), (gradient * direction).sum(), rtol=4e-4, atol=1e-7)
