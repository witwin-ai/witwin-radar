"""Adaptive motion against the independently executed ADC discovery route."""

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


def test_adaptive_options_refuse_invalid_values():
    with pytest.raises(ValueError):
        AdaptiveMotionSpec(phase_error_rad=float("nan"))
    with pytest.raises(ValueError):
        AdaptiveMotionSpec(batch_observations=0)


@pytest.mark.parametrize("baseline", [0, 1])
def test_short_lived_topology_event_is_refined_without_blending_path_identities(baseline):
    from types import SimpleNamespace

    from witwin.radar.simulation import _adaptive_fmcw
    from witwin.radar.synthesis.assembly import FmcwSpec

    times = tuple(index * 1e-6 for index in range(33))
    spec = FmcwSpec(33, 1, 1e-6, 1e-3, 0.0, 0.0, 77e9, carrier_rate_hz=77e9, output_domain="beat")

    def evaluate(queries):
        records = []
        for time in queries:
            count = baseline + int(12e-6 <= time <= 20e-6)
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
    expected = torch.tensor([float(baseline + int(12e-6 <= t <= 20e-6)) for t in times], device="cuda")
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
