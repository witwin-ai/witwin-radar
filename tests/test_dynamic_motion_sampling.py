"""Independent geometric phase oracles for the scene-driven motion sampler."""

import math

import pytest
import torch
from support import multi_endpoint_driver as drv
from test_phase11_simulate_entry import _radar, _static_scene

from witwin.radar import Motion, PointTargets

pytestmark = pytest.mark.gpu


def _targets(positions, *, trajectory=None):
    """The fixture scatterer, authored as the dimensionless strength directly."""

    return PointTargets(
        positions=positions, amplitude=drv.FIXTURE_AMPLITUDE, phase=drv.FIXTURE_PHASE_RAD, trajectory=trajectory
    )


class Orbit:
    def __init__(self, device, *, radius=0.12, frequency=40.0):
        self.device, self.radius, self.frequency = device, radius, frequency

    def __call__(self, t):
        angle = 2 * math.pi * self.frequency * t
        r = self.radius
        return torch.tensor([[2 + r * math.cos(angle), r * math.sin(angle), 0]], device=self.device)


def test_orbit_phase_and_tdm_match_independent_geometric_delay():
    radar = _radar()
    trajectory = Orbit(radar.device)
    result = radar.simulate(
        _static_scene(),
        _targets(trajectory(0.0), trajectory=trajectory),
        times=(0.0,),
        los=True,
        reflections=0,
        motion=Motion.chirp(),
    )
    spec = radar.waveform_spec()
    # Invert only the declared spectrum. Test the first ADC sample of each TX.
    beat = torch.fft.ifft(result.cube[0], dim=-1)
    for tx in range(spec.num_tx):
        for rx in range(spec.num_rx):
            times = [result.sample_times_s[0][c * spec.num_tx + tx] for c in range(spec.num_chirps)]
            positions = torch.cat([trajectory(t) for t in times]).double()
            delay = (
                (positions - radar.tx_pos[tx].double()).norm(dim=-1)
                + (positions - radar.rx_pos[rx].double()).norm(dim=-1)
            ) / 299792458.0
            cycles = spec.reference_frequency_hz * delay + spec.slope_hz_per_s * delay * (spec.t_start_s - delay / 2)
            phase_steps = torch.angle(beat[tx, rx, 1:, 0] * beat[tx, rx, :-1, 0].conj())
            expected = torch.angle(torch.exp(2j * math.pi * (cycles[1:] - cycles[:-1])))
            torch.testing.assert_close(phase_steps.double(), expected, atol=2e-3, rtol=0)
    assert result.path_set_complete
    assert result.discovery_count == spec.num_chirps * spec.num_tx


def test_dynamic_default_discovers_endpoint_born_reflections():
    radar = _radar()

    def crossing(t):
        return torch.tensor([[2.0, 2.4 - 1800 * t, 0.0]], device=radar.device)

    targets = _targets(crossing(0.0), trajectory=crossing)
    times = (0.0, 0.001)
    complete = radar.simulate(_static_scene(), targets, times=times, motion=Motion.chirp())
    held = radar.simulate(_static_scene(), targets, times=times, motion=Motion.chirp(rediscover_every_frames=10))
    assert complete.path_set_complete and not held.path_set_complete
    assert complete.discovery_count > held.discovery_count == 1
    assert not torch.allclose(complete.cube, held.cube, atol=0, rtol=1e-3)


@pytest.mark.parametrize("times", [(0.0, float("nan")), (1.0, 0.0), (0.0, 0.0)])
def test_invalid_frame_times_are_refused(times):
    radar = _radar()
    targets = _targets(torch.tensor([[2.0, 0.2, 0.0]], device=radar.device))
    with pytest.raises(ValueError, match="finite|increasing"):
        radar.simulate(_static_scene(), targets, times=times)


def test_moving_scene_parameter_jvp_preserves_primal_and_matches_reverse():
    from dataclasses import replace

    import torch.autograd.forward_ad as ad

    # A small complete ADC grid, including both transmitters.
    radar = _radar()
    radar = radar.replace(waveform=replace(radar.waveform, chirps_per_frame=1, samples_per_chirp=2))
    base = torch.tensor([[2.0, 0.6, 0.0]], device=radar.device)

    def solve(origin, mode):
        velocity = torch.tensor([[0.5, 0.2, 0.0]], device=origin.device)
        return radar.simulate(
            _static_scene(),
            _targets(origin, trajectory=lambda t: origin + t * velocity),
            times=(0.0,),
            los=True,
            reflections=0,
            grad=mode,
        ).cube

    reference = solve(base, "none")
    leaf = base.clone().requires_grad_()
    reverse = torch.autograd.grad(solve(leaf, "vjp").real.sum(), leaf)[0]
    direction = torch.tensor([[0.3, -0.4, 0.0]], device=base.device)
    for scale in (0.0, 1.0, -2.0):
        with ad.dual_level():
            primal, tangent = ad.unpack_dual(solve(ad.make_dual(base, scale * direction), "jvp"))
            torch.testing.assert_close(primal, reference, rtol=0, atol=0)
            torch.testing.assert_close(tangent.real.sum(), (reverse * (scale * direction)).sum(), rtol=3e-5, atol=1e-8)


def test_moving_sensor_endpoint_binding_and_missing_mapping_refusal():
    from dataclasses import replace

    from witwin.core.dynamics import DynamicScene, LinearTrajectory
    from witwin.core.scene import AntennaState, Scene

    from witwin.radar.simulation import SensorEndpointIds

    radar = _radar()
    radar = radar.replace(waveform=replace(radar.waveform, chirps_per_frame=1, samples_per_chirp=2))
    endpoints = [
        AntennaState(77110 + i, "tx" if i < 2 else "rx", p.cpu())
        for i, p in enumerate(torch.cat([radar.tx_pos, radar.rx_pos]))
    ]
    scene = Scene(structures=(), endpoints=endpoints)
    trajectories = {e.antenna_id: LinearTrajectory(origin=(0, 0, 0), velocity=(0.3, 0, 0)) for e in endpoints}
    dynamic = DynamicScene(scene, endpoint_trajectories=trajectories)
    targets = _targets(torch.tensor([[2.0, 0.0, 0.0]], device=radar.device))
    args = {"times": (0.0,), "los": True, "reflections": 0}
    with pytest.raises(ValueError, match="sensor_endpoints"):
        radar.simulate(dynamic, targets, **args)
    result = radar.simulate(dynamic, targets, endpoints=SensorEndpointIds((77110, 77111), (77112, 77113)), **args)
    last_time = result.sample_times_s[0][-1]
    expected_tx = radar.tx_pos + torch.tensor([0.3 * last_time, 0, 0], device=radar.device)
    torch.testing.assert_close(result.last_propagation.inbound.departure_origin_m, expected_tx, rtol=0, atol=1e-8)
    assert result.path_set_complete and result.motion_sampling == "adc"


def test_moving_wall_round_trip_phase_matches_independent_image_geometry():
    from dataclasses import replace

    from support import multi_endpoint_world as world

    radar = _radar()
    radar = radar.replace(waveform=replace(radar.waveform, chirps_per_frame=1, samples_per_chirp=1))
    scene = world.make_dynamic_scene(wall_velocity=(4.0, 0.0, 0.0))
    site = torch.tensor([[2.0, 0.6, 0.0]], device=radar.device)
    results = [radar.simulate(scene, _targets(site), times=(t,)) for t in (0.0, 1e-4)]
    predicted = []
    for result in results:
        paths, legs = result.last_radar_paths, result.last_propagation
        t = result.sample_times_s[0][-1]
        mirror = site.double().clone()
        wall_x = torch.tensor(4 + 4 * t, dtype=torch.float32, device=radar.device).double()
        mirror[:, 0] = 2 * wall_x - mirror[:, 0]
        inbound = legs.inbound.depth[paths.topology.inbound_row] > 0
        outbound = legs.outbound.depth[paths.topology.outbound_row] > 0
        tx_target = torch.where(inbound[:, None], mirror, site.double())
        rx_target = torch.where(outbound[:, None], mirror, site.double())
        pair = paths.sensor_pair_index
        length = (tx_target - radar.tx_pos[pair % 2].double()).norm(dim=-1)
        length += (rx_target - radar.rx_pos[pair // 2].double()).norm(dim=-1)
        delay = length / 299792458
        torch.testing.assert_close(paths.total_delay_s.double(), delay, rtol=2e-7, atol=1e-14)
        predicted.append(delay)
    first, last = (result.last_radar_paths for result in results)
    torch.testing.assert_close(first.topology.inbound_row, last.topology.inbound_row)
    torch.testing.assert_close(first.topology.outbound_row, last.topology.outbound_row)
    expected = -77e9 * (predicted[1] - predicted[0]) / 1e-4
    measured = torch.angle(last.complex_transfer_ref * first.complex_transfer_ref.conj()).double() / (
        2 * math.pi * 1e-4
    )
    assert float(expected.abs().max()) > 4000
    torch.testing.assert_close(measured, expected, rtol=0, atol=2.0)
