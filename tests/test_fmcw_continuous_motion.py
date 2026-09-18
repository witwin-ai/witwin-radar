"""Continuous-delay dechirp and scene sampling against independent oracles."""

import math

import pytest
import torch
from support import multi_endpoint_driver as drv
from support import multi_endpoint_geometry as geo
from test_phase11_simulate_entry import _static_scene

from witwin.radar import PointTargets, Radar
from witwin.radar.synthesis.assembly import FmcwSpec
from witwin.radar.synthesis.fmcw import synthesize_fmcw_rows

pytestmark = pytest.mark.gpu


def _oracle(tau, rate, weight, spec):
    slot = torch.arange(spec.num_chirps, device=tau.device).double()[:, None] * spec.chirp_period_s
    fast = spec.t_start_s + torch.arange(spec.num_samples, device=tau.device).double()[None] * spec.sample_period_s
    delay = tau.double() + rate.double() * (slot + fast)
    # Difference of transmitted and delayed chirp phase; carrier at tau0 lives in weight.
    cycles = spec.reference_frequency_hz * rate.double() * (slot + fast)
    cycles += spec.slope_hz_per_s * delay * (fast - delay / 2)
    beat = weight.to(torch.complex128) * torch.exp(2j * math.pi * cycles)
    return beat if spec.output_domain == "beat" else torch.fft.fft(beat, norm="forward", dim=-1)


@pytest.mark.parametrize("domain", ["beat", "spectrum"])
@pytest.mark.parametrize("velocity", [0.0, 2.0, -20.0])
def test_continuous_delay_primal_and_both_derivatives(domain, velocity):
    spec = FmcwSpec(32, 3, 1 / 5e6, 60e-6, 6e13, 6e-6, 77e9, carrier_rate_hz=77e9, output_domain=domain)
    tau = torch.tensor([2 * 3.7 / 299792458], device="cuda", requires_grad=True)
    rate = torch.tensor([2 * velocity / 299792458], device="cuda", requires_grad=True)
    weight = torch.tensor([0.7 - 0.2j], device="cuda", requires_grad=True)
    offsets = torch.tensor([0, 1], device="cuda", dtype=torch.int64)
    out = synthesize_fmcw_rows(tau, rate, weight, offsets, spec)[:, 0]
    expected = _oracle(tau, rate, weight, spec)
    torch.testing.assert_close(out.to(torch.complex128), expected, rtol=2e-5, atol=3e-6)
    loss_weight = torch.linspace(-0.5, 0.7, out.numel(), device="cuda").reshape(out.shape)
    loss = (out.real * loss_weight + 0.3 * out.imag).sum()
    ref_loss = (expected.real * loss_weight + 0.3 * expected.imag).sum()
    actual_grad = torch.autograd.grad(loss, (tau, rate, weight))
    reference_grad = torch.autograd.grad(ref_loss, (tau, rate, weight))
    for measured, reference in zip(actual_grad, reference_grad, strict=True):
        torch.testing.assert_close(measured, reference, rtol=3e-5, atol=3e-4)
    direction = torch.full_like(rate, 1e-8)
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(rate.detach(), direction)
        measured = torch.autograd.forward_ad.unpack_dual(
            synthesize_fmcw_rows(tau.detach(), dual, weight.detach(), offsets, spec)
        ).tangent[:, 0]
    dt = 1e-3
    finite_difference = (
        _oracle(tau.detach(), rate.detach() + dt * direction, weight.detach(), spec)
        - _oracle(tau.detach(), rate.detach() - dt * direction, weight.detach(), spec)
    ) / (2 * dt)
    torch.testing.assert_close(measured.to(torch.complex128), finite_difference, atol=2e-4, rtol=3e-4)


def test_scene_adc_sampling_matches_radial_motion_including_fast_time():
    config = dict(geo.FIXTURE_RADAR_CONFIG)
    config.update(chirp_per_frame=2, adc_samples=8)
    # No pattern is declared: the element pattern now defaults to isotropic,
    # which is the unit gain this oracle's closed-form phase assumes. The
    # polarization is declared rather than pose-derived so the transverse axis
    # is the fixture's own, which is what the closed form was measured against.
    radar = Radar.from_dict(config, position=(0, 0, 0), look_at=(1, 0, 0), polarization=geo.POLARIZATION)

    def trajectory(t):
        return torch.tensor([[2 + 2 * t, 0.6, 0.0]], device=radar.device)

    result = radar.simulate(
        _static_scene(),
        PointTargets(
            positions=trajectory(0.0),
            amplitude=drv.FIXTURE_AMPLITUDE,
            phase=drv.FIXTURE_PHASE_RAD,
            trajectory=trajectory,
        ),
        times=(0.0,),
        los=True,
        reflections=0,
    )
    spec = radar.waveform_spec()
    assert result.motion_sampling == "adc"
    assert result.discovery_count == spec.num_chirps * spec.num_tx * spec.num_samples
    beat = torch.fft.ifft(result.cube[0], norm="forward", dim=-1)
    for tx in range(spec.num_tx):
        for rx in range(spec.num_rx):
            phases = []
            for chirp in range(spec.num_chirps):
                for m in range(spec.num_samples):
                    local = spec.t_start_s + m * spec.sample_period_s
                    t = (chirp * spec.num_tx + tx) * spec.chirp_period_s + local
                    point = trajectory(t)[0].double()
                    tau = (
                        (point - radar.tx_pos[tx].double()).norm() + (point - radar.rx_pos[rx].double()).norm()
                    ) / 299792458
                    phases.append(
                        2
                        * math.pi
                        * (spec.reference_frequency_hz * tau + spec.slope_hz_per_s * tau * (local - tau / 2))
                    )
            expected = torch.exp(1j * torch.stack(phases)).reshape(spec.num_chirps, spec.num_samples)
            measured = beat[tx, rx] / beat[tx, rx, 0, 0]
            expected = expected / expected[0, 0]
            torch.testing.assert_close((measured / measured.abs()).to(torch.complex128), expected, atol=2e-3, rtol=0)
