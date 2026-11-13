"""D-5 / F-4: time-domain OFDM with a cyclic prefix, pilots and seeded QAM data (R-ADR-046).

Oracles:

* a float64 Torch evaluation of the received samples, written from the
  equation in ``ofdm_time.cu`` (the useful waveform as an explicit subcarrier
  sum, the prefix as its periodic extension, the previous symbol past the
  prefix), against the kernel's forward, VJP and JVP;
* the identity that makes the CFR the CFR: for a still echo inside the prefix
  the unitary transform of the samples divided by the symbols is exactly the
  closed-form CFR kernel's cube;
* inter-carrier interference against the textbook law. A delay walking at
  ``tau_dot`` puts a normalised frequency offset ``eps = f_c tau_dot / df`` on
  every subcarrier; the demodulated bin keeps ``|I_0|^2 = (sin(pi eps) / (N
  sin(pi eps / N)))^2`` of its power and leaks ``1 - |I_0|^2`` to the others,
  so the signal-to-ICI ratio is ``|I_0|^2 / (1 - |I_0|^2)``.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
import torch
import torch.autograd.forward_ad as forward_ad
from support import oracle_scenes as sc

from witwin.radar import Motion, Ofdm, PointTargets
from witwin.radar.policy import SPEED_OF_LIGHT_M_PER_S as C0
from witwin.radar.processing import ofdm_cfr_estimate
from witwin.radar.synthesis.assembly import OfdmSpec
from witwin.radar.synthesis.ofdm import synthesize_cfr_rows, synthesize_ofdm_time_rows

pytestmark = pytest.mark.gpu

F_REF_HZ = 77.0e9

SPEC = OfdmSpec(
    num_subcarriers=16,
    num_symbols=3,
    subcarrier_spacing_hz=1.0e6,
    cyclic_prefix_s=0.25e-6,
    reference_frequency_hz=F_REF_HZ,
    max_expected_delay_s=0.2e-6,
    carrier_rate_hz=F_REF_HZ,
    modulation="qam16",
    pilot_spacing=4,
    symbol_seed=11,
    output_domain="time",
)


def oracle(tau0, rate, weight, offsets, spec: OfdmSpec) -> torch.Tensor:
    """``[symbol, segment, sample]`` received samples in float64, from the equation."""

    table = torch.from_numpy(spec.symbol_table())
    n = spec.num_subcarriers
    t_s = spec.waveform_sample_period_s
    t_sym = spec.symbol_period_s
    sub = torch.arange(n, dtype=torch.float64)
    out = torch.zeros((spec.num_symbols, len(offsets) - 1, spec.samples_per_symbol), dtype=torch.complex128)
    samples = torch.arange(spec.samples_per_symbol, dtype=torch.float64)
    for symbol in range(spec.num_symbols):
        t_loc = spec.cyclic_prefix_s + samples * t_s
        t_abs = symbol * t_sym + t_loc
        for segment in range(len(offsets) - 1):
            for k in range(offsets[segment], offsets[segment + 1]):
                drift = rate[k] * t_abs
                tau = tau0[k] + drift
                u = t_loc - tau
                shift = torch.floor(u.detach() / t_sym)
                j = symbol + shift.to(torch.int64)
                t = u - shift * t_sym - spec.cyclic_prefix_s
                live = (j >= 0) & (j < spec.num_symbols)
                rows = table[j.clamp(0, spec.num_symbols - 1)]
                x = (rows * torch.exp(2j * math.pi * spec.subcarrier_spacing_hz * t[:, None] * sub[None, :])).sum(
                    -1
                ) / math.sqrt(n)
                cycles = -(spec.carrier_rate_hz * drift + spec.carrier_hz * tau)
                out[symbol, segment] = out[symbol, segment] + weight[k] * x * torch.exp(2j * math.pi * cycles) * live
    return out


DELAYS = (0.13e-6, 0.41e-6)  # the second is past the prefix: ISI
RATES = (2.0e-7, -1.5e-7)
WEIGHTS = (0.6 - 0.3j, -0.2 + 0.45j)
OFFSETS = (0, 1, 2)


def _inputs(device, dtype, cdtype):
    return (
        torch.tensor(DELAYS, dtype=dtype, device=device),
        torch.tensor(RATES, dtype=dtype, device=device),
        torch.tensor(WEIGHTS, dtype=cdtype, device=device),
    )


def _kernel(tau, rate, weight, spec=SPEC):
    return synthesize_ofdm_time_rows(tau, rate, weight, torch.tensor(OFFSETS, device="cuda"), spec)


def test_the_symbol_table_is_seeded_unit_power_with_a_pilot_comb():
    table = SPEC.symbol_table()
    assert np.array_equal(table, SPEC.symbol_table())
    assert not np.array_equal(table, replace(SPEC, symbol_seed=12).symbol_table())
    assert np.all(table[:, ::4] == 1.0)
    big = replace(SPEC, num_subcarriers=512, num_symbols=64, pilot_spacing=0).symbol_table()
    assert float(np.mean(np.abs(big) ** 2)) == pytest.approx(1.0, rel=0.02)
    assert set(np.round(big.real * math.sqrt(10)).astype(int).ravel()) == {-3, -1, 1, 3}


def test_the_kernel_is_the_float64_closed_form_including_isi():
    tau, rate, weight = _inputs("cuda", torch.float32, torch.complex64)
    cube = _kernel(tau, rate, weight).cpu().to(torch.complex128)
    o = _inputs("cpu", torch.float64, torch.complex128)
    expected = oracle(o[0].float().double(), o[1].float().double(), o[2], list(OFFSETS), SPEC)
    assert float((cube - expected).abs().max()) < 2e-5 * float(expected.abs().max())


def test_a_still_echo_inside_the_prefix_demodulates_to_the_closed_form_cfr():
    spec = replace(SPEC, num_subcarriers=64, num_symbols=4)
    tau = torch.tensor((0.13e-6, 0.21e-6), device="cuda")
    weight = torch.tensor(WEIGHTS, dtype=torch.complex64, device="cuda")
    offsets = torch.tensor(OFFSETS, device="cuda")
    samples = synthesize_ofdm_time_rows(tau, None, weight, offsets, spec)
    symbols = torch.from_numpy(spec.symbol_table()).cuda()
    estimate = ofdm_cfr_estimate(samples.permute(1, 0, 2).to(torch.complex128), symbols).permute(1, 0, 2)
    ideal = synthesize_cfr_rows(tau, None, weight, offsets, replace(spec, modulation="ones", output_domain="cfr"))
    assert float((estimate - ideal).abs().max()) < 2e-5


def test_the_native_vjp_and_jvp_follow_the_oracle():
    torch.manual_seed(9)
    target = torch.randn((SPEC.num_symbols, 2, SPEC.num_subcarriers), dtype=torch.complex128)
    o_tau, o_rate, o_weight = (x.clone().requires_grad_(True) for x in _inputs("cpu", torch.float64, torch.complex128))
    (oracle(o_tau, o_rate, o_weight, list(OFFSETS), SPEC) - target).abs().pow(2).sum().backward()
    tau, rate, weight = (x.clone().requires_grad_(True) for x in _inputs("cuda", torch.float32, torch.complex64))
    (_kernel(tau, rate, weight).cpu().to(torch.complex128) - target).abs().pow(2).sum().backward()
    for got, want in ((tau.grad, o_tau.grad), (rate.grad, o_rate.grad), (weight.grad, o_weight.grad)):
        got = got.cpu().to(want.dtype)
        assert float((got - want).abs().max()) < 2e-3 * float(want.abs().max())

    d_tau = torch.tensor((1e-9, -2e-9))
    d_rate = torch.tensor((1e-7, -1e-7))
    d_weight = torch.tensor((0.1 + 0.2j, -0.3 + 0.05j))
    with forward_ad.dual_level():
        cube = _kernel(
            forward_ad.make_dual(tau.detach(), d_tau.cuda()),
            forward_ad.make_dual(rate.detach(), d_rate.cuda()),
            forward_ad.make_dual(weight.detach(), d_weight.to(torch.complex64).cuda()),
        )
        tangent = forward_ad.unpack_dual(cube).tangent.cpu().to(torch.complex128)
    reference = torch.autograd.functional.jvp(
        lambda a, b, c: oracle(a, b, c, list(OFFSETS), SPEC),
        (o_tau.detach(), o_rate.detach(), o_weight.detach()),
        (d_tau.double(), d_rate.double(), d_weight),
    )[1]
    assert float((tangent - reference).abs().max()) < 2e-3 * float(reference.abs().max())


# --------------------------------------------------------------------------
# Inter-carrier interference against the textbook law
# --------------------------------------------------------------------------


def _signal_to_ici(samples: torch.Tensor, spec: OfdmSpec, tau: float) -> float:
    """``|projection|^2 / residual`` of the demodulated bins on the still-echo CFR."""

    symbols = torch.from_numpy(spec.symbol_table()).to(samples.device)
    received = torch.fft.fft(samples.to(torch.complex128), dim=-1, norm="ortho")
    ideal = symbols * torch.exp(
        -2j * math.pi * spec.subcarrier_spacing_hz * tau * torch.arange(spec.num_subcarriers, device=samples.device)
    )
    alpha = (received * ideal.conj()).sum(-1, keepdim=True) / (ideal.abs() ** 2).sum(-1, keepdim=True)
    residual = (received - alpha * ideal).abs().pow(2).sum()
    return float((alpha.abs() ** 2 * (ideal.abs() ** 2).sum(-1, keepdim=True)).sum() / residual)


@pytest.mark.parametrize("epsilon", [0.05, 0.1, 0.2, 0.3])
def test_ici_power_follows_the_normalised_doppler_law(epsilon):
    spec = replace(SPEC, num_subcarriers=256, num_symbols=8, subcarrier_spacing_hz=120e3, cyclic_prefix_s=1e-6)
    spec = replace(spec, max_expected_delay_s=0.5e-6, pilot_spacing=0)
    rate = epsilon * spec.subcarrier_spacing_hz / F_REF_HZ
    tau = 0.3e-6
    samples = synthesize_ofdm_time_rows(
        torch.tensor((tau,), device="cuda"),
        torch.tensor((rate,), device="cuda"),
        torch.ones(1, dtype=torch.complex64, device="cuda"),
        torch.tensor((0, 1), device="cuda"),
        spec,
    )[:, 0]
    n = spec.num_subcarriers
    i0 = (math.sin(math.pi * epsilon) / (n * math.sin(math.pi * epsilon / n))) ** 2
    expected_db = 10 * math.log10(i0 / (1.0 - i0))
    measured_db = 10 * math.log10(_signal_to_ici(samples, spec, tau))
    # The delay also walks across each symbol (2e-12 s), far below df^-1.
    assert measured_db == pytest.approx(expected_db, abs=0.5)


def test_a_still_echo_has_no_ici_and_a_moving_scene_shows_it_end_to_end():
    waveform = Ofdm(
        subcarrier_spacing=120e3,
        num_subcarriers=64,
        cyclic_prefix=1e-6,
        num_symbols=4,
        max_expected_delay=0.5e-6,
        modulation="qpsk",
        symbol_seed=5,
        output="time",
    )
    radar = sc.radar(waveform)
    range_m = 3.0
    origin = torch.tensor([[range_m, 0.0, 0.0]], device=radar.device)
    # eps = 0.2 at 77 GHz: tau_dot = 0.2 df / f_c, a radial speed of 47 m/s.
    speed = 0.2 * 120e3 / sc.CARRIER_HZ * C0 / 2.0
    moving = PointTargets(
        positions=origin,
        rcs=1.0,
        trajectory=lambda t: origin + speed * t * torch.tensor([[1.0, 0.0, 0.0]], device=radar.device),
    )
    result = radar.simulate(sc.empty_scene(), moving, times=(0.0,), los=True, reflections=0, motion=Motion.adc())
    assert result.observation_schedule == "fast_time"
    spec = radar.waveform_spec()
    samples = result.cube[0, 0, 0]
    tau = 2.0 * range_m / C0
    i0 = (math.sin(math.pi * 0.2) / (64 * math.sin(math.pi * 0.2 / 64))) ** 2
    measured_db = 10 * math.log10(_signal_to_ici(samples, spec, tau))
    assert measured_db == pytest.approx(10 * math.log10(i0 / (1.0 - i0)), abs=1.0)

    still = radar.simulate(
        sc.empty_scene(), PointTargets(positions=origin, rcs=1.0), times=(0.0,), los=True, reflections=0
    )
    assert 10 * math.log10(_signal_to_ici(still.cube[0, 0, 0], spec, tau)) > 60.0
