"""Coded, periodic and stepped pulses on the native pulsed family (R-ADR-046).

Three oracles, none of which shares an expression with the kernel:

* a float64 Torch evaluation of the closed form in ``pulsed_echo.cu``, written
  here from the equation (chip index, chip phase, chip frequency, stepped
  carrier, periodic wrap) and itself checked against float64 central
  differences before the native VJP and JVP are compared against it;
* textbook compression results: an m-sequence's periodic autocorrelation is
  ``N`` at lag zero and ``-1`` elsewhere, so the cyclic matched filter peaks at
  exactly ``C`` with every sidelobe at ``-C / N``; a stepped-frequency train's
  inverse transform across pulses peaks at ``C`` in bin ``L df tau``;
* separate launches, one per segment, against one launch with a per-segment
  schedule: bit for bit.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
import torch
import torch.autograd.forward_ad as forward_ad
from support import fd

from witwin.radar import Pulsed
from witwin.radar.processing import matched_filter, stepped_frequency_profile
from witwin.radar.synthesis.assembly import PulsedSpec
from witwin.radar.synthesis.pulsed import synthesize_echo_rows

pytestmark = pytest.mark.gpu

F_REF_HZ = 77.0e9


def _spec(waveform: Pulsed, **overrides) -> PulsedSpec:
    spec = waveform.to_spec(carrier=F_REF_HZ)
    return replace(spec, **overrides) if overrides else spec


def fsk_spec(**overrides) -> PulsedSpec:
    """A 6-chip Costas FSK pulse, 0.4 us chips, 2.5 MHz tone spacing, 50 MS/s."""

    tones = tuple(2.5e6 * value for value in Pulsed.costas(7))
    waveform = Pulsed.fsk(tones, chip_width=0.4e-6, pri=20e-6, num_pulses=3, sample_rate=50e6, num_samples=256)
    return _spec(waveform, **overrides)


def stepped_spec(**overrides) -> PulsedSpec:
    waveform = Pulsed.stepped_frequency(
        step=2.0e6, num_steps=16, pulse_width=1.0e-6, pri=10e-6, sample_rate=20e6, num_samples=64
    )
    return _spec(waveform, **overrides)


def pmcw_spec(order: int = 7, **overrides) -> PulsedSpec:
    waveform = Pulsed.pmcw(Pulsed.m_sequence(order), chip_rate=50e6, repetitions=4, sample_rate=50e6)
    return _spec(waveform, **overrides)


# --------------------------------------------------------------------------
# The float64 oracle
# --------------------------------------------------------------------------


def oracle_cube(tau_rt, rate, weight, offsets, spec: PulsedSpec, schedule=None) -> torch.Tensor:
    """The closed form of ``pulsed_echo.cu`` in float64 Torch, differentiable."""

    segments = len(offsets) - 1
    pulses = torch.arange(spec.num_pulses, dtype=torch.float64).reshape(-1, 1)
    samples = torch.arange(spec.num_samples, dtype=torch.float64).reshape(1, -1)
    t_l = pulses * spec.pri_s
    width = spec.pulse_width_s
    chips = [complex(c) for c in spec.chips]
    freqs = list(spec.chip_frequencies_hz) or [0.0] * len(chips)
    out = torch.zeros((spec.num_pulses, segments, spec.num_samples), dtype=torch.complex128)
    for segment in range(segments):
        if schedule is None:
            start = spec.range_gate_start_s
            step = pulses * spec.frequency_step_hz
        else:
            start, step = float(schedule[segment][0]), torch.full_like(pulses, float(schedule[segment][1]))
        t_m = start + samples * spec.sample_period_s
        for row in range(offsets[segment], offsets[segment + 1]):
            drift = rate[row] * t_l
            tau = tau_rt[row] + drift
            u = t_m - tau
            if spec.periodic:
                u = u - torch.floor(u.detach() / width) * width
            cycles = -(spec.carrier_rate_hz * drift + (spec.carrier_hz + step) * tau)
            inside = ((u >= 0.0) & (u < width)).to(torch.float64)
            amplitude = inside * spec.pulse_amplitude
            if spec.is_coded:
                chip_width = width / len(chips)
                index = torch.clamp(torch.floor(u.detach() / chip_width), 0, len(chips) - 1).to(torch.int64)
                a = torch.tensor(chips, dtype=torch.complex128)[index]
                f = torch.tensor(freqs, dtype=torch.float64)[index]
                cycles = cycles + f * (u - index.to(torch.float64) * chip_width)
                envelope = amplitude * a
            elif spec.is_linear_fm:
                cycles = cycles + 0.5 * spec.bandwidth_hz * u * u / width
                envelope = amplitude.to(torch.complex128)
            else:
                envelope = amplitude.to(torch.complex128)
            out[:, segment, :] = out[:, segment, :] + weight[row] * envelope * torch.exp(2j * math.pi * cycles)
    return out


def _loss(cube, target):
    delta = cube.to(torch.complex128) - target
    return (delta.real**2 + delta.imag**2).sum()


DELAYS = (1.01e-6, 1.523e-6)
RATES = (3.0e-8, -2.0e-8)
WEIGHTS = (0.6 - 0.3j, -0.2 + 0.45j)
OFFSETS = (0, 1, 2)


def _inputs(device, dtype, cdtype):
    return (
        torch.tensor(DELAYS, dtype=dtype, device=device),
        torch.tensor(RATES, dtype=dtype, device=device),
        torch.tensor(WEIGHTS, dtype=cdtype, device=device),
    )


@pytest.mark.parametrize("make", [fsk_spec, stepped_spec, pmcw_spec], ids=["fsk", "stepped", "pmcw"])
def test_the_kernel_is_the_float64_closed_form(make):
    spec = make()
    tau, rate, weight = _inputs("cuda", torch.float32, torch.complex64)
    offsets = torch.tensor(OFFSETS, dtype=torch.int64, device="cuda")
    cube = synthesize_echo_rows(tau, rate, weight, offsets, spec).cpu().to(torch.complex128)
    o_tau, o_rate, o_weight = _inputs("cpu", torch.float64, torch.complex128)
    expected = oracle_cube(o_tau.float().double(), o_rate.float().double(), o_weight, list(OFFSETS), spec)
    scale = float(expected.abs().max())
    assert scale > 0.5
    assert float((cube - expected).abs().max()) < 2e-4 * scale


def test_a_coded_envelope_only_rotates_and_scales_a_chip():
    """A chip of magnitude 0.5 halves the sample; a phase of 90 degrees rotates it."""

    spec = _spec(
        Pulsed.arbitrary((1.0, 0.5j, -0.25), rate=10e6, pri=10e-6, num_pulses=1, sample_rate=10e6, num_samples=8)
    )
    offsets = torch.tensor((0, 1), dtype=torch.int64, device="cuda")
    # Half a chip past sample 2, so every sample reads one chip unambiguously.
    tau = torch.tensor((1.5e-7,), device="cuda")
    cube = synthesize_echo_rows(tau, None, torch.ones(1, dtype=torch.complex64, device="cuda"), offsets, spec)[0, 0]
    # The carrier rate is the production placement, so a static row carries no
    # carrier phase of its own: samples 2, 3, 4 are the three chips verbatim.
    expected = torch.tensor((0, 0, 1.0, 0.5j, -0.25, 0, 0, 0), dtype=torch.complex64)
    assert torch.allclose(cube.cpu(), expected, atol=1e-6)


# --------------------------------------------------------------------------
# Compression oracles
# --------------------------------------------------------------------------


def test_an_m_sequence_compresses_to_c_with_every_cyclic_sidelobe_at_minus_c_over_n():
    spec = pmcw_spec(7)
    n = len(spec.chips)
    assert spec.num_samples == n
    weight = torch.tensor((0.3 - 0.4j,), dtype=torch.complex64, device="cuda")
    lag = 37
    # A thousandth of a sample short of the lag: float32 rounding cannot move a
    # sample across a chip boundary, and each sample reads chip ``m - lag``.
    tau = torch.tensor(((lag - 1e-3) * spec.sample_period_s,), dtype=torch.float32, device="cuda")
    offsets = torch.tensor((0, 1), dtype=torch.int64, device="cuda")
    cube = synthesize_echo_rows(tau, None, weight, offsets, spec)
    profile = matched_filter(cube[0, 0].to(torch.complex128), spec).cpu()
    expected = torch.full((n,), -complex(weight[0]) / n, dtype=torch.complex128)
    expected[lag] = complex(weight[0])
    assert torch.allclose(profile, expected, atol=2e-6)
    # Every repetition sees the same cyclic shift: the code never stops.
    assert torch.allclose(cube[1:].cpu(), cube[:1].expand(3, -1, -1).cpu(), atol=1e-6)


def test_a_costas_fsk_pulse_compresses_to_its_coefficient_at_its_delay():
    spec = fsk_spec()
    weight = torch.tensor((0.7 + 0.2j,), dtype=torch.complex64, device="cuda")
    on_grid = 50
    tau = torch.tensor(((on_grid - 1e-3) * spec.sample_period_s,), dtype=torch.float32, device="cuda")
    offsets = torch.tensor((0, 1), dtype=torch.int64, device="cuda")
    cube = synthesize_echo_rows(tau, None, weight, offsets, spec)
    profile = matched_filter(cube[0, 0].to(torch.complex128), spec).cpu()
    peak = int(profile.abs().argmax())
    assert peak == on_grid
    # The thousandth-sample offset costs f_i 2e-11 s of tone phase, below 2e-3 rad.
    assert abs(complex(profile[peak]) - complex(weight[0])) < 5e-3 * abs(complex(weight[0]))
    # The mainlobe is about T_c / N wide (3.3 samples here); beyond two of
    # those, a Costas code's zero-Doppler response stays far below the peak.
    off_peak = torch.cat((profile[: peak - 7], profile[peak + 8 :])).abs().max()
    assert float(off_peak) < 0.3 * abs(complex(weight[0]))


def test_a_stepped_frequency_train_inverts_to_a_synthetic_range_profile():
    spec = stepped_spec()
    weight = torch.tensor((0.4 - 0.1j,), dtype=torch.complex64, device="cuda")
    # 3 synthetic bins of 1 / (16 x 2 MHz) = 31.25 ns, inside the 1 us pulse.
    tau_s = 3 * (1.0 / (16 * spec.frequency_step_hz))
    tau = torch.tensor((tau_s,), dtype=torch.float32, device="cuda")
    offsets = torch.tensor((0, 1), dtype=torch.int64, device="cuda")
    cube = synthesize_echo_rows(tau, None, weight, offsets, spec)
    profile, delays = stepped_frequency_profile(cube[:, 0, :].to(torch.complex128), spec)
    sample = 10  # inside the echo: tau + 10 T_s < tau + T_p
    column = profile[:, sample].cpu()
    peak = int(column.abs().argmax())
    assert float(delays[peak]) == pytest.approx(tau_s, rel=1e-9)
    assert abs(complex(column[peak]) - complex(weight[0])) < 2e-4


# --------------------------------------------------------------------------
# AD: the oracle, then the native companions against it
# --------------------------------------------------------------------------


@pytest.mark.parametrize("make", [fsk_spec, stepped_spec], ids=["fsk", "stepped"])
def test_the_native_vjp_and_jvp_follow_the_oracle(make):
    spec = make()
    torch.manual_seed(5)
    target = torch.randn((spec.num_pulses, 2, spec.num_samples), dtype=torch.complex128)

    o_tau, o_rate, o_weight = (x.clone().requires_grad_(True) for x in _inputs("cpu", torch.float64, torch.complex128))
    _loss(oracle_cube(o_tau, o_rate, o_weight, list(OFFSETS), spec), target).backward()
    # The oracle against float64 central differences first. The steps keep every
    # sample on one side of every chip boundary (clearance >= 1e-10 s here).
    for index in range(2):
        measured = fd.central_difference(
            lambda v: _loss(oracle_cube(v, o_rate.detach(), o_weight.detach(), list(OFFSETS), spec), target),
            o_tau.detach(),
            index,
            1e-13,
        )
        assert fd.relative_error(measured, float(o_tau.grad[index]), floor=1e-6) < 1e-4

    tau, rate, weight = (x.clone().requires_grad_(True) for x in _inputs("cuda", torch.float32, torch.complex64))
    offsets = torch.tensor(OFFSETS, dtype=torch.int64, device="cuda")
    _loss(synthesize_echo_rows(tau, rate, weight, offsets, spec).cpu(), target).backward()
    assert torch.allclose(tau.grad.cpu().double(), o_tau.grad, rtol=2e-3, atol=1e-3 * float(o_tau.grad.abs().max()))
    assert torch.allclose(rate.grad.cpu().double(), o_rate.grad, rtol=2e-3, atol=1e-3 * float(o_rate.grad.abs().max()))
    assert torch.allclose(weight.grad.cpu().to(torch.complex128), o_weight.grad, rtol=2e-3, atol=1e-3)

    # JVP against the oracle's directional derivative (reverse-mode identity).
    d_tau = torch.tensor((1.0e-9, -2.0e-9))
    d_rate = torch.tensor((1.0e-8, 0.5e-8))
    d_weight = torch.tensor((0.1 + 0.2j, -0.3 + 0.05j), dtype=torch.complex64)
    with forward_ad.dual_level():
        cube = synthesize_echo_rows(
            forward_ad.make_dual(tau.detach(), d_tau.cuda()),
            forward_ad.make_dual(rate.detach(), d_rate.cuda()),
            forward_ad.make_dual(weight.detach(), d_weight.cuda()),
            offsets,
            spec,
        )
        tangent = forward_ad.unpack_dual(cube).tangent.cpu().to(torch.complex128)
    reference = torch.autograd.functional.jvp(
        lambda a, b, c: oracle_cube(a, b, c, list(OFFSETS), spec),
        (o_tau.detach(), o_rate.detach(), o_weight.detach()),
        (d_tau.double(), d_rate.double(), d_weight.to(torch.complex128)),
    )[1]
    scale = float(reference.abs().max())
    assert scale > 0
    assert float((tangent - reference).abs().max()) < 2e-3 * scale


# --------------------------------------------------------------------------
# The per-segment schedule: one launch equals one launch per segment
# --------------------------------------------------------------------------


SCHEDULE = ((1.2e-6, 0.0), (1.2e-6 + 3 * 0.02e-6, 4.0e6))


def _per_segment(spec, tau, rate, weight):
    cubes = []
    for segment, (start, offset) in enumerate(SCHEDULE):
        single = replace(spec, range_gate_start_s=start, frequency_step_hz=offset)
        row = slice(OFFSETS[segment], OFFSETS[segment + 1])
        local = torch.tensor((0, 1), dtype=torch.int64, device="cuda")
        cubes.append(synthesize_echo_rows(tau[row], rate[row], weight[row], local, single))
    return torch.cat(cubes, dim=1)


def _scheduled(spec, tau, rate, weight):
    offsets = torch.tensor(OFFSETS, dtype=torch.int64, device="cuda")
    table = torch.tensor(SCHEDULE, dtype=torch.float64, device="cuda")
    return synthesize_echo_rows(tau, rate, weight, offsets, spec, segment_schedule=table)


def test_a_segment_schedule_is_one_launch_per_segment_bit_for_bit():
    # One pulse: the stepped offset is pulse 1's own carrier on the per-segment
    # launch and the scheduled one alike.
    spec = replace(fsk_spec(), num_pulses=1, num_samples=8)
    tau, rate, weight = (x.requires_grad_(True) for x in _inputs("cuda", torch.float32, torch.complex64))
    separate = _per_segment(spec, tau, rate, weight)
    together = _scheduled(spec, tau, rate, weight)
    assert torch.equal(separate, together)

    cotangent = torch.randn_like(separate)
    grads_a = torch.autograd.grad((separate * cotangent.conj()).real.sum(), (tau, rate, weight))
    grads_b = torch.autograd.grad((together * cotangent.conj()).real.sum(), (tau, rate, weight))
    for a, b in zip(grads_a, grads_b, strict=True):
        assert torch.equal(a, b)

    with forward_ad.dual_level():
        duals = [forward_ad.make_dual(x.detach(), torch.ones_like(x)) for x in (tau, rate, weight)]
        tangent_a = forward_ad.unpack_dual(_per_segment(spec, *duals)).tangent
        tangent_b = forward_ad.unpack_dual(_scheduled(spec, *duals)).tangent
    assert torch.equal(tangent_a, tangent_b)


def test_a_schedule_of_the_wrong_shape_or_with_a_derivative_is_refused():
    spec = replace(fsk_spec(), num_pulses=1)
    tau, rate, weight = _inputs("cuda", torch.float32, torch.complex64)
    offsets = torch.tensor(OFFSETS, dtype=torch.int64, device="cuda")
    with pytest.raises(ValueError, match="segment_schedule"):
        synthesize_echo_rows(
            tau, rate, weight, offsets, spec, segment_schedule=torch.zeros(3, 2, dtype=torch.float64, device="cuda")
        )
    with pytest.raises(RuntimeError, match="schedule"):
        synthesize_echo_rows(
            tau,
            rate,
            weight,
            offsets,
            spec,
            segment_schedule=torch.zeros(2, 2, dtype=torch.float64, device="cuda", requires_grad=True),
        )


# --------------------------------------------------------------------------
# Spec refusals (CPU)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"pulse_kind": "coded"}, "at least one chip"),
        ({"pulse_kind": "coded", "chips": (1.5 + 0j,)}, "exceeds one"),
        ({"pulse_kind": "coded", "chips": (1.0 + 0j,), "chip_frequencies_hz": (1.0, 2.0)}, "chip_frequencies_hz"),
        ({"chips": (1.0 + 0j,)}, "only a coded pulse"),
        ({"periodic": True}, "pri_s must equal pulse_width_s"),
    ],
)
def test_a_coded_spec_refuses_what_it_cannot_transmit(fields, message):
    base = {
        "num_pulses": 2,
        "num_samples": 64,
        "sample_period_s": 1 / 50e6,
        "pri_s": 10e-6,
        "range_gate_start_s": 0.0,
        "pulse_kind": "rect",
        "pulse_width_s": 1e-6,
        "bandwidth_hz": 1e6,
        "reference_frequency_hz": F_REF_HZ,
        "max_expected_delay_rate": 0.0,
        "carrier_rate_hz": F_REF_HZ,
    }
    base.update(fields)
    with pytest.raises(ValueError, match=message):
        PulsedSpec(**base)


def test_the_m_sequence_and_costas_generators_have_their_defining_properties():
    for order in (3, 7, 10):
        code = np.asarray(Pulsed.m_sequence(order), dtype=float)
        n = (1 << order) - 1
        correlation = np.real(np.fft.ifft(np.fft.fft(code) * np.conj(np.fft.fft(code))))
        assert len(code) == n and correlation[0] == pytest.approx(n)
        assert np.allclose(correlation[1:], -1.0)
    order = Pulsed.costas(11)
    assert sorted(order) == list(range(1, 11))
    # Costas: every displacement vector between two dots is distinct.
    vectors = {(j - i, order[j] - order[i]) for i in range(len(order)) for j in range(len(order)) if i != j}
    assert len(vectors) == len(order) * (len(order) - 1)
