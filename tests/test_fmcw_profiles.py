"""F-4 FMCW extensions (R-ADR-047): chirp sequences, start offsets, ramp phase error, an in-phase-only ADC.

Oracles:

* a float64 Torch evaluation of the beat phase with every extension, written
  from the equation (``f0 tau + E(u) - E(u - tau)`` on top of the uniform
  ramp, each slot at its own ramp start, slope and ADC start), against the
  kernel's forward, VJP and JVP;
* the uniform ramp is bit-identical with and without the extension machinery;
* end to end: each profile of a sequence puts a still target at its own beat
  frequency ``S tau``; an in-phase receiver publishes ``Re`` of the IQ cube, so
  its spectrum is Hermitian with each image at half the amplitude.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch
import torch.autograd.forward_ad as forward_ad
from support import oracle_scenes as sc

from witwin.radar import Chirp, Fmcw, PointTargets
from witwin.radar.policy import SPEED_OF_LIGHT_M_PER_S as C0
from witwin.radar.synthesis.assembly import FmcwSpec
from witwin.radar.synthesis.fmcw import synthesize_fmcw_rows

pytestmark = pytest.mark.gpu

F_REF_HZ = 77.0e9

UNIFORM = FmcwSpec(
    num_samples=16,
    num_chirps=3,
    sample_period_s=1.0 / 10e6,
    chirp_period_s=20e-6,
    slope_hz_per_s=30e12,
    t_start_s=2e-6,
    reference_frequency_hz=F_REF_HZ,
    carrier_rate_hz=F_REF_HZ,
    num_tx=2,
    num_rx=1,
    output_domain="beat",
)

#: Two profiles: the second steeper, 150 MHz higher, with a longer idle and a
#: later ADC start. The cubic ramp error is about 0.4 rad at the ramp end.
SEQUENCED = replace(
    UNIFORM,
    sequence=((30e12, 0.0, 4e-6, 2e-6, 16e-6), (45e12, 150e6, 6e-6, 3e-6, 18e-6)),
    phase_error_cycles=(0.0, 1.5e13),
)

DELAYS = (4.0e-8, 6.5e-8, 5.1e-8)
RATES = (3.0e-7, -2.0e-7, 1.0e-7)
WEIGHTS = (0.6 - 0.3j, -0.2 + 0.45j, 0.3 + 0.1j)
OFFSETS = (0, 2, 3)


def oracle(tau0, rate, weight, spec: FmcwSpec) -> torch.Tensor:
    """``[chirp, pair, sample]`` beat cube in float64, from the equation."""

    profiles = torch.from_numpy(spec.slot_profiles())
    poly = list(spec.phase_error_cycles)

    def error(t):
        return sum(c * t ** (i + 2) for i, c in enumerate(poly)) if poly else torch.zeros_like(t)

    samples = torch.arange(spec.num_samples, dtype=torch.float64)
    out = torch.zeros((spec.num_chirps, len(OFFSETS) - 1, spec.num_samples), dtype=torch.complex128)
    for segment in range(len(OFFSETS) - 1):
        tx = segment % spec.num_tx
        for chirp in range(spec.num_chirps):
            ramp_start, slope, offset, adc = profiles[chirp * spec.num_tx + tx]
            u = adc + samples * spec.sample_period_s
            for k in range(OFFSETS[segment], OFFSETS[segment + 1]):
                drift = rate[k] * (ramp_start + u)
                tau = tau0[k] + drift
                cycles = (
                    spec.carrier_hz * tau
                    + spec.carrier_rate_hz * drift
                    + slope * tau * (u - 0.5 * tau)
                    + offset * tau
                    + error(u)
                    - error(u - tau)
                )
                out[chirp, segment] = out[chirp, segment] + weight[k] * torch.exp(2j * math.pi * cycles)
    return out


def _inputs(device, dtype, cdtype):
    return (
        torch.tensor(DELAYS, dtype=dtype, device=device),
        torch.tensor(RATES, dtype=dtype, device=device),
        torch.tensor(WEIGHTS, dtype=cdtype, device=device),
    )


def _kernel(spec, tau, rate, weight):
    offsets = torch.tensor(OFFSETS, dtype=torch.int64, device="cuda")
    tx = torch.tensor((0, 1), dtype=torch.int32, device="cuda")
    return synthesize_fmcw_rows(tau, rate, weight, offsets, spec, segment_tx_index=tx)


def test_the_slot_profiles_accumulate_each_profiles_period():
    profiles = SEQUENCED.slot_profiles()
    assert profiles.shape == (6, 4)
    # Slot s uses profile s % 2; ramp starts are the running sum of idle + ramp end.
    assert profiles[:, 0].tolist() == pytest.approx([0.0, 20e-6, 44e-6, 64e-6, 88e-6, 108e-6])
    assert profiles[1, 1] == 45e12 and profiles[1, 2] == 150e6 and profiles[1, 3] == 3e-6
    assert SEQUENCED.slot_period_s == pytest.approx(132e-6 / 3)


def test_the_extended_beat_is_the_float64_closed_form():
    tau, rate, weight = _inputs("cuda", torch.float32, torch.complex64)
    cube = _kernel(SEQUENCED, tau, rate, weight).cpu().to(torch.complex128)
    o = _inputs("cpu", torch.float64, torch.complex128)
    expected = oracle(o[0].float().double(), o[1].float().double(), o[2], SEQUENCED)
    assert float((cube - expected).abs().max()) < 2e-4 * float(expected.abs().max())


def test_the_uniform_ramp_is_bit_identical_through_an_empty_extension():
    tau, _, weight = _inputs("cuda", torch.float32, torch.complex64)
    plain = _kernel(UNIFORM, tau, None, weight)
    # A zero-offset single-profile sequence is a slot table holding the scalars
    # (still rows: the table's summed ramp starts and the kernel's product
    # s * T may differ in the last bit, and a still row never reads them).
    tabled = _kernel(replace(UNIFORM, sequence=((30e12, 0.0, 0.0, 2e-6, 20e-6),)), tau, None, weight)
    assert torch.equal(plain, tabled)


def test_the_native_vjp_and_jvp_follow_the_extended_oracle():
    torch.manual_seed(3)
    target = torch.randn((SEQUENCED.num_chirps, 2, SEQUENCED.num_samples), dtype=torch.complex128)
    o_tau, o_rate, o_weight = (x.clone().requires_grad_(True) for x in _inputs("cpu", torch.float64, torch.complex128))
    loss = (oracle(o_tau, o_rate, o_weight, SEQUENCED) - target).abs().pow(2).sum()
    loss.backward()
    tau, rate, weight = (x.clone().requires_grad_(True) for x in _inputs("cuda", torch.float32, torch.complex64))
    (_kernel(SEQUENCED, tau, rate, weight).cpu().to(torch.complex128) - target).abs().pow(2).sum().backward()
    for got, want in ((tau.grad, o_tau.grad), (rate.grad, o_rate.grad), (weight.grad, o_weight.grad)):
        got = got.cpu().to(want.dtype)
        assert float((got - want).abs().max()) < 3e-3 * float(want.abs().max())

    directions = (torch.tensor((1e-9, -2e-9, 0.5e-9)), torch.tensor((1e-7, 0.0, -1e-7)))
    d_weight = torch.tensor((0.1 + 0.2j, -0.3 + 0.05j, 0.2 - 0.1j))
    with forward_ad.dual_level():
        cube = _kernel(
            SEQUENCED,
            forward_ad.make_dual(tau.detach(), directions[0].cuda()),
            forward_ad.make_dual(rate.detach(), directions[1].cuda()),
            forward_ad.make_dual(weight.detach(), d_weight.to(torch.complex64).cuda()),
        )
        tangent = forward_ad.unpack_dual(cube).tangent.cpu().to(torch.complex128)
    reference = torch.autograd.functional.jvp(
        lambda a, b, c: oracle(a, b, c, SEQUENCED),
        (o_tau.detach(), o_rate.detach(), o_weight.detach()),
        (directions[0].double(), directions[1].double(), d_weight),
    )[1]
    assert float((tangent - reference).abs().max()) < 3e-3 * float(reference.abs().max())


def test_the_closed_form_spectrum_refuses_what_it_cannot_describe():
    tau, rate, weight = _inputs("cuda", torch.float32, torch.complex64)
    with pytest.raises(ValueError, match="closed-form range spectrum"):
        _kernel(replace(SEQUENCED, output_domain="spectrum"), tau, None, weight)


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------

RANGE_M = 6.0


def _fmcw(**fields):
    base = {
        "slope": 30e12,
        "sample_rate": 10e6,
        "samples_per_chirp": 64,
        "chirps_per_frame": 4,
        "adc_start": 1e-6,
        "idle": 5e-6,
        "ramp_end": 8e-6,
        "output": "beat",
    }
    base.update(fields)
    return Fmcw(**base)


def _simulate(waveform, range_m=RANGE_M, **radar_fields):
    radar = sc.radar(waveform, **radar_fields)
    return radar, radar.simulate(
        sc.empty_scene(), PointTargets(positions=[[range_m, 0.0, 0.0]], rcs=1.0), times=(0.0,), los=True, reflections=0
    )


def test_each_profile_of_a_sequence_beats_at_its_own_slope():
    waveform = _fmcw(sequence=(Chirp(), Chirp(slope=60e12, start_offset=200e6)))
    radar, result = _simulate(waveform)
    tau = 2.0 * RANGE_M / C0
    spectrum = torch.fft.fft(result.cube[0, 0, 0].to(torch.complex128), dim=-1).abs()
    for chirp in range(4):
        slope = 30e12 if chirp % 2 == 0 else 60e12
        expected_bin = slope * tau * 64 / 10e6
        assert int(spectrum[chirp].argmax()) == round(expected_bin), (chirp, expected_bin)


def test_a_real_adc_publishes_the_in_phase_channel():
    # On bin 8 exactly, so the image at bin -8 does not leak into it.
    on_bin = 8 * 10e6 / (30e12 * 64) * C0 / 2.0
    radar, iq = _simulate(_fmcw(), on_bin)
    _, real = _simulate(_fmcw(sampling="real"), on_bin)
    assert torch.equal(real.cube.imag, torch.zeros_like(real.cube.imag))
    assert torch.allclose(real.cube.real, iq.cube.real, rtol=0, atol=1e-12)
    _, real_spectrum = _simulate(_fmcw(sampling="real", output="spectrum"), on_bin)
    _, iq_spectrum = _simulate(_fmcw(output="spectrum"), on_bin)
    row = real_spectrum.cube[0, 0, 0, 0].to(torch.complex128)
    peak = int(iq_spectrum.cube[0, 0, 0, 0].abs().argmax())
    # Hermitian: the image sits at -k with half the IQ amplitude in each.
    assert peak == 8
    assert float(row[peak].abs()) == pytest.approx(0.5 * float(iq_spectrum.cube[0, 0, 0, 0, peak].abs()), rel=1e-3)
    assert float(row[(-peak) % 64].abs()) == pytest.approx(float(row[peak].abs()), rel=1e-3)
