"""Receiver impairments: compression, baseband DC and flicker, dynamic AGC, coupling.

Every stage is off by default and each test turns on exactly one. The oracles
are physical statements written here, in float64 where they are numbers:

* compression: the 1-dB point and the two-tone third-order intercept land at
  the declared input-referred figures;
* the time-dynamic AGC: the per-block gain is the first-order loop's closed
  form, and its derivatives match a float64 Torch evaluation of the same loop;
* baseband: the DC offset is exact and the flicker density is ``1/f`` with the
  thermal density at the declared corner;
* coupling: the cube is ``C_tx^T H C_rx^T`` and it is applied before noise.
"""

from __future__ import annotations

import math

import pytest
import torch
from support.dsp_ledger import DspLedger

from witwin.radar.frontend import Adc, Agc, Compression, FrontendChain, FrontendSpec, Noise

pytestmark = pytest.mark.gpu

IMPEDANCE = 50.0


def _chain(**fields) -> FrontendChain:
    return FrontendChain(FrontendSpec(impedance=IMPEDANCE, **fields))


def _watts(dbm: float) -> float:
    return 1e-3 * 10.0 ** (dbm / 10.0)


def _sqrt_watt_tone(power_dbm: float, count: int, bin_index: int) -> torch.Tensor:
    """A complex tone of ``power_dbm`` at the port, in sqrt(W)."""

    phase = 2 * math.pi * bin_index * torch.arange(count, dtype=torch.float64) / count
    amplitude = math.sqrt(_watts(power_dbm))
    return (amplitude * torch.exp(1j * phase)).to(torch.complex64).cuda()


# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lna", [None, 20.0])
@pytest.mark.parametrize("p1db", [None, -22.0])
def test_the_gain_is_one_db_down_at_the_declared_input_compression_point(lna, p1db):
    """M4-8: the 1-dB point is input-referred and independent of the LNA gain."""

    stage = Compression(iip3=-10.0, p1db=p1db)
    expected = -10.0 - 9.6357 if p1db is None else p1db
    chain = _chain(compression=stage, lna=lna)
    linear = _chain(lna=lna)
    tone = _sqrt_watt_tone(expected, 256, 5)
    out = chain.apply(tone).signal
    reference = linear.apply(tone).signal
    gain_db = 20 * math.log10(float(out.abs().mean()) / float(reference.abs().mean()))
    assert gain_db == pytest.approx(-1.0, abs=0.01)


def test_the_two_tone_third_order_intercept_is_the_declared_iip3():
    """IM3 at ``2 f1 - f2`` extrapolates to the declared IIP3 within 0.1 dB.

    Each tone carries ``P`` at the input; the product's power is
    ``P^3 / IIP3^2``, so ``IIP3 = P + (P_fund - P_im3) / 2`` in dB.
    """

    iip3 = -5.0
    count = 4096
    tone_dbm = -40.0
    chain = _chain(compression=Compression(iip3=iip3))
    signal = _sqrt_watt_tone(tone_dbm, count, 100) + _sqrt_watt_tone(tone_dbm, count, 110)
    out = chain.apply(signal).signal.to(torch.complex128) / math.sqrt(IMPEDANCE)
    spectrum = torch.fft.fft(out) / count
    fundamental = float(spectrum[100].abs() ** 2)
    product = float(spectrum[90].abs() ** 2)
    extrapolated = tone_dbm + 10 * math.log10(fundamental / product) / 2
    assert extrapolated == pytest.approx(iip3, abs=0.1)


def test_am_pm_rotates_the_phase_by_the_declared_angle_at_the_one_db_point():
    stage = Compression(iip3=-10.0, am_pm=5.0)
    tone = _sqrt_watt_tone(-10.0 - 9.6357, 64, 3)
    out = _chain(compression=stage).apply(tone).signal
    rotation = torch.angle(out * tone.conj() / tone.abs() ** 2)
    assert float(rotation.mean()) == pytest.approx(math.radians(5.0), abs=1e-4)


def test_the_output_amplitude_saturates_rather_than_folding_back():
    chain = _chain(compression=Compression(iip3=-10.0))
    levels = torch.linspace(-40.0, 10.0, 101)
    out = torch.stack([chain.apply(_sqrt_watt_tone(float(level), 8, 1)).signal.abs().mean() for level in levels])
    assert bool((out[1:] >= out[:-1] * (1 - 1e-6)).all())


def _compression_oracle(x: torch.Tensor, stage: Compression) -> torch.Tensor:
    """The stage in float64 Torch: the independent expression its kernel is pinned to."""

    c5, k, u_sat = stage.coefficients
    scale = stage.input_power_scale(impedance=IMPEDANCE, lna_voltage_gain=1.0)
    u = scale * (x.real**2 + x.imag**2)
    held = torch.clamp(u, max=u_sat)
    gain = (1 - held + c5 * held**2) * torch.exp(1j * k * held)
    gain = torch.where(u > u_sat, gain * torch.sqrt(u_sat / u), gain)
    return x * gain


@pytest.mark.parametrize("mode", ["vjp", "jvp"])
def test_the_compression_derivatives_match_a_float64_oracle(mode):
    stage = Compression(iip3=-20.0, p1db=-31.0, am_pm=8.0)
    generator = torch.Generator().manual_seed(3)
    volts = math.sqrt(_watts(-20.0) * IMPEDANCE)
    x = (torch.randn(512, generator=generator, dtype=torch.complex128) * volts).cuda()
    direction = torch.randn(512, generator=generator, dtype=torch.complex128).cuda()
    chain = _chain(compression=stage)

    def production(value):
        return chain.apply((value / math.sqrt(IMPEDANCE)).to(torch.complex64)).signal

    def oracle(value):
        return _compression_oracle(value, stage)

    if mode == "vjp":
        leaf = x.to(torch.complex64).clone().requires_grad_(True)
        (production(leaf) * direction.conj().to(torch.complex64)).real.sum().backward()
        reference_leaf = x.clone().requires_grad_(True)
        (oracle(reference_leaf) * direction.conj()).real.sum().backward()
        torch.testing.assert_close(leaf.grad.to(torch.complex128), reference_leaf.grad, rtol=2e-4, atol=1e-6)
    else:
        import torch.autograd.forward_ad as fwad

        with fwad.dual_level():
            dual = fwad.make_dual(x.to(torch.complex64), direction.to(torch.complex64))
            tangent = fwad.unpack_dual(production(dual)).tangent
        _, reference = torch.func.jvp(oracle, (x,), (direction,))
        torch.testing.assert_close(tangent.to(torch.complex128), reference, rtol=2e-4, atol=1e-6)


def test_a_p1db_without_a_saturation_point_is_refused_by_name():
    with pytest.raises(ValueError, match="no saturation point|beyond this stage's saturation"):
        Compression(iip3=0.0, p1db=-1.0)
    with pytest.raises(ValueError, match="below iip3"):
        Compression(iip3=0.0, p1db=1.0)


# ---------------------------------------------------------------------------
# Time-dynamic AGC
# ---------------------------------------------------------------------------


def _agc_oracle(x: torch.Tensor, agc: Agc) -> tuple[torch.Tensor, torch.Tensor]:
    """The first-order loop in float64, per (rx, block), on a (TX, RX, S, N) cube."""

    beta = agc.loop_coefficient
    log_min = math.log(agc.min_voltage_gain)
    log_max = math.log(agc.max_voltage_gain)
    power = (x.real**2 + x.imag**2).mean(dim=(0, 3))  # (RX, S)
    ideal = torch.log(agc.target_rms / torch.sqrt(power.clamp_min(1e-24)))
    levels = []
    previous = None
    for k in range(x.shape[2]):
        raw = ideal[:, 0] if k == 0 else (1 - beta) * previous + beta * ideal[:, k - 1]
        level = raw.clamp(log_min, log_max)
        levels.append(level)
        previous = level
    gain = torch.exp(torch.stack(levels, dim=1))  # (RX, S)
    return x * gain[None, :, :, None], gain


def _stepped_cube(seed: int = 1) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    # Unit-modulus samples with random phases: every block's RMS is exactly its level.
    x = torch.exp(2j * math.pi * torch.rand(2, 3, 16, 32, generator=generator, dtype=torch.float64))
    step = torch.ones(16, dtype=torch.float64)
    step[6:] = 8.0
    return x * step[None, None, :, None] * 1e-3


def test_the_dynamic_agc_follows_the_first_order_loop_closed_form():
    """A step in input level is tracked with the declared time constant.

    With a constant input after the step, ``l_k - l_inf`` decays as
    ``(1 - beta)^k``, ``beta = 1 - exp(-T_block / tau)``.
    """

    agc = Agc(target_rms=0.1, time_constant=4e-3, block_period=1e-3)
    x = _stepped_cube()
    output = _chain(agc=agc).apply((x / math.sqrt(IMPEDANCE)).to(torch.complex64).cuda())
    reference, gain = _agc_oracle(x, agc)
    measured = output.diagnostics.agc_gain.reshape(3, 16).double().cpu()
    torch.testing.assert_close(measured, gain.cpu(), rtol=2e-5, atol=0.0)
    torch.testing.assert_close(output.signal.cpu().to(torch.complex128), reference, rtol=2e-5, atol=1e-9)
    # The decay after the step is the declared time constant's: with a
    # constant input, successive log-gain increments shrink by 1 - beta.
    steps = torch.diff(torch.log(measured[0, 7:15]))
    ratio = steps[1:] / steps[:-1]
    torch.testing.assert_close(ratio, torch.full_like(ratio, math.exp(-1e-3 / 4e-3)), rtol=1e-3, atol=0.0)


@pytest.mark.parametrize("mode", ["vjp", "jvp"])
def test_the_dynamic_agc_derivatives_match_a_float64_oracle(mode):
    agc = Agc(target_rms=0.1, time_constant=2e-3, block_period=1e-3, min_gain=-20.0, max_gain=60.0)
    x = _stepped_cube(2)
    direction = _stepped_cube(5)
    chain = _chain(agc=agc)

    def production(value):
        return chain.apply((value / math.sqrt(IMPEDANCE)).to(torch.complex64)).signal

    def oracle(value):
        return _agc_oracle(value, agc)[0]

    if mode == "vjp":
        leaf = x.to(torch.complex64).cuda().requires_grad_(True)
        (production(leaf) * direction.conj().to(torch.complex64).cuda()).real.sum().backward()
        reference_leaf = x.clone().requires_grad_(True)
        (oracle(reference_leaf) * direction.conj()).real.sum().backward()
        torch.testing.assert_close(leaf.grad.cpu().to(torch.complex128), reference_leaf.grad, rtol=2e-3, atol=1e-6)
    else:
        import torch.autograd.forward_ad as fwad

        with fwad.dual_level():
            dual = fwad.make_dual(x.to(torch.complex64).cuda(), direction.to(torch.complex64).cuda())
            tangent = fwad.unpack_dual(production(dual)).tangent
        _, reference = torch.func.jvp(oracle, (x,), (direction,))
        torch.testing.assert_close(tangent.cpu().to(torch.complex128), reference, rtol=2e-3, atol=1e-7)


def test_a_frame_static_agc_is_unchanged_by_the_dynamic_fields():
    x = _stepped_cube().to(torch.complex64).cuda()
    static = _chain(agc=Agc(target_rms=0.1)).apply(x)
    assert static.diagnostics.agc_gain.numel() == 3


# ---------------------------------------------------------------------------
# Baseband DC offset and flicker
# ---------------------------------------------------------------------------


def test_the_dc_offset_lands_exactly_on_each_receiver():
    offsets = (1e-3 + 2e-3j, -4e-3 + 0j)
    chain = _chain(dc_offset=offsets)
    zeros = torch.zeros(2, 2, 4, 8, dtype=torch.complex64, device="cuda")
    out = chain.apply(zeros).signal
    for rx, value in enumerate(offsets):
        assert torch.equal(out[:, rx], torch.full_like(out[:, rx], complex(value)))


def test_the_baseband_flicker_density_is_one_over_f_with_the_thermal_density_at_the_corner():
    """Welch PSD: ``kT_sys R f_c / f`` below the corner, within 1.5 dB per octave band."""

    rate = 1e6
    corner = 2e4
    noise = Noise(figure=0.0, flicker_corner=corner).resolved(bandwidth=rate, sample_rate=rate)
    plain = Noise(figure=0.0).resolved(bandwidth=rate, sample_rate=rate)
    count = 1 << 18
    signal = torch.zeros(1, 1, 1, count, dtype=torch.complex64, device="cuda")
    # Isolate the flicker: the same thermal draws with and without it differ by it alone.
    with_flicker = _chain(noise=noise).apply(signal).signal
    without = _chain(noise=plain).apply(signal).signal
    flicker = with_flicker - without
    values = flicker.reshape(-1).to(torch.complex128).cpu()
    segment = 1 << 14
    window = torch.hann_window(segment, periodic=True, dtype=torch.float64)
    blocks = values[: (count // segment) * segment].reshape(-1, segment)
    blocks = blocks - blocks.mean(dim=1, keepdim=True)
    psd = (torch.fft.fft(blocks * window).abs() ** 2).mean(dim=0) / (rate * (window**2).sum())
    freqs = torch.fft.fftfreq(segment, d=1.0 / rate)
    density = 1.380649e-23 * 290.0 * IMPEDANCE
    for low in (500.0, 2000.0, 8000.0, 32000.0):
        band = (freqs >= low) & (freqs < 2 * low)
        measured = float(psd[band].mean())
        expected = density * corner * math.log(2.0) / low  # mean of 1/f over the octave is ln2 / f_low
        assert 10 * math.log10(measured / expected) == pytest.approx(0.0, abs=1.5), low


def test_the_baseband_stage_passes_derivatives_through_unchanged():
    chain = _chain(dc_offset=(1e-3 + 0j,))
    leaf = torch.randn(1, 1, 4, 8, dtype=torch.complex64, device="cuda").requires_grad_(True)
    chain.apply(leaf).signal.real.sum().backward()
    expected = torch.full_like(leaf, math.sqrt(IMPEDANCE))
    torch.testing.assert_close(leaf.grad, expected)


# ---------------------------------------------------------------------------
# Mutual coupling
# ---------------------------------------------------------------------------


def test_coupling_mixes_the_ports_as_the_matrices_declare_and_before_the_noise():
    generator = torch.Generator().manual_seed(9)
    tx = torch.eye(2, dtype=torch.complex128) + 0.1j * torch.randn(2, 2, generator=generator, dtype=torch.float64)
    rx = torch.eye(3, dtype=torch.complex128) + 0.2 * torch.randn(3, 3, generator=generator, dtype=torch.complex128)
    cube = torch.randn(2, 3, 4, 8, generator=generator, dtype=torch.complex128)
    chain = FrontendChain(
        FrontendSpec(
            impedance=1.0, tx_coupling=tuple(map(tuple, tx.tolist())), rx_coupling=tuple(map(tuple, rx.tolist()))
        )
    )
    out = chain.apply(cube.to(torch.complex64).cuda()).signal.cpu().to(torch.complex128)
    expected = torch.einsum("kt,kjsn,rj->trsn", tx, cube, rx)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)

    # Noise enters after the coupling: a silent coupled array has the plain
    # receiver's noise power, not the coupling matrices' row norms times it.
    noise = Noise(figure=3.0).resolved(bandwidth=1e6, sample_rate=1e6)
    zeros = torch.zeros(2, 3, 64, 256, dtype=torch.complex64, device="cuda")
    coupled = FrontendChain(
        FrontendSpec(impedance=1.0, noise=noise, tx_coupling=tuple(map(tuple, (2 * tx).tolist())))
    ).apply(zeros)
    plain = FrontendChain(FrontendSpec(impedance=1.0, noise=noise)).apply(zeros)
    assert torch.equal(coupled.signal, plain.signal)


# ---------------------------------------------------------------------------
# Frames and launch budgets
# ---------------------------------------------------------------------------


def test_each_frame_draws_its_own_thermal_realisation_and_frame_zero_keeps_the_seed():
    noise = Noise(figure=3.0).resolved(bandwidth=1e6, sample_rate=1e6)
    chain = _chain(noise=noise, adc=None)
    zeros = torch.zeros(1, 1, 4, 64, dtype=torch.complex64, device="cuda")
    first = chain.apply(zeros, frame_time_s=0.0).signal
    second = chain.apply(zeros, frame_time_s=0.05).signal
    assert not torch.equal(first, second)
    assert torch.equal(first, chain.apply(zeros, seed_base=chain.spec.seed).signal)
    assert torch.equal(second, chain.apply(zeros, frame_time_s=0.05).signal)


IMPAIRMENT_OPERATORS = (
    "frontend_compress_forward",
    "frontend_baseband_forward",
    "frontend_agc_dynamic_forward",
    "frontend_noise_forward",
    "frontend_quantize_forward",
)


def test_every_impairment_is_one_native_operator_call_per_frame(monkeypatch):
    noise = Noise(figure=6.0, flicker_corner=1e4).resolved(bandwidth=5e6, sample_rate=5e6)
    spec = FrontendSpec(
        impedance=IMPEDANCE,
        noise=noise,
        lna=20.0,
        compression=Compression(iip3=-15.0),
        dc_offset=(1e-4 + 0j, 0j),
        agc=Agc(target_rms=0.2, time_constant=1e-3, block_period=1e-4),
        adc=Adc(bits=10, full_scale=1.0),
        timeline=(1e-4, 0.0, 2e-7),
    )
    chain = FrontendChain(spec)
    signal = torch.randn(2, 2, 4, 32, dtype=torch.complex64, device="cuda") * 1e-5
    chain.apply(signal)
    from witwin.radar.cuda import runtime

    ledger = DspLedger(monkeypatch, IMPAIRMENT_OPERATORS, runtime.build_extension())
    chain.apply(signal)
    assert ledger.launches == dict.fromkeys(IMPAIRMENT_OPERATORS, 1), ledger.launches
