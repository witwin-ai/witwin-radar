"""The receive chain: physical levels, a fixed order, and exactly one ADC.

Every assertion here is against a number derived independently in
``tests/support/reference_frontend.py`` rather than against the production
expression. The chain from a noise figure to a per-component standard deviation
has four places to lose a factor of two, and an oracle that imported the
production formula would agree with all four mistakes.

The order test is the important one. Two independently callable runtimes made
the composite order whatever a caller happened to do, and thermal noise landing
after the LNA instead of before it is a factor of ``g_lna^2`` in output noise
power - twenty decibels here, invisible in every plot anyone would draw.
"""

from __future__ import annotations

import ast
import math
import pathlib

import pytest
import torch
from support.reference_frontend import (
    agc_gain,
    quantize,
    single_sideband_psd,
    thermal_sigma_volts,
    wiener_innovation_sigma_rad,
)

pytestmark = pytest.mark.gpu

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
IMPEDANCE = 50.0


def _specs():
    from witwin.radar.frontend import (
        AdcSpec,
        AgcSpec,
        FrontendChain,
        FrontendSpec,
        LnaSpec,
        NoiseSpec,
        PortSpec,
        SeedSpec,
    )

    return (AdcSpec, AgcSpec, FrontendChain, FrontendSpec, LnaSpec, NoiseSpec, PortSpec, SeedSpec)


def _zeros(count: int) -> torch.Tensor:
    return torch.zeros(count, dtype=torch.complex64, device="cuda")


# ---------------------------------------------------------------------------
# T4.5 - thermal noise level
# ---------------------------------------------------------------------------


def test_the_thermal_noise_level_is_kTsysBR():
    """Both components at ``sigma^2``, and the total at ``k T_sys B R``.

    Four million samples give a standard error of about 0.07 percent on a
    variance estimate, so ``3e-3`` is a comfortable four-sigma band rather than a
    tolerance chosen to make the test pass. The identity ``T_sys == T0 F`` when
    ``T_ant == T0`` is asserted to ``1e-12`` because it is exact arithmetic, not
    an estimate: it is the definition of a noise figure and a violation would
    mean the system temperature was assembled wrongly.
    """

    _, _, FrontendChain, FrontendSpec, _, NoiseSpec, PortSpec, SeedSpec = _specs()

    port = PortSpec(reference_impedance_ohm=IMPEDANCE)
    noise = NoiseSpec(noise_figure_db=6.0, antenna_temperature_k=290.0, bandwidth_hz=5e6)
    assert math.isclose(noise.system_noise_temperature_k, 290.0 * noise.noise_factor, rel_tol=1e-12)

    sigma = thermal_sigma_volts(
        noise_figure_db=6.0, antenna_temperature_k=290.0, bandwidth_hz=5e6, reference_impedance_ohm=IMPEDANCE
    )
    assert math.isclose(noise.thermal_sigma_volts(port), sigma, rel_tol=1e-12)

    chain = FrontendChain(FrontendSpec(port=port, noise=noise, seed=SeedSpec(7)))
    out = chain.apply(_zeros(1 << 22)).signal
    real_variance = float(out.real.double().var(unbiased=False))
    imag_variance = float(out.imag.double().var(unbiased=False))
    total = float((out.real.double() ** 2 + out.imag.double() ** 2).mean())

    assert math.isclose(real_variance, sigma**2, rel_tol=3e-3)
    assert math.isclose(imag_variance, sigma**2, rel_tol=3e-3)
    assert math.isclose(total, noise.noise_power_watts * IMPEDANCE, rel_tol=3e-3)


# ---------------------------------------------------------------------------
# T4.8 - the chain ORDER is fixed
# ---------------------------------------------------------------------------


def test_thermal_noise_is_input_referred_so_the_lna_amplifies_it():
    """Output noise power is ``g^2 k T_sys B R``, not ``k T_sys B R``.

    This is the assertion that kills the split-runtime hazard. A chain that let
    the caller run the receiver stage first would add thermal noise AFTER the
    gain and land on the second number, which differs from the first by exactly
    ``g_lna^2``. Both are plausible-looking noise floors; only one is physics.

    The comparison is between two runs at the same seed, so the two realisations
    are the same draws and the ratio is the gain rather than an estimate of it.
    """

    _, _, FrontendChain, FrontendSpec, LnaSpec, NoiseSpec, PortSpec, SeedSpec = _specs()

    port = PortSpec(reference_impedance_ohm=IMPEDANCE)
    noise = NoiseSpec(noise_figure_db=6.0, bandwidth_hz=5e6)
    lna = LnaSpec(gain_db=20.0)
    signal = _zeros(1 << 20)

    plain = FrontendChain(FrontendSpec(port=port, noise=noise, seed=SeedSpec(7))).apply(signal).signal
    amplified = FrontendChain(FrontendSpec(port=port, noise=noise, lna=lna, seed=SeedSpec(7))).apply(signal).signal

    plain_power = float((plain.real.double() ** 2 + plain.imag.double() ** 2).mean())
    amplified_power = float((amplified.real.double() ** 2 + amplified.imag.double() ** 2).mean())
    assert math.isclose(amplified_power / plain_power, lna.voltage_gain**2, rel_tol=1e-6)
    assert math.isclose(amplified_power, lna.voltage_gain**2 * noise.noise_power_watts * IMPEDANCE, rel_tol=3e-3)


def test_the_stage_order_is_published_and_the_runtime_follows_it():
    from witwin.radar.frontend import FRONTEND_STAGE_ORDER

    _, AgcSpec, FrontendChain, FrontendSpec, LnaSpec, NoiseSpec, PortSpec, _ = _specs()
    assert FRONTEND_STAGE_ORDER == ("port", "phase", "thermal", "lna", "agc", "adc")

    from witwin.radar.frontend import AdcSpec

    chain = FrontendChain(
        FrontendSpec(
            port=PortSpec(IMPEDANCE),
            noise=NoiseSpec(
                noise_figure_db=3.0,
                bandwidth_hz=1e6,
                phase_noise_dbc_per_hz=-90.0,
                phase_offset_hz=1e5,
                phase_sample_rate_hz=1e6,
            ),
            lna=LnaSpec(gain_db=10.0),
            agc=AgcSpec(target_rms=1.0, mode="global"),
            adc=AdcSpec(bits=10, full_scale=1.0),
        )
    )
    assert chain.enabled_stages == FRONTEND_STAGE_ORDER

    # A stage with a zero standard deviation is REPORTED as off. That is not
    # cosmetic: the phase scan still runs and still consumes its own Philox
    # stream, so the thermal realisation is unchanged either way, and reporting
    # a silent stage as enabled would suggest the two were coupled.
    quiet = FrontendChain(
        FrontendSpec(port=PortSpec(IMPEDANCE), noise=NoiseSpec(noise_figure_db=3.0, bandwidth_hz=1e6))
    )
    assert quiet.enabled_stages == ("port", "thermal")


def test_the_port_conversion_happens_exactly_once():
    """``v = sqrt(W) sqrt(R)``, and nowhere else in the chain.

    With every noise stage off, the whole chain is that one factor, so the ratio
    of output to input is exactly ``sqrt(R)``. A second conversion hidden in a
    transmit gain - which is where it used to live - would show up here as ``R``.
    """

    _, _, FrontendChain, FrontendSpec, _, _, PortSpec, _ = _specs()

    signal = torch.complex(torch.randn(256, device="cuda"), torch.randn(256, device="cuda")).to(torch.complex64)
    out = FrontendChain(FrontendSpec(port=PortSpec(IMPEDANCE))).apply(signal).signal
    assert torch.allclose(out, signal * math.sqrt(IMPEDANCE), rtol=1e-6, atol=1e-7)


# ---------------------------------------------------------------------------
# T4.6 - quantisation
# ---------------------------------------------------------------------------


def test_the_quantization_error_variance_is_the_step_squared_over_twelve():
    _, _, FrontendChain, FrontendSpec, _, _, PortSpec, _ = _specs()
    from witwin.radar.frontend import AdcSpec

    adc = AdcSpec(bits=10, full_scale=1.0)
    generator = torch.Generator(device="cpu").manual_seed(19)
    busy = (
        torch.complex(
            torch.rand(1 << 20, generator=generator) * 1.2 - 0.6, torch.rand(1 << 20, generator=generator) * 1.2 - 0.6
        )
        .to(torch.complex64)
        .cuda()
    )

    output = FrontendChain(FrontendSpec(port=PortSpec(1.0), adc=adc)).apply(busy)
    error = output.signal - busy
    assert math.isclose(float(error.real.double().var(unbiased=False)), adc.quantization_variance, rel_tol=1e-2)
    assert math.isclose(float(error.imag.double().var(unbiased=False)), adc.quantization_variance, rel_tol=1e-2)
    # The busy signal is inside full scale, so nothing clips. The count is a
    # DEVICE tensor and it is published rather than suppressed, because a
    # nonzero value is the only visible symptom of an AGC misconfiguration.
    assert output.diagnostics.clipped_components is not None
    assert int(output.diagnostics.clipped_components.item()) == 0


def test_the_quantizer_matches_the_reference_grid_and_clips_symmetrically():
    """Same code everywhere off the ties, and the same clipping either side.

    The grid, not the tie-breaking, is what this asserts, and the distinction is
    deliberate. Exactly half way between two codes the decision comes down to one
    unit in the last place of ``(x + FS) / step``, and the kernel and Torch reach
    that quotient by different routes: nvcc contracts the reconstruction into a
    fused multiply-add and Torch evaluates it as two separately rounded
    operations. Asserting a tie would be asserting which of two equally defensible
    roundings a compiler happened to choose. Off the ties there is nothing to
    choose, and the codes agree exactly - which is the property a quantiser
    actually has to have.

    Both clipping directions are covered, and the count is asserted rather than
    merely non-``None``: a clipped-sample count is the only visible symptom of an
    AGC misconfiguration, so a count that silently stopped incrementing would be
    worse than no diagnostic at all.
    """

    _, _, FrontendChain, FrontendSpec, _, _, PortSpec, _ = _specs()
    from witwin.radar.frontend import AdcSpec

    adc = AdcSpec(bits=8, full_scale=1.0)
    # A quarter of a step off every boundary: unambiguous on both sides, and
    # spanning past full scale in both directions so the clipping is exercised.
    quarter = adc.step / 4.0
    codes = torch.arange(-260, 261, dtype=torch.float32) * adc.step + quarter
    signal = torch.complex(codes, -codes).to(torch.complex64).cuda()
    output = FrontendChain(FrontendSpec(port=PortSpec(1.0), adc=adc)).apply(signal)
    reference = quantize(signal, bits=adc.bits, full_scale=adc.full_scale)

    def _code(values: torch.Tensor) -> torch.Tensor:
        return torch.round((values + adc.full_scale) / adc.step)

    assert torch.equal(_code(output.signal.real), _code(reference.real))
    assert torch.equal(_code(output.signal.imag), _code(reference.imag))
    # The reconstructed voltage is allowed one unit in the last place, for the
    # fused-multiply-add reason above.
    assert torch.allclose(output.signal, reference, rtol=0.0, atol=2e-7)

    below = int((codes < -adc.full_scale).sum())
    above = int((codes > adc.full_scale).sum())
    assert below > 0 and above > 0
    # One count per COMPONENT, and the imaginary part is the negation of the
    # real one, so every out-of-range sample contributes exactly twice.
    assert int(output.diagnostics.clipped_components.item()) == 2 * (below + above)
    # The extreme codes land on full scale to within one unit in the last
    # place, again because the reconstruction is a fused multiply-add. The
    # statement being made is that nothing escapes the grid, not that the top
    # code is bit-exactly 1.0.
    ulp = adc.full_scale * 1e-6
    assert float(output.signal.real.max()) <= adc.full_scale + ulp
    assert float(output.signal.real.min()) >= -adc.full_scale - ulp


def test_the_full_scale_sine_sqnr_matches_the_textbook_figure():
    _, _, FrontendChain, FrontendSpec, _, _, PortSpec, _ = _specs()
    from witwin.radar.frontend import AdcSpec

    adc = AdcSpec(bits=10, full_scale=1.0)
    count = 1 << 16
    phase = 2 * math.pi * 97 * torch.arange(count, dtype=torch.float64) / count
    signal = torch.complex(phase.cos().float(), phase.sin().float()).to(torch.complex64).cuda()
    output = FrontendChain(FrontendSpec(port=PortSpec(1.0), adc=adc)).apply(signal).signal
    error_power = float(((output.real - signal.real).double() ** 2).mean())
    signal_power = float((signal.real.double() ** 2).mean())
    measured = 10.0 * math.log10(signal_power / error_power)
    assert abs(measured - adc.full_scale_sine_sqnr_db) < 0.3


# ---------------------------------------------------------------------------
# T4.7 - phase-noise PSD asymptote
# ---------------------------------------------------------------------------


def test_the_phase_noise_spectrum_follows_the_free_running_asymptote():
    """``L(f) = sigma_w^2 fs / (4 pi^2 f^2)``, to 1 dB over two octaves each way.

    What is asserted is the GENERATOR, not a datasheet. The model is a
    free-running oscillator: a ``-20 dB/decade`` slope with no close-in ``1/f^3``
    region and no far-out floor. Two further limitations are recorded on
    ``NoiseSpec`` and are deliberately NOT tested, because the model cannot
    produce them: the Wiener accumulation assumes a uniform sample spacing while
    a real FMCW time base has an idle gap, and range correlation is absent, so a
    homodyne receiver's close-range phase noise is grossly overstated. Writing an
    absolute close-range level test against this model would be asserting a
    number the physics does not claim.
    """

    _, _, FrontendChain, FrontendSpec, _, NoiseSpec, PortSpec, SeedSpec = _specs()

    sample_rate = 5e6
    offset = 1e5
    level_dbc = -90.0
    noise = NoiseSpec(
        noise_figure_db=0.0,
        bandwidth_hz=0.0,
        phase_noise_dbc_per_hz=level_dbc,
        phase_offset_hz=offset,
        phase_sample_rate_hz=sample_rate,
    )
    expected_sigma = wiener_innovation_sigma_rad(
        level_dbc_per_hz=level_dbc, offset_hz=offset, sample_rate_hz=sample_rate
    )
    assert math.isclose(noise.phase_innovation_sigma_rad, expected_sigma, rel_tol=1e-12)

    output = FrontendChain(FrontendSpec(port=PortSpec(1.0), noise=noise, seed=SeedSpec(11))).apply(_zeros(1 << 20))
    phase = output.diagnostics.phase_rad
    assert phase is not None and phase.device.type == "cuda"

    frequencies, psd = single_sideband_psd(phase, sample_rate_hz=sample_rate, segment=4096)
    deviations = []
    for target in (offset / 4, offset / 2, offset, 2 * offset, 4 * offset):
        index = int(torch.argmin((frequencies - target).abs()))
        # Average a few neighbouring bins: one periodogram bin of a random walk
        # is chi-squared with two degrees of freedom even after segment
        # averaging, and the band is flat enough over seven bins to make this a
        # variance reduction rather than a bias.
        measured = float(psd[max(index - 3, 1) : index + 4].mean())
        predicted = 10.0 ** (noise.single_sideband_dbc_per_hz(float(frequencies[index])) / 10.0)
        deviations.append(10.0 * math.log10(measured / predicted))
    assert max(abs(value) for value in deviations) < 1.0, deviations

    # And the slope itself, over the whole decade and a half.
    low = int(torch.argmin((frequencies - offset / 4).abs()))
    high = int(torch.argmin((frequencies - 4 * offset).abs()))
    decades = math.log10(float(frequencies[high]) / float(frequencies[low]))
    slope = (
        10.0 * math.log10(float(psd[high - 3 : high + 4].mean()))
        - 10.0 * math.log10(float(psd[low - 3 : low + 4].mean()))
    ) / decades
    assert abs(slope + 20.0) < 1.0, slope


# ---------------------------------------------------------------------------
# T4.9 - exactly one quantizer
# ---------------------------------------------------------------------------


def test_there_is_exactly_one_call_site_of_the_quantizer():
    """AST scan: one in ``frontend.py``, none in ``radar.py``.

    The old code had two quantiser owners and a runtime refusal to configure
    both. The refusal is unnecessary now because the second quantiser stopped
    existing, and this is the assertion that keeps it that way - a third one
    would be added by someone who could not see the first two.
    """

    call_sites = []
    for path in [REPO_ROOT / "witwin" / "radar" / "frontend.py"]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            name = function.attr if isinstance(function, ast.Attribute) else getattr(function, "id", "")
            if name == "frontend_quantize_forward":
                call_sites.append(path.name)
    assert call_sites == ["frontend.py"], call_sites

    radar_source = (REPO_ROOT / "witwin" / "radar" / "radar.py").read_text(encoding="utf-8")
    assert "frontend_quantize_forward" not in radar_source


# ---------------------------------------------------------------------------
# T4.12 - AGC breaks linearity, and the physics tests know it
# ---------------------------------------------------------------------------


def test_the_agc_is_nonlinear_and_the_chain_without_it_is_linear():
    """Both halves, so the invariant's caveat is a tested fact.

    With AGC on, doubling the input does NOT double the output: the measured RMS
    doubles too and the gain halves, so the output is unchanged up to the clamp.
    That is why the cross-waveform linearity invariant is stated with AGC off
    rather than with a tolerance.
    """

    _, AgcSpec, FrontendChain, FrontendSpec, LnaSpec, _, PortSpec, _ = _specs()

    generator = torch.Generator(device="cpu").manual_seed(23)
    signal = (
        torch.complex(torch.randn(1024, generator=generator), torch.randn(1024, generator=generator))
        .to(torch.complex64)
        .cuda()
    )

    with_agc = FrontendChain(FrontendSpec(port=PortSpec(1.0), agc=AgcSpec(target_rms=1.0, mode="global")))
    assert not torch.allclose(with_agc.apply(2 * signal).signal, 2 * with_agc.apply(signal).signal, rtol=1e-3)

    without_agc = FrontendChain(FrontendSpec(port=PortSpec(1.0), lna=LnaSpec(gain_db=6.0)))
    assert torch.allclose(
        without_agc.apply(2 * signal).signal, 2 * without_agc.apply(signal).signal, rtol=1e-6, atol=1e-7
    )


def test_the_agc_gain_matches_the_reference_and_hits_the_target_rms():
    _, AgcSpec, FrontendChain, FrontendSpec, _, _, PortSpec, _ = _specs()

    generator = torch.Generator(device="cpu").manual_seed(29)
    signal = (
        torch.complex(torch.randn(4096, generator=generator), torch.randn(4096, generator=generator))
        .to(torch.complex64)
        .cuda()
    )
    # The signal is deliberately near unit RMS so the required gain sits well
    # inside [min_gain, max_gain]. Scaled to 1e-3 the gain would be 1414 against
    # a 60 dB ceiling of 1000, and this would be measuring the CLAMP rather than
    # the gain - a real behaviour, but a different assertion.
    agc = AgcSpec(target_rms=2.0, mode="global")
    output = FrontendChain(FrontendSpec(port=PortSpec(1.0), agc=agc)).apply(signal)
    expected_gain, expected_rms = agc_gain(
        signal, target_rms=agc.target_rms, min_gain=agc.min_gain, max_gain=agc.max_gain
    )
    assert math.isclose(float(output.diagnostics.agc_gain[0]), expected_gain, rel_tol=1e-5)
    assert math.isclose(float(output.diagnostics.agc_rms[0]), expected_rms, rel_tol=1e-5)
    achieved = float((output.signal.real.double() ** 2 + output.signal.imag.double() ** 2).mean().sqrt())
    assert math.isclose(achieved, agc.target_rms, rel_tol=1e-5)


# ---------------------------------------------------------------------------
# T4.13 - the AGC costs no host observation
# ---------------------------------------------------------------------------


def test_the_agc_reads_nothing_to_the_host(monkeypatch):
    """The measured gain stays a device tensor. Every counter is zero.

    Reading it to build a Python scalar would be a per-frame device-to-host
    transfer, and it is the kind that hides well: a gain is one number, and one
    number looks free right up until it is inside a frame loop.
    """

    _, AgcSpec, FrontendChain, FrontendSpec, _, _, PortSpec, _ = _specs()

    counters = {"item": 0, "cpu": 0, "tolist": 0, "numpy": 0, "synchronize": 0}
    for name in ("item", "cpu", "tolist", "numpy"):
        original = getattr(torch.Tensor, name)

        def _wrapped(self, *args, _name=name, _original=original, **kwargs):
            counters[_name] += 1
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, name, _wrapped)

    original_synchronize = torch.cuda.synchronize

    def _synchronize(*args, **kwargs):
        counters["synchronize"] += 1
        return original_synchronize(*args, **kwargs)

    monkeypatch.setattr(torch.cuda, "synchronize", _synchronize)

    signal = torch.complex(torch.randn(2048, device="cuda"), torch.randn(2048, device="cuda")).to(torch.complex64)
    chain = FrontendChain(FrontendSpec(port=PortSpec(1.0), agc=AgcSpec(target_rms=1.0, mode="global")))
    output = chain.apply(signal)
    assert output.diagnostics.agc_gain.device.type == "cuda"
    assert counters == {"item": 0, "cpu": 0, "tolist": 0, "numpy": 0, "synchronize": 0}, counters


# ---------------------------------------------------------------------------
# T4.17 - AD
# ---------------------------------------------------------------------------


def test_the_frontend_jvp_matches_a_central_finite_difference():
    """The whole differentiable chain - phase, thermal, LNA, AGC - at once.

    The noise DRAWS are constants with respect to the derivative, which is what
    makes a finite difference of this chain meaningful at all: the same seed
    gives the same realisation on both sides of the difference, so what is being
    measured is the operator and not the noise.
    """

    from torch.autograd.forward_ad import dual_level, make_dual, unpack_dual

    _, AgcSpec, FrontendChain, FrontendSpec, LnaSpec, NoiseSpec, PortSpec, SeedSpec = _specs()

    chain = FrontendChain(
        FrontendSpec(
            port=PortSpec(IMPEDANCE),
            noise=NoiseSpec(
                noise_figure_db=3.0,
                bandwidth_hz=1e6,
                phase_noise_dbc_per_hz=-80.0,
                phase_offset_hz=1e5,
                phase_sample_rate_hz=1e6,
            ),
            lna=LnaSpec(gain_db=10.0),
            agc=AgcSpec(target_rms=1e-3, mode="global"),
            seed=SeedSpec(5),
        )
    )
    generator = torch.Generator(device="cpu").manual_seed(31)

    def _random(scale: float) -> torch.Tensor:
        return (
            torch.complex(torch.randn(4096, generator=generator), torch.randn(4096, generator=generator))
            .to(torch.complex64)
            .cuda()
            * scale
        )

    base = _random(1e-4)
    tangent = _random(1e-4)

    with dual_level():
        jvp = unpack_dual(chain.apply(make_dual(base, tangent)).signal).tangent
    assert jvp is not None, "the forward-mode tangent was swallowed"

    step = 1e-3
    difference = (chain.apply(base + step * tangent).signal - chain.apply(base - step * tangent).signal) / (2 * step)
    assert float((jvp - difference).abs().max() / difference.abs().max()) < 2e-3


def test_the_frontend_vjp_is_the_adjoint_of_its_jvp():
    """``<cotangent, JVP(tangent)> == <VJP(cotangent), tangent>``.

    The AGC is the reason this is worth asserting separately. Its derivative is a
    rank-one update through a group reduction, so a transposed or dropped term is
    invisible in the primal and in the forward mode taken alone.
    """

    from torch.autograd.forward_ad import dual_level, make_dual, unpack_dual

    _, AgcSpec, FrontendChain, FrontendSpec, LnaSpec, NoiseSpec, PortSpec, SeedSpec = _specs()

    chain = FrontendChain(
        FrontendSpec(
            port=PortSpec(IMPEDANCE),
            noise=NoiseSpec(noise_figure_db=3.0, bandwidth_hz=1e6),
            lna=LnaSpec(gain_db=10.0),
            agc=AgcSpec(target_rms=1e-3, mode="global"),
            seed=SeedSpec(5),
        )
    )
    generator = torch.Generator(device="cpu").manual_seed(37)

    def _random(scale: float) -> torch.Tensor:
        return (
            torch.complex(torch.randn(2048, generator=generator), torch.randn(2048, generator=generator))
            .to(torch.complex64)
            .cuda()
            * scale
        )

    base = _random(1e-4)
    tangent = _random(1e-4)
    cotangent = _random(1.0)

    with dual_level():
        jvp = unpack_dual(chain.apply(make_dual(base, tangent)).signal).tangent
        forward = float((jvp.real * cotangent.real).sum() + (jvp.imag * cotangent.imag).sum())

    real = base.real.contiguous().clone().requires_grad_(True)
    imag = base.imag.contiguous().clone().requires_grad_(True)
    output = chain.apply(torch.complex(real, imag)).signal
    loss = (output.real * cotangent.real).sum() + (output.imag * cotangent.imag).sum()
    loss.backward()
    reverse = float((real.grad * tangent.real).sum() + (imag.grad * tangent.imag).sum())
    assert math.isclose(forward, reverse, rel_tol=2e-5)


def test_the_quantizer_refuses_a_differentiable_input():
    """Fail loud, naming Phase 9 and the non-differentiability of ``round``.

    Silently detaching would return a number with no gradient where the caller
    asked for one, and a straight-through surrogate is a modelling decision the
    frontend is not entitled to make on its own.
    """

    from torch.autograd.forward_ad import dual_level, make_dual

    _, _, FrontendChain, FrontendSpec, _, _, PortSpec, _ = _specs()
    from witwin.radar.frontend import AdcSpec

    chain = FrontendChain(FrontendSpec(port=PortSpec(1.0), adc=AdcSpec(bits=8, full_scale=1.0)))
    signal = torch.complex(torch.randn(64, device="cuda"), torch.randn(64, device="cuda")).to(torch.complex64)

    with pytest.raises(RuntimeError, match="Phase-9"):
        chain.apply(signal.clone().requires_grad_(True))

    with dual_level():
        with pytest.raises(RuntimeError, match="not differentiable"):
            chain.apply(make_dual(signal, torch.ones_like(signal)))
