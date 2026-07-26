"""The dirichlet_spectrum family with complex path weights.

Plan work item 2: the native backend supports complex path weights, and the
existing real-amplitude Radar baseline is the special case ``a_im = 0``. Two
switches land with the weight, in the same change, because they are the same
act: a family that gains complex Channel weights inherits the carrier
double-count the moment it does, and a family that consumes a round-trip delay
must stop deriving one from a distance.

The oracle here is a float64 closed form of the kernel's own documented
expression, written independently in Torch. It is used three ways - primal,
VJP by ``torch.autograd``, and JVP by forward-mode - so a single reference
statement covers all three companions and none of them is checked against
another kernel.

Recorded deviation from the stage brief, item T0.9: the brief asks that
``grad_a_im`` be "exactly zero for a purely real output projection". It is not,
and should not be. ``dL/da_im = -gout_re * R.im + gout_im * R.re``, so a
cotangent with ``gout_im = 0`` still produces ``-gout_re * R.im``, which is
nonzero wherever the response has an imaginary part - that is the whole content
of a complex weight. What IS asserted is that the component is live for a
complex weight and that all three gradients match the float64 reference.

Recorded deviation, item T0.3: bit-identity against the PRE-widening binary was
measured out of tree, by capturing ``forward_chunked`` and
``forward_mimo_linear_chunked`` on a fixed input before and after the ABI
change (``torch.equal`` on both, four arrays, max absolute difference 0.0). It
cannot be re-run from inside the repository, because the pre-change binary no
longer exists to build. What this file pins durably is the property that made
that equality possible: a zero imaginary component is inert, statement for
statement.
"""

from __future__ import annotations

import math

import pytest
import torch

from witwin.radar.synthesis.dirichlet_spectrum import (
    DirichletSpectrumSpec,
    chunked_spectra,
    spectrum_vjp,
    spectrum_vjp_per_bin,
    spectrum_vjp_single_block,
)

pytestmark = pytest.mark.gpu


C0 = 299792458.0

NUM_SAMPLES = 32
N_FFT = 128
NUM_BINS = 64
SAMPLE_RATE_HZ = 5.0e6
SLOPE_HZ_PER_S = 5.0e13
FC_HZ = 1.0e9
T_START_S = 0.0

# Deliberately a modest carrier. The production geometry puts 77 GHz through a
# float32 delay, which costs order 1e-2 rad of absolute phase and makes a
# pointwise comparison against a float64 reference meaningless - the existing
# suite compares that configuration by correlation instead. Here the question
# is whether the complex algebra and its derivatives are right, so the
# configuration is chosen to keep float32 honest and let the assertions be
# pointwise.


def _requires_cuda():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")


def _spec(*, tau_is_seconds: int = 0) -> DirichletSpectrumSpec:
    k0_per_meter = (SLOPE_HZ_PER_S * 2.0 / C0) * N_FFT / SAMPLE_RATE_HZ
    k0_per_second = SLOPE_HZ_PER_S * N_FFT / SAMPLE_RATE_HZ
    return DirichletSpectrumSpec(
        n=(NUM_SAMPLES - 1) / 2,
        k0_per_meter=k0_per_second if tau_is_seconds else k0_per_meter,
        num_bins=NUM_BINS,
        n_fft=N_FFT,
        fc=FC_HZ,
        slope_hz_per_s=SLOPE_HZ_PER_S,
        t_start_s=T_START_S,
        tau_is_seconds=tau_is_seconds,
    )


def _inputs(*, device: str = "cuda"):
    d = torch.tensor([1.25, 2.5, 3.75, 4.5], dtype=torch.float32, device=device)
    a_re = torch.tensor([0.7, -0.4, 1.0, 0.25], dtype=torch.float32, device=device)
    a_im = torch.tensor([-0.3, 0.85, 0.0, 0.6], dtype=torch.float32, device=device)
    return d, a_re, a_im


def _reference_spectrum(
    path_values: torch.Tensor,
    a_re: torch.Tensor,
    a_im: torch.Tensor,
    spec: DirichletSpectrumSpec,
) -> torch.Tensor:
    """The kernel's documented closed form, in float64, written independently.

    Differentiable in all three inputs, so the same expression serves as the
    primal oracle, the VJP oracle, and the JVP oracle.
    """

    n = spec.n
    tau = path_values if spec.tau_is_seconds else 2.0 * path_values / C0
    phi0 = 2.0 * math.pi * (
        spec.fc * tau + spec.slope_hz_per_s * tau * (spec.t_start_s - 0.5 * tau)
    )
    k0 = path_values * spec.k0_per_meter
    bins = torch.arange(spec.num_bins, dtype=path_values.dtype, device=path_values.device)
    x = 2.0 * math.pi * (bins.unsqueeze(0) - k0.unsqueeze(1)) / spec.n_fft
    scale = torch.sin((n + 0.5) * x) / torch.sin(0.5 * x)
    response = scale * torch.exp(-1j * n * x) * torch.exp(1j * phi0).unsqueeze(1)
    weight = torch.complex(a_re, a_im)
    return (weight.unsqueeze(1) * response).sum(dim=0)


def _one_spectrum(d, a_re, a_im, spec, *, targets_per_spectrum=None):
    per = targets_per_spectrum or int(d.shape[0])
    return chunked_spectra(
        d, a_re, a_im, spec=spec, targets_per_spectrum=per
    ).sum(dim=0)


# ---------------------------------------------------------------------------
# Primal
# ---------------------------------------------------------------------------


def test_the_complex_spectrum_matches_the_float64_closed_form():
    _requires_cuda()
    spec = _spec()
    d, a_re, a_im = _inputs()

    actual = _one_spectrum(d, a_re, a_im, spec)
    reference = _reference_spectrum(
        d.to(torch.float64), a_re.to(torch.float64), a_im.to(torch.float64), spec
    )

    peak = reference.abs().max()
    error = (actual.to(torch.complex128) - reference).abs().max()
    assert error / peak < 1e-4, float(error / peak)


def test_a_zero_imaginary_weight_is_inert():
    """The real path is not a branch; it is the value ``a_im = 0``.

    Bit equality, not closeness: the imaginary contribution is accumulated as
    its own pair of statements precisely so that zero times anything leaves the
    real accumulation untouched. Fusing it into ``a_re*R.re - a_im*R.im`` would
    be one rounding instead of two and would break exactly this.
    """

    _requires_cuda()
    spec = _spec()
    d, a_re, _ = _inputs()

    implicit = chunked_spectra(d, a_re, spec=spec, targets_per_spectrum=4)
    explicit = chunked_spectra(
        d, a_re, torch.zeros_like(a_re), spec=spec, targets_per_spectrum=4
    )
    assert torch.equal(implicit, explicit)


def test_a_nonzero_imaginary_weight_changes_the_answer():
    """Non-vacuity guard for the test above."""

    _requires_cuda()
    spec = _spec()
    d, a_re, a_im = _inputs()

    real_only = chunked_spectra(d, a_re, spec=spec, targets_per_spectrum=4)
    complex_weight = chunked_spectra(
        d, a_re, a_im, spec=spec, targets_per_spectrum=4
    )
    assert not torch.equal(real_only, complex_weight)


def test_a_negative_weight_flips_the_sign_exactly():
    """The reflection flip. A ``complex(abs(a), 0)`` promotion fails this."""

    _requires_cuda()
    spec = _spec()
    d, a_re, _ = _inputs()

    positive = chunked_spectra(d, a_re, spec=spec, targets_per_spectrum=4)
    negated = chunked_spectra(d, -a_re, spec=spec, targets_per_spectrum=4)
    assert torch.equal(negated, -positive)


@pytest.mark.parametrize("psi", [0.0, math.pi / 4, math.pi / 2, 2.4])
def test_a_complex_weight_rotates_the_output(psi):
    """Linearity in the complex weight, stated as a rotation.

    ``sum_k |a| e^{j psi} R_k == e^{j psi} sum_k |a| R_k`` is exactly the
    statement that the phase of a material or target response reaches the IQ
    unmodified, which is half of the plan's criterion A3.
    """

    _requires_cuda()
    spec = _spec()
    d, magnitude, _ = _inputs()
    magnitude = magnitude.abs()

    base = _one_spectrum(d, magnitude, torch.zeros_like(magnitude), spec)
    rotated = _one_spectrum(
        d,
        magnitude * math.cos(psi),
        magnitude * math.sin(psi),
        spec,
    )
    expected = base * complex(math.cos(psi), math.sin(psi))

    tolerance = 1e-6 * float(expected.abs().max())
    torch.testing.assert_close(rotated, expected, rtol=1e-5, atol=tolerance)


def test_tau_in_seconds_is_the_same_physics_as_a_one_way_distance():
    """The two inputs differ only in where the float32 division happens.

    A caller that already holds a round-trip delay must not have to turn it back
    into a distance so the kernel can halve it again. The matching k0 scale is
    part of the contract: ``slope * n_fft / fs`` instead of
    ``(slope * 2 / c0) * n_fft / fs``.
    """

    _requires_cuda()
    d, a_re, a_im = _inputs()
    delays = (d.to(torch.float64) * 2.0 / C0).to(torch.float32)

    by_distance = _one_spectrum(d, a_re, a_im, _spec())
    by_delay = _one_spectrum(delays, a_re, a_im, _spec(tau_is_seconds=1))

    tolerance = 1e-6 * float(by_distance.abs().max())
    torch.testing.assert_close(by_delay, by_distance, rtol=1e-6, atol=tolerance)


def test_a_zero_carrier_leaves_the_absolute_phase_to_the_weight():
    """``fc`` is the carrier home, exactly as ``carrier_hz`` is in the beat family.

    With ``fc = 0`` the kernel applies no absolute reference-frequency phase,
    so the two settings differ by exactly ``exp(+j 2 pi fc tau)`` per row. That
    is what lets a Channel-sourced weight, which already carries that factor,
    be consumed without counting it twice.
    """

    _requires_cuda()
    with_carrier = _spec()
    without_carrier = DirichletSpectrumSpec(
        n=with_carrier.n,
        k0_per_meter=with_carrier.k0_per_meter,
        num_bins=with_carrier.num_bins,
        n_fft=with_carrier.n_fft,
        fc=0.0,
        slope_hz_per_s=with_carrier.slope_hz_per_s,
        t_start_s=with_carrier.t_start_s,
    )

    d = torch.tensor([2.5], dtype=torch.float32, device="cuda")
    a_re = torch.tensor([0.8], dtype=torch.float32, device="cuda")
    a_im = torch.tensor([-0.35], dtype=torch.float32, device="cuda")

    tau = 2.0 * float(d[0]) / C0
    turn = 2.0 * math.pi * FC_HZ * tau
    carried = _one_spectrum(d, a_re, a_im, with_carrier)
    # Applying the carrier to the WEIGHT instead of in the kernel must give the
    # same spectrum.
    on_weight = _one_spectrum(
        d,
        a_re * math.cos(turn) - a_im * math.sin(turn),
        a_re * math.sin(turn) + a_im * math.cos(turn),
        without_carrier,
    )

    tolerance = 1e-4 * float(carried.abs().max())
    torch.testing.assert_close(on_weight, carried, rtol=1e-4, atol=tolerance)


# ---------------------------------------------------------------------------
# VJP
# ---------------------------------------------------------------------------


def _reference_gradients(d, a_re, a_im, spec, gout_re, gout_im):
    ref_d = d.detach().to(torch.float64).requires_grad_(True)
    ref_a_re = a_re.detach().to(torch.float64).requires_grad_(True)
    ref_a_im = a_im.detach().to(torch.float64).requires_grad_(True)
    spectrum = _reference_spectrum(ref_d, ref_a_re, ref_a_im, spec)
    loss = (
        spectrum.real * gout_re.to(torch.float64)
        + spectrum.imag * gout_im.to(torch.float64)
    ).sum()
    return torch.autograd.grad(loss, (ref_d, ref_a_re, ref_a_im))


@pytest.mark.parametrize(
    "operator",
    ["backward_parallel_bins", "backward_per_bin", "backward"],
)
def test_every_backward_operator_matches_the_float64_reference(operator):
    _requires_cuda()
    spec = _spec()
    d, a_re, a_im = _inputs()
    gout_re = torch.linspace(0.2, 1.3, NUM_BINS, dtype=torch.float32, device="cuda")
    gout_im = torch.linspace(-0.7, 0.4, NUM_BINS, dtype=torch.float32, device="cuda")

    dispatch = {
        "backward_parallel_bins": spectrum_vjp,
        "backward_per_bin": spectrum_vjp_per_bin,
        "backward": spectrum_vjp_single_block,
    }[operator]
    grad_d, grad_a_re, grad_a_im = dispatch(
        d, a_re, gout_re, gout_im, spec=spec, weight_im=a_im
    )

    ref_d, ref_a_re, ref_a_im = _reference_gradients(
        d, a_re, a_im, spec, gout_re, gout_im
    )
    torch.testing.assert_close(grad_d.to(torch.float64), ref_d, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(
        grad_a_re.to(torch.float64), ref_a_re, rtol=1e-3, atol=1e-3
    )
    torch.testing.assert_close(
        grad_a_im.to(torch.float64), ref_a_im, rtol=1e-3, atol=1e-3
    )


def test_the_imaginary_weight_gradient_is_live():
    _requires_cuda()
    spec = _spec()
    d, a_re, a_im = _inputs()
    gout_re = torch.linspace(0.2, 1.3, NUM_BINS, dtype=torch.float32, device="cuda")
    gout_im = torch.linspace(-0.7, 0.4, NUM_BINS, dtype=torch.float32, device="cuda")

    _, _, grad_a_im = spectrum_vjp(
        d, a_re, gout_re, gout_im, spec=spec, weight_im=a_im
    )
    assert torch.isfinite(grad_a_im).all()
    assert grad_a_im.abs().max() > 0.0


def test_the_float64_reference_passes_gradcheck():
    """Validate the oracle before trusting it as one."""

    _requires_cuda()
    spec = _spec()
    d, a_re, a_im = _inputs(device="cpu")
    gout_re = torch.linspace(0.2, 1.3, NUM_BINS, dtype=torch.float64)
    gout_im = torch.linspace(-0.7, 0.4, NUM_BINS, dtype=torch.float64)

    def scalar_loss(path_values, weight_re, weight_im):
        spectrum = _reference_spectrum(path_values, weight_re, weight_im, spec)
        return (spectrum.real * gout_re + spectrum.imag * gout_im).sum()

    inputs = tuple(
        value.to(torch.float64).requires_grad_(True) for value in (d, a_re, a_im)
    )
    assert torch.autograd.gradcheck(scalar_loss, inputs, eps=1e-7, atol=1e-6)


def test_the_public_chirp_gradient_still_flows_through_the_moved_owner():
    """The Python owner move must not have detached the production path."""

    _requires_cuda()
    from conftest import MINIMAL_CONFIG
    from witwin.radar import Radar, RadarConfig

    radar = Radar(
        RadarConfig.from_dict(
            {
                **MINIMAL_CONFIG,
                "adc_samples": 64,
                "num_range_bins": 64,
                "chirp_per_frame": 2,
                "num_doppler_bins": 2,
            }
        ),
        device="cuda",
    )
    distances = torch.tensor(
        [1.1, 2.4], dtype=torch.float32, device="cuda", requires_grad=True
    )
    amplitudes = torch.tensor(
        [0.9, 0.6], dtype=torch.float32, device="cuda", requires_grad=True
    )
    radar.chirp(distances, amplitudes).abs().square().sum().backward()

    assert distances.grad is not None and torch.isfinite(distances.grad).all()
    assert amplitudes.grad is not None and torch.isfinite(amplitudes.grad).all()
    assert distances.grad.abs().max() > 0.0


# ---------------------------------------------------------------------------
# JVP
# ---------------------------------------------------------------------------

# Central-difference steps, one per variable, chosen so the truncation error
# stays well under the 2e-3 tolerance while the perturbation is still many
# float32 ulps of the operand. The distance step is the tight one: the phase
# slope here is 2 pi * (2/c0) * fc ~ 42 rad per metre, so 1e-4 m is 4e-3 rad,
# and the quadratic term is under 1e-5 of the derivative.
FD_STEPS = {"path": 1.0e-4, "weight_re": 1.0e-3, "weight_im": 1.0e-3}


def _jvp(d, a_re, a_im, spec, tan_d, tan_a_re, tan_a_im):
    with torch.autograd.forward_ad.dual_level():
        dual_d = torch.autograd.forward_ad.make_dual(d, tan_d)
        dual_a_re = torch.autograd.forward_ad.make_dual(a_re, tan_a_re)
        dual_a_im = torch.autograd.forward_ad.make_dual(a_im, tan_a_im)
        out = chunked_spectra(
            dual_d,
            dual_a_re,
            dual_a_im,
            spec=spec,
            targets_per_spectrum=int(d.shape[0]),
        )
        tangent_re = torch.autograd.forward_ad.unpack_dual(out.real).tangent
        tangent_im = torch.autograd.forward_ad.unpack_dual(out.imag).tangent
    return torch.complex(tangent_re, tangent_im).sum(dim=0)


@pytest.mark.parametrize("variable", ["path", "weight_re", "weight_im"])
def test_dirichlet_jvp_matches_central_finite_differences(variable):
    """Finite differences are the TEST oracle and appear nowhere else.

    A forward-only dual has ``requires_grad == False``. The eager shortcut this
    family used to carry took that as "no derivative wanted" and returned a
    plain tensor, so this test is also the regression guard for the shortcut's
    deletion: it can only pass if the spectrum routes through
    ``Function.apply`` unconditionally.
    """

    _requires_cuda()
    spec = _spec()
    d, a_re, a_im = _inputs()
    direction = torch.tensor(
        [1.0, -0.5, 0.25, 0.75], dtype=torch.float32, device="cuda"
    )
    zero = torch.zeros_like(direction)
    tangents = {
        "path": (direction, zero, zero),
        "weight_re": (zero, direction, zero),
        "weight_im": (zero, zero, direction),
    }[variable]

    analytic = _jvp(d, a_re, a_im, spec, *tangents)

    step = FD_STEPS[variable]
    offsets = [component * step for component in tangents]
    plus = _reference_spectrum(
        (d + offsets[0]).to(torch.float64),
        (a_re + offsets[1]).to(torch.float64),
        (a_im + offsets[2]).to(torch.float64),
        spec,
    )
    minus = _reference_spectrum(
        (d - offsets[0]).to(torch.float64),
        (a_re - offsets[1]).to(torch.float64),
        (a_im - offsets[2]).to(torch.float64),
        spec,
    )
    finite = (plus - minus) / (2.0 * step)

    error = (analytic.to(torch.complex128) - finite).abs().max()
    scale = finite.abs().max()
    assert error / scale < 2e-3, (variable, float(error / scale))


def test_a_forward_dual_survives_the_public_spectrum_entry():
    _requires_cuda()
    spec = _spec()
    d, a_re, a_im = _inputs()
    tangent = torch.ones_like(d)
    zero = torch.zeros_like(d)

    result = _jvp(d, a_re, a_im, spec, tangent, zero, zero)
    assert torch.isfinite(result).all()
    assert result.abs().max() > 0.0
