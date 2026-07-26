"""Operator-level AD for the native OFDM CFR primitive.

The oracle is the float64 pure-Torch cube in ``tests/support/reference_ofdm``,
which is FD-validated in float64 FIRST. ``gradcheck`` on the float32 kernel runs
too, but only as corroboration with explicitly loose tolerances: the production
cube is float32 and the OFDM subcarrier phase is a small increment sitting on a
large frozen carrier phase, so a naive central difference on the production
chain subtracts two nearly equal float32 numbers and can return an exactly zero
derivative that looks like a real answer.

Every finite difference in this file is a TEST ORACLE. Production derivatives
are the registered ``ofdm_cfr_backward`` and ``ofdm_cfr_jvp`` companions and
nothing else.

The differentiable inputs are ``tau_rt``, ``tau_rate``, ``weight_re``, and
``weight_im``. The one that needs its own test is ``tau_rate``: its derivative
is ``-2 pi t_l (n df + f_c + f_r)``, which is NOT ``d/dtau_rt`` times ``t_l``,
because the carrier rate multiplies the drift and not the full delay. At the
production carrier placement the difference between the two forms is a factor of
about ten thousand.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.autograd.forward_ad as forward_ad

from support import fd  # noqa: E402
from support import reference_ofdm as ref  # noqa: E402
from witwin.radar.synthesis.contracts import (  # noqa: E402
    SPEED_OF_LIGHT_M_PER_S,
    OfdmCfrSpec,
)
from witwin.radar.synthesis.ofdm_cfr import synthesize_cfr_rows  # noqa: E402


pytestmark = pytest.mark.gpu


C0 = SPEED_OF_LIGHT_M_PER_S
F_REF_HZ = 77.0e9
DF_HZ = 120.0e3
NUM_SUBCARRIERS = 64


# The PRODUCTION carrier placement: the absolute carrier phase lives in the
# Channel weight and carrier_rate_hz supplies the inter-symbol Doppler term the
# frozen weight cannot carry. Deriving the operator AD against this setting is
# deliberate - carrier_rate_hz is exactly what makes d(phi)/d(tau_rate) differ
# from d(phi)/d(tau_rt) * t_l, and a spec with it zeroed would never exercise it.
SPEC = OfdmCfrSpec(
    num_subcarriers=NUM_SUBCARRIERS,
    num_symbols=5,
    subcarrier_spacing_hz=DF_HZ,
    cyclic_prefix_s=2.0e-6,
    reference_frequency_hz=F_REF_HZ,
    max_expected_delay_s=1.0e-6,
    carrier_hz=0.0,
    carrier_rate_hz=F_REF_HZ,
)

DELAYS = (2.4683743e-8, 6.6712819e-7)
RATES = (8.0055e-8, -3.1e-8)
WEIGHTS = (0.6 - 0.3j, -0.2 + 0.45j)

# Steps, chosen per variable and recorded. The scale each one has to clear is
# the phase swing it produces:
#   tau_rt    d(phi)/d(tau)  = 2 pi n df          up to 4.75e7 rad/s
#   tau_rate  d(phi)/d(rate) = 2 pi t_l (n df + f_ref)  up to 1.6e8 rad
#   weight    enters linearly
STEP_TAU_S = 1.0e-13
STEP_RATE = 1.0e-13
STEP_WEIGHT = 1.0e-6


def _imaginary_central_difference(evaluate, value, index, step):
    """``d(loss)/d(Im w)`` by a central difference along the imaginary axis.

    ``support.fd.central_difference`` divides by the step, and a purely
    imaginary step would make the quotient complex; the directional derivative
    along ``i`` is the real quotient over the REAL step length. Kept here rather
    than widened into the shared helper because it is the one place a complex
    parameter is differentiated component-wise.
    """

    plus = value.clone()
    minus = value.clone()
    plus[index] = plus[index] + 1j * step
    minus[index] = minus[index] - 1j * step
    return float((evaluate(plus) - evaluate(minus)) / (2.0 * step))


def _cpu_inputs():
    return (
        torch.tensor(DELAYS, dtype=torch.float64),
        torch.tensor(RATES, dtype=torch.float64),
        torch.tensor(WEIGHTS, dtype=torch.complex128),
        torch.tensor([0, len(DELAYS)], dtype=torch.int64),
    )


def _cuda_inputs():
    return (
        torch.tensor(DELAYS, dtype=torch.float32, device="cuda"),
        torch.tensor(RATES, dtype=torch.float32, device="cuda"),
        torch.tensor(WEIGHTS, dtype=torch.complex64, device="cuda"),
        torch.tensor([0, len(DELAYS)], dtype=torch.int64, device="cuda"),
    )


def _reference_loss(tau, rate, weight, offsets, target, spec=SPEC):
    return ref.cfr_loss(ref.cfr_cube(tau, rate, weight, offsets, spec), target)


def _production_loss(tau, rate, weight, offsets, target, spec=SPEC):
    cube = synthesize_cfr_rows(tau, rate, weight, offsets, spec)
    return ref.cfr_loss(cube.cpu(), target)


@pytest.fixture(scope="module")
def target_cube():
    torch.manual_seed(20260725)
    return torch.randn(
        (SPEC.num_symbols, 1, SPEC.num_subcarriers), dtype=torch.complex128
    )


# --------------------------------------------------------------------------
# Validate the oracle before anything is compared against it
# --------------------------------------------------------------------------


def test_oracle_gradients_agree_with_float64_finite_differences(target_cube):
    tau, rate, weight, offsets = _cpu_inputs()
    tau = tau.clone().requires_grad_(True)
    rate = rate.clone().requires_grad_(True)
    weight = weight.clone().requires_grad_(True)
    _reference_loss(tau, rate, weight, offsets, target_cube).backward()

    for index in range(len(DELAYS)):
        measured = fd.central_difference(
            lambda value: _reference_loss(
                value, rate.detach(), weight.detach(), offsets, target_cube
            ),
            tau.detach(),
            index,
            STEP_TAU_S,
        )
        assert fd.relative_error(measured, float(tau.grad[index]), floor=1e-6) < 1e-4

        measured_rate = fd.central_difference(
            lambda value: _reference_loss(
                tau.detach(), value, weight.detach(), offsets, target_cube
            ),
            rate.detach(),
            index,
            STEP_RATE,
        )
        assert (
            fd.relative_error(measured_rate, float(rate.grad[index]), floor=1e-6)
            < 1e-4
        )

        measured_re = fd.central_difference(
            lambda value: _reference_loss(
                tau.detach(), rate.detach(), value, offsets, target_cube
            ),
            weight.detach(),
            index,
            STEP_WEIGHT,
        )
        # Torch's complex autograd convention: .grad holds the conjugate
        # Wirtinger derivative, so d(loss)/d(Re w) is +Re(grad).
        assert (
            fd.relative_error(measured_re, float(weight.grad[index].real), floor=1e-9)
            < 1e-5
        )

        measured_im = _imaginary_central_difference(
            lambda value: _reference_loss(
                tau.detach(), rate.detach(), value, offsets, target_cube
            ),
            weight.detach(),
            index,
            STEP_WEIGHT,
        )
        assert (
            fd.relative_error(measured_im, float(weight.grad[index].imag), floor=1e-9)
            < 1e-5
        )


# --------------------------------------------------------------------------
# T2.10  VJP
# --------------------------------------------------------------------------


def test_native_vjp_matches_the_oracle(target_cube):
    tau, rate, weight, offsets = _cuda_inputs()
    tau = tau.clone().requires_grad_(True)
    rate = rate.clone().requires_grad_(True)
    weight = weight.clone().requires_grad_(True)
    _production_loss(tau, rate, weight, offsets, target_cube).backward()

    o_tau, o_rate, o_weight, o_offsets = _cpu_inputs()
    o_tau = o_tau.clone().requires_grad_(True)
    o_rate = o_rate.clone().requires_grad_(True)
    o_weight = o_weight.clone().requires_grad_(True)
    _reference_loss(o_tau, o_rate, o_weight, o_offsets, target_cube).backward()

    for index in range(len(DELAYS)):
        assert (
            fd.relative_error(
                float(tau.grad[index]), float(o_tau.grad[index]), floor=1e-6
            )
            < 1e-3
        ), index
        assert (
            fd.relative_error(
                float(rate.grad[index]), float(o_rate.grad[index]), floor=1e-6
            )
            < 1e-3
        ), index
        assert (
            fd.relative_error(
                float(weight.grad[index].real),
                float(o_weight.grad[index].real),
                floor=1e-9,
            )
            < 1e-3
        ), index
        assert (
            fd.relative_error(
                float(weight.grad[index].imag),
                float(o_weight.grad[index].imag),
                floor=1e-9,
            )
            < 1e-3
        ), index

    # Every row must actually be carrying gradient, or the comparison above is
    # satisfied by zeros on both sides.
    assert float(tau.grad.abs().min()) > 0.0
    assert float(rate.grad.abs().min()) > 0.0
    assert float(weight.grad.abs().min()) > 0.0


MULTI_DELAYS = (2.4683743e-8, 6.6712819e-7, 1.7e-8, 3.1e-8, 9.0e-9)
MULTI_RATES = (8.0055e-8, -3.1e-8, 4.0e-9, -2.2e-8, 7.5e-9)
MULTI_WEIGHTS = (0.6 - 0.3j, -0.2 + 0.45j, 0.15 + 0.8j, -0.5 - 0.1j, 0.33 + 0.22j)
# Three segments with different row counts. Row 2 sits exactly on a boundary,
# which is the index where the half-open partition is decided, and the trailing
# segment is empty.
MULTI_OFFSETS = (0, 2, 5, 5)


@pytest.fixture(scope="module")
def multi_target_cube():
    torch.manual_seed(20260726)
    return torch.randn(
        (SPEC.num_symbols, len(MULTI_OFFSETS) - 1, SPEC.num_subcarriers),
        dtype=torch.complex128,
    )


def test_multi_segment_vjp_matches_the_oracle(multi_target_cube):
    """The backward kernel's per-path segment mapping, under gradient.

    ``segment_of_each_row`` feeds ONLY the backward kernel; forward and JVP read
    ``pair_offsets`` directly. A single-segment gradient test cannot see it,
    because there the mapping is the constant zero and cannot be wrong.
    """

    tau = torch.tensor(MULTI_DELAYS, dtype=torch.float32, device="cuda")
    rate = torch.tensor(MULTI_RATES, dtype=torch.float32, device="cuda")
    weight = torch.tensor(MULTI_WEIGHTS, dtype=torch.complex64, device="cuda")
    offsets = torch.tensor(MULTI_OFFSETS, dtype=torch.int64, device="cuda")

    tau = tau.clone().requires_grad_(True)
    rate = rate.clone().requires_grad_(True)
    weight = weight.clone().requires_grad_(True)
    cube = synthesize_cfr_rows(tau, rate, weight, offsets, SPEC)
    ref.cfr_loss(cube.cpu(), multi_target_cube).backward()

    o_tau = torch.tensor(MULTI_DELAYS, dtype=torch.float64).requires_grad_(True)
    o_rate = torch.tensor(MULTI_RATES, dtype=torch.float64).requires_grad_(True)
    o_weight = torch.tensor(MULTI_WEIGHTS, dtype=torch.complex128).requires_grad_(
        True
    )
    o_offsets = torch.tensor(MULTI_OFFSETS, dtype=torch.int64)
    o_cube = ref.cfr_cube(o_tau, o_rate, o_weight, o_offsets, SPEC)
    ref.cfr_loss(o_cube, multi_target_cube).backward()

    for index in range(len(MULTI_DELAYS)):
        assert (
            fd.relative_error(
                float(tau.grad[index]), float(o_tau.grad[index]), floor=1e-6
            )
            < 1e-3
        ), index
        assert (
            fd.relative_error(
                float(rate.grad[index]), float(o_rate.grad[index]), floor=1e-6
            )
            < 1e-3
        ), index
        assert (
            fd.relative_error(
                float(weight.grad[index].real),
                float(o_weight.grad[index].real),
                floor=1e-9,
            )
            < 1e-3
        ), index
        assert (
            fd.relative_error(
                float(weight.grad[index].imag),
                float(o_weight.grad[index].imag),
                floor=1e-9,
            )
            < 1e-3
        ), index

    assert float(weight.grad.abs().min()) > 1e-9


def test_the_segment_mapping_is_the_half_open_partition():
    """The mapping itself, stated directly rather than only through gradients.

    An offsets table is a half-open partition ``[start, end)``, so a row whose
    index equals a boundary belongs to the NEXT segment. ``right=False`` puts it
    in the previous one, which is wrong for exactly one row per boundary.
    """

    from witwin.radar.synthesis.assembly import segment_of_each_row

    offsets = torch.tensor(MULTI_OFFSETS, dtype=torch.int64, device="cuda")
    mapping = segment_of_each_row(offsets, len(MULTI_DELAYS))
    assert mapping.tolist() == [0, 0, 1, 1, 1]
    # An empty trailing segment claims no rows.
    assert 2 not in mapping.tolist()


# --------------------------------------------------------------------------
# T2.10  JVP
# --------------------------------------------------------------------------


def test_native_jvp_matches_the_oracle_and_a_float64_finite_difference(target_cube):
    tau, rate, weight, offsets = _cuda_inputs()
    d_tau = torch.tensor([1.0e-9, -3.0e-10], dtype=torch.float32, device="cuda")
    d_rate = torch.tensor([2.0e-10, 5.0e-11], dtype=torch.float32, device="cuda")

    with forward_ad.dual_level():
        dual_tau = forward_ad.make_dual(tau, d_tau)
        dual_rate = forward_ad.make_dual(rate, d_rate)
        cube = synthesize_cfr_rows(dual_tau, dual_rate, weight, offsets, SPEC)
        loss = ref.cfr_loss(cube.cpu(), target_cube)
        tangent = forward_ad.unpack_dual(loss).tangent
        assert tangent is not None, "the native jvp companion was not reached"
        measured = float(tangent)

    o_tau, o_rate, o_weight, o_offsets = _cpu_inputs()
    o_d_tau = d_tau.double().cpu()
    o_d_rate = d_rate.double().cpu()

    with forward_ad.dual_level():
        oracle_tangent = forward_ad.unpack_dual(
            _reference_loss(
                forward_ad.make_dual(o_tau, o_d_tau),
                forward_ad.make_dual(o_rate, o_d_rate),
                o_weight,
                o_offsets,
                target_cube,
            )
        ).tangent
    oracle = float(oracle_tangent)

    # FD validates the oracle. The step is scaled DOWN from the tangent: a unit
    # step along a 1e-9 s delay tangent swings the top subcarrier's phase by
    # 0.05 rad and the drift term by far more, so the difference quotient there
    # would not be a derivative.
    directional = fd.directional_derivative(
        lambda t, r: _reference_loss(t, r, o_weight, o_offsets, target_cube),
        (o_tau, o_rate),
        (o_d_tau, o_d_rate),
        1e-4,
    )
    assert fd.relative_error(directional, oracle, floor=1e-9) < 1e-4
    assert fd.relative_error(measured, oracle, floor=1e-9) < 2e-3


@pytest.mark.parametrize("variable", ["tau_rt", "tau_rate", "weight_re", "weight_im"])
def test_the_jvp_of_each_differentiable_input_matches_a_central_difference(variable):
    """All four inputs, on the CUBE rather than through a scalar loss.

    A loss collapses four gradients into one number and can hide a sign error in
    one of them behind the others. This compares the whole complex cube.

    The finite difference is taken on the FLOAT64 ORACLE at the same
    float32-rounded operating point, never on the production kernel. Differencing
    the production cube would subtract two nearly equal float32 numbers: at a
    1e-6 weight step the signal is 1e-6 against a 6e-8 float32 noise floor, a
    6 percent error that has nothing to do with the derivative under test. That
    conditioning trap is the reason the oracle exists at all.
    """

    tau, rate, weight, offsets = _cuda_inputs()
    steps = {
        "tau_rt": STEP_TAU_S,
        "tau_rate": STEP_RATE,
        "weight_re": STEP_WEIGHT,
        "weight_im": STEP_WEIGHT,
    }
    step = steps[variable]
    direction = torch.tensor([1.0, -0.5], dtype=torch.float32, device="cuda")

    # The oracle's operating point is the float32 one the kernel was handed,
    # promoted without moving.
    o_tau = tau.double().cpu()
    o_rate = rate.double().cpu()
    o_weight = weight.to(torch.complex128).cpu()
    o_offsets = offsets.cpu()
    o_direction = direction.double().cpu()

    def evaluate(offset: float) -> torch.Tensor:
        moved_tau = o_tau
        moved_rate = o_rate
        moved_weight = o_weight
        delta = (offset * step) * o_direction
        if variable == "tau_rt":
            moved_tau = o_tau + delta
        elif variable == "tau_rate":
            moved_rate = o_rate + delta
        elif variable == "weight_re":
            moved_weight = o_weight + torch.complex(delta, torch.zeros_like(delta))
        else:
            moved_weight = o_weight + torch.complex(torch.zeros_like(delta), delta)
        return ref.cfr_cube(moved_tau, moved_rate, moved_weight, o_offsets, SPEC)

    with forward_ad.dual_level():
        tangent_tau = torch.zeros_like(tau)
        tangent_rate = torch.zeros_like(rate)
        tangent_weight = torch.zeros_like(weight)
        if variable == "tau_rt":
            tangent_tau = direction.clone()
        elif variable == "tau_rate":
            tangent_rate = direction.clone()
        elif variable == "weight_re":
            tangent_weight = torch.complex(direction, torch.zeros_like(direction))
        else:
            tangent_weight = torch.complex(torch.zeros_like(direction), direction)
        cube = synthesize_cfr_rows(
            forward_ad.make_dual(tau, tangent_tau),
            forward_ad.make_dual(rate, tangent_rate),
            forward_ad.make_dual(weight, tangent_weight),
            offsets,
            SPEC,
        )
        jvp = forward_ad.unpack_dual(cube).tangent
        assert jvp is not None, variable
        jvp = (jvp.cpu() * step).to(torch.complex128)

    expected = 0.5 * (evaluate(1.0) - evaluate(-1.0))
    scale = float(expected.abs().max())
    assert scale > 0.0, variable
    torch.testing.assert_close(
        jvp,
        expected,
        rtol=2e-3,
        atol=2e-3 * scale,
        msg=lambda text: f"{variable}: {text}",
    )


def test_the_rate_derivative_is_not_the_delay_derivative_scaled_by_slow_time():
    """``d/dtau_rate = -2 pi t_l (n df + f_c + f_r)``, not ``d/dtau_rt * t_l``.

    ``carrier_rate_hz`` multiplies the DRIFT, which depends on ``tau_rate`` but
    not on ``tau_rt``, so the rate derivative carries an extra
    ``-2 pi t_l f_r`` that the base-delay derivative does not. At the production
    placement, where ``f_r = f_ref = 77 GHz`` and ``n df`` reaches 7.56 MHz, the
    naive product understates the rate derivative by a factor of about ten
    thousand - and the primal is completely unaffected, so nothing else would
    show it.

    Measured directly on one grid point: a pure ``tau_rate`` tangent rotates the
    phasor, so ``tangent / H`` is exactly ``j * dphi/dtau_rate``.
    """

    single = torch.tensor([2.4683743e-8], dtype=torch.float32, device="cuda")
    rate = torch.tensor([8.0055e-8], dtype=torch.float32, device="cuda")
    weight = torch.ones(1, dtype=torch.complex64, device="cuda")
    offsets = torch.tensor([0, 1], dtype=torch.int64, device="cuda")

    with forward_ad.dual_level():
        dual_rate = forward_ad.make_dual(rate, torch.ones_like(rate))
        cube = synthesize_cfr_rows(single, dual_rate, weight, offsets, SPEC)
        primal, tangent = forward_ad.unpack_dual(cube)
        ratio = (tangent.to(torch.complex128) / primal.to(torch.complex128)).cpu()

    symbol = SPEC.num_symbols - 1
    subcarrier = NUM_SUBCARRIERS - 1
    t_l = symbol * SPEC.symbol_period_s
    f_sub = subcarrier * SPEC.subcarrier_spacing_hz

    measured = float(ratio[symbol, 0, subcarrier].imag)
    analytic = -2.0 * math.pi * t_l * (f_sub + SPEC.carrier_hz + SPEC.carrier_rate_hz)
    naive = -2.0 * math.pi * t_l * (f_sub + SPEC.carrier_hz)

    assert measured == pytest.approx(analytic, rel=1e-4)
    assert abs(analytic / naive) == pytest.approx(
        1.0 + F_REF_HZ / f_sub, rel=1e-9
    )
    assert abs(analytic / naive) > 1000.0
    # The real part is zero: a pure rate tangent is a rotation and nothing else.
    assert abs(float(ratio[symbol, 0, subcarrier].real)) < 1e-6 * abs(measured)


def test_a_forward_only_dual_is_not_dropped_at_the_facade():
    """Guards against an eager ``requires_grad`` shortcut.

    An ADR-038 forward-only dual has ``requires_grad == False``. A facade that
    checked ``requires_grad`` before deciding whether to use autograd would
    return a plain tensor here and the Doppler tangent would vanish silently.
    """

    tau, rate, weight, offsets = _cuda_inputs()
    assert not tau.requires_grad
    with forward_ad.dual_level():
        dual = forward_ad.make_dual(tau, torch.ones_like(tau) * 1e-12)
        assert not dual.requires_grad
        cube = synthesize_cfr_rows(dual, rate, weight, offsets, SPEC)
        assert forward_ad.unpack_dual(cube).tangent is not None


def test_gradcheck_corroborates_both_modes():
    """Corroboration only, at tiny size and with stated float32 tolerances."""

    spec = OfdmCfrSpec(
        num_subcarriers=4,
        num_symbols=2,
        subcarrier_spacing_hz=DF_HZ,
        cyclic_prefix_s=2.0e-6,
        reference_frequency_hz=F_REF_HZ,
        max_expected_delay_s=1.0e-6,
        carrier_hz=0.0,
        carrier_rate_hz=F_REF_HZ,
    )
    offsets = torch.tensor([0, 2], dtype=torch.int64, device="cuda")
    rate = torch.zeros(2, dtype=torch.float32, device="cuda")
    tau = torch.tensor([1.0e-8, 2.0e-8], dtype=torch.float32, device="cuda")
    weight = torch.tensor(
        [0.6 - 0.3j, -0.2 + 0.45j], dtype=torch.complex64, device="cuda"
    ).requires_grad_(True)

    def run(w):
        return synthesize_cfr_rows(tau, rate, w, offsets, spec)

    # Only the weight is checked: it enters linearly, so a float32 central
    # difference is well conditioned. tau enters through a phase of order 1e7
    # rad/s and has no usable float32 perturbation scale; its derivative is
    # covered against the float64 oracle above, which is the stronger check.
    assert torch.autograd.gradcheck(
        run, (weight,), eps=1e-3, atol=2e-2, rtol=2e-2, nondet_tol=1e-5
    )
    assert torch.autograd.gradcheck(
        run,
        (weight,),
        eps=1e-3,
        atol=2e-2,
        rtol=2e-2,
        nondet_tol=1e-5,
        check_forward_ad=True,
        check_backward_ad=False,
        check_undefined_grad=False,
        check_batched_grad=False,
    )
