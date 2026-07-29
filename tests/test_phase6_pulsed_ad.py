"""Operator-level AD for the native pulsed echo primitive.

The oracle is the float64 pure-Torch train in ``tests/support/reference_pulsed``,
which is FD-validated in float64 FIRST. ``gradcheck`` on the float32 kernel runs
too, but only as corroboration with explicitly loose tolerances: the production
train is float32 and the LFM's own phase reaches a hundred cycles, so a naive
central difference on the production chain subtracts two nearly equal float32
numbers and can return an exactly zero derivative that looks like a real answer.

Every finite difference in this file is a TEST ORACLE. Production derivatives are
the registered ``pulsed_echo_backward`` and ``pulsed_echo_jvp`` companions and
nothing else.

**The envelope is where this waveform's AD differs from the other two.** The
geometry enters ``p(t_g + m T_s - tau)`` through a support test as well as
through a phase, and a support test is not differentiable at its two edges. The
kernel returns the ALMOST-EVERYWHERE derivative: exactly zero envelope gradient,
never a delta. Two consequences, both tested rather than assumed:

* every finite difference here must keep each sample on the same side of both
  edges, and each test asserts its step against the measured clearance rather
  than hoping;
* with a rectangular pulse AND the production carrier placement, the a.e.
  derivative with respect to ``tau_rt`` is EXACTLY ZERO, because a rectangle has
  no phase for the delay to move. That is a real property of the model, not a
  defect, and it is why the analytic AD tests use the LFM.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.autograd.forward_ad as forward_ad

from support import fd  # noqa: E402
from support import reference_pulsed as ref  # noqa: E402
from support.pulsed_grid import (  # noqa: E402
    F_REF_HZ,
    SAMPLE_PERIOD_S,
    reference_spec,
    rect_spec,
)
from witwin.radar.synthesis.pulsed import synthesize_echo_rows  # noqa: E402


pytestmark = pytest.mark.gpu


# A small grid: the backward kernel gives one thread per PATH and loops the whole
# (pulse, sample) product, so the reference grid's 32 x 1024 would make a
# two-row gradient test spend 32768 iterations in two threads for no extra
# coverage. Every structural feature that matters - live drift, two segments, a
# pulse fully inside the gate - survives the shrink.
#
# The PRODUCTION carrier placement: the absolute carrier phase lives in the
# Channel weight and carrier_rate_hz supplies the inter-pulse Doppler term the
# frozen weight cannot carry. Deriving the operator AD against this setting is
# deliberate - carrier_rate_hz is exactly what makes d(phi)/d(tau_rate) differ
# from d(phi)/d(tau_rt) * t_l, and a spec with it zeroed would never exercise it.
SPEC = reference_spec(
    num_pulses=4,
    num_samples=256,
    pri_s=20.0e-6,
    pulse_width_s=2.0e-6,
    max_expected_delay_rate=1.0e-7,
)

RECT_SPEC = rect_spec(
    num_pulses=4,
    num_samples=256,
    pri_s=20.0e-6,
    pulse_width_s=2.0e-6,
    max_expected_delay_rate=1.0e-7,
)

# Both pulses land entirely inside the 5.12 us gate, and neither sits on an
# envelope edge: the measured clearance is 0.3 ns, three orders of magnitude
# above every finite-difference step below.
DELAYS = (1.0003e-6, 2.5007e-6)
RATES = (8.0055e-8, -3.1e-8)
WEIGHTS = (0.6 - 0.3j, -0.2 + 0.45j)

# Steps, chosen per variable and recorded. The scale each one has to clear is
# the phase swing it produces:
#   tau_rt    d(phi)/d(tau)  = 2 pi B u / T_p        up to 1.3e8 rad/s
#   tau_rate  d(phi)/d(rate) = 2 pi t_l (f_ref + B u / T_p)  up to 2.9e7 rad
#   weight    enters linearly
# and the ceiling every geometry step has to stay under is the 0.3 ns envelope
# clearance, which is 3e-10 s.
STEP_TAU_S = 1.0e-13
STEP_RATE = 1.0e-13
STEP_WEIGHT = 1.0e-6


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
    return ref.echo_loss(ref.echo_cube(tau, rate, weight, offsets, spec), target)


def _production_loss(tau, rate, weight, offsets, target, spec=SPEC):
    cube = synthesize_echo_rows(tau, rate, weight, offsets, spec)
    return ref.echo_loss(cube.cpu(), target)


def _imaginary_central_difference(evaluate, value, index, step):
    """``d(loss)/d(Im w)`` by a central difference along the imaginary axis.

    ``support.fd.central_difference`` divides by the step, and a purely
    imaginary step would make the quotient complex; the directional derivative
    along ``i`` is the real quotient over the REAL step length.
    """

    plus = value.clone()
    minus = value.clone()
    plus[index] = plus[index] + 1j * step
    minus[index] = minus[index] - 1j * step
    return float((evaluate(plus) - evaluate(minus)) / (2.0 * step))


@pytest.fixture(scope="module")
def target_cube():
    torch.manual_seed(20260725)
    return torch.randn(
        (SPEC.num_pulses, 1, SPEC.num_samples), dtype=torch.complex128
    )


# --------------------------------------------------------------------------
# The envelope clearance, which every finite difference below depends on
# --------------------------------------------------------------------------


def test_no_sample_sits_near_a_pulse_edge_at_the_operating_point():
    """The precondition for every finite difference in this file.

    A step that moves a sample across ``u = 0`` or ``u = T_p`` measures the
    envelope switching on or off, which is a real discontinuity, and disagrees
    with the almost-everywhere derivative that both the kernel and the oracle
    return. Asserted rather than hoped for, because the failure mode is a
    plausible-looking finite-difference mismatch that reads as a broken kernel.
    """

    tau, rate, _, _ = _cpu_inputs()
    clearance = ref.envelope_clearance_s(tau, rate, SPEC)
    assert clearance == pytest.approx(3.0e-10, rel=0.2)
    assert clearance > 1000.0 * STEP_TAU_S
    assert clearance > 1000.0 * STEP_RATE * SPEC.pri_s * SPEC.num_pulses


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
# T3.11  VJP
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


MULTI_DELAYS = (1.0003e-6, 2.5007e-6, 1.7003e-6, 0.9007e-6, 2.1005e-6)
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
        (SPEC.num_pulses, len(MULTI_OFFSETS) - 1, SPEC.num_samples),
        dtype=torch.complex128,
    )


def test_multi_segment_vjp_matches_the_oracle(multi_target_cube):
    """The backward kernel's per-path segment mapping, under gradient.

    ``segment_of_each_row`` feeds ONLY the backward kernel; forward and JVP read
    ``pair_offsets`` directly. A single-segment gradient test cannot see it,
    because there the mapping is the constant zero and cannot be wrong.
    """

    o_tau = torch.tensor(MULTI_DELAYS, dtype=torch.float64)
    o_rate = torch.tensor(MULTI_RATES, dtype=torch.float64)
    assert ref.envelope_clearance_s(o_tau, o_rate, SPEC) > 1000.0 * STEP_TAU_S

    tau = torch.tensor(MULTI_DELAYS, dtype=torch.float32, device="cuda")
    rate = torch.tensor(MULTI_RATES, dtype=torch.float32, device="cuda")
    weight = torch.tensor(MULTI_WEIGHTS, dtype=torch.complex64, device="cuda")
    offsets = torch.tensor(MULTI_OFFSETS, dtype=torch.int64, device="cuda")

    tau = tau.clone().requires_grad_(True)
    rate = rate.clone().requires_grad_(True)
    weight = weight.clone().requires_grad_(True)
    cube = synthesize_echo_rows(tau, rate, weight, offsets, SPEC)
    ref.echo_loss(cube.cpu(), multi_target_cube).backward()

    o_tau = o_tau.requires_grad_(True)
    o_rate = o_rate.requires_grad_(True)
    o_weight = torch.tensor(MULTI_WEIGHTS, dtype=torch.complex128).requires_grad_(True)
    o_offsets = torch.tensor(MULTI_OFFSETS, dtype=torch.int64)
    o_cube = ref.echo_cube(o_tau, o_rate, o_weight, o_offsets, SPEC)
    ref.echo_loss(o_cube, multi_target_cube).backward()

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


# --------------------------------------------------------------------------
# T3.11  JVP
# --------------------------------------------------------------------------


def test_native_jvp_matches_the_oracle_and_a_float64_finite_difference(target_cube):
    tau, rate, weight, offsets = _cuda_inputs()
    d_tau = torch.tensor([1.0e-9, -3.0e-10], dtype=torch.float32, device="cuda")
    d_rate = torch.tensor([2.0e-10, 5.0e-11], dtype=torch.float32, device="cuda")

    with forward_ad.dual_level():
        dual_tau = forward_ad.make_dual(tau, d_tau)
        dual_rate = forward_ad.make_dual(rate, d_rate)
        cube = synthesize_echo_rows(dual_tau, dual_rate, weight, offsets, SPEC)
        loss = ref.echo_loss(cube.cpu(), target_cube)
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

    # FD validates the oracle. The step is scaled DOWN from the tangent so that
    # the realised geometry step, 1e-4 * 1e-9 s, stays three orders of magnitude
    # below the 3e-10 s envelope clearance.
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
    """All four inputs, on the TRAIN rather than through a scalar loss.

    A loss collapses four gradients into one number and can hide a sign error in
    one of them behind the others. This compares the whole complex train.

    The finite difference is taken on the FLOAT64 ORACLE at the same
    float32-rounded operating point, never on the production kernel. Differencing
    the production train would subtract two nearly equal float32 numbers: at a
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
        return ref.echo_cube(moved_tau, moved_rate, moved_weight, o_offsets, SPEC)

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
        cube = synthesize_echo_rows(
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
    """``d/dtau_rate = -2 pi t_l (f_c + f_r + B u / T_p)``, not ``d/dtau_rt * t_l``.

    ``carrier_rate_hz`` multiplies the DRIFT, which depends on ``tau_rate`` but
    not on ``tau_rt``, so the rate derivative carries an extra
    ``-2 pi t_l f_r`` that the base-delay derivative does not. At the production
    placement, where ``f_r = f_ref = 77 GHz`` and ``B u / T_p`` reaches 20 MHz,
    the naive product understates the rate derivative by a factor of about four
    thousand - and the primal is completely unaffected, so nothing else would
    show it.

    Measured directly on one grid point: a pure ``tau_rate`` tangent rotates the
    phasor, so ``tangent / y`` is exactly ``j * dphi/dtau_rate``.
    """

    single = torch.tensor([DELAYS[0]], dtype=torch.float32, device="cuda")
    rate = torch.tensor([RATES[0]], dtype=torch.float32, device="cuda")
    weight = torch.ones(1, dtype=torch.complex64, device="cuda")
    offsets = torch.tensor([0, 1], dtype=torch.int64, device="cuda")

    with forward_ad.dual_level():
        dual_rate = forward_ad.make_dual(rate, torch.ones_like(rate))
        cube = synthesize_echo_rows(single, dual_rate, weight, offsets, SPEC)
        primal, tangent = forward_ad.unpack_dual(cube)
        ratio = (tangent.to(torch.complex128) / primal.to(torch.complex128)).cpu()

    pulse = SPEC.num_pulses - 1
    t_l = pulse * SPEC.pri_s
    stored_tau = float(single.cpu())
    # A sample near the trailing edge of the pulse, where the LFM's own
    # contribution is largest and the two forms differ least - so the assertion
    # is made where it is hardest to pass.
    sample = int((stored_tau + 0.9 * SPEC.pulse_width_s) / SAMPLE_PERIOD_S)
    envelope_time = sample * SAMPLE_PERIOD_S - (stored_tau + rate.item() * t_l)
    assert 0.0 < envelope_time < SPEC.pulse_width_s
    chirp_hz = SPEC.instantaneous_pulse_frequency_hz(envelope_time)

    measured = float(ratio[pulse, 0, sample].imag)
    analytic = -math.tau * t_l * (SPEC.carrier_hz + SPEC.carrier_rate_hz + chirp_hz)
    naive = -math.tau * t_l * (SPEC.carrier_hz + chirp_hz)

    assert measured == pytest.approx(analytic, rel=1e-4)
    assert abs(analytic / naive) == pytest.approx(1.0 + F_REF_HZ / chirp_hz, rel=1e-6)
    assert abs(analytic / naive) > 1000.0
    # The real part is zero: a pure rate tangent is a rotation and nothing else.
    assert abs(float(ratio[pulse, 0, sample].real)) < 1e-6 * abs(measured)


# --------------------------------------------------------------------------
# T3.11  the rectangular envelope's edge, stated as a property
# --------------------------------------------------------------------------


def test_the_rectangular_envelope_has_exactly_zero_delay_gradient():
    """The a.e. derivative, and it is not an approximation.

    With a rectangular pulse and the production carrier placement, the train's
    dependence on ``tau_rt`` is ENTIRELY through the support test: a rectangle
    has no phase for the delay to move, and the frozen weight already owns the
    carrier. The almost-everywhere derivative is therefore exactly zero, and the
    kernel returns exactly zero rather than a delta at the two edges.

    This is a real property of the model, not a defect, and it is the reason the
    analytic AD tests above use the LFM. It also says something a user needs to
    know: a rectangular pulse gives gradient-based range estimation nothing to
    descend on, and the fix is a pulse with phase, not a smoothed envelope.
    """

    tau = torch.tensor(DELAYS, dtype=torch.float32, device="cuda").requires_grad_(True)
    rate = torch.tensor(RATES, dtype=torch.float32, device="cuda").requires_grad_(True)
    weight = torch.tensor(
        WEIGHTS, dtype=torch.complex64, device="cuda"
    ).requires_grad_(True)
    offsets = torch.tensor([0, len(DELAYS)], dtype=torch.int64, device="cuda")

    cube = synthesize_echo_rows(tau, rate, weight, offsets, RECT_SPEC)
    (cube.real.sum() + cube.imag.sum()).backward()

    assert float(tau.grad.abs().max()) == 0.0
    # The rate still moves the CARRIER, so its gradient is not zero, and the
    # weight enters linearly. Only the base delay vanishes.
    assert float(rate.grad.abs().min()) > 0.0
    assert float(weight.grad.abs().min()) > 0.0

    # The same run with an LFM pulse has a nonzero delay gradient, which is what
    # makes the assertion above a statement about the ENVELOPE rather than about
    # a dead gradient path.
    tau_lfm = torch.tensor(
        DELAYS, dtype=torch.float32, device="cuda"
    ).requires_grad_(True)
    lfm = synthesize_echo_rows(
        tau_lfm,
        torch.tensor(RATES, dtype=torch.float32, device="cuda"),
        torch.tensor(WEIGHTS, dtype=torch.complex64, device="cuda"),
        offsets,
        SPEC,
    )
    (lfm.real.sum() + lfm.imag.sum()).backward()
    assert float(tau_lfm.grad.abs().min()) > 0.0


def test_a_finite_difference_across_a_rectangular_edge_disagrees_and_should():
    """Documented, not worked around: the FD oracle must avoid the edge.

    At a delay that puts a sample exactly on ``u = 0`` the support is half-open,
    so that sample is INSIDE. Moving the delay by a positive quarter-sample drops
    it out and moving by a negative quarter-sample keeps it, so a central
    difference of THAT SAMPLE straddles a jump: the quotient is the envelope's
    whole height divided by the step, and it diverges as the step shrinks rather
    than converging to anything.

    The a.e. derivative is exactly zero and both the kernel and the oracle
    return it. The disagreement below is correct behaviour, and asserting it
    here is what stops a later reader from "fixing" the kernel to match a finite
    difference that never had a limit.

    Note what is NOT discontinuous: the total energy. The half-open support keeps
    the sampled pulse exactly ``M_p`` samples long at every delay, so a sample
    leaving at one edge is matched by one arriving at the other and the sum of
    ``|y|^2`` does not move at all. The discontinuity is per sample, and a test
    written on the energy would have found nothing and concluded the edge was
    smooth.
    """

    on_grid_tau = 50 * SAMPLE_PERIOD_S
    tau = torch.tensor([on_grid_tau], dtype=torch.float64)
    rate = torch.zeros(1, dtype=torch.float64)
    weight = torch.ones(1, dtype=torch.complex128)
    offsets = torch.tensor([0, 1], dtype=torch.int64)
    assert ref.envelope_clearance_s(tau, rate, RECT_SPEC) == 0.0

    def edge_sample(shift: float) -> float:
        cube = ref.echo_cube(tau + shift, rate, weight, offsets, RECT_SPEC)
        return float(cube[0, 0, 50].abs() ** 2)

    step = 0.25 * SAMPLE_PERIOD_S
    assert edge_sample(-step) == pytest.approx(1.0 / RECT_SPEC.pulse_width_s, rel=1e-12)
    assert edge_sample(step) == 0.0
    quotient = (edge_sample(step) - edge_sample(-step)) / (2.0 * step)
    assert abs(quotient) > 1.0e6

    # It diverges rather than converging: halving the step doubles the quotient.
    halved = (edge_sample(0.5 * step) - edge_sample(-0.5 * step)) / step
    assert halved == pytest.approx(2.0 * quotient, rel=1e-12)

    # The energy, by contrast, does not move at all: the half-open support keeps
    # the sample count constant.
    def energy(shift: float) -> float:
        cube = ref.echo_cube(tau + shift, rate, weight, offsets, RECT_SPEC)
        return float((cube.abs() ** 2).sum())

    assert energy(step) == pytest.approx(energy(-step), rel=1e-15)

    # Away from the edge, the same per-sample finite difference is exactly zero,
    # which is the a.e. derivative the kernel returns.
    off_grid = torch.tensor([on_grid_tau + 0.5 * SAMPLE_PERIOD_S], dtype=torch.float64)

    def off_edge_sample(shift: float) -> float:
        cube = ref.echo_cube(off_grid + shift, rate, weight, offsets, RECT_SPEC)
        return float(cube[0, 0, 60].abs() ** 2)

    tiny = 1.0e-12
    assert (off_edge_sample(tiny) - off_edge_sample(-tiny)) == 0.0


# --------------------------------------------------------------------------
# Dispatch contracts
# --------------------------------------------------------------------------


def test_a_forward_only_dual_is_not_dropped_at_the_facade():
    """Guards against an eager ``requires_grad`` shortcut.

    An ADR-038 forward-only dual has ``requires_grad == False``. A facade that
    checked ``requires_grad`` before deciding whether to use autograd would
    return a plain tensor here and the Doppler tangent would vanish silently.
    """

    tau, rate, weight, offsets = _cuda_inputs()
    assert not tau.requires_grad
    with forward_ad.dual_level():
        dual = forward_ad.make_dual(rate, torch.ones_like(rate) * 1e-12)
        assert not dual.requires_grad
        cube = synthesize_echo_rows(tau, dual, weight, offsets, SPEC)
        assert forward_ad.unpack_dual(cube).tangent is not None


def test_gradcheck_corroborates_both_modes():
    """Corroboration only, at tiny size and with stated float32 tolerances."""

    spec = reference_spec(
        num_pulses=2,
        num_samples=64,
        pri_s=20.0e-6,
        pulse_width_s=0.5e-6,
        max_expected_delay_rate=1.0e-7,
    )
    offsets = torch.tensor([0, 2], dtype=torch.int64, device="cuda")
    rate = torch.zeros(2, dtype=torch.float32, device="cuda")
    tau = torch.tensor([0.2003e-6, 0.5007e-6], dtype=torch.float32, device="cuda")
    weight = torch.tensor(
        [0.6 - 0.3j, -0.2 + 0.45j], dtype=torch.complex64, device="cuda"
    ).requires_grad_(True)

    def run(w):
        return synthesize_echo_rows(tau, rate, w, offsets, spec)

    # Only the weight is checked: it enters linearly, so a float32 central
    # difference is well conditioned. tau enters through a phase of order 1e8
    # rad/s AND through a support test, so it has no usable float32
    # perturbation scale at all; its derivative is covered against the float64
    # oracle above, which is the stronger check.
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
