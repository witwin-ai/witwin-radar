"""Operator-level AD for the native FMCW beat primitive.

The oracle is the float64 pure-Torch chain in ``tests/support/reference_chain``,
which is FD-validated in float64 first. ``gradcheck`` on the float32 kernel runs
too, but only as corroboration with explicitly loose tolerances.
"""

from __future__ import annotations

import pytest
import torch
import torch.autograd.forward_ad as forward_ad

from support import fd  # noqa: E402
from support import phase4_geometry as geo  # noqa: E402
from support import reference_chain as ref  # noqa: E402
from witwin.radar.synthesis.contracts import FmcwBeatSpec  # noqa: E402
from witwin.radar.synthesis.fmcw_beat import synthesize_beat_rows  # noqa: E402


pytestmark = pytest.mark.gpu


# The PRODUCTION carrier placement: the absolute carrier phase lives in the
# Channel weight, and carrier_rate_hz supplies the intra-frame Doppler term the
# frozen weight cannot carry. Deriving the operator AD against this setting is
# deliberate - carrier_rate_hz makes d(phi)/d(tau_rate) differ from
# d(phi)/d(tau_rt) * t_c, and a spec with it zeroed would never exercise that.
SPEC = FmcwBeatSpec(
    num_samples=32,
    num_chirps=3,
    sample_period_s=1.0 / 4.4e6,
    chirp_period_s=65.0e-6,
    slope_hz_per_s=60.012e12,
    t_start_s=6.0e-6,
    carrier_hz=0.0,
    carrier_rate_hz=geo.REFERENCE_FREQUENCY_HZ,
)

DELAYS = (geo.round_trip_delay_s(), 2.4e-8)
RATES = (2.385e-8, -1.1e-8)
WEIGHTS = (0.6 - 0.3j, -0.2 + 0.45j)
OFFSETS = (0, 2)


def _cpu_inputs():
    return (
        torch.tensor(DELAYS, dtype=torch.float64),
        torch.tensor(RATES, dtype=torch.float64),
        torch.tensor(WEIGHTS, dtype=torch.complex128),
        torch.tensor(OFFSETS + (len(DELAYS),), dtype=torch.int64)[
            [0, 2]
        ],  # one segment holding both rows
    )


def _cuda_inputs():
    tau = torch.tensor(DELAYS, dtype=torch.float32, device="cuda")
    rate = torch.tensor(RATES, dtype=torch.float32, device="cuda")
    weight = torch.tensor(WEIGHTS, dtype=torch.complex64, device="cuda")
    offsets = torch.tensor([0, len(DELAYS)], dtype=torch.int64, device="cuda")
    return tau, rate, weight, offsets


def _reference_loss(tau, rate, weight, offsets, target):
    iq = ref.beat_samples(tau, rate, weight, offsets, SPEC)
    return ref.radar_loss(iq, target)


@pytest.fixture(scope="module")
def target_iq():
    torch.manual_seed(20260724)
    return torch.randn(
        (SPEC.num_chirps, 1, SPEC.num_samples), dtype=torch.complex128
    )


def _production_loss(tau, rate, weight, offsets, target):
    iq = synthesize_beat_rows(tau, rate, weight, offsets, SPEC)
    return ref.radar_loss(iq.cpu(), target)


def test_oracle_gradients_agree_with_float64_finite_differences(target_iq):
    """Validate the oracle before anything is compared against it."""

    tau, rate, weight, offsets = _cpu_inputs()
    tau = tau.clone().requires_grad_(True)
    rate = rate.clone().requires_grad_(True)
    weight = weight.clone().requires_grad_(True)
    _reference_loss(tau, rate, weight, offsets, target_iq).backward()

    for index in range(len(DELAYS)):
        measured = fd.central_difference(
            lambda value: _reference_loss(
                value, rate.detach(), weight.detach(), offsets, target_iq
            ),
            tau.detach(),
            index,
            1e-14,
        )
        assert fd.relative_error(
            measured, float(tau.grad[index]), floor=1e-6
        ) < 1e-4

        measured_rate = fd.central_difference(
            lambda value: _reference_loss(
                tau.detach(), value, weight.detach(), offsets, target_iq
            ),
            rate.detach(),
            index,
            1e-14,
        )
        assert fd.relative_error(
            measured_rate, float(rate.grad[index]), floor=1e-6
        ) < 1e-4

        measured_re = fd.central_difference(
            lambda value: _reference_loss(
                tau.detach(), rate.detach(), value, offsets, target_iq
            ),
            weight.detach(),
            index,
            1e-6,
        )
        # Torch's complex autograd convention: .grad holds the conjugate
        # Wirtinger derivative, so d(loss)/d(Re w) is +Re(grad).
        assert fd.relative_error(
            measured_re, float(weight.grad[index].real), floor=1e-9
        ) < 1e-5


def test_native_vjp_matches_the_oracle(target_iq):
    tau, rate, weight, offsets = _cuda_inputs()
    tau = tau.clone().requires_grad_(True)
    rate = rate.clone().requires_grad_(True)
    weight = weight.clone().requires_grad_(True)
    _production_loss(tau, rate, weight, offsets, target_iq).backward()

    o_tau, o_rate, o_weight, o_offsets = _cpu_inputs()
    o_tau = o_tau.clone().requires_grad_(True)
    o_rate = o_rate.clone().requires_grad_(True)
    o_weight = o_weight.clone().requires_grad_(True)
    _reference_loss(o_tau, o_rate, o_weight, o_offsets, target_iq).backward()

    for index in range(len(DELAYS)):
        assert fd.relative_error(
            float(tau.grad[index]), float(o_tau.grad[index]), floor=1e-6
        ) < 1e-3
        assert fd.relative_error(
            float(rate.grad[index]), float(o_rate.grad[index]), floor=1e-6
        ) < 1e-3
        assert fd.relative_error(
            float(weight.grad[index].real),
            float(o_weight.grad[index].real),
            floor=1e-9,
        ) < 1e-3
        assert fd.relative_error(
            float(weight.grad[index].imag),
            float(o_weight.grad[index].imag),
            floor=1e-9,
        ) < 1e-3


MULTI_DELAYS = (geo.round_trip_delay_s(), 2.4e-8, 1.7e-8, 3.1e-8, 9.0e-9)
MULTI_RATES = (2.385e-8, -1.1e-8, 4.0e-9, -2.2e-8, 7.5e-9)
MULTI_WEIGHTS = (0.6 - 0.3j, -0.2 + 0.45j, 0.15 + 0.8j, -0.5 - 0.1j, 0.33 + 0.22j)
# Three segments with different row counts. Row 2 sits exactly on a boundary,
# which is the index where the half-open partition is decided.
MULTI_OFFSETS = (0, 2, 5, 5)


@pytest.fixture(scope="module")
def multi_target_iq():
    torch.manual_seed(20260725)
    return torch.randn(
        (SPEC.num_chirps, len(MULTI_OFFSETS) - 1, SPEC.num_samples),
        dtype=torch.complex128,
    )


def test_multi_segment_vjp_matches_the_oracle(multi_target_iq):
    """The backward kernel's per-path segment mapping, under gradient.

    ``_segment_of_each_path`` feeds ONLY the backward kernel; forward and JVP
    read ``path_offsets`` directly. Every other gradient test in this suite is
    single-segment, where the mapping is the constant zero and cannot be wrong.
    That left the one deviation the implementation actually made -
    ``bucketize(..., right=True)`` - verified by hand and by nothing else: the
    mutation ``right=True -> right=False`` passed all 32 kernel, AD, composition
    and end-to-end tests.

    Three segments of unequal size, including an EMPTY trailing segment and a
    row whose index equals a boundary, which is precisely where ``right``
    changes the answer.
    """

    tau = torch.tensor(MULTI_DELAYS, dtype=torch.float32, device="cuda")
    rate = torch.tensor(MULTI_RATES, dtype=torch.float32, device="cuda")
    weight = torch.tensor(MULTI_WEIGHTS, dtype=torch.complex64, device="cuda")
    offsets = torch.tensor(MULTI_OFFSETS, dtype=torch.int64, device="cuda")

    tau = tau.clone().requires_grad_(True)
    rate = rate.clone().requires_grad_(True)
    weight = weight.clone().requires_grad_(True)
    iq = synthesize_beat_rows(tau, rate, weight, offsets, SPEC)
    ref.radar_loss(iq.cpu(), multi_target_iq).backward()

    o_tau = torch.tensor(MULTI_DELAYS, dtype=torch.float64).requires_grad_(True)
    o_rate = torch.tensor(MULTI_RATES, dtype=torch.float64).requires_grad_(True)
    o_weight = torch.tensor(
        MULTI_WEIGHTS, dtype=torch.complex128
    ).requires_grad_(True)
    o_offsets = torch.tensor(MULTI_OFFSETS, dtype=torch.int64)
    o_iq = ref.beat_samples(o_tau, o_rate, o_weight, o_offsets, SPEC)
    ref.radar_loss(o_iq, multi_target_iq).backward()

    for index in range(len(MULTI_DELAYS)):
        assert fd.relative_error(
            float(tau.grad[index]), float(o_tau.grad[index]), floor=1e-6
        ) < 1e-3, index
        assert fd.relative_error(
            float(rate.grad[index]), float(o_rate.grad[index]), floor=1e-6
        ) < 1e-3, index
        assert fd.relative_error(
            float(weight.grad[index].real),
            float(o_weight.grad[index].real),
            floor=1e-9,
        ) < 1e-3, index
        assert fd.relative_error(
            float(weight.grad[index].imag),
            float(o_weight.grad[index].imag),
            floor=1e-9,
        ) < 1e-3, index

    # Every row must actually be carrying signal, or the comparison above is
    # satisfied by zeros on both sides.
    assert float(tau.grad.abs().min()) > 1e-6
    assert float(weight.grad.abs().min()) > 1e-9


def test_the_segment_mapping_is_the_half_open_partition():
    """The mapping itself, stated directly rather than only through gradients.

    An offsets table is a half-open partition ``[start, end)``, so a row whose
    index equals a boundary belongs to the NEXT segment. ``right=False`` puts it
    in the previous one.
    """

    from witwin.radar.synthesis.fmcw_beat import _segment_of_each_path

    offsets = torch.tensor(MULTI_OFFSETS, dtype=torch.int64, device="cuda")
    mapping = _segment_of_each_path(offsets, len(MULTI_DELAYS))
    assert mapping.tolist() == [0, 0, 1, 1, 1]
    # An empty trailing segment claims no rows.
    assert 2 not in mapping.tolist()


def test_native_jvp_matches_the_oracle(target_iq):
    tau, rate, weight, offsets = _cuda_inputs()
    d_tau = torch.tensor([1.0e-9, -3.0e-10], dtype=torch.float32, device="cuda")
    d_rate = torch.tensor([2.0e-10, 5.0e-11], dtype=torch.float32, device="cuda")

    with forward_ad.dual_level():
        dual_tau = forward_ad.make_dual(tau, d_tau)
        dual_rate = forward_ad.make_dual(rate, d_rate)
        iq = synthesize_beat_rows(dual_tau, dual_rate, weight, offsets, SPEC)
        loss = ref.radar_loss(iq.cpu(), target_iq)
        tangent = forward_ad.unpack_dual(loss).tangent
        assert tangent is not None, "the native jvp companion was not reached"
        measured = float(tangent)

    o_tau, o_rate, o_weight, o_offsets = _cpu_inputs()
    o_d_tau = d_tau.double().cpu()
    o_d_rate = d_rate.double().cpu()

    # The oracle's own forward-mode derivative, in float64.
    with forward_ad.dual_level():
        oracle_tangent = forward_ad.unpack_dual(
            _reference_loss(
                forward_ad.make_dual(o_tau, o_d_tau),
                forward_ad.make_dual(o_rate, o_d_rate),
                o_weight,
                o_offsets,
                target_iq,
            )
        ).tangent
    oracle = float(oracle_tangent)

    # FD validates the oracle, and the step is scaled DOWN from the tangent:
    # a unit step along a 1e-9 s delay tangent swings the beat phase by about
    # 2.3 rad, so the difference quotient there is not a derivative at all.
    directional = fd.directional_derivative(
        lambda t, r: _reference_loss(t, r, o_weight, o_offsets, target_iq),
        (o_tau, o_rate),
        (o_d_tau, o_d_rate),
        1e-4,
    )
    assert fd.relative_error(directional, oracle, floor=1e-9) < 1e-4

    assert fd.relative_error(measured, oracle, floor=1e-9) < 1e-3


def test_a_forward_only_dual_is_not_dropped_at_the_facade():
    """Guards against re-introducing the eager ``requires_grad`` shortcut.

    An ADR-038 forward-only dual has ``requires_grad == False``. A facade that
    checks ``requires_grad`` before deciding whether to use autograd would
    return a plain tensor here, and the Doppler tangent would vanish silently.
    """

    tau, rate, weight, offsets = _cuda_inputs()
    assert not tau.requires_grad
    with forward_ad.dual_level():
        dual = forward_ad.make_dual(
            tau, torch.ones_like(tau) * 1e-9
        )
        assert not dual.requires_grad
        iq = synthesize_beat_rows(dual, rate, weight, offsets, SPEC)
        assert forward_ad.unpack_dual(iq).tangent is not None


def test_gradcheck_corroborates_both_modes():
    """Corroboration only, at tiny size and with stated float32 tolerances."""

    spec = FmcwBeatSpec(
        num_samples=4,
        num_chirps=2,
        sample_period_s=1.0 / 4.4e6,
        chirp_period_s=65.0e-6,
        slope_hz_per_s=60.012e12,
        t_start_s=0.0,
        carrier_hz=0.0,
    )
    offsets = torch.tensor([0, 2], dtype=torch.int64, device="cuda")
    rate = torch.zeros(2, dtype=torch.float32, device="cuda")
    tau = torch.tensor([1.0e-8, 2.0e-8], dtype=torch.float32, device="cuda")
    weight = torch.tensor(
        [0.6 - 0.3j, -0.2 + 0.45j], dtype=torch.complex64, device="cuda"
    ).requires_grad_(True)

    def run(w):
        return synthesize_beat_rows(tau, rate, w, offsets, spec)

    # Only the weight is checked: it enters linearly, so a float32 central
    # difference is well conditioned. tau enters through a phase of order 1e10
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
