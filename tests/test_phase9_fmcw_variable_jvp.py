"""FMCW forward mode, per input variable, and the rate derivative in closed form.

The asymmetry this closes: OFDM (``test_phase6_ofdm_ad.py``) and pulsed
(``test_phase6_pulsed_ad.py``) each have a per-variable jvp test on the CUBE and
a closed-form check that the rate derivative is not the delay derivative scaled
by slow time. FMCW - the oldest and most used of the three, and the only one
whose kernel already returned ``grad_tau_rate`` - had neither. Its only forward
test collapses ``tau_rt`` and ``tau_rate`` into one scalar loss, which cannot
tell a sign error in one apart from a sign error in the other.

The structure below mirrors the OFDM file deliberately, so that the three
families read the same and a reader who has understood one has understood all
three. Two things are genuinely different and are asserted rather than copied:

* the beat cube is the ONE waveform published in the conjugate of Channel's
  convention, so its rate derivative is POSITIVE where OFDM's and pulsed's are
  negative;
* the term the naive product drops is ``carrier_rate_hz`` against a fast-time
  chirp frequency ``slope * (t_start - tau + t_m)`` rather than against a
  subcarrier offset, so the understatement is a function of the fast-time
  sample and is smallest at the far end of the sweep. The assertion is made at
  the LAST sample, where it is hardest to pass.

The oracle is the float64 CPU chain in ``tests/support/reference_chain``, which
``test_phase4_fmcw_beat_ad.py`` FD-validates in float64 first. Differencing the
production float32 cube instead would subtract two nearly equal numbers at any
step small enough to be a derivative; that conditioning trap is why the oracle
exists at all.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.autograd.forward_ad as forward_ad
from support import phase4_geometry as geo  # noqa: E402
from support import reference_chain as ref  # noqa: E402

from witwin.radar.synthesis.assembly import FmcwSpec  # noqa: E402
from witwin.radar.synthesis.fmcw import synthesize_fmcw_rows  # noqa: E402

pytestmark = pytest.mark.gpu


F_REF_HZ = geo.REFERENCE_FREQUENCY_HZ
SAMPLE_PERIOD_S = 1.0 / 4.4e6
SLOPE_HZ_PER_S = 60.012e12
T_START_S = 6.0e-6

#: The PRODUCTION carrier placement, matching ``test_phase4_fmcw_beat_ad.py``:
#: the absolute carrier lives in the Channel weight and ``carrier_rate_hz``
#: supplies the intra-frame Doppler the frozen weight cannot carry. A spec with
#: ``carrier_rate_hz = 0`` would make the second test below vacuous, because the
#: naive product it refutes would be correct.
SPEC = FmcwSpec(
    num_samples=32,
    num_chirps=3,
    sample_period_s=SAMPLE_PERIOD_S,
    chirp_period_s=65.0e-6,
    slope_hz_per_s=SLOPE_HZ_PER_S,
    t_start_s=T_START_S,
    reference_frequency_hz=F_REF_HZ,
    carrier_hz=0.0,
    carrier_rate_hz=F_REF_HZ,
    output_domain="beat",
)

DELAYS = (geo.round_trip_delay_s(), 2.4e-8)
RATES = (2.385e-8, -1.1e-8)
WEIGHTS = (0.6 - 0.3j, -0.2 + 0.45j)

#: Steps, chosen per variable and recorded. The scale each one has to clear is
#: the phase swing it produces over the cube:
#:
#:   tau_rt    d(cycles)/d(tau)      = slope * (t_start - tau + t_m)  ~ 8e8 /s
#:   tau_rate  d(cycles)/d(rate)     = t_c * (f_ref + that)           ~ 1.5e7
#:   weight    enters linearly
#:
#: The delay step is a hundredth of the OFDM one because the FMCW argument moves
#: two orders of magnitude faster in ``tau``: 1e-13 s already swings the beat
#: phase by 5e-4 cycles, and 1e-11 would be half a turn.
STEP_TAU_S = 1.0e-13
STEP_RATE = 1.0e-11
STEP_WEIGHT = 1.0e-6


def _cuda_inputs():
    tau = torch.tensor(DELAYS, dtype=torch.float32, device="cuda")
    rate = torch.tensor(RATES, dtype=torch.float32, device="cuda")
    weight = torch.tensor(WEIGHTS, dtype=torch.complex64, device="cuda")
    offsets = torch.tensor([0, len(DELAYS)], dtype=torch.int64, device="cuda")
    return tau, rate, weight, offsets


# --------------------------------------------------------------------------
# 1. The per-variable jvp, on the cube
# --------------------------------------------------------------------------


@pytest.mark.parametrize("variable", ["tau_rt", "tau_rate", "weight_re", "weight_im"])
def test_the_jvp_of_each_differentiable_input_matches_a_central_difference(variable):
    """All four inputs, on the CUBE rather than through a scalar loss.

    A loss collapses four derivatives into one number and can hide a sign error
    in one of them behind the others. This compares the whole complex cube, so a
    variable whose tangent is dropped entirely - which is what a missing entry
    in the kernel's tangent list looks like - fails here and nowhere else.

    The finite difference is taken on the FLOAT64 ORACLE at the same
    float32-rounded operating point, never on the production kernel.
    """

    tau, rate, weight, offsets = _cuda_inputs()
    steps = {"tau_rt": STEP_TAU_S, "tau_rate": STEP_RATE, "weight_re": STEP_WEIGHT, "weight_im": STEP_WEIGHT}
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
        return ref.beat_samples(moved_tau, moved_rate, moved_weight, o_offsets, SPEC)

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
        cube = synthesize_fmcw_rows(
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
    torch.testing.assert_close(jvp, expected, rtol=2e-3, atol=2e-3 * scale, msg=lambda text: f"{variable}: {text}")


# --------------------------------------------------------------------------
# 2. The rate derivative in closed form
# --------------------------------------------------------------------------


def test_the_rate_derivative_is_not_the_delay_derivative_scaled_by_slow_time():
    """``d/dtau_rate = +2 pi t_c (f_c + f_r + S (t_start - tau + t_m))``.

    ``carrier_rate_hz`` multiplies the DRIFT ``tau_rate * t_c``, which depends
    on ``tau_rate`` and not on ``tau_rt``, so the rate derivative carries a
    ``+2 pi t_c f_r`` term that the base-delay derivative does not. At the
    production placement, where ``f_r = f_ref = 77 GHz``, the naive product
    ``d/dtau_rt * t_c`` understates the rate derivative by two orders of
    magnitude - and the PRIMAL is completely unaffected, so no magnitude plot,
    range profile or Doppler map would show it.

    Measured directly on one grid point: a pure ``tau_rate`` tangent rotates the
    phasor, so ``tangent / s`` is exactly ``j dphi/dtau_rate``.

    The sign is POSITIVE here and negative in the OFDM and pulsed families,
    which is the beat cube's single conjugation showing up in the derivative.
    That is asserted rather than absorbed into an ``abs``: a family that lost
    its conjugation would still pass every magnitude test in the tree.
    """

    single = torch.tensor([DELAYS[0]], dtype=torch.float32, device="cuda")
    rate = torch.tensor([RATES[0]], dtype=torch.float32, device="cuda")
    weight = torch.ones(1, dtype=torch.complex64, device="cuda")
    offsets = torch.tensor([0, 1], dtype=torch.int64, device="cuda")

    with forward_ad.dual_level():
        dual_rate = forward_ad.make_dual(rate, torch.ones_like(rate))
        cube = synthesize_fmcw_rows(single, dual_rate, weight, offsets, SPEC)
        primal, tangent = forward_ad.unpack_dual(cube)
        ratio = (tangent.to(torch.complex128) / primal.to(torch.complex128)).cpu()

    chirp = SPEC.num_chirps - 1
    # The LAST fast-time sample: the chirp frequency is largest there, so the
    # dropped carrier term is the smallest fraction of the total and the
    # assertion is made where it is hardest to pass.
    sample = SPEC.num_samples - 1
    t_c = chirp * SPEC.chirp_period_s * SPEC.num_tx
    t_m = sample * SPEC.sample_period_s
    stored_tau = float(single.cpu()) + float(rate.cpu()) * t_c
    chirp_hz = SPEC.slope_hz_per_s * (SPEC.t_start_s - stored_tau + t_m)

    measured = float(ratio[chirp, 0, sample].imag)
    analytic = math.tau * t_c * (SPEC.carrier_hz + SPEC.carrier_rate_hz + chirp_hz)
    naive = math.tau * t_c * (SPEC.carrier_hz + chirp_hz)

    assert measured == pytest.approx(analytic, rel=1e-4)
    assert analytic > 0.0
    assert analytic / naive == pytest.approx(1.0 + F_REF_HZ / chirp_hz, rel=1e-9)
    assert analytic / naive > 90.0
    # The real part is zero: a pure rate tangent is a rotation and nothing else.
    assert abs(float(ratio[chirp, 0, sample].real)) < 1e-6 * abs(measured)


def test_the_dropped_term_is_smallest_at_the_far_end_of_the_sweep():
    """Why the assertion above is made at the last sample and not the first.

    The understatement factor is ``1 + f_ref / (S (t_start - tau + t_m))`` and
    ``t_m`` runs from 0 to ``(M - 1) T_s``, so the fast-time sample at the START
    of the gate is where the naive product is worst and the LAST sample is where
    it is least wrong. Asserting at the easy end would be a weaker test that
    looked identical; recording the span is what shows the choice was made.

    Measured on this spec: 215.3 at the first sample and 99.5 at the last, which
    is the ``21x to 215x`` span ``FmcwSpec``'s own docstring quotes - so
    this also pins that documented number against the arithmetic rather than
    leaving it as prose.
    """

    t_c = (SPEC.num_chirps - 1) * SPEC.chirp_period_s * SPEC.num_tx
    tau = DELAYS[0] + RATES[0] * t_c

    def factor(sample: int) -> float:
        chirp_hz = SPEC.slope_hz_per_s * (SPEC.t_start_s - tau + sample * SPEC.sample_period_s)
        return 1.0 + F_REF_HZ / chirp_hz

    first = factor(0)
    last = factor(SPEC.num_samples - 1)
    assert first > last > 1.0
    assert first == pytest.approx(215.3, rel=1e-3)
    assert last == pytest.approx(99.46, rel=1e-3)


# --------------------------------------------------------------------------
# 3. The rate tangent is genuinely reaching the kernel
# --------------------------------------------------------------------------


def test_a_rate_only_tangent_is_not_the_zero_tangent():
    """The falsifier for the whole file.

    A kernel that ignored ``tan_tau_rate`` entirely would publish an exactly
    zero tangent here, and both tests above would then be comparing zero against
    a central difference of a function that barely moves. Asserting the tangent
    is a substantial fraction of the primal's own magnitude is what rules that
    out, and the number is large: at the last chirp a unit rate tangent turns
    the phasor by about 1.5e7 radians per unit rate.
    """

    single = torch.tensor([DELAYS[0]], dtype=torch.float32, device="cuda")
    rate = torch.tensor([RATES[0]], dtype=torch.float32, device="cuda")
    weight = torch.ones(1, dtype=torch.complex64, device="cuda")
    offsets = torch.tensor([0, 1], dtype=torch.int64, device="cuda")

    with forward_ad.dual_level():
        cube = synthesize_fmcw_rows(single, forward_ad.make_dual(rate, torch.ones_like(rate)), weight, offsets, SPEC)
        primal, tangent = forward_ad.unpack_dual(cube)
        magnitude = float(tangent.abs().max())
        reference = float(primal.abs().max())

    assert magnitude > 1.0e6 * reference

    # And the FIRST chirp's tangent is exactly zero, because ``t_c = 0`` there:
    # the drift has had no slow time to accumulate. A kernel that applied the
    # rate as a constant offset would fail this and pass everything above.
    with forward_ad.dual_level():
        cube = synthesize_fmcw_rows(single, forward_ad.make_dual(rate, torch.ones_like(rate)), weight, offsets, SPEC)
        first_chirp = forward_ad.unpack_dual(cube).tangent[0]
    assert float(first_chirp.abs().max()) == 0.0
