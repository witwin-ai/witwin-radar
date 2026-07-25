"""The join's own AD, finite-difference verified at a scalar loss, both modes.

This is the rigorous gate the native join is accepted on. It runs on synthetic
legs on purpose: with the legs' payloads as free parameters the join is pure
algebra, so the float64 oracle is exact, the central difference is perfectly
conditioned, and every derivative the kernel claims - inbound delay, outbound
delay, both leg coefficients, and the per-site response - is checked
independently rather than through a chain that could hide a cancelled term.

The order is deliberate. The float64 Torch oracle is validated by float64
central differences FIRST; only then are the production float32 gradients
compared against it. A float32 finite difference on the production loss is not
a usable oracle: the composed transfer is a triple product spanning several
orders of magnitude, and differencing two nearly equal float32 numbers can
return a confident zero.

The loss is deliberately phase sensitive. ``|C_rt|^2`` would pass with the
complex conjugation inverted anywhere in the chain; ``Re(conj(g) . C_rt)`` with
a fixed complex ``g`` does not.
"""

from __future__ import annotations

import pytest
import torch
import torch.autograd.forward_ad as forward_ad

from witwin.radar.paths import TwoWayComposer

from reference.two_way_torch import PerSiteResponse, join_reference  # noqa: E402
from support import fd  # noqa: E402
from support import join_fixture as fx  # noqa: E402


pytestmark = pytest.mark.gpu

SOURCES = [10, 11]
SINKS = [30]
SITES = [20, 21]
COMPONENTS = [0, 1]

# tau_rt is nanosecond scale, so it enters the loss through a 1e8 scaling that
# puts every term at order one. Without it the delay contribution would be
# 1e-8 of the transfer contribution and the finite difference would be
# measuring rounding noise instead of a derivative.
DELAY_SCALE = 1.0e8
STEP = {"tau": 1.0e-12, "coefficient": 1.0e-6, "response": 1.0e-6}
ZERO_FLOOR = 1.0e-9


def _composer(device: str = "cuda"):
    inbound = fx.frozen_leg(fx.leg_rows(SOURCES, SITES, COMPONENTS), device=device)
    outbound = fx.frozen_leg(fx.leg_rows(SITES, SINKS, COMPONENTS), device=device)
    return TwoWayComposer.freeze(
        inbound,
        outbound,
        torch.tensor(SITES, dtype=torch.int64, device=device),
        radar_source_ids=SOURCES,
        radar_sink_ids=SINKS,
        reference_frequency_hz=77.0e9,
    )


def _parameters(composer, *, device: str = "cuda"):
    tau_in, rate_in, c_in = fx.payload(
        composer.inbound_row_count, seed=101, device=device
    )
    tau_out, rate_out, c_out = fx.payload(
        composer.outbound_row_count, seed=102, device=device
    )
    _, _, site = fx.payload(composer.site_count, seed=103, device=device)
    return {
        "tau_in": tau_in,
        "tau_out": tau_out,
        "rate_in": rate_in,
        "rate_out": rate_out,
        "c_in": c_in,
        "c_out": c_out,
        "response": site,
    }


def _loss_weights(composer, *, device: str = "cuda"):
    generator = torch.Generator().manual_seed(4242)
    rows = composer.path_count

    def sample():
        return (torch.rand(rows, generator=generator, dtype=torch.float64) - 0.5).to(
            device=device
        )

    return {
        "linear": sample(),
        "quadratic": 1.0 + sample(),
        "transfer": torch.complex(sample(), sample()),
    }


def _scalar_loss(tau_rt, transfer, weights):
    """A phase-sensitive scalar loss with a real second-order term in tau.

    The quadratic delay term is what makes the finite difference a genuine
    test of the delay chain rather than an identity: a loss linear in every
    parameter is differenced exactly by construction and would pass even if the
    curvature were wrong.
    """

    scaled = tau_rt.to(torch.float64) * DELAY_SCALE
    delay_term = (weights["linear"] * scaled).sum() + 0.5 * (
        weights["quadratic"] * scaled * scaled
    ).sum()
    transfer_term = (
        torch.conj(weights["transfer"]) * transfer.to(torch.complex128)
    ).real.sum()
    return delay_term + transfer_term


def _oracle_loss(composer, values, weights, row_valid):
    tau_rt, _, transfer = join_reference(
        tau_in=values["tau_in"],
        tau_out=values["tau_out"],
        rate_in=values["rate_in"],
        rate_out=values["rate_out"],
        c_in=values["c_in"],
        c_out=values["c_out"],
        response=values["response"],
        idx_in=composer.inbound_row,
        idx_out=composer.outbound_row,
        idx_s=composer.response_slot,
        row_valid=row_valid,
    )
    return _scalar_loss(tau_rt, transfer, weights)


def _oracle_gradients(composer, base, weights, row_valid):
    values = {
        name: value.clone().requires_grad_(True) for name, value in base.items()
    }
    loss = _oracle_loss(composer, values, weights, row_valid)
    loss.backward()
    return {name: value.grad for name, value in values.items()}, float(
        loss.detach()
    )


def _production_batch(composer, base, *, row_valid=None, requires_grad=False):
    def real(name):
        tensor = base[name].to(torch.float32).clone()
        return tensor.requires_grad_(requires_grad)

    def complex_(name):
        tensor = base[name].to(torch.complex64).clone()
        return tensor.requires_grad_(requires_grad)

    valid_in = valid_out = None
    if row_valid is not None:
        valid_in, valid_out = row_valid
    live = {
        "tau_in": real("tau_in"),
        "tau_out": real("tau_out"),
        "c_in": complex_("c_in"),
        "c_out": complex_("c_out"),
        "response": complex_("response"),
    }
    inbound = fx.leg_batch(
        live["tau_in"],
        live["c_in"],
        rate=base["rate_in"].to(torch.float32),
        row_valid=valid_in,
    )
    outbound = fx.leg_batch(
        live["tau_out"],
        live["c_out"],
        rate=base["rate_out"].to(torch.float32),
        row_valid=valid_out,
    )
    return live, inbound, outbound, PerSiteResponse(live["response"])


# --------------------------------------------------------------------------
# The oracle first
# --------------------------------------------------------------------------


def test_the_float64_oracle_matches_its_own_finite_differences():
    composer = _composer()
    base = _parameters(composer)
    weights = _loss_weights(composer)
    gradients, _ = _oracle_gradients(composer, base, weights, None)

    for name, step in (
        ("tau_in", STEP["tau"]),
        ("tau_out", STEP["tau"]),
    ):
        for index in (0, 3, int(base[name].shape[0]) - 1):
            measured = fd.central_difference(
                lambda value, key=name: _oracle_loss(
                    composer, {**base, key: value}, weights, None
                ),
                base[name],
                index,
                step,
            )
            assert (
                fd.relative_error(
                    measured, float(gradients[name][index]), floor=ZERO_FLOOR
                )
                < 1e-5
            ), (name, index, measured, float(gradients[name][index]))

    for name in ("c_in", "c_out", "response"):
        for index in (0, int(base[name].shape[0]) - 1):
            for part, expected in (
                ("real", float(gradients[name][index].real)),
                ("imag", float(gradients[name][index].imag)),
            ):
                offset = 1.0 if part == "real" else 1.0j

                def evaluate(value, key=name, offset=offset, index=index):
                    perturbed = base[key].clone()
                    perturbed[index] = base[key][index] + offset * value
                    return _oracle_loss(
                        composer, {**base, key: perturbed}, weights, None
                    )

                measured = fd.central_difference(
                    evaluate,
                    torch.zeros(1, dtype=torch.float64, device=base[name].device),
                    0,
                    STEP["coefficient"],
                )
                assert (
                    fd.relative_error(measured, expected, floor=ZERO_FLOOR) < 1e-5
                ), (name, index, part, measured, expected)


# --------------------------------------------------------------------------
# Reverse mode through the native VJP
# --------------------------------------------------------------------------


def test_reverse_mode_join_gradients_match_the_oracle():
    composer = _composer()
    base = _parameters(composer)
    weights = _loss_weights(composer)
    oracle, oracle_loss = _oracle_gradients(composer, base, weights, None)

    live, inbound, outbound, response = _production_batch(
        composer, base, requires_grad=True
    )
    composed = composer.compose(inbound, outbound, response)
    loss = _scalar_loss(composed.total_delay_s, composed.complex_transfer_ref, weights)
    loss.backward()

    assert fd.relative_error(float(loss.detach()), oracle_loss, floor=ZERO_FLOOR) < 1e-5
    for name in ("tau_in", "tau_out", "c_in", "c_out", "response"):
        measured = live[name].grad
        assert measured is not None, name
        assert torch.isfinite(measured.view(torch.float32 if measured.is_complex() else measured.dtype)).all()
        expected = oracle[name]
        # Not vacuous: every one of these gradients is orders of magnitude
        # above the zero floor, so agreement means agreement on a number.
        assert float(expected.abs().min()) > 1.0e-3, name
        for index in range(int(expected.shape[0])):
            if measured.is_complex():
                for part in ("real", "imag"):
                    assert (
                        fd.relative_error(
                            float(getattr(measured[index], part)),
                            float(getattr(expected[index], part)),
                            floor=ZERO_FLOOR,
                        )
                        < 1e-4
                    ), (name, index, part)
            else:
                assert (
                    fd.relative_error(
                        float(measured[index]),
                        float(expected[index]),
                        floor=ZERO_FLOOR,
                    )
                    < 1e-4
                ), (name, index)


def test_each_coefficient_gradient_family_matches_a_hand_derived_reduction():
    """One independent check per native VJP gradient slot family.

    ``test_reverse_mode_join_gradients_match_the_oracle`` compares against the
    retained Torch composition differentiated by autograd. That is the right
    primary guard, but it is a SINGLE guard: zeroing one family's store in the
    backward kernel is caught by that test and nothing else, and the
    permutation test - which compares gradients to gradients - happily compares
    zeros to zeros.

    So this derives the three reductions by hand instead, from the definition
    of the composition and Torch's real-pair convention for a complex
    parameter (``grad.real = dL/dRe(z)``, ``grad.imag = dL/dIm(z)``):

        C_rt[k] = C_out[o] * S[s] * C_in[i],  L includes Re(conj(g) . C_rt)

        grad_C_out[j] = sum over live rows with o == j of g . conj(S . C_in)
        grad_C_in[j]  = sum over live rows with i == j of g . conj(C_out . S)
        grad_S[j]     = sum over live rows with s == j of g . conj(C_out . C_in)

    A dead row is excluded, so the mask is part of the derivation rather than a
    separate assertion. The delay terms of the loss touch no coefficient and
    drop out.
    """

    composer = _composer()
    base = _parameters(composer)
    weights = _loss_weights(composer)

    valid_in = torch.ones(composer.inbound_row_count, dtype=torch.bool, device="cuda")
    valid_in[1] = False
    valid_out = torch.ones(
        composer.outbound_row_count, dtype=torch.bool, device="cuda"
    )
    valid_out[0] = False

    live, inbound, outbound, response = _production_batch(
        composer, base, row_valid=(valid_in, valid_out), requires_grad=True
    )
    composed = composer.compose(inbound, outbound, response)
    _scalar_loss(
        composed.total_delay_s, composed.complex_transfer_ref, weights
    ).backward()

    # The join tables and the mask, read on the host: this is a test, and the
    # tables are freeze-time values the production path never reads back.
    idx_in = composer.inbound_row.tolist()
    idx_out = composer.outbound_row.tolist()
    idx_s = composer.response_slot.tolist()
    alive_in = valid_in.tolist()
    alive_out = valid_out.tolist()
    g = weights["transfer"].to(torch.complex128).cpu()
    c_in = base["c_in"].to(torch.complex128).cpu()
    c_out = base["c_out"].to(torch.complex128).cpu()
    site = base["response"].to(torch.complex128).cpu()

    expected = {
        "c_in": torch.zeros(composer.inbound_row_count, dtype=torch.complex128),
        "c_out": torch.zeros(composer.outbound_row_count, dtype=torch.complex128),
        "response": torch.zeros(composer.site_count, dtype=torch.complex128),
    }
    live_rows = 0
    for row, (i, o, s) in enumerate(zip(idx_in, idx_out, idx_s, strict=True)):
        if not (alive_in[i] and alive_out[o]):
            continue
        live_rows += 1
        expected["c_out"][o] += g[row] * torch.conj(site[s] * c_in[i])
        expected["c_in"][i] += g[row] * torch.conj(c_out[o] * site[s])
        expected["response"][s] += g[row] * torch.conj(c_out[o] * c_in[i])
    # The mask is doing something and the sums are not empty.
    assert 0 < live_rows < composer.path_count

    for name, reference in expected.items():
        measured = live[name].grad
        assert measured is not None, name
        for index in range(int(reference.shape[0])):
            for part in ("real", "imag"):
                target = float(getattr(reference[index], part))
                assert (
                    fd.relative_error(
                        float(getattr(measured[index], part)),
                        target,
                        floor=ZERO_FLOOR,
                    )
                    < 1e-4
                ), (name, index, part)
        # Every family must carry real signal somewhere, or "matches" would be
        # a comparison of zeros.
        assert float(reference.abs().max()) > 1.0e-3, name


def test_a_dead_row_carries_no_gradient_back_to_either_leg():
    """Dead rows are data, and data with a zero derivative.

    A row that stops existing must not leak a gradient into the leg payloads it
    would have been built from. With every row of one inbound leg entry dead,
    that entry's gradient is exactly zero - not small.
    """

    composer = _composer()
    base = _parameters(composer)
    weights = _loss_weights(composer)

    valid_in = torch.ones(
        composer.inbound_row_count, dtype=torch.bool, device="cuda"
    )
    valid_in[2] = False
    valid_out = torch.ones(
        composer.outbound_row_count, dtype=torch.bool, device="cuda"
    )
    live, inbound, outbound, response = _production_batch(
        composer, base, row_valid=(valid_in, valid_out), requires_grad=True
    )
    composed = composer.compose(inbound, outbound, response)
    _scalar_loss(
        composed.total_delay_s, composed.complex_transfer_ref, weights
    ).backward()

    assert float(live["tau_in"].grad[2]) == 0.0
    assert complex(live["c_in"].grad[2]) == 0j
    # And the surviving entries did receive gradient, so the zero above is the
    # mask talking rather than a severed tape.
    assert float(live["tau_in"].grad.abs().sum()) > 0.0
    assert float(live["c_in"].grad.abs().sum()) > 0.0


# --------------------------------------------------------------------------
# Forward mode through the native JVP
# --------------------------------------------------------------------------


def test_forward_mode_join_tangent_matches_the_oracle_and_a_finite_difference():
    composer = _composer()
    base = _parameters(composer)
    weights = _loss_weights(composer)
    oracle, _ = _oracle_gradients(composer, base, weights, None)

    generator = torch.Generator().manual_seed(777)

    def direction(shape, complex_valued):
        real = torch.rand(shape, generator=generator, dtype=torch.float64) - 0.5
        if not complex_valued:
            return real.to(device="cuda")
        imag = torch.rand(shape, generator=generator, dtype=torch.float64) - 0.5
        return torch.complex(real, imag).to(device="cuda")

    directions = {
        "tau_in": direction(composer.inbound_row_count, False) * 1.0e-8,
        "tau_out": direction(composer.outbound_row_count, False) * 1.0e-8,
        "c_in": direction(composer.inbound_row_count, True),
        "c_out": direction(composer.outbound_row_count, True),
        "response": direction(composer.site_count, True),
    }

    live, inbound, outbound, response = _production_batch(composer, base)
    with forward_ad.dual_level():
        dual_inbound = fx.leg_batch(
            forward_ad.make_dual(
                live["tau_in"], directions["tau_in"].to(torch.float32)
            ),
            forward_ad.make_dual(
                live["c_in"], directions["c_in"].to(torch.complex64)
            ),
            rate=base["rate_in"].to(torch.float32),
        )
        dual_outbound = fx.leg_batch(
            forward_ad.make_dual(
                live["tau_out"], directions["tau_out"].to(torch.float32)
            ),
            forward_ad.make_dual(
                live["c_out"], directions["c_out"].to(torch.complex64)
            ),
            rate=base["rate_out"].to(torch.float32),
        )
        dual_response = PerSiteResponse(
            forward_ad.make_dual(
                live["response"], directions["response"].to(torch.complex64)
            )
        )
        composed = composer.compose(dual_inbound, dual_outbound, dual_response)
        loss = _scalar_loss(
            composed.total_delay_s, composed.complex_transfer_ref, weights
        )
        tangent = forward_ad.unpack_dual(loss).tangent
        assert tangent is not None, "the forward tape did not reach the loss"
        measured = float(tangent)

    # The same directional derivative from the reverse-mode oracle. For a
    # complex parameter the projection is Re(conj(grad) . direction), which is
    # exactly the real inner product of the (real, imag) pairs.
    expected = 0.0
    for name, gradient in oracle.items():
        if name not in directions:
            continue
        step = directions[name]
        if gradient.is_complex():
            expected += float((torch.conj(gradient) * step).real.sum())
        else:
            expected += float((gradient * step).sum())
    assert fd.relative_error(measured, expected, floor=ZERO_FLOOR) < 1e-4

    # And by float64 central difference along the same direction, so the
    # forward path is checked against a difference and not only against the
    # reverse path it is supposed to be independent of.
    def along(scale: float) -> float:
        moved = {
            name: base[name] + scale * directions[name]
            if name in directions
            else base[name]
            for name in base
        }
        return float(_oracle_loss(composer, moved, weights, None))

    step = 1.0e-4
    differenced = (along(step) - along(-step)) / (2.0 * step)
    assert fd.relative_error(measured, differenced, floor=ZERO_FLOOR) < 1e-4


def test_the_composed_rate_tangent_is_structurally_zero():
    """delay_rate is primal by contract, so its tangent is zero, not absent.

    The contract severs ``d(delay_rate)/dx`` deliberately. Publishing a nonzero
    tangent there would resurrect a second-order term nothing else in the chain
    claims.

    "Present and exactly zero" is asserted rather than "zero or missing". The
    kernel writes the rate tangent unconditionally, so autograd does attach
    one; accepting ``None`` as well would mean this test could not tell a zero
    the kernel computed from a tangent autograd dropped on the way out.
    """

    composer = _composer()
    base = _parameters(composer)
    live, inbound, outbound, response = _production_batch(composer, base)
    with forward_ad.dual_level():
        dual_inbound = fx.leg_batch(
            forward_ad.make_dual(live["tau_in"], torch.ones_like(live["tau_in"])),
            live["c_in"],
            rate=base["rate_in"].to(torch.float32),
        )
        composed = composer.compose(dual_inbound, outbound, response)
        delay_tangent = forward_ad.unpack_dual(composed.total_delay_s).tangent
        rate_tangent = forward_ad.unpack_dual(composed.delay_rate).tangent
        assert delay_tangent is not None
        assert float(delay_tangent.abs().sum()) > 0.0
        assert rate_tangent is not None, "the rate tangent was dropped, not zeroed"
        assert rate_tangent.shape == composed.delay_rate.shape
        assert float(rate_tangent.abs().sum()) == 0.0
