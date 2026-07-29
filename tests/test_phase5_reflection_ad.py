"""Reverse and forward mode THROUGH a reflection leg, against finite differences.

Phase 4 verified the AD chain on line-of-sight legs, where the transport is
free-space and the site derivative is a pure geometry term. Phase 5 added
reflection legs, whose derivative also runs through the specular point and the
lossy-dielectric Fresnel coefficient - and nothing exercised reverse mode there
at all. Forward-mode Doppler was checked, but a Doppler tangent is one
direction of one term.

The oracle is a finite difference of the PRODUCTION chain at perturbed
positions, not a reimplementation. An independent oracle for Channel's Fresnel
coefficient would duplicate a RayD/Channel numerical owner, which is exactly
what the ownership rules forbid; a finite difference is allowed in tests and
owns nothing.

Two things make that finite difference honest at float32:

* The loss weights are ZERO on the one combined row that joins two
  line-of-sight legs. Every remaining row carries at least one reflection, so
  the whole gradient is the reflection rows' derivative rather than a
  free-space derivative with a reflection contribution somewhere inside it.
* The stencil is fourth order and its denominator is the realized float32 step.
  The second-order difference at the same step is still 2-6% from truncation
  alone, which is the same size as the disagreement it would be trying to
  detect.

The site sits off the fixture position and off the reflection plane, at
``(2.31, 0.83, 0.27)``, so all three axes carry signal and none of them is zero
by symmetry.
"""

from __future__ import annotations

import pytest
import torch
import torch.autograd.forward_ad as forward_ad

pytest.importorskip("witwin.channel")

from support import fd  # noqa: E402
from support import phase4_geometry as geo  # noqa: E402
from support import spike_driver as drv  # noqa: E402

pytestmark = pytest.mark.gpu

MULTIPATH_COMPONENTS = frozenset({"los", "reflection"})
SITE_M = (2.31, 0.83, 0.27)
# Nanosecond delays enter the loss through a 1e8 scale so the delay and
# transfer terms are the same order; without it the finite difference would be
# measuring the transfer term alone.
DELAY_SCALE = 1.0e8
STEP_M = 2.0e-4
# The float32 production chain's own noise floor, measured: the fourth-order
# difference agrees to 0.04-1.4% across steps of 1e-4, 2e-4 and 4e-4 m, and the
# second-order difference at the same steps converges quadratically toward the
# same value. A wrong term would not sit inside this band on all three axes.
FD_RTOL = 2.0e-2
# The transfer-only control drops the delay terms, which shrinks the loss to
# ~2e-4 and its smallest gradient component to ~8e-3. Measured across steps of
# 1e-4 to 8e-4 m that component's fourth-order difference wanders between 0.5%
# and 2.2% NON-monotonically, which is a noise floor rather than truncation, so
# it gets its own stated tolerance instead of borrowing the one above.
CONTROL_FD_RTOL = 4.0e-2
ZERO_FLOOR = 1.0e-6


@pytest.fixture(scope="module")
def spike():
    return drv.Phase4Spike(components=MULTIPATH_COMPONENTS, max_depth=1)


@pytest.fixture(scope="module")
def endpoints():
    tx, _, rx = drv.positions()
    return tx, rx


def _site(values, *, requires_grad: bool = False) -> torch.Tensor:
    tensor = torch.tensor([tuple(values)], dtype=torch.float32, device="cuda")
    return tensor.requires_grad_(requires_grad)


def _reflection_mask(spike, composed) -> torch.Tensor:
    """One per composed row, zero on the row that joins two los legs."""

    names = {geo.LOS_COMPONENT_ID: "los", geo.REFLECTION_COMPONENT_ID: "reflection"}

    def component(frozen, row):
        return names[int(frozen.component_id[row])]

    keys = [
        (
            component(spike.inbound, int(composed.topology.inbound_row[row])),
            component(spike.outbound, int(composed.topology.outbound_row[row])),
        )
        for row in range(composed.path_count)
    ]
    assert set(keys) == {("los", "los"), ("los", "reflection"), ("reflection", "los"), ("reflection", "reflection")}
    return torch.tensor(
        [0.0 if key == ("los", "los") else 1.0 for key in keys], dtype=torch.float64, device=composed.device
    )


def _weights(mask: torch.Tensor) -> dict[str, torch.Tensor]:
    """Deterministic, phase-sensitive loss weights, masked to reflection rows.

    ``|C|^2`` would pass with the transfer's phase conjugated anywhere in the
    chain. ``Re(conj(g) . C)`` with a fixed complex ``g`` does not. The
    quadratic delay term gives the delay chain real curvature, so the
    difference is testing a derivative rather than an identity.
    """

    generator = torch.Generator().manual_seed(9091)
    rows = int(mask.shape[0])

    def sample() -> torch.Tensor:
        return (torch.rand(rows, generator=generator, dtype=torch.float64) - 0.5).to(device=mask.device)

    return {
        "linear": sample() * mask,
        "quadratic": (1.0 + sample()) * mask,
        "transfer": torch.complex(sample(), sample()) * mask,
    }


def _loss(composed, weights) -> torch.Tensor:
    scaled = composed.total_delay_s.to(torch.float64) * DELAY_SCALE
    delay_term = (weights["linear"] * scaled).sum() + 0.5 * (weights["quadratic"] * scaled * scaled).sum()
    transfer_term = (torch.conj(weights["transfer"]) * composed.complex_transfer_ref.to(torch.complex128)).real.sum()
    return delay_term + transfer_term


@pytest.fixture(scope="module")
def weights(spike, endpoints):
    tx, rx = endpoints
    composed, _, _ = spike.paths(tx, _site(SITE_M), rx, drv.make_response())
    assert composed.path_count == 4
    assert bool(composed.row_valid.all()), "all four rows must be live here"
    return _weights(_reflection_mask(spike, composed))


def _evaluate(spike, endpoints, weights, values) -> float:
    tx, rx = endpoints
    composed, _, _ = spike.paths(tx, _site(values), rx, drv.make_response())
    return float(_loss(composed, weights))


def test_the_reflection_rows_carry_the_whole_loss(spike, endpoints, weights):
    """The premise, asserted before anything is differentiated.

    If the masked-out line-of-sight row still moved the loss, the finite
    differences below would be verifying a free-space derivative wearing a
    reflection label.
    """

    tx, rx = endpoints
    composed, inbound, outbound = spike.paths(tx, _site(SITE_M), rx, drv.make_response())
    assert float(weights["transfer"].abs().sum()) > 0.0
    assert int((weights["transfer"].abs() == 0.0).sum()) == 1

    # Both legs really do carry a reflection row with a nontrivial coefficient,
    # so "through a reflection leg" is a fact about this fixture.
    for name, legs in (("inbound", inbound), ("outbound", outbound)):
        magnitudes = legs.coefficient.abs()
        assert int(legs.leg_count) == 2, name
        assert float(magnitudes.min()) > 0.0, name
    baseline = _evaluate(spike, endpoints, weights, SITE_M)
    assert abs(baseline) > ZERO_FLOOR


def test_reverse_mode_site_gradients_match_finite_differences(spike, endpoints, weights):
    tx, rx = endpoints
    site = _site(SITE_M, requires_grad=True)
    composed, _, _ = spike.paths(tx, site, rx, drv.make_response(), ad_mode="vjp")
    _loss(composed, weights).backward()

    gradient = site.grad.reshape(3)
    assert torch.isfinite(gradient).all()

    for axis in range(3):
        samples = {}
        realized = {}
        for offset in (-2, -1, 1, 2):
            moved = list(SITE_M)
            moved[axis] += offset * STEP_M
            samples[offset] = _evaluate(spike, endpoints, weights, moved)
            realized[offset] = float(_site(moved)[0, axis])
        measured = fd.fourth_order_difference(samples, (realized[1] - realized[-1]) / 2.0)
        expected = float(gradient[axis])
        # Every axis carries real signal at this off-plane site, so none of
        # these comparisons is a zero against a zero.
        assert abs(expected) > 1.0e-1, (axis, expected)
        assert fd.relative_error(measured, expected, floor=ZERO_FLOOR) < FD_RTOL, (axis, measured, expected)


def test_forward_mode_site_tangent_matches_reverse_mode_and_a_difference(spike, endpoints, weights):
    """The same derivative three ways: JVP, VJP projected, and a difference.

    JVP against VJP is the cross-check that the two native companions agree;
    both against the difference is what makes the agreement mean correct rather
    than consistently wrong.
    """

    tx, rx = endpoints
    direction = torch.tensor([[-0.5, 0.4, 0.3]], dtype=torch.float32, device="cuda")

    site = _site(SITE_M, requires_grad=True)
    composed, _, _ = spike.paths(tx, site, rx, drv.make_response(), ad_mode="vjp")
    _loss(composed, weights).backward()
    projected = float(torch.dot(site.grad.reshape(3).double(), direction.reshape(3).double()))

    with forward_ad.dual_level():
        dual, _, _ = spike.paths(
            tx,
            forward_ad.make_dual(_site(SITE_M), direction),
            rx,
            drv.make_response(),
            ad_mode="jvp",
            include_delay_rate=False,
        )
        tangent = forward_ad.unpack_dual(_loss(dual, weights)).tangent
        assert tangent is not None, "the forward tape did not reach the loss"
        measured = float(tangent)

    assert abs(projected) > 1.0e-1
    # The two AD modes evaluate the same native transport, so they agree far
    # more tightly than either agrees with a difference.
    assert fd.relative_error(measured, projected, floor=ZERO_FLOOR) < 1e-5

    unit = direction.reshape(3).double().cpu().tolist()
    samples = {}
    for offset in (-2, -1, 1, 2):
        moved = [SITE_M[axis] + offset * STEP_M * unit[axis] for axis in range(3)]
        samples[offset] = _evaluate(spike, endpoints, weights, moved)
    differenced = fd.fourth_order_difference(samples, STEP_M)
    assert fd.relative_error(measured, differenced, floor=ZERO_FLOOR) < FD_RTOL, (measured, differenced)


def test_the_gradient_reaches_the_reflection_leg_and_not_only_the_delay(spike, endpoints):
    """A term-level control on what the gradient above is made of.

    With the delay terms removed the loss depends on the site ONLY through the
    complex transfers of rows that carry a reflection. That is the statement
    this control makes: the reflection TRANSPORT is differentiated, not just
    the round-trip length.

    It is deliberately not a claim about the Fresnel coefficient in isolation.
    The transfer's site dependence is dominated by the specular-point
    propagation phase, and separating the lossy-dielectric coefficient's own
    derivative would need an independent oracle for it - which would duplicate
    a Channel/RayD numerical owner. That remains a recorded gap, not something
    this test quietly claims to have closed.
    """

    tx, rx = endpoints
    composed, _, _ = spike.paths(tx, _site(SITE_M), rx, drv.make_response())
    mask = _reflection_mask(spike, composed)
    weights = _weights(mask)
    weights = {
        "linear": torch.zeros_like(weights["linear"]),
        "quadratic": torch.zeros_like(weights["quadratic"]),
        "transfer": weights["transfer"],
    }

    site = _site(SITE_M, requires_grad=True)
    composed, _, _ = spike.paths(tx, site, rx, drv.make_response(), ad_mode="vjp")
    loss = _loss(composed, weights)
    loss.backward()
    gradient = site.grad.reshape(3)
    assert float(gradient.abs().min()) > ZERO_FLOOR

    for axis in range(3):
        samples = {}
        realized = {}
        for offset in (-2, -1, 1, 2):
            moved = list(SITE_M)
            moved[axis] += offset * STEP_M
            samples[offset] = _evaluate(spike, endpoints, weights, moved)
            realized[offset] = float(_site(moved)[0, axis])
        measured = fd.fourth_order_difference(samples, (realized[1] - realized[-1]) / 2.0)
        assert fd.relative_error(measured, float(gradient[axis]), floor=ZERO_FLOOR) < CONTROL_FD_RTOL, (
            axis,
            measured,
            float(gradient[axis]),
        )
