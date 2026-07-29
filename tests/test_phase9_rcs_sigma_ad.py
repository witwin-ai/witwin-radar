"""A radar cross section as a leaf, all the way to a synthesized FMCW cube.

The one genuine capability ADD in the Phase-9 Radar work. Everything else this
stage does is a refusal or a test of something that already worked;
``ScalarRcsResponse.from_rcs`` could not carry a derivative at all, because
``rcs_amplitude`` was ``math.sqrt`` on the host. There was no refusal either, so
a caller who marked a cross section got a plain float back and no signal that the
leaf had been dropped on the floor - the amplitude leaf beside it has worked
since Phase 4.

``sigma`` is the canonical inverse-design question in radar: how large does this
target have to be. Supporting it costs one ``torch.sqrt`` in RESULT
CONSTRUCTION, off the per-path loop, so the mechanism is ``torch-orchestration``
and no physics moved into Torch. The capability matrix records it that way on
purpose - the distinction between "a Torch expression built a result" and "a
Torch expression evaluated physics" is the one the compute policy turns on.

**The oracle here is exact, not a finite difference, and that is the point.**
The whole chain is LINEAR in the response amplitude - the join multiplies by it,
the beat kernel is linear in the weight, and the loss is the squared magnitude -
so with ``|S| = sqrt(4 pi sigma) / lambda`` the loss is exactly proportional to
``sigma`` and

    d(loss)/d(sigma) = loss / sigma

to machine precision. A finite difference is run as well, because the closed
form alone would be satisfied by a chain that ignored sigma's VALUE and by any
constant multiple of the true law; the value is pinned separately against
``RCS_AMPLITUDE_LAW`` and against a measured ``d(loss)/d(amplitude)``, which is
what fixes the ``0.5 / sqrt(sigma)`` factor rather than merely the linearity.

Measured on this fixture at ``sigma = 3.5``:

    vjp                                3.658581660e-9
    loss / sigma  (exact)              3.658582263e-9    rel 1.6e-7
    d(loss)/d(amplitude) chain rule    3.658581729e-9    rel 1.9e-8
    jvp                                3.658581882e-9    rel 6.1e-8
    central difference, h = 1e-3       3.659739178e-9    rel 3.2e-4
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.autograd.forward_ad as forward_ad

pytest.importorskip("witwin.channel")

from support import multi_endpoint_driver as drv  # noqa: E402
from support import multi_endpoint_geometry as geo  # noqa: E402
from support.synthesis_batch import to_synthesis  # noqa: E402

from witwin.radar.scattering import ScalarRcsResponse  # noqa: E402
from witwin.radar.scattering import (  # noqa: E402
    SPEED_OF_LIGHT_M_PER_S,
    rcs_amplitude,
)
from witwin.radar.synthesis import synthesize_fmcw  # noqa: E402


pytestmark = pytest.mark.gpu


#: Square metres. Away from 1.0 so that a chain which confused ``sigma`` with
#: ``amplitude`` somewhere could not pass by coincidence.
SIGMA_M2 = 3.5

#: Non-zero so that the phase tape is exercised alongside the amplitude one.
PHASE_RAD = 0.7

WAVELENGTH_M = SPEED_OF_LIGHT_M_PER_S / geo.REFERENCE_FREQUENCY_HZ

#: Central-difference step, in square metres. ``sigma`` is order 1 and the loss
#: is exactly linear in it, so the only error here is float32 cancellation and
#: the step is chosen large enough to clear it: measured relative agreement at
#: 1e-4, 1e-3 and 1e-2 is 3.5e-3, 3.2e-4 and 2.6e-4.
SIGMA_STEP = 1.0e-3
SIGMA_FD_RTOL = 2.0e-3


@pytest.fixture(scope="module")
def spec():
    return drv.make_spec(num_chirps=2)


@pytest.fixture(scope="module")
def spike():
    return drv.MultiEndpointSpike()


def _loss(spike, spec, response) -> torch.Tensor:
    """freeze -> reevaluate -> compose -> synthesize -> scalar, once."""

    composed, _, _ = spike.frame(
        None, response, ad_mode="none", include_delay_rate=False
    )
    cube = synthesize_fmcw(to_synthesis(composed), spec)
    return cube.abs().square().sum()


def _sigma_tensor(value: float, *, requires_grad: bool = False) -> torch.Tensor:
    tensor = torch.tensor(float(value), dtype=torch.float32, device="cuda")
    return tensor.requires_grad_(requires_grad)


def _response_from_sigma(sigma: torch.Tensor) -> ScalarRcsResponse:
    return ScalarRcsResponse.from_rcs(
        sigma,
        reference_frequency_hz=geo.REFERENCE_FREQUENCY_HZ,
        phase_rad=PHASE_RAD,
    )


# --------------------------------------------------------------------------
# 1. The law, and the host-float path that must not move
# --------------------------------------------------------------------------


def test_the_tensor_route_reproduces_the_host_float_amplitude():
    """The compatibility statement: existing callers see the same number.

    A tensor cross section is a new ROUTE through the same law, not a new law.
    If the two disagreed, every existing absolute-level test would still pass -
    they all go through the host float - and the differentiable route would be
    quietly calibrated differently by up to whatever the discrepancy was.
    """

    host = rcs_amplitude(SIGMA_M2, WAVELENGTH_M)
    device = rcs_amplitude(_sigma_tensor(SIGMA_M2), WAVELENGTH_M)
    assert isinstance(device, torch.Tensor)
    closed_form = math.sqrt(4.0 * math.pi * SIGMA_M2) / WAVELENGTH_M
    assert host == pytest.approx(closed_form, rel=1e-12)
    assert float(device) == pytest.approx(closed_form, rel=1e-6)


def test_a_marked_cross_section_produces_a_graph_bearing_response():
    sigma = _sigma_tensor(SIGMA_M2, requires_grad=True)
    response = _response_from_sigma(sigma)
    assert response.amplitude.grad_fn is not None
    assert response.amplitude.dtype == torch.float32
    assert response.amplitude.device.type == "cuda"
    # The phase follows the amplitude's placement and stays a constant.
    assert response.phase_rad.device == response.amplitude.device
    assert response.phase_rad.grad_fn is None


# --------------------------------------------------------------------------
# 2. Reverse mode against three independent oracles
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def reverse(spike, spec):
    """One reverse pass, reused by the four assertions that read it."""

    sigma = _sigma_tensor(SIGMA_M2, requires_grad=True)
    loss = _loss(spike, spec, _response_from_sigma(sigma))
    loss.backward()
    return float(loss.detach()), float(sigma.grad)


def test_the_cross_section_gradient_equals_the_exact_closed_form(reverse):
    """``d(loss)/d(sigma) = loss / sigma``, because the chain is linear in ``S``.

    This is a stronger statement than any finite difference could make and it
    is available only because the response is a pure scale: the join multiplies
    by it, the beat kernel is linear in the weight, and ``|IQ|^2`` is quadratic
    in the amplitude, which is itself ``sqrt(sigma)`` times a constant. The two
    square roots cancel exactly.
    """

    loss, gradient = reverse
    assert loss > 0.0
    assert gradient == pytest.approx(loss / SIGMA_M2, rel=1e-5)


def test_the_cross_section_gradient_matches_a_central_difference(spike, spec, reverse):
    """FD as the independent oracle, in case the closed form is the wrong one.

    The linearity argument above is a claim ABOUT the chain. If it were false -
    if something downstream were not linear in the weight - the closed form and
    the analytic gradient could agree with each other and both be wrong. A
    difference of two complete recomputations cannot make that mistake.
    """

    _, gradient = reverse

    def value(sigma: float) -> float:
        return float(_loss(spike, spec, _response_from_sigma(_sigma_tensor(sigma))))

    difference = (
        value(SIGMA_M2 + SIGMA_STEP) - value(SIGMA_M2 - SIGMA_STEP)
    ) / (2.0 * SIGMA_STEP)
    assert difference == pytest.approx(gradient, rel=SIGMA_FD_RTOL)


def test_the_gradient_is_the_amplitude_gradient_through_the_square_root(
    spike, spec, reverse
):
    """The brief's explicit check: ``dL/dsigma = dL/damp * 0.5 * amp / sigma``.

    This is what actually pins the ``0.5 / sqrt(sigma)`` factor. The closed form
    and the finite difference are both satisfied by ANY chain that is linear in
    sigma, including one that used ``sqrt(4 pi sigma)`` with the wrong
    constant; comparing against a separately measured ``d(loss)/d(amplitude)``
    at the same operating point fixes the derivative of the map itself.

    ``amplitude`` is the leaf covered since Phase 4
    (``test_phase4_spike_e2e.py``), so this also states that the new leaf
    composes with the old one rather than sitting beside it.
    """

    _, gradient = reverse
    amplitude = float(rcs_amplitude(SIGMA_M2, WAVELENGTH_M))
    response = ScalarRcsResponse.from_values(
        amplitude, PHASE_RAD, device="cuda", requires_grad=True
    )
    _loss(spike, spec, response).backward()
    d_loss_d_amplitude = float(response.amplitude.grad)

    chain = d_loss_d_amplitude * 0.5 * amplitude / SIGMA_M2
    assert chain == pytest.approx(gradient, rel=1e-4)
    # And the factor is not 1: a chain that forgot the square root entirely
    # would be this far out.
    assert abs(d_loss_d_amplitude / gradient - 1.0) > 0.5


# --------------------------------------------------------------------------
# 3. Forward mode, and the two modes against each other
# --------------------------------------------------------------------------


def test_the_forward_tangent_matches_the_reverse_gradient(spike, spec, reverse):
    """One leaf, so ``<grad, v>`` and the tangent are the same number at ``v = 1``.

    Both modes are asserted because they are separate code paths through the
    join and the beat kernel - the forward one is a native ``jvp`` companion,
    the reverse a native ``backward`` - and the phase's acceptance criterion is
    that they agree on one frozen topology, not that each agrees with FD alone.
    """

    _, gradient = reverse
    with forward_ad.dual_level():
        primal = _sigma_tensor(SIGMA_M2)
        dual = forward_ad.make_dual(primal, torch.ones_like(primal))
        assert not dual.requires_grad
        loss = _loss(spike, spec, _response_from_sigma(dual))
        tangent = forward_ad.unpack_dual(loss).tangent
        assert tangent is not None, "the sigma tangent died before the cube"
        measured = float(tangent)
    assert measured == pytest.approx(gradient, rel=1e-5)


def test_a_detached_cross_section_carries_no_gradient_at_all(spike, spec):
    """The falsifier. Without it the tests above measure an unknown chain.

    ``detach`` here reproduces exactly the state of the tree before this change:
    the amplitude arrives as a plain number and the whole frame runs. If that
    still produced a gradient, the graph under test would be coming from
    somewhere other than the cross section.
    """

    sigma = _sigma_tensor(SIGMA_M2, requires_grad=True)
    response = _response_from_sigma(sigma.detach())
    loss = _loss(spike, spec, response)
    assert not loss.requires_grad
    assert sigma.grad is None


# --------------------------------------------------------------------------
# 4. The refusals that keep the new route honest
# --------------------------------------------------------------------------


def test_marking_the_derived_amplitude_as_well_is_refused():
    """``requires_grad=True`` with a tensor sigma names the leaf instead.

    Torch's own message for marking a non-leaf mentions neither this
    constructor nor the law it applies, and the request is a real confusion
    rather than a typo: the caller wants a gradient and is saying so twice.
    """

    with pytest.raises(ValueError) as excinfo:
        ScalarRcsResponse.from_rcs(
            _sigma_tensor(SIGMA_M2, requires_grad=True),
            reference_frequency_hz=geo.REFERENCE_FREQUENCY_HZ,
            requires_grad=True,
        )
    assert "sigma_m2" in str(excinfo.value)
    assert "from_values" in str(excinfo.value)


def test_a_non_scalar_cross_section_is_refused():
    """One complex number per target, broadcast across that target's rows.

    A per-row cross section is an aspect-dependent response and has its own
    owner (``AspectScatterResponse``); accepting a vector here would silently
    broadcast the first element or fail deep inside the join.
    """

    vector = torch.tensor([1.0, 2.0], dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError) as excinfo:
        ScalarRcsResponse.from_rcs(
            vector, reference_frequency_hz=geo.REFERENCE_FREQUENCY_HZ
        )
    assert "0-dim" in str(excinfo.value)


def test_the_host_float_route_keeps_its_negative_refusal():
    """Unchanged behaviour for scalar callers, including the error they get."""

    with pytest.raises(ValueError) as excinfo:
        ScalarRcsResponse.from_rcs(
            -1.0, reference_frequency_hz=geo.REFERENCE_FREQUENCY_HZ
        )
    assert "cannot be negative" in str(excinfo.value)
