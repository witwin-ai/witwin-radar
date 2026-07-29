"""An endpoint leaf reaching an OFDM cube and a pulsed train, both modes.

Every end-to-end AD test in the tree from Phase 4 to Phase 8 synthesizes FMCW.
The OFDM and pulsed families are covered at the OPERATOR level - synthetic row
tensors, a float64 oracle, four inputs each - and nowhere else, so "does a Core
leaf reach an OFDM cube" had no test at all. That is not a small gap: the route
from an endpoint position to a cube runs through the Channel consumer's
fixed-topology reevaluation, the native two-way join, the scatter response and
one of three kernels, and only the last of those is shared with the operator
tests. The surveyor probed both chains working and neither was pinned, so any
regression in the first three stages was invisible on two of the three
waveforms.

Six cells: ``{ofdm, pulsed} x {sites, transmitters, receivers}``, each in both
AD modes, each validated against a fourth-order finite difference of the whole
production chain - a fresh reevaluation, a fresh composition and a fresh
synthesis per sample, which is what makes the difference an oracle rather than
a replay of the same graph.

**Why a fourth-order stencil and why these directions.** The loss carries the
77 GHz propagation phase, so it turns a full cycle every 1.95 mm of round-trip
path change and its higher derivatives are large. Second order is not enough:
measured relative disagreement at a 1e-4 m step is 2.8 percent second order
against 0.10 percent fourth order on the OFDM site cell. The perturbation
directions are chosen so that the live components ADD rather than cancel;
with an arbitrary direction the six live components of the OFDM site gradient
cancel to a twentieth of their own magnitude and amplify the difference's noise
by the same factor, which was measured at 2 percent before the directions were
chosen and 0.1 percent after.

Worst measured fourth-order agreement over all six cells, at ``STEP_M``: 0.73
percent. The tolerance is 3 percent, a factor of four.

**One capability boundary is recorded here rather than worked around.** A
forward-mode request whose dual covers only the transmitters is REFUSED by
Radar's dead-tangent guard, because the outbound leg then has no live input and
publishes no ``delay_s`` tangent. The supported shape is one dual level covering
all three endpoint sets, which is exactly what
``witwin.radar.propagation.two_way_duals`` does for the same reason.
The last section pins both halves.
"""

from __future__ import annotations

import pytest
import torch
import torch.autograd.forward_ad as forward_ad

pytest.importorskip("witwin.channel")

from support import fd  # noqa: E402
from support import multi_endpoint_driver as drv  # noqa: E402
from support import waveform_chains as wc  # noqa: E402

pytestmark = pytest.mark.gpu


WAVEFORMS = ("ofdm", "pulsed")
LEAVES = ("sites", "transmitters", "receivers")

#: Metres. See the module docstring for the sweep behind it.
STEP_M = 1.0e-4
FD_RTOL = 3.0e-2

#: The row-wise magnitudes every direction is built from. Two rows because every
#: endpoint set in this fixture has exactly two members, and the two rows differ
#: so that a chain which summed the endpoints instead of indexing them could not
#: pass.
DIRECTION_SCALE = ((0.4, 0.8, 0.0), (0.6, 0.3, 0.0))

#: The SIGN pattern per cell, chosen so the live components add. The zeros are
#: structural rather than cosmetic: ``TX_B`` publishes no rows at all in this
#: fixture, and no endpoint has a live ``z`` component because the whole
#: fixture is planar. Both facts are asserted below instead of being left as
#: an unexplained pattern of zeros here.
DIRECTION_SIGNS = {
    ("ofdm", "sites"): ((-1, -1, 0), (1, 1, 0)),
    ("ofdm", "transmitters"): ((1, -1, 0), (0, 0, 0)),
    ("ofdm", "receivers"): ((1, -1, 0), (1, -1, 0)),
    ("pulsed", "sites"): ((-1, 1, 0), (-1, -1, 0)),
    ("pulsed", "transmitters"): ((-1, 1, 0), (0, 0, 0)),
    ("pulsed", "receivers"): ((1, -1, 0), (-1, 1, 0)),
}

CELLS = tuple((waveform, leaf) for waveform in WAVEFORMS for leaf in LEAVES)


@pytest.fixture(scope="module")
def spike():
    return drv.MultiEndpointSpike()


@pytest.fixture(scope="module")
def specs():
    return {waveform: wc.make_spec(waveform) for waveform in WAVEFORMS}


def _bases(spike) -> dict:
    return {
        "sites": spike.site_tensor(),
        "transmitters": spike.transmitter_tensor(),
        "receivers": spike.receiver_tensor(),
    }


def _direction(waveform: str, leaf: str) -> torch.Tensor:
    signs = torch.tensor(DIRECTION_SIGNS[(waveform, leaf)], dtype=torch.float32, device="cuda")
    scale = torch.tensor(DIRECTION_SCALE, dtype=torch.float32, device="cuda")
    return signs * scale


def _reverse(spike, spec, waveform: str, leaf: str):
    """One reverse pass on ``leaf``; returns the loss and the gradient."""

    base = _bases(spike)[leaf]
    live = base.clone().requires_grad_(True)
    loss = wc.chain_loss(spike, waveform, spec, ad_mode="vjp", **{leaf: live})
    loss.backward()
    return float(loss.detach()), live.grad.detach()


# --------------------------------------------------------------------------
# 1. The premise: these really are two other waveforms, on the real fixture
# --------------------------------------------------------------------------


def test_the_two_cubes_have_their_own_shapes_and_are_not_the_beat_cube(spike, specs):
    """A cheap guard against the whole file testing FMCW three times.

    The three synthesis entries take the same batch, so a dispatch mistake in
    the fixture would produce a perfectly plausible cube and every gradient
    below would still agree with its own finite difference.
    """

    ofdm = wc.synthesize("ofdm", spike.frame(None, drv.make_response(), include_delay_rate=False)[0], specs["ofdm"])
    pulsed = wc.synthesize(
        "pulsed", spike.frame(None, drv.make_response(), include_delay_rate=False)[0], specs["pulsed"]
    )
    pairs = specs["ofdm"].num_subcarriers
    assert tuple(ofdm.shape) == (specs["ofdm"].num_symbols, 4, pairs)
    assert tuple(pulsed.shape) == (specs["pulsed"].num_pulses, 4, specs["pulsed"].num_samples)
    assert ofdm.dtype == torch.complex64 and pulsed.dtype == torch.complex64
    assert float(ofdm.abs().max()) > 0.0 and float(pulsed.abs().max()) > 0.0

    # And they are published in the CHANNEL convention, unconjugated, where the
    # beat cube is the one waveform that is conjugated. A fixture that had
    # dispatched to FMCW would carry the other convention.
    from witwin.radar.synthesis.assembly import BEAT_PHASOR, CHANNEL_PHASOR

    assert specs["ofdm"].phasor == CHANNEL_PHASOR
    assert specs["pulsed"].phasor == CHANNEL_PHASOR
    assert drv.make_spec(num_chirps=2).phasor == BEAT_PHASOR


def test_the_pulsed_fixture_uses_an_lfm_and_says_why(spike, specs):
    """A rectangular pulse would make every pulsed test below vacuous.

    With a rectangle and the production carrier placement the train's dependence
    on ``tau_rt`` is entirely through the support test, so the
    almost-everywhere delay derivative is EXACTLY zero - pinned as a property by
    ``test_phase6_pulsed_ad.py``. A pulsed end-to-end AD test built on one would
    assert that zero equals zero and look exactly like a test of the chain.
    """

    assert specs["pulsed"].pulse_kind == "lfm"
    assert specs["pulsed"].is_linear_fm
    _, gradient = _reverse(spike, specs["pulsed"], "pulsed", "sites")
    assert float(gradient.abs().max()) > 0.0


# --------------------------------------------------------------------------
# 2. Reverse mode against a fourth-order difference of the whole chain
# --------------------------------------------------------------------------


@pytest.mark.parametrize("waveform,leaf", CELLS, ids=[f"{w}-{leaf}" for w, leaf in CELLS])
def test_the_endpoint_gradient_matches_a_fourth_order_difference(spike, specs, waveform, leaf):
    spec = specs[waveform]
    base = _bases(spike)[leaf]
    direction = _direction(waveform, leaf)
    _, gradient = _reverse(spike, spec, waveform, leaf)

    analytic = float((gradient * direction).sum())
    assert abs(analytic) > 0.0

    # The parts add rather than cancel, which is what keeps the difference's
    # relative error at the level the tolerance assumes.
    contributions = (gradient * direction).sum(dim=1)
    live = contributions[contributions.abs() > 0.0]
    assert live.numel() > 0
    assert bool((live > 0.0).all()) or bool((live < 0.0).all())

    samples = {
        offset: float(wc.chain_loss(spike, waveform, spec, **{leaf: base + (offset * STEP_M) * direction}))
        for offset in (-2, -1, 1, 2)
    }
    difference = fd.fourth_order_difference(samples, STEP_M)
    assert fd.relative_error(difference, analytic, floor=1e-30) < FD_RTOL


# --------------------------------------------------------------------------
# 3. Forward mode, against the reverse gradient it must reproduce
# --------------------------------------------------------------------------


@pytest.mark.parametrize("waveform,leaf", CELLS, ids=[f"{w}-{leaf}" for w, leaf in CELLS])
def test_the_forward_tangent_reproduces_the_reverse_directional_derivative(spike, specs, waveform, leaf):
    """``<grad, v>`` from one reverse pass against the jvp along the same ``v``.

    Both modes are exercised on ONE frozen topology - the same compiled scene,
    the same frozen legs, the same frozen join - which is the phase's acceptance
    criterion. The tolerance is tight because these are two evaluations of the
    same derivative rather than two approximations of it: measured agreement is
    exact on the site cells and 3.5e-7 relative at worst on the others.
    """

    spec = specs[waveform]
    bases = _bases(spike)
    direction = _direction(waveform, leaf)
    _, gradient = _reverse(spike, spec, waveform, leaf)
    analytic = float((gradient * direction).sum())

    with forward_ad.dual_level():
        duals = {
            name: forward_ad.make_dual(value, direction if name == leaf else torch.zeros_like(value))
            for name, value in bases.items()
        }
        loss = wc.chain_loss(spike, waveform, spec, ad_mode="jvp", **duals)
        tangent = forward_ad.unpack_dual(loss).tangent
        assert tangent is not None, f"{waveform}/{leaf}: the tangent died"
        measured = float(tangent)

    assert fd.relative_error(measured, analytic, floor=1e-30) < 1e-5


# --------------------------------------------------------------------------
# 4. The structural zeros, which are answers rather than gaps
# --------------------------------------------------------------------------


@pytest.mark.parametrize("waveform", WAVEFORMS)
def test_the_silent_transmitter_has_an_exactly_zero_gradient(spike, specs, waveform):
    """``TX_B`` publishes no rows, so its gradient is exactly zero, not small.

    It sits behind the facet from both sites and its image shares a plane with
    them, so neither a line of sight nor a specular path exists. The empty pair
    segments that follow are the hardest part of the join's contract, and an
    exact zero here is what says the gradient respected them rather than
    smearing one endpoint's contribution across the batch.
    """

    _, gradient = _reverse(spike, specs[waveform], waveform, "transmitters")
    assert float(gradient[0].abs().max()) > 0.0
    assert float(gradient[1].abs().max()) == 0.0


@pytest.mark.parametrize("waveform,leaf", CELLS, ids=[f"{w}-{leaf}" for w, leaf in CELLS])
def test_the_out_of_plane_gradient_is_exactly_zero(spike, specs, waveform, leaf):
    """Every endpoint and the whole facet lie in ``z = 0``, so ``dL/dz`` is zero.

    Exact, because the geometry is symmetric about the plane and a specular path
    stays in it: moving an endpoint out of plane is a second-order change in
    every path length. Asserting EXACT rather than small is the point - a
    float32 chain that leaked a fraction of the in-plane derivative into ``z``
    would produce a small number here and pass an approximate check.
    """

    _, gradient = _reverse(spike, specs[waveform], waveform, leaf)
    assert float(gradient[:, 2].abs().max()) == 0.0


@pytest.mark.parametrize("waveform", WAVEFORMS)
def test_a_detached_endpoint_tensor_carries_no_gradient(spike, specs, waveform):
    """The falsifier: without it the tests above measure an unknown graph."""

    base = _bases(spike)["sites"]
    live = base.clone().requires_grad_(True)
    loss = wc.chain_loss(spike, waveform, specs[waveform], ad_mode="vjp", sites=live.detach())
    assert not loss.requires_grad
    assert live.grad is None


# --------------------------------------------------------------------------
# 5. The forward-mode leg-coverage boundary, recorded
# --------------------------------------------------------------------------


def test_a_transmitter_only_forward_dual_is_refused_by_the_dead_tangent_guard(spike, specs):
    """An honest refusal, and the reason it is not a defect in this file.

    The outbound leg runs sites to receivers. A dual that covers only the
    transmitters leaves that leg with no live input, so its ``delay_s`` carries
    no tangent, and ``channel_consumer._delay_rate`` refuses rather than
    publishing ``delay_rate = 0`` - which is exactly what a correct static scene
    publishes and would therefore be indistinguishable from success.

    The guard is stricter than the physics strictly requires and S1 recorded
    that as an upstream question for a later decision; what matters here is that
    it fails LOUDLY and before a result, which is the behaviour this phase
    demands. The supported shape is the next test.
    """

    base = _bases(spike)["transmitters"]
    with forward_ad.dual_level():
        dual = forward_ad.make_dual(base, torch.ones_like(base))
        with pytest.raises(RuntimeError) as excinfo:
            wc.chain_loss(spike, "ofdm", specs["ofdm"], ad_mode="jvp", transmitters=dual)
    assert "no delay_s tangent" in str(excinfo.value)


def test_one_dual_level_over_all_three_endpoint_sets_is_the_supported_shape(spike, specs):
    """A zero tangent is a statement, and it is the one both legs need.

    Covering all three endpoint sets in ONE ``dual_level`` - the transmitters
    and receivers with a zero tangent when only the sites move - is what
    ``kinematics.two_way_duals`` already does, for the same reason: an inbound
    leg's rate needs the transmitter and the site while the outbound leg's needs
    the site and the receiver, so a round trip whose two ends both move cannot
    be expressed one tensor at a time.

    The resulting tangent is the transmitter's alone, which is asserted against
    the reverse gradient by the parametrized test above; here the statement is
    only that the call is ACCEPTED where the narrow one was refused.
    """

    bases = _bases(spike)
    with forward_ad.dual_level():
        duals = {
            name: forward_ad.make_dual(
                value, torch.ones_like(value) if name == "transmitters" else torch.zeros_like(value)
            )
            for name, value in bases.items()
        }
        loss = wc.chain_loss(spike, "ofdm", specs["ofdm"], ad_mode="jvp", **duals)
        tangent = forward_ad.unpack_dual(loss).tangent
    assert tangent is not None
    assert float(tangent) != 0.0
