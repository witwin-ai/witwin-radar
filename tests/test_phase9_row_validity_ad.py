"""A row that stops existing is a complete answer, and its gradient is zero.

Phase 9's fourth constraint: a reflection row that stops existing at the
perturbed endpoints publishes ``row_valid=False`` and contributes EXACT zero to
every gradient - a complete answer, not a failure. ``test_phase7_invalidation.py``
proved that for the LEG payload when a wall moved. What it did not touch is the
question this file exists for: what happens to the GRADIENT, on the composed
rows, through a synthesized cube, when the endpoints move instead of the world.

Everything below runs on ONE frozen topology - one compiled scene, one pair of
frozen legs, one frozen join - and moves only the site the replay was frozen
for. Two configurations:

* ``MOVED_SITES`` slides ``SITE_P`` 1 m along ``-y``. Its specular point to
  ``RX_B`` walks off the 1.2 m facet, so exactly two of the eleven composed rows
  stop existing and nine survive, including six of ``SITE_P``'s own. That is
  the interesting case: a site whose gradient is still live while two of its
  rows are dead.
* ``OCCLUDED_SITES`` puts ``SITE_P`` at ``x = 6``, behind the wall. Every one of
  its rows dies. The replay answers with exact zeros and an exactly zero
  gradient; a FRESH freeze at the same geometry cannot be built at all, because
  ``TwoWayComposer.freeze`` refuses a site with no outbound row. Under-reporting
  with a complete published answer is the accepted behaviour and refusing to
  freeze is the accepted alternative; a stale answer is neither.

**The strongest statement in the file** is that the frozen 11-row replay and a
freshly discovered 9-row topology at the same geometry produce a BITWISE
identical site gradient. That is the exact sense in which the dead rows changed
nothing about the live ones - not "small", not "close", the same bits.

Finite differences are not used here. Every oracle is an exact zero, a bitwise
comparison against an independently discovered topology, or a refusal.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch
import torch.autograd.forward_ad as forward_ad

pytest.importorskip("witwin.channel")

from support import ad_matrix as mx  # noqa: E402
from support import multi_endpoint_driver as drv  # noqa: E402
from support import multi_endpoint_geometry as geo  # noqa: E402
from support import waveform_chains as wc  # noqa: E402
from support.synthesis_batch import to_synthesis  # noqa: E402

pytestmark = pytest.mark.gpu


#: Metres along ``-y``. Swept over the fixture: at 1 m exactly two composed rows
#: die and nine survive; at 3 m five die; at ``x + 4`` the whole site goes.
PARTIAL_DEATH_SHIFT_M = 1.0

#: The two composed rows that stop existing at ``MOVED_SITES``, by identity.
#: Both are ``SITE_P -> RX_B`` through the wall, and both are OUTBOUND
#: reflections: the outbound specular point leaves the facet while every
#: inbound one stays on it.
DEAD_ROW_KEYS = (
    (geo.TX_A_STABLE_ID, geo.SITE_P_STABLE_ID, geo.RX_B_STABLE_ID, "los", "reflection"),
    (geo.TX_A_STABLE_ID, geo.SITE_P_STABLE_ID, geo.RX_B_STABLE_ID, "reflection", "reflection"),
)

#: A payload no valid row could produce, written into the dead rows to prove the
#: waveform kernels gate on ``row_valid`` rather than merely multiplying by a
#: weight that happens to be zero.
POISON_TRANSFER = complex(1.0e3, 5.0e2)
POISON_DELAY_S = 3.0e-8


@pytest.fixture(scope="module")
def spike():
    return drv.MultiEndpointSpike()


@pytest.fixture(scope="module")
def values(spike):
    return mx.base_values(spike)


def _shifted(values, *, dy=0.0, x=None):
    sites = values["sites"].clone()
    sites[0, 1] = sites[0, 1] + dy
    if x is not None:
        sites[0, 0] = x
    moved = dict(values)
    moved["sites"] = sites
    return moved


@pytest.fixture(scope="module")
def moved(values):
    return _shifted(values, dy=-PARTIAL_DEATH_SHIFT_M)


@pytest.fixture(scope="module")
def occluded(values):
    return _shifted(values, x=6.0)


def _reverse(spike, kind, values):
    live = mx.marked(values, ("sites",))
    composed = mx.replay(spike, live, ad_mode="vjp")
    cube = wc.synthesize(kind, composed, wc.make_spec(kind))
    cube.abs().square().sum().backward()
    return live["sites"].grad.detach(), composed


# --------------------------------------------------------------------------
# 1. The premise: which rows die, and that the rest do not
# --------------------------------------------------------------------------


def test_moving_one_site_kills_exactly_two_rows_and_keeps_nine(spike, moved):
    """The configuration every test below leans on, named by row identity.

    Nine live rows including six of the moved site's own. A configuration that
    killed the whole site - which ``OCCLUDED_SITES`` does deliberately - could
    not distinguish "the dead rows contribute zero" from "this site contributes
    zero".
    """

    composed = mx.replay(spike, moved)
    keys = drv.composed_keys(spike, composed)
    valid = composed.row_valid.tolist()
    dead = tuple(key for key, alive in zip(keys, valid, strict=True) if not alive)
    live = [key for key, alive in zip(keys, valid, strict=True) if alive]

    assert composed.path_count == 11
    assert dead == DEAD_ROW_KEYS, dead
    assert len(live) == 9
    assert sum(1 for key in live if key[1] == geo.SITE_P_STABLE_ID) == 6, live


def test_the_base_configuration_has_no_dead_row_at_all(spike, values):
    """The falsifier for every "dead" assertion below.

    Without it, a fixture in which nothing was ever alive would satisfy the
    exact-zero statements trivially.
    """

    composed = mx.replay(spike, values)
    assert bool(composed.row_valid.all())
    assert float(composed.complex_transfer_ref.abs().min()) > 0.0


# --------------------------------------------------------------------------
# 2. A dead row is inert, on both channels
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ad_mode", ("none", "vjp"))
def test_a_dead_row_publishes_an_exactly_zero_payload(spike, moved, ad_mode):
    """Exact zeros on the delay and the transfer, and a live payload elsewhere.

    Exact rather than small. A row whose payload is merely tiny is a row that
    was computed at a geometry the topology no longer describes and then scaled;
    a row that is exactly zero was published as absent.
    """

    live = mx.marked(moved, ("sites",)) if ad_mode == "vjp" else moved
    composed = mx.replay(spike, live, ad_mode=ad_mode)
    dead = ~composed.row_valid
    transfer = composed.complex_transfer_ref.detach()
    delay = composed.total_delay_s.detach()

    assert bool(dead.any()) and not bool(dead.all())
    assert torch.equal(transfer[dead], torch.zeros_like(transfer[dead]))
    assert torch.equal(delay[dead], torch.zeros_like(delay[dead]))
    assert float(transfer[~dead].abs().min()) > 0.0
    assert float(delay[~dead].min()) > 0.0


@pytest.mark.parametrize("kind", mx.WAVEFORMS)
def test_a_dead_row_carries_an_exactly_zero_forward_tangent(spike, moved, kind):
    """The forbidden middle: valid-and-tangent-free, or dead-and-tangent-bearing.

    ``test_phase7_invalidation.py`` pins this for the LEG under a moving world.
    Here the endpoints move instead, the rows are the COMPOSED ones, and the
    tangent is followed all the way into the synthesized cube - so a stage
    between the join and the kernel that lost the mask would show up.
    """

    with forward_ad.dual_level():
        duals = dict(moved)
        # The SITES carry the tangent and the two ends carry an explicit zero.
        # Seeding all three with ones would be a RIGID TRANSLATION of every
        # endpoint, which leaves a line-of-sight distance exactly unchanged and
        # publishes a legitimately zero tangent on four of the nine live rows -
        # a fixture that would make "every live row carries a tangent" false for
        # a reason that has nothing to do with validity.
        for name in ("sites", "transmitters", "receivers"):
            seed = torch.ones_like(moved[name]) if name == "sites" else torch.zeros_like(moved[name])
            duals[name] = forward_ad.make_dual(moved[name].clone(), seed)
        composed = mx.replay(spike, duals, ad_mode="jvp")
        dead = ~composed.row_valid
        for tensor in (composed.total_delay_s, composed.complex_transfer_ref):
            tangent = forward_ad.unpack_dual(tensor).tangent
            assert tangent is not None
            assert torch.equal(tangent[dead], torch.zeros_like(tangent[dead]))
            assert float(tangent[~dead].abs().min()) > 0.0
        cube = wc.synthesize(kind, composed, wc.make_spec(kind))
        assert forward_ad.unpack_dual(cube).tangent is not None


# --------------------------------------------------------------------------
# 3. A dead row contributes exactly zero to every gradient
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", mx.WAVEFORMS)
def test_the_dead_rows_contribute_exactly_zero_to_the_gradient(spike, moved, kind):
    """Masking the loss to the live rows changes the gradient by zero bits.

    Three statements, and the third is the one that could fail on its own: the
    loss taken over the DEAD rows alone has an exactly zero gradient, so the
    zero above is a property of the rows rather than of the mask that hid them.
    """

    live = mx.marked(moved, ("sites",))
    composed = mx.replay(spike, live, ad_mode="vjp")
    transfer = composed.complex_transfer_ref
    valid = composed.row_valid
    weight = transfer.abs().square()

    (whole,) = torch.autograd.grad(weight.sum(), live["sites"], retain_graph=True)
    (masked,) = torch.autograd.grad((weight * valid).sum(), live["sites"], retain_graph=True)
    (dead_only,) = torch.autograd.grad(
        (weight * (~valid).to(weight.dtype)).sum(), live["sites"], retain_graph=True, allow_unused=True
    )

    assert float(whole.abs().max()) > 0.0
    assert torch.equal(whole, masked)
    assert dead_only is None or torch.equal(dead_only, torch.zeros_like(whole))


@pytest.mark.parametrize("kind", mx.WAVEFORMS)
def test_a_poisoned_dead_row_cannot_change_the_cube(spike, moved, kind):
    """The waveform kernels gate on ``row_valid``, not on a zero weight.

    Overwrite the dead rows' transfer with a value four orders of magnitude
    above every live one and their delay with a plausible one, then synthesize.
    The cube must be BITWISE what it was. Without this the exact zeros above
    would only say that a zero weight contributes nothing - which is true of any
    kernel and says nothing about the mask.
    """

    composed = mx.replay(spike, moved)
    batch = to_synthesis(composed)
    valid = composed.row_valid
    poisoned = dataclasses.replace(
        batch,
        complex_transfer_ref=torch.where(
            valid, batch.complex_transfer_ref, torch.full_like(batch.complex_transfer_ref, POISON_TRANSFER)
        ),
        total_delay_s=torch.where(valid, batch.total_delay_s, torch.full_like(batch.total_delay_s, POISON_DELAY_S)),
    )
    spec = wc.make_spec(kind)
    reference = wc.synthesize(kind, composed, spec)
    from witwin.radar.synthesis import synthesize_fmcw, synthesize_ofdm, synthesize_pulsed

    owner = {"fmcw": synthesize_fmcw, "ofdm": synthesize_ofdm, "pulsed": synthesize_pulsed}[kind]
    assert torch.equal(owner(poisoned, spec), reference)
    # And the poison really was large enough to be seen: applied to a LIVE row
    # the same value moves the cube.
    all_poisoned = dataclasses.replace(
        batch, complex_transfer_ref=torch.full_like(batch.complex_transfer_ref, POISON_TRANSFER)
    )
    assert not torch.equal(owner(all_poisoned, spec), reference)


# --------------------------------------------------------------------------
# 4. The surviving rows are exactly the rows a fresh discovery finds
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", mx.WAVEFORMS)
def test_the_frozen_replay_and_a_fresh_discovery_agree_bit_for_bit(spike, moved, kind):
    """Nine dead-carrying rows against nine freshly discovered ones.

    The frozen topology publishes eleven rows of which two are dead; a topology
    discovered AT the moved geometry has nine rows and no dead ones. If the dead
    rows changed anything about the live ones - a reduction that included them,
    a pair segmentation that counted them, a cotangent that leaked into them -
    the two gradients would differ. They do not differ approximately; they are
    the same bits.

    This is also the statement that no row was BORN: the two identity lists are
    compared as ordered sequences, so an extra row on either side fails.
    """

    fresh = drv.MultiEndpointSpike(
        compiled=spike.compiled,
        sites=tuple((stable_id, tuple(moved["sites"][row].tolist())) for row, stable_id in enumerate(spike.site_ids)),
    )
    frozen_keys = drv.composed_keys(spike, mx.replay(spike, moved))
    fresh_composed = mx.replay(fresh, moved)
    assert bool(fresh_composed.row_valid.all())
    assert drv.composed_keys(fresh, fresh_composed) == [
        key for key, alive in zip(frozen_keys, mx.replay(spike, moved).row_valid.tolist(), strict=True) if alive
    ]

    frozen_gradient, _ = _reverse(spike, kind, moved)
    fresh_gradient, _ = _reverse(fresh, kind, moved)
    assert float(frozen_gradient.abs().max()) > 0.0
    assert torch.equal(frozen_gradient, fresh_gradient)


# --------------------------------------------------------------------------
# 5. The selection boundary: a whole site disappears
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", mx.WAVEFORMS)
def test_a_fully_occluded_site_answers_with_an_exactly_zero_gradient(spike, occluded, kind):
    """Past the selection boundary the replay still answers, completely.

    Every ``SITE_P`` row is gone and its gradient is EXACTLY zero - not a small
    number, not a stale one carried over from the geometry the topology was
    frozen at. ``SITE_Q``'s gradient is unaffected and stays live, which is what
    says the zero is about the rows rather than about the call failing quietly.
    """

    gradient, composed = _reverse(spike, kind, occluded)
    keys = drv.composed_keys(spike, composed)
    valid = composed.row_valid.tolist()
    surviving_sites = {key[1] for key, alive in zip(keys, valid, strict=True) if alive}
    assert surviving_sites == {geo.SITE_Q_STABLE_ID}
    assert torch.equal(gradient[0], torch.zeros_like(gradient[0]))
    assert float(gradient[1].abs().max()) > 0.0


def test_a_fresh_freeze_at_the_occluded_geometry_refuses_instead(spike, occluded):
    """The alternative, and it is a refusal rather than a smaller answer.

    A frozen replay under-reports with a complete published answer; a fresh
    discovery at the same geometry cannot even be composed, because the join
    refuses a declared site with no outbound row. Both are acceptable Phase-9
    behaviours and they are the ONLY two: a stale answer is neither.
    """

    with pytest.raises(ValueError, match="no outbound leg row"):
        drv.MultiEndpointSpike(
            compiled=spike.compiled,
            sites=tuple(
                (stable_id, tuple(occluded["sites"][row].tolist())) for row, stable_id in enumerate(spike.site_ids)
            ),
        )


@pytest.mark.parametrize("kind", mx.WAVEFORMS)
def test_the_replay_never_answers_with_the_geometry_it_was_frozen_at(spike, values, moved, kind):
    """The stale-answer class, ruled out directly.

    The moved frame must differ from the frozen-at frame in the surviving rows'
    numbers as well as in the dead rows' validity. A replay that had quietly
    returned its discovery-time answer would pass every exact-zero test above
    and fail here.
    """

    base_composed = mx.replay(spike, values)
    moved_composed = mx.replay(spike, moved)
    alive = moved_composed.row_valid
    assert not torch.equal(moved_composed.total_delay_s[alive], base_composed.total_delay_s[alive])
    base_cube = wc.synthesize(kind, base_composed, wc.make_spec(kind))
    moved_cube = wc.synthesize(kind, moved_composed, wc.make_spec(kind))
    assert not torch.equal(moved_cube, base_cube)
