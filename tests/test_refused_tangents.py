"""Every unsupported tangent, refused before a cube exists, per waveform.

The Phase-9 acceptance criterion for the unsupported half of the matrix is that
a cell that cannot be answered "fails before any partial result". "Before" is
the load-bearing word and it is not asserted by catching an exception: a stage
that computes a whole frame and then raises has already spent it, and a caller
who catches the exception is holding half an answer.

So every refusal below is driven through the SAME production chain the supported
cells use - compile, freeze, replay, compose, synthesize - with the three
waveform owners replaced by counting stand-ins, and the assertion is that the
count is exactly zero. The instrument is calibrated against the same chain
running normally, so a zero can never be vacuous.

Six refusal families, each a row in the capability matrix:

* an ``ad_mode`` outside the closed vocabulary;
* ``sources.powers_w``, refused in both modes;
* endpoint ``polarizations``, refused in both modes;
* the ``diffraction`` and ``transmission`` components, which cannot be frozen at
  all;
* a waveform spec scalar as a tensor, which is S2's host-float rule seen from
  the chain rather than from the constructor;
* every second-order request at every boundary the chain crosses.

The polarization BASIS is not a Radar-side input at all: ``RadarEndpointSpec``
carries ``stable_ids``, ``positions_m``, ``polarizations`` and ``powers_w`` and
nothing else, so there is no basis tensor a Radar caller could mark. Channel
owns that refusal and its own matrix carries the row.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch
import torch.autograd.forward_ad as forward_ad

pytest.importorskip("witwin.channel")

from support import ad_boundaries
from support import ad_matrix as mx
from support import multi_endpoint_driver as drv
from support import multi_endpoint_geometry as geo
from support import multi_endpoint_world as world
from support import waveform_chains as wc

from witwin.radar.channel import ChannelPropagationAdapter

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def spike():
    return drv.MultiEndpointSpike()


@pytest.fixture(scope="module")
def values(spike):
    return mx.base_values(spike)


class _SynthesisCounter:
    """Counting stand-ins for the three waveform owners.

    Installed on the three owner modules. That is enough BECAUSE ``Radar``
    resolves the owners inside its dispatch method at call time rather than at
    import time, so the patched attribute is the one it reaches - and
    ``test_the_no_cube_instrument_counts_a_real_synthesis`` proves it rather
    than leaving it to inspection of the fixture.
    """

    def __init__(self, monkeypatch):
        self.calls = 0
        from witwin.radar.synthesis import fmcw, ofdm, pulsed

        for owner, name in ((fmcw, "synthesize_fmcw"), (ofdm, "synthesize_ofdm"), (pulsed, "synthesize_pulsed")):
            original = getattr(owner, name)

            def counting(*args, _original=original, **kwargs):
                self.calls += 1
                return _original(*args, **kwargs)

            monkeypatch.setattr(owner, name, counting)


@pytest.fixture
def counter(monkeypatch):
    return _SynthesisCounter(monkeypatch)


def _chain(spike, values, *, ad_mode="none", kind="fmcw"):
    return mx.loss_of(kind, values, ad_mode=ad_mode, spike=spike)


# --------------------------------------------------------------------------
# The instrument, calibrated
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", mx.WAVEFORMS)
def test_the_no_cube_instrument_counts_a_real_synthesis(spike, values, counter, kind):
    """A zero count means nothing unless a working chain makes it nonzero."""

    loss = _chain(spike, values, kind=kind)
    assert float(loss) != 0.0
    assert counter.calls == 1


# --------------------------------------------------------------------------
# The ad_mode vocabulary
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", mx.WAVEFORMS)
def test_an_ad_mode_outside_the_vocabulary_is_refused_before_any_cube(spike, values, counter, kind):
    """A closed vocabulary, and the message names the three legal values.

    ``"reverse"`` rather than nonsense: a caller who knows Torch reaches for
    that word first, and a silent fallback to ``"none"`` would publish a primal
    with no derivative and no complaint.
    """

    with pytest.raises(NotImplementedError) as raised:
        _chain(spike, values, ad_mode="reverse", kind=kind)
    assert "unsupported ad_mode 'reverse'" in str(raised.value)
    assert "['jvp', 'none', 'vjp']" in str(raised.value)
    assert counter.calls == 0


# --------------------------------------------------------------------------
# The primal-only endpoint inputs
# --------------------------------------------------------------------------


def _sources(spike, **overrides):
    spec = world.endpoint_batch(
        [position for _, position in spike.transmitters],
        spike.transmitter_ids,
        power_w=geo.TX_POWER_W,
        device=spike.device,
    )
    return dataclasses.replace(spec, **overrides)


def _replay(spike, sources, *, ad_mode):
    return spike.adapter.reevaluate_slots(
        spike.inbound, sources, spike._site_batch(spike.site_tensor(), role="sink"), slot_count=1, ad_mode=ad_mode
    )


@pytest.mark.parametrize("field", ("powers_w", "polarizations"))
@pytest.mark.parametrize("mode", ("vjp", "jvp"))
def test_a_primal_only_endpoint_input_is_refused_in_both_modes(spike, counter, field, mode):
    """Transmit power and polarization, marked and dualled, both refused.

    The message points at ``capabilities().primal_only_ad_inputs`` rather than
    listing the inputs, because the list is Channel's and a copy of it here
    would be the thing that rots.
    """

    base = {
        "powers_w": torch.full((2,), geo.TX_POWER_W, dtype=torch.float32, device=spike.device),
        "polarizations": torch.tensor([geo.POLARIZATION] * 2, dtype=torch.float32, device=spike.device),
    }[field]

    if mode == "vjp":
        sources = _sources(spike, **{field: base.clone().requires_grad_(True)})
        with pytest.raises(NotImplementedError) as raised:
            _replay(spike, sources, ad_mode="vjp")
    else:
        with forward_ad.dual_level():
            sources = _sources(spike, **{field: forward_ad.make_dual(base.clone(), torch.ones_like(base))})
            with pytest.raises(NotImplementedError) as raised:
                _replay(spike, sources, ad_mode="jvp")

    assert f"sources.{field} is primal-only" in str(raised.value)
    assert "primal_only_ad_inputs" in str(raised.value)
    assert counter.calls == 0


def test_the_endpoint_spec_carries_no_polarization_basis_to_mark(spike):
    """Why there is no basis row on the Radar side, stated as a fact.

    ``RadarEndpointSpec`` has exactly four fields. The transverse basis is
    Channel's, derived inside the consumer, and a Radar caller cannot reach it -
    so the refusal for it belongs to Channel's matrix and not to this one. If a
    basis field ever appears here this test fails, which is the point.
    """

    from witwin.radar.propagation import RadarEndpointSpec

    assert tuple(field.name for field in dataclasses.fields(RadarEndpointSpec)) == (
        "stable_ids",
        "positions_m",
        "polarizations",
        "powers_w",
    )


# --------------------------------------------------------------------------
# The components that cannot be frozen
# --------------------------------------------------------------------------


@pytest.mark.parametrize("component", ("diffraction", "transmission"))
def test_an_unfreezable_component_is_refused_before_any_discovery(spike, counter, component):
    """Item 7's deliverable: a pre-compute refusal, never an implementation.

    Channel's ADR-043 record narrows ``component_ad_modes["diffraction"]`` to
    ``{"none"}``, and the adapter refuses either component at CONSTRUCTION -
    before discovery, before any frozen handle exists, and therefore before any
    question about its derivative can be asked. The dormant ADR-029/030
    capacity artifacts are not reachable from here and no Radar cell exists for
    them.
    """

    with pytest.raises(NotImplementedError) as raised:
        ChannelPropagationAdapter(
            spike.compiled,
            reference_frequency_hz=geo.REFERENCE_FREQUENCY_HZ,
            components=frozenset({component}),
            max_depth=1,
        )
    assert component in str(raised.value)
    assert "cannot be frozen for reevaluation" in str(raised.value)
    assert counter.calls == 0


# --------------------------------------------------------------------------
# A waveform spec scalar, seen from the chain
# --------------------------------------------------------------------------


SPEC_SCALARS = {"fmcw": "slope_hz_per_s", "ofdm": "subcarrier_spacing_hz", "pulsed": "pulse_width_s"}


@pytest.mark.parametrize("kind", mx.WAVEFORMS)
@pytest.mark.parametrize("marked", (True, False))
def test_a_tensor_waveform_scalar_is_refused_before_the_spec_exists(counter, kind, marked):
    """S2's host-float rule from the consumer's side rather than the owner's.

    S2 pins the rule at every ``__post_init__``. What this adds is that the
    refusal lands before a spec object exists at all, so a chain built on one
    cannot have run: there is nothing to synthesize with.
    """

    field = SPEC_SCALARS[kind]
    value = torch.tensor(float(getattr(wc.make_spec(kind), field)))
    if marked:
        value = value.requires_grad_(True)
    with pytest.raises(TypeError) as raised:
        dataclasses.replace(wc.make_spec(kind), **{field: value})
    assert f"{field} must be a host float" in str(raised.value)
    assert counter.calls == 0


# --------------------------------------------------------------------------
# Higher order, at every boundary this chain crosses
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", mx.WAVEFORMS)
def test_a_grad_of_grad_request_through_the_whole_chain_is_refused(spike, values, kind):
    """``create_graph=True`` on the production chain, not on an isolated kernel.

    S3 pins the rule at each of the seven registered boundaries. This asks the
    question the way a caller would - one ``torch.autograd.grad`` over the whole
    compile-to-cube chain - and the refusal has to survive every stage between.
    The owner named in the message is whichever backward the chain reaches
    first, which is a property of the graph rather than of this test, so the
    assertion is on the rule's wording.
    """

    live = mx.marked(values, ("sites",))
    loss = mx.loss_of(kind, live, ad_mode="vjp", spike=spike)
    with pytest.raises(NotImplementedError) as raised:
        torch.autograd.grad(loss, live["sites"], create_graph=True)
    assert "is first-order only" in str(raised.value)
    assert "create_graph=True" in str(raised.value)


@pytest.mark.parametrize("kind", mx.WAVEFORMS)
def test_the_first_order_request_over_the_same_chain_still_works(spike, values, kind):
    """Over-refusing is the opposite mistake and just as easy to make."""

    live = mx.marked(values, ("sites",))
    loss = mx.loss_of(kind, live, ad_mode="vjp", spike=spike)
    (gradient,) = torch.autograd.grad(loss, live["sites"])
    assert float(gradient.abs().sum()) > 0.0


@pytest.mark.parametrize("name", ad_boundaries.BOUNDARY_NAMES)
def test_a_cotangent_carrying_a_forward_tangent_is_refused_at_every_boundary(name):
    """Forward-over-reverse, at each of the seven registered boundaries.

    S3 owns this rule; what is added here is that the matrix's per-waveform
    chains all terminate at boundaries that carry it, so a mixed second
    derivative cannot arrive through any of them. Without the guard the value
    comes back correct and its tangent is ``None`` - an exact zero with no
    error, which is the worst single result either survey found.
    """

    boundary = ad_boundaries.boundary(name)
    leaf = boundary.leaf.clone().requires_grad_(True)
    output = boundary.loss(leaf)
    with forward_ad.dual_level():
        seed = torch.ones_like(output)
        dual = forward_ad.make_dual(torch.ones_like(output), seed)
        with pytest.raises(NotImplementedError) as raised:
            torch.autograd.grad(output, leaf, grad_outputs=dual)
    assert "is first-order only" in str(raised.value)
    assert "a forward tangent" in str(raised.value)
