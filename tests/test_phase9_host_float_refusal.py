"""Every configuration scalar refuses a tensor, at construction, by name.

The defect this closes is uniform across eight specs and about forty fields:
each one is a host float, each one is read with ``float(...)`` or an arithmetic
expression that ``float`` swallows, and each one used to accept a 0-dim
``requires_grad`` tensor, run the whole frame, and return ``grad = None``. A
missing derivative that looks exactly like a successful optimisation step is
worse than a crash, because nothing in the run reports it.

**Why the rule is "any tensor" and not "any marked tensor".** Two reasons, both
tested below. A tensor that happens not to require grad today is the input that
starts requiring grad tomorrow, at which point the silence comes back; and
``float()`` on a device tensor is a host synchronisation on a per-frame object,
which no spec may hide either. Refusing the TYPE is the only version of the rule
that stays true, and it is cheap because a spec is constructed once.

**No legitimate caller was displaced.** Every one of the 85 spec construction
sites in the tree was scanned before the rule landed, and the whole suite -
default and ``--gpu`` - was run after it: no production module and no test
passes a tensor into any of these fields. The refusal removes a capability
nobody had.

The one deliberate asymmetry is ``ScalarRcsResponse.from_rcs(sigma_m2)``, which
DOES accept a tensor because a radar cross section is scene state rather than a
device or waveform declaration. ``test_phase9_rcs_sigma_ad.py`` owns that cell,
and the last test here pins the two rules against each other so that neither can
drift into the other's territory.
"""

from __future__ import annotations

import math

import pytest
import torch

from witwin.radar.frontend import AdcSpec, AgcSpec, LnaSpec, NoiseSpec, PortSpec, SeedSpec  # noqa: E402
from witwin.radar.policy import require_host_float, require_host_floats  # noqa: E402
from witwin.radar.synthesis.assembly import FmcwSpec, OfdmSpec, PulsedSpec  # noqa: E402

#: A working set of keyword arguments per spec, and the fields that must each
#: refuse a tensor. The valid set is what makes this a test of the refusal
#: rather than of some unrelated validation firing first: every case below
#: constructs successfully with the base arguments and fails ONLY because one
#: field was replaced.
F_REF_HZ = 77.0e9

FMCW_BASE = {
    "num_samples": 32,
    "num_chirps": 3,
    "sample_period_s": 1.0 / 4.4e6,
    "chirp_period_s": 65.0e-6,
    "slope_hz_per_s": 60.012e12,
    "t_start_s": 6.0e-6,
    "reference_frequency_hz": F_REF_HZ,
    "carrier_hz": 0.0,
    "carrier_rate_hz": F_REF_HZ,
    "num_tx": 1,
    "num_rx": 1,
}

OFDM_BASE = {
    "num_subcarriers": 8,
    "num_symbols": 2,
    "subcarrier_spacing_hz": 120.0e3,
    "cyclic_prefix_s": 2.0e-6,
    "reference_frequency_hz": F_REF_HZ,
    "max_expected_delay_s": 1.0e-6,
    "carrier_hz": 0.0,
    "carrier_rate_hz": F_REF_HZ,
}

PULSED_BASE = {
    "num_pulses": 2,
    "num_samples": 32,
    "sample_period_s": 2.0e-8,
    "pri_s": 1.0e-5,
    "range_gate_start_s": 0.0,
    "pulse_kind": "rect",
    "pulse_width_s": 1.0e-7,
    "bandwidth_hz": 1.0e7,
    "reference_frequency_hz": F_REF_HZ,
    "max_expected_delay_rate": 0.0,
    "carrier_hz": 0.0,
    "carrier_rate_hz": F_REF_HZ,
}

NOISE_BASE = {
    "noise_figure_db": 6.0,
    "antenna_temperature_k": 290.0,
    "bandwidth_hz": 5.0e6,
    "phase_noise_dbc_per_hz": -90.0,
    "phase_offset_hz": 1.0e6,
    "phase_sample_rate_hz": 4.4e6,
}

#: ``(owner label, factory, base kwargs, refusing fields)``. The field lists are
#: exhaustive per spec: every scalar the owner declares is here, so a new field
#: added without a decision about its derivative shows up as a missing case.
SPEC_CASES = (
    ("PortSpec", PortSpec, {"reference_impedance_ohm": 50.0}, ("reference_impedance_ohm",)),
    (
        "NoiseSpec",
        NoiseSpec,
        NOISE_BASE,
        (
            "noise_figure_db",
            "antenna_temperature_k",
            "bandwidth_hz",
            "phase_noise_dbc_per_hz",
            "phase_offset_hz",
            "phase_sample_rate_hz",
        ),
    ),
    ("LnaSpec", LnaSpec, {"gain_db": 20.0}, ("gain_db",)),
    ("AgcSpec", AgcSpec, {"target_rms": 1.0}, ("target_rms", "min_gain_db", "max_gain_db")),
    ("AdcSpec", AdcSpec, {"bits": 10, "full_scale": 1.0}, ("bits", "full_scale")),
    (
        "FmcwSpec",
        FmcwSpec,
        FMCW_BASE,
        (
            "num_samples",
            "num_chirps",
            "sample_period_s",
            "chirp_period_s",
            "slope_hz_per_s",
            "t_start_s",
            "reference_frequency_hz",
            "carrier_hz",
            "carrier_rate_hz",
            "num_tx",
            "num_rx",
        ),
    ),
    (
        "OfdmSpec",
        OfdmSpec,
        OFDM_BASE,
        (
            "num_subcarriers",
            "num_symbols",
            "subcarrier_spacing_hz",
            "cyclic_prefix_s",
            "reference_frequency_hz",
            "max_expected_delay_s",
            "carrier_hz",
            "carrier_rate_hz",
        ),
    ),
    (
        "PulsedSpec",
        PulsedSpec,
        PULSED_BASE,
        (
            "num_pulses",
            "num_samples",
            "sample_period_s",
            "pri_s",
            "range_gate_start_s",
            "pulse_width_s",
            "bandwidth_hz",
            "reference_frequency_hz",
            "max_expected_delay_rate",
            "carrier_hz",
            "carrier_rate_hz",
        ),
    ),
)

FIELD_CASES = tuple((label, factory, base, field) for label, factory, base, fields in SPEC_CASES for field in fields)


def _identify(case) -> str:
    return f"{case[0]}.{case[3]}"


def _declared_value(factory, base, field) -> float:
    """The value the working spec actually holds for ``field``.

    Read off the CONSTRUCTED object rather than out of ``base``, because a
    field with a default - ``AgcSpec.min_gain_db``, ``FmcwSpec.num_tx`` -
    is legitimately absent from the base kwargs and must still be covered. A
    zero is replaced by one so that the substituted tensor is a plausible
    value and the refusal is the only reason the construction fails.
    """

    value = getattr(factory(**base), field)
    return float(value) if value else 1.0


# --------------------------------------------------------------------------
# 1. The premise: every base kwarg set actually constructs
# --------------------------------------------------------------------------


@pytest.mark.parametrize("label,factory,base,fields", SPEC_CASES, ids=[case[0] for case in SPEC_CASES])
def test_the_base_specification_still_constructs(label, factory, base, fields):
    """Without this the refusal tests could all be passing for the wrong reason.

    A parametrized "it raises" test is vacuous if the base arguments were
    already invalid: the spec would raise on every case regardless of the
    substituted field, and a deleted refusal would go unnoticed.
    """

    spec = factory(**base)
    assert spec is not None
    for name in fields:
        assert not isinstance(getattr(spec, name), torch.Tensor)


# --------------------------------------------------------------------------
# 2. Every field refuses a marked tensor, and names itself
# --------------------------------------------------------------------------


@pytest.mark.parametrize("label,factory,base,field", FIELD_CASES, ids=[_identify(c) for c in FIELD_CASES])
def test_every_configuration_scalar_refuses_a_marked_tensor(label, factory, base, field):
    """The exact request that used to run a whole frame and return None."""

    marked = torch.tensor(_declared_value(factory, base, field), requires_grad=True)
    with pytest.raises(TypeError) as excinfo:
        factory(**{**base, field: marked})
    message = str(excinfo.value)
    assert f"{label}.{field}" in message
    assert "requires_grad tensor" in message
    # The message must say why, not only that. A bare type error sends the
    # reader looking for a bug in the plumbing rather than for the decision.
    assert "None" in message


@pytest.mark.parametrize("label,factory,base,field", FIELD_CASES, ids=[_identify(c) for c in FIELD_CASES])
def test_every_configuration_scalar_refuses_an_unmarked_tensor_too(label, factory, base, field):
    """The tomorrow case, which is why the rule is on the TYPE.

    A caller who threads an unmarked tensor through a spec today has already
    built the pipe; the day something upstream of it starts requiring grad, the
    silence returns with no code change to blame. Refusing here is what makes
    that impossible rather than unlikely.
    """

    plain = torch.tensor(_declared_value(factory, base, field))
    assert not plain.requires_grad
    with pytest.raises(TypeError) as excinfo:
        factory(**{**base, field: plain})
    assert f"{label}.{field}" in str(excinfo.value)


# --------------------------------------------------------------------------
# 3. The refusal is a REFUSAL: it happens before anything else
# --------------------------------------------------------------------------


def test_the_refusal_precedes_every_other_validation():
    """A tensor is refused even when its VALUE would also be rejected.

    ``FmcwSpec`` refuses a non-positive ``sample_period_s``. Ordering the
    tensor check second would make ``sample_period_s=tensor(-1.0)`` raise a
    ValueError about positivity - true, but it would send the caller to fix the
    sign of a tensor that was never going to be accepted, and it would leave a
    VALID tensor slipping through wherever no range check exists.
    """

    bad = torch.tensor(-1.0, requires_grad=True)
    with pytest.raises(TypeError) as excinfo:
        FmcwSpec(**{**FMCW_BASE, "sample_period_s": bad})
    assert "must be a host float" in str(excinfo.value)


def test_no_partial_specification_survives_a_refusal():
    """Nothing is published: the constructor raises, so no object exists.

    ``__post_init__`` raising means ``__init__`` raises, so there is no
    half-built spec anywhere for a caller to pick up from an exception handler.
    Stated as a test because "fails before a partial result" is the phase's
    acceptance criterion and this is the frontend/synthesis half of it.
    """

    captured = None
    try:
        captured = AgcSpec(target_rms=torch.tensor(1.0, requires_grad=True))
    except TypeError:
        pass
    assert captured is None


def test_a_forward_dual_is_refused_as_well():
    """A forward-only dual is a tensor, so the type rule covers it for free.

    Worth pinning rather than assuming: the ADR-038 route marks nothing, so a
    rule written against ``requires_grad`` would have let a dual through and
    published a spec-independent tangent of exactly zero.
    """

    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(torch.tensor(20.0), torch.tensor(1.0))
        assert not dual.requires_grad
        with pytest.raises(TypeError):
            LnaSpec(gain_db=dual)


# --------------------------------------------------------------------------
# 4. The validator itself
# --------------------------------------------------------------------------


def test_the_validator_passes_every_non_tensor_through():
    """Ints, floats, None and bools are all legal spec values."""

    require_host_floats(
        "Probe", "a reason.", an_int=3, a_float=1.5, a_none=None, a_bool=True, a_numpy_free_scalar=math.pi
    )


def test_the_validator_reports_the_first_offender_in_declaration_order():
    """The message names ONE field, and it is the first one declared.

    Dict iteration order is insertion order, so a spec that lists its fields in
    declaration order gets a stable message. Without this the same broken call
    could blame a different field between runs and two bug reports of the same
    defect would not look alike.
    """

    with pytest.raises(TypeError) as excinfo:
        require_host_floats("Probe", "a reason.", first=torch.tensor(1.0), second=torch.tensor(2.0))
    assert "Probe.first" in str(excinfo.value)
    assert "Probe.second" not in str(excinfo.value)


def test_the_single_field_validator_carries_the_owner_and_the_reason():
    with pytest.raises(TypeError) as excinfo:
        require_host_float("field", torch.tensor(1.0), owner="Owner", reason="Because so.")
    message = str(excinfo.value)
    assert "Owner.field" in message
    assert "Because so." in message


# --------------------------------------------------------------------------
# 5. The boundary against the one supported configuration leaf
# --------------------------------------------------------------------------


def test_a_radar_cross_section_is_deliberately_on_the_other_side_of_this_rule():
    """``sigma_m2`` accepts a tensor while every spec scalar refuses one.

    This is the asymmetry the whole matrix turns on and it is a physical
    distinction, not an oversight: a cross section describes the WORLD, and a
    sample period, a subcarrier spacing or an LNA gain describes the
    INSTRUMENT. Pinning both directions in one test is what stops a later
    tidy-up from making the package uniform in the wrong direction.
    """

    from witwin.radar.scattering import ScalarRcsResponse

    sigma = torch.tensor(3.5, dtype=torch.float32, requires_grad=True)
    response = ScalarRcsResponse.from_rcs(sigma, reference_frequency_hz=F_REF_HZ)
    assert response.amplitude.grad_fn is not None

    with pytest.raises(TypeError):
        LnaSpec(gain_db=torch.tensor(20.0, requires_grad=True))


def test_the_seed_is_refused_by_its_own_older_type_rule():
    """``SeedSpec`` needed no new check and this says why, once.

    ``seed_base`` is an int with an ``isinstance`` guard that predates this
    phase, so a tensor was already refused there. Recording it means the field
    is accounted for rather than merely absent from the tables above.
    """

    with pytest.raises(TypeError):
        SeedSpec(seed_base=torch.tensor(3))
