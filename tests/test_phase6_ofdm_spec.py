"""``OfdmSpec``: the OFDM grid, its derived quantities, and its refusals.

Every assertion here runs on the CPU. That is deliberate and it matches
``test_phase6_synthesis_contract``: the cyclic-prefix contract and the
carrier-home rule are the part of OFDM that decides whether a cube is wrong by a
factor of ten thousand, and a check that needs a GPU is a check that does not
run in the default suite.

Reference grid, the one the kernel tests use, restated so the numbers can be
recomputed by hand::

    f_ref = 77 GHz     df    = 120 kHz     N_sc = 64
    T_cp  = 2 us       L     = 32
    T_u   = 1 / df      = 8.3333 us
    T_sym = T_u + T_cp  = 10.3333 us
    T_s   = 1 / (N_sc df) = 130.208 ns
"""

from __future__ import annotations

import ast
import dataclasses
import pathlib

import pytest

from witwin.radar.synthesis.assembly import (
    CHANNEL_PHASOR,
    CHANNEL_TIME_DEPENDENCE,
    SPEED_OF_LIGHT_M_PER_S,
    SUBCARRIER_ORIGIN_F_REF_AT_N0,
    FmcwSpec,
    OfdmSpec,
    WaveformSpecProtocol,
    require_single_carrier_home,
)

C0 = SPEED_OF_LIGHT_M_PER_S
F_REF = 77.0e9
DF_HZ = 120.0e3
NUM_SUBCARRIERS = 64
CYCLIC_PREFIX_S = 2.0e-6
NUM_SYMBOLS = 32
MAX_DELAY_S = 1.0e-6


def _spec(**overrides) -> OfdmSpec:
    fields = {
        "num_subcarriers": NUM_SUBCARRIERS,
        "num_symbols": NUM_SYMBOLS,
        "subcarrier_spacing_hz": DF_HZ,
        "cyclic_prefix_s": CYCLIC_PREFIX_S,
        "reference_frequency_hz": F_REF,
        "max_expected_delay_s": MAX_DELAY_S,
        "carrier_hz": 0.0,
        "carrier_rate_hz": F_REF,
    }
    fields.update(overrides)
    return OfdmSpec(**fields)


# --------------------------------------------------------------------------
# Derived quantities, one assertion each
# --------------------------------------------------------------------------


def test_the_useful_symbol_time_is_the_reciprocal_spacing():
    assert _spec().useful_symbol_time_s == pytest.approx(1.0 / DF_HZ, rel=1e-12)
    assert _spec().useful_symbol_time_s == pytest.approx(8.333333333333334e-6, rel=1e-12)


def test_the_symbol_period_includes_the_cyclic_prefix():
    """``T_sym = T_u + T_cp``, and the prefix is not optional in it.

    The CP does not change the CFR closed form, but slow time is sampled once
    per SYMBOL, prefix included. Dropping it from ``T_sym`` would overstate the
    unambiguous velocity by 24% at this grid and understate every measured
    Doppler slope by the same factor.
    """

    spec = _spec()
    assert spec.symbol_period_s == pytest.approx(1.0 / DF_HZ + CYCLIC_PREFIX_S, rel=1e-12)
    assert spec.symbol_period_s == pytest.approx(1.0333333333333333e-5, rel=1e-12)
    assert spec.symbol_period_s > spec.useful_symbol_time_s


def test_the_waveform_sample_period_is_the_reciprocal_bandwidth():
    spec = _spec()
    assert spec.occupied_bandwidth_hz == pytest.approx(NUM_SUBCARRIERS * DF_HZ, rel=1e-12)
    assert spec.waveform_sample_period_s == pytest.approx(1.302083333333333e-7, rel=1e-12)
    assert spec.delay_resolution_s == spec.waveform_sample_period_s


def test_the_range_resolution_halves_the_delay_resolution():
    """``c0 / (2 N_sc df)``. The factor of two is the round trip.

    Reading a range resolution straight off ``c0 * T_s`` is exactly where that
    factor goes missing, so the two are asserted against each other rather than
    each against its own literal.
    """

    spec = _spec()
    assert spec.range_resolution_m == pytest.approx(C0 / (2.0 * NUM_SUBCARRIERS * DF_HZ), rel=1e-12)
    assert spec.range_resolution_m == pytest.approx(0.5 * C0 * spec.waveform_sample_period_s, rel=1e-12)
    assert spec.range_resolution_m == pytest.approx(19.517738151041666, rel=1e-12)


def test_the_unambiguous_delay_is_one_useful_symbol_time():
    spec = _spec()
    assert spec.max_unambiguous_delay_s == spec.useful_symbol_time_s
    assert spec.max_unambiguous_delay_s == pytest.approx(
        spec.num_subcarriers * spec.waveform_sample_period_s, rel=1e-12
    )


def test_the_unambiguous_speed_is_the_closed_form(monkeypatch):
    """``c0 / (4 f_ref T_sym)``, equivalently ``lambda / (4 T_sym)``.

    Both forms are asserted because they are the two ways this bound is written
    in the literature and a discrepancy between them would mean the spec's
    wavelength and its reference frequency had drifted apart.
    """

    spec = _spec()
    assert spec.max_unambiguous_speed_mps == pytest.approx(C0 / (4.0 * F_REF * spec.symbol_period_s), rel=1e-12)
    assert spec.max_unambiguous_speed_mps == pytest.approx(spec.wavelength_m / (4.0 * spec.symbol_period_s), rel=1e-12)
    assert spec.max_unambiguous_speed_mps == pytest.approx(94.19536803519063, rel=1e-12)


def test_the_subcarrier_grid_is_pinned_to_the_reference_frequency():
    spec = _spec()
    assert spec.subcarrier_origin == SUBCARRIER_ORIGIN_F_REF_AT_N0
    assert spec.subcarrier_frequency_hz(0) == F_REF
    assert spec.subcarrier_frequency_hz(NUM_SUBCARRIERS - 1) == pytest.approx(
        F_REF + (NUM_SUBCARRIERS - 1) * DF_HZ, rel=1e-15
    )


def test_the_subcarrier_phase_step_is_negative_and_names_the_delay():
    """The Channel convention, in the one place a reader will look for it.

    ``-2 pi df tau``: the sign is what says this cube is not conjugated.
    """

    spec = _spec()
    tau = 2.0 * 3.7 / C0
    assert spec.subcarrier_phase_step_rad(tau) < 0.0
    assert spec.subcarrier_phase_step_rad(tau) == pytest.approx(-2.0 * 3.141592653589793 * DF_HZ * tau, rel=1e-12)
    assert spec.cir_peak_sample(tau) == pytest.approx(tau * NUM_SUBCARRIERS * DF_HZ, rel=1e-12)


def test_the_published_convention_is_carried_as_data():
    """Not documentation: a consumer reads it off the spec.

    OFDM and FMCW publish different conventions, so "which one is this" has to
    be answerable without knowing which module produced the cube.
    """

    assert OfdmSpec.phasor == CHANNEL_PHASOR == "exp(-j*k*d)"
    assert OfdmSpec.time_dependence == CHANNEL_TIME_DEPENDENCE
    assert OfdmSpec.applies_spreading is False
    assert isinstance(_spec(), WaveformSpecProtocol)


# --------------------------------------------------------------------------
# T2.7  the cyclic-prefix contract, fail-loud
# --------------------------------------------------------------------------


def _batch():
    """The smallest valid frozen-weight batch this spec can be checked against."""

    import torch

    from witwin.radar.paths import RadarPathTopology
    from witwin.radar.synthesis import SlowTimeMode, SynthesisPathBatch

    zeros = torch.zeros(1, dtype=torch.int64)
    return SynthesisPathBatch(
        sensor_pair_count=1,
        path_count=1,
        sensor_pair_index=torch.zeros(1, dtype=torch.int64),
        pair_offsets=torch.tensor([0, 1], dtype=torch.int64),
        total_delay_s=torch.full((1,), 2.4683743e-8, dtype=torch.float32),
        delay_rate=torch.zeros(1, dtype=torch.float32),
        complex_transfer_ref=torch.ones(1, dtype=torch.complex64),
        reference_frequency_hz=F_REF,
        frequency_response=None,
        frequency_offsets_hz=None,
        topology=RadarPathTopology(zeros, zeros, zeros, zeros, zeros),
        row_valid=None,
        join_mode="multipath",
        weight_includes_reference_phase=True,
        weight_includes_spreading=True,
        weight_includes_tx_power=True,
        slow_time_mode=SlowTimeMode.FROZEN_WEIGHT_WITH_CARRIER_RATE,
    )


@pytest.mark.parametrize("delay_s", [CYCLIC_PREFIX_S, 1.05 * CYCLIC_PREFIX_S])
def test_an_echo_window_wider_than_the_cyclic_prefix_is_refused(delay_s):
    """Fail loud, naming ``cyclic_prefix_s``, with no clamped mode to fall to.

    Outside the CP window ``Y / X`` is no longer ``exp(-j 2 pi n df tau)``: the
    response gains an inter-symbol term the closed form does not have. The
    result would still be a smooth, plausible-looking frequency response, which
    is why this is a refusal rather than a warning.

    The boundary itself is refused: at ``max_expected_delay = T_cp`` the last
    echo arrives exactly as the prefix ends, which is already outside the strict
    inequality the single-tap form needs.
    """

    from witwin.radar.synthesis.assembly import require_ofdm_compatible

    spec = _spec(max_expected_delay_s=delay_s)
    with pytest.raises(ValueError, match="cyclic_prefix_s"):
        require_ofdm_compatible(_batch(), spec)


def test_an_echo_window_inside_the_cyclic_prefix_is_accepted():
    from witwin.radar.synthesis.assembly import require_ofdm_compatible

    require_ofdm_compatible(_batch(), _spec(max_expected_delay_s=0.999 * CYCLIC_PREFIX_S))


def test_there_is_no_clamping_path_in_the_ofdm_contract():
    """The absence, asserted rather than assumed.

    A clamp would turn the refusal above into a silently reduced-accuracy cube.
    Scanned with the AST rather than as text: the docstring says the words
    "clamp" and "reduced-accuracy mode" on purpose, to state the rule, and a
    text scan would forbid the explanation along with the act.
    """

    module = pathlib.Path(__file__).resolve().parents[1] / "witwin/radar/synthesis/assembly.py"
    tree = ast.parse(module.read_text(encoding="utf-8"))
    function = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "require_ofdm_compatible"
    )

    called = {
        node.func.id if isinstance(node.func, ast.Name) else node.func.attr
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name | ast.Attribute)
    }
    for forbidden in ("clamp", "clip", "min", "max", "warn", "replace"):
        assert forbidden not in called, forbidden

    # The only two outcomes are "return None" and "raise". Nothing is returned,
    # nothing is mutated, and nothing is written back onto the spec.
    assert not any(isinstance(node, ast.Return) and node.value is not None for node in ast.walk(function))
    assert not any(isinstance(node, ast.Assign | ast.AugAssign) for node in ast.walk(function))
    # Three refusals, counted rather than described, so that a fourth has to be
    # added here deliberately: the foreign-spec TypeError, the wideband
    # column-count mismatch, and the cyclic-prefix bound. The count went from
    # two to three in Phase 8 when the band arrived; it is a count of REFUSALS,
    # and every one of them is still all-or-nothing.
    raises = [node for node in ast.walk(function) if isinstance(node, ast.Raise)]
    assert len(raises) == 3, len(raises)


def test_a_zero_length_cyclic_prefix_is_refused():
    with pytest.raises(ValueError, match="cyclic_prefix_s"):
        _spec(cyclic_prefix_s=0.0)


def test_the_cyclic_prefix_check_reads_configuration_and_never_a_tensor():
    """The bound is a CONFIGURED window, not a measured maximum delay.

    A measured ``max(tau_k)`` would be a per-frame device-to-host transfer -
    exactly what the fixed-topology capability exists to avoid - so the check
    must be expressible without the batch's tensors at all. Asserted by giving
    it a batch whose actual delay violates the bound while the configured window
    does not: the check passes, because it is not looking at the tensor.
    """

    import torch

    from witwin.radar.synthesis.assembly import require_ofdm_compatible

    batch = _batch()
    long_echo = dataclasses.replace(batch, total_delay_s=torch.full((1,), 5.0e-6, dtype=torch.float32))
    require_ofdm_compatible(long_echo, _spec(max_expected_delay_s=1.0e-6))


# --------------------------------------------------------------------------
# The carrier-home rule, shared with FMCW rather than restated
# --------------------------------------------------------------------------


def test_naming_the_carrier_in_both_homes_is_refused():
    with pytest.raises(ValueError, match="double counts"):
        _spec(carrier_hz=F_REF, carrier_rate_hz=F_REF)


def test_both_supported_carrier_homes_construct():
    assert _spec(carrier_hz=0.0, carrier_rate_hz=F_REF).carrier_rate_hz == F_REF
    assert _spec(carrier_hz=F_REF, carrier_rate_hz=0.0).carrier_hz == F_REF


def test_the_carrier_home_rule_is_one_helper_and_not_two_copies():
    """One rule, one implementation, checked structurally.

    Two specs that each spelled the check out would be two places to change and
    one place to forget, and the failure mode of forgetting is a cube that is
    wrong by a constant phase per symbol - invisible in magnitude. The AST scan
    asserts both ``__post_init__`` bodies call the shared helper and that
    neither raises a carrier error of its own.
    """

    module = pathlib.Path(__file__).resolve().parents[1] / "witwin/radar/synthesis/assembly.py"
    tree = ast.parse(module.read_text(encoding="utf-8"))
    for spec_name in ("FmcwSpec", "OfdmSpec"):
        cls = next(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == spec_name)
        post_init = next(
            node for node in ast.walk(cls) if isinstance(node, ast.FunctionDef) and node.name == "__post_init__"
        )
        calls = {
            node.func.id
            for node in ast.walk(post_init)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "require_single_carrier_home" in calls, spec_name

    # And the two produce byte-identical messages, because it is one message.
    messages = []
    for spec in (
        lambda: FmcwSpec(
            num_samples=4,
            num_chirps=1,
            sample_period_s=1.0e-7,
            chirp_period_s=1.0e-5,
            slope_hz_per_s=1.0e13,
            t_start_s=0.0,
            reference_frequency_hz=F_REF,
            carrier_hz=F_REF,
            carrier_rate_hz=F_REF,
        ),
        lambda: _spec(carrier_hz=F_REF, carrier_rate_hz=F_REF),
    ):
        with pytest.raises(ValueError) as raised:
            spec()
        messages.append(str(raised.value))
    assert messages[0] == messages[1]


def test_the_helper_accepts_every_single_home_combination():
    require_single_carrier_home(0.0, 0.0)
    require_single_carrier_home(F_REF, 0.0)
    require_single_carrier_home(0.0, F_REF)


# --------------------------------------------------------------------------
# Field validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"num_subcarriers": 0}, "num_subcarriers"),
        ({"num_symbols": 0}, "num_symbols"),
        ({"subcarrier_spacing_hz": 0.0}, "subcarrier_spacing_hz"),
        ({"max_expected_delay_s": -1.0e-9}, "max_expected_delay_s"),
        ({"reference_frequency_hz": 0.0}, "reference_frequency_hz"),
        ({"subcarrier_origin": "centred"}, "subcarrier_origin"),
    ],
)
def test_a_malformed_grid_is_refused_at_construction(overrides, match):
    with pytest.raises(ValueError, match=match):
        _spec(**overrides)


def test_a_centred_band_is_refused_rather_than_relabelled():
    """A centred band is a different frequency grid, not a naming choice.

    With ``n`` running ``-N/2 .. N/2-1`` the kernel's ``f_sub = n * df`` term
    would have to be signed, and ``H[0][p][0]`` would no longer be ``C_rt``.
    Accepting the string without changing the kernel is the silent version of
    that.
    """

    with pytest.raises(ValueError, match="f_ref_at_n0"):
        _spec(subcarrier_origin="centred_on_f_ref")
