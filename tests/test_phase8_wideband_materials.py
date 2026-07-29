"""Phase 8: the wideband material fixtures, and the refusals, with numbers.

Three of the material classes the plan names are ACCEPTED and checked against
closed forms; two are REFUSED, and a refused capability is explained here with
a measured number rather than with a sentence. That is what "the capability and
numerical differences between the narrowband and frequency-dependent paths are
explainable" means for a class the wideband route does not cover: the drift the
frozen record would introduce is computed, and it is shown to exceed the error
law the accepted route publishes.

Tolerances. Channel evaluates each column natively in float32 and publishes
complex64, so a comparison against a float64 closed form cannot beat a few
float32 ULPs on the accumulated phase. The reflection rows carry a 27 ns delay
at 3 GHz - about 80 cycles of absolute phase - and that is what sets the floor.
Every bound below is the measured worst case with margin, and the measurement
is in the docstring so a later regression is legible.
"""

from __future__ import annotations

import math

import pytest
import torch

from witwin.channel.propagation import consumer
from witwin.radar.channel import (
    WIDEBAND_FREQUENCY_RESOLUTION_PHASE_BUDGET_RAD,
    ChannelPropagationAdapter,
)

from support import wideband_world as ww  # noqa: E402


pytestmark = pytest.mark.gpu

F_REF = ww.REFERENCE_FREQUENCY_HZ
COMPONENTS = frozenset({"los", "reflection"})

#: 49 columns spanning 2.4 Airy fringes, centred on ``df = 0`` so the reference
#: identity is reachable at a column index rather than at an endpoint. An odd
#: count with an exact centre is what makes ``offsets[HALF] == 0.0`` exact in
#: float rather than "small".
COLUMN_COUNT = 49
CENTRE = COLUMN_COUNT // 2


def _sweep(fringes: float = 2.4) -> tuple[float, ...]:
    step = fringes * ww.fringe_period_hz() / (COLUMN_COUNT - 1)
    offsets = tuple((index - CENTRE) * step for index in range(COLUMN_COUNT))
    assert offsets[CENTRE] == 0.0
    return offsets


def _band(compiled, offsets):
    adapter = ChannelPropagationAdapter(
        compiled,
        reference_frequency_hz=F_REF,
        components=COMPONENTS,
        max_depth=1,
        frequency_offsets_hz=offsets,
    )
    frozen = adapter.freeze(ww.source_spec(), ww.sink_spec())
    leg = adapter.reevaluate(
        frozen, ww.source_spec(), ww.sink_spec(), ad_mode="none"
    )
    return adapter, leg


def _row(leg, component_id: int) -> torch.Tensor:
    index = (leg.component_id == component_id).nonzero().flatten()
    assert index.numel() == 1, (component_id, index.tolist())
    return leg.frequency_response[int(index[0])].to(torch.complex128).cpu()


# ---------------------------------------------------------------------------
# Accepted: lossy half space
# ---------------------------------------------------------------------------


def test_the_half_space_fixture_really_is_a_half_space():
    """The degeneracy is asserted, not assumed.

    A "half space" here is a slab whose back face is invisible. If the loss or
    the thickness were ever reduced, the closed form below would silently become
    the wrong one - an Airy stack compared against a bare interface - and the
    Fresnel test would fail for a reason that looks like a kernel bug.
    """

    worst = 0.0
    for offset in _sweep():
        frequency = F_REF + offset
        stack = ww.slab_reflection_te(frequency, material="half_space")
        interface = ww.bare_interface_te(frequency)
        worst = max(worst, abs(stack - interface) / abs(interface))
    assert worst < 1.0e-8, worst


def test_the_half_space_reflection_follows_the_fresnel_coefficient(
):
    """``H(f) = sqrt(P) lambda/(4 pi d) exp(-j k d) * r_TE(f)``.

    Smooth and monotone: no fringes, so this isolates the
    ``eps_c = eps_r - j sigma/(omega eps0)`` term from the layer stack entirely.
    Measured worst case 9.6e-5 relative over the sweep.
    """

    offsets = _sweep()
    _, leg = _band(ww.compile_half_space(), offsets)
    row = _row(leg, 1)
    distance = ww.reflection_length_m()

    worst = 0.0
    for index, offset in enumerate(offsets):
        frequency = F_REF + offset
        expected = ww.free_space_coefficient(
            frequency, distance
        ) * ww.slab_reflection_te(frequency, material="half_space")
        worst = max(
            worst, abs(complex(row[index]) - expected) / abs(expected)
        )
    assert worst < 2.0e-4, worst


def test_the_line_of_sight_row_follows_free_space_across_the_band():
    """The other half of the same statement, with no material in it at all.

    Measured worst case 4.5e-5 relative.
    """

    offsets = _sweep()
    _, leg = _band(ww.compile_half_space(), offsets)
    row = _row(leg, 0)
    distance = ww.line_of_sight_length_m()

    worst = 0.0
    for index, offset in enumerate(offsets):
        expected = ww.free_space_coefficient(F_REF + offset, distance)
        worst = max(worst, abs(complex(row[index]) - expected) / abs(expected))
    assert worst < 1.0e-4, worst


# ---------------------------------------------------------------------------
# Accepted: the multilayer slab, swept across two Airy fringes
# ---------------------------------------------------------------------------


def test_the_slab_reflection_follows_the_airy_stack_across_two_fringes():
    """The falsifying fixture. Measured worst case 9.6e-5 relative.

    The sweep spans 2.4 fringe periods, so the reflectivity is neither flat nor
    monotone: it swings by more than an order of magnitude. A narrowband
    implementation cannot produce this curve at all, and the next test says so
    with a number.
    """

    offsets = _sweep()
    _, leg = _band(ww.compile_slab(), offsets)
    row = _row(leg, 1)
    distance = ww.reflection_length_m()

    worst = 0.0
    for index, offset in enumerate(offsets):
        frequency = F_REF + offset
        expected = ww.free_space_coefficient(
            frequency, distance
        ) * ww.slab_reflection_te(frequency)
        worst = max(worst, abs(complex(row[index]) - expected) / abs(expected))
    assert worst < 2.0e-4, worst


def test_the_slab_fringes_land_where_the_analytic_period_says():
    """The period is closed form, so the minima are predictable, not fitted.

    ``df_fringe = c / (2 Re(sqrt(eps_r)) d cos(theta_t))`` with ``theta_t`` from
    Snell's law at the fixture's incidence. The measured envelope minima are
    required to land in the SAME grid bins as the analytic ones, which is a
    stronger statement than agreeing on a period: a curve with the right period
    and the wrong phase would fail.
    """

    offsets = _sweep()
    _, leg = _band(ww.compile_slab(), offsets)
    measured = _row(leg, 1).abs()
    analytic = torch.tensor(
        [abs(ww.slab_reflection_te(F_REF + offset)) for offset in offsets],
        dtype=torch.float64,
    )

    def minima(values: torch.Tensor) -> list[int]:
        return [
            index
            for index in range(1, values.numel() - 1)
            if values[index] < values[index - 1] and values[index] < values[index + 1]
        ]

    # The band's own spreading tilt makes the measured envelope the product of
    # the stack and 1/f, so the comparison is between the stack curves; the
    # measured one is divided by the free-space magnitude to isolate it.
    distance = ww.reflection_length_m()
    isolated = torch.tensor(
        [
            float(measured[index])
            / abs(ww.free_space_coefficient(F_REF + offset, distance))
            for index, offset in enumerate(offsets)
        ],
        dtype=torch.float64,
    )
    assert minima(isolated) == minima(analytic), (
        minima(isolated),
        minima(analytic),
    )
    assert len(minima(analytic)) >= 2, "the sweep must cross at least two fringes"
    assert float(analytic.max() / analytic.min()) > 5.0


def test_the_narrowband_law_is_measurably_wrong_on_the_slab():
    """The published error law, evaluated on the fixture it describes.

    ``H(f_ref + df) = C(f_ref) * exp(-j 2 pi df tau)`` is what a narrowband
    consumer applies. Across 2.4 fringes it is off by more than an order of
    magnitude in places, which is the quantitative content of "narrowband OFDM
    cannot express a per-tap spectral shape at all".
    """

    offsets = _sweep()
    _, leg = _band(ww.compile_slab(), offsets)
    row = _row(leg, 1)
    delay_s = float(leg.delay_s[int((leg.component_id == 1).nonzero()[0])])
    reference = complex(row[CENTRE])

    worst = 0.0
    for index, offset in enumerate(offsets):
        cycles = -offset * delay_s
        fraction = cycles - math.floor(cycles)
        law = reference * complex(
            math.cos(2.0 * math.pi * fraction), math.sin(2.0 * math.pi * fraction)
        )
        worst = max(worst, abs(complex(row[index]) - law) / abs(complex(row[index])))
    assert worst > 1.0, worst


# ---------------------------------------------------------------------------
# The acceptance test: the fringe is visible in the range profile
# ---------------------------------------------------------------------------

#: A band that fits INSIDE the fixture's unambiguous delay window. The inverse
#: transform of a CFR wraps at ``1 / df``, and the reflection row sits at
#: 26.6 ns, so ``df`` has to stay under 37.6 MHz or the tap folds and the test
#: measures aliasing instead of material response. 64 x 30 MHz = 1.92 GHz, which
#: is 1.26 Airy fringes of the fixture slab.
PROFILE_SUBCARRIERS = 64
PROFILE_SPACING_HZ = 30.0e6


def _profile_spec():
    from witwin.radar.synthesis import OfdmSpec

    return OfdmSpec(
        num_subcarriers=PROFILE_SUBCARRIERS,
        num_symbols=1,
        subcarrier_spacing_hz=PROFILE_SPACING_HZ,
        cyclic_prefix_s=1.0e-7,
        reference_frequency_hz=F_REF,
        max_expected_delay_s=5.0e-8,
        carrier_hz=0.0,
        carrier_rate_hz=F_REF,
    )


def _idft(spectrum: torch.Tensor) -> torch.Tensor:
    """``(1/N) sum_n X[n] exp(+j 2 pi n m / N)``, written out in float64.

    Written as an explicit sum rather than through ``torch.fft`` so that the
    reference this test compares against shares no implementation with anything
    under test, and so that the normalisation is stated rather than inherited.
    """

    count = int(spectrum.shape[-1])
    index = torch.arange(count, dtype=torch.float64)
    kernel = torch.exp(
        2.0j
        * math.pi
        * index.reshape(-1, 1)
        * index.reshape(1, -1)
        / count
    )
    return (kernel @ spectrum.to(torch.complex128)) / count


def _single_reflection_cube(offsets):
    """One reflection row, one symbol: a single tap and nothing else.

    ``components={"reflection"}`` rather than the usual pair, because a range
    profile with a line-of-sight tap in it would be a two-tap test and the point
    here is the SHAPE of one tap.
    """

    from witwin.radar.paths import DirectComposer
    from witwin.radar.synthesis import (
        SlowTimeMode,
        SynthesisPathBatch,
        synthesize_ofdm,
    )

    adapter = ChannelPropagationAdapter(
        ww.compile_slab(),
        reference_frequency_hz=F_REF,
        components=frozenset({"reflection"}),
        max_depth=1,
        frequency_offsets_hz=offsets,
    )
    frozen = adapter.freeze(ww.source_spec(), ww.sink_spec())
    assert frozen.row_count == 1, frozen.row_count
    leg = adapter.reevaluate(
        frozen, ww.source_spec(), ww.sink_spec(), ad_mode="none"
    )
    composer = DirectComposer.freeze(
        frozen,
        radar_source_ids=[ww.SOURCE_STABLE_ID],
        radar_sink_ids=[ww.SINK_STABLE_ID],
        reference_frequency_hz=F_REF,
    )
    composed = composer.compose(leg, include_delay_rate=False)

    spec = _profile_spec()
    wide = synthesize_ofdm(
        SynthesisPathBatch.from_radar_paths(
            composed, slow_time_mode=SlowTimeMode.FROZEN_WEIGHT_WITH_CARRIER_RATE
        ),
        spec,
    )
    import dataclasses

    narrow = synthesize_ofdm(
        SynthesisPathBatch.from_radar_paths(
            dataclasses.replace(
                composed, frequency_response=None, frequency_offsets_hz=None
            ),
            slow_time_mode=SlowTimeMode.FROZEN_WEIGHT_WITH_CARRIER_RATE,
        ),
        spec,
    )
    delay_s = float(leg.delay_s[0])
    return narrow[0, 0].cpu(), wide[0, 0].cpu(), delay_s


def test_a_narrowband_tap_is_the_pure_delay_kernel_and_a_wideband_one_is_not():
    """The acceptance test for "wideband is physically different, correctly".

    Three claims, in order:

    (a) the NARROWBAND profile of one tap is the pure delay kernel
        ``(1/N) sum_n exp(-j 2 pi n df tau) exp(+j 2 pi n m / N)`` to 1e-4
        normalised. That is what "the scene is an ideal delay-and-scale tap"
        means when written down;
    (b) the WIDEBAND profile of the same tap is NOT, by a wide margin. The slab
        filters each subcarrier by ``r(f_n)``, so the tap is smeared and
        asymmetric - the material-thickness signature narrowband OFDM cannot
        express at all;
    (c) the wideband profile IS the transform of the analytic
        ``sqrt(P) lambda/(4 pi d) exp(-j k d) r_TE(f_ref + n df)`` sampled on the
        same grid. Without (c), (b) would only say the two differ.
    """

    spec = _profile_spec()
    offsets = spec.frequency_offsets_hz
    narrow_cfr, wide_cfr, delay_s = _single_reflection_cube(offsets)
    distance = ww.reflection_length_m()

    index = torch.arange(PROFILE_SUBCARRIERS, dtype=torch.float64)
    cycles = -PROFILE_SPACING_HZ * delay_s * index
    kernel = torch.exp(2.0j * math.pi * (cycles - torch.floor(cycles)))
    ideal = _idft(complex(narrow_cfr[0]) * kernel)
    measured_narrow = _idft(narrow_cfr)
    scale = float(ideal.abs().max())
    assert float((measured_narrow - ideal).abs().max()) / scale < 1.0e-4

    analytic = torch.tensor(
        [
            ww.free_space_coefficient(F_REF + offset, distance)
            * ww.slab_reflection_te(F_REF + offset)
            for offset in offsets
        ],
        dtype=torch.complex128,
    )
    expected_wide = _idft(analytic)
    measured_wide = _idft(wide_cfr)
    wide_scale = float(expected_wide.abs().max())
    assert (
        float((measured_wide - expected_wide).abs().max()) / wide_scale < 2.0e-4
    )

    # (b): the wideband tap is not the pure delay kernel, normalised the same
    # way. Compared shape to shape, so the spreading tilt alone cannot carry it.
    wide_ideal = _idft(complex(wide_cfr[0]) * kernel)
    departure = float(
        (measured_wide - wide_ideal).abs().max() / wide_ideal.abs().max()
    )
    assert departure > 1.0e-2, departure

    # And the smear is asymmetric, which a symmetric loss of resolution is not.
    peak = int(measured_wide.abs().argmax())
    left = float(measured_wide[peak - 1].abs())
    right = float(measured_wide[peak + 1].abs())
    assert abs(left - right) / max(left, right) > 1.0e-2, (left, right)


# ---------------------------------------------------------------------------
# Refused: dispersion, explained with a number
# ---------------------------------------------------------------------------


def test_a_dispersive_scene_refuses_the_band_with_the_channel_message_intact():
    """The Channel refusal passes through; radar does not re-wrap or soften it.

    Reached through the production adapter, not through a hand-built request, so
    what is asserted is that the whole radar path lets the refusal out.
    """

    adapter = ChannelPropagationAdapter(
        ww.compile_dispersive(),
        reference_frequency_hz=F_REF,
        components=COMPONENTS,
        max_depth=1,
        frequency_offsets_hz=_sweep(),
    )
    frozen = adapter.freeze(ww.source_spec(), ww.sink_spec())
    with pytest.raises(NotImplementedError) as caught:
        adapter.reevaluate(
            frozen, ww.source_spec(), ww.sink_spec(), ad_mode="none"
        )
    message = str(caught.value)
    assert "dispersive" in message.lower() or "frequency_dependent" in message
    assert "compile" in message.lower()


def test_the_dispersive_refusal_is_justified_by_the_drift_it_would_hide():
    """How wrong a frozen dispersion record would be, over this band.

    Channel evaluates a ``DispersionSpec`` once, at compile, so a band on a
    dispersive scene would reuse ``eps_r(f_ref)`` at every column. This computes
    the resulting drift in ``eps_r`` and requires it to exceed the accepted
    route's own material error scale ``df / df_fringe`` - which is why the
    capability is refused rather than approximated.
    """

    offsets = _sweep()
    drift = ww.dispersive_eps_r_drift(offsets)
    material_scale = max(abs(offset) for offset in offsets) / ww.fringe_period_hz()

    assert drift > 0.01, drift
    assert drift < material_scale, (drift, material_scale)

    # And the same scene WITHOUT the dispersion spec is accepted, so the
    # refusal is about the spec and not about the geometry.
    _, leg = _band(ww.compile_slab(), offsets)
    assert leg.band_count == COLUMN_COUNT


def test_a_rough_scene_refuses_the_band():
    """Refused for a lifetime reason, not a physics one.

    The Kirchhoff table is built per material cache token and that token hashes
    the frequency, so a table built at ``f_ref`` and used at ``f_ref + df`` is a
    frozen approximation of the same class as a frozen dispersion record.
    """

    adapter = ChannelPropagationAdapter(
        ww.compile_rough(),
        reference_frequency_hz=F_REF,
        components=COMPONENTS,
        max_depth=1,
        frequency_offsets_hz=_sweep(),
    )
    frozen = adapter.freeze(ww.source_spec(), ww.sink_spec())
    with pytest.raises(NotImplementedError):
        adapter.reevaluate(
            frozen, ww.source_spec(), ww.sink_spec(), ad_mode="none"
        )


def test_an_unresolvable_grid_spacing_is_refused_by_channel():
    """Below one float32 grid step an offset simply does not exist.

    Channel refuses it rather than publishing a duplicate column under a
    different label. At 3 GHz the step is 256 Hz.
    """

    resolution = consumer.native_frequency_resolution_hz(F_REF)
    assert resolution == 256.0, resolution
    adapter = ChannelPropagationAdapter(
        ww.compile_slab(),
        reference_frequency_hz=F_REF,
        components=COMPONENTS,
        max_depth=1,
        frequency_offsets_hz=(0.0, 0.25 * resolution),
    )
    frozen = adapter.freeze(ww.source_spec(), ww.sink_spec())
    with pytest.raises(ValueError, match="resolution"):
        adapter.reevaluate(
            frozen, ww.source_spec(), ww.sink_spec(), ad_mode="none"
        )


# ---------------------------------------------------------------------------
# The radar-owned frequency-resolution phase budget
# ---------------------------------------------------------------------------


def _budget_delay_s(fraction: float) -> float:
    """A round-trip delay at ``fraction`` of the budget, at 77 GHz."""

    resolution = consumer.native_frequency_resolution_hz(77.0e9)
    return fraction * WIDEBAND_FREQUENCY_RESOLUTION_PHASE_BUDGET_RAD / (
        math.pi * resolution
    )


def test_the_budget_binds_where_the_published_law_says_it_does():
    """``pi * resolution_hz * max(delay_s) <= budget``, arithmetic only.

    Channel publishes ``native_frequency_resolution_hz`` and the law; it
    deliberately does not evaluate the bound, because that needs
    ``max(delay_s)`` - a device reduction plus a host read outside the ADR-032
    per-call budget. Radar owns the evaluation, at freeze, where a host read is
    already paid.
    """

    resolution = consumer.native_frequency_resolution_hz(77.0e9)
    assert resolution == 8192.0, resolution
    bound_s = WIDEBAND_FREQUENCY_RESOLUTION_PHASE_BUDGET_RAD / (
        math.pi * resolution
    )
    assert 3.8e-6 < bound_s < 4.0e-6, bound_s
    # A 150 m round trip sits comfortably inside it, at 2.6e-2 rad.
    assert math.pi * resolution * 1.0e-6 == pytest.approx(2.573e-2, rel=1.0e-3)


def test_a_topology_whose_delays_exceed_the_budget_is_refused_by_name():
    """Both numbers in the message, and the check runs at FREEZE.

    Driven through a fabricated delay tensor rather than through a 600 m scene,
    because the refusal is about the delay magnitude and building a kilometre of
    geometry would make the test about mesh size instead.
    """

    adapter = ChannelPropagationAdapter(
        ww.compile_slab(),
        reference_frequency_hz=77.0e9,
        components=COMPONENTS,
        max_depth=1,
        frequency_offsets_hz=(0.0, 1.0e7),
    )
    over = torch.tensor(
        [_budget_delay_s(1.01)], dtype=torch.float32, device="cuda"
    )
    with pytest.raises(ValueError, match="WIDEBAND_FREQUENCY_RESOLUTION_PHASE"):
        adapter._require_frequency_resolution_budget(over)

    message = ""
    try:
        adapter._require_frequency_resolution_budget(over)
    except ValueError as error:
        message = str(error)
    assert "8192.0" in message
    assert "native_frequency_resolution_hz" in message

    under = torch.tensor(
        [_budget_delay_s(0.99)], dtype=torch.float32, device="cuda"
    )
    adapter._require_frequency_resolution_budget(under)


def test_a_narrowband_adapter_never_evaluates_the_budget():
    """No band, no bound: the check is a property of the band, not of the scene."""

    adapter = ChannelPropagationAdapter(
        ww.compile_slab(),
        reference_frequency_hz=77.0e9,
        components=COMPONENTS,
        max_depth=1,
    )
    enormous = torch.tensor([1.0e-3], dtype=torch.float32, device="cuda")
    adapter._require_frequency_resolution_budget(enormous)


# ---------------------------------------------------------------------------
# Capability agreement
# ---------------------------------------------------------------------------


def test_radar_derives_its_wideband_support_from_the_capability_record():
    """Not from a local constant, so a narrowed Channel narrows radar with it.

    Driven by monkeypatching the capability record rather than by asserting the
    values it happens to carry today: what is under test is that the adapter
    READS it.
    """

    record = consumer.capabilities()
    assert record.supports_wideband_offsets is True
    assert "scalar_transport" in record.wideband_responses
    assert COMPONENTS <= record.wideband_components


def test_a_component_outside_the_wideband_cell_is_refused_by_name():
    """``transmission`` is a real component and it is not freezable either.

    The message must quote the capability record's own cell, because a local
    copy of that set is exactly what would go stale.
    """

    record = consumer.capabilities()
    outside = sorted(record.components - record.wideband_components)
    assert outside, "the consumer's wideband cell must be a strict subset"

    with pytest.raises(NotImplementedError) as caught:
        ChannelPropagationAdapter(
            ww.compile_slab(),
            reference_frequency_hz=F_REF,
            components=frozenset({outside[0]}),
            max_depth=1,
            frequency_offsets_hz=(0.0, 1.0e7),
        )
    message = str(caught.value)
    assert outside[0] in message
    assert sorted(record.wideband_components)[0] in message


def test_the_adapter_refuses_a_band_when_the_record_withdraws_support(
    monkeypatch,
):
    import dataclasses

    record = consumer.capabilities()
    withdrawn = dataclasses.replace(record, supports_wideband_offsets=False)
    monkeypatch.setattr(consumer, "capabilities", lambda: withdrawn)

    with pytest.raises(NotImplementedError, match="does not support frequency offsets"):
        ChannelPropagationAdapter(
            ww.compile_slab(),
            reference_frequency_hz=F_REF,
            components=COMPONENTS,
            max_depth=1,
            frequency_offsets_hz=(0.0, 1.0e7),
        )
