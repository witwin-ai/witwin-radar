"""Closed-form OFDM acceptance: subcarrier phase, CIR peak, Doppler, sign.

Every expected value here is written down from the physics, in float64, in the
test itself. Nothing is compared against a previous run of the code under test.
The primary delay assertion is a PHASE SLOPE, not an IDFT argmax: at the
reference geometry the CIR peak sits at sample 0.19, so an argmax is 0 and would
pass against a kernel with no delay term at all.

Reference grid, the physics survey's OFDM probe grid, restated in SI::

    f_ref = 77 GHz     df   = 120 kHz    N_sc  = 64
    T_cp  = 2 us       L    = 32         T_sym = 10.3333 us
    d  = 3.7 m one way   ->  tau_rt   = 2 d / c0   = 24.6837 ns
    v_r = 12 m/s away    ->  tau_rate = 2 v_r / c0 = 80.0554 ns/s

The cube is in the CHANNEL phasor convention ``exp(-j k d)`` and is NOT
conjugated. Every sign below follows from that one fact, and the sign tests are
the ones that would survive a magnitude-only review.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from witwin.radar.synthesis.contracts import (  # noqa: E402
    SPEED_OF_LIGHT_M_PER_S,
    OfdmCfrSpec,
)
from witwin.radar.synthesis.ofdm_cfr import synthesize_cfr_rows  # noqa: E402


pytestmark = pytest.mark.gpu


C0 = SPEED_OF_LIGHT_M_PER_S
F_REF_HZ = 77.0e9
DF_HZ = 120.0e3
NUM_SUBCARRIERS = 64
CYCLIC_PREFIX_S = 2.0e-6
NUM_SYMBOLS = 32
MAX_DELAY_S = 1.0e-6
RANGE_M = 3.7
RADIAL_SPEED_MPS = 12.0

TAU_RT_S = 2.0 * RANGE_M / C0
TAU_RATE = 2.0 * RADIAL_SPEED_MPS / C0

# A target far enough out that the CIR peak clears four samples of the 130.2 ns
# delay grid; at 3.7 m the peak sits at sample 0.19 and an argmax says nothing.
FAR_RANGE_M = 100.0
TAU_FAR_S = 2.0 * FAR_RANGE_M / C0


def _spec(**overrides) -> OfdmCfrSpec:
    """The production carrier placement unless a test says otherwise."""

    fields = dict(
        num_subcarriers=NUM_SUBCARRIERS,
        num_symbols=1,
        subcarrier_spacing_hz=DF_HZ,
        cyclic_prefix_s=CYCLIC_PREFIX_S,
        reference_frequency_hz=F_REF_HZ,
        max_expected_delay_s=MAX_DELAY_S,
        carrier_hz=0.0,
        carrier_rate_hz=F_REF_HZ,
    )
    fields.update(overrides)
    return OfdmCfrSpec(**fields)


def _frozen_channel_weight(amplitude: complex = 1.0 + 0.0j, tau: float = TAU_RT_S):
    """A Channel-sourced coefficient at this geometry.

    ``C_rt`` carries ``exp(-j 2 pi f_ref tau_rt)`` at the FROZEN per-frame
    delay. It is handed to the CFR kernel unconjugated, unlike the FMCW beat
    weight, because the OFDM product stays in Channel's convention.
    """

    phase = -2.0 * math.pi * F_REF_HZ * tau
    return amplitude * complex(math.cos(phase), math.sin(phase))


def _rows(delays, weights, rates, offsets=None):
    tau = torch.tensor(delays, dtype=torch.float32, device="cuda")
    rate = torch.tensor(rates, dtype=torch.float32, device="cuda")
    transfer = torch.tensor(weights, dtype=torch.complex64, device="cuda")
    if offsets is None:
        offsets = [0, len(delays)]
    table = torch.tensor(offsets, dtype=torch.int64, device="cuda")
    return tau, rate, transfer, table


def _stored(value: float) -> float:
    """The float32 value the kernel actually sees.

    ``total_delay_s`` is float32 by contract, so rounding it costs up to half an
    ulp, about 6e-8 relative. That is a property of the CONTRACT, not of the
    kernel, and separating the two is what lets the kernel be asserted tighter
    than its input can be represented.
    """

    return float(torch.tensor([value], dtype=torch.float32))


def _cube(spec, delays, weights, rates, offsets=None) -> torch.Tensor:
    tau, rate, transfer, table = _rows(delays, weights, rates, offsets)
    return synthesize_cfr_rows(tau, rate, transfer, table, spec).cpu().to(
        torch.complex128
    )


def _unwrapped(values: torch.Tensor) -> torch.Tensor:
    return torch.from_numpy(np.unwrap(torch.angle(values).numpy()))


def _lsq_slope(phase: torch.Tensor) -> float:
    """Least-squares slope of an unwrapped phase sequence, per index step."""

    index = torch.arange(phase.numel(), dtype=torch.float64)
    index = index - index.mean()
    return float((index * (phase - phase.mean())).sum() / (index * index).sum())


# --------------------------------------------------------------------------
# T2.1  the subcarrier phase slope - the primary delay assertion
# --------------------------------------------------------------------------


def test_the_subcarrier_phase_slope_reports_the_round_trip_delay_in_seconds():
    """``arg H[n] - arg H[n-1] = -2 pi df tau_rt`` for every ``n``.

    Bin-free and exact for any delay, which is why it and not the CIR peak is
    the primary assertion. Two statements: every adjacent step matches the
    analytic one, and the least-squares fit over the whole band recovers
    ``tau_rt`` IN SECONDS. Asserting a CIR bin index instead would pass against
    a kernel that doubled the delay and halved the bandwidth.

    The per-step tolerance is absolute, in radians, because the step itself is
    small (0.0186 rad here). The kernel wraps its cycle count in double and then
    rounds the ``[0, 2 pi)`` argument to float32 before one ``sincosf``, which
    costs up to about 2.4e-7 rad per sample; the measured worst adjacent-step
    deviation at this geometry is 3.57e-7 rad.
    """

    spec = _spec()
    row = _cube(spec, [TAU_RT_S], [_frozen_channel_weight()], [0.0])[0, 0]

    analytic_step = -2.0 * math.pi * DF_HZ * _stored(TAU_RT_S)
    assert analytic_step == pytest.approx(-0.018611103798605002, rel=1e-9)
    assert spec.subcarrier_phase_step_rad(_stored(TAU_RT_S)) == pytest.approx(
        analytic_step, rel=1e-12
    )

    steps = torch.angle(row[1:] * torch.conj(row[:-1]))
    assert steps.numel() == NUM_SUBCARRIERS - 1
    assert float((steps - analytic_step).abs().max()) < 2.0e-6

    slope = _lsq_slope(_unwrapped(row))
    tau_estimate = -slope / (2.0 * math.pi * DF_HZ)
    assert tau_estimate == pytest.approx(TAU_RT_S, rel=1e-5)
    assert C0 * tau_estimate / 2.0 == pytest.approx(RANGE_M, rel=1e-5)

    # The resolution the tolerance sits inside, stated rather than assumed: the
    # assertion above is five orders of magnitude finer than one CIR sample.
    assert spec.range_resolution_m == pytest.approx(19.517738151041666, rel=1e-12)


def test_the_delay_is_round_trip_and_is_never_doubled():
    """Doubling ``tau`` doubles the subcarrier phase slope.

    ``dirichlet.cu`` takes a ONE-WAY distance and forms ``tau = 2 d / c0``
    internally. A synthesis kernel that repeated that on a round-trip delay
    would report exactly twice the range, self-consistently, at every geometry
    and with no other symptom.
    """

    spec = _spec()
    single = _lsq_slope(
        _unwrapped(_cube(spec, [TAU_RT_S], [_frozen_channel_weight()], [0.0])[0, 0])
    )
    doubled = _lsq_slope(
        _unwrapped(
            _cube(
                spec,
                [2.0 * TAU_RT_S],
                [_frozen_channel_weight(tau=2.0 * TAU_RT_S)],
                [0.0],
            )[0, 0]
        )
    )
    assert doubled == pytest.approx(2.0 * single, rel=1e-5)
    assert -single / (2.0 * math.pi * DF_HZ) == pytest.approx(TAU_RT_S, rel=1e-5)


# --------------------------------------------------------------------------
# T2.2  the CIR peak, as a secondary check
# --------------------------------------------------------------------------


def test_the_channel_impulse_response_peaks_at_the_delay_in_samples():
    """``h[m] = IDFT_n{H[n]}`` peaks at ``m_peak = tau_rt / T_s``.

    Secondary to the phase slope and deliberately run at a FAR target: at 3.7 m
    the peak is at sample 0.19, below one bin, and its argmax is 0 - a value a
    kernel with no delay term would also produce. At 100 m it is at 5.12, four
    samples clear of zero and off-grid, so the interpolated location is a real
    measurement.

    The fit is on a 16x zero-padded transform. The same parabolic fit on the
    unpadded CIR is biased by 2.2e-2 relative at this geometry, three orders of
    magnitude outside the tolerance, so the padded assertion is measuring the
    kernel and the unpadded one is measuring the fit. Both are asserted, in
    opposite directions.
    """

    spec = _spec()
    m_peak = spec.cir_peak_sample(TAU_FAR_S)
    assert m_peak == pytest.approx(5.1235445022436155, rel=1e-12)
    assert m_peak > 4.0

    row = _cube(spec, [TAU_FAR_S], [_frozen_channel_weight(tau=TAU_FAR_S)], [0.0])[
        0, 0
    ]

    def _interpolated_peak(pad: int) -> float:
        magnitude = torch.fft.ifft(row, n=NUM_SUBCARRIERS * pad).abs()
        peak = int(magnitude.argmax())
        left = float(magnitude[peak - 1])
        centre = float(magnitude[peak])
        right = float(magnitude[(peak + 1) % magnitude.numel()])
        offset = 0.5 * (left - right) / (left - 2.0 * centre + right)
        return (peak + offset) / pad

    assert _interpolated_peak(16) == pytest.approx(m_peak, rel=1e-3)
    assert abs(_interpolated_peak(1) - m_peak) / m_peak > 1e-2

    # And the delay resolution the bin is measured in, in metres.
    assert C0 * spec.waveform_sample_period_s / 2.0 == pytest.approx(
        spec.range_resolution_m, rel=1e-12
    )


# --------------------------------------------------------------------------
# T2.3  the exact reference identity
# --------------------------------------------------------------------------


@pytest.mark.parametrize("weight", [0.25 - 0.5j, -1.0 + 0.0j, 3.5 + 2.25j])
def test_the_first_subcarrier_of_the_first_symbol_is_the_coefficient(weight):
    """``H[0][p][0] == C_rt``, exactly.

    This is what pinning ``n = 0`` to ``f_ref`` and publishing in Channel's
    convention buy: at ``n = 0`` there is no subcarrier phase, at ``l = 0``
    there is no drift, and the weight already holds the absolute carrier phase,
    so the kernel applies a phase of exactly zero. A conjugation anywhere in the
    family, a centred band, or a carrier applied to the full delay instead of
    the drift would each break this identity and nothing else visible.
    """

    spec = _spec()
    cube = _cube(spec, [TAU_RT_S], [weight], [0.0])
    measured = complex(cube[0, 0, 0])
    reference = complex(torch.tensor(weight, dtype=torch.complex64))
    assert measured == pytest.approx(reference, rel=1e-6, abs=1e-7 * abs(reference))
    # At this grid point the kernel's phase is identically zero, so the identity
    # is bit-exact rather than merely inside the tolerance.
    assert measured == reference


def test_a_conjugated_cube_would_fail_the_reference_identity():
    """Non-vacuity for the test above: the two conventions are distinguishable.

    A weight whose imaginary part is nonzero is required, and that is why the
    parametrisation above is not all real.
    """

    weight = 0.25 - 0.5j
    cube = _cube(_spec(), [TAU_RT_S], [weight], [0.0])
    assert complex(cube[0, 0, 0]) != complex(weight).conjugate()


# --------------------------------------------------------------------------
# T2.4  slow-time Doppler, two-sided
# --------------------------------------------------------------------------


def _analytic_symbol_slope(spec: OfdmCfrSpec, subcarrier: int, rate: float) -> float:
    """``-2 pi (f_ref + n df) tau_rate T_sym`` radians per symbol."""

    return (
        -2.0
        * math.pi
        * spec.subcarrier_frequency_hz(subcarrier)
        * rate
        * spec.symbol_period_s
    )


def _subcarrier_only_slope(spec: OfdmCfrSpec, subcarrier: int, rate: float) -> float:
    """What a frozen weight with NO carrier-rate term would leave behind."""

    return (
        -2.0
        * math.pi
        * subcarrier
        * spec.subcarrier_spacing_hz
        * rate
        * spec.symbol_period_s
    )


def _measured_symbol_slope(spec, subcarrier: int, rate: float) -> float:
    cube = _cube(spec, [TAU_RT_S], [_frozen_channel_weight()], [rate])
    return _lsq_slope(_unwrapped(cube[:, 0, subcarrier]))


def test_the_slow_time_slope_carries_the_whole_carrier_not_just_the_subcarrier():
    """The two-sided guard against a silent 1e4 Doppler understatement.

    One side: the measured per-symbol slope equals ``-2 pi f_ref tau_rate
    T_sym``. The other side: that value is enormously larger than the
    ``-2 pi n df tau_rate T_sym`` a kernel that reused the frozen weight without
    a carrier-rate term would leave. At ``n = 0`` the subcarrier-only value is
    exactly ZERO, so the whole Doppler disappears; at the top subcarrier it
    survives at one part in ten thousand. A one-sided test passes in both cases,
    because a cube with 1e-4 of the right Doppler still looks like a radar cube.
    """

    spec = _spec(num_symbols=64)
    analytic = _analytic_symbol_slope(spec, 0, _stored(TAU_RATE))
    assert analytic == pytest.approx(-0.400222582752, rel=1e-9)
    measured = _measured_symbol_slope(spec, 0, TAU_RATE)
    assert measured == pytest.approx(analytic, rel=1e-5)

    # Side two, at n = 0: the frozen-weight bug leaves exactly nothing.
    assert _subcarrier_only_slope(spec, 0, _stored(TAU_RATE)) == 0.0
    assert abs(measured) > 1000.0 * 1e-5 * abs(analytic)

    # And at the top subcarrier, where the bug leaves the most it ever can.
    top = NUM_SUBCARRIERS - 1
    understatement = _analytic_symbol_slope(
        spec, top, _stored(TAU_RATE)
    ) / _subcarrier_only_slope(spec, top, _stored(TAU_RATE))
    assert understatement == pytest.approx(10186.1852, rel=1e-6)
    assert F_REF_HZ / (top * DF_HZ) == pytest.approx(10185.1852, rel=1e-6)
    assert understatement > 1000.0


def test_the_slow_time_slope_carries_the_subcarrier_dependent_correction():
    """``slope(n) - slope(0) = -2 pi n df tau_rate T_sym``.

    The ``n`` dependence is the part that says the kernel applies the subcarrier
    phase to the FULL delay ``tau_k(l)`` and not to ``tau_rt`` alone. It is
    small - one part in ten thousand of the slope itself - so the estimate is a
    least-squares fit over 64 symbols; at 32 symbols the float32 phase noise is
    1.4e-4 of the quantity being measured and the assertion would be measuring
    the estimator.
    """

    spec = _spec(num_symbols=64)
    top = NUM_SUBCARRIERS - 1
    measured = _measured_symbol_slope(spec, top, TAU_RATE) - _measured_symbol_slope(
        spec, 0, TAU_RATE
    )
    analytic = _subcarrier_only_slope(spec, top, _stored(TAU_RATE))
    assert analytic == pytest.approx(-3.929458e-05, rel=1e-5)
    assert measured == pytest.approx(analytic, rel=1e-4)


@pytest.mark.parametrize("subcarrier", [0, NUM_SUBCARRIERS - 1])
def test_both_carrier_homes_produce_the_same_slow_time_slope(subcarrier):
    """``(f_ref, 0)`` and ``(0, f_ref)`` differ only by a constant.

    A kernel-owned carrier multiplies the FULL ``tau_k(l)`` and therefore
    already walks across symbols; a weight-owned one is frozen at ``tau_rt`` and
    needs the rate term to walk at all. They are exactly equivalent in slow time
    and differ by the constant ``f_ref tau_rt`` the weight holds on the
    production route.
    """

    production = _spec(num_symbols=64)
    kernel_owned = _spec(num_symbols=64, carrier_hz=F_REF_HZ, carrier_rate_hz=0.0)
    assert production.carrier_hz == 0.0
    assert kernel_owned.carrier_rate_hz == 0.0

    from_weight = _measured_symbol_slope(production, subcarrier, TAU_RATE)
    from_kernel = _lsq_slope(
        _unwrapped(
            _cube(kernel_owned, [TAU_RT_S], [1.0 + 0.0j], [TAU_RATE])[
                :, 0, subcarrier
            ]
        )
    )
    analytic = _analytic_symbol_slope(production, subcarrier, _stored(TAU_RATE))
    assert from_weight == pytest.approx(analytic, rel=1e-5)
    assert from_kernel == pytest.approx(analytic, rel=1e-5)


# --------------------------------------------------------------------------
# T2.5  Doppler SIGN
# --------------------------------------------------------------------------


def test_a_receding_site_puts_the_cfr_tone_at_negative_doppler():
    """The SIGN, and it is the OPPOSITE of the FMCW beat cube's.

    Physical Doppler in Channel's ``exp(-j k d)`` convention is
    ``f_D = -f_ref tau_rate``, so a receding site (``tau_rate > 0``) is a
    NEGATIVE shift. OFDM publishes that convention unchanged, so its slow-time
    tone sits at ``-f_ref tau_rate``. The FMCW cube is conjugated once and its
    tone sits at ``+f_ref tau_rate``. Both are correct and they point opposite
    ways; a consumer that assumed one convention for both would read every OFDM
    target as approaching when it recedes.
    """

    num_symbols = 64
    spec = _spec(num_symbols=num_symbols)
    cube = _cube(spec, [TAU_RT_S], [_frozen_channel_weight()], [TAU_RATE])
    slow = cube[:, 0, 0]

    spectrum = torch.fft.fftshift(torch.fft.fft(slow)).abs()
    frequencies = torch.fft.fftshift(
        torch.fft.fftfreq(num_symbols, d=spec.symbol_period_s)
    )
    peak_hz = float(frequencies[int(spectrum.argmax())])
    bin_hz = float(frequencies[1] - frequencies[0])

    physical_doppler_hz = -F_REF_HZ * TAU_RATE
    assert physical_doppler_hz == pytest.approx(-6164.264479261849, rel=1e-9)
    assert peak_hz < 0.0
    assert abs(peak_hz - physical_doppler_hz) <= 0.5 * bin_hz
    assert C0 * TAU_RATE / 2.0 == pytest.approx(RADIAL_SPEED_MPS, rel=1e-12)
    assert RADIAL_SPEED_MPS < spec.max_unambiguous_speed_mps

    # The two waveforms disagree on sign because one of them is conjugated, and
    # that is asserted rather than left to the reader.
    from witwin.radar.synthesis.fmcw_beat import channel_phasor_to_beat_weight

    coefficient = torch.tensor(
        [_frozen_channel_weight()], dtype=torch.complex64, device="cuda"
    )
    beat = channel_phasor_to_beat_weight(coefficient)
    assert not torch.equal(beat, coefficient)
    assert torch.equal(beat, torch.conj(coefficient).resolve_conj())


def test_an_approaching_site_reverses_the_slow_time_slope():
    spec = _spec(num_symbols=64)
    receding = _measured_symbol_slope(spec, 0, TAU_RATE)
    approaching = _measured_symbol_slope(spec, 0, -TAU_RATE)
    assert receding < 0.0 < approaching
    assert approaching == pytest.approx(-receding, rel=1e-5)


# --------------------------------------------------------------------------
# T2.6  unambiguous velocity
# --------------------------------------------------------------------------


def test_a_speed_past_the_unambiguous_bound_aliases():
    """At 1.05x the bound the measured slope wraps and changes SIGN.

    The bound is half a cycle of Doppler phase per symbol, so at 1.05x the true
    slope is -3.2987 rad and the measurable one is +2.9845: a receding target
    reads as an approaching one. The velocity is not merely imprecise past the
    bound, it is unrecoverable, which is why the assertion is on the sign flip
    and not on an error magnitude.
    """

    spec = _spec(num_symbols=16)
    assert spec.max_unambiguous_speed_mps == pytest.approx(
        C0 / (4.0 * F_REF_HZ * spec.symbol_period_s), rel=1e-12
    )

    speed = 1.05 * spec.max_unambiguous_speed_mps
    rate = 2.0 * speed / C0
    true_slope = _analytic_symbol_slope(spec, 0, _stored(rate))
    assert true_slope == pytest.approx(-1.05 * math.pi, rel=1e-4)

    cube = _cube(spec, [TAU_RT_S], [_frozen_channel_weight()], [rate])
    slow = cube[:, 0, 0]
    wrapped = float(torch.angle(slow[1:] * torch.conj(slow[:-1])).mean())
    assert true_slope < 0.0 < wrapped
    assert wrapped == pytest.approx(true_slope + 2.0 * math.pi, rel=1e-4)

    # Just inside the bound the same estimator recovers the true slope.
    inside_rate = 2.0 * (0.95 * spec.max_unambiguous_speed_mps) / C0
    inside = _cube(spec, [TAU_RT_S], [_frozen_channel_weight()], [inside_rate])[
        :, 0, 0
    ]
    measured = float(torch.angle(inside[1:] * torch.conj(inside[:-1])).mean())
    assert measured == pytest.approx(
        _analytic_symbol_slope(spec, 0, _stored(inside_rate)), rel=1e-5
    )


# --------------------------------------------------------------------------
# T2.7  the cyclic prefix, through the production entry point
# --------------------------------------------------------------------------


def test_the_cyclic_prefix_is_checked_before_any_launch(monkeypatch):
    """The refusal happens BEFORE the kernel runs, not after.

    A check that ran after the launch would still raise, but it would already
    have spent the frame and, worse, would leave the door open to a later
    "return what we have" branch. Asserted by counting launches across the
    refusal.
    """

    from support import multi_endpoint_driver as drv  # noqa: F401  (import guard)
    from witwin.radar.synthesis import ofdm_cfr

    launches = {"count": 0}
    operators = ofdm_cfr._ops()
    original = operators.ofdm_cfr_forward

    def counting(*args, **kwargs):
        launches["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(operators, "ofdm_cfr_forward", counting)

    batch = _cuda_batch()
    spec = _spec(max_expected_delay_s=3.0e-6)
    with pytest.raises(ValueError, match="cyclic_prefix_s"):
        ofdm_cfr.synthesize_ofdm_cfr(batch, spec)
    assert launches["count"] == 0

    ofdm_cfr.synthesize_ofdm_cfr(batch, _spec())
    assert launches["count"] == 1


# --------------------------------------------------------------------------
# T2.8  dead rows, empty segments, zero rows
# --------------------------------------------------------------------------


def _cuda_batch(*, row_valid=None, path_count: int = 1, pair_count: int = 1):
    from witwin.radar.paths.contracts import RadarPathTopology
    from witwin.radar.synthesis import SlowTimeMode, SynthesisPathBatch

    zeros = torch.zeros(path_count, dtype=torch.int64, device="cuda")
    offsets = torch.tensor(
        [0] + [path_count] * pair_count, dtype=torch.int64, device="cuda"
    )
    return SynthesisPathBatch(
        sensor_pair_count=pair_count,
        path_count=path_count,
        sensor_pair_index=torch.zeros(path_count, dtype=torch.int64, device="cuda"),
        pair_offsets=offsets,
        total_delay_s=torch.full(
            (path_count,), TAU_RT_S, dtype=torch.float32, device="cuda"
        ),
        delay_rate=torch.full(
            (path_count,), TAU_RATE, dtype=torch.float32, device="cuda"
        ),
        complex_transfer_ref=torch.full(
            (path_count,),
            _frozen_channel_weight(),
            dtype=torch.complex64,
            device="cuda",
        ),
        reference_frequency_hz=F_REF_HZ,
        frequency_response=None,
        frequency_offsets_hz=None,
        topology=RadarPathTopology(zeros, zeros, zeros, zeros, zeros),
        row_valid=row_valid,
        join_mode="multipath",
        weight_includes_reference_phase=True,
        weight_includes_spreading=True,
        weight_includes_tx_power=True,
        slow_time_mode=SlowTimeMode.FROZEN_WEIGHT_WITH_CARRIER_RATE,
    )


def test_a_dead_row_contributes_exactly_zero_and_carries_no_gradient():
    """Zeroed on the WEIGHT, not on the output.

    Zeroing the output afterwards produces the same primal and leaves a live
    gradient path back through a row that does not exist, so the assertion is on
    both the value and the gradient.
    """

    from witwin.radar.synthesis.ofdm_cfr import synthesize_ofdm_cfr

    spec = _spec(num_symbols=4)
    alive = _cuda_batch(
        row_valid=torch.ones(1, dtype=torch.bool, device="cuda")
    )
    dead = _cuda_batch(row_valid=torch.zeros(1, dtype=torch.bool, device="cuda"))

    assert float(synthesize_ofdm_cfr(alive, spec).abs().sum()) > 0.0
    assert float(synthesize_ofdm_cfr(dead, spec).abs().sum()) == 0.0

    import dataclasses

    weight = torch.full(
        (1,), _frozen_channel_weight(), dtype=torch.complex64, device="cuda"
    ).requires_grad_(True)
    live = dataclasses.replace(dead, complex_transfer_ref=weight)
    cube = synthesize_ofdm_cfr(live, spec)
    (cube.real.sum() + cube.imag.sum()).backward()
    assert float(weight.grad.abs().max()) == 0.0


def test_an_empty_pair_segment_produces_an_exact_zero_column():
    """A pair that discovered nothing keeps its channel and publishes zeros.

    Renumbering it away would shorten the pair axis and mis-steer every angle
    downstream. Three segments, of which the MIDDLE one is empty, because a
    trailing empty segment is the easy case.
    """

    spec = _spec(num_symbols=3)
    cube = _cube(
        spec,
        [TAU_RT_S, TAU_FAR_S, 3.0e-8],
        [_frozen_channel_weight(), 0.5 + 0.25j, -0.75 + 0.1j],
        [0.0, 0.0, 0.0],
        offsets=[0, 1, 1, 3],
    )
    assert tuple(cube.shape) == (3, 3, NUM_SUBCARRIERS)
    assert float(cube[:, 1, :].abs().max()) == 0.0
    assert float(cube[:, 0, :].abs().min()) > 0.0
    assert float(cube[:, 2, :].abs().min()) > 0.0


def test_a_batch_with_no_rows_produces_an_all_zero_cube_of_the_right_shape():
    spec = _spec(num_symbols=5)
    tau = torch.zeros(0, dtype=torch.float32, device="cuda")
    rate = torch.zeros(0, dtype=torch.float32, device="cuda")
    transfer = torch.zeros(0, dtype=torch.complex64, device="cuda")
    offsets = torch.tensor([0, 0, 0], dtype=torch.int64, device="cuda")
    cube = synthesize_cfr_rows(tau, rate, transfer, offsets, spec)
    assert tuple(cube.shape) == (5, 2, NUM_SUBCARRIERS)
    assert float(cube.abs().max()) == 0.0


# --------------------------------------------------------------------------
# T2.9  linearity in the coefficient
# --------------------------------------------------------------------------


def test_the_cfr_is_linear_in_the_transfer_coefficients():
    """``synth({a, b}) == synth({a}) + synth({b})``.

    The whole waveform is a sum of per-row phasors scaled by the coefficient, so
    linearity is structural. It is asserted because it is the property the
    cross-waveform invariant work depends on and the first thing an accidental
    normalisation by the row count would break.
    """

    spec = _spec(num_symbols=4)
    a = ([TAU_RT_S], [_frozen_channel_weight(0.75 - 0.2j)], [TAU_RATE])
    b = ([TAU_FAR_S], [0.3 + 0.9j], [-0.5 * TAU_RATE])
    together = _cube(
        spec,
        a[0] + b[0],
        a[1] + b[1],
        a[2] + b[2],
    )
    separate = _cube(spec, *a) + _cube(spec, *b)
    scale = float(together.abs().max())
    assert scale > 0.0
    torch.testing.assert_close(together, separate, rtol=1e-5, atol=1e-5 * scale)


def test_permuting_the_rows_of_one_segment_leaves_the_cube_unchanged():
    """Row order is not a physical fact; the segment sum is order-free.

    Up to float32 accumulation order, which is what the tolerance is for.
    """

    spec = _spec(num_symbols=3)
    delays = [TAU_RT_S, TAU_FAR_S, 3.0e-8]
    weights = [_frozen_channel_weight(), 0.5 + 0.25j, -0.75 + 0.1j]
    rates = [TAU_RATE, 0.0, -TAU_RATE]
    order = [2, 0, 1]
    straight = _cube(spec, delays, weights, rates)
    permuted = _cube(
        spec,
        [delays[i] for i in order],
        [weights[i] for i in order],
        [rates[i] for i in order],
    )
    scale = float(straight.abs().max())
    torch.testing.assert_close(straight, permuted, rtol=1e-6, atol=1e-6 * scale)


# --------------------------------------------------------------------------
# The float64 oracle, and the closed form as built
# --------------------------------------------------------------------------


def test_the_kernel_matches_the_float64_reference_cube():
    """Independent reimplementation, three segments, live Doppler.

    The oracle is in ``tests/support/reference_ofdm`` and shares no expression
    with the beat oracle: the two waveforms disagree about the phasor sign,
    about which quantity the carrier rate multiplies, and about what slow time
    means.
    """

    from support import reference_ofdm as ref

    spec = _spec(num_symbols=5)
    delays = [TAU_RT_S, TAU_FAR_S, 3.0e-8, 1.1e-8, 7.0e-9]
    weights = [
        _frozen_channel_weight(),
        0.5 + 0.25j,
        -0.75 + 0.1j,
        0.2 - 0.6j,
        1.3 + 0.05j,
    ]
    rates = [TAU_RATE, -TAU_RATE, 0.0, 2.0 * TAU_RATE, -0.5 * TAU_RATE]
    offsets = [0, 2, 2, 5]

    measured = _cube(spec, delays, weights, rates, offsets)
    expected = ref.cfr_cube(
        torch.tensor(delays, dtype=torch.float32).double(),
        torch.tensor(rates, dtype=torch.float32).double(),
        torch.tensor(weights, dtype=torch.complex128),
        torch.tensor(offsets, dtype=torch.int64),
        spec,
    )
    scale = float(expected.abs().max())
    assert scale > 0.0
    torch.testing.assert_close(measured, expected, rtol=1e-5, atol=1e-6 * scale)


def test_the_carrier_rate_multiplies_the_drift_and_the_subcarrier_the_full_delay():
    """The asymmetry, isolated: it is the likeliest error in this kernel.

    Two probes of the same cube at ``l = 0``, where the drift is exactly zero:

    * the subcarrier phase is fully present, because ``n * df`` multiplies
      ``tau_rt`` even though nothing has drifted;
    * the carrier-rate phase is exactly absent, because it multiplies the drift.

    A kernel that gave the carrier rate the full delay would put a constant
    ``-2 pi f_ref tau_rt`` on top of a weight that already carries it, breaking
    the ``H[0][p][0] == C_rt`` identity. A kernel that gave the subcarrier term
    the drift would leave ``H[0][p][n]`` independent of ``n``, which is a CFR
    with no range information in it at all.
    """

    spec = _spec(num_symbols=2)
    cube = _cube(spec, [TAU_RT_S], [_frozen_channel_weight()], [TAU_RATE])

    # Drift is zero at l = 0: the carrier-rate term contributes nothing.
    assert complex(cube[0, 0, 0]) == pytest.approx(
        complex(torch.tensor(_frozen_channel_weight(), dtype=torch.complex64)),
        rel=1e-6,
    )
    # ...while the subcarrier term is fully present at the same symbol.
    step = float(torch.angle(cube[0, 0, 1] * torch.conj(cube[0, 0, 0])))
    assert step == pytest.approx(
        -2.0 * math.pi * DF_HZ * _stored(TAU_RT_S), abs=2.0e-6
    )
    # A CFR with a drift-only subcarrier term would be flat across n.
    assert abs(step) > 1e-3


# --------------------------------------------------------------------------
# One real multi-endpoint frame, with its launch, host and memory budget
# --------------------------------------------------------------------------


HOST_OBSERVERS = ("item", "cpu", "tolist", "numpy")

OFDM_OPERATORS = ("ofdm_cfr_forward", "ofdm_cfr_backward", "ofdm_cfr_jvp")


class _FrameLedger:
    """Count native launches and host observations while it is active."""

    def __init__(self, monkeypatch, operators) -> None:
        self.launches = dict.fromkeys(OFDM_OPERATORS, 0)
        self.host = dict.fromkeys((*HOST_OBSERVERS, "synchronize"), 0)
        for name in self.launches:
            original = getattr(operators, name)

            def counting(*args, _name=name, _original=original, **kwargs):
                self.launches[_name] += 1
                return _original(*args, **kwargs)

            monkeypatch.setattr(operators, name, counting)
        for name in HOST_OBSERVERS:
            original_method = getattr(torch.Tensor, name)

            def observing(
                tensor, *args, _name=name, _original=original_method, **kwargs
            ):
                self.host[_name] += 1
                return _original(tensor, *args, **kwargs)

            monkeypatch.setattr(torch.Tensor, name, observing)
        original_sync = torch.cuda.synchronize

        def counting_sync(*args, **kwargs):
            self.host["synchronize"] += 1
            return original_sync(*args, **kwargs)

        monkeypatch.setattr(torch.cuda, "synchronize", counting_sync)


@pytest.fixture(scope="module")
def multi_endpoint_spike():
    pytest.importorskip("witwin.channel")
    from support import multi_endpoint_driver as drv

    return drv.MultiEndpointSpike()


def _fixture_spec(num_symbols: int) -> OfdmCfrSpec:
    from support import multi_endpoint_geometry as geo

    return OfdmCfrSpec(
        num_subcarriers=NUM_SUBCARRIERS,
        num_symbols=num_symbols,
        subcarrier_spacing_hz=DF_HZ,
        cyclic_prefix_s=CYCLIC_PREFIX_S,
        reference_frequency_hz=geo.REFERENCE_FREQUENCY_HZ,
        max_expected_delay_s=MAX_DELAY_S,
        carrier_hz=0.0,
        carrier_rate_hz=geo.REFERENCE_FREQUENCY_HZ,
    )


def test_a_real_multi_endpoint_frame_synthesizes_and_assembles(multi_endpoint_spike):
    """2 TX x 2 RX, eleven composed rows, four pairs of which two are empty.

    The same frozen topology the FMCW stage uses, through the OFDM owner. The
    rank-4 packing is shared - it is structural, not waveform-specific - so an
    OFDM frame lands in ``(TX, RX, symbol, subcarrier)`` through exactly the
    same call, and the empty pairs survive as empty CHANNELS.
    """

    from support import multi_endpoint_driver as drv
    from witwin.radar.synthesis.assembly import assemble_frame_cube
    from witwin.radar.synthesis.ofdm_cfr import synthesize_ofdm_cfr

    composed, _, _ = multi_endpoint_spike.frame()
    spec = _fixture_spec(3)
    batch = drv.to_synthesis(composed)
    assert batch.sensor_pair_count == 4

    cube = synthesize_ofdm_cfr(batch, spec)
    assert tuple(cube.shape) == (3, 4, NUM_SUBCARRIERS)
    assert cube.dtype == torch.complex64

    frame = assemble_frame_cube(cube, num_tx=2, num_rx=2)
    assert tuple(frame.shape) == (2, 2, 3, NUM_SUBCARRIERS)
    for rx in range(2):
        assert float(frame[0, rx].abs().sum()) > 0.0
        assert float(frame[1, rx].abs().sum()) == 0.0


def test_one_ofdm_frame_is_one_launch_and_no_host_observation(
    multi_endpoint_spike, monkeypatch
):
    """Acceptance: exactly one ``ofdm_cfr_forward`` per frame, and no D2H.

    The host budget is measured over synthesis and assembly only, because the
    frame's two sanctioned ``.item()`` copies belong to the two frozen legs and
    are already pinned by ``test_phase5_budget``.
    """

    from support import multi_endpoint_driver as drv
    from witwin.radar.synthesis import ofdm_cfr
    from witwin.radar.synthesis.assembly import assemble_frame_cube

    composed, _, _ = multi_endpoint_spike.frame()
    spec = _fixture_spec(3)
    batch = drv.to_synthesis(composed)
    operators = ofdm_cfr._ops()

    ledger = _FrameLedger(monkeypatch, operators)
    cube = ofdm_cfr.synthesize_ofdm_cfr(batch, spec)
    assemble_frame_cube(cube, num_tx=2, num_rx=2)

    assert ledger.launches == {
        "ofdm_cfr_forward": 1,
        "ofdm_cfr_backward": 0,
        "ofdm_cfr_jvp": 0,
    }, ledger.launches
    assert ledger.host == dict.fromkeys((*HOST_OBSERVERS, "synchronize"), 0), (
        ledger.host
    )


def test_one_backward_launch_per_forward_launch(multi_endpoint_spike, monkeypatch):
    from support import multi_endpoint_driver as drv
    from witwin.radar.synthesis import ofdm_cfr

    composed, _, _ = multi_endpoint_spike.frame(
        response=drv.make_response(requires_grad=True)
    )
    spec = _fixture_spec(3)
    batch = drv.to_synthesis(composed)
    operators = ofdm_cfr._ops()

    ledger = _FrameLedger(monkeypatch, operators)
    cube = ofdm_cfr.synthesize_ofdm_cfr(batch, spec)
    (cube.real.sum() + cube.imag.sum()).backward()
    assert ledger.launches["ofdm_cfr_forward"] == 1
    assert ledger.launches["ofdm_cfr_backward"] == 1
    assert ledger.launches["ofdm_cfr_jvp"] == 0


def test_the_forward_allocates_no_per_path_per_subcarrier_intermediate():
    """Peak allocation stays within 2x of the output plus the inputs.

    A ``K x L x N_sc`` materialisation - the shape a Torch replay of this sum
    would produce - fails this immediately at any interesting row count, which
    is the point of measuring it rather than asserting the absence of a loop.
    """

    spec = _spec(num_symbols=32)
    rows = 512
    generator = torch.Generator(device="cuda").manual_seed(20260725)
    tau = (
        torch.rand(rows, generator=generator, device="cuda", dtype=torch.float32)
        * 5.0e-7
    )
    rate = torch.zeros(rows, dtype=torch.float32, device="cuda")
    transfer = torch.ones(rows, dtype=torch.complex64, device="cuda")
    offsets = torch.tensor([0, rows], dtype=torch.int64, device="cuda")

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    cube = synthesize_cfr_rows(tau, rate, transfer, offsets, spec)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - before

    output_bytes = 2 * 4 * spec.num_symbols * 1 * spec.num_subcarriers
    # The complex recombination doubles the output once, which is the whole of
    # the allowed factor; the per-row inputs are already resident.
    budget = 2.0 * (output_bytes + 2 * 4 * rows) + 2 * output_bytes
    assert tuple(cube.shape) == (32, 1, NUM_SUBCARRIERS)
    assert peak <= budget, (peak, budget)
    # Non-vacuity: one K x L x N_sc float32 intermediate would be 4 MB.
    assert 4 * rows * spec.num_symbols * spec.num_subcarriers > budget
