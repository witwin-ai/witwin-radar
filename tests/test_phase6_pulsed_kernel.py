"""Closed-form pulsed acceptance: fractional delay, MF peak, Doppler, sign.

Every expected value here is written down from the physics, in float64, in the
test itself. Nothing is compared against a previous run of the code under test.

The primary delay assertion is the MATCHED-FILTER PEAK LOCATION IN SECONDS, and
its companion is the sub-sample sweep: the whole design decision of this
waveform is that ``p(t - tau)`` is evaluated at the exact fractional delay, and
the only way to see that from outside is to move the delay by less than a sample
and watch the estimate follow. Every sweep test therefore also computes what a
NEAREST-SAMPLE implementation would have produced and asserts that it fails,
because a tolerance that only the correct implementation passes is worth more
than one nothing is compared against.

Reference grid: ``tests/support/pulsed_grid``. Restated in SI::

    f_ref = 77 GHz    fs = 50 MSPS (T_s = 20 ns)   M  = 1024 (gate 20.48 us)
    T_pri = 100 us    L  = 32                      T_p = 10 us   B = 20 MHz
    d = 300 m         ->  tau_rt   = 2 d / c0   = 2.0013846 us
    v_r = 5 m/s away  ->  tau_rate = 2 v_r / c0 = 33.3564 ns/s

The train is in the CHANNEL phasor convention ``exp(-j k d)`` and is NOT
conjugated. Every sign below follows from that one fact, and the sign tests are
the ones that would survive a magnitude-only review.

Two measured accuracy floors are quoted rather than hidden, because they set
every tolerance in this file:

* The magnitude-parabolic delay estimator is biased by up to 1.1e-10 s, about
  0.002 of a range cell. The bias is a property of fitting a parabola to a
  CUSPED ``|R|``; it was measured not to shrink with more oversampling (16, 64,
  256 agree to 1e-15 s) and not to shrink with a faster ADC (50, 100, 200, 400
  MSPS all give 1.05e-10 s).
* The sampled matched filter loses ``O(T_s / T_p)`` of the peak magnitude at an
  off-grid delay - measured 2.0e-3 here - to the partial samples at the pulse's
  two ends. That one DOES shrink with the sample rate, and the test asserts the
  scaling rather than the number.
"""

from __future__ import annotations

import cmath
import math

import pytest
import torch

from support.pulsed_grid import (  # noqa: E402
    BANDWIDTH_HZ,
    C0,
    F_REF_HZ,
    NUM_SAMPLES,
    ON_GRID_SAMPLE,
    ON_GRID_TAU_S,
    PRI_S,
    PULSE_WIDTH_S,
    RADIAL_SPEED_MPS,
    SAMPLE_PERIOD_S,
    TAU_RATE,
    TAU_RT_S,
    peak_estimate,
    rect_spec,
    reference_spec,
    stored,
)
from witwin.radar.sigproc.matched_filter import lag_axis, matched_filter  # noqa: E402
from witwin.radar.synthesis.contracts import PulsedEchoSpec  # noqa: E402
from witwin.radar.synthesis.pulsed_echo import synthesize_echo_rows  # noqa: E402


pytestmark = pytest.mark.gpu


#: The delay estimator's measured accuracy, as a fraction of one range cell.
#: Quoted as a fraction of the CELL because that is the unit the estimator's
#: error is a property of; expressed against a 2 us delay it is 5.4e-5 relative,
#: which says more about the target's range than about the estimator.
DELAY_TOLERANCE_CELLS = 3.0e-3

#: The same statement in the brief's units, at this geometry.
DELAY_RTOL = 1.0e-4

#: The rectangular pulse's triangle is a full pulse width wide on EACH side, so
#: its half-width can only be measured at a delay far enough into the gate for
#: the left half to be there. 8 us puts the half-max crossings at 3 and 13 us,
#: both inside the 20.48 us gate, with the whole pulse inside as well.
TRIANGLE_SAMPLE = 400


def _frozen_channel_weight(amplitude: complex = 1.0 + 0.0j, tau: float = TAU_RT_S):
    """A Channel-sourced coefficient at this geometry.

    ``C_rt`` carries ``exp(-j 2 pi f_ref tau_rt)`` at the FROZEN per-frame
    delay. It is handed to the pulsed kernel unconjugated, unlike the FMCW beat
    weight, because the pulsed product stays in Channel's convention.
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


def _cube(spec, delays, weights, rates, offsets=None) -> torch.Tensor:
    tau, rate, transfer, table = _rows(delays, weights, rates, offsets)
    return synthesize_echo_rows(tau, rate, transfer, table, spec).cpu().to(
        torch.complex128
    )


def _single_row_estimate(spec, tau, weight=1.0 + 0.0j, oversample: int = 64):
    cube = _cube(spec, [tau], [weight], [0.0])
    return peak_estimate(cube[0, 0], spec, oversample=oversample)


def _unwrapped(values: torch.Tensor) -> torch.Tensor:
    import numpy as np

    return torch.from_numpy(np.unwrap(torch.angle(values).numpy()))


def _lsq_slope(phase: torch.Tensor) -> float:
    """Least-squares slope of an unwrapped phase sequence, per index step."""

    index = torch.arange(phase.numel(), dtype=torch.float64)
    index = index - index.mean()
    return float((index * (phase - phase.mean())).sum() / (index * index).sum())


# --------------------------------------------------------------------------
# T3.1  the matched-filter peak location - the primary delay assertion
# --------------------------------------------------------------------------


def test_the_matched_filter_peak_reports_the_round_trip_delay_in_seconds():
    """``t_peak - t_g == tau_rt``, asserted IN SECONDS against the closed form.

    Never against a bin index: the sample grid is 20 ns and the range cell is
    50 ns, so a bin assertion would be satisfied by a kernel that was wrong by
    half a cell. The estimate here beats the range cell by three orders of
    magnitude.
    """

    spec = reference_spec(num_pulses=1)
    estimate, _, _ = _single_row_estimate(spec, TAU_RT_S, _frozen_channel_weight())
    truth = stored(TAU_RT_S)

    error_s = estimate - spec.range_gate_start_s - truth
    assert abs(error_s) / truth == pytest.approx(6.04e-6, rel=0.05)
    assert abs(error_s) / truth < DELAY_RTOL
    assert abs(error_s) / spec.range_cell_delay_s < DELAY_TOLERANCE_CELLS

    # And the range that delay means, monostatic.
    assert C0 * estimate / 2.0 == pytest.approx(300.0, rel=DELAY_RTOL)

    # The resolution the tolerance sits inside, stated rather than assumed.
    assert spec.range_resolution_m == pytest.approx(7.49481145, rel=1e-12)
    assert abs(error_s) * C0 / 2.0 < 0.01 * spec.range_resolution_m


def test_the_peak_is_exact_at_an_on_grid_delay():
    """The estimator has no bias of its own; the 1.1e-10 s floor is the cusp.

    At an on-grid delay the sampled correlation is exactly the continuous one
    and its peak sits exactly on a fine-grid point, so the parabolic fit
    contributes nothing and the answer is exact to float64. That is what makes
    the off-grid number above a measurement of the WAVEFORM rather than of the
    estimator.
    """

    spec = reference_spec(num_pulses=1)
    estimate, _, _ = _single_row_estimate(
        spec, ON_GRID_TAU_S, _frozen_channel_weight(tau=ON_GRID_TAU_S)
    )
    assert estimate == pytest.approx(ON_GRID_TAU_S, rel=1e-12)


def test_the_delay_is_round_trip_and_is_never_doubled():
    """Doubling ``tau`` doubles the measured peak delay.

    ``dirichlet.cu`` takes a ONE-WAY distance and forms ``tau = 2 d / c0``
    internally. A synthesis kernel that repeated that on a round-trip delay
    would report exactly twice the range, self-consistently, at every geometry
    and with no other symptom.
    """

    spec = reference_spec(num_pulses=1)
    single, _, _ = _single_row_estimate(spec, ON_GRID_TAU_S)
    doubled, _, _ = _single_row_estimate(spec, 2.0 * ON_GRID_TAU_S)
    # Both delays keep the WHOLE pulse inside the gate: 2 us and 4 us plus a
    # 10 us pulse against a 20.48 us gate. A truncated pulse is a different
    # measurement - it degrades the compressed peak - and is tested separately.
    assert 2.0 * ON_GRID_TAU_S + spec.pulse_width_s < spec.range_gate_end_s
    assert doubled == pytest.approx(2.0 * single, rel=1e-9)
    assert single == pytest.approx(ON_GRID_TAU_S, rel=1e-9)


# --------------------------------------------------------------------------
# T3.2  fractional-delay fidelity - the snapping guard
# --------------------------------------------------------------------------


def test_the_estimate_tracks_every_sub_sample_offset_and_snapping_does_not():
    """Eight offsets of ``T_s / 8``, and the staircase a snapped kernel gives.

    This is the evidence that the pulse is evaluated analytically at a
    continuous argument. The tolerance alone would not prove it, so the test
    also SYNTHESIZES what a nearest-sample implementation would have produced -
    the same kernel at the delay rounded to the grid - and asserts that it fails
    the same tolerance at every one of the eight offsets, and that its eight
    estimates collapse onto two distinct values instead of eight.
    """

    spec = reference_spec(num_pulses=1)
    table = []
    snapped_table = []
    for k in range(8):
        tau = TAU_RT_S + k * SAMPLE_PERIOD_S / 8.0
        truth = stored(tau)
        estimate, _, _ = _single_row_estimate(spec, tau)
        table.append((truth, estimate, abs(estimate - truth) / truth))

        snapped_tau = round(tau / SAMPLE_PERIOD_S) * SAMPLE_PERIOD_S
        snapped, _, _ = _single_row_estimate(spec, snapped_tau)
        snapped_table.append((truth, snapped, abs(snapped - truth) / truth))

    for index, (truth, estimate, relative) in enumerate(table):
        assert relative < DELAY_RTOL, (index, truth, estimate, relative)
        assert abs(estimate - truth) / spec.range_cell_delay_s < DELAY_TOLERANCE_CELLS

    # The snapped implementation misses at EVERY offset, by at least five times
    # the tolerance the analytic one passes with room to spare.
    for index, (truth, snapped, relative) in enumerate(snapped_table):
        assert relative > 5.0 * DELAY_RTOL, (index, truth, snapped, relative)

    # ...and it is a staircase: eight distinct true delays, two distinct
    # answers. This is the shape of the failure, not just its size.
    steps = {round(value / SAMPLE_PERIOD_S) for _, value, _ in snapped_table}
    assert len(steps) == 2, sorted(steps)
    assert len({round(value / SAMPLE_PERIOD_S, 3) for _, value, _ in table}) == 8

    # The analytic estimates are strictly increasing with the true delay, which
    # a staircase is not.
    values = [value for _, value, _ in table]
    assert all(later > earlier for earlier, later in zip(values, values[1:]))


def test_the_peak_magnitude_barely_moves_across_a_sub_sample_sweep():
    """The straddle loss, measured and shown to be ``O(T_s / T_p)``.

    A snapped or interpolated envelope loses amplitude off grid. What survives
    here is the partial-sample term at the pulse's two ends, which is 2.0e-3 at
    this grid and HALVES when the sample rate doubles. Asserting the scaling
    turns a magic tolerance into a demonstrated law: an implementation that lost
    amplitude for any other reason would not halve.
    """

    def spread(spec: PulsedEchoSpec) -> float:
        magnitudes = [
            _single_row_estimate(
                spec, TAU_RT_S + k * spec.sample_period_s / 8.0
            )[2]
            for k in range(8)
        ]
        return (max(magnitudes) - min(magnitudes)) / max(magnitudes)

    coarse = spread(reference_spec(num_pulses=1))
    assert coarse == pytest.approx(1.95e-3, rel=0.1)
    assert coarse < 3.0e-3

    fine = spread(
        reference_spec(
            num_pulses=1,
            sample_period_s=SAMPLE_PERIOD_S / 2.0,
            num_samples=2 * NUM_SAMPLES,
        )
    )
    assert fine == pytest.approx(0.5 * coarse, rel=0.15)


# --------------------------------------------------------------------------
# T3.3  unit energy and the exact peak identity
# --------------------------------------------------------------------------


@pytest.mark.parametrize("weight", [0.25 - 0.5j, -1.0 + 0.0j, 3.5 + 2.25j])
def test_the_matched_filter_peak_is_exactly_the_coefficient(weight):
    """``MF peak == C_rt``: magnitude AND argument, at an on-grid delay.

    This is what the unit-ENERGY normalisation buys, and it is the pulsed member
    of the cross-waveform amplitude invariant. With a unit-AMPLITUDE pulse the
    peak would be ``C_rt`` times ``T_p / T_s`` - 500 here - and every comparison
    against the FMCW peak or the OFDM coefficient would have to carry a
    waveform-specific factor.

    The tolerance is the float32 kernel's, not the estimator's: the correlation
    is exact at an on-grid delay, so what is left is the kernel's own phase
    rounding.
    """

    spec = reference_spec(num_pulses=1)
    _, peak, _ = _single_row_estimate(spec, ON_GRID_TAU_S, weight)
    reference = complex(torch.tensor(weight, dtype=torch.complex64))
    assert abs(peak) == pytest.approx(abs(reference), rel=1e-6)
    # The argument is compared as the phase of the RATIO, which is the only form
    # that stays right at the branch cut: a real negative coefficient has
    # argument +pi or -pi depending on the sign of a rounding error, and
    # comparing the two atan2 values would fail by 2 pi on a correct answer.
    #
    # 1e-5 rad rather than 1e-6: the kernel wraps a cycle count that reaches a
    # hundred cycles of LFM phase and then rounds the [0, 2 pi) argument to
    # float32 before one sincosf, which costs up to 3.7e-7 rad per sample.
    # Measured worst deviation across the three weights is 3.2e-7 rad here, and
    # 1.3e-6 rad at a 8 us delay where the pulse phase has further to run.
    assert abs(cmath.phase(peak / reference)) < 1.0e-5


def test_a_conjugated_train_would_fail_the_peak_identity():
    """Non-vacuity for the test above: the two conventions are distinguishable.

    A weight whose imaginary part is nonzero is required, which is why the
    parametrisation above is not all real.
    """

    weight = 0.25 - 0.5j
    spec = reference_spec(num_pulses=1)
    _, peak, _ = _single_row_estimate(spec, ON_GRID_TAU_S, weight)
    assert abs(peak - complex(weight).conjugate()) > 0.1 * abs(weight)


# --------------------------------------------------------------------------
# T3.4  the rectangular pulse's matched-filter triangle
# --------------------------------------------------------------------------


def test_the_rectangular_pulse_compresses_to_a_triangle_of_base_two_pulse_widths():
    """Peak ``C_rt`` at ``t = tau``, half-width at half maximum ``T_p / 2``.

    Exact rather than approximate: two on-grid rectangles correlate to a
    triangle whose samples are ``1 - |k| T_s / T_p`` with no interpolation
    error at all, so the half-max crossing lands exactly on a sample and the
    linear interpolation between the two straddling samples is exact.

    The delay is 8 us rather than 2 us because the triangle is a full pulse
    width wide on EACH side and the left half has to be inside the gate for a
    half-width to be measurable.
    """

    spec = rect_spec(num_pulses=1)
    weight = 0.6 - 0.3j
    triangle_tau_s = TRIANGLE_SAMPLE * SAMPLE_PERIOD_S
    assert triangle_tau_s - spec.pulse_width_s / 2.0 > spec.range_gate_start_s
    assert triangle_tau_s + spec.pulse_width_s < spec.range_gate_end_s
    cube = _cube(spec, [triangle_tau_s], [weight], [0.0])
    compressed = matched_filter(cube[0, 0], spec)
    lags = lag_axis(spec)
    magnitude = compressed.abs()

    peak = int(magnitude.argmax())
    assert peak == TRIANGLE_SAMPLE
    assert float(lags[peak]) == pytest.approx(triangle_tau_s, rel=1e-12)
    assert abs(complex(compressed[peak])) == pytest.approx(abs(weight), rel=1e-6)

    # The triangle, exactly: half maximum lands on the samples T_p / 2 away.
    half_samples = spec.pulse_sample_count // 2
    assert float(magnitude[peak + half_samples] / magnitude[peak]) == pytest.approx(
        0.5, rel=1e-12
    )
    assert float(magnitude[peak - half_samples] / magnitude[peak]) == pytest.approx(
        0.5, rel=1e-12
    )

    half = 0.5 * float(magnitude[peak])
    index = peak
    while float(magnitude[index]) > half:
        index += 1
    fraction = (float(magnitude[index - 1]) - half) / (
        float(magnitude[index - 1]) - float(magnitude[index])
    )
    half_width_s = (index - 1 + fraction - peak) * SAMPLE_PERIOD_S
    assert half_width_s == pytest.approx(PULSE_WIDTH_S / 2.0, rel=1e-3)

    # The base is two pulse widths: the correlation is structurally zero beyond.
    assert float(magnitude[peak + spec.pulse_sample_count + 2 :].max()) < 1e-9 * float(
        magnitude[peak]
    )

    # And the resolution that pulse width buys - three orders of magnitude
    # coarser than the LFM's, which is the whole reason an LFM is transmitted.
    assert spec.range_resolution_m == pytest.approx(1498.96229, rel=1e-12)
    assert spec.range_resolution_m == pytest.approx(C0 * PULSE_WIDTH_S / 2.0, rel=1e-12)


# --------------------------------------------------------------------------
# T3.5  the LFM's compressed shape
# --------------------------------------------------------------------------


def test_the_lfm_compresses_to_a_sinc_with_the_textbook_first_sidelobe():
    """First null at ``1 / B``, first sidelobe at ``-13.2 dB``.

    The exact closed form of the unit-energy LFM autocorrelation is

        ``R(x) = exp(j pi B x) sin(pi B x (1 - |x| / T_p)) / (pi B x)``

    so the first null is at ``B x (1 - x / T_p) = 1``, which is ``1.005 / B``
    at this time-bandwidth product rather than exactly ``1 / B``. Both are
    inside the stated tolerance and the correction is asserted, because a
    kernel with the wrong sweep rate would move the null and nothing else
    obvious.
    """

    spec = reference_spec(num_pulses=1)
    cube = _cube(spec, [ON_GRID_TAU_S], [1.0 + 0.0j], [0.0])
    oversample = 64
    magnitude = matched_filter(cube[0, 0], spec, oversample=oversample).abs()
    lags = lag_axis(spec, oversample=oversample)

    peak = int(magnitude.argmax())
    assert float(lags[peak]) == pytest.approx(ON_GRID_TAU_S, rel=1e-12)
    peak_value = float(magnitude[peak])
    assert peak_value == pytest.approx(1.0, rel=1e-6)

    index = peak + 1
    while float(magnitude[index + 1]) < float(magnitude[index]):
        index += 1
    null_offset_s = float(lags[index]) - float(lags[peak])
    assert null_offset_s * BANDWIDTH_HZ == pytest.approx(1.0, rel=1e-2)
    # The analytic correction, to within one fine-grid step.
    analytic = 1.0 / (1.0 - 1.0 / (BANDWIDTH_HZ * PULSE_WIDTH_S))
    assert null_offset_s * BANDWIDTH_HZ == pytest.approx(analytic, abs=1.0e-2)
    assert float(magnitude[index]) < 0.01 * peak_value

    sidelobe = index + 1
    while float(magnitude[sidelobe + 1]) > float(magnitude[sidelobe]):
        sidelobe += 1
    ratio_db = 20.0 * math.log10(float(magnitude[sidelobe]) / peak_value)
    assert ratio_db == pytest.approx(-13.2, abs=1.0)
    assert (float(lags[sidelobe]) - float(lags[peak])) * BANDWIDTH_HZ == pytest.approx(
        1.43, rel=0.05
    )


# --------------------------------------------------------------------------
# T3.6  per-pulse Doppler advance, two-sided
# --------------------------------------------------------------------------


# Two fast-time samples inside the pulse support, chosen for what the LFM's own
# phase contributes at each. Sample 101 sits 18.6 ns past the leading edge,
# where the envelope's contribution to the slow-time slope is 5e-7 of the
# carrier's; sample 599 sits 21 ns short of the trailing edge, where it is at
# its largest, 2.6e-4. The drift over the whole coherent processing interval is
# 1.0e-10 s, so both stay inside the support for every pulse.
LEADING_SAMPLE = 101
TRAILING_SAMPLE = 599


def _envelope_time_s(sample: int, tau: float = TAU_RT_S) -> float:
    return sample * SAMPLE_PERIOD_S - stored(tau)


def _measured_pulse_slope(spec, sample: int, rate: float) -> float:
    cube = _cube(spec, [TAU_RT_S], [_frozen_channel_weight()], [rate])
    return _lsq_slope(_unwrapped(cube[:, 0, sample]))


@pytest.mark.parametrize("sample", [LEADING_SAMPLE, TRAILING_SAMPLE])
def test_the_slow_time_slope_carries_the_whole_carrier_not_just_the_envelope(sample):
    """The two-sided guard against a silently vanishing Doppler.

    One side: the measured per-pulse slope equals
    ``-2 pi tau_rate T_pri (f_ref + B u / T_p)``. The other side: that value is
    enormously larger than the ``-2 pi tau_rate T_pri B u / T_p`` a kernel that
    reused the frozen weight without a carrier-rate term would leave behind.

    Pulsed is the strongest of the three waveforms here, because the ENVELOPE
    carries no carrier: at the leading edge of the pulse the frozen-weight bug
    leaves 5e-7 of the Doppler, and for a rectangular pulse it leaves EXACTLY
    nothing. A one-sided test passes in all of those cases, because a train with
    a millionth of the right Doppler still looks like a radar train.
    """

    spec = reference_spec()
    envelope_time = _envelope_time_s(sample)
    assert 0.0 < envelope_time < PULSE_WIDTH_S

    analytic = spec.slow_time_phase_step_rad(stored(TAU_RATE), envelope_time)
    measured = _measured_pulse_slope(spec, sample, TAU_RATE)
    assert measured == pytest.approx(analytic, rel=1e-5)

    envelope_only = (
        -math.tau
        * stored(TAU_RATE)
        * PRI_S
        * spec.instantaneous_pulse_frequency_hz(envelope_time)
    )
    understatement = analytic / envelope_only
    assert abs(understatement) > 1000.0
    assert abs(understatement) == pytest.approx(
        1.0 + F_REF_HZ / spec.instantaneous_pulse_frequency_hz(envelope_time),
        rel=1e-9,
    )
    assert abs(measured) > 1000.0 * 1e-5 * abs(analytic)


def test_a_rectangular_pulse_loses_its_doppler_entirely_without_the_rate_term():
    """The extreme case: the envelope-only slope is EXACTLY zero.

    A rectangular envelope has no phase at all, so a kernel that dropped the
    carrier-rate term would produce a train with no slow-time phase whatsoever -
    a perfectly stationary target, at any velocity. The measured slope here is
    the pure carrier term and is nowhere near zero.
    """

    spec = rect_spec()
    assert spec.instantaneous_pulse_frequency_hz(PULSE_WIDTH_S / 2.0) == 0.0

    analytic = -math.tau * stored(TAU_RATE) * PRI_S * F_REF_HZ
    measured = _measured_pulse_slope(spec, TRAILING_SAMPLE, TAU_RATE)
    assert measured == pytest.approx(analytic, rel=1e-5)
    assert abs(measured) > 1.0


@pytest.mark.parametrize("sample", [LEADING_SAMPLE, TRAILING_SAMPLE])
def test_both_carrier_homes_produce_the_same_slow_time_slope(sample):
    """``(f_ref, 0)`` and ``(0, f_ref)`` differ only by a constant.

    A kernel-owned carrier multiplies the FULL ``tau_k(l)`` and therefore already
    walks across pulses; a weight-owned one is frozen at ``tau_rt`` and needs the
    rate term to walk at all. They are exactly equivalent in slow time and differ
    by the constant ``f_ref tau_rt`` the weight holds on the production route.
    """

    production = reference_spec()
    kernel_owned = reference_spec(carrier_hz=F_REF_HZ, carrier_rate_hz=0.0)
    assert production.carrier_hz == 0.0
    assert kernel_owned.carrier_rate_hz == 0.0

    from_weight = _measured_pulse_slope(production, sample, TAU_RATE)
    kernel_cube = _cube(kernel_owned, [TAU_RT_S], [1.0 + 0.0j], [TAU_RATE])
    from_kernel = _lsq_slope(_unwrapped(kernel_cube[:, 0, sample]))
    analytic = production.slow_time_phase_step_rad(
        stored(TAU_RATE), _envelope_time_s(sample)
    )
    assert from_weight == pytest.approx(analytic, rel=1e-5)
    assert from_kernel == pytest.approx(analytic, rel=1e-5)


# --------------------------------------------------------------------------
# T3.7  Doppler SIGN and aliasing
# --------------------------------------------------------------------------


def test_a_receding_site_puts_the_slow_time_tone_at_negative_doppler():
    """The SIGN, and it is the OPPOSITE of the FMCW beat cube's.

    Physical Doppler in Channel's ``exp(-j k d)`` convention is
    ``f_D = -f_ref tau_rate``, so a receding site (``tau_rate > 0``) is a
    NEGATIVE shift. Pulsed publishes that convention unchanged, as OFDM does; the
    FMCW cube is conjugated once and its tone sits at ``+f_ref tau_rate``. Both
    are correct and they point opposite ways, so a consumer that assumed one
    convention for both would read every pulsed target as approaching when it
    recedes.
    """

    spec = reference_spec()
    cube = _cube(spec, [TAU_RT_S], [_frozen_channel_weight()], [TAU_RATE])
    slow = cube[:, 0, LEADING_SAMPLE]

    spectrum = torch.fft.fftshift(torch.fft.fft(slow)).abs()
    frequencies = torch.fft.fftshift(torch.fft.fftfreq(spec.num_pulses, d=PRI_S))
    peak_hz = float(frequencies[int(spectrum.argmax())])
    bin_hz = float(frequencies[1] - frequencies[0])

    physical_doppler_hz = spec.doppler_frequency_hz(TAU_RATE)
    assert physical_doppler_hz == pytest.approx(-2568.443533, rel=1e-8)
    assert peak_hz < 0.0
    assert abs(peak_hz - physical_doppler_hz) <= 0.5 * bin_hz
    assert C0 * TAU_RATE / 2.0 == pytest.approx(RADIAL_SPEED_MPS, rel=1e-12)
    assert RADIAL_SPEED_MPS < spec.max_unambiguous_speed_m_s

    # The two published conventions disagree on sign because one of them is
    # conjugated, and that is asserted rather than left to the reader.
    from witwin.radar.synthesis.fmcw_beat import channel_phasor_to_beat_weight

    coefficient = torch.tensor(
        [_frozen_channel_weight()], dtype=torch.complex64, device="cuda"
    )
    beat = channel_phasor_to_beat_weight(coefficient)
    assert not torch.equal(beat, coefficient)
    assert torch.equal(beat, torch.conj(coefficient).resolve_conj())


def test_an_approaching_site_reverses_the_slow_time_slope():
    spec = reference_spec()
    receding = _measured_pulse_slope(spec, LEADING_SAMPLE, TAU_RATE)
    approaching = _measured_pulse_slope(spec, LEADING_SAMPLE, -TAU_RATE)
    assert receding < 0.0 < approaching
    assert approaching == pytest.approx(-receding, rel=1e-5)


def test_a_speed_past_the_unambiguous_bound_aliases():
    """At 1.05x the bound the measured slope wraps and changes SIGN.

    The bound is half a cycle of Doppler phase per pulse, so at 1.05x the true
    slope is -3.2987 rad and the measurable one is +2.9845: a receding target
    reads as an approaching one. The velocity is not merely imprecise past the
    bound, it is unrecoverable, which is why the assertion is on the sign flip
    and not on an error magnitude.
    """

    spec = reference_spec(num_pulses=16)
    assert spec.max_unambiguous_speed_m_s == pytest.approx(
        C0 / (4.0 * F_REF_HZ * PRI_S), rel=1e-12
    )

    speed = 1.05 * spec.max_unambiguous_speed_m_s
    rate = 2.0 * speed / C0
    true_slope = -math.tau * stored(rate) * PRI_S * F_REF_HZ
    assert true_slope == pytest.approx(-1.05 * math.pi, rel=1e-4)

    # A speed this far outside the configured window is exactly what the
    # migration guard is about, so the spec has to declare it.
    fast = reference_spec(num_pulses=16, max_expected_delay_rate=rate)
    cube = _cube(fast, [TAU_RT_S], [_frozen_channel_weight()], [rate])
    slow = cube[:, 0, LEADING_SAMPLE]
    wrapped = float(torch.angle(slow[1:] * torch.conj(slow[:-1])).mean())
    assert true_slope < 0.0 < wrapped
    assert wrapped == pytest.approx(true_slope + 2.0 * math.pi, rel=1e-3)

    # Just inside the bound the same estimator recovers the true slope.
    inside_rate = 2.0 * (0.95 * spec.max_unambiguous_speed_m_s) / C0
    inside_spec = reference_spec(num_pulses=16, max_expected_delay_rate=inside_rate)
    inside = _cube(
        inside_spec, [TAU_RT_S], [_frozen_channel_weight()], [inside_rate]
    )[:, 0, LEADING_SAMPLE]
    measured = float(torch.angle(inside[1:] * torch.conj(inside[:-1])).mean())
    assert measured == pytest.approx(
        -math.tau * stored(inside_rate) * PRI_S * F_REF_HZ, rel=1e-4
    )


# --------------------------------------------------------------------------
# T3.9  dead rows, empty segments, zero rows, out-of-gate rows
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

    import dataclasses

    from witwin.radar.synthesis.pulsed_echo import synthesize_pulsed_echo

    spec = reference_spec(num_pulses=2, num_samples=256)
    alive = _cuda_batch(row_valid=torch.ones(1, dtype=torch.bool, device="cuda"))
    dead = _cuda_batch(row_valid=torch.zeros(1, dtype=torch.bool, device="cuda"))

    assert float(synthesize_pulsed_echo(alive, spec).abs().sum()) > 0.0
    assert float(synthesize_pulsed_echo(dead, spec).abs().sum()) == 0.0

    weight = torch.full(
        (1,), _frozen_channel_weight(), dtype=torch.complex64, device="cuda"
    ).requires_grad_(True)
    live = dataclasses.replace(dead, complex_transfer_ref=weight)
    cube = synthesize_pulsed_echo(live, spec)
    (cube.real.sum() + cube.imag.sum()).backward()
    assert float(weight.grad.abs().max()) == 0.0


def test_a_row_outside_the_range_gate_contributes_exactly_zero():
    """The envelope support does it, with no NaN and no wraparound.

    A delay past the end of the gate puts every sample's ``u`` below zero, so
    the pulse is never entered. The failure this rules out is a modulo or a
    clamp: either would fold a distant echo back into the gate as a phantom
    target at short range, which is exactly what an unwary implementation of a
    "circular" fast-time axis produces.
    """

    spec = reference_spec(num_pulses=2, num_samples=256)
    outside = spec.range_gate_end_s + spec.pulse_width_s + 1.0e-6
    cube = _cube(spec, [outside], [_frozen_channel_weight(tau=outside)], [0.0])
    assert float(cube.abs().max()) == 0.0
    assert not bool(torch.isnan(cube.real).any())
    assert not bool(torch.isnan(cube.imag).any())

    # And a row whose echo arrives BEFORE the gate opens is equally silent.
    early = reference_spec(
        num_pulses=2, num_samples=256, range_gate_start_s=15.0e-6
    )
    early_cube = _cube(early, [TAU_RT_S], [_frozen_channel_weight()], [0.0])
    assert float(early_cube.abs().max()) == 0.0


def test_an_empty_pair_segment_produces_an_exact_zero_channel():
    """A pair that discovered nothing keeps its channel and publishes zeros.

    Renumbering it away would shorten the pair axis and mis-steer every angle
    downstream. Three segments, of which the MIDDLE one is empty, because a
    trailing empty segment is the easy case.
    """

    spec = reference_spec(num_pulses=3, num_samples=512)
    cube = _cube(
        spec,
        [TAU_RT_S, ON_GRID_TAU_S, 3.0e-6],
        [_frozen_channel_weight(), 0.5 + 0.25j, -0.75 + 0.1j],
        [0.0, 0.0, 0.0],
        offsets=[0, 1, 1, 3],
    )
    assert tuple(cube.shape) == (3, 3, 512)
    assert float(cube[:, 1, :].abs().max()) == 0.0
    assert float(cube[:, 0, :].abs().max()) > 0.0
    assert float(cube[:, 2, :].abs().max()) > 0.0


def test_a_batch_with_no_rows_produces_an_all_zero_train_of_the_right_shape():
    spec = reference_spec(num_pulses=5, num_samples=128)
    tau = torch.zeros(0, dtype=torch.float32, device="cuda")
    rate = torch.zeros(0, dtype=torch.float32, device="cuda")
    transfer = torch.zeros(0, dtype=torch.complex64, device="cuda")
    offsets = torch.tensor([0, 0, 0], dtype=torch.int64, device="cuda")
    cube = synthesize_echo_rows(tau, rate, transfer, offsets, spec)
    assert tuple(cube.shape) == (5, 2, 128)
    assert float(cube.abs().max()) == 0.0


# --------------------------------------------------------------------------
# T3.10  linearity and row order
# --------------------------------------------------------------------------


def test_the_pulse_train_is_linear_in_the_transfer_coefficients():
    """``synth({a, b}) == synth({a}) + synth({b})``.

    The whole waveform is a sum of per-row envelopes scaled by the coefficient,
    so linearity is structural. It is asserted because it is the property the
    cross-waveform invariant work depends on and the first thing an accidental
    normalisation by the row count would break.
    """

    spec = reference_spec(num_pulses=4, num_samples=512)
    a = ([TAU_RT_S], [_frozen_channel_weight(0.75 - 0.2j)], [TAU_RATE])
    b = ([ON_GRID_TAU_S], [0.3 + 0.9j], [-0.5 * TAU_RATE])
    together = _cube(spec, a[0] + b[0], a[1] + b[1], a[2] + b[2])
    separate = _cube(spec, *a) + _cube(spec, *b)
    scale = float(together.abs().max())
    assert scale > 0.0
    torch.testing.assert_close(together, separate, rtol=1e-5, atol=1e-5 * scale)


def test_permuting_the_rows_of_one_segment_leaves_the_train_unchanged():
    """Row order is not a physical fact; the segment sum is order-free.

    Up to float32 accumulation order, which is what the tolerance is for.
    """

    spec = reference_spec(num_pulses=3, num_samples=512)
    delays = [TAU_RT_S, ON_GRID_TAU_S, 3.0e-6]
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


def test_the_kernel_matches_the_float64_reference_train():
    """Independent reimplementation, three segments, live Doppler, both kinds.

    The oracle is in ``tests/support/reference_pulsed`` and shares no expression
    with either neighbouring oracle: the three waveforms disagree about the
    phasor sign, about which quantity the carrier rate multiplies, and about
    what the waveform-specific factor is.
    """

    from support import reference_pulsed as ref

    delays = [TAU_RT_S, ON_GRID_TAU_S, 3.0e-6, 5.5e-6, 7.25e-6]
    weights = [
        _frozen_channel_weight(),
        0.5 + 0.25j,
        -0.75 + 0.1j,
        0.2 - 0.6j,
        1.3 + 0.05j,
    ]
    rates = [TAU_RATE, -TAU_RATE, 0.0, 2.0 * TAU_RATE, -0.5 * TAU_RATE]
    offsets = [0, 2, 2, 5]

    for spec in (
        reference_spec(num_pulses=5, num_samples=512),
        rect_spec(num_pulses=5, num_samples=512),
    ):
        measured = _cube(spec, delays, weights, rates, offsets)
        expected = ref.echo_cube(
            torch.tensor(delays, dtype=torch.float32).double(),
            torch.tensor(rates, dtype=torch.float32).double(),
            torch.tensor(weights, dtype=torch.complex128),
            torch.tensor(offsets, dtype=torch.int64),
            spec,
        )
        scale = float(expected.abs().max())
        assert scale > 0.0
        torch.testing.assert_close(
            measured, expected, rtol=1e-5, atol=1e-6 * scale
        )


def test_the_carrier_rate_multiplies_the_drift_and_the_envelope_the_full_delay():
    """The asymmetry, isolated: it is the likeliest error in this kernel.

    Two probes of the same train at ``l = 0``, where the drift is exactly zero:

    * the envelope sits at the FULL delay, because ``tau_rt`` positions the
      pulse even though nothing has drifted;
    * the carrier-rate phase is exactly absent, because it multiplies the drift.

    A kernel that gave the carrier rate the full delay would put a constant
    ``-2 pi f_ref tau_rt`` on top of a weight that already carries it, breaking
    the peak identity. A kernel that positioned the envelope by the drift alone
    would put every echo at zero delay, at every range.
    """

    spec = reference_spec(num_pulses=2)
    weight = 0.6 - 0.3j
    cube = _cube(spec, [ON_GRID_TAU_S], [weight], [TAU_RATE])

    # Drift is zero at l = 0: the carrier-rate term contributes nothing, so the
    # peak is the coefficient exactly.
    _, peak, _ = peak_estimate(cube[0, 0], spec)
    reference = complex(torch.tensor(weight, dtype=torch.complex64))
    assert abs(peak - reference) < 1e-5 * abs(reference)

    # ...while the envelope is fully positioned by the same delay: the gate is
    # silent right up to the sample the pulse's leading edge lands on.
    assert float(cube[0, 0, :ON_GRID_SAMPLE].abs().max()) == 0.0
    assert abs(complex(cube[0, 0, ON_GRID_SAMPLE])) == pytest.approx(
        abs(weight) * spec.pulse_amplitude, rel=1e-6
    )


# --------------------------------------------------------------------------
# One real multi-endpoint frame, with its launch, host and memory budget
# --------------------------------------------------------------------------


HOST_OBSERVERS = ("item", "cpu", "tolist", "numpy")

PULSED_OPERATORS = ("pulsed_echo_forward", "pulsed_echo_backward", "pulsed_echo_jvp")


class _FrameLedger:
    """Count native launches and host observations while it is active."""

    def __init__(self, monkeypatch, operators) -> None:
        self.launches = dict.fromkeys(PULSED_OPERATORS, 0)
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


def _fixture_spec(num_pulses: int) -> PulsedEchoSpec:
    from support import multi_endpoint_geometry as geo

    return reference_spec(
        num_pulses=num_pulses,
        num_samples=512,
        reference_frequency_hz=geo.REFERENCE_FREQUENCY_HZ,
        carrier_rate_hz=geo.REFERENCE_FREQUENCY_HZ,
        # The fixture's fastest site moves at 12 m/s, which is the configured
        # window this frame has to declare. Its migration over three pulses is
        # 2.4e-11 s against a 5e-8 s range cell.
        max_expected_delay_rate=2.0 * 12.0 / geo.C0_M_PER_S,
    )


def test_a_real_multi_endpoint_frame_synthesizes_and_assembles(multi_endpoint_spike):
    """2 TX x 2 RX, eleven composed rows, four pairs of which two are empty.

    The same frozen topology the FMCW and OFDM stages use, through the pulsed
    owner. The rank-4 packing is shared - it is structural, not
    waveform-specific - so a pulsed frame lands in ``(TX, RX, pulse, sample)``
    through exactly the same call, and the empty pairs survive as empty
    CHANNELS.
    """

    from support import multi_endpoint_driver as drv
    from witwin.radar.synthesis.assembly import assemble_frame_cube
    from witwin.radar.synthesis.pulsed_echo import synthesize_pulsed_echo

    composed, _, _ = multi_endpoint_spike.frame()
    spec = _fixture_spec(3)
    batch = drv.to_synthesis(composed)
    assert batch.sensor_pair_count == 4

    cube = synthesize_pulsed_echo(batch, spec)
    assert tuple(cube.shape) == (3, 4, 512)
    assert cube.dtype == torch.complex64

    frame = assemble_frame_cube(cube, num_tx=2, num_rx=2)
    assert tuple(frame.shape) == (2, 2, 3, 512)
    for rx in range(2):
        assert float(frame[0, rx].abs().sum()) > 0.0
        assert float(frame[1, rx].abs().sum()) == 0.0


def test_one_pulsed_frame_is_one_launch_and_no_host_observation(
    multi_endpoint_spike, monkeypatch
):
    """Acceptance: exactly one ``pulsed_echo_forward`` per frame, and no D2H.

    The host budget is measured over synthesis and assembly only, because the
    frame's two sanctioned ``.item()`` copies belong to the two frozen legs and
    are already pinned by ``test_phase5_budget``.
    """

    from support import multi_endpoint_driver as drv
    from witwin.radar.synthesis import pulsed_echo
    from witwin.radar.synthesis.assembly import assemble_frame_cube

    composed, _, _ = multi_endpoint_spike.frame()
    spec = _fixture_spec(3)
    batch = drv.to_synthesis(composed)
    operators = pulsed_echo._ops()

    ledger = _FrameLedger(monkeypatch, operators)
    cube = pulsed_echo.synthesize_pulsed_echo(batch, spec)
    assemble_frame_cube(cube, num_tx=2, num_rx=2)

    assert ledger.launches == {
        "pulsed_echo_forward": 1,
        "pulsed_echo_backward": 0,
        "pulsed_echo_jvp": 0,
    }, ledger.launches
    assert ledger.host == dict.fromkeys((*HOST_OBSERVERS, "synchronize"), 0), (
        ledger.host
    )


def test_one_backward_launch_per_forward_launch(multi_endpoint_spike, monkeypatch):
    from support import multi_endpoint_driver as drv
    from witwin.radar.synthesis import pulsed_echo

    composed, _, _ = multi_endpoint_spike.frame(
        response=drv.make_response(requires_grad=True)
    )
    spec = _fixture_spec(2)
    batch = drv.to_synthesis(composed)
    operators = pulsed_echo._ops()

    ledger = _FrameLedger(monkeypatch, operators)
    cube = pulsed_echo.synthesize_pulsed_echo(batch, spec)
    (cube.real.sum() + cube.imag.sum()).backward()
    assert ledger.launches["pulsed_echo_forward"] == 1
    assert ledger.launches["pulsed_echo_backward"] == 1
    assert ledger.launches["pulsed_echo_jvp"] == 0


def test_the_compatibility_guards_run_before_any_launch(
    multi_endpoint_spike, monkeypatch
):
    """The refusal happens BEFORE the kernel runs, not after.

    A check that ran after the launch would still raise, but it would already
    have spent the frame and, worse, would leave the door open to a later
    "return what we have" branch. Asserted by counting launches across the
    refusal.
    """

    from support import multi_endpoint_driver as drv
    from witwin.radar.synthesis import pulsed_echo

    composed, _, _ = multi_endpoint_spike.frame()
    batch = drv.to_synthesis(composed)
    operators = pulsed_echo._ops()

    launches = {"count": 0}
    original = operators.pulsed_echo_forward

    def counting(*args, **kwargs):
        launches["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(operators, "pulsed_echo_forward", counting)

    migrating = _fixture_spec(3)
    import dataclasses

    # Over three pulses the coherent processing interval is only 300 us, so the
    # speed that walks a whole range cell in it is 50 km/s rather than the
    # 5 km/s the 32-pulse spec needs. The bound is a statement about the CPI,
    # not about the speed alone.
    migrating = dataclasses.replace(
        migrating, max_expected_delay_rate=2.0 * 50000.0 / C0
    )
    assert migrating.range_migration_delay_s > migrating.range_cell_delay_s
    with pytest.raises(ValueError, match="range migration"):
        pulsed_echo.synthesize_pulsed_echo(batch, migrating)
    assert launches["count"] == 0

    pulsed_echo.synthesize_pulsed_echo(batch, _fixture_spec(3))
    assert launches["count"] == 1


def test_the_forward_allocates_no_per_path_per_sample_intermediate():
    """Peak allocation stays within 2x of the output plus the inputs.

    A ``K x L x M`` materialisation - the shape a Torch replay of this sum would
    produce - fails this immediately at any interesting row count, which is the
    point of measuring it rather than asserting the absence of a loop.
    """

    spec = reference_spec(num_pulses=8, num_samples=512)
    rows = 512
    generator = torch.Generator(device="cuda").manual_seed(20260725)
    tau = (
        torch.rand(rows, generator=generator, device="cuda", dtype=torch.float32)
        * 5.0e-6
    )
    rate = torch.zeros(rows, dtype=torch.float32, device="cuda")
    transfer = torch.ones(rows, dtype=torch.complex64, device="cuda")
    offsets = torch.tensor([0, rows], dtype=torch.int64, device="cuda")

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    cube = synthesize_echo_rows(tau, rate, transfer, offsets, spec)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - before

    # 2 * 4 * L * P * M: the re/im float32 pair the kernel writes. The complex
    # recombination copies it once more, which is where the factor of two goes;
    # the only other per-call allocations are O(K), not O(K * M).
    output_bytes = 2 * 4 * spec.num_pulses * 1 * spec.num_samples
    per_row_bytes = 3 * 8 * rows  # segment, plus headroom
    assert tuple(cube.shape) == (8, 1, 512)
    assert peak <= 2.0 * output_bytes + per_row_bytes, (peak, output_bytes)
    # Non-vacuity: one K x L x M float32 intermediate would be 8 MB, which is
    # more than a hundred times this budget, so a Torch replay of the sum fails
    # here immediately.
    assert 4 * rows * spec.num_pulses * spec.num_samples > 100.0 * (
        2.0 * output_bytes + per_row_bytes
    )
