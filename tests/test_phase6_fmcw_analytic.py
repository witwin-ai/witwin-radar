"""Closed-form FMCW acceptance: beat frequency, range, Doppler slope and sign.

Every expected value here is written down from the physics, in float64, in the
test itself. Nothing is compared against a previous run of the code under test,
and nothing is asserted in FFT bins: a bin is a property of the transform, and
a range or a velocity that is only right to within a bin has not been checked.

The reference geometry is the physics survey's probe geometry, restated in SI
so that the numbers can be recomputed by hand:

    fc = 77 GHz          S  = 60 MHz/us = 6.0e13 Hz/s
    fs = 5 MSPS          N  = 256 samples
    t0 = 6 us            Tc = 60 us
    d  = 3.7 m one way   ->  tau_rt = 2 d / c0 = 24.6837 ns
    v_r = 12 m/s away    ->  tau_rate = 2 v_r / c0 = 80.0554 ns/s

The whole file runs at ``num_tx = 1``; the TDM slot axis is
``test_phase6_fmcw_tdm``.
"""

from __future__ import annotations

import math

import pytest
import torch

from witwin.radar.synthesis.assembly import SPEED_OF_LIGHT_M_PER_S, FmcwSpec  # noqa: E402
from witwin.radar.synthesis.fmcw import synthesize_fmcw_rows  # noqa: E402

pytestmark = pytest.mark.gpu


C0 = SPEED_OF_LIGHT_M_PER_S
FC_HZ = 77.0e9
SLOPE_HZ_PER_S = 6.0e13
SAMPLE_RATE_HZ = 5.0e6
NUM_SAMPLES = 256
T_START_S = 6.0e-6
CHIRP_PERIOD_S = 60.0e-6
RANGE_M = 3.7
RADIAL_SPEED_MPS = 12.0

TAU_RT_S = 2.0 * RANGE_M / C0
TAU_RATE = 2.0 * RADIAL_SPEED_MPS / C0


def _spec(**overrides) -> FmcwSpec:
    """The production carrier placement unless a test says otherwise."""

    fields = {
        "num_samples": NUM_SAMPLES,
        "num_chirps": 1,
        "sample_period_s": 1.0 / SAMPLE_RATE_HZ,
        "chirp_period_s": CHIRP_PERIOD_S,
        "slope_hz_per_s": SLOPE_HZ_PER_S,
        "t_start_s": T_START_S,
        "reference_frequency_hz": FC_HZ,
        "carrier_hz": 0.0,
        "carrier_rate_hz": FC_HZ,
        "output_domain": "beat",
    }
    fields.update(overrides)
    return FmcwSpec(**fields)


def _frozen_channel_weight(amplitude: complex = 1.0 + 0.0j) -> complex:
    """A Channel-sourced beat weight at this geometry.

    The Channel coefficient carries ``exp(-j 2 pi fc tau_rt)`` at the FROZEN
    per-frame delay; the beat weight is its conjugate. It is constant across
    chirps, which is exactly why ``carrier_rate_hz`` has to exist.
    """

    phase = 2.0 * math.pi * FC_HZ * TAU_RT_S
    return amplitude * complex(math.cos(phase), math.sin(phase))


def _one_row(delay_s: float, weight: complex, rate: float = 0.0):
    tau = torch.tensor([delay_s], dtype=torch.float32, device="cuda")
    tau_rate = torch.tensor([rate], dtype=torch.float32, device="cuda")
    w = torch.tensor([weight], dtype=torch.complex64, device="cuda")
    offsets = torch.tensor([0, 1], dtype=torch.int64, device="cuda")
    return tau, tau_rate, w, offsets


def _stored(delay_s: float) -> float:
    """The float32 value the kernel actually sees.

    ``total_delay_s`` is float32 by contract, so rounding it costs up to half
    an ulp, about 6e-8 relative. That is a property of the CONTRACT, not of the
    kernel, and separating the two is what lets the kernel be asserted an order
    of magnitude tighter than the input can be represented.
    """

    return float(torch.tensor([delay_s], dtype=torch.float32))


def _fast_time(spec: FmcwSpec, delay_s: float, weight: complex, rate=0.0):
    tau, tau_rate, w, offsets = _one_row(delay_s, weight, rate)
    cube = synthesize_fmcw_rows(tau, tau_rate, w, offsets, spec)
    return cube[0, 0].cpu().to(torch.complex128)


def _phase_slope_beat_frequency(samples: torch.Tensor) -> float:
    """``f_beat`` from the sample-to-sample phase step, exactly.

    With ``tau_rate = 0`` the only sample-dependent term in the phase is
    ``2 pi S tau t_m``, so consecutive samples differ by exactly
    ``2 pi S tau T_s``. This estimator has no window bias and no bin
    quantisation; it is the one that carries the tight assertion.
    """

    steps = samples[1:] * torch.conj(samples[:-1])
    return float(torch.angle(steps).mean()) / (2.0 * math.pi) * SAMPLE_RATE_HZ


def _parabolic_peak_hz(samples: torch.Tensor, pad: int) -> float:
    """``f_beat`` from a parabolic fit to the zero-padded FFT magnitude peak."""

    n_fft = samples.numel() * pad
    magnitude = torch.fft.fft(samples, n=n_fft).abs()
    peak = int(magnitude.argmax())
    left = float(magnitude[peak - 1])
    centre = float(magnitude[peak])
    right = float(magnitude[peak + 1])
    offset = 0.5 * (left - right) / (left - 2.0 * centre + right)
    return (peak + offset) * SAMPLE_RATE_HZ / n_fft


# --------------------------------------------------------------------------
# T1.1  beat frequency and range
# --------------------------------------------------------------------------


def test_the_beat_tone_reports_the_round_trip_delay_in_seconds():
    """``f_beat = S tau_rt`` with NO factor of two, and ``R = c0 tau_rt / 2``.

    Two estimators, because they fail differently. The phase-slope estimator is
    exact for a single unwindowed tone and pins the physics. The zero-padded
    parabolic FFT peak is what a real range processor does, and pinning it too
    is what says the tone the processor finds is the tone the kernel wrote:
    without padding the same fit is off by 2.0e-3 relative, which is 7.5 mm of
    range and would hide a genuine error of the same size.

    The assertion is on ``tau`` in SECONDS. Asserting a bin index would pass
    against a kernel that doubled the delay and halved the slope.
    """

    spec = _spec()
    samples = _fast_time(spec, TAU_RT_S, 1.0 + 0.0j)

    f_beat = SLOPE_HZ_PER_S * TAU_RT_S
    assert f_beat == pytest.approx(1481024.582679795, rel=1e-12)
    assert spec.beat_frequency_hz(TAU_RT_S) == pytest.approx(f_beat, rel=1e-12)
    assert spec.beat_bin(TAU_RT_S) == pytest.approx(f_beat * NUM_SAMPLES / SAMPLE_RATE_HZ, rel=1e-12)

    exact = _phase_slope_beat_frequency(samples)
    # Against the delay the kernel was HANDED: this is the kernel's own
    # exactness, with the float32 contract's rounding taken out of the way.
    assert exact == pytest.approx(SLOPE_HZ_PER_S * _stored(TAU_RT_S), rel=1e-8)
    # And against the geometry, where float32 storage of tau costs up to 6e-8.
    assert exact / SLOPE_HZ_PER_S == pytest.approx(TAU_RT_S, rel=1e-7)

    padded = _parabolic_peak_hz(samples, pad=16)
    assert padded / SLOPE_HZ_PER_S == pytest.approx(TAU_RT_S, rel=1e-6)
    assert C0 * (padded / SLOPE_HZ_PER_S) / 2.0 == pytest.approx(RANGE_M, rel=1e-6)

    # Non-vacuity: the same fit without padding is three orders of magnitude
    # worse, so the padded assertion is measuring the kernel and not the fit.
    raw = _parabolic_peak_hz(samples, pad=1)
    assert abs(raw - f_beat) / f_beat > 1e-4

    # The resolution the tolerance sits inside, stated rather than assumed.
    assert C0 * SAMPLE_RATE_HZ / (2.0 * SLOPE_HZ_PER_S * NUM_SAMPLES) == pytest.approx(0.048794345377604166, rel=1e-12)
    assert C0 * SAMPLE_RATE_HZ / (2.0 * SLOPE_HZ_PER_S) == pytest.approx(12.491352416666667, rel=1e-12)


def test_the_delay_is_round_trip_and_is_never_doubled():
    """Doubling ``tau`` doubles ``f_beat``; the kernel must not do it for you.

    ``dirichlet.cu`` takes a ONE-WAY distance and forms ``tau = 2 d / c0``
    internally. A synthesis kernel that repeated that on a round-trip delay
    would report exactly twice the range, self-consistently, at every geometry.

    The comparison runs at 2 m rather than at the file's 3.7 m so that the
    DOUBLED tone still fits under the Nyquist frequency: at 3.7 m the doubled
    beat tone is 2.96 MHz against a 2.5 MHz limit and the estimator would be
    measuring the alias rather than the kernel.
    """

    spec = _spec()
    short_delay = 2.0 * 2.0 / C0
    assert 2.0 * SLOPE_HZ_PER_S * short_delay < 0.5 * SAMPLE_RATE_HZ

    single = _phase_slope_beat_frequency(_fast_time(spec, short_delay, 1.0 + 0.0j))
    doubled = _phase_slope_beat_frequency(_fast_time(spec, 2.0 * short_delay, 1.0 + 0.0j))
    assert single == pytest.approx(SLOPE_HZ_PER_S * _stored(short_delay), rel=1e-8)
    assert doubled == pytest.approx(SLOPE_HZ_PER_S * _stored(2.0 * short_delay), rel=1e-8)
    assert doubled == pytest.approx(2.0 * single, rel=1e-6)


# --------------------------------------------------------------------------
# T1.2  on-grid peak magnitude
# --------------------------------------------------------------------------


def test_an_on_grid_beat_tone_peaks_at_exactly_n_times_the_weight():
    """``|peak| = N |W|`` when the tone sits on an FFT bin.

    Off grid the exact value is the Dirichlet kernel, which is what the legacy
    spectrum path computes in closed form; choosing an integer bin removes that
    term and leaves a statement purely about the weight. The weight is complex
    and its magnitude is not one, so a kernel that dropped the imaginary part
    or normalised the sum would fail here.
    """

    spec = _spec()
    weight = 0.25 - 0.5j
    for target_bin in (76, 100):
        delay = target_bin * SAMPLE_RATE_HZ / (SLOPE_HZ_PER_S * NUM_SAMPLES)
        assert spec.beat_bin(delay) == pytest.approx(float(target_bin), rel=1e-12)

        magnitude = torch.fft.fft(_fast_time(spec, delay, weight)).abs()
        assert int(magnitude.argmax()) == target_bin
        assert float(magnitude[target_bin]) == pytest.approx(NUM_SAMPLES * abs(weight), rel=1e-5)


# --------------------------------------------------------------------------
# T1.3 / T1.4  slow-time Doppler
# --------------------------------------------------------------------------


def _slow_time_slope(spec: FmcwSpec, weight: complex, sample: int) -> float:
    tau, rate, w, offsets = _one_row(TAU_RT_S, weight, TAU_RATE)
    cube = synthesize_fmcw_rows(tau, rate, w, offsets, spec).cpu()
    slow = cube[:, 0, sample].to(torch.complex128)
    steps = slow[1:] * torch.conj(slow[:-1])
    return float(torch.angle(steps).mean())


def _analytic_slope(sample: int) -> float:
    """``d(phase)/d(chirp) = 2 pi tau_rate (fc + S (t0 - tau + t_m)) Tc``.

    The bracket is the exact derivative of the whole phase with respect to the
    delay: the ramp contributes ``S (t0 - tau)`` because ``d/dtau`` of
    ``S tau (t0 - tau/2)`` is ``S (t0 - tau)``, and ``S t_m`` because the beat
    tone itself moves with the delay.
    """

    t_m = sample / SAMPLE_RATE_HZ
    ramp = SLOPE_HZ_PER_S * (T_START_S - TAU_RT_S + t_m)
    return 2.0 * math.pi * TAU_RATE * CHIRP_PERIOD_S * (FC_HZ + ramp)


def _analytic_ramp_only_slope(sample: int) -> float:
    t_m = sample / SAMPLE_RATE_HZ
    ramp = SLOPE_HZ_PER_S * (T_START_S - TAU_RT_S + t_m)
    return 2.0 * math.pi * TAU_RATE * CHIRP_PERIOD_S * ramp


@pytest.mark.parametrize("sample", [0, NUM_SAMPLES - 1])
def test_the_slow_time_slope_carries_the_whole_carrier_not_just_the_ramp(sample):
    """The two-sided guard against a silent 215x Doppler understatement.

    One side: the measured slope equals the analytic one. The other side: the
    analytic one is more than a hundred times the value a kernel that ignored
    ``carrier_rate_hz`` would produce. A one-sided test passes when the term is
    dropped, because the ramp alone still produces a plausible Doppler cube -
    that is exactly how the bug survived until it was measured.
    """

    spec = _spec(num_chirps=16)
    analytic = _analytic_slope(sample)
    measured = _slow_time_slope(spec, _frozen_channel_weight(), sample)
    assert measured == pytest.approx(analytic, rel=1e-5)

    ramp_only = _analytic_ramp_only_slope(sample)
    assert abs(analytic / ramp_only) > 100.0 * 1e-5 * 1e4
    expected_factor = {0: 215.7725, NUM_SAMPLES - 1: 23.5244}[sample]
    assert analytic / ramp_only == pytest.approx(expected_factor, rel=1e-4)


@pytest.mark.parametrize("sample", [0, NUM_SAMPLES - 1])
def test_both_carrier_homes_produce_the_same_slow_time_slope(sample):
    """``(fc, 0)`` and ``(0, fc)`` differ only by a constant, never by a slope.

    A kernel-owned carrier multiplies the FULL ``tau(t)`` and therefore already
    walks; a weight-owned one is frozen at ``tau_rt`` and needs the rate term to
    walk at all. The two are exactly equivalent in slow time and differ by the
    constant ``fc tau_rt``, which the weight holds on the production path. That
    equivalence is what lets the legacy real-amplitude path and the Channel path
    be the same physics, so it is asserted rather than assumed.
    """

    production = _spec(num_chirps=16)
    kernel_carrier = _spec(num_chirps=16, carrier_hz=FC_HZ, carrier_rate_hz=0.0)
    assert production.carrier_hz == 0.0
    assert kernel_carrier.carrier_rate_hz == 0.0

    from_weight = _slow_time_slope(production, _frozen_channel_weight(), sample)
    from_kernel = _slow_time_slope(kernel_carrier, 1.0 + 0.0j, sample)
    analytic = _analytic_slope(sample)
    assert from_weight == pytest.approx(analytic, rel=1e-6)
    assert from_kernel == pytest.approx(analytic, rel=1e-6)
    assert from_weight == pytest.approx(from_kernel, rel=1e-6)


def test_naming_the_carrier_in_both_homes_is_refused():
    with pytest.raises(ValueError, match="double counts"):
        _spec(carrier_hz=FC_HZ, carrier_rate_hz=FC_HZ)


# --------------------------------------------------------------------------
# T1.5  Doppler sign
# --------------------------------------------------------------------------


def test_a_receding_site_puts_the_beat_cube_tone_at_positive_doppler():
    """The SIGN, not the magnitude.

    Physical Doppler in Channel's ``exp(-j k d)`` convention is
    ``f_D = -fc tau_rate``, so a receding site (``tau_rate > 0``) is a NEGATIVE
    shift. The beat cube is conjugated once, at
    ``channel_phasor_to_beat_weight``, so its slow-time tone sits at
    ``+fc tau_rate``. A sign error here is completely invisible in a
    magnitude-only range-Doppler map and shows up much later as a target that
    approaches when it should recede.
    """

    num_chirps = 64
    spec = _spec(num_chirps=num_chirps)
    tau, rate, weight, offsets = _one_row(TAU_RT_S, _frozen_channel_weight(), TAU_RATE)
    cube = synthesize_fmcw_rows(tau, rate, weight, offsets, spec).cpu()
    slow = cube[:, 0, 0].to(torch.complex128)

    spectrum = torch.fft.fftshift(torch.fft.fft(slow)).abs()
    frequencies = torch.fft.fftshift(torch.fft.fftfreq(num_chirps, d=CHIRP_PERIOD_S))
    peak_hz = float(frequencies[int(spectrum.argmax())])
    bin_hz = float(frequencies[1] - frequencies[0])

    analytic_tone = TAU_RATE * (FC_HZ + SLOPE_HZ_PER_S * (T_START_S - TAU_RT_S))
    assert peak_hz > 0.0
    assert abs(peak_hz - analytic_tone) <= 0.5 * bin_hz

    physical_doppler_hz = -FC_HZ * TAU_RATE
    assert physical_doppler_hz == pytest.approx(-6164.264479261849, rel=1e-9)
    assert physical_doppler_hz < 0.0 < peak_hz
    assert C0 * TAU_RATE / 2.0 == pytest.approx(RADIAL_SPEED_MPS, rel=1e-12)

    # And the tone is inside the unambiguous band, so its sign is meaningful.
    assert RADIAL_SPEED_MPS < spec.max_unambiguous_speed_mps


# --------------------------------------------------------------------------
# T1.7  unambiguous velocity
# --------------------------------------------------------------------------


def test_the_unambiguous_speed_bound_is_owned_by_the_fmcw_spec():
    """``lambda / (4 Tc num_tx)``, derived in one place and only one place.

    The spec is the only owner of this bound; both fixture arrays pin its formula.
    """

    from support import multi_endpoint_geometry as multi
    from support import phase4_geometry as single

    from witwin.radar import RadarConfig

    assert _spec().max_unambiguous_speed_mps == pytest.approx(16.222535606060607, rel=1e-12)

    for geometry in (single, multi):
        config = RadarConfig.from_dict(dict(geometry.FIXTURE_RADAR_CONFIG))
        spec = FmcwSpec.from_radar_config(config)
        assert spec.num_tx == config.num_tx
        assert spec.num_rx == config.num_rx
        assert spec.slot_period_s == pytest.approx(spec.chirp_period_s * config.num_tx, rel=1e-12)
