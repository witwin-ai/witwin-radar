"""Analytic fixtures for the native FMCW beat synthesis primitive.

Everything here is checked against a closed form computed in float64, or
against the existing Dirichlet path, never against a previous run of the code
under test.
"""

from __future__ import annotations

import math

import pytest
import torch

from support import phase4_geometry as geo  # noqa: E402
from support import reference_chain as ref  # noqa: E402
from witwin.radar.synthesis.contracts import FmcwBeatSpec  # noqa: E402
from witwin.radar.synthesis.fmcw_beat import (  # noqa: E402
    channel_phasor_to_beat_weight,
    synthesize_beat_rows,
)


pytestmark = pytest.mark.gpu


def _spec(**overrides) -> FmcwBeatSpec:
    """Fixture spec.

    ``carrier_hz`` goes through ``from_radar_config`` rather than ``replace``
    because the two carrier parameters are a pair: the factory derives
    ``carrier_rate_hz`` from the carrier placement, and replacing only
    ``carrier_hz`` on a production spec is the both-nonzero double count the
    contract refuses.
    """

    from witwin.radar import RadarConfig

    config = RadarConfig.from_dict(dict(geo.FIXTURE_RADAR_CONFIG))
    carrier_hz = overrides.pop("carrier_hz", 0.0)
    spec = FmcwBeatSpec.from_radar_config(config, carrier_hz=carrier_hz)
    if overrides:
        from dataclasses import replace

        spec = replace(spec, **overrides)
    return spec


def _rows(delays, weights, *, rates=None, segments=1):
    device = "cuda"
    tau = torch.tensor(delays, dtype=torch.float32, device=device)
    rate = torch.zeros_like(tau) if rates is None else torch.tensor(
        rates, dtype=torch.float32, device=device
    )
    weight = torch.tensor(weights, dtype=torch.complex64, device=device)
    count = tau.numel()
    per = count // segments
    offsets = torch.tensor(
        [per * i for i in range(segments)] + [count], dtype=torch.int64, device=device
    )
    return tau, rate, weight, offsets


def test_primal_matches_the_float64_oracle():
    spec = _spec(num_chirps=3)
    tau, rate, weight, offsets = _rows(
        [geo.round_trip_delay_s(), 2.1e-8],
        [0.5 + 0.25j, -0.125 + 0.75j],
        rates=[1.0e-9, -4.0e-10],
    )
    measured = synthesize_beat_rows(tau, rate, weight, offsets, spec)
    expected = ref.beat_samples(
        tau.cpu(), rate.cpu(), weight.cpu(), offsets.cpu(), spec
    )
    torch.testing.assert_close(
        measured.cpu().to(torch.complex128), expected, rtol=2e-5, atol=2e-5
    )


def test_beat_frequency_and_peak_bin_match_the_closed_form():
    spec = _spec(num_chirps=1)
    tau_rt = geo.round_trip_delay_s()
    tau, rate, weight, offsets = _rows([tau_rt], [1.0 + 0.0j])
    iq = synthesize_beat_rows(tau, rate, weight, offsets, spec)

    spectrum = torch.fft.fft(iq[0, 0].cpu().to(torch.complex128))
    peak = int(spectrum.abs().argmax())
    expected_bin = spec.beat_bin(tau_rt)
    assert abs(peak - expected_bin) <= 1.0, (peak, expected_bin)
    assert abs(spec.beat_frequency_hz(tau_rt) - spec.slope_hz_per_s * tau_rt) < 1e-6


def test_tau_is_the_round_trip_delay_and_is_never_doubled():
    """The exact-2x-wrong-range regression.

    Feeding the round-trip delay must place the tone at ``slope * tau``. If the
    kernel ever reintroduced the Dirichlet path's ``2 * d / c0`` doubling, the
    result would be internally consistent and plausible, and every range would
    be twice what it should be.
    """

    spec = _spec(num_chirps=1)
    tau_rt = geo.round_trip_delay_s()
    single, rate, weight, offsets = _rows([tau_rt], [1.0 + 0.0j])
    doubled, rate2, weight2, offsets2 = _rows([2.0 * tau_rt], [1.0 + 0.0j])

    peak_single = int(
        torch.fft.fft(
            synthesize_beat_rows(single, rate, weight, offsets, spec)[0, 0].cpu()
        )
        .abs()
        .argmax()
    )
    peak_doubled = int(
        torch.fft.fft(
            synthesize_beat_rows(doubled, rate2, weight2, offsets2, spec)[0, 0].cpu()
        )
        .abs()
        .argmax()
    )
    assert abs(peak_single - spec.beat_bin(tau_rt)) <= 1.0
    assert abs(peak_doubled - 2 * peak_single) <= 1.0


def test_matches_the_dirichlet_path_when_carrier_is_the_carrier():
    """``carrier_hz = fc`` reproduces the existing solver's phase structure.

    The Dirichlet solver works in the frequency domain over a padded spectrum,
    so the comparison is made in the domain both can reach: the DFT of this
    kernel's time samples against the Dirichlet spectrum at the matching bin
    scale. What is being pinned is the phase convention and the ``t_start`` /
    RVP terms, which are shared.
    """

    from witwin.radar import Radar, RadarConfig

    config = RadarConfig.from_dict(dict(geo.FIXTURE_RADAR_CONFIG))
    radar = Radar(config, device="cuda")
    solver = radar.solver

    distance_m = 3.0
    tau_rt = 2.0 * distance_m / geo.C0_M_PER_S
    spec = _spec(num_chirps=1, carrier_hz=config.fc)

    tau, rate, weight, offsets = _rows([tau_rt], [1.0 + 0.0j])
    samples = synthesize_beat_rows(tau, rate, weight, offsets, spec)[0, 0]
    measured = torch.fft.fft(samples.cpu().to(torch.complex128))

    dirichlet = solver.chirp_mimo(
        torch.tensor([distance_m], dtype=torch.float32, device="cuda"),
        torch.tensor([1.0], dtype=torch.float32, device="cuda"),
    ).cpu()

    peak_measured = int(measured.abs().argmax())
    peak_dirichlet = int(dirichlet.abs().argmax())
    assert peak_measured == peak_dirichlet

    a = complex(measured[peak_measured])
    b = complex(dirichlet[peak_dirichlet])
    # Same peak bin, same peak phase: the shared convention is what is pinned.
    assert abs(abs(a) - abs(b)) <= 2e-3 * abs(b)
    phase_gap = math.remainder(
        math.atan2(a.imag, a.real) - math.atan2(b.imag, b.real), 2.0 * math.pi
    )
    assert abs(phase_gap) < 2e-2, phase_gap


def test_multi_chirp_slow_time_phase_slope_carries_doppler():
    """Slow-time phase advances by ``2 pi f_c * tau_rate * T_chirp`` per chirp.

    This is pre-existing package behaviour: the distance advances inside the
    same ``+j phi`` the primal uses, so the Doppler-FFT bin sits at
    ``+f_c * d(tau)/dt``, which is the negative of the physical Doppler shift.
    It is asserted, not flipped.
    """

    carrier = geo.REFERENCE_FREQUENCY_HZ
    spec = _spec(num_chirps=16, carrier_hz=carrier)
    tau_rt = geo.round_trip_delay_s()
    rate_value = 2.385e-8
    tau, rate, weight, offsets = _rows(
        [tau_rt], [1.0 + 0.0j], rates=[rate_value]
    )
    iq = synthesize_beat_rows(tau, rate, weight, offsets, spec).cpu()

    # Take one fast-time sample and look at how its phase walks across chirps.
    slow_time = iq[:, 0, 0].to(torch.complex128)
    steps = slow_time[1:] * torch.conj(slow_time[:-1])
    measured = float(torch.angle(steps).mean())

    # The full d(phi)/d(tau) at sample zero, not just the carrier term: the
    # ramp contributes slope * (t_start - tau), which is 0.47% of the carrier
    # here and is far larger than the tolerance. Keeping the carrier-only
    # approximation would have made this test fail against a correct kernel.
    dphi_dtau = 2.0 * math.pi * (
        carrier + spec.slope_hz_per_s * (spec.t_start_s - tau_rt)
    )
    expected = dphi_dtau * rate_value * spec.chirp_period_s
    assert abs(math.remainder(measured - expected, 2.0 * math.pi)) < 1e-4
    assert expected > 0.0  # positive slow-time slope for a receding site

    carrier_only = 2.0 * math.pi * carrier * rate_value * spec.chirp_period_s
    assert abs(measured - carrier_only) > 1e-3  # the ramp terms are really there


def test_production_carrier_placement_carries_the_same_doppler():
    """The two carrier homes must agree on the slow-time phase walk.

    This is the regression for a silent 215x Doppler understatement. On the
    production path the Channel weight holds ``exp(+j 2 pi fc tau_rt)`` at the
    FROZEN per-frame delay, so that phase does not advance across chirps. With
    ``carrier_rate_hz = 0`` the kernel would then keep only the ramp's
    contribution ``slope * (t_start - tau + t_m) * tau_rate``, which is 1/215 of
    the true slope at sample 0 and 1/21 at the last sample - a plausible,
    silently wrong Doppler cube. ``carrier_rate_hz = fc`` restores exactly the
    missing ``fc * tau_rate * t_c``.

    Both placements are compared against the SAME analytic slope, so neither one
    is the other's oracle.
    """

    carrier = geo.REFERENCE_FREQUENCY_HZ
    tau_rt = geo.round_trip_delay_s()
    rate_value = 2.385e-8

    kernel_carrier = _spec(num_chirps=16, carrier_hz=carrier)
    production = _spec(num_chirps=16)
    assert kernel_carrier.carrier_rate_hz == 0.0
    assert production.carrier_hz == 0.0
    assert production.carrier_rate_hz == pytest.approx(carrier)

    def slow_time_slope(spec, weight_value):
        tau, rate, weight, offsets = _rows(
            [tau_rt], [weight_value], rates=[rate_value]
        )
        iq = synthesize_beat_rows(tau, rate, weight, offsets, spec).cpu()
        slow = iq[:, 0, 0].to(torch.complex128)
        steps = slow[1:] * torch.conj(slow[:-1])
        return float(torch.angle(steps).mean())

    # A Channel-sourced beat weight: the conjugated exp(-j 2 pi fc tau_rt),
    # frozen at the per-frame delay and therefore constant across chirps.
    frozen_carrier_phase = 2.0 * math.pi * carrier * tau_rt
    weight = complex(math.cos(frozen_carrier_phase), math.sin(frozen_carrier_phase))

    expected = (
        2.0
        * math.pi
        * (carrier + production.slope_hz_per_s * (production.t_start_s - tau_rt))
        * rate_value
        * production.chirp_period_s
    )
    measured_kernel = slow_time_slope(kernel_carrier, 1.0 + 0.0j)
    measured_production = slow_time_slope(production, weight)

    assert abs(math.remainder(measured_kernel - expected, 2.0 * math.pi)) < 1e-4
    assert abs(math.remainder(measured_production - expected, 2.0 * math.pi)) < 1e-4

    # And the term really is dominant: dropping it loses more than 99% of the
    # slope. Without this assertion the test would still pass against a kernel
    # that ignored carrier_rate_hz if the tolerance were ever loosened.
    ramp_only = (
        2.0
        * math.pi
        * production.slope_hz_per_s
        * (production.t_start_s - tau_rt)
        * rate_value
        * production.chirp_period_s
    )
    assert abs(expected / ramp_only) > 200.0


def test_the_two_carrier_homes_cannot_both_be_used():
    """Both nonzero double counts the carrier, so the contract refuses it."""

    with pytest.raises(ValueError, match="double counts"):
        FmcwBeatSpec(
            num_samples=8,
            num_chirps=2,
            sample_period_s=1.0 / 4.4e6,
            chirp_period_s=65.0e-6,
            slope_hz_per_s=60.012e12,
            t_start_s=0.0,
            reference_frequency_hz=geo.REFERENCE_FREQUENCY_HZ,
            carrier_hz=geo.REFERENCE_FREQUENCY_HZ,
            carrier_rate_hz=geo.REFERENCE_FREQUENCY_HZ,
        )


def test_segments_are_independent_and_offsets_partition_the_rows():
    spec = _spec(num_chirps=2)
    tau, rate, weight, offsets = _rows(
        [1.0e-8, 2.0e-8, 3.0e-8, 4.0e-8],
        [1.0 + 0.0j, 0.5 - 0.5j, -0.25 + 0.0j, 0.75 + 0.25j],
        segments=2,
    )
    both = synthesize_beat_rows(tau, rate, weight, offsets, spec)
    assert tuple(both.shape) == (2, 2, spec.num_samples)

    first = synthesize_beat_rows(
        tau[:2].contiguous(),
        rate[:2].contiguous(),
        weight[:2].contiguous(),
        torch.tensor([0, 2], dtype=torch.int64, device="cuda"),
        spec,
    )
    torch.testing.assert_close(both[:, 0:1, :], first)


def test_the_conjugation_sign_is_anchored_to_a_hand_computed_sample():
    """An ABSOLUTE anchor for the Channel-to-beat conjugation.

    The existing conjugation coverage is pairwise: production and the oracle
    each make the same ``conj`` choice independently, so a coordinated inversion
    of both would agree with itself. The Dirichlet equivalence test uses a real
    weight, where conjugation is invisible.

    Here the expected value is written out by hand from the two published
    conventions, with a deliberately complex weight and a delay chosen so the
    beat phase is a known fraction of a cycle. Nothing in the package is
    consulted for the expected number.
    """

    # One row, one chirp, zero rate, no ramp contribution at sample 0 because
    # t_start is zero: the whole phase is the carrier term, 0.25 of a cycle.
    carrier = 1.0e9
    tau_rt = 0.25 / carrier
    spec = FmcwBeatSpec(
        num_samples=1,
        num_chirps=1,
        sample_period_s=1.0 / 4.4e6,
        chirp_period_s=65.0e-6,
        slope_hz_per_s=0.0,
        t_start_s=0.0,
        reference_frequency_hz=carrier,
        carrier_hz=carrier,
    )

    # A Channel-convention coefficient with a distinctly signed imaginary part.
    channel = torch.tensor([0.0 + 1.0j], dtype=torch.complex64, device="cuda")
    beat = channel_phasor_to_beat_weight(channel)
    tau = torch.tensor([tau_rt], dtype=torch.float32, device="cuda")
    rate = torch.zeros_like(tau)
    offsets = torch.tensor([0, 1], dtype=torch.int64, device="cuda")
    measured = complex(
        synthesize_beat_rows(tau, rate, beat, offsets, spec)[0, 0, 0].cpu()
    )

    # By hand: conj(0 + 1j) = -1j, and exp(+j 2 pi * 0.25) = +1j.
    # (-1j) * (+1j) = +1. An inverted conjugation would give -1 instead.
    expected = complex(1.0, 0.0)
    assert abs(measured - expected) < 1e-5, (measured, expected)
    # State the failure mode explicitly so the anchor cannot be read as a
    # magnitude check: the inverted convention has the same magnitude.
    assert abs(measured - (-expected)) > 1.0


def test_conjugation_is_the_only_channel_to_beat_conversion():
    coefficient = torch.tensor([0.25 - 0.5j], dtype=torch.complex64, device="cuda")
    beat = channel_phasor_to_beat_weight(coefficient)
    torch.testing.assert_close(beat, torch.conj(coefficient).resolve_conj())
    assert not beat.is_conj()
    with pytest.raises(TypeError, match="must be complex"):
        channel_phasor_to_beat_weight(torch.ones(1, device="cuda"))


def test_loading_the_extension_is_free_and_side_effect_free_after_the_first_call():
    """Regression: the loader must be called once per process, not per launch.

    On Windows the loader prepends the MSVC tool directories to PATH every time
    it runs. Calling it per synthesis launch grew PATH until Windows rejected it
    with "the environment variable is longer than 32767 characters", which
    surfaced as 25 unrelated CUDA tests failing partway through a full session.
    """

    import os

    from witwin.radar.cuda import build
    from witwin.radar.synthesis import fmcw_beat

    first = build.build_extension()
    path_length = len(os.environ.get("PATH", ""))
    for _ in range(8):
        assert build.build_extension() is first
        assert fmcw_beat._ops() is first
    assert len(os.environ.get("PATH", "")) == path_length


def test_zero_weight_row_contributes_exactly_zero():
    spec = _spec(num_chirps=2)
    tau, rate, weight, offsets = _rows(
        [1.0e-8, 2.0e-8], [1.0 + 0.0j, 0.0 + 0.0j]
    )
    with_dead = synthesize_beat_rows(tau, rate, weight, offsets, spec)
    alive = synthesize_beat_rows(
        tau[:1].contiguous(),
        rate[:1].contiguous(),
        weight[:1].contiguous(),
        torch.tensor([0, 1], dtype=torch.int64, device="cuda"),
        spec,
    )
    torch.testing.assert_close(with_dead, alive)
