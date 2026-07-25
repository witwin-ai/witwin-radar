"""CPU unit conversions for the FMCW beat waveform description.

The radar config carries engineering units and the kernel wants SI. That
conversion is exactly the kind of thing that is wrong once and then wrong
everywhere, so it lives in a pure function and is checked without a GPU.
"""

from __future__ import annotations

import pytest

from support import phase4_geometry as geo
from witwin.radar import RadarConfig
from witwin.radar.synthesis import FmcwBeatSpec


@pytest.fixture
def spec() -> FmcwBeatSpec:
    config = RadarConfig.from_dict(dict(geo.FIXTURE_RADAR_CONFIG))
    return FmcwBeatSpec.from_radar_config(config)


def test_units_are_converted_to_si(spec):
    # sample_rate is kSPS, so 4400 -> 4.4 MHz.
    assert spec.sample_rate_hz == pytest.approx(4.4e6)
    assert spec.sample_period_s == pytest.approx(1.0 / 4.4e6)
    # idle_time and ramp_end_time are microseconds.
    assert spec.chirp_period_s == pytest.approx(65.0e-6)
    # slope is MHz per microsecond, i.e. 1e12 Hz per second.
    assert spec.slope_hz_per_s == pytest.approx(60.012e12)
    # adc_start_time is microseconds.
    assert spec.t_start_s == pytest.approx(6.0e-6)
    assert spec.num_samples == 256
    assert spec.num_chirps == 8


def test_carrier_placement_defaults_to_the_weight(spec):
    # The Channel coefficient already carries exp(-j 2 pi f tau), so the
    # default asks the kernel not to apply the ABSOLUTE carrier a second time.
    assert spec.carrier_hz == 0.0
    # ... but that weight is frozen at the per-frame tau_rt and cannot express
    # intra-frame Doppler, so the factory pairs it with the rate-only carrier.
    # A default of zero here would silently understate Doppler by up to 215x.
    assert spec.carrier_rate_hz == pytest.approx(geo.REFERENCE_FREQUENCY_HZ)

    config = RadarConfig.from_dict(dict(geo.FIXTURE_RADAR_CONFIG))
    explicit = FmcwBeatSpec.from_radar_config(config, carrier_hz=config.fc)
    assert explicit.carrier_hz == pytest.approx(geo.REFERENCE_FREQUENCY_HZ)
    # The kernel owns the whole carrier here, so the rate term must NOT be
    # applied as well; that would double count it.
    assert explicit.carrier_rate_hz == 0.0


def test_the_two_carrier_homes_are_mutually_exclusive(spec):
    from dataclasses import replace

    # Overriding only carrier_hz on a production spec is the double count, and
    # it is refused rather than silently keeping both terms.
    with pytest.raises(ValueError, match="double counts"):
        replace(spec, carrier_hz=geo.REFERENCE_FREQUENCY_HZ)
    # Stating both halves of the pair is how a caller switches placement.
    switched = replace(
        spec, carrier_hz=geo.REFERENCE_FREQUENCY_HZ, carrier_rate_hz=0.0
    )
    assert switched.carrier_hz == pytest.approx(geo.REFERENCE_FREQUENCY_HZ)


def test_beat_frequency_has_no_factor_of_two(spec):
    tau_rt = geo.round_trip_delay_s()
    assert spec.beat_frequency_hz(tau_rt) == pytest.approx(60.012e12 * tau_rt)
    # The fixture round trip lands just under bin 47 of 256.
    assert spec.beat_bin(tau_rt) == pytest.approx(
        spec.beat_frequency_hz(tau_rt) * 256 / 4.4e6
    )
    assert 46.0 < spec.beat_bin(tau_rt) < 47.0


def test_round_trip_delay_is_the_sum_of_two_legs():
    d_in, d_out = geo.leg_distances_m()
    assert d_in == pytest.approx(4.36**0.5)
    assert d_out == pytest.approx(3.7825**0.5)
    assert geo.round_trip_delay_s() == pytest.approx(
        (d_in + d_out) / geo.C0_M_PER_S
    )


def test_spec_rejects_a_degenerate_grid():
    with pytest.raises(ValueError, match="num_samples must be positive"):
        FmcwBeatSpec(0, 1, 1e-6, 1e-4, 1e12, 0.0)
    with pytest.raises(ValueError, match="num_chirps must be positive"):
        FmcwBeatSpec(4, 0, 1e-6, 1e-4, 1e12, 0.0)
    with pytest.raises(ValueError, match="sample_period_s must be positive"):
        FmcwBeatSpec(4, 1, 0.0, 1e-4, 1e12, 0.0)
    with pytest.raises(ValueError, match="chirp_period_s must be positive"):
        FmcwBeatSpec(4, 1, 1e-6, 0.0, 1e12, 0.0)
