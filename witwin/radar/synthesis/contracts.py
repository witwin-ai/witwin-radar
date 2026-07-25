"""Waveform description for FMCW beat synthesis.

This module is pure and CPU-testable on purpose: the unit conversions between
the radar config's engineering units and SI are exactly the kind of thing that
is wrong once and then wrong everywhere, and they should not require a GPU to
check.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FmcwBeatSpec:
    """One chirp frame's sampling grid and ramp, in SI units.

    ``carrier_hz`` selects where the carrier phase ``2 * pi * f_c * tau`` is
    applied, and both settings are exact:

    * ``carrier_hz = fc`` reproduces the Dirichlet solver's phase structure
      exactly, which is what the equivalence test uses.
    * ``carrier_hz = 0`` is the production path for Channel-sourced weights,
      where the carrier already sits inside the natively computed coefficient.
      That is the more accurate placement here, because the coefficient's
      phase was formed against a float64 delay inside the native kernel, while
      a float32 ``tau`` re-multiplied by 77 GHz loses roughly 2e-4 rad at 2 m
      and 1e-2 rad at 100 m.

    Neither setting is a fallback for the other.
    """

    num_samples: int
    num_chirps: int
    sample_period_s: float
    chirp_period_s: float
    slope_hz_per_s: float
    t_start_s: float
    carrier_hz: float = 0.0

    def __post_init__(self) -> None:
        if self.num_samples < 1:
            raise ValueError("num_samples must be positive")
        if self.num_chirps < 1:
            raise ValueError("num_chirps must be positive")
        if self.sample_period_s <= 0.0:
            raise ValueError("sample_period_s must be positive")
        if self.chirp_period_s <= 0.0:
            raise ValueError("chirp_period_s must be positive")

    @classmethod
    def from_radar_config(cls, config, *, carrier_hz: float = 0.0) -> "FmcwBeatSpec":
        """Convert a :class:`witwin.radar.RadarConfig` into SI units.

        The config carries engineering units: ``sample_rate`` in kSPS,
        ``idle_time`` / ``ramp_end_time`` / ``adc_start_time`` in microseconds,
        and ``slope`` in MHz per microsecond, which is 1e12 Hz per second.
        """

        return cls(
            num_samples=int(config.adc_samples),
            num_chirps=int(config.chirp_per_frame),
            sample_period_s=1.0 / (float(config.sample_rate) * 1e3),
            chirp_period_s=(float(config.idle_time) + float(config.ramp_end_time))
            * 1e-6,
            slope_hz_per_s=float(config.slope) * 1e12,
            t_start_s=float(config.adc_start_time) * 1e-6,
            carrier_hz=float(carrier_hz),
        )

    @property
    def sample_rate_hz(self) -> float:
        return 1.0 / self.sample_period_s

    def beat_frequency_hz(self, round_trip_delay_s: float) -> float:
        """``f_beat = slope * tau``, with ``tau`` the ROUND-TRIP delay.

        There is no factor of two here. A two-leg round trip already knows its
        own total delay; doubling it would be a monostatic assumption that this
        contract does not make.
        """

        return self.slope_hz_per_s * float(round_trip_delay_s)

    def beat_bin(self, round_trip_delay_s: float) -> float:
        """Fractional FFT bin of the beat tone over ``num_samples``."""

        return (
            self.beat_frequency_hz(round_trip_delay_s)
            * self.num_samples
            / self.sample_rate_hz
        )


__all__ = ["FmcwBeatSpec"]
