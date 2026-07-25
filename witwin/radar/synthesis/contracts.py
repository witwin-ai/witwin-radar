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

    The carrier phase ``2 * pi * f_c * tau`` has two legitimate homes, and the
    two carrier parameters together say which one. Exactly one of them is
    nonzero:

    * ``carrier_hz = fc``, ``carrier_rate_hz = 0``  -  the kernel owns the whole
      carrier phase. This reproduces the Dirichlet solver's phase structure
      exactly, which is what the equivalence test uses.
    * ``carrier_hz = 0``, ``carrier_rate_hz = fc``  -  the production path for
      Channel-sourced weights, where the absolute carrier phase already sits
      inside the natively computed coefficient. That placement is more accurate,
      because the coefficient's phase was formed against a float64 delay inside
      the native kernel, while a float32 ``tau`` re-multiplied by 77 GHz loses
      roughly 2e-4 rad at 2 m and 1e-2 rad at 100 m.

    ``carrier_rate_hz`` is not a second copy of the carrier and not a tuning
    knob. A Channel coefficient is frozen at the per-frame ``tau_rt``, so the
    carrier phase it holds does NOT advance across chirps. Without this term the
    slow-time phase walk keeps only ``slope * (t_start - tau + t_m) * tau_rate``
    and understates intra-frame Doppler by 21x to 215x across the fast-time axis
    - silently, because the primal still looks like a plausible radar cube.
    ``carrier_rate_hz`` applies the carrier to the delay CHANGE
    ``(tau - tau_rt)`` only, which is exactly the missing term.

    Setting both to ``fc`` double counts the carrier and is refused. Both
    supported settings are exact; neither is a fallback for the other.
    """

    num_samples: int
    num_chirps: int
    sample_period_s: float
    chirp_period_s: float
    slope_hz_per_s: float
    t_start_s: float
    carrier_hz: float = 0.0
    carrier_rate_hz: float = 0.0

    def __post_init__(self) -> None:
        if self.num_samples < 1:
            raise ValueError("num_samples must be positive")
        if self.num_chirps < 1:
            raise ValueError("num_chirps must be positive")
        if self.sample_period_s <= 0.0:
            raise ValueError("sample_period_s must be positive")
        if self.chirp_period_s <= 0.0:
            raise ValueError("chirp_period_s must be positive")
        if self.carrier_hz != 0.0 and self.carrier_rate_hz != 0.0:
            raise ValueError(
                "carrier_hz and carrier_rate_hz name the same carrier in two "
                "different homes; setting both double counts it. Use "
                "carrier_hz=fc with carrier_rate_hz=0 when the kernel owns the "
                "carrier phase, or carrier_hz=0 with carrier_rate_hz=fc when a "
                "Channel-sourced weight already carries it."
            )

    @classmethod
    def from_radar_config(cls, config, *, carrier_hz: float = 0.0) -> "FmcwBeatSpec":
        """Convert a :class:`witwin.radar.RadarConfig` into SI units.

        The config carries engineering units: ``sample_rate`` in kSPS,
        ``idle_time`` / ``ramp_end_time`` / ``adc_start_time`` in microseconds,
        and ``slope`` in MHz per microsecond, which is 1e12 Hz per second.

        ``carrier_rate_hz`` is derived, not passed: it is ``config.fc`` on the
        production path (``carrier_hz = 0``, weight owns the carrier) and zero
        when the caller puts the carrier in the kernel. Deriving it here is what
        makes the default configuration Doppler-correct; a caller that overrides
        ``carrier_hz`` through ``dataclasses.replace`` will hit the both-nonzero
        error rather than silently losing the rate term.
        """

        carrier = float(carrier_hz)
        return cls(
            num_samples=int(config.adc_samples),
            num_chirps=int(config.chirp_per_frame),
            sample_period_s=1.0 / (float(config.sample_rate) * 1e3),
            chirp_period_s=(float(config.idle_time) + float(config.ramp_end_time))
            * 1e-6,
            slope_hz_per_s=float(config.slope) * 1e12,
            t_start_s=float(config.adc_start_time) * 1e-6,
            carrier_hz=carrier,
            carrier_rate_hz=0.0 if carrier != 0.0 else float(config.fc),
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
