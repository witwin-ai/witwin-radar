"""The radar: its parameters, its pose, and the entry points that use them.

One flat immutable record. The fields are what a datasheet or a configuration
file lists, in SI units, one field each. A sub-record appears only where a
field has variants (:class:`Fmcw` / :class:`Ofdm` / :class:`Pulsed`,
:class:`~witwin.radar.sensors.Pattern`) or where several numbers are coupled
(:class:`~witwin.radar.frontend.Noise`, ``Agc``, ``Adc``). Everything else -
the antenna layout, the receive chain's one-number stages, the pose - is a
field here, because a wrapper whose only job is grouping makes the caller
write two constructors to say one thing.

No field name carries a unit suffix; every field's unit is on its docstring
row. That rule covers this surface only. :mod:`witwin.radar.processing` is
frozen by R-ADR-017 and keeps ``range_m`` and its siblings, and the internal
records (``FmcwSpec.sample_period_s``, ``RadarPathBatch.total_delay_s``) keep
theirs, because those are the equation-owning modules where R-ADR-021 requires
the unit to be in the identifier.

This module owns no propagation and no synthesis physics. It holds the
parameter record, the pose transforms every consumer shares, and the
derivation of the internal block specs those consumers are handed.
``Radar.simulate`` delegates to :mod:`witwin.radar.simulation`.
"""

from __future__ import annotations

import cmath
import dataclasses
import json
import math
import os
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, ClassVar

import torch

from .frontend import Adc, Agc, Compression, FrontendChain, FrontendSpec, Iq, Noise
from .policy import SPEED_OF_LIGHT_M_PER_S, _carries_derivative, refuse_derivative, resolve_device
from .sensors import Coupling, Pattern, SensorArraySpec, watts_from_dbm
from .synthesis.assembly import (
    PULSE_NORMALIZATION_UNIT_AMPLITUDE,
    SUBCARRIER_ORIGIN_F_REF_AT_N0,
    FmcwSpec,
    OfdmSpec,
    PulsedSpec,
    pair_tx_index,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .motion import Motion
    from .paths import RadarPathBatch
    from .results import Result
    from .simulation import Paths
    from .synthesis import SynthesisResult
    from .targets import MeshTargets, PointTargets, StructureTargets

WAVEFORM_FMCW = "fmcw"
WAVEFORM_OFDM = "ofdm"
WAVEFORM_PULSED = "pulsed"
WAVEFORM_KINDS = (WAVEFORM_FMCW, WAVEFORM_OFDM, WAVEFORM_PULSED)

#: What ``antenna_unit`` may say. Half-wavelength offsets are the convention
#: every TI-style configuration uses and the one the array keeps internally:
#: the same description then means the same beam pattern at a different
#: carrier, because an array is defined by its electrical size.
ANTENNA_UNITS = ("m", "half_wavelength")

#: What ``polarization`` may name instead of a world vector. Both are derived
#: from the pose, so they are transverse to the boresight by construction - the
#: property a hand-written vector silently loses, producing a cube of exact
#: zeros with nothing raised.
POLARIZATION_ALIASES = ("up", "right")


# ---------------------------------------------------------------------------
# Waveforms
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Chirp:
    """One chirp profile of an FMCW sequence, in SI units (R-ADR-047).

    A field left at ``None`` is the :class:`Fmcw` ramp's own value. The
    sequence is a TI-style chirp configuration: one profile per slot of a loop,
    repeated across the frame, so several profiles share a frame. Every profile
    keeps the waveform's sample rate and sample count, because the frame is one
    cube.
    """

    #: Ramp slope, Hz/s.
    slope: float | None = None
    #: Start frequency relative to ``Radar.carrier``, Hz.
    start_offset: float = 0.0
    #: Idle time after the ramp, s.
    idle: float | None = None
    #: Delay from the ramp start to the first ADC sample, s.
    adc_start: float | None = None
    #: Ramp end time, s.
    ramp_end: float | None = None

    def __post_init__(self) -> None:
        for name in ("slope", "idle", "adc_start", "ramp_end"):
            if getattr(self, name) is not None:
                _finite(f"Chirp.{name}", getattr(self, name))
        _finite("Chirp.start_offset", self.start_offset)


@dataclass(frozen=True, slots=True)
class Fmcw:
    """One FMCW ramp and its ADC window, in SI units.

    The vendor units a millimetre-wave datasheet quotes - MHz/us of slope, kSPS
    of sample rate, microseconds of timing - are read by :meth:`from_ti`, which
    is the one conversion site in the package. Everything downstream of this
    record is SI.
    """

    kind: ClassVar[str] = WAVEFORM_FMCW

    #: Ramp slope, Hz/s. A 60.012 MHz/us ramp is ``60.012e12``.
    slope: float
    #: ADC sample rate, Hz.
    sample_rate: float
    #: ADC samples taken per chirp.
    samples_per_chirp: int
    #: Chirps per frame, per transmitter.
    chirps_per_frame: int
    #: Delay from the chirp start to the first ADC sample, s.
    adc_start: float
    #: Idle time between ramps, s.
    idle: float
    #: Ramp end time, s. The chirp period is ``idle + ramp_end``.
    ramp_end: float
    #: ``"spectrum"`` for the normalized range spectrum, ``"beat"`` for the
    #: synthesized time-domain beat samples. Spectrum is the default and the
    #: beat route is an opt-in output domain, never a fallback.
    output: str = "spectrum"
    #: A chirp sequence: one :class:`Chirp` profile per slot of a loop, repeated
    #: across the frame's slots. Empty is the one ramp the fields above describe.
    sequence: tuple[Chirp, ...] = ()
    #: The transmitted ramp's phase error, rad: coefficients ``c_2, c_3, ...``
    #: of ``phi(t) = sum_k c_k t^k`` over the time since the ramp start
    #: (rad/s^k). A quadratic frequency error is a cubic phase.
    phase_error: tuple[float, ...] = ()
    #: ``"complex"`` for an IQ receiver, ``"real"`` for one that digitises the
    #: in-phase channel only (R-ADR-047).
    sampling: str = "complex"
    #: ``"tdm"``, ``"ddm"`` or ``"bpm"``: how the transmitters share the frame
    #: (R-ADR-048). DDM and BPM fire every transmitter every chirp and publish
    #: one transmit channel; :func:`witwin.radar.processing.demultiplex`
    #: separates them.
    mimo: str = "tdm"
    #: DDM only: the number of Doppler sub-bands, at least the transmitter
    #: count. Zero is the transmitter count.
    ddm_bands: int = 0

    def __post_init__(self) -> None:
        if self.sampling not in ("complex", "real"):
            raise ValueError(f"Fmcw.sampling must be 'complex' or 'real', got {self.sampling!r}")
        if not all(isinstance(chirp, Chirp) for chirp in self.sequence):
            raise TypeError("Fmcw.sequence holds Chirp records")
        object.__setattr__(self, "sequence", tuple(self.sequence))
        object.__setattr__(self, "phase_error", tuple(_finite("Fmcw.phase_error", value) for value in self.phase_error))
        _positive("Fmcw.sample_rate", self.sample_rate)
        _positive("Fmcw.ramp_end", self.ramp_end)
        _positive_int("Fmcw.samples_per_chirp", self.samples_per_chirp)
        _positive_int("Fmcw.chirps_per_frame", self.chirps_per_frame)
        _finite("Fmcw.slope", self.slope)
        _finite("Fmcw.adc_start", self.adc_start)
        _non_negative("Fmcw.idle", self.idle)
        if self.output not in ("spectrum", "beat"):
            raise ValueError(f"Fmcw.output must be 'spectrum' or 'beat', got {self.output!r}")

    @property
    def chirp_period(self) -> float:
        """``idle + ramp_end``, s."""

        return float(self.idle) + float(self.ramp_end)

    @classmethod
    def from_ti(
        cls,
        *,
        slope_mhz_per_us: float,
        sample_rate_ksps: float,
        samples_per_chirp: int,
        chirps_per_frame: int,
        adc_start_us: float,
        idle_us: float,
        ramp_end_us: float,
        output: str = "spectrum",
    ) -> Fmcw:
        """Build from the units a TI-style configuration file quotes.

        The parameter names carry their vendor units here, and only here,
        because that is the whole job of this constructor: a caller reading a
        datasheet needs to see which column each number came from.
        """

        return cls(
            slope=float(slope_mhz_per_us) * 1e12,
            sample_rate=float(sample_rate_ksps) * 1e3,
            samples_per_chirp=int(samples_per_chirp),
            chirps_per_frame=int(chirps_per_frame),
            adc_start=float(adc_start_us) * 1e-6,
            idle=float(idle_us) * 1e-6,
            ramp_end=float(ramp_end_us) * 1e-6,
            output=str(output),
        )

    def to_spec(self, *, carrier: float, num_tx: int = 1, num_rx: int = 1, offset: float = 0.0) -> FmcwSpec:
        """The SI synthesis spec. ``offset`` is a wideband band offset, Hz."""

        carrier_hz = float(offset)
        return FmcwSpec(
            num_samples=int(self.samples_per_chirp),
            num_chirps=int(self.chirps_per_frame),
            sample_period_s=1.0 / float(self.sample_rate),
            chirp_period_s=self.chirp_period,
            slope_hz_per_s=float(self.slope),
            t_start_s=float(self.adc_start),
            reference_frequency_hz=float(carrier),
            carrier_hz=carrier_hz,
            carrier_rate_hz=0.0 if carrier_hz != 0.0 else float(carrier),
            num_tx=int(num_tx),
            num_rx=int(num_rx),
            output_domain=self.output,
            sequence=tuple(self._profile(chirp) for chirp in self.sequence),
            phase_error_cycles=tuple(value / math.tau for value in self.phase_error),
            real_adc=self.sampling == "real",
            mimo=str(self.mimo),
            ddm_bands=int(self.ddm_bands),
        )

    def _profile(self, chirp: Chirp) -> tuple[float, float, float, float, float]:
        """``(slope, start offset, idle, ADC start, ramp end)`` with this ramp's values filling the gaps."""

        def pick(value, default):
            return float(default if value is None else value)

        return (
            pick(chirp.slope, self.slope),
            float(chirp.start_offset),
            pick(chirp.idle, self.idle),
            pick(chirp.adc_start, self.adc_start),
            pick(chirp.ramp_end, self.ramp_end),
        )


@dataclass(frozen=True, slots=True)
class Ofdm:
    """One OFDM symbol grid, in SI units."""

    kind: ClassVar[str] = WAVEFORM_OFDM

    #: Subcarrier spacing, Hz.
    subcarrier_spacing: float
    num_subcarriers: int
    #: Cyclic prefix duration, s.
    cyclic_prefix: float
    num_symbols: int
    #: The longest round-trip delay the cyclic prefix must cover, s.
    max_expected_delay: float
    subcarrier_origin: str = SUBCARRIER_ORIGIN_F_REF_AT_N0
    #: The subcarrier constellation: ``"ones"`` (the unmodulated comb the ideal
    #: CFR assumes), ``"bpsk"``, ``"qpsk"``, ``"qam16"`` or ``"qam64"``. A
    #: modulated stream is synthesized as received time samples, so intra-symbol
    #: motion produces inter-carrier interference (R-ADR-046).
    modulation: str = "ones"
    #: Every ``pilot_spacing``-th subcarrier carries the known pilot ``1``.
    pilot_spacing: int = 0
    #: Seed of the data symbols.
    symbol_seed: int = 0
    #: ``"cfr"`` publishes the per-subcarrier estimate ``Y / X``; ``"time"``
    #: publishes the received post-prefix samples.
    output: str = "cfr"

    def __post_init__(self) -> None:
        _positive("Ofdm.subcarrier_spacing", self.subcarrier_spacing)
        _positive("Ofdm.cyclic_prefix", self.cyclic_prefix)
        _positive_int("Ofdm.num_subcarriers", self.num_subcarriers)
        _positive_int("Ofdm.num_symbols", self.num_symbols)
        _non_negative("Ofdm.max_expected_delay", self.max_expected_delay)

    @property
    def bandwidth(self) -> float:
        """``N_sc * df``, the occupied band, Hz."""

        return float(self.subcarrier_spacing) * int(self.num_subcarriers)

    @property
    def sample_rate(self) -> float:
        """The symbol grid's complex sample rate ``N_sc * df``, Hz.

        A CFR bin is one output of a unitary ``N_sc``-point transform of time
        samples at this rate, so its thermal variance is ``k T_sys`` times this
        rate: the receive chain's per-sample noise contract holds bin by bin.
        """

        return self.bandwidth

    def to_spec(self, *, carrier: float, offset: float = 0.0) -> OfdmSpec:
        carrier_hz = float(offset)
        return OfdmSpec(
            subcarrier_spacing_hz=float(self.subcarrier_spacing),
            num_subcarriers=int(self.num_subcarriers),
            cyclic_prefix_s=float(self.cyclic_prefix),
            num_symbols=int(self.num_symbols),
            max_expected_delay_s=float(self.max_expected_delay),
            reference_frequency_hz=float(carrier),
            carrier_hz=carrier_hz,
            # Derived, never passed: the weight owns the carrier on the
            # production path, so the rate is the reference frequency there and
            # zero when the caller puts the carrier in the kernel instead.
            # Dropping it understates Doppler by the whole carrier term.
            carrier_rate_hz=0.0 if carrier_hz != 0.0 else float(carrier),
            subcarrier_origin=self.subcarrier_origin,
            modulation=str(self.modulation),
            pilot_spacing=int(self.pilot_spacing),
            symbol_seed=int(self.symbol_seed),
            output_domain=str(self.output),
        )


@dataclass(frozen=True, slots=True)
class Pulsed:
    """One pulse train and its range gate, in SI units.

    The directly sampled family: the receiver mixes the echo down with a fixed
    local oscillator and samples it, and a matched filter compresses it. Four
    constructors spell the coded and stepped members (R-ADR-046):
    :meth:`pmcw` (a binary or polyphase code transmitted continuously),
    :meth:`fsk` (a frequency-hopped pulse), :meth:`stepped_frequency` (a carrier
    that steps from pulse to pulse) and :meth:`arbitrary` (a user-sampled
    complex envelope, held for one chip per sample like a DAC holds it).
    """

    kind: ClassVar[str] = WAVEFORM_PULSED

    #: ``"rect"``, ``"lfm"`` or ``"coded"``.
    pulse_kind: str
    #: Pulse width, s. For a coded pulse, ``num_chips`` chip widths.
    pulse_width: float
    #: Chirp bandwidth of an LFM pulse, or the chip rate of a coded one, Hz.
    bandwidth: float
    #: Pulse repetition interval, s.
    pri: float
    num_pulses: int
    #: Receive sample rate, Hz.
    sample_rate: float
    num_samples: int
    #: Delay from the pulse start to the first range gate sample, s.
    range_gate_start: float
    #: The largest ``d(tau)/dt`` the matched filter must tolerate,
    #: dimensionless.
    max_expected_delay_rate: float = 0.0
    pulse_normalization: str = PULSE_NORMALIZATION_UNIT_AMPLITUDE
    #: ``"coded"`` only: complex chip amplitudes, each of magnitude at most one
    #: (one is the full transmit amplitude).
    chips: tuple[complex, ...] = ()
    #: ``"coded"`` only: each chip's frequency, Hz. Empty is all zero.
    chip_frequencies: tuple[float, ...] = ()
    #: Carrier step between consecutive pulses, Hz. Zero is a fixed carrier.
    frequency_step: float = 0.0
    #: The code repeats with no gap, one period per slow-time sample.
    periodic: bool = False

    def __post_init__(self) -> None:
        _positive("Pulsed.pulse_width", self.pulse_width)
        _positive("Pulsed.bandwidth", self.bandwidth)
        _positive("Pulsed.pri", self.pri)
        _positive("Pulsed.sample_rate", self.sample_rate)
        _positive_int("Pulsed.num_pulses", self.num_pulses)
        _positive_int("Pulsed.num_samples", self.num_samples)
        _non_negative("Pulsed.range_gate_start", self.range_gate_start)
        _non_negative("Pulsed.max_expected_delay_rate", self.max_expected_delay_rate)
        _finite("Pulsed.frequency_step", self.frequency_step)
        object.__setattr__(self, "chips", tuple(complex(chip) for chip in self.chips))
        object.__setattr__(self, "chip_frequencies", tuple(float(value) for value in self.chip_frequencies))

    def to_spec(self, *, carrier: float, offset: float = 0.0) -> PulsedSpec:
        carrier_hz = float(offset)
        return PulsedSpec(
            pulse_kind=str(self.pulse_kind),
            pulse_width_s=float(self.pulse_width),
            bandwidth_hz=float(self.bandwidth),
            pri_s=float(self.pri),
            num_pulses=int(self.num_pulses),
            sample_period_s=1.0 / float(self.sample_rate),
            num_samples=int(self.num_samples),
            range_gate_start_s=float(self.range_gate_start),
            max_expected_delay_rate=float(self.max_expected_delay_rate),
            reference_frequency_hz=float(carrier),
            carrier_hz=carrier_hz,
            carrier_rate_hz=0.0 if carrier_hz != 0.0 else float(carrier),
            pulse_normalization=self.pulse_normalization,
            chips=self.chips,
            chip_frequencies_hz=self.chip_frequencies,
            frequency_step_hz=float(self.frequency_step),
            periodic=bool(self.periodic),
        )

    # -- the coded and stepped members --------------------------------------

    @classmethod
    def pmcw(
        cls, code, *, chip_rate: float, repetitions: int, sample_rate: float, max_expected_delay_rate: float = 0.0
    ) -> Pulsed:
        """A phase code transmitted continuously, one code period per slow-time sample.

        ``code`` is the chip sequence: ``+1``/``-1`` for a binary code such as
        :meth:`m_sequence`, or unit complex values for a polyphase one. The
        range gate is one code period, ``len(code) * sample_rate / chip_rate``
        samples, which must be a whole number; the matched filter is the
        cyclic correlation over it.
        """

        chips = tuple(complex(value) for value in code)
        period = len(chips) / float(chip_rate)
        samples = round(period * float(sample_rate))
        if samples < 1 or abs(samples / float(sample_rate) - period) > 1e-9 * period:
            raise ValueError(
                f"one code period ({len(chips)} chips at {chip_rate} Hz) is not a whole number of samples at "
                f"{sample_rate} Hz; the cyclic matched filter needs the gate to hold exactly one period"
            )
        return cls(
            pulse_kind="coded",
            pulse_width=period,
            bandwidth=float(chip_rate),
            pri=period,
            num_pulses=int(repetitions),
            sample_rate=float(sample_rate),
            num_samples=int(samples),
            range_gate_start=0.0,
            max_expected_delay_rate=max_expected_delay_rate,
            chips=chips,
            periodic=True,
        )

    @classmethod
    def fsk(
        cls,
        frequencies,
        *,
        chip_width: float,
        pri: float,
        num_pulses: int,
        sample_rate: float,
        num_samples: int,
        range_gate_start: float = 0.0,
        max_expected_delay_rate: float = 0.0,
    ) -> Pulsed:
        """A frequency-hopped pulse: chip ``i`` is a tone at ``frequencies[i]`` Hz.

        Each chip starts at zero phase. The declared bandwidth is the hop span
        plus one chip's own ``1 / chip_width``.
        """

        tones = tuple(float(value) for value in frequencies)
        if not tones:
            raise ValueError("an FSK pulse needs at least one frequency")
        return cls(
            pulse_kind="coded",
            pulse_width=len(tones) * float(chip_width),
            bandwidth=max(tones) - min(tones) + 1.0 / float(chip_width),
            pri=float(pri),
            num_pulses=int(num_pulses),
            sample_rate=float(sample_rate),
            num_samples=int(num_samples),
            range_gate_start=float(range_gate_start),
            max_expected_delay_rate=max_expected_delay_rate,
            chips=(1.0 + 0.0j,) * len(tones),
            chip_frequencies=tones,
        )

    @classmethod
    def stepped_frequency(
        cls,
        *,
        step: float,
        num_steps: int,
        pulse_width: float,
        pri: float,
        sample_rate: float,
        num_samples: int,
        range_gate_start: float = 0.0,
        max_expected_delay_rate: float = 0.0,
    ) -> Pulsed:
        """Rectangular pulses whose carrier steps by ``step`` Hz from pulse to pulse.

        The synthetic range profile is the inverse transform across the
        ``num_steps`` pulses
        (:func:`~witwin.radar.processing.stepped_frequency_profile`): its
        resolution is ``c / (2 num_steps step)`` and its unambiguous range
        ``c / (2 step)``.
        """

        return cls(
            pulse_kind="rect",
            pulse_width=float(pulse_width),
            bandwidth=1.0 / float(pulse_width),
            pri=float(pri),
            num_pulses=int(num_steps),
            sample_rate=float(sample_rate),
            num_samples=int(num_samples),
            range_gate_start=float(range_gate_start),
            max_expected_delay_rate=max_expected_delay_rate,
            frequency_step=float(step),
        )

    @classmethod
    def arbitrary(
        cls,
        samples,
        *,
        rate: float,
        pri: float,
        num_pulses: int,
        sample_rate: float,
        num_samples: int,
        bandwidth: float | None = None,
        range_gate_start: float = 0.0,
        max_expected_delay_rate: float = 0.0,
    ) -> Pulsed:
        """A user-sampled complex envelope, each sample held for ``1 / rate`` s.

        ``samples`` are relative to the full transmit amplitude, so their
        magnitudes may not exceed one. ``bandwidth`` defaults to ``rate``.
        """

        chips = tuple(complex(value) for value in samples)
        if not chips:
            raise ValueError("an arbitrary envelope needs at least one sample")
        return cls(
            pulse_kind="coded",
            pulse_width=len(chips) / float(rate),
            bandwidth=float(rate) if bandwidth is None else float(bandwidth),
            pri=float(pri),
            num_pulses=int(num_pulses),
            sample_rate=float(sample_rate),
            num_samples=int(num_samples),
            range_gate_start=float(range_gate_start),
            max_expected_delay_rate=max_expected_delay_rate,
            chips=chips,
        )

    @staticmethod
    def m_sequence(order: int) -> tuple[int, ...]:
        """The maximal-length binary sequence of length ``2**order - 1``, as ``+1``/``-1``.

        Generated by a Fibonacci shift register on a primitive trinomial or
        pentanomial (orders 2-16). Its periodic autocorrelation is ``N`` at lag
        zero and ``-1`` at every other lag.
        """

        taps = _M_SEQUENCE_TAPS.get(int(order))
        if taps is None:
            raise ValueError(f"m-sequence order must be one of {sorted(_M_SEQUENCE_TAPS)}, got {order!r}")
        length = (1 << order) - 1
        state = [1] * order
        out = []
        for _ in range(length):
            out.append(state[-1])
            feedback = 0
            for tap in taps:
                feedback ^= state[tap - 1]
            state = [feedback, *state[:-1]]
        return tuple(1 - 2 * bit for bit in out)

    @staticmethod
    def costas(prime: int) -> tuple[int, ...]:
        """A Welch-Costas frequency order of length ``prime - 1``: ``g**i mod prime``.

        ``g`` is the smallest primitive root. Multiply by a tone spacing of
        ``1 / chip_width`` for :meth:`fsk`; the result has a thumbtack
        ambiguity function.
        """

        p = int(prime)
        if p < 3 or any(p % d == 0 for d in range(2, int(math.isqrt(p)) + 1)):
            raise ValueError(f"costas needs an odd prime, got {prime!r}")
        for g in range(2, p):
            if len({pow(g, i, p) for i in range(1, p)}) == p - 1:
                return tuple(pow(g, i, p) for i in range(1, p))
        raise AssertionError("every prime has a primitive root")


#: Feedback taps of a primitive polynomial per register length, counted from
#: one: ``x^n + x^k + 1`` and the pentanomials where no trinomial is primitive.
_M_SEQUENCE_TAPS = {
    2: (2, 1),
    3: (3, 2),
    4: (4, 3),
    5: (5, 3),
    6: (6, 5),
    7: (7, 6),
    8: (8, 6, 5, 4),
    9: (9, 5),
    10: (10, 7),
    11: (11, 9),
    12: (12, 6, 4, 1),
    13: (13, 4, 3, 1),
    14: (14, 5, 3, 1),
    15: (15, 14),
    16: (16, 15, 13, 4),
}


Waveform = Fmcw | Ofdm | Pulsed


# ---------------------------------------------------------------------------
# Internal block specs
#
# These are what a consumer is HANDED: the propagation adapter sees the
# propagation block and nothing else, the sensor stage sees the array. They are
# derived from the radar rather than authored, which is why they keep their
# unit-suffixed field names and are absent from the public surface.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SensorConfig:
    """The array and its transmit and receive element patterns."""

    array: SensorArraySpec
    pattern: Pattern
    rx_pattern: Pattern | None = None

    @property
    def receive_pattern(self) -> Pattern:
        return self.pattern if self.rx_pattern is None else self.rx_pattern


@dataclass(frozen=True, slots=True)
class PropagationConfig:
    """The ONLY block a propagation adapter is ever handed.

    Folding a waveform field in here is what the boundary test exists to
    prevent, and it asserts the request keyword set by EQUALITY rather than by
    containment, because a containment check passes when a field is added.
    """

    reference_frequency_hz: float
    components: frozenset[str] = frozenset({"los", "reflection"})
    max_depth: int = 1

    def __post_init__(self) -> None:
        if not self.reference_frequency_hz > 0.0:
            raise ValueError("reference_frequency_hz must be positive")
        if self.max_depth < 0:
            raise ValueError("max_depth must be non-negative")


@dataclass(frozen=True, slots=True)
class RadarSystemConfig:
    """The four blocks, with the waveform discriminator stored rather than read.

    ``waveform.kind`` is the dispatch key. It is a class attribute of the
    waveform record, so a block cannot be built with the wrong one and a caller
    cannot infer a different one by looking for a ``slope``.
    """

    waveform: Waveform
    sensors: SensorConfig
    propagation: PropagationConfig
    frontend: FrontendSpec | None = None

    def __post_init__(self) -> None:
        if self.waveform.kind not in WAVEFORM_KINDS:
            raise ValueError(f"waveform.kind must be one of {list(WAVEFORM_KINDS)}, got {self.waveform.kind!r}")
        if self.sensors.array.reference_frequency_hz != self.propagation.reference_frequency_hz:
            raise ValueError(
                "the array's reference frequency and the propagation reference "
                "frequency are the same physical quantity and must agree"
            )

    @property
    def kind(self) -> str:
        return self.waveform.kind

    def waveform_spec(self, *, carrier_hz: float = 0.0):
        """The SI synthesis spec for whichever waveform this configuration is.

        ``__post_init__`` already refused any kind outside ``WAVEFORM_KINDS``,
        so the only branch here is the FMCW spec's extra array arguments.
        """

        array = self.sensors.array
        reference = self.propagation.reference_frequency_hz
        if self.waveform.kind == WAVEFORM_FMCW:
            return self.waveform.to_spec(carrier=reference, num_tx=array.num_tx, num_rx=array.num_rx, offset=carrier_hz)
        return self.waveform.to_spec(carrier=reference, offset=carrier_hz)

    def with_propagation(self, *, components=None, max_depth: int | None = None) -> RadarSystemConfig:
        """A copy whose propagation block carries these two knobs.

        A per-solve request returns a new configuration rather than mutating
        this one, because an override that edited the radar's stored
        configuration would silently change every LATER solve as well.

        ``reference_frequency_hz`` is deliberately not overridable: it is tied
        to the array's element spacing and to the compiled scene by Channel, so
        changing it here would produce a configuration that is refused later
        rather than one that means something else.
        """

        if components is None and max_depth is None:
            return self
        current = self.propagation
        return replace(
            self,
            propagation=PropagationConfig(
                reference_frequency_hz=current.reference_frequency_hz,
                components=(current.components if components is None else frozenset(components)),
                max_depth=(current.max_depth if max_depth is None else int(max_depth)),
            ),
        )


# ---------------------------------------------------------------------------
# Primitive validators
# ---------------------------------------------------------------------------


def _scalar_leaf(name: str, value: Any) -> Any:
    """A leaf-capable scalar field: a host number, or a 0-dim floating tensor (read once, for validation)."""

    if isinstance(value, torch.Tensor):
        if value.dim() != 0 or not value.is_floating_point():
            raise ValueError(f"a tensor {name} must be a 0-dim floating tensor, got shape {tuple(value.shape)}")
        return float(value.detach())
    return value


def _host_scalar(value: Any) -> float:
    return float(value.detach()) if isinstance(value, torch.Tensor) else float(value)


def _host_or_leaf(value: Any) -> Any:
    return value if isinstance(value, torch.Tensor) else float(value)


def _finite(name: str, value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite float, got {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be a finite float, got {value!r}")
    return parsed


def _non_negative(name: str, value: Any) -> float:
    parsed = _finite(name, value)
    if parsed < 0.0:
        raise ValueError(f"{name} must be non-negative, got {parsed}")
    return parsed


def _positive(name: str, value: Any) -> float:
    parsed = _finite(name, value)
    if parsed <= 0.0:
        raise ValueError(f"{name} must be positive, got {parsed}")
    return parsed


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive int, got {value!r}")
    return value


def vec3_tensor(value, *, name: str) -> torch.Tensor:
    """Coerce a pose vector to a CPU float32 tensor of shape (3,).

    A tensor keeps its tape: the radar's pose is a leaf (R-ADR-052), and every
    element position, the pattern frame and the point cloud's world mapping
    are differentiable expressions of it. Nothing here detaches.
    """

    if isinstance(value, torch.Tensor):
        tensor = value.to(device="cpu", dtype=torch.float32).reshape(-1)
    else:
        tensor = torch.tensor(tuple(float(component) for component in value), dtype=torch.float32)
    if tensor.shape != (3,):
        raise ValueError(f"{name} must contain exactly three values.")
    return tensor


def _elements(value, *, name: str) -> tuple[tuple[float, float, float], ...]:
    if isinstance(value, torch.Tensor):
        rows = value.detach().to(device="cpu", dtype=torch.float32).tolist()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        rows = list(value)
    else:
        raise TypeError(f"{name} must be a sequence of 3-element positions, got {type(value).__name__}")
    if not rows:
        raise ValueError(f"{name} must name at least one element")
    out: list[tuple[float, float, float]] = []
    for index, row in enumerate(rows):
        if isinstance(row, torch.Tensor):
            row = row.detach().to(device="cpu", dtype=torch.float32).tolist()
        if not isinstance(row, Sequence) or len(row) != 3:
            raise ValueError(f"{name}[{index}] must be a 3-element position")
        out.append(tuple(_finite(f"{name}[{index}][{axis}]", row[axis]) for axis in range(3)))
    return tuple(out)


# ---------------------------------------------------------------------------
# Radar
# ---------------------------------------------------------------------------


def _pair_table(name: str, value) -> float | tuple[tuple[float, ...], ...]:
    """A scalar, or a ``[num_tx][num_rx]`` table, of finite host floats."""

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and value:
        table = tuple(tuple(float(entry) for entry in row) for row in value)
        if len({len(row) for row in table}) != 1 or not table[0]:
            raise ValueError(f"Leakage.{name} must be a scalar or a rectangular [num_tx][num_rx] table")
        return table
    raise ValueError(f"Leakage.{name} must be a float or a [num_tx][num_rx] table, got {value!r}")


@dataclass(frozen=True, slots=True)
class Leakage:
    """Transmit-to-receive antenna coupling: the echo that never left the board.

    A fraction of every transmitted chirp reaches the receiver directly, and
    after de-chirping it is a tone at ``slope * delay_s`` - a target at a range
    of centimetres, tens of dB above anything the scene returns. Two things make
    it matter rather than being a constant to subtract:

    * it spends the receiver's dynamic range, so the AGC gain and the ADC
      headroom a real system has are the ones left over after it; and
    * the range transform of a rectangular window puts its skirt across the
      near bins, which for a target inside a metre is where the target is.

    A synthetic set without it teaches a model that the near bins are clean.

    ``isolation_db`` is the transmit-to-receive isolation, positive, so the
    coupled amplitude is ``sqrt(P_tx) * 10^(-isolation_db/20)`` in the sqrt(W)
    the synthesis families publish. ``delay_s`` is the coupling path's own
    round-trip delay, which sets the beat tone and therefore which bin the
    skirt radiates from. ``phase`` is the coupling coefficient's argument at
    the reference frequency, rad, in Channel's phasor convention.

    Each of the three is either one number for every transmitter-receiver pair
    or a ``[num_tx][num_rx]`` table, because a board's coupling is a measured
    property of each pair: neighbouring elements couple more strongly than
    distant ones.

    This is one more path row per pair, not a new physics owner: it is
    synthesized by the same waveform kernel the scene is, with the same phase
    law, so nothing here can drift from the echo it sits on top of. That holds
    for every waveform: FMCW de-chirps it into a near-range tone, OFDM sees it
    as a short-delay CFR term, and a pulsed receiver sees it only where its
    range gate is open while the pulse is still being transmitted.
    """

    #: Transmit-to-receive isolation, dB, scalar or ``[num_tx][num_rx]``.
    isolation_db: Any
    #: Coupling path delay, s, scalar or ``[num_tx][num_rx]``.
    delay_s: Any
    #: Coupling phase at the reference frequency, rad, scalar or table.
    phase: Any = 0.0

    def __post_init__(self) -> None:
        refuse_derivative(
            "Leakage", _LEAKAGE_REASON, isolation_db=self.isolation_db, delay_s=self.delay_s, phase=self.phase
        )
        for name in ("isolation_db", "delay_s", "phase"):
            object.__setattr__(self, name, _pair_table(name, getattr(self, name)))
        for value in self._flat("isolation_db"):
            if not math.isfinite(value) or not value > 0.0:
                raise ValueError(f"isolation_db must be positive, got {value}")
        for value in self._flat("delay_s"):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"delay_s must not be negative, got {value}")
        for value in self._flat("phase"):
            if not math.isfinite(value):
                raise ValueError(f"phase must be finite, got {value}")

    def _flat(self, name: str) -> tuple[float, ...]:
        value = getattr(self, name)
        return (value,) if isinstance(value, float) else tuple(entry for row in value for entry in row)

    def amplitude(self, power_dbm: float) -> float:
        """Coupled amplitude in sqrt(W) of a scalar isolation, from the transmit power."""

        if not isinstance(self.isolation_db, float):
            raise ValueError("a per-pair isolation has one amplitude per pair; use pair_rows")
        return math.sqrt(watts_from_dbm(_host_scalar(power_dbm))) * 10.0 ** (-float(self.isolation_db) / 20.0)

    def pair_rows(self, *, num_tx: int, num_rx: int, power_dbm: float) -> tuple[list[float], list[complex]]:
        """Delay and complex coupling coefficient per sensor pair, in pair-rank order.

        Pair rank is sink major (``pair = rx * num_tx + tx``), the partition the
        synthesis families group rows by. A table is indexed ``[tx][rx]``.
        """

        def entry(name: str, tx: int, rx: int) -> float:
            value = getattr(self, name)
            if isinstance(value, float):
                return value
            if len(value) != num_tx or len(value[0]) != num_rx:
                raise ValueError(
                    f"Leakage.{name} is a {len(value)} x {len(value[0])} table but the array is {num_tx} x {num_rx}"
                )
            return value[tx][rx]

        root_power = math.sqrt(watts_from_dbm(_host_scalar(power_dbm)))
        delays: list[float] = []
        weights: list[complex] = []
        for rx in range(num_rx):
            for tx in range(num_tx):
                magnitude = root_power * 10.0 ** (-entry("isolation_db", tx, rx) / 20.0)
                delays.append(entry("delay_s", tx, rx))
                weights.append(magnitude * cmath.exp(1j * entry("phase", tx, rx)))
        return delays, weights


@dataclass(frozen=True)
class Radar:
    """One radar: its waveform, its array, its receive chain and its pose.

    Immutable. :meth:`replace` returns a new radar rather than editing this
    one, so a radar captured in a closure or held by a result cannot change
    underneath it.

    The physically meaningful scalars and vectors are leaves (R-ADR-052): a
    tensor ``position``, ``look_at`` or ``up`` carries its derivative through
    every element position and the pattern frame (with a world-vector
    polarization; an ``"up"``/``"right"`` one turns with the pose and is
    refused for a pose leaf), a tensor ``power`` through every transmitted
    amplitude, a tensor ``lna_gain`` through the receive chain, and a
    ``Pattern`` built from tensors through its table values and its gain.

    Four verbs use it, and they are two halves and their fusion.
    :meth:`trace` runs the world half and keeps the composed round trips;
    :meth:`echo` runs the instrument half over those rows. :meth:`simulate`
    fuses the two and stacks every frame; :meth:`stream` fuses them and yields
    one frame at a time, so a sequence too long to hold as a stacked cube is
    still producible. All four share one session loop and one synthesis route,
    so ``echo(trace(...))`` and ``simulate(...)`` are the same numbers and not
    merely the same physics.
    """

    #: Reference frequency, Hz. The carrier the array spacing, the propagation
    #: solve and the synthesis all refer to; they are one physical quantity.
    carrier: float
    waveform: Waveform
    #: Transmit element positions, in ``antenna_unit``.
    tx: tuple[tuple[float, float, float], ...]
    #: Receive element positions, in ``antenna_unit``.
    rx: tuple[tuple[float, float, float], ...]
    #: Transmit power, dBm. A 0-dim tensor is a leaf (:meth:`_transmit_amplitude_scale`).
    power: float
    antenna_unit: str = "m"
    #: The element pattern, with its absolute peak gain (``Pattern.gain``,
    #: dBi). Isotropic at 0 dBi by default: an unchosen dipole attenuates every
    #: off-boresight return by a number nobody asked for. It is the transmit
    #: element's, and the receive element's unless ``rx_pattern`` is given.
    pattern: Pattern = field(default_factory=Pattern.isotropic)
    #: The receive element pattern. ``None`` is the same as ``pattern``.
    rx_pattern: Pattern | None = None
    #: Thermal and oscillator noise. ``None`` is an ideal receiver.
    noise: Noise | None = None
    #: Low-noise amplifier voltage gain, dB. ``None`` is no LNA stage. A 0-dim
    #: tensor is a leaf of the receive chain.
    lna_gain: float | None = None
    agc: Agc | None = None
    #: ``None`` is no quantisation.
    adc: Adc | None = None
    #: Quadrature demodulator imbalance. ``None`` is a perfect demodulator.
    iq: Iq | None = None
    #: Transmit-to-receive antenna coupling. ``None`` is perfect isolation,
    #: which is what a scene-only dataset assumes whether it says so or not.
    leakage: Leakage | None = None
    #: Receiver gain compression after the LNA. ``None`` is a linear receiver.
    compression: Compression | None = None
    #: Baseband DC offset at the mixer output, V: one complex value for every
    #: receiver or one per receiver. ``None`` is no offset.
    dc_offset: Any = None
    #: Mutual coupling between array elements. ``None`` is an uncoupled array.
    coupling: Coupling | None = None
    #: Receive port reference impedance, ohm.
    impedance: float = 50.0
    #: The Philox base seed every receiver stage derives its own stream from.
    seed: int = 0
    #: Radar origin in world coordinates, m. A tensor is a leaf.
    position: Any = (0.0, 0.0, 0.0)
    #: The point the boresight looks at, world coordinates, m.
    look_at: Any = (0.0, 0.0, -1.0)
    #: World-space up vector, used to complete the frame.
    up: Any = (0.0, 1.0, 0.0)
    #: ``"up"``, ``"right"`` or a world vector. The two aliases are derived
    #: from the pose and are therefore transverse to the boresight; a vector
    #: parallel to the boresight radiates nothing and is refused. It is the
    #: transmit elements' polarization and, unless ``rx_polarization`` is
    #: given, the receive elements'.
    polarization: Any = "up"
    #: The receive elements' polarization, in the same vocabulary. ``None`` is
    #: ``polarization``; a different value measures a cross-polar channel.
    rx_polarization: Any = None
    device: Any = "cuda"

    # -- derived, built once in __post_init__ ------------------------------
    system_config: RadarSystemConfig = field(init=False, repr=False, compare=False)
    frontend: Any = field(init=False, repr=False, compare=False)
    tx_pos: torch.Tensor = field(init=False, repr=False, compare=False)
    rx_pos: torch.Tensor = field(init=False, repr=False, compare=False)
    polarization_vector: tuple[float, float, float] = field(init=False, repr=False, compare=False)
    rx_polarization_vector: tuple[float, float, float] = field(init=False, repr=False, compare=False)

    # A radar holds no run state. The four typed per-frame diagnostics live on
    # the result that produced them, which is the record that can honestly say
    # which frame they describe; a copy here would be a second owner, and a
    # call that raised part way through would leave it describing a world the
    # failed call never simulated.

    def __post_init__(self) -> None:
        set_ = object.__setattr__
        _positive("Radar.carrier", self.carrier)
        _finite("Radar.power", _scalar_leaf("Radar.power", self.power))
        if self.lna_gain is not None:
            _finite("Radar.lna_gain", _scalar_leaf("Radar.lna_gain", self.lna_gain))
        _positive("Radar.impedance", self.impedance)
        if self.antenna_unit not in ANTENNA_UNITS:
            raise ValueError(f"Radar.antenna_unit must be one of {list(ANTENNA_UNITS)}, got {self.antenna_unit!r}")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError(f"Radar.seed must be a non-negative int, got {self.seed!r}")
        if not isinstance(self.pattern, Pattern):
            raise TypeError(f"Radar.pattern must be a Pattern, got {type(self.pattern).__name__}")
        if self.rx_pattern is not None and not isinstance(self.rx_pattern, Pattern):
            raise TypeError(f"Radar.rx_pattern must be a Pattern, got {type(self.rx_pattern).__name__}")

        set_(self, "device", resolve_device(self.device, owner="Radar"))
        set_(self, "tx", _elements(self.tx, name="Radar.tx"))
        set_(self, "rx", _elements(self.rx, name="Radar.rx"))

        # The array keeps half-wavelength offsets, so a metre-authored layout is
        # divided by the half wavelength exactly here and nowhere else.
        half_wavelength = SPEED_OF_LIGHT_M_PER_S / float(self.carrier) / 2.0
        scale = 1.0 if self.antenna_unit == "half_wavelength" else 1.0 / half_wavelength
        array = SensorArraySpec(
            num_tx=len(self.tx),
            num_rx=len(self.rx),
            tx_loc=tuple(tuple(v * scale for v in row) for row in self.tx),
            rx_loc=tuple(tuple(v * scale for v in row) for row in self.rx),
            reference_frequency_hz=float(self.carrier),
        )

        set_(
            self,
            "system_config",
            RadarSystemConfig(
                waveform=self.waveform,
                sensors=SensorConfig(array=array, pattern=self.pattern, rx_pattern=self.rx_pattern),
                propagation=PropagationConfig(reference_frequency_hz=float(self.carrier)),
                frontend=self._build_frontend_spec(),
            ),
        )
        self._place_antennas()
        set_(self, "frontend", self._build_frontend_chain())

    # -- construction helpers ---------------------------------------------

    def _build_frontend_spec(self) -> FrontendSpec | None:
        """The internal receive-chain record, or ``None`` for an ideal receiver.

        A chain exists when any stage is configured. ``Noise`` arrives with its
        two waveform-dependent numbers possibly unset and leaves resolved, so
        nothing downstream of here ever has to infer a bandwidth - inferring it
        in three places is how those three quietly disagree.

        The noise contract is one rule for every waveform (R-ADR-029): samples
        are physical amplitudes in sqrt(W) and the default noise bandwidth is
        the waveform's complex SAMPLE RATE, which is the equivalent noise
        bandwidth of a white complex sample stream. A pulsed radar's sweep
        ``bandwidth`` is a property of the transmitted pulse, not of the
        receiver's sampling, and is deliberately not used here.
        """

        stages = (
            self.noise,
            self.lna_gain,
            self.agc,
            self.adc,
            self.iq,
            self.compression,
            self.dc_offset,
            self.coupling,
        )
        if all(stage is None for stage in stages):
            return None
        if self.noise is not None and not isinstance(self.noise, Noise):
            raise TypeError(f"Radar.noise must be a Noise, got {type(self.noise).__name__}")
        if self.agc is not None and not isinstance(self.agc, Agc):
            raise TypeError(f"Radar.agc must be an Agc, got {type(self.agc).__name__}")
        if self.adc is not None and not isinstance(self.adc, Adc):
            raise TypeError(f"Radar.adc must be an Adc, got {type(self.adc).__name__}")
        if self.compression is not None and not isinstance(self.compression, Compression):
            raise TypeError(f"Radar.compression must be a Compression, got {type(self.compression).__name__}")
        if self.coupling is not None:
            if not isinstance(self.coupling, Coupling):
                raise TypeError(f"Radar.coupling must be a Coupling, got {type(self.coupling).__name__}")
            self.coupling.require_array(len(self.tx), len(self.rx))
        noise = self.noise
        if noise is not None:
            rate = float(self.waveform.sample_rate)
            noise = noise.resolved(bandwidth=rate, sample_rate=rate)
        timeline = self._receive_timeline()
        baseband = self.dc_offset is not None or (noise is not None and noise.flicker_corner is not None)
        if baseband and self.waveform.kind == WAVEFORM_OFDM:
            raise NotImplementedError(
                "a DC offset and baseband flicker are time-domain impairments, and the OFDM cube is a "
                "frequency-domain CFR; mapping them onto subcarriers needs the symbol's own transform"
            )
        return FrontendSpec(
            noise=noise,
            lna=None if self.lna_gain is None else _host_or_leaf(self.lna_gain),
            iq=self.iq,
            agc=None if self.agc is None else self.agc.resolved(block_period=timeline[0]),
            adc=self.adc,
            impedance=float(self.impedance),
            seed=int(self.seed),
            compression=self.compression,
            dc_offset=self._dc_offsets(),
            tx_coupling=None if self.coupling is None else self.coupling.tx,
            rx_coupling=None if self.coupling is None else self.coupling.rx,
            timeline=timeline,
        )

    def _receive_timeline(self) -> tuple[float, float, float]:
        """``(slow_period_s, fast_start_s, fast_period_s)`` of one frame's cube.

        The slow period is one slow-time index: a whole TDM cycle of
        ``num_tx`` chirps for FMCW, a symbol (prefix included) for OFDM and a
        PRI for a pulsed radar. It is the time-dynamic AGC's block and the
        baseband flicker's clock.
        """

        waveform = self.waveform
        if waveform.kind == WAVEFORM_FMCW:
            spec = waveform.to_spec(carrier=float(self.carrier), num_tx=len(self.tx), num_rx=len(self.rx))
            return (spec.slot_period_s, float(waveform.adc_start), 1.0 / float(waveform.sample_rate))
        if waveform.kind == WAVEFORM_OFDM:
            symbol = 1.0 / float(waveform.subcarrier_spacing) + float(waveform.cyclic_prefix)
            return (symbol, 0.0, 1.0 / float(waveform.sample_rate))
        return (float(waveform.pri), float(waveform.range_gate_start), 1.0 / float(waveform.sample_rate))

    def _dc_offsets(self) -> tuple[complex, ...] | None:
        if self.dc_offset is None:
            return None
        refuse_derivative("Radar.dc_offset", _BASEBAND_REASON, dc_offset=self.dc_offset)
        values = self.dc_offset
        if isinstance(values, (int, float, complex)) and not isinstance(values, bool):
            values = (values,) * len(self.rx)
        offsets = tuple(complex(value) for value in values)
        if len(offsets) != len(self.rx):
            raise ValueError(f"Radar.dc_offset holds {len(offsets)} values but the array has {len(self.rx)} receivers")
        if not all(math.isfinite(value.real) and math.isfinite(value.imag) for value in offsets):
            raise ValueError("Radar.dc_offset must be finite")
        return offsets

    def _build_frontend_chain(self):
        if self.system_config.frontend is None:
            return None
        return FrontendChain(self.system_config.frontend)

    def _place_antennas(self) -> None:
        """Validate the pose and put the elements in the world.

        The refusals are all here rather than at first use: a pose whose up is
        collinear with the boresight has no frame, and a polarization parallel
        to the boresight radiates nothing and would otherwise publish a cube of
        exact zeros with nothing raised.
        """

        set_ = object.__setattr__
        position = vec3_tensor(self.position, name="Radar.position")
        look_at = vec3_tensor(self.look_at, name="Radar.look_at")
        up = vec3_tensor(self.up, name="Radar.up")
        self._require_pose_leaf_polarization()
        forward = look_at - position
        if torch.linalg.norm(forward) <= 1e-12:
            raise ValueError("Radar.look_at must differ from Radar.position.")
        if torch.linalg.norm(up) <= 1e-12:
            raise ValueError("Radar.up must be non-zero.")
        if torch.linalg.norm(torch.cross(forward, up, dim=0)) <= 1e-12:
            raise ValueError("Radar.up must not be collinear with the viewing direction.")

        _, world_from_local = self._world_from_local_matrix(device="cpu", dtype=torch.float32)
        right, true_up, forward = world_from_local[:, 0], world_from_local[:, 1], -world_from_local[:, 2]

        def resolve(value: Any, name: str) -> tuple[float, float, float]:
            if value == "up":
                vector = true_up
            elif value == "right":
                vector = right
            elif isinstance(value, str):
                raise ValueError(f"{name} must be one of {list(POLARIZATION_ALIASES)} or a world vector, got {value!r}")
            else:
                refuse_derivative("witwin.radar.Radar", _POLARIZATION_REASON, **{name: value})
                vector = vec3_tensor(value, name=name)
                norm = torch.linalg.norm(vector)
                if norm <= 1e-12:
                    raise ValueError(f"{name} must be non-zero.")
                vector = vector / norm
                if float(torch.linalg.norm(torch.cross(vector, forward.detach(), dim=0))) <= 1e-6:
                    raise ValueError(
                        f"{name} is parallel to the boresight, so the field radiates nothing and every "
                        "transport would come back exactly zero. Use 'up' or 'right' for a vector that is "
                        "transverse by construction."
                    )
            return tuple(float(v) for v in vector)  # type: ignore[return-value]

        set_(self, "polarization_vector", resolve(self.polarization, "Radar.polarization"))
        set_(
            self,
            "rx_polarization_vector",
            self.polarization_vector
            if self.rx_polarization is None
            else resolve(self.rx_polarization, "Radar.rx_polarization"),
        )

        array = self.system_config.sensors.array
        tx_local, rx_local = array.local_offsets_m(device=self.device)
        world_from_local = world_from_local.to(device=self.device)
        origin = position.to(device=self.device)
        set_(self, "tx_pos", (tx_local @ world_from_local.transpose(0, 1) + origin).contiguous())
        set_(self, "rx_pos", (rx_local @ world_from_local.transpose(0, 1) + origin).contiguous())

    def _require_pose_leaf_polarization(self) -> None:
        """Refuse a pose leaf whose polarization the pose would turn.

        An ``"up"`` or ``"right"`` polarization is derived from the pose, so
        a derivative of the position, ``look_at`` or ``up`` would also move
        the polarization every leg is launched with. Channel's fixed-topology
        replay takes endpoint polarizations as primal-only constants and
        publishes no derivative for them, so that part of the derivative
        cannot be delivered; carrying the rest would be a silently partial
        gradient. A world-vector polarization does not turn with the pose and
        the derivative through the element positions and the pattern frame is
        then complete.
        """

        marked = [f"Radar.{name}" for name in ("position", "look_at", "up") if _carries_derivative(getattr(self, name))]
        if not marked:
            return
        aliased = [
            f"Radar.{name}={value!r}"
            for name, value in (("polarization", self.polarization), ("rx_polarization", self.rx_polarization))
            if isinstance(value, str)
        ]
        if aliased:
            raise RuntimeError(
                f"{', '.join(marked)} carries a derivative, but {', '.join(aliased)} is derived from the pose, so "
                "the polarization every leg is launched with would turn with it. Channel's replay treats an "
                "endpoint polarization as a constant and publishes no derivative for it, so that part of the "
                "pose derivative cannot be delivered and the rest would be silently partial. Declare the "
                "polarization as a world vector (for example polarization=(0.0, 1.0, 0.0)), which does not "
                "turn with the pose, or keep the pose a host declaration."
            )

    def _require_pose_ad_mode(self, grad: str) -> None:
        """A pose leaf moves every element, so Channel's replay must carry its mode.

        The element positions reach both legs through Channel's fixed-topology
        replay, which publishes the position derivative only in the AD mode
        the verb asks for: ``grad="vjp"`` for a gradient and ``grad="jvp"`` for
        a forward tangent. Any other pairing would leave the legs' part out
        and publish the pattern stage's and the join's part alone, a silently
        partial derivative, so it is refused before any work.
        """

        for name in ("position", "look_at", "up"):
            carrier = _carries_derivative(getattr(self, name))
            if carrier is None:
                continue
            wanted = "vjp" if carrier == "a gradient" else "jvp"
            if grad != wanted:
                raise RuntimeError(
                    f"Radar.{name} carries {carrier}, which moves every element position, and the legs carry "
                    f"that derivative only when the verb asks Channel for it: pass grad={wanted!r} (got "
                    f"grad={grad!r})."
                )

    def _transmit_amplitude_scale(self, *, device) -> torch.Tensor | None:
        """``sqrt(P / P_host)`` as a 0-dim tensor when ``power`` is a leaf, else ``None``.

        Channel launches the transmit leg with the HOST power (endpoint powers
        are primal-only there), so the field it returns is exactly the field at
        the authored power. The leaf enters as this factor, whose value is
        exactly one - ``10^((p - p) / 20)`` - and whose derivative is
        ``ln(10) / 20`` per dB: the amplitude of every transmitted wave is
        ``sqrt(P)``. The sensor stage folds it into the transmit element's
        peak and the leakage rows take it too, so every row the transmitter
        launches carries it exactly once.
        """

        if not isinstance(self.power, torch.Tensor):
            return None
        power = self.power.to(device=device, dtype=torch.float32)
        return torch.pow(10.0, (power - power.detach()) / 20.0)

    # -- derived reads -----------------------------------------------------

    @property
    def num_tx(self) -> int:
        """Transmit element count, which is just how many ``tx`` names."""

        return len(self.tx)

    @property
    def num_rx(self) -> int:
        """Receive element count, which is just how many ``rx`` names."""

        return len(self.rx)

    @property
    def wavelength(self) -> float:
        """``c0 / carrier``, m."""

        return SPEED_OF_LIGHT_M_PER_S / float(self.carrier)

    @property
    def transmit_power_watts(self) -> float:
        """``power`` in watts, which is what a source endpoint's field takes."""

        return watts_from_dbm(_host_scalar(self.power))

    def waveform_spec(self, *, offset: float = 0.0):
        """The SI synthesis spec this radar's waveform and array describe."""

        return self.system_config.waveform_spec(carrier_hz=offset)

    # -- rebuilding --------------------------------------------------------

    def replace(self, **fields: Any) -> Radar:
        """A new radar with these fields changed. Nothing here is mutated."""

        unknown = sorted(set(fields) - {f.name for f in self.__dataclass_fields__.values() if f.init})
        if unknown:
            raise TypeError(f"Radar.replace got unknown fields: {', '.join(unknown)}")
        return replace(self, **fields)

    def to(self, device: Any) -> Radar:
        """A new radar on another device."""

        return self.replace(device=device)

    # -- loaders -----------------------------------------------------------

    @classmethod
    def from_dict(cls, config: Mapping[str, Any], **overrides: Any) -> Radar:
        """Build from the flat configuration format (its JSON Schema is :func:`config_schema`).

        The waveform is either the TI-style FMCW keys - the only place vendor
        units are read: ``slope`` in MHz/us, ``sample_rate`` in kSPS, the three
        timings in microseconds - or a ``waveform`` block of the waveform
        record's SI fields. ``fc`` is in Hz, ``power`` in dBm and
        ``tx_loc``/``rx_loc`` in ``antenna_unit`` (half wavelengths unless the
        mapping says ``"m"``); every other radar field is a key of its own
        name, a receive-chain record as a block of its fields and a complex
        number as ``{"re": x, "im": y}``. Keyword overrides are applied
        afterwards and use the SI field names; ``device`` is always a keyword.
        """

        return _radar_from_flat_config(config, overrides)

    @classmethod
    def from_json(cls, path: str | os.PathLike[str], **overrides: Any) -> Radar:
        with open(path, encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle), **overrides)

    def to_dict(self) -> dict[str, Any]:
        """This radar as the flat configuration :meth:`from_dict` reads, JSON-ready.

        The waveform is written as the SI ``waveform`` block, because SI values
        round-trip exactly and vendor units do not, and fields at their default
        are left out. ``Radar.from_dict(radar.to_dict(), device=radar.device)``
        equals ``radar``. The device is not written: it is where a radar runs,
        not what it is. A pose given as a tensor is written as its values.
        """

        return _radar_to_flat_config(self)

    def to_json(self, path: str | os.PathLike[str]) -> None:
        """Write :meth:`to_dict` to ``path`` as JSON; :meth:`from_json` reads it back."""

        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
            handle.write("\n")

    # -- pose transforms, shared by every consumer -------------------------

    def _world_from_local_matrix(self, *, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        position = vec3_tensor(self.position, name="Radar.position").to(device=device, dtype=dtype)
        look_at = vec3_tensor(self.look_at, name="Radar.look_at").to(device=device, dtype=dtype)
        up = vec3_tensor(self.up, name="Radar.up").to(device=device, dtype=dtype)
        forward = look_at - position
        forward = forward / torch.linalg.norm(forward)
        right = torch.cross(forward, up, dim=0)
        right = right / torch.linalg.norm(right)
        true_up = torch.cross(right, forward, dim=0)
        true_up = true_up / torch.linalg.norm(true_up)
        return position, torch.stack((right, true_up, -forward), dim=1)

    # -- instrument half ---------------------------------------------------

    def _apply_signal_models(
        self, signal: torch.Tensor, *, phase_in_signal: bool = False, frame_time_s: float = 0.0
    ) -> torch.Tensor:
        """Run the receive chain, if one is configured, keyed by the frame's instant."""

        if self.frontend is None:
            return signal
        return self.frontend.apply(signal, phase_in_signal=phase_in_signal, frame_time_s=frame_time_s).signal

    def _synthesize(self, paths: RadarPathBatch, *, slow_time_mode, spec=None) -> SynthesisResult:
        """Synthesize one frame with whichever waveform this radar declares.

        Dispatch is a dict lookup on the STORED ``waveform.kind``. It is not a
        ``try``/``except``, not a capability probe, and not an inference from a
        ``slope``: ``RadarSystemConfig`` refused every kind without an owner
        at construction, so the lookup cannot miss.

        ``slow_time_mode`` has no default for the reason it has none anywhere
        else: only the caller knows whether it froze the weight for the frame or
        refreshes it per slot, and defaulting it makes the collision a silent
        wrong answer instead of a refusal.
        """

        from .synthesis import SynthesisPathBatch, SynthesisResult
        from .synthesis.fmcw import synthesize_fmcw
        from .synthesis.ofdm import synthesize_ofdm
        from .synthesis.pulsed import synthesize_pulsed

        owners = {
            WAVEFORM_FMCW: (synthesize_fmcw, SynthesisResult.from_fmcw),
            WAVEFORM_OFDM: (synthesize_ofdm, SynthesisResult.from_ofdm),
            WAVEFORM_PULSED: (synthesize_pulsed, SynthesisResult.from_pulsed),
        }
        batch = SynthesisPathBatch.from_radar_paths(paths, slow_time_mode=slow_time_mode)
        synthesize, build_result = owners[self.system_config.kind]
        spec = self.system_config.waveform_spec() if spec is None else spec
        cube = synthesize(batch, spec)
        return build_result(cube + self._leakage_cube(spec, batch.sensor_pair_count, cube), spec)

    def _leakage_cube(self, spec, sensor_pair_count: int, cube: torch.Tensor):
        """The coupled transmit signal, synthesized by the waveform's own kernel.

        Zero when no coupling is declared, which is the default. One row per
        sensor pair carries that pair's coupling delay and complex coupling
        coefficient, and the same row entry that produced the echo produces
        this, so the phase law has one owner rather than a second copy in
        Torch. The coefficient is in Channel's convention; FMCW consumes the
        conjugated beat weight like every other row.
        """

        if self.leakage is None:
            return cube.new_zeros(())
        device = cube.device
        rows = int(sensor_pair_count)
        delays, weights = self.leakage.pair_rows(num_tx=len(self.tx), num_rx=len(self.rx), power_dbm=self.power)
        if len(delays) != rows:
            raise ValueError(f"leakage describes {len(delays)} sensor pairs but this cube has {rows}")
        delay = torch.tensor(delays, dtype=torch.float32, device=device)
        weight = torch.tensor(weights, dtype=torch.complex64, device=device)
        scale = self._transmit_amplitude_scale(device=device)
        if scale is not None:
            # The coupled wave is a transmitted wave: its amplitude is sqrt(P) too.
            weight = weight * scale
        offsets = torch.arange(rows + 1, dtype=torch.int64, device=device)
        kind = self.system_config.kind
        if kind == WAVEFORM_FMCW:
            from .synthesis.fmcw import synthesize_fmcw_rows

            return synthesize_fmcw_rows(
                delay,
                None,
                weight.conj(),
                offsets,
                spec,
                segment_tx_index=pair_tx_index(
                    num_tx=len(self.tx), num_rx=len(self.rx), sensor_pair_count=rows, device=device
                ),
            )
        if kind == WAVEFORM_OFDM:
            from .synthesis.ofdm import synthesize_cfr_rows, synthesize_ofdm_time_rows

            if spec.time_domain:
                return synthesize_ofdm_time_rows(delay, None, weight, offsets, spec)
            return synthesize_cfr_rows(delay, None, weight, offsets, spec)
        from .synthesis.pulsed import synthesize_echo_rows

        return synthesize_echo_rows(delay, None, weight, offsets, spec)

    # -- entry points ------------------------------------------------------

    def trace(
        self,
        scene,
        targets: PointTargets | StructureTargets | MeshTargets,
        *,
        times,
        los: bool = True,
        reflections: int = 1,
        transmission: bool = False,
        diffraction: bool = False,
        environment: bool = False,
        coupling: bool = False,
        discovery: str = "auto",
        max_paths: int | None = None,
        max_round_trips: int | None = None,
        wideband: bool = False,
        motion: Motion | None = None,
        grad: str = "none",
        endpoints=None,
        pose=None,
    ) -> Paths:
        """Compose this radar's round trips over a Core world and keep them.

        The world half of the pipeline: sampling the world at the waveform's
        observation instants, compiling or reusing the Channel epoch,
        discovering the topology, joining the two legs through the scatter
        response, and weighting by the array's pattern. It stops there. The
        waveform, the receive chain and the output domain belong to
        :meth:`echo`, which is what lets a receiver sweep re-use one trace.

        Arguments are :meth:`simulate`'s. What differs is the retention:
        :class:`~witwin.radar.simulation.Paths` holds every evaluated
        observation's rows, so an ADC-refreshed sequence costs real device
        memory. Read that record's docstring before tracing a long one.
        """

        from .simulation import trace_scene

        session = self._session(
            targets,
            times,
            los,
            reflections,
            motion,
            grad,
            endpoints,
            transmission=transmission,
            diffraction=diffraction,
            environment=environment,
            coupling=coupling,
            discovery=discovery,
            max_paths=max_paths,
            max_round_trips=max_round_trips,
            wideband=wideband,
            pose=pose,
        )
        return trace_scene(self, scene, **session)

    def echo(self, paths: Paths) -> Result:
        """Run this radar's instrument half over already-composed rows.

        Synthesizes the waveform at each traced observation, applies the
        receive chain, lands the frame in the declared output domain and
        assembles the typed result. ``radar.echo(radar.trace(...))`` is
        bit-identical to ``radar.simulate(...)``.

        The radar echoing does not have to be the one that traced. Its receive
        chain, its seed and its FMCW output domain are free; anything the
        observation schedule was derived from is not, and is refused by name
        rather than replayed against a schedule that no longer describes it.
        """

        from .simulation import echo_paths

        return echo_paths(self, paths)

    def simulate(
        self,
        scene,
        targets: PointTargets | StructureTargets | MeshTargets,
        *,
        times,
        los: bool = True,
        reflections: int = 1,
        transmission: bool = False,
        diffraction: bool = False,
        environment: bool = False,
        coupling: bool = False,
        discovery: str = "auto",
        max_paths: int | None = None,
        max_round_trips: int | None = None,
        wideband: bool = False,
        motion: Motion | None = None,
        grad: str = "none",
        endpoints=None,
        pose=None,
    ) -> Result:
        """Simulate this radar over a Core world and return the frame cubes.

        ``scene`` is a ``witwin.core.Scene`` or a
        ``witwin.core.dynamics.DynamicScene``; ``times`` is the sequence of
        frame instants in seconds; ``targets`` names where the scatterers are
        and how strongly they scatter, and it is required because every default
        for it would be an unchosen statement about the world.

        ``los`` and ``reflections`` are the propagation request for THIS call
        and do not edit the radar. ``transmission`` adds paths through walls on
        the straight segment and ``diffraction`` edge-diffracted paths; Channel
        evaluates both only at discovery, so they require ``grad="none"`` and a
        discovery at every evaluated instant (the default). ``environment``
        adds site-less transmitter-to-receiver rows for the scene's own
        returns (reflections, diffraction), and ``coupling`` the elements'
        straight-segment coupling; both are off by default, and coupling is
        refused for co-located transmit and receive elements, where free-space
        transport is undefined. ``discovery`` is Channel's reflection candidate
        source (``"auto"``, ``"exhaustive"`` or ``"rays"``; anything but a
        complete image tree clears ``path_set_complete``), ``max_paths`` bounds
        one discovery's rows (Channel refuses, never truncates), and
        ``max_round_trips`` bounds one sensor pair's composed rows, refused by
        name before the join. ``wideband`` evaluates each OFDM subcarrier's
        transport (spreading, material and layer response) instead of shifting
        the reference coefficient; it replays every frozen row at every
        subcarrier and is refused for the other waveforms. ``motion`` selects
        how often the world is
        resampled inside a frame and defaults to
        :meth:`~witwin.radar.motion.Motion.auto`. ``grad`` is ``"none"``,
        ``"vjp"`` or ``"jvp"``.

        ``pose`` is the radar's ego-motion, ``pose(t) -> (rotation, translation)``:
        a ``(3, 3)`` local-to-world rotation composed on the left of the
        authored orientation, about the radar's authored position, and a
        ``(3,)`` world translation in metres. The elements' positions, their
        polarizations and the antenna pattern's orientation all follow it, at
        every observation instant. ``None`` holds the authored pose.

        The whole assembly lives in :mod:`witwin.radar.simulation` and its
        docstring is the contract; this method delegates rather than
        reimplementing so there is one owner of the frame loop.
        """

        from .simulation import simulate_scene

        session = self._session(
            targets,
            times,
            los,
            reflections,
            motion,
            grad,
            endpoints,
            transmission=transmission,
            diffraction=diffraction,
            environment=environment,
            coupling=coupling,
            discovery=discovery,
            max_paths=max_paths,
            max_round_trips=max_round_trips,
            wideband=wideband,
            pose=pose,
        )
        return simulate_scene(self, scene, **session)

    def stream(
        self,
        scene,
        targets: PointTargets | StructureTargets | MeshTargets,
        *,
        times,
        los: bool = True,
        reflections: int = 1,
        transmission: bool = False,
        diffraction: bool = False,
        environment: bool = False,
        coupling: bool = False,
        discovery: str = "auto",
        max_paths: int | None = None,
        max_round_trips: int | None = None,
        wideband: bool = False,
        motion: Motion | None = None,
        grad: str = "none",
        endpoints=None,
        pose=None,
    ) -> Iterator[Result]:
        """Simulate the same session as :meth:`simulate`, one frame at a time.

        Yields a one-frame result per instant in ``times``, so a sequence long
        enough to exhaust device memory as a single stacked cube can still be
        produced and consumed. The physics, the session state and the per-frame
        cubes are the same; only the retention differs, and a caller that keeps
        every yielded result has spent more memory than :meth:`simulate` would
        have, not less. Arguments are validated when iteration starts rather
        than when this returns, because this is a generator.
        """

        from .simulation import stream_scene

        session = self._session(
            targets,
            times,
            los,
            reflections,
            motion,
            grad,
            endpoints,
            transmission=transmission,
            diffraction=diffraction,
            environment=environment,
            coupling=coupling,
            discovery=discovery,
            max_paths=max_paths,
            max_round_trips=max_round_trips,
            wideband=wideband,
            pose=pose,
        )
        yield from stream_scene(self, scene, **session)

    def _session(
        self,
        targets,
        times,
        los,
        reflections,
        motion,
        grad,
        endpoints,
        *,
        transmission: bool = False,
        diffraction: bool = False,
        environment: bool = False,
        coupling: bool = False,
        discovery: str = "auto",
        max_paths: int | None = None,
        max_round_trips: int | None = None,
        wideband: bool = False,
        pose=None,
    ) -> dict[str, Any]:
        """Turn the public call arguments into what the frame loop takes."""

        from .motion import Motion
        from .targets import as_session_targets

        self._require_pose_ad_mode(grad)
        if not isinstance(reflections, int) or isinstance(reflections, bool) or reflections < 0:
            raise ValueError(f"reflections must be a non-negative int, got {reflections!r}")
        for name, value in (("transmission", transmission), ("diffraction", diffraction)):
            if type(value) is not bool:
                raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
        if not (los or reflections or transmission or diffraction):
            raise ValueError(
                "a solve with neither the line of sight nor any reflection, transmission or diffraction asks for "
                "no propagation at all; set los=True, reflections>0, or both"
            )
        if max_round_trips is not None and (type(max_round_trips) is not int or max_round_trips < 1):
            raise ValueError(f"max_round_trips must be a positive int or None, got {max_round_trips!r}")
        components = set()
        if los:
            components.add("los")
        if reflections > 0:
            components.add("reflection")
        if transmission:
            components.add("transmission")
        if diffraction:
            components.add("diffraction")
        sites, response = as_session_targets(targets, radar=self)
        resolved = Motion.auto() if motion is None else motion
        return {
            "times": times,
            "response": response,
            "sites": sites,
            "components": frozenset(components),
            # Channel's one depth bound also limits how many walls a
            # transmission path crosses and how many edges a diffracted one
            # meets, so those components keep at least one interaction.
            "max_depth": max(int(reflections), 1 if (transmission or diffraction) else 0),
            "ad_mode": grad,
            "polarization": self.polarization_vector,
            "rx_polarization": self.rx_polarization_vector,
            "antenna_pattern": self.system_config.sensors.pattern,
            "sensor_endpoints": endpoints,
            "motion": resolved,
            "environment": environment,
            "coupling": coupling,
            "discovery": discovery,
            "max_paths": max_paths,
            "max_round_trips": max_round_trips,
            "wideband": wideband,
            "pose": pose,
        }


# ---------------------------------------------------------------------------
# The flat configuration file format
# ---------------------------------------------------------------------------

_LEAKAGE_REASON = "antenna coupling is a measured board property, not a quantity a scene differentiates through"

_POLARIZATION_REASON = (
    "a radar polarization is launched by Channel's replay as a primal-only endpoint constant, which publishes no "
    "derivative for it, so a polarization leaf could not be carried."
)

_BASEBAND_REASON = "a DC offset is a measured property of the receiver, not a quantity a scene differentiates through"

#: The value of the optional ``format`` key :meth:`Radar.to_dict` writes.
CONFIG_FORMAT = "witwin.radar.config/1"

#: The TI-style FMCW keys, in vendor units: the hand-written way to declare the waveform.
TI_WAVEFORM_KEYS = (
    "slope",
    "sample_rate",
    "adc_samples",
    "chirp_per_frame",
    "adc_start_time",
    "idle_time",
    "ramp_end_time",
)

_REQUIRED = ("num_tx", "num_rx", "fc", "power", "tx_loc", "rx_loc")

#: Receive-chain and coupling records, each a block of its own field names.
_RECORD_BLOCKS: dict[str, type] = {
    "noise": Noise,
    "agc": Agc,
    "adc": Adc,
    "iq": Iq,
    "compression": Compression,
    "coupling": Coupling,
}

#: Radar fields written as plain values (numbers, strings, vectors, complex numbers).
_PLAIN_FIELDS = (
    "antenna_unit",
    "lna_gain",
    "dc_offset",
    "impedance",
    "seed",
    "position",
    "look_at",
    "up",
    "polarization",
    "rx_polarization",
)

#: One row per top-level key: (key, JSON Schema type, unit, meaning). :func:`config_schema` is
#: generated from it, and the reader refuses any key not here.
CONFIG_FIELDS: tuple[tuple[str, Any, str, str], ...] = (
    ("format", "string", "", f"optional format tag; {CONFIG_FORMAT!r} when written by Radar.to_dict"),
    ("num_tx", "integer", "", "transmit element count; must equal len(tx_loc)"),
    ("num_rx", "integer", "", "receive element count; must equal len(rx_loc)"),
    ("fc", "number", "Hz", "carrier (reference frequency)"),
    ("power", "number", "dBm", "transmit power"),
    (
        "tx_loc",
        "array",
        "antenna_unit",
        "transmit element positions, [x, y, z] each, radar-local (right, up, -boresight)",
    ),
    ("rx_loc", "array", "antenna_unit", "receive element positions, as tx_loc"),
    ("antenna_unit", "string", "", "'half_wavelength' (default) or 'm'"),
    ("slope", "number", "MHz/us", "TI-style FMCW: ramp slope"),
    ("sample_rate", "number", "ksps", "TI-style FMCW: ADC sample rate"),
    ("adc_samples", "integer", "", "TI-style FMCW: samples per chirp"),
    ("chirp_per_frame", "integer", "", "TI-style FMCW: chirps per frame per transmitter"),
    ("adc_start_time", "number", "us", "TI-style FMCW: ramp start to first ADC sample"),
    ("idle_time", "number", "us", "TI-style FMCW: idle time between ramps"),
    ("ramp_end_time", "number", "us", "TI-style FMCW: ramp end time"),
    ("output_domain", "string", "", "TI-style FMCW: 'spectrum' (default) or 'beat'"),
    ("waveform", "object", "SI", "{'kind': 'fmcw'|'ofdm'|'pulsed', <the waveform record's fields>}"),
    ("antenna_pattern", "object", "deg, dBi", "transmit (and default receive) element pattern"),
    ("rx_antenna_pattern", "object", "deg, dBi", "receive element pattern; absent is antenna_pattern"),
    ("noise", "object", "SI", "Noise record fields"),
    ("lna_gain", "number", "dB", "LNA voltage gain"),
    ("agc", "object", "SI", "Agc record fields"),
    ("adc", "object", "SI", "Adc record fields"),
    ("iq", "object", "dB, deg", "Iq record fields"),
    ("leakage", "object", "dB, s, rad", "Leakage record fields; scalars or [num_tx][num_rx] tables"),
    ("compression", "object", "dBm, deg", "Compression record fields"),
    ("dc_offset", ["number", "object", "array"], "V", "one complex value or one per receiver"),
    ("coupling", "object", "", "Coupling record: tx and/or rx complex matrices"),
    ("impedance", "number", "ohm", "receive port reference impedance"),
    ("seed", "integer", "", "receive-chain base seed"),
    ("position", "array", "m", "radar origin, world frame"),
    ("look_at", "array", "m", "boresight target point, world frame"),
    ("up", "array", "", "world up vector completing the pose"),
    ("polarization", ["string", "array"], "", "'up', 'right' or a world vector"),
    ("rx_polarization", ["string", "array", "null"], "", "receive polarization; absent is polarization"),
)

_KNOWN_KEYS = frozenset(row[0] for row in CONFIG_FIELDS)


# Value codec: JSON lists and {"re", "im"} objects <-> tuples and complex numbers.


def _encode(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.requires_grad:
            raise ValueError("a configuration value carrying a gradient is scene state, not configuration")
        return _encode(value.detach().cpu().tolist())
    if isinstance(value, complex):
        return {"re": float(value.real), "im": float(value.imag)}
    if isinstance(value, (tuple, list)):
        return [_encode(entry) for entry in value]
    if isinstance(value, bool) or value is None or isinstance(value, (int, float, str)):
        return value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _record_to_dict(value)
    raise TypeError(f"{type(value).__name__} has no configuration form")


def _decode(value: Any) -> Any:
    if isinstance(value, Mapping) and set(value) == {"re", "im"}:
        return complex(float(value["re"]), float(value["im"]))
    if isinstance(value, list):
        return tuple(_decode(entry) for entry in value)
    return value


def _record_to_dict(record: Any) -> dict[str, Any]:
    """Every init field of a frozen record whose value differs from its default."""

    out: dict[str, Any] = {}
    for entry in dataclasses.fields(record):
        if not entry.init:
            continue
        value = getattr(record, entry.name)
        if entry.default is not dataclasses.MISSING and value == entry.default:
            continue
        out[entry.name] = _encode(value)
    return out


def _record_from_dict(cls: type, block: Any, *, key: str) -> Any:
    if not isinstance(block, Mapping):
        raise ValueError(f"{key} must be a mapping of {cls.__name__} fields, got {block!r}")
    names = {entry.name for entry in dataclasses.fields(cls) if entry.init}
    unknown = sorted(set(block) - names)
    if unknown:
        raise ValueError(f"{key} has unsupported keys: {', '.join(unknown)}; {cls.__name__} has {sorted(names)}")
    return cls(**{name: _decode(value) for name, value in block.items()})


def _pattern_to_dict(pattern: Pattern) -> dict[str, Any]:
    out: dict[str, Any] = {"kind": pattern.kind, "x_angles_deg": list(pattern.x_angles)}
    out["y_angles_deg"] = list(pattern.y_angles)
    if pattern.kind == "separable":
        out["x_values"] = _encode(pattern.x_gain)
        out["y_values"] = _encode(pattern.y_gain)
    else:
        out["values"] = _encode(pattern.gain_table)
    if pattern.quantity != "power":
        out["quantity"] = pattern.quantity
    if pattern.gain != 0.0:
        out["gain"] = float(pattern.gain)
    return out


def _pattern_from_dict(config: Mapping[str, Any]) -> Pattern:
    """Read an element-pattern block: ``kind``, the angle axes in degrees, the values, ``quantity`` and ``gain``."""

    if not isinstance(config, Mapping):
        raise ValueError(f"an antenna pattern must be a mapping, got {config!r}")
    kind = config.get("kind")
    if kind is None:
        kind = "map" if "values" in config else "separable"
    quantity = config.get("quantity", "power")
    gain = float(config.get("gain", 0.0))
    is_field = quantity == "field"
    if quantity not in ("power", "field"):
        raise ValueError(f"antenna_pattern quantity must be 'power' or 'field', got {quantity!r}")
    if kind == "separable":
        build = Pattern.field_separable if is_field else Pattern.separable
        x_values, y_values = _decode(config["x_values"]), _decode(config["y_values"])
        return build(config["x_angles_deg"], x_values, config["y_angles_deg"], y_values, gain=gain)
    if kind == "map":
        build = Pattern.field_table if is_field else Pattern.table
        return build(config["x_angles_deg"], config["y_angles_deg"], _decode(config["values"]), gain=gain)
    raise ValueError(f"antenna_pattern kind must be 'separable' or 'map', got {kind!r}")


def _waveform_classes() -> dict[str, type]:
    return {WAVEFORM_FMCW: Fmcw, WAVEFORM_OFDM: Ofdm, WAVEFORM_PULSED: Pulsed}


def _waveform_to_dict(waveform: Any) -> dict[str, Any]:
    return {"kind": waveform.kind, **_record_to_dict(waveform)}


def _waveform_from_dict(block: Any) -> Any:
    if not isinstance(block, Mapping) or "kind" not in block:
        raise ValueError(f"waveform must be a mapping with a 'kind' of fmcw, ofdm or pulsed, got {block!r}")
    classes = _waveform_classes()
    kind = block["kind"]
    if kind not in classes:
        raise ValueError(f"waveform kind must be one of {sorted(classes)}, got {kind!r}")
    fields = {key: value for key, value in block.items() if key != "kind"}
    sequence = fields.pop("sequence", None)
    waveform = _record_from_dict(classes[kind], fields, key=f"waveform ({kind})")
    if sequence is not None:
        if kind != "fmcw":
            raise ValueError("waveform.sequence is a list of FMCW Chirp profiles; only an fmcw waveform has one")
        chirps = tuple(_record_from_dict(Chirp, entry, key="waveform.sequence[]") for entry in sequence)
        waveform = dataclasses.replace(waveform, sequence=chirps)
    return waveform


def _ti_waveform(config: Mapping[str, Any]) -> Fmcw:
    return Fmcw.from_ti(
        slope_mhz_per_us=_finite("slope", config["slope"]),
        sample_rate_ksps=_positive("sample_rate", config["sample_rate"]),
        samples_per_chirp=_positive_int("adc_samples", config["adc_samples"]),
        chirps_per_frame=_positive_int("chirp_per_frame", config["chirp_per_frame"]),
        adc_start_us=_finite("adc_start_time", config["adc_start_time"]),
        idle_us=_non_negative("idle_time", config["idle_time"]),
        ramp_end_us=_positive("ramp_end_time", config["ramp_end_time"]),
        output=str(config.get("output_domain", "spectrum")),
    )


def _radar_from_flat_config(config: Mapping[str, Any], overrides: Mapping[str, Any]) -> Radar:
    """Build a radar from the flat configuration; ``overrides`` are SI :class:`Radar` fields applied last."""

    if not isinstance(config, Mapping):
        raise TypeError(f"a radar configuration is a mapping, got {type(config).__name__}")
    unknown = sorted(set(config) - _KNOWN_KEYS)
    if unknown:
        raise ValueError(
            f"Radar config has unsupported keys: {', '.join(unknown)}. The flat configuration accepts "
            f"{', '.join(sorted(_KNOWN_KEYS))}; see witwin.radar.config.config_schema(). A device is a keyword."
        )
    ti_keys = [key for key in (*TI_WAVEFORM_KEYS, "output_domain") if key in config]
    if "waveform" in config and ti_keys:
        raise ValueError(
            f"the waveform is declared twice: a 'waveform' block and the TI-style FMCW keys {ti_keys}. "
            "Give one: the TI-style keys in vendor units, or the waveform block in SI units"
        )
    required = _REQUIRED if "waveform" in config else (*_REQUIRED, *TI_WAVEFORM_KEYS)
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Radar config is missing required keys: {', '.join(missing)}")
    if config.get("format", CONFIG_FORMAT) != CONFIG_FORMAT:
        raise ValueError(f"Radar config format {config['format']!r} is not {CONFIG_FORMAT!r}")

    num_tx = _positive_int("num_tx", config["num_tx"])
    num_rx = _positive_int("num_rx", config["num_rx"])
    tx = _elements(config["tx_loc"], name="tx_loc")
    rx = _elements(config["rx_loc"], name="rx_loc")
    if len(tx) != num_tx:
        raise ValueError(f"tx_loc holds {len(tx)} entries but num_tx is {num_tx}")
    if len(rx) != num_rx:
        raise ValueError(f"rx_loc holds {len(rx)} entries but num_rx is {num_rx}")

    fields: dict[str, Any] = {
        "carrier": _positive("fc", config["fc"]),
        "waveform": _waveform_from_dict(config["waveform"]) if "waveform" in config else _ti_waveform(config),
        "tx": tx,
        "rx": rx,
        "antenna_unit": "half_wavelength",
        "power": _finite("power", config["power"]),
    }
    if config.get("antenna_pattern") is not None:
        fields["pattern"] = _pattern_from_dict(config["antenna_pattern"])
    if config.get("rx_antenna_pattern") is not None:
        fields["rx_pattern"] = _pattern_from_dict(config["rx_antenna_pattern"])
    for key, cls in _RECORD_BLOCKS.items():
        if config.get(key) is not None:
            fields[key] = _record_from_dict(cls, config[key], key=key)
    if config.get("leakage") is not None:
        fields["leakage"] = _record_from_dict(Leakage, config["leakage"], key="leakage")
    for key in _PLAIN_FIELDS:
        if key in config:
            fields[key] = _decode(config[key])
    fields.update(overrides)
    return Radar(**fields)


def _radar_to_flat_config(radar: Radar) -> dict[str, Any]:
    """The flat configuration of ``radar``: JSON-ready, SI waveform block, defaults omitted."""

    defaults = {entry.name: entry for entry in dataclasses.fields(Radar)}
    out: dict[str, Any] = {
        "format": CONFIG_FORMAT,
        "num_tx": radar.num_tx,
        "num_rx": radar.num_rx,
        "fc": float(radar.carrier),
        "power": float(radar.power),
        "antenna_unit": radar.antenna_unit,
        "tx_loc": _encode(radar.tx),
        "rx_loc": _encode(radar.rx),
        "waveform": _waveform_to_dict(radar.waveform),
    }
    if radar.pattern != Pattern.isotropic():
        out["antenna_pattern"] = _pattern_to_dict(radar.pattern)
    if radar.rx_pattern is not None:
        out["rx_antenna_pattern"] = _pattern_to_dict(radar.rx_pattern)
    for key in (*_RECORD_BLOCKS, "leakage"):
        record = getattr(radar, key)
        if record is not None:
            out[key] = _record_to_dict(record)
    for key in _PLAIN_FIELDS:
        value = getattr(radar, key)
        default = defaults[key].default
        if key == "antenna_unit" or (default is not dataclasses.MISSING and _same(value, default)):
            continue
        out[key] = _encode(value)
    return out


def _same(value: Any, default: Any) -> bool:
    if isinstance(value, torch.Tensor) or isinstance(default, torch.Tensor):
        return False
    try:
        return bool(_encode(value) == _encode(default))
    except TypeError:
        return False


def config_schema() -> dict[str, Any]:
    """The JSON Schema (draft 2020-12) of the flat configuration, generated from :data:`CONFIG_FIELDS`.

    The receive-chain and waveform blocks list their record's field names, so a
    field added to a record appears here without an edit.
    """

    properties: dict[str, Any] = {}
    for key, kind, unit, meaning in CONFIG_FIELDS:
        entry: dict[str, Any] = {"type": kind, "description": meaning + (f" [{unit}]" if unit else "")}
        cls = Leakage if key == "leakage" else _RECORD_BLOCKS.get(key)
        if cls is not None:
            entry["properties"] = {entry.name: {} for entry in dataclasses.fields(cls) if entry.init}
            entry["additionalProperties"] = False
        properties[key] = entry
    waveform_variants = []
    for kind, cls in _waveform_classes().items():
        names = {entry.name: {} for entry in dataclasses.fields(cls) if entry.init}
        names["kind"] = {"const": kind}
        waveform_variants.append({"properties": names, "required": ["kind"], "additionalProperties": False})
    properties["waveform"]["oneOf"] = waveform_variants
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": CONFIG_FORMAT,
        "title": "witwin-radar flat configuration",
        "type": "object",
        "properties": properties,
        "required": list(_REQUIRED),
        "oneOf": [{"required": ["waveform"]}, {"required": list(TI_WAVEFORM_KEYS)}],
        "additionalProperties": False,
    }


__all__ = ["Chirp", "Fmcw", "Ofdm", "Pulsed", "Radar", "config_schema"]
