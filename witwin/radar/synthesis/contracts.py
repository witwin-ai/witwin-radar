"""The input contract every waveform synthesis kernel consumes, and the
waveform descriptions that go with it.

This module is pure and CPU-testable on purpose: the unit conversions between
the radar config's engineering units and SI are exactly the kind of thing that
is wrong once and then wrong everywhere, and they should not require a GPU to
check. The same is true of the provenance rules below, which decide whether a
weight and a waveform spec may be used together at all.

Two contracts live here and they are not the same statement:

* :class:`~witwin.radar.paths.contracts.RadarPathBatch` is what the two-way
  composer PRODUCED.
* :class:`SynthesisPathBatch` is what a waveform kernel is ALLOWED TO ASSUME
  about a weight.

The difference is provenance. Every double-count hazard the Phase-6 physics
survey found is a combination of a weight and a spec that nobody validates
against each other: a Channel coefficient already carries
``exp(-j 2 pi f_ref tau_rt)``, ``lambda/(4 pi d)`` per leg, and
``sqrt(tx_power)``, so a kernel that applies any of them again is silently
wrong by a factor nobody notices. Recording that on the batch and validating
the spec against it at construction turns six documented hazards into four
impossible states, which is the difference between a rule and a comment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Protocol, runtime_checkable

import torch

from ..paths.contracts import JOIN_MODES, JoinMode, RadarPathBatch, RadarPathTopology


#: Exact SI definition, in metres per second. Named here because the FMCW spec
#: derives its unambiguous-velocity bound from a wavelength and must agree with
#: ``Radar.max_doppler`` to the last bit.
SPEED_OF_LIGHT_M_PER_S = 299792458.0

#: Channel's phasor convention, which the OFDM CFR cube is published in
#: unchanged. Quoted verbatim from ``witwin.channel.constants.PHASOR``.
CHANNEL_PHASOR = "exp(-j*k*d)"

#: Channel's time-dependence convention, quoted verbatim from
#: ``witwin.channel.constants.TIME_DEPENDENCE``.
CHANNEL_TIME_DEPENDENCE = "exp(+j*2*pi*f*t)"

#: The FMCW beat cube's convention, which is the CONJUGATE of Channel's.
#: De-chirping multiplies the echo by the conjugate of the transmitted chirp, so
#: the beat-domain phasor advances with ``+j`` while a Channel transport advances
#: with ``-j``. Both are correct and they are different products; naming the beat
#: one here is what lets a result carry its convention as data instead of a
#: reader inferring it from the waveform's name.
BEAT_PHASOR = "conj(exp(-j*k*d))"



def require_single_carrier_home(carrier_hz: float, carrier_rate_hz: float) -> None:
    """Refuse a spec that names the absolute carrier in both of its homes.

    Shared by every waveform spec, because it is one rule about one physical
    quantity rather than a per-waveform convention: the absolute
    reference-frequency phase either lives in the weight, in which case the
    kernel applies it only to the delay CHANGE (``carrier_rate_hz``), or it
    lives in the kernel, which applies it to the full delay (``carrier_hz``)
    and therefore already walks across slow time. Naming both counts the
    carrier twice.

    This is a spec-INTERNAL consistency check. The batch-versus-spec checks are
    rules R1-R4 of :func:`require_compatible`, and neither subsumes the other:
    this one catches a self-contradictory spec before a batch is even in hand.
    """

    if carrier_hz != 0.0 and carrier_rate_hz != 0.0:
        raise ValueError(
            "carrier_hz and carrier_rate_hz name the same carrier in two "
            "different homes; setting both double counts it. Use "
            "carrier_hz=fc with carrier_rate_hz=0 when the kernel owns the "
            "carrier phase, or carrier_hz=0 with carrier_rate_hz=fc when a "
            "Channel-sourced weight already carries it."
        )


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

    ``num_tx`` and ``num_rx`` describe the TDM-MIMO array. They belong on the
    spec rather than on the batch because they are a property of the WAVEFORM's
    time structure: TDM fires the transmitters sequentially, so the slow-time
    coordinate of a sensor pair is its slot ``chirp * num_tx + tx``, not the
    chirp index. ``num_tx = 1`` is the degenerate single-transmitter case where
    slot and chirp coincide.
    """

    #: The beat kernel has no ``lambda / (4 pi d)`` term at all: free-space
    #: spreading is Channel transport's, per leg, once. This is a statement
    #: about the kernel and not a setting, so it is a class attribute rather
    #: than a field nobody may change.
    applies_spreading: ClassVar[bool] = False

    #: The convention the published cube is in, carried as data so that a
    #: consumer never has to infer it from the waveform's name. This is the ONE
    #: waveform whose product is conjugated relative to Channel's, and stating
    #: it here is what makes a cross-waveform phase comparison writable.
    phasor: ClassVar[str] = BEAT_PHASOR
    time_dependence: ClassVar[str] = CHANNEL_TIME_DEPENDENCE

    num_samples: int
    num_chirps: int
    sample_period_s: float
    chirp_period_s: float
    slope_hz_per_s: float
    t_start_s: float
    reference_frequency_hz: float
    carrier_hz: float = 0.0
    carrier_rate_hz: float = 0.0
    num_tx: int = 1
    num_rx: int = 1

    def __post_init__(self) -> None:
        if self.num_samples < 1:
            raise ValueError("num_samples must be positive")
        if self.num_chirps < 1:
            raise ValueError("num_chirps must be positive")
        if self.sample_period_s <= 0.0:
            raise ValueError("sample_period_s must be positive")
        if self.chirp_period_s <= 0.0:
            raise ValueError("chirp_period_s must be positive")
        if not self.reference_frequency_hz > 0.0:
            raise ValueError(
                "reference_frequency_hz must be positive; it is the frequency "
                "the weight this spec will consume was evaluated at, and "
                "require_compatible refuses a mismatch"
            )
        if self.num_tx < 1:
            raise ValueError("num_tx must be positive")
        if self.num_rx < 1:
            raise ValueError("num_rx must be positive")
        require_single_carrier_home(self.carrier_hz, self.carrier_rate_hz)

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
            reference_frequency_hz=float(config.fc),
            carrier_hz=carrier,
            carrier_rate_hz=0.0 if carrier != 0.0 else float(config.fc),
            num_tx=int(config.num_tx),
            num_rx=int(config.num_rx),
        )

    @property
    def sample_rate_hz(self) -> float:
        return 1.0 / self.sample_period_s

    @property
    def sensor_pair_count(self) -> int:
        """The TDM-MIMO virtual array size the pair partition must span."""

        return self.num_tx * self.num_rx

    @property
    def wavelength_m(self) -> float:
        return SPEED_OF_LIGHT_M_PER_S / self.reference_frequency_hz

    @property
    def slot_period_s(self) -> float:
        """Slow-time spacing between two chirps of the SAME transmitter.

        With ``num_tx`` transmitters sharing the frame in TDM, a given
        transmitter revisits its slot once every ``num_tx`` chirp periods. This
        is the period the Doppler FFT actually samples at, and it is why TDM
        costs a factor ``num_tx`` of unambiguous velocity.
        """

        return self.chirp_period_s * self.num_tx

    @property
    def max_unambiguous_speed_mps(self) -> float:
        """``lambda / (4 * T_chirp * num_tx)``, the aliasing bound on ``|v_r|``.

        Half a wavelength of two-way path change per slow-time sample is half a
        cycle of Doppler phase; beyond it the sign of the velocity is not
        recoverable.
        """

        return self.wavelength_m / (4.0 * self.slot_period_s)

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


#: The only supported OFDM subcarrier origin: ``f_n = f_ref + n * df`` with
#: ``n`` running ``[0, N_sc)``, so subcarrier 0 sits exactly at the frequency
#: the weight was evaluated at. Centring the band instead would put a
#: half-band phase offset between ``H[0][p][0]`` and ``C_rt`` and force a phase
#: correction into every cross-waveform amplitude comparison.
SUBCARRIER_ORIGIN_F_REF_AT_N0 = "f_ref_at_n0"


@dataclass(frozen=True, slots=True)
class OfdmCfrSpec:
    """One OFDM frame's subcarrier grid and symbol timing, in SI units.

    The product is a channel frequency response cube
    ``H[symbol, sensor_pair, subcarrier]``, not a time-domain waveform. A CFR is
    the exact analogue of the FMCW beat cube: it is what per-subcarrier
    equalisation ``H = Y / X`` leaves after the transmitted symbols are removed,
    it needs no per-sample IFFT inside the kernel, and it is what the Phase-6
    plan names. A time-domain OFDM waveform, if one is ever needed, is a
    downstream IFFT plus cyclic-prefix insertion in DSP glue, not synthesis
    physics.

    **Phasor convention.** This cube is published in the CHANNEL convention
    :data:`CHANNEL_PHASOR`, NOT conjugated. Equalisation removes the transmitted
    symbol but not the carrier convention, so there is nothing to conjugate and
    no conversion site anywhere in the OFDM family. The FMCW beat cube IS
    conjugated, because de-chirping multiplies by the conjugate of the
    transmitted chirp. The two are different products rather than an
    inconsistency, and both carry their convention as data.

    **Subcarrier origin.** ``n = 0`` is pinned to ``reference_frequency_hz``, so
    ``f_n = f_ref + n * df``. With ``carrier_hz = 0``, ``carrier_rate_hz =
    f_ref`` and a stationary row, ``H[0][p][0]`` is exactly the Channel
    coefficient ``C_rt``. That identity is what pinning the origin buys.

    **Narrowband, by construction and on purpose.** Channel's consumer has no
    frequency-offset input: it publishes one coefficient per row at one
    reference frequency plus the narrowband offset law
    ``H(f_ref + df) = C(f_ref) * exp(-j 2 pi df delay_s)``. Phase-6 OFDM applies
    exactly that law, which means the material and antenna response is FROZEN at
    ``f_ref`` across the whole band - only the propagation delay is
    frequency-dependent. A per-subcarrier material response is Phase-8 work;
    :class:`SynthesisPathBatch` declares ``frequency_response`` so that this
    assumption is explicit and rule R8 refuses it so that it cannot be silently
    ignored.

    **Cyclic prefix.** ``max_expected_delay_s`` is a CONFIGURED bound - the
    range window the radar is set up for - and never a measured maximum delay,
    which would be a per-frame device-to-host transfer.
    :func:`require_ofdm_compatible` refuses ``max_expected_delay_s >=
    cyclic_prefix_s`` outright. There is no clamp, no warning, and no
    reduced-accuracy mode, because outside the CP window the single-tap
    per-subcarrier form gains an inter-symbol term this kernel does not have.

    ``carrier_hz`` and ``carrier_rate_hz`` are the same two carrier homes as in
    :class:`FmcwBeatSpec` and obey the same rule through the same helper: the
    absolute phase either lives in the weight, and the kernel applies the
    carrier to the delay CHANGE only, or it lives in the kernel, which applies
    it to the full delay. Naming both counts it twice. Dropping the rate term on
    a frozen Channel weight leaves only the ``n * df`` slow-time phase and
    understates Doppler by ``f_ref / (n * df)`` - a factor of about 1e4 at the
    top of a 64 x 120 kHz band at 77 GHz, and infinite at ``n = 0``.
    """

    #: The CFR kernel has no ``lambda / (4 pi d)`` term at all: free-space
    #: spreading is Channel transport's, per leg, once. A statement about the
    #: kernel rather than a setting, so a class attribute rather than a field.
    applies_spreading: ClassVar[bool] = False

    #: The convention the published cube is in, carried as data so that a
    #: consumer never has to infer it from the waveform's name.
    phasor: ClassVar[str] = CHANNEL_PHASOR
    time_dependence: ClassVar[str] = CHANNEL_TIME_DEPENDENCE

    num_subcarriers: int
    num_symbols: int
    subcarrier_spacing_hz: float
    cyclic_prefix_s: float
    reference_frequency_hz: float
    max_expected_delay_s: float
    carrier_hz: float = 0.0
    carrier_rate_hz: float = 0.0
    subcarrier_origin: str = SUBCARRIER_ORIGIN_F_REF_AT_N0

    def __post_init__(self) -> None:
        if self.num_subcarriers < 1:
            raise ValueError("num_subcarriers must be positive")
        if self.num_symbols < 1:
            raise ValueError("num_symbols must be positive")
        if self.subcarrier_spacing_hz <= 0.0:
            raise ValueError("subcarrier_spacing_hz must be positive")
        if self.cyclic_prefix_s <= 0.0:
            raise ValueError(
                "cyclic_prefix_s must be positive; a zero-length cyclic prefix "
                "cannot contain any echo, so the single-tap per-subcarrier form "
                "this spec describes would never be valid"
            )
        if self.max_expected_delay_s < 0.0:
            raise ValueError("max_expected_delay_s must be non-negative")
        if not self.reference_frequency_hz > 0.0:
            raise ValueError(
                "reference_frequency_hz must be positive; it is the frequency "
                "subcarrier 0 sits at and the frequency the weight this spec "
                "will consume was evaluated at"
            )
        if self.subcarrier_origin != SUBCARRIER_ORIGIN_F_REF_AT_N0:
            raise ValueError(
                "subcarrier_origin must be "
                f"{SUBCARRIER_ORIGIN_F_REF_AT_N0!r}; a centred band is a "
                "different frequency grid and would need its own kernel term, "
                "not a relabelling"
            )
        require_single_carrier_home(self.carrier_hz, self.carrier_rate_hz)

    @property
    def useful_symbol_time_s(self) -> float:
        """``T_u = 1 / df``, the FFT window one symbol is transformed over."""

        return 1.0 / self.subcarrier_spacing_hz

    @property
    def symbol_period_s(self) -> float:
        """``T_sym = T_u + T_cp``, the slow-time sampling period.

        The cyclic prefix does not change the CFR closed form, but it does
        lengthen the symbol, which is why it appears in the unambiguous-velocity
        bound: slow time is sampled once per SYMBOL, prefix included.
        """

        return self.useful_symbol_time_s + self.cyclic_prefix_s

    @property
    def waveform_sample_period_s(self) -> float:
        """``T_s = 1 / (N_sc * df)``, the time grid the CIR lands on."""

        return 1.0 / (self.num_subcarriers * self.subcarrier_spacing_hz)

    @property
    def delay_resolution_s(self) -> float:
        """One CIR sample: the delay resolution the whole band buys.

        Identical to :attr:`waveform_sample_period_s` by construction; both
        names exist because one is a property of the waveform's time grid and
        the other is the estimator's resolution, and reading a range resolution
        off a sample period is exactly the step where a factor of two goes
        missing.
        """

        return self.waveform_sample_period_s

    @property
    def occupied_bandwidth_hz(self) -> float:
        """``N_sc * df``, the band the delay resolution comes from."""

        return self.num_subcarriers * self.subcarrier_spacing_hz

    @property
    def range_resolution_m(self) -> float:
        """``c0 / (2 * N_sc * df)`` - the round trip halves the delay."""

        return SPEED_OF_LIGHT_M_PER_S / (2.0 * self.occupied_bandwidth_hz)

    @property
    def max_unambiguous_delay_s(self) -> float:
        """``1 / df``: beyond one useful symbol time the CIR wraps."""

        return self.useful_symbol_time_s

    @property
    def wavelength_m(self) -> float:
        return SPEED_OF_LIGHT_M_PER_S / self.reference_frequency_hz

    @property
    def max_unambiguous_speed_mps(self) -> float:
        """``c0 / (4 * f_ref * T_sym)``, the aliasing bound on ``|v_r|``.

        Equivalently ``lambda / (4 * T_sym)``. Half a wavelength of two-way path
        change per slow-time sample is half a cycle of Doppler phase; beyond it
        the sign of the velocity is not recoverable and a receding target reads
        as an approaching one.
        """

        return SPEED_OF_LIGHT_M_PER_S / (
            4.0 * self.reference_frequency_hz * self.symbol_period_s
        )

    def subcarrier_frequency_hz(self, subcarrier: int) -> float:
        """``f_n = f_ref + n * df`` under the pinned origin."""

        return self.reference_frequency_hz + subcarrier * self.subcarrier_spacing_hz

    def subcarrier_phase_step_rad(self, round_trip_delay_s: float) -> float:
        """``-2 pi df tau``, the phase step between adjacent subcarriers.

        The sign is negative because the cube is in Channel's ``exp(-j k d)``
        convention. This is the exact, bin-free delay statement a CFR carries;
        an IDFT peak is the same information quantised to ``T_s``.
        """

        return -math.tau * self.subcarrier_spacing_hz * float(round_trip_delay_s)

    def cir_peak_sample(self, round_trip_delay_s: float) -> float:
        """``tau / T_s``, the fractional CIR sample the echo peaks at."""

        return float(round_trip_delay_s) / self.waveform_sample_period_s


#: The two analytic pulse shapes. Both are closed-form functions of a
#: CONTINUOUS argument, which is the constraint that matters: the kernel
#: evaluates ``p(t - tau)`` at the exact fractional delay, so there is no
#: lookup table, no gather, and no interpolation anywhere in the family. A
#: sampled or tabulated pulse is a different design and needs its own decision.
PULSE_KIND_RECT = "rect"
PULSE_KIND_LFM = "lfm"
PULSE_KINDS = frozenset({PULSE_KIND_RECT, PULSE_KIND_LFM})

#: ``integral |p(t)|^2 dt = 1``. Named as an explicit field on the spec rather
#: than left as a constant inside a kernel, because it is what makes the
#: matched-filter peak exactly ``C_rt`` with no ``N``-dependent factor, and
#: therefore what makes the cross-waveform amplitude invariant assertable
#: without waveform-specific bookkeeping.
PULSE_NORMALIZATION_UNIT_ENERGY = "unit_energy"


@dataclass(frozen=True, slots=True)
class PulsedEchoSpec:
    """One pulsed frame's fast-time gate, pulse shape, and PRI, in SI units.

    The product is the complex baseband received pulse train
    ``y[pulse, sensor_pair, sample]``  -  the matched-filter INPUT, not its
    output. The matched filter itself is a correlation and lives in DSP glue
    (:mod:`witwin.radar.sigproc.matched_filter`), because synthesis owns the
    received waveform and processing owns the filter. Putting the correlation
    in the kernel would fuse a modelling decision (which replica, which window,
    which oversampling) into the physics.

    **The pulse is evaluated at the exact fractional delay.** ``u = t_g + m T_s
    - tau_k(l)`` is a continuous number and ``p(u)`` is evaluated from its
    analytic form there, never snapped to the nearest sample. Snapping would
    quantise the delay by ``T_s / 2``, which at 50 MSPS is 10 ns, three metres
    of range, and it would destroy the closed form that every assertion below
    is written against. This is why both supported pulse kinds are analytic.

    **Phasor convention.** Like the OFDM CFR cube and unlike the FMCW beat cube,
    this train is published in the CHANNEL convention :data:`CHANNEL_PHASOR`.
    There is no de-chirping here, so there is nothing to conjugate and no
    conversion site anywhere in the family.

    **Structural parallel.** The three waveforms differ only in one factor. FMCW
    contributes ``exp(+j 2 pi S tau t_m)``, OFDM contributes
    ``exp(-j 2 pi n df tau)``, and this family contributes ``p(t - tau)``. The
    slow-time factor - the carrier applied to the delay CHANGE - is identical in
    all three, which is the whole point of a shared input contract.

    ``bandwidth_hz`` is the LFM sweep for ``pulse_kind = "lfm"`` and ``1 / T_p``
    for ``pulse_kind = "rect"``, which is the rectangular pulse's own
    matched-filter bandwidth. It is a declared field rather than an inferred one
    because it sets the range cell, the range-migration bound, and (in Phase-6
    stage S4) the receiver's noise bandwidth, and inferring it differently in
    three places is how those three quietly disagree.

    **The pulse support is half-open**, ``0 <= u < T_p``, and that is a contract
    rather than an accident of writing the comparison one way. A closed support
    puts exactly one extra sample inside the pulse whenever the delay lands on
    the sample grid and not otherwise, so the received pulse would be
    ``M_p + 1`` samples long at one delay and ``M_p`` at the next. The matched
    filter's replica has a fixed length, so that one sample is a mismatched tap:
    it costs about 0.2 percent of the peak magnitude and biases the estimated
    delay by nearly two thousandths of a sample, at one delay in every ``M_p``.
    Half-open makes the sampled pulse exactly :attr:`pulse_sample_count` samples
    long at EVERY delay, and it leaves the continuous unit-energy integral
    unchanged because a single point has measure zero.

    ``max_expected_delay_rate`` is a CONFIGURED bound on ``|d(tau_rt)/dt|`` - the
    velocity window the radar is set up for - and never a measured maximum,
    which would be a per-frame device-to-host transfer.
    :func:`require_pulsed_compatible` uses it for the range-migration check.
    """

    #: This kernel has no ``lambda / (4 pi d)`` term at all: free-space spreading
    #: is Channel transport's, per leg, once. A statement about the kernel rather
    #: than a setting, so a class attribute rather than a field.
    applies_spreading: ClassVar[bool] = False

    #: The convention the published train is in, carried as data so that a
    #: consumer never has to infer it from the waveform's name.
    phasor: ClassVar[str] = CHANNEL_PHASOR
    time_dependence: ClassVar[str] = CHANNEL_TIME_DEPENDENCE

    num_pulses: int
    num_samples: int
    sample_period_s: float
    pri_s: float
    range_gate_start_s: float
    pulse_kind: str
    pulse_width_s: float
    bandwidth_hz: float
    reference_frequency_hz: float
    max_expected_delay_rate: float
    carrier_hz: float = 0.0
    carrier_rate_hz: float = 0.0
    pulse_normalization: str = PULSE_NORMALIZATION_UNIT_ENERGY

    def __post_init__(self) -> None:
        if self.num_pulses < 1:
            raise ValueError("num_pulses must be positive")
        if self.num_samples < 1:
            raise ValueError("num_samples must be positive")
        if self.sample_period_s <= 0.0:
            raise ValueError("sample_period_s must be positive")
        if self.pri_s <= 0.0:
            raise ValueError("pri_s must be positive")
        if self.range_gate_start_s < 0.0:
            raise ValueError("range_gate_start_s must be non-negative")
        if self.pulse_width_s <= 0.0:
            raise ValueError("pulse_width_s must be positive")
        if self.bandwidth_hz <= 0.0:
            raise ValueError("bandwidth_hz must be positive")
        if self.max_expected_delay_rate < 0.0:
            raise ValueError("max_expected_delay_rate must be non-negative")
        if not self.reference_frequency_hz > 0.0:
            raise ValueError(
                "reference_frequency_hz must be positive; it is the frequency "
                "the weight this spec will consume was evaluated at, and "
                "require_compatible refuses a mismatch"
            )
        if self.pulse_kind not in PULSE_KINDS:
            raise ValueError(
                f"pulse_kind must be one of {sorted(PULSE_KINDS)}, got "
                f"{self.pulse_kind!r}; both supported kinds are ANALYTIC, "
                "because the kernel evaluates the pulse at a continuous "
                "fractional delay. A sampled or tabulated pulse is a different "
                "design and needs its own decision"
            )
        if self.pulse_normalization != PULSE_NORMALIZATION_UNIT_ENERGY:
            raise ValueError(
                "pulse_normalization must be "
                f"{PULSE_NORMALIZATION_UNIT_ENERGY!r}; unit ENERGY is what makes "
                "the matched-filter peak exactly C_rt with no N-dependent "
                "factor. A unit-amplitude pulse would put a T_p / T_s factor in "
                "the peak and force every amplitude comparison to carry it"
            )
        require_single_carrier_home(self.carrier_hz, self.carrier_rate_hz)

    @property
    def sample_rate_hz(self) -> float:
        return 1.0 / self.sample_period_s

    @property
    def wavelength_m(self) -> float:
        return SPEED_OF_LIGHT_M_PER_S / self.reference_frequency_hz

    @property
    def is_linear_fm(self) -> bool:
        return self.pulse_kind == PULSE_KIND_LFM

    @property
    def pulse_amplitude(self) -> float:
        """``1 / sqrt(T_p)``, the unit-energy envelope height.

        Both kinds share it: the LFM differs from the rectangle only by a phase,
        and a phase does not change ``|p|``. Passed to the kernel as a scalar so
        that the normalisation lives on this spec rather than inside the kernel.
        """

        return 1.0 / math.sqrt(self.pulse_width_s)

    @property
    def pulse_sample_count(self) -> int:
        """How many fast-time samples the pulse spans, rounded to the grid.

        Used by the matched-filter replica, not by the kernel: the kernel never
        discretises the pulse. The discrete replica has EXACTLY unit energy when
        ``T_p`` is a whole number of samples, which is what
        :attr:`pulse_grid_is_commensurate` reports.
        """

        return int(round(self.pulse_width_s / self.sample_period_s))

    @property
    def pulse_grid_is_commensurate(self) -> bool:
        """Whether ``T_p`` is an exact whole number of sample periods.

        When it is, the sampled replica's discrete energy
        ``sum_m |p[m]|^2 T_s`` is exactly 1 and the matched-filter peak is
        exactly ``C_rt``. When it is not, the replica's energy is
        ``round(T_p / T_s) * T_s / T_p`` and the peak carries that same factor -
        a property of the DISCRETE replica, not of the kernel, which is why it is
        reported rather than silently corrected.
        """

        return (
            abs(self.pulse_sample_count * self.sample_period_s - self.pulse_width_s)
            <= 1e-12 * self.pulse_width_s
        )

    @property
    def range_gate_end_s(self) -> float:
        """``t_g + M T_s``, the last fast-time instant the gate observes."""

        return self.range_gate_start_s + self.num_samples * self.sample_period_s

    @property
    def duty_cycle(self) -> float:
        """``T_p / T_pri``."""

        return self.pulse_width_s / self.pri_s

    @property
    def range_resolution_m(self) -> float:
        """``c0 / (2 B)`` for the LFM, ``c0 T_p / 2`` for the rectangle.

        Two expressions rather than one because they are two different physical
        statements: the LFM's resolution comes from its SWEEP and is independent
        of its length, while the rectangle's comes from its length alone. The
        two coincide only at ``B = 1 / T_p``, which is exactly the value this
        spec requires a rectangular pulse to declare, so the branch is a
        readability choice and not a numerical one.
        """

        if self.is_linear_fm:
            return SPEED_OF_LIGHT_M_PER_S / (2.0 * self.bandwidth_hz)
        return SPEED_OF_LIGHT_M_PER_S * self.pulse_width_s / 2.0

    @property
    def max_unambiguous_range_m(self) -> float:
        """``c0 T_pri / 2``: beyond it an echo lands in the next PRI."""

        return SPEED_OF_LIGHT_M_PER_S * self.pri_s / 2.0

    @property
    def max_unambiguous_speed_m_s(self) -> float:
        """``c0 / (4 f_ref T_pri)``, equivalently ``lambda / (4 T_pri)``.

        Half a wavelength of two-way path change per pulse is half a cycle of
        Doppler phase; beyond it the sign of the velocity is not recoverable and
        a receding target reads as an approaching one.
        """

        return SPEED_OF_LIGHT_M_PER_S / (
            4.0 * self.reference_frequency_hz * self.pri_s
        )

    @property
    def coherent_processing_interval_s(self) -> float:
        """``L T_pri``, the span the slow-time transform is taken over."""

        return self.num_pulses * self.pri_s

    @property
    def range_cell_delay_s(self) -> float:
        """``1 / B``, one range cell expressed as a delay."""

        return 1.0 / self.bandwidth_hz

    @property
    def range_migration_delay_s(self) -> float:
        """How far the delay walks over one coherent processing interval.

        ``max_expected_delay_rate * L * T_pri``. Compared against
        :attr:`range_cell_delay_s` by :func:`require_pulsed_compatible`: if the
        walk exceeds one range cell the peak smears across cells and the
        single-cell closed form this family is written against stops holding.
        """

        return self.max_expected_delay_rate * self.coherent_processing_interval_s

    def instantaneous_pulse_frequency_hz(self, envelope_time_s: float) -> float:
        """``B u / T_p`` for the LFM, ``0`` for the rectangle.

        The derivative of the pulse's own phase with respect to its argument,
        expressed as a frequency. It is the pulsed analogue of the OFDM
        subcarrier offset ``n df``: a small, envelope-position-dependent
        addition to the carrier that the slow-time phase step carries and that a
        constant-envelope model would miss.
        """

        if not self.is_linear_fm:
            return 0.0
        return self.bandwidth_hz * float(envelope_time_s) / self.pulse_width_s

    def slow_time_phase_step_rad(
        self, delay_rate: float, envelope_time_s: float = 0.0
    ) -> float:
        """Phase advance per pulse at a fixed fast-time sample, in radians.

        ``-2 pi tau_rate T_pri (f_c + f_r + B u / T_p)``. The first two terms are
        the carrier in whichever of its two homes it lives in - they enter the
        same way because a kernel-owned carrier multiplies the full delay and
        therefore already walks. The third is the LFM's own phase moving with the
        drifting envelope position, which is a correction of relative size
        ``B u / (T_p f_ref)``: about 1.3e-4 at the middle of a 20 MHz, 10 us
        sweep at 77 GHz. Small, but larger than the tolerance the slow-time slope
        is asserted to, so it is part of the closed form rather than noise.
        """

        return (
            -math.tau
            * float(delay_rate)
            * self.pri_s
            * (
                self.carrier_hz
                + self.carrier_rate_hz
                + self.instantaneous_pulse_frequency_hz(envelope_time_s)
            )
        )

    def doppler_frequency_hz(self, delay_rate: float) -> float:
        """``-f_ref tau_rate``: the physical Doppler in Channel's convention.

        Negative for a receding row. The published train is in that same
        convention, so this IS the tone the slow-time transform shows - unlike
        the FMCW beat cube, whose single conjugation puts its tone at ``+f_ref
        tau_rate``.
        """

        return -self.reference_frequency_hz * float(delay_rate)


class SlowTimeMode(str, Enum):
    """How the weight and the slow-time axis divide the Doppler phase.

    These two are mutually exclusive, and they are ONE enum rather than two
    independently-settable fields because the combination "the caller refreshed
    the weight at every slot AND the kernel still applies a carrier rate"
    applies Doppler twice and looks like a plausible radar cube while doing it.
    Phase 6 always uses the frozen mode; Phase 7 owns dynamics and is the reason
    the refreshed mode is named now.
    """

    #: The weight was computed once, at the frame's ``tau_rt``, and does not
    #: walk across chirps/symbols/pulses. The slow-time carrier phase is the
    #: waveform kernel's job.
    FROZEN_WEIGHT_WITH_CARRIER_RATE = "frozen_weight_with_carrier_rate"

    #: The weight is re-evaluated at every slow-time slot, so it already walked.
    #: A carrier-rate term on top of it would double the Doppler.
    REFRESHED_WEIGHT_NO_RATE = "refreshed_weight_no_rate"


@runtime_checkable
class WaveformSpecProtocol(Protocol):
    """Exactly the attributes :func:`require_compatible` reads.

    Declared as a Protocol rather than a base class because the three waveform
    specs have nothing else in common: an FMCW ramp, an OFDM subcarrier grid,
    and a pulse envelope share no fields. What they DO share is a position on
    the four questions that decide whether a weight may be handed to them.

    ``tx_power_mode`` is deliberately absent: it belongs to the sensor-weight
    owner rather than to a waveform, and :func:`require_compatible` reads it
    only when a spec chooses to declare it.
    """

    #: Absolute reference-frequency carrier the KERNEL applies, in Hz. Zero
    #: means the kernel applies none because the weight already carries it.
    carrier_hz: float

    #: Reference frequency applied to the delay CHANGE only, in Hz. Zero means
    #: the kernel applies none.
    carrier_rate_hz: float

    #: The frequency the weight was evaluated at. Must equal the batch's.
    reference_frequency_hz: float

    #: Whether the waveform owner multiplies by ``lambda/(4 pi d)`` itself.
    applies_spreading: bool


def _require_tensor(
    name: str,
    value: object,
    *,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}")
    if value.dtype != dtype:
        raise TypeError(f"{name} must use {dtype}, got {value.dtype}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
    if not value.is_contiguous():
        raise ValueError(
            f"{name} must be contiguous; a synthesis kernel indexes it linearly "
            "and the contract will not hide a copy on the hot path"
        )
    if value.device != device:
        raise ValueError(
            f"{name} is on {value.device} but the batch is on {device}; a "
            "synthesis batch is single-device by contract"
        )
    return value


@dataclass(frozen=True, slots=True, eq=False)
class SynthesisPathBatch:
    """What a waveform synthesis kernel may assume about a set of path rows.

    Geometry is ``total_delay_s`` (the ROUND-TRIP delay ``tau_rt``, in seconds,
    never a one-way distance) and ``delay_rate`` (``d(tau_rt)/dt``,
    dimensionless). A kernel consumes those two and nothing else about the
    geometry: it may never reconstruct a distance, and it may never re-apply a
    ``1/(4 pi d)``.

    ``complex_transfer_ref`` is in the CHANNEL phasor convention,
    ``exp(-j k d)`` under ``exp(+j 2 pi f t)`` time dependence, evaluated at
    ``reference_frequency_hz``. It is NOT a beat weight; converting to one is
    the FMCW owner's single call site.

    The four provenance fields are the whole reason this type exists. They say
    what is ALREADY inside the weight, so :func:`require_compatible` can refuse
    a spec that would apply it a second time. They are set by the two
    classmethods below rather than by a caller, because a caller that could
    assert its own provenance could assert the convenient one.

    Validation is host-only: shapes, dtypes, contiguity, device, and flags. It
    reads no tensor VALUE, so constructing this contract costs no
    device-to-host transfer and no synchronization. In particular
    ``pair_offsets[0] == 0`` and ``pair_offsets[-1] == path_count`` are a
    documented producer obligation, exactly as in ``RadarPathBatch``, not a
    device read.
    """

    # ---- cardinality (host ints, already published by the compact contract) --
    sensor_pair_count: int
    path_count: int

    # ---- row -> segment partition -------------------------------------------
    sensor_pair_index: torch.Tensor
    pair_offsets: torch.Tensor

    # ---- geometry ------------------------------------------------------------
    total_delay_s: torch.Tensor
    delay_rate: torch.Tensor | None

    # ---- transfer ------------------------------------------------------------
    complex_transfer_ref: torch.Tensor
    reference_frequency_hz: float
    frequency_response: torch.Tensor | None
    frequency_offsets_hz: torch.Tensor | None

    # ---- identity ------------------------------------------------------------
    topology: RadarPathTopology
    row_valid: torch.Tensor | None
    join_mode: JoinMode

    # ---- provenance ----------------------------------------------------------
    weight_includes_reference_phase: bool
    weight_includes_spreading: bool
    weight_includes_tx_power: bool
    slow_time_mode: SlowTimeMode

    def __post_init__(self) -> None:
        if self.join_mode not in JOIN_MODES:
            raise ValueError(
                f"join_mode must be one of {sorted(JOIN_MODES)}, got "
                f"{self.join_mode!r}"
            )
        if not isinstance(self.slow_time_mode, SlowTimeMode):
            raise TypeError(
                "slow_time_mode must be a SlowTimeMode member, got "
                f"{self.slow_time_mode!r}"
            )
        if type(self.sensor_pair_count) is not int or self.sensor_pair_count < 1:
            raise ValueError("sensor_pair_count must be a positive int")
        if type(self.path_count) is not int or self.path_count < 0:
            raise ValueError("path_count must be a non-negative int")
        for name in (
            "weight_includes_reference_phase",
            "weight_includes_spreading",
            "weight_includes_tx_power",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool")
        if (
            type(self.reference_frequency_hz) is not float
            or not self.reference_frequency_hz > 0.0
        ):
            raise ValueError(
                "reference_frequency_hz must be a positive float; it is the "
                "frequency the weight was evaluated at, not an optional label"
            )

        device = self.total_delay_s.device
        rows = (self.path_count,)
        _require_tensor(
            "total_delay_s",
            self.total_delay_s,
            dtype=torch.float32,
            shape=rows,
            device=device,
        )
        _require_tensor(
            "sensor_pair_index",
            self.sensor_pair_index,
            dtype=torch.int64,
            shape=rows,
            device=device,
        )
        _require_tensor(
            "pair_offsets",
            self.pair_offsets,
            dtype=torch.int64,
            shape=(self.sensor_pair_count + 1,),
            device=device,
        )
        _require_tensor(
            "complex_transfer_ref",
            self.complex_transfer_ref,
            dtype=torch.complex64,
            shape=rows,
            device=device,
        )
        if self.delay_rate is not None:
            _require_tensor(
                "delay_rate",
                self.delay_rate,
                dtype=torch.float32,
                shape=rows,
                device=device,
            )
        if self.row_valid is not None:
            _require_tensor(
                "row_valid",
                self.row_valid,
                dtype=torch.bool,
                shape=rows,
                device=device,
            )
        if (self.frequency_response is None) != (self.frequency_offsets_hz is None):
            raise ValueError(
                "frequency_response and frequency_offsets_hz are one statement "
                "and must be supplied together; a response without its "
                "frequency grid says nothing"
            )
        if self.frequency_response is not None:
            if self.frequency_offsets_hz.dim() != 1:
                raise ValueError("frequency_offsets_hz must have shape (F,)")
            bands = (self.path_count, int(self.frequency_offsets_hz.shape[0]))
            _require_tensor(
                "frequency_response",
                self.frequency_response,
                dtype=torch.complex64,
                shape=bands,
                device=device,
            )
            _require_tensor(
                "frequency_offsets_hz",
                self.frequency_offsets_hz,
                dtype=torch.float32,
                shape=(bands[1],),
                device=device,
            )
        if self.topology.row_count != self.path_count:
            raise ValueError("topology must have exactly path_count rows")

    @property
    def device(self) -> torch.device:
        return self.total_delay_s.device

    @classmethod
    def from_radar_paths(
        cls,
        paths: RadarPathBatch,
        *,
        slow_time_mode: SlowTimeMode,
    ) -> "SynthesisPathBatch":
        """Wrap a composed round-trip batch, zero-copy, with Channel provenance.

        Every tensor passes through by reference. Nothing is cloned, made
        contiguous, or moved: row identity, row order, storage aliasing, stride,
        dtype, device, and gradient state are all preserved, and a test asserts
        object identity rather than value equality.

        The three provenance booleans are Channel's published contract, not a
        caller's opinion, which is why they are written here:

        * ``coefficient_reference = "includes_reference_frequency_phase"``
        * ``FREE_SPACE_AMPLITUDE = "sqrt(tx_power)*wavelength/(4*pi*distance)"``

        ``slow_time_mode`` is the one thing the caller must say, because only
        the caller knows whether it froze the weight for the frame or refreshes
        it per slot. It has no default: defaulting it would make the Phase-7
        collision a silent wrong answer instead of a refusal.
        """

        if not isinstance(paths, RadarPathBatch):
            raise TypeError(
                f"from_radar_paths needs a RadarPathBatch, got {type(paths).__name__}"
            )
        return cls(
            sensor_pair_count=paths.sensor_pair_count,
            path_count=paths.path_count,
            sensor_pair_index=paths.sensor_pair_index,
            pair_offsets=paths.pair_offsets,
            total_delay_s=paths.total_delay_s,
            delay_rate=paths.delay_rate,
            complex_transfer_ref=paths.complex_transfer_ref,
            reference_frequency_hz=float(paths.reference_frequency_hz),
            frequency_response=None,
            frequency_offsets_hz=None,
            topology=paths.topology,
            row_valid=paths.row_valid,
            join_mode=paths.join_mode,
            weight_includes_reference_phase=True,
            weight_includes_spreading=True,
            weight_includes_tx_power=True,
            slow_time_mode=slow_time_mode,
        )

    @classmethod
    def from_real_amplitudes(
        cls,
        one_way_distances_m: torch.Tensor,
        amplitudes: torch.Tensor,
        *,
        pair_offsets: torch.Tensor,
        topology: RadarPathTopology,
        c0: float,
        reference_frequency_hz: float,
        delay_rate: torch.Tensor | None = None,
        join_mode: JoinMode = "multipath",
    ) -> "SynthesisPathBatch":
        """Embed the legacy real-amplitude path as the complex special case.

        This is the whole of the real-compatibility criterion: the existing
        Radar baseline is not a second code path, it is ``C = amp + 0j`` with
        the monostatic delay written down once, here, in Python, explicitly.

        Two traps are encoded rather than commented:

        * ``torch.complex(amplitudes, zeros)``, never
          ``complex(abs(amplitudes), 0)``. The SIGN of a legacy amplitude is the
          only phase a real amplitude can carry - it is the reflection flip -
          and discarding it is a silent 180-degree error that no magnitude plot
          shows.
        * ``weight_includes_reference_phase = False``. A real amplitude carries
          no phase at all, so rule R2 forces the spec to own the carrier, which
          is exactly the legacy Dirichlet phase structure. The complex-weight
          switch and the carrier-home switch are therefore the same act.

        ``one_way_distances_m`` is doubled here because the legacy input is a
        one-way distance and every contract downstream of this one speaks
        round-trip delay. Making that conversion visible at the boundary is the
        point: the legacy kernel did it internally, where a caller that already
        had a round-trip delay could not tell.
        """

        if one_way_distances_m.shape != amplitudes.shape:
            raise ValueError(
                "one_way_distances_m and amplitudes must have the same shape"
            )
        if amplitudes.dtype != torch.float32:
            raise TypeError(
                f"amplitudes must be float32, got {amplitudes.dtype}"
            )
        if not c0 > 0.0:
            raise ValueError("c0 must be positive")
        total_delay_s = one_way_distances_m * (2.0 / float(c0))
        complex_transfer_ref = torch.complex(
            amplitudes, torch.zeros_like(amplitudes)
        ).to(torch.complex64)
        path_count = int(amplitudes.shape[0])
        sensor_pair_count = int(pair_offsets.shape[0]) - 1
        rows = torch.arange(
            path_count, device=pair_offsets.device, dtype=torch.int64
        )
        sensor_pair_index = torch.bucketize(rows, pair_offsets[1:], right=True)
        return cls(
            sensor_pair_count=sensor_pair_count,
            path_count=path_count,
            sensor_pair_index=sensor_pair_index.contiguous(),
            pair_offsets=pair_offsets.contiguous(),
            total_delay_s=total_delay_s.contiguous(),
            delay_rate=None if delay_rate is None else delay_rate.contiguous(),
            complex_transfer_ref=complex_transfer_ref.contiguous(),
            reference_frequency_hz=float(reference_frequency_hz),
            frequency_response=None,
            frequency_offsets_hz=None,
            topology=topology,
            row_valid=None,
            join_mode=join_mode,
            weight_includes_reference_phase=False,
            weight_includes_spreading=True,
            weight_includes_tx_power=True,
            slow_time_mode=SlowTimeMode.FROZEN_WEIGHT_WITH_CARRIER_RATE,
        )


#: The axis names each waveform's rank-3 product is indexed by. Published as
#: data because "the second axis is the sensor pair" is exactly the sort of
#: shared assumption that a consumer eventually gets wrong in one place.
FMCW_AXES = ("chirp", "sensor_pair", "sample")
OFDM_AXES = ("symbol", "sensor_pair", "subcarrier")
PULSED_AXES = ("pulse", "sensor_pair", "sample")


@dataclass(frozen=True, slots=True)
class SynthesisResult:
    """What one synthesis call produced, with its conventions carried as data.

    Data only. The waveform owner constructs it through one of the three
    classmethods below, which is the repository's standing shape for a result:
    the producer knows which product it made, and a consumer never has to infer
    a phasor convention from a waveform's name.

    ``phasor`` matters more here than it looks. The FMCW beat cube is
    CONJUGATED relative to Channel's convention and the other two are not, so a
    cross-waveform phase comparison that ignored this field would find a sign
    error in two of the three and conclude that the physics disagreed.
    """

    cube: torch.Tensor
    kind: str
    axes: tuple[str, ...]
    phasor: str
    time_dependence: str
    reference_frequency_hz: float

    def __post_init__(self) -> None:
        if self.cube.dim() != len(self.axes):
            raise ValueError(
                f"a {self.kind} cube has {len(self.axes)} axes {self.axes}, got "
                f"shape {tuple(self.cube.shape)}"
            )

    @classmethod
    def from_fmcw_beat(cls, cube: torch.Tensor, spec: FmcwBeatSpec) -> "SynthesisResult":
        return cls(
            cube=cube,
            kind="fmcw",
            axes=FMCW_AXES,
            phasor=spec.phasor,
            time_dependence=spec.time_dependence,
            reference_frequency_hz=float(spec.reference_frequency_hz),
        )

    @classmethod
    def from_ofdm_cfr(cls, cube: torch.Tensor, spec: OfdmCfrSpec) -> "SynthesisResult":
        return cls(
            cube=cube,
            kind="ofdm",
            axes=OFDM_AXES,
            phasor=spec.phasor,
            time_dependence=spec.time_dependence,
            reference_frequency_hz=float(spec.reference_frequency_hz),
        )

    @classmethod
    def from_pulsed_echo(
        cls, cube: torch.Tensor, spec: PulsedEchoSpec
    ) -> "SynthesisResult":
        return cls(
            cube=cube,
            kind="pulsed",
            axes=PULSED_AXES,
            phasor=spec.phasor,
            time_dependence=spec.time_dependence,
            reference_frequency_hz=float(spec.reference_frequency_hz),
        )


def require_compatible(batch: SynthesisPathBatch, spec: WaveformSpecProtocol) -> None:
    """Refuse any weight/spec pair that would count a factor twice.

    Called by every waveform entry point before any kernel launch. Each rule
    below names the hazard it prevents, because the failure it prevents is
    always a plausible-looking number rather than a crash, and an error message
    that only says "invalid configuration" would send the reader looking for a
    bug in the physics.

    One deviation from the Phase-6 design document is recorded here rather than
    buried: the design states R3 as "the frozen mode requires
    ``carrier_rate_hz == reference_frequency_hz``", full stop. That is
    unsatisfiable for the legacy real-amplitude batch, which is frozen AND has
    ``weight_includes_reference_phase = False``: R2 then forces
    ``carrier_hz = f_ref``, and a spec with both carrier parameters nonzero
    double counts the carrier and is refused by the spec itself. The physics
    resolves it - differentiating the FMCW phase with respect to slow time gives
    the same bracket for ``(f_ref, 0)`` and ``(0, f_ref)``, because a
    kernel-owned carrier multiplies the FULL ``tau(t)`` and therefore already
    walks. So R3 is enforced as "the delay change has exactly one owner, chosen
    by the provenance": the weight's carrier home decides which of the two
    parameters must equal ``f_ref``.
    """

    if not isinstance(batch, SynthesisPathBatch):
        raise TypeError(
            f"require_compatible needs a SynthesisPathBatch, got {type(batch).__name__}"
        )
    for attribute in (
        "carrier_hz",
        "carrier_rate_hz",
        "reference_frequency_hz",
        "applies_spreading",
    ):
        if not hasattr(spec, attribute):
            raise TypeError(
                f"{type(spec).__name__} does not declare {attribute!r}, so it "
                "cannot be checked against a weight's provenance; a waveform "
                "spec must satisfy WaveformSpecProtocol"
            )

    carrier_hz = float(spec.carrier_hz)
    carrier_rate_hz = float(spec.carrier_rate_hz)
    f_ref = batch.reference_frequency_hz

    # R1 - hazard H1: the Channel coefficient already holds
    # exp(-j 2 pi f_ref tau_rt).
    if batch.weight_includes_reference_phase and carrier_hz != 0.0:
        raise ValueError(
            "double-counted carrier phase: the weight already carries "
            "exp(-j*2*pi*f_ref*tau_rt) (coefficient_reference = "
            "'includes_reference_frequency_phase'), so carrier_hz must be 0; "
            f"got carrier_hz={carrier_hz}"
        )

    # R2 - the mirror image: nobody owns the absolute carrier at all.
    if (
        not batch.weight_includes_reference_phase
        and carrier_hz == 0.0
        and carrier_rate_hz == 0.0
    ):
        raise ValueError(
            "missing carrier phase: this weight carries no reference-frequency "
            "phase, and neither carrier_hz nor carrier_rate_hz is set, so the "
            "absolute carrier has no owner and the synthesized IQ would have no "
            "range phase at all"
        )

    # R3 - hazard H4, first half: a frozen weight does not walk across slow
    # time, so the delay CHANGE needs exactly one owner and the provenance says
    # which parameter it is.
    if batch.slow_time_mode is SlowTimeMode.FROZEN_WEIGHT_WITH_CARRIER_RATE:
        if batch.weight_includes_reference_phase:
            if carrier_rate_hz != f_ref:
                raise ValueError(
                    "understated Doppler: the weight is frozen at the frame's "
                    "tau_rt and carries the reference phase, so the delay "
                    f"change has no other home; carrier_rate_hz must be {f_ref}, "
                    f"got {carrier_rate_hz}. Dropping it understates intra-frame "
                    "Doppler by one to two orders of magnitude while still "
                    "producing a plausible-looking cube"
                )
        elif carrier_hz != f_ref:
            raise ValueError(
                "understated Doppler: the weight carries no reference phase, so "
                "the kernel must own the absolute carrier and thereby the delay "
                f"change; carrier_hz must be {f_ref}, got {carrier_hz}"
            )

    # R4 - hazard H4, second half: a refreshed weight already walked.
    if batch.slow_time_mode is SlowTimeMode.REFRESHED_WEIGHT_NO_RATE:
        if carrier_rate_hz != 0.0:
            raise ValueError(
                "double-counted Doppler: this weight is re-evaluated at every "
                "slow-time slot, so it already carries the delay change; "
                f"carrier_rate_hz must be 0, got {carrier_rate_hz}"
            )
        if batch.delay_rate is not None:
            raise ValueError(
                "double-counted Doppler: a refreshed weight walks by itself, so "
                "the batch must not also publish delay_rate for a kernel to "
                "apply"
            )

    # R5 - hazard F1: free-space spreading is Channel transport's, per leg,
    # once.
    if batch.weight_includes_spreading and bool(spec.applies_spreading):
        raise ValueError(
            "double-counted free-space spreading: the weight already contains "
            "wavelength/(4*pi*distance) per leg (FREE_SPACE_AMPLITUDE), so the "
            "waveform owner must not apply it again; set applies_spreading=False"
        )

    # R6 - hazard F4: TX power reaches physics through powers_w and nowhere
    # else. tx_power_mode belongs to the sensor-weight owner, so it is checked
    # only when a spec declares it.
    tx_power_mode = getattr(spec, "tx_power_mode", None)
    if (
        batch.weight_includes_tx_power
        and tx_power_mode is not None
        and tx_power_mode != "already_in_weight"
    ):
        raise ValueError(
            "double-counted transmit power: the weight already contains "
            "sqrt(tx_power) from the source endpoint's powers_w, so the sensor "
            "weight owner must run with tx_power_mode='already_in_weight'; got "
            f"{tx_power_mode!r}"
        )

    # R7 - the weight was evaluated at one frequency and means nothing at
    # another. This mirrors Channel's own request/compile frequency rule.
    if float(spec.reference_frequency_hz) != f_ref:
        raise ValueError(
            "reference frequency mismatch: the weight was evaluated at "
            f"{f_ref} Hz but the waveform spec declares "
            f"{float(spec.reference_frequency_hz)} Hz; a narrowband coefficient "
            "is not transferable between reference frequencies"
        )

    # R8 - wideband material response is Phase 8.
    if batch.frequency_response is not None:
        raise ValueError(
            "wideband material response is Phase 8 work: this contract declares "
            "frequency_response/frequency_offsets_hz so that the Phase-6 "
            "narrowband assumption is explicit, and refuses a non-None value so "
            "that it cannot be silently ignored by a kernel that only knows the "
            "narrowband law H(f_ref+df) = C(f_ref)*exp(-j*2*pi*df*delay_s)"
        )


def require_ofdm_compatible(
    batch: SynthesisPathBatch, spec: OfdmCfrSpec
) -> None:
    """The shared provenance rules, plus OFDM's cyclic-prefix contract.

    Called by :func:`~witwin.radar.synthesis.ofdm_cfr.synthesize_ofdm_cfr`
    before any kernel launch. Two checks beyond :func:`require_compatible`, both
    on CONFIGURED values and never on measured device delays - reading a
    per-frame maximum delay to the host would be exactly the hot-path
    device-to-host transfer the fixed-topology capability exists to avoid:

    1. ``max_expected_delay_s < cyclic_prefix_s``. This is the standard
       OFDM-radar assumption: there is no timing synchronisation to the echo, so
       the whole echo has to land inside the cyclic-prefix window for ``Y / X``
       to be exactly ``exp(-j 2 pi n df tau)``. Outside it the response gains an
       inter-symbol term that the closed form does not have, and the cube would
       be wrong in a way that looks like a slightly defocused range profile.

    2. The delay SPREAD bound ``max_k(tau_k) - min_k(tau_k) < T_cp``, which the
       single-tap-per-subcarrier form also requires. It is the same inequality:
       every round-trip delay is non-negative and bounded above by
       ``max_expected_delay_s``, so check 1 implies it. It is documented here
       rather than checked separately because checking it for real would mean
       reducing over the device delays.

    Both refusals name ``cyclic_prefix_s``. There is no clamp and no
    reduced-accuracy mode.
    """

    if not isinstance(spec, OfdmCfrSpec):
        raise TypeError(
            f"require_ofdm_compatible needs an OfdmCfrSpec, got "
            f"{type(spec).__name__}"
        )
    require_compatible(batch, spec)
    if spec.max_expected_delay_s >= spec.cyclic_prefix_s:
        raise ValueError(
            "the configured echo window does not fit inside the cyclic prefix: "
            f"max_expected_delay_s={spec.max_expected_delay_s} is not less than "
            f"cyclic_prefix_s={spec.cyclic_prefix_s}. The single-tap "
            "per-subcarrier response H = Y/X is exactly "
            "exp(-j*2*pi*n*df*tau) only while the whole echo lands inside the "
            "cyclic-prefix window; beyond it there is an inter-symbol term this "
            "kernel does not model. Shorten the range window or lengthen "
            "cyclic_prefix_s - there is no clamped or reduced-accuracy mode"
        )


def require_pulsed_compatible(
    batch: SynthesisPathBatch, spec: PulsedEchoSpec
) -> None:
    """The shared provenance rules, plus the pulsed timing and migration bounds.

    Called by :func:`~witwin.radar.synthesis.pulsed_echo.synthesize_pulsed_echo`
    before any kernel launch. Three checks beyond :func:`require_compatible`, all
    on CONFIGURED values and never on measured device delays - reducing over the
    device delays to find a maximum would be exactly the hot-path
    device-to-host transfer the fixed-topology capability exists to avoid:

    1. ``pulse_width_s < pri_s``. A pulse at least as long as the repetition
       interval never stops transmitting, so there is no receive window at all.

    2. ``range_gate_start_s + num_samples * sample_period_s <= pri_s``. The gate
       must close before the next pulse fires; a gate that overruns is observing
       the next transmission, not an echo.

    3. ``max_expected_delay_rate * num_pulses * pri_s < 1 / bandwidth_hz``. Within
       a coherent processing interval of ``L`` pulses the delay walks by
       ``tau_rate L T_pri``. If that walk exceeds one range cell the echo migrates
       between cells, the peak smears, and the single-cell closed form this
       family is written against stops holding - quietly, as a loss of range
       resolution that looks like a defocused target. A fixture must satisfy the
       bound or assert the migration explicitly; there is no clamp and no
       reduced-accuracy mode.
    """

    if not isinstance(spec, PulsedEchoSpec):
        raise TypeError(
            f"require_pulsed_compatible needs a PulsedEchoSpec, got "
            f"{type(spec).__name__}"
        )
    require_compatible(batch, spec)
    if spec.pulse_width_s >= spec.pri_s:
        raise ValueError(
            f"pulse_width_s={spec.pulse_width_s} is not shorter than "
            f"pri_s={spec.pri_s}; a pulse at least as long as the repetition "
            "interval leaves no receive window, so there is no echo to gate"
        )
    if spec.range_gate_end_s > spec.pri_s:
        raise ValueError(
            "the range gate overruns the pulse repetition interval: "
            f"range_gate_start_s + num_samples * sample_period_s="
            f"{spec.range_gate_end_s} exceeds pri_s={spec.pri_s}. The gate must "
            "close before the next pulse fires; past that instant the samples "
            "observe the next transmission rather than an echo"
        )
    if spec.range_migration_delay_s >= spec.range_cell_delay_s:
        raise ValueError(
            "range migration over the coherent processing interval: the delay "
            f"walks by max_expected_delay_rate * num_pulses * pri_s="
            f"{spec.range_migration_delay_s} s, which is not less than one range "
            f"cell 1 / bandwidth_hz={spec.range_cell_delay_s} s. The echo then "
            "moves between range cells within one coherent processing interval, "
            "the matched-filter peak smears, and the single-cell closed form "
            "this family is written against stops holding. Shorten the coherent "
            "processing interval, narrow the velocity window, or widen the range "
            "cell - there is no clamped or reduced-accuracy mode"
        )


__all__ = [
    "CHANNEL_PHASOR",
    "CHANNEL_TIME_DEPENDENCE",
    "PULSE_KINDS",
    "PULSE_KIND_LFM",
    "PULSE_KIND_RECT",
    "PULSE_NORMALIZATION_UNIT_ENERGY",
    "SPEED_OF_LIGHT_M_PER_S",
    "SUBCARRIER_ORIGIN_F_REF_AT_N0",
    "BEAT_PHASOR",
    "FMCW_AXES",
    "OFDM_AXES",
    "PULSED_AXES",
    "FmcwBeatSpec",
    "OfdmCfrSpec",
    "PulsedEchoSpec",
    "SlowTimeMode",
    "SynthesisPathBatch",
    "SynthesisResult",
    "WaveformSpecProtocol",
    "require_compatible",
    "require_ofdm_compatible",
    "require_pulsed_compatible",
    "require_single_carrier_home",
]
