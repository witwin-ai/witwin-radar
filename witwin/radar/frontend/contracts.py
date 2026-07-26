"""The receive chain, described once, in SI units.

ONE runtime with a fixed order, not a set of independently callable stages.
Two objects that each know part of the chain let the CALLER decide where
thermal noise lands relative to the LNA, and that decision is worth a factor of
``g_lna^2`` in output noise power. It also lets both of them own a quantiser, so
configuring both quantises twice. Neither state is representable here: the order
lives in :class:`~witwin.radar.frontend.chain.FrontendChain` and there is
exactly one ADC slot.

Every derived scalar below is computed in Python once per frame from
configuration and handed to the kernel as a number. None of them is inferred:

    k      = 1.380649e-23 J/K exactly,  T0 = 290 K
    F      = 10^(NF_dB/10)
    T_sys  = T_ant + T0 * (F - 1)
    sigma_component = sqrt(k * T_sys * B * R / 2)             [volts]
    sigma_w^2       = 10^(L_dBc/10) * 4 pi^2 f_off^2 / fs     [rad^2 per step]
    g_lna  = 10^(gain_dB/20)
    step   = 2 * full_scale / (2^bits - 1),  Var_q = step^2/12 per component

``bandwidth_hz`` in particular is an EXPLICIT named field and is never inferred
from a waveform. It is the ADC sample rate (or a narrower IF filter) for FMCW,
the matched-filter bandwidth for pulsed - ``1/T_p`` for a rectangle, ``B`` for
an LFM - and the subcarrier spacing for an OFDM channel-frequency-response cube
or the whole occupied band for time-domain OFDM samples. Getting it wrong is a
pure scale error in SNR, which is exactly the kind of mistake that survives
every relative test, so it must be stated rather than guessed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


#: Boltzmann's constant, exact in SI since 2019.
BOLTZMANN_J_PER_K = 1.380649e-23

#: The IEEE reference temperature for a noise figure.
REFERENCE_TEMPERATURE_K = 290.0

#: Philox stage identifiers. They are part of the reproducibility contract:
#: keying the counter by the stage is what makes toggling one stage leave every
#: other stage's realisation bit-identical. Renumbering them changes every
#: realisation and is a numerical change.
STAGE_PHASE_NOISE = 0
STAGE_THERMAL_NOISE = 1

#: The fixed stage order, published as data so a reader never has to infer it
#: from the order of ``if`` statements. The runtime asserts it against itself.
FRONTEND_STAGE_ORDER = ("port", "phase", "thermal", "lna", "agc", "adc")

AGC_MODE_GLOBAL = "global"
AGC_MODE_PER_RX = "per_rx"
AGC_MODES = (AGC_MODE_GLOBAL, AGC_MODE_PER_RX)


@dataclass(frozen=True, slots=True)
class PortSpec:
    """The sqrt(W) to volt conversion, applied exactly once at stage 0.

    A synthesis cube is in sqrt(W) at the receive antenna port and everything
    downstream of the LNA is in volts. ``v = sqrt(W) * sqrt(R)`` is that
    conversion, and it happens here and nowhere else. The old code did it inside
    a transmit gain, where it was multiplied onto a weight that already carried
    transmit power - two errors that hid each other.
    """

    reference_impedance_ohm: float = 50.0

    def __post_init__(self) -> None:
        if not self.reference_impedance_ohm > 0.0:
            raise ValueError("reference_impedance_ohm must be positive")

    @property
    def volts_per_sqrt_watt(self) -> float:
        return math.sqrt(self.reference_impedance_ohm)


@dataclass(frozen=True, slots=True)
class NoiseSpec:
    """Thermal and oscillator noise, in physical units rather than raw sigmas.

    **Two limitations of the phase-noise model are RECORDED here, not fixed.**
    Changing either is a numerical change that needs its own decision, and
    writing a number down that the model cannot produce is worse than leaving
    the model as it is:

    1. The Wiener accumulation assumes a UNIFORM sample spacing. A real FMCW
       time base is not uniform - ``chirp_period_s = idle + ramp_end`` exceeds
       ``num_samples * sample_period_s`` - and a Wiener step scales as
       ``sigma_w^2 = 2 pi linewidth dt`` with the ACTUAL ``dt``, so the idle gap
       needs a larger step than a sample gap. As implemented, the slow-time
       phase-noise correlation is wrong by the duty-cycle factor.

    2. RANGE CORRELATION IS ABSENT. In a homodyne FMCW receiver the local
       oscillator and the echo come from the same source, so phase noise at the
       beat output is suppressed by ``4 sin^2(pi f tau)`` and is essentially
       cancelled at short range. An uncorrelated random walk therefore grossly
       OVERSTATES close-range phase noise. No absolute phase-noise level test
       may be written against this model until that is decided; the tested claim
       is the ``-20 dB/decade`` asymptote of the generator itself, which is what
       the generator actually promises.

    The model is a free-running oscillator, whose single-sideband phase-noise
    spectrum is ``L(f) = sigma_w^2 fs / (4 pi^2 f^2)`` for ``f`` well above the
    linewidth. That is a ``-20 dB/decade`` slope and it does NOT model the
    close-in ``1/f^3`` region or the far-out noise floor. A multi-region mask
    needs a shaped-PSD generator, which is a different model.
    """

    noise_figure_db: float = 0.0
    antenna_temperature_k: float = REFERENCE_TEMPERATURE_K
    bandwidth_hz: float = 0.0
    phase_noise_dbc_per_hz: float | None = None
    phase_offset_hz: float = 0.0
    phase_sample_rate_hz: float = 0.0

    def __post_init__(self) -> None:
        if self.antenna_temperature_k < 0.0:
            raise ValueError("antenna_temperature_k must be non-negative")
        if self.bandwidth_hz < 0.0:
            raise ValueError("bandwidth_hz must be non-negative")
        if self.phase_noise_dbc_per_hz is not None:
            if not self.phase_offset_hz > 0.0:
                raise ValueError(
                    "phase_offset_hz must be positive when phase_noise_dbc_per_hz "
                    "is given; L(f_off) says nothing without the offset it was "
                    "measured at"
                )
            if not self.phase_sample_rate_hz > 0.0:
                raise ValueError(
                    "phase_sample_rate_hz must be positive when "
                    "phase_noise_dbc_per_hz is given; a Wiener step is a rate, "
                    "and the rate is what turns dBc/Hz into a per-step variance"
                )

    @property
    def noise_factor(self) -> float:
        """``F = 10^(NF_dB/10)``, linear."""

        return 10.0 ** (float(self.noise_figure_db) / 10.0)

    @property
    def system_noise_temperature_k(self) -> float:
        """``T_sys = T_ant + T0 (F - 1)``, input-referred.

        With ``T_ant = T0`` this collapses to ``T0 F``, which is the identity
        every noise-figure datasheet quotes.
        """

        return self.antenna_temperature_k + REFERENCE_TEMPERATURE_K * (
            self.noise_factor - 1.0
        )

    @property
    def noise_power_watts(self) -> float:
        """``k T_sys B``, the total noise power in the stated bandwidth."""

        return (
            BOLTZMANN_J_PER_K * self.system_noise_temperature_k * float(self.bandwidth_hz)
        )

    def thermal_sigma_volts(self, port: PortSpec) -> float:
        """``sqrt(k T_sys B R / 2)``: the per-COMPONENT standard deviation.

        The noise is circularly symmetric complex Gaussian with total variance
        ``k T_sys B R``, split evenly between the real and imaginary parts, so
        each component gets half. Returning the per-component value rather than
        the total is deliberate: the kernel draws two independent normals per
        element and this is what it multiplies them by.
        """

        return math.sqrt(
            0.5 * self.noise_power_watts * port.reference_impedance_ohm
        )

    @property
    def phase_innovation_sigma_rad(self) -> float:
        """``sqrt(10^(L/10) 4 pi^2 f_off^2 / fs)``, the Wiener step.

        Zero when no phase noise is configured, which the runtime turns into a
        scan of exactly zero phase rather than a skipped stage - so that the
        thermal realisation is unaffected either way.
        """

        if self.phase_noise_dbc_per_hz is None:
            return 0.0
        level = 10.0 ** (float(self.phase_noise_dbc_per_hz) / 10.0)
        variance = (
            level
            * 4.0
            * math.pi**2
            * float(self.phase_offset_hz) ** 2
            / float(self.phase_sample_rate_hz)
        )
        return math.sqrt(variance)

    def single_sideband_dbc_per_hz(self, offset_hz: float) -> float:
        """The model's own ``L(f)``, in dBc/Hz, at an arbitrary offset.

        ``L(f) = sigma_w^2 fs / (4 pi^2 f^2)``. Published so a test can assert
        the ``-20 dB/decade`` asymptote against the generator rather than
        against a second copy of the formula.
        """

        sigma = self.phase_innovation_sigma_rad
        if sigma <= 0.0 or offset_hz <= 0.0:
            raise ValueError(
                "single_sideband_dbc_per_hz needs a configured phase noise and a "
                "positive offset"
            )
        level = (
            sigma**2
            * float(self.phase_sample_rate_hz)
            / (4.0 * math.pi**2 * float(offset_hz) ** 2)
        )
        return 10.0 * math.log10(level)


@dataclass(frozen=True, slots=True)
class LnaSpec:
    """A voltage gain in dB. Applied AFTER thermal noise, always."""

    gain_db: float = 0.0

    @property
    def voltage_gain(self) -> float:
        return 10.0 ** (float(self.gain_db) / 20.0)


@dataclass(frozen=True, slots=True)
class AgcSpec:
    """Automatic gain control, and the reason physics tests turn it off.

    The gain depends on the signal's own RMS, so the frontend is NOT linear in
    the signal and the cross-waveform linearity invariant does not hold with AGC
    on. That is a tested fact rather than a footnote. The measured gain stays a
    DEVICE tensor: reading it to build a Python scalar would be a per-frame
    device-to-host transfer.
    """

    target_rms: float
    mode: str = AGC_MODE_PER_RX
    min_gain_db: float = -60.0
    max_gain_db: float = 60.0

    def __post_init__(self) -> None:
        if not self.target_rms > 0.0:
            raise ValueError("target_rms must be positive")
        if self.mode not in AGC_MODES:
            raise ValueError(f"mode must be one of {list(AGC_MODES)}, got {self.mode!r}")
        if self.min_gain_db > self.max_gain_db:
            raise ValueError("min_gain_db must not exceed max_gain_db")

    @property
    def min_gain(self) -> float:
        return 10.0 ** (float(self.min_gain_db) / 20.0)

    @property
    def max_gain(self) -> float:
        return 10.0 ** (float(self.max_gain_db) / 20.0)


@dataclass(frozen=True, slots=True)
class AdcSpec:
    """Uniform mid-tread quantisation, and the ONLY quantiser in the chain.

    ``round`` is not differentiable and this family has no backward and no jvp
    on purpose. A straight-through surrogate is a Phase-9 modelling decision,
    not a detail a frontend may choose, so the owner raises on a grad-enabled or
    forward-dual input rather than silently detaching.
    """

    bits: int
    full_scale: float

    def __post_init__(self) -> None:
        if self.bits < 1 or self.bits > 30:
            raise ValueError("bits must lie in [1, 30]")
        if not self.full_scale > 0.0:
            raise ValueError("full_scale must be positive")

    @property
    def step(self) -> float:
        """``2 FS / (2^b - 1)``."""

        return 2.0 * float(self.full_scale) / (2**int(self.bits) - 1)

    @property
    def quantization_variance(self) -> float:
        """``step^2 / 12`` PER COMPONENT, for a busy non-overloaded signal."""

        return self.step**2 / 12.0

    @property
    def full_scale_sine_sqnr_db(self) -> float:
        """``6.02 b + 1.76`` dB, the textbook full-scale sine figure."""

        return 6.02 * int(self.bits) + 1.76


@dataclass(frozen=True, slots=True)
class SeedSpec:
    """One base seed; every stage derives its own stream from it.

    Never one generator threaded through the chain. A shared generator consumes
    draws as it goes, so enabling phase noise SHIFTS the thermal realisation and
    a differential measurement ends up comparing two different noise
    realisations while believing it isolated one stage.
    """

    seed_base: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.seed_base, int) or isinstance(self.seed_base, bool):
            raise TypeError("seed_base must be an int")
        if self.seed_base < 0:
            raise ValueError("seed_base must be non-negative")


@dataclass(frozen=True, slots=True)
class FrontendSpec:
    """The whole receive chain. Every stage optional; the ORDER is not.

    Enabling a stage is a non-``None`` field. The sequence they run in is fixed
    by the runtime and is not expressible here, which is the difference between
    this and the two runtimes it replaces.
    """

    port: PortSpec = PortSpec()
    noise: NoiseSpec | None = None
    lna: LnaSpec | None = None
    agc: AgcSpec | None = None
    adc: AdcSpec | None = None
    seed: SeedSpec = SeedSpec()

    @property
    def applies_noise_stage(self) -> bool:
        """Whether the fused phase/thermal/LNA operator has anything to do."""

        return self.noise is not None or self.lna is not None

    def thermal_sigma_volts(self) -> float:
        if self.noise is None:
            return 0.0
        return self.noise.thermal_sigma_volts(self.port)

    def phase_sigma_rad(self) -> float:
        if self.noise is None:
            return 0.0
        return self.noise.phase_innovation_sigma_rad

    def lna_voltage_gain(self) -> float:
        return 1.0 if self.lna is None else self.lna.voltage_gain


__all__ = [
    "AGC_MODES",
    "AGC_MODE_GLOBAL",
    "AGC_MODE_PER_RX",
    "BOLTZMANN_J_PER_K",
    "FRONTEND_STAGE_ORDER",
    "REFERENCE_TEMPERATURE_K",
    "STAGE_PHASE_NOISE",
    "STAGE_THERMAL_NOISE",
    "AdcSpec",
    "AgcSpec",
    "FrontendSpec",
    "LnaSpec",
    "NoiseSpec",
    "PortSpec",
    "SeedSpec",
]
