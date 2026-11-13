"""Typed receiver frontend and its fixed physical signal chain.

The module owns frontend configuration, deterministic stage seeding, native
phase/thermal/LNA and AGC operators, and final ADC quantization.
"""

from __future__ import annotations

import cmath
import math
import struct
from dataclasses import dataclass, field, replace

import torch

from .cuda import native_ops as _ops
from .policy import first_order_only, refuse_derivative, require_host_floats

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

AGC_MODE_GLOBAL = "global"
AGC_MODE_PER_RX = "per_rx"
AGC_MODES = (AGC_MODE_GLOBAL, AGC_MODE_PER_RX)

#: Launch width of the elementwise passes, threads per block. The Philox
#: realisation does not depend on it - a test builds a chain at another width
#: and asserts bit equality - and the Wiener scan is not affected by it at all,
#: because its accumulation order is part of the realisation.
DEFAULT_BLOCK_SIZE = 256


#: Why the port impedance has no derivative. It converts sqrt(W) to volts and
#: is a UNIT CONVENTION: differentiating a loss with respect to the units it is
#: expressed in is not a question about the world.
_PORT_REASON = (
    "reference_impedance_ohm is the unit convention that turns sqrt(W) into volts, applied exactly once at stage 0."
)

#: Why every noise scalar has no derivative. All four parameterise a
#: counter-based Philox draw in :class:`_FrontendNoise`; a pathwise derivative
#: through an RNG stream is not defined by any accepted contract here, and a
#: reparameterised noise model is a separate decision with its own ADR.
_NOISE_REASON = (
    "every scalar on Noise parameterises a counter-based Philox draw - the "
    "thermal sigma and the Wiener step are the standard deviations of a "
    "realisation, not a smooth function of the signal. A pathwise derivative "
    "through an RNG stream is not defined by any contract this package "
    "accepts, and a reparameterised noise model is a separate decision with "
    "its own ADR."
)

#: Why the LNA gain has no derivative TODAY. This one is a named deferral
#: rather than a statement that the derivative does not exist: the gain is a
#: smooth multiplicative factor and a leaf would be perfectly well defined. It
#: would need a new tangent and gradient slot in the native frontend operator,
#: and no consumer asks for one.

#: Why the AGC setpoint has no derivative. It is a control target, and the
#: stage is already non-linear in the signal because its gain depends on the
#: signal's own RMS - which is why every physics test turns AGC off.
_AGC_REASON = (
    "target_rms is a control setpoint. The AGC gain depends on the signal's "
    "own RMS, so the stage is not linear in the signal at all, and every "
    "physics invariant in this package is asserted with AGC off."
)

#: Why the ADC grid has no derivative. Both fields define ``round``'s step and
#: clip level, and :meth:`FrontendChain._quantize` already refuses a
#: differentiable SIGNAL. Refusing the grid as well keeps the wall in one
#: piece: a full-scale leaf would be a derivative of a staircase's placement.
_IQ_REASON = (
    "a demodulator imbalance is a measured property of the receiver, not a quantity a scene differentiates through"
)

_COMPRESSION_REASON = (
    "the intercept, the compression point and the AM/PM figure are measured device properties, not quantities a "
    "scene differentiates through; the stage stays differentiable in the signal it compresses"
)

_ADC_REASON = (
    "bits and full_scale define the quantiser's grid, and `round` has a zero "
    "derivative almost everywhere and an undefined one at every code "
    "boundary. The ADC stage already refuses a differentiable signal; the "
    "grid sits behind that same wall."
)


@dataclass(frozen=True, slots=True)
class Noise:
    """Thermal, oscillator and baseband flicker noise, in physical units.

    The scene-driven FMCW route evaluates one common oscillator at receive
    and delayed transmit times for each path before coherent summation. Idle
    gaps and frame gaps use their actual SI timestamps. Standalone ``apply``
    can model a free-running oscillator on an explicit grid, or on the uniform
    grid declared by ``phase_sample_rate``. These are different observables
    of the same continuous-time realization, not independent generators.

    The oscillator has three single-sideband regions, each optional and each
    declared by its own level at its own offset (R-ADR-031):

    * ``1/f^3`` (flicker FM): ``phase_flicker_density`` dBc/Hz at
      ``phase_flicker_offset``;
    * ``1/f^2`` (white FM, the free-running Wiener oscillator):
      ``phase_density`` dBc/Hz at ``phase_offset``,
      ``L(f) = sigma_w^2 fs / (4 pi^2 f^2)``;
    * the white floor: ``phase_floor`` dBc/Hz, held over cells of
      ``1 / phase_sample_rate``.

    The composite mask is their sum. All three are functions of absolute time,
    so the range-correlation of ``phi(t) - phi(t - tau)`` holds for every
    region; the ``1/f^2`` region is an exact Brownian bridge and is
    bit-identical whether the other two are declared or not.

    ``flicker_corner`` is the baseband (mixer and IF amplifier) ``1/f`` noise:
    an additive noise whose density equals the white thermal density at that
    frequency and rises as ``1/f`` below it. It needs the thermal stage, since
    the corner is defined relative to it.
    """

    #: Receiver noise figure, dB.
    figure: float = 0.0
    #: Antenna noise temperature, K.
    antenna_temperature: float = REFERENCE_TEMPERATURE_K
    #: Thermal noise bandwidth, Hz. ``None`` means the waveform's complex sample
    #: rate, which :class:`~witwin.radar.radar.Radar` resolves once when it
    #: builds the chain: every waveform publishes physical sqrt(W) samples and
    #: a white complex stream at ``f_s`` has noise bandwidth ``f_s``. An
    #: explicit value is an equivalent noise bandwidth whose variance ``k T B``
    #: is drawn per sample; the draws stay white, so a value above ``f_s``
    #: describes aliased noise and one below it a filter this model does not
    #: correlate. Zero is refused, because a zero bandwidth is a silently
    #: noiseless receiver.
    bandwidth: float | None = None
    #: Single-sideband oscillator phase-noise density in the ``1/f^2`` region,
    #: at ``phase_offset``, dBc/Hz. ``None`` means no white-FM noise.
    phase_density: float | None = None
    #: The offset ``phase_density`` was measured at, Hz.
    phase_offset: float | None = None
    #: Wiener step rate and floor hold-cell rate, Hz. ``None`` means the
    #: waveform's sample rate.
    phase_sample_rate: float | None = None
    #: Single-sideband density in the ``1/f^3`` region at
    #: ``phase_flicker_offset``, dBc/Hz. ``None`` means no flicker FM.
    phase_flicker_density: float | None = None
    #: The offset ``phase_flicker_density`` was measured at, Hz.
    phase_flicker_offset: float | None = None
    #: Far-out white phase-noise floor, dBc/Hz. ``None`` means no floor.
    phase_floor: float | None = None
    #: Baseband ``1/f`` corner frequency, Hz. ``None`` means no flicker.
    flicker_corner: float | None = None

    def __post_init__(self) -> None:
        require_host_floats(
            "Noise",
            _NOISE_REASON,
            figure=self.figure,
            antenna_temperature=self.antenna_temperature,
            bandwidth=self.bandwidth,
            phase_density=self.phase_density,
            phase_offset=self.phase_offset,
            phase_sample_rate=self.phase_sample_rate,
            phase_flicker_density=self.phase_flicker_density,
            phase_flicker_offset=self.phase_flicker_offset,
            phase_floor=self.phase_floor,
            flicker_corner=self.flicker_corner,
        )
        if self.antenna_temperature < 0.0:
            raise ValueError("antenna_temperature must be non-negative")
        if self.bandwidth is not None and not self.bandwidth > 0.0:
            raise ValueError(
                "bandwidth must be positive, or None to take the waveform's "
                "sampling bandwidth; zero is a receiver that adds no thermal "
                "noise while claiming a noise figure"
            )
        if self.phase_sample_rate is not None and not self.phase_sample_rate > 0.0:
            raise ValueError("phase_sample_rate must be positive, or None to take the waveform's sample rate")
        for density, offset, name in (
            (self.phase_density, self.phase_offset, "phase"),
            (self.phase_flicker_density, self.phase_flicker_offset, "phase_flicker"),
        ):
            if density is not None and offset is None:
                raise ValueError(
                    f"{name}_offset is required when {name}_density is given; "
                    "L(f_off) says nothing without the offset it was measured at"
                )
            if offset is not None and not offset > 0.0:
                raise ValueError(f"{name}_offset must be positive")
        if self.flicker_corner is not None and not self.flicker_corner > 0.0:
            raise ValueError("flicker_corner must be positive, or None for no baseband flicker noise")

    def resolved(self, *, bandwidth: float, sample_rate: float) -> Noise:
        """This record with its two waveform-dependent defaults filled in.

        ``bandwidth`` and ``sample_rate`` are the waveform's, supplied by the
        radar. An explicitly configured value always wins: the point of the
        ``None`` default is that the common case needs no number, not that the
        waveform overrules a caller who gave one.
        """

        if self.bandwidth is not None and self.phase_sample_rate is not None:
            return self
        return replace(
            self,
            bandwidth=self.bandwidth if self.bandwidth is not None else float(bandwidth),
            phase_sample_rate=(self.phase_sample_rate if self.phase_sample_rate is not None else float(sample_rate)),
        )

    @property
    def noise_factor(self) -> float:
        """``F = 10^(NF_dB/10)``, linear."""

        return 10.0 ** (float(self.figure) / 10.0)

    @property
    def system_noise_temperature_k(self) -> float:
        """``T_sys = T_ant + T0 (F - 1)``, input-referred.

        With ``T_ant = T0`` this collapses to ``T0 F``, which is the identity
        every noise-figure datasheet quotes.
        """

        return self.antenna_temperature + REFERENCE_TEMPERATURE_K * (self.noise_factor - 1.0)

    @property
    def noise_power_watts(self) -> float:
        """``k T_sys B``, the total noise power in the stated bandwidth."""

        if self.bandwidth is None:
            raise ValueError("bandwidth is unresolved; Radar fills it from the waveform before building a chain")
        return BOLTZMANN_J_PER_K * self.system_noise_temperature_k * float(self.bandwidth)

    def thermal_sigma_volts(self, impedance: float) -> float:
        """``sqrt(k T_sys B R / 2)``: the per-COMPONENT standard deviation.

        The noise is circularly symmetric complex Gaussian with total variance
        ``k T_sys B R``, split evenly between the real and imaginary parts, so
        each component gets half. Returning the per-component value rather than
        the total is deliberate: the kernel draws two independent normals per
        element and this is what it multiplies them by.
        """

        return math.sqrt(0.5 * self.noise_power_watts * float(impedance))

    def flicker_level_volts(self, impedance: float) -> float:
        """``h = sqrt(ln2 f_c k T_sys R)`` V, the per-component Haar height.

        The baseband flicker generator is a sum of Haar wavelets of height ``h``
        in every octave; its per-component two-sided density is
        ``h^2 / (2 ln2 f)``, so the complex density ``h^2 / (ln2 f)`` equals the
        thermal density ``k T_sys R`` at ``f = f_c``. Zero when undeclared.
        """

        if self.flicker_corner is None:
            return 0.0
        density = BOLTZMANN_J_PER_K * self.system_noise_temperature_k * float(impedance)
        return math.sqrt(math.log(2.0) * float(self.flicker_corner) * density)

    @property
    def phase_innovation_sigma_rad(self) -> float:
        """``sqrt(10^(L/10) 4 pi^2 f_off^2 / fs)``, the Wiener step.

        Zero when no ``1/f^2`` phase noise is configured.
        """

        if self.phase_density is None:
            return 0.0
        level = 10.0 ** (float(self.phase_density) / 10.0)
        variance = level * 4.0 * math.pi**2 * float(self.phase_offset) ** 2 / float(self.phase_sample_rate)
        return math.sqrt(variance)

    @property
    def phase_flicker_scale_rad(self) -> float:
        """``c = sqrt(2 pi^2 ln2 f3^3 L(f3))``, the flicker-FM tent amplitude per second."""

        if self.phase_flicker_density is None:
            return 0.0
        level = 10.0 ** (float(self.phase_flicker_density) / 10.0)
        return math.sqrt(2.0 * math.pi**2 * math.log(2.0) * float(self.phase_flicker_offset) ** 3 * level)

    @property
    def phase_floor_sigma_rad(self) -> float:
        """``s = sqrt(L_floor f_cell)``, the floor's per-cell standard deviation."""

        if self.phase_floor is None:
            return 0.0
        return math.sqrt(10.0 ** (float(self.phase_floor) / 10.0) * float(self.phase_sample_rate))

    @property
    def has_oscillator_noise(self) -> bool:
        """Whether any of the three oscillator regions is declared."""

        return self.phase_density is not None or self.phase_flicker_density is not None or self.phase_floor is not None

    def phase_difference(
        self,
        times_s,
        delays_s,
        *,
        seed_base: int,
        sign: float = 1.0,
        block_size: int = DEFAULT_BLOCK_SIZE,
        invalid: torch.Tensor | None = None,
    ):
        """Common-oscillator ``phi(t)-phi(t-tau)`` in radians, on actual SI time.

        A white-frequency-noise oscillator has diffusion q=4*pi^2*f0^2*L(f0)
        [rad^2/s]. The phase-difference covariance is q times the intersection
        length of the two delay intervals; its PSD is 4*sin^2(pi*f*tau) times
        the oscillator PSD. The native expansion shares one realization across
        paths, antennas, queries and frames, independent of query order; the
        flicker and floor regions are keyed by the same absolute time.

        Oscillator phase is not differentiable in time/delay. Geometry AD is
        explicitly refused; signal/RCS derivatives at fixed delays remain valid.
        Forty dyadic levels resolve 0.91 ps. Oracles:
        test_correlated_phase_noise.py and test_shaped_phase_noise.py.

        A non-finite time or delay, or a negative delay, is never asserted on
        the device. The kernel writes NaN for that query and counts it in
        ``invalid``. Without ``invalid`` the count is read back here and a
        nonzero one raises ``ValueError`` (one host synchronisation); a caller
        that passes its own counter decodes it later with
        :func:`raise_on_invalid_oscillator_queries`, once per frame.
        """
        refuse_derivative(
            "frontend oscillator time queries",
            "oscillator phase is nowhere differentiable in time; fixed-delay signal derivatives remain supported.",
            timestamps=times_s,
            path_delays=delays_s,
        )
        if times_s.shape != delays_s.shape or times_s.device != delays_s.device:
            raise ValueError("oscillator times and delays must share shape and device")
        if not times_s.is_cuda or seed_base < 0 or sign not in (-1.0, 1.0):
            raise ValueError("oscillator needs CUDA times, a nonnegative seed and sign +/-1")
        times = times_s.to(torch.float64).contiguous()
        delays = delays_s.to(torch.float64).contiguous()
        counter = torch.zeros(1, dtype=torch.int32, device=times.device) if invalid is None else invalid
        phase = torch.empty(times.shape, device=times.device, dtype=torch.float32)
        diffusion = self.phase_innovation_sigma_rad**2 * float(self.phase_sample_rate)
        _ops().oscillator_phase_forward(
            times,
            delays,
            phase,
            counter,
            diffusion,
            self.phase_flicker_scale_rad,
            self.phase_floor_sigma_rad,
            float(self.phase_sample_rate),
            sign,
            seed_base,
            _checked_block_size(block_size),
        )
        if invalid is None:
            raise_on_invalid_oscillator_queries(counter)
        return phase


def raise_on_invalid_oscillator_queries(counter: torch.Tensor) -> None:
    """Decode the oscillator's device counter on the host, and raise if nonzero.

    This is the one host read of the oscillator family. It replaces a device
    assertion, which would leave the CUDA context unusable for the rest of the
    process; a counter read leaves the context intact and names the problem.
    """

    count = int(counter.item())
    if count:
        counter.zero_()
        raise ValueError(
            f"{count} oscillator queries had a non-finite time or delay, or a negative delay; their phase is NaN. "
            "Oscillator times and path delays must be finite and delays non-negative"
        )


@dataclass(frozen=True, slots=True)
class Agc:
    """Automatic gain control, and the reason physics tests turn it off.

    The gain depends on the signal's own RMS, so the frontend is NOT linear in
    the signal and the cross-waveform linearity invariant does not hold with AGC
    on. That is a tested fact rather than a footnote. The measured gain stays a
    DEVICE tensor: reading it to build a Python scalar would be a per-frame
    device-to-host transfer.

    ``time_constant`` selects the loop. ``None`` is a frame-static AGC: one gain
    per group, measured over the whole frame. A positive value is a
    time-dynamic AGC: one gain per group and slow-time block (a chirp index, an
    OFDM symbol or a pulse), driven by a causal first-order loop

        ``l_k = clamp((1 - beta) l_{k-1} + beta ln(target / r_{k-1}))``

    in log gain, with ``beta = 1 - exp(-T_block / time_constant)`` and the
    first block acquired at its own ideal gain. ``block_period`` is the block
    duration ``T_block``; ``None`` takes the waveform's slow-time period, which
    the radar resolves. The loop state does not carry across frames.
    """

    #: Target output RMS, volts.
    target_rms: float
    mode: str = AGC_MODE_PER_RX
    #: Gain limits, dB. The linear factors the kernel takes are the two
    #: ``*_voltage_gain`` properties.
    min_gain: float = -60.0
    max_gain: float = 60.0
    #: Loop time constant, s. ``None`` is a frame-static AGC.
    time_constant: float | None = None
    #: Slow-time block duration, s. ``None`` takes the waveform's.
    block_period: float | None = None

    def __post_init__(self) -> None:
        require_host_floats(
            "Agc",
            _AGC_REASON,
            target_rms=self.target_rms,
            min_gain=self.min_gain,
            max_gain=self.max_gain,
            time_constant=self.time_constant,
            block_period=self.block_period,
        )
        if not self.target_rms > 0.0:
            raise ValueError("target_rms must be positive")
        if self.mode not in AGC_MODES:
            raise ValueError(f"mode must be one of {list(AGC_MODES)}, got {self.mode!r}")
        if self.min_gain > self.max_gain:
            raise ValueError("min_gain must not exceed max_gain")
        if self.time_constant is not None and not self.time_constant > 0.0:
            raise ValueError("time_constant must be positive, or None for a frame-static AGC")
        if self.block_period is not None and not self.block_period > 0.0:
            raise ValueError("block_period must be positive, or None to take the waveform's slow-time period")

    def resolved(self, *, block_period: float) -> Agc:
        """This record with the waveform's block period filled in, if it is dynamic."""

        if self.time_constant is None or self.block_period is not None:
            return self
        return replace(self, block_period=float(block_period))

    @property
    def is_dynamic(self) -> bool:
        return self.time_constant is not None

    @property
    def loop_coefficient(self) -> float:
        """``beta = 1 - exp(-T_block / tau)``, the per-block loop update weight."""

        if self.time_constant is None or self.block_period is None:
            raise ValueError("a frame-static or unresolved AGC has no loop coefficient")
        return -math.expm1(-float(self.block_period) / float(self.time_constant))

    @property
    def min_voltage_gain(self) -> float:
        return 10.0 ** (float(self.min_gain) / 20.0)

    @property
    def max_voltage_gain(self) -> float:
        return 10.0 ** (float(self.max_gain) / 20.0)


@dataclass(frozen=True, slots=True)
class Compression:
    """Receiver gain compression: a memoryless polynomial with AM/PM.

    Placed after the LNA, where a receiver compresses, and referred to the
    receiver INPUT like a datasheet's figures: with ``u = P_in / P_IIP3`` the
    complex gain is

        ``G(u) = (1 - u + c5 u^2) exp(j k u)``

    and the output amplitude is held at its first maximum beyond it (a soft
    limiter), so the stage is monotone with a continuous derivative. The cubic
    term is the intercept's: a cubic alone puts the 1-dB compression point
    9.64 dB below ``iip3``. A declared ``p1db`` solves ``c5`` so that
    ``|G| = -1 dB`` at it; ``am_pm`` is the phase rotation at that point,
    degrees (``k = am_pm / u_1dB``). Powers are the complex envelope's
    ``|x|^2 / R`` at the receive port, in dBm.

    The stage is differentiable in the signal (an analytic 2x2 Jacobian in the
    native kernel). Its parameters are device configuration and refuse a
    derivative.
    """

    #: Input-referred third-order intercept point, dBm.
    iip3: float
    #: Input-referred 1-dB compression point, dBm. ``None`` is the cubic's own.
    p1db: float | None = None
    #: AM/PM conversion at the 1-dB compression point, degrees.
    am_pm: float = 0.0

    def __post_init__(self) -> None:
        require_host_floats("Compression", _COMPRESSION_REASON, iip3=self.iip3, p1db=self.p1db, am_pm=self.am_pm)
        if self.p1db is not None and not self.p1db < self.iip3:
            raise ValueError("p1db must lie below iip3; a 1-dB point above the intercept is not a compressing stage")
        self.coefficients  # noqa: B018 - validates that the declared pair has a saturation point

    @property
    def one_db_ratio(self) -> float:
        """``u_1dB = P1dB / P_IIP3``, linear."""

        if self.p1db is None:
            return 1.0 - 10.0 ** (-1.0 / 20.0)
        return 10.0 ** ((float(self.p1db) - float(self.iip3)) / 10.0)

    @property
    def coefficients(self) -> tuple[float, float, float]:
        """``(c5, k, u_sat)``: the quintic term, the AM/PM slope and the hold point."""

        u1 = self.one_db_ratio
        c5 = 0.0 if self.p1db is None else (10.0 ** (-1.0 / 20.0) - 1.0 + u1) / (u1 * u1)
        # The first stationary point of sqrt(u) (1 - u + c5 u^2): 1 - 3u + 5 c5 u^2 = 0.
        if c5 == 0.0:
            u_sat = 1.0 / 3.0
        else:
            discriminant = 9.0 - 20.0 * c5
            if discriminant <= 0.0:
                raise ValueError(
                    f"p1db={self.p1db} dBm with iip3={self.iip3} dBm has no saturation point: the solved quintic "
                    "term makes the output grow without bound. Move p1db closer to iip3 - 9.64 dB"
                )
            u_sat = (3.0 - math.sqrt(discriminant)) / (10.0 * c5)
        if not u1 < u_sat or not 1.0 - u_sat + c5 * u_sat * u_sat > 0.0:
            raise ValueError(f"p1db={self.p1db} dBm lies beyond this stage's saturation point; it cannot be reached")
        k = math.radians(float(self.am_pm)) / u1
        return c5, k, u_sat

    def input_power_scale(self, *, impedance: float, lna_voltage_gain: float) -> float:
        """``1 / (g^2 P_IIP3 R)`` [1/V^2]: ``u`` from the post-LNA volts the stage sees."""

        watts = 1e-3 * 10.0 ** (float(self.iip3) / 10.0)
        return 1.0 / (float(lna_voltage_gain) ** 2 * watts * float(impedance))


@dataclass(frozen=True, slots=True)
class Iq:
    """Quadrature demodulator gain and phase imbalance.

    A perfect quadrature split produces ``y = x``. A real one splits with a
    gain ratio ``g`` and a phase error ``phi`` between the two branches, which
    leaks a conjugate copy of the signal through:

        ``y = alpha * x + beta * conj(x)``
        ``alpha = (1 + g * exp(-j phi)) / 2``
        ``beta  = (1 - g * exp(+j phi)) / 2``

    The observable is the conjugate term, and it matters because ``conj`` is a
    reflection of the Doppler axis: a target approaching at ``+v`` also appears
    receding at ``-v``, at ``image_rejection_db`` below the true return. A
    classifier whose label IS the sign of the velocity can learn to trust a
    clean mirror axis that no real receiver has.

    ``gain`` is ``20 log10(g)``, dB, and ``phase`` is ``phi``, rad. Both zero is
    a perfect demodulator, which is what ``None`` on the radar already means,
    so a zeroed record is accepted rather than refused: it is the degenerate
    end of a sweep, not a mistake.
    """

    #: Branch gain ratio, dB.
    gain: float = 0.0
    #: Branch phase error, rad.
    phase: float = 0.0

    def __post_init__(self) -> None:
        require_host_floats("Iq", _IQ_REASON, gain=self.gain, phase=self.phase)

    @property
    def alpha(self) -> complex:
        """The direct coefficient."""

        ratio = 10.0 ** (float(self.gain) / 20.0)
        phi = float(self.phase)
        return (1.0 + ratio * cmath.exp(-1j * phi)) / 2.0

    @property
    def beta(self) -> complex:
        """The conjugate (image) coefficient."""

        ratio = 10.0 ** (float(self.gain) / 20.0)
        phi = float(self.phase)
        return (1.0 - ratio * cmath.exp(1j * phi)) / 2.0

    @property
    def image_rejection_db(self) -> float:
        """``-20 log10|beta / alpha|``: how far the mirror sits below the truth.

        Published so a test can assert the image's measured level against the
        declaration rather than against a number spelled twice.
        """

        magnitude = abs(self.beta) / abs(self.alpha)
        return math.inf if magnitude == 0.0 else -20.0 * math.log10(magnitude)


@dataclass(frozen=True, slots=True)
class Adc:
    """Uniform mid-tread quantisation, and the ONLY quantiser in the chain.

    The grid is a two's-complement converter's, per component:
    ``step = 2 FS / 2^b`` and the codes are ``k = -2^(b-1) ... 2^(b-1) - 1``,
    so the levels are ``k * step``. Zero IS a code - an idle channel reads
    exactly zero, with no half-step DC offset - and the grid spans
    ``[-FS, FS - step]``. An input whose code falls outside that range
    saturates at the end code and is counted as clipped; the no-overload range
    is therefore ``[-FS - step/2, FS - step/2)``. Rounding is half-to-even.

    ``round`` is not differentiable and this family has no backward and no jvp
    on purpose. A straight-through surrogate is a modelling decision, not a
    detail a frontend may choose, so the owner raises on a grad-enabled or
    forward-dual input rather than silently detaching.
    """

    bits: int
    full_scale: float

    def __post_init__(self) -> None:
        require_host_floats("Adc", _ADC_REASON, bits=self.bits, full_scale=self.full_scale)
        if self.bits < 2 or self.bits > 30:
            raise ValueError(
                "bits must lie in [2, 30]; a one-bit mid-tread converter has the codes {-FS, 0} and "
                "cannot represent a positive sample"
            )
        if not self.full_scale > 0.0:
            raise ValueError("full_scale must be positive")

    @property
    def step(self) -> float:
        """``2 FS / 2^b``, volts per code."""

        return 2.0 * float(self.full_scale) / (2 ** int(self.bits))


@dataclass(frozen=True, slots=True)
class FrontendSpec:
    """The whole receive chain. Every stage optional; the ORDER is not.

    Internal: :class:`~witwin.radar.radar.Radar` builds one of these from its
    own flat fields and hands it to :class:`FrontendChain`. It is not a public
    constructor, which is why the one-number stages are fields here rather
    than records a caller would have to assemble.

    Enabling a stage is a non-``None`` field. The sequence they run in is fixed
    by the runtime and is not expressible here.

    ``seed`` is ONE base seed and every stage derives its own stream from it,
    never one generator threaded through the chain. A shared generator consumes
    draws as it goes, so enabling phase noise would SHIFT the thermal
    realisation and a differential measurement would compare two different
    realisations while believing it isolated one stage.

    ``impedance`` is the sqrt(W) to volt conversion applied exactly once, at
    stage 0: a synthesis cube is in sqrt(W) at the receive antenna port and
    everything downstream of the LNA is in volts, so ``v = sqrt(W) sqrt(R)``.

    ``lna`` is a voltage gain in dB: a host float, or a 0-dim floating tensor
    that is a LEAF. Every stage between the LNA and the AGC is homogeneous of
    degree one in the gain - the thermal noise is input-referred, compression
    is input-referred, the quadrature split is linear and the flicker level
    scales with it - so the chain's output there is
    ``(g / g_host) (y(g_host) - dc) + dc`` exactly, with the DC offset the one
    term the gain does not scale. The chain runs at the host value and applies
    that unit factor (value exactly one) once; the AGC and the converter then
    see a differentiable signal like any other (R-ADR-052).

    ``tx_coupling`` and ``rx_coupling`` are the array's port coupling matrices
    (``C[i][j]``: element ``i``'s port response to a unit excitation of
    element ``j``), ``dc_offset`` is one complex baseband offset per receiver
    in volts at the mixer output, and ``timeline`` is ``(slow_period_s,
    fast_start_s, fast_period_s)``: the receive timestamps the baseband flicker
    process is evaluated on, ``t = t_frame + s T_slow + t_0 + m T_fast``.
    """

    noise: Noise | None = None
    lna: float | None = None
    #: Quadrature imbalance. ``None`` is a perfect demodulator.
    iq: Iq | None = None
    agc: Agc | None = None
    adc: Adc | None = None
    impedance: float = 50.0
    seed: int = 0
    compression: Compression | None = None
    dc_offset: tuple[complex, ...] | None = None
    tx_coupling: tuple[tuple[complex, ...], ...] | None = None
    rx_coupling: tuple[tuple[complex, ...], ...] | None = None
    timeline: tuple[float, float, float] | None = None
    #: The LNA gain's host value, dB, read once at construction.
    lna_host: float | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        require_host_floats("FrontendSpec", _PORT_REASON, impedance=self.impedance)
        if not self.impedance > 0.0:
            raise ValueError("impedance must be positive")
        if self.lna is not None:
            if isinstance(self.lna, torch.Tensor):
                if self.lna.dim() != 0 or not self.lna.is_floating_point():
                    raise ValueError("a tensor lna must be a 0-dim floating tensor, dB")
                host = float(self.lna.detach())
            else:
                host = float(self.lna)
            if not math.isfinite(host):
                raise ValueError("lna must be finite, dB")
            object.__setattr__(self, "lna_host", host)
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("seed must be an int")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.agc is not None and self.agc.is_dynamic and self.agc.block_period is None:
            raise ValueError("a time-dynamic Agc needs its block_period resolved before a chain is built")
        if self.noise is not None and self.noise.flicker_corner is not None and self.noise.bandwidth is None:
            raise ValueError("baseband flicker is defined against the thermal density; resolve Noise first")

    @property
    def volts_per_sqrt_watt(self) -> float:
        return math.sqrt(float(self.impedance))

    @property
    def applies_noise_stage(self) -> bool:
        """Whether the fused phase/thermal/LNA operator has anything to do."""

        return self.noise is not None or self.lna is not None

    @property
    def applies_baseband_stage(self) -> bool:
        return self.dc_offset is not None or (self.noise is not None and self.noise.flicker_corner is not None)

    def thermal_sigma_volts(self) -> float:
        if self.noise is None:
            return 0.0
        return self.noise.thermal_sigma_volts(self.impedance)

    def lna_voltage_gain(self) -> float:
        return 1.0 if self.lna_host is None else 10.0 ** (self.lna_host / 20.0)

    def lna_leaf_scale(self, device) -> torch.Tensor | None:
        """``g / g_host`` as a 0-dim tensor (value exactly one) for a tensor ``lna``, else ``None``."""

        if not isinstance(self.lna, torch.Tensor):
            return None
        gain = self.lna.to(device=device, dtype=torch.float32)
        return torch.pow(10.0, (gain - gain.detach()) / 20.0)


def _checked_block_size(block_size: int) -> int:
    """A launch width the elementwise kernels accept: a power of two in [32, 1024]."""

    block = int(block_size)
    if block < 32 or block > 1024 or (block & (block - 1)) != 0:
        raise ValueError(f"block_size must be a power of two between 32 and 1024, got {block}")
    return block


def _require_no_derivative(signal: torch.Tensor, stage: str) -> None:
    """Refuse a differentiable input to a non-differentiable stage, loudly.

    Silently detaching would return a number with no gradient where the caller
    asked for one, which is the failure a fail-loud contract exists to prevent.

    The check itself is :func:`witwin.radar.policy.refuse_derivative`, the
    ONE owner of the non-differentiability wall; what stays here is the ADC's
    own reason - why ``round`` has no derivative.
    """

    refuse_derivative(
        f"the frontend {stage} stage",
        "`round` has a zero derivative almost everywhere and an undefined one "
        "at every code boundary, and a straight-through surrogate is a "
        "modelling decision rather than something the frontend may choose: "
        "detach the signal before the ADC, or run without one.",
        signal=signal,
    )


def frame_seed(seed: int, frame_time_s: float) -> int:
    """The Philox key of one frame: the base seed mixed with the frame instant.

    Every frame of a run draws its own thermal, floor and flicker realisation,
    and that realisation is a function of the frame's TIME alone, so
    ``simulate``, ``stream`` and ``echo`` agree frame by frame and a frame at
    ``t = 0`` keeps the base seed exactly. The mix is MurmurHash3's 64-bit
    finaliser on the IEEE-754 bits, which maps zero to zero.
    """

    bits = int.from_bytes(struct.pack("<d", float(frame_time_s)), "little")
    mask = (1 << 64) - 1
    bits ^= bits >> 33
    bits = (bits * 0xFF51AFD7ED558CCD) & mask
    bits ^= bits >> 33
    bits = (bits * 0xC4CEB9FE1A85EC53) & mask
    bits ^= bits >> 33
    return (int(seed) ^ bits) & ((1 << 63) - 1)


@dataclass(frozen=True, slots=True)
class _NoisePlan:
    num_outer: int
    num_phase: int
    phase: torch.Tensor
    thermal_sigma: float
    lna_gain: float
    seed_base: int
    block_size: int


@dataclass(frozen=True, slots=True)
class _AgcPlan:
    dim0: int
    num_groups: int
    dim2: int
    target_rms: float
    min_gain: float
    max_gain: float
    block_size: int
    #: Slow-time blocks per group; 1 for a frame-static AGC.
    num_blocks: int = 1
    beta: float = 0.0


@dataclass(frozen=True, slots=True)
class _CompressionPlan:
    scale: float
    c5: float
    k_pm: float
    u_sat: float
    block_size: int

    def arguments(self) -> tuple:
        return (self.scale, self.c5, self.k_pm, self.u_sat, self.block_size)


class _FrontendNoise(torch.autograd.Function):
    """Phase rotation, thermal addition, and the LNA gain, in one launch pair."""

    @staticmethod
    def forward(x_re, x_im, plan):
        out_re = torch.empty_like(x_re)
        out_im = torch.empty_like(x_im)
        phase_rad = plan.phase
        _ops().frontend_noise_forward(
            x_re,
            x_im,
            out_re,
            out_im,
            phase_rad,
            plan.num_outer,
            plan.num_phase,
            plan.thermal_sigma,
            plan.lna_gain,
            plan.seed_base,
            STAGE_THERMAL_NOISE,
            plan.block_size,
        )
        return out_re, out_im, phase_rad

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, _, plan = inputs
        ctx.plan = plan
        # The derivative is taken at the phase the primal actually used, never
        # at a regenerated one: saving it makes the two exactly consistent and
        # removes a second copy of the generator from the backward path.
        ctx.save_for_backward(output[2])
        ctx.save_for_forward(output[2])
        ctx.mark_non_differentiable(output[2])

    @staticmethod
    @first_order_only
    def backward(ctx, grad_out_re, grad_out_im, grad_phase):
        (phase_rad,) = ctx.saved_tensors
        plan = ctx.plan
        zeros = None
        if grad_out_re is None or grad_out_im is None:
            zeros = torch.zeros((plan.num_outer, plan.num_phase), dtype=torch.float32, device=phase_rad.device)
        grad_re = zeros if grad_out_re is None else grad_out_re.contiguous()
        grad_im = zeros if grad_out_im is None else grad_out_im.contiguous()
        grad_x_re = torch.empty_like(grad_re)
        grad_x_im = torch.empty_like(grad_im)
        _ops().frontend_noise_backward(
            phase_rad,
            grad_re,
            grad_im,
            grad_x_re,
            grad_x_im,
            plan.num_outer,
            plan.num_phase,
            plan.lna_gain,
            plan.block_size,
        )
        return grad_x_re, grad_x_im, None

    @staticmethod
    def jvp(ctx, tan_x_re, tan_x_im, tan_plan):
        (phase_rad,) = ctx.saved_tensors
        plan = ctx.plan
        shape = (plan.num_outer, plan.num_phase)
        zeros = torch.zeros(shape, dtype=torch.float32, device=phase_rad.device)
        tan_re = zeros if tan_x_re is None else tan_x_re.contiguous()
        tan_im = zeros if tan_x_im is None else tan_x_im.contiguous()
        tan_out_re = torch.empty_like(tan_re)
        tan_out_im = torch.empty_like(tan_im)
        _ops().frontend_noise_jvp(
            phase_rad,
            tan_re,
            tan_im,
            tan_out_re,
            tan_out_im,
            plan.num_outer,
            plan.num_phase,
            plan.lna_gain,
            plan.block_size,
        )
        return tan_out_re, tan_out_im, None


class _FrontendCompress(torch.autograd.Function):
    """Receiver compression, with the analytic 2x2 Jacobian in both modes."""

    @staticmethod
    def forward(x_re, x_im, plan):
        out_re = torch.empty_like(x_re)
        out_im = torch.empty_like(x_im)
        _ops().frontend_compress_forward(x_re, x_im, out_re, out_im, *plan.arguments())
        return out_re, out_im

    @staticmethod
    def setup_context(ctx, inputs, output):
        x_re, x_im, plan = inputs
        ctx.plan = plan
        ctx.save_for_backward(x_re, x_im)
        ctx.save_for_forward(x_re, x_im)

    @staticmethod
    @first_order_only
    def backward(ctx, grad_out_re, grad_out_im):
        x_re, x_im = ctx.saved_tensors
        zeros = torch.zeros_like(x_re)
        grad_re = zeros if grad_out_re is None else grad_out_re.contiguous()
        grad_im = zeros if grad_out_im is None else grad_out_im.contiguous()
        grad_x_re = torch.empty_like(x_re)
        grad_x_im = torch.empty_like(x_im)
        _ops().frontend_compress_backward(x_re, x_im, grad_re, grad_im, grad_x_re, grad_x_im, *ctx.plan.arguments())
        return grad_x_re, grad_x_im, None

    @staticmethod
    def jvp(ctx, tan_x_re, tan_x_im, tan_plan):
        x_re, x_im = ctx.saved_tensors
        zeros = torch.zeros_like(x_re)
        tan_re = zeros if tan_x_re is None else tan_x_re.contiguous()
        tan_im = zeros if tan_x_im is None else tan_x_im.contiguous()
        tan_out_re = torch.empty_like(x_re)
        tan_out_im = torch.empty_like(x_im)
        _ops().frontend_compress_jvp(x_re, x_im, tan_re, tan_im, tan_out_re, tan_out_im, *ctx.plan.arguments())
        return tan_out_re, tan_out_im


class _FrontendAgc(torch.autograd.Function):
    """A measured gain and its application, with the gain kept on the device."""

    @staticmethod
    def forward(x_re, x_im, plan):
        out_re = torch.empty_like(x_re)
        out_im = torch.empty_like(x_im)
        gain = torch.empty(plan.num_groups, dtype=torch.float32, device=x_re.device)
        rms = torch.empty_like(gain)
        _ops().frontend_agc_forward(
            x_re,
            x_im,
            out_re,
            out_im,
            gain,
            rms,
            plan.dim0,
            plan.num_groups,
            plan.dim2,
            plan.target_rms,
            plan.min_gain,
            plan.max_gain,
            plan.block_size,
        )
        return out_re, out_im, gain, rms

    @staticmethod
    def setup_context(ctx, inputs, output):
        x_re, x_im, plan = inputs
        ctx.plan = plan
        ctx.save_for_backward(x_re, x_im, output[2], output[3])
        ctx.save_for_forward(x_re, x_im, output[2], output[3])
        ctx.mark_non_differentiable(output[2], output[3])

    @staticmethod
    @first_order_only
    def backward(ctx, grad_out_re, grad_out_im, grad_gain, grad_rms):
        x_re, x_im, gain, rms = ctx.saved_tensors
        plan = ctx.plan
        zeros = torch.zeros_like(x_re)
        grad_re = zeros if grad_out_re is None else grad_out_re.contiguous()
        grad_im = zeros if grad_out_im is None else grad_out_im.contiguous()
        grad_x_re = torch.empty_like(x_re)
        grad_x_im = torch.empty_like(x_im)
        inner = torch.empty_like(gain)
        _ops().frontend_agc_backward(
            x_re,
            x_im,
            gain,
            rms,
            grad_re,
            grad_im,
            grad_x_re,
            grad_x_im,
            inner,
            plan.dim0,
            plan.num_groups,
            plan.dim2,
            plan.target_rms,
            plan.min_gain,
            plan.max_gain,
            plan.block_size,
        )
        return grad_x_re, grad_x_im, None

    @staticmethod
    def jvp(ctx, tan_x_re, tan_x_im, tan_plan):
        x_re, x_im, gain, rms = ctx.saved_tensors
        plan = ctx.plan
        zeros = torch.zeros_like(x_re)
        tan_re = zeros if tan_x_re is None else tan_x_re.contiguous()
        tan_im = zeros if tan_x_im is None else tan_x_im.contiguous()
        tan_out_re = torch.empty_like(x_re)
        tan_out_im = torch.empty_like(x_im)
        inner = torch.empty_like(gain)
        _ops().frontend_agc_jvp(
            x_re,
            x_im,
            gain,
            rms,
            tan_re,
            tan_im,
            tan_out_re,
            tan_out_im,
            inner,
            plan.dim0,
            plan.num_groups,
            plan.dim2,
            plan.target_rms,
            plan.min_gain,
            plan.max_gain,
            plan.block_size,
        )
        return tan_out_re, tan_out_im, None, None


class _FrontendAgcDynamic(torch.autograd.Function):
    """The per-block AGC loop: gains, RMS and clamp mask stay on the device."""

    @staticmethod
    def _grid(plan) -> tuple:
        return (plan.dim0, plan.num_groups, plan.num_blocks, plan.dim2)

    @staticmethod
    def forward(x_re, x_im, plan):
        out_re = torch.empty_like(x_re)
        out_im = torch.empty_like(x_im)
        slots = plan.num_groups * plan.num_blocks
        gain = torch.empty(slots, dtype=torch.float32, device=x_re.device)
        rms = torch.empty_like(gain)
        free = torch.empty_like(gain)
        _ops().frontend_agc_dynamic_forward(
            x_re,
            x_im,
            out_re,
            out_im,
            gain,
            rms,
            free,
            *_FrontendAgcDynamic._grid(plan),
            plan.target_rms,
            plan.min_gain,
            plan.max_gain,
            plan.beta,
            plan.block_size,
        )
        return out_re, out_im, gain, rms, free

    @staticmethod
    def setup_context(ctx, inputs, output):
        x_re, x_im, plan = inputs
        ctx.plan = plan
        ctx.save_for_backward(x_re, x_im, output[2], output[3], output[4])
        ctx.save_for_forward(x_re, x_im, output[2], output[3], output[4])
        ctx.mark_non_differentiable(output[2], output[3], output[4])

    @staticmethod
    def _linear(saved, plan, a_re, a_im, operator):
        x_re, x_im, gain, rms, free = saved
        zeros = torch.zeros_like(x_re)
        a_re = zeros if a_re is None else a_re.contiguous()
        a_im = zeros if a_im is None else a_im.contiguous()
        out_re = torch.empty_like(x_re)
        out_im = torch.empty_like(x_im)
        scratch = torch.empty_like(gain)
        operator(
            x_re,
            x_im,
            gain,
            rms,
            free,
            a_re,
            a_im,
            out_re,
            out_im,
            scratch,
            *_FrontendAgcDynamic._grid(plan),
            plan.beta,
            plan.block_size,
        )
        return out_re, out_im

    @staticmethod
    @first_order_only
    def backward(ctx, grad_out_re, grad_out_im, grad_gain, grad_rms, grad_free):
        grad_re, grad_im = _FrontendAgcDynamic._linear(
            ctx.saved_tensors, ctx.plan, grad_out_re, grad_out_im, _ops().frontend_agc_dynamic_backward
        )
        return grad_re, grad_im, None

    @staticmethod
    def jvp(ctx, tan_x_re, tan_x_im, tan_plan):
        tan_re, tan_im = _FrontendAgcDynamic._linear(
            ctx.saved_tensors, ctx.plan, tan_x_re, tan_x_im, _ops().frontend_agc_dynamic_jvp
        )
        return tan_re, tan_im, None, None, None


@dataclass(frozen=True, slots=True)
class FrontendDiagnostics:
    """Device-side diagnostics from one chain application.

    Every field stays on the device. Reading the AGC gain to build a Python
    scalar would be a per-frame device-to-host transfer, and suppressing the
    clipped-component count would hide an AGC misconfiguration behind a signal
    that merely looks compressed. A time-dynamic AGC publishes one gain and one
    RMS per (group, slow-time block), group major.
    """

    phase_rad: torch.Tensor | None
    agc_gain: torch.Tensor | None
    agc_rms: torch.Tensor | None
    clipped_components: torch.Tensor | None


@dataclass(frozen=True, slots=True)
class FrontendOutput:
    """Data only: the processed signal and what the stages measured."""

    signal: torch.Tensor
    diagnostics: FrontendDiagnostics


class FrontendChain:
    """The single receive-chain runtime.

    ``apply`` runs, in this order and nothing else:

        port, coupling, phase, thermal, LNA, compression, IQ, baseband
        (DC offset and flicker), AGC, ADC

    There is no argument that reorders it, no second quantiser, and no way to
    run the gain before the noise it references. The order is the physical
    one: coupling is in the antenna array ahead of any active device; thermal
    noise is input-referred; the LNA and mixer compress what they amplify,
    noise included; the quadrature split follows the mixer; the DC offset and
    the flicker noise belong to the mixer output and the IF amplifiers; the
    AGC and the converter see all of it.

    ``block_size`` is the launch width of the elementwise kernels, threads per
    block. It is a scheduling choice and never a numerical one.
    """

    def __init__(self, spec: FrontendSpec, *, block_size: int = DEFAULT_BLOCK_SIZE) -> None:
        if not isinstance(spec, FrontendSpec):
            raise TypeError(f"FrontendChain needs a FrontendSpec, got {type(spec).__name__}")
        self.spec = spec
        self.block_size = _checked_block_size(block_size)
        # Per-device constant tables (coupling matrices, DC offsets), built at
        # first use and kept, so a frame costs no host-to-device copy.
        self._resident: dict = {}
        # The oscillator's invalid-query counter for the per-path route. It is
        # accumulated on the device by every per-observation call and decoded
        # once per frame by ``apply``; ``_oscillator_pending`` is a host flag,
        # not a device read.
        self._oscillator_invalid: torch.Tensor | None = None
        self._oscillator_pending = False

    @property
    def has_phase_noise(self) -> bool:
        return self.spec.noise is not None and self.spec.noise.has_oscillator_noise

    def apply_path_phase(self, paths, time_s):
        """Apply the common LO difference before coherent path summation.

        Paths use Channel's negative phasor, so the phase rotates negatively;
        FMCW conjugation produces tx*conj(rx) with the positive LO difference.
        """
        if not self.has_phase_noise or paths.path_count == 0:
            return paths
        times = torch.full_like(paths.total_delay_s, float(time_s), dtype=torch.float64)
        weight = self._apply_path_phase_rows(paths.total_delay_s, paths.complex_transfer_ref, times)
        return replace(paths, complex_transfer_ref=weight)

    def _oscillator_counter(self, device) -> torch.Tensor:
        counter = self._oscillator_invalid
        if counter is None or counter.device != torch.device(device):
            counter = torch.zeros(1, dtype=torch.int32, device=device)
            self._oscillator_invalid = counter
        self._oscillator_pending = True
        return counter

    def _apply_path_phase_rows(self, delays, weight, times):
        """The row form the ADC-refreshed route calls directly.

        The guard is repeated here rather than left to the caller because this is
        a second entry point, not a helper: ``simulation`` reaches it per
        observation. The batch entry keeps its own guard so a chain without phase
        noise allocates no timestamp tensor per frame. Invalid queries are
        counted on the device and decoded by the frame's :meth:`apply`.
        """
        if not self.has_phase_noise or not len(delays):
            return weight
        phase = self.spec.noise.phase_difference(
            times,
            delays,
            seed_base=self.spec.seed,
            sign=-1.0,
            block_size=self.block_size,
            invalid=self._oscillator_counter(delays.device),
        )
        plan = _NoisePlan(1, len(delays), phase, 0.0, 1.0, self.spec.seed, self.block_size)
        real, imaginary, _ = _FrontendNoise.apply(weight.real.contiguous(), weight.imag.contiguous(), plan)
        return torch.complex(real, imaginary)

    def _run_times(self, signal: torch.Tensor, num_run: int, *, times_s, frame_time_s: float) -> torch.Tensor:
        """Absolute receive timestamps of one run, float64 seconds, on the device."""

        if times_s is not None:
            if times_s.numel() != num_run:
                raise ValueError("receive timestamps must cover the complete receive timeline")
            return times_s.to(device=signal.device, dtype=torch.float64).contiguous()
        index = torch.arange(num_run, device=signal.device, dtype=torch.float64)
        timeline = self.spec.timeline
        if timeline is not None and signal.ndim >= 4:
            slow_period, fast_start, fast_period = timeline
            samples = int(signal.shape[-1])
            slow = torch.div(index, samples, rounding_mode="floor")
            return frame_time_s + fast_start + slow * slow_period + (index - slow * samples) * fast_period
        return frame_time_s + index / float(self.spec.noise.phase_sample_rate)

    def _noise_plan(
        self,
        signal: torch.Tensor,
        *,
        thermal_seed: int,
        process_seed: int,
        times_s=None,
        phase_in_signal=False,
        frame_time_s=0.0,
    ) -> _NoisePlan:
        """The fused stage's plan.

        ``thermal_seed`` keys the frame's white draws; ``process_seed`` keys the
        oscillator, which is ONE process over absolute time and therefore keeps
        the base seed across frames.
        """

        spec = self.spec
        num_phase = _phase_run_length(signal)
        phase = torch.zeros(num_phase, dtype=torch.float32, device=signal.device)
        if self.has_phase_noise and not phase_in_signal:
            constructed = times_s is None
            if constructed:
                times_s = torch.arange(num_phase, device=signal.device, dtype=torch.float64)
                times_s = frame_time_s + times_s / spec.noise.phase_sample_rate
            if times_s.numel() != num_phase:
                raise ValueError("phase timestamps must cover the complete receive timeline")
            # A timeline this chain constructed is finite and ordered by
            # construction, so its counter is never read; a caller's is.
            counter = self._discarded_counter(signal.device) if constructed else None
            phase = spec.noise.phase_difference(
                times_s, times_s - times_s[0], seed_base=process_seed, block_size=self.block_size, invalid=counter
            )
        return _NoisePlan(
            num_outer=signal.numel() // num_phase,
            num_phase=num_phase,
            phase=phase,
            thermal_sigma=spec.thermal_sigma_volts(),
            lna_gain=spec.lna_voltage_gain(),
            seed_base=int(thermal_seed),
            block_size=self.block_size,
        )

    def _discarded_counter(self, device) -> torch.Tensor:
        key = ("discarded_counter", str(device))
        counter = self._resident.get(key)
        if counter is None:
            counter = torch.zeros(1, dtype=torch.int32, device=device)
            self._resident[key] = counter
        return counter

    def _agc_plan(self, signal: torch.Tensor, agc: Agc) -> _AgcPlan:
        rank4 = signal.ndim == 4
        if agc.mode == AGC_MODE_PER_RX and rank4:
            dim0 = int(signal.shape[0])
            groups = int(signal.shape[1])
        elif rank4:
            dim0 = int(signal.shape[0] * signal.shape[1])
            groups = 1
        else:
            dim0 = 1
            groups = 1
        if agc.is_dynamic and rank4:
            blocks = int(signal.shape[2])
            inner = int(signal.shape[3])
            beta = agc.loop_coefficient
        else:
            blocks = 1
            inner = int(signal.numel()) // (dim0 * groups)
            beta = 0.0
        return _AgcPlan(
            dim0=dim0,
            num_groups=groups,
            dim2=inner,
            target_rms=float(agc.target_rms),
            min_gain=agc.min_voltage_gain,
            max_gain=agc.max_voltage_gain,
            block_size=self.block_size,
            num_blocks=blocks,
            beta=beta,
        )

    def _dc_broadcast(self, signal: torch.Tensor) -> torch.Tensor:
        """The per-receiver DC offset shaped to broadcast over ``signal`` (receivers on axis 1 of a rank-4 cube)."""

        dc = self._resident_tensor("dc_offset", self.spec.dc_offset, signal.device)
        return dc.view(1, -1, 1, 1) if signal.ndim == 4 else dc.reshape(())

    def _resident_tensor(self, name: str, values, device) -> torch.Tensor:
        if isinstance(values, torch.Tensor):
            # A tensor table (a coupling leaf) is used as given, so its tape reaches the cube.
            return values.to(device=device, dtype=torch.complex64)
        key = (name, str(device))
        tensor = self._resident.get(key)
        if tensor is None:
            tensor = torch.as_tensor(values, dtype=torch.complex64, device=device).contiguous()
            self._resident[key] = tensor
        return tensor

    def _couple(self, signal: torch.Tensor) -> torch.Tensor:
        """``Y = C_tx H C_rx^T`` over the (TX, RX) axes of a rank-4 cube.

        ``H[t, r]`` is the response with element ``t`` transmitting and element
        ``r`` receiving. With coupling, exciting port ``t`` drives element ``k``
        with ``C_tx[k, t]`` and port ``r`` reads ``C_rx[r, j]`` of element
        ``j``, so ``Y[t, r] = SUM_kj C_tx[k, t] H[k, j] C_rx[r, j]``. Two cuBLAS
        products; the derivative is the transposed products, carried by
        ``torch.matmul``. Within a TDM frame the neighbouring elements' slots
        are taken at the port's own slot: the coupled response is assembled
        from one frame's cube, not re-synthesised per slot.
        """

        spec = self.spec
        if signal.ndim != 4:
            raise ValueError("mutual coupling needs the rank-4 (TX, RX, slow, fast) cube")
        num_tx, num_rx, slow, fast = (int(v) for v in signal.shape)
        working = signal
        if spec.tx_coupling is not None:
            matrix = self._resident_tensor("tx_coupling", spec.tx_coupling, signal.device)
            if tuple(matrix.shape) != (num_tx, num_tx):
                raise ValueError(f"tx_coupling is {tuple(matrix.shape)} but the cube has {num_tx} transmitters")
            working = torch.matmul(matrix.transpose(0, 1), working.reshape(num_tx, -1)).reshape(signal.shape)
        if spec.rx_coupling is not None:
            matrix = self._resident_tensor("rx_coupling", spec.rx_coupling, signal.device)
            if tuple(matrix.shape) != (num_rx, num_rx):
                raise ValueError(f"rx_coupling is {tuple(matrix.shape)} but the cube has {num_rx} receivers")
            working = torch.matmul(matrix, working.reshape(num_tx, num_rx, slow * fast)).reshape(signal.shape)
        return working

    def _baseband(self, signal: torch.Tensor, *, seed_base: int, times_s, frame_time_s: float) -> torch.Tensor:
        spec = self.spec
        num_run = _phase_run_length(signal)
        num_outer = signal.numel() // num_run
        num_rx = int(signal.shape[1]) if signal.ndim == 4 else 1
        offsets = spec.dc_offset if spec.dc_offset is not None else (0j,) * num_rx
        if len(offsets) != num_rx:
            raise ValueError(f"dc_offset holds {len(offsets)} values but the cube has {num_rx} receivers")
        dc = self._resident_tensor("dc_offset", offsets, signal.device)
        noise = spec.noise
        level = 0.0 if noise is None else noise.flicker_level_volts(spec.impedance) * spec.lna_voltage_gain()
        levels = 0
        times = torch.zeros(1, dtype=torch.float64, device=signal.device)
        if level > 0.0:
            times = self._run_times(signal, num_run, times_s=times_s, frame_time_s=frame_time_s)
            # Octaves from 1 s down to the coarsest one not finer than two samples.
            rate = float(noise.phase_sample_rate)
            levels = max(1, int(math.floor(math.log2(rate / 2.0))) + 1)
        else:
            times = times.expand(num_run).contiguous()
        # The kernel writes the additive term; adding it here carries the
        # stage's identity derivative in both modes with no companion kernel.
        add_re = torch.empty((num_outer, num_run), dtype=torch.float32, device=signal.device)
        add_im = torch.empty_like(add_re)
        _ops().frontend_baseband_forward(
            add_re,
            add_im,
            times,
            dc.real.contiguous(),
            dc.imag.contiguous(),
            num_outer,
            num_run,
            num_rx,
            level,
            levels,
            int(seed_base),
            self.block_size,
        )
        return signal + torch.complex(add_re, add_im).reshape(signal.shape)

    def apply(
        self,
        signal: torch.Tensor,
        *,
        seed_base: int | None = None,
        times_s=None,
        phase_in_signal=False,
        frame_time_s: float = 0.0,
    ) -> FrontendOutput:
        """Run the whole chain, in order, once.

        ``frame_time_s`` is the frame's start instant: it keys the frame's
        noise realisation (:func:`frame_seed`) and places the baseband flicker
        on absolute time. An explicit ``seed_base`` overrides the key.
        """

        if not signal.is_complex():
            raise TypeError(f"the frontend consumes a complex signal, got {signal.dtype}")
        if signal.dtype != torch.complex64:
            raise TypeError(
                f"the frontend consumes complex64, got {signal.dtype}; the "
                "synthesis families all publish complex64 and a silent promotion "
                "here would hide a dtype mistake upstream"
            )
        if self._oscillator_pending:
            self._oscillator_pending = False
            raise_on_invalid_oscillator_queries(self._oscillator_invalid)
        spec = self.spec
        # The white draws are keyed per frame; the time-keyed processes (the
        # oscillator and the baseband flicker) are one realisation over
        # absolute time and keep the base seed. An explicit seed_base keys both.
        process_seed = spec.seed if seed_base is None else int(seed_base)
        seed = frame_seed(spec.seed, frame_time_s) if seed_base is None else int(seed_base)

        shape = signal.shape
        # Stage 0: the sqrt(W) to volt conversion, applied exactly once.
        working = signal * spec.volts_per_sqrt_watt
        phase_rad = None
        agc_gain = None
        agc_rms = None
        clipped = None

        if spec.tx_coupling is not None or spec.rx_coupling is not None:
            working = self._couple(working)

        if spec.applies_noise_stage:
            plan = self._noise_plan(
                working,
                thermal_seed=seed,
                process_seed=process_seed,
                times_s=times_s,
                phase_in_signal=phase_in_signal,
                frame_time_s=frame_time_s,
            )
            flat = working.reshape(plan.num_outer, plan.num_phase)
            out_re, out_im, phase_rad = _FrontendNoise.apply(flat.real.contiguous(), flat.imag.contiguous(), plan)
            working = torch.complex(out_re, out_im).reshape(shape)

        if spec.compression is not None:
            c5, k_pm, u_sat = spec.compression.coefficients
            plan = _CompressionPlan(
                scale=spec.compression.input_power_scale(
                    impedance=spec.impedance, lna_voltage_gain=spec.lna_voltage_gain()
                ),
                c5=c5,
                k_pm=k_pm,
                u_sat=u_sat,
                block_size=self.block_size,
            )
            flat = working.reshape(-1)
            out_re, out_im = _FrontendCompress.apply(flat.real.contiguous(), flat.imag.contiguous(), plan)
            working = torch.complex(out_re, out_im).reshape(shape)

        if spec.iq is not None:
            # After the noise and before the gain: the imbalance is a property of
            # the quadrature split that follows the mixer, so it acts on the noisy
            # signal, and the AGC and the quantiser then see the image as part of
            # the waveform - which is how a real receiver spends dynamic range on
            # it. Torch rather than a kernel because this is two complex multiplies
            # over the whole cube, with no per-path loop to fuse it into.
            working = spec.iq.alpha * working + spec.iq.beta * working.conj()

        if spec.applies_baseband_stage:
            working = self._baseband(working, seed_base=process_seed, times_s=times_s, frame_time_s=frame_time_s)

        lna_scale = spec.lna_leaf_scale(working.device)
        if lna_scale is not None:
            # The LNA leaf: the chain up to here ran at the host gain and is
            # homogeneous of degree one in it except for the DC offset, so
            # s (y - dc) + dc is the output at the leaf's value (s is exactly
            # one) with the exact derivative. Two library calls over the cube.
            working = working * lna_scale
            if spec.dc_offset is not None:
                working = working + (1.0 - lna_scale) * self._dc_broadcast(working)

        if spec.agc is not None:
            plan = self._agc_plan(working, spec.agc)
            if plan.num_blocks > 1 or spec.agc.is_dynamic:
                flat = working.reshape(plan.dim0, plan.num_groups * plan.num_blocks, plan.dim2)
                out_re, out_im, agc_gain, agc_rms, _ = _FrontendAgcDynamic.apply(
                    flat.real.contiguous(), flat.imag.contiguous(), plan
                )
            else:
                flat = working.reshape(plan.dim0, plan.num_groups, plan.dim2)
                out_re, out_im, agc_gain, agc_rms = _FrontendAgc.apply(
                    flat.real.contiguous(), flat.imag.contiguous(), plan
                )
            working = torch.complex(out_re, out_im).reshape(shape)

        if spec.adc is not None:
            working, clipped = self._quantize(working, spec.adc)

        return FrontendOutput(
            signal=working,
            diagnostics=FrontendDiagnostics(
                phase_rad=phase_rad, agc_gain=agc_gain, agc_rms=agc_rms, clipped_components=clipped
            ),
        )

    def _quantize(self, signal: torch.Tensor, adc: Adc) -> tuple[torch.Tensor, torch.Tensor]:
        """The ONLY call site of the quantizer, in the whole package."""

        _require_no_derivative(signal, "ADC")
        flat = signal.reshape(-1)
        out_re = torch.empty(flat.shape, dtype=torch.float32, device=flat.device)
        out_im = torch.empty_like(out_re)
        clipped = torch.zeros(1, dtype=torch.int32, device=flat.device)
        _ops().frontend_quantize_forward(
            flat.real.contiguous(),
            flat.imag.contiguous(),
            out_re,
            out_im,
            clipped,
            int(flat.numel()),
            int(adc.bits),
            float(adc.full_scale),
            self.block_size,
        )
        return torch.complex(out_re, out_im).reshape(signal.shape), clipped


def _phase_run_length(signal: torch.Tensor) -> int:
    """How many samples one oscillator run spans.

    A rank-4 ``(TX, RX, chirp, sample)`` cube shares one oscillator across the
    virtual array, so the phase walks over ``chirp * sample`` and is broadcast
    over the array.
    Anything else is treated as a single run over the whole tensor, which is the
    honest reading of a signal whose slow-time axis this runtime was not told
    about.
    """

    if signal.ndim >= 4:
        return int(signal.shape[-2] * signal.shape[-1])
    return int(signal.numel())


__all__ = ["Adc", "Agc", "Compression", "Noise"]
