"""Typed receiver frontend and its fixed physical signal chain.

The module owns frontend configuration, deterministic stage seeding, native
phase/thermal/LNA and AGC operators, and final ADC quantization. Former
``frontend.*`` submodule paths are intentionally not retained.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import torch

from .policy import first_order_only, refuse_derivative, require_host_floats

__all__ = [
    "AdcSpec",
    "AgcSpec",
    "FrontendSpec",
    "LnaSpec",
    "NoiseSpec",
    "PortSpec",
    "SeedSpec",
]

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


#: Why the port impedance has no derivative. It converts sqrt(W) to volts and
#: is a UNIT CONVENTION: differentiating a loss with respect to the units it is
#: expressed in is not a question about the world.
_PORT_REASON = (
    "reference_impedance_ohm is the unit convention that turns sqrt(W) into "
    "volts, applied exactly once at stage 0."
)

#: Why every noise scalar has no derivative. All four parameterise a
#: counter-based Philox draw in ``frontend.py``; a pathwise derivative
#: through an RNG stream is not defined by any accepted contract here, and a
#: reparameterised noise model is a separate decision with its own ADR.
_NOISE_REASON = (
    "every scalar on NoiseSpec parameterises a counter-based Philox draw - the "
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
_LNA_REASON = (
    "gain_db is device configuration rather than scene state. The native "
    "frontend operator carries no tangent or gradient slot for it and no "
    "consumer asks for one; adding the slot is a named Phase-9 deferral "
    "recorded in docs/dev/radar-ad-capability-matrix.md."
)

#: Why the AGC setpoint has no derivative. It is a control target, and the
#: stage is already non-linear in the signal because its gain depends on the
#: signal's own RMS - which is why every physics test turns AGC off.
_AGC_REASON = (
    "target_rms is a control setpoint. The AGC gain depends on the signal's "
    "own RMS, so the stage is not linear in the signal at all, and every "
    "physics invariant in this package is asserted with AGC off."
)

#: Why the ADC grid has no derivative. Both fields define ``round``'s step and
#: clip level, and the quantiser already refuses a differentiable SIGNAL at
#: ``frontend.py``. Refusing the grid as well keeps the wall in one
#: piece: a full-scale leaf would be a derivative of a staircase's placement.
_ADC_REASON = (
    "bits and full_scale define the quantiser's grid, and `round` has a zero "
    "derivative almost everywhere and an undefined one at every code "
    "boundary. The ADC stage already refuses a differentiable signal; the "
    "grid sits behind that same wall."
)


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
        require_host_floats(
            "PortSpec",
            _PORT_REASON,
            reference_impedance_ohm=self.reference_impedance_ohm,
        )
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
        require_host_floats(
            "NoiseSpec",
            _NOISE_REASON,
            noise_figure_db=self.noise_figure_db,
            antenna_temperature_k=self.antenna_temperature_k,
            bandwidth_hz=self.bandwidth_hz,
            phase_noise_dbc_per_hz=self.phase_noise_dbc_per_hz,
            phase_offset_hz=self.phase_offset_hz,
            phase_sample_rate_hz=self.phase_sample_rate_hz,
        )
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
    """A voltage gain in dB. Applied AFTER thermal noise, always.

    The gain is the one frontend scalar whose derivative would be perfectly
    well defined - it is a smooth multiplicative factor on the whole signal -
    and it is refused anyway, because the native operator has no slot for it.
    Refusing it is the honest state; ``float()``-ing a marked tensor and
    returning ``grad = None`` is not.
    """

    gain_db: float = 0.0

    def __post_init__(self) -> None:
        require_host_floats("LnaSpec", _LNA_REASON, gain_db=self.gain_db)

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
        require_host_floats(
            "AgcSpec",
            _AGC_REASON,
            target_rms=self.target_rms,
            min_gain_db=self.min_gain_db,
            max_gain_db=self.max_gain_db,
        )
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
        require_host_floats(
            "AdcSpec", _ADC_REASON, bits=self.bits, full_scale=self.full_scale
        )
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


#: Launch width for the elementwise passes. An environment override exists so a
#: test can prove that the Philox realisation does not depend on it; production
#: never sets it. The Wiener scan is deliberately NOT affected, because its
#: accumulation order is part of the realisation.
_BLOCK_SIZE_ENV = "WITWIN_RADAR_FRONTEND_BLOCK"
_DEFAULT_BLOCK_SIZE = 256


def frontend_block_size() -> int:
    raw = os.environ.get(_BLOCK_SIZE_ENV)
    if raw is None:
        return _DEFAULT_BLOCK_SIZE
    block = int(raw)
    if block < 32 or block > 1024 or (block & (block - 1)) != 0:
        raise ValueError(
            f"{_BLOCK_SIZE_ENV} must be a power of two between 32 and 1024, got "
            f"{block}"
        )
    return block


_OPS = None


def _ops():
    """The native operator table, resolved once per process."""

    global _OPS
    if _OPS is None:
        from .cuda import runtime as build

        _OPS = build.build_extension()
    return _OPS


def _require_no_derivative(signal: torch.Tensor, stage: str) -> None:
    """Refuse a differentiable input to a non-differentiable stage, loudly.

    Silently detaching would return a number with no gradient where the caller
    asked for one, which is the failure a fail-loud contract exists to prevent.

    The check itself is :func:`witwin.radar.policy.refuse_derivative`,
    the ONE owner of the non-differentiability wall. This wording was the model
    that owner was generalised from, so what stays here is the ADC's own reason
    - why ``round`` has no derivative - rather than a second copy of the rule.
    """

    refuse_derivative(
        f"the frontend {stage} stage",
        "`round` has a zero derivative almost everywhere and an undefined one "
        "at every code boundary, and a Phase-9 straight-through surrogate is a "
        "modelling decision rather than something the frontend may choose: "
        "detach the signal before the ADC, or run without one.",
        signal=signal,
    )


@dataclass(frozen=True, slots=True)
class _NoisePlan:
    num_outer: int
    num_phase: int
    phase_sigma: float
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


class _FrontendNoise(torch.autograd.Function):
    """Phase rotation, thermal addition, and the LNA gain, in one launch pair."""

    @staticmethod
    def forward(x_re, x_im, plan):
        out_re = torch.empty_like(x_re)
        out_im = torch.empty_like(x_im)
        phase_rad = torch.empty(
            plan.num_phase, dtype=torch.float32, device=x_re.device
        )
        _ops().frontend_noise_forward(
            x_re,
            x_im,
            out_re,
            out_im,
            phase_rad,
            plan.num_outer,
            plan.num_phase,
            plan.phase_sigma,
            plan.thermal_sigma,
            plan.lna_gain,
            plan.seed_base,
            STAGE_PHASE_NOISE,
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
            zeros = torch.zeros(
                (plan.num_outer, plan.num_phase),
                dtype=torch.float32,
                device=phase_rad.device,
            )
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


@dataclass(frozen=True, slots=True)
class FrontendDiagnostics:
    """Device-side diagnostics from one chain application.

    Every field stays on the device. Reading the AGC gain to build a Python
    scalar would be a per-frame device-to-host transfer, and suppressing the
    clipped-component count would hide an AGC misconfiguration behind a signal
    that merely looks compressed.
    """

    phase_rad: torch.Tensor | None
    agc_gain: torch.Tensor | None
    agc_rms: torch.Tensor | None
    clipped_components: torch.Tensor | None

    @classmethod
    def empty(cls) -> "FrontendDiagnostics":
        return cls(
            phase_rad=None, agc_gain=None, agc_rms=None, clipped_components=None
        )


@dataclass(frozen=True, slots=True)
class FrontendOutput:
    """Data only: the processed signal and what the stages measured."""

    signal: torch.Tensor
    diagnostics: FrontendDiagnostics
    stages: tuple[str, ...]


class FrontendChain:
    """The single receive-chain runtime.

    ``apply`` runs :data:`FRONTEND_STAGE_ORDER` and nothing else. There is no
    argument that reorders it, no second quantiser, and no way to run the gain
    before the noise it references.
    """

    def __init__(self, spec: FrontendSpec) -> None:
        if not isinstance(spec, FrontendSpec):
            raise TypeError(
                f"FrontendChain needs a FrontendSpec, got {type(spec).__name__}"
            )
        self.spec = spec

    @property
    def enabled_stages(self) -> tuple[str, ...]:
        """Which stages will run, in the fixed order, for reporting."""

        spec = self.spec
        active = {
            "port": True,
            "phase": spec.phase_sigma_rad() > 0.0,
            "thermal": spec.thermal_sigma_volts() > 0.0,
            "lna": spec.lna is not None,
            "agc": spec.agc is not None,
            "adc": spec.adc is not None,
        }
        return tuple(name for name in FRONTEND_STAGE_ORDER if active[name])

    def _noise_plan(self, signal: torch.Tensor, *, seed_base: int) -> _NoisePlan:
        spec = self.spec
        num_phase = _phase_run_length(signal)
        return _NoisePlan(
            num_outer=signal.numel() // num_phase,
            num_phase=num_phase,
            phase_sigma=spec.phase_sigma_rad(),
            thermal_sigma=spec.thermal_sigma_volts(),
            lna_gain=spec.lna_voltage_gain(),
            seed_base=int(seed_base),
            block_size=frontend_block_size(),
        )

    def _agc_plan(self, signal: torch.Tensor, agc: AgcSpec) -> _AgcPlan:
        if agc.mode == AGC_MODE_PER_RX and signal.ndim == 4:
            dim0 = int(signal.shape[0])
            groups = int(signal.shape[1])
            inner = int(signal.shape[2] * signal.shape[3])
        else:
            dim0 = 1
            groups = 1
            inner = int(signal.numel())
        return _AgcPlan(
            dim0=dim0,
            num_groups=groups,
            dim2=inner,
            target_rms=float(agc.target_rms),
            min_gain=agc.min_gain,
            max_gain=agc.max_gain,
            block_size=frontend_block_size(),
        )

    def apply(
        self, signal: torch.Tensor, *, seed_base: int | None = None
    ) -> FrontendOutput:
        """Run the whole chain, in order, once."""

        if not signal.is_complex():
            raise TypeError(
                f"the frontend consumes a complex signal, got {signal.dtype}"
            )
        if signal.dtype != torch.complex64:
            raise TypeError(
                f"the frontend consumes complex64, got {signal.dtype}; the "
                "synthesis families all publish complex64 and a silent promotion "
                "here would hide a dtype mistake upstream"
            )
        spec = self.spec
        seed = spec.seed.seed_base if seed_base is None else int(seed_base)

        shape = signal.shape
        # Stage 0: the sqrt(W) to volt conversion, applied exactly once.
        working = signal * spec.port.volts_per_sqrt_watt
        phase_rad = None
        agc_gain = None
        agc_rms = None
        clipped = None

        if spec.applies_noise_stage:
            plan = self._noise_plan(working, seed_base=seed)
            flat = working.reshape(plan.num_outer, plan.num_phase)
            out_re, out_im, phase_rad = _FrontendNoise.apply(
                flat.real.contiguous(), flat.imag.contiguous(), plan
            )
            working = torch.complex(out_re, out_im).reshape(shape)

        if spec.agc is not None:
            plan = self._agc_plan(working, spec.agc)
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
                phase_rad=phase_rad,
                agc_gain=agc_gain,
                agc_rms=agc_rms,
                clipped_components=clipped,
            ),
            stages=self.enabled_stages,
        )

    def _quantize(
        self, signal: torch.Tensor, adc: AdcSpec
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
            frontend_block_size(),
        )
        return torch.complex(out_re, out_im).reshape(signal.shape), clipped


def _phase_run_length(signal: torch.Tensor) -> int:
    """How many samples one Wiener run spans.

    A rank-4 ``(TX, RX, chirp, sample)`` cube shares one oscillator across the
    virtual array, so the phase walks over ``chirp * sample`` and is broadcast
    over the array - which is exactly what the expression this replaces did.
    Anything else is treated as a single run over the whole tensor, which is the
    honest reading of a signal whose slow-time axis this runtime was not told
    about.
    """

    if signal.ndim >= 4:
        return int(signal.shape[-2] * signal.shape[-1])
    return int(signal.numel())


__all__ = [
    "AdcSpec",
    "AgcSpec",
    "FrontendSpec",
    "LnaSpec",
    "NoiseSpec",
    "PortSpec",
    "SeedSpec",
]