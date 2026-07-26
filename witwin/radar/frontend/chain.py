"""One receive chain, one order, one ADC.

``NoiseModelRuntime`` and ``ReceiverChainRuntime`` are MERGED here rather than
kept side by side. Their composite order was whatever the caller happened to do,
and they each owned a quantiser. Both of those are unrepresentable now:

    0. port      x <- x * sqrt(R)                sqrt(W) -> volts, exactly ONCE
    1. phase     x <- x * exp(j theta)           Wiener scan, stage 0
    2. thermal   x <- x + n,  n ~ CN(0, 2 s^2)   stage 1, INPUT-REFERRED
    3. lna       x <- x * g_lna
    4. agc       x <- x * clamp(target/rms, ...)
    5. adc       x <- clip and round             ALWAYS last

Stages 1 to 3 are ONE native operator. Thermal noise physically enters at the
antenna and LNA input, so it is added before the gain, and the output noise
power is ``g_lna^2 k T_sys B R`` rather than ``k T_sys B R``. An implementation
that let the caller run the receiver chain first is wrong by exactly that
factor, silently, in a quantity nobody plots.

**Per-stage seeds, never a shared generator.** The native draws are
counter-based Philox keyed by ``(seed_base, stage_id, linear element index)``.
Toggling one stage therefore leaves every other stage's realisation
bit-identical, which is what "reproducible" has to mean for a differential
measurement. It also makes the realisation independent of the launch
configuration: ``block_size`` is a real argument, and a test sets it to two
different values and asserts ``torch.equal``.

Draw order is part of the contract and a refactor that fuses two draws into one
changes every realisation:

* thermal draws one Philox call per element in linear index order, real
  component FIRST then imaginary;
* phase draws one call per slow-time sample in linear index order.

**Noise is optional and OFF by default.** ``FrontendSpec`` is ``None`` on a
``RadarConfig`` unless a caller asks for one, and every physics test runs
without it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from torch.autograd.function import once_differentiable

from .contracts import (
    AGC_MODE_PER_RX,
    FRONTEND_STAGE_ORDER,
    STAGE_PHASE_NOISE,
    STAGE_THERMAL_NOISE,
    AdcSpec,
    AgcSpec,
    FrontendSpec,
)


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
        from ..cuda import build

        _OPS = build.build_extension()
    return _OPS


def _require_no_derivative(signal: torch.Tensor, stage: str) -> None:
    """Refuse a differentiable input to a non-differentiable stage, loudly.

    Silently detaching would return a number with no gradient where the caller
    asked for one, which is the failure a fail-loud contract exists to prevent.
    """

    tangent = torch.autograd.forward_ad.unpack_dual(signal).tangent
    if signal.requires_grad or tangent is not None:
        raise RuntimeError(
            f"the frontend {stage} stage is not differentiable: `round` has a "
            "zero derivative almost everywhere and an undefined one at every "
            "code boundary, so this family ships no backward and no jvp. A "
            "straight-through surrogate is a Phase-9 modelling decision rather "
            "than something the frontend may choose. Detach the signal before "
            "the ADC, or run without one."
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
    @once_differentiable
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
    @once_differentiable
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
    "FrontendChain",
    "FrontendDiagnostics",
    "FrontendOutput",
    "frontend_block_size",
]
