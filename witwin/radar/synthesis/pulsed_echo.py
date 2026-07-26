"""Native pulsed echo synthesis: the Radar-owned pulse-train loop.

The per-path, per-sample sum

    y[l][p][m] = sum_k C[k] * p(t_g + m * T_s - tau_k(l))
                 * exp(+j * 2 * pi * cycles(tau[k], t_l))

runs entirely inside a CUDA kernel. Torch's role here is validation, buffer
allocation, autograd dispatch, and result assembly; it never evaluates the pulse
or the phasor sum. Three registered operators  -  forward, backward, jvp  -  have
exactly one Python owner, this module.

**This module publishes the matched-filter INPUT.** The received pulse train is
the physics; correlating it against the transmitted replica is signal
processing, and it lives in :mod:`witwin.radar.sigproc.matched_filter` under the
plan's Torch/FFT exception. The split is not cosmetic: a filter fuses a
modelling choice - which replica, which window, how much oversampling - into
whatever it is part of, and those choices belong to a processing chain that a
user may replace, not to a propagation model that they may not.

**The pulse is evaluated at the exact fractional delay.** ``u = t_g + m T_s -
tau_k(l)`` is a continuous number and the kernel evaluates ``p(u)`` from its
closed form there. Nothing is snapped to the nearest sample, which is why both
supported pulse kinds are analytic and why there is no table, no gather, and no
interpolation anywhere in the family.

Three structural contracts, each with a test:

* **No conjugation, anywhere.** The train is published in the CHANNEL phasor
  convention ``exp(-j k d)``. There is no de-chirping in a pulsed receiver, so
  there is nothing to convert; ``channel_phasor_to_beat_weight`` belongs to the
  FMCW owner and is never called from here.
* The facade ALWAYS routes through ``Function.apply``. An ADR-038 forward-only
  dual has ``requires_grad == False``, so an eager ``requires_grad`` shortcut
  would silently swallow its Doppler tangent and return a plain tensor.
* No complex tensor crosses the autograd boundary. The public entry splits the
  transfer coefficient into real and imaginary parts with Torch's own
  autograd-aware accessors and recombines the output the same way, which makes
  the conjugate-Wirtinger convention question structurally impossible to get
  wrong at the seam.

Slow time here is per PULSE, not per TDM slot. The FMCW slot table exists
because TDM shares one transmitter chain in time across chirps; a pulsed frame
illuminates every sensor pair on the same PRI grid, so ``t_l = l * T_pri`` for
every pair and there is no per-segment time offset to carry.
"""

from __future__ import annotations

import torch
from torch.autograd.function import once_differentiable

from .assembly import segment_of_each_row
from .contracts import (
    PULSE_KIND_LFM,
    PULSE_KIND_RECT,
    PulsedEchoSpec,
    SynthesisPathBatch,
    require_pulsed_compatible,
)


#: The kernel's pulse-kind selector, mirroring the constants on the spec. An
#: integer crosses the ABI because a string would need an allocation and a
#: comparison per launch to say something the spec already validated once.
_PULSE_KIND_CODE = {PULSE_KIND_RECT: 0, PULSE_KIND_LFM: 1}


_OPS = None


def _ops():
    """The native operator table, resolved once per process.

    Held here as well as in the build module because this runs on every
    synthesis call, forward and backward: a per-launch import plus function call
    is pure overhead on the hot path.
    """

    global _OPS
    if _OPS is None:
        from ..cuda import build

        _OPS = build.build_extension()
    return _OPS


def _kernel_arguments(spec: PulsedEchoSpec) -> tuple:
    """The waveform scalars, in the order all three operators take them.

    Written once because the three signatures agree by construction and a
    divergence between them would be a silently different pulse in the gradient
    than in the primal.
    """

    return (
        spec.num_pulses,
        spec.num_samples,
        spec.sample_period_s,
        spec.pri_s,
        spec.range_gate_start_s,
        _PULSE_KIND_CODE[spec.pulse_kind],
        spec.pulse_width_s,
        spec.bandwidth_hz,
        spec.pulse_amplitude,
        spec.carrier_hz,
        spec.carrier_rate_hz,
    )


class _PulsedEchoSynthesis(torch.autograd.Function):
    """Autograd bridge for the three native pulsed operators."""

    @staticmethod
    def forward(tau_rt, tau_rate, weight_re, weight_im, offsets, segment, spec):
        num_paths = int(tau_rt.shape[0])
        num_segments = int(offsets.shape[0]) - 1
        out_re = torch.empty(
            (spec.num_pulses, num_segments, spec.num_samples),
            dtype=torch.float32,
            device=tau_rt.device,
        )
        out_im = torch.empty_like(out_re)
        _ops().pulsed_echo_forward(
            tau_rt,
            tau_rate,
            weight_re,
            weight_im,
            offsets,
            out_re,
            out_im,
            num_paths,
            num_segments,
            *_kernel_arguments(spec),
        )
        return out_re, out_im

    @staticmethod
    def setup_context(ctx, inputs, output):
        (
            tau_rt,
            tau_rate,
            weight_re,
            weight_im,
            offsets,
            segment,
            spec,
        ) = inputs
        ctx.spec = spec
        ctx.num_segments = int(offsets.shape[0]) - 1
        ctx.save_for_backward(tau_rt, tau_rate, weight_re, weight_im, segment)
        ctx.save_for_forward(tau_rt, tau_rate, weight_re, weight_im, offsets)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out_re, grad_out_im):
        tau_rt, tau_rate, weight_re, weight_im, segment = ctx.saved_tensors
        spec = ctx.spec
        grad_tau_rt = torch.empty_like(tau_rt)
        grad_tau_rate = torch.empty_like(tau_rate)
        grad_weight_re = torch.empty_like(weight_re)
        grad_weight_im = torch.empty_like(weight_im)
        _ops().pulsed_echo_backward(
            tau_rt,
            tau_rate,
            weight_re,
            weight_im,
            segment,
            grad_out_re.contiguous(),
            grad_out_im.contiguous(),
            grad_tau_rt,
            grad_tau_rate,
            grad_weight_re,
            grad_weight_im,
            int(tau_rt.shape[0]),
            ctx.num_segments,
            *_kernel_arguments(spec),
        )
        return (
            grad_tau_rt,
            grad_tau_rate,
            grad_weight_re,
            grad_weight_im,
            None,
            None,
            None,
        )

    @staticmethod
    def jvp(
        ctx,
        tan_tau_rt,
        tan_tau_rate,
        tan_weight_re,
        tan_weight_im,
        tan_offsets,
        tan_segment,
        tan_spec,
    ):
        tau_rt, tau_rate, weight_re, weight_im, offsets = ctx.saved_tensors
        spec = ctx.spec
        zero = torch.zeros_like(tau_rt)
        tan_tau_rt = zero if tan_tau_rt is None else tan_tau_rt.contiguous()
        tan_tau_rate = zero if tan_tau_rate is None else tan_tau_rate.contiguous()
        tan_weight_re = zero if tan_weight_re is None else tan_weight_re.contiguous()
        tan_weight_im = zero if tan_weight_im is None else tan_weight_im.contiguous()
        tan_out_re = torch.empty(
            (spec.num_pulses, ctx.num_segments, spec.num_samples),
            dtype=torch.float32,
            device=tau_rt.device,
        )
        tan_out_im = torch.empty_like(tan_out_re)
        _ops().pulsed_echo_jvp(
            tau_rt,
            tau_rate,
            weight_re,
            weight_im,
            offsets,
            tan_tau_rt,
            tan_tau_rate,
            tan_weight_re,
            tan_weight_im,
            tan_out_re,
            tan_out_im,
            int(tau_rt.shape[0]),
            ctx.num_segments,
            *_kernel_arguments(spec),
        )
        return tan_out_re, tan_out_im


def synthesize_echo_rows(
    total_delay_s: torch.Tensor,
    delay_rate: torch.Tensor | None,
    transfer_ref: torch.Tensor,
    pair_offsets: torch.Tensor,
    spec: PulsedEchoSpec,
) -> torch.Tensor:
    """Synthesize a pulse train from Channel-convention transfer coefficients.

    ``transfer_ref`` is the coefficient AS CHANNEL PUBLISHES IT - the entry
    performs no conjugation, because the pulsed product stays in that
    convention. The FMCW entry is the one that takes an already-converted beat
    weight; the families deliberately do not share a weight type, so a
    coefficient cannot be conjugated on the way into the wrong one.

    Returns ``complex64[num_pulses, num_segments, num_samples]``.
    """

    path_count = int(total_delay_s.shape[0])
    if transfer_ref.shape != total_delay_s.shape:
        raise ValueError("transfer_ref and total_delay_s must have the same shape")
    if delay_rate is None:
        rate = torch.zeros_like(total_delay_s)
    else:
        if delay_rate.shape != total_delay_s.shape:
            raise ValueError("delay_rate and total_delay_s must have the same shape")
        rate = delay_rate
    segment = segment_of_each_row(pair_offsets, path_count)
    # Torch-owned, autograd-aware accessors: the real pair crosses the
    # boundary, never the complex tensor.
    out_re, out_im = _PulsedEchoSynthesis.apply(
        total_delay_s.contiguous(),
        rate.contiguous(),
        transfer_ref.real.contiguous(),
        transfer_ref.imag.contiguous(),
        pair_offsets.contiguous(),
        segment,
        spec,
    )
    return torch.complex(out_re, out_im)


def synthesize_pulsed_echo(
    batch: SynthesisPathBatch, spec: PulsedEchoSpec
) -> torch.Tensor:
    """Synthesize one coherent processing interval's received pulse train.

    Returns ``complex64[num_pulses, sensor_pair_count, num_samples]`` in the
    CHANNEL phasor convention (``spec.phasor``).
    ``assembly.assemble_frame_cube`` turns it into the rank-4
    ``(TX, RX, pulse, sample)`` layout an array processor indexes.

    :func:`~witwin.radar.synthesis.contracts.require_pulsed_compatible` runs
    FIRST, so a weight and a spec that would count the carrier, the spreading,
    or the Doppler twice - or a gate that overruns the PRI, or a velocity window
    whose range migration exceeds a range cell - are refused before any kernel
    launch rather than producing a plausible train that is wrong in a way that
    reads as a defocused target.

    A dead row contributes exactly zero. That is enforced on the WEIGHT, with
    ``torch.where``, so the row is inert in the primal and carries no gradient
    to anything it was built from. Zeroing the output afterwards would leave a
    live gradient path back through a row that does not exist.
    """

    require_pulsed_compatible(batch, spec)
    transfer = batch.complex_transfer_ref
    if batch.row_valid is not None:
        transfer = torch.where(
            batch.row_valid, transfer, torch.zeros_like(transfer)
        )
    return synthesize_echo_rows(
        batch.total_delay_s,
        batch.delay_rate,
        transfer,
        batch.pair_offsets,
        spec,
    )


__all__ = [
    "PulsedEchoSpec",
    "synthesize_echo_rows",
    "synthesize_pulsed_echo",
]
