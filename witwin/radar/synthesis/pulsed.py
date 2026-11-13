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
processing, and it lives in :mod:`witwin.radar.processing.range_doppler` under the
processing Torch/FFT exception (cuFFT through ``torch.fft``). The split is not cosmetic: a filter fuses a
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

import functools

import torch

from ..cuda import native_ops as _ops
from ..policy import first_order_only, refuse_derivative
from .assembly import (
    PULSE_KIND_CODED,
    PULSE_KIND_LFM,
    PULSE_KIND_RECT,
    CrossRouting,
    PulsedSpec,
    SynthesisPathBatch,
    delay_residual,
    require_pulsed_compatible,
    segment_of_each_row,
    synthesize_cross,
)

#: The kernel's pulse-kind selector, mirroring the constants on the spec. An
#: integer crosses the ABI because a string would need an allocation and a
#: comparison per launch to say something the spec already validated once.
_PULSE_KIND_CODE = {PULSE_KIND_RECT: 0, PULSE_KIND_LFM: 1, PULSE_KIND_CODED: 2}


@functools.lru_cache(maxsize=32)
def _chip_table(spec: PulsedSpec, device: torch.device) -> torch.Tensor:
    """The spec's chip table on ``device``, uploaded once per distinct spec.

    ``float64[num_chips, 3]`` for a coded pulse and an empty table otherwise. A
    spec is frozen and hashable, so a frame loop that synthesizes the same
    waveform every frame crosses to the device once rather than once a launch.
    """

    rows = spec.chip_table() if spec.is_coded else ()
    return torch.tensor(rows, dtype=torch.float64, device=device).reshape(len(rows), 3).contiguous()


def _empty_table(device: torch.device) -> torch.Tensor:
    return torch.empty((0, 2), dtype=torch.float64, device=device)


def _kernel_arguments(spec: PulsedSpec) -> tuple:
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
        spec.frequency_step_hz,
        int(spec.periodic),
    )


class _PulsedEchoSynthesis(torch.autograd.Function):
    """Autograd bridge for the three native pulsed operators."""

    @staticmethod
    def forward(tau_rt, tau_lo, tau_rate, weight_re, weight_im, offsets, segment, schedule, spec):
        num_paths = int(tau_rt.shape[0])
        num_segments = int(offsets.shape[0]) - 1
        out_re = torch.empty(
            (spec.num_pulses, num_segments, spec.num_samples), dtype=torch.float32, device=tau_rt.device
        )
        out_im = torch.empty_like(out_re)
        _ops().pulsed_echo_forward(
            tau_rt,
            tau_lo,
            tau_rate,
            weight_re,
            weight_im,
            offsets,
            _chip_table(spec, tau_rt.device),
            schedule,
            out_re,
            out_im,
            num_paths,
            num_segments,
            *_kernel_arguments(spec),
        )
        return out_re, out_im

    @staticmethod
    def setup_context(ctx, inputs, output):
        (tau_rt, tau_lo, tau_rate, weight_re, weight_im, offsets, segment, schedule, spec) = inputs
        ctx.spec = spec
        ctx.num_segments = int(offsets.shape[0]) - 1
        ctx.save_for_backward(tau_rt, tau_lo, tau_rate, weight_re, weight_im, segment, schedule)
        ctx.save_for_forward(tau_rt, tau_lo, tau_rate, weight_re, weight_im, offsets, schedule)

    @staticmethod
    @first_order_only
    def backward(ctx, grad_out_re, grad_out_im):
        tau_rt, tau_lo, tau_rate, weight_re, weight_im, segment, schedule = ctx.saved_tensors
        spec = ctx.spec
        grad_tau_rt = torch.empty_like(tau_rt)
        grad_tau_rate = torch.empty_like(tau_rate)
        grad_weight_re = torch.empty_like(weight_re)
        grad_weight_im = torch.empty_like(weight_im)
        _ops().pulsed_echo_backward(
            tau_rt,
            tau_lo,
            tau_rate,
            weight_re,
            weight_im,
            segment,
            _chip_table(spec, tau_rt.device),
            schedule,
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
        return (grad_tau_rt, None, grad_tau_rate, grad_weight_re, grad_weight_im, None, None, None, None)

    @staticmethod
    def jvp(
        ctx,
        tan_tau_rt,
        tan_tau_lo,
        tan_tau_rate,
        tan_weight_re,
        tan_weight_im,
        tan_offsets,
        tan_segment,
        tan_schedule,
        tan_spec,
    ):
        tau_rt, tau_lo, tau_rate, weight_re, weight_im, offsets, schedule = ctx.saved_tensors
        spec = ctx.spec
        zero = torch.zeros_like(tau_rt)
        tan_tau_rt = zero if tan_tau_rt is None else tan_tau_rt.contiguous()
        tan_tau_rate = zero if tan_tau_rate is None else tan_tau_rate.contiguous()
        tan_weight_re = zero if tan_weight_re is None else tan_weight_re.contiguous()
        tan_weight_im = zero if tan_weight_im is None else tan_weight_im.contiguous()
        tan_out_re = torch.empty(
            (spec.num_pulses, ctx.num_segments, spec.num_samples), dtype=torch.float32, device=tau_rt.device
        )
        tan_out_im = torch.empty_like(tan_out_re)
        _ops().pulsed_echo_jvp(
            tau_rt,
            tau_lo,
            tau_rate,
            weight_re,
            weight_im,
            offsets,
            _chip_table(spec, tau_rt.device),
            schedule,
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
    spec: PulsedSpec,
    *,
    segment_schedule: torch.Tensor | None = None,
    total_delay_residual_s: torch.Tensor | None = None,
) -> torch.Tensor:
    """Synthesize a pulse train from Channel-convention transfer coefficients.

    ``segment_schedule`` is ``float64[segments, 2]``: each segment's own
    fast-time gate start [s] and carrier offset [Hz], in place of
    ``spec.range_gate_start_s`` and ``l * spec.frequency_step_hz``. It is how one
    launch synthesizes segments that are different fast-time instants of a
    refreshed frame (each segment's value is the one a launch with that gate
    start and offset would produce). Schedule data: a derivative in it is
    refused.

    ``total_delay_residual_s`` is the composed rows' non-differentiable delay
    residual (R-ADR-040), read as ``tau = total_delay_s + residual`` in
    float64; ``None`` reads the float32 delay alone, bit for bit as before.

    ``transfer_ref`` is the coefficient AS CHANNEL PUBLISHES IT - the entry
    performs no conjugation, because the pulsed product stays in that
    convention. The FMCW entry is the one that takes an already-converted beat
    weight; the families deliberately do not share a weight type, so a
    coefficient cannot be conjugated on the way into the wrong one.

    This is the ROW-LEVEL entry, and what it does NOT check is part of its
    contract. There is no :class:`SynthesisPathBatch` here, so the eight
    provenance rules do not run: nothing verifies that this weight's spreading,
    transmit power, and reference phase are absent from the spec's owners.
    Spec-internal refusals still apply. The validated route is
    :func:`synthesize_pulsed`, which asks the batch first; use this one
    only where the caller owns the single-count rule itself. R-ADR-010 records
    the split.

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
    num_segments = int(pair_offsets.shape[0]) - 1
    if segment_schedule is None:
        schedule = _empty_table(total_delay_s.device)
    else:
        refuse_derivative("pulsed segment schedule", "gate instants are fixed metadata", schedule=segment_schedule)
        if segment_schedule.dtype != torch.float64 or tuple(segment_schedule.shape) != (num_segments, 2):
            raise ValueError(
                f"segment_schedule must be float64 with shape ({num_segments}, 2), got "
                f"{segment_schedule.dtype} {tuple(segment_schedule.shape)}"
            )
        schedule = segment_schedule.contiguous()
    segment = segment_of_each_row(pair_offsets, path_count)
    # Torch-owned, autograd-aware accessors: the real pair crosses the
    # boundary, never the complex tensor.
    out_re, out_im = _PulsedEchoSynthesis.apply(
        total_delay_s.contiguous(),
        delay_residual(total_delay_s, total_delay_residual_s),
        rate.contiguous(),
        transfer_ref.real.contiguous(),
        transfer_ref.imag.contiguous(),
        pair_offsets.contiguous(),
        segment,
        schedule,
        spec,
    )
    return torch.complex(out_re, out_im)


def synthesize_pulsed(batch: SynthesisPathBatch, spec: PulsedSpec) -> torch.Tensor:
    """Synthesize one coherent processing interval's received pulse train.

    Returns ``complex64[num_pulses, sensor_pair_count, num_samples]`` in the
    CHANNEL phasor convention (``spec.phasor``).
    ``assembly.assemble_frame_cube`` turns it into the rank-4
    ``(TX, RX, pulse, sample)`` layout an array processor indexes.

    :func:`~witwin.radar.synthesis.assembly.require_pulsed_compatible` runs
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
        transfer = torch.where(batch.row_valid, transfer, torch.zeros_like(transfer))
    return synthesize_echo_rows(
        batch.total_delay_s,
        batch.delay_rate,
        transfer,
        batch.pair_offsets,
        spec,
        total_delay_residual_s=batch.total_delay_residual_s,
    )


#: Columns of one pulsed cross-radar link row (R-ADR-051), in the kernel's order.
PULSED_CROSS_COLUMNS = (
    "tx_frame_shift_s",
    "tx_offset_hz",
    "rx_offset_hz",
    "rx_frame_start_s",
    "tx_pri_s",
    "tx_pulse_count",
    "tx_pulse_kind",
    "tx_pulse_width_s",
    "tx_bandwidth_hz",
    "tx_amplitude",
    "tx_first_chip",
    "tx_chip_count",
    "tx_periodic",
    "tx_frequency_step_hz",
)


def pulse_kind_code(spec: PulsedSpec) -> int:
    """The kernel's selector for ``spec``'s pulse kind."""

    return _PULSE_KIND_CODE[spec.pulse_kind]


def synthesize_pulsed_cross(
    transfer: torch.Tensor,
    delay: torch.Tensor,
    residual: torch.Tensor | None,
    valid: torch.Tensor | None,
    routing: CrossRouting,
) -> torch.Tensor:
    """A sampled receiver's samples of other radars' pulse trains and of emitters, ``complex64[cells, samples]``.

    Channel convention (R-ADR-051). ``routing.link_table`` rows follow
    :data:`PULSED_CROSS_COLUMNS`; ``routing.aux`` is the transmitters' chip
    tables, ``float64[chips, 3]``. See
    :func:`~witwin.radar.synthesis.assembly.synthesize_cross`.
    """

    return synthesize_cross("pulsed", transfer, delay, residual, valid, routing)


__all__ = [
    "PULSED_CROSS_COLUMNS",
    "PulsedSpec",
    "pulse_kind_code",
    "synthesize_echo_rows",
    "synthesize_pulsed",
    "synthesize_pulsed_cross",
]
