"""Native FMCW beat synthesis: the Radar-owned waveform hot loop.

The per-path, per-sample sum

    s[c][p][m] = sum_k C[k] * exp(+j * 2 * pi * cycles(tau[k], t_c, t_m))

runs entirely inside a CUDA kernel. Torch's role here is validation, buffer
allocation, autograd dispatch, and result assembly; it never evaluates the
phasor sum. Beat, spectrum and refreshed-observation families each provide
forward, backward and jvp operators, all owned by this module.

Two structural contracts, each with a test:

* The facade ALWAYS routes through ``Function.apply``. An eager shortcut that
  checks ``requires_grad`` and bypasses autograd when it is false is forbidden:
  an ADR-038 forward-only dual has ``requires_grad == False``, so that shortcut
  silently swallows its tangent and returns a plain tensor.
* No complex tensor crosses the autograd boundary. The public entry splits a
  complex weight into real and imaginary parts with Torch's own autograd-aware
  accessors and recombines the output the same way, which makes the
  conjugate-Wirtinger convention question structurally impossible to get wrong.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np
import torch

from ..cuda import native_ops as _ops
from ..policy import first_order_only, refuse_derivative
from .assembly import (
    FMCW_OUTPUT_BEAT,
    FMCW_OUTPUT_SPECTRUM,
    FmcwSpec,
    SynthesisPathBatch,
    delay_residual,
    pair_tx_index,
    require_compatible,
    segment_of_each_row,
)


def _forward_op(domain: str):
    ops = _ops()
    return ops.fmcw_spectrum_forward if domain == "spectrum" else ops.fmcw_beat_forward


def _backward_op(domain: str):
    ops = _ops()
    return ops.fmcw_spectrum_backward if domain == "spectrum" else ops.fmcw_beat_backward


def _jvp_op(domain: str):
    ops = _ops()
    return ops.fmcw_spectrum_jvp if domain == "spectrum" else ops.fmcw_beat_jvp


def channel_phasor_to_beat_weight(coefficient: torch.Tensor) -> torch.Tensor:
    """Convert a Channel transfer coefficient into an FMCW beat weight.

    Channel publishes ``exp(-j * k * d)`` under an ``exp(+j * 2 * pi * f * t)``
    time dependence. FMCW de-chirping multiplies the transmitted chirp by the
    conjugate of the received signal, so the beat-domain phasor advances with
    ``+j``. The two conventions are therefore conjugates, and a Channel
    coefficient becomes a beat weight by conjugation.

    A complex target response is authored in the Channel convention, because
    it multiplies transports authored there. The unfused route converts here;
    fused adaptive synthesis performs the same single conjugation after its
    interpolation, inside the FMCW native owner.
    """

    if coefficient.dtype not in (torch.complex64, torch.complex128):
        raise TypeError(f"a Channel transfer coefficient must be complex, got {coefficient.dtype}")
    return torch.conj(coefficient).resolve_conj()


@functools.lru_cache(maxsize=32)
def _ramp_tables(spec: FmcwSpec, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """The spec's slot table and phase-error polynomial on ``device``, uploaded once per spec.

    The slot table is empty for the uniform ramp, which the kernel then
    evaluates from its scalars exactly as before; it is ``float64[slots, 4]``
    when the spec carries a sequence or a start offset.
    """

    slots = (
        torch.empty((0, 4), dtype=torch.float64, device=device)
        if spec.uniform
        else torch.from_numpy(spec.slot_profiles()).to(device=device, dtype=torch.float64).contiguous()
    )
    poly = torch.tensor(spec.phase_error_cycles, dtype=torch.float64, device=device).reshape(-1)
    return slots, poly


def _beat_tables(spec: FmcwSpec, segment_table: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The beat operators' three tables, in their argument order: slot, segment, polynomial."""

    slots, poly = _ramp_tables(spec, segment_table.device)
    return slots, segment_table, poly


def _kernel_transmitters(spec: FmcwSpec) -> int:
    """The ``num_tx`` the kernel's slot index uses: one slot per chirp when every transmitter fires."""

    return 1 if spec.simultaneous else spec.num_tx


class _FmcwSynthesis(torch.autograd.Function):
    """Autograd bridge for the selected native FMCW output domain."""

    @staticmethod
    def forward(tau_rt, tau_lo, tau_rate, weight_re, weight_im, offsets, segment, tx_index, starts, spec):
        num_paths = int(tau_rt.shape[0])
        num_segments = int(offsets.shape[0]) - 1
        out_re = torch.empty(
            (spec.num_chirps, num_segments, spec.num_samples), dtype=torch.float32, device=tau_rt.device
        )
        out_im = torch.empty_like(out_re)
        spectrum = spec.output_domain == FMCW_OUTPUT_SPECTRUM
        delay = () if spectrum else (tau_rate,)
        _forward_op(spec.output_domain)(
            tau_rt,
            tau_lo,
            *delay,
            weight_re,
            weight_im,
            offsets,
            tx_index,
            *(() if spectrum else _beat_tables(spec, starts)),
            out_re,
            out_im,
            num_paths,
            num_segments,
            _kernel_transmitters(spec),
            spec.num_chirps,
            spec.num_samples,
            spec.sample_period_s,
            spec.chirp_period_s,
            spec.slope_hz_per_s,
            spec.carrier_hz,
            spec.carrier_rate_hz,
            spec.t_start_s,
        )
        return out_re, out_im

    @staticmethod
    def setup_context(ctx, inputs, output):
        (tau_rt, tau_lo, tau_rate, weight_re, weight_im, offsets, segment, tx_index, starts, spec) = inputs
        ctx.spec = spec
        ctx.num_segments = int(offsets.shape[0]) - 1
        ctx.save_for_backward(tau_rt, tau_lo, tau_rate, weight_re, weight_im, segment, tx_index, starts)
        ctx.save_for_forward(tau_rt, tau_lo, tau_rate, weight_re, weight_im, offsets, tx_index, starts)

    @staticmethod
    @first_order_only
    def backward(ctx, grad_out_re, grad_out_im):
        (tau_rt, tau_lo, tau_rate, weight_re, weight_im, segment, tx_index, starts) = ctx.saved_tensors
        spec = ctx.spec
        spectrum = spec.output_domain == FMCW_OUTPUT_SPECTRUM
        grad_tau_rt = torch.empty_like(tau_rt)
        # The spectrum family models no delay walk, so there is no rate to carry
        # a cotangent: its slot in the returned tuple is None rather than zeros,
        # which is what tells autograd the input was not used at all.
        grad_tau_rate = None if spectrum else torch.empty_like(tau_rate)
        grad_weight_re = torch.empty_like(weight_re)
        grad_weight_im = torch.empty_like(weight_im)
        _backward_op(spec.output_domain)(
            tau_rt,
            tau_lo,
            *(() if spectrum else (tau_rate,)),
            weight_re,
            weight_im,
            segment,
            tx_index,
            *(() if spectrum else _beat_tables(spec, starts)),
            grad_out_re.contiguous(),
            grad_out_im.contiguous(),
            grad_tau_rt,
            *(() if spectrum else (grad_tau_rate,)),
            grad_weight_re,
            grad_weight_im,
            int(tau_rt.shape[0]),
            ctx.num_segments,
            _kernel_transmitters(spec),
            spec.num_chirps,
            spec.num_samples,
            spec.sample_period_s,
            spec.chirp_period_s,
            spec.slope_hz_per_s,
            spec.carrier_hz,
            spec.carrier_rate_hz,
            spec.t_start_s,
        )
        return (grad_tau_rt, None, grad_tau_rate, grad_weight_re, grad_weight_im, None, None, None, None, None)

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
        tan_tx_index,
        tan_starts,
        tan_spec,
    ):
        (tau_rt, tau_lo, tau_rate, weight_re, weight_im, offsets, tx_index, starts) = ctx.saved_tensors
        spec = ctx.spec
        zero = torch.zeros_like(tau_rt)
        tan_tau_rt = zero if tan_tau_rt is None else tan_tau_rt.contiguous()
        tan_tau_rate = zero if tan_tau_rate is None else tan_tau_rate.contiguous()
        tan_weight_re = zero if tan_weight_re is None else tan_weight_re.contiguous()
        tan_weight_im = zero if tan_weight_im is None else tan_weight_im.contiguous()
        tan_out_re = torch.empty(
            (spec.num_chirps, ctx.num_segments, spec.num_samples), dtype=torch.float32, device=tau_rt.device
        )
        tan_out_im = torch.empty_like(tan_out_re)
        spectrum = spec.output_domain == FMCW_OUTPUT_SPECTRUM
        _jvp_op(spec.output_domain)(
            tau_rt,
            tau_lo,
            *(() if spectrum else (tau_rate,)),
            weight_re,
            weight_im,
            offsets,
            tx_index,
            *(() if spectrum else _beat_tables(spec, starts)),
            tan_tau_rt,
            *(() if spectrum else (tan_tau_rate,)),
            tan_weight_re,
            tan_weight_im,
            tan_out_re,
            tan_out_im,
            int(tau_rt.shape[0]),
            ctx.num_segments,
            _kernel_transmitters(spec),
            spec.num_chirps,
            spec.num_samples,
            spec.sample_period_s,
            spec.chirp_period_s,
            spec.slope_hz_per_s,
            spec.carrier_hz,
            spec.carrier_rate_hz,
            spec.t_start_s,
        )
        return tan_out_re, tan_out_im


class _FmcwObservations(torch.autograd.Function):
    """Native CSR observation synthesis with fixed ADC scheduling."""

    @staticmethod
    def forward(values, offsets, segment, slope, carrier):
        result = values.new_empty((len(offsets) - 1, 2))
        _ops().fmcw_observation_forward(values, offsets, segment, values.new_empty(0), result, slope, carrier)
        return result

    @staticmethod
    def setup_context(ctx, inputs, output):
        values, offsets, segment, ctx.slope, ctx.carrier = inputs
        ctx.save_for_backward(values, offsets, segment)
        ctx.save_for_forward(values, offsets, segment)

    @staticmethod
    @first_order_only
    def backward(ctx, gradient):
        values, offsets, segment = ctx.saved_tensors
        result = torch.empty_like(values)
        _ops().fmcw_observation_backward(
            values, offsets, segment, gradient.contiguous(), result, ctx.slope, ctx.carrier
        )
        return result, None, None, None, None

    @staticmethod
    def jvp(ctx, tangent, offsets_tangent, segment_tangent, slope_tangent, carrier_tangent):
        values, offsets, segment = ctx.saved_tensors
        result = values.new_empty((len(offsets) - 1, 2))
        _ops().fmcw_observation_jvp(values, offsets, segment, tangent.contiguous(), result, ctx.slope, ctx.carrier)
        return result


def synthesize_fmcw_observations(delay, weight, offsets, adc_time, spec, *, segment=None):
    """One complex beat value per CSR segment, using refreshed path rows.

    Delay [s] and conjugated Channel weight are differentiable. Chirp-local
    ADC times [s] are fixed schedule metadata. There is no delay-rate term:
    each row already describes its observation instant.
    ``segment`` may reuse the caller's CSR inverse when it expanded these
    rows itself; otherwise this owner derives it from ``offsets``.
    """

    refuse_derivative("FMCW observation schedule", "ADC times are fixed metadata", adc_time=adc_time)
    if delay.shape != weight.shape or delay.shape != adc_time.shape:
        raise ValueError("observation delay, weight and ADC time must have the same shape")
    if segment is None:
        segment = segment_of_each_row(offsets, len(delay))
    values = torch.stack([delay.double(), weight.real.double(), weight.imag.double(), adc_time.double()], dim=1)
    result = _FmcwObservations.apply(values, offsets.contiguous(), segment, spec.slope_hz_per_s, spec.carrier_hz)
    return torch.complex(result[:, 0].float(), result[:, 1].float())


@dataclass(frozen=True)
class _AdaptiveObservationSpec:
    waveform: FmcwSpec
    interpolation_carrier: float
    begin: int
    stop: int
    precise_delay: bool = False

    def native_args(self):
        spec = self.waveform
        return (
            self.begin,
            self.stop,
            spec.num_tx,
            spec.num_samples,
            self.interpolation_carrier,
            spec.slope_hz_per_s,
            spec.carrier_hz,
            int(self.precise_delay),
        )

    def output_shape(self, bounds):
        return ((self.stop - self.begin) * ((bounds.shape[1] - 1) // self.waveform.num_tx), 2)


class _AdaptiveFmcw(torch.autograd.Function):
    """Fused compact interpolation and ADC synthesis, with native first-order AD."""

    @staticmethod
    def forward(samples, validity, starts, basis, bounds, owners, clock, plan):
        result = samples.new_empty(plan.output_shape(bounds))
        _ops().fmcw_adaptive_forward(
            samples, validity, starts, basis, bounds, owners, clock, samples.new_empty(0), result, *plan.native_args()
        )
        return result

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.plan = inputs[-1]
        ctx.save_for_backward(*inputs[:-1])
        ctx.save_for_forward(*inputs[:-1])

    @staticmethod
    @first_order_only
    def backward(ctx, gradient):
        samples, *metadata = ctx.saved_tensors
        result = torch.zeros_like(samples)
        _ops().fmcw_adaptive_backward(samples, *metadata, gradient.contiguous(), result, *ctx.plan.native_args())
        return result, None, None, None, None, None, None, None

    @staticmethod
    def jvp(
        ctx,
        tangent,
        validity_tangent,
        starts_tangent,
        basis_tangent,
        bounds_tangent,
        owners_tangent,
        clock_tangent,
        plan_tangent,
    ):
        samples, *metadata = ctx.saved_tensors
        bounds = metadata[3]
        result = samples.new_empty(ctx.plan.output_shape(bounds))
        tangent = torch.zeros_like(samples) if tangent is None else tangent.contiguous()
        _ops().fmcw_adaptive_jvp(samples, *metadata, tangent, result, *ctx.plan.native_args())
        return result


def synthesize_adaptive_fmcw(
    samples, validity, starts, basis, bounds, owners, clock, spec, carrier_hz, begin, stop, *, precise_delay=False
):
    """Direct complete ADC values from compact adaptive probe rows.

    All routing and timestamps are discrete metadata. Samples retain native
    VJP/JVP; interpolation's float32 boundary and serial path sum are retained,
    except that ``precise_delay`` keeps the interpolated delay in float64 for
    probes that carry the join's float64 delay (R-ADR-040).
    A row-dependent receiver effect must be applied before using observation
    synthesis instead; this operator does not absorb oscillator phase noise.
    """
    refuse_derivative("adaptive observation schedule", "partition and ADC times are fixed", basis=basis, clock=clock)
    plan = _AdaptiveObservationSpec(spec, carrier_hz, begin, stop, bool(precise_delay))
    result = _AdaptiveFmcw.apply(samples, validity, starts, basis, bounds, owners, clock, plan)
    return torch.complex(result[:, 0].float(), result[:, 1].float())


def synthesize_fmcw_rows(
    total_delay_s: torch.Tensor,
    delay_rate: torch.Tensor | None,
    beat_weight: torch.Tensor,
    pair_offsets: torch.Tensor,
    spec: FmcwSpec,
    *,
    segment_tx_index: torch.Tensor | None = None,
    segment_t_start_s: torch.Tensor | None = None,
    segment_schedule: torch.Tensor | None = None,
    total_delay_residual_s: torch.Tensor | None = None,
) -> torch.Tensor:
    """Synthesize the configured FMCW output from conjugated Channel weights.

    ``beat_weight`` is in the BEAT convention. Use
    :func:`channel_phasor_to_beat_weight` to get there from a Channel transfer
    coefficient; this entry does not convert, so a caller cannot accidentally
    conjugate twice.

    ``segment_tx_index`` names which transmitter drives each sensor-pair
    segment, which is what turns the slow-time axis into TDM slot time. It may
    be omitted only when ``spec.num_tx == 1``, where every segment is in slot
    ``chirp`` and the question does not arise; a multi-TX spec must say, because
    guessing it would put a whole chirp period of Doppler walk on the wrong
    channel and still produce a plausible cube. :func:`synthesize_fmcw`
    derives it from the array layout.

    ``segment_t_start_s`` is ``float64[segments]``: each segment's own ADC
    window start in seconds, in place of ``spec.t_start_s``. It is how one
    launch synthesizes segments that are different ADC instants of a refreshed
    frame; each segment's value is the one a launch with that start as
    ``spec.t_start_s`` would produce. Beat domain only, and schedule data: a
    derivative in it is refused. ``segment_schedule`` is the same statement for
    a chirp sequence, ``float64[segments, 3]``: each segment's window start [s],
    slope [Hz/s] and start offset [Hz]
    (:func:`~witwin.radar.synthesis.assembly.refreshed_schedule`).

    ``total_delay_residual_s`` is the composed rows' float32 delay residual
    (R-ADR-040): the kernels read ``tau = total_delay_s + residual`` in
    float64, so the bandwidth term ``2 pi B tau`` is not quantized to the
    float32 ulp of the delay. It carries no derivative; ``None`` reads the
    float32 delay alone, bit for bit as before.

    Under DDM and BPM every transmitter fires every chirp, so the slow-time
    coordinate of every segment is its chirp and ``segment_tx_index`` is not
    read; :func:`multiplex_transmitters` then sums the pairs each receiver hears.

    This is the ROW-LEVEL entry, and what it does NOT check is part of its
    contract. There is no :class:`SynthesisPathBatch` here, so the eight
    provenance rules do not run: nothing verifies that this weight's spreading,
    transmit power, and reference phase are absent from the spec's owners.
    Spec-internal refusals still apply - a spec that puts the carrier in two
    homes is refused here as well. The validated route is
    :func:`synthesize_fmcw`, which asks the batch first; use this one only
    where the caller owns the single-count rule itself. R-ADR-010 records the
    split.
    """

    path_count = int(total_delay_s.shape[0])
    if beat_weight.shape != total_delay_s.shape:
        raise ValueError("beat_weight and total_delay_s must have the same shape")
    if spec.output_domain == FMCW_OUTPUT_SPECTRUM and not (spec.admits_direct_spectrum and spec.uniform):
        raise ValueError(
            "the closed-form range spectrum needs one linear ramp on every chirp and a complex receiver; a chirp "
            'sequence, a start offset, a ramp phase error or a real ADC is synthesized with output_domain="beat" '
            "and transformed by processing"
        )
    if delay_rate is None:
        rate = torch.zeros_like(total_delay_s)
    else:
        if spec.output_domain == FMCW_OUTPUT_SPECTRUM:
            raise ValueError(
                "the range spectrum models no delay walk: its closed form exists "
                "because one delay holds for the whole chirp. Synthesize a walking "
                f'delay with output_domain="{FMCW_OUTPUT_BEAT}" and take the range '
                "transform in processing, which evaluates the same linear-delay "
                "model for N log N instead of N per bin"
            )
        if delay_rate.shape != total_delay_s.shape:
            raise ValueError("delay_rate and total_delay_s must have the same shape")
        rate = delay_rate
    num_segments = int(pair_offsets.shape[0]) - 1
    if spec.simultaneous:
        tx_index = torch.zeros(num_segments, dtype=torch.int32, device=total_delay_s.device)
    elif segment_tx_index is None:
        if spec.num_tx != 1:
            raise ValueError(
                f"this spec declares num_tx={spec.num_tx}, so every sensor-pair "
                "segment must name the transmitter that drives it; pass "
                "segment_tx_index (witwin.radar.synthesis.assembly.pair_tx_index "
                "derives it from the array layout)"
            )
        tx_index = torch.zeros(num_segments, dtype=torch.int32, device=total_delay_s.device)
    else:
        if tuple(segment_tx_index.shape) != (num_segments,):
            raise ValueError(
                "segment_tx_index must hold one transmitter index per sensor-pair "
                f"segment, expected shape ({num_segments},), got "
                f"{tuple(segment_tx_index.shape)}"
            )
        tx_index = segment_tx_index
    if segment_t_start_s is not None and segment_schedule is not None:
        raise ValueError("segment_t_start_s and segment_schedule are two spellings of one schedule; pass one")
    if segment_t_start_s is None and segment_schedule is None:
        starts = torch.empty((0, 3), dtype=torch.float64, device=total_delay_s.device)
    else:
        if spec.output_domain == FMCW_OUTPUT_SPECTRUM:
            raise ValueError(
                "per-segment ADC window starts belong to the beat domain; the range spectrum's closed form "
                "assumes one uniform window"
            )
        table = segment_t_start_s if segment_schedule is None else segment_schedule
        refuse_derivative("FMCW segment window starts", "ADC times are fixed metadata", starts=table)
        width = (num_segments,) if segment_schedule is None else (num_segments, 3)
        if table.dtype != torch.float64 or tuple(table.shape) != width:
            name = "segment_t_start_s" if segment_schedule is None else "segment_schedule"
            raise ValueError(f"{name} must be float64 with shape {width}, got {table.dtype} {tuple(table.shape)}")
        if segment_schedule is None:
            # The uniform ramp's slope and offset beside each start.
            table = torch.stack(
                (table, torch.full_like(table, spec.slope_hz_per_s), torch.full_like(table, spec.start_offset_hz)),
                dim=1,
            )
        starts = table.contiguous()
    segment = segment_of_each_row(pair_offsets, path_count)
    # Torch-owned, autograd-aware accessors: the real pair crosses the
    # boundary, never the complex tensor.
    out_re, out_im = _FmcwSynthesis.apply(
        total_delay_s.contiguous(),
        delay_residual(total_delay_s, total_delay_residual_s),
        rate.contiguous(),
        beat_weight.real.contiguous(),
        beat_weight.imag.contiguous(),
        pair_offsets.contiguous(),
        segment,
        tx_index.contiguous(),
        starts,
        spec,
    )
    return torch.complex(out_re, out_im)


def multiplex_transmitters(cube: torch.Tensor, spec: FmcwSpec) -> torch.Tensor:
    """``[chirp, pair, sample]`` -> ``[chirp, rx, sample]``: what each receiver hears under DDM or BPM.

    Every transmitter fires every chirp with its code
    (:meth:`FmcwSpec.transmit_code`), so a receiver records the coded sum over
    transmitters. The beat cube carries the CONJUGATE of the physical code,
    because the receiver mixes the echo against an uncoded ramp. One contraction
    over the transmitter axis of the sink-major pair rank ``rx * num_tx + tx``.
    TDM is returned as it is.
    """

    if not spec.simultaneous:
        return cube
    chirps, pairs, samples = (int(value) for value in cube.shape)
    code = torch.from_numpy(np.conj(spec.transmit_code())).to(device=cube.device, dtype=cube.dtype)
    return torch.einsum("crts,ct->crs", cube.reshape(chirps, spec.num_rx, spec.num_tx, samples), code)


def synthesize_fmcw(batch: SynthesisPathBatch, spec: FmcwSpec) -> torch.Tensor:
    """Synthesize one FMCW frame; direct spectrum is the default domain.

    Returns ``complex64[num_chirps, sensor_pair_count, num_samples]``, one
    rank-3 cube in the BEAT convention. ``assembly.assemble_frame_cube`` turns
    it into the rank-4 ``(TX, RX, chirp, sample)`` layout processing consumes.

    :func:`~witwin.radar.synthesis.assembly.require_compatible` runs FIRST, so
    a weight and a spec that would count the carrier, the spreading, or the
    Doppler twice are refused before any kernel launch rather than producing a
    plausible cube that is wrong by a factor nobody notices.

    A dead row contributes exactly zero. That is enforced on the WEIGHT, with
    ``torch.where``, so the row is inert in the primal and carries no gradient
    to anything it was built from. Zeroing the output afterwards would leave a
    live gradient path back through a row that does not exist.
    """

    require_compatible(batch, spec)
    weight = channel_phasor_to_beat_weight(batch.complex_transfer_ref)
    if batch.row_valid is not None:
        weight = torch.where(batch.row_valid, weight, torch.zeros_like(weight))
    return synthesize_fmcw_rows(
        batch.total_delay_s,
        batch.delay_rate,
        weight,
        batch.pair_offsets,
        spec,
        segment_tx_index=pair_tx_index(
            num_tx=spec.num_tx, num_rx=spec.num_rx, sensor_pair_count=batch.sensor_pair_count, device=batch.device
        ),
        total_delay_residual_s=batch.total_delay_residual_s,
    )


#: Columns of one cross-radar link row (R-ADR-051), in the kernel's order.
CROSS_LINK_COLUMNS = (
    "tx_slope_hz_per_s",
    "tx_start_offset_hz",
    "tx_ramp_end_s",
    "tx_chirp_period_s",
    "tx_slot_count",
    "tx_transmitters",
    "tx_frame_shift_s",
    "rx_slope_hz_per_s",
    "rx_start_offset_hz",
    "phase0_cycles",
    "rx_if_bandwidth_hz",
)


@dataclass(frozen=True, slots=True, eq=False)
class CrossRouting:
    """The constant routing of one cross-radar beat launch (R-ADR-051).

    ``row_link`` and ``row_tx`` (``int32[rows_per_slot]``) give each row of one
    slot its link in ``link_table`` (``float64[links, 11]``, columns
    :data:`CROSS_LINK_COLUMNS`; ``-1`` is a row this receiver does not hear
    through this family) and its transmitter's index within its own radar.
    ``segment_rows`` (``int64[cells, 2]``) is each receive cell's half-open row
    range and ``segment_schedule`` (``float64[cells, 2]``) its slot's ramp start
    in the receiver's frame and its first ADC instant into the ramp, s.
    ``row_segment_offsets`` / ``row_segments`` are the inverse map the backward
    reads. Every member is schedule data and carries no derivative.
    """

    row_link: torch.Tensor
    row_tx: torch.Tensor
    link_table: torch.Tensor
    segment_rows: torch.Tensor
    segment_schedule: torch.Tensor
    row_segment_offsets: torch.Tensor
    row_segments: torch.Tensor
    rows_per_slot: int
    num_samples: int
    sample_period_s: float

    @property
    def cell_count(self) -> int:
        return int(self.segment_schedule.shape[0])


class _FmcwCross(torch.autograd.Function):
    """Autograd bridge for the native cross-radar beat family."""

    @staticmethod
    def forward(transfer, tau_rt, tau_lo, valid, routing):
        out = transfer.new_empty((routing.cell_count, routing.num_samples, 2))
        _ops().fmcw_cross_forward(
            transfer,
            tau_rt,
            tau_lo,
            valid,
            routing.row_link,
            routing.row_tx,
            routing.link_table,
            routing.segment_rows,
            routing.segment_schedule,
            out,
            routing.rows_per_slot,
            routing.num_samples,
            routing.sample_period_s,
        )
        return out

    @staticmethod
    def setup_context(ctx, inputs, output):
        transfer, tau_rt, tau_lo, valid, routing = inputs
        ctx.routing = routing
        ctx.save_for_backward(transfer, tau_rt, tau_lo, valid)
        ctx.save_for_forward(transfer, tau_rt, tau_lo, valid)

    @staticmethod
    @first_order_only
    def backward(ctx, grad_out):
        transfer, tau_rt, tau_lo, valid = ctx.saved_tensors
        routing = ctx.routing
        grad_transfer = torch.empty_like(transfer)
        grad_tau = torch.empty_like(tau_rt)
        _ops().fmcw_cross_backward(
            transfer,
            tau_rt,
            tau_lo,
            valid,
            routing.row_link,
            routing.row_tx,
            routing.link_table,
            routing.row_segment_offsets,
            routing.row_segments,
            routing.segment_schedule,
            grad_out.contiguous(),
            grad_transfer,
            grad_tau,
            routing.rows_per_slot,
            routing.num_samples,
            routing.sample_period_s,
        )
        return grad_transfer, grad_tau, None, None, None

    @staticmethod
    def jvp(ctx, tan_transfer, tan_tau_rt, tan_tau_lo, tan_valid, tan_routing):
        transfer, tau_rt, tau_lo, valid = ctx.saved_tensors
        routing = ctx.routing
        tan_transfer = torch.zeros_like(transfer) if tan_transfer is None else tan_transfer.contiguous()
        tan_tau_rt = torch.zeros_like(tau_rt) if tan_tau_rt is None else tan_tau_rt.contiguous()
        tan_out = transfer.new_empty((routing.cell_count, routing.num_samples, 2))
        _ops().fmcw_cross_jvp(
            transfer,
            tau_rt,
            tau_lo,
            valid,
            routing.row_link,
            routing.row_tx,
            routing.link_table,
            routing.segment_rows,
            routing.segment_schedule,
            tan_transfer,
            tan_tau_rt,
            tan_out,
            routing.rows_per_slot,
            routing.num_samples,
            routing.sample_period_s,
        )
        return tan_out


def synthesize_fmcw_cross(
    transfer: torch.Tensor,
    delay: torch.Tensor,
    residual: torch.Tensor | None,
    valid: torch.Tensor | None,
    routing: CrossRouting,
) -> torch.Tensor:
    """Every receive cell's cross-radar beat, ``complex64[cells, num_samples]`` (R-ADR-051).

    ``transfer`` is the rows' Channel coefficient (``complex64[rows]``, Channel
    convention; the kernel conjugates it into the beat weight), ``delay`` the
    float32 round-trip delay and ``residual`` its float64 remainder or
    ``None``. A row whose ``valid`` entry is false contributes exactly zero and
    receives no gradient. Differentiable in ``transfer`` and ``delay`` in both
    modes; the gates are discrete and contribute their almost-everywhere
    derivative.
    """

    if transfer.dtype != torch.complex64 or delay.dtype != torch.float32:
        raise TypeError("cross-radar rows are a complex64 transfer and a float32 delay")
    empty = delay.new_empty(0)
    flags = torch.empty(0, dtype=torch.bool, device=delay.device) if valid is None else valid.contiguous()
    out = _FmcwCross.apply(
        torch.view_as_real(transfer.contiguous()),
        delay.contiguous(),
        empty if residual is None else residual.contiguous(),
        flags,
        routing,
    )
    return torch.view_as_complex(out)


__all__ = [
    "CROSS_LINK_COLUMNS",
    "CrossRouting",
    "FmcwSpec",
    "channel_phasor_to_beat_weight",
    "multiplex_transmitters",
    "synthesize_fmcw_cross",
    "synthesize_fmcw_observations",
    "synthesize_fmcw_rows",
    "synthesize_fmcw",
]
