"""Native OFDM channel-frequency-response synthesis: the Radar-owned CFR loop.

The per-path, per-subcarrier sum

    H[l][p][n] = sum_k C[k] * exp(+j * 2 * pi * cycles(tau[k], t_l, n * df))

runs entirely inside a CUDA kernel. Torch's role here is validation, buffer
allocation, autograd dispatch, and result assembly; it never evaluates the
phasor sum. Three registered operators  -  forward, backward, jvp  -  have
exactly one Python owner, this module.

Three structural contracts, each with a test:

* **No conjugation, anywhere.** The cube is published in the CHANNEL phasor
  convention ``exp(-j k d)``. OFDM demodulation is per-subcarrier equalisation
  ``H = Y / X``, which removes the transmitted symbol but not the carrier
  convention, so there is nothing to convert. ``channel_phasor_to_beat_weight``
  belongs to the FMCW owner and is never called from here; the convention
  travels as data on :data:`~witwin.radar.synthesis.assembly.CHANNEL_PHASOR`
  and ``OfdmSpec.phasor``.
* The facade ALWAYS routes through ``Function.apply``. An ADR-038 forward-only
  dual has ``requires_grad == False``, so an eager ``requires_grad`` shortcut
  would silently swallow its Doppler tangent and return a plain tensor.
* No complex tensor crosses the autograd boundary. The public entry splits the
  transfer coefficient into real and imaginary parts with Torch's own
  autograd-aware accessors and recombines the output the same way, which makes
  the conjugate-Wirtinger convention question structurally impossible to get
  wrong at the seam.

Slow time here is per SYMBOL, not per TDM slot. The FMCW slot table exists
because TDM shares one transmitter chain in time across chirps; an OFDM frame
samples every sensor pair on the same symbol grid, so ``t_l = l * T_sym`` for
every pair and there is no per-segment time offset to carry.
"""

from __future__ import annotations

import functools
from typing import Any

import numpy as np
import torch

from ..cuda import native_ops as _ops
from ..policy import first_order_only, refuse_derivative
from .assembly import (
    CrossRouting,
    OfdmSpec,
    SynthesisPathBatch,
    delay_residual,
    require_ofdm_compatible,
    segment_of_each_row,
    synthesize_cross,
)


class _OfdmCfrSynthesis(torch.autograd.Function):
    """Autograd bridge for the three native CFR operators."""

    @staticmethod
    def forward(
        tau_rt: torch.Tensor,
        tau_lo: torch.Tensor,
        tau_rate: torch.Tensor,
        weight_re: torch.Tensor,
        weight_im: torch.Tensor,
        offsets: torch.Tensor,
        segment: torch.Tensor,
        spec: OfdmSpec,
        columns: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_paths = int(tau_rt.shape[0])
        num_segments = int(offsets.shape[0]) - 1
        out_re = torch.empty(
            (spec.num_symbols, num_segments, spec.num_subcarriers), dtype=torch.float32, device=tau_rt.device
        )
        out_im = torch.empty_like(out_re)
        _ops().ofdm_cfr_forward(
            tau_rt,
            tau_lo,
            tau_rate,
            weight_re,
            weight_im,
            offsets,
            out_re,
            out_im,
            num_paths,
            num_segments,
            spec.num_symbols,
            spec.num_subcarriers,
            columns,
            spec.subcarrier_spacing_hz,
            spec.symbol_period_s,
            spec.carrier_hz,
            spec.carrier_rate_hz,
        )
        return out_re, out_im

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: tuple[torch.Tensor, torch.Tensor]) -> None:
        (tau_rt, tau_lo, tau_rate, weight_re, weight_im, offsets, segment, spec, columns) = inputs
        ctx.spec = spec
        ctx.columns = columns
        ctx.num_segments = int(offsets.shape[0]) - 1
        ctx.save_for_backward(tau_rt, tau_lo, tau_rate, weight_re, weight_im, segment)
        ctx.save_for_forward(tau_rt, tau_lo, tau_rate, weight_re, weight_im, offsets)

    @staticmethod
    @first_order_only
    def backward(ctx: Any, grad_out_re: torch.Tensor, grad_out_im: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        tau_rt, tau_lo, tau_rate, weight_re, weight_im, segment = ctx.saved_tensors
        spec = ctx.spec
        grad_tau_rt = torch.empty_like(tau_rt)
        grad_tau_rate = torch.empty_like(tau_rate)
        grad_weight_re = torch.empty_like(weight_re)
        grad_weight_im = torch.empty_like(weight_im)
        _ops().ofdm_cfr_backward(
            tau_rt,
            tau_lo,
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
            spec.num_symbols,
            spec.num_subcarriers,
            ctx.columns,
            spec.subcarrier_spacing_hz,
            spec.symbol_period_s,
            spec.carrier_hz,
            spec.carrier_rate_hz,
        )
        return (grad_tau_rt, None, grad_tau_rate, grad_weight_re, grad_weight_im, None, None, None, None)

    @staticmethod
    def jvp(
        ctx: Any,
        tan_tau_rt: Any,
        tan_tau_lo: Any,
        tan_tau_rate: Any,
        tan_weight_re: Any,
        tan_weight_im: Any,
        tan_offsets: Any,
        tan_segment: Any,
        tan_spec: Any,
        tan_columns: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tau_rt, tau_lo, tau_rate, weight_re, weight_im, offsets = ctx.saved_tensors
        spec = ctx.spec
        zero = torch.zeros_like(tau_rt)
        zero_weight = torch.zeros_like(weight_re)
        tan_tau_rt = zero if tan_tau_rt is None else tan_tau_rt.contiguous()
        tan_tau_rate = zero if tan_tau_rate is None else tan_tau_rate.contiguous()
        tan_weight_re = zero_weight if tan_weight_re is None else tan_weight_re.contiguous()
        tan_weight_im = zero_weight if tan_weight_im is None else tan_weight_im.contiguous()
        tan_out_re = torch.empty(
            (spec.num_symbols, ctx.num_segments, spec.num_subcarriers), dtype=torch.float32, device=tau_rt.device
        )
        tan_out_im = torch.empty_like(tan_out_re)
        _ops().ofdm_cfr_jvp(
            tau_rt,
            tau_lo,
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
            spec.num_symbols,
            spec.num_subcarriers,
            ctx.columns,
            spec.subcarrier_spacing_hz,
            spec.symbol_period_s,
            spec.carrier_hz,
            spec.carrier_rate_hz,
        )
        return tan_out_re, tan_out_im


def synthesize_cfr_rows(
    total_delay_s: torch.Tensor,
    delay_rate: torch.Tensor | None,
    transfer_ref: torch.Tensor,
    pair_offsets: torch.Tensor,
    spec: OfdmSpec,
    *,
    total_delay_residual_s: torch.Tensor | None = None,
) -> torch.Tensor:
    """Synthesize a CFR cube from Channel-convention transfer coefficients.

    ``transfer_ref`` is the coefficient AS CHANNEL PUBLISHES IT - the entry
    performs no conjugation, because the OFDM product stays in that convention.
    The FMCW entry is the one that takes an already-converted beat weight; the
    two families deliberately do not share a weight type, so a coefficient
    cannot be conjugated on the way into the wrong one.

    Two accepted shapes, and the shape IS the statement:

    * ``[K]`` - one coefficient per row, evaluated at ``f_ref``. Every
      subcarrier reuses it and the kernel owns the whole ``n * df`` phase. This
      is the narrowband route and it is unchanged, bit for bit.
    * ``[K, num_subcarriers]`` - the ADR-042 wideband response, column ``n``
      evaluated at ``f_ref + n * df``. Each column already carries its own
      absolute phase at the frozen delay, so the kernel applies only that
      phase's slow-time change.

    A ``[K, F]`` weight with ``F != num_subcarriers`` is refused before any
    launch: the kernel pairs column ``n`` with subcarrier ``n`` and there is no
    interpolation to fill a coarser grid in.

    This is the ROW-LEVEL entry, and what it does NOT check is part of its
    contract. There is no :class:`SynthesisPathBatch` here, so the provenance
    rules do not run: nothing verifies that this weight's spreading, transmit
    power, and reference phase are absent from the spec's owners.
    Spec-internal refusals still apply. The validated route is
    :func:`synthesize_ofdm`, which asks the batch first; use this one only
    where the caller owns the single-count rule itself. R-ADR-010 records the
    split.

    ``total_delay_residual_s`` is the composed rows' non-differentiable delay
    residual (R-ADR-040), read as ``tau = total_delay_s + residual`` in
    float64; ``None`` reads the float32 delay alone.

    Returns ``complex64[num_symbols, num_segments, num_subcarriers]``.
    """

    path_count = int(total_delay_s.shape[0])
    if transfer_ref.ndim == 1:
        columns = 1
        if transfer_ref.shape != total_delay_s.shape:
            raise ValueError("transfer_ref and total_delay_s must have the same shape")
    elif transfer_ref.ndim == 2:
        columns = int(transfer_ref.shape[1])
        if int(transfer_ref.shape[0]) != path_count:
            raise ValueError(
                f"a wideband transfer_ref must carry one row per path; got "
                f"{int(transfer_ref.shape[0])} rows for {path_count} paths"
            )
        if columns != spec.num_subcarriers:
            raise ValueError(
                f"a wideband transfer_ref must carry one column per subcarrier; "
                f"got {columns} columns for num_subcarriers="
                f"{spec.num_subcarriers}. The kernel pairs column n with "
                "subcarrier n and does not interpolate between them"
            )
    else:
        raise ValueError(
            "transfer_ref must be [paths] (narrowband) or [paths, subcarriers] "
            f"(wideband), got rank {transfer_ref.ndim}"
        )
    if delay_rate is None:
        rate = torch.zeros_like(total_delay_s)
    else:
        if delay_rate.shape != total_delay_s.shape:
            raise ValueError("delay_rate and total_delay_s must have the same shape")
        rate = delay_rate
    segment = segment_of_each_row(pair_offsets, path_count)
    # Torch-owned, autograd-aware accessors: the real pair crosses the
    # boundary, never the complex tensor.
    out_re, out_im = _OfdmCfrSynthesis.apply(
        total_delay_s.contiguous(),
        delay_residual(total_delay_s, total_delay_residual_s),
        rate.contiguous(),
        transfer_ref.real.contiguous(),
        transfer_ref.imag.contiguous(),
        pair_offsets.contiguous(),
        segment,
        spec,
        columns,
    )
    return torch.complex(out_re, out_im)


@functools.lru_cache(maxsize=32)
def symbol_tensor(spec: OfdmSpec, device: torch.device) -> torch.Tensor:
    """The spec's symbol table as the kernel reads it, ``float32[L, N, 2]``, uploaded once per spec."""

    table = spec.symbol_table()
    return torch.from_numpy(np.stack((table.real, table.imag), axis=-1)).to(device=device, dtype=torch.float32)


class _OfdmTimeSynthesis(torch.autograd.Function):
    """Autograd bridge for the three native time-domain OFDM operators."""

    @staticmethod
    def forward(
        tau_rt: torch.Tensor,
        tau_lo: torch.Tensor,
        tau_rate: torch.Tensor,
        weight_re: torch.Tensor,
        weight_im: torch.Tensor,
        offsets: torch.Tensor,
        segment: torch.Tensor,
        symbols: torch.Tensor,
        schedule: torch.Tensor,
        spec: OfdmSpec,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out_re = torch.empty(
            (spec.num_symbols, int(offsets.shape[0]) - 1, spec.samples_per_symbol),
            dtype=torch.float32,
            device=tau_rt.device,
        )
        out_im = torch.empty_like(out_re)
        _ops().ofdm_time_forward(
            tau_rt,
            tau_lo,
            tau_rate,
            weight_re,
            weight_im,
            offsets,
            symbols,
            schedule,
            out_re,
            out_im,
            int(tau_rt.shape[0]),
            int(offsets.shape[0]) - 1,
            *_time_arguments(spec),
        )
        return out_re, out_im

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: tuple[torch.Tensor, torch.Tensor]) -> None:
        (tau_rt, tau_lo, tau_rate, weight_re, weight_im, offsets, segment, symbols, schedule, spec) = inputs
        ctx.spec = spec
        ctx.num_segments = int(offsets.shape[0]) - 1
        ctx.save_for_backward(tau_rt, tau_lo, tau_rate, weight_re, weight_im, segment, symbols, schedule)
        ctx.save_for_forward(tau_rt, tau_lo, tau_rate, weight_re, weight_im, offsets, symbols, schedule)

    @staticmethod
    @first_order_only
    def backward(ctx: Any, grad_re: torch.Tensor, grad_im: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        tau_rt, tau_lo, tau_rate, weight_re, weight_im, segment, symbols, schedule = ctx.saved_tensors
        spec = ctx.spec
        paths = int(tau_rt.shape[0])
        partial = torch.zeros((paths, spec.num_symbols, 4), dtype=torch.float64, device=tau_rt.device)
        _ops().ofdm_time_backward(
            tau_rt,
            tau_lo,
            tau_rate,
            weight_re,
            weight_im,
            segment,
            symbols,
            schedule,
            grad_re.contiguous(),
            grad_im.contiguous(),
            partial,
            paths,
            ctx.num_segments,
            *_time_arguments(spec),
        )
        # One reduction over the symbol axis of the kernel's per-(path, symbol)
        # partial sums, in a fixed order.
        total = partial.sum(dim=1).to(torch.float32)
        return (total[:, 0], None, total[:, 1], total[:, 2], total[:, 3], None, None, None, None, None)

    @staticmethod
    def jvp(
        ctx: Any,
        t_tau: Any,
        t_lo: Any,
        t_rate: Any,
        t_re: Any,
        t_im: Any,
        t_offsets: Any,
        t_segment: Any,
        t_symbols: Any,
        t_schedule: Any,
        t_spec: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tau_rt, tau_lo, tau_rate, weight_re, weight_im, offsets, symbols, schedule = ctx.saved_tensors
        spec = ctx.spec
        zero = torch.zeros_like(tau_rt)
        out_re = torch.empty(
            (spec.num_symbols, ctx.num_segments, spec.samples_per_symbol), dtype=torch.float32, device=tau_rt.device
        )
        out_im = torch.empty_like(out_re)
        _ops().ofdm_time_jvp(
            tau_rt,
            tau_lo,
            tau_rate,
            weight_re,
            weight_im,
            offsets,
            symbols,
            schedule,
            zero if t_tau is None else t_tau.contiguous(),
            zero if t_rate is None else t_rate.contiguous(),
            zero if t_re is None else t_re.contiguous(),
            zero if t_im is None else t_im.contiguous(),
            out_re,
            out_im,
            int(tau_rt.shape[0]),
            ctx.num_segments,
            *_time_arguments(spec),
        )
        return out_re, out_im


def _time_arguments(spec: OfdmSpec) -> tuple:
    return (
        spec.num_symbols,
        spec.num_subcarriers,
        spec.samples_per_symbol,
        spec.subcarrier_spacing_hz,
        spec.cyclic_prefix_s,
        spec.carrier_hz,
        spec.carrier_rate_hz,
    )


def synthesize_ofdm_time_rows(
    total_delay_s: torch.Tensor,
    delay_rate: torch.Tensor | None,
    transfer_ref: torch.Tensor,
    pair_offsets: torch.Tensor,
    spec: OfdmSpec,
    *,
    symbols: torch.Tensor | None = None,
    segment_schedule: torch.Tensor | None = None,
    total_delay_residual_s: torch.Tensor | None = None,
) -> torch.Tensor:
    """The received post-prefix time samples of a modulated OFDM stream.

    Returns ``complex64[num_symbols, segments, samples_per_symbol]`` in the
    CHANNEL convention; ``ofdm_time.cu`` states the closed form. The delay walks
    inside each symbol with the row's delay rate, so motion produces
    inter-carrier interference, and an echo longer than the prefix reads the
    previous symbol.

    ``symbols`` is the frame's symbol table ``float32[L, N, 2]`` (default: this
    spec's own, :func:`symbol_tensor`); a refreshed slot batch passes the
    frame's table with a one-symbol spec. ``segment_schedule`` is
    ``float64[segments, 2]``: each segment's window start within its symbol [s]
    and its symbol index. Both are waveform and schedule data; a derivative in
    either is refused. ``total_delay_residual_s`` is the composed rows'
    non-differentiable delay residual (R-ADR-040); ``None`` reads the float32
    delay alone. Narrowband weights only. Row-level
    entry: the provenance rules of :func:`synthesize_ofdm` do not run here.
    """

    if transfer_ref.shape != total_delay_s.shape:
        raise ValueError("the time-domain route takes a narrowband [paths] transfer_ref")
    rate = torch.zeros_like(total_delay_s) if delay_rate is None else delay_rate
    if rate.shape != total_delay_s.shape:
        raise ValueError("delay_rate and total_delay_s must have the same shape")
    device = total_delay_s.device
    table = symbol_tensor(spec, device) if symbols is None else symbols
    refuse_derivative("OFDM symbol table", "the transmitted symbols are waveform data", symbols=table)
    num_segments = int(pair_offsets.shape[0]) - 1
    if segment_schedule is None:
        schedule = torch.empty((0, 2), dtype=torch.float64, device=device)
    else:
        refuse_derivative("OFDM segment schedule", "sample instants are fixed metadata", schedule=segment_schedule)
        if segment_schedule.dtype != torch.float64 or tuple(segment_schedule.shape) != (num_segments, 2):
            raise ValueError(
                f"segment_schedule must be float64 with shape ({num_segments}, 2), got "
                f"{segment_schedule.dtype} {tuple(segment_schedule.shape)}"
            )
        schedule = segment_schedule.contiguous()
    segment = segment_of_each_row(pair_offsets, int(total_delay_s.shape[0]))
    out_re, out_im = _OfdmTimeSynthesis.apply(
        total_delay_s.contiguous(),
        delay_residual(total_delay_s, total_delay_residual_s),
        rate.contiguous(),
        transfer_ref.real.contiguous(),
        transfer_ref.imag.contiguous(),
        pair_offsets.contiguous(),
        segment,
        table.contiguous(),
        schedule,
        spec,
    )
    return torch.complex(out_re, out_im)


def synthesize_ofdm(batch: SynthesisPathBatch, spec: OfdmSpec) -> torch.Tensor:
    """Synthesize one OFDM frame's channel frequency response cube.

    Returns ``complex64[num_symbols, sensor_pair_count, num_subcarriers]`` in
    the CHANNEL phasor convention (``spec.phasor``).
    ``assembly.assemble_frame_cube`` turns it into the rank-4
    ``(TX, RX, symbol, subcarrier)`` layout an array processor indexes.

    A modulated stream or a time output (``spec.time_domain``) is synthesized
    by :func:`synthesize_ofdm_time_rows`; the unmodulated comb as a CFR keeps
    the closed-form CFR kernel.

    :func:`~witwin.radar.synthesis.assembly.require_ofdm_compatible` runs
    FIRST, so a weight and a spec that would count the carrier, the spreading,
    or the Doppler twice - or an echo window that does not fit inside the cyclic
    prefix - are refused before any kernel launch rather than producing a
    plausible cube that is wrong by a factor nobody notices.

    A dead row contributes exactly zero. That is enforced on the WEIGHT, with
    ``torch.where``, so the row is inert in the primal and carries no gradient
    to anything it was built from. Zeroing the output afterwards would leave a
    live gradient path back through a row that does not exist.
    """

    require_ofdm_compatible(batch, spec)
    transfer = batch.complex_transfer_ref
    if batch.complex_transfer_band is not None:
        # A wideband batch's column ``n`` was evaluated at f_ref + n * df, the
        # subcarrier the kernel pairs it with; any other grid is refused.
        grid = tuple(float(n) * spec.subcarrier_spacing_hz for n in range(spec.num_subcarriers))
        if batch.frequency_offsets_hz != grid:
            raise ValueError(
                "a wideband batch's frequency columns must be this spec's subcarriers, f_ref + n * df for "
                f"n in [0, {spec.num_subcarriers})"
            )
        transfer = batch.complex_transfer_band
    if batch.row_valid is not None:
        valid = batch.row_valid if transfer.ndim == 1 else batch.row_valid[:, None]
        transfer = torch.where(valid, transfer, torch.zeros_like(transfer))
    if spec.time_domain:
        if transfer.ndim != 1:
            raise NotImplementedError(
                "a wideband batch carries one weight per subcarrier and only the OFDM CFR family consumes it; the "
                "time-domain OFDM family synthesizes each row as one tap and would use the reference column alone"
            )
        return synthesize_ofdm_time_rows(
            batch.total_delay_s,
            batch.delay_rate,
            transfer,
            batch.pair_offsets,
            spec,
            total_delay_residual_s=batch.total_delay_residual_s,
        )
    return synthesize_cfr_rows(
        batch.total_delay_s,
        batch.delay_rate,
        transfer,
        batch.pair_offsets,
        spec,
        total_delay_residual_s=batch.total_delay_residual_s,
    )


#: Columns of one OFDM cross-radar link row (R-ADR-051), in the kernel's order.
OFDM_CROSS_COLUMNS = (
    "tx_frame_shift_s",
    "tx_offset_hz",
    "rx_offset_hz",
    "rx_frame_start_s",
    "tx_symbol_period_s",
    "tx_symbol_count",
    "tx_cyclic_prefix_s",
    "tx_subcarrier_spacing_hz",
    "tx_subcarrier_count",
    "tx_first_symbol_entry",
    "tx_inverse_sqrt_n",
)


def synthesize_ofdm_cross(
    transfer: torch.Tensor,
    delay: torch.Tensor,
    residual: torch.Tensor | None,
    valid: torch.Tensor | None,
    routing: CrossRouting,
) -> torch.Tensor:
    """A sampled receiver's samples of other radars' OFDM streams, ``complex64[cells, num_samples]`` (R-ADR-051).

    Channel convention. ``routing.link_table`` rows follow
    :data:`OFDM_CROSS_COLUMNS`; ``routing.aux`` is the transmitters' symbol
    tables, ``float32[entries, 2]``. See
    :func:`~witwin.radar.synthesis.assembly.synthesize_cross`.
    """

    return synthesize_cross("ofdm", transfer, delay, residual, valid, routing)


__all__ = [
    "OFDM_CROSS_COLUMNS",
    "OfdmSpec",
    "symbol_tensor",
    "synthesize_cfr_rows",
    "synthesize_ofdm",
    "synthesize_ofdm_cross",
    "synthesize_ofdm_time_rows",
]
