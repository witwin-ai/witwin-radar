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
  travels as data on :data:`~witwin.radar.synthesis.contracts.CHANNEL_PHASOR`
  and ``OfdmCfrSpec.phasor``.
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

import torch
from torch.autograd.function import once_differentiable

from .assembly import segment_of_each_row
from .contracts import (
    OfdmCfrSpec,
    SynthesisPathBatch,
    require_ofdm_compatible,
)


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


class _OfdmCfrSynthesis(torch.autograd.Function):
    """Autograd bridge for the three native CFR operators."""

    @staticmethod
    def forward(
        tau_rt, tau_rate, weight_re, weight_im, offsets, segment, spec, columns
    ):
        num_paths = int(tau_rt.shape[0])
        num_segments = int(offsets.shape[0]) - 1
        out_re = torch.empty(
            (spec.num_symbols, num_segments, spec.num_subcarriers),
            dtype=torch.float32,
            device=tau_rt.device,
        )
        out_im = torch.empty_like(out_re)
        _ops().ofdm_cfr_forward(
            tau_rt,
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
    def setup_context(ctx, inputs, output):
        (
            tau_rt,
            tau_rate,
            weight_re,
            weight_im,
            offsets,
            segment,
            spec,
            columns,
        ) = inputs
        ctx.spec = spec
        ctx.columns = columns
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
        _ops().ofdm_cfr_backward(
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
            spec.num_symbols,
            spec.num_subcarriers,
            ctx.columns,
            spec.subcarrier_spacing_hz,
            spec.symbol_period_s,
            spec.carrier_hz,
            spec.carrier_rate_hz,
        )
        return (
            grad_tau_rt,
            grad_tau_rate,
            grad_weight_re,
            grad_weight_im,
            None,
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
        tan_columns,
    ):
        tau_rt, tau_rate, weight_re, weight_im, offsets = ctx.saved_tensors
        spec = ctx.spec
        zero = torch.zeros_like(tau_rt)
        zero_weight = torch.zeros_like(weight_re)
        tan_tau_rt = zero if tan_tau_rt is None else tan_tau_rt.contiguous()
        tan_tau_rate = zero if tan_tau_rate is None else tan_tau_rate.contiguous()
        tan_weight_re = (
            zero_weight if tan_weight_re is None else tan_weight_re.contiguous()
        )
        tan_weight_im = (
            zero_weight if tan_weight_im is None else tan_weight_im.contiguous()
        )
        tan_out_re = torch.empty(
            (spec.num_symbols, ctx.num_segments, spec.num_subcarriers),
            dtype=torch.float32,
            device=tau_rt.device,
        )
        tan_out_im = torch.empty_like(tan_out_re)
        _ops().ofdm_cfr_jvp(
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
    spec: OfdmCfrSpec,
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
    :func:`synthesize_ofdm_cfr`, which asks the batch first; use this one only
    where the caller owns the single-count rule itself. R-ADR-010 records the
    split.

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
        rate.contiguous(),
        transfer_ref.real.contiguous(),
        transfer_ref.imag.contiguous(),
        pair_offsets.contiguous(),
        segment,
        spec,
        columns,
    )
    return torch.complex(out_re, out_im)


def synthesize_ofdm_cfr(
    batch: SynthesisPathBatch, spec: OfdmCfrSpec
) -> torch.Tensor:
    """Synthesize one OFDM frame's channel frequency response cube.

    Returns ``complex64[num_symbols, sensor_pair_count, num_subcarriers]`` in
    the CHANNEL phasor convention (``spec.phasor``).
    ``assembly.assemble_frame_cube`` turns it into the rank-4
    ``(TX, RX, symbol, subcarrier)`` layout an array processor indexes.

    :func:`~witwin.radar.synthesis.contracts.require_ofdm_compatible` runs
    FIRST, so a weight and a spec that would count the carrier, the spreading,
    or the Doppler twice - or an echo window that does not fit inside the cyclic
    prefix - are refused before any kernel launch rather than producing a
    plausible cube that is wrong by a factor nobody notices.

    When the batch carries a ``frequency_response`` the cube is WIDEBAND: each
    subcarrier consumes its own column, evaluated by Channel at that
    subcarrier's frequency, instead of the reference coefficient. The choice is
    made by the batch, not by an argument here, because a band that arrived and
    was not consumed is exactly the silent narrowband answer rule R8 exists to
    prevent.

    A dead row contributes exactly zero. That is enforced on the WEIGHT, with
    ``torch.where``, so the row is inert in the primal and carries no gradient
    to anything it was built from. Zeroing the output afterwards would leave a
    live gradient path back through a row that does not exist. The mask is
    ``[K]`` and broadcasts over the band: whether a row exists is geometry, not
    frequency.
    """

    require_ofdm_compatible(batch, spec)
    if batch.frequency_response is None:
        transfer = batch.complex_transfer_ref
        mask = batch.row_valid
    else:
        transfer = batch.frequency_response
        mask = None if batch.row_valid is None else batch.row_valid.unsqueeze(1)
    if mask is not None:
        transfer = torch.where(mask, transfer, torch.zeros_like(transfer))
    return synthesize_cfr_rows(
        batch.total_delay_s,
        batch.delay_rate,
        transfer,
        batch.pair_offsets,
        spec,
    )


__all__ = [
    "OfdmCfrSpec",
    "synthesize_cfr_rows",
    "synthesize_ofdm_cfr",
]
