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
    def forward(tau_rt, tau_rate, weight_re, weight_im, offsets, segment, spec):
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

    Returns ``complex64[num_symbols, num_segments, num_subcarriers]``.
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
    out_re, out_im = _OfdmCfrSynthesis.apply(
        total_delay_s.contiguous(),
        rate.contiguous(),
        transfer_ref.real.contiguous(),
        transfer_ref.imag.contiguous(),
        pair_offsets.contiguous(),
        segment,
        spec,
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

    A dead row contributes exactly zero. That is enforced on the WEIGHT, with
    ``torch.where``, so the row is inert in the primal and carries no gradient
    to anything it was built from. Zeroing the output afterwards would leave a
    live gradient path back through a row that does not exist.
    """

    require_ofdm_compatible(batch, spec)
    transfer = batch.complex_transfer_ref
    if batch.row_valid is not None:
        transfer = torch.where(
            batch.row_valid, transfer, torch.zeros_like(transfer)
        )
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
