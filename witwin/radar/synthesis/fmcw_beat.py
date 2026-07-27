"""Native FMCW beat synthesis: the Radar-owned waveform hot loop.

The per-path, per-sample sum

    s[c][p][m] = sum_k C[k] * exp(+j * 2 * pi * cycles(tau[k], t_c, t_m))

runs entirely inside a CUDA kernel. Torch's role here is validation, buffer
allocation, autograd dispatch, and result assembly; it never evaluates the
phasor sum. Three registered operators  -  forward, backward, jvp  -  have
exactly one Python owner, this module.

Two structural contracts, each with a test:

* The facade ALWAYS routes through ``Function.apply``. It must never replicate
  the eager shortcut in ``solvers/solver_dirichlet.py``, which checks
  ``requires_grad`` and bypasses autograd when it is false. An ADR-038
  forward-only dual has ``requires_grad == False``, so that shortcut silently
  swallows its tangent and returns a plain tensor.
* No complex tensor crosses the autograd boundary. The public entry splits a
  complex weight into real and imaginary parts with Torch's own autograd-aware
  accessors and recombines the output the same way, which makes the
  conjugate-Wirtinger convention question structurally impossible to get wrong.
"""

from __future__ import annotations

import torch
from ..ad_contracts import first_order_only

from .assembly import pair_tx_index
from .contracts import FmcwBeatSpec, SynthesisPathBatch, require_compatible


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


def channel_phasor_to_beat_weight(coefficient: torch.Tensor) -> torch.Tensor:
    """Convert a Channel transfer coefficient into an FMCW beat weight.

    Channel publishes ``exp(-j * k * d)`` under an ``exp(+j * 2 * pi * f * t)``
    time dependence. FMCW de-chirping multiplies the received signal by the
    conjugate of the transmitted chirp, so the beat-domain phasor advances with
    ``+j``. The two conventions are therefore conjugates, and a Channel
    coefficient becomes a beat weight by conjugation.

    This is the ONE conversion site. A complex target response is authored in
    the Channel convention, because it multiplies transports authored there;
    it is converted here along with everything else it multiplies.
    """

    if coefficient.dtype not in (torch.complex64, torch.complex128):
        raise TypeError(
            f"a Channel transfer coefficient must be complex, got {coefficient.dtype}"
        )
    return torch.conj(coefficient).resolve_conj()


def _segment_of_each_path(
    path_offsets: torch.Tensor, path_count: int
) -> torch.Tensor:
    """Map each compact row to its sensor-pair segment.

    ``right=True`` is required: an offsets table is a half-open partition, so
    a row whose index equals a boundary belongs to the NEXT segment. The output
    shape comes from ``path_count``, a host int the compact contract already
    published, so this adds no cardinality observation.
    """

    rows = torch.arange(path_count, device=path_offsets.device, dtype=torch.int64)
    return torch.bucketize(rows, path_offsets[1:], right=True)


class _FmcwBeatSynthesis(torch.autograd.Function):
    """Autograd bridge for the three native beat operators."""

    @staticmethod
    def forward(
        tau_rt, tau_rate, weight_re, weight_im, offsets, segment, tx_index, spec
    ):
        num_paths = int(tau_rt.shape[0])
        num_segments = int(offsets.shape[0]) - 1
        out_re = torch.empty(
            (spec.num_chirps, num_segments, spec.num_samples),
            dtype=torch.float32,
            device=tau_rt.device,
        )
        out_im = torch.empty_like(out_re)
        _ops().fmcw_beat_forward(
            tau_rt,
            tau_rate,
            weight_re,
            weight_im,
            offsets,
            tx_index,
            out_re,
            out_im,
            num_paths,
            num_segments,
            spec.num_tx,
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
        (
            tau_rt,
            tau_rate,
            weight_re,
            weight_im,
            offsets,
            segment,
            tx_index,
            spec,
        ) = inputs
        ctx.spec = spec
        ctx.num_segments = int(offsets.shape[0]) - 1
        ctx.save_for_backward(
            tau_rt, tau_rate, weight_re, weight_im, segment, tx_index
        )
        ctx.save_for_forward(
            tau_rt, tau_rate, weight_re, weight_im, offsets, tx_index
        )

    @staticmethod
    @first_order_only
    def backward(ctx, grad_out_re, grad_out_im):
        (
            tau_rt,
            tau_rate,
            weight_re,
            weight_im,
            segment,
            tx_index,
        ) = ctx.saved_tensors
        spec = ctx.spec
        grad_tau_rt = torch.empty_like(tau_rt)
        grad_tau_rate = torch.empty_like(tau_rate)
        grad_weight_re = torch.empty_like(weight_re)
        grad_weight_im = torch.empty_like(weight_im)
        _ops().fmcw_beat_backward(
            tau_rt,
            tau_rate,
            weight_re,
            weight_im,
            segment,
            tx_index,
            grad_out_re.contiguous(),
            grad_out_im.contiguous(),
            grad_tau_rt,
            grad_tau_rate,
            grad_weight_re,
            grad_weight_im,
            int(tau_rt.shape[0]),
            ctx.num_segments,
            spec.num_tx,
            spec.num_chirps,
            spec.num_samples,
            spec.sample_period_s,
            spec.chirp_period_s,
            spec.slope_hz_per_s,
            spec.carrier_hz,
            spec.carrier_rate_hz,
            spec.t_start_s,
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
        tan_tx_index,
        tan_spec,
    ):
        (
            tau_rt,
            tau_rate,
            weight_re,
            weight_im,
            offsets,
            tx_index,
        ) = ctx.saved_tensors
        spec = ctx.spec
        zero = torch.zeros_like(tau_rt)
        tan_tau_rt = zero if tan_tau_rt is None else tan_tau_rt.contiguous()
        tan_tau_rate = zero if tan_tau_rate is None else tan_tau_rate.contiguous()
        tan_weight_re = zero if tan_weight_re is None else tan_weight_re.contiguous()
        tan_weight_im = zero if tan_weight_im is None else tan_weight_im.contiguous()
        tan_out_re = torch.empty(
            (spec.num_chirps, ctx.num_segments, spec.num_samples),
            dtype=torch.float32,
            device=tau_rt.device,
        )
        tan_out_im = torch.empty_like(tan_out_re)
        _ops().fmcw_beat_jvp(
            tau_rt,
            tau_rate,
            weight_re,
            weight_im,
            offsets,
            tx_index,
            tan_tau_rt,
            tan_tau_rate,
            tan_weight_re,
            tan_weight_im,
            tan_out_re,
            tan_out_im,
            int(tau_rt.shape[0]),
            ctx.num_segments,
            spec.num_tx,
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


def synthesize_beat_rows(
    total_delay_s: torch.Tensor,
    delay_rate: torch.Tensor | None,
    beat_weight: torch.Tensor,
    pair_offsets: torch.Tensor,
    spec: FmcwBeatSpec,
    *,
    segment_tx_index: torch.Tensor | None = None,
) -> torch.Tensor:
    """Synthesize complex IQ from already-conjugated beat weights.

    ``beat_weight`` is in the BEAT convention. Use
    :func:`channel_phasor_to_beat_weight` to get there from a Channel transfer
    coefficient; this entry does not convert, so a caller cannot accidentally
    conjugate twice.

    ``segment_tx_index`` names which transmitter drives each sensor-pair
    segment, which is what turns the slow-time axis into TDM slot time. It may
    be omitted only when ``spec.num_tx == 1``, where every segment is in slot
    ``chirp`` and the question does not arise; a multi-TX spec must say, because
    guessing it would put a whole chirp period of Doppler walk on the wrong
    channel and still produce a plausible cube. :func:`synthesize_fmcw_beat`
    derives it from the array layout.

    This is the ROW-LEVEL entry, and what it does NOT check is part of its
    contract. There is no :class:`SynthesisPathBatch` here, so the eight
    provenance rules do not run: nothing verifies that this weight's spreading,
    transmit power, and reference phase are absent from the spec's owners.
    Spec-internal refusals still apply - a spec that puts the carrier in two
    homes is refused here as well. The validated route is
    :func:`synthesize_fmcw_beat`, which asks the batch first; use this one only
    where the caller owns the single-count rule itself. R-ADR-010 records the
    split.
    """

    path_count = int(total_delay_s.shape[0])
    if beat_weight.shape != total_delay_s.shape:
        raise ValueError("beat_weight and total_delay_s must have the same shape")
    if delay_rate is None:
        rate = torch.zeros_like(total_delay_s)
    else:
        if delay_rate.shape != total_delay_s.shape:
            raise ValueError("delay_rate and total_delay_s must have the same shape")
        rate = delay_rate
    num_segments = int(pair_offsets.shape[0]) - 1
    if segment_tx_index is None:
        if spec.num_tx != 1:
            raise ValueError(
                f"this spec declares num_tx={spec.num_tx}, so every sensor-pair "
                "segment must name the transmitter that drives it; pass "
                "segment_tx_index (witwin.radar.synthesis.assembly.pair_tx_index "
                "derives it from the array layout)"
            )
        tx_index = torch.zeros(
            num_segments, dtype=torch.int32, device=total_delay_s.device
        )
    else:
        if tuple(segment_tx_index.shape) != (num_segments,):
            raise ValueError(
                "segment_tx_index must hold one transmitter index per sensor-pair "
                f"segment, expected shape ({num_segments},), got "
                f"{tuple(segment_tx_index.shape)}"
            )
        tx_index = segment_tx_index
    segment = _segment_of_each_path(pair_offsets, path_count)
    # Torch-owned, autograd-aware accessors: the real pair crosses the
    # boundary, never the complex tensor.
    out_re, out_im = _FmcwBeatSynthesis.apply(
        total_delay_s.contiguous(),
        rate.contiguous(),
        beat_weight.real.contiguous(),
        beat_weight.imag.contiguous(),
        pair_offsets.contiguous(),
        segment,
        tx_index.contiguous(),
        spec,
    )
    return torch.complex(out_re, out_im)


def synthesize_fmcw_beat(
    batch: SynthesisPathBatch, spec: FmcwBeatSpec
) -> torch.Tensor:
    """Synthesize one frame of complex IQ from a synthesis path batch.

    Returns ``complex64[num_chirps, sensor_pair_count, num_samples]``, one
    rank-3 cube in the BEAT convention. ``assembly.assemble_frame_cube`` turns
    it into the rank-4 ``(TX, RX, chirp, sample)`` layout ``sigproc`` consumes.

    :func:`~witwin.radar.synthesis.contracts.require_compatible` runs FIRST, so
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
        weight = torch.where(
            batch.row_valid, weight, torch.zeros_like(weight)
        )
    return synthesize_beat_rows(
        batch.total_delay_s,
        batch.delay_rate,
        weight,
        batch.pair_offsets,
        spec,
        segment_tx_index=pair_tx_index(
            num_tx=spec.num_tx,
            num_rx=spec.num_rx,
            sensor_pair_count=batch.sensor_pair_count,
            device=batch.device,
        ),
    )


__all__ = [
    "FmcwBeatSpec",
    "channel_phasor_to_beat_weight",
    "synthesize_beat_rows",
    "synthesize_fmcw_beat",
]
