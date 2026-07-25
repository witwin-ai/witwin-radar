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
from torch.autograd.function import once_differentiable

from ..paths.contracts import RadarPathBatch
from .contracts import FmcwBeatSpec


def _ops():
    from ..cuda import build

    return build.build_extension()


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
    def forward(tau_rt, tau_rate, weight_re, weight_im, offsets, segment, spec):
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
            out_re,
            out_im,
            num_paths,
            num_segments,
            spec.num_chirps,
            spec.num_samples,
            spec.sample_period_s,
            spec.chirp_period_s,
            spec.slope_hz_per_s,
            spec.carrier_hz,
            spec.t_start_s,
        )
        return out_re, out_im

    @staticmethod
    def setup_context(ctx, inputs, output):
        tau_rt, tau_rate, weight_re, weight_im, offsets, segment, spec = inputs
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
        _ops().fmcw_beat_backward(
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
            spec.num_chirps,
            spec.num_samples,
            spec.sample_period_s,
            spec.chirp_period_s,
            spec.slope_hz_per_s,
            spec.carrier_hz,
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
            tan_tau_rt,
            tan_tau_rate,
            tan_weight_re,
            tan_weight_im,
            tan_out_re,
            tan_out_im,
            int(tau_rt.shape[0]),
            ctx.num_segments,
            spec.num_chirps,
            spec.num_samples,
            spec.sample_period_s,
            spec.chirp_period_s,
            spec.slope_hz_per_s,
            spec.carrier_hz,
            spec.t_start_s,
        )
        return tan_out_re, tan_out_im


def synthesize_beat_rows(
    total_delay_s: torch.Tensor,
    delay_rate: torch.Tensor | None,
    beat_weight: torch.Tensor,
    pair_offsets: torch.Tensor,
    spec: FmcwBeatSpec,
) -> torch.Tensor:
    """Synthesize complex IQ from already-conjugated beat weights.

    ``beat_weight`` is in the BEAT convention. Use
    :func:`channel_phasor_to_beat_weight` to get there from a Channel transfer
    coefficient; this entry does not convert, so a caller cannot accidentally
    conjugate twice.
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
        spec,
    )
    return torch.complex(out_re, out_im)


def synthesize_fmcw_beat(paths: RadarPathBatch, spec: FmcwBeatSpec) -> torch.Tensor:
    """Synthesize one frame of complex IQ from composed round-trip rows.

    Returns ``complex64[num_chirps, sensor_pair_count, num_samples]``.

    A dead row contributes exactly zero. That is enforced on the WEIGHT, with
    ``torch.where``, so the row is inert in the primal and carries no gradient
    to anything it was built from. Zeroing the output afterwards would leave a
    live gradient path back through a row that does not exist.
    """

    weight = channel_phasor_to_beat_weight(paths.complex_transfer_ref)
    if paths.row_valid is not None:
        weight = torch.where(
            paths.row_valid, weight, torch.zeros_like(weight)
        )
    return synthesize_beat_rows(
        paths.total_delay_s,
        paths.delay_rate,
        weight,
        paths.pair_offsets,
        spec,
    )


__all__ = [
    "FmcwBeatSpec",
    "channel_phasor_to_beat_weight",
    "synthesize_beat_rows",
    "synthesize_fmcw_beat",
]
