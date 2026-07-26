"""The single Python owner of the ``dirichlet_spectrum`` native family.

Nine registered operators - ``forward_chunked``,
``forward_mimo_linear_chunked``, ``dirichlet_jvp``, ``mimo_linear_backward``,
``mimo_linear_jvp``, ``backward``, ``backward_batched``,
``backward_parallel_bins``, ``backward_per_bin`` - are named, dispatched, and
buffered here and nowhere else.

This module used to be half of ``solvers/solver_dirichlet.py``, which is the
wrong home for two reasons. The first is ownership: the Dirichlet spectrum is
Radar's oldest waveform synthesis backend, and synthesis is where a waveform
backend belongs. The second is coupling: the old autograd bridge kept a live
``ctx.solver`` reference and read ``solver.radar.config`` inside ``backward``,
which means a backward pass read whatever the config happened to be at the time
it ran rather than what the forward was computed with. Everything this module
needs now arrives as a value, in :class:`DirichletSpectrumSpec`.

What stays in the solver is orchestration: the ``pad_factor`` / ``N_fft`` /
``num_bins`` / ``n`` / ``k0_per_meter`` derivation, the slot-group chunking, the
``torch.fft.ifft``, and the public method surface.

One structural rule, with a test: the spectrum ALWAYS routes through
``Function.apply``. The eager ``requires_grad`` shortcut this module replaced
bypassed autograd whenever no input required grad - and an ADR-038 forward-only
dual has ``requires_grad == False``, so that shortcut silently swallowed its
tangent and handed back a plain tensor. ``synthesis/fmcw_beat.py`` never had
the shortcut; neither does this.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.autograd.function import once_differentiable


_OPS = None


def _ops():
    """The native operator table, resolved once per process.

    Cached here as well as in the build module because this runs on every
    spectrum call, forward and backward: a per-launch import plus function call
    is pure overhead on the hot path. It is the same object the build module
    caches, so a test that monkeypatches one sees the other.
    """

    global _OPS
    if _OPS is None:
        from ..cuda import build

        _OPS = build.build_extension()
    return _OPS


@dataclass(frozen=True, slots=True)
class DirichletSpectrumSpec:
    """Every configuration scalar the family's kernels read, by value.

    By value is the point. The autograd context that this replaced held a
    reference to the live solver and reached through it to ``radar.config``
    during ``backward``; a config mutated between the forward and the backward
    silently produced a gradient of a different function than the one that was
    evaluated.

    ``tau_is_seconds`` selects what the path tensor holds, and the caller must
    supply the matching ``k0_per_meter`` scale:

    * ``0`` - a ONE-WAY distance in metres. ``tau = 2 d / c0`` (monostatic), and
      ``k0_per_meter = (slope * 2 / c0) * n_fft / fs``. This is the legacy
      Dirichlet input and is bit-identical to what the family produced before
      the flag existed.
    * ``1`` - a ROUND-TRIP delay in seconds, consumed directly, with
      ``k0_per_meter = slope * n_fft / fs``. Every Phase-6 contract speaks
      round-trip delay, and turning one back into a distance so the kernel can
      halve it again is how a path becomes self-consistently 2x wrong.

    ``fc`` is the carrier home, mirroring ``carrier_hz`` in the beat family:
    nonzero means this kernel owns ``2 pi fc tau``; zero means the weight
    already carries it. ``synthesis/contracts.py`` refuses the combination that
    would apply it twice, before any launch.
    """

    n: float
    k0_per_meter: float
    num_bins: int
    n_fft: int
    fc: float
    slope_hz_per_s: float
    t_start_s: float
    tau_is_seconds: int = 0


def _zeros_like_weight(weight: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(weight)


class _DirichletSpectrum(torch.autograd.Function):
    """Autograd bridge for the chunked Dirichlet spectrum.

    The complex weight crosses as ``(a_re, a_im)``, never as a complex tensor,
    matching the beat and join families: the conjugate-Wirtinger convention
    cannot be got wrong at a seam that has no complex tensor on it.
    """

    @staticmethod
    def forward(d, a_re, a_im, targets_per_spectrum, shared_gradient, spec):
        num_targets = int(d.shape[0])
        num_spectra = (num_targets + targets_per_spectrum - 1) // targets_per_spectrum
        output_re = torch.empty(
            (num_spectra, spec.num_bins), dtype=torch.float32, device=d.device
        )
        output_im = torch.empty_like(output_re)
        if num_targets:
            _ops().forward_chunked(
                d,
                a_re,
                a_im,
                output_re,
                output_im,
                spec.n,
                spec.k0_per_meter,
                spec.num_bins,
                spec.n_fft,
                num_targets,
                targets_per_spectrum,
                spec.fc,
                spec.slope_hz_per_s,
                spec.t_start_s,
                spec.tau_is_seconds,
            )
        return output_re, output_im

    @staticmethod
    def setup_context(ctx, inputs, output):
        d, a_re, a_im, targets_per_spectrum, shared_gradient, spec = inputs
        ctx.spec = spec
        ctx.targets_per_spectrum = targets_per_spectrum
        ctx.shared_gradient = shared_gradient
        ctx.save_for_backward(d, a_re, a_im)
        ctx.save_for_forward(d, a_re, a_im)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out_re, grad_out_im):
        d, a_re, a_im = ctx.saved_tensors
        spec = ctx.spec
        grad_d = torch.empty_like(d)
        grad_a_re = torch.empty_like(a_re)
        grad_a_im = torch.empty_like(a_im)
        if d.numel():
            gout_re = grad_out_re.contiguous()
            gout_im = grad_out_im.contiguous()
            if ctx.shared_gradient:
                # One spectrum shared by every target: the caller summed the
                # chunks, so every chunk sees the same cotangent row.
                _ops().backward_parallel_bins(
                    d,
                    a_re,
                    a_im,
                    gout_re[0],
                    gout_im[0],
                    grad_d,
                    grad_a_re,
                    grad_a_im,
                    spec.n,
                    spec.k0_per_meter,
                    spec.num_bins,
                    spec.n_fft,
                    d.numel(),
                    spec.fc,
                    spec.slope_hz_per_s,
                    spec.t_start_s,
                    spec.tau_is_seconds,
                )
            else:
                _ops().backward_batched(
                    d,
                    a_re,
                    a_im,
                    gout_re,
                    gout_im,
                    grad_d,
                    grad_a_re,
                    grad_a_im,
                    spec.n,
                    spec.k0_per_meter,
                    spec.num_bins,
                    spec.n_fft,
                    d.numel(),
                    ctx.targets_per_spectrum,
                    spec.fc,
                    spec.slope_hz_per_s,
                    spec.t_start_s,
                    spec.tau_is_seconds,
                )
        return grad_d, grad_a_re, grad_a_im, None, None, None

    @staticmethod
    def jvp(ctx, tan_d, tan_a_re, tan_a_im, tan_tps, tan_shared, tan_spec):
        d, a_re, a_im = ctx.saved_tensors
        spec = ctx.spec
        zero = torch.zeros_like(d)
        tan_d = zero if tan_d is None else tan_d.contiguous()
        tan_a_re = zero if tan_a_re is None else tan_a_re.contiguous()
        tan_a_im = zero if tan_a_im is None else tan_a_im.contiguous()
        num_targets = int(d.shape[0])
        num_spectra = (
            num_targets + ctx.targets_per_spectrum - 1
        ) // ctx.targets_per_spectrum
        tan_out_re = torch.empty(
            (num_spectra, spec.num_bins), dtype=torch.float32, device=d.device
        )
        tan_out_im = torch.empty_like(tan_out_re)
        if num_targets:
            _ops().dirichlet_jvp(
                d,
                a_re,
                a_im,
                tan_d,
                tan_a_re,
                tan_a_im,
                tan_out_re,
                tan_out_im,
                spec.n,
                spec.k0_per_meter,
                spec.num_bins,
                spec.n_fft,
                num_targets,
                ctx.targets_per_spectrum,
                spec.fc,
                spec.slope_hz_per_s,
                spec.t_start_s,
                spec.tau_is_seconds,
            )
        return tan_out_re, tan_out_im


def chunked_spectra(
    path_values: torch.Tensor,
    weight_re: torch.Tensor,
    weight_im: torch.Tensor | None = None,
    *,
    spec: DirichletSpectrumSpec,
    targets_per_spectrum: int,
    shared_gradient: bool = False,
) -> torch.Tensor:
    """Chunked Dirichlet spectra, shape ``(num_spectra, spec.num_bins)``.

    ``path_values`` is a one-way distance or a round-trip delay according to
    ``spec.tau_is_seconds``. ``weight_im`` defaults to zeros, which is exactly
    the legacy real-amplitude call and produces bit-identical output.
    """

    if targets_per_spectrum <= 0:
        raise ValueError("targets_per_spectrum must be positive.")
    if path_values.shape != weight_re.shape:
        raise ValueError("path values and weights must have the same shape.")
    if weight_im is None:
        weight_im = _zeros_like_weight(weight_re)
    elif weight_im.shape != weight_re.shape:
        raise ValueError("weight components must have the same shape.")
    out_re, out_im = _DirichletSpectrum.apply(
        path_values,
        weight_re,
        weight_im,
        targets_per_spectrum,
        shared_gradient,
        spec,
    )
    return torch.complex(out_re, out_im)


@dataclass(frozen=True, slots=True)
class MimoLinearFramePlan:
    """The TDM frame layout the fused MIMO launch and both companions share.

    One record, because the primal, the backward, and the jvp must agree on
    every one of these numbers exactly. Passing them separately three times is
    how a backward ends up integrating over a different frame than the forward.
    """

    targets_per_pair: int
    num_pairs: int
    chirp_per_frame: int
    chirp_period_s: float
    num_tx: int
    range_loss_update: bool

    def kernel_tail(self) -> tuple:
        return (
            self.targets_per_pair,
            self.chirp_per_frame,
            self.chirp_period_s,
            self.num_tx,
            1 if self.range_loss_update else 0,
        )


class _MimoLinearSpectrum(torch.autograd.Function):
    """Autograd bridge for the fused TDM-frame MIMO launch.

    Every mode goes through the native family. The Torch expression that used
    to serve the reverse-mode case - ``dist = d0 + rate * t`` with the
    ``d0 / clamp(dist)`` range-loss update, followed by a chunked forward over
    the expanded frame - was a second implementation of this kernel's physics.
    It ran in a different dtype, expanded a ``chirps x pairs x targets``
    intermediate the fused kernel never materialises, and was reachable only
    when an input happened to require grad, so the gradient and the value came
    from different code. It is deleted.
    """

    @staticmethod
    def forward(d0, d_rate, a_re, a_im, plan, spec):
        output_re = torch.zeros(
            (plan.chirp_per_frame, plan.num_pairs, spec.num_bins),
            dtype=torch.float32,
            device=d0.device,
        )
        output_im = torch.zeros_like(output_re)
        _ops().forward_mimo_linear_chunked(
            d0,
            d_rate,
            a_re,
            a_im,
            output_re,
            output_im,
            spec.n,
            spec.k0_per_meter,
            spec.num_bins,
            spec.n_fft,
            *plan.kernel_tail(),
            spec.fc,
            spec.slope_hz_per_s,
            spec.t_start_s,
            spec.tau_is_seconds,
        )
        return output_re, output_im

    @staticmethod
    def setup_context(ctx, inputs, output):
        d0, d_rate, a_re, a_im, plan, spec = inputs
        ctx.plan = plan
        ctx.spec = spec
        ctx.save_for_backward(d0, d_rate, a_re, a_im)
        ctx.save_for_forward(d0, d_rate, a_re, a_im)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out_re, grad_out_im):
        d0, d_rate, a_re, a_im = ctx.saved_tensors
        plan = ctx.plan
        spec = ctx.spec
        grad_d0 = torch.empty_like(d0)
        grad_d_rate = torch.empty_like(d_rate)
        grad_a_re = torch.empty_like(a_re)
        grad_a_im = torch.empty_like(a_im)
        _ops().mimo_linear_backward(
            d0,
            d_rate,
            a_re,
            a_im,
            grad_out_re.contiguous(),
            grad_out_im.contiguous(),
            grad_d0,
            grad_d_rate,
            grad_a_re,
            grad_a_im,
            spec.n,
            spec.k0_per_meter,
            spec.num_bins,
            spec.n_fft,
            *plan.kernel_tail(),
            spec.fc,
            spec.slope_hz_per_s,
            spec.t_start_s,
            spec.tau_is_seconds,
        )
        return grad_d0, grad_d_rate, grad_a_re, grad_a_im, None, None

    @staticmethod
    def jvp(ctx, tan_d0, tan_d_rate, tan_a_re, tan_a_im, tan_plan, tan_spec):
        d0, d_rate, a_re, a_im = ctx.saved_tensors
        plan = ctx.plan
        spec = ctx.spec
        zero = torch.zeros_like(d0)
        tan_d0 = zero if tan_d0 is None else tan_d0.contiguous()
        tan_d_rate = zero if tan_d_rate is None else tan_d_rate.contiguous()
        tan_a_re = zero if tan_a_re is None else tan_a_re.contiguous()
        tan_a_im = zero if tan_a_im is None else tan_a_im.contiguous()
        tan_out_re = torch.zeros(
            (plan.chirp_per_frame, plan.num_pairs, spec.num_bins),
            dtype=torch.float32,
            device=d0.device,
        )
        tan_out_im = torch.zeros_like(tan_out_re)
        _ops().mimo_linear_jvp(
            d0,
            d_rate,
            a_re,
            a_im,
            tan_d0,
            tan_d_rate,
            tan_a_re,
            tan_a_im,
            tan_out_re,
            tan_out_im,
            spec.n,
            spec.k0_per_meter,
            spec.num_bins,
            spec.n_fft,
            *plan.kernel_tail(),
            spec.fc,
            spec.slope_hz_per_s,
            spec.t_start_s,
            spec.tau_is_seconds,
        )
        return tan_out_re, tan_out_im


def mimo_linear_spectra(
    path_values: torch.Tensor,
    path_rates: torch.Tensor,
    weight_re: torch.Tensor,
    weight_im: torch.Tensor | None = None,
    *,
    spec: DirichletSpectrumSpec,
    plan: MimoLinearFramePlan,
) -> torch.Tensor:
    """One fused launch over the whole TDM frame, ``(chirps, pairs, bins)``.

    Primal, VJP, and JVP all belong to the ``dirichlet_spectrum`` family. There
    is no ``requires_grad`` branch here and no Torch replay behind one: the
    route a gradient takes is the route the value took.
    """

    if weight_im is None:
        weight_im = _zeros_like_weight(weight_re)
    out_re, out_im = _MimoLinearSpectrum.apply(
        path_values, path_rates, weight_re, weight_im, plan, spec
    )
    return torch.complex(out_re, out_im)


def spectrum_vjp(
    path_values: torch.Tensor,
    weight_re: torch.Tensor,
    grad_output_re: torch.Tensor,
    grad_output_im: torch.Tensor,
    *,
    spec: DirichletSpectrumSpec,
    weight_im: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One block per target, parallel bin reduction. Returns three gradients."""

    if weight_im is None:
        weight_im = _zeros_like_weight(weight_re)
    num_targets = int(path_values.shape[0])
    grad_d = torch.empty_like(path_values)
    grad_a_re = torch.empty_like(weight_re)
    grad_a_im = torch.empty_like(weight_im)
    _ops().backward_parallel_bins(
        path_values,
        weight_re,
        weight_im,
        grad_output_re,
        grad_output_im,
        grad_d,
        grad_a_re,
        grad_a_im,
        spec.n,
        spec.k0_per_meter,
        spec.num_bins,
        spec.n_fft,
        num_targets,
        spec.fc,
        spec.slope_hz_per_s,
        spec.t_start_s,
        spec.tau_is_seconds,
    )
    return grad_d, grad_a_re, grad_a_im


def spectrum_vjp_per_bin(
    path_values: torch.Tensor,
    weight_re: torch.Tensor,
    grad_output_re: torch.Tensor,
    grad_output_im: torch.Tensor,
    *,
    spec: DirichletSpectrumSpec,
    weight_im: torch.Tensor | None = None,
    bins_per_chunk: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One thread per spectrum bin, atomics into per-chunk slots.

    The per-chunk partial sums are reduced here rather than in the kernel, so
    the kernel keeps its slot ownership and the caller keeps a single summation
    order to reason about.
    """

    if weight_im is None:
        weight_im = _zeros_like_weight(weight_re)
    num_targets = int(path_values.shape[0])
    num_chunks = (spec.num_bins + bins_per_chunk - 1) // bins_per_chunk
    shape = (num_chunks, num_targets)
    grad_d = torch.zeros(shape, dtype=torch.float32, device=path_values.device)
    grad_a_re = torch.zeros_like(grad_d)
    grad_a_im = torch.zeros_like(grad_d)
    _ops().backward_per_bin(
        path_values,
        weight_re,
        weight_im,
        grad_output_re,
        grad_output_im,
        grad_d,
        grad_a_re,
        grad_a_im,
        spec.n,
        spec.k0_per_meter,
        spec.num_bins,
        spec.n_fft,
        num_targets,
        bins_per_chunk,
        spec.fc,
        spec.slope_hz_per_s,
        spec.t_start_s,
        spec.tau_is_seconds,
    )
    return grad_d.sum(dim=0), grad_a_re.sum(dim=0), grad_a_im.sum(dim=0)


def spectrum_vjp_single_block(
    path_values: torch.Tensor,
    weight_re: torch.Tensor,
    grad_output_re: torch.Tensor,
    grad_output_im: torch.Tensor,
    *,
    spec: DirichletSpectrumSpec,
    weight_im: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One thread per target, serial bin loop - the ``backward`` operator.

    Kept because it is a registered ABI symbol with its own contract test. It
    has no production caller: ``DirichletSolver.backward`` dispatches
    ``backward_parallel_bins``. The manifest used to name that method as this
    symbol's end-to-end caller, which was false; it now records the symbol as
    ``caller_status: test_only`` and the manifest test caps the number of
    caller-free symbols at this one. Recorded rather than silently deleted,
    because removing a manifested symbol is a deliberate change to the kernel,
    the extension, and the load probe, not a side effect of a review.
    """

    if weight_im is None:
        weight_im = _zeros_like_weight(weight_re)
    num_targets = int(path_values.shape[0])
    grad_d = torch.empty_like(path_values)
    grad_a_re = torch.empty_like(weight_re)
    grad_a_im = torch.empty_like(weight_im)
    _ops().backward(
        path_values,
        weight_re,
        weight_im,
        grad_output_re,
        grad_output_im,
        grad_d,
        grad_a_re,
        grad_a_im,
        spec.n,
        spec.k0_per_meter,
        spec.num_bins,
        spec.n_fft,
        num_targets,
        spec.fc,
        spec.slope_hz_per_s,
        spec.t_start_s,
        spec.tau_is_seconds,
    )
    return grad_d, grad_a_re, grad_a_im


__all__ = [
    "DirichletSpectrumSpec",
    "MimoLinearFramePlan",
    "chunked_spectra",
    "mimo_linear_spectra",
    "spectrum_vjp",
    "spectrum_vjp_per_bin",
    "spectrum_vjp_single_block",
]
