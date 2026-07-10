"""
Dirichlet kernel solver backend.

Computes the range spectrum directly in the frequency domain and keeps all
backend-specific state on the solver instance.
"""

from __future__ import annotations

import torch
from torch.autograd.function import once_differentiable

from ..path_cache import MimoPathCache
from . import Solver
from .common import (
    collect_interpolated_samples,
    compute_path_amplitudes,
    compute_slot_path_tensors,
    compute_total_path_lengths,
    normalize_interpolated_sample,
    samples_require_grad,
)


def _load_module():
    from ..cuda import build

    return build.build_extension()


def _to_f32(solver: "DirichletSolver", value: torch.Tensor) -> torch.Tensor:
    return value.to(dtype=torch.float32, device=solver.device).contiguous()


def _same_tensor_storage(a: torch.Tensor | None, b: torch.Tensor | None) -> bool:
    if a is None or b is None:
        return a is b
    return a.shape == b.shape and a.data_ptr() == b.data_ptr()


def _samples_share_storage(samples) -> bool:
    if not samples:
        return False
    first = samples[0]
    return all(
        _same_tensor_storage(first.points, sample.points)
        and _same_tensor_storage(first.intensities, sample.intensities)
        and _same_tensor_storage(first.entry_points, sample.entry_points)
        and _same_tensor_storage(first.fixed_path_lengths, sample.fixed_path_lengths)
        and _same_tensor_storage(first.depths, sample.depths)
        and _same_tensor_storage(first.normals, sample.normals)
        for sample in samples[1:]
    )


def _reshape_pair_targets(value: torch.Tensor, num_pairs: int, n_targets: int) -> torch.Tensor:
    return value.reshape(num_pairs, n_targets).contiguous()


class _NativeChunkedSpectrum(torch.autograd.Function):
    """Autograd bridge for chunked native Dirichlet spectra."""

    @staticmethod
    def forward(
        ctx,
        solver,
        distances: torch.Tensor,
        amplitudes: torch.Tensor,
        targets_per_spectrum: int,
        num_bins: int,
        n_fft: int,
        k0_per_meter: float,
        shared_gradient: bool,
    ) -> torch.Tensor:
        d = _to_f32(solver, distances.reshape(-1))
        a = _to_f32(solver, amplitudes.reshape(-1))
        if d.shape != a.shape:
            raise ValueError("distances and amplitudes must have the same shape.")
        if targets_per_spectrum <= 0:
            raise ValueError("targets_per_spectrum must be positive.")

        num_targets = d.numel()
        num_spectra = (num_targets + targets_per_spectrum - 1) // targets_per_spectrum
        output_re = torch.empty((num_spectra, num_bins), dtype=torch.float32, device=solver.device)
        output_im = torch.empty_like(output_re)
        if num_targets:
            cfg = solver.radar.config
            solver._module.forward_chunked(
                d,
                a,
                output_re,
                output_im,
                solver.n,
                k0_per_meter,
                num_bins,
                n_fft,
                num_targets,
                targets_per_spectrum,
                cfg.fc,
                cfg.slope * 1e12,
                cfg.adc_start_time * 1e-6,
            )

        ctx.solver = solver
        ctx.targets_per_spectrum = targets_per_spectrum
        ctx.num_bins = num_bins
        ctx.n_fft = n_fft
        ctx.k0_per_meter = k0_per_meter
        ctx.shared_gradient = shared_gradient
        ctx.distance_dtype = distances.dtype
        ctx.amplitude_dtype = amplitudes.dtype
        ctx.distance_shape = distances.shape
        ctx.amplitude_shape = amplitudes.shape
        ctx.save_for_backward(d, a)
        return torch.complex(output_re, output_im)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor):
        d, a = ctx.saved_tensors
        needs_d = ctx.needs_input_grad[1]
        needs_a = ctx.needs_input_grad[2]
        grad_d = grad_a = None
        if (needs_d or needs_a) and d.numel():
            solver = ctx.solver
            cfg = solver.radar.config
            gout_re = grad_output.real.to(dtype=torch.float32, device=solver.device).contiguous()
            gout_im = grad_output.imag.to(dtype=torch.float32, device=solver.device).contiguous()
            if ctx.shared_gradient:
                native_grad_d = torch.empty_like(d)
                native_grad_a = torch.empty_like(a)
                solver._module.backward_parallel_bins(
                    d,
                    a,
                    gout_re[0],
                    gout_im[0],
                    native_grad_d,
                    native_grad_a,
                    solver.n,
                    ctx.k0_per_meter,
                    ctx.num_bins,
                    ctx.n_fft,
                    d.numel(),
                    cfg.fc,
                    cfg.slope * 1e12,
                    cfg.adc_start_time * 1e-6,
                )
            else:
                native_grad_d = torch.empty_like(d)
                native_grad_a = torch.empty_like(a)
                solver._module.backward_batched(
                    d,
                    a,
                    gout_re,
                    gout_im,
                    native_grad_d,
                    native_grad_a,
                    solver.n,
                    ctx.k0_per_meter,
                    ctx.num_bins,
                    ctx.n_fft,
                    d.numel(),
                    ctx.targets_per_spectrum,
                    cfg.fc,
                    cfg.slope * 1e12,
                    cfg.adc_start_time * 1e-6,
                )
            if needs_d:
                grad_d = native_grad_d.to(ctx.distance_dtype).reshape(ctx.distance_shape)
            if needs_a:
                grad_a = native_grad_a.to(ctx.amplitude_dtype).reshape(ctx.amplitude_shape)
        else:
            if needs_d:
                grad_d = torch.empty(ctx.distance_shape, dtype=ctx.distance_dtype, device=d.device)
            if needs_a:
                grad_a = torch.empty(ctx.amplitude_shape, dtype=ctx.amplitude_dtype, device=a.device)
        return None, grad_d, grad_a, None, None, None, None, None


def _native_chunked_spectra(
    solver: "DirichletSolver",
    distances: torch.Tensor,
    amplitudes: torch.Tensor,
    *,
    targets_per_spectrum: int,
    num_bins: int,
    n_fft: int,
    k0_per_meter: float,
    shared_gradient: bool = False,
) -> torch.Tensor:
    if not (torch.is_grad_enabled() and (distances.requires_grad or amplitudes.requires_grad)):
        d = _to_f32(solver, distances.reshape(-1))
        a = _to_f32(solver, amplitudes.reshape(-1))
        if d.shape != a.shape:
            raise ValueError("distances and amplitudes must have the same shape.")
        num_targets = d.numel()
        num_spectra = (num_targets + targets_per_spectrum - 1) // targets_per_spectrum
        output_re = torch.empty((num_spectra, num_bins), dtype=torch.float32, device=solver.device)
        output_im = torch.empty_like(output_re)
        if num_targets:
            cfg = solver.radar.config
            solver._module.forward_chunked(
                d,
                a,
                output_re,
                output_im,
                solver.n,
                k0_per_meter,
                num_bins,
                n_fft,
                num_targets,
                targets_per_spectrum,
                cfg.fc,
                cfg.slope * 1e12,
                cfg.adc_start_time * 1e-6,
            )
        return torch.complex(output_re, output_im)
    return _NativeChunkedSpectrum.apply(
        solver,
        distances,
        amplitudes,
        targets_per_spectrum,
        num_bins,
        n_fft,
        k0_per_meter,
        shared_gradient,
    )


def spectrum(
    solver: "DirichletSolver",
    distances,
    amplitudes,
    *,
    targets_per_chunk: int = 256,
    num_bins: int | None = None,
    n_fft: int | None = None,
    k0_per_meter: float | None = None,
):
    """Dirichlet spectrum with optional overrides for MIMO-mode calls."""
    num_bins = solver.num_bins if num_bins is None else num_bins
    n_fft = solver.N_fft if n_fft is None else n_fft
    k0_per_meter = solver.k0_per_meter if k0_per_meter is None else k0_per_meter

    num_targets = distances.shape[0]
    d = _to_f32(solver, distances)
    a = _to_f32(solver, amplitudes)

    if num_targets == 0:
        return torch.zeros(num_bins, dtype=torch.complex64, device=solver.device)
    chunks = _native_chunked_spectra(
        solver,
        d,
        a,
        targets_per_spectrum=targets_per_chunk,
        num_bins=num_bins,
        n_fft=n_fft,
        k0_per_meter=k0_per_meter,
        shared_gradient=True,
    )
    return chunks.sum(dim=0)


def backward(solver: "DirichletSolver", distances, amplitudes, grad_output_re, grad_output_im):
    """Backward pass with one block per target and parallel bin reduction."""
    cfg = solver.radar.config
    num_targets = distances.shape[0]
    d = _to_f32(solver, distances)
    a = _to_f32(solver, amplitudes)
    g_re = grad_output_re.to(dtype=torch.float32, device=solver.device).contiguous()
    g_im = grad_output_im.to(dtype=torch.float32, device=solver.device).contiguous()

    grad_d = torch.empty(num_targets, dtype=torch.float32, device=solver.device)
    grad_a = torch.empty(num_targets, dtype=torch.float32, device=solver.device)

    solver._module.backward_parallel_bins(
        d,
        a,
        g_re,
        g_im,
        grad_d,
        grad_a,
        solver.n,
        solver.k0_per_meter,
        solver.num_bins,
        solver.N_fft,
        num_targets,
        cfg.fc,
        cfg.slope * 1e12,
        cfg.adc_start_time * 1e-6,
    )

    return grad_d, grad_a


def backward_per_bin(
    solver: "DirichletSolver",
    distances,
    amplitudes,
    grad_output_re,
    grad_output_im,
    *,
    bins_per_chunk: int = 256,
):
    """Backward pass with one thread per spectrum bin."""
    cfg = solver.radar.config
    num_targets = distances.shape[0]
    d = _to_f32(solver, distances)
    a = _to_f32(solver, amplitudes)
    g_re = grad_output_re.to(dtype=torch.float32, device=solver.device).contiguous()
    g_im = grad_output_im.to(dtype=torch.float32, device=solver.device).contiguous()

    num_chunks = (solver.num_bins + bins_per_chunk - 1) // bins_per_chunk
    grad_d = torch.zeros((num_chunks, num_targets), dtype=torch.float32, device=solver.device)
    grad_a = torch.zeros((num_chunks, num_targets), dtype=torch.float32, device=solver.device)

    solver._module.backward_per_bin(
        d,
        a,
        g_re,
        g_im,
        grad_d,
        grad_a,
        solver.n,
        solver.k0_per_meter,
        solver.num_bins,
        solver.N_fft,
        num_targets,
        bins_per_chunk,
        cfg.fc,
        cfg.slope * 1e12,
        cfg.adc_start_time * 1e-6,
    )

    return grad_d.sum(dim=0), grad_a.sum(dim=0)


class DirichletSolver(Solver):
    """Direct frequency-domain spectrum solver."""

    def __init__(self, radar, pad_factor: int = 16):
        super().__init__(radar)
        self._cuda_module = None

        cfg = radar.config
        fs = cfg.sample_rate * 1e3
        slope_hz = cfg.slope * 1e12

        self.pad_factor = int(pad_factor)
        self.N_fft = cfg.adc_samples * self.pad_factor
        self.num_bins = self.N_fft // 2
        self.n = (cfg.adc_samples - 1) / 2
        self.k0_per_meter = (slope_hz * 2 / radar.c0) * self.N_fft / fs

        self.mimo_N_fft = cfg.adc_samples
        self.mimo_num_bins = cfg.adc_samples
        self.mimo_k0_per_meter = (slope_hz * 2 / radar.c0) * self.mimo_N_fft / fs

    @property
    def _module(self):
        if self.device.type != "cuda":
            raise RuntimeError("The Dirichlet solver requires CUDA tensors; construct Radar with device='cuda'.")
        if self._cuda_module is None:
            self._cuda_module = _load_module()
        return self._cuda_module

    def chirp(self, distances, amplitudes):
        """High-resolution Dirichlet spectrum (pad_factor bins)."""
        return spectrum(self, distances, amplitudes)

    def chirp_mimo(self, distances, amplitudes):
        """Dirichlet spectrum at adc_samples resolution for MIMO output."""
        return spectrum(
            self,
            distances,
            amplitudes,
            num_bins=self.mimo_num_bins,
            n_fft=self.mimo_N_fft,
            k0_per_meter=self.mimo_k0_per_meter,
        )

    def frame(self, interpolator, t0=0):
        r = self.radar
        cfg = r.config
        chirp_period = (cfg.idle_time + cfg.ramp_end_time) * 1e-6
        tx0 = r.tx_pos[0:1].contiguous()
        rx0 = r.rx_pos[0:1].contiguous()

        result = []
        for chirp_id in range(cfg.chirp_per_frame):
            time_in_frame = chirp_id * chirp_period * cfg.num_tx
            sample = normalize_interpolated_sample(interpolator(t0 + time_in_frame), device=self.device)
            total_lengths = compute_total_path_lengths(sample, tx0, rx0)
            one_way = total_lengths.squeeze(0).squeeze(0) * 0.5
            amp = compute_path_amplitudes(r, sample, total_lengths, tx_pos=tx0, rx_pos=rx0).squeeze(0).squeeze(0)
            result.append(self.chirp(one_way, amp))

        return torch.stack(result)

    def mimo(self, interpolator, t0=0, **options):
        """Generate a MIMO frame; each TX samples the scene at its TDM slot time."""
        freq_domain = self._pop_bool_option(options, "freq_domain", False)
        fast = self._pop_bool_option(options, "fast", False)
        self._ensure_no_options(options)

        r = self.radar
        samples = collect_interpolated_samples(r, interpolator, t0)
        if fast and not samples_require_grad(samples) and _samples_share_storage(samples):
            return self._mimo_from_sample_static(samples[0], freq_domain=freq_domain)

        frame = self._mimo_from_slot_samples(samples, freq_domain=freq_domain)
        return frame

    # Slot-group size cap: bounds the (slots, N, RX, 3) geometry transients in
    # compute_slot_path_tensors to roughly 256 MB of float32.
    _SLOT_GROUP_ELEMENT_BUDGET = 1 << 26

    def _mimo_from_slot_samples(self, samples, *, freq_domain: bool):
        """One forward_chunked launch per slot group over padded per-slot paths."""
        r = self.radar
        cfg = r.config
        num_rx = cfg.num_rx
        num_slots = len(samples)
        n_max = max(int(sample.points.shape[0]) for sample in samples)
        if n_max == 0:
            return torch.zeros(
                (cfg.num_tx, cfg.num_rx, cfg.chirp_per_frame, cfg.adc_samples),
                dtype=torch.complex64,
                device=self.device,
            )

        spectrum_groups = []
        group = max(1, self._SLOT_GROUP_ELEMENT_BUDGET // max(1, n_max * num_rx * 3))
        for start in range(0, num_slots, group):
            stop = min(start + group, num_slots)
            one_way, amplitudes = compute_slot_path_tensors(r, samples[start:stop], first_slot=start)
            if one_way is None:
                spectrum_groups.append(
                    torch.zeros(
                        ((stop - start) * num_rx, self.mimo_num_bins),
                        dtype=torch.complex64,
                        device=self.device,
                    )
                )
                continue
            _, _, group_n = one_way.shape
            spectrum_groups.append(
                _native_chunked_spectra(
                    self,
                    one_way.reshape(-1),
                    amplitudes.reshape(-1),
                    targets_per_spectrum=group_n,
                    num_bins=self.mimo_num_bins,
                    n_fft=self.mimo_N_fft,
                    k0_per_meter=self.mimo_k0_per_meter,
                )
            )

        spectra = torch.cat(spectrum_groups, dim=0)
        if not freq_domain:
            spectra = torch.fft.ifft(spectra, dim=-1)
        return (
            spectra.view(cfg.chirp_per_frame, cfg.num_tx, num_rx, cfg.adc_samples)
            .permute(1, 2, 0, 3)
            .contiguous()
        )

    def mimo_from_trace(self, trace, *, velocities=None, t0=0.0, **options):
        """Generate a MIMO frame from one pre-traced frame.

        ``velocities`` is optional per-path world velocity with shape ``(N, 3)``.
        When omitted, the trace is treated as static across all chirps.
        """
        freq_domain = self._pop_bool_option(options, "freq_domain", False)
        amplitude_update = options.pop("amplitude_update", "range_loss")
        self._ensure_no_options(options)
        sample = normalize_interpolated_sample(trace, device=self.device)
        if velocities is None:
            cache = self._path_cache_from_sample(sample)
            return self.mimo_from_paths(cache, freq_domain=freq_domain, amplitude_update=amplitude_update)

        velocity_t = torch.as_tensor(velocities, dtype=torch.float32, device=self.device)
        if velocity_t.shape != sample.points.shape:
            raise ValueError(
                "velocities must have shape (N, 3), matching trace.points; "
                f"got {tuple(velocity_t.shape)} for {tuple(sample.points.shape)}."
            )
        cache = self._path_cache_from_sample(sample, velocity_t.contiguous())
        return self.mimo_from_paths(cache, freq_domain=freq_domain, amplitude_update=amplitude_update)

    def path_cache_from_trace(self, trace, *, velocities=None) -> MimoPathCache:
        """Precompute one frame's per-pair distances, amplitudes, and optional rates."""
        sample = normalize_interpolated_sample(trace, device=self.device)
        velocity_t = None
        if velocities is not None:
            velocity_t = torch.as_tensor(velocities, dtype=torch.float32, device=self.device).contiguous()
            if velocity_t.shape != sample.points.shape:
                raise ValueError(
                    "velocities must have shape (N, 3), matching trace.points; "
                    f"got {tuple(velocity_t.shape)} for {tuple(sample.points.shape)}."
                )
        return self._path_cache_from_sample(sample, velocity_t)

    def _path_cache_from_sample(self, sample, velocities: torch.Tensor | None = None) -> MimoPathCache:
        r = self.radar
        total_lengths = compute_total_path_lengths(sample, r.tx_pos, r.rx_pos)
        one_way = (total_lengths * 0.5).contiguous()
        amplitudes = compute_path_amplitudes(
            r,
            sample,
            total_lengths,
            tx_pos=r.tx_pos,
            rx_pos=r.rx_pos,
        ).contiguous()
        one_way_rates = None
        if velocities is not None:
            one_way_rates = (
                self._total_path_length_rates(sample, velocities, tx_pos=r.tx_pos, rx_pos=r.rx_pos) * 0.5
            ).contiguous()
        return MimoPathCache(
            one_way_distances=one_way,
            amplitudes=amplitudes,
            one_way_distance_rates=one_way_rates,
        )

    def mimo_from_paths(
        self,
        cache: MimoPathCache,
        *,
        freq_domain: bool = False,
        amplitude_update: str = "range_loss",
    ):
        if amplitude_update not in {"constant", "range_loss"}:
            raise ValueError("amplitude_update must be 'constant' or 'range_loss'.")
        distances = cache.one_way_distances.to(dtype=torch.float32, device=self.device).contiguous()
        amplitudes = cache.amplitudes.to(dtype=torch.float32, device=self.device).contiguous()
        if distances.shape != amplitudes.shape or distances.ndim != 3:
            raise ValueError(
                "MimoPathCache one_way_distances and amplitudes must both have shape "
                "(num_tx, num_rx, num_paths)."
            )
        cfg = self.radar.config
        expected_shape = (cfg.num_tx, cfg.num_rx)
        if distances.shape[:2] != expected_shape:
            raise ValueError(
                "MimoPathCache antenna dimensions must match radar config; "
                f"got {tuple(distances.shape[:2])}, expected {expected_shape}."
            )
        n_targets = distances.shape[2]
        if n_targets == 0:
            return torch.zeros(
                (cfg.num_tx, cfg.num_rx, cfg.chirp_per_frame, cfg.adc_samples),
                dtype=torch.complex64,
                device=self.device,
            )
        num_pairs = cfg.num_tx * cfg.num_rx
        if cache.one_way_distance_rates is None:
            spectra = self._mimo_spectra_static(
                distances.reshape(-1).contiguous(),
                amplitudes.reshape(-1).contiguous(),
                num_pairs,
                n_targets,
            )
            if not freq_domain:
                spectra = torch.fft.ifft(spectra, dim=-1)
            chirp = spectra.view(cfg.num_tx, cfg.num_rx, cfg.adc_samples)
            return chirp.unsqueeze(2).expand(-1, -1, cfg.chirp_per_frame, -1).clone()

        rates = cache.one_way_distance_rates.to(dtype=torch.float32, device=self.device).contiguous()
        if rates.shape != distances.shape:
            raise ValueError("MimoPathCache one_way_distance_rates must match one_way_distances shape.")
        return self._mimo_from_path_tensors_linear(
            distances,
            amplitudes,
            rates,
            freq_domain=freq_domain,
            amplitude_update=amplitude_update,
        )

    def _mimo_from_sample_static(self, sample, *, freq_domain: bool):
        r = self.radar
        cfg = r.config
        tx_pos = r.tx_pos
        rx_pos = r.rx_pos
        num_pairs = cfg.num_tx * cfg.num_rx
        n_targets = sample.points.shape[0]
        if n_targets == 0:
            return torch.zeros(
                (cfg.num_tx, cfg.num_rx, cfg.chirp_per_frame, cfg.adc_samples),
                dtype=torch.complex64,
                device=self.device,
            )

        total_lengths = compute_total_path_lengths(sample, tx_pos, rx_pos)
        one_way = _reshape_pair_targets(total_lengths, num_pairs, n_targets) * 0.5
        all_d = one_way.reshape(-1).contiguous()
        amp = compute_path_amplitudes(r, sample, total_lengths, tx_pos=tx_pos, rx_pos=rx_pos)
        all_a = amp.reshape(-1).contiguous()
        spectra = self._mimo_spectra_static(all_d, all_a, num_pairs, n_targets)
        if not freq_domain:
            spectra = torch.fft.ifft(spectra, dim=-1)
        chirp = spectra.view(cfg.num_tx, cfg.num_rx, cfg.adc_samples)
        return chirp.unsqueeze(2).expand(-1, -1, cfg.chirp_per_frame, -1).clone()

    def _mimo_spectra_static(
        self,
        all_d: torch.Tensor,
        all_a: torch.Tensor,
        num_pairs: int,
        n_targets: int,
        *,
        targets_per_chunk: int = 256,
    ) -> torch.Tensor:
        chunks_per_pair = (n_targets + targets_per_chunk - 1) // targets_per_chunk
        n_padded = chunks_per_pair * targets_per_chunk
        if n_padded != n_targets:
            d = torch.zeros((num_pairs, n_padded), dtype=torch.float32, device=self.device)
            a = torch.zeros_like(d)
            d[:, :n_targets] = all_d.view(num_pairs, n_targets)
            a[:, :n_targets] = all_a.view(num_pairs, n_targets)
            all_d = d.reshape(-1)
            all_a = a.reshape(-1)

        spectra = _native_chunked_spectra(
            self,
            all_d,
            all_a,
            targets_per_spectrum=targets_per_chunk,
            num_bins=self.mimo_num_bins,
            n_fft=self.mimo_N_fft,
            k0_per_meter=self.mimo_k0_per_meter,
        )
        return spectra.view(num_pairs, chunks_per_pair, self.mimo_num_bins).sum(dim=1)

    def _mimo_from_path_tensors_linear(
        self,
        distances: torch.Tensor,
        amplitudes: torch.Tensor,
        rates: torch.Tensor,
        *,
        freq_domain: bool,
        amplitude_update: str,
    ):
        if distances.requires_grad or amplitudes.requires_grad or rates.requires_grad:
            return self._mimo_from_path_tensors_linear_autograd(
                distances,
                amplitudes,
                rates,
                freq_domain=freq_domain,
                amplitude_update=amplitude_update,
            )
        cfg = self.radar.config
        num_pairs = cfg.num_tx * cfg.num_rx
        n_targets = distances.shape[2]
        output_re = torch.zeros(
            (cfg.chirp_per_frame, num_pairs, self.mimo_num_bins),
            dtype=torch.float32,
            device=self.device,
        )
        output_im = torch.zeros_like(output_re)
        chirp_period = (cfg.idle_time + cfg.ramp_end_time) * 1e-6
        self._module.forward_mimo_linear_chunked(
            distances.reshape(-1).contiguous(),
            rates.reshape(-1).contiguous(),
            amplitudes.reshape(-1).contiguous(),
            output_re,
            output_im,
            self.n,
            self.mimo_k0_per_meter,
            self.mimo_num_bins,
            self.mimo_N_fft,
            n_targets,
            cfg.chirp_per_frame,
            chirp_period,
            cfg.num_tx,
            1 if amplitude_update == "range_loss" else 0,
            cfg.fc,
            cfg.slope * 1e12,
            cfg.adc_start_time * 1e-6,
        )

        spectra = torch.complex(output_re, output_im)
        if not freq_domain:
            spectra = torch.fft.ifft(spectra, dim=-1)
        return spectra.view(cfg.chirp_per_frame, cfg.num_tx, cfg.num_rx, cfg.adc_samples).permute(1, 2, 0, 3).contiguous()

    def _mimo_from_path_tensors_linear_autograd(
        self,
        distances: torch.Tensor,
        amplitudes: torch.Tensor,
        rates: torch.Tensor,
        *,
        freq_domain: bool,
        amplitude_update: str,
    ):
        cfg = self.radar.config
        num_pairs = cfg.num_tx * cfg.num_rx
        n_targets = distances.shape[2]
        chirp_period = (cfg.idle_time + cfg.ramp_end_time) * 1e-6
        chirp_ids = torch.arange(
            cfg.chirp_per_frame,
            dtype=distances.dtype,
            device=self.device,
        ).view(-1, 1)
        tx_ids = torch.arange(cfg.num_tx, dtype=distances.dtype, device=self.device).repeat_interleave(
            cfg.num_rx
        ).view(1, -1)
        times = (chirp_ids * cfg.num_tx + tx_ids) * chirp_period
        base_distances = distances.reshape(1, num_pairs, n_targets)
        frame_distances = base_distances + rates.reshape(1, num_pairs, n_targets) * times.unsqueeze(-1)
        frame_amplitudes = amplitudes.reshape(1, num_pairs, n_targets).expand_as(frame_distances)
        if amplitude_update == "range_loss":
            frame_amplitudes = frame_amplitudes * base_distances / torch.clamp(frame_distances, min=1e-6)
        spectra = _native_chunked_spectra(
            self,
            frame_distances.reshape(-1),
            frame_amplitudes.reshape(-1),
            targets_per_spectrum=n_targets,
            num_bins=self.mimo_num_bins,
            n_fft=self.mimo_N_fft,
            k0_per_meter=self.mimo_k0_per_meter,
        )
        if not freq_domain:
            spectra = torch.fft.ifft(spectra, dim=-1)
        return (
            spectra.view(cfg.chirp_per_frame, cfg.num_tx, cfg.num_rx, cfg.adc_samples)
            .permute(1, 2, 0, 3)
            .contiguous()
        )

    def _total_path_length_rates(self, sample, velocities: torch.Tensor, *, tx_pos, rx_pos):
        entry_vectors = sample.entry_points.unsqueeze(0) - tx_pos.unsqueeze(1)
        point_vectors = sample.points.unsqueeze(0) - rx_pos.unsqueeze(1)
        entry_dist = torch.clamp(torch.linalg.norm(entry_vectors, dim=-1), min=1e-6)
        point_dist = torch.clamp(torch.linalg.norm(point_vectors, dim=-1), min=1e-6)
        entry_rates = (entry_vectors * velocities.unsqueeze(0)).sum(dim=-1) / entry_dist
        point_rates = (point_vectors * velocities.unsqueeze(0)).sum(dim=-1) / point_dist
        return entry_rates.unsqueeze(1) + point_rates.unsqueeze(0)

    def backward(self, distances, amplitudes, grad_output_re, grad_output_im):
        return backward(self, distances, amplitudes, grad_output_re, grad_output_im)

    def backward_per_bin(self, distances, amplitudes, grad_output_re, grad_output_im, bins_per_chunk: int = 256):
        return backward_per_bin(
            self,
            distances,
            amplitudes,
            grad_output_re,
            grad_output_im,
            bins_per_chunk=bins_per_chunk,
        )
