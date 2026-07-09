"""
Dirichlet kernel solver backend.

Computes the range spectrum directly in the frequency domain and keeps all
backend-specific state on the solver instance.
"""

from __future__ import annotations

import os

import slangtorch
import torch

from ..path_cache import MimoPathCache
from . import Solver
from .common import (
    collect_interpolated_samples,
    compute_path_amplitudes,
    compute_total_path_lengths,
    ensure_cuda_build_env,
    ensure_current_env_on_path,
    normalize_interpolated_sample,
    pytorch_chirp_reference,
    pytorch_mimo_from_samples,
    samples_require_grad,
)

_SOLVERS_DIR = os.path.dirname(__file__)


def _load_module():
    ensure_current_env_on_path()
    slang_path = os.path.join(_SOLVERS_DIR, "dirichlet.slang")
    return slangtorch.loadModule(slang_path, includePaths=ensure_cuda_build_env())


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
    cfg = solver.radar.config
    num_bins = solver.num_bins if num_bins is None else num_bins
    n_fft = solver.N_fft if n_fft is None else n_fft
    k0_per_meter = solver.k0_per_meter if k0_per_meter is None else k0_per_meter

    num_targets = distances.shape[0]
    d = _to_f32(solver, distances)
    a = _to_f32(solver, amplitudes)

    num_chunks = (num_targets + targets_per_chunk - 1) // targets_per_chunk
    output_re = torch.zeros((num_chunks, num_bins), dtype=torch.float32, device=solver.device)
    output_im = torch.zeros((num_chunks, num_bins), dtype=torch.float32, device=solver.device)

    solver._module.forward_chunked(
        d=d,
        a=a,
        output_re=output_re,
        output_im=output_im,
        n=solver.n,
        k0_per_meter=k0_per_meter,
        num_bins=num_bins,
        N_fft=n_fft,
        num_targets=num_targets,
        targets_per_chunk=targets_per_chunk,
        fc=cfg.fc,
        slope=cfg.slope * 1e12,
        t_start=cfg.adc_start_time * 1e-6,
    ).launchRaw(
        blockSize=(256, 1, 1),
        gridSize=((num_bins + 255) // 256, num_chunks, 1),
    )

    return torch.complex(output_re.sum(dim=0), output_im.sum(dim=0))


def backward(solver: "DirichletSolver", distances, amplitudes, grad_output_re, grad_output_im):
    """Backward pass with one thread per target."""
    cfg = solver.radar.config
    num_targets = distances.shape[0]
    d = _to_f32(solver, distances)
    a = _to_f32(solver, amplitudes)
    g_re = grad_output_re.to(dtype=torch.float32, device=solver.device).contiguous()
    g_im = grad_output_im.to(dtype=torch.float32, device=solver.device).contiguous()

    grad_d = torch.zeros(num_targets, dtype=torch.float32, device=solver.device)
    grad_a = torch.zeros(num_targets, dtype=torch.float32, device=solver.device)

    solver._module.backward(
        d=d,
        a=a,
        grad_output_re=g_re,
        grad_output_im=g_im,
        grad_d=grad_d,
        grad_a=grad_a,
        n=solver.n,
        k0_per_meter=solver.k0_per_meter,
        num_bins=solver.num_bins,
        N_fft=solver.N_fft,
        num_targets=num_targets,
        fc=cfg.fc,
        slope=cfg.slope * 1e12,
        t_start=cfg.adc_start_time * 1e-6,
    ).launchRaw(
        blockSize=(256, 1, 1),
        gridSize=((num_targets + 255) // 256, 1, 1),
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
        d=d,
        a=a,
        grad_output_re=g_re,
        grad_output_im=g_im,
        grad_d=grad_d,
        grad_a=grad_a,
        n=solver.n,
        k0_per_meter=solver.k0_per_meter,
        num_bins=solver.num_bins,
        N_fft=solver.N_fft,
        num_targets=num_targets,
        bins_per_chunk=bins_per_chunk,
        fc=cfg.fc,
        slope=cfg.slope * 1e12,
        t_start=cfg.adc_start_time * 1e-6,
    ).launchRaw(
        blockSize=(bins_per_chunk, 1, 1),
        gridSize=(num_chunks, 1, 1),
    )

    return grad_d.sum(dim=0), grad_a.sum(dim=0)


class DirichletSolver(Solver):
    """Direct frequency-domain spectrum solver."""

    def __init__(self, radar, pad_factor: int = 16):
        super().__init__(radar)
        self._module = _load_module()

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

    def chirp(self, distances, amplitudes):
        """High-resolution Dirichlet spectrum (pad_factor bins)."""
        signal = spectrum(self, distances, amplitudes)
        if distances.requires_grad or amplitudes.requires_grad:
            reference = pytorch_chirp_reference(self.radar, distances, amplitudes)
            reference = torch.fft.fft(reference, n=self.N_fft)[: self.N_fft // 2]
            signal = signal.to(reference.dtype)
            return signal.detach() + (reference - reference.detach())
        return signal

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
        """Generate MIMO frame with batched TX/RX kernel launches."""
        freq_domain = self._pop_bool_option(options, "freq_domain", False)
        fast = self._pop_bool_option(options, "fast", False)
        self._ensure_no_options(options)

        r = self.radar
        cfg = r.config
        samples = collect_interpolated_samples(r, interpolator, t0)
        if fast and not samples_require_grad(samples) and _samples_share_storage(samples):
            return self._mimo_from_sample_static(samples[0], freq_domain=freq_domain)

        tx_pos = r.tx_pos
        rx_pos = r.rx_pos
        num_pairs = cfg.num_tx * cfg.num_rx

        frame = torch.zeros(
            (cfg.num_tx, cfg.num_rx, cfg.chirp_per_frame, cfg.adc_samples),
            dtype=torch.complex64,
            device=self.device,
        )

        for chirp_id, sample in enumerate(samples):
            n_targets = sample.points.shape[0]
            if n_targets == 0:
                continue

            total_lengths = compute_total_path_lengths(sample, tx_pos, rx_pos)
            one_way = total_lengths.reshape(num_pairs, n_targets) * 0.5
            all_d = one_way.reshape(-1).contiguous()
            amp = compute_path_amplitudes(r, sample, total_lengths, tx_pos=tx_pos, rx_pos=rx_pos)
            all_a = amp.reshape(-1).contiguous()

            output_re = torch.zeros((num_pairs, self.mimo_num_bins), dtype=torch.float32, device=self.device)
            output_im = torch.zeros((num_pairs, self.mimo_num_bins), dtype=torch.float32, device=self.device)

            self._module.forward_chunked(
                d=all_d,
                a=all_a,
                output_re=output_re,
                output_im=output_im,
                n=self.n,
                k0_per_meter=self.mimo_k0_per_meter,
                num_bins=self.mimo_num_bins,
                N_fft=self.mimo_N_fft,
                num_targets=num_pairs * n_targets,
                targets_per_chunk=n_targets,
                fc=cfg.fc,
                slope=cfg.slope * 1e12,
                t_start=cfg.adc_start_time * 1e-6,
            ).launchRaw(
                blockSize=(256, 1, 1),
                gridSize=((self.mimo_num_bins + 255) // 256, num_pairs, 1),
            )

            spectra = torch.complex(output_re, output_im)
            if not freq_domain:
                spectra = torch.fft.ifft(spectra, dim=-1)
            frame[:, :, chirp_id, :] = spectra.view(cfg.num_tx, cfg.num_rx, cfg.adc_samples)

        if samples_require_grad(samples):
            reference = pytorch_mimo_from_samples(r, samples)
            if freq_domain:
                reference = torch.fft.fft(reference, dim=-1)
            frame = frame.to(reference.dtype)
            return frame.detach() + (reference - reference.detach())
        return frame

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
            if samples_require_grad([sample]):
                return self.mimo(lambda _t: trace, t0=t0, fast=False, freq_domain=freq_domain)
            cache = self._path_cache_from_sample(sample)
            return self.mimo_from_paths(cache, freq_domain=freq_domain, amplitude_update=amplitude_update)

        velocity_t = torch.as_tensor(velocities, dtype=torch.float32, device=self.device)
        if velocity_t.shape != sample.points.shape:
            raise ValueError(
                "velocities must have shape (N, 3), matching trace.points; "
                f"got {tuple(velocity_t.shape)} for {tuple(sample.points.shape)}."
            )
        if samples_require_grad([sample]) or velocity_t.requires_grad:
            return self._mimo_from_trace_linear_reference(
                trace,
                velocity_t,
                t0=t0,
                freq_domain=freq_domain,
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

    def _mimo_from_trace_linear_reference(self, trace, velocities: torch.Tensor, *, t0: float, freq_domain: bool):
        from ..trace_result import TraceResult

        base_points = trace.points.to(dtype=torch.float32, device=self.device)
        base_entry_points = trace.entry_points.to(dtype=torch.float32, device=self.device)

        def interpolator(t):
            dt = float(t) - float(t0)
            return TraceResult(
                base_points + velocities * dt,
                trace.intensities,
                entry_points=base_entry_points + velocities * dt,
                fixed_path_lengths=trace.fixed_path_lengths,
                depths=trace.depths,
                normals=trace.normals,
            )

        return self.mimo(interpolator, t0=t0, fast=False, freq_domain=freq_domain)

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
    ) -> torch.Tensor:
        cfg = self.radar.config
        output_re = torch.zeros((num_pairs, self.mimo_num_bins), dtype=torch.float32, device=self.device)
        output_im = torch.zeros((num_pairs, self.mimo_num_bins), dtype=torch.float32, device=self.device)
        self._module.forward_chunked(
            d=all_d,
            a=all_a,
            output_re=output_re,
            output_im=output_im,
            n=self.n,
            k0_per_meter=self.mimo_k0_per_meter,
            num_bins=self.mimo_num_bins,
            N_fft=self.mimo_N_fft,
            num_targets=num_pairs * n_targets,
            targets_per_chunk=n_targets,
            fc=cfg.fc,
            slope=cfg.slope * 1e12,
            t_start=cfg.adc_start_time * 1e-6,
        ).launchRaw(
            blockSize=(256, 1, 1),
            gridSize=((self.mimo_num_bins + 255) // 256, num_pairs, 1),
        )
        return torch.complex(output_re, output_im)

    def _mimo_from_path_tensors_linear(
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
        output_re = torch.zeros(
            (cfg.chirp_per_frame, num_pairs, self.mimo_num_bins),
            dtype=torch.float32,
            device=self.device,
        )
        output_im = torch.zeros_like(output_re)
        chirp_period = (cfg.idle_time + cfg.ramp_end_time) * 1e-6
        self._module.forward_mimo_linear_chunked(
            d0=distances.reshape(-1).contiguous(),
            d_rate=rates.reshape(-1).contiguous(),
            a0=amplitudes.reshape(-1).contiguous(),
            output_re=output_re,
            output_im=output_im,
            n=self.n,
            k0_per_meter=self.mimo_k0_per_meter,
            num_bins=self.mimo_num_bins,
            N_fft=self.mimo_N_fft,
            targets_per_pair=n_targets,
            chirp_per_frame=cfg.chirp_per_frame,
            chirp_period=chirp_period,
            num_tx=cfg.num_tx,
            range_loss_update=1 if amplitude_update == "range_loss" else 0,
            fc=cfg.fc,
            slope=cfg.slope * 1e12,
            t_start=cfg.adc_start_time * 1e-6,
        ).launchRaw(
            blockSize=(256, 1, 1),
            gridSize=(
                (self.mimo_num_bins + 255) // 256,
                num_pairs,
                cfg.chirp_per_frame,
            ),
        )

        spectra = torch.complex(output_re, output_im)
        if not freq_domain:
            spectra = torch.fft.ifft(spectra, dim=-1)
        return spectra.view(cfg.chirp_per_frame, cfg.num_tx, cfg.num_rx, cfg.adc_samples).permute(1, 2, 0, 3).contiguous()

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
