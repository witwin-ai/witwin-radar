"""
Slang CUDA solver backend — high-throughput time-domain beat signal computation.

Two parallelization strategies for chirp:
- chirp_slang (default): parallel over time samples AND target chunks
- chirp_slang_per_target: each thread = one target, loops time samples (atomicAdd)

Plus frameCuda for full MIMO frames.
"""

import os
import torch
import slangtorch

from . import Solver
from ._runtime import (
    collect_interpolated_samples,
    compute_path_amplitudes,
    compute_total_path_lengths,
    ensure_current_env_on_path,
    normalize_interpolated_sample,
    pytorch_chirp_reference,
    pytorch_mimo_from_samples,
    samples_require_grad,
)

# .slang files live alongside solver modules
_SOLVERS_DIR = os.path.dirname(__file__)


def init():
    """Load the radar.slang module for the solver instance."""
    ensure_current_env_on_path()
    slang_path = os.path.join(_SOLVERS_DIR, 'radar.slang')
    return slangtorch.loadModule(slang_path)


# ------------------------------------------------------------------
# Chirp kernels
# ------------------------------------------------------------------

def chirp_slang(solver, distances, amplitudes, targets_per_chunk=256):
    """Chunked chirp: parallel over time samples AND target chunks (default)."""
    radar = solver.radar
    T = radar.adc_samples
    num_targets = distances.shape[0]

    distances = distances.to(dtype=torch.float32, device=solver.device).contiguous()
    amplitudes = amplitudes.to(dtype=torch.float32, device=solver.device).contiguous()

    num_chunks = (num_targets + targets_per_chunk - 1) // targets_per_chunk

    out_real = torch.zeros((num_chunks, T), dtype=torch.float64, device=solver.device)
    out_imag = torch.zeros((num_chunks, T), dtype=torch.float64, device=solver.device)

    block_size = 256
    grid_x = (T + block_size - 1) // block_size

    solver._module.chirp_kernel_chunked(
        t_sample=radar.t_sample,
        distances=distances, amplitudes=amplitudes,
        fc=radar.fc, slope=radar.slope * 1e12,
        num_targets=num_targets, targets_per_chunk=targets_per_chunk,
        out_real=out_real, out_imag=out_imag,
    ).launchRaw(blockSize=(block_size, 1, 1), gridSize=(grid_x, num_chunks, 1))

    return out_real.sum(dim=0) + 1j * out_imag.sum(dim=0)


def chirp_slang_per_target(solver, distances, amplitudes, targets_per_chunk=256):
    """Per-target chirp: each thread = one target, loops all time samples (atomicAdd)."""
    radar = solver.radar
    T = radar.adc_samples
    num_targets = distances.shape[0]

    distances = distances.to(dtype=torch.float32, device=solver.device).contiguous()
    amplitudes = amplitudes.to(dtype=torch.float32, device=solver.device).contiguous()

    num_chunks = (num_targets + targets_per_chunk - 1) // targets_per_chunk

    out_real = torch.zeros((num_chunks, T), dtype=torch.float32, device=solver.device)
    out_imag = torch.zeros((num_chunks, T), dtype=torch.float32, device=solver.device)

    solver._module.chirp_kernel_per_target(
        t_sample=radar.t_sample,
        distances=distances, amplitudes=amplitudes,
        fc=radar.fc, slope=radar.slope * 1e12,
        num_targets=num_targets, T=T, targets_per_chunk=targets_per_chunk,
        out_real=out_real, out_imag=out_imag,
    ).launchRaw(blockSize=(targets_per_chunk, 1, 1), gridSize=(num_chunks, 1, 1))

    return out_real.sum(dim=0).to(torch.float64) + 1j * out_imag.sum(dim=0).to(torch.float64)


# ------------------------------------------------------------------
# MIMO frame generation
# ------------------------------------------------------------------

def frameCuda(solver, samples):
    """Generate a full MIMO frame using Slang CUDA kernels from sampled chirps."""
    radar = solver.radar
    TX, RX, F, T = radar.num_tx, radar.num_rx, radar.chirp_per_frame, radar.adc_samples

    frame_real = torch.zeros((TX, RX, F, T), dtype=torch.float64, device=solver.device)
    frame_imag = torch.zeros_like(frame_real)

    t_sample = (
        torch.arange(0, T, dtype=torch.float64, device=solver.device) / (radar.sample_rate * 1e3)
        + radar.adc_start_time * 1e-6
    )

    tx_pos = torch.as_tensor(radar.tx_pos, dtype=torch.float32, device=solver.device)
    rx_pos = torch.as_tensor(radar.rx_pos, dtype=torch.float32, device=solver.device)

    max_points = 0
    all_total_lengths = []
    all_amplitudes = []
    for sample in samples:
        N = sample.points.shape[0]
        if N > max_points:
            max_points = N
        if N == 0:
            all_total_lengths.append(None)
            all_amplitudes.append(None)
            continue
        total_lengths = compute_total_path_lengths(sample, tx_pos, rx_pos)
        amplitudes = compute_path_amplitudes(radar, sample, total_lengths, tx_pos=tx_pos, rx_pos=rx_pos)
        all_total_lengths.append(total_lengths)
        all_amplitudes.append(amplitudes)

    if max_points == 0:
        return frame_real + 1j * frame_imag

    total_lengths_tensor = torch.zeros((F, TX, RX, max_points), dtype=torch.float32, device=solver.device)
    amplitudes_tensor = torch.zeros((F, TX, RX, max_points), dtype=torch.float32, device=solver.device)
    for i in range(F):
        if all_total_lengths[i] is None:
            continue
        N = all_total_lengths[i].shape[-1]
        total_lengths_tensor[i, :, :, :N] = all_total_lengths[i]
        amplitudes_tensor[i, :, :, :N] = all_amplitudes[i]

    block_size = 256
    grid_t = (T + block_size - 1) // block_size

    solver._module.frameMIMO_kernel(
        t_sample=t_sample.to(torch.float64),
        total_lengths=total_lengths_tensor,
        amplitudes=amplitudes_tensor,
        fc=radar.fc, slope=radar.slope * 1e6 * 1e6,
        phi=0.0, max_N=max_points,
        out_real=frame_real.to(torch.float64),
        out_imag=frame_imag.to(torch.float64),
    ).launchRaw(blockSize=(block_size, 1, 1), gridSize=(grid_t, TX * RX, F))

    return frame_real + 1j * frame_imag


# ------------------------------------------------------------------
# Solver class
# ------------------------------------------------------------------

class SlangSolver(Solver):
    """Slang CUDA kernel solver — high-throughput time-domain computation."""

    def __init__(self, radar):
        super().__init__(radar)
        self._module = init()

    def chirp(self, distances, amplitudes):
        d_rt = distances * 2  # one-way -> round-trip
        signal = chirp_slang(self, d_rt, amplitudes)
        if distances.requires_grad or amplitudes.requires_grad:
            reference = pytorch_chirp_reference(self.radar, distances, amplitudes)
            signal = signal.to(reference.dtype)
            return signal.detach() + (reference - reference.detach())
        return signal

    def frame(self, interpolator, t0=0):
        r = self.radar
        T_chirp = (r.idle_time + r.ramp_end_time) * 1e-6
        tx0 = torch.as_tensor(r.tx_pos[0:1], dtype=torch.float32, device=r.device)
        rx0 = torch.as_tensor(r.rx_pos[0:1], dtype=torch.float32, device=r.device)

        result = []
        for chirp_id in range(r.chirp_per_frame):
            time_in_frame = chirp_id * T_chirp * r.num_tx
            sample = normalize_interpolated_sample(interpolator(t0 + time_in_frame), device=r.device)
            total_lengths = compute_total_path_lengths(sample, tx0, rx0)
            one_way = total_lengths.squeeze(0).squeeze(0) * 0.5
            amp = compute_path_amplitudes(r, sample, total_lengths, tx_pos=tx0, rx_pos=rx0).squeeze(0).squeeze(0)
            result.append(self.chirp(one_way, amp))

        return torch.stack(result)

    def mimo(self, interpolator, t0=0, **options):
        self._ensure_no_options(options)
        samples = collect_interpolated_samples(self.radar, interpolator, t0)
        signal = frameCuda(self, samples)
        if samples_require_grad(samples):
            reference = pytorch_mimo_from_samples(self.radar, samples)
            signal = signal.to(reference.dtype)
            return signal.detach() + (reference - reference.detach())
        return signal
