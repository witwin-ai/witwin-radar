"""Shared runtime helpers for radar solver backends."""

from __future__ import annotations

from dataclasses import dataclass
import os
import sys

import math
import torch


def ensure_current_env_on_path() -> None:
    """Prepend the active interpreter environment to PATH for subprocess tools."""
    env_root = os.path.dirname(sys.executable)
    candidates = [
        env_root,
        os.path.join(env_root, "Scripts"),
        os.path.join(env_root, "Library", "bin"),
    ]
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    prepend = [path for path in candidates if os.path.isdir(path) and path not in path_entries]
    if prepend:
        os.environ["PATH"] = os.pathsep.join(prepend + path_entries)


@dataclass(frozen=True)
class PathSample:
    intensities: torch.Tensor
    points: torch.Tensor
    entry_points: torch.Tensor
    fixed_path_lengths: torch.Tensor
    depths: torch.Tensor
    normals: torch.Tensor | None


def normalize_interpolated_sample(sample, *, device: str) -> PathSample:
    """Normalize legacy tuples and TraceResult-like objects into path samples."""
    if hasattr(sample, "points") and hasattr(sample, "intensities"):
        intensities = sample.intensities
        points = sample.points
        entry_points = getattr(sample, "entry_points", None)
        fixed_path_lengths = getattr(sample, "fixed_path_lengths", None)
        depths = getattr(sample, "depths", None)
        normals = getattr(sample, "normals", None)
    else:
        intensities, points = sample
        entry_points = None
        fixed_path_lengths = None
        depths = None
        normals = None

    points = points.to(dtype=torch.float32, device=device)
    intensities = intensities.to(dtype=torch.float32, device=device)

    if entry_points is None:
        entry_points = points
    else:
        entry_points = entry_points.to(dtype=torch.float32, device=device)

    if fixed_path_lengths is None:
        fixed_path_lengths = torch.zeros(points.shape[0], dtype=torch.float32, device=device)
    else:
        fixed_path_lengths = fixed_path_lengths.to(dtype=torch.float32, device=device)

    if depths is None:
        depths = torch.zeros(points.shape[0], dtype=torch.int32, device=device)
    else:
        depths = depths.to(dtype=torch.int32, device=device)

    if normals is not None:
        normals = normals.to(dtype=torch.float32, device=device)

    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError("Interpolated sample points must have shape (N, 3).")
    if entry_points.shape != points.shape:
        raise ValueError("Interpolated sample entry_points must match points shape.")
    if intensities.shape != fixed_path_lengths.shape:
        raise ValueError("Interpolated sample intensities and fixed_path_lengths must share shape (N,).")
    if depths.shape != intensities.shape:
        raise ValueError("Interpolated sample depths must have shape (N,).")
    if normals is not None and normals.shape != points.shape:
        raise ValueError("Interpolated sample normals must match points shape.")

    return PathSample(
        intensities=intensities,
        points=points,
        entry_points=entry_points,
        fixed_path_lengths=fixed_path_lengths,
        depths=depths,
        normals=normals,
    )


def collect_interpolated_samples(radar, interpolator, t0=0.0):
    """Evaluate the scene interpolator once per chirp and keep tensors on-graph."""
    chirp_period = (radar.idle_time + radar.ramp_end_time) * 1e-6
    samples = []
    for chirp_id in range(radar.chirp_per_frame):
        time_in_frame = chirp_id * chirp_period * radar.num_tx
        sample = interpolator(t0 + time_in_frame)
        samples.append(normalize_interpolated_sample(sample, device=radar.device))
    return samples


def samples_require_grad(samples) -> bool:
    return any(
        sample.intensities.requires_grad
        or sample.points.requires_grad
        or sample.entry_points.requires_grad
        or sample.fixed_path_lengths.requires_grad
        or (sample.normals is not None and sample.normals.requires_grad)
        for sample in samples
    )


def compute_total_path_lengths(sample: PathSample, tx_pos: torch.Tensor, rx_pos: torch.Tensor) -> torch.Tensor:
    """Return total path lengths with shape (TX, RX, N)."""
    dist_tx = torch.cdist(sample.entry_points, tx_pos).transpose(0, 1).unsqueeze(1)
    dist_rx = torch.cdist(sample.points, rx_pos).transpose(0, 1).unsqueeze(0)
    return dist_tx + sample.fixed_path_lengths.view(1, 1, -1) + dist_rx


def compute_antenna_pattern_gains(
    radar,
    sample: PathSample,
    tx_pos: torch.Tensor,
    rx_pos: torch.Tensor,
) -> torch.Tensor | None:
    """Return per-path power gains from the configured TX/RX antenna pattern."""
    pattern = getattr(radar, "antenna_pattern", None)
    if pattern is None:
        return None

    tx_vectors = sample.entry_points.unsqueeze(0) - tx_pos.unsqueeze(1)
    rx_vectors = sample.points.unsqueeze(0) - rx_pos.unsqueeze(1)
    if hasattr(radar, "local_from_world_vectors"):
        tx_vectors = radar.local_from_world_vectors(tx_vectors)
        rx_vectors = radar.local_from_world_vectors(rx_vectors)
    tx_gains = pattern.evaluate_vectors(tx_vectors).unsqueeze(1)
    rx_gains = pattern.evaluate_vectors(rx_vectors).unsqueeze(0)
    return tx_gains * rx_gains


def _normalize_vectors(vectors: torch.Tensor) -> torch.Tensor:
    return vectors / torch.clamp(torch.linalg.norm(vectors, dim=-1, keepdim=True), min=1e-12)


def compute_polarization_amplitudes(radar, sample: PathSample) -> torch.Tensor | None:
    """Return signed TX/RX polarization projection factors for each path."""
    polarization = getattr(radar, "polarization", None)
    if polarization is None:
        return None
    if sample.normals is None:
        raise ValueError("Radar polarization requires per-path surface normals in the interpolated sample.")

    normals = _normalize_vectors(sample.normals)
    tx_world = _normalize_vectors(polarization.tx_world.to(device=normals.device, dtype=normals.dtype))
    rx_world = _normalize_vectors(polarization.rx_world.to(device=normals.device, dtype=normals.dtype))

    reflected_tx = tx_world.unsqueeze(1)
    if polarization.reflection_flip:
        reflected_tx = reflected_tx - 2.0 * (reflected_tx * normals.unsqueeze(0)).sum(dim=-1, keepdim=True) * normals.unsqueeze(0)
    reflected_tx = _normalize_vectors(reflected_tx)
    return (reflected_tx.unsqueeze(1) * rx_world.view(1, rx_world.shape[0], 1, 3)).sum(dim=-1)


def compute_path_amplitudes(
    radar,
    sample: PathSample,
    total_path_lengths: torch.Tensor,
    *,
    tx_pos: torch.Tensor | None = None,
    rx_pos: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convert power-domain material coefficients to amplitude-domain weights with FSPL."""
    fspl_amp = radar._lambda / (4.0 * math.pi * torch.clamp(total_path_lengths, min=1e-6))
    scatter_power = torch.clamp(sample.intensities, min=0.0).view(1, 1, -1)
    if tx_pos is None:
        tx_pos = getattr(radar, "tx_pos", radar.tx_loc)
    if rx_pos is None:
        rx_pos = getattr(radar, "rx_pos", radar.rx_loc)

    tx_pos = torch.as_tensor(tx_pos, dtype=torch.float32, device=total_path_lengths.device)
    rx_pos = torch.as_tensor(rx_pos, dtype=torch.float32, device=total_path_lengths.device)
    pattern_gains = compute_antenna_pattern_gains(radar, sample, tx_pos, rx_pos)
    if pattern_gains is not None:
        scatter_power = scatter_power * torch.clamp(pattern_gains, min=0.0)
    amplitudes = radar.gain * torch.sqrt(scatter_power) * fspl_amp
    polarization_factor = compute_polarization_amplitudes(radar, sample)
    if polarization_factor is not None:
        amplitudes = amplitudes * polarization_factor
    return amplitudes


def pytorch_chirp_reference(radar, distances, amplitudes):
    d_rt = (distances * 2).unsqueeze(-1)
    toa = d_rt / radar.c0
    rx = radar.waveform(radar.t_sample - toa)
    rx_weighted = rx * amplitudes.unsqueeze(-1)
    rx_combined = rx_weighted.sum(dim=0)
    return radar.tx_waveform * torch.conj(rx_combined)


def pytorch_mimo_from_samples(radar, samples):
    frame = torch.zeros(
        (radar.chirp_per_frame, radar.num_tx, radar.num_rx, radar.adc_samples),
        dtype=torch.complex128,
        device=radar.device,
    )
    tx_pos = torch.as_tensor(radar.tx_pos, device=radar.device, dtype=torch.float32)
    rx_pos = torch.as_tensor(radar.rx_pos, device=radar.device, dtype=torch.float32)

    for chirp_id, sample in enumerate(samples):
        distances = compute_total_path_lengths(sample, tx_pos, rx_pos).unsqueeze(-1)
        toa = distances / radar.c0
        rx = radar.waveform(radar.t_sample - toa)
        amplitudes = compute_path_amplitudes(radar, sample, distances.squeeze(-1), tx_pos=tx_pos, rx_pos=rx_pos)
        rx_weighted = rx * amplitudes.view(1, 1, -1, 1)
        rx_combined = torch.sum(rx_weighted, dim=-2)
        frame[chirp_id] = radar.tx_waveform * torch.conj(rx_combined)

    return frame.permute(1, 2, 0, 3)
