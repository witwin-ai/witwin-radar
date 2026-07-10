"""Shared solver helpers used by all radar backends."""

from __future__ import annotations

from dataclasses import dataclass

import math
import torch


@dataclass(frozen=True)
class PathSample:
    intensities: torch.Tensor
    points: torch.Tensor
    entry_points: torch.Tensor
    fixed_path_lengths: torch.Tensor
    depths: torch.Tensor
    normals: torch.Tensor | None


def normalize_interpolated_sample(sample, *, device: str | torch.device) -> PathSample:
    """Promote a TraceResult-like sample (or legacy ``(intensities, points)`` tuple) onto ``device``."""
    if isinstance(sample, tuple):
        intensities, points = sample
        points = points.to(dtype=torch.float32, device=device)
        intensities = intensities.to(dtype=torch.float32, device=device)
        return PathSample(
            intensities=intensities,
            points=points,
            entry_points=points,
            fixed_path_lengths=torch.zeros(points.shape[0], dtype=torch.float32, device=device),
            depths=torch.zeros(points.shape[0], dtype=torch.int32, device=device),
            normals=None,
        )

    points = sample.points.to(dtype=torch.float32, device=device)
    intensities = sample.intensities.to(dtype=torch.float32, device=device)
    entry_points = sample.entry_points.to(dtype=torch.float32, device=device)
    fixed_path_lengths = sample.fixed_path_lengths.to(dtype=torch.float32, device=device)
    depths = sample.depths.to(dtype=torch.int32, device=device)
    normals = sample.normals.to(dtype=torch.float32, device=device) if sample.normals is not None else None

    return PathSample(
        intensities=intensities,
        points=points,
        entry_points=entry_points,
        fixed_path_lengths=fixed_path_lengths,
        depths=depths,
        normals=normals,
    )


def collect_interpolated_samples(radar, interpolator, t0=0.0):
    """Evaluate the scene interpolator once per TDM chirp slot and keep tensors on-graph.

    TDM-MIMO fires TX antennas sequentially: slot ``chirp_id * num_tx + tx_id``
    starts ``slot * chirp_period`` into the frame. The returned list holds
    ``chirp_per_frame * num_tx`` samples in slot order, so per-TX motion phase
    (the phase ``_compensate_tdm_phase`` removes downstream) is simulated.
    """
    cfg = radar.config
    chirp_period = (cfg.idle_time + cfg.ramp_end_time) * 1e-6
    samples = []
    for slot in range(cfg.chirp_per_frame * cfg.num_tx):
        sample = interpolator(t0 + slot * chirp_period)
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
    tx_vectors = radar.local_from_world_vectors(sample.entry_points.unsqueeze(0) - tx_pos.unsqueeze(1))
    rx_vectors = radar.local_from_world_vectors(sample.points.unsqueeze(0) - rx_pos.unsqueeze(1))
    tx_gains = radar.evaluate_antenna_pattern_vectors(tx_vectors).unsqueeze(1)
    rx_gains = radar.evaluate_antenna_pattern_vectors(rx_vectors).unsqueeze(0)
    return tx_gains * rx_gains


def _normalize_vectors(vectors: torch.Tensor) -> torch.Tensor:
    return vectors / torch.clamp(torch.linalg.norm(vectors, dim=-1, keepdim=True), min=1e-12)


def compute_polarization_amplitudes(radar, sample: PathSample) -> torch.Tensor | None:
    """Return signed TX/RX polarization projection factors for each path."""
    polarization = radar.polarization
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
    tx_index: int | None = None,
) -> torch.Tensor:
    """Convert power-domain material coefficients to amplitude-domain weights with FSPL.

    ``tx_index`` selects a single TX row of the polarization factors when
    ``tx_pos`` holds only that antenna (per-TDM-slot evaluation).
    """
    fspl_amp = radar._lambda / (4.0 * math.pi * torch.clamp(total_path_lengths, min=1e-6))
    scatter_power = torch.clamp(sample.intensities, min=0.0).view(1, 1, -1)
    if tx_pos is None:
        tx_pos = radar.tx_pos
    if rx_pos is None:
        rx_pos = radar.rx_pos
    pattern_gains = compute_antenna_pattern_gains(radar, sample, tx_pos, rx_pos)
    if pattern_gains is not None:
        scatter_power = scatter_power * torch.clamp(pattern_gains, min=0.0)
    amplitudes = radar.gain * torch.sqrt(scatter_power) * fspl_amp
    polarization_factor = compute_polarization_amplitudes(radar, sample)
    if polarization_factor is not None:
        if tx_index is not None:
            polarization_factor = polarization_factor[tx_index : tx_index + 1]
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
    """Float64 time-domain MIMO reference over per-TDM-slot samples.

    Slot ``chirp_id * num_tx + tx_id`` contributes the (tx_id, :, chirp_id)
    rows of the frame from its own scene state and TX antenna.
    """
    cfg = radar.config
    frame = torch.zeros(
        (cfg.num_tx, cfg.num_rx, cfg.chirp_per_frame, cfg.adc_samples),
        dtype=torch.complex128,
        device=radar.device,
    )
    rx_pos = radar.rx_pos

    for slot, sample in enumerate(samples):
        chirp_id, tx_id = divmod(slot, cfg.num_tx)
        if sample.points.shape[0] == 0:
            continue
        tx_pos = radar.tx_pos[tx_id : tx_id + 1]
        distances = compute_total_path_lengths(sample, tx_pos, rx_pos).unsqueeze(-1)
        toa = distances / radar.c0
        rx = radar.waveform(radar.t_sample - toa)
        amplitudes = compute_path_amplitudes(
            radar,
            sample,
            distances.squeeze(-1),
            tx_pos=tx_pos,
            rx_pos=rx_pos,
            tx_index=tx_id,
        )
        rx_combined = torch.sum(rx * amplitudes.unsqueeze(-1), dim=-2)
        frame[tx_id, :, chirp_id] = radar.tx_waveform * torch.conj(rx_combined.squeeze(0))

    return frame


def _stack_slot_samples(samples, *, with_normals: bool):
    """Stack per-slot sample fields into zero-padded (slots, N_max, ...) tensors.

    Padded rows keep zero intensity, so downstream amplitude math zeroes them.
    Fields remain on-graph so the native spectrum backward can propagate
    distance and amplitude gradients into scene parameters.
    """
    counts = [int(sample.points.shape[0]) for sample in samples]
    n_max = max(counts)
    if n_max == 0:
        return None
    num_slots = len(samples)
    device = samples[0].points.device

    fields = [
        [sample.points for sample in samples],
        [sample.entry_points for sample in samples],
        [sample.fixed_path_lengths for sample in samples],
        [sample.intensities for sample in samples],
    ]
    if with_normals:
        if any(sample.normals is None for sample in samples):
            raise ValueError("Radar polarization requires per-path surface normals in the interpolated sample.")
        fields.append([sample.normals for sample in samples])

    if all(count == n_max for count in counts):
        stacked = [torch.stack(field) for field in fields]
    else:
        counts_t = torch.tensor(counts, device=device)
        slot_ids = torch.repeat_interleave(torch.arange(num_slots, device=device), counts_t)
        starts = torch.cumsum(counts_t, dim=0) - counts_t
        within = torch.arange(int(counts_t.sum()), device=device) - starts[slot_ids]
        flat_idx = slot_ids * n_max + within

        stacked = []
        for field in fields:
            flat = torch.cat(field, dim=0)
            out = torch.zeros((num_slots * n_max, *flat.shape[1:]), dtype=flat.dtype, device=device)
            out[flat_idx] = flat
            stacked.append(out.view(num_slots, n_max, *flat.shape[1:]))

    if not with_normals:
        stacked.append(None)
    return stacked


def _slot_polarization_factors(radar, normals: torch.Tensor, slot_tx: torch.Tensor) -> torch.Tensor:
    """Per-slot polarization projection factors, shape (slots, N, RX)."""
    polarization = radar.polarization
    normals_n = _normalize_vectors(normals)
    tx_world = _normalize_vectors(polarization.tx_world.to(device=normals.device, dtype=normals.dtype))
    rx_world = _normalize_vectors(polarization.rx_world.to(device=normals.device, dtype=normals.dtype))

    reflected_tx = tx_world[slot_tx].unsqueeze(1)
    if polarization.reflection_flip:
        reflected_tx = reflected_tx - 2.0 * (reflected_tx * normals_n).sum(dim=-1, keepdim=True) * normals_n
    reflected_tx = _normalize_vectors(reflected_tx)
    return (reflected_tx.unsqueeze(2) * rx_world.view(1, 1, -1, 3)).sum(dim=-1)


def compute_slot_path_tensors(radar, samples, *, first_slot: int = 0):
    """Batched per-slot path geometry for the native MIMO forward path.

    Slot ``first_slot + s`` transmits from TX antenna ``(first_slot + s) % num_tx``.
    Returns ``(one_way_distances, amplitudes)`` with shape (slots, RX, N_max),
    or ``(None, None)`` when every slot is empty. Padded entries carry zero
    amplitude, which the CUDA kernels skip.
    """
    cfg = radar.config
    device = radar.device
    num_slots = len(samples)
    with_normals = radar.polarization is not None
    packed = _stack_slot_samples(samples, with_normals=with_normals)
    if packed is None:
        return None, None
    points, entry_points, fixed_path_lengths, intensities, normals = packed

    slot_tx = (torch.arange(num_slots, device=device) + first_slot) % cfg.num_tx
    tx_sel = radar.tx_pos[slot_tx]
    rx_pos = radar.rx_pos

    dist_tx = torch.cdist(entry_points, tx_sel.unsqueeze(1)).squeeze(-1)
    dist_rx = torch.cdist(points, rx_pos.unsqueeze(0).expand(num_slots, -1, -1))
    total = dist_tx.unsqueeze(-1) + fixed_path_lengths.unsqueeze(-1) + dist_rx

    fspl_amp = radar._lambda / (4.0 * math.pi * torch.clamp(total, min=1e-6))
    scatter_power = torch.clamp(intensities, min=0.0).unsqueeze(-1)

    tx_vectors = radar.local_from_world_vectors(entry_points - tx_sel.unsqueeze(1))
    rx_vectors = radar.local_from_world_vectors(points.unsqueeze(2) - rx_pos.view(1, 1, -1, 3))
    tx_gains = radar.evaluate_antenna_pattern_vectors(tx_vectors)
    rx_gains = radar.evaluate_antenna_pattern_vectors(rx_vectors)
    scatter_power = scatter_power * torch.clamp(tx_gains.unsqueeze(-1) * rx_gains, min=0.0)

    amplitudes = radar.gain * torch.sqrt(scatter_power) * fspl_amp
    if with_normals:
        amplitudes = amplitudes * _slot_polarization_factors(radar, normals, slot_tx)

    one_way = (total * 0.5).transpose(1, 2).contiguous()
    amplitudes = amplitudes.transpose(1, 2).contiguous()
    return one_way, amplitudes
