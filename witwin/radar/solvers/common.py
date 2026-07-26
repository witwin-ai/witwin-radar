"""What is left of the shared solver helpers, and why so little is left.

Plan work item 8 moved this module's hot path to a native owner. The five Torch
expressions that used to live here - ``compute_total_path_lengths``,
``compute_antenna_pattern_gains``, ``compute_polarization_amplitudes``,
``compute_path_amplitudes``, and the batched ``compute_slot_path_tensors`` -
evaluated distance fields, an antenna interpolation, free-space spreading, and a
polarization projection once per frame in Torch. They are now one CUDA kernel,
the ``sensor_weight`` family, whose Python owner is
:mod:`witwin.radar.sensors.weights` and whose row assembly is
:mod:`witwin.radar.sensors.legacy_paths`.

What remains is not physics:

* :class:`PathSample` is a contract.
* :func:`normalize_interpolated_sample` is dtype and device glue.
* :func:`samples_require_grad` is a predicate, and it must never gate a call to
  ``Function.apply``: an ADR-038 forward-only dual has ``requires_grad ==
  False``, so a route chosen by this predicate would swallow its tangent.
* :func:`_stack_slot_samples` is structural packing - ``repeat_interleave``,
  ``cumsum``, and a scatter - with no expression in it that has a unit.

``tests/test_phase5_removed_entry_points.py`` freezes exactly that set.
"""

from __future__ import annotations

from dataclasses import dataclass

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


def samples_require_grad(samples) -> bool:
    return any(
        sample.intensities.requires_grad
        or sample.points.requires_grad
        or sample.entry_points.requires_grad
        or sample.fixed_path_lengths.requires_grad
        or (sample.normals is not None and sample.normals.requires_grad)
        for sample in samples
    )


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
