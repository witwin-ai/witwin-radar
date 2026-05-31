"""Trace result container shared by radar tracing and solver code."""

from __future__ import annotations

import torch


class TraceResult:
    """Opaque trace result. Supports ``points, intensities = tracer.trace()``."""

    __slots__ = (
        "points",
        "intensities",
        "entry_points",
        "fixed_path_lengths",
        "depths",
        "normals",
        "_tri_indices",
    )

    def __init__(
        self,
        points,
        intensities,
        tri_indices=None,
        *,
        entry_points=None,
        fixed_path_lengths=None,
        depths=None,
        normals=None,
    ):
        self.points = points
        self.intensities = intensities
        self.entry_points = points if entry_points is None else entry_points
        if fixed_path_lengths is None:
            fixed_path_lengths = torch.zeros(points.shape[0], dtype=torch.float32, device=points.device)
        self.fixed_path_lengths = fixed_path_lengths
        if depths is None:
            depths = torch.zeros(points.shape[0], dtype=torch.int32, device=points.device)
        self.depths = depths
        self.normals = normals
        self._tri_indices = tri_indices

    def __iter__(self):
        yield self.points
        yield self.intensities

    def __repr__(self):
        return f"TraceResult({self.points.shape[0]} points)"


def empty_trace(device: torch.device, *, include_tri_indices: bool = False) -> TraceResult:
    tri_indices = None
    if include_tri_indices:
        tri_indices = torch.empty((0,), dtype=torch.int64, device=device)
    return TraceResult(
        torch.empty((0, 3), dtype=torch.float32, device=device),
        torch.empty((0,), dtype=torch.float32, device=device),
        tri_indices,
        entry_points=torch.empty((0, 3), dtype=torch.float32, device=device),
        fixed_path_lengths=torch.empty((0,), dtype=torch.float32, device=device),
        depths=torch.empty((0,), dtype=torch.int32, device=device),
        normals=torch.empty((0, 3), dtype=torch.float32, device=device),
    )
