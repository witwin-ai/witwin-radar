"""Radar simulation result containers."""

from __future__ import annotations

from typing import Any

import torch


class Result:
    def __init__(
        self,
        *,
        method: str,
        scene: Any,
        signal: torch.Tensor,
        trace: Any,
        radar: Any,
        renderer: Any,
        metadata: dict[str, Any] | None = None,
    ):
        self.method = str(method)
        self.scene = scene
        self._signal = signal
        self._trace = trace
        self.radar = radar
        self.renderer = renderer
        self.metadata = dict(metadata or {})

    @property
    def trace(self) -> Any:
        return self._trace

    def signal(self) -> torch.Tensor:
        return self._signal

    def trace_points(self) -> torch.Tensor:
        return self._trace.points

    def trace_intensities(self) -> torch.Tensor:
        return self._trace.intensities

    def trace_entry_points(self) -> torch.Tensor:
        return self._trace.entry_points

    def trace_fixed_path_lengths(self) -> torch.Tensor:
        return self._trace.fixed_path_lengths

    def trace_depths(self) -> torch.Tensor:
        return self._trace.depths

    def trace_normals(self) -> torch.Tensor | None:
        return self._trace.normals

    def tensor(self, name: str) -> torch.Tensor:
        key = str(name).lower()
        if key in {"signal", "frame", "mimo"}:
            return self.signal()
        if key in {"points", "trace_points"}:
            return self.trace_points()
        if key in {"intensities", "trace_intensities"}:
            return self.trace_intensities()
        if key in {"entry_points", "trace_entry_points"}:
            return self.trace_entry_points()
        if key in {"fixed_path_lengths", "trace_fixed_path_lengths", "path_lengths", "trace_path_lengths"}:
            return self.trace_fixed_path_lengths()
        if key in {"depths", "trace_depths"}:
            return self.trace_depths()
        if key in {"normals", "trace_normals"}:
            normals = self.trace_normals()
            if normals is None:
                raise KeyError("Trace normals are unavailable for this result.")
            return normals
        raise KeyError(f"Unknown radar result tensor '{name}'.")


class MultiResult:
    def __init__(self, *, results: dict[str, Result], metadata: dict[str, Any] | None = None):
        self._results = dict(results)
        self.metadata = dict(metadata or {})

    def __getitem__(self, name: str) -> Result:
        return self._results[name]

    def __len__(self) -> int:
        return len(self._results)

    def names(self) -> tuple[str, ...]:
        return tuple(self._results.keys())

    def result(self, name: str) -> Result:
        return self._results[name]

    def signal(self, name: str) -> torch.Tensor:
        return self._results[name].signal()

    def trace(self, name: str) -> Any:
        return self._results[name].trace

    def items(self):
        return self._results.items()

    def values(self):
        return self._results.values()

    def keys(self):
        return self._results.keys()
