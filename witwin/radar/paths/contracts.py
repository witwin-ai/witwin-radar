"""Radar round-trip path contracts.

A leg is one source-to-sink propagation segment published by the Channel
consumer. A radar path is a composed round trip: radar source -> scatter site
-> radar sink. This module holds the composed row identity and the composed
row payload; the composition itself belongs to
:mod:`witwin.radar.paths.two_way`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


JoinMode = Literal["direct", "multipath"]

JOIN_MODES: frozenset[str] = frozenset({"direct", "multipath"})


def _require_tensor(
    name: str,
    value: object,
    *,
    dtype: torch.dtype,
    shape: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}")
    if value.dtype != dtype:
        raise TypeError(f"{name} must use {dtype}, got {value.dtype}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
    return value


@dataclass(frozen=True, slots=True, eq=False)
class RadarPathTopology:
    """The identity of each composed round-trip row.

    The tuple ``(radar_source_id, site_id, radar_sink_id)`` is the row's
    identity and is stable across a frozen sequence. ``inbound_row`` and
    ``outbound_row`` record which frozen leg rows were joined, so a composed
    result can always be traced back to the two legs that produced it.

    A DIRECT row - radar source straight to radar sink, with no scatter site -
    uses ``site_id = -1`` and ``outbound_row = -1``. Those are sentinels, not
    missing data: a direct path has exactly one leg, and giving it a fabricated
    second one would make it indistinguishable from a round trip through a
    target with unit response.

    Identity is what the join uses. Joining by array position instead would be
    silently wrong the moment a leg publishes its rows in a different order,
    and the resulting error looks like a physics bug rather than a bookkeeping
    one.
    """

    radar_source_id: torch.Tensor
    site_id: torch.Tensor
    radar_sink_id: torch.Tensor
    inbound_row: torch.Tensor
    outbound_row: torch.Tensor

    def __post_init__(self) -> None:
        rows = (int(self.radar_source_id.shape[0]),)
        for name in (
            "radar_source_id",
            "site_id",
            "radar_sink_id",
            "inbound_row",
            "outbound_row",
        ):
            _require_tensor(name, getattr(self, name), dtype=torch.int64, shape=rows)

    @property
    def row_count(self) -> int:
        return int(self.radar_source_id.shape[0])


@dataclass(frozen=True, slots=True, eq=False)
class RadarPathBatch:
    """Composed round-trip rows ready for waveform synthesis.

    ``complex_transfer_ref`` is published in the CHANNEL phasor convention,
    ``exp(-j * k * d)`` with ``exp(+j * 2 * pi * f * t)`` time dependence,
    at ``reference_frequency_hz``. It is NOT a beat weight. FMCW de-chirping
    conjugates the received phasor, and that conversion has exactly one call
    site, in the synthesis facade.

    ``delay_rate`` is ``d(total_delay_s)/dt`` and is primal-valued: it arrives
    as an unpacked forward tangent, so consuming it here deliberately severs
    the second-order ``d(delay_rate)/dx`` term.

    RETARDATION, stated so an absurd-velocity test cannot be misread as a bug.
    ``delay_rate`` is ``rate_in + rate_out`` with BOTH legs evaluated at the
    same world instant ``t``. The exact two-way rate evaluates the outbound leg
    at ``t + tau_in``, where the target has moved on, and carries a
    ``(1 - v_r/c)`` factor from the same retardation. The relative error of the
    same-instant form is therefore ``O(v/c)``: about ``4e-8`` at 12 m/s, which
    is five orders of magnitude below the float32 delay quantisation these rows
    are published at. It is an approximation, it is named here rather than left
    implicit, and it is not corrected because the correction is smaller than the
    representation. A test driven at a relativistic velocity measures this
    approximation; it has not found a defect.

    ``row_valid`` is the sole authority on whether a row means anything. A
    dead row is a complete answer contributing exactly zero, never an error,
    and validity is never inferred from a zero payload.

    ``join_mode`` records which composer produced these rows. It is stored
    rather than inferred so that "which paths am I looking at" is a checkable
    property of the result and never a guess from its shape. Both modes publish
    THIS contract, so a consumer downstream of it - synthesis, in particular -
    needs no branch; the choice is made once, by the caller, upstream.
    """

    sensor_pair_count: int
    path_count: int
    sensor_pair_index: torch.Tensor
    pair_offsets: torch.Tensor
    total_delay_s: torch.Tensor
    delay_rate: torch.Tensor | None
    complex_transfer_ref: torch.Tensor
    reference_frequency_hz: float
    row_valid: torch.Tensor | None
    topology: RadarPathTopology
    join_mode: JoinMode

    def __post_init__(self) -> None:
        if self.join_mode not in JOIN_MODES:
            raise ValueError(
                f"join_mode must be one of {sorted(JOIN_MODES)}, got "
                f"{self.join_mode!r}"
            )
        if type(self.sensor_pair_count) is not int or self.sensor_pair_count < 1:
            raise ValueError("sensor_pair_count must be a positive int")
        if type(self.path_count) is not int or self.path_count < 0:
            raise ValueError("path_count must be a non-negative int")
        rows = (self.path_count,)
        _require_tensor(
            "sensor_pair_index", self.sensor_pair_index, dtype=torch.int64, shape=rows
        )
        _require_tensor(
            "pair_offsets",
            self.pair_offsets,
            dtype=torch.int64,
            shape=(self.sensor_pair_count + 1,),
        )
        _require_tensor(
            "total_delay_s", self.total_delay_s, dtype=torch.float32, shape=rows
        )
        _require_tensor(
            "complex_transfer_ref",
            self.complex_transfer_ref,
            dtype=torch.complex64,
            shape=rows,
        )
        if self.delay_rate is not None:
            _require_tensor(
                "delay_rate", self.delay_rate, dtype=torch.float32, shape=rows
            )
        if self.row_valid is not None:
            _require_tensor(
                "row_valid", self.row_valid, dtype=torch.bool, shape=rows
            )
        if self.topology.row_count != self.path_count:
            raise ValueError("topology must have exactly path_count rows")

    @property
    def device(self) -> torch.device:
        return self.total_delay_s.device


__all__ = ["JOIN_MODES", "JoinMode", "RadarPathBatch", "RadarPathTopology"]
