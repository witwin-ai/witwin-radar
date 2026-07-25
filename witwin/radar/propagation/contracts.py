"""Radar-shaped propagation contracts.

These types are the Radar side of the Channel consumer boundary. They are
deliberately free of any ``witwin.channel`` import so that
``witwin.radar.propagation`` can be imported on a machine that has no
``witwin-channel`` installed; only :mod:`witwin.radar.propagation.channel_consumer`
reaches across the boundary.

A leg is one source-to-sink propagation segment. A radar round trip is composed
from two legs by :mod:`witwin.radar.paths.two_way`; this module has no opinion
about that composition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


EndpointRole = Literal["source", "sink"]


def _require_tensor(
    name: str,
    value: object,
    *,
    dtype: torch.dtype,
    ndim: int | None = None,
    shape: tuple[int, ...] | None = None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}")
    if value.dtype != dtype:
        raise TypeError(f"{name} must use {dtype}, got {value.dtype}")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}, got {value.ndim}")
    if shape is not None and tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    return value


@dataclass(frozen=True, slots=True, eq=False)
class RadarEndpointSpec:
    """One batch of radar endpoints in world coordinates.

    ``positions_m`` is the only differentiable member. It may carry
    ``requires_grad`` for reverse mode or a forward-AD tangent for the
    ADR-038 forward-only dual; the remaining members are primal-only because
    the native field companions treat them as constants.

    Structural validation runs here and is device-agnostic on purpose: the
    CUDA requirement belongs to the Channel endpoint contract, so a caller
    gets the shape or dtype complaint it actually made rather than a device
    complaint that hides it.
    """

    stable_ids: torch.Tensor
    positions_m: torch.Tensor
    polarizations: torch.Tensor
    powers_w: torch.Tensor | None = None

    def __post_init__(self) -> None:
        positions = _require_tensor(
            "positions_m", self.positions_m, dtype=torch.float32, ndim=2
        )
        if positions.shape[1] != 3:
            raise ValueError(
                f"positions_m must have shape (N, 3), got {tuple(positions.shape)}"
            )
        rows = int(positions.shape[0])
        _require_tensor(
            "stable_ids", self.stable_ids, dtype=torch.int64, shape=(rows,)
        )
        _require_tensor(
            "polarizations", self.polarizations, dtype=torch.float32, shape=(rows, 3)
        )
        if self.powers_w is not None:
            _require_tensor(
                "powers_w", self.powers_w, dtype=torch.float32, shape=(rows,)
            )
        device = positions.device
        for name, value in (
            ("stable_ids", self.stable_ids),
            ("polarizations", self.polarizations),
            ("powers_w", self.powers_w),
        ):
            if value is not None and value.device != device:
                raise ValueError(
                    f"{name} must share the positions_m device {device}, "
                    f"got {value.device}"
                )

    @property
    def count(self) -> int:
        return int(self.positions_m.shape[0])

    @property
    def device(self) -> torch.device:
        return self.positions_m.device


def require_endpoint_role(spec: RadarEndpointSpec, role: EndpointRole) -> None:
    """Enforce the Channel source/sink power contract before any native work.

    A source radiates and therefore carries ``powers_w``; a sink receives and
    must not. Getting this wrong is rejected by the consumer anyway, but the
    Radar-side message names the leg endpoint the caller actually passed.
    """

    if role not in ("source", "sink"):
        raise ValueError(f"role must be 'source' or 'sink', got {role!r}")
    if role == "source" and spec.powers_w is None:
        raise ValueError("a source endpoint requires powers_w")
    if role == "sink" and spec.powers_w is not None:
        raise ValueError("a sink endpoint must not carry powers_w")


@dataclass(frozen=True, slots=True, eq=False)
class RadarLegBatch:
    """One reevaluated propagation leg in Radar vocabulary.

    ``delay_s`` and ``coefficient`` ALIAS the consumer tensors: same storage,
    same stride, same gradient state. Copying them would silently break the
    zero-copy discipline the compact contract exists to provide, so a change
    here has to preserve object identity.

    ``row_valid`` is the sole authority on whether a row's payload means
    anything. A dead row is a complete answer that this frozen path does not
    exist at these endpoint positions, contributing exactly zero; it is never
    an error and validity is never inferred from a zero payload.

    ``delay_rate`` is ``d(delay_s)/dt`` in seconds per second, unpacked from a
    forward-only dual and published as a PRIMAL value. It is the Doppler
    primitive; consuming it as a primal deliberately severs the second-order
    ``d(delay_rate)/dx`` term, which this contract does not claim.

    ``source_id``, ``sink_id``, ``primitive_sequence``, ``material_sequence``
    and ``interaction_type`` are the row's stable IDENTITY. They come straight
    off the frozen topology, so they are the same tensor objects on every frame
    of a frozen sequence and cost nothing to publish. A two-way composer joins
    on them; the sequences in particular are ADR-037 frozen labels rather than
    re-validated hits, which is exactly what makes them a stable key.
    """

    leg_count: int
    pair_count: int
    pair_index: torch.Tensor
    pair_offsets: torch.Tensor
    source_index: torch.Tensor
    sink_index: torch.Tensor
    depth: torch.Tensor
    component_id: torch.Tensor
    source_id: torch.Tensor
    sink_id: torch.Tensor
    primitive_sequence: torch.Tensor
    material_sequence: torch.Tensor
    interaction_type: torch.Tensor
    delay_s: torch.Tensor
    coefficient: torch.Tensor
    delay_rate: torch.Tensor | None
    row_valid: torch.Tensor | None
    diagnostics: object

    def __post_init__(self) -> None:
        if type(self.leg_count) is not int or self.leg_count < 0:
            raise ValueError("leg_count must be a non-negative int")
        if type(self.pair_count) is not int or self.pair_count < 0:
            raise ValueError("pair_count must be a non-negative int")
        rows = (self.leg_count,)
        _require_tensor("pair_index", self.pair_index, dtype=torch.int64, shape=rows)
        _require_tensor(
            "pair_offsets",
            self.pair_offsets,
            dtype=torch.int64,
            shape=(self.pair_count + 1,),
        )
        for name in ("source_index", "sink_index", "depth", "component_id"):
            _require_tensor(
                name, getattr(self, name), dtype=torch.int32, shape=rows
            )
        for name in ("source_id", "sink_id"):
            _require_tensor(
                name, getattr(self, name), dtype=torch.int64, shape=rows
            )
        width = int(self.primitive_sequence.shape[1]) if (
            isinstance(self.primitive_sequence, torch.Tensor)
            and self.primitive_sequence.ndim == 2
        ) else -1
        if width < 0:
            raise ValueError("primitive_sequence must have shape (rows, width)")
        for name in (
            "primitive_sequence",
            "material_sequence",
            "interaction_type",
        ):
            _require_tensor(
                name,
                getattr(self, name),
                dtype=torch.int32,
                shape=(self.leg_count, width),
            )
        _require_tensor("delay_s", self.delay_s, dtype=torch.float32, shape=rows)
        _require_tensor(
            "coefficient", self.coefficient, dtype=torch.complex64, shape=rows
        )
        if self.delay_rate is not None:
            _require_tensor(
                "delay_rate", self.delay_rate, dtype=torch.float32, shape=rows
            )
        if self.row_valid is not None:
            _require_tensor(
                "row_valid", self.row_valid, dtype=torch.bool, shape=rows
            )

    @property
    def device(self) -> torch.device:
        return self.delay_s.device


__all__ = [
    "EndpointRole",
    "RadarEndpointSpec",
    "RadarLegBatch",
    "require_endpoint_role",
]
