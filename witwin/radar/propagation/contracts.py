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

    ``field_direction`` is the row's PROPAGATION direction, a unit vector in
    world coordinates, aliased from the consumer's ``PropagationGeometry``. It
    is the direction of the row's FINAL segment, so it is the direction the
    field arrives at the sink travelling in - which for a line-of-sight row is
    also the direction it left the source in, and for a higher-order row is
    not. An aspect-dependent scatter response consumes it and is responsible
    for saying which of the two meanings it needs; see
    :mod:`witwin.radar.scattering.aspect`, which refuses an outbound leg whose
    rows are not line of sight rather than reading a departure direction off a
    row that does not carry one.

    It is optional only because a fabricated leg row - a test that builds a
    batch by hand to reach a validation path - has no geometry behind it. Every
    batch the adapter publishes carries it, and a consumer that needs it
    refuses a batch without it by name rather than inventing one.

    ``slot_count`` states how many time slots - TDM slots, OFDM symbols or
    pulses - this batch carries. The default ``1`` is one instant and is what
    every single-shot reevaluation publishes. A batch with ``slot_count > 1``
    is SLOT MAJOR and FROZEN-ROW MINOR: row ``t * rows_per_slot + r`` is frozen
    row ``r`` at slot ``t``, and the pair partition is block diagonal, so pair
    ``t * pairs_per_slot + p`` is slot ``t``'s pair ``p``. That is the Channel
    consumer's ``slot_pair_layout``, restated here rather than reinvented; the
    whole point of the layout is that ``pair_count`` grows LINEARLY in the slot
    count instead of quadratically. :meth:`slot` is the only supported way to
    address one slot, so no consumer has to rederive the arithmetic.
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
    slot_count: int = 1
    field_direction: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if type(self.leg_count) is not int or self.leg_count < 0:
            raise ValueError("leg_count must be a non-negative int")
        if type(self.pair_count) is not int or self.pair_count < 0:
            raise ValueError("pair_count must be a non-negative int")
        if type(self.slot_count) is not int or self.slot_count < 1:
            raise ValueError("slot_count must be a positive int")
        for name, total in (
            ("leg_count", self.leg_count),
            ("pair_count", self.pair_count),
        ):
            if total % self.slot_count:
                raise ValueError(
                    f"{name} {total} is not divisible by slot_count "
                    f"{self.slot_count}; a slot-major batch carries the same "
                    "frozen rows and the same pair partition in every slot"
                )
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
        if self.field_direction is not None:
            _require_tensor(
                "field_direction",
                self.field_direction,
                dtype=torch.float32,
                shape=(self.leg_count, 3),
            )

    @property
    def device(self) -> torch.device:
        return self.delay_s.device

    @property
    def rows_per_slot(self) -> int:
        return self.leg_count // self.slot_count

    @property
    def pairs_per_slot(self) -> int:
        return self.pair_count // self.slot_count

    def slot(self, index: int) -> "RadarLegBatch":
        """One slot of a slot-major batch, as a single-slot batch.

        The payload members are NARROWED, so ``delay_s``, ``coefficient``,
        ``delay_rate`` and ``row_valid`` still alias the batched storage and a
        gradient flows straight back through them. Only the two partition
        tables are rebased, because a slot's pair ranks have to start at zero
        for the slice to be a partition of that slot's rows; rebasing them is
        int64 metadata arithmetic and reads no payload value.

        This exists so that a consumer written against the single-slot contract
        - the two-way join, in particular - can be driven per slot WITHOUT a
        second statement of the block-diagonal layout living in the caller.
        """

        if type(index) is not int or not 0 <= index < self.slot_count:
            raise ValueError(
                f"slot index must be an int in [0, {self.slot_count}), "
                f"got {index!r}"
            )
        rows = self.rows_per_slot
        pairs = self.pairs_per_slot
        start = index * rows
        stop = start + rows
        base = index * pairs

        def narrow(value):
            return None if value is None else value[start:stop]

        return RadarLegBatch(
            leg_count=rows,
            pair_count=pairs,
            pair_index=self.pair_index[start:stop] - base,
            pair_offsets=(
                self.pair_offsets[base : base + pairs + 1]
                - self.pair_offsets[base]
            ),
            source_index=narrow(self.source_index),
            sink_index=narrow(self.sink_index),
            depth=narrow(self.depth),
            component_id=narrow(self.component_id),
            source_id=narrow(self.source_id),
            sink_id=narrow(self.sink_id),
            primitive_sequence=narrow(self.primitive_sequence),
            material_sequence=narrow(self.material_sequence),
            interaction_type=narrow(self.interaction_type),
            delay_s=narrow(self.delay_s),
            coefficient=narrow(self.coefficient),
            delay_rate=narrow(self.delay_rate),
            row_valid=narrow(self.row_valid),
            diagnostics=self.diagnostics,
            slot_count=1,
            field_direction=narrow(self.field_direction),
        )


__all__ = [
    "EndpointRole",
    "RadarEndpointSpec",
    "RadarLegBatch",
    "require_endpoint_role",
]
