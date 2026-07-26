"""Row assembly for the legacy real-amplitude Radar route (work item 8).

``solvers/common.py`` used to hold five Torch expressions - two ``cdist``
distance fields, the free-space spreading term, the antenna-pattern
interpolation, and the polarization projection - and evaluate them once per
frame. Plan work item 8 moves that hot path to a native owner. The owner is the
``sensor_weight`` family; what is left here is the STRUCTURAL half of the move:
naming which transmitter and which receiver each row belongs to, and expanding
per-path scene tensors across the antenna pairs.

Everything in this module is packing. There is no distance, no angle, no
interpolation, and no projection: those are all inside one CUDA kernel now, and
this module's job is to hand that kernel a row set and read its answer back.

The three mode flags are fixed for this route and the reason is the single-count
rule. A legacy real amplitude carries no free-space spreading, no transmit
power, and no polarization projection - the Torch expression it replaces applied
all three itself - so all three are ON here. A Channel-sourced weight carries
all three already and arrives through
:meth:`SynthesisPathBatch.from_radar_paths` with all three OFF. The flags are
never a caller's choice in either case.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .contracts import AntennaPatternSpec, SPEED_OF_LIGHT_M_PER_S
from .weights import (
    ROW_KIND_VIA,
    SensorWeightGeometry,
    SensorWeightModes,
    SensorWeightPlan,
    SensorWeightResult,
    evaluate_sensor_weights,
)

#: The polarization vector handed to the kernel when a radar declares none. It
#: is never read - ``legacy_real_polarization`` is 0 in that case - but the ABI
#: takes a tensor rather than an optional, because a null pointer that is never
#: dereferenced is still a pointer every check has to reason about.
_UNUSED_POLARIZATION = (0.0, 1.0, 0.0)


@dataclass(frozen=True, slots=True)
class LegacySensorContext:
    """One radar's sensor description, resolved once per call site.

    Built from the radar facade rather than from a config mapping because the
    pose is what turns a local element offset and a local polarization into a
    world vector, and the pose belongs to the radar.
    """

    num_tx: int
    num_rx: int
    tx_pos: torch.Tensor
    rx_pos: torch.Tensor
    pol_tx: torch.Tensor
    pol_rx: torch.Tensor
    local_axes: torch.Tensor
    plan: SensorWeightPlan
    device: torch.device

    @classmethod
    def from_radar(cls, radar, *, tx_count: int | None = None, rx_count: int | None = None):
        """Resolve the sensor description, optionally for a leading subarray.

        ``tx_count`` / ``rx_count`` exist for ``DirichletSolver.frame``, which
        is a single-transmitter, single-receiver product. Slicing the arrays
        here rather than at the call site keeps the transmitter index the kernel
        receives and the polarization row it looks up in step.
        """

        device = radar.device
        num_tx = int(radar.config.num_tx if tx_count is None else tx_count)
        num_rx = int(radar.config.num_rx if rx_count is None else rx_count)
        polarization = radar.polarization
        if polarization is None:
            pol_tx = torch.tensor(
                [_UNUSED_POLARIZATION] * num_tx, dtype=torch.float32, device=device
            )
            pol_rx = torch.tensor(
                [_UNUSED_POLARIZATION] * num_rx, dtype=torch.float32, device=device
            )
            reflection_flip = True
        else:
            pol_tx = polarization.tx_world[:num_tx].contiguous()
            pol_rx = polarization.rx_world[:num_rx].contiguous()
            reflection_flip = bool(polarization.reflection_flip)

        # ``local_from_world_vectors`` is ``v @ world_from_local``, so the local
        # components are dot products with that matrix's COLUMNS. The kernel
        # takes those columns as its rows; that transpose IS the frame change.
        _, world_from_local = radar._world_from_local_matrix(
            device=device, dtype=torch.float32
        )
        modes = SensorWeightModes(
            spreading=True,
            tx_power=True,
            legacy_real_polarization=polarization is not None,
            reflection_flip=reflection_flip,
        )
        plan = SensorWeightPlan.build(
            AntennaPatternSpec.from_config(radar.antenna_pattern_config),
            modes=modes,
            wavelength_m=float(radar.c0) / float(radar.config.fc),
            tx_amplitude=float(radar.transmit_amplitude),
            c0=float(radar.c0),
            device=device,
        )
        return cls(
            num_tx=num_tx,
            num_rx=num_rx,
            tx_pos=radar.tx_pos[:num_tx].contiguous(),
            rx_pos=radar.rx_pos[:num_rx].contiguous(),
            pol_tx=pol_tx,
            pol_rx=pol_rx,
            local_axes=world_from_local.transpose(0, 1).contiguous(),
            plan=plan,
            device=device,
        )

    @property
    def uses_polarization(self) -> bool:
        return self.plan.modes.legacy_real_polarization


def _expand(values: torch.Tensor, repeats: int) -> torch.Tensor:
    """Tile a per-path tensor across ``repeats`` antenna pairs, row-major."""

    return values.unsqueeze(0).expand(repeats, *values.shape).reshape(
        repeats * values.shape[0], *values.shape[1:]
    ).contiguous()


def _normals_or_zeros(normals: torch.Tensor | None, count: int, device) -> torch.Tensor:
    if normals is not None:
        return normals
    return torch.zeros(count, 3, dtype=torch.float32, device=device)


def _evaluate(
    context: LegacySensorContext,
    *,
    site_in: torch.Tensor,
    site_out: torch.Tensor,
    fixed_length_m: torch.Tensor,
    intensity: torch.Tensor,
    normals: torch.Tensor,
    site_velocity: torch.Tensor,
    tx_index: torch.Tensor,
    rx_index: torch.Tensor,
) -> SensorWeightResult:
    rows = int(intensity.shape[0])
    geometry = SensorWeightGeometry(
        num_tx=context.num_tx,
        num_rx=context.num_rx,
        tx_velocity=torch.zeros(context.num_tx, 3, dtype=torch.float32, device=context.device),
        rx_velocity=torch.zeros(context.num_rx, 3, dtype=torch.float32, device=context.device),
        site_velocity=site_velocity,
        fixed_length_m=fixed_length_m,
        tx_index=tx_index,
        rx_index=rx_index,
        row_kind=torch.full(
            (rows,), ROW_KIND_VIA, dtype=torch.int32, device=context.device
        ),
        normals=normals,
        pol_tx=context.pol_tx,
        pol_rx=context.pol_rx,
        local_axes=context.local_axes,
    )
    return evaluate_sensor_weights(
        tx_pos=context.tx_pos,
        rx_pos=context.rx_pos,
        site_in=site_in,
        site_out=site_out,
        intensity=intensity,
        weight=torch.ones(rows, dtype=torch.complex64, device=context.device),
        geometry=geometry,
        plan=context.plan,
    )


def evaluate_pair_rows(
    context: LegacySensorContext,
    sample,
    *,
    velocities: torch.Tensor | None = None,
) -> SensorWeightResult:
    """Every ``(tx, rx, path)`` row of one scene sample, in that order.

    The row order is exactly the ``(TX, RX, N)`` layout the Torch expression
    produced, so a caller reshapes rather than permutes and the frame cube's
    pair axis keeps its meaning.
    """

    count = int(sample.points.shape[0])
    pairs = context.num_tx * context.num_rx
    device = context.device
    tx_index = (
        torch.arange(context.num_tx, device=device)
        .view(-1, 1, 1)
        .expand(context.num_tx, context.num_rx, count)
        .reshape(-1)
        .contiguous()
    )
    rx_index = (
        torch.arange(context.num_rx, device=device)
        .view(1, -1, 1)
        .expand(context.num_tx, context.num_rx, count)
        .reshape(-1)
        .contiguous()
    )
    if velocities is None:
        velocities = torch.zeros(count, 3, dtype=torch.float32, device=device)
    return _evaluate(
        context,
        site_in=_expand(sample.entry_points, pairs),
        site_out=_expand(sample.points, pairs),
        fixed_length_m=_expand(sample.fixed_path_lengths, pairs),
        intensity=_expand(sample.intensities, pairs),
        normals=_expand(_normals_or_zeros(sample.normals, count, device), pairs),
        site_velocity=_expand(velocities, pairs),
        tx_index=tx_index,
        rx_index=rx_index,
    )


def evaluate_slot_rows(
    context: LegacySensorContext,
    packed,
    *,
    first_slot: int,
) -> SensorWeightResult:
    """Every ``(slot, rx, path)`` row of one padded TDM slot group.

    Slot ``first_slot + s`` transmits from antenna ``(first_slot + s) % num_tx``,
    which is the TDM firing order the frame cube is indexed by. Padded rows carry
    zero intensity, so their weight is exactly zero and the spectrum kernels skip
    them.
    """

    points, entry_points, fixed_path_lengths, intensities, normals = packed
    num_slots, n_max = points.shape[0], points.shape[1]
    num_rx = context.num_rx
    device = context.device
    slot_tx = (torch.arange(num_slots, device=device) + first_slot) % context.num_tx

    tx_index = (
        slot_tx.view(-1, 1, 1).expand(num_slots, num_rx, n_max).reshape(-1).contiguous()
    )
    rx_index = (
        torch.arange(num_rx, device=device)
        .view(1, -1, 1)
        .expand(num_slots, num_rx, n_max)
        .reshape(-1)
        .contiguous()
    )

    def per_row(values: torch.Tensor) -> torch.Tensor:
        tail = values.shape[2:]
        return (
            values.unsqueeze(1)
            .expand(num_slots, num_rx, n_max, *tail)
            .reshape(num_slots * num_rx * n_max, *tail)
            .contiguous()
        )

    rows = num_slots * num_rx * n_max
    if normals is None:
        normals = torch.zeros(num_slots, n_max, 3, dtype=torch.float32, device=device)
    return _evaluate(
        context,
        site_in=per_row(entry_points),
        site_out=per_row(points),
        fixed_length_m=per_row(fixed_path_lengths),
        intensity=per_row(intensities),
        normals=per_row(normals),
        site_velocity=torch.zeros(rows, 3, dtype=torch.float32, device=device),
        tx_index=tx_index,
        rx_index=rx_index,
    )


__all__ = [
    "LegacySensorContext",
    "evaluate_pair_rows",
    "evaluate_slot_rows",
]
