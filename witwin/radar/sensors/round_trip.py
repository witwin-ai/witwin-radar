"""Apply this radar's antenna pattern to composed round-trip rows.

Until Phase 11 the ``sensor_weight`` family had exactly one importer,
``sensors/legacy_paths.py``, which belongs to the Dirichlet route that this
phase deletes. The new pipeline applied NO antenna pattern at all: a composed
weight came out of the two-way join and went straight into synthesis, and
``SynthesisPathBatch.from_radar_paths`` asserted three provenance booleans
without any of them being able to say whether the array's pattern had been
counted. This module is the production owner that closes both gaps.

**The physics stays where it already was.** Nothing here evaluates a direction,
an angle, an interpolation or a gain. The whole weighting is the existing
``sensor_weight`` CUDA family, reached through
:func:`witwin.radar.sensors.weights.evaluate_sensor_weights`; what this module
adds is the row packing that family needs and the typed result it publishes.
There is no second `torch.autograd.Function` here, so the AD tape ledger gains
no owner - the forward, backward and jvp companions of the existing family are
what carry the derivative.

**All three mode flags are OFF, and that is the single-count rule.** A composed
weight is Channel-sourced: it already carries ``wavelength / (4 pi d)`` per leg,
``sqrt(P_tx)`` from the source endpoint's ``powers_w``, and the endpoint
polarization projection. Turning any of them on here would apply it a second
time. What is left of the kernel's ``scale`` is therefore

    scale = sqrt(max(intensity, 0)) * sqrt(max(G_t * G_r, 0))

with ``intensity = 1``, which is exactly the AMPLITUDE gain of the transmit and
receive patterns and nothing else.

**An isotropic pattern is bit-for-bit a no-op.** With
:data:`ISOTROPIC_PATTERN` both lookups return exactly ``1.0`` (the interpolation
is ``y0 + w * (y1 - y0)`` with ``y0 == y1 == 1``), so ``scale`` is exactly
``1.0f`` and the kernel publishes ``weight * 1.0f``, which is the identity for
every finite float. That is what lets this stage be introduced without moving a
single existing number, and
``tests/test_phase11_antenna_pattern_route.py`` asserts it with
``torch.equal`` rather than a tolerance.

**The stage is opt-in.** ``Radar.simulate(..., antenna_pattern=...)`` defaults to
``None``, which builds no stage and launches no kernel. The default is NOT
``radar.system_config.sensors.pattern``: that spec defaults to a half-wave
dipole, so adopting it silently would attenuate every existing result by a
number nobody asked for. A caller who wants the configured pattern passes it.

**The row tables are frozen once per topology epoch.** Which transmitter and
which receiver a composed row belongs to is a function of its sensor-pair rank
under :data:`witwin.radar.synthesis.assembly.PAIR_RANK_LAYOUT`, and which site
it visits is its frozen response slot. Both are decided when the join is frozen,
so :meth:`RoundTripPatternStage.freeze` computes them once and
:meth:`RoundTripPatternStage.apply` only gathers positions and launches.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..paths.contracts import RadarPathBatch
from .contracts import (
    PATTERN_KIND_SEPARABLE,
    SPEED_OF_LIGHT_M_PER_S,
    AntennaPatternSpec,
)
from .weights import (
    ROW_KIND_VIA,
    SensorWeightGeometry,
    SensorWeightModes,
    SensorWeightPlan,
    evaluate_sensor_weights,
)


#: The pattern that changes nothing, published as data so the no-op claim is a
#: value a test can pass rather than a sentence.
#:
#: Both axes span the whole range ``atan2`` can produce, so no direction ever
#: falls outside the support and takes the zero-outside branch, and both values
#: are ``1.0`` at each knot, so the interpolation returns exactly ``1.0``. It is
#: an ISOTROPIC pattern in the only sense this family has: unit power gain in
#: every direction.
ISOTROPIC_PATTERN = AntennaPatternSpec(
    kind=PATTERN_KIND_SEPARABLE,
    x_angles_deg=(-180.0, 180.0),
    y_angles_deg=(-180.0, 180.0),
    x_values=(1.0, 1.0),
    y_values=(1.0, 1.0),
)

#: The polarization vectors the ABI takes but the kernel never reads here.
#:
#: ``legacy_real_polarization`` is 0 for every Channel-sourced weight, so these
#: rows are dead arguments. They are a tensor rather than an optional for the
#: reason ``legacy_paths`` gives: a null pointer that is never dereferenced is
#: still a pointer every ABI check has to reason about.
_UNUSED_POLARIZATION = (0.0, 1.0, 0.0)

#: The mode set every Channel-sourced composed weight uses. Written once, as
#: data, because the three booleans ARE the single-count rule and a call site
#: that could choose them would eventually choose the convenient ones.
CHANNEL_SOURCED_MODES = SensorWeightModes(
    spreading=False,
    tx_power=False,
    legacy_real_polarization=False,
    reflection_flip=False,
)


def _pattern_plan(
    pattern: AntennaPatternSpec,
    *,
    reference_frequency_hz: float,
    device: torch.device,
) -> SensorWeightPlan:
    if not isinstance(pattern, AntennaPatternSpec):
        raise TypeError(
            "antenna_pattern must be a witwin.radar.sensors.AntennaPatternSpec, "
            f"got {type(pattern).__name__}; pass "
            "radar.system_config.sensors.pattern for the configured one, or "
            "witwin.radar.sensors.ISOTROPIC_PATTERN for none"
        )
    return SensorWeightPlan.build(
        pattern,
        modes=CHANNEL_SOURCED_MODES,
        wavelength_m=SPEED_OF_LIGHT_M_PER_S / float(reference_frequency_hz),
        tx_amplitude=1.0,
        c0=SPEED_OF_LIGHT_M_PER_S,
        device=device,
    )


def _site_rank_to_array_index(
    site_ids: tuple[int, ...], *, device: torch.device
) -> torch.Tensor:
    """Map a composer response slot back to a row of the site position tensor.

    ``TwoWayComposer.freeze`` sorts the declared site IDs, so its
    ``response_slot`` is a rank in ASCENDING ID order while
    ``RadarWorldBinding.site_positions_m`` is in the order the binding published.
    The two coincide for the default allocator and diverge the moment a caller
    declares its own stable IDs, which is exactly the case where getting it
    wrong would look like a physics bug: every row would take its pattern angle
    from another target.

    Built on the host from the binding's own host tuple, once per epoch, so no
    device tensor is read back to get it.
    """

    listed = [int(value) for value in site_ids]
    if len(set(listed)) != len(listed):
        raise ValueError(f"site_ids must not repeat a stable ID, got {listed}")
    order = sorted(range(len(listed)), key=lambda index: listed[index])
    return torch.tensor(order, dtype=torch.int64, device=device)


@dataclass(frozen=True, slots=True, eq=False)
class RoundTripPatternStage:
    """One frozen topology's antenna-pattern application.

    Every tensor here is a CONSTANT of the frozen join: which transmitter and
    receiver each composed row belongs to, which site it visits, and the two
    zero-filled descriptions the family's ABI takes for quantities this route
    does not model (velocities, a fixed leg length, a surface normal). They are
    allocated once, at :meth:`freeze`, so a frame costs one gather and one
    launch.

    The velocities are zero and that is a statement rather than a placeholder:
    the delay and the delay rate this family computes are DISCARDED here.
    Channel owns the round-trip delay and the two-way join owns its rate, and
    recomputing either from the array geometry would put a second owner on a
    number the batch already carries. Only the weight is consumed.
    """

    num_tx: int
    num_rx: int
    row_count: int
    site_count: int
    tx_index: torch.Tensor
    rx_index: torch.Tensor
    site_slot: torch.Tensor
    row_kind: torch.Tensor
    zero_rows: torch.Tensor
    zero_vectors: torch.Tensor
    unit_intensity: torch.Tensor
    tx_velocity: torch.Tensor
    rx_velocity: torch.Tensor
    pol_tx: torch.Tensor
    pol_rx: torch.Tensor
    local_axes: torch.Tensor
    plan: SensorWeightPlan

    @classmethod
    def freeze(
        cls,
        radar,
        composer,
        *,
        site_ids,
        pattern: AntennaPatternSpec,
    ) -> "RoundTripPatternStage":
        """Build the constant tables for one frozen :class:`TwoWayComposer`.

        ``site_ids`` is the binding's host tuple, in the order its site position
        tensor is laid out. Passing it rather than reading the composer's
        ``topology.site_id`` back to the host is deliberate: the host tuple is
        already there, and the device column is not.
        """

        array = radar.system_config.sensors.array
        num_tx = int(array.num_tx)
        num_rx = int(array.num_rx)
        pair_index = composer.sensor_pair_index
        device = pair_index.device
        if composer.sensor_pair_count != num_tx * num_rx:
            raise ValueError(
                f"this join spans {composer.sensor_pair_count} sensor pairs but "
                f"the array is {num_tx} x {num_rx}; the pattern stage looks a "
                "transmitter and a receiver up by pair rank and the two must be "
                "the same front end"
            )
        site_rank_to_index = _site_rank_to_array_index(
            tuple(site_ids), device=device
        )
        if int(site_rank_to_index.shape[0]) != composer.site_count:
            raise ValueError(
                f"the binding declares {int(site_rank_to_index.shape[0])} sites "
                f"but this join was frozen against {composer.site_count}"
            )
        rows = int(composer.path_count)
        # PAIR_RANK_LAYOUT is sink major - pair = rx_rank * num_tx + tx_rank -
        # so the transmitter is the REMAINDER and the receiver is the quotient.
        # Getting these two the wrong way round steers every pattern lookup at
        # the wrong element and still produces a plausible cube.
        tx_index = torch.remainder(pair_index, num_tx).contiguous()
        rx_index = torch.div(pair_index, num_tx, rounding_mode="floor").contiguous()
        zero_rows = torch.zeros(rows, dtype=torch.float32, device=device)
        return cls(
            num_tx=num_tx,
            num_rx=num_rx,
            row_count=rows,
            site_count=int(composer.site_count),
            tx_index=tx_index,
            rx_index=rx_index,
            site_slot=site_rank_to_index.index_select(
                0, composer.response_slot
            ).contiguous(),
            row_kind=torch.full(
                (rows,), ROW_KIND_VIA, dtype=torch.int32, device=device
            ),
            zero_rows=zero_rows,
            zero_vectors=torch.zeros(rows, 3, dtype=torch.float32, device=device),
            unit_intensity=torch.ones(rows, dtype=torch.float32, device=device),
            tx_velocity=torch.zeros(num_tx, 3, dtype=torch.float32, device=device),
            rx_velocity=torch.zeros(num_rx, 3, dtype=torch.float32, device=device),
            pol_tx=torch.tensor(
                [_UNUSED_POLARIZATION] * num_tx, dtype=torch.float32, device=device
            ),
            pol_rx=torch.tensor(
                [_UNUSED_POLARIZATION] * num_rx, dtype=torch.float32, device=device
            ),
            # ``local_from_world_vectors`` is ``v @ world_from_local``, so the
            # local components are dot products with that matrix's COLUMNS. The
            # kernel takes those columns as its rows; the transpose IS the frame
            # change, and it is the same one legacy_paths performs.
            local_axes=radar._world_from_local_matrix(
                device=device, dtype=torch.float32
            )[1].transpose(0, 1).contiguous(),
            plan=_pattern_plan(
                pattern,
                reference_frequency_hz=array.reference_frequency_hz,
                device=device,
            ),
        )

    def _geometry(self) -> SensorWeightGeometry:
        return SensorWeightGeometry(
            num_tx=self.num_tx,
            num_rx=self.num_rx,
            tx_velocity=self.tx_velocity,
            rx_velocity=self.rx_velocity,
            site_velocity=self.zero_vectors,
            fixed_length_m=self.zero_rows,
            tx_index=self.tx_index,
            rx_index=self.rx_index,
            row_kind=self.row_kind,
            normals=self.zero_vectors,
            pol_tx=self.pol_tx,
            pol_rx=self.pol_rx,
            local_axes=self.local_axes,
        )

    def apply(
        self,
        paths: RadarPathBatch,
        *,
        tx_pos: torch.Tensor,
        rx_pos: torch.Tensor,
        site_positions_m: torch.Tensor,
    ) -> RadarPathBatch:
        """Publish ``paths`` with the transmit and receive pattern gains applied.

        The three position tensors are the binding's own objects and are passed
        through by reference, so a ``requires_grad`` leaf or a forward-AD dual on
        the radar's elements or on a site reaches the native companions.
        ``site_in`` and ``site_out`` are the SAME gathered tensor: a site is one
        point, the transmit pattern reads the direction to it and the receive
        pattern reads the direction from it, and autograd accumulates both
        gradients into the one leaf. Building the gather twice would halve that
        gradient and zero half of a tangent.

        Everything except the weight passes through untouched - the same
        objects, not copies - so row identity, row order, storage aliasing,
        dtype, device and the delay's gradient state all survive.
        """

        if not isinstance(paths, RadarPathBatch):
            raise TypeError(
                "the antenna-pattern stage consumes a RadarPathBatch, got "
                f"{type(paths).__name__}"
            )
        if paths.join_mode != "multipath":
            raise NotImplementedError(
                f"the antenna-pattern stage is frozen against a two-way join and "
                f"these rows declare join_mode {paths.join_mode!r}. A direct row "
                "has no scatter site, so its transmit and receive directions are "
                "the other endpoint rather than a site, and applying this stage's "
                "site-based row kind to it would look up the pattern along a "
                "direction the row does not have. A direct-leakage pattern is a "
                "separate capability with its own row kind"
            )
        if paths.weight_includes_antenna_pattern:
            raise ValueError(
                "these rows already record weight_includes_antenna_pattern; "
                "applying the array pattern twice squares its gain and is "
                "invisible in any magnitude plot, so it is refused here rather "
                "than counted"
            )
        if paths.path_count != self.row_count:
            raise ValueError(
                f"these rows carry {paths.path_count} paths but this stage was "
                f"frozen against {self.row_count}; the batch does not belong to "
                "this frozen topology"
            )
        site = site_positions_m.index_select(0, self.site_slot)
        geometry = self._geometry()
        weight = evaluate_sensor_weights(
            tx_pos=tx_pos,
            rx_pos=rx_pos,
            site_in=site,
            site_out=site,
            intensity=self.unit_intensity,
            weight=paths.complex_transfer_ref,
            geometry=geometry,
            plan=self.plan,
        ).weight
        return RadarPathBatch(
            sensor_pair_count=paths.sensor_pair_count,
            path_count=paths.path_count,
            sensor_pair_index=paths.sensor_pair_index,
            pair_offsets=paths.pair_offsets,
            total_delay_s=paths.total_delay_s,
            delay_rate=paths.delay_rate,
            complex_transfer_ref=weight,
            reference_frequency_hz=paths.reference_frequency_hz,
            row_valid=paths.row_valid,
            topology=paths.topology,
            join_mode=paths.join_mode,
            frequency_response=self._apply_band(
                paths, geometry, tx_pos, rx_pos, site
            ),
            frequency_offsets_hz=paths.frequency_offsets_hz,
            weight_includes_antenna_pattern=True,
        )

    def _apply_band(
        self,
        paths: RadarPathBatch,
        geometry: SensorWeightGeometry,
        tx_pos: torch.Tensor,
        rx_pos: torch.Tensor,
        site: torch.Tensor,
    ) -> torch.Tensor | None:
        """The same real gain, applied to every column of a composed band.

        The frequency axis is a PYTHON LOOP over the existing ``[K]`` primitive
        rather than a strided ``[K, F]`` kernel, which is the boundary
        ``TwoWayComposer._compose_band`` already draws for the same reason:
        widening a native family means widening its primal, its jvp and its vjp
        together, and that needs a measured reason first.

        The pattern is applied to the band at all - rather than only to the
        reference column - because this family's gain has no frequency axis. A
        band whose reference column carried the pattern and whose columns did not
        would be two different antennas in one batch.
        """

        if paths.frequency_response is None:
            return None
        columns = [
            evaluate_sensor_weights(
                tx_pos=tx_pos,
                rx_pos=rx_pos,
                site_in=site,
                site_out=site,
                intensity=self.unit_intensity,
                weight=paths.frequency_response[:, index],
                geometry=geometry,
                plan=self.plan,
            ).weight
            for index in range(int(paths.frequency_response.shape[1]))
        ]
        return torch.stack(columns, dim=1)


__all__ = [
    "CHANNEL_SOURCED_MODES",
    "ISOTROPIC_PATTERN",
    "RoundTripPatternStage",
]
