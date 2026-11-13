"""Ground truth: what each target is doing at a frame instant, and what each composed path is.

Two products, both computed from the simulated geometry rather than from the
cube, so a detector or a dataset can be scored against them.

:class:`TargetLabels` holds, per frame and per target, the target's position,
range, closing-positive radial velocity, azimuth and elevation in the radar's
frame, its monostatic cross section and its ID. :class:`GroundTruth` evaluates
them at any instant from the same three sources the session reads - the target
record's site policy, the Core scene's snapshot at that instant and the radar's
pose - and :func:`ground_truth` stacks them over a frame sequence.

:class:`PathSummary` lists one evaluated observation's composed round trips:
which component and site each row is, how many interactions it made, its
round-trip delay, its Doppler shift and its amplitude. :func:`path_summary`
reads it out of a :class:`~witwin.radar.Paths`.

Frames and conventions, stated once:

* The radar frame is the PROCESSING frame of
  :class:`~witwin.radar.processing.PointCloud`: origin at the radar's position
  at that instant, ``x`` to its right, ``y`` along its boresight, ``z`` up.
  ``azimuth = atan2(x, y)`` (positive to the right) and
  ``elevation = atan2(z, hypot(x, y))`` (positive up), rad.
* ``range_m`` is the one-way distance from the radar's position to the target,
  which is what the range axis of a monostatic radar measures; an element's
  offset from the radar's position is not in it.
* ``radial_velocity_mps`` is ``-d(range)/dt``: positive closing, the sign
  :class:`~witwin.radar.processing.RangeDopplerMap` publishes. It is the
  central difference of the range over ``t +- velocity_step_s``, evaluated from
  two more snapshots of the world, which also captures the radar's own motion.
  It is a label, not a physics input: no simulated quantity reads it.
* ``rcs_m2`` is the declared cross section seen monostatically at that
  instant: a scalar strength as declared, an :class:`~witwin.radar.Aspect`
  lobe at the target-to-radar direction, an :class:`~witwin.radar.RcsTable` at
  the monostatic aspect angle. A :class:`~witwin.radar.Fluctuation` is reported
  at its mean. A :class:`~witwin.radar.MeshTargets` structure has no declared
  cross section (the facets' coherent sum IS the simulation) and reports NaN.

Per-target work is a fixed number of batched tensor operations per instant;
nothing loops over targets in Python.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields, replace
from typing import Any

import numpy as np
import torch

from .binding import DEFAULT_SITE_ID_BASE
from .display import compact_repr
from .paths import COMPONENT_NAMES
from .targets import MeshTargets, PointTargets, RcsTable, StructureTargets, as_session_targets

#: The default half-step of the radial-velocity central difference, s.
VELOCITY_STEP_S = 1e-3

#: ``TargetLabels.kind`` per target record.
LABEL_KINDS = ("point", "structure", "mesh")


@dataclass(frozen=True, slots=True, eq=False)
class TargetLabels:
    """Per-frame, per-target ground truth; every tensor is float64 or int64 on the CPU.

    Leading axes are ``[frame, target]``. Targets are in the record's
    declaration order: ``positions`` order for point targets, ascending
    structure ID for structure and mesh targets.
    """

    __repr__ = compact_repr

    #: ``"point"``, ``"structure"`` or ``"mesh"``.
    kind: str
    #: Frame instants, s, ``[frame]``.
    time_s: torch.Tensor
    #: ``[target]``: the site's stable world ID (the ``site_id`` a composed row
    #: names) for point and structure targets, the structure ID for a mesh.
    target_id: torch.Tensor
    #: ``[frame, target, 3]``, m, world frame.
    world_position_m: torch.Tensor
    #: ``[frame, target, 3]``, m, the radar frame ``(right, boresight, up)``.
    position_m: torch.Tensor
    range_m: torch.Tensor
    #: Closing positive, m/s.
    radial_velocity_mps: torch.Tensor
    #: rad, ``atan2(x, y)`` in the radar frame.
    azimuth_rad: torch.Tensor
    #: rad, ``atan2(z, hypot(x, y))`` in the radar frame.
    elevation_rad: torch.Tensor
    #: m^2, monostatic, NaN for a mesh target.
    rcs_m2: torch.Tensor
    #: ``[site]``: every scatter site's stable ID, and the ``target_id`` it belongs to,
    #: so a composed row's ``site_id`` resolves to a target. For a merged mesh
    #: site holding facets of two structures, the first facet's structure.
    site_id: torch.Tensor
    site_target: torch.Tensor
    #: Mesh only (else ``None``): ``[frame, target]`` facet area, m^2, and
    #: ``[frame, target, 2]`` the nearest and farthest facet-centroid range, m.
    area_m2: torch.Tensor | None = None
    range_extent_m: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.kind not in LABEL_KINDS:
            raise ValueError(f"kind must be one of {LABEL_KINDS}, got {self.kind!r}")
        frames, targets = int(self.time_s.shape[0]), int(self.target_id.shape[0])
        for name in ("range_m", "radial_velocity_mps", "azimuth_rad", "elevation_rad", "rcs_m2"):
            if tuple(getattr(self, name).shape) != (frames, targets):
                raise ValueError(f"{name} must be [{frames}, {targets}], got {tuple(getattr(self, name).shape)}")

    @property
    def frame_count(self) -> int:
        return int(self.time_s.shape[0])

    @property
    def target_count(self) -> int:
        return int(self.target_id.shape[0])

    def tensors(self) -> dict[str, torch.Tensor]:
        """Every member as a flat ``{name: tensor}`` mapping (``kind`` as a 0-dim code), the saved form."""

        out: dict[str, torch.Tensor] = {"kind": torch.tensor(LABEL_KINDS.index(self.kind))}
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, torch.Tensor):
                out[field.name] = value
        return out

    @classmethod
    def from_tensors(cls, mapping: dict[str, torch.Tensor]) -> TargetLabels:
        """Rebuild from :meth:`tensors`."""

        values = dict(mapping)
        kind = LABEL_KINDS[int(values.pop("kind"))]
        return cls(kind=kind, **values)

    @classmethod
    def stack(cls, frames: Sequence[TargetLabels]) -> TargetLabels:
        """Concatenate one-frame (or several-frame) labels along the frame axis."""

        if not frames:
            raise ValueError("no labels to stack")
        first = frames[0]
        per_frame = {"target_id", "site_id", "site_target", "kind"}
        values: dict[str, Any] = {}
        for field in fields(cls):
            value = getattr(first, field.name)
            if field.name in per_frame or value is None:
                values[field.name] = value
            else:
                values[field.name] = torch.cat([getattr(frame, field.name) for frame in frames], dim=0)
        return cls(**values)


class GroundTruth:
    """Evaluates :class:`TargetLabels` for one radar, one world and one target record, at any instant.

    ``pose`` is the radar's ego-motion, the same callable the verbs take. Build
    one per sequence and call :meth:`at` per frame: the site policy and a
    mesh's facet model are prepared once. Each call reads the world three
    times (at ``t`` and ``t +- velocity_step_s``).
    """

    def __init__(
        self,
        radar: Any,
        scene: Any,
        targets: PointTargets | StructureTargets | MeshTargets,
        *,
        pose: Callable[[float], Any] | None = None,
        velocity_step_s: float = VELOCITY_STEP_S,
    ) -> None:
        from witwin.core.dynamics import DynamicScene

        if not float(velocity_step_s) > 0.0:
            raise ValueError("velocity_step_s must be positive")
        self.radar = radar
        self.targets = targets
        self.pose = pose
        self.velocity_step_s = float(velocity_step_s)
        self.world = scene if isinstance(scene, DynamicScene) else DynamicScene(scene)
        self.policy, _ = as_session_targets(targets, radar=radar, ad_mode=None)
        self.kind = (
            "mesh"
            if isinstance(targets, MeshTargets)
            else "point"
            if isinstance(targets, PointTargets)
            else "structure"
        )
        self.device = radar.tx_pos.device
        position, world_from_local = radar._world_from_local_matrix(device="cpu", dtype=torch.float64)
        self._origin = position.reshape(3)
        self._basis = world_from_local

    # -- geometry at one instant ---------------------------------------------

    def _radar_frame(self, time_s: float) -> tuple[torch.Tensor, torch.Tensor]:
        """The radar's world position and ``(right, boresight, up)`` world axes at ``time_s``, float64."""

        origin, basis = self._origin, self._basis
        if self.pose is not None:
            rotation, translation = self.pose(float(time_s))
            rotation = torch.as_tensor(rotation, dtype=torch.float64).detach().cpu().reshape(3, 3)
            translation = torch.as_tensor(translation, dtype=torch.float64).detach().cpu().reshape(3)
            basis = rotation @ basis
            origin = origin + translation
        right, up, back = basis[:, 0], basis[:, 1], basis[:, 2]
        return origin, torch.stack((right, -back, up), dim=1)

    def _points(self, time_s: float) -> tuple[torch.Tensor, Any]:
        """World positions of the targets (or, for a mesh, of the facets) at ``time_s``, float64 CPU."""

        snapshot = self.world.at(float(time_s))
        positions, facets = self.policy.resolve_frame(snapshot, device=self.device)
        if facets is None:
            return positions.detach().to(device="cpu", dtype=torch.float64), None
        return facets.corners.detach().to(device="cpu", dtype=torch.float64).reshape(-1, 3, 3), facets

    def at(self, time_s: float) -> TargetLabels:
        """The labels of one instant, as a one-frame :class:`TargetLabels`."""

        step = self.velocity_step_s
        instants = (float(time_s) - step, float(time_s), float(time_s) + step)
        samples = [self._points(instant) for instant in instants]
        frames = [self._radar_frame(instant) for instant in instants]
        if self.kind == "mesh":
            return self._mesh_labels(float(time_s), samples, frames)
        world = torch.stack([points for points, _ in samples])
        local, ranges = _in_radar_frame(world, frames)
        site_count = int(world.shape[1])
        ids = self.policy.stable_ids
        target_id = torch.tensor(
            list(ids) if ids is not None else range(DEFAULT_SITE_ID_BASE, DEFAULT_SITE_ID_BASE + site_count),
            dtype=torch.int64,
        )
        rcs = self._rcs(world[1], frames[1][0], float(time_s))
        return _labels(
            self.kind,
            float(time_s),
            target_id,
            world[1],
            local[1],
            ranges,
            rcs,
            step,
            site_id=target_id,
            site_target=target_id,
        )

    def _rcs(self, world: torch.Tensor, origin: torch.Tensor, time_s: float) -> torch.Tensor:
        """``[target]`` monostatic cross section, m^2, at the target-to-radar direction."""

        targets: Any = self.targets
        count = int(world.shape[0])
        wavelength = float(self.radar.wavelength)
        toward = origin.reshape(1, 3) - world
        toward = toward / torch.linalg.vector_norm(toward, dim=1, keepdim=True)
        rcs = targets.rcs
        if isinstance(rcs, RcsTable):
            # The scatter response interpolates the complex amplitude linearly in theta
            # (R-ADR-045); the cross section is that amplitude's square, not an
            # interpolated square.
            table = torch.as_tensor(
                rcs.amplitude_at(float(self.radar.reference_frequency_hz), wavelength), dtype=torch.complex128
            )
            axis = _host(rcs.axis).reshape(count, 3)
            cosine = (axis * toward).sum(dim=1).clamp(-1.0, 1.0)
            position = torch.arccos(cosine) / math.pi * (int(table.shape[0]) - 1)
            lower = position.floor().clamp(max=int(table.shape[0]) - 2).long()
            weight = position - lower
            amplitude = table[lower] * (1.0 - weight) + table[lower + 1] * weight
            return amplitude.abs() ** 2 * wavelength**2 / (4.0 * math.pi)
        if rcs is not None:
            sigma = _host(rcs).expand(count).clone()
        else:
            amplitude = _host(targets.amplitude).expand(count)
            sigma = amplitude**2 * wavelength**2 / (4.0 * math.pi)
        aspect = getattr(targets, "aspect", None)
        if aspect is not None:
            axis = _host(aspect.axis(time_s) if callable(aspect.axis) else aspect.axis).reshape(count, 3)
            lobe = (axis * toward).sum(dim=1).clamp(min=0.0) ** (2.0 * float(aspect.exponent))
            sigma = sigma * lobe**2
        return sigma

    def _mesh_labels(
        self, time_s: float, samples: list[tuple[torch.Tensor, Any]], frames: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> TargetLabels:
        """Per-structure aggregates of a mesh target: area-weighted facet centroid, area, range extent."""

        model = samples[1][1].model
        structures = torch.tensor(model.structure_ids, dtype=torch.int64)
        bounds = torch.tensor(np.cumsum(model.vertex_counts), dtype=torch.int64)
        facet_structure = torch.searchsorted(bounds, model.faces[:, 0].cpu(), right=True)
        count = int(structures.shape[0])
        centres, areas = [], []
        for corners, _ in samples:
            edge_a, edge_b = corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
            area = 0.5 * torch.linalg.vector_norm(torch.cross(edge_a, edge_b, dim=1), dim=1)
            weighted = torch.zeros((count, 3), dtype=torch.float64).index_add_(
                0, facet_structure, corners.mean(dim=1) * area[:, None]
            )
            total = torch.zeros(count, dtype=torch.float64).index_add_(0, facet_structure, area)
            centres.append(weighted / total[:, None])
            areas.append(total)
        world = torch.stack(centres)
        local, ranges = _in_radar_frame(world, frames)
        centroid_range = torch.linalg.vector_norm(samples[1][0].mean(dim=1) - frames[1][0].reshape(1, 3), dim=1)
        extent = torch.stack(
            (
                torch.full((count,), math.inf, dtype=torch.float64).scatter_reduce(
                    0, facet_structure, centroid_range, reduce="amin"
                ),
                torch.full((count,), -math.inf, dtype=torch.float64).scatter_reduce(
                    0, facet_structure, centroid_range, reduce="amax"
                ),
            ),
            dim=1,
        )
        first_facet = model.site_facets[model.site_offsets[:-1]].cpu()
        site_count = int(model.site_offsets.shape[0]) - 1
        site_id = torch.arange(DEFAULT_SITE_ID_BASE, DEFAULT_SITE_ID_BASE + site_count, dtype=torch.int64)
        labels = _labels(
            "mesh",
            time_s,
            structures,
            world[1],
            local[1],
            ranges,
            torch.full((count,), math.nan, dtype=torch.float64),
            self.velocity_step_s,
            site_id=site_id,
            site_target=structures[facet_structure[first_facet]],
        )
        return replace(labels, area_m2=areas[1].reshape(1, count), range_extent_m=extent.reshape(1, count, 2))


def _host(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", dtype=torch.float64)
    return torch.as_tensor(value, dtype=torch.float64)


def _in_radar_frame(
    world: torch.Tensor, frames: Sequence[tuple[torch.Tensor, torch.Tensor]]
) -> tuple[torch.Tensor, torch.Tensor]:
    """``[instant, target, 3]`` world points into each instant's radar frame, and their ranges."""

    origins = torch.stack([origin for origin, _ in frames])
    axes = torch.stack([basis for _, basis in frames])
    local = torch.einsum("itk,ikj->itj", world - origins[:, None, :], axes)
    return local, torch.linalg.vector_norm(local, dim=2)


def _labels(
    kind: str,
    time_s: float,
    target_id: torch.Tensor,
    world: torch.Tensor,
    local: torch.Tensor,
    ranges: torch.Tensor,
    rcs: torch.Tensor,
    step: float,
    *,
    site_id: torch.Tensor,
    site_target: torch.Tensor,
) -> TargetLabels:
    x, y, z = local[:, 0], local[:, 1], local[:, 2]
    return TargetLabels(
        kind=kind,
        time_s=torch.tensor([time_s], dtype=torch.float64),
        target_id=target_id,
        world_position_m=world[None],
        position_m=local[None],
        range_m=ranges[1][None],
        radial_velocity_mps=(-(ranges[2] - ranges[0]) / (2.0 * step))[None],
        azimuth_rad=torch.atan2(x, y)[None],
        elevation_rad=torch.atan2(z, torch.hypot(x, y))[None],
        rcs_m2=rcs.to(torch.float64)[None],
        site_id=site_id,
        site_target=site_target,
    )


def ground_truth(
    radar: Any,
    scene: Any,
    targets: PointTargets | StructureTargets | MeshTargets,
    *,
    times: Sequence[float],
    pose: Callable[[float], Any] | None = None,
    velocity_step_s: float = VELOCITY_STEP_S,
) -> TargetLabels:
    """:class:`TargetLabels` at every instant of ``times``: the frame instants a verb was given."""

    labeler = GroundTruth(radar, scene, targets, pose=pose, velocity_step_s=velocity_step_s)
    return TargetLabels.stack([labeler.at(float(time_s)) for time_s in times])


# ---------------------------------------------------------------------------
# Path summary
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, eq=False)
class PathSummary:
    """One evaluated observation's composed round trips, one entry per row, on the CPU.

    ``component`` indexes ``component_names`` (Radar's component classes:
    target, environment clutter, direct leakage, multi-interaction);
    ``leg_component`` is ``[row, 2]``, the inbound and outbound legs' Channel
    component IDs as Channel encodes them (``-1`` for a site-less row's absent
    second leg). ``depth`` is the round trip's interaction count, both legs
    summed, the scatter at the site not counted.

    ``delay_s`` is the float64 round-trip delay. ``doppler_hz`` is
    ``-f_ref d(delay)/dt``, positive closing: the delay difference between the
    frame's first and last evaluated observations over their time difference,
    when both belong to the same frozen topology. A frame that evaluated the
    world once is a frame held still - the waveform saw no motion, so its
    Doppler is exactly zero - and a row the two observations do not share (a
    rediscovery between them, or a row invalid in either) is NaN, as is every
    row of an adaptive frame, which retains one observation. ``amplitude`` is
    ``|C|`` of the composed transfer in sqrt(W), ``phase_rad`` its argument in
    Channel's phasor convention.
    """

    __repr__ = compact_repr

    time_s: float
    source_id: torch.Tensor
    site_id: torch.Tensor
    sink_id: torch.Tensor
    #: Sink-major sensor pair, ``rx * num_tx + tx``.
    sensor_pair: torch.Tensor
    component: torch.Tensor
    component_names: tuple[str, ...]
    leg_component: torch.Tensor
    depth: torch.Tensor
    delay_s: torch.Tensor
    doppler_hz: torch.Tensor
    amplitude: torch.Tensor
    phase_rad: torch.Tensor
    valid: torch.Tensor

    @property
    def row_count(self) -> int:
        return int(self.delay_s.shape[0])


def _delays(batch: Any) -> torch.Tensor:
    delay = batch.total_delay_s.detach().to(torch.float64)
    if batch.total_delay_residual_s is not None:
        delay = delay + batch.total_delay_residual_s.detach().to(torch.float64)
    return delay


def _valid(batch: Any) -> torch.Tensor:
    if batch.row_valid is None:
        return torch.ones(int(batch.path_count), dtype=torch.bool, device=batch.total_delay_s.device)
    return batch.row_valid.detach().to(torch.bool)


def path_summary(paths: Any, frame: int = 0, observation: int = -1) -> PathSummary:
    """Summarise observation ``observation`` of frame ``frame`` of ``paths`` (a :class:`~witwin.radar.Paths`)."""

    batch = paths.rows(frame, observation)
    identity = torch.from_numpy(np.asarray(paths.row_identity(frame, observation), dtype=np.int64))
    index = paths.component_index(frame, observation)
    times = paths.observation_times(frame)
    count = int(batch.path_count)
    delay = _delays(batch)
    valid = _valid(batch)
    doppler = torch.full((count,), math.nan, dtype=torch.float64, device=delay.device)
    if len(times) == 1 and paths.kind == "static":
        doppler = torch.zeros_like(doppler)
    elif len(times) > 1 and paths.kind != "adaptive":
        first, last = paths.rows(frame, 0), paths.rows(frame, len(times) - 1)
        if first.topology is last.topology and last.topology is batch.topology and times[-1] != times[0]:
            rate = (_delays(last) - _delays(first)) / float(times[-1] - times[0])
            shared = _valid(first) & _valid(last)
            doppler = torch.where(shared, -float(batch.reference_frequency_hz) * rate, doppler)
    transfer = batch.complex_transfer_ref.detach()
    return PathSummary(
        time_s=float(times[observation]),
        source_id=identity[:, 0],
        site_id=identity[:, 1],
        sink_id=identity[:, 2],
        sensor_pair=batch.sensor_pair_index.detach().to(device="cpu", dtype=torch.int64),
        component=index.class_id.detach().to(device="cpu", dtype=torch.int64),
        component_names=tuple(index.names) if index.names else COMPONENT_NAMES,
        leg_component=identity[:, 3:5],
        depth=identity[:, 5] + identity[:, 6].clamp(min=0),
        delay_s=delay.cpu(),
        doppler_hz=doppler.cpu(),
        amplitude=transfer.abs().to(device="cpu", dtype=torch.float32),
        phase_rad=transfer.angle().to(device="cpu", dtype=torch.float32),
        valid=valid.cpu(),
    )


__all__ = ["GroundTruth", "PathSummary", "TargetLabels", "ground_truth", "path_summary"]
