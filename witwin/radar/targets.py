"""What the radar is looking at: where the scatterers are and how they scatter.

Two records, and the restriction is the design rather than an unfinished edge.
:class:`PointTargets` names world positions outright; :class:`StructureTargets`
puts one scatterer at each world anchor Core publishes for a moving structure.
What is deliberately absent is any rule that DERIVES a scatterer from geometry
- a surface sample, a centroid, a bounding-box centre, a visibility-weighted
set. Every one of those is a geometry algorithm, and a geometry algorithm
written in Torch on the production path is what this architecture exists to
keep out. R-ADR-020 records the deferral and names what closing it would need.

Position and strength are one record because they are one statement about the
world. Splitting them, as the two-object form did, made a caller repeat the
radar's own carrier and device back at it just to say how large a target is.

This module owns the target vocabulary. The scattering COEFFICIENT is owned by
:mod:`witwin.radar.scattering` and the site binding by
:mod:`witwin.radar.simulation`; nothing here computes either.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

__all__ = ["Aspect", "PointTargets", "StructureTargets"]


@dataclass(frozen=True, slots=True)
class Aspect:
    """Aspect-dependent scattering, for a target that is not a sphere.

    The response is
    ``amplitude * max(-dot(dir_in, axis), 0)^n * max(dot(dir_out, axis), 0)^n``,
    so ``axis`` is the direction of maximum return and ``exponent`` is how
    sharply the return falls away from it. Attaching one of these to
    :class:`PointTargets` makes the strength depend on the geometry of each
    round trip; without it the cross section is isotropic.
    """

    #: One unit axis per target, world frame, shape ``(S, 3)``.
    axis: Any
    #: Lobe sharpness ``n``, dimensionless. Larger is narrower.
    exponent: float
    #: How long one aspect realisation stays coherent, s.
    coherent_interval: float
    #: Aspect phase drift, rad/s.
    phase_rate: float = 0.0

    def __post_init__(self) -> None:
        if not float(self.exponent) >= 0.0:
            raise ValueError("Aspect.exponent must be non-negative")
        if not float(self.coherent_interval) > 0.0:
            raise ValueError("Aspect.coherent_interval must be positive")


@dataclass(frozen=True, slots=True)
class PointTargets:
    """Scatterers at world positions the caller names, with their cross section.

    ``positions`` is passed through UNTOUCHED when it is already a tensor, so a
    ``requires_grad`` leaf or a forward-AD dual keeps its tape all the way into
    both propagation legs. ``rcs`` may be a 0-dim tensor for the same reason:
    "how large does this target have to be" is the canonical inverse-design
    question, and the radar cross section is the one configuration scalar in
    this package that is genuine scene state.

    The device and the carrier are NOT fields. They are the radar's, and it
    supplies them when the session starts; a target set is a statement about
    the world, not about the instrument looking at it.
    """

    #: World positions, m, shape ``(S, 3)`` or a sequence of triples.
    positions: Any
    #: Radar cross section, m^2. One value for every target: a per-target
    #: vector needs an :class:`Aspect`, because the scalar response broadcasts
    #: one strength across every composed row.
    rcs: Any
    #: Scattering phase, rad.
    phase: float = 0.0
    #: ``trajectory(time)`` returns this same ordered set of material points at
    #: that instant, shape ``(S, 3)`` in metres. Rotation and articulation must
    #: MOVE the points: an angular-velocity label alone cannot. ``None`` is a
    #: target set that does not move.
    trajectory: Callable[[float], Any] | None = None
    #: Stable world IDs, one per target. ``None`` allocates them.
    ids: tuple[int, ...] | None = None
    aspect: Aspect | None = None

    def __post_init__(self) -> None:
        if self.trajectory is not None and not callable(self.trajectory):
            raise TypeError("PointTargets.trajectory must be callable as trajectory(time) -> (S, 3) positions")
        if self.aspect is not None and not isinstance(self.aspect, Aspect):
            raise TypeError(f"PointTargets.aspect must be an Aspect, got {type(self.aspect).__name__}")
        if isinstance(self.rcs, torch.Tensor) and self.rcs.dim() > 0 and self.aspect is None:
            raise ValueError(
                "a per-target rcs vector needs an Aspect: the isotropic response carries ONE strength and "
                "broadcasts it across every composed row, so a vector here would silently take its first entry"
            )

    @property
    def count(self) -> int:
        if isinstance(self.positions, torch.Tensor):
            return int(self.positions.shape[0])
        return len(self.positions)


@dataclass(frozen=True, slots=True)
class StructureTargets:
    """One scatterer at each world anchor Core publishes for a structure.

    The anchor is a Core-owned quantity read as it stands; Radar computes
    nothing from the mesh. A structure with no rigid motion has no such anchor
    and is refused by name, which is the design (R-ADR-020) rather than a gap:
    the alternative is a mesh-sampling rule, and that is a geometry algorithm.
    """

    #: Radar cross section, m^2, shared by every selected structure.
    rcs: Any
    #: Which structures to place a scatterer on. ``None`` takes every structure
    #: the snapshot carries. Selection and ordering are by ascending structure
    #: ID rather than by the scene's tuple order, so the array order is a
    #: function of world identity and survives a reordered scene.
    structure_ids: tuple[int, ...] | None = None
    #: Scattering phase, rad.
    phase: float = 0.0
    #: Stable world IDs, one per selected structure. ``None`` allocates them.
    ids: tuple[int, ...] | None = None


class _PositionTrajectory:
    """Adapt ``trajectory(time) -> positions`` to the site policy's protocol.

    The policy consumes a ``Kinematics``, which pairs positions with
    velocities, and the frame loop reads only the positions: a site velocity is
    never differenced into physics, because the delay rate comes from the
    propagation solve at each observation instant. The velocities are therefore
    published as exact zeros rather than estimated from a finite difference,
    which would be a second, disagreeing owner of the same quantity.
    """

    __slots__ = ("_positions",)

    def __init__(self, positions: Callable[[float], Any]) -> None:
        self._positions = positions

    def at(self, time_s: float):
        from .propagation import Kinematics

        positions = self._positions(float(time_s))
        if not isinstance(positions, torch.Tensor):
            positions = torch.tensor([[float(v) for v in row] for row in positions], dtype=torch.float32)
        return Kinematics(positions_m=positions, velocities_m_per_s=torch.zeros_like(positions))


def as_session_targets(targets: PointTargets | StructureTargets, *, radar) -> tuple[Any, Any]:
    """Split a target record into the site policy and the scatter response.

    The two halves go to two different owners - the binding places the sites,
    the composer multiplies the round trip by the response - and this is the
    one place that knows they came from one caller statement. The carrier and
    the device are taken from ``radar`` here, which is why neither is a field
    of a target record.
    """

    from .scattering import AspectScatterResponse, ScalarRcsResponse, rcs_amplitude
    from .simulation import ScatterSitePolicy

    if isinstance(targets, PointTargets):
        positions = targets.positions
        if not isinstance(positions, torch.Tensor):
            positions = torch.tensor(
                [[float(v) for v in row] for row in positions], dtype=torch.float32, device=radar.device
            )
        policy = ScatterSitePolicy.explicit(
            positions,
            stable_ids=targets.ids,
            trajectory=None if targets.trajectory is None else _PositionTrajectory(targets.trajectory),
        )
    elif isinstance(targets, StructureTargets):
        policy = ScatterSitePolicy.structure_anchor(structure_ids=targets.structure_ids, stable_ids=targets.ids)
    else:
        raise TypeError(
            f"targets must be PointTargets or StructureTargets, got {type(targets).__name__}; where the "
            "scatterers are is a declaration, not a search"
        )

    aspect = getattr(targets, "aspect", None)
    if aspect is None:
        response = ScalarRcsResponse.from_rcs(
            targets.rcs,
            reference_frequency_hz=float(radar.carrier),
            phase_rad=float(targets.phase),
            device=radar.device,
        )
    else:
        amplitude = rcs_amplitude(targets.rcs, radar.wavelength)
        count = targets.count
        if not isinstance(amplitude, torch.Tensor):
            amplitude = torch.full((count,), float(amplitude), dtype=torch.float32, device=radar.device)
        elif amplitude.dim() == 0:
            amplitude = amplitude.expand(count)
        response = AspectScatterResponse(
            axis=_axis_tensor(aspect.axis, count=count, device=radar.device),
            amplitude=amplitude.to(device=radar.device),
            phase_rad=torch.full((count,), float(targets.phase), dtype=torch.float32, device=radar.device),
            exponent=float(aspect.exponent),
            coherent_interval_s=float(aspect.coherent_interval),
            aspect_phase_rate_rad_per_s=float(aspect.phase_rate),
        )
    return policy, response


def _axis_tensor(axis: Any, *, count: int, device) -> torch.Tensor:
    tensor = (
        axis
        if isinstance(axis, torch.Tensor)
        else torch.tensor([[float(v) for v in row] for row in axis], dtype=torch.float32)
    )
    tensor = tensor.to(device=device)
    if tensor.shape != (count, 3):
        raise ValueError(f"Aspect.axis must have shape ({count}, 3) for {count} targets, got {tuple(tensor.shape)}")
    return tensor
