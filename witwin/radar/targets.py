"""What the radar is looking at: where the scatterers are and how they scatter.

Three records. :class:`PointTargets` names world positions outright;
:class:`StructureTargets` puts one scatterer at each world anchor Core
publishes for a moving structure; :class:`MeshTargets` makes the facets of
Core structures the scatterers, each a physical-optics facet of the structure's
own material (R-ADR-042). The facet extraction is native and lives in
:mod:`witwin.radar.facets`; nothing here computes geometry.

Position and strength are one record because they are one statement about the
world; the carrier and the device are the radar's and are supplied when the
session starts.

This module owns the target vocabulary. The scattering COEFFICIENT is owned by
:mod:`witwin.radar.scattering` and the site binding by
:mod:`witwin.radar.simulation`; nothing here computes either.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any, ClassVar, cast

import torch

from .binding import ScatterSitePolicy
from .policy import require_real
from .propagation import Kinematics
from .scattering import (
    AspectScatterResponse,
    FacetScatterResponse,
    FluctuatingResponse,
    ScalarRcsResponse,
    ScatteringMatrixResponse,
    TableScatterResponse,
    rcs_amplitude,
)

__all__ = ["Aspect", "Fluctuation", "MeshTargets", "PointTargets", "RcsTable", "ScatteringMatrix", "StructureTargets"]


@dataclass(frozen=True, slots=True)
class ScatteringMatrix:
    """How a target scatters each polarization, relative to its strength.

    ``hh``, ``hv``, ``vh`` and ``vv`` (receive polarization first) are complex
    factors on the target's ``sqrt(4 pi rcs) / wavelength`` strength, in the
    site's local backscatter-alignment basis: for a wave propagating into the
    site along ``k``, ``h = (up x k)/|up x k|`` and ``v = h x k``. ``up`` is
    the site axis in world coordinates. Without one a target scatters as the
    identity dyadic, which on a monostatic geometry is :meth:`sphere`.
    """

    hh: complex = 1.0
    hv: complex = 0.0
    vh: complex = 0.0
    vv: complex = 1.0
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)

    def __post_init__(self) -> None:
        for name in ("hh", "hv", "vh", "vv"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, numbers.Complex):
                raise TypeError(f"ScatteringMatrix.{name} must be a complex number, got {type(value).__name__}")
            if not (math.isfinite(complex(value).real) and math.isfinite(complex(value).imag)):
                raise ValueError(f"ScatteringMatrix.{name} must be finite, got {value!r}")
        up = _vector3("ScatteringMatrix.up", self.up)
        if not any(up):
            raise ValueError(f"ScatteringMatrix.up must be a non-zero 3-vector, got {self.up!r}")

    @classmethod
    def sphere(cls, *, up: tuple[float, float, float] = (0.0, 0.0, 1.0)) -> ScatteringMatrix:
        """Odd-bounce scattering (sphere, plate, trihedral): ``diag(1, 1)``."""

        return cls(1.0, 0.0, 0.0, 1.0, up=up)

    @classmethod
    def dihedral(cls, *, up: tuple[float, float, float] = (0.0, 0.0, 1.0)) -> ScatteringMatrix:
        """Even-bounce scattering with the fold along ``h``: ``diag(1, -1)``."""

        return cls(1.0, 0.0, 0.0, -1.0, up=up)

    def values(self) -> tuple[complex, complex, complex, complex]:
        return (complex(self.hh), complex(self.hv), complex(self.vh), complex(self.vv))


@dataclass(frozen=True, slots=True)
class Aspect:
    """Aspect-dependent scattering, for a target that is not a sphere.

    The response is
    ``amplitude * max(-dot(dir_in, axis), 0)^n * max(dot(dir_out, axis), 0)^n``,
    so ``axis`` is the direction of maximum return and ``exponent`` is how
    sharply the return falls away from it. Attaching one of these to
    :class:`PointTargets` makes the strength depend on the geometry of each
    round trip; without it the cross section is isotropic.

    ``axis`` may be a function of time, ``axis(t) -> (S, 3)``: a rotating
    lobe or a blade flash. It is then evaluated at every observation instant
    the session samples, the session counts as moving even if no point does,
    and the lobe the kernel sees at an instant is the axis at that instant.
    The function must return unit vectors; a tensor it returns keeps its
    tape, so a gradient reaches whatever parameterises the rotation.
    """

    #: One unit axis per target, world frame, shape ``(S, 3)``; or a function
    #: of time returning one.
    axis: Any
    #: Lobe sharpness ``n``, dimensionless, at least 1. Larger is narrower.
    exponent: float
    #: How long one aspect realisation stays coherent, s.
    coherent_interval: float
    #: Aspect phase drift, rad/s.
    phase_rate: float = 0.0

    def __post_init__(self) -> None:
        if not require_real("Aspect.exponent", self.exponent) >= 1.0:
            raise ValueError("Aspect.exponent must be at least 1")
        if not require_real("Aspect.coherent_interval", self.coherent_interval) > 0.0:
            raise ValueError("Aspect.coherent_interval must be positive")
        require_real("Aspect.phase_rate", self.phase_rate)
        if not callable(self.axis):
            _require_axis_rows("Aspect.axis", self.axis, count=None)


@dataclass(frozen=True, slots=True)
class RcsTable:
    """A tabulated cross section, ``sigma(theta, f)``, as a target's ``rcs``.

    ``theta`` is the aspect angle between the target's ``axis`` and the
    bisector of the directions from the target to the transmitter and to the
    receiver (the monostatic aspect when the two coincide), sampled uniformly
    on ``[0, pi]`` by the table's last axis. ``rcs`` is ``(A,)`` for one
    frequency or ``(F, A)`` at ``frequencies``; ``phase`` (same shape,
    or a scalar) is the scattering phase. The table is reduced to the radar's
    carrier once, by linear interpolation of the complex amplitude
    ``sqrt(4 pi sigma) / lambda exp(-j phase)`` in frequency, and evaluated per
    composed row by linear interpolation in ``theta`` (R-ADR-045). It is host
    data and takes no derivative; the aspect it is read at does.
    """

    #: One unit axis per target, world frame, shape ``(S, 3)``.
    axis: Any
    #: Cross section, m^2, ``(A,)`` or ``(F, A)``, ``A >= 2`` aspect samples on ``[0, pi]``.
    rcs: Any
    #: The frequencies of the table's rows, Hz, ascending; required with an ``(F, A)`` table.
    frequencies: Any = None
    #: Scattering phase, rad, broadcast against ``rcs``.
    phase: Any = 0.0

    def __post_init__(self) -> None:
        import numpy as np

        _require_axis_rows("RcsTable.axis", self.axis, count=None)
        table = np.asarray(self.rcs, dtype=np.float64)
        if table.ndim not in (1, 2) or table.shape[-1] < 2:
            raise ValueError("RcsTable.rcs must be (A,) or (F, A) with at least two aspect samples")
        if np.any(table < 0.0) or not np.all(np.isfinite(table)):
            raise ValueError("RcsTable.rcs is a cross section in square metres and must be finite and non-negative")
        if (table.ndim == 2) != (self.frequencies is not None):
            raise ValueError("RcsTable.frequencies is required with, and only with, an (F, A) table")
        if table.ndim == 2:
            frequencies = np.asarray(self.frequencies, dtype=np.float64)
            if frequencies.shape != (table.shape[0],) or np.any(np.diff(frequencies) <= 0.0):
                raise ValueError("RcsTable.frequencies must be ascending, one per table row")
        np.broadcast_to(np.asarray(self.phase, dtype=np.float64), table.shape)

    def amplitude_at(self, carrier_hz: float, wavelength_m: float) -> Any:
        """The complex amplitude table at the carrier, ``(A,)`` complex128 on the host."""

        import numpy as np

        table = np.asarray(self.rcs, dtype=np.float64)
        phase = np.broadcast_to(np.asarray(self.phase, dtype=np.float64), table.shape)
        amplitude = np.sqrt(4.0 * math.pi * table) / float(wavelength_m) * np.exp(-1j * phase)
        if table.ndim == 1:
            return amplitude
        frequencies = np.asarray(self.frequencies, dtype=np.float64)
        carrier = float(carrier_hz)
        if not frequencies[0] <= carrier <= frequencies[-1]:
            raise ValueError(
                f"the carrier {carrier} Hz is outside the table's frequencies [{frequencies[0]}, {frequencies[-1]}] "
                "Hz; a tabulated cross section is not extrapolated"
            )
        upper = int(min(cast(int, np.searchsorted(frequencies, carrier, side="right")), len(frequencies) - 1))
        lower = max(upper - 1, 0)
        span = frequencies[upper] - frequencies[lower]
        weight = 0.0 if span == 0.0 else (carrier - frequencies[lower]) / span
        return (1.0 - weight) * amplitude[lower] + weight * amplitude[upper]


@dataclass(frozen=True, slots=True)
class Fluctuation:
    """Swerling fluctuation of every target's cross section (R-ADR-045).

    Each site's power is multiplied by an independent draw ``X`` with
    ``E[X] = 1``: exponential (chi-square with 2 degrees of freedom) for
    Swerling I and II, chi-square with 4 degrees of freedom for III and IV.
    Swerling I and III hold one draw for every observation inside one
    ``coherent_interval`` (a scan) and redraw at each interval boundary; II and
    IV redraw at every observation instant (pulse to pulse). A draw is a pure
    function of ``seed``, the interval's start (or the instant) and the site,
    keyed like the frontend's noise, so every verb sees the same realisation.
    The phase is not fluctuated.
    """

    swerling: int
    #: The decorrelation interval of Swerling I and III, s.
    coherent_interval: float
    seed: int = 0

    def __post_init__(self) -> None:
        if self.swerling not in (1, 2, 3, 4) or type(self.swerling) is not int:
            raise ValueError(f"Fluctuation.swerling must be 1, 2, 3 or 4, got {self.swerling!r}")
        if not require_real("Fluctuation.coherent_interval", self.coherent_interval) > 0.0:
            raise ValueError("Fluctuation.coherent_interval must be positive")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("Fluctuation.seed must be a non-negative int")


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

    #: The fields that may be tensors carrying a derivative, and the AD modes
    #: each supports. A trajectory's returned tensor keeps its tape too.
    LEAVES: ClassVar[dict[str, tuple[str, ...]]] = {
        "positions": ("jvp", "vjp"),
        "rcs": ("jvp", "vjp"),
        "amplitude": ("jvp", "vjp"),
        "phase": ("jvp", "vjp"),
    }

    #: World positions, m, shape ``(S, 3)`` or a sequence of triples.
    positions: Any
    #: Radar cross section, m^2: one value shared by every target, or one per
    #: target in ``positions`` order (a sequence or a 1-dim tensor). Without an
    #: :class:`Aspect` every target is isotropic at its own strength.
    rcs: Any = None
    #: The dimensionless target strength, ``sqrt(4 pi sigma) / lambda``, for a
    #: caller who has it rather than a cross section. Exactly one of this and
    #: ``rcs`` is given: they are the same quantity on two sides of the law
    #: that converts them, and accepting both would leave which one wins
    #: undefined. A test oracle or an optimiser that treats the strength itself
    #: as the leaf wants this one, because going through the cross section
    #: would put the square root in its gradient.
    amplitude: Any = None
    #: Scattering phase, rad: one value, or one per target in ``positions`` order.
    phase: Any = 0.0
    #: ``trajectory(time)`` returns this same ordered set of material points at
    #: that instant, shape ``(S, 3)`` in metres. Rotation and articulation must
    #: MOVE the points: an angular-velocity label alone cannot. ``None`` is a
    #: target set that does not move.
    trajectory: Callable[[float], Any] | None = None
    #: Stable world IDs, one per target. ``None`` allocates them.
    ids: tuple[int, ...] | None = None
    aspect: Aspect | None = None
    #: Polarimetric scattering. ``None`` scatters as the identity dyadic.
    scattering: ScatteringMatrix | None = None
    #: Swerling fluctuation of the cross section; ``None`` holds it fixed.
    fluctuation: Fluctuation | None = None

    def __post_init__(self) -> None:
        _require_positions("PointTargets.positions", self.positions)
        _require_one_strength("PointTargets", rcs=self.rcs, amplitude=self.amplitude)
        if self.scattering is not None and not isinstance(self.scattering, ScatteringMatrix):
            raise TypeError(f"PointTargets.scattering must be a ScatteringMatrix, got {type(self.scattering).__name__}")
        if self.scattering is not None and self.aspect is not None:
            raise NotImplementedError(
                "PointTargets.scattering and PointTargets.aspect cannot be combined: the aspect lobe is a "
                "per-row scalar and the scattering matrix a per-site 2x2, and no response composes both"
            )
        if isinstance(self.rcs, RcsTable) and self.scattering is not None:
            raise NotImplementedError("an RcsTable is a scalar aspect law; it does not compose with a ScatteringMatrix")
        if isinstance(self.rcs, RcsTable) and self.aspect is not None:
            raise ValueError("an RcsTable is its own aspect law; give a table or an Aspect, not both")
        if self.fluctuation is not None and not isinstance(self.fluctuation, Fluctuation):
            raise TypeError("PointTargets.fluctuation must be a Fluctuation")
        if self.trajectory is not None and not callable(self.trajectory):
            raise TypeError("PointTargets.trajectory must be callable as trajectory(time) -> (S, 3) positions")
        if self.aspect is not None and not isinstance(self.aspect, Aspect):
            raise TypeError(f"PointTargets.aspect must be an Aspect, got {type(self.aspect).__name__}")
        for name in ("rcs", "amplitude", "phase"):
            if not isinstance(getattr(self, name), RcsTable):
                _require_per_target_values(f"PointTargets.{name}", getattr(self, name), non_negative=name == "rcs")
                _require_per_target_length(f"PointTargets.{name}", getattr(self, name), self.count)
        _require_ids("PointTargets.ids", self.ids, count=self.count)
        if isinstance(self.rcs, RcsTable):
            _require_axis_rows("RcsTable.axis", self.rcs.axis, count=self.count)
        if self.aspect is not None and not callable(self.aspect.axis):
            _require_axis_rows("Aspect.axis", self.aspect.axis, count=self.count)

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

    #: Radar cross section, m^2: one value shared by every selected structure,
    #: or one per selected structure in ascending structure-ID order.
    rcs: Any = None
    #: The dimensionless strength instead of a cross section; exactly one of
    #: the two, for the reason :class:`PointTargets` gives.
    amplitude: Any = None
    #: Which structures to place a scatterer on. ``None`` takes every structure
    #: the snapshot carries. Selection and ordering are by ascending structure
    #: ID rather than by the scene's tuple order, so the array order is a
    #: function of world identity and survives a reordered scene.
    structure_ids: tuple[int, ...] | None = None
    #: Scattering phase, rad: one value, or one per selected structure in
    #: ascending structure-ID order.
    phase: Any = 0.0
    #: Stable world IDs, one per selected structure. ``None`` allocates them.
    ids: tuple[int, ...] | None = None
    #: Swerling fluctuation of the cross section; ``None`` holds it fixed.
    fluctuation: Fluctuation | None = None

    def __post_init__(self) -> None:
        _require_one_strength("StructureTargets", rcs=self.rcs, amplitude=self.amplitude)
        if isinstance(self.rcs, RcsTable):
            raise ValueError("an RcsTable needs a per-target axis; use PointTargets for a tabulated cross section")
        if self.fluctuation is not None and not isinstance(self.fluctuation, Fluctuation):
            raise TypeError("StructureTargets.fluctuation must be a Fluctuation")
        _require_ids("StructureTargets.structure_ids", self.structure_ids, count=None)
        count = None if self.structure_ids is None else len(self.structure_ids)
        for name in ("rcs", "amplitude", "phase"):
            _require_per_target_values(f"StructureTargets.{name}", getattr(self, name), non_negative=name == "rcs")
            _require_per_target_length(f"StructureTargets.{name}", getattr(self, name), count)
        _require_ids("StructureTargets.ids", self.ids, count=count)


@dataclass(frozen=True, slots=True)
class MeshTargets:
    """The facets of Core structures, each scattering as a physical-optics facet.

    The selected structures stay in the scene, so the legs see them: a facet
    hidden by the rest of the world contributes nothing, and the target's own
    multi-bounce (a corner reflector's) comes from the scene's reflections, so
    it needs ``reflections`` at least the number of bounces before the last.
    What a facet returns is fixed by its area, its orientation to the two legs
    and the material its structure carries (R-ADR-042); there is no cross
    section to declare. A facet scatters from its normal side only, the side
    the right-handed corner order points to.

    Every facet is a site by default. ``cluster_size`` merges the facets
    whose centroids share one cubic cell of that edge into one site, and
    ``max_sites`` picks the finest such cell that keeps the site count within
    the budget; a merged site keeps every facet's own term and phase, valid
    while the cell is well inside the far field of the radar
    (``k D^2 / (8 R)`` small for a cell of extent ``D`` at range ``R``) and
    small against the surface's curvature, and is occluded as one point, at
    the member facet nearest the cell's mean.
    """

    #: Which structures scatter. ``None`` takes every structure in the scene.
    structure_ids: tuple[int, ...] | None = None
    #: Cubic cell edge, m, of one merged site; ``None`` keeps one facet per site.
    cluster_size: float | None = None
    #: The largest site count to allow; ``None`` declares no budget.
    max_sites: int | None = None
    #: Seed of the frozen per-facet phase of the diffuse (rough-surface) term.
    seed: int = 0
    #: Stable world IDs are allocated per site; a mesh target names none.

    def __post_init__(self) -> None:
        _require_ids("MeshTargets.structure_ids", self.structure_ids, count=None)
        if self.cluster_size is not None and self.max_sites is not None:
            raise ValueError("MeshTargets takes cluster_size or max_sites, not both")
        if self.cluster_size is not None and not require_real("MeshTargets.cluster_size", self.cluster_size) > 0.0:
            raise ValueError("MeshTargets.cluster_size must be positive")
        if self.max_sites is not None and (type(self.max_sites) is not int or self.max_sites < 1):
            raise ValueError("MeshTargets.max_sites must be a positive int")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("MeshTargets.seed must be a non-negative int")


def _require_one_strength(owner: str, *, rcs: Any, amplitude: Any) -> None:
    """Exactly one of the two ways to say how strongly a target scatters."""

    if (rcs is None) == (amplitude is None):
        raise ValueError(
            f"{owner} needs exactly one of rcs and amplitude. They are the same quantity on two sides of "
            "amplitude = sqrt(4 pi rcs) / wavelength, so giving both leaves it undefined which one the "
            "response is built from, and giving neither says nothing about how strongly the target scatters."
        )


def _as_host(value: Any) -> Any:
    """A NumPy array as nested lists, so host sequences are checked one way; anything else unchanged."""

    if not isinstance(value, torch.Tensor) and hasattr(value, "__array__") and hasattr(value, "tolist"):
        return value.tolist()
    return value


def _vector3(name: str, value: Any) -> tuple[float, float, float]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().reshape(-1).tolist()
    value = _as_host(value)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 3:
        raise ValueError(f"{name} must be a 3-vector, got {value!r}")
    x, y, z = (require_real(f"{name}[{axis}]", value[axis]) for axis in range(3))
    return x, y, z


def _require_positions(name: str, value: Any) -> None:
    """``(S, 3)`` world positions, at least one: a floating tensor, or a sequence of finite triples.

    A tensor is checked for its shape and dtype only. Reading its values would
    be a device synchronization at construction, and a tensor is how a caller
    hands over a leaf whose values the session reads anyway.
    """

    value = _as_host(value)
    if isinstance(value, torch.Tensor):
        if value.dim() != 2 or int(value.shape[1]) != 3 or int(value.shape[0]) < 1:
            raise ValueError(f"{name} must have shape (S, 3) with at least one target, got {tuple(value.shape)}")
        if not value.is_floating_point():
            raise TypeError(f"{name} must be a floating tensor, got {value.dtype}")
        return
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(
            f"{name} must be a (S, 3) tensor or a sequence of 3-element positions, got {type(value).__name__}"
        )
    if not value:
        raise ValueError(f"{name} must name at least one position")
    for index, row in enumerate(value):
        _vector3(f"{name}[{index}]", row)


def _require_axis_rows(name: str, value: Any, *, count: int | None) -> None:
    """One axis per target, shape ``(S, 3)``; ``count`` is ``S`` when the owner knows it."""

    value = _as_host(value)
    if isinstance(value, torch.Tensor):
        shape = tuple(int(size) for size in value.shape)
    else:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise TypeError(f"{name} must be a (S, 3) tensor or a sequence of 3-vectors, got {type(value).__name__}")
        for index, row in enumerate(value):
            _vector3(f"{name}[{index}]", row)
        shape = (len(value), 3)
    if len(shape) != 2 or shape[1] != 3 or (count is not None and shape[0] != count):
        expected = "(S, 3)" if count is None else f"({count}, 3) for {count} targets"
        raise ValueError(f"{name} must have shape {expected}, got {shape}")


def _require_per_target_values(name: str, value: Any, *, non_negative: bool) -> None:
    """A strength or phase: ``None``, a finite real, a sequence of them, or a 0/1-dim tensor.

    Host values are checked here; a cross section is a power and is also
    refused below zero. A tensor's values are not read (no device
    synchronization at construction): the square-root law refuses a negative
    cross section when the session builds the response.
    """

    value = _as_host(value)
    if value is None or isinstance(value, torch.Tensor):
        return
    entries = value if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else (value,)
    for index, entry in enumerate(entries):
        label = name if entries is not value else f"{name}[{index}]"
        parsed = require_real(label, entry)
        if non_negative and parsed < 0.0:
            raise ValueError(f"{label} is a cross section in square metres and must be non-negative, got {parsed}")


def _require_ids(name: str, value: Any, *, count: int | None) -> None:
    """Distinct non-negative int IDs, ``count`` of them when the owner knows how many it needs."""

    if value is None:
        return
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a tuple of ints, got {type(value).__name__}")
    for entry in value:
        if isinstance(entry, bool) or not isinstance(entry, numbers.Integral) or entry < 0:
            raise ValueError(f"{name} must hold non-negative ints, got {entry!r}")
    if len(set(value)) != len(value):
        raise ValueError(f"{name} must not repeat an ID, got {tuple(value)}")
    if count is not None and len(value) != count:
        raise ValueError(f"{name} names {len(value)} IDs but the record declares {count} targets")


def _per_target_length(value: Any) -> int | None:
    """How many targets a strength or phase names; ``None`` for one shared value."""

    if value is None or isinstance(value, (int, float)):
        return None
    if isinstance(value, torch.Tensor):
        if value.dim() > 1:
            raise ValueError(f"a per-target value must be 0-dim or 1-dim, got shape {tuple(value.shape)}")
        return None if value.dim() == 0 else int(value.shape[0])
    return len(value)


def _require_per_target_length(name: str, value: Any, count: int | None) -> None:
    length = _per_target_length(value)
    if length is not None and count is not None and length != count:
        raise ValueError(f"{name} names {length} targets but the record declares {count}")


def _per_site_tensor(value: Any, *, order: torch.Tensor | None, device: Any) -> torch.Tensor:
    """A shared value as a 0-dim tensor, or a per-target vector in the join's site order.

    The join orders sites by ascending stable ID. Caller IDs need not be
    ascending, so a per-target vector is permuted by ``order`` - an
    ``index_select``, which keeps a caller's tape - rather than trusted to
    already be in join order.
    """

    if isinstance(value, torch.Tensor):
        tensor = value.to(device=device)
    elif isinstance(value, (int, float)):
        tensor = torch.tensor(float(value), dtype=torch.float32, device=device)
    else:
        tensor = torch.tensor([float(entry) for entry in value], dtype=torch.float32, device=device)
    if tensor.dim() == 1 and order is not None:
        tensor = tensor.index_select(0, order)
    return tensor


def _join_order(ids: tuple[int, ...] | None, *, device: Any) -> torch.Tensor | None:
    """The permutation from declaration order to ascending stable ID; ``None`` when they agree."""

    if ids is None or list(ids) == sorted(ids):
        return None
    ranked = sorted(range(len(ids)), key=lambda index: ids[index])
    return torch.tensor(ranked, dtype=torch.int64, device=device)


class _PositionTrajectory:
    """Adapt ``trajectory(time) -> positions`` to the site policy's protocol.

    The policy consumes a ``Kinematics`` of positions. A site velocity is never
    differenced into physics: the delay rate comes from the propagation solve
    at each observation instant.
    """

    __slots__ = ("_positions",)

    def __init__(self, positions: Callable[[float], Any]) -> None:
        self._positions = positions

    def at(self, time_s: float) -> Kinematics:
        positions = self._positions(float(time_s))
        if not isinstance(positions, torch.Tensor):
            positions = torch.tensor([[float(v) for v in row] for row in positions], dtype=torch.float32)
        return Kinematics(positions_m=positions)


def as_session_targets(targets: PointTargets | StructureTargets | MeshTargets, *, radar: Any) -> tuple[Any, Any]:
    """Split a target record into the site policy and the scatter response.

    The two halves go to two different owners - the binding places the sites,
    the composer multiplies the round trip by the response - and this is the
    one place that knows they came from one caller statement. The carrier and
    the device are taken from ``radar`` here, which is why neither is a field
    of a target record.
    """

    if isinstance(targets, MeshTargets):
        from .facets import FacetSites

        facets = FacetSites(
            structure_ids=targets.structure_ids,
            carrier_hz=float(radar.reference_frequency_hz),
            cluster_size_m=targets.cluster_size,
            max_sites=targets.max_sites,
        )
        response: Any = FacetScatterResponse(
            wavenumber_rad_per_m=2.0 * math.pi / float(radar.wavelength), seed=targets.seed
        )
        return ScatterSitePolicy.from_facets(facets), response
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
            f"targets must be PointTargets, StructureTargets or MeshTargets, got {type(targets).__name__}; where the "
            "scatterers are is a declaration, not a search"
        )

    order = _join_order(targets.ids, device=radar.device)
    if isinstance(targets.rcs, RcsTable):
        table = targets.rcs.amplitude_at(float(radar.reference_frequency_hz), radar.wavelength)
        # An RcsTable is refused on every record but PointTargets.
        axis = _axis_tensor(targets.rcs.axis, count=cast(PointTargets, targets).count, device=radar.device)
        response = TableScatterResponse(
            axis=(axis if order is None else axis.index_select(0, order)).contiguous(),
            table_re=torch.tensor(table.real, dtype=torch.float32, device=radar.device),
            table_im=torch.tensor(table.imag, dtype=torch.float32, device=radar.device),
        )
        return policy, _fluctuated(response, targets.fluctuation)
    phase = _per_site_tensor(targets.phase, order=order, device=radar.device)
    if isinstance(targets.rcs, torch.Tensor):
        # The tensor law keeps a cross-section leaf's tape.
        sigma = _per_site_tensor(targets.rcs, order=order, device=radar.device)
        amplitude = _per_site_tensor(rcs_amplitude(sigma, radar.wavelength), order=None, device=radar.device)
    elif targets.rcs is not None:
        # The host law refuses a negative cross section by name.
        amplitude = _per_site_tensor(rcs_amplitude(targets.rcs, radar.wavelength), order=order, device=radar.device)
    else:
        amplitude = _per_site_tensor(targets.amplitude, order=order, device=radar.device)
    aspect = targets.aspect if isinstance(targets, PointTargets) else None
    scattering = targets.scattering if isinstance(targets, PointTargets) else None
    if scattering is not None:
        if amplitude.dim() > 0 or phase.dim() > 0:
            raise ValueError("a scattering matrix takes one strength and one phase shared by every target")
        factors = torch.tensor(scattering.values(), dtype=torch.complex64, device=radar.device)
        rotation = torch.exp(-1j * phase.to(torch.complex64))
        response = ScatteringMatrixResponse(
            matrix=(factors * rotation * amplitude).to(torch.complex64),
            up=torch.tensor(scattering.up, dtype=torch.float32, device=radar.device),
        )
    elif aspect is None:
        response = ScalarRcsResponse(amplitude=amplitude, phase_rad=phase)
    else:
        count = cast(PointTargets, targets).count
        timed = callable(aspect.axis)
        axis = _axis_tensor(aspect.axis(0.0) if timed else aspect.axis, count=count, device=radar.device)
        response = AspectScatterResponse(
            axis=axis if order is None else axis.index_select(0, order),
            amplitude=amplitude.expand(count).contiguous(),
            phase_rad=phase.expand(count).contiguous(),
            exponent=float(aspect.exponent),
            coherent_interval_s=float(aspect.coherent_interval),
            aspect_phase_rate_rad_per_s=float(aspect.phase_rate),
        )
        if timed:
            if targets.fluctuation is not None:
                raise NotImplementedError(
                    "a Fluctuation on a target whose Aspect axis is a function of time is not composed: the "
                    "time-resolved aspect response is rebuilt per replay group and the draw is bound per observation"
                )
            return policy, TimedAspectResponse(
                base=response, axis=aspect.axis, count=count, device=radar.device, order=order
            )
    return policy, _fluctuated(response, targets.fluctuation)


def _fluctuated(response: Any, fluctuation: Fluctuation | None) -> Any:
    if fluctuation is None:
        return response
    return FluctuatingResponse(
        base=response,
        degrees=2 if fluctuation.swerling in (1, 2) else 4,
        per_observation=fluctuation.swerling in (2, 4),
        coherent_interval_s=float(fluctuation.coherent_interval),
        seed=fluctuation.seed,
    )


@dataclass(frozen=True, slots=True, eq=False)
class TimedAspectResponse:
    """An aspect response whose axis is a function of time.

    Not a scatter response itself: the session resolves it at the instants of
    each replay group with :meth:`at`, which returns an ordinary
    :class:`~witwin.radar.scattering.AspectScatterResponse` with one site per
    (instant, target), slot major. The slot-batched join indexes those sites
    per slot, so every row's lobe is the axis at that row's own instant and
    the native aspect kernel does every per-row evaluation.
    """

    base: AspectScatterResponse
    axis: Callable[[float], Any]
    count: int
    device: Any
    #: The permutation from declaration order to the join's site order, or ``None``.
    order: torch.Tensor | None = None

    @property
    def is_geometry_dependent(self) -> bool:
        """Always: the lobe it resolves to is evaluated per composed row."""

        return True

    def at(self, times: Sequence[float]) -> AspectScatterResponse:
        """The response at ``times``: ``len(times) * count`` sites, slot major."""

        axes = [_axis_tensor(self.axis(float(time_s)), count=self.count, device=self.device) for time_s in times]
        if self.order is not None:
            axes = [axis.index_select(0, self.order) for axis in axes]
        joined = torch.cat(axes)
        slots = len(times)
        return replace(
            self.base,
            axis=joined.contiguous(),
            amplitude=self.base.amplitude.repeat(slots),
            phase_rad=self.base.phase_rad.repeat(slots),
        )


def _axis_tensor(axis: Any, *, count: int, device: Any) -> torch.Tensor:
    tensor = (
        axis
        if isinstance(axis, torch.Tensor)
        else torch.tensor([[float(v) for v in row] for row in axis], dtype=torch.float32)
    )
    tensor = tensor.to(device=device)
    if tensor.shape != (count, 3):
        raise ValueError(f"Aspect.axis must have shape ({count}, 3) for {count} targets, got {tuple(tensor.shape)}")
    return tensor
