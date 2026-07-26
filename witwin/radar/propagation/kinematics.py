"""Core kinematics to forward-AD velocity duals: the delay-rate seam.

The Doppler chain already exists end to end. What did not exist is its INPUT.
``witwin.core.dynamics.RigidMotion`` has declared ``velocity`` and
``angular_velocity`` since it was written and, until this module, had exactly
zero consumers anywhere in the platform: Channel's compiler reads only
``rotation`` and ``translation``, and every Doppler test built its tangent by
hand from a literal. So the half of the delay-rate story that says "from
endpoint and target kinematics" had no implementation at all.

This module is that implementation, and it is the SINGLE Radar owner of the
conversion. It answers one question - given a Core snapshot, what is the
velocity of each endpoint and each scatter site - and hands the answer to
``torch.autograd.forward_ad`` as a tangent. It owns no physics: every number
downstream of it is produced by a native Channel kernel or by the native join.

Three sources of velocity, and they are not interchangeable:

* an ENDPOINT rides ``EndpointState.rigid_motion.velocity`` directly;
* a rigid-body SITE rides ``v(p) = v_cm + omega x (p - c)``, which is the only
  route by which ``angular_velocity`` can enter any computation and therefore
  the only reason a rotating target or a rotor blade can produce a Doppler
  spread at all;
* a DEFORMING structure has no analytic derivative in Core (``DeformationState``
  carries vertices or offsets and no velocity descriptor), so it must supply one
  through :class:`DeformationVelocity`. Differencing two snapshots is a finite
  difference: allowed inside a test as an independent oracle, never in
  production.

**The dead-tangent trap, and why this module owns the dual level.** A forward-AD
tangent dies silently. ``make_dual(p, v)`` followed by a rebuild of ``p`` from
Python values - a list comprehension, a fresh ``torch.tensor``, a detach, a
``.contiguous()`` copy taken outside the level - produces a perfectly ordinary
position tensor with no tangent, and the whole chain then publishes
``delay_rate = 0``. Zero is exactly what a correct stationary scene publishes,
so the failure is indistinguishable from success by inspection. Two consequences
are baked into this module:

* :func:`two_way_duals` covers the transmitter, site and receiver tensors in ONE
  ``dual_level``, because an inbound leg's rate needs ``v_tx`` and ``v_site``
  while the outbound leg's needs ``v_site`` and ``v_rx``. Opening one level per
  tensor cannot express a round trip whose two ends both move.
* slot replication is :func:`replicate_slots`, an ``index_select`` ON the dual
  tensor. Rebuilding a slot stack from values is the same trap wearing a
  different hat.

Every test that exercises this seam must carry a non-zero RADIAL component for
the same reason. A purely transverse fixture cannot tell a dead tangent from a
correct zero.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Iterator, Protocol, Sequence, runtime_checkable

import torch
import torch.autograd.forward_ad as forward_ad


@runtime_checkable
class DeformationVelocity(Protocol):
    """An analytic ``d(vertices)/dt`` for a deforming structure.

    Core's ``DeformationState`` states WHERE the vertices are and never how fast
    they are moving, so a deforming mesh has no time derivative anywhere in
    Core. Production finite differences are forbidden, which leaves exactly one
    supported route: the descriptor that produced the deformation also states
    its rate in closed form.

    ``velocity_at`` returns a ``(V, 3)`` world-frame velocity in metres per
    second, one row per authored vertex, in authored vertex order. A caller that
    tracks a subset of vertices as scatter sites index-selects from it; see
    :func:`deformation_kinematics`.

    This is a Radar-side contract because the gap it fills is a Core gap that
    Radar is not permitted to patch. If Core later grows a velocity descriptor
    on ``DeformationState``, an adapter implementing this protocol over it is
    the whole migration.
    """

    def velocity_at(self, time_s: float) -> torch.Tensor:
        ...


def _require_positions(name: str, value: object) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}")
    if value.ndim != 2 or int(value.shape[1]) != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {tuple(value.shape)}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} must use torch.float32, got {value.dtype}")
    return value


def _vector3(
    name: str,
    value: object,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """One world-frame 3-vector, or zeros when the caller declared nothing.

    ``None`` means "this quantity is not part of the motion", which is a
    statement about the world and not a missing argument, so it resolves to a
    real zero vector rather than raising. A Core ``RigidMotion`` with no
    ``angular_velocity`` describes a purely translating body.
    """

    if value is None:
        return torch.zeros(3, dtype=dtype, device=device)
    tensor = (
        value
        if isinstance(value, torch.Tensor)
        else torch.tensor(value, dtype=dtype, device=device)
    )
    if tuple(tensor.shape) != (3,):
        raise ValueError(f"{name} must have shape (3,), got {tuple(tensor.shape)}")
    return tensor.to(device=device, dtype=dtype)


@dataclass(frozen=True, slots=True, eq=False)
class Kinematics:
    """One ordered endpoint set's positions and velocities.

    ``positions_m`` is the tensor that becomes a forward-AD PRIMAL and
    ``velocities_m_per_s`` is the tensor that becomes its TANGENT. They are held
    together in one object because handing them to ``make_dual`` in the wrong
    order, or pairing a position tensor with a velocity tensor built for a
    different endpoint order, produces a completely plausible Doppler spectrum
    for a world that does not exist.

    Both are ``float32``, both are contiguous, and both live on the same device.
    That is the Channel endpoint contract restated at the point where the
    tensors are BUILT rather than at the point where they are rejected.
    """

    positions_m: torch.Tensor
    velocities_m_per_s: torch.Tensor

    def __post_init__(self) -> None:
        positions = _require_positions("positions_m", self.positions_m)
        velocities = _require_positions(
            "velocities_m_per_s", self.velocities_m_per_s
        )
        if tuple(positions.shape) != tuple(velocities.shape):
            raise ValueError(
                f"positions_m has shape {tuple(positions.shape)} and "
                f"velocities_m_per_s has shape {tuple(velocities.shape)}; a "
                "tangent must name the same endpoints in the same order as the "
                "primal it rides"
            )
        if positions.device != velocities.device:
            raise ValueError(
                f"positions_m is on {positions.device} and velocities_m_per_s "
                f"is on {velocities.device}"
            )
        if not positions.is_contiguous() or not velocities.is_contiguous():
            raise ValueError("Kinematics tensors must be contiguous")

    @property
    def count(self) -> int:
        return int(self.positions_m.shape[0])

    @property
    def device(self) -> torch.device:
        return self.positions_m.device


def rigid_site_velocities(
    positions_m: torch.Tensor,
    *,
    velocity=None,
    angular_velocity=None,
    centre_m=None,
) -> torch.Tensor:
    """``v(p) = v_cm + omega x (p - c)`` for world points riding a rigid body.

    This is the whole of rigid-body micro-Doppler. Two points of the same body
    at opposite ends of a rotor get equal and opposite projected velocities, and
    the resulting Doppler pair spread is the blade-flash signature; a model that
    only carried ``v_cm`` would give both points the same shift and could not
    produce the signature at all.

    ``centre_m`` is the instantaneous rotation centre and it is explicit on
    purpose - see :func:`rotation_centre_m` for the value that matches the way
    Channel composes a snapshot's rigid motion. Guessing it wrong shows up as a
    velocity offset that is uniform over the body, which looks exactly like a
    platform velocity.
    """

    positions = _require_positions("positions_m", positions_m)
    device, dtype = positions.device, positions.dtype
    linear = _vector3("velocity", velocity, device, dtype)
    omega = _vector3("angular_velocity", angular_velocity, device, dtype)
    centre = _vector3("centre_m", centre_m, device, dtype)
    offset = positions - centre
    spin = torch.linalg.cross(omega.expand(offset.shape), offset)
    return (linear + spin).contiguous()


def rotation_centre_m(rigid_motion, *, device=None, dtype=torch.float32):
    """The point Channel's snapshot composition actually rotates a structure about.

    Channel builds a moved structure as ``vertices @ R.T + t``
    (``scene/compiler.py``): the authored WORLD vertices are rotated about the
    world origin and the translation is applied afterwards. Differentiating that
    gives ``dp/dt = omega x (p - t) + t_dot``, so the instantaneous rotation
    centre is the CURRENT TRANSLATION, not the authored pose position.

    That distinction is worth a named function because the authored pose centre
    is the intuitive answer and it is wrong: using it puts a spurious
    ``omega x (t - pose)`` offset on every site of the body, uniform over the
    body and therefore indistinguishable from a platform velocity.
    """

    translation = None if rigid_motion is None else rigid_motion.translation
    resolved = (
        torch.device(device)
        if device is not None
        else (
            translation.device
            if isinstance(translation, torch.Tensor)
            else torch.device("cpu")
        )
    )
    return _vector3("translation", translation, resolved, dtype)


def structure_site_kinematics(
    state, positions_m: torch.Tensor
) -> Kinematics:
    """Rigid-body kinematics of world points riding one Core ``StructureState``.

    ``positions_m`` are the world positions of the tracked points - scatter
    sites, in the Radar architecture, where a site is an ENDPOINT of the two
    legs rather than a piece of geometry. The structure supplies the motion; the
    caller supplies which points of it it cares about.

    A structure with no ``rigid_motion`` is stationary and gets exact zeros,
    which is a complete answer rather than a missing one.
    """

    positions = _require_positions("positions_m", positions_m)
    motion = getattr(state, "rigid_motion", None)
    return Kinematics(
        positions_m=positions.contiguous(),
        velocities_m_per_s=rigid_site_velocities(
            positions,
            velocity=None if motion is None else motion.velocity,
            angular_velocity=None if motion is None else motion.angular_velocity,
            centre_m=rotation_centre_m(motion, device=positions.device),
        ),
    )


def deformation_kinematics(
    positions_m: torch.Tensor,
    descriptor: DeformationVelocity,
    time_s: float,
    *,
    vertex_index: torch.Tensor | None = None,
) -> Kinematics:
    """Kinematics of tracked vertices of a deforming structure.

    ``vertex_index`` selects which authored vertices the tracked points are; it
    is an ``index_select`` rather than a slice because a limb's sites are not
    contiguous in authored vertex order. Omitting it means every vertex is
    tracked, in authored order.

    The descriptor's rate is ANALYTIC. This function will not difference two
    snapshots to obtain one, in production or anywhere else: a finite difference
    would silently answer with a truncation error that grows with the step and
    would put a forbidden approximation inside a production hot path.
    """

    positions = _require_positions("positions_m", positions_m)
    velocities = descriptor.velocity_at(time_s)
    if not isinstance(velocities, torch.Tensor):
        raise TypeError(
            f"{type(descriptor).__name__}.velocity_at must return a torch."
            f"Tensor, got {type(velocities).__name__}"
        )
    if vertex_index is not None:
        if vertex_index.dtype != torch.int64:
            raise TypeError(
                f"vertex_index must use torch.int64, got {vertex_index.dtype}"
            )
        velocities = velocities.index_select(
            0, vertex_index.to(device=velocities.device)
        )
    velocities = velocities.to(
        device=positions.device, dtype=torch.float32
    ).contiguous()
    return Kinematics(
        positions_m=positions.contiguous(), velocities_m_per_s=velocities
    )


def endpoint_kinematics(
    snapshot_or_states,
    antenna_ids: Sequence[int] | None = None,
    *,
    device: str | torch.device = "cuda",
) -> Kinematics:
    """``(positions, velocities)`` for an ordered set of Core endpoint states.

    ``snapshot_or_states`` is a ``SceneSnapshot`` or any sequence of
    ``EndpointState``. ``antenna_ids`` declares the ENDPOINT BATCH ORDER: it is
    the order the positions and the velocities are both built in, and it is the
    order the Channel leg rows will name. Omitting it keeps the snapshot's own
    declaration order, which is fine for a single caller and wrong the moment
    two callers disagree, so a batch that will be joined by identity should
    always declare it.

    Position resolution follows Core's own composition: the authored antenna
    position plus the snapshot's additional world-frame ``translation``. An
    endpoint's ``rotation`` is orientation and does not move its phase centre;
    an array element that ORBITS a rotating platform is a rigid-body site and
    belongs in :func:`rigid_site_velocities` with the platform centre, not here.

    Velocity is ``rigid_motion.velocity`` verbatim - the first consumer that
    field has ever had. An endpoint with no motion contributes exact zeros.
    """

    states = getattr(snapshot_or_states, "endpoints", snapshot_or_states)
    ordered = list(states)
    if antenna_ids is not None:
        by_id = {int(state.antenna.antenna_id): state for state in ordered}
        missing = [
            stable_id for stable_id in antenna_ids if int(stable_id) not in by_id
        ]
        if missing:
            raise KeyError(
                f"the snapshot declares no endpoint for antenna IDs {missing}; "
                f"it carries {sorted(by_id)}"
            )
        ordered = [by_id[int(stable_id)] for stable_id in antenna_ids]
    if not ordered:
        raise ValueError("endpoint_kinematics requires at least one endpoint")

    resolved = torch.device(device)
    positions = []
    velocities = []
    for state in ordered:
        motion = getattr(state, "rigid_motion", None)
        position = state.antenna.position.to(
            device=resolved, dtype=torch.float32
        )
        if tuple(position.shape) != (3,):
            raise ValueError(
                f"antenna position must have shape (3,), got "
                f"{tuple(position.shape)}"
            )
        translation = _vector3(
            "translation",
            None if motion is None else motion.translation,
            resolved,
            torch.float32,
        )
        positions.append(position + translation)
        velocities.append(
            _vector3(
                "velocity",
                None if motion is None else motion.velocity,
                resolved,
                torch.float32,
            )
        )
    return Kinematics(
        positions_m=torch.stack(positions).contiguous(),
        velocities_m_per_s=torch.stack(velocities).contiguous(),
    )


def replicate_slots(positions: torch.Tensor, slot_count: int) -> torch.Tensor:
    """Repeat one endpoint set once per slot, SLOT MAJOR, on a live tensor.

    ``index_select`` rather than a rebuild, and that is the whole point: a
    forward-AD dual survives a differentiable op and dies the moment its values
    are read back into Python. The slot-major layout - slot ``t`` owning rows
    ``[t * n, (t + 1) * n)`` - is the Channel consumer's ``slot_pair_layout``,
    which is why the arithmetic is stated here once instead of in every caller.

    A slot stack of a MOVING endpoint set is a different expression: build the
    per-slot displacement as a differentiable function of the base positions and
    keep the tangent flowing through it. This function is the STATIC case, where
    every slot sees the same positions and the frozen-mode kernel owns the
    slow-time carrier.
    """

    if type(slot_count) is not int or slot_count < 1:
        raise ValueError(f"slot_count must be a positive int, got {slot_count!r}")
    if slot_count == 1:
        return positions
    rows = int(positions.shape[0])
    index = torch.arange(rows, device=positions.device).repeat(slot_count)
    return positions.index_select(0, index)


@dataclass(frozen=True, slots=True, eq=False)
class TwoWayDuals:
    """The three position tensors of a radar round trip, dualised together.

    Valid only inside the :func:`two_way_duals` block that produced them. A
    forward tangent belongs to its level; reading one after the level exits is
    undefined, which is why the adapter clones the delay tangent inside the
    level and why this object is yielded rather than returned.
    """

    transmitters: torch.Tensor | None
    sites: torch.Tensor
    receivers: torch.Tensor | None
    slot_count: int


@contextlib.contextmanager
def two_way_duals(
    *,
    sites: Kinematics,
    transmitters: Kinematics | None = None,
    receivers: Kinematics | None = None,
    slot_count: int = 1,
) -> Iterator[TwoWayDuals]:
    """Dualise transmitter, site and receiver positions in ONE level.

    One level, not three. The inbound leg's delay rate is
    ``d|p_site - p_tx|/dt`` and needs both tangents live at once; the outbound
    leg's needs the site and the receiver. Nesting a level per tensor would make
    each leg see one moving end and one frozen end and would publish a round
    trip whose two halves describe different worlds.

    ``sites`` is required because a radar round trip without a target is not a
    round trip. The endpoints are optional: a static front end is the common
    case and passing ``None`` says so, rather than passing a zero tangent that
    a reader has to check.

    ``slot_count`` replicates each tensor slot major INSIDE the level, so the
    batched replay of a whole TDM frame carries the same live tangent in every
    slot.
    """

    if not isinstance(sites, Kinematics):
        raise TypeError(
            f"sites must be a Kinematics, got {type(sites).__name__}"
        )
    for name, value in (("transmitters", transmitters), ("receivers", receivers)):
        if value is not None and not isinstance(value, Kinematics):
            raise TypeError(
                f"{name} must be a Kinematics or None, got {type(value).__name__}"
            )
    if type(slot_count) is not int or slot_count < 1:
        raise ValueError(f"slot_count must be a positive int, got {slot_count!r}")

    def dual(value: Kinematics | None) -> torch.Tensor | None:
        if value is None:
            return None
        return replicate_slots(
            forward_ad.make_dual(value.positions_m, value.velocities_m_per_s),
            slot_count,
        )

    with forward_ad.dual_level():
        yield TwoWayDuals(
            transmitters=dual(transmitters),
            sites=dual(sites),
            receivers=dual(receivers),
            slot_count=slot_count,
        )


__all__ = [
    "DeformationVelocity",
    "Kinematics",
    "TwoWayDuals",
    "deformation_kinematics",
    "endpoint_kinematics",
    "replicate_slots",
    "rigid_site_velocities",
    "rotation_centre_m",
    "structure_site_kinematics",
    "two_way_duals",
]
