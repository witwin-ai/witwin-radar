"""The frame-to-frame detection contract. An interface, not a tracker.

Nothing in this repository had any notion of a detection that persists across
frames: there is no track, no association, no Kalman filter, and Phase 8 does
not add one. What it adds is the CONTRACT a tracker plugs into, because that is
the thing the simulator owes an external tracker and the thing a tracker cannot
supply for itself.

Three pieces:

* :class:`DetectionFrame` - one frame's detections plus the world time they were
  taken at and the frame index they came from. Time in SECONDS, not a frame
  count, because a tracker's motion model integrates seconds and a frame rate
  that changes mid-run would otherwise be invisible.
* :class:`TrackHandoff` - accumulates frames and hands them to an external
  associator. It stores, it does not decide.
* :func:`nearest_neighbour_associator` - ONE reference associator, constant
  velocity plus nearest neighbour, so that the contract has a worked example and
  the acceptance test has something to run. It is labelled reference and it is
  not a recommendation.

**This whole module is explicitly NON-DIFFERENTIABLE, and it enforces it.**
Phase-9 item 4 already names CFAR, peak selection and tracking as the
non-differentiable stages. A detection is the output of a threshold comparison
and an ``argwhere``; a gradient through an association decision is a gradient
through a discrete choice that does not have one. A derivative-carrying tensor
is REFUSED at the boundary rather than silently detached: a silent detach is how
a caller ends up with a zero gradient and a plausible number, which is worse
than an error.

Two Phase-9 corrections to that enforcement, both of which mattered:

* it checked ``requires_grad`` only, so a FORWARD DUAL walked straight through
  with a live tangent. It now goes through
  :func:`witwin.radar.policy.refuse_derivative`, which checks both modes,
  so this module and the wall speak with one voice and one wording;
* it fired LATE. A :class:`DetectionFrame` is built from a
  :class:`~witwin.radar.processing.detection.PointCloud` that already exists,
  so by the time this refusal ran the frame had been computed in full. The
  point-cloud stage now refuses at ITS entry, which makes this check
  unreachable in the normal flow. It is kept anyway: an unreachable guard on
  the second door is the right shape for a wall, and a caller who hands this
  class a hand-built tensor still meets it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from ..policy import refuse_derivative
from .detection import POINT_CLOUD_COLUMNS, PointCloud

#: Why a detection frame has no derivative.
_ASSOCIATION_REASON = (
    "a detection is the output of a threshold comparison and an argwhere, and "
    "an association is a discrete assignment between two frames; neither has a "
    "derivative, and the frame this class publishes is non-differentiable in "
    "every field."
)


@dataclass(frozen=True, slots=True, eq=False)
class DetectionFrame:
    """One frame of detections, timestamped, ready to hand to a tracker.

    The fields are exactly what a constant-velocity tracker needs and nothing
    more: where, how fast along the line of sight, how strong, and when.
    """

    time_s: float
    xyz: torch.Tensor
    velocity_mps: torch.Tensor
    energy: torch.Tensor
    frame_index: int

    def __post_init__(self) -> None:
        refuse_derivative(
            "witwin.radar.processing.tracking.DetectionFrame",
            _ASSOCIATION_REASON,
            xyz=self.xyz,
            velocity_mps=self.velocity_mps,
            energy=self.energy,
        )
        if self.xyz.dim() != 2 or int(self.xyz.shape[1]) != 3:
            raise ValueError(f"xyz must be [N, 3]; got {tuple(self.xyz.shape)}")
        count = int(self.xyz.shape[0])
        for name in ("velocity_mps", "energy"):
            value = getattr(self, name)
            if value.dim() != 1 or int(value.shape[0]) != count:
                raise ValueError(f"{name} must be [{count}] to match xyz; got {tuple(value.shape)}")
        if type(self.frame_index) is not int or self.frame_index < 0:
            raise ValueError(f"frame_index must be a non-negative int, got {self.frame_index!r}")

    def __len__(self) -> int:
        return int(self.xyz.shape[0])

    @classmethod
    def from_point_cloud(cls, cloud: PointCloud, *, time_s: float, frame_index: int) -> DetectionFrame:
        """The bridge from the point-cloud stage. No values are recomputed."""

        if not isinstance(cloud, PointCloud):
            raise TypeError(f"from_point_cloud takes a PointCloud, got {type(cloud).__name__}")
        return cls(
            time_s=float(time_s),
            xyz=cloud.xyz,
            velocity_mps=cloud.velocity_mps,
            energy=cloud.energy,
            frame_index=int(frame_index),
        )

    def as_fixed_size(self, size: int, *, generator: torch.Generator | None = None) -> torch.Tensor:
        """``[size, 6]`` in :data:`POINT_CLOUD_COLUMNS` order.

        The fixed-shape batch a learned consumer wants, replacing ``reg_data``'s
        ``numpy`` / ``np.random`` path. Three cases, all on the input device and
        all in torch:

        * no detections: exact zeros, so a consumer sees an empty frame rather
          than a random one;
        * fewer than ``size``: every detection is placed once at a random slot
          and the remaining slots are filled by sampling detections with
          replacement, which is what ``reg_data`` did;
        * more than ``size``: ``size`` detections are sampled without
          replacement.

        ``generator`` is an explicit argument. ``reg_data`` drew from the global
        ``numpy`` random state, so two runs of the same simulation produced
        different batches and nothing said so.
        """

        if type(size) is not int or size < 1:
            raise ValueError(f"size must be a positive int, got {size!r}")
        device = self.xyz.device
        columns = len(POINT_CLOUD_COLUMNS)
        count = len(self)
        batch = torch.zeros((size, columns), dtype=torch.float64, device=device)
        if count == 0:
            return batch
        data = torch.stack(
            (
                self.xyz[:, 0],
                self.xyz[:, 1],
                self.xyz[:, 2],
                self.velocity_mps,
                self.energy,
                self.xyz.square().sum(dim=1).sqrt(),
            ),
            dim=1,
        ).to(torch.float64)
        if count < size:
            slots = torch.randperm(size, generator=generator, device=device)
            batch[slots[:count]] = data
            duplicates = torch.randint(count, (size - count,), generator=generator, device=device)
            batch[slots[count:]] = data.index_select(0, duplicates)
            return batch
        chosen = torch.randperm(count, generator=generator, device=device)[:size]
        return data.index_select(0, chosen)


#: An associator takes the previous frame, the next frame, and the elapsed time,
#: and returns ``[N_next]`` int64 track ids, ``-1`` for "no continuation".
Associator = Callable[[DetectionFrame | None, DetectionFrame, float], torch.Tensor]


@dataclass(slots=True, eq=False)
class TrackHandoff:
    """Accumulates :class:`DetectionFrame`s and runs an external associator.

    This class stores and sequences. It contains no motion model, no gate, and
    no track lifetime policy, because those are exactly the decisions a tracker
    exists to make and a simulator has no business making on its behalf.

    ``associator`` is any callable matching :data:`Associator`. The default is
    :func:`nearest_neighbour_associator`, which is a REFERENCE, chosen so the
    contract has a runnable example rather than because it is a good tracker.
    """

    associator: Associator | None = None
    frames: list[DetectionFrame] = field(default_factory=list)
    assignments: list[torch.Tensor] = field(default_factory=list)
    _next_track_id: int = 0

    def push(self, frame: DetectionFrame) -> torch.Tensor:
        """Append a frame and return its ``[N]`` int64 track assignment."""

        if not isinstance(frame, DetectionFrame):
            raise TypeError(f"push takes a DetectionFrame, got {type(frame).__name__}")
        if self.frames and frame.time_s <= self.frames[-1].time_s:
            raise ValueError(
                f"frame {frame.frame_index} is stamped {frame.time_s} s, which is "
                f"not after the previous frame's {self.frames[-1].time_s} s; a "
                "handoff is an ordered stream"
            )
        associate = self.associator or nearest_neighbour_associator
        previous = self.frames[-1] if self.frames else None
        elapsed = 0.0 if previous is None else frame.time_s - previous.time_s
        continued = associate(previous, frame, elapsed)
        if not isinstance(continued, torch.Tensor) or continued.dtype != torch.int64:
            raise TypeError(
                "an associator returns an int64 tensor of previous-frame indices, "
                f"with -1 for a new track; got {type(continued).__name__}"
            )
        if int(continued.shape[0]) != len(frame):
            raise ValueError(
                f"the associator returned {int(continued.shape[0])} assignments for {len(frame)} detections"
            )
        assignment = torch.full((len(frame),), -1, dtype=torch.int64, device=continued.device)
        previous_ids = (
            self.assignments[-1] if self.assignments else torch.zeros((0,), dtype=torch.int64, device=continued.device)
        )
        for index in range(len(frame)):
            source = int(continued[index])
            if source >= 0 and source < int(previous_ids.shape[0]):
                assignment[index] = previous_ids[source]
            else:
                assignment[index] = self._next_track_id
                self._next_track_id += 1
        self.frames.append(frame)
        self.assignments.append(assignment)
        return assignment

    @property
    def track_count(self) -> int:
        """How many distinct tracks have been opened."""

        return self._next_track_id

    def track(self, track_id: int) -> list[tuple[float, torch.Tensor]]:
        """``[(time_s, xyz)]`` for one track, in frame order."""

        history = []
        for frame, assignment in zip(self.frames, self.assignments, strict=True):
            hits = torch.argwhere(assignment == int(track_id)).reshape(-1)
            for row in hits.tolist():
                history.append((frame.time_s, frame.xyz[row]))
        return history


def nearest_neighbour_associator(
    previous: DetectionFrame | None, current: DetectionFrame, elapsed_s: float, *, gate_m: float = 1.0
) -> torch.Tensor:
    """REFERENCE constant-velocity nearest-neighbour association.

    Each previous detection is predicted forward by ``elapsed_s`` along its own
    radial velocity - the only velocity component a radar measures - and each
    current detection is matched to the nearest prediction within ``gate_m``.
    Matching is greedy in order of distance, so one prediction claims at most
    one detection.

    Reference, and labelled so: there is no track initiation logic, no track
    death, no covariance, and no multi-hypothesis anything. It exists so the
    handoff contract has a runnable example. It is NOT differentiable and it does
    not pretend to be.
    """

    device = current.xyz.device
    result = torch.full((len(current),), -1, dtype=torch.int64, device=device)
    if previous is None or len(previous) == 0 or len(current) == 0:
        return result

    # Radial closing velocity moves a target ALONG its own line of sight, which
    # is the unit vector from the array to where it was.
    radius = previous.xyz.square().sum(dim=1, keepdim=True).sqrt().clamp(min=1e-12)
    direction = previous.xyz / radius
    predicted = previous.xyz - direction * (previous.velocity_mps.reshape(-1, 1) * float(elapsed_s))
    # Written out rather than through ``torch.cdist``: a pairwise distance is
    # arithmetic here, and this package does not reach for a geometry primitive
    # that the physics packages are statically forbidden from naming.
    distance = (current.xyz.unsqueeze(1) - predicted.unsqueeze(0)).square().sum(dim=-1).sqrt()
    claimed = torch.zeros((len(previous),), dtype=torch.bool, device=device)
    order = torch.argsort(distance.reshape(-1))
    columns = int(distance.shape[1])
    for flat in order.tolist():
        row, column = divmod(flat, columns)
        if result[row] >= 0 or claimed[column]:
            continue
        if float(distance[row, column]) > float(gate_m):
            break
        result[row] = column
        claimed[column] = True
    return result


__all__ = ["Associator", "DetectionFrame", "TrackHandoff", "nearest_neighbour_associator"]
