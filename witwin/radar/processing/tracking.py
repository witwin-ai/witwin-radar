"""The frame-to-frame detection contract, a reference associator and a Kalman tracker.

The CONTRACT a tracker plugs into comes first, because that is the thing the
simulator owes an external tracker and the thing a tracker cannot supply for
itself. Four pieces:

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
* :class:`KalmanTracker` - a multi-target constant-velocity Kalman tracker
  with Mahalanobis gating, greedy association and a miss-count track
  lifetime, vectorised over tracks.

**This whole module is explicitly NON-DIFFERENTIABLE, and it enforces it.**
A detection is the output of a threshold comparison and an ``argwhere``; a
gradient through an association decision is a gradient through a discrete
choice that does not have one. A derivative-carrying tensor is REFUSED at the
boundary rather than silently detached: a silent detach is how a caller ends up
with a zero gradient and a plausible number, which is worse than an error.

The refusal goes through :func:`witwin.radar.policy.refuse_derivative`, which
checks both reverse-mode ``requires_grad`` and a forward dual's tangent, so this
module and the wall speak with one voice and one wording. In the normal flow the
point-cloud stage has already refused at ITS entry, so the check on
:class:`DetectionFrame` is unreachable; it is kept because an unreachable guard
on the second door is the right shape for a wall, and a caller who hands this
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
        # One host read per frame: the assignment is a per-detection decision
        # on the host, so the source list crosses once and the ids go back once.
        previous_ids = self.assignments[-1].tolist() if self.assignments else []
        ids = []
        for source in continued.tolist():
            if 0 <= source < len(previous_ids):
                ids.append(previous_ids[source])
            else:
                ids.append(self._next_track_id)
                self._next_track_id += 1
        assignment = torch.tensor(ids, dtype=torch.int64, device=continued.device)
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
    if previous is None or len(previous) == 0 or len(current) == 0:
        return torch.full((len(current),), -1, dtype=torch.int64, device=device)

    # Radial closing velocity moves a target ALONG its own line of sight, which
    # is the unit vector from the array to where it was.
    radius = previous.xyz.square().sum(dim=1, keepdim=True).sqrt().clamp(min=1e-12)
    direction = previous.xyz / radius
    predicted = previous.xyz - direction * (previous.velocity_mps.reshape(-1, 1) * float(elapsed_s))
    # Written out rather than through ``torch.cdist``: a pairwise distance is
    # arithmetic here, and this package does not reach for a geometry primitive
    # that the physics packages are statically forbidden from naming.
    distance = (current.xyz.unsqueeze(1) - predicted.unsqueeze(0)).square().sum(dim=-1).sqrt()
    return greedy_assignment(distance, gate=float(gate_m))


def greedy_assignment(cost: torch.Tensor, *, gate: float) -> torch.Tensor:
    """``[N, T]`` costs -> ``[N]`` int64 column per row, ``-1`` for none.

    Global greedy: the cheapest remaining pair within ``gate`` is matched, its
    row and column leave, and so on. Evaluated in rounds rather than pair by
    pair: in each round every MUTUALLY cheapest pair - the row's best column
    whose best row is that row - is matched at once. The cheapest remaining
    pair is always mutual, and matching one mutual pair never makes another
    one non-mutual, so the rounds produce exactly the pair-by-pair greedy
    matching, in a number of rounds bounded by the matching's chain length
    rather than by the number of detections.
    """

    rows, columns = int(cost.shape[0]), int(cost.shape[1])
    result = torch.full((rows,), -1, dtype=torch.int64, device=cost.device)
    if rows == 0 or columns == 0:
        return result
    live = torch.where(cost <= gate, cost.to(torch.float64), torch.full_like(cost, torch.inf, dtype=torch.float64))
    row_index = torch.arange(rows, device=cost.device)
    for _ in range(min(rows, columns)):
        best_column = live.argmin(dim=1)
        best_row = live.argmin(dim=0)
        finite = torch.isfinite(live[row_index, best_column])
        mutual = finite & (best_row[best_column] == row_index)
        if not bool(mutual.any()):
            break
        result = torch.where(mutual, best_column, result)
        taken = torch.zeros(columns, dtype=torch.bool, device=cost.device)
        taken[best_column[mutual]] = True
        live = torch.where(mutual.reshape(-1, 1) | taken.reshape(1, -1), torch.inf, live)
    return result


@dataclass(frozen=True, slots=True, eq=False)
class TrackState:
    """The live tracks of a :class:`KalmanTracker`, ``T`` of them.

    ``position``/``velocity`` are ``[T, 3]`` float64 in the frame the detection
    frames were in (metres, metres per second), ``covariance`` is the
    ``[T, 6, 6]`` state covariance over ``(position, velocity)``. ``hits``
    counts the frames a track was updated in and ``misses`` the consecutive
    frames it was not.
    """

    ids: torch.Tensor
    position: torch.Tensor
    velocity: torch.Tensor
    covariance: torch.Tensor
    hits: torch.Tensor
    misses: torch.Tensor

    def __len__(self) -> int:
        return int(self.ids.shape[0])


class KalmanTracker:
    """Multi-target constant-velocity Kalman tracking of :class:`DetectionFrame` positions.

    State ``x = (p, v)`` in the detection frame's coordinates. Between frames
    ``dt`` seconds apart, ``p <- p + v dt`` with white-acceleration process
    noise of spectral density ``q = process_noise_mps2 ** 2``:

        ``Q = q [[dt^3 / 3 I, dt^2 / 2 I], [dt^2 / 2 I, dt I]]``

    Each detection measures position with covariance
    ``measurement_noise_m ** 2 I``. A detection may continue a track when its
    Mahalanobis distance ``sqrt(y^T S^-1 y)`` to the track's prediction is at
    most ``gate``; within the gate, pairs are matched by
    :func:`greedy_assignment` on that distance. A matched track is updated
    (Joseph form, so the covariance stays symmetric positive definite); an
    unmatched track coasts and is dropped after more than ``max_misses``
    consecutive misses; an unmatched detection opens a track at its position
    with zero velocity and ``initial_velocity_std_mps`` uncertainty.

    Every per-track and per-detection step is one batched tensor operation;
    the only Python loop is the assignment's round loop. Not differentiable:
    an association is a discrete choice, and a derivative-carrying frame is
    refused by :class:`DetectionFrame` itself.
    """

    def __init__(
        self,
        *,
        process_noise_mps2: float = 1.0,
        measurement_noise_m: float = 0.1,
        gate: float = 3.5,
        initial_velocity_std_mps: float = 5.0,
        max_misses: int = 2,
    ) -> None:
        for name, value in (
            ("process_noise_mps2", process_noise_mps2),
            ("measurement_noise_m", measurement_noise_m),
            ("gate", gate),
            ("initial_velocity_std_mps", initial_velocity_std_mps),
        ):
            if not float(value) > 0.0:
                raise ValueError(f"{name} must be positive, got {value!r}")
        if type(max_misses) is not int or max_misses < 0:
            raise ValueError(f"max_misses must be a non-negative int, got {max_misses!r}")
        self.process_noise_mps2 = float(process_noise_mps2)
        self.measurement_noise_m = float(measurement_noise_m)
        self.gate = float(gate)
        self.initial_velocity_std_mps = float(initial_velocity_std_mps)
        self.max_misses = max_misses
        self._state: torch.Tensor | None = None
        self._covariance: torch.Tensor | None = None
        self._ids: torch.Tensor | None = None
        self._hits: torch.Tensor | None = None
        self._misses: torch.Tensor | None = None
        self._time_s: float | None = None
        self._next_id = 0

    @property
    def tracks(self) -> TrackState:
        """The live tracks after the last :meth:`push`."""

        if self._state is None:
            empty = torch.zeros((0,), dtype=torch.int64)
            return TrackState(
                ids=empty,
                position=torch.zeros((0, 3), dtype=torch.float64),
                velocity=torch.zeros((0, 3), dtype=torch.float64),
                covariance=torch.zeros((0, 6, 6), dtype=torch.float64),
                hits=empty,
                misses=empty,
            )
        return TrackState(
            ids=self._ids,
            position=self._state[:, :3],
            velocity=self._state[:, 3:],
            covariance=self._covariance,
            hits=self._hits,
            misses=self._misses,
        )

    def push(self, frame: DetectionFrame) -> torch.Tensor:
        """Advance to ``frame`` and return each detection's ``[N]`` int64 track id."""

        if not isinstance(frame, DetectionFrame):
            raise TypeError(f"push takes a DetectionFrame, got {type(frame).__name__}")
        if self._time_s is not None and not frame.time_s > self._time_s:
            raise ValueError(
                f"frame {frame.frame_index} is stamped {frame.time_s} s, which is not after the previous frame's "
                f"{self._time_s} s"
            )
        device = frame.xyz.device
        measured = frame.xyz.to(torch.float64)
        count = len(frame)
        if self._state is None:
            self._state = torch.zeros((0, 6), dtype=torch.float64, device=device)
            self._covariance = torch.zeros((0, 6, 6), dtype=torch.float64, device=device)
            self._ids = torch.zeros((0,), dtype=torch.int64, device=device)
            self._hits = torch.zeros((0,), dtype=torch.int64, device=device)
            self._misses = torch.zeros((0,), dtype=torch.int64, device=device)
        elapsed = 0.0 if self._time_s is None else frame.time_s - self._time_s
        self._time_s = frame.time_s
        self._predict(elapsed)

        noise = self.measurement_noise_m**2 * torch.eye(3, dtype=torch.float64, device=device)
        innovation_cov = self._covariance[:, :3, :3] + noise
        # [T, 3, N]: every detection's innovation against every prediction.
        innovation = measured.transpose(0, 1).unsqueeze(0) - self._state[:, :3].unsqueeze(-1)
        whitened = torch.linalg.solve(innovation_cov, innovation)
        distance = (innovation * whitened).sum(dim=1).clamp(min=0.0).sqrt().transpose(0, 1)
        assigned = greedy_assignment(distance, gate=self.gate)

        matched = assigned >= 0
        track_of = assigned[matched]
        hit = torch.zeros(int(self._ids.shape[0]), dtype=torch.bool, device=device)
        hit[track_of] = True
        if bool(matched.any()):
            self._update(track_of, measured[matched], noise)
        self._hits = self._hits + hit.to(torch.int64)
        self._misses = torch.where(hit, torch.zeros_like(self._misses), self._misses + 1)

        ids = torch.full((count,), -1, dtype=torch.int64, device=device)
        ids[matched] = self._ids[track_of]
        fresh = ~matched
        born = int(fresh.sum())
        if born:
            new_ids = torch.arange(self._next_id, self._next_id + born, dtype=torch.int64, device=device)
            self._next_id += born
            ids[fresh] = new_ids
            self._birth(measured[fresh], new_ids, noise)
        keep = self._misses <= self.max_misses
        self._state, self._covariance = self._state[keep], self._covariance[keep]
        self._ids, self._hits, self._misses = self._ids[keep], self._hits[keep], self._misses[keep]
        return ids

    def _predict(self, dt: float) -> None:
        if dt <= 0.0 or int(self._state.shape[0]) == 0:
            return
        device = self._state.device
        eye = torch.eye(3, dtype=torch.float64, device=device)
        transition = torch.eye(6, dtype=torch.float64, device=device)
        transition[:3, 3:] = dt * eye
        q = self.process_noise_mps2**2
        process = torch.zeros((6, 6), dtype=torch.float64, device=device)
        process[:3, :3] = q * dt**3 / 3.0 * eye
        process[:3, 3:] = q * dt**2 / 2.0 * eye
        process[3:, :3] = q * dt**2 / 2.0 * eye
        process[3:, 3:] = q * dt * eye
        self._state = self._state @ transition.transpose(0, 1)
        self._covariance = transition @ self._covariance @ transition.transpose(0, 1) + process

    def _update(self, tracks: torch.Tensor, measured: torch.Tensor, noise: torch.Tensor) -> None:
        state = self._state[tracks]
        covariance = self._covariance[tracks]
        innovation_cov = covariance[:, :3, :3] + noise
        # K = P H^T S^-1, solved rather than inverted: S K^T = H P.
        gain = torch.linalg.solve(innovation_cov, covariance[:, :3, :]).transpose(1, 2)
        innovation = measured - state[:, :3]
        state = state + (gain @ innovation.unsqueeze(-1)).squeeze(-1)
        identity = torch.eye(6, dtype=torch.float64, device=state.device)
        reduce = identity - torch.cat((gain, torch.zeros_like(gain)), dim=2)
        covariance = reduce @ covariance @ reduce.transpose(1, 2) + gain @ noise @ gain.transpose(1, 2)
        self._state = self._state.index_copy(0, tracks, state)
        self._covariance = self._covariance.index_copy(0, tracks, covariance)

    def _birth(self, measured: torch.Tensor, ids: torch.Tensor, noise: torch.Tensor) -> None:
        count = int(measured.shape[0])
        device = measured.device
        state = torch.cat((measured, torch.zeros_like(measured)), dim=1)
        covariance = torch.zeros((count, 6, 6), dtype=torch.float64, device=device)
        covariance[:, :3, :3] = noise
        covariance[:, 3:, 3:] = self.initial_velocity_std_mps**2 * torch.eye(3, dtype=torch.float64, device=device)
        zero = torch.zeros(count, dtype=torch.int64, device=device)
        self._state = torch.cat((self._state, state))
        self._covariance = torch.cat((self._covariance, covariance))
        self._ids = torch.cat((self._ids, ids))
        self._hits = torch.cat((self._hits, zero + 1))
        self._misses = torch.cat((self._misses, zero))


__all__ = [
    "Associator",
    "DetectionFrame",
    "KalmanTracker",
    "TrackHandoff",
    "TrackState",
    "greedy_assignment",
    "nearest_neighbour_associator",
]
