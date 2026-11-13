"""Path births and deaths: the per-frame event log of a moving session.

A geometric-optics path appears and disappears as a step. This module records
WHEN that happened, which round trip it was, and how strong it was at the
step, so a caller can see a birth or a shadow-boundary crossing rather than
infer one from a broadband smear in a range-Doppler map.

Detection is a native kernel (``path_events`` in
``witwin/radar/cuda/path_events.cu``). The host authors, once per pair of
frozen joins, which composed row of the earlier observation is the same round
trip as which row of the later one; the kernel compares the two rows'
existence and appends one record per change through an atomic counter. A
frame's records stay on the device until :class:`PathEvents` is first read,
so recording costs no synchronization in the frame loop.

Events are DISCRETE. Nothing here is differentiable and every payload is read
detached; an event's amplitude is a diagnostic magnitude, not a quantity a
loss should be built on.

Resolution: an event is known only to lie between two evaluated observations,
``previous_time_s`` (the last instant the old state was seen) and ``time_s``
(the first instant the new state was seen). On the exhaustive ADC route the
two are adjacent ADC instants; on the adaptive route they are adjacent probes.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from .cuda import native_ops as _ops

if TYPE_CHECKING:
    from .paths import RadarPathBatch, TwoWayComposer

#: Records one frame can hold. A frame that detects more reports how many it
#: dropped in :attr:`PathEvents.dropped` rather than losing them silently.
EVENT_CAPACITY = 65536

#: How many host candidate tables (one per pair of joins and slot layout) and
#: their device uploads a session keeps.
TABLE_CACHE = 64

#: Columns of :attr:`TwoWayComposer.identity_host`.
_SOURCE, _SITE, _SINK, _IN_COMPONENT, _OUT_COMPONENT, _IN_DEPTH, _OUT_DEPTH = range(7)


def _padded(values: np.ndarray, width: int) -> np.ndarray:
    """``values`` widened to ``width`` columns with the no-interaction label ``-1``."""

    if values.shape[1] == width:
        return values
    pad = np.full((values.shape[0], width - values.shape[1]), -1, dtype=np.int64)
    return np.concatenate((values, pad), axis=1)


def _row_keys(composer: TwoWayComposer, in_width: int, out_width: int) -> np.ndarray:
    return np.concatenate(
        (
            composer.identity_host,
            _padded(composer.inbound_sequence_host, in_width),
            _padded(composer.outbound_sequence_host, out_width),
        ),
        axis=1,
    )


@dataclass(frozen=True, slots=True, eq=False)
class Transition:
    """Which rows of one join are which rows of the next, as candidate pairs.

    ``before[i]`` and ``after[i]`` are local rows of the earlier and later
    join, or ``-1`` where the round trip does not exist on that side. Rows of
    one join are unique by key (the join refuses a repeated key), so the match
    is a bijection between the rows the two joins share.
    """

    before: np.ndarray
    after: np.ndarray
    #: ``int64[candidates, 7]``: each candidate's identity, from whichever side has it.
    identity: np.ndarray

    @property
    def count(self) -> int:
        return int(self.before.shape[0])

    @classmethod
    def between(cls, earlier: TwoWayComposer, later: TwoWayComposer) -> Transition:
        """Match two joins' rows by their whole key, with array operations only."""

        if earlier is later:
            rows = np.arange(later.path_count, dtype=np.int64)
            return cls(before=rows, after=rows, identity=later.identity_host)
        in_width = max(earlier.inbound_sequence_host.shape[1], later.inbound_sequence_host.shape[1])
        out_width = max(earlier.outbound_sequence_host.shape[1], later.outbound_sequence_host.shape[1])
        keys_a = _row_keys(earlier, in_width, out_width)
        keys_b = _row_keys(later, in_width, out_width)
        count_a, count_b = len(keys_a), len(keys_b)
        partner = np.full(count_b, -1, dtype=np.int64)
        if count_a and count_b:
            _, labels = np.unique(np.concatenate((keys_a, keys_b)), axis=0, return_inverse=True)
            labels = labels.reshape(-1)
            row_of_label = np.full(int(labels.max()) + 1, -1, dtype=np.int64)
            row_of_label[labels[:count_a]] = np.arange(count_a)
            partner = row_of_label[labels[count_a:]]
        kept = np.zeros(count_a, dtype=bool)
        kept[partner[partner >= 0]] = True
        died = np.flatnonzero(~kept)
        # Every later row is a candidate (matched, or born when unmatched);
        # every earlier row without a partner is a candidate death.
        before = np.concatenate((partner, died)).astype(np.int64)
        after = np.concatenate((np.arange(count_b), np.full(len(died), -1))).astype(np.int64)
        identity = np.concatenate((later.identity_host, earlier.identity_host[died])).reshape(-1, 7)
        return cls(before=before, after=after, identity=identity)


@dataclass(frozen=True, slots=True, eq=False)
class _Launch:
    """One kernel launch's host decoding table.

    Candidate ``base + c`` belongs to transition ``t`` where
    ``starts[t] <= c < starts[t + 1]``; that transition moved from the
    observation at ``previous_time_s[t]`` to the one at ``time_s[t]``.
    """

    base: int
    starts: np.ndarray
    time_s: np.ndarray
    previous_time_s: np.ndarray
    observation: np.ndarray
    transitions: tuple[Transition, ...]


class PathEvents:
    """One frame's path births and deaths, materialised on first read.

    Every array has one entry per event, ordered by the instant the event was
    seen and then by row identity. ``birth`` is true for a path that exists
    at ``time_s`` and did not at ``previous_time_s``, false for the reverse;
    the true instant lies in ``(previous_time_s, time_s]``. ``amplitude`` is
    ``|C|`` of the live side in the composed transfer's sqrt(W) units: the new
    row for a birth, the last live row for a death.

    ``radar_source_id``, ``site_id`` and ``radar_sink_id`` name the round
    trip; ``component`` is ``int32[events, 2]``, the inbound and outbound legs'
    Channel component IDs, and ``depth`` the two legs' interaction counts
    summed. ``observation`` is the index of ``time_s`` in the frame's
    observation schedule (``sample_times_s``).

    The first observation of a session has no predecessor, so the paths it
    starts with are not births.
    """

    __slots__ = ("_code", "_amplitude", "_counter", "_launches", "_cache", "_candidates", "_lag")

    def __init__(self) -> None:
        self._lag = 0.0
        self._code: torch.Tensor | None = None
        self._amplitude: torch.Tensor | None = None
        self._counter: torch.Tensor | None = None
        self._launches: list[_Launch] = []
        self._candidates = 0
        self._cache: dict[str, Any] | None = None

    def _buffers(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._code is None:
            self._code = torch.empty(EVENT_CAPACITY, dtype=torch.int64, device=device)
            self._amplitude = torch.empty(EVENT_CAPACITY, dtype=torch.float32, device=device)
            self._counter = torch.zeros(1, dtype=torch.int32, device=device)
        assert self._amplitude is not None and self._counter is not None
        return self._code, self._amplitude, self._counter

    def _launch(
        self,
        previous: Pool | None,
        current: Pool,
        tables: torch.Tensor,
        transitions: Sequence[Transition],
        time_s: np.ndarray,
        previous_time_s: np.ndarray,
        observation: np.ndarray,
    ) -> None:
        """Detect one launch's events: ``tables`` is ``int64[2, candidates]`` on the device."""

        candidates = int(tables.shape[1])
        if candidates == 0:
            return
        current_valid, current_transfer = current
        code, amplitude, counter = self._buffers(current_valid.device)
        previous_valid, previous_transfer = (current_valid[:0], current_transfer[:0]) if previous is None else previous
        _ops().path_events_forward(
            previous_valid,
            previous_transfer,
            current_valid,
            current_transfer,
            tables[0],
            tables[1],
            code,
            amplitude,
            counter,
            self._candidates,
        )
        starts = np.concatenate(([0], np.cumsum([transition.count for transition in transitions])))
        self._launches.append(
            _Launch(
                base=self._candidates,
                starts=starts,
                time_s=np.asarray(time_s, dtype=np.float64),
                previous_time_s=np.asarray(previous_time_s, dtype=np.float64),
                observation=np.asarray(observation, dtype=np.int64),
                transitions=tuple(transitions),
            )
        )
        self._candidates += candidates

    def _materialised(self) -> dict[str, Any]:
        if self._cache is not None:
            return self._cache
        columns: dict[str, Any] = {
            "time_s": np.zeros(0),
            "previous_time_s": np.zeros(0),
            "observation": np.zeros(0, dtype=np.int64),
            "birth": np.zeros(0, dtype=bool),
            "radar_source_id": np.zeros(0, dtype=np.int64),
            "site_id": np.zeros(0, dtype=np.int64),
            "radar_sink_id": np.zeros(0, dtype=np.int64),
            "component": np.zeros((0, 2), dtype=np.int32),
            "depth": np.zeros(0, dtype=np.int32),
            "amplitude": np.zeros(0, dtype=np.float32),
            "dropped": 0,
        }
        if self._code is None or not self._launches:
            self._cache = columns
            return columns
        assert self._amplitude is not None and self._counter is not None
        # The frame's one synchronization, paid by the first reader.
        found = int(self._counter.item())
        kept = min(found, EVENT_CAPACITY)
        code = self._code[:kept].cpu().numpy()
        amplitude = self._amplitude[:kept].cpu().numpy()
        candidate, birth = code >> 1, (code & 1).astype(bool)
        bases = np.asarray([launch.base for launch in self._launches], dtype=np.int64)
        launch_of = np.searchsorted(bases, candidate, side="right") - 1
        time_s = np.empty(kept)
        previous_time_s = np.empty(kept)
        observation = np.empty(kept, dtype=np.int64)
        identity = np.empty((kept, 7), dtype=np.int64)
        for index in np.unique(launch_of):
            launch = self._launches[int(index)]
            members = np.flatnonzero(launch_of == index)
            local = candidate[members] - launch.base
            transition_of = np.searchsorted(launch.starts, local, side="right") - 1
            time_s[members] = launch.time_s[transition_of]
            previous_time_s[members] = launch.previous_time_s[transition_of]
            observation[members] = launch.observation[transition_of]
            for position in np.unique(transition_of):
                chosen = members[transition_of == position]
                transition = launch.transitions[int(position)]
                identity[chosen] = transition.identity[candidate[chosen] - launch.base - launch.starts[position]]
        order = np.lexsort((identity[:, _SINK], identity[:, _SITE], identity[:, _SOURCE], ~birth, time_s))
        identity = identity[order]
        columns.update(
            time_s=time_s[order],
            previous_time_s=previous_time_s[order],
            observation=observation[order],
            birth=birth[order],
            radar_source_id=identity[:, _SOURCE],
            site_id=identity[:, _SITE],
            radar_sink_id=identity[:, _SINK],
            component=identity[:, [_IN_COMPONENT, _OUT_COMPONENT]].astype(np.int32),
            depth=(identity[:, _IN_DEPTH] + identity[:, _OUT_DEPTH]).astype(np.int32),
            amplitude=amplitude[order],
            dropped=found - kept,
        )
        self._cache = columns
        # The device buffers are no longer needed once decoded.
        self._code = self._amplitude = self._counter = None
        self._launches = []
        return columns

    @property
    def birth_lag_bound_s(self) -> float:
        """How late a birth seen in this frame can have been seen, s.

        The longest stretch this frame relied on one topology discovery: the
        largest gap between consecutive discoveries ending inside the frame,
        and the age of the topology at the frame's last evaluated observation.
        A path born just after a discovery is first seen at the next one, so
        no birth this frame reports was seen later than this after it
        happened. Zero when the candidate family was certified complete or
        nothing in the world moves. Host bookkeeping, known without reading
        the device.
        """

        return self._lag

    def __len__(self) -> int:
        return int(self._materialised()["time_s"].shape[0])

    @property
    def time_s(self) -> np.ndarray:
        """``float64[events]``: the first observation instant the new state was seen at, s."""

        return self._materialised()["time_s"]

    @property
    def previous_time_s(self) -> np.ndarray:
        """``float64[events]``: the last observation instant the old state was seen at, s."""

        return self._materialised()["previous_time_s"]

    @property
    def observation(self) -> np.ndarray:
        """``int64[events]``: the index of :attr:`time_s` in the frame's schedule."""

        return self._materialised()["observation"]

    @property
    def birth(self) -> np.ndarray:
        """``bool[events]``: true for a birth, false for a death."""

        return self._materialised()["birth"]

    @property
    def radar_source_id(self) -> np.ndarray:
        return self._materialised()["radar_source_id"]

    @property
    def site_id(self) -> np.ndarray:
        return self._materialised()["site_id"]

    @property
    def radar_sink_id(self) -> np.ndarray:
        return self._materialised()["radar_sink_id"]

    @property
    def component(self) -> np.ndarray:
        """``int32[events, 2]``: the inbound and outbound legs' Channel component IDs."""

        return self._materialised()["component"]

    @property
    def depth(self) -> np.ndarray:
        """``int32[events]``: the round trip's interaction count, both legs summed."""

        return self._materialised()["depth"]

    @property
    def amplitude(self) -> np.ndarray:
        """``float32[events]``: ``|C|`` of the live side at the event, sqrt(W)."""

        return self._materialised()["amplitude"]

    @property
    def dropped(self) -> int:
        """Events detected past :data:`EVENT_CAPACITY` and not recorded."""

        return int(self._materialised()["dropped"])

    def __repr__(self) -> str:
        if self._cache is None:
            return f"PathEvents(<{self._candidates} candidates, not read yet>)"
        return f"PathEvents({len(self)} events, {int(self.birth.sum())} births)"


#: One observation pool as the kernel reads it: ``bool[rows]`` validity and the
#: ``float32[rows, 2]`` real view of the complex transfer, both detached.
Pool = tuple[torch.Tensor, torch.Tensor]


def batch_pool(paths: RadarPathBatch) -> Pool:
    """The detached pool of a composed batch. Views only; nothing is copied."""

    if paths.row_valid is None:
        raise ValueError("an observed batch always carries row_valid; a replayed one always publishes it")
    return paths.row_valid, torch.view_as_real(paths.complex_transfer_ref.detach())


@dataclass(frozen=True, slots=True, eq=False)
class Observed:
    """One evaluated observation, as the recorder reads it.

    ``pool`` is the composed batch whose rows ``offset : offset + rows`` are
    this observation's; ``composer`` is the single-slot join those rows follow.
    ``discovered`` says the topology was rediscovered (or certified complete)
    at this instant, which is what bounds the birth lag.
    """

    time_s: float
    observation: int
    pool: object
    offset: int
    composer: TwoWayComposer
    discovered: bool


class EventRecorder:
    """A session's event detection: the previous observation and the table cache.

    Holds ONE previous observation across calls and frames, so a birth across
    a replay-group or frame boundary is still a birth. Device candidate tables
    are cached by the joins and offsets they were built from; a frame whose
    topology repeats builds none.
    """

    def __init__(self) -> None:
        self._previous: Observed | None = None
        self._previous_pool: Pool | None = None
        self._last_discovery: float | None = None
        self._last_time: float | None = None
        self._tables: OrderedDict[tuple, tuple[torch.Tensor, tuple[Transition, ...], tuple[object, ...]]] = (
            OrderedDict()
        )
        self._transitions: OrderedDict[tuple[int, int], tuple[object, object, Transition]] = OrderedDict()

    def close_frame(self, frame: PathEvents) -> None:
        """Fold the topology's age at the frame's last observation into its lag bound."""

        if self._last_time is not None and self._last_discovery is not None:
            frame._lag = max(frame._lag, self._last_time - self._last_discovery)

    def _transition(self, earlier: TwoWayComposer, later: TwoWayComposer) -> Transition:
        key = (id(earlier), id(later))
        cached = self._transitions.get(key)
        # The cache holds both joins, so their ids cannot be reused while
        # the entry lives.
        if cached is not None and cached[0] is earlier and cached[1] is later:
            self._transitions.move_to_end(key)
            return cached[2]
        transition = Transition.between(earlier, later)
        self._transitions[key] = (earlier, later, transition)
        while len(self._transitions) > TABLE_CACHE:
            self._transitions.popitem(last=False)
        return transition

    def _note_discovery(self, frame: PathEvents, entry: Observed) -> None:
        if entry.discovered:
            if self._last_discovery is not None:
                frame._lag = max(frame._lag, entry.time_s - self._last_discovery)
            self._last_discovery = entry.time_s
        self._last_time = entry.time_s

    def record(self, frame: PathEvents, entries: Sequence[Observed], pool_of: Callable[[object], Pool]) -> None:
        """Detect events across ``entries``, which are in time order.

        One launch per run of consecutive entries that share a pool. The
        first entry of a run compares against the previous entry, which may
        live in another pool; every later one against its neighbour in the
        same pool.
        """

        begin = 0
        while begin < len(entries):
            stop = begin + 1
            while stop < len(entries) and entries[stop].pool is entries[begin].pool:
                stop += 1
            self._record_run(frame, entries[begin:stop], pool_of)
            begin = stop

    def _record_run(self, frame: PathEvents, run: Sequence[Observed], pool_of: Callable[[object], Pool]) -> None:
        previous, previous_pool = self._previous, self._previous_pool
        current_pool = pool_of(run[0].pool)
        external = previous is not None and previous.pool is not run[0].pool
        chain = ([previous] if previous is not None else []) + list(run)
        pairs = list(zip(chain, chain[1:], strict=False))
        for entry in run:
            self._note_discovery(frame, entry)
        self._previous, self._previous_pool = run[-1], current_pool
        if not pairs:
            return
        key = (external, tuple((id(a.composer), a.offset, id(b.composer), b.offset) for a, b in pairs))
        joins = tuple(join for a, b in pairs for join in (a.composer, b.composer))
        cached = self._tables.get(key)
        # The entry holds the joins its key names by id, so a hit whose joins
        # are the same objects cannot be a recycled id.
        if cached is not None and all(x is y for x, y in zip(cached[2], joins, strict=True)):
            self._tables.move_to_end(key)
            tables, transitions = cached[0], cached[1]
        else:
            transitions = tuple(self._transition(a.composer, b.composer) for a, b in pairs)
            before, after = [], []
            for index, ((a, b), transition) in enumerate(zip(pairs, transitions, strict=True)):
                local_before = transition.before
                shifted = np.where(local_before >= 0, local_before + a.offset, -1)
                if index > 0 or not external:
                    # The neighbour is in the CURRENT pool: the kernel's
                    # negative encoding names it.
                    shifted = np.where(local_before >= 0, -2 - shifted, -1)
                before.append(shifted)
                after.append(np.where(transition.after >= 0, transition.after + b.offset, -1))
            host = np.stack((np.concatenate(before), np.concatenate(after))).astype(np.int64)
            tables = torch.from_numpy(host).to(current_pool[0].device)
            self._tables[key] = (tables, transitions, joins)
            while len(self._tables) > TABLE_CACHE:
                self._tables.popitem(last=False)
        frame._launch(
            previous_pool if external else None,
            current_pool,
            tables,
            transitions,
            time_s=np.asarray([b.time_s for _, b in pairs]),
            previous_time_s=np.asarray([a.time_s for a, _ in pairs]),
            observation=np.asarray([b.observation for _, b in pairs]),
        )


__all__ = ["EVENT_CAPACITY", "EventRecorder", "Observed", "PathEvents", "Transition", "batch_pool"]
