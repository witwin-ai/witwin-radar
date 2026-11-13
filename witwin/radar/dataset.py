"""Dataset writing: run a list of scenes and seeds, and stream cubes, labels and metadata to disk.

:func:`write_dataset` runs one :class:`Episode` after another through
:meth:`Radar.stream <witwin.radar.Radar.stream>`, so only one frame is ever on
the device, and writes each frame as soon as it exists. The layout is::

    root/dataset.json            format, version, the radar's flat configuration, the episode index
    root/<episode>/              one saved result (:mod:`witwin.radar.storage`), whose frame files
                                 also carry a "labels" mapping (:meth:`TargetLabels.tensors`)

So every episode directory opens with :meth:`Result.load <witwin.radar.Result.load>`,
and :func:`load_episode` returns it with its labels. ``dataset.json`` and each
episode's ``metadata.json`` are written last, so a directory missing them is an
interrupted write.

An episode's ``seed`` replaces the radar's receive-chain seed for that episode
(``Radar.replace(seed=...)``); ``None`` keeps the radar's. The labels are
:class:`~witwin.radar.labels.GroundTruth` at each frame instant.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from .labels import VELOCITY_STEP_S, GroundTruth, TargetLabels
from .storage import (
    json_value,
    metadata_record,
    prepare_directory,
    read_frame,
    read_metadata,
    write_axes,
    write_frame,
    write_metadata,
)

#: The format name of a dataset's ``dataset.json``.
DATASET_FORMAT = "witwin.radar.dataset"
DATASET_FORMAT_VERSION = 1
DATASET_INDEX = "dataset.json"

#: Per-frame members accumulated across a streamed episode, in the saved result's vocabulary.
_PER_FRAME = ("times", "epochs", "rediscovery_reasons")


@dataclass(frozen=True)
class Episode:
    """One scene to simulate: the world, what is in it, when, and with which receiver seed."""

    #: A ``witwin.core.Scene`` or ``DynamicScene``.
    scene: Any
    #: ``PointTargets``, ``StructureTargets`` or ``MeshTargets``.
    targets: Any
    #: Frame instants, s.
    times: Sequence[float]
    #: The radar's receive-chain seed for this episode; ``None`` keeps the radar's.
    seed: int | None = None
    #: The episode's directory name; ``None`` is ``episode_<index>``.
    name: str | None = None
    #: The radar's ego-motion for this episode, as the verbs take it.
    pose: Callable[[float], Any] | None = None
    #: Keyword arguments for :meth:`Radar.stream` beyond the dataset-wide ones (``motion``, ``reflections``...).
    options: Mapping[str, Any] = field(default_factory=dict)


def write_dataset(
    path: str | os.PathLike[str],
    radar: Any,
    episodes: Sequence[Episode],
    *,
    overwrite: bool = False,
    labels: bool = True,
    velocity_step_s: float = VELOCITY_STEP_S,
    **options: Any,
) -> Path:
    """Simulate every episode and write it under ``path``; returns the root directory.

    ``options`` are :meth:`Radar.stream` keywords shared by every episode
    (``los``, ``reflections``, ``motion``, ...); an episode's own ``options``
    override them. An ``ad_mode`` other than ``"none"`` is refused: a dataset is data, and a frame written
    to disk keeps no graph. An existing non-empty ``path`` is refused unless
    ``overwrite`` and it holds a dataset this function wrote.
    """

    if options.get("ad_mode", "none") != "none" or any(e.options.get("ad_mode", "none") != "none" for e in episodes):
        raise ValueError("a dataset is written without a derivative; grad must be 'none'")
    names = [episode.name or f"episode_{index:05d}" for index, episode in enumerate(episodes)]
    if len(set(names)) != len(names):
        raise ValueError(f"episode names must be unique, got {names}")
    if any(Path(name).name != name or name in ("", ".", "..") for name in names):
        raise ValueError(f"an episode name is one directory name, got {names}")
    root = Path(path)
    if root.exists() and any(root.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{root} is not empty; pass overwrite=True to replace a dataset")
        if not (root / DATASET_INDEX).is_file():
            raise FileExistsError(f"{root} is not empty and holds no dataset; refusing to overwrite it")
        _remove_dataset(root)
    root.mkdir(parents=True, exist_ok=True)

    index = []
    for name, episode in zip(names, episodes, strict=True):
        frames = _write_episode(
            root / name, radar, episode, labels=labels, velocity_step_s=velocity_step_s, options=options
        )
        index.append(
            {"name": name, "seed": episode.seed, "frames": frames, "times_s": [float(t) for t in episode.times]}
        )
    record = {
        "format": DATASET_FORMAT,
        "version": DATASET_FORMAT_VERSION,
        "radar": radar.to_dict(),
        "options": json_value({key: value for key, value in options.items() if key != "motion"}),
        "motion": repr(options["motion"]) if "motion" in options else None,
        "labels": bool(labels),
        "episodes": index,
    }
    staging = root / (DATASET_INDEX + ".partial")
    staging.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(staging, root / DATASET_INDEX)
    return root


def _write_episode(
    directory: Path, radar: Any, episode: Episode, *, labels: bool, velocity_step_s: float, options: Mapping[str, Any]
) -> int:
    """Stream one episode to ``directory``; returns its frame count."""

    prepare_directory(directory, overwrite=True)
    run_radar = radar if episode.seed is None else radar.replace(seed=int(episode.seed))
    arguments = {**options, **episode.options}
    labeler = (
        GroundTruth(run_radar, episode.scene, episode.targets, pose=episode.pose, velocity_step_s=velocity_step_s)
        if labels
        else None
    )
    per_frame: dict[str, list[Any]] = {name: [] for name in _PER_FRAME}
    adaptive: list[Any] = []
    first: Any = None
    last: Any = None
    complete = {"path_set_complete": True, "motion_sampling_exhaustive": True, "discovery_complete": True}
    dropped = 0
    count = 0
    stream = run_radar.stream(episode.scene, episode.targets, times=episode.times, pose=episode.pose, **arguments)
    for count, frame in enumerate(stream, start=1):
        extra = {} if labeler is None else {"labels": labeler.at(float(frame.times[0])).tensors()}
        write_frame(directory, count - 1, frame.cube[0], frame.sample_times[0], **extra)
        if first is None:
            first = frame
            write_axes(directory, frame.axes)
        for name in _PER_FRAME:
            per_frame[name].append(getattr(frame, name)[0])
        for name in complete:
            complete[name] = complete[name] and bool(getattr(frame, name))
        dropped = max(dropped, int(frame.transmission_dropped_rows))
        adaptive.extend(frame.adaptive_diagnostics)
        last = frame
    if first is None or last is None:
        raise ValueError(f"episode {directory.name} has no frame instants")
    run = {
        "kind": last.kind,
        "axis_names": list(last.axis_names),
        "phasor": last.phasor,
        "time_dependence": last.time_dependence,
        "reference_frequency_hz": float(last.reference_frequency_hz),
        "times": [float(value) for value in per_frame["times"]],
        "epochs": [int(value) for value in per_frame["epochs"]],
        "rediscovery_reasons": list(per_frame["rediscovery_reasons"]),
        "compile_count": int(last.compile_count),
        "discovery_count": int(last.discovery_count),
        "motion_sampling": last.motion_sampling,
        "output_domain": last.output_domain,
        "observation_schedule": last.observation_schedule,
        "transmission_dropped_rows": dropped,
        "adaptive_diagnostics": json_value(adaptive),
        **complete,
    }
    record = metadata_record(
        run, axes=first.axes, cube=first.cube[0], frame_count=count, extra={"episode": {"seed": episode.seed}}
    )
    write_metadata(directory, record)
    return count


def _remove_dataset(root: Path) -> None:
    """Remove a dataset's own episode directories and index, and nothing else."""

    record = json.loads((root / DATASET_INDEX).read_text(encoding="utf-8"))
    for entry in record.get("episodes", []):
        directory = root / str(entry["name"])
        if (directory / "metadata.json").is_file():
            prepare_directory(directory, overwrite=True)
            if not any(directory.iterdir()):
                directory.rmdir()
    (root / DATASET_INDEX).unlink()


def read_dataset(path: str | os.PathLike[str]) -> dict[str, Any]:
    """The ``dataset.json`` index of a dataset :func:`write_dataset` finished."""

    file = Path(path) / DATASET_INDEX
    if not file.is_file():
        raise FileNotFoundError(f"{Path(path)} holds no {DATASET_INDEX}: not a dataset, or its write did not finish")
    record = json.loads(file.read_text(encoding="utf-8"))
    if record.get("format") != DATASET_FORMAT:
        raise ValueError(f"{file} declares format {record.get('format')!r}, not {DATASET_FORMAT!r}")
    version = record.get("version")
    if type(version) is not int or not 1 <= version <= DATASET_FORMAT_VERSION:
        raise ValueError(f"{file} is format version {version!r}; this reads 1 to {DATASET_FORMAT_VERSION}")
    return record


def load_episode(
    path: str | os.PathLike[str], *, device: torch.device | str = "cpu"
) -> tuple[Any, TargetLabels | None]:
    """One episode directory as ``(Result, TargetLabels or None)``."""

    from .results import Result

    directory = Path(path)
    result = Result.load(directory, device=device)
    record = read_metadata(directory)
    frames = [read_frame(directory, index) for index in range(int(record["frame_count"]))]
    if not all("labels" in frame for frame in frames):
        return result, None
    return result, TargetLabels.stack([TargetLabels.from_tensors(frame["labels"]) for frame in frames])


__all__ = ["Episode", "load_episode", "read_dataset", "write_dataset"]
