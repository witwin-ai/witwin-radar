"""The saved form of a :class:`~witwin.radar.Result`: a directory, versioned, with no pickled objects.

Layout of a saved result at ``path``::

    path/metadata.json        format, version, producer, every scalar of the result and its axes
    path/axes.pt              the axes record's tensors, a flat {name: tensor} mapping
    path/frames/000000.pt     one frame: {"cube": [tx, rx, slow, fast], "sample_times_s": float64[observations]}
    path/frames/000001.pt     ...

Every ``.pt`` file is :func:`torch.save` of a flat mapping from a string to a
plain tensor (or to a mapping of such), and it is read back with
``torch.load(weights_only=True)``, which refuses anything else. No Radar,
Channel or Core object is pickled, so a saved result opens on a machine with
any version of those packages, and a file from an untrusted source cannot run
code when it is read. Everything that is not a tensor is in ``metadata.json``.

The cube is one file per frame so that a writer can stream: a frame is written
as soon as it exists, and ``metadata.json`` - which lists the frames - is
written last, so a directory without it is an incomplete write, never a
shorter result. :mod:`witwin.radar.dataset` writes its episodes in this format
and adds a ``labels`` mapping to each frame file.

What is NOT saved, by design: the closing frame's world state
(``last_snapshot``, ``last_compiled_scene``, ``last_propagation``,
``last_radar_paths``) and the per-frame path events. They are live handles on
a compiled scene and device buffers, they are not data, and a loaded result
carries ``None`` and ``()`` in their place. Gradients are not saved; a loaded
cube is a leaf.

Versioning: ``metadata.json`` carries ``"format": "witwin.radar.result"`` and an
integer ``"version"``. A reader refuses a format it does not know and a version
newer than its own, by name.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

#: The format name every saved result declares.
RESULT_FORMAT = "witwin.radar.result"
#: The format version this module writes, and the newest it reads.
RESULT_FORMAT_VERSION = 1

METADATA_FILE = "metadata.json"
AXES_FILE = "axes.pt"
FRAMES_DIRECTORY = "frames"

#: The result members a saved result does not carry, and why, recorded in the metadata.
NOT_SAVED = {
    "last_snapshot": "a Core scene snapshot: a live world object, not data",
    "last_compiled_scene": "a Channel compiled scene: a device handle",
    "last_propagation": "the closing frame's leg batches: device buffers tied to a compiled scene",
    "last_radar_paths": "the closing frame's composed rows: device buffers tied to a compiled scene",
    "events": "per-frame path events decode against the session's joins",
}

#: Result members that are scalars or tuples of scalars, written to the metadata as they stand.
_RESULT_SCALARS = (
    "kind",
    "axis_names",
    "phasor",
    "time_dependence",
    "reference_frequency_hz",
    "times_s",
    "epochs",
    "rediscovery_reasons",
    "compile_count",
    "discovery_count",
    "path_set_complete",
    "motion_sampling_exhaustive",
    "motion_sampling",
    "output_domain",
    "observation_schedule",
    "discovery_complete",
    "transmission_dropped_rows",
)


def frame_file(index: int) -> str:
    """The relative path of frame ``index`` inside a saved result."""

    return f"{FRAMES_DIRECTORY}/{int(index):06d}.pt"


def json_value(value: Any) -> Any:
    """A JSON-representable copy of ``value``: tuples become lists, arrays and tensors their lists.

    A non-finite float is written as the string ``"nan"``, ``"inf"`` or
    ``"-inf"``, because JSON has no such numbers and Python's encoder would
    otherwise write a token other readers reject.
    """

    if isinstance(value, torch.Tensor):
        return json_value(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, complex):
        return {"re": json_value(value.real), "im": json_value(value.imag)}
    if isinstance(value, Mapping):
        return {str(key): json_value(entry) for key, entry in value.items()}
    if isinstance(value, Iterable):
        return [json_value(entry) for entry in value]
    return str(value)


def _tuples(value: Any) -> Any:
    """Nested JSON lists back to nested tuples."""

    if isinstance(value, list):
        return tuple(_tuples(entry) for entry in value)
    return value


def _axes_record(axes: Any) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Split a ``ProcessingAxes`` into its JSON scalars and its tensors, field by field."""

    scalars: dict[str, Any] = {}
    tensors: dict[str, torch.Tensor] = {}
    for field in dataclasses.fields(axes):
        value = getattr(axes, field.name)
        if isinstance(value, torch.Tensor):
            tensors[field.name] = value.detach().cpu()
        else:
            scalars[field.name] = json_value(value)
    return scalars, tensors


def _rebuild_axes(scalars: Mapping[str, Any], tensors: Mapping[str, torch.Tensor]) -> Any:
    from .processing import ProcessingAxes

    known = {field.name for field in dataclasses.fields(ProcessingAxes)}
    unknown = sorted((set(scalars) | set(tensors)) - known)
    if unknown:
        raise ValueError(f"the saved axes record names fields this version does not know: {unknown}")
    fields: dict[str, Any] = {name: _tuples(value) for name, value in scalars.items()}
    fields.update(tensors)
    for name in known - set(fields):
        default = next(field for field in dataclasses.fields(ProcessingAxes) if field.name == name)
        if default.default is dataclasses.MISSING:
            raise ValueError(f"the saved axes record is missing {name!r}")
    return ProcessingAxes(**fields)


def run_record(result: Any) -> dict[str, Any]:
    """A result's saved scalars - per-frame tuples and run-level statements - as JSON values."""

    record = {name: json_value(getattr(result, name)) for name in _RESULT_SCALARS}
    record["adaptive_diagnostics"] = json_value(result.adaptive_diagnostics)
    return record


def metadata_record(
    run: Mapping[str, Any], *, axes: Any, cube: torch.Tensor, frame_count: int, extra: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """The ``metadata.json`` record of a saved result of ``frame_count`` frames.

    ``run`` is :func:`run_record`'s mapping, or the same keys accumulated by a
    streaming writer; ``cube`` is any one frame's ``[tx, rx, slow, fast]`` cube,
    read for its dtype and shape only.
    """

    from .deployment import package_version

    run = dict(run)
    record: dict[str, Any] = {
        "format": RESULT_FORMAT,
        "version": RESULT_FORMAT_VERSION,
        "producer": {"package": "witwin-radar", "version": package_version()},
        "frame_count": int(frame_count),
        "frames": [frame_file(index) for index in range(frame_count)],
        "adaptive_diagnostics": run.pop("adaptive_diagnostics", []),
        "result": run,
        "axes": _axes_record(axes)[0],
        "axes_tensors": sorted(_axes_record(axes)[1]),
        "cube": {"dtype": str(cube.dtype).removeprefix("torch."), "frame_shape": list(cube.shape)},
        "not_saved": dict(NOT_SAVED),
    }
    if extra:
        record.update(extra)
    return record


def write_axes(directory: Path, axes: Any) -> None:
    torch.save(_axes_record(axes)[1], directory / AXES_FILE)


def write_frame(directory: Path, index: int, cube: torch.Tensor, sample_times_s: Any, **extra: Any) -> None:
    """Write frame ``index``: its ``[tx, rx, slow, fast]`` cube, its observation instants and any extra mappings."""

    payload: dict[str, Any] = {
        "cube": cube.detach().cpu().contiguous(),
        "sample_times_s": torch.as_tensor(np.asarray(sample_times_s, dtype=np.float64)),
    }
    payload.update(extra)
    path = directory / frame_file(index)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def write_metadata(directory: Path, record: Mapping[str, Any]) -> None:
    """Write ``metadata.json`` through a temporary file, so a reader never sees half of one."""

    staging = directory / (METADATA_FILE + ".partial")
    staging.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(staging, directory / METADATA_FILE)


def prepare_directory(path: str | os.PathLike[str], *, overwrite: bool) -> Path:
    """Create ``path`` for a new saved result; refuse a non-empty one unless ``overwrite``."""

    directory = Path(path)
    if directory.exists():
        if not directory.is_dir():
            raise FileExistsError(f"{directory} exists and is not a directory")
        if any(directory.iterdir()):
            if not overwrite:
                raise FileExistsError(f"{directory} is not empty; pass overwrite=True to replace a saved result")
            if not (directory / METADATA_FILE).is_file():
                raise FileExistsError(
                    f"{directory} is not empty and holds no saved result; refusing to overwrite a directory "
                    "this format did not write"
                )
            _remove_saved(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _remove_saved(directory: Path) -> None:
    """Remove a saved result's own files and nothing else."""

    record = read_metadata(directory)
    for relative in record.get("frames", []):
        (directory / relative).unlink(missing_ok=True)
    (directory / AXES_FILE).unlink(missing_ok=True)
    (directory / METADATA_FILE).unlink()
    frames = directory / FRAMES_DIRECTORY
    if frames.is_dir() and not any(frames.iterdir()):
        frames.rmdir()


def save_result(result: Any, path: str | os.PathLike[str], *, overwrite: bool = False) -> Path:
    """Write ``result`` to the directory ``path`` in the saved-result format."""

    directory = prepare_directory(path, overwrite=overwrite)
    frames = int(result.cube.shape[0])
    write_axes(directory, result.axes)
    for index in range(frames):
        write_frame(directory, index, result.cube[index], result.sample_times_s[index])
    write_metadata(
        directory, metadata_record(run_record(result), axes=result.axes, cube=result.cube[0], frame_count=frames)
    )
    return directory


def read_metadata(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read and check a saved result's ``metadata.json``."""

    directory = Path(path)
    file = directory / METADATA_FILE
    if not file.is_file():
        raise FileNotFoundError(
            f"{directory} holds no {METADATA_FILE}: it is not a saved result, or its write did not finish"
        )
    record = json.loads(file.read_text(encoding="utf-8"))
    if record.get("format") != RESULT_FORMAT:
        raise ValueError(f"{file} declares format {record.get('format')!r}, not {RESULT_FORMAT!r}")
    version = record.get("version")
    if type(version) is not int or version < 1 or version > RESULT_FORMAT_VERSION:
        raise ValueError(
            f"{file} is format version {version!r}; this witwin-radar reads versions 1 to {RESULT_FORMAT_VERSION}"
        )
    return record


def read_frame(path: str | os.PathLike[str], index: int, *, device: torch.device | str = "cpu") -> dict[str, Any]:
    """One frame file's mapping, tensors on ``device``."""

    return torch.load(Path(path) / frame_file(index), map_location=device, weights_only=True)


def load_result_fields(path: str | os.PathLike[str], *, device: torch.device | str = "cpu") -> dict[str, Any]:
    """The keyword arguments that rebuild the saved result at ``path``, tensors on ``device``."""

    directory = Path(path)
    record = read_metadata(directory)
    tensors = torch.load(directory / AXES_FILE, map_location=device, weights_only=True)
    axes = _rebuild_axes(record["axes"], tensors)
    frames = [read_frame(directory, index, device=device) for index in range(int(record["frame_count"]))]
    if not frames:
        raise ValueError(f"{directory} records no frame")
    scalars = record["result"]
    fields: dict[str, Any] = {name: _tuples(scalars[name]) for name in _RESULT_SCALARS}
    for name in ("reference_frequency_hz",):
        fields[name] = float(fields[name])
    fields["times_s"] = tuple(float(value) for value in fields["times_s"])
    fields.update(
        cube=torch.stack([frame["cube"] for frame in frames]),
        axes=axes,
        sample_times_s=tuple(frame["sample_times_s"].cpu().numpy() for frame in frames),
        adaptive_diagnostics=tuple(record.get("adaptive_diagnostics", ())),
        last_snapshot=None,
        last_compiled_scene=None,
        last_propagation=None,
        last_radar_paths=None,
        events=(),
    )
    return fields


__all__ = [
    "RESULT_FORMAT",
    "RESULT_FORMAT_VERSION",
    "load_result_fields",
    "read_frame",
    "read_metadata",
    "metadata_record",
    "run_record",
    "save_result",
    "write_frame",
    "write_metadata",
]
