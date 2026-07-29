"""Freeze the reset Radar root surface, including signatures and defaults.

The target owner inventory lives in ``ci/public-api-manifest.json``. This file
keeps the executable signature snapshot for the part of that target that is
already live: the package root and the public members of ``Radar``. Owner
facades join this generator when their concept-axis move lands.
"""

from __future__ import annotations

import importlib
import inspect
import json
from pathlib import Path
import re
import sys
import types

import witwin.radar


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "ci" / "public-api-snapshot.json"
MANIFEST = json.loads(
    (ROOT / "ci" / "public-api-manifest.json").read_text(encoding="utf-8")
)
PUBLIC_MODULES = tuple(MANIFEST["modules"])
PUBLIC_CLASSES = tuple(MANIFEST["root_class_members"])


def _kind(obj: object) -> str:
    if inspect.isclass(obj):
        return "class"
    if inspect.isfunction(obj) or inspect.isbuiltin(obj):
        return "function"
    if isinstance(obj, types.UnionType):
        return "union"
    if isinstance(obj, property):
        return "property"
    return "value"


def _target(name: str, obj: object) -> str:
    module = getattr(obj, "__module__", None)
    qualname = getattr(obj, "__qualname__", None)
    if module and qualname:
        return f"{module}.{qualname}"
    return f"<{type(obj).__name__}> {name}"


def _signature(obj: object) -> str | None:
    if not (inspect.isclass(obj) or inspect.isfunction(obj)):
        return None
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return None


def _export(name: str, obj: object) -> dict[str, object]:
    entry: dict[str, object] = {
        "name": name,
        "kind": _kind(obj),
        "target": _target(name, obj),
    }
    if (signature := _signature(obj)) is not None:
        entry["signature"] = signature
    return entry


def _class_members(cls: type) -> list[dict[str, object]]:
    members = []
    for name, member in sorted(vars(cls).items()):
        if name.startswith("_"):
            continue
        entry: dict[str, object] = {"name": name, "kind": _kind(member)}
        if isinstance(member, property):
            entry["doc_first_line"] = (member.__doc__ or "").strip().splitlines()[0]
        elif (signature := _signature(member)) is not None:
            entry["signature"] = signature
        members.append(entry)
    return members


def build_snapshot() -> dict[str, object]:
    modules = []
    for module_name in PUBLIC_MODULES:
        module = importlib.import_module(module_name)
        modules.append(
            {
                "module": module_name,
                "exports": [
                    _export(name, getattr(module, name))
                    for name in sorted(module.__all__)
                ],
            }
        )
    classes = []
    for dotted in PUBLIC_CLASSES:
        module_name, _, attribute = dotted.rpartition(".")
        cls = getattr(importlib.import_module(module_name), attribute)
        classes.append({"class": dotted, "members": _class_members(cls)})
    return {
        "schema_version": 2,
        "generator": "tests/test_public_api_snapshot.py::build_snapshot/v2",
        "modules": modules,
        "classes": classes,
    }


def test_the_public_surface_matches_the_frozen_snapshot() -> None:
    expected = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    assert expected["schema_version"] == 2
    assert [entry["module"] for entry in expected["modules"]] == list(PUBLIC_MODULES)
    assert build_snapshot() == expected


def _source_files() -> tuple[list[Path], list[Path]]:
    production = sorted((ROOT / "witwin" / "radar").rglob("*.py"))
    consumers = sorted((ROOT / "tests").rglob("*.py"))
    consumers += sorted((ROOT / "examples").glob("*.py"))
    return production, consumers


def test_every_root_export_has_a_consumer() -> None:
    module = witwin.radar
    declaring = Path(module.__file__).resolve()
    production, consumers = _source_files()
    texts = [
        path.read_text(encoding="utf-8")
        for path in production + consumers
        if path.resolve() != declaring
    ]
    unconsumed = [
        name
        for name in sorted(module.__all__)
        if not any(re.search(rf"\b{re.escape(name)}\b", text) for text in texts)
    ]
    assert unconsumed == []


def test_the_root_is_exactly_the_system_api() -> None:
    assert witwin.radar.__all__ == ["Radar", "RadarConfig"]
    assert witwin.radar.Radar.__module__ == "witwin.radar.radar"
    assert witwin.radar.RadarConfig.__module__ == "witwin.radar.radar"


if __name__ == "__main__":
    SNAPSHOT.write_text(
        json.dumps(build_snapshot(), indent=2) + "\n", encoding="utf-8"
    )
    print(f"regenerated {SNAPSHOT}")
