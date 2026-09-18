"""Freeze the reset Radar root surface, including signatures and defaults.

The owner inventory lives in ``ci/public-api-manifest.json``. This file keeps
the executable signature snapshot of it. What the snapshot covers is not
decided here: the scope is exactly the ``modules`` list in that manifest, plus
the public members of ``Radar`` named in ``root_class_members``, both read at
import below. A module joins or leaves the snapshot by being added to or
removed from that list.
"""

from __future__ import annotations

import importlib
import inspect
import json
import re
import types
from pathlib import Path

import witwin.radar

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "ci" / "public-api-snapshot.json"
MANIFEST = json.loads((ROOT / "ci" / "public-api-manifest.json").read_text(encoding="utf-8"))
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
    entry: dict[str, object] = {"name": name, "kind": _kind(obj), "target": _target(name, obj)}
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
            # A public property with no docstring used to fail here as an
            # IndexError on an empty split, which names neither the class nor
            # the member. The requirement is real - the snapshot pins the first
            # line - so it is stated instead of relaxed.
            lines = (member.__doc__ or "").strip().splitlines()
            assert lines, f"public property {cls.__name__}.{name} has no docstring to pin"
            entry["doc_first_line"] = lines[0]
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
                "exports": [_export(name, getattr(module, name)) for name in sorted(module.__all__)],
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
    texts = [path.read_text(encoding="utf-8") for path in production + consumers if path.resolve() != declaring]
    unconsumed = [
        name for name in sorted(module.__all__) if not any(re.search(rf"\b{re.escape(name)}\b", text) for text in texts)
    ]
    assert unconsumed == []


def test_the_root_is_exactly_the_happy_path() -> None:
    """The root carries every name a complete session needs, and no more.

    It is checked against the manifest rather than a literal list here, because
    the manifest is what the three CI gates read; a second literal would be a
    second owner of the same statement. What this adds is that every root name
    resolves to the type its manifest entry names, so a facade cannot quietly
    start re-exporting something else.
    """

    assert witwin.radar.__all__ == sorted(MANIFEST["modules"]["witwin.radar"])
    for name, target in MANIFEST["modules"]["witwin.radar"].items():
        obj = getattr(witwin.radar, name)
        owner = getattr(obj, "__module__", None) or obj.__name__
        qualified = owner if owner == target else f"{owner}.{getattr(obj, '__qualname__', '')}"
        assert qualified == target, f"{name} resolves to {qualified}, not {target}"


def test_the_radar_holds_no_run_state() -> None:
    """No ``last_*`` retention site survives on the radar.

    The four typed diagnostics live on the result that produced them. A copy on
    the radar would be a second owner, and a call that raised part way through
    would leave it describing a world that call never simulated. The fifth name
    checked below, ``last_result``, has never existed on either object; it is
    here because it is the obvious name for the convenience handle that would
    reintroduce run state.
    """

    for name in ("last_result", "last_snapshot", "last_compiled_scene", "last_propagation", "last_radar_paths"):
        assert not hasattr(witwin.radar.Radar, name), f"Radar still exposes {name}"


if __name__ == "__main__":
    SNAPSHOT.write_text(json.dumps(build_snapshot(), indent=2) + "\n", encoding="utf-8")
    print(f"regenerated {SNAPSHOT}")
