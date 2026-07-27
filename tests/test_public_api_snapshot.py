"""Freeze the public radar surface, and refuse an export nobody consumes.

Two questions, one file, because they are the same question asked from opposite
ends.

**Did the public surface move?** `ci/public-api-snapshot.json` records every
name in `__all__` for the three public modules, where each name resolves to, and
- for callables - its exact signature, plus the public method and property set
of `Radar`. Channel has had this since ADR-003 (`channel/ci/public-api-snapshot.json`);
radar has not, which is why "public APIs, docs, examples and package metadata
consistent" was unfalsifiable here. A diff in this file is not a failure, it is a
question: regenerate it deliberately (`python tests/test_public_api_snapshot.py`)
in the same commit as the change and the migration note.

**Is the surface bigger than the tree?** Every exported name must be reachable
from something other than the `__init__.py` that exports it: a production module
that imports it, or a test that exercises it. `SamplingMode`, `MotionSampling`
and `Timeline` all sat in `witwin.radar.__all__` through four phases with zero
consumers of any kind - deleted in Phase 11 by hand, after a manual survey.
Nothing in CI would have noticed.

The nine `witwin.core` geometry names re-exported from the radar root are the
one interesting case. They have no production consumer by construction: they are
a convenience alias so a caller can write `from witwin.radar import Box`. Their
contract is therefore an identity pin - each name must BE the Core object, not a
radar-side copy - and that pin is the last test in this file. It is a real
assertion about the re-export, not a mention that satisfies the scan: if radar
ever grew its own `Box`, the pin fails and acceptance criterion 1 ("one logical
Core Scene") fails with it.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import re
import sys
import types

import pytest

import witwin.core
import witwin.radar
import witwin.radar.processing
import witwin.radar.sigproc


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "ci" / "public-api-snapshot.json"

#: The modules whose `__all__` is a compatibility promise. `witwin.radar` is the
#: package root; `witwin.radar.processing` is the Phase-8 post-processing facade
#: that the examples and the pipeline guide use; `witwin.radar.sigproc` is the
#: pre-Phase-8 surface retained as re-export adapters over it.
PUBLIC_MODULES = (
    "witwin.radar",
    "witwin.radar.processing",
    "witwin.radar.sigproc",
)

#: Classes whose public member set is frozen alongside the module exports. A
#: class in `__all__` freezes its name and its module; without this, `Radar`
#: could lose `simulate` and the snapshot would not move.
PUBLIC_CLASSES = ("witwin.radar.Radar",)

#: `witwin.core` names re-exported from the radar root for convenience. They are
#: exempt from the consumer scan and covered by the identity pin instead.
CORE_RE_EXPORTS = {
    "Material": "PhysicalMaterial",
    "Structure": "Structure",
    "GeometryBase": "GeometryBase",
    "Mesh": "Mesh",
    "Box": "Box",
    "Sphere": "Sphere",
    "Cylinder": "Cylinder",
    "Cone": "Cone",
    "Ellipsoid": "Ellipsoid",
    "Pyramid": "Pyramid",
    "Prism": "Prism",
    "Torus": "Torus",
    "HollowBox": "HollowBox",
}


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
    if isinstance(obj, types.UnionType):
        return " | ".join(
            f"{part.__module__}.{part.__qualname__}" for part in obj.__args__
        )
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
    signature = _signature(obj)
    if signature is not None:
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
        else:
            signature = _signature(member)
            if signature is not None:
                entry["signature"] = signature
        members.append(entry)
    return members


def build_snapshot() -> dict[str, object]:
    modules = []
    for module_name in PUBLIC_MODULES:
        module = sys.modules[module_name]
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
        cls = getattr(sys.modules[module_name], attribute)
        classes.append({"class": dotted, "members": _class_members(cls)})
    return {
        "schema_version": 1,
        "generator": "tests/test_public_api_snapshot.py::build_snapshot/v1",
        "modules": modules,
        "classes": classes,
    }


def test_the_public_surface_matches_the_frozen_snapshot() -> None:
    expected = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    assert expected["schema_version"] == 1
    assert [entry["module"] for entry in expected["modules"]] == list(PUBLIC_MODULES)
    assert build_snapshot() == expected


def _source_files() -> tuple[list[Path], list[Path]]:
    production = sorted((ROOT / "witwin" / "radar").rglob("*.py"))
    consumers = sorted((ROOT / "tests").rglob("*.py"))
    consumers += sorted((ROOT / "examples").glob("*.py"))
    return production, consumers


@pytest.mark.parametrize("module_name", PUBLIC_MODULES)
def test_every_export_has_a_consumer_or_a_contract_test(module_name: str) -> None:
    """No name may sit in `__all__` with nothing anywhere using it."""

    module = sys.modules[module_name]
    declaring = Path(module.__file__).resolve()
    production, consumers = _source_files()
    texts = [
        (path, path.read_text(encoding="utf-8"))
        for path in production + consumers
        if path.resolve() != declaring
    ]

    unconsumed = []
    for name in sorted(module.__all__):
        if module_name == "witwin.radar" and name in CORE_RE_EXPORTS:
            continue
        pattern = re.compile(rf"\b{re.escape(name)}\b")
        if not any(pattern.search(text) for _, text in texts):
            unconsumed.append(name)

    assert not unconsumed, (
        f"{module_name}.__all__ exports names that nothing in witwin/radar/, "
        f"tests/ or examples/ names: {unconsumed}. Either give the export a "
        f"consumer or a contract test, or delete it from __all__."
    )


def test_the_core_re_exports_are_the_core_objects() -> None:
    """The radar root aliases Core geometry; it must never shadow it."""

    for exported, core_name in CORE_RE_EXPORTS.items():
        assert getattr(witwin.radar, exported) is getattr(witwin.core, core_name), (
            f"witwin.radar.{exported} is not witwin.core.{core_name}. The radar "
            f"root re-exports Core's world types as an alias; a radar-side copy "
            f"would be a second owner of the logical world."
        )


def test_the_union_alias_is_core_geometry_plus_the_radar_body() -> None:
    """`Geometry` is the one radar-side widening of a Core type."""

    members = set(witwin.radar.Geometry.__args__)
    assert witwin.radar.SMPLBody in members
    assert members - {witwin.radar.SMPLBody} == set(witwin.core.Geometry.__args__)


def test_a_lazy_export_means_the_same_object_on_every_access() -> None:
    """`capabilities` is both a submodule name and the function it exports.

    Importing `witwin.radar.capabilities` binds the SUBMODULE onto the package,
    so a lazy `__getattr__` that only returned the function handed out the
    function once and the module thereafter - and `from witwin.radar import
    capabilities` got the module from the start, because the fromlist `hasattr`
    probe performed the shadowing import before the name was read. The
    resolution is memoised into the package globals after the import for
    exactly this reason.
    """

    import subprocess

    for name in ("build_info", "capabilities", "runtime_diagnostics"):
        first = getattr(witwin.radar, name)
        assert getattr(witwin.radar, name) is first
        assert getattr(witwin.radar, name) is first
        assert not inspect.ismodule(first)

    # A fresh process, because the memoisation above is process state and the
    # `from ... import` path is a different opcode.
    probe = (
        "from witwin.radar import capabilities;"
        "import witwin.radar;"
        "assert capabilities is witwin.radar.capabilities;"
        "assert callable(capabilities) and not hasattr(capabilities, '__file__');"
        "assert capabilities()['schema_version'] == 1;"
        "print('ok')"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"


def test_removed_names_raise_with_a_replacement_rather_than_resolving() -> None:
    """A deleted export must not come back as a bare AttributeError."""

    for name in ("Tracer", "Scene", "Timeline", "Solver", "TraceResult"):
        with pytest.raises(AttributeError) as raised:
            getattr(witwin.radar, name)
        assert "has been removed" in str(raised.value)
        assert name not in witwin.radar.__all__


if __name__ == "__main__":
    SNAPSHOT.write_text(
        json.dumps(build_snapshot(), indent=2) + "\n", encoding="utf-8"
    )
    print(f"regenerated {SNAPSHOT}")
