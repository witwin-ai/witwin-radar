"""G3: exactly one module reaches the dispatcher, and the callers are frozen.

Phase 10 work item 7 forbids raw native access. Radar has no
`runtime.symbols`-style indirection the way Channel does; every kernel facade
imports
ative_ops as _ops` from the lazy CUDA boundary and reaches symbols
through the validated table it returns. That is fine only while the imported callable is
the loader that validates identity first. The failure this gate exists to stop
is a new module writing `torch.ops._radar_native.<x>` directly, which loads
nothing, validates nothing, and works - right up to the first stale binary.

Two frozen statements, both by EQUALITY rather than containment:

* **the dispatcher owner set.** Every AST reference to `torch.ops` or to
  `torch.utils.cpp_extension`, anywhere under `witwin/`, must come from
  `witwin/radar/cuda/runtime.py`. Not "must be in an allowlist" - must equal
  that one module. A new owner fails, and so does the owner disappearing.
* **the loader's consumers.** The modules that call
  `witwin.radar.cuda.runtime.build_extension()` are recorded one by one with
  the reason each one holds a handle. A tenth consumer is a decision; a stale
  entry for a deleted one is a hole.

`witwin/radar/cuda/runtime.py` is the sole dispatcher, JIT, and identity
validation owner. It remains importable without CUDA because dispatcher loading is
lazy; every load route must validate sidecars and runtime identity before calling
`torch.ops.load_library`. The exact owner set and exact consumer set are frozen
here.

Docstrings are not access. This gate scans executable expressions; prose that names
`torch.ops` is governed separately.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The single module allowed to name the dispatcher or the JIT compiler in
#: code. Frozen by equality against the measured set.
DISPATCHER_OWNERS = frozenset({"witwin/radar/cuda/runtime.py"})

#: The loader's sibling, which must never gain dispatcher access: it runs
#: before the library is loaded and must import without CUDA.

#: Every module that takes a handle from the loader, and why. Frozen by
#: equality. Seven kernel facades bind the lazy `native_ops` bridge as `_ops`;
#: `deployment.py` is the identity reporter - `build_info()` must come from the
#: loader that VALIDATED the record, not from a re-read of the sidecar, which
#: would answer a different question.
EXPECTED_LOADER_CONSUMERS = {
    "witwin/radar/cuda/__init__.py": "single lazy bridge from kernel facades to the validated runtime",
    "witwin/radar/deployment.py": "public build_info(), from the validated record",
    "witwin/radar/frontend.py": "frontend_chain facade",
    "witwin/radar/paths.py": "two_way_join facade",
    "witwin/radar/scattering.py": "scatter_response_aspect facade",
    "witwin/radar/sensors.py": "sensor_weight facade",
    "witwin/radar/synthesis/fmcw.py": "fmcw_beat_synthesis facade",
    "witwin/radar/synthesis/ofdm.py": "ofdm_cfr_synthesis facade",
    "witwin/radar/synthesis/pulsed.py": "pulsed_echo_synthesis facade",
}


def _dotted(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def production_modules(root: Path) -> list[Path]:
    package = root / "witwin"
    return sorted(path for path in package.rglob("*.py") if "__pycache__" not in path.parts)


def _dispatcher_references(tree: ast.Module) -> list[tuple[int, str]]:
    """`(line, expression)` for every dispatcher or JIT-compiler reference."""

    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            name = _dotted(node)
            if name.startswith("torch.ops") or name.startswith("torch.utils.cpp_extension"):
                found.append((node.lineno, name))
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("torch.utils.cpp_extension"):
                    found.append((node.lineno, alias.name))
            continue
        if isinstance(node, ast.ImportFrom) and not node.level:
            base = node.module or ""
            if base.startswith("torch.utils.cpp_extension") or base == "torch.utils":
                for alias in node.names:
                    if base.startswith("torch.utils.cpp_extension") or (alias.name == "cpp_extension"):
                        found.append((node.lineno, f"{base}.{alias.name}"))
    # `torch.ops.x.y` yields nested Attribute nodes; keep the outermost per line.
    longest: dict[int, str] = {}
    for line, name in found:
        if len(name) > len(longest.get(line, "")):
            longest[line] = name
    return sorted(longest.items())


def _calls_build_extension(tree: ast.Module) -> bool:
    """True when the module asks the loader for a handle.

    The dot boundary matters: `build.py`'s own private `_build_extension` and
    `_jit_build_extension` are the loader implementing itself, not a consumer
    taking a handle, and a suffix match without the boundary would report the
    owner as its own tenth consumer.
    """

    imported_names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        module = node.module or ""
        for alias in node.names:
            if (module, alias.name) in {("cuda.runtime", "build_extension"), ("cuda", "native_ops")}:
                imported_names.add(alias.asname or alias.name)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _dotted(node.func)
        if name in imported_names or name == "build_extension" or name.endswith(".build_extension"):
            return True
    return False


def scan(root: Path) -> tuple[dict[str, list[tuple[int, str]]], set[str]]:
    dispatcher: dict[str, list[tuple[int, str]]] = {}
    consumers: set[str] = set()
    for path in production_modules(root):
        relative = path.relative_to(root).as_posix()
        source = path.read_text(encoding="utf-8").lstrip(chr(0xFEFF))
        tree = ast.parse(source, filename=str(path))
        references = _dispatcher_references(tree)
        if references:
            dispatcher[relative] = references
        if _calls_build_extension(tree):
            consumers.add(relative)
    return dispatcher, consumers


def check(root: Path) -> list[str]:
    dispatcher, consumers = scan(root)
    failures: list[str] = []

    owners = set(dispatcher)
    for module in sorted(owners - DISPATCHER_OWNERS):
        lines = ", ".join(f"{line}:{name}" for line, name in dispatcher[module])
        failures.append(
            f"{module}: reaches the dispatcher directly ({lines}); the loader "
            "at witwin/radar/cuda/runtime.py is the only owner"
        )
    for module in sorted(DISPATCHER_OWNERS - owners):
        failures.append(
            f"{module}: recorded as the dispatcher owner but names neither "
            "torch.ops nor torch.utils.cpp_extension; the record is stale"
        )

    expected = set(EXPECTED_LOADER_CONSUMERS)
    for module in sorted(consumers - expected):
        failures.append(
            f"{module}: calls build_extension() but is not a recorded consumer; "
            "add it to EXPECTED_LOADER_CONSUMERS with its reason"
        )
    for module in sorted(expected - consumers):
        failures.append(
            f"{module}: recorded as a loader consumer "
            f"({EXPECTED_LOADER_CONSUMERS[module]}) but calls no "
            "build_extension(); the record is stale"
        )
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    arguments = parser.parse_args(argv)

    root = arguments.root.resolve()
    failures = check(root)
    if failures:
        print(f"check_raw_native_access: {len(failures)} violation(s) under {root}", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1

    print(
        "check_raw_native_access: dispatcher owner "
        f"{sorted(DISPATCHER_OWNERS)[0]}; "
        f"{len(EXPECTED_LOADER_CONSUMERS)} recorded loader consumers; "
        "identity validation and dispatcher loading share the single runtime owner"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
