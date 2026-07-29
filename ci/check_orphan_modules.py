#!/usr/bin/env python
"""Fail when a production module under `witwin/radar/` is unreachable.

Ruff already reports an unused *import*. It cannot report an unused *module*,
and that gap is not hypothetical: `witwin/radar/timeline.py` shipped for four
phases after its last production consumer went away, because every import
inside it was used and nothing anywhere asked whether anyone imported the file.
The Phase-11 deletion found it by hand. This gate is what finds the next one.

The question it asks is reachability, not "does anyone import this". A pair of
dead modules that import each other both have an importer, and a whole dead
subpackage whose `__init__` imports its own members is entirely self-supporting.
So the check starts at the declared entry points and walks the import graph:

  - a module reaches every module it imports (absolute or relative);
  - a module reaches its parent package, because importing `a.b.c` imports
    `a.b` first - this is what keeps package `__init__.py` files honest instead
    of blanket-exempt;
Anything left unvisited is an orphan. `ENTRY_POINTS` is the allowlist, and it is
deliberately short: each entry is a module a *user* imports directly, so nothing
in the tree needs to. Adding to it is a decision about the public surface, which
is why every entry carries its reason.

Tests are not importers. A module kept alive only by its own tests is exactly
the thing this gate exists to surface; if such a module is genuinely public, it
belongs in `ENTRY_POINTS` with a reason.
"""

from __future__ import annotations

import argparse
import ast
from collections import deque
from pathlib import Path
import sys


PACKAGE = "witwin.radar"

# Modules a user imports directly, so no in-tree production module has to.
ENTRY_POINTS: dict[str, str] = {
    "witwin.radar": "minimal Radar/RadarConfig system facade",
    "witwin.radar.capabilities": "public capability report owner",
    "witwin.radar.deployment": "public deployment/runtime report owner",
    "witwin.radar.frontend": "public receiver-frontend owner",
    "witwin.radar.smpl": "public SMPL authoring facade",
    "witwin.radar.processing": "public signal-processing facade",
    "witwin.radar.scattering": "public scatter-response owner",
    "witwin.radar.sensors": "public sensor contract owner",
    "witwin.radar.simulation": "public simulation/session result owner",
    "witwin.radar.synthesis": "public waveform synthesis facade",
}


def module_name(root: Path, path: Path) -> str:
    parts = list(path.relative_to(root).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _package_of(name: str, is_package: bool) -> str:
    if is_package:
        return name
    return name.rpartition(".")[0]


def _resolve(base_package: str, level: int, module: str | None) -> str:
    parts = base_package.split(".")
    if level > 1:
        parts = parts[: -(level - 1)]
    base = ".".join(parts)
    if not module:
        return base
    return f"{base}.{module}"


def edges_from(path: Path, name: str, known: set[str]) -> set[str]:
    """Modules that importing `path` also imports."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    is_package = path.name == "__init__.py"
    base_package = _package_of(name, is_package)
    out: set[str] = set()

    def note(candidate: str) -> None:
        if candidate in known and candidate != name:
            out.add(candidate)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                note(alias.name)
        elif isinstance(node, ast.ImportFrom):
            target = (
                _resolve(base_package, node.level, node.module)
                if node.level
                else (node.module or "")
            )
            note(target)
            for alias in node.names:
                note(f"{target}.{alias.name}")

    # Importing a submodule imports its parent package first.
    parent = name.rpartition(".")[0]
    if parent and parent.startswith(PACKAGE):
        note(parent)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=None, help="repository root")
    arguments = parser.parse_args(argv)

    repo = Path(arguments.root) if arguments.root else Path(__file__).resolve().parents[1]
    package_dir = repo / PACKAGE.replace(".", "/")
    paths = {
        module_name(repo, path): path for path in sorted(package_dir.rglob("*.py"))
    }
    known = set(paths)

    unknown_entries = sorted(set(ENTRY_POINTS) - known)
    if unknown_entries:
        print(
            "ci/check_orphan_modules.py: ENTRY_POINTS names modules that do not "
            "exist; delete the stale entries:",
            file=sys.stderr,
        )
        for name in unknown_entries:
            print(f"  {name}", file=sys.stderr)
        return 1

    graph = {name: edges_from(path, name, known) for name, path in paths.items()}

    visited: set[str] = set()
    queue = deque(sorted(ENTRY_POINTS))
    while queue:
        name = queue.popleft()
        if name in visited:
            continue
        visited.add(name)
        queue.extend(sorted(graph[name] - visited))

    orphans = sorted(known - visited)
    if orphans:
        print(
            "ci/check_orphan_modules.py: unreachable production module(s). No "
            "production module imports these, directly or transitively, from "
            "any entry point. Delete them, or - if a user imports one "
            "directly - add it to ENTRY_POINTS with the reason:",
            file=sys.stderr,
        )
        for name in orphans:
            print(f"  {name}  ({paths[name].relative_to(repo).as_posix()})", file=sys.stderr)
        return 1

    print(
        f"ci/check_orphan_modules.py: OK - {len(known)} production modules, "
        f"all reachable from {len(ENTRY_POINTS)} declared entry points."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
