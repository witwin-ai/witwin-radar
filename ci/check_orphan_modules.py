#!/usr/bin/env python
"""Fail when a production module under `witwin/radar/` is unreachable.

Ruff already reports an unused *import*. It cannot report an unused *module*:
a file whose every import is used, and which nothing imports, passes every
linter. This gate is what finds it.

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
import sys
from collections import deque
from pathlib import Path

from _ast_scan import import_candidates, module_name

PACKAGE = "witwin.radar"

# Modules a user imports directly, so no in-tree production module has to.
ENTRY_POINTS: dict[str, str] = {
    "witwin.radar": "the root facade: the flat Radar record, its waveforms and its four verbs",
    "witwin.radar.deployment": "public deployment/runtime report owner",
    "witwin.radar.smpl": "public SMPL authoring facade",
    "witwin.radar.processing": "public signal-processing facade",
    "witwin.radar.scattering": "public scatter-response owner",
    "witwin.radar.simulation": "public simulation/session result owner",
    "witwin.radar.synthesis": "public waveform synthesis facade",
}


def edges_from(path: Path, name: str, known: set[str]) -> set[str]:
    """Modules that importing `path` also imports."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    package = name if path.name == "__init__.py" else name.rpartition(".")[0]
    out = {candidate for candidate in import_candidates(tree, package) if candidate in known and candidate != name}

    # Importing a submodule imports its parent package first.
    parent = name.rpartition(".")[0]
    if parent and parent.startswith(PACKAGE) and parent in known:
        out.add(parent)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=None, help="repository root")
    arguments = parser.parse_args(argv)

    repo = Path(arguments.root) if arguments.root else Path(__file__).resolve().parents[1]
    package_dir = repo / PACKAGE.replace(".", "/")
    paths = {module_name(repo, path): path for path in sorted(package_dir.rglob("*.py"))}
    known = set(paths)

    unknown_entries = sorted(set(ENTRY_POINTS) - known)
    if unknown_entries:
        print(
            "ci/check_orphan_modules.py: ENTRY_POINTS names modules that do not exist; delete the stale entries:",
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
