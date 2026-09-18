#!/usr/bin/env python
"""Validate the target Radar module inventory and executable import graph."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import Counter
from pathlib import Path

from _ast_scan import import_candidates, module_name


def _edges(path: Path, name: str, known: set[str]) -> tuple[set[str], bool]:
    """Internal modules `path` imports, and whether it reaches `witwin.channel`."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    package = name if path.name == "__init__.py" else name.rpartition(".")[0]
    internal: set[str] = set()
    imports_channel = False
    for candidate in import_candidates(tree, package):
        if candidate == "witwin.channel" or candidate.startswith("witwin.channel."):
            imports_channel = True
        cursor = candidate
        while cursor:
            if cursor in known and cursor != name:
                internal.add(cursor)
                break
            cursor = cursor.rpartition(".")[0]
    return internal, imports_channel


def _cycles(graph: dict[str, set[str]]) -> list[tuple[str, ...]]:
    found: set[tuple[str, ...]] = set()
    active: list[str] = []
    active_set: set[str] = set()
    done: set[str] = set()

    def visit(node: str) -> None:
        if node in done:
            return
        if node in active_set:
            index = active.index(node)
            cycle = active[index:] + [node]
            body = cycle[:-1]
            start = min(range(len(body)), key=lambda i: body[i])
            canonical = tuple(body[start:] + body[:start])
            found.add(canonical)
            return
        active.append(node)
        active_set.add(node)
        for child in sorted(graph.get(node, ())):
            visit(child)
        active.pop()
        active_set.remove(node)
        done.add(node)

    for module in sorted(graph):
        visit(module)
    return sorted(found)


def _audit_surface_split(repo: Path, manifest: dict, target: set[str]) -> list[str]:
    """Tie ``public_facades`` and ``internal_modules`` to the manifests that decide them.

    Both keys sat here with no reader at all, so the split they declare could
    disagree with `ci/public-api-manifest.json` indefinitely and nothing would
    say so - which is how `witwin.radar.frontend` came to be called a public
    owner in one file and an internal module in this one. The public list is
    checked for EQUALITY against the public API manifest's modules, because a
    facade this file names and that file does not export is a contradiction in
    one direction and an unlisted public module in the other. The internal list
    is checked for the weaker property it actually claims: target modules, not
    public ones.

    Both keys are optional, so a caller auditing a partial tree with ``--root``
    is not forced to restate a surface it is not testing.
    """

    errors: list[str] = []
    facades = manifest.get("public_facades")
    internal = manifest.get("internal_modules")
    if facades is not None:
        public_manifest = repo / "ci" / "public-api-manifest.json"
        if not public_manifest.is_file():
            errors.append("public_facades is declared but ci/public-api-manifest.json is missing")
        else:
            exported = list(json.loads(public_manifest.read_text(encoding="utf-8"))["modules"])
            if sorted(facades) != sorted(exported):
                errors.append(f"public_facades differ from the public API manifest: {sorted(facades)} vs {exported}")
    if internal is not None:
        errors.extend(f"internal module is not a target module: {name}" for name in sorted(set(internal) - target))
        if facades is not None:
            both = sorted(set(internal) & set(facades))
            errors.extend(f"module is declared both public and internal: {name}" for name in both)
    return errors


def _audit_public_owners(repo: Path, target: set[str]) -> list[str]:
    """Every public export must resolve to a symbol owned by a target module.

    `ci/check_public_api_manifest.py` already rejects one target exposed under
    two names; this is the other direction, an exposure whose owner is not in
    the module inventory at all.
    """

    public_manifest = repo / "ci" / "public-api-manifest.json"
    if not public_manifest.is_file():
        return []
    errors: list[str] = []
    for module, exports in json.loads(public_manifest.read_text(encoding="utf-8"))["modules"].items():
        for name, canonical in exports.items():
            owner = canonical.rpartition(".")[0]
            if owner not in target:
                errors.append(f"public exposure {module}.{name} names non-target owner module {owner}")
    return errors


def audit(repo: Path) -> list[str]:
    manifest = json.loads((repo / "ci" / "architecture-manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        return ["architecture manifest schema_version must be 1"]
    paths = {module_name(repo, path): path for path in sorted((repo / "witwin" / "radar").rglob("*.py"))}
    known = set(paths)
    target = set(manifest["target_modules"])
    errors = [
        *(f"missing target module: {name}" for name in sorted(target - known)),
        *(f"unexpected production module: {name}" for name in sorted(known - target)),
    ]
    owners = manifest.get("concept_owners", {})
    duplicate_owners = [name for name, count in Counter(owners.values()).items() if count > 1]
    errors.extend(
        f"one module owns multiple declared concepts without an explicit merge: {name}"
        for name in sorted(duplicate_owners)
    )
    for concept, owner in sorted(owners.items()):
        if owner not in target:
            errors.append(f"concept {concept!r} names non-target owner {owner!r}")
    errors.extend(_audit_surface_split(repo, manifest, target))
    errors.extend(_audit_public_owners(repo, target))

    graph: dict[str, set[str]] = {}
    channel_importers = []
    for name, path in paths.items():
        edges, imports_channel = _edges(path, name, known)
        graph[name] = edges
        if imports_channel:
            channel_importers.append(name)
    expected = [manifest["channel_importer"]]
    if sorted(channel_importers) != expected:
        errors.append(f"Channel executable importers differ: expected {expected}, got {sorted(channel_importers)}")
    for cycle in _cycles(graph):
        errors.append("internal import cycle: " + " -> ".join((*cycle, cycle[0])))
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None)
    args = parser.parse_args(argv)
    repo = Path(args.root) if args.root else Path(__file__).resolve().parents[1]
    errors = audit(repo)
    if errors:
        for error in errors:
            print(f"ci/check_architecture.py: {error}", file=sys.stderr)
        return 1
    print("ci/check_architecture.py: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
