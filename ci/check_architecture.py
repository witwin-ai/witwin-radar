#!/usr/bin/env python
"""Validate the target Radar module inventory and executable import graph."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys


def _module(repo: Path, path: Path) -> str:
    parts = list(path.relative_to(repo).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _resolve(package: str, level: int, module: str | None) -> str:
    parts = package.split(".")
    if level > 1:
        parts = parts[: -(level - 1)]
    base = ".".join(parts)
    return base if not module else f"{base}.{module}"


def _edges(path: Path, name: str, known: set[str]) -> tuple[set[str], bool]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    package = name if path.name == "__init__.py" else name.rpartition(".")[0]
    internal: set[str] = set()
    imports_channel = False

    def note(candidate: str) -> None:
        nonlocal imports_channel
        if candidate == "witwin.channel" or candidate.startswith("witwin.channel."):
            imports_channel = True
        cursor = candidate
        while cursor:
            if cursor in known and cursor != name:
                internal.add(cursor)
                break
            cursor = cursor.rpartition(".")[0]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                note(alias.name)
        elif isinstance(node, ast.ImportFrom):
            target = (
                _resolve(package, node.level, node.module)
                if node.level
                else (node.module or "")
            )
            note(target)
            for alias in node.names:
                note(f"{target}.{alias.name}")
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "importlib"
            and node.func.attr == "import_module"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            value = node.args[0].value
            note(_resolve(package, 1, value[1:]) if value.startswith(".") else value)
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


def audit(repo: Path) -> list[str]:
    manifest = json.loads(
        (repo / "ci" / "architecture-manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("schema_version") != 1:
        return ["architecture manifest schema_version must be 1"]
    paths = {
        _module(repo, path): path
        for path in sorted((repo / "witwin" / "radar").rglob("*.py"))
    }
    known = set(paths)
    target = set(manifest["target_modules"])
    errors = [
        *(f"missing target module: {name}" for name in sorted(target - known)),
        *(f"unexpected production module: {name}" for name in sorted(known - target)),
    ]
    owners = manifest.get("concept_owners", {})
    duplicate_owners = [
        name
        for name, count in __import__("collections").Counter(owners.values()).items()
        if count > 1
    ]
    errors.extend(
        f"one module owns multiple declared concepts without an explicit merge: {name}"
        for name in sorted(duplicate_owners)
    )
    for concept, owner in sorted(owners.items()):
        if owner not in target:
            errors.append(f"concept {concept!r} names non-target owner {owner!r}")

    graph: dict[str, set[str]] = {}
    channel_importers = []
    for name, path in paths.items():
        edges, imports_channel = _edges(path, name, known)
        graph[name] = edges
        if imports_channel:
            channel_importers.append(name)
    expected = [manifest["channel_importer"]]
    if sorted(channel_importers) != expected:
        errors.append(
            "Channel executable importers differ: "
            f"expected {expected}, got {sorted(channel_importers)}"
        )
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
