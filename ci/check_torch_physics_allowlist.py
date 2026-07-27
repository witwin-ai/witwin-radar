"""G4: the Torch-physics allowlist covers the whole tree and cannot grow quietly.

`tests/test_phase6_no_torch_physics.py` scanned ONE package - `solvers/`, which
Phase 11 has since deleted - for the seven Torch calls that evaluate geometry or
a phase. Four of eleven production packages were nominally in scope and only one
was actually scanned, so `processing/`, `sensors/`, `geometry/`, `utils/`,
`timeline.py` and the rest were outside it. That is the failure mode this gate exists for and it is worth
naming precisely: **narrowing the scan is a silent way to grow the allowlist**.
Nothing had to be added to a list; a hit simply had to land in a directory
nobody was looking at.

So the scope here is the whole `witwin/` tree with an EMPTY exclusion list, and
the exclusion list is itself frozen. Every match that scope
produces is recorded in `ci/torch-physics-allowlist.json` with a category, a
reason and the ADR that permits it, keyed by
`(module, function, call, occurrences)`. Equality in both directions: an
unrecorded match fails, and a recorded match that no longer exists fails too.

The occurrence count is deliberate. `(module, function, call)` alone would let
a second `torch.cos` appear inside a function that already has one, which is
precisely how a windowing helper becomes a phase evaluator.

Three lists that live in pytest are re-frozen here from the JSON, so that
editing the test constant without editing the record - or the reverse - fails:
`FORBIDDEN_TORCH_CALLS` and `RADAR_FACADE_TORCH_PHYSICS` from
`tests/test_phase6_no_torch_physics.py`, and `FENCE_ALLOWANCES` from
`tests/processing/test_cutover.py`.

Finally `FROZEN_BASELINE_DIGEST` below covers the whole allowlist document, in
the style of `channel/ci/check_import_graph.py`. It lives in this source file
rather than in the JSON, so widening the allowlist changes a value in a second
file that a reviewer has to update on purpose.

**Classification, not repair.** A recorded entry is a statement about what the
expression IS, and several of them are recorded DEBT rather than approval. This
gate does not fix physics: moving an expression into a kernel is a numerical
change and belongs in its own commit with its own evidence.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST_PATH = REPO_ROOT / "ci" / "torch-physics-allowlist.json"

#: sha256 over the canonical JSON of the WHOLE allowlist document. It is held
#: here rather than inside the JSON on purpose: a digest a document carries is
#: a checksum, and a digest a second file carries is a decision. Recomputed and
#: printed on failure, so an intentional widening costs one copy.
FROZEN_BASELINE_DIGEST = (
    "c3d85ab84dcfd310e6291ca2176571131762e072e66397f23c321e6ce192c227"
)

SCHEMA_VERSION = 1

TOP_LEVEL_KEYS = frozenset(
    {
        "comment",
        "schema_version",
        "scanned_root",
        "excluded_paths",
        "forbidden_torch_calls",
        "fence_allowances",
        "radar_facade_torch_physics",
        "categories",
        "entries",
    }
)

ENTRY_KEYS = frozenset(
    {"module", "function", "call", "occurrences", "category", "reason", "adr"}
)


def _dotted(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _enclosing_functions(tree: ast.Module) -> dict[int, str]:
    names: dict[int, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for line in range(node.lineno, (node.end_lineno or node.lineno) + 1):
            names.setdefault(line, node.name)
    return names


def scanned_modules(root: Path, scanned_root: str, excluded: tuple[str, ...]):
    package = root / scanned_root
    modules = []
    for path in sorted(package.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(root).as_posix()
        if any(
            relative == item or relative.startswith(f"{item.rstrip('/')}/")
            for item in excluded
        ):
            continue
        modules.append(path)
    return modules


def scan(
    root: Path,
    *,
    scanned_root: str,
    excluded: tuple[str, ...],
    forbidden: tuple[str, ...],
) -> dict[tuple[str, str, str], int]:
    """`(module, function, call) -> occurrences` over the scanned scope."""

    counts: dict[tuple[str, str, str], int] = {}
    for path in scanned_modules(root, scanned_root, excluded):
        relative = path.relative_to(root).as_posix()
        source = path.read_text(encoding="utf-8").lstrip(chr(0xFEFF))
        tree = ast.parse(source, filename=str(path))
        functions = _enclosing_functions(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _dotted(node.func)
            if not name.startswith("torch."):
                continue
            if name[len("torch."):] not in forbidden:
                continue
            key = (relative, functions.get(node.lineno, "<module>"), name)
            counts[key] = counts.get(key, 0) + 1
    return counts


def allowlist_digest(document: dict) -> str:
    canonical = json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _check_schema(document: dict) -> list[str]:
    failures: list[str] = []
    if document.get("schema_version") != SCHEMA_VERSION:
        failures.append(
            f"schema_version is {document.get('schema_version')!r}, "
            f"expected {SCHEMA_VERSION}"
        )
    unknown = sorted(set(document) - TOP_LEVEL_KEYS)
    if unknown:
        failures.append(f"unknown top-level key(s): {unknown}")
    missing = sorted(TOP_LEVEL_KEYS - set(document))
    if missing:
        failures.append(f"missing top-level key(s): {missing}")

    categories = set(document.get("categories", {}))
    seen: set[tuple[str, str, str]] = set()
    for index, entry in enumerate(document.get("entries", [])):
        label = f"entries[{index}]"
        if set(entry) != ENTRY_KEYS:
            failures.append(
                f"{label}: keys {sorted(entry)}, expected {sorted(ENTRY_KEYS)}"
            )
            continue
        key = (entry["module"], entry["function"], entry["call"])
        if key in seen:
            failures.append(f"{label}: duplicate entry for {key}")
        seen.add(key)
        if entry["category"] not in categories:
            failures.append(
                f"{label}: category {entry['category']!r} is not described in "
                "'categories'"
            )
        if not str(entry["reason"]).strip():
            failures.append(f"{label}: empty reason")
        if not str(entry["adr"]).strip():
            failures.append(f"{label}: empty adr")
        if not isinstance(entry["occurrences"], int) or entry["occurrences"] < 1:
            failures.append(f"{label}: occurrences must be a positive integer")
    return failures


def _check_pytest_constants(root: Path, document: dict) -> list[str]:
    """Re-freeze the three lists that live in pytest against the record.

    Read by AST rather than imported: this gate must run before pytest, on a
    tree whose `witwin` package may not even be importable.
    """

    failures: list[str] = []

    def literals(path: Path, name: str):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == name
                for target in node.targets
            ):
                return ast.literal_eval(node.value)
        raise KeyError(f"{path}: {name} not found")

    physics = root / "tests" / "test_phase6_no_torch_physics.py"
    cutover = root / "tests" / "processing" / "test_cutover.py"

    forbidden = tuple(literals(physics, "FORBIDDEN_TORCH_CALLS"))
    recorded = tuple(document["forbidden_torch_calls"])
    if forbidden != recorded:
        failures.append(
            "tests/test_phase6_no_torch_physics.py FORBIDDEN_TORCH_CALLS is "
            f"{list(forbidden)}; the allowlist records {list(recorded)}"
        )

    facade = {tuple(pair) for pair in literals(physics, "RADAR_FACADE_TORCH_PHYSICS")}
    recorded_facade = {
        tuple(pair) for pair in document["radar_facade_torch_physics"]
    }
    if facade != recorded_facade:
        failures.append(
            "RADAR_FACADE_TORCH_PHYSICS disagrees with the allowlist: "
            f"{sorted(facade ^ recorded_facade)}"
        )

    fence = set(literals(cutover, "FENCE_ALLOWANCES"))
    recorded_fence = set(document["fence_allowances"])
    if fence != recorded_fence:
        failures.append(
            "tests/processing/test_cutover.py FENCE_ALLOWANCES disagrees with "
            f"the allowlist: {sorted(fence ^ recorded_fence)}"
        )
    return failures


def check(root: Path, allowlist_path: Path) -> list[str]:
    document = json.loads(allowlist_path.read_text(encoding="utf-8"))
    failures = _check_schema(document)
    if failures:
        return failures

    digest = allowlist_digest(document)
    if digest != FROZEN_BASELINE_DIGEST:
        failures.append(
            "the allowlist changed: FROZEN_BASELINE_DIGEST in "
            "ci/check_torch_physics_allowlist.py is "
            f"{FROZEN_BASELINE_DIGEST}, the document hashes to {digest}. "
            "Update the constant deliberately, in the same change that widens "
            "the allowlist."
        )

    measured = scan(
        root,
        scanned_root=str(document["scanned_root"]),
        excluded=tuple(document["excluded_paths"]),
        forbidden=tuple(document["forbidden_torch_calls"]),
    )
    recorded = {
        (entry["module"], entry["function"], entry["call"]): entry["occurrences"]
        for entry in document["entries"]
    }

    for key in sorted(set(measured) - set(recorded)):
        module, function, call = key
        failures.append(
            f"{module}: {function}() calls {call} {measured[key]} time(s) and is "
            "not in the allowlist; classify it (record it with a reason and an "
            "ADR, or move the expression into a kernel)"
        )
    for key in sorted(set(recorded) - set(measured)):
        module, function, call = key
        failures.append(
            f"{module}: the allowlist still records {function}() calling {call}, "
            "which no longer exists; delete the entry"
        )
    for key in sorted(set(recorded) & set(measured)):
        if recorded[key] != measured[key]:
            module, function, call = key
            failures.append(
                f"{module}: {function}() calls {call} {measured[key]} time(s), "
                f"the allowlist records {recorded[key]}"
            )

    failures.extend(_check_pytest_constants(root, document))
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--allowlist", type=Path, default=None)
    arguments = parser.parse_args(argv)

    root = arguments.root.resolve()
    allowlist_path = (
        arguments.allowlist.resolve()
        if arguments.allowlist is not None
        else root / "ci" / "torch-physics-allowlist.json"
    )

    failures = check(root, allowlist_path)
    if failures:
        print(
            f"check_torch_physics_allowlist: {len(failures)} violation(s) under {root}",
            file=sys.stderr,
        )
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1

    document = json.loads(allowlist_path.read_text(encoding="utf-8"))
    modules = scanned_modules(
        root,
        str(document["scanned_root"]),
        tuple(document["excluded_paths"]),
    )
    total = sum(entry["occurrences"] for entry in document["entries"])
    print(
        f"check_torch_physics_allowlist: {len(modules)} modules scanned under "
        f"{document['scanned_root']} with "
        f"{len(document['excluded_paths'])} exclusions; "
        f"{len(document['entries'])} recorded expressions, {total} occurrences; "
        f"digest {FROZEN_BASELINE_DIGEST[:16]}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
