"""Reject exact, non-trivial function clones in production radar modules.

The check compares normalized Python ASTs, so formatting, comments, function
names, and source locations do not hide a duplicated implementation. Functions
with fewer than three executable statements are excluded because their
repetition is usually a language-level protocol shape rather than duplicated
domain logic.
"""

from __future__ import annotations

import ast
import copy
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "witwin" / "radar"


def _fingerprint(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    body = node.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        if isinstance(body[0].value.value, str):
            body = body[1:]
    if len(body) < 3:
        return None
    normalized = copy.deepcopy(node)
    normalized.name = "_"
    normalized.body = body
    normalized.decorator_list = []
    return ast.dump(normalized, annotate_fields=True, include_attributes=False)


def find_duplicates() -> list[list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        relative = path.relative_to(ROOT).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            fingerprint = _fingerprint(node)
            if fingerprint is not None:
                groups[fingerprint].append(f"{relative}:{node.lineno}:{node.name}")
    return sorted((owners for owners in groups.values() if len(owners) > 1), key=lambda owners: owners[0])


def main() -> int:
    duplicates = find_duplicates()
    if duplicates:
        for owners in duplicates:
            print("ci/check_duplicate_code.py: exact function clone:", file=sys.stderr)
            for owner in owners:
                print(f"  - {owner}", file=sys.stderr)
        return 1
    print("ci/check_duplicate_code.py: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
