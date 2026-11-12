"""Find the package's ``torch.autograd.Function`` owners, and the modules the native manifest says must hold them.

The structural AD tests check every owner rather than a hand-counted inventory.
What keeps such a scan from passing vacuously is a second, independent answer:
every Python module that calls a native ``_backward`` operator, as
``ci/native-binding-manifest.json`` derives it from the registrations, must
define at least one ``Function``, and no other module may.
"""

from __future__ import annotations

import ast
import json
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
PACKAGE = REPO_ROOT / "witwin" / "radar"


def production_modules() -> list[pathlib.Path]:
    return sorted(path for path in PACKAGE.rglob("*.py") if "__pycache__" not in path.parts)


def autograd_functions() -> list[tuple[str, ast.ClassDef]]:
    """``(path relative to the package, class)`` for every ``*.Function`` subclass."""

    found = []
    for path in production_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and any(
                isinstance(base, ast.Attribute) and base.attr == "Function" for base in node.bases
            ):
                found.append((path.relative_to(PACKAGE).as_posix(), node))
    return found


def native_backward_owners() -> set[str]:
    """Package-relative modules that call a native backward, per the generated binding manifest."""

    manifest = json.loads((REPO_ROOT / "ci" / "native-binding-manifest.json").read_text(encoding="utf-8"))
    prefix = "witwin/radar/"
    return {row["python_owner"].removeprefix(prefix) for row in manifest["operators"] if row["ad_role"] == "backward"}


def method(function: ast.ClassDef, name: str) -> ast.FunctionDef | None:
    matches = [node for node in function.body if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(matches) <= 1, (function.name, name)
    return matches[0] if matches else None
