"""AST helpers shared by the static gates that read the production tree."""

from __future__ import annotations

import ast
from collections.abc import Callable
from pathlib import Path


def dotted(node: ast.AST) -> str:
    """``torch.ops.x`` for an Attribute chain rooted at a Name; ``""`` otherwise."""

    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def production_modules(root: Path) -> list[Path]:
    """Every Python file that ships inside the ``witwin`` package under ``root``."""

    package = root / "witwin"
    return sorted(path for path in package.rglob("*.py") if "__pycache__" not in path.parts)


def imported_modules(node: ast.Import | ast.ImportFrom, is_root: Callable[[str], bool]) -> list[str]:
    """The absolute module names an import statement reaches.

    A ``from X import a, b`` yields ``X`` alone when ``is_root(X)`` holds -
    the base module is already the match - and ``X.a``, ``X.b`` otherwise, so a
    name imported from a package that is itself clean can still be matched.
    Relative imports stay inside the scanned package and yield nothing.
    """

    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if node.level:
        return []
    base = node.module or ""
    if not base:
        return []
    if is_root(base):
        return [base]
    return [f"{base}.{alias.name}" for alias in node.names]


def module_name(repo: Path, path: Path) -> str:
    """``witwin.radar.x`` for ``repo/witwin/radar/x.py``; packages drop ``__init__``."""

    parts = list(path.relative_to(repo).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def resolve_relative(package: str, level: int, module: str | None) -> str:
    """The absolute target of ``from <level dots><module> import ...`` inside ``package``."""

    parts = package.split(".")
    if level > 1:
        parts = parts[: -(level - 1)]
    base = ".".join(parts)
    return base if not module else f"{base}.{module}"


def import_candidates(tree: ast.Module, package: str) -> list[str]:
    """Every dotted name the module's imports may resolve to, in source order.

    ``from X import a`` contributes both ``X`` and ``X.a`` because the caller
    cannot tell a submodule from an attribute without the file list; it filters
    against the modules it knows. ``importlib.import_module("...")`` with a
    string literal counts as an import, relative or absolute.
    """

    candidates: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            candidates.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            target = resolve_relative(package, node.level, node.module) if node.level else (node.module or "")
            candidates.append(target)
            candidates.extend(f"{target}.{alias.name}" for alias in node.names)
        elif (
            isinstance(node, ast.Call)
            and dotted(node.func) == "importlib.import_module"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            value = node.args[0].value
            candidates.append(resolve_relative(package, 1, value[1:]) if value.startswith(".") else value)
    return candidates
