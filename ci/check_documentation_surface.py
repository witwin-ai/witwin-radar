#!/usr/bin/env python
"""Check living Radar documentation against retired paths and current files."""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

PATH_TOKEN = re.compile(r"`((?:witwin/radar|ci|tests|examples|docs)/[^`\s:]+(?:\.[A-Za-z0-9]+)?)")


def _python_symbols(path: Path) -> set[str]:
    """Return module-level and class-qualified Python definitions."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()

    def visit(body: list[ast.stmt], prefix: str = "") -> None:
        for node in body:
            if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            qualified = f"{prefix}.{node.name}" if prefix else node.name
            found.add(qualified)
            if isinstance(node, ast.ClassDef):
                visit(node.body, qualified)

    visit(tree.body)
    return found


def _audit_symbol_owner_tables(repo: Path, manifest: dict) -> list[str]:
    """Resolve every ``path.py::symbol`` owner in configured living tables."""

    errors: list[str] = []
    cache: dict[Path, set[str]] = {}
    for table in manifest.get("symbol_owner_tables", ()):
        relative = table["path"]
        document = repo / relative
        if not document.is_file():
            errors.append(f"symbol-owner document is missing: {relative}")
            continue
        expected_columns = int(table["columns"])
        owner_column = int(table["owner_column"])
        for number, raw in enumerate(document.read_text(encoding="utf-8").splitlines(), 1):
            line = raw.strip()
            if not line.startswith("|") or line.startswith("|---"):
                continue
            cells = [cell.strip().strip("`") for cell in line.strip("|").split("|")]
            if len(cells) != expected_columns or "owner" in cells[owner_column].lower():
                continue
            owner = cells[owner_column]
            if "::" not in owner:
                errors.append(f"{relative}:{number} owner is not path.py::symbol: {owner!r}")
                continue
            path_text, symbol = owner.split("::", 1)
            source = repo / path_text
            if not source.is_file():
                errors.append(f"{relative}:{number} owner path is missing: {path_text!r}")
                continue
            try:
                if source not in cache:
                    cache[source] = _python_symbols(source)
                available = cache[source]
            except (OSError, SyntaxError) as error:
                errors.append(f"{relative}:{number} cannot parse owner {path_text!r}: {error}")
                continue
            if symbol not in available:
                errors.append(f"{relative}:{number} owner symbol is missing: {owner!r}")
    return errors


def audit(repo: Path) -> list[str]:
    manifest = json.loads((repo / "ci" / "documentation-manifest.json").read_text(encoding="utf-8"))
    errors: list[str] = []
    living = manifest["living"]
    for relative in living:
        path = repo / relative
        if not path.is_file():
            errors.append(f"living document is missing: {relative}")
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        retired_entry = manifest.get("retired_token_exemptions", {}).get(relative, {})
        missing_entry = manifest.get("missing_path_exemptions", {}).get(relative, {})
        retired_exemptions = set(retired_entry.get("tokens", ()))
        missing_exemptions = set(missing_entry.get("paths", ()))
        for kind, entry in (("retired token", retired_entry), ("missing path", missing_entry)):
            if entry and not str(entry.get("reason", "")).strip():
                errors.append(f"{relative} has a {kind} exemption without a reason")
        for token in manifest["retired_living_tokens"]:
            if token in text and token not in retired_exemptions:
                errors.append(f"{relative} names retired current token {token!r}")
        for match in PATH_TOKEN.finditer(text):
            claimed = match.group(1).rstrip(".,)")
            if "*" in claimed or claimed.endswith("/"):
                continue
            if claimed in missing_exemptions:
                continue
            if not (repo / claimed).exists():
                errors.append(f"{relative} names missing current path {claimed!r}")

    errors.extend(_audit_symbol_owner_tables(repo, manifest))

    prefixes = tuple(manifest["historical_prefixes"])
    overlap = [
        relative for relative in living if any(relative == prefix or relative.startswith(prefix) for prefix in prefixes)
    ]
    if overlap:
        errors.append(f"documents classified as both living and historical: {overlap}")
    return sorted(set(errors))


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    errors = audit(repo)
    if errors:
        for error in errors:
            print(f"ci/check_documentation_surface.py: {error}", file=sys.stderr)
        return 1
    print("ci/check_documentation_surface.py: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
