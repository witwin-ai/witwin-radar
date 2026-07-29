"""G1: the radar production tree names no ray-tracing runtime, by any route.

Phase 10 work item 7 forbids `drjit`, `rayd.drjit` and `@dr.wrap` from
production. Four separate scans, because an import statement is only the
easiest of the four ways to reach one:

* **imports** (AST). `import drjit`, `from rayd.drjit import x`,
  `from rayd import drjit`. Prefix-matched, so `rayd.drjit.foo` is caught by
  the `rayd` entry and does not need its own.
* **decorators** (AST). `@dr.wrap(...)` is how a Dr.Jit boundary is declared,
  and it is an ATTRIBUTE expression, not an import - a module that received
  `dr` as a parameter, or aliased it, would carry a live Dr.Jit boundary past
  an import-only scan.
* **string literals** (AST). `importlib.import_module("drjit")` is an import
  that no import scan sees, and it is exactly what a "temporary" fallback
  looks like. The cost of closing that hole is that ordinary PROSE about the
  removal matches too, so the two prose occurrences that exist today are
  frozen by equality with the reason each one is there. A third occurrence
  fails until somebody records it, and a recorded one that disappears fails
  too - a stale allowlist entry is a hole nothing reports.
* **declared distributions** (`pyproject.toml`). No import is needed at all:
  one requirement line makes pip install a ray-tracing runtime beside this
  package, in the base list or in an extra. Criterion A8 is a property of the
  METADATA as much as of the code, so it is checked here rather than only in a
  packaging test.

`tests/test_phase4_import_boundary.py` already probes a live process for
`drjit` in `sys.modules`. That catches an import that RUNS. This catches one
that is merely present, in a module the probe's import never reaches, and it
does so without importing anything.

Run it directly, or through `tests/test_phase10_static_gates.py`, which also
proves it FIRES by writing a violation into a temporary copy of the tree.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Top-level distributions whose presence in production would reintroduce a
#: second tracing backend. Matched as a module prefix, so `rayd` covers
#: `rayd.drjit` and `rayd.torch`.
FORBIDDEN_MODULES = ("drjit", "mitsuba", "rayd", "sionna")

#: Decorators that declare a Dr.Jit boundary. Matched on the trailing dotted
#: form so `dr.wrap`, `drjit.wrap` and `something.dr.wrap` all match.
FORBIDDEN_DECORATOR_SUFFIXES = ("dr.wrap", "drjit.wrap")

#: Lower-cased substrings searched for in every string constant.
FORBIDDEN_TOKENS = ("drjit", "mitsuba", "rayd", "sionna", "dr.wrap")

#: No production dependency-vocabulary exceptions remain.
ALLOWED_TOKEN_OCCURRENCES = frozenset()


@dataclass(frozen=True, slots=True)
class Violation:
    module: str
    line: int
    kind: str
    detail: str

    def __str__(self) -> str:
        return f"{self.module}:{self.line}: {self.kind}: {self.detail}"


def _dotted(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _is_forbidden_module(module: str) -> bool:
    return any(module == forbidden or module.startswith(f"{forbidden}.") for forbidden in FORBIDDEN_MODULES)


def _imported_modules(node: ast.Import | ast.ImportFrom) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if node.level:
        return []
    base = node.module or ""
    if not base:
        return []
    if _is_forbidden_module(base):
        return [base]
    return [f"{base}.{alias.name}" for alias in node.names]


def production_modules(root: Path) -> list[Path]:
    """Every Python file that ships inside the `witwin` package."""

    package = root / "witwin"
    return sorted(path for path in package.rglob("*.py") if "__pycache__" not in path.parts)


def scan_module(path: Path, root: Path) -> tuple[list[Violation], dict[str, int]]:
    """Violations, plus the string-token census this module contributes."""

    relative = path.relative_to(root).as_posix()
    source = path.read_text(encoding="utf-8").lstrip(chr(0xFEFF))
    tree = ast.parse(source, filename=str(path))
    violations: list[Violation] = []
    tokens: dict[str, int] = {}

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for module in _imported_modules(node):
                if _is_forbidden_module(module):
                    violations.append(Violation(relative, node.lineno, "import", module))
            continue

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for decorator in node.decorator_list:
                target = decorator.func if isinstance(decorator, ast.Call) else decorator
                name = _dotted(target)
                if any(name == suffix or name.endswith(f".{suffix}") for suffix in FORBIDDEN_DECORATOR_SUFFIXES):
                    violations.append(Violation(relative, node.lineno, "decorator", f"@{name}"))
            continue

        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            lowered = node.value.lower()
            for token in FORBIDDEN_TOKENS:
                count = lowered.count(token)
                if count:
                    tokens[token] = tokens.get(token, 0) + count

    return violations, tokens


def scan(root: Path) -> tuple[list[Violation], set[tuple[str, str, int]]]:
    violations: list[Violation] = []
    census: set[tuple[str, str, int]] = set()
    for path in production_modules(root):
        module_violations, tokens = scan_module(path, root)
        violations.extend(module_violations)
        relative = path.relative_to(root).as_posix()
        for token, count in tokens.items():
            census.add((relative, token, count))
    return violations, census


def _distribution_name(requirement: str) -> str:
    """The PEP 503 normalized project name at the head of a requirement."""

    head = requirement.strip()
    for stop in ("[", "(", ";", "<", ">", "=", "!", "~", " ", "@"):
        head = head.split(stop, 1)[0]
    return re.sub(r"[-_.]+", "-", head).lower()


def declared_distributions(root: Path) -> list[tuple[str, str]]:
    """Every distribution this package declares, as ``(where, requirement)``."""

    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    project = data.get("project", {})
    declared = [("project.dependencies", requirement) for requirement in project.get("dependencies", [])]
    for extra, requirements in project.get("optional-dependencies", {}).items():
        declared.extend((f"project.optional-dependencies.{extra}", requirement) for requirement in requirements)
    declared.extend(
        ("build-system.requires", requirement) for requirement in data.get("build-system", {}).get("requires", [])
    )
    return declared


def check_declared_dependencies(root: Path) -> list[str]:
    """No ray-tracing runtime enters through the METADATA either.

    The three scans above read code. A wheel can acquire a `rayd` runtime
    without a single production import: one line in `pyproject.toml` does it,
    and pip then installs the thing criterion A8 exists to keep out. Extras
    count too - an extra is an installable route, not a comment.
    `witwin-channel` is the one dependency allowed to reach RayD, and it does
    so as its own build-time source link, never as a Radar runtime
    requirement.

    Before this check the property had exactly one guard in the default test
    set, `tests/test_phase10_wheel_packaging.py`; the phase-5 metadata scan
    knows only the `drjit` and `rayd-drjit` spellings and is blind to a bare
    `rayd`. A frozen property belongs in a gate that runs in `quick`.
    """

    failures = []
    for where, requirement in declared_distributions(root):
        name = _distribution_name(requirement)
        if any(name == forbidden or name.startswith(f"{forbidden}-") for forbidden in FORBIDDEN_MODULES):
            failures.append(
                f"pyproject.toml [{where}] declares {requirement!r}: a "
                f"ray-tracing runtime distribution ({name}) must never be a "
                "Radar requirement, in the base list or in an extra"
            )
    return failures


def check(root: Path) -> list[str]:
    violations, census = scan(root)
    failures = [str(violation) for violation in violations]
    failures.extend(check_declared_dependencies(root))

    unrecorded = sorted(census - set(ALLOWED_TOKEN_OCCURRENCES))
    for module, token, count in unrecorded:
        failures.append(
            f"{module}: string literals mention '{token}' {count} time(s); "
            "record the occurrence in ALLOWED_TOKEN_OCCURRENCES with its reason "
            "or remove it"
        )

    stale = sorted(set(ALLOWED_TOKEN_OCCURRENCES) - census)
    for module, token, count in stale:
        failures.append(
            f"{module}: ALLOWED_TOKEN_OCCURRENCES still records "
            f"'{token}' x{count}, which no longer exists; delete the entry"
        )
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    arguments = parser.parse_args(argv)

    root = arguments.root.resolve()
    failures = check(root)
    if failures:
        print(f"check_production_dependencies: {len(failures)} violation(s) under {root}", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1

    modules = production_modules(root)
    declared = declared_distributions(root)
    print(
        f"check_production_dependencies: {len(modules)} production modules name "
        f"none of {', '.join(FORBIDDEN_MODULES)} or "
        f"{', '.join('@' + name for name in FORBIDDEN_DECORATOR_SUFFIXES)}; "
        f"{len(ALLOWED_TOKEN_OCCURRENCES)} recorded prose occurrence(s); "
        f"{len(declared)} declared distribution(s) clean"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
