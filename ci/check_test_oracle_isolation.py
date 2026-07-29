"""G2: `tests/` is an oracle, not a dependency, and it does not ship.

`tests/reference/` holds real Torch CPU implementations - `path_math.py` and
`two_way_torch.py`. R-ADR-007 allows them there and only
there: an independent reference stops being independent the moment production
can reach it, and a CPU reference inside a shipped wheel is a fallback waiting
for the first loader failure that someone decides to "handle".

Both halves of that statement are structurally true today and neither was
asserted anywhere. Channel writes the same argument down at
`ci/check_import_graph.py`; Radar said it nowhere, which meant the isolation
survived by nobody happening to break it.

Three checks:

1. **no production import.** No module under `witwin/` imports `tests`,
   `tests.reference` or `tests.<anything>`, and no string constant names one
   (`importlib.import_module("tests.reference.dsp_oracles")` is an import that
   an import scan cannot see).
2. **not packaged.** `[tool.hatch.build.targets.wheel].packages` is exactly
   `["witwin"]` and no `artifacts` entry reaches outside `witwin/`. A wheel
   configuration is the only thing standing between the oracle and the sdist
   consumer, so it is checked rather than assumed.
3. **not in the built wheel** - when one is supplied with `--wheel`. The
   configuration check answers "what did we ask for"; this answers "what came
   out", and they are not the same question when a build hook is involved.
   `ci/wheel_smoke.py` also asserts no `tests/` member, on a wheel it installs;
   this runs on a wheel that is merely present, so `quick` can check a stored
   artifact without an install.
"""

from __future__ import annotations

import argparse
import ast
import sys
import zipfile
from pathlib import Path

import tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Import roots that belong to the test tree. `tests` is the package name the
#: repository's own `tests/` directory takes when the repository root is on
#: `sys.path`, which is how pytest runs it.
TEST_ROOTS = ("tests",)

#: The wheel package list this repository is allowed to declare. Equality: a
#: second entry is how `tests` would ship, and a missing entry is how the
#: package would stop shipping.
EXPECTED_WHEEL_PACKAGES = ("witwin",)


def _is_test_module(module: str) -> bool:
    return any(module == root or module.startswith(f"{root}.") for root in TEST_ROOTS)


def _imported_modules(node: ast.Import | ast.ImportFrom) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if node.level:
        return []
    base = node.module or ""
    if not base:
        return []
    if _is_test_module(base):
        return [base]
    return [f"{base}.{alias.name}" for alias in node.names]


def production_modules(root: Path) -> list[Path]:
    package = root / "witwin"
    return sorted(path for path in package.rglob("*.py") if "__pycache__" not in path.parts)


def check_imports(root: Path) -> list[str]:
    failures: list[str] = []
    for path in production_modules(root):
        relative = path.relative_to(root).as_posix()
        source = path.read_text(encoding="utf-8").lstrip(chr(0xFEFF))
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for module in _imported_modules(node):
                    if _is_test_module(module):
                        failures.append(
                            f"{relative}:{node.lineno}: production imports "
                            f"'{module}'; the oracle must stay under tests/"
                        )
                continue
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                value = node.value
                if _is_test_module(value):
                    failures.append(
                        f"{relative}:{node.lineno}: production names the module "
                        f"'{value}' as a string; a lazily imported oracle is "
                        "still an imported oracle"
                    )
    return failures


def check_packaging(root: Path) -> list[str]:
    configuration = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    wheel = configuration.get("tool", {}).get("hatch", {}).get("build", {}).get("targets", {}).get("wheel", {})
    failures: list[str] = []

    packages = tuple(wheel.get("packages", ()))
    if packages != EXPECTED_WHEEL_PACKAGES:
        failures.append(
            f"pyproject.toml: wheel packages are {list(packages)}; expected {list(EXPECTED_WHEEL_PACKAGES)}"
        )

    for artifact in wheel.get("artifacts", ()):
        if not str(artifact).startswith("witwin/"):
            failures.append(f"pyproject.toml: wheel artifact '{artifact}' reaches outside witwin/")
    return failures


def check_wheel(wheel_path: Path) -> list[str]:
    with zipfile.ZipFile(wheel_path) as archive:
        names = archive.namelist()
    offenders = sorted(
        name for name in names if any(name == root or name.startswith(f"{root}/") for root in TEST_ROOTS)
    )
    if offenders:
        return [f"{wheel_path.name}: ships {len(offenders)} test member(s), first {offenders[0]}"]
    return []


def _resolve_wheel(candidate: Path) -> Path:
    if candidate.is_dir():
        wheels = sorted(candidate.glob("*.whl"))
        if len(wheels) != 1:
            raise SystemExit(f"expected exactly one .whl in {candidate}, found {len(wheels)}")
        return wheels[0]
    return candidate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--wheel", type=Path, default=None, help="a built wheel, or a directory holding exactly one")
    arguments = parser.parse_args(argv)

    root = arguments.root.resolve()
    failures = check_imports(root) + check_packaging(root)

    wheel_checked = "no wheel supplied"
    if arguments.wheel is not None:
        wheel = _resolve_wheel(arguments.wheel.resolve())
        failures.extend(check_wheel(wheel))
        wheel_checked = wheel.name

    if failures:
        print(f"check_test_oracle_isolation: {len(failures)} violation(s) under {root}", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1

    print(
        f"check_test_oracle_isolation: {len(production_modules(root))} production "
        f"modules import no test module; wheel packages "
        f"{list(EXPECTED_WHEEL_PACKAGES)}; wheel members: {wheel_checked}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
