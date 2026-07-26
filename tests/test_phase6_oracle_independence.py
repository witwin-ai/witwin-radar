"""The float64 DSP oracles are independent of the code they check.

Phase 6 migrates ``witwin/radar/solvers/common.py``'s Torch path geometry and
amplitude expressions into a native owner (plan work item 8), and the
acceptance criterion for that migration is that the real-amplitude Radar
baseline is preserved. Until Phase 6, ``tests/reference/dsp_oracles.py``
imported ``compute_path_amplitudes`` and ``compute_total_path_lengths`` from
exactly that module, so "the reference still agrees" would have meant "the
module still agrees with itself" - true by construction and worth nothing.

The two expressions now live in ``tests/reference/path_math.py``, copied
verbatim. This file is the structural guard that keeps them copied: an AST scan
over every module in the reference package, asserting that none of them names
``witwin.radar.solvers``.

The scan is on the AST rather than on text because the module docstrings
legitimately talk about ``witwin.radar.solvers.common`` - naming the rule is
not breaking it.
"""

from __future__ import annotations

import ast
import pathlib
import textwrap

import pytest


TESTS_ROOT = pathlib.Path(__file__).resolve().parent
REFERENCE_ROOT = TESTS_ROOT / "reference"

FORBIDDEN_PREFIX = "witwin.radar.solvers"


def _imported_module_names(path: pathlib.Path) -> set[str]:
    """Every module name a file imports, absolute and relative alike."""

    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            root = node.module or ""
            if node.level:
                # A relative import inside tests/reference/ can never reach
                # witwin.radar; record it under the package it resolves into so
                # the assertion below stays a statement about absolute names.
                root = f"reference.{root}" if root else "reference"
            found.add(root)
            found.update(f"{root}.{alias.name}" for alias in node.names)
    return found


def _reference_modules() -> list[pathlib.Path]:
    return sorted(REFERENCE_ROOT.glob("*.py"))


def test_the_reference_package_has_the_modules_this_scan_assumes():
    names = {path.name for path in _reference_modules()}
    assert "dsp_oracles.py" in names
    assert "path_math.py" in names


@pytest.mark.parametrize(
    "module", _reference_modules(), ids=lambda path: path.name
)
def test_no_reference_oracle_imports_the_module_it_checks(module: pathlib.Path):
    offenders = sorted(
        name
        for name in _imported_module_names(module)
        if name == FORBIDDEN_PREFIX or name.startswith(FORBIDDEN_PREFIX + ".")
    )
    assert offenders == [], (module.name, offenders)


def test_the_oracles_still_import_the_two_copied_expressions():
    """Independence must not have been won by deleting the call.

    ``dsp_oracles`` still evaluates the same two expressions; it just gets them
    from the copy. If a future edit inlines or drops them, this fails and the
    scan above stops meaning anything.
    """

    from reference import dsp_oracles, path_math

    assert dsp_oracles.compute_path_amplitudes is path_math.compute_path_amplitudes
    assert (
        dsp_oracles.compute_total_path_lengths is path_math.compute_total_path_lengths
    )


def test_the_copied_expressions_still_match_the_production_ones():
    """A copy that has drifted is worse than an import.

    Independence is about who owns the expression, not about permission to
    change it behind the production module's back. While ``solvers/common.py``
    still exists, the two sources must be textually identical after the
    docstring, so a Phase-6 edit to one and not the other is caught here rather
    than as a mysterious tolerance failure. When work item 8 deletes the
    production functions, this test goes with them and ``path_math`` becomes the
    sole record of the legacy expression.
    """

    import inspect

    from witwin.radar.solvers import common
    from reference import path_math

    for name in ("compute_total_path_lengths", "compute_path_amplitudes",
                 "compute_polarization_amplitudes", "compute_antenna_pattern_gains"):
        produced = inspect.getsource(getattr(common, name))
        copied = inspect.getsource(getattr(path_math, name))
        assert _body_without_docstring(produced) == _body_without_docstring(copied), name


def _body_without_docstring(source: str) -> str:
    function = ast.parse(textwrap.dedent(source)).body[0]
    assert isinstance(function, ast.FunctionDef)
    body = function.body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    return "\n".join(ast.dump(node) for node in body)
