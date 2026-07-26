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


def test_the_production_module_no_longer_holds_the_copied_expressions():
    """The copy is now the SOLE record of the legacy expression.

    This test used to assert that ``solvers/common.py`` and
    ``reference/path_math.py`` were textually identical, and its own docstring
    said it would go with the production functions when work item 8 deleted
    them. It has: the four expressions live only under ``tests/`` now, and what
    is asserted is that no production module grew them back. A drift check
    against a module that no longer has the function would silently pass.
    """

    import inspect

    from witwin.radar.solvers import common
    from reference import path_math

    for name in ("compute_total_path_lengths", "compute_path_amplitudes",
                 "compute_polarization_amplitudes", "compute_antenna_pattern_gains"):
        assert not hasattr(common, name), name
        assert callable(getattr(path_math, name)), name
        # And the copy is still a real expression rather than a re-export.
        assert inspect.getmodule(getattr(path_math, name)) is path_math, name


# --------------------------------------------------------------------------
# D5.8 - the three per-waveform references are independent too
# --------------------------------------------------------------------------

SUPPORT_ROOT = TESTS_ROOT / "support"

#: The three float64 references the waveform families are checked against, one
#: per waveform. An oracle that imported the module it validates would be
#: checking that module against itself, which is the defect this whole file
#: exists to prevent - and it was a real one until Phase 6.
WAVEFORM_ORACLES = (
    "reference_chain.py",
    "reference_ofdm.py",
    "reference_pulsed.py",
)

FORBIDDEN_ORACLE_PREFIXES = ("witwin.radar.solvers", "witwin.radar.synthesis")


def test_the_three_waveform_oracles_exist():
    names = {path.name for path in SUPPORT_ROOT.glob("*.py")}
    for oracle in WAVEFORM_ORACLES:
        assert oracle in names, oracle


@pytest.mark.parametrize("oracle", WAVEFORM_ORACLES)
def test_no_waveform_oracle_imports_the_domain_it_validates(oracle: str):
    """Neither the synthesis owners nor the solvers may appear in an oracle.

    A spec dataclass is not a numerical implementation, so a reference may
    CONSTRUCT one - a grid has to agree with the kernel about what a symbol
    period is - but it may not import an expression from the package under
    test. The scan therefore allows ``witwin.radar.synthesis.contracts`` by
    name and forbids everything else in that package.
    """

    allowed = {
        "witwin.radar.synthesis.contracts",
        "witwin.radar.synthesis",
    }
    offenders = sorted(
        name
        for name in _imported_module_names(SUPPORT_ROOT / oracle)
        if any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in FORBIDDEN_ORACLE_PREFIXES
        )
        and name not in allowed
        and not name.startswith("witwin.radar.synthesis.contracts.")
    )
    assert offenders == [], (oracle, offenders)
