"""The float64 oracles are independent of the code they check.

The native sensor family is checked against independently copied path-length, path-rate and antenna-pattern expressions. Importing those expressions from the production owner would make the reference agree by construction and prove nothing.

The expressions live in ``tests/reference/path_math.py``, copied verbatim.
This file is the structural guard that keeps them copied: an AST scan over
every module in the reference package, asserting that none of them names the
production owner of the family it validates.

**Phase 11 moved the target of that assertion.** ``witwin.radar.solvers`` no
longer exists, so naming it would make the scan vacuous. The family
``path_math`` still checks is the LIVE ``sensor_weight`` one, whose owners are
``witwin.radar.sensors`` and ``witwin.radar.synthesis``; those are what the
oracle may not import. The ``dsp_oracles`` half of the file went with the
Dirichlet route it checked.

The scan is on the AST rather than on text because module docstrings
legitimately talk about the owner package - naming the rule is not breaking it.
"""

from __future__ import annotations

import ast
import pathlib

import pytest


TESTS_ROOT = pathlib.Path(__file__).resolve().parent
REFERENCE_ROOT = TESTS_ROOT / "reference"

#: The production owners of the families the reference package checks. An
#: oracle that imported one of these would be checking a module against itself.
#: ``witwin.radar.solvers`` used to be the single entry and is gone; the
#: sensor-weight owner replaced it as the thing ``path_math`` is a copy of.
FORBIDDEN_PREFIXES = ("witwin.radar.sensors", "witwin.radar.synthesis")


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
    assert "path_math.py" in names
    # And the Dirichlet oracle is gone rather than merely unused: it evaluated
    # a chirp and a MIMO cube for a family that no longer ships.
    assert "dsp_oracles.py" not in names


@pytest.mark.parametrize(
    "module", _reference_modules(), ids=lambda path: path.name
)
def test_no_reference_oracle_imports_the_module_it_checks(module: pathlib.Path):
    offenders = sorted(
        name
        for name in _imported_module_names(module)
        if any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in FORBIDDEN_PREFIXES
        )
    )
    assert offenders == [], (module.name, offenders)


def test_the_copy_is_still_a_real_expression_and_has_a_live_consumer():
    """Independence must not have been won by deleting the call.

    Two claims. First, each copied expression is defined in ``path_math``
    rather than re-exported from somewhere - a re-export would make the scan
    above pass while checking a module against itself through one more hop.
    Second, the copy is still USED: ``tests/test_phase6_sensor_weight.py`` is
    the contract test of the live ``sensor_weight`` family and drives all three as its reference. An oracle nobody calls is not independent, it
    is dead.
    """

    import inspect

    from reference import path_math

    names = (
        "compute_total_path_lengths",
        "compute_total_path_length_rates",
        "compute_antenna_pattern_gains",
    )
    for name in names:
        assert callable(getattr(path_math, name)), name
        assert inspect.getmodule(getattr(path_math, name)) is path_math, name

    consumer = (TESTS_ROOT / "test_phase6_sensor_weight.py").read_text(
        encoding="utf-8"
    )
    for name in names:
        assert name in consumer, name


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

FORBIDDEN_ORACLE_PREFIXES = ("witwin.radar.sensors", "witwin.radar.synthesis")


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
    test. The scan therefore allows ``witwin.radar.synthesis.assembly`` by
    name and forbids everything else in that package.
    """

    allowed = {
        "witwin.radar.synthesis.assembly",
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
        and not name.startswith("witwin.radar.synthesis.assembly.")
    )
    assert offenders == [], (oracle, offenders)
