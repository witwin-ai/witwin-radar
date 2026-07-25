"""Every registered native operator has an owner, a test, and a caller.

A symbol without a production caller is cleanup debt, not a feature. Keeping
the manifest honest is cheap here and expensive later.
"""

from __future__ import annotations

import json
import pathlib
import re

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "ci" / "native-binding-manifest.json"
EXTENSION = REPO_ROOT / "witwin" / "radar" / "cuda" / "extension.cpp"


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _declared_operators() -> set[str]:
    source = EXTENSION.read_text(encoding="utf-8")
    return set(re.findall(r'm\.def\(\s*"(\w+)\(', source))


def _implemented_operators() -> set[str]:
    found: set[str] = set()
    for name in ("dirichlet.cu", "fmcw_beat.cu"):
        source = (REPO_ROOT / "witwin" / "radar" / "cuda" / "kernels" / name).read_text(
            encoding="utf-8"
        )
        found.update(re.findall(r'm\.impl\(\s*"(\w+)"', source))
    return found


def test_declared_implemented_and_manifested_operators_agree(manifest):
    manifested = {entry["symbol"] for entry in manifest["operators"]}
    declared = _declared_operators()
    implemented = _implemented_operators()
    assert declared == manifested, (sorted(declared), sorted(manifested))
    assert implemented == manifested, (sorted(implemented), sorted(manifested))


def test_every_operator_has_an_owner_a_test_and_a_caller(manifest):
    for entry in manifest["operators"]:
        owner = REPO_ROOT / entry["python_owner"]
        assert owner.exists(), entry["symbol"]
        contract = REPO_ROOT / entry["contract_test"]
        assert contract.exists(), entry["symbol"]
        assert entry["end_to_end_caller"].startswith("witwin.radar."), entry["symbol"]
        # The owner must actually name the symbol it claims to own.
        assert entry["symbol"] in owner.read_text(encoding="utf-8"), entry["symbol"]


def test_every_manifested_source_is_a_build_input(manifest):
    from witwin.radar.cuda import build

    sources = {
        str(path.relative_to(REPO_ROOT)).replace("\\", "/")
        for path in build.extension_sources()
    }
    assert sources == set(manifest["sources"]), (sorted(sources), manifest["sources"])


def test_the_load_check_covers_every_operator_family(manifest):
    """A stale binary must fail at load, not deep inside a kernel call."""

    from witwin.radar.cuda import build

    families = {entry["family"] for entry in manifest["operators"]}
    checked_families = {
        entry["family"]
        for entry in manifest["operators"]
        if entry["symbol"] in build._REQUIRED_OPERATORS
    }
    assert checked_families == families, sorted(families - checked_families)
