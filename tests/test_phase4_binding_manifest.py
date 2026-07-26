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
    """Every symbol implemented anywhere under ``kernels/``.

    The directory is globbed rather than listed. A hard-coded list makes a NEW
    translation unit invisible to this gate exactly when it is least reviewed -
    the file would be a build input and a manifest entry while its ``m.impl``
    registrations went unchecked - and the glob is strictly stronger: a kernel
    file that is not a build input still has to appear in the manifest.
    """

    kernels = REPO_ROOT / "witwin" / "radar" / "cuda" / "kernels"
    found: set[str] = set()
    for path in sorted(kernels.glob("*.cu")):
        found.update(
            re.findall(r'm\.impl\(\s*"(\w+)"', path.read_text(encoding="utf-8"))
        )
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


def test_the_jit_build_directory_is_keyed_by_the_source_set():
    """Two checkouts with different sources must not share a build directory.

    They used to. The main radar checkout compiles two sources and this one
    compiles three into the same `stable_abi_v1` directory, so ninja relinked on
    every alternating run and the binary on disk was whichever checkout ran last.
    Observed live: two distinct link command hashes in one `.ninja_log`, and a
    `fmcw_beat_forward` attribute error that passed on an identical rerun.
    """

    from witwin.radar.cuda import build

    fingerprint = build.source_fingerprint()
    assert len(fingerprint) == 16
    assert fingerprint in build.default_build_directory().name
    # Deterministic: the same sources always key the same directory.
    assert build.source_fingerprint() == fingerprint

    # Content, not just the file list: an edited kernel keys a new directory.
    original = build.extension_sources
    kernel = REPO_ROOT / "witwin" / "radar" / "cuda" / "kernels" / "fmcw_beat.cu"
    edited = pathlib.Path(__file__).parent / "support" / "__init__.py"
    build.extension_sources = lambda: [kernel, edited]
    try:
        assert build.source_fingerprint() != fingerprint
    finally:
        build.extension_sources = original
    assert build.source_fingerprint() == fingerprint


def test_every_load_route_validates_the_required_operators(monkeypatch):
    """Including the JIT route, which used to return an unvalidated module.

    A just-in-time build can hand back a stale library: `load` reuses whatever is
    already linked in its build directory. Returning that without checking turns
    a missing operator family into a failure deep inside a kernel call.
    """

    import torch

    from witwin.radar.cuda import build

    class Empty:
        def __getattr__(self, name):
            raise AttributeError(name)

    monkeypatch.setattr(torch.ops, "witwin_radar_dirichlet_cuda", Empty())
    with pytest.raises(ImportError, match="stale"):
        build._require_operators(pathlib.Path("fake.pyd"))

    # And the JIT route goes through that same gate rather than around it.
    source = (REPO_ROOT / "witwin" / "radar" / "cuda" / "build.py").read_text(
        encoding="utf-8"
    )
    assert "return _require_operators(Path(library_path))" in source
    assert "return _StableOpsModule(Path(library_path))" not in source
