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

    kernels = REPO_ROOT / "witwin" / "radar" / "cuda"
    found: set[str] = set()
    for path in sorted(kernels.glob("*.cu")):
        found.update(re.findall(r'm\.impl\(\s*"(\w+)"', path.read_text(encoding="utf-8")))
    return found


def test_declared_implemented_and_manifested_operators_agree(manifest):
    manifested = {entry["symbol"] for entry in manifest["operators"]}
    declared = _declared_operators()
    implemented = _implemented_operators()
    assert declared == manifested, (sorted(declared), sorted(manifested))
    assert implemented == manifested, (sorted(implemented), sorted(manifested))


#: Symbols that are registered, tested, and called by NOTHING in production.
#: A caller-free symbol is cleanup debt, and the manifest is allowed to say so.
#: What it is not allowed to do is name a caller that does not call it. This
#: set is the budget: it may shrink, and growing it is a test failure.
#:
#: Phase 11 emptied it. The one entry was ``backward``, in the
#: ``dirichlet_spectrum`` family, and it was deleted with that family's
#: translation unit rather than given a caller. Every registered symbol now
#: names a production entry point, which is acceptance criterion 6's "no orphan
#: binding" written as an equality.
CALLER_FREE_SYMBOLS = set()


def test_every_operator_has_an_owner_a_test_and_a_caller(manifest):
    """And a caller-free symbol must SAY it is caller-free.

    The entry shape survives an empty budget on purpose. It exists because the
    ``backward`` row named ``DirichletSolver.backward`` while that method
    dispatched ``backward_parallel_bins`` through ``spectrum_vjp``: a manifest
    whose only job is accuracy cannot carry a claim the owner's own docstring
    contradicts, so a null caller plus a recorded reason is a first-class entry
    shape. What changed is the budget. A new caller-free symbol is now an
    explicit decision recorded above rather than an addition that costs nothing.
    """

    caller_free = set()
    for entry in manifest["operators"]:
        owner = REPO_ROOT / entry["python_owner"]
        assert owner.exists(), entry["symbol"]
        contract = REPO_ROOT / entry["contract_test"]
        assert contract.exists(), entry["symbol"]
        caller = entry["end_to_end_caller"]
        if caller is None:
            assert entry.get("caller_status") == "test_only", entry["symbol"]
            assert entry.get("caller_note"), entry["symbol"]
            caller_free.add(entry["symbol"])
        else:
            assert caller.startswith("witwin.radar."), entry["symbol"]
        # The owner must actually name the symbol it claims to own.
        assert entry["symbol"] in owner.read_text(encoding="utf-8"), entry["symbol"]
    assert caller_free == CALLER_FREE_SYMBOLS, sorted(caller_free)


def test_a_named_end_to_end_caller_resolves_to_something_that_exists(manifest):
    """The checkable half of the claim: the dotted path is real.

    Resolving the attribute chain does not prove the caller reaches the symbol -
    that would need a call graph, which this file is not going to grow - but it
    does stop an entry from naming a method that was renamed or deleted, which
    is one of the two ways the ``backward`` inaccuracy could have arisen.
    """

    import importlib

    for entry in manifest["operators"]:
        caller = entry["end_to_end_caller"]
        if caller is None:
            continue
        parts = caller.split(".")
        module = None
        attributes: list[str] = []
        for split in range(len(parts) - 1, 2, -1):
            try:
                module = importlib.import_module(".".join(parts[:split]))
            except ImportError:
                continue
            attributes = parts[split:]
            break
        assert module is not None, caller
        target = module
        for attribute in attributes:
            assert hasattr(target, attribute), (caller, attribute)
            target = getattr(target, attribute)


def test_every_manifested_source_is_a_build_input(manifest):
    from witwin.radar.cuda import runtime as build

    sources = {str(path.relative_to(REPO_ROOT)).replace("\\", "/") for path in build.extension_sources()}
    assert sources == set(manifest["sources"]), (sorted(sources), manifest["sources"])


def test_the_load_check_covers_every_operator_family(manifest):
    """A stale binary must fail at load, not deep inside a kernel call."""

    from witwin.radar.cuda import runtime as build

    families = {entry["family"] for entry in manifest["operators"]}
    checked_families = {
        entry["family"] for entry in manifest["operators"] if entry["symbol"] in build._REQUIRED_OPERATORS
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

    from witwin.radar.cuda import runtime as build

    fingerprint = build.source_fingerprint()
    assert len(fingerprint) == 16
    assert fingerprint in build.default_build_directory().name
    # Deterministic: the same sources always key the same directory.
    assert build.source_fingerprint() == fingerprint

    # Content, not just the file list: an edited kernel keys a new directory.
    original = build.extension_sources
    kernel = REPO_ROOT / "witwin" / "radar" / "cuda" / "fmcw_beat.cu"
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

    from witwin.radar.cuda import runtime as build

    class Empty:
        def __getattr__(self, name):
            raise AttributeError(name)

    monkeypatch.setattr(torch.ops, "_radar_native", Empty())
    with pytest.raises(ImportError, match="stale"):
        build._require_operators(pathlib.Path("fake.pyd"))

    # And the JIT route goes through that same gate rather than around it.
    source = (REPO_ROOT / "witwin" / "radar" / "cuda" / "runtime.py").read_text(encoding="utf-8")
    assert "return _require_operators(Path(library_path))" in source
    assert "return _StableOpsModule(Path(library_path))" not in source
