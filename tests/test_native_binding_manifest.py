"""The native binding manifest is generated from the registrations, and its checks fire.

``ci/check_native_bindings.py`` derives every operator row from ``m.def`` in
``extension.cpp``, ``m.impl`` in the kernel sources and the Python module that
calls each operator. The tests here run it in process on the real tree, then on
edited copies of the inputs, one per defect the derived checks exist to catch.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "ci" / "native-binding-manifest.json"

sys.path.insert(0, str(REPO_ROOT / "ci"))

import check_native_bindings as gate


@pytest.fixture(scope="module")
def committed() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def tree() -> gate.Tree:
    return gate.read_tree()


def _hand(manifest: dict) -> dict:
    return {name: manifest[name] for name in gate.HAND_SECTIONS}


def _failures(tree: gate.Tree, hand: dict) -> list[str]:
    failures: list[str] = []
    gate.generate(hand, tree, failures)
    return failures


def _edit_kernel(tree: gate.Tree, symbol: str, replacement: str) -> gate.Tree:
    kernels = {
        path: text.replace(f'm.impl("{symbol}"', replacement) if f'm.impl("{symbol}"' in text else text
        for path, text in tree.kernels.items()
    }
    assert kernels != tree.kernels, symbol
    return dataclasses.replace(tree, kernels=kernels)


def test_the_committed_manifest_is_what_the_registrations_generate(committed, tree):
    failures: list[str] = []
    generated = gate.generate(_hand(committed), tree, failures)
    assert failures == []
    assert MANIFEST.read_text(encoding="utf-8").replace("\r\n", "\n") == gate.render(generated)


def test_every_operator_has_one_python_owner_and_a_build_translation_unit(committed):
    sources = set(committed["sources"])
    for row in committed["operators"]:
        assert row["native_tu"] in sources, row
        assert (REPO_ROOT / row["python_owner"]).is_file(), row
        assert row["python_callers"], row


def test_an_unimplemented_declaration_fails(committed, tree):
    edited = _edit_kernel(tree, "two_way_join_jvp", 'm.impl("renamed_jvp"')
    failures = _failures(edited, _hand(committed))
    assert any("declared and implemented operators differ" in failure for failure in failures), failures


def test_a_group_with_a_backward_but_no_jvp_fails(committed, tree):
    extension = tree.extension.replace('"two_way_join_jvp(', '"two_way_join_tangent(')
    edited = dataclasses.replace(
        _edit_kernel(tree, "two_way_join_jvp", 'm.impl("two_way_join_tangent"'), extension=extension
    )
    failures = _failures(edited, _hand(committed))
    assert any("ends in _forward, _backward or _jvp" in failure for failure in failures), failures
    assert any("without its other companion" in failure for failure in failures), failures


def test_a_forward_only_group_must_be_declared(committed, tree):
    hand = _hand(committed)
    hand["no_backward"] = {
        group: reason for group, reason in hand["no_backward"].items() if group != "oscillator_phase"
    }
    failures = _failures(tree, hand)
    assert any("'oscillator_phase' registers no backward and no jvp" in failure for failure in failures), failures


def test_a_stale_or_unexplained_no_backward_entry_fails(committed, tree):
    hand = _hand(committed)
    hand["no_backward"] = {**hand["no_backward"], "sensor_weight": "", "deleted_family": "gone"}
    failures = _failures(tree, hand)
    assert any("'sensor_weight', which registers a backward and a jvp" in failure for failure in failures), failures
    assert any("no_backward entry 'sensor_weight' gives no reason" in failure for failure in failures), failures
    assert any("'deleted_family', which is not a registered AD group" in failure for failure in failures), failures


def test_an_operator_nobody_calls_fails(committed, tree):
    python = {path: text.replace(".sensor_weight_jvp(", ".renamed(") for path, text in tree.python.items()}
    failures = _failures(dataclasses.replace(tree, python=python), _hand(committed))
    assert any("sensor_weight_jvp: no production Python module calls it" in failure for failure in failures), failures


def test_an_error_owner_naming_a_missing_module_fails(committed, tree):
    hand = _hand(committed)
    hand["error_owners"] = [dict(hand["error_owners"][0], owner_module="witwin/radar/nowhere.py")]
    failures = _failures(tree, hand)
    assert any("does not exist" in failure for failure in failures), failures


def test_a_stale_manifest_fails_the_gate(tmp_path, committed, capsys):
    stale = dict(committed, operators=committed["operators"][:-1])
    path = tmp_path / "manifest.json"
    path.write_text(gate.render(stale), encoding="utf-8")
    assert gate.main(["--manifest", str(path)]) == 1
    assert "is stale" in capsys.readouterr().out


def test_the_manifest_symbol_set_matches_the_packaged_sidecar(committed):
    """Manifest, shipped binary and loader agree."""

    from witwin.radar.cuda import runtime

    binary = runtime.prebuilt_extension_path()
    if not binary.is_file():
        pytest.skip("no packaged prebuilt in this checkout")
    record = runtime.read_build_info(binary)
    assert set(record["operator_symbols"]) == set(gate.operator_symbols(committed))


def test_the_load_check_covers_every_native_translation_unit(committed):
    """A stale binary must fail at load, not deep inside a kernel call."""

    from witwin.radar.cuda import runtime

    units = {row["native_tu"] for row in committed["operators"]}
    checked = {row["native_tu"] for row in committed["operators"] if row["symbol"] in runtime._REQUIRED_OPERATORS}
    assert checked == units, sorted(units - checked)


def test_the_jit_build_directory_is_keyed_by_the_source_set(monkeypatch):
    """Two checkouts with different sources must not share a build directory."""

    from witwin.radar.cuda import runtime

    fingerprint = runtime.source_fingerprint()
    assert len(fingerprint) == 16
    assert fingerprint in runtime.default_build_directory().name
    assert runtime.source_fingerprint() == fingerprint

    # Content, not just the file list: an edited kernel keys a new directory.
    kernel = REPO_ROOT / "witwin" / "radar" / "cuda" / "fmcw_beat.cu"
    edited = pathlib.Path(__file__).parent / "support" / "__init__.py"
    monkeypatch.setattr(runtime, "extension_sources", lambda: [kernel, edited])
    assert runtime.source_fingerprint() != fingerprint
    monkeypatch.undo()
    assert runtime.source_fingerprint() == fingerprint


def test_a_library_missing_a_required_operator_is_refused(monkeypatch):
    import torch

    from witwin.radar.cuda import runtime

    class Empty:
        def __getattr__(self, name):
            raise AttributeError(name)

    monkeypatch.setattr(torch.ops, "_radar_native", Empty())
    with pytest.raises(ImportError, match="stale"):
        runtime._require_operators(pathlib.Path("fake.pyd"))


def test_every_fingerprinted_source_is_checked_out_with_lf_endings():
    """The source fingerprint hashes raw bytes, so a CRLF checkout would refuse every binary built elsewhere."""

    import shutil
    import subprocess

    from witwin.radar.cuda import runtime

    if shutil.which("git") is None or not (REPO_ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    sources = [*runtime.extension_sources(), *sorted(runtime.source_root().glob("*.cuh"))]
    relative = [pathlib.Path(path).resolve().relative_to(REPO_ROOT).as_posix() for path in sources]
    completed = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "check-attr", "eol", "--", *relative], capture_output=True, text=True, check=True
    )
    attributes = dict(line.split(": eol: ") for line in completed.stdout.splitlines())
    assert attributes == dict.fromkeys(relative, "lf")
