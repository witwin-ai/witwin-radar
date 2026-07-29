"""The extension boundary, checked against the shipped binaries.

Acceptance criterion A5. ``ci/check_extension_boundary.py`` parses the PE
import table (or the ELF ``DT_NEEDED`` list) of the radar native library and
asserts it against a frozen allowlist. This file runs that gate on the packaged
prebuilt, proves the gate FIRES on a mutated allowlist, and pins the property a
reader would otherwise have to take on trust: that the byte scan can find a
stem when one IS present, so "neither binary names the other" is a measurement
rather than a scan that does nothing.

The ELF branch has no case here because no ELF radar binary exists on a Windows
developer machine. It was verified by hand against two real ELF64 libraries and
the result is recorded in ``docs/dev/audit/phase10-extension-boundary.md``; the
manylinux cells of the release matrix are what will exercise it in anger.

Every case SKIPS rather than silently passes when the artifact it needs is not
in the checkout. A boundary test that quietly passes with no binary to check is
worse than no test, because it reports a green that means nothing.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
GATE = REPO_ROOT / "ci" / "check_extension_boundary.py"

sys.path.insert(0, str(REPO_ROOT / "ci"))

import check_extension_boundary as gate  # noqa: E402


def _radar_binary() -> Path:
    try:
        return gate.discover_radar_binary()
    except gate.BoundaryError as error:
        pytest.skip(f"no packaged radar prebuilt: {error}")


def test_the_gate_passes_on_the_packaged_prebuilt():
    binary = _radar_binary()
    completed = subprocess.run(
        [sys.executable, str(GATE), "--radar-binary", str(binary)],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "extension boundary OK" in completed.stdout


def test_exactly_one_native_member_sits_under_the_prebuilt_directory():
    binary = _radar_binary()
    members = [entry for entry in binary.parent.iterdir() if entry.is_file() and entry.suffix in gate.NATIVE_SUFFIXES]
    assert members == [binary]


def test_the_radar_imports_are_a_subset_of_the_frozen_allowlist():
    binary = _radar_binary()
    imports = {name.lower() for name in gate.read_imports(binary)}
    allowed = gate.allowlist_for(binary)
    assert imports <= allowed, sorted(imports - allowed)
    # Not vacuous: the library really does import the Torch runtime it
    # registers into, so an empty or unparsed table would fail here.
    assert "torch_cpu.dll" in imports or "libtorch_cpu.so" in imports


def test_no_rayd_or_drjit_runtime_reaches_the_radar_binary():
    """Criterion A8 at the binary level rather than at the import-graph level."""

    binary = _radar_binary()
    for name in gate.read_imports(binary):
        lowered = name.lower()
        for token in gate.FORBIDDEN_IMPORT_TOKENS:
            assert token not in lowered, (name, token)


def test_the_radar_library_holds_no_python_c_api_reference():
    binary = _radar_binary()
    for name in gate.read_imports(binary):
        lowered = name.lower()
        for token in gate.PYTHON_IMPORT_TOKENS:
            assert not lowered.startswith(token), name


def test_the_binary_name_scan_can_actually_find_a_name():
    """Calibration. Without this, 'names neither' could mean 'finds nothing'."""

    binary = _radar_binary()
    assert gate.names_binary(binary, gate.stem_of(binary))
    assert not gate.names_binary(binary, "_channel")
    assert not gate.names_binary(binary, "drjit")


def test_the_gate_fires_when_an_import_leaves_the_allowlist(monkeypatch):
    """A passing gate that cannot fail is not a gate."""

    binary = _radar_binary()
    narrowed = frozenset(name for name in gate.WINDOWS_ALLOWLIST if not name.startswith("torch_"))
    monkeypatch.setattr(gate, "WINDOWS_ALLOWLIST", narrowed)
    monkeypatch.setattr(
        gate, "LINUX_ALLOWLIST", frozenset(name for name in gate.LINUX_ALLOWLIST if not name.startswith("libtorch"))
    )
    report = gate.check_boundary(binary)
    assert not report["ok"]
    assert any("allowlist" in failure for failure in report["failures"])


def test_the_gate_fires_when_the_two_extensions_name_each_other(tmp_path):
    """A cross-extension private ABI, simulated by planting the other stem."""

    binary = _radar_binary()
    planted = tmp_path / "_channel.cp311-win_amd64.pyd"
    planted.write_bytes(binary.read_bytes() + gate.stem_of(binary).encode("ascii"))
    report = gate.check_boundary(binary, planted)
    assert not report["ok"]
    assert any("cross-extension" in failure for failure in report["failures"])


def test_the_channel_binary_is_audited_when_one_is_available():
    """Optional by design: Channel is not a Radar test dependency.

    The path comes from ``WITWIN_CHANNEL_EXTENSION_PATH``, which is Channel's
    own developer-override variable, rather than from a hard-coded machine
    path. A checkout that has no Channel binary skips, and says so.
    """

    raw = os.environ.get("WITWIN_CHANNEL_EXTENSION_PATH")
    if not raw:
        pytest.skip("WITWIN_CHANNEL_EXTENSION_PATH is not set")
    channel = Path(raw)
    if not channel.is_file():
        pytest.skip(f"{channel} does not exist")
    report = gate.check_boundary(_radar_binary(), channel)
    assert report["ok"], report["failures"]
