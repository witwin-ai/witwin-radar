"""Radar reports its own build identity and runtime state.

Each extension fails loudly AND reports its full build identity. Three
properties, each of which has a way of quietly regressing:

* ``runtime_diagnostics()`` must NEVER raise. It is what a bug report pastes,
  so it is exercised here with the extension deliberately unloadable - in a
  subprocess, because the loader memoizes and a poisoned in-process load would
  leak into every later test.
* ``build_info()`` must raise in exactly that situation. A diagnostic that
  degrades and an identity call that degrades are opposite requirements, and
  proving one without the other proves nothing.
* importing ``witwin.radar`` must load neither the native extension nor
  ``witwin.channel``. That is measured in a subprocess by counting
  ``sys.modules``, because an in-process check runs in a session where either
  may already be loaded by some earlier test.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

import witwin.radar.deployment as deployment

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(source: str, env: dict[str, str] | None = None) -> dict:
    child_env = {key: value for key, value in os.environ.items() if not key.startswith("WITWIN_RADAR_")}
    child_env.update(env or {})
    completed = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, cwd=str(REPO_ROOT), env=child_env, check=False
    )
    for line in completed.stdout.splitlines():
        if line.startswith("PHASE10DIAG "):
            return json.loads(line[len("PHASE10DIAG ") :])
    raise AssertionError(f"probe produced no result\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}")


# --------------------------------------------------------------------------
# runtime_diagnostics
# --------------------------------------------------------------------------

DECLARED_KEYS = (
    "deployment_abi",
    "radar_abi_version",
    "package_version",
    "python",
    "python_executable",
    "platform",
    "declared_sm_architectures",
    "verified_sm_architectures",
    "sm_matrix_status",
    "ptx_forward_compatibility_sm",
    "native_build",
    "errors",
)


def test_runtime_diagnostics_returns_every_declared_key():
    diagnostics = deployment.runtime_diagnostics()
    for key in DECLARED_KEYS:
        assert key in diagnostics, key
    assert diagnostics["deployment_abi"] == "witwin.radar.deployment.v1"


_UNLOADABLE_PROBE = r"""
import json, sys
from pathlib import Path

from witwin.radar.cuda import runtime

# Make every load route fail: no packaged binary, no override, no build request.
runtime.prebuilt_extension_path = lambda: Path("does-not-exist.pyd")

import witwin.radar.deployment as deployment

result = {}
try:
    diagnostics = deployment.runtime_diagnostics()
    result["diagnostics_raised"] = None
    result["keys"] = sorted(diagnostics)
    result["errors"] = diagnostics["errors"]
    result["native_build"] = diagnostics["native_build"]
except BaseException as exc:  # noqa: BLE001
    result["diagnostics_raised"] = f"{type(exc).__name__}: {exc}"

try:
    deployment.build_info()
    result["build_info_raised"] = None
except BaseException as exc:  # noqa: BLE001
    result["build_info_raised"] = type(exc).__name__
    result["build_info_message"] = str(exc)

print("PHASE10DIAG " + json.dumps(result))
"""


def test_runtime_diagnostics_survives_an_unloadable_extension():
    result = _run(_UNLOADABLE_PROBE)
    assert result["diagnostics_raised"] is None, result["diagnostics_raised"]
    for key in DECLARED_KEYS:
        assert key in result["keys"], key
    assert result["native_build"] is None
    assert result["errors"], "a broken native load must be reported, not hidden"
    assert any("scripts/build_radar_cuda_prebuilt.py" in error for error in result["errors"]), result["errors"]


def test_build_info_fails_loudly_where_diagnostics_degrades():
    """The same process, the same breakage, the opposite requirement."""

    result = _run(_UNLOADABLE_PROBE)
    assert result["build_info_raised"] is not None
    assert "RadarExtensionLoadError" in result["build_info_raised"]
    assert "does-not-exist.pyd" in result["build_info_message"]


def test_build_info_reports_the_full_validated_identity():
    from witwin.radar.cuda import runtime

    if not runtime.prebuilt_extension_path().is_file():
        pytest.skip("no packaged prebuilt in this checkout")
    info = deployment.build_info()
    assert info["origin"] == "packaged"
    assert info["radar_abi_version"] == runtime.RADAR_ABI_VERSION
    record = info["native_build"]
    assert record is not None
    for name, _ in runtime.BUILD_INFO_FIELDS:
        assert name in record, name


def test_require_supported_runtime_agrees_with_the_declared_matrix():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    diagnostics = deployment.require_supported_runtime()
    device = diagnostics["device"]
    assert device["declared_supported"]
    assert device["sm"] in deployment.DECLARED_SM_ARCHITECTURES


def test_the_declared_sm_matrix_is_the_release_gencode_list():
    """One list, two consumers, and a test between them.

    ``scripts/verify_cuda_binary_arches.py`` is what a release actually runs
    against the built wheel. If the runtime record and the verifier disagree,
    one of them is lying to an operator about what is inside the binary.
    """

    source = (REPO_ROOT / "scripts" / "verify_cuda_binary_arches.py").read_text(encoding="utf-8")
    expected = re.search(r"EXPECTED_SASS = \(([^)]*)\)", source)
    assert expected is not None
    architectures = tuple(int(entry.strip().strip('"')) for entry in expected.group(1).split(",") if entry.strip())
    assert architectures == deployment.DECLARED_SM_ARCHITECTURES

    ptx = re.search(r'EXPECTED_PTX_TARGET = "sm_(\d+)"', source)
    assert ptx is not None
    assert int(ptx.group(1)) == deployment.PTX_FORWARD_COMPATIBILITY_SM


def test_verified_architectures_are_a_subset_of_declared_ones():
    assert set(deployment.VERIFIED_SM_ARCHITECTURES) <= set(deployment.DECLARED_SM_ARCHITECTURES)


# --------------------------------------------------------------------------
# the root import
# --------------------------------------------------------------------------


_ROOT_IMPORT_PROBE = r"""
import json, sys

import witwin.radar

print("PHASE10DIAG " + json.dumps({
    "channel": [n for n in sys.modules if n.startswith("witwin.channel")],
    "build_loaded": "witwin.radar.cuda.runtime" in sys.modules,
    "exports": sorted(witwin.radar.__all__),
}))
"""


def test_the_minimal_root_does_not_load_native_or_channel():
    result = _run(_ROOT_IMPORT_PROBE)
    assert result["channel"] == []
    assert result["build_loaded"] is False
    assert result["exports"] == [
        "Adc",
        "Agc",
        "Aspect",
        "Fmcw",
        "Frame",
        "Iq",
        "Leakage",
        "Motion",
        "Noise",
        "Ofdm",
        "Paths",
        "Pattern",
        "PointTargets",
        "Pulsed",
        "Radar",
        "Result",
        "StructureTargets",
        "processing",
    ]


def test_removed_names_receive_an_ordinary_attribute_error():
    import witwin.radar

    for name in ("Tracer", "Scene", "Timeline", "Solver", "TraceResult"):
        with pytest.raises(AttributeError, match="has no attribute"):
            getattr(witwin.radar, name)
