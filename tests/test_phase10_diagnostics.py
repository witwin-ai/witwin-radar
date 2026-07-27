"""Radar reports its own build identity, capability and runtime state.

Acceptance criterion A3 (each extension fails loudly AND reports full build
identity) and the Radar half of item 3. Three properties, each of which has a
way of quietly regressing:

* ``runtime_diagnostics()`` must NEVER raise. It is what a bug report pastes,
  so it is exercised here with the extension deliberately unloadable - in a
  subprocess, because the loader memoizes and a poisoned in-process load would
  leak into every later test.
* ``build_info()`` must raise in exactly that situation. A diagnostic that
  degrades and an identity call that degrades are opposite requirements, and
  proving one without the other proves nothing.
* ``capabilities()`` must not import ``witwin.channel``. That is measured in a
  subprocess by counting ``sys.modules``, because an in-process check runs in a
  session where Channel may already be loaded by some earlier test.

The AD summary inside the capability record is pinned against
``docs/dev/radar-ad-capability-matrix.md``, which is the authority. Two places,
one direction of truth.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

from witwin.radar import capabilities as capability_record
from witwin.radar import deployment


REPO_ROOT = Path(__file__).resolve().parents[1]
MATRIX = REPO_ROOT / "docs" / "dev" / "radar-ad-capability-matrix.md"


def _run(source: str, env: dict[str, str] | None = None) -> dict:
    child_env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("WITWIN_RADAR_")
    }
    child_env.update(env or {})
    completed = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=child_env,
        check=False,
    )
    for line in completed.stdout.splitlines():
        if line.startswith("PHASE10DIAG "):
            return json.loads(line[len("PHASE10DIAG ") :])
    raise AssertionError(
        f"probe produced no result\nstdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )


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

from witwin.radar.cuda import build

# Make every load route fail: no packaged binary, no override, no build request.
build.prebuilt_extension_path = lambda: Path("does-not-exist.pyd")

from witwin.radar import deployment

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
    assert any(
        "scripts/build_radar_cuda_prebuilt.py" in error for error in result["errors"]
    ), result["errors"]


def test_build_info_fails_loudly_where_diagnostics_degrades():
    """The same process, the same breakage, the opposite requirement."""

    result = _run(_UNLOADABLE_PROBE)
    assert result["build_info_raised"] is not None
    assert "RadarExtensionLoadError" in result["build_info_raised"]
    assert "does-not-exist.pyd" in result["build_info_message"]


def test_build_info_reports_the_full_validated_identity():
    from witwin.radar.cuda import build, identity

    if not build.prebuilt_extension_path().is_file():
        pytest.skip("no packaged prebuilt in this checkout")
    info = deployment.build_info()
    assert info["origin"] == "packaged"
    assert info["radar_abi_version"] == identity.RADAR_ABI_VERSION
    record = info["native_build"]
    assert record is not None
    for name, _ in identity.BUILD_INFO_FIELDS:
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

    source = (REPO_ROOT / "scripts" / "verify_cuda_binary_arches.py").read_text(
        encoding="utf-8"
    )
    expected = re.search(r"EXPECTED_SASS = \(([^)]*)\)", source)
    assert expected is not None
    architectures = tuple(
        int(entry.strip().strip('"')) for entry in expected.group(1).split(",")
        if entry.strip()
    )
    assert architectures == deployment.DECLARED_SM_ARCHITECTURES

    ptx = re.search(r'EXPECTED_PTX_TARGET = "sm_(\d+)"', source)
    assert ptx is not None
    assert int(ptx.group(1)) == deployment.PTX_FORWARD_COMPATIBILITY_SM


def test_verified_architectures_are_a_subset_of_declared_ones():
    assert set(deployment.VERIFIED_SM_ARCHITECTURES) <= set(
        deployment.DECLARED_SM_ARCHITECTURES
    )


# --------------------------------------------------------------------------
# capabilities
# --------------------------------------------------------------------------


def test_the_capability_record_is_versioned_and_names_its_abi():
    from witwin.radar.cuda.identity import RADAR_ABI_VERSION

    record = capability_record()
    assert record["schema_version"] == 1
    assert record["radar_abi_version"] == RADAR_ABI_VERSION
    assert record["native_library"]["numerical_owner"] == "radar"


def test_the_capability_families_are_the_manifest_families():
    manifest = json.loads(
        (REPO_ROOT / "ci" / "native-binding-manifest.json").read_text(encoding="utf-8")
    )
    record = capability_record()
    assert set(record["native_library"]["operator_families"]) == {
        entry["family"] for entry in manifest["operators"]
    }


def test_the_ad_summary_agrees_with_the_capability_matrix_document():
    """The record summarizes the matrix; the matrix stays the authority."""

    text = MATRIX.read_text(encoding="utf-8")
    record = capability_record()["ad_contract"]
    assert record["matrix_document"] == "docs/dev/radar-ad-capability-matrix.md"
    for state in record["states"]:
        assert f"| `{state}` |" in text, state
    assert "SILENT is not one of them" in text
    assert record["production_finite_differences"] is False
    assert record["first_order_only"] is True
    # The matrix's own statement of the velocity rule (ADR-038).
    assert "velocity is a forward-AD tangent DIRECTION, never a leaf" in text


def test_the_processing_wall_stages_are_the_matrix_wall_stages():
    text = MATRIX.read_text(encoding="utf-8")
    wall = capability_record()["processing_wall"]
    for stage in ("range_profile", "range_doppler", "beam_cube", "matched_filter"):
        assert stage in wall["differentiable_stages"]
        assert f"processing/{stage}" in text
    for stage in ("ca_cfar", "os_cfar", "point_cloud", "fft2_aoa"):
        assert stage in wall["refusing_stages"]
        assert f"processing/{stage}" in text
    assert not set(wall["differentiable_stages"]) & set(wall["refusing_stages"])


def test_the_refused_components_are_refused_by_the_adapter():
    record = capability_record()["propagation_request"]
    source = (REPO_ROOT / record["refusal_site"]).read_text(encoding="utf-8")
    assert "not_freezable" in source
    assert set(record["components"]) == {"los", "reflection"}
    assert not set(record["components"]) & set(record["refused_components"])


_CAPABILITY_PROBE = r"""
import json, sys

import witwin.radar

before = [name for name in sys.modules if name.startswith("witwin.channel")]
record = witwin.radar.capabilities()
after = [name for name in sys.modules if name.startswith("witwin.channel")]

print("PHASE10DIAG " + json.dumps({
    "before": before,
    "after": after,
    "consumer_status": record["propagation_consumer"]["status"],
    "schema_version": record["schema_version"],
}))
"""


def test_reading_the_capability_record_never_imports_channel():
    """Criterion A2's most likely regression, measured rather than reviewed."""

    result = _run(_CAPABILITY_PROBE)
    assert result["before"] == []
    assert result["after"] == [], result["after"]
    assert result["consumer_status"] == "not_loaded"
    assert result["schema_version"] == 1


_CONSUMER_PRESENT_PROBE = r"""
import json, sys

import witwin.channel.propagation.consumer  # noqa: F401
import witwin.radar

record = witwin.radar.capabilities()["propagation_consumer"]
print("PHASE10DIAG " + json.dumps({
    "status": record["status"],
    "contract_version": record.get("contract_version"),
    "components": record.get("components"),
}))
"""


def test_the_consumer_record_is_embedded_when_channel_is_already_loaded():
    """The other half: not_loaded must mean absent, not never-reported."""

    pytest.importorskip("witwin.channel")
    result = _run(_CONSUMER_PRESENT_PROBE)
    assert result["status"] == "loaded"
    assert isinstance(result["contract_version"], int)
    assert "los" in result["components"]


_ROOT_LAZY_PROBE = r"""
import json, sys

import witwin.radar

print("PHASE10DIAG " + json.dumps({
    "channel": [n for n in sys.modules if n.startswith("witwin.channel")],
    "build_loaded": "witwin.radar.cuda.build" in sys.modules,
    "has_build_info": hasattr(witwin.radar, "build_info"),
    "has_capabilities": hasattr(witwin.radar, "capabilities"),
    "has_runtime_diagnostics": hasattr(witwin.radar, "runtime_diagnostics"),
    "build_loaded_after_access": "witwin.radar.cuda.build" in sys.modules,
}))
"""


def test_the_root_exports_are_lazy_and_do_not_load_the_extension():
    """``hasattr`` resolves them, which is the point: they exist, unloaded.

    ``witwin.radar.cuda.build`` being absent from ``sys.modules`` after a bare
    import is the property the whole Phase-10 loader contract rests on - a
    broken prebuilt must not be able to fail an ordinary package import.
    """

    result = _run(_ROOT_LAZY_PROBE)
    assert result["channel"] == []
    assert result["build_loaded"] is False
    assert result["has_build_info"] is True
    assert result["has_capabilities"] is True
    assert result["has_runtime_diagnostics"] is True


def test_the_root_still_refuses_the_removed_names():
    """The lazy hook must not have swallowed the removed-name messages."""

    import witwin.radar

    with pytest.raises(AttributeError, match="Channel consumer"):
        witwin.radar.Tracer
    with pytest.raises(AttributeError, match="has no attribute"):
        witwin.radar.not_a_real_name
