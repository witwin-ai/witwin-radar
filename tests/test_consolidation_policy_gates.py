"""Calibration tests for consolidation policy and inventory gates."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    # The gates import their shared helpers as siblings, the way `python ci/x.py` finds them.
    if str(ROOT / "ci") not in sys.path:
        sys.path.insert(0, str(ROOT / "ci"))
    path = ROOT / "ci" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _json(path: Path, value: object) -> None:
    _write(path, json.dumps(value))


def test_architecture_gate_rejects_a_public_export_owned_outside_the_inventory(tmp_path: Path) -> None:
    gate = _load("check_architecture")
    _json(
        tmp_path / "ci" / "public-api-manifest.json",
        {"modules": {"witwin.radar": {"Radar": "witwin.radar.radar.Radar", "Ghost": "witwin.radar.gone.Ghost"}}},
    )
    errors = gate._audit_public_owners(tmp_path, {"witwin.radar.radar"})
    assert errors == ["public exposure witwin.radar.Ghost names non-target owner module witwin.radar.gone"]


def test_public_api_manifest_gate_rejects_duplicate_target() -> None:
    gate = _load("check_public_api_manifest")
    errors = gate.audit_manifest(
        {
            "modules": {
                "witwin.radar": {"Radar": "witwin.radar.radar.Radar", "RadarAlias": "witwin.radar.radar.Radar"}
            },
            "root_class_members": {},
        }
    )
    assert errors == ["target witwin.radar.radar.Radar exposed twice: witwin.radar.Radar and witwin.radar.RadarAlias"]


def test_duplicate_code_gate_rejects_renamed_clone(tmp_path: Path, monkeypatch) -> None:
    gate = _load("check_duplicate_code")
    package = tmp_path / "witwin" / "radar"
    implementation = (
        "def {name}(value):\n    {doc!r}\n    shifted = value + 1\n    scaled = shifted * 2\n    return scaled\n"
    )
    _write(package / "first.py", implementation.format(name="first", doc="First wording."))
    _write(package / "second.py", implementation.format(name="second", doc="Different wording."))
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    monkeypatch.setattr(gate, "PACKAGE", package)
    duplicates = gate.find_duplicates()
    assert len(duplicates) == 1
    assert [owner.rsplit(":", 1)[-1] for owner in duplicates[0]] == ["first", "second"]


def test_public_api_manifest_gate_handles_declared_value_exports(monkeypatch) -> None:
    gate = _load("check_public_api_manifest")
    module = SimpleNamespace(__all__=("ISOTROPIC_PATTERN",), ISOTROPIC_PATTERN=object())
    monkeypatch.setattr(gate.importlib, "import_module", lambda _name: module)
    manifest = {
        "modules": {"witwin.radar.sensors": {"ISOTROPIC_PATTERN": "witwin.radar.sensors.ISOTROPIC_PATTERN"}},
        "root_class_members": {},
        "value_exports": ["witwin.radar.sensors.ISOTROPIC_PATTERN"],
    }
    assert gate.audit_live(manifest) == []


def test_release_claim_gate_rejects_retired_policy(tmp_path: Path) -> None:
    gate = _load("check_release_claims")
    _json(
        tmp_path / "ci" / "release-policy.json",
        {"manylinux_policy": "manylinux_2_28", "stable_abi_cross_torch_claim": False},
    )
    _write(tmp_path / "README.md", "manylinux_2_35 cross-Torch Stable ABI\n")
    _write(tmp_path / "FEATURE_LIST.md", "manylinux_2_28\n")
    _write(tmp_path / "docs" / "dev" / "plans" / "phase10-deferred-release-matrix.md", "manylinux_2_28\n")
    errors = gate.audit(tmp_path)
    assert any("retired manylinux_2_35" in error for error in errors)
    assert any("cross-Torch Stable ABI" in error for error in errors)


def test_workflow_policy_rejects_a_wheel_smoke_shadowed_by_the_checkout(tmp_path: Path) -> None:
    gate = _load("check_workflow_policy")
    source = (ROOT / ".github" / "workflows" / "publish-witwin-radar.yml").read_text(encoding="utf-8")
    mutated = source.replace("python -I - <<'PY'", "python - <<'PY'", 1)
    assert mutated != source
    workflow = tmp_path / "publish-witwin-radar.yml"
    _write(workflow, mutated)
    failures = gate.check_workflow(workflow)
    assert any("repository checkout can shadow the installed wheel" in failure for failure in failures)


def test_workflow_policy_rejects_raw_compressed_platform_tag_comparison(tmp_path: Path) -> None:
    gate = _load("check_workflow_policy")
    source = (ROOT / ".github" / "workflows" / "publish-witwin-radar.yml").read_text(encoding="utf-8")
    mutated = source.replace('[-1].split(".")) for wheel', "[-1]) for wheel", 1)
    assert mutated != source
    workflow = tmp_path / "publish-witwin-radar.yml"
    _write(workflow, mutated)
    failures = gate.check_workflow(workflow)
    assert any("parse compressed wheel platform tags" in failure for failure in failures)
