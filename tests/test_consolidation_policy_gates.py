"""Calibration tests for consolidation policy and inventory gates."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
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


def test_single_definition_gate_rejects_shared_owner_and_target(tmp_path: Path) -> None:
    gate = _load("check_single_definition")
    _json(
        tmp_path / "ci" / "architecture-manifest.json",
        {
            "target_modules": ["witwin.radar.radar"],
            "concept_owners": {"configuration": "witwin.radar.radar", "session": "witwin.radar.radar"},
        },
    )
    _json(
        tmp_path / "ci" / "public-api-manifest.json",
        {"modules": {"witwin.radar": {"Radar": "witwin.radar.radar.Radar", "RadarAlias": "witwin.radar.radar.Radar"}}},
    )
    errors = gate.audit(tmp_path)
    assert any("owns multiple concept axes" in error for error in errors)
    assert any("has multiple exposures" in error for error in errors)


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


def test_release_claim_gate_rejects_retired_policy_and_false_success(tmp_path: Path) -> None:
    gate = _load("check_release_claims")
    _json(
        tmp_path / "ci" / "release-policy.json",
        {
            "manylinux_policy": "manylinux_2_28",
            "stable_abi_cross_torch_claim": False,
            "expected_loader_refusal_is_release_success": False,
        },
    )
    _write(tmp_path / "README.md", "manylinux_2_35 cross-Torch Stable ABI\n")
    _write(tmp_path / "FEATURE_LIST.md", "manylinux_2_28\n")
    _write(tmp_path / "docs" / "dev" / "plans" / "phase10-deferred-release-matrix.md", "manylinux_2_28\n")
    _write(
        tmp_path / ".github" / "workflows" / "publish-witwin-radar.yml",
        "script: |\n"
        "  try:\n"
        "      load()\n"
        "  except build.RadarExtensionABIError:\n"
        "      print('This cell measures deviation P3, not a passing Stable ABI cell.')\n"
        "      raise SystemExit(0)\n",
    )
    errors = gate.audit(tmp_path)
    assert any("retired manylinux_2_35" in error for error in errors)
    assert any("cross-Torch Stable ABI" in error for error in errors)
    assert any("loader refusal" in error for error in errors)


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


def test_governance_inventory_gate_rejects_open_and_unproven_rows(tmp_path: Path) -> None:
    gate = _load("check_governance_inventory")
    header = "| ID | Debt | Owner | Phase | Falsifier | Scope | Status | Evidence |\n"
    divider = "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
    rows = []
    for index in range(1, 28):
        status = "open" if index == 1 else "closed"
        evidence = "—" if index in {1, 2} else "proof"
        rows.append(f"| GOV-{index:03d} | debt | owner | G | gate | scope | {status} | {evidence} |\n")
    path = tmp_path / "docs" / "dev" / "audit" / "radar-governance-debt-and-drift-inventory.md"
    _write(path, header + divider + "".join(rows))
    assert gate.audit(tmp_path, require_closed=False) == ["GOV-002 is closed without evidence"]
    errors = gate.audit(tmp_path, require_closed=True)
    assert "GOV-001 is not closed: open" in errors
    assert "GOV-002 is closed without evidence" in errors
