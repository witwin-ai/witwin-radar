#!/usr/bin/env python
"""Validate the symbol-level target API, ownership and live facades."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path


def _exposures(manifest: dict[str, object]) -> dict[str, str]:
    return {
        f"{module}.{name}": target
        for module, exports in manifest["modules"].items()
        for name, target in exports.items()
    }


def audit_manifest(manifest: dict[str, object]) -> list[str]:
    errors: list[str] = []
    seen_targets: dict[str, str] = {}
    for module, exports in manifest["modules"].items():
        if not exports:
            errors.append(f"public module has no exports: {module}")
        for name, target in exports.items():
            key = f"{module}.{name}"
            if target in seen_targets:
                errors.append(f"target {target} exposed twice: {seen_targets[target]} and {key}")
            seen_targets[target] = key
    for class_name, members in manifest["root_class_members"].items():
        if len(members) != len(set(members)):
            errors.append(f"duplicate public class member in {class_name}")
    return errors


def audit_policy(manifest: dict[str, object], repo: Path) -> list[str]:
    errors: list[str] = []
    modules = set(manifest["modules"])
    contracts = manifest.get("module_contracts", {})
    if set(contracts) != modules:
        errors.append("module_contracts must cover every public module exactly")
    for module, contract in contracts.items():
        caller = repo / contract.get("primary_caller", "")
        if not caller.is_file():
            errors.append(f"{module} names missing primary caller {caller}")
        if not str(contract.get("reason", "")).strip():
            errors.append(f"{module} has no public-retention reason")

    exposures = _exposures(manifest)
    values = set(manifest.get("value_exports", ()))
    unknown_values = sorted(values - set(exposures))
    errors.extend(f"value export is not public: {name}" for name in unknown_values)

    snapshot = repo / str(manifest.get("signature_snapshot", ""))
    if not snapshot.is_file():
        errors.append(f"signature snapshot is missing: {snapshot}")
    else:
        record = json.loads(snapshot.read_text(encoding="utf-8"))
        if record.get("schema_version") != 2:
            errors.append("signature snapshot schema_version must be 2")
        snapshot_modules = [entry.get("module") for entry in record.get("modules", [])]
        if snapshot_modules != list(manifest["modules"]):
            errors.append("signature snapshot modules must cover public modules exactly")
        snapshot_classes = [entry.get("class") for entry in record.get("classes", [])]
        if snapshot_classes != list(manifest["root_class_members"]):
            errors.append("signature snapshot classes must cover root_class_members exactly")

    for exposure, axes in manifest.get("result_contracts", {}).items():
        if exposure not in exposures:
            errors.append(f"result contract names non-public exposure {exposure}")
        if not isinstance(axes, list) or not axes:
            errors.append(f"result contract {exposure} has no axes")
    return errors


def audit_live(manifest: dict[str, object]) -> list[str]:
    errors: list[str] = []
    value_exports = set(manifest.get("value_exports", ()))
    for module_name, expected in manifest["modules"].items():
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            errors.append(f"cannot import public module {module_name}: {exc}")
            continue
        actual = tuple(sorted(getattr(module, "__all__", ())))
        wanted = tuple(sorted(expected))
        if actual != wanted:
            errors.append(f"{module_name}.__all__: expected {wanted}, got {actual}")
            continue
        for name, target in expected.items():
            exposure = f"{module_name}.{name}"
            obj = getattr(module, name)
            if exposure in value_exports:
                continue
            actual_target = f"{getattr(obj, '__module__', '')}.{getattr(obj, '__qualname__', '')}"
            if actual_target != target:
                errors.append(f"{exposure}: expected owner {target}, got {actual_target}")
    for dotted, expected_members in manifest["root_class_members"].items():
        module_name, _, name = dotted.rpartition(".")
        try:
            cls = getattr(importlib.import_module(module_name), name)
        except Exception as exc:
            errors.append(f"cannot resolve public class {dotted}: {exc}")
            continue
        actual = sorted(member for member in vars(cls) if not member.startswith("_"))
        if actual != sorted(expected_members):
            errors.append(f"{dotted} public members: expected {sorted(expected_members)}, got {actual}")
    return errors


def main(argv: list[str] | None = None) -> int:
    repo = Path(__file__).resolve().parents[1]
    manifest = json.loads((repo / "ci" / "public-api-manifest.json").read_text(encoding="utf-8"))
    errors = []
    if manifest.get("schema_version") != 2:
        errors.append("public API manifest schema_version must be 2")
    errors.extend(audit_manifest(manifest))
    errors.extend(audit_policy(manifest, repo))
    if "--manifest-only" not in (argv if argv is not None else sys.argv[1:]):
        errors.extend(audit_live(manifest))
    if errors:
        for error in errors:
            print(f"ci/check_public_api_manifest.py: {error}", file=sys.stderr)
        return 1
    print("ci/check_public_api_manifest.py: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
