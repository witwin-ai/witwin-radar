#!/usr/bin/env python
"""Reject retired Radar Python/root/native compatibility surfaces."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

RETIRED_PATHS = ("witwin/radar/sigproc", "witwin/radar/processing/adapters.py")
RETIRED_NAMES = {
    "FmcwBeatSpec",
    "PolarizationSpec",
    "RadarAxes",
    "SensorWeightModes",
    "from_real_amplitudes",
    "synthesize_fmcw_beat",
}
RETIRED_ROOT_NAMES = {
    "Box",
    "Cone",
    "Cylinder",
    "DetectorType",
    "Ellipsoid",
    "Geometry",
    "GeometryBase",
    "HollowBox",
    "Material",
    "Mesh",
    "Prism",
    "Pyramid",
    "RadarPropagationLegs",
    "RadarSimulationResult",
    "RadarWorldBinding",
    "SMPLBody",
    "ScatterSitePolicy",
    "Sphere",
    "StableIdAllocator",
    "Structure",
    "Torus",
    "bind_radar_world",
    "build_info",
    "capabilities",
    "require_supported_runtime",
    "runtime_diagnostics",
}
ROOT_PROXY_NAMES = {"_LAZY", "_REMOVED", "__getattr__"}
RETIRED_NATIVE_TOKENS = (
    "legacy_real_polarization",
    "spreading_mode",
    "tx_power_mode",
    "reflection_flip",
    "normals",
    "pol_tx",
    "pol_rx",
    "local_axes",
)


def _literal_all(tree: ast.Module) -> tuple[str, ...] | None:
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(target, ast.Name) and target.id == "__all__" for target in targets):
            continue
        value = node.value
        if isinstance(value, (ast.List, ast.Tuple)) and all(
            isinstance(item, ast.Constant) and isinstance(item.value, str) for item in value.elts
        ):
            return tuple(item.value for item in value.elts)
    return None


def _root_bindings(tree: ast.Module) -> set[str]:
    bindings: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bindings.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bindings.add(alias.asname or alias.name.rpartition(".")[2])
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            bindings.update(target.id for target in targets if isinstance(target, ast.Name))
    return bindings


def audit(repo: Path) -> list[str]:
    errors: list[str] = []
    for relative in RETIRED_PATHS:
        if (repo / relative).exists():
            errors.append(f"retired path exists: {relative}")

    public = json.loads((repo / "ci" / "public-api-manifest.json").read_text(encoding="utf-8"))
    expected_root = tuple(sorted(public["modules"]["witwin.radar"]))
    root_path = repo / "witwin" / "radar" / "__init__.py"
    root_tree = ast.parse(root_path.read_text(encoding="utf-8"), filename=str(root_path))
    actual_root = _literal_all(root_tree)
    if actual_root is None or tuple(sorted(actual_root)) != expected_root:
        errors.append(f"root __all__ must be exactly {expected_root}, got {actual_root}")
    root_bindings = _root_bindings(root_tree)
    for name in sorted(root_bindings & RETIRED_ROOT_NAMES):
        errors.append(f"retired root API remains bound outside __all__: {name}")
    for name in sorted(root_bindings & ROOT_PROXY_NAMES):
        errors.append(f"root compatibility proxy remains bound: {name}")

    for path in sorted((repo / "witwin" / "radar").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        relative = path.relative_to(repo).as_posix()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name in RETIRED_NAMES:
                    errors.append(f"retired definition {node.name}: {relative}:{node.lineno}")
            elif isinstance(node, ast.Name) and node.id in {"_REMOVED", "_LAZY"}:
                errors.append(f"compatibility proxy {node.id}: {relative}:{node.lineno}")
            elif isinstance(node, ast.Name) and node.id == "DeprecationWarning":
                errors.append(f"deprecation shim warning: {relative}:{node.lineno}")
        text = path.read_text(encoding="utf-8")
        if "witwin.radar.sigproc" in text or "witwin/radar/sigproc" in text:
            errors.append(f"retired sigproc path named by production: {relative}")

    for path in sorted((repo / "witwin" / "radar" / "cuda").rglob("*")):
        if path.suffix not in {".cpp", ".cu", ".cuh", ".h"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for token in RETIRED_NATIVE_TOKENS:
            if token in text:
                errors.append(f"retired native compatibility token {token!r}: {path.relative_to(repo).as_posix()}")
    return sorted(set(errors))


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    errors = audit(repo)
    if errors:
        for error in errors:
            print(f"ci/check_no_compatibility.py: {error}", file=sys.stderr)
        return 1
    print("ci/check_no_compatibility.py: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
