"""Calibration tests for the concept-axis consolidation governance gates."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


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


def test_architecture_gate_finds_a_second_channel_importer(tmp_path: Path) -> None:
    gate = _load("check_architecture")
    modules = ("witwin.radar", "witwin.radar.channel")
    _json(
        tmp_path / "ci" / "architecture-manifest.json",
        {
            "schema_version": 1,
            "package": "witwin.radar",
            "channel_importer": "witwin.radar.channel",
            "target_modules": modules,
            "concept_owners": {"channel_boundary": "witwin.radar.channel"},
        },
    )
    _write(tmp_path / "witwin" / "radar" / "__init__.py", "")
    _write(tmp_path / "witwin" / "radar" / "channel.py", "import witwin.channel\n")
    assert gate.audit(tmp_path) == []

    _write(tmp_path / "witwin" / "radar" / "extra.py", "import witwin.channel\n")
    errors = gate.audit(tmp_path)
    assert any("unexpected production module" in error for error in errors)
    assert any("Channel executable importers differ" in error for error in errors)


def test_architecture_gate_does_not_treat_plain_strings_as_edges(tmp_path: Path) -> None:
    gate = _load("check_architecture")
    _write(tmp_path / "owner.py", 'TOKEN = "witwin.radar.hidden"\n')
    edges, imports_channel = gate._edges(
        tmp_path / "owner.py",
        "witwin.radar.owner",
        {"witwin.radar.owner", "witwin.radar.hidden"},
    )
    assert edges == set()
    assert imports_channel is False


def test_no_compatibility_gate_finds_python_and_native_shims(tmp_path: Path) -> None:
    gate = _load("check_no_compatibility")
    _json(
        tmp_path / "ci" / "public-api-manifest.json",
        {
            "modules": {
                "witwin.radar": {
                    "Radar": "witwin.radar.radar.Radar",
                    "RadarConfig": "witwin.radar.radar.RadarConfig",
                }
            }
        },
    )
    _write(
        tmp_path / "witwin" / "radar" / "__init__.py",
        "__all__ = ['Radar', 'RadarConfig']\n",
    )
    assert gate.audit(tmp_path) == []

    _write(
        tmp_path / "witwin" / "radar" / "__init__.py",
        "__all__ = ['Radar', 'RadarConfig']\n"
        "Box = object()\n"
        "_LAZY = {}\n"
        "def __getattr__(name):\n"
        "    raise AttributeError(name)\n",
    )
    root_errors = gate.audit(tmp_path)
    assert any(
        "retired root API remains bound outside __all__: Box" in error
        for error in root_errors
    )
    assert any(
        "root compatibility proxy remains bound: _LAZY" in error
        for error in root_errors
    )
    assert any(
        "root compatibility proxy remains bound: __getattr__" in error
        for error in root_errors
    )

    _write(
        tmp_path / "witwin" / "radar" / "__init__.py",
        "__all__ = ['Radar', 'RadarConfig']\n",
    )
    _write(tmp_path / "witwin" / "radar" / "sigproc" / "__init__.py", "")
    _write(
        tmp_path / "witwin" / "radar" / "cuda" / "sensor.cu",
        "bool legacy_real_polarization;\n"
        "int spreading_mode;\n"
        "float *normals;\n",
    )
    errors = gate.audit(tmp_path)
    assert any("retired path exists" in error for error in errors)
    assert sum("retired native compatibility token" in error for error in errors) == 3

def test_documentation_gate_finds_retired_and_missing_paths(tmp_path: Path) -> None:
    gate = _load("check_documentation_surface")
    _json(
        tmp_path / "ci" / "documentation-manifest.json",
        {
            "living": ["README.md"],
            "historical_prefixes": ["docs/history/"],
            "retired_living_tokens": ["witwin.radar.sigproc"],
        },
    )
    _write(
        tmp_path / "README.md",
        "Use witwin.radar.sigproc and `witwin/radar/missing.py`.\n",
    )
    errors = gate.audit(tmp_path)
    assert any("retired current token" in error for error in errors)
    assert any("missing current path" in error for error in errors)


def test_workflow_reference_gate_finds_a_missing_script(tmp_path: Path) -> None:
    gate = _load("check_workflow_references")
    _write(
        tmp_path / ".github" / "workflows" / "quality.yml",
        "run: python tools/does_not_exist.py\n",
    )
    assert gate.audit(tmp_path) == [
        ".github/workflows/quality.yml invokes missing script tools/does_not_exist.py"
    ]

    _write(
        tmp_path / ".github" / "workflows" / "quality.yml",
        "# run: python tools/does_not_exist.py\n",
    )
    assert gate.audit(tmp_path) == []


def test_required_channel_gate_checks_install_fingerprint_and_skip_budget(
    tmp_path: Path,
) -> None:
    gate = _load("check_required_channel_coverage")
    relative = ".github/workflows/quality.yml"
    _json(
        tmp_path / "ci" / "required-integration-tests.json",
        {
            "required_workflows": [relative],
            "allowed_channel_skips": 0,
        },
    )
    _write(tmp_path / relative, "run: pip install .[dev]\n")
    errors = gate.audit(tmp_path)
    assert len(errors) == 3

    _write(
        tmp_path / relative,
        "# run: pip install .[dev,channel]\n"
        "# WITWIN_CHANNEL_FINGERPRINT build_info build_fingerprint witwin.channel\n"
        "# WITWIN_REQUIRED_CHANNEL_SKIP_BUDGET\n",
    )
    assert len(gate.audit(tmp_path)) == 3

    _write(
        tmp_path / relative,
        "env:\n"
        "  WITWIN_CHANNEL_FINGERPRINT: required\n"
        "  WITWIN_REQUIRED_CHANNEL_SKIP_BUDGET: '0'\n"
        "run: |\n"
        "  python -m pip install .[dev,channel]\n"
        "  python -c \"import witwin.channel; from witwin.channel import build_info; print(build_info()['build_fingerprint'])\"\n"
        "  echo $WITWIN_CHANNEL_FINGERPRINT\n"
        "  echo $WITWIN_REQUIRED_CHANNEL_SKIP_BUDGET\n",
    )
    assert gate.audit(tmp_path) == []
