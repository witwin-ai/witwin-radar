"""The repository gates fail on the defects they exist to catch.

Each gate passes on the checkout in the quick tier; these tests show it fires,
in process, on a copy of its inputs with one defect written in.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "ci"))

import check_architecture
import check_documentation_surface
import check_workflows as lint

PUBLISH = REPO_ROOT / ".github" / "workflows" / "publish-witwin-radar.yml"
POLICY = json.loads((REPO_ROOT / "ci" / "release-policy.json").read_text(encoding="utf-8"))


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """The files the lint reads, copied so one can be broken."""

    for relative in (
        ".github",
        "ci/release-policy.json",
        "ci/required-integration-tests.json",
        "README.md",
        "FEATURE_LIST.md",
        "pyproject.toml",
    ):
        source, target = REPO_ROOT / relative, tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        (shutil.copytree if source.is_dir() else shutil.copy2)(source, target)
    return tmp_path


def _edit(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert old in text, old
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def test_the_checkout_passes():
    assert lint.audit(REPO_ROOT) == []


def test_a_push_trigger_on_the_publish_workflow_fails(tmp_path: Path):
    workflow = tmp_path / "publish.yml"
    workflow.write_text(PUBLISH.read_text(encoding="utf-8").replace("on:\n", "on:\n  push:\n", 1), encoding="utf-8")
    assert any("forbidden trigger" in failure for failure in lint.check_publish_workflow(workflow, POLICY))


def test_a_dropped_release_architecture_fails(tmp_path: Path):
    workflow = tmp_path / "publish.yml"
    workflow.write_text(PUBLISH.read_text(encoding="utf-8").replace("8.7;8.9", "8.9", 1), encoding="utf-8")
    assert any("FULL_GENCODE_ARCHES" in failure for failure in lint.check_publish_workflow(workflow, POLICY))


def test_a_missing_script_reference_fails(tree: Path):
    failures: list[str] = []
    lint.check_script_references(tree, failures)
    assert any("invokes missing script ci/run_ci_tier.py" in failure for failure in failures), failures


def test_a_required_workflow_that_skips_the_channel_identity_fails(tree: Path):
    _edit(tree / ".github" / "workflows" / "quality.yml", "run: python ci/check_channel_identity.py", "run: true")
    failures: list[str] = []
    lint.check_required_channel_coverage(tree, failures)
    assert failures == [".github/workflows/quality.yml does not run ci/check_channel_identity.py"]


def test_a_torch_range_the_loader_does_not_accept_fails(tree: Path):
    _edit(tree / "pyproject.toml", f'"torch=={POLICY["torch_version"]}"', '"torch>=2.10"')
    failures: list[str] = []
    lint.check_torch_requirement(tree, POLICY, failures)
    assert failures and "the loader accepts a packaged binary only on" in failures[0]


def test_a_retired_manylinux_claim_fails(tree: Path):
    _edit(tree / "README.md", POLICY["manylinux_policy"], "manylinux_2_35")
    failures: list[str] = []
    lint.check_release_claims(tree, POLICY, failures)
    assert failures == ["README.md names a manylinux policy other than manylinux_2_28"]


@pytest.fixture
def package(tmp_path: Path) -> Path:
    for relative in ("witwin", "ci"):
        shutil.copytree(
            REPO_ROOT / relative, tmp_path / relative, ignore=shutil.ignore_patterns("__pycache__", "*.so", "*.pyd")
        )
    return tmp_path


def test_a_second_channel_importer_fails_the_architecture_gate(package: Path):
    target = package / "witwin" / "radar" / "paths.py"
    target.write_text("import witwin.channel\n" + target.read_text(encoding="utf-8"), encoding="utf-8")
    assert any("Channel executable importers differ" in error for error in check_architecture.audit(package))


@pytest.mark.parametrize("access", ["X = torch.ops._radar_native\n", "from torch.utils import cpp_extension\n"])
def test_a_second_dispatcher_owner_fails_the_architecture_gate(package: Path, access: str):
    target = package / "witwin" / "radar" / "paths.py"
    target.write_text("import torch\n" + access + target.read_text(encoding="utf-8"), encoding="utf-8")
    errors = check_architecture.audit(package)
    assert any("torch.ops or torch.utils.cpp_extension differ" in error for error in errors), errors


def test_the_documentation_gate_checks_tracked_paths_and_skips_build_products(tmp_path: Path):
    """A gitignored build product is never asked for; a missing tracked path still fails."""

    import subprocess

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("*.pyd\n", encoding="utf-8")
    (tmp_path / "ci").mkdir()
    (tmp_path / "README.md").write_text(
        "Built into `witwin/radar/cuda/prebuilt/_radar_native.pyd`; see `docs/missing.md`.\n", encoding="utf-8"
    )
    manifest = {"living": ["README.md"], "historical_prefixes": [], "retired_living_tokens": []}
    (tmp_path / "ci" / "documentation-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert check_documentation_surface.audit(tmp_path) == ["README.md names missing current path 'docs/missing.md'"]
