"""The two Phase-11 repository gates pass here, and FAIL on a planted violation.

`tests/test_phase10_static_gates.py` established the rule this file follows: a
gate asserted only against a clean tree proves the tree is clean today, not that
the gate would notice tomorrow. So each new gate runs twice - once against the
checkout, once against a mirror of it carrying exactly one violation - and both
halves invoke the gate the way CI does, as a subprocess with its own `--root`.

The two gates are the dead-code half of acceptance criterion 8:

* `ci/check_orphan_modules.py` - a production module with no path to it from a
  declared entry point. This is the shape `witwin/radar/timeline.py` had for
  four phases: every import inside it used, no importer above it, ruff silent.
* `tests/test_public_api_snapshot.py` - the frozen public surface. Here the
  planted violation is an export added to `__all__`, which must move the
  snapshot AND trip the unused-export scan.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
ORPHAN_GATE = REPO_ROOT / "ci" / "check_orphan_modules.py"
SNAPSHOT = REPO_ROOT / "ci" / "public-api-snapshot.json"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ORPHAN_GATE), *args],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture
def mirror(tmp_path: Path) -> Path:
    """A copy of `witwin/` deep enough for the orphan gate to run over it."""

    root = tmp_path / "mirror"
    (root / "witwin").mkdir(parents=True)
    shutil.copytree(
        REPO_ROOT / "witwin" / "radar",
        root / "witwin" / "radar",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyd", "prebuilt"),
    )
    return root


def test_the_orphan_gate_passes_on_the_real_tree() -> None:
    completed = _run()
    assert completed.returncode == 0, completed.stderr
    assert "all reachable" in completed.stdout


def test_the_orphan_gate_passes_on_the_untouched_mirror(mirror: Path) -> None:
    completed = _run("--root", str(mirror))
    assert completed.returncode == 0, completed.stderr


def test_the_orphan_gate_fires_on_a_module_nobody_imports(mirror: Path) -> None:
    planted = mirror / "witwin" / "radar" / "timeline.py"
    planted.write_text(
        "from __future__ import annotations\n\n"
        "import torch\n\n\n"
        "def sample(times: torch.Tensor) -> torch.Tensor:\n"
        "    return times.clone()\n",
        encoding="utf-8",
    )

    completed = _run("--root", str(mirror))
    assert completed.returncode == 1
    assert "witwin.radar.timeline" in completed.stderr
    assert "unreachable production module" in completed.stderr


def test_the_orphan_gate_fires_on_a_whole_dead_subpackage(mirror: Path) -> None:
    """Two modules that import each other are still unreachable."""

    package = mirror / "witwin" / "radar" / "legacy"
    package.mkdir()
    (package / "__init__.py").write_text("from .tracer import Tracer\n", encoding="utf-8")
    (package / "tracer.py").write_text(
        "from . import __name__ as _pkg\n\n\nclass Tracer:\n    pass\n", encoding="utf-8"
    )

    completed = _run("--root", str(mirror))
    assert completed.returncode == 1
    assert "witwin.radar.legacy" in completed.stderr
    assert "witwin.radar.legacy.tracer" in completed.stderr


def test_the_orphan_gate_refuses_a_stale_entry_point(mirror: Path) -> None:
    """An allowlist entry that names a deleted module is itself a defect."""

    shutil.rmtree(mirror / "witwin" / "radar" / "sigproc")
    completed = _run("--root", str(mirror))
    assert completed.returncode == 1
    assert "ENTRY_POINTS names modules that do not exist" in completed.stderr


def test_the_snapshot_file_is_the_generator_output_verbatim() -> None:
    """Regenerating must be a no-op, so a diff is always a real surface change."""

    from test_public_api_snapshot import build_snapshot

    on_disk = SNAPSHOT.read_text(encoding="utf-8")
    assert on_disk == json.dumps(build_snapshot(), indent=2) + "\n"


def test_the_snapshot_notices_a_new_export() -> None:
    """The frozen surface is a real comparison, not a length check."""

    from test_public_api_snapshot import build_snapshot

    frozen = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    mutated = json.loads(json.dumps(frozen))
    mutated["modules"][0]["exports"].append(
        {"name": "Tracer", "kind": "class", "target": "witwin.radar.trace.Tracer"}
    )
    assert build_snapshot() != mutated
    assert build_snapshot() == frozen
