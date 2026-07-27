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

The file also pins the cutover migration note against the removals it claims to
document. The note had no consumer at all: deleting it outright left every gate
and every test green, so the one artifact a migrating caller reads could rot or
vanish silently while the API breaks stayed pinned.
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
MIGRATION_NOTE = (
    REPO_ROOT / "docs" / "dev" / "migration" / "phase11-cutover-migration-note.md"
)

#: Removed public names that predate Phase 11 and are documented by the phase
#: that removed them. `Tracer` and `fresnel` went with the Dr.Jit tracer in
#: Phase 5; this note covers the cutover, so it does not restate them.
_DOCUMENTED_ELSEWHERE = frozenset({"Tracer", "fresnel"})

#: Breaks that are not a name in `_REMOVED` - a deleted constructor parameter
#: and a deleted method leave no refusal message behind, so nothing but the
#: note tells a caller what happened.
_UNNAMED_BREAKS = ("pad_factor", "simulate_group", "last_trace")


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


def test_the_migration_note_documents_every_removed_export() -> None:
    """Nothing else consumes the note, so nothing else notices it rotting.

    A removed export raises an `AttributeError` that names its replacement, so
    the API break itself is pinned. The note is the only place a caller learns
    what happened without first writing the broken call, and deleting it - or
    adding a removal without a paragraph for it - was invisible to every gate
    in this repository.
    """

    from witwin.radar import _REMOVED

    text = MIGRATION_NOTE.read_text(encoding="utf-8")
    undocumented = sorted(
        name
        for name in _REMOVED
        if name not in _DOCUMENTED_ELSEWHERE and name not in text
    )
    assert undocumented == []


def test_the_migration_note_documents_the_breaks_that_leave_no_refusal() -> None:
    text = MIGRATION_NOTE.read_text(encoding="utf-8")
    assert [name for name in _UNNAMED_BREAKS if name not in text] == []


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
