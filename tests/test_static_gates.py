"""The two production static gates pass on a clean copy and FAIL on a planted violation.

G1 (``ci/check_production_dependencies.py``) forbids a ray-tracing runtime in
production code or in a declared dependency; G2
(``ci/check_test_oracle_isolation.py``) forbids production code or the wheel
from carrying the test oracles. The quick tier runs both on the checkout; what
is asserted here is that each one notices a violation. Each gate runs as a
SUBPROCESS through its own ``--root`` against a MIRROR of the files it reads,
because that is how CI invokes it.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CI_ROOT = REPO_ROOT / "ci"

if str(CI_ROOT) not in sys.path:
    sys.path.insert(0, str(CI_ROOT))


GATE_SCRIPTS = ("check_production_dependencies.py", "check_test_oracle_isolation.py")

#: What the two gates read: the scanned package and the wheel configuration.
MIRRORED = ("witwin", "pyproject.toml")


def _mirror(destination: Path) -> Path:
    for relative in MIRRORED:
        source, target = REPO_ROOT / relative, destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyd", "*.so"))
        else:
            shutil.copy2(source, target)
    return destination


def _run(script: str, root: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(CI_ROOT / script), "--root", str(root), *extra],
        capture_output=True,
        text=True,
        check=False,
    )


def _append_to_channel_extra(pyproject: Path, requirement: str) -> None:
    source = pyproject.read_text(encoding="utf-8")
    mutated = source.replace("channel = [\n", f'channel = [\n    "{requirement}",\n', 1)
    assert mutated != source
    pyproject.write_text(mutated, encoding="utf-8")


@pytest.fixture
def mirror(tmp_path: Path) -> Path:
    return _mirror(tmp_path / "tree")


@pytest.mark.parametrize("script", GATE_SCRIPTS)
def test_every_gate_passes_on_the_untouched_mirror(script: str, mirror: Path) -> None:
    """Without this, a mirror missing a file a gate reads would make every case below fail for the wrong reason."""

    completed = _run(script, mirror)
    assert completed.returncode == 0, completed.stderr


# ---------------------------------------------------------------------------
# G1: forbidden runtimes
# ---------------------------------------------------------------------------


def test_g1_fires_on_a_forbidden_import(mirror: Path) -> None:
    target = mirror / "witwin" / "radar" / "radar.py"
    target.write_text("import drjit\n" + target.read_text(encoding="utf-8"), encoding="utf-8")
    completed = _run("check_production_dependencies.py", mirror)
    assert completed.returncode == 1
    assert "import: drjit" in completed.stderr


def test_g1_fires_on_a_declared_ray_tracing_dependency(mirror: Path) -> None:
    """The route that needs no import at all.

    A `rayd` requirement in `pyproject.toml` installs a ray-tracing runtime
    beside Radar without a single production module mentioning it, which is
    exactly the shape criterion A8 forbids.
    """

    target = mirror / "pyproject.toml"
    source = target.read_text(encoding="utf-8")
    mutated = source.replace('"scipy>=1.10",', '"scipy>=1.10",\n    "rayd>=0.1",', 1)
    assert mutated != source
    target.write_text(mutated, encoding="utf-8")
    completed = _run("check_production_dependencies.py", mirror)
    assert completed.returncode == 1
    assert "project.dependencies" in completed.stderr
    assert "must never be a Radar requirement" in completed.stderr


def test_g1_fires_on_a_ray_tracing_extra(mirror: Path) -> None:
    """An extra is an installable route, not a comment."""

    _append_to_channel_extra(mirror / "pyproject.toml", "rayd-torch>=0.1")
    completed = _run("check_production_dependencies.py", mirror)
    assert completed.returncode == 1
    assert "optional-dependencies.channel" in completed.stderr


# ---------------------------------------------------------------------------
# G2: test-oracle isolation
# ---------------------------------------------------------------------------


def test_g2_fires_on_a_production_import_of_the_oracle(mirror: Path) -> None:
    target = mirror / "witwin" / "radar" / "processing" / "__init__.py"
    target.write_text(
        "from tests.reference import dsp_oracles\n" + target.read_text(encoding="utf-8"), encoding="utf-8"
    )
    completed = _run("check_test_oracle_isolation.py", mirror)
    assert completed.returncode == 1
    assert "production imports 'tests.reference'" in completed.stderr


def test_g2_fires_on_a_wheel_that_carries_a_test_member(mirror: Path, tmp_path: Path) -> None:
    """The configuration and the artifact are different questions.

    A build hook runs between them. This is the half that would catch a hook
    that packed something the configuration never named.
    """

    wheel = tmp_path / "witwin_radar-0.0.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("witwin/radar/__init__.py", "")
        archive.writestr("tests/reference/dsp_oracles.py", "")
    completed = _run("check_test_oracle_isolation.py", mirror, "--wheel", str(wheel))
    assert completed.returncode == 1
    assert "ships 1 test member(s)" in completed.stderr
