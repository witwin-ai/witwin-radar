"""The four production static gates pass here, and FAIL on a planted violation.

Work item 7 asks for four prohibitions. Asserting that each gate exits 0 on the
real tree proves only that the tree is clean today - it does not prove the gate
would notice if it stopped being clean, and a gate that cannot be shown to fail
is a comment with an exit code.

So every gate is exercised twice. Once against the checkout, which is the
regression half. Once against a MIRROR of the checkout under `tmp_path` with a
single violation written into it, which is the calibration half:

* a `import drjit` and a `@dr.wrap` and an `importlib.import_module("drjit")`
  for G1;
* a `from tests.reference import dsp_oracles` in a production module, and a
  wheel carrying a `tests/` member, for G2;
* a tenth module reaching `torch.ops._radar_native` directly, and an eleventh
  taking a loader handle, for G3;
* a `torch.cdist` planted in `processing/`, a mutated allowlist document, and a
  pytest constant edited out from under the record, for G4.

The mirror is a real directory tree rather than a monkeypatched scan, because
each gate is also run as a SUBPROCESS through its own `--root`. That is how CI
invokes it, and an in-process call can pass while `main()` returns 0 anyway.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
CI_ROOT = REPO_ROOT / "ci"

if str(CI_ROOT) not in sys.path:
    sys.path.insert(0, str(CI_ROOT))

import check_production_dependencies  # noqa: E402
import check_raw_native_access  # noqa: E402
import check_test_oracle_isolation  # noqa: E402
import check_torch_physics_allowlist  # noqa: E402


GATE_SCRIPTS = (
    "check_production_dependencies.py",
    "check_test_oracle_isolation.py",
    "check_raw_native_access.py",
    "check_torch_physics_allowlist.py",
)

#: What the mirror needs for all four gates to be meaningful. `witwin/` is the
#: scanned tree; `pyproject.toml` carries the wheel configuration; the two test
#: modules hold the three constants G4 re-freezes; the JSON is G4's record.
MIRRORED = (
    "witwin",
    "pyproject.toml",
    "ci/torch-physics-allowlist.json",
    "tests/test_phase6_no_torch_physics.py",
    "tests/processing/test_cutover.py",
)


def _mirror(destination: Path) -> Path:
    """A copy of everything the four gates read, and nothing else."""

    for relative in MIRRORED:
        source = REPO_ROOT / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(
                source,
                target,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyd", "*.so"),
            )
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


@pytest.fixture
def mirror(tmp_path: Path) -> Path:
    return _mirror(tmp_path / "tree")


# ---------------------------------------------------------------------------
# The regression half: all four gates pass on the checkout.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("script", GATE_SCRIPTS)
def test_every_gate_passes_on_the_real_tree(script: str) -> None:
    completed = _run(script, REPO_ROOT)
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("script", GATE_SCRIPTS)
def test_every_gate_passes_on_the_untouched_mirror(script: str, mirror: Path) -> None:
    """Calibration for the calibration.

    Every "fires on a violation" assertion below is only worth something if the
    mirror passes BEFORE the violation is written. Without this, a mirror that
    was missing one of the files a gate reads would make all of them fail for
    the wrong reason.
    """

    completed = _run(script, mirror)
    assert completed.returncode == 0, completed.stderr


# ---------------------------------------------------------------------------
# G1: forbidden runtimes
# ---------------------------------------------------------------------------


def test_g1_fires_on_a_forbidden_import(mirror: Path) -> None:
    target = mirror / "witwin" / "radar" / "utils" / "vector.py"
    target.write_text(
        "import drjit\n" + target.read_text(encoding="utf-8"), encoding="utf-8"
    )
    completed = _run("check_production_dependencies.py", mirror)
    assert completed.returncode == 1
    assert "import: drjit" in completed.stderr


def test_g1_fires_on_a_dr_wrap_decorator(mirror: Path) -> None:
    """The scan an import-only gate cannot make.

    `@dr.wrap` needs no import statement in the module that carries it - `dr`
    can arrive by any route - so a Dr.Jit boundary can exist in a file whose
    import list is clean.
    """

    target = mirror / "witwin" / "radar" / "utils" / "vector.py"
    target.write_text(
        target.read_text(encoding="utf-8")
        + '\n\n@dr.wrap(source="torch", target="drjit")\ndef _boundary(x):\n    return x\n',
        encoding="utf-8",
    )
    completed = _run("check_production_dependencies.py", mirror)
    assert completed.returncode == 1
    assert "decorator: @dr.wrap" in completed.stderr


def test_g1_fires_on_a_lazily_imported_token(mirror: Path) -> None:
    target = mirror / "witwin" / "radar" / "utils" / "vector.py"
    target.write_text(
        target.read_text(encoding="utf-8")
        + '\n\ndef _late():\n    import importlib\n\n    return importlib.import_module("drjit")\n',
        encoding="utf-8",
    )
    completed = _run("check_production_dependencies.py", mirror)
    assert completed.returncode == 1
    assert "string literals mention 'drjit'" in completed.stderr


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

    target = mirror / "pyproject.toml"
    source = target.read_text(encoding="utf-8")
    mutated = source.replace(
        '"witwin-channel>=0.4,<0.5",',
        '"witwin-channel>=0.4,<0.5",\n    "rayd-torch>=0.1",',
        1,
    )
    assert mutated != source
    target.write_text(mutated, encoding="utf-8")
    completed = _run("check_production_dependencies.py", mirror)
    assert completed.returncode == 1
    assert "optional-dependencies.channel" in completed.stderr


def test_g1_accepts_the_channel_extra_it_is_meant_to_allow() -> None:
    """The one dependency that reaches RayD - through Channel's own build."""

    declared = [
        requirement
        for _, requirement in check_production_dependencies.declared_distributions(
            REPO_ROOT
        )
    ]
    assert "witwin-channel>=0.4,<0.5" in declared
    assert check_production_dependencies.check_declared_dependencies(REPO_ROOT) == []


def test_g1_fires_when_a_recorded_prose_occurrence_disappears(mirror: Path) -> None:
    """A stale allowlist entry is a hole that nothing else reports."""

    target = mirror / "witwin" / "radar" / "propagation" / "epochs.py"
    source = target.read_text(encoding="utf-8")
    assert "RayD" in source
    target.write_text(source.replace("RayD", "the tracer"), encoding="utf-8")
    completed = _run("check_production_dependencies.py", mirror)
    assert completed.returncode == 1
    assert "which no longer exists" in completed.stderr


# ---------------------------------------------------------------------------
# G2: test-oracle isolation
# ---------------------------------------------------------------------------


def test_g2_fires_on_a_production_import_of_the_oracle(mirror: Path) -> None:
    target = mirror / "witwin" / "radar" / "sigproc" / "__init__.py"
    target.write_text(
        "from tests.reference import dsp_oracles\n"
        + target.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    completed = _run("check_test_oracle_isolation.py", mirror)
    assert completed.returncode == 1
    assert "production imports 'tests.reference'" in completed.stderr


def test_g2_fires_when_the_wheel_configuration_would_ship_tests(mirror: Path) -> None:
    target = mirror / "pyproject.toml"
    target.write_text(
        target.read_text(encoding="utf-8").replace(
            'packages = ["witwin"]', 'packages = ["witwin", "tests"]'
        ),
        encoding="utf-8",
    )
    completed = _run("check_test_oracle_isolation.py", mirror)
    assert completed.returncode == 1
    assert "wheel packages are ['witwin', 'tests']" in completed.stderr


def test_g2_fires_on_a_wheel_that_carries_a_test_member(
    mirror: Path, tmp_path: Path
) -> None:
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


# ---------------------------------------------------------------------------
# G3: raw native access
# ---------------------------------------------------------------------------


def test_g3_fires_on_a_direct_dispatcher_call(mirror: Path) -> None:
    target = mirror / "witwin" / "radar" / "sigproc" / "__init__.py"
    target.write_text(
        target.read_text(encoding="utf-8")
        + "\n\ndef _straight_to_the_dispatcher(x):\n"
        "    import torch\n\n"
        "    return torch.ops._radar_native.two_way_join_forward(x)\n",
        encoding="utf-8",
    )
    completed = _run("check_raw_native_access.py", mirror)
    assert completed.returncode == 1
    assert "reaches the dispatcher directly" in completed.stderr


def test_g3_fires_on_an_unrecorded_loader_consumer(mirror: Path) -> None:
    target = mirror / "witwin" / "radar" / "sigproc" / "__init__.py"
    target.write_text(
        target.read_text(encoding="utf-8")
        + "\n\ndef _tenth_handle():\n"
        "    from ..cuda import build\n\n"
        "    return build.build_extension()\n",
        encoding="utf-8",
    )
    completed = _run("check_raw_native_access.py", mirror)
    assert completed.returncode == 1
    assert "is not a recorded consumer" in completed.stderr


def test_g3_fires_when_identity_gains_dispatcher_access(mirror: Path) -> None:
    """The one module whose zero-access property is asserted positively.

    `identity.py` validates before `torch.ops.load_library` and must import on
    a machine with no CUDA. A `torch.ops` reference there is how validation
    quietly moves to after the load.
    """

    target = mirror / "witwin" / "radar" / "cuda" / "identity.py"
    target.write_text(
        target.read_text(encoding="utf-8")
        + "\n\ndef _peek():\n    import torch\n\n    return torch.ops.loaded_libraries\n",
        encoding="utf-8",
    )
    completed = _run("check_raw_native_access.py", mirror)
    assert completed.returncode == 1
    assert "may hold no dispatcher access" in completed.stderr


# ---------------------------------------------------------------------------
# G4: the Torch-physics allowlist
# ---------------------------------------------------------------------------


def test_g4_fires_on_torch_physics_in_a_previously_unscanned_package(
    mirror: Path,
) -> None:
    """The exact hit the old one-package scan could not see.

    `processing/` was outside the Phase-6 scan entirely, so this expression
    would have landed in production without failing anything.
    """

    target = mirror / "witwin" / "radar" / "processing" / "primitives.py"
    target.write_text(
        target.read_text(encoding="utf-8")
        + "\n\ndef _range_field(a, b):\n    import torch\n\n    return torch.cdist(a, b)\n",
        encoding="utf-8",
    )
    completed = _run("check_torch_physics_allowlist.py", mirror)
    assert completed.returncode == 1
    assert "_range_field() calls torch.cdist 1 time(s)" in completed.stderr
    assert "not in the allowlist" in completed.stderr


def test_g4_fires_when_an_allowed_expression_gains_a_sibling(mirror: Path) -> None:
    """The reason the record carries an occurrence COUNT.

    `(module, function, call)` alone would let a second `torch.cos` appear
    inside a helper that legitimately has one, which is how a window function
    becomes a phase evaluator.
    """

    target = mirror / "witwin" / "radar" / "utils" / "vector.py"
    source = target.read_text(encoding="utf-8")
    original = (
        "    return vectors / torch.clamp("
        "torch.linalg.norm(vectors, dim=-1, keepdim=True), min=1e-12)"
    )
    assert original in source
    target.write_text(
        source.replace(
            original,
            "    lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)\n"
            "    scale = torch.linalg.norm(vectors, dim=-1, keepdim=True)\n"
            "    return vectors / torch.clamp(lengths * scale / scale, min=1e-12)",
        ),
        encoding="utf-8",
    )
    completed = _run("check_torch_physics_allowlist.py", mirror)
    assert completed.returncode == 1
    assert "the allowlist records 1" in completed.stderr


def test_g4_fires_when_the_allowlist_itself_is_edited(mirror: Path) -> None:
    """Widening the record must cost a second, deliberate edit."""

    path = mirror / "ci" / "torch-physics-allowlist.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["entries"].append(
        {
            "module": "witwin/radar/processing/primitives.py",
            "function": "_smuggled",
            "call": "torch.cdist",
            "occurrences": 1,
            "category": "dsp_window",
            "reason": "not really",
            "adr": "R-ADR-007",
        }
    )
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    completed = _run("check_torch_physics_allowlist.py", mirror)
    assert completed.returncode == 1
    assert "FROZEN_BASELINE_DIGEST" in completed.stderr


def test_g4_fires_when_the_scan_scope_is_narrowed(mirror: Path) -> None:
    """The failure mode the whole gate exists for, reproduced.

    Excluding a directory is how the previous scan grew its allowlist without
    adding to a list. Here it changes the digest AND strands every entry under
    the excluded path, so it fails twice over.
    """

    path = mirror / "ci" / "torch-physics-allowlist.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["excluded_paths"] = ["witwin/radar/processing"]
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    completed = _run("check_torch_physics_allowlist.py", mirror)
    assert completed.returncode == 1
    assert "FROZEN_BASELINE_DIGEST" in completed.stderr
    assert "no longer exists" in completed.stderr


def test_g4_fires_when_a_pytest_constant_drifts_from_the_record(mirror: Path) -> None:
    """The three lists have two homes and must agree."""

    target = mirror / "tests" / "test_phase6_no_torch_physics.py"
    source = target.read_text(encoding="utf-8")
    assert '("_set_pose_fields", "torch.linalg.norm"),' in source
    target.write_text(
        source.replace('("_set_pose_fields", "torch.linalg.norm"),', ""),
        encoding="utf-8",
    )
    completed = _run("check_torch_physics_allowlist.py", mirror)
    assert completed.returncode == 1
    assert "RADAR_FACADE_TORCH_PHYSICS disagrees with the allowlist" in completed.stderr


def test_g4_fires_when_the_fence_allowance_list_drifts(mirror: Path) -> None:
    """Phase 11 emptied the list, so the drift under test is an ADDITION.

    The mutation used to append a second entry beside the Dirichlet solver's.
    That entry is gone with its route, and adding the FIRST allowance is the
    sharper test anyway: it is exactly the shape of change this gate exists to
    catch.
    """

    target = mirror / "tests" / "processing" / "test_cutover.py"
    source = target.read_text(encoding="utf-8")
    empty = "FENCE_ALLOWANCES = {}"
    assert empty in source
    smuggled = (
        "FENCE_ALLOWANCES = {\n"
        '    "witwin/radar/processing/aoa.py": "smuggled in",\n'
        "}"
    )
    target.write_text(source.replace(empty, smuggled, 1), encoding="utf-8")
    completed = _run("check_torch_physics_allowlist.py", mirror)
    assert completed.returncode == 1
    assert "FENCE_ALLOWANCES disagrees with the allowlist" in completed.stderr


# ---------------------------------------------------------------------------
# The record itself
# ---------------------------------------------------------------------------


def test_the_allowlist_records_every_measured_expression_with_a_reason() -> None:
    """No entry may be a bare line item.

    A classification whose reason is empty is an allowlist entry, and an
    allowlist entry that nobody had to justify is how the list grows.
    """

    document = json.loads(
        (REPO_ROOT / "ci" / "torch-physics-allowlist.json").read_text(encoding="utf-8")
    )
    measured = check_torch_physics_allowlist.scan(
        REPO_ROOT,
        scanned_root=document["scanned_root"],
        excluded=tuple(document["excluded_paths"]),
        forbidden=tuple(document["forbidden_torch_calls"]),
    )
    assert len(document["entries"]) == len(measured)
    assert sum(entry["occurrences"] for entry in document["entries"]) == sum(
        measured.values()
    )
    for entry in document["entries"]:
        assert len(entry["reason"]) > 30, entry
        assert entry["adr"].startswith("R-ADR-"), entry


def test_the_recorded_debt_is_named_as_debt() -> None:
    """Debt is not approval, and the record must keep saying so.

    There were two debt categories. `work_item_8_survivor` is CLOSED: its two
    entries were `Radar.waveform` and `NoiseModelRuntime._apply_phase_noise`,
    and Phase 11 deleted both expressions rather than reclassifying them. The
    category is asserted ABSENT from the description map and from the entries,
    because "the debt disappeared from the record" and "the debt was paid" look
    identical in a subset check.

    `freeze_time_pattern_oracle` is still debt. If a later change quietly
    reclassified it as an ordinary allowlist entry, the gate would still pass
    and the debt would disappear from the record.
    """

    document = json.loads(
        (REPO_ROOT / "ci" / "torch-physics-allowlist.json").read_text(encoding="utf-8")
    )
    debt = {"freeze_time_pattern_oracle"}
    assert debt <= set(document["categories"])
    for name in debt:
        assert "DEBT" in document["categories"][name]
    recorded = {entry["category"] for entry in document["entries"]}
    assert debt <= recorded
    assert "work_item_8_survivor" not in document["categories"]
    assert "work_item_8_survivor" not in recorded


def test_the_dispatcher_owner_set_is_a_single_module() -> None:
    assert check_raw_native_access.DISPATCHER_OWNERS == frozenset(
        {"witwin/radar/cuda/build.py"}
    )
    dispatcher, consumers = check_raw_native_access.scan(REPO_ROOT)
    assert set(dispatcher) == check_raw_native_access.DISPATCHER_OWNERS
    assert consumers == set(check_raw_native_access.EXPECTED_LOADER_CONSUMERS)


def test_the_production_token_census_has_exactly_two_prose_entries() -> None:
    violations, census = check_production_dependencies.scan(REPO_ROOT)
    assert violations == []
    assert census == set(check_production_dependencies.ALLOWED_TOKEN_OCCURRENCES)


def test_the_wheel_package_list_is_witwin_alone() -> None:
    assert check_test_oracle_isolation.check_packaging(REPO_ROOT) == []
