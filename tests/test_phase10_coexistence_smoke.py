"""Prove that ``ci/coexistence_smoke.py`` can fail.

The smoke itself needs three built wheels and about two minutes, so it is a
nightly gate rather than a unit test. What IS worth asserting on every run is
that its judgement is real: that the forbidden-runtime matcher matches, that
the requirement parser reads extras, that a scenario which prints the wrong
thing is rejected rather than shrugged at, and that the generated scenario
scripts are syntactically valid Python before a subprocess ever sees them.

A gate nobody has watched fail is a gate nobody knows the shape of.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _load_smoke():
    path = REPOSITORY_ROOT / "ci" / "coexistence_smoke.py"
    spec = importlib.util.spec_from_file_location("radar_coexistence_smoke", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


smoke = _load_smoke()


class TestTheForbiddenRuntimeMatcher:
    def test_it_names_every_forbidden_runtime(self):
        assert smoke._forbidden(
            ["numpy", "rayd", "drjit", "mitsuba", "sionna", "torch"]
        ) == ["drjit", "mitsuba", "rayd", "sionna"]

    def test_it_catches_a_dashed_distribution_of_a_forbidden_runtime(self):
        assert smoke._forbidden(["rayd-torch", "rayd-drjit"]) == [
            "rayd-drjit",
            "rayd-torch",
        ]

    def test_it_does_not_fire_on_a_name_that_merely_starts_the_same(self):
        # ``raydium`` is not ``rayd``. A prefix match without the separator
        # would make this gate reject unrelated packages and get relaxed.
        assert smoke._forbidden(["raydium", "drjitter"]) == []

    def test_the_witwin_closure_is_clean(self):
        assert smoke._forbidden(["witwin", "witwin-channel", "witwin-radar"]) == []


class TestTheRequirementParser:
    def test_it_reads_extras_as_well_as_base_requirements(self):
        metadata = (
            "Metadata-Version: 2.4\n"
            "Name: witwin-radar\n"
            "Requires-Dist: torch>=2.10\n"
            "Requires-Dist: witwin-channel<0.5,>=0.4; extra == 'channel'\n"
            "Requires-Dist: pytest; extra == 'dev'\n"
        )
        assert smoke._requirement_names(metadata) == [
            "torch",
            "witwin-channel",
            "pytest",
        ]

    def test_it_normalizes_the_distribution_name(self):
        metadata = "Requires-Dist: Witwin_Channel >= 0.4\n"
        assert smoke._requirement_names(metadata) == ["witwin-channel"]

    def test_it_refuses_a_requirement_it_cannot_read(self):
        with pytest.raises(smoke.CoexistenceError):
            smoke._requirement_names("Requires-Dist: ==1.0\n")


class TestTheWheelReaders:
    def _wheel(self, tmp_path: Path, members: dict[str, bytes]) -> Path:
        wheel = tmp_path / "witwin_radar-0.3.0-py3-none-win_amd64.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            for name, payload in members.items():
                archive.writestr(name, payload)
        return wheel

    def test_a_wheel_without_a_native_member_is_refused(self, tmp_path: Path):
        wheel = self._wheel(tmp_path, {"witwin/radar/__init__.py": b""})
        with pytest.raises(smoke.CoexistenceError, match="no radar native member"):
            smoke._wheel_native_record(wheel)

    def test_a_native_member_without_its_build_record_is_refused(
        self, tmp_path: Path
    ):
        wheel = self._wheel(
            tmp_path, {"witwin/radar/cuda/prebuilt/_radar_native.pyd": b"MZ"}
        )
        with pytest.raises(smoke.CoexistenceError, match="build-info.json"):
            smoke._wheel_native_record(wheel)

    def test_the_build_record_is_read_from_beside_the_binary(self, tmp_path: Path):
        record = {"source_fingerprint": "ab" * 32, "build_type": "developer"}
        wheel = self._wheel(
            tmp_path,
            {
                "witwin/radar/cuda/prebuilt/_radar_native.pyd": b"MZ",
                "witwin/radar/cuda/prebuilt/_radar_native.build-info.json": json.dumps(
                    record
                ).encode("ascii"),
            },
        )
        assert smoke._wheel_native_record(wheel) == record

    def test_a_wheel_with_two_dist_info_metadata_members_is_refused(
        self, tmp_path: Path
    ):
        wheel = self._wheel(
            tmp_path,
            {
                "witwin_radar-0.3.0.dist-info/METADATA": b"Name: witwin-radar\n",
                "witwin_other-0.1.0.dist-info/METADATA": b"Name: witwin-other\n",
            },
        )
        with pytest.raises(smoke.CoexistenceError, match="dist-info METADATA"):
            smoke._wheel_metadata(wheel)


class TestTheScenarioHarness:
    def _run(self, tmp_path: Path, body: str) -> dict[str, object]:
        scratch = tmp_path / "scenarios"
        temp_root = tmp_path / "temp"
        for directory in (scratch, temp_root, tmp_path / "target"):
            directory.mkdir(parents=True, exist_ok=True)
        return smoke._run_scenario(
            name="X",
            code=body,
            target=tmp_path / "target",
            scratch=scratch,
            temp_root=temp_root,
        )

    def test_a_scenario_that_raises_fails_the_smoke(self, tmp_path: Path):
        with pytest.raises(smoke.CoexistenceError, match="scenario X failed"):
            self._run(tmp_path, '\nraise SystemExit("the property did not hold")\n')

    def test_a_scenario_that_prints_nothing_fails_the_smoke(self, tmp_path: Path):
        with pytest.raises(smoke.CoexistenceError, match="printed 0 lines"):
            self._run(tmp_path, "\npass\n")

    def test_a_scenario_that_prints_twice_fails_the_smoke(self, tmp_path: Path):
        with pytest.raises(smoke.CoexistenceError, match="printed 2 lines"):
            self._run(tmp_path, "\nemit(a=1)\nemit(b=2)\n")

    def test_a_scenario_that_prints_non_json_fails_the_smoke(self, tmp_path: Path):
        with pytest.raises(smoke.CoexistenceError, match="did not emit JSON"):
            self._run(tmp_path, '\nprint("looks fine to me")\n')

    def test_a_passing_scenario_returns_its_emitted_record(self, tmp_path: Path):
        assert self._run(tmp_path, "\nemit(measured=7)\n") == {"measured": 7}

    def test_the_subprocess_environment_carries_no_loader_override(self):
        env = smoke._scenario_env(Path("E:/nowhere"))
        leaked = sorted(
            name
            for name in env
            if name.startswith(smoke._SCRUBBED_ENV_PREFIXES)
        )
        assert leaked == []
        assert env["PYTHONNOUSERSITE"] == "1"
        assert env["TMP"] == env["TEMP"] == str(Path("E:/nowhere"))


class TestTheGeneratedScenarioScripts:
    @pytest.mark.parametrize(
        "name",
        ["A", "B", "C", "D", "E", "F", "G", "I"],
    )
    def test_every_scenario_script_is_valid_python(self, name: str, tmp_path: Path):
        bodies = {
            "A": smoke._scenario_a,
            "B": smoke._scenario_b,
            "C": smoke._scenario_c,
            "D": smoke._scenario_d,
            "E": smoke._scenario_e,
            "F": smoke._scenario_f,
            "G": lambda: smoke._scenario_g(tmp_path),
            "I": lambda: smoke._scenario_i({"source_fingerprint": "ab" * 32}),
        }
        source = smoke._preamble(tmp_path) + bodies[name]()
        ast.parse(source, filename=f"scenario_{name}.py")
        source.encode("ascii")

    def test_the_preamble_strips_every_editable_finder_this_environment_has(self):
        source = smoke._preamble(Path("E:/target"))
        for prefix in ("_editable_impl_witwin", "__editable__", "_witwin_channel"):
            assert prefix in source

    def test_the_scenarios_named_in_the_docstring_are_the_scenarios_run(self):
        # The nine-scenario claim in the module docstring and the set the runner
        # actually executes must not drift apart.
        source = (REPOSITORY_ROOT / "ci" / "coexistence_smoke.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        generators = {
            node.name.rsplit("_", 1)[-1].upper()
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name.startswith("_scenario_")
        }
        assert generators == {"A", "B", "C", "D", "E", "F", "G", "H", "I", "ENV"}


class TestTheEvidenceContract:
    def test_the_recorded_evidence_version_is_the_one_in_the_filename(self):
        # ``artifacts/phase10/coexistence.v1.json`` and EVIDENCE_VERSION are two
        # statements of one number; a reader who trusts the filename has to be
        # right.
        assert smoke.EVIDENCE_VERSION == 1

    @pytest.mark.skipif(
        not (REPOSITORY_ROOT / "artifacts" / "phase10" / "coexistence.v1.json").is_file(),
        reason="no coexistence evidence in this checkout; the smoke is a nightly gate",
    )
    def test_a_present_evidence_file_reports_all_nine_scenarios(self):
        evidence = json.loads(
            (REPOSITORY_ROOT / "artifacts" / "phase10" / "coexistence.v1.json").read_text(
                encoding="utf-8"
            )
        )
        assert evidence["evidence_version"] == smoke.EVIDENCE_VERSION
        assert sorted(evidence["scenarios"]) == list("ABCDEFGHI")
        assert set(evidence["wheels"]) == {"core", "channel", "radar"}


def test_the_smoke_module_imports_no_witwin_package():
    # It has to run before anything is installed, and inside an environment
    # whose ambient witwin packages are exactly what it is trying to exclude.
    assert "witwin" not in sys.modules or smoke.__file__ is not None
    tree = ast.parse((REPOSITORY_ROOT / "ci" / "coexistence_smoke.py").read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])
    assert "witwin" not in imported
    assert "torch" not in imported
