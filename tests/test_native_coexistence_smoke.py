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
        assert smoke._forbidden(["numpy", "rayd", "drjit", "mitsuba", "sionna", "torch"]) == [
            "drjit",
            "mitsuba",
            "rayd",
            "sionna",
        ]

    def test_it_does_not_fire_on_a_name_that_merely_starts_the_same(self):
        # ``raydium`` is not ``rayd``. A prefix match without the separator
        # would make this gate reject unrelated packages and get relaxed.
        assert smoke._forbidden(["raydium", "drjitter"]) == []


class TestTheRequirementParser:
    def test_it_reads_extras_as_well_as_base_requirements(self):
        metadata = (
            "Metadata-Version: 2.4\n"
            "Name: witwin-radar\n"
            "Requires-Dist: torch>=2.10\n"
            "Requires-Dist: witwin-channel<0.7,>=0.6.0; python_version < \"3.14\" and extra == 'channel'\n"
            "Requires-Dist: pytest; extra == 'dev'\n"
        )
        assert smoke._requirement_names(metadata) == ["torch", "witwin-channel", "pytest"]


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

    def test_a_native_member_without_its_build_record_is_refused(self, tmp_path: Path):
        wheel = self._wheel(tmp_path, {"witwin/radar/cuda/prebuilt/_radar_native.pyd": b"MZ"})
        with pytest.raises(smoke.CoexistenceError, match="build-info.json"):
            smoke._wheel_native_record(wheel)


class TestTheScenarioHarness:
    def _run(self, tmp_path: Path, body: str) -> dict[str, object]:
        scratch = tmp_path / "scenarios"
        temp_root = tmp_path / "temp"
        for directory in (scratch, temp_root, tmp_path / "target"):
            directory.mkdir(parents=True, exist_ok=True)
        return smoke._run_scenario(
            name="X", code=body, target=tmp_path / "target", scratch=scratch, temp_root=temp_root
        )

    def test_a_scenario_that_raises_fails_the_smoke(self, tmp_path: Path):
        with pytest.raises(smoke.CoexistenceError, match="scenario X failed"):
            self._run(tmp_path, '\nraise SystemExit("the property did not hold")\n')

    def test_a_scenario_that_prints_nothing_fails_the_smoke(self, tmp_path: Path):
        with pytest.raises(smoke.CoexistenceError, match="printed 0 lines"):
            self._run(tmp_path, "\npass\n")

    def test_the_subprocess_environment_carries_no_loader_override(self):
        env = smoke._scenario_env(Path("E:/nowhere"))
        leaked = sorted(name for name in env if name.startswith(smoke._SCRUBBED_ENV_PREFIXES))
        assert leaked == []
        assert env["PYTHONNOUSERSITE"] == "1"
        assert env["TMP"] == env["TEMP"] == str(Path("E:/nowhere"))


class TestTheGeneratedScenarioScripts:
    @pytest.mark.parametrize("name", ["A", "B", "C", "D", "E", "F", "G", "I"])
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


def test_the_audited_source_members_are_the_translation_units_the_runtime_compiles():
    """The coexistence audit must reproduce the native build fingerprint, so it names every translation unit."""

    spec = importlib.util.spec_from_file_location(
        "coexistence_runtime_sources", REPOSITORY_ROOT / "witwin" / "radar" / "cuda" / "runtime.py"
    )
    assert spec is not None and spec.loader is not None
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    expected = tuple(f"witwin/radar/cuda/{path.name}" for path in runtime.extension_sources())
    assert smoke._RADAR_SOURCE_MEMBERS == expected
