"""The binding manifest is an ownership registry, and the gate that says so.

Schema 2 added seven per-operator columns and three top-level keys. Every one
of them is checked here, and - more importantly - the GATE is checked: a test
that only runs a passing gate proves the gate ran, not that it works. Each
enforcement test below feeds ``ci/check_native_bindings.py`` a mutated copy of
the manifest and asserts it exits non-zero for the right reason.

The mutation cases are deliberately the ones that describe a real defect:
a RayD-owned family registered as a Radar primitive (Phase-10 work item 4's
explicit prohibition), a companion whose primal disappeared, a translation unit
that is not a build input, and a symbol set that has drifted from the shipped
binary.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "ci" / "native-binding-manifest.json"
GATE = REPO_ROOT / "ci" / "check_native_bindings.py"

sys.path.insert(0, str(REPO_ROOT / "ci"))

import check_native_bindings as gate  # noqa: E402


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _run_gate(manifest_path: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(GATE), "--manifest", str(manifest_path)],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        check=False,
    )


def _mutated(tmp_path: Path, mutate) -> Path:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    mutate(data)
    path = tmp_path / "mutated-manifest.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# The registry itself
# --------------------------------------------------------------------------


def test_the_manifest_declares_schema_two_and_the_live_abi_version(manifest):
    from witwin.radar.cuda.runtime import RADAR_ABI_VERSION

    assert manifest["schema_version"] == 2
    assert manifest["radar_abi_version"] == RADAR_ABI_VERSION


def test_every_operator_carries_every_column(manifest):
    known = set(gate.REQUIRED_OPERATOR_COLUMNS) | set(gate.OPTIONAL_OPERATOR_COLUMNS)
    for entry in manifest["operators"]:
        missing = [name for name in gate.REQUIRED_OPERATOR_COLUMNS if name not in entry]
        assert not missing, (entry.get("symbol"), missing)
        assert not set(entry) - known, (entry["symbol"], sorted(set(entry) - known))


def test_symbols_are_unique(manifest):
    symbols = [entry["symbol"] for entry in manifest["operators"]]
    assert len(symbols) == len(set(symbols))
    assert len(symbols) == 28


def test_every_native_tu_is_a_build_input(manifest):
    sources = set(manifest["sources"])
    for entry in manifest["operators"]:
        assert entry["native_tu"] in sources, entry["symbol"]
    # And every kernel TU owns at least one symbol: a build input that owns
    # nothing is either dead code or an unregistered family.
    owning = {entry["native_tu"] for entry in manifest["operators"]}
    kernels = {name for name in sources if name.endswith(".cu")}
    assert owning == kernels, sorted(kernels ^ owning)


def test_radar_owns_the_numerics_of_every_registered_symbol(manifest):
    """Item 4's prohibition, as a value rather than an unwritten rule."""

    owners = {entry["numerical_owner"] for entry in manifest["operators"]}
    assert owners == {"radar"}, sorted(owners)


def test_every_ad_group_has_exactly_one_primal(manifest):
    groups: dict[str, list[str]] = {}
    for entry in manifest["operators"]:
        groups.setdefault(entry["ad_group"], []).append(entry["ad_role"])
    for group, roles in sorted(groups.items()):
        assert roles.count("primal") == 1, (group, roles)
        assert all(role in gate.AD_ROLES for role in roles), (group, roles)


def test_launch_counts_and_host_observations_are_recorded(manifest):
    """R-ADR-006's 'Radar adds zero host observations', made machine-readable.

    The launch counts are transcribed from the kernel sources; the assertion
    here is that every row HAS one and that it is positive, because an operator
    that launches nothing is not an operator.
    """

    for entry in manifest["operators"]:
        assert entry["launches"] >= 1, entry["symbol"]
        assert entry["host_observations"] == 0, entry["symbol"]
        assert entry["fused_stages"], entry["symbol"]


def test_the_error_owners_registry_names_modules_that_exist(manifest):
    domains = [entry["domain"] for entry in manifest["error_owners"]]
    assert len(domains) == len(set(domains))
    assert "native_load" in domains
    for entry in manifest["error_owners"]:
        assert set(entry) == gate.ERROR_OWNER_KEYS
        assert (REPO_ROOT / entry["owner_module"]).is_file(), entry["domain"]


def test_the_manifest_symbol_set_matches_the_packaged_sidecar(manifest):
    """Registry, shipped binary and loader agree, or this fails (A4)."""

    from witwin.radar.cuda import runtime as build

    identity = build

    binary = build.prebuilt_extension_path()
    if not binary.is_file():
        pytest.skip("no packaged prebuilt in this checkout")
    record = identity.read_build_info(binary)
    assert set(record["operator_symbols"]) == {entry["symbol"] for entry in manifest["operators"]}


def test_the_gate_passes_on_the_real_manifest():
    completed = _run_gate(MANIFEST)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "OK" in completed.stdout


# --------------------------------------------------------------------------
# The gate fires
# --------------------------------------------------------------------------


def test_a_rayd_owned_family_registered_here_fails_the_gate(tmp_path):
    """The one mutation Phase-10 work item 4 explicitly names."""

    def mutate(data):
        data["operators"][0]["numerical_owner"] = "rayd"

    completed = _run_gate(_mutated(tmp_path, mutate))
    assert completed.returncode == 1, completed.stdout
    assert "numerical_owner" in completed.stdout
    assert "shared primitive" in completed.stdout


def test_a_companion_without_its_primal_fails_the_gate(tmp_path):
    def mutate(data):
        for entry in data["operators"]:
            if entry["symbol"] == "fmcw_beat_forward":
                entry["ad_group"] = "fmcw_beat_renamed"

    completed = _run_gate(_mutated(tmp_path, mutate))
    assert completed.returncode == 1, completed.stdout
    assert "primal" in completed.stdout


def test_a_native_tu_outside_the_build_input_set_fails_the_gate(tmp_path):
    def mutate(data):
        data["operators"][0]["native_tu"] = "witwin/radar/cuda/ghost.cu"

    completed = _run_gate(_mutated(tmp_path, mutate))
    assert completed.returncode == 1, completed.stdout
    assert "native_tu" in completed.stdout


def test_a_missing_column_fails_the_gate(tmp_path):
    def mutate(data):
        del data["operators"][0]["fused_stages"]

    completed = _run_gate(_mutated(tmp_path, mutate))
    assert completed.returncode == 1, completed.stdout
    assert "missing columns" in completed.stdout


def test_a_drifted_symbol_set_fails_the_gate(tmp_path):
    """The registry cannot quietly describe a binary it does not match."""

    from witwin.radar.cuda import runtime as build

    if not build.prebuilt_extension_path().is_file():
        pytest.skip("no packaged prebuilt in this checkout")

    def mutate(data):
        entry = dict(data["operators"][0])
        entry["symbol"] = "invented_operator"
        entry["python_owner"] = "witwin/radar/cuda/runtime.py"
        data["operators"].append(entry)

    completed = _run_gate(_mutated(tmp_path, mutate))
    assert completed.returncode == 1, completed.stdout
    assert "different operator set" in completed.stdout


def test_a_stale_schema_version_fails_the_gate(tmp_path):
    def mutate(data):
        data["schema_version"] = 1

    completed = _run_gate(_mutated(tmp_path, mutate))
    assert completed.returncode == 1, completed.stdout
    assert "schema_version" in completed.stdout


def test_an_error_owner_naming_a_missing_module_fails_the_gate(tmp_path):
    def mutate(data):
        data["error_owners"][0]["owner_module"] = "witwin/radar/nowhere.py"

    completed = _run_gate(_mutated(tmp_path, mutate))
    assert completed.returncode == 1, completed.stdout
    assert "does not exist" in completed.stdout


def test_a_contract_test_that_names_neither_the_symbol_nor_its_owner_fails(tmp_path):
    """File existence is not coverage.

    Re-pointing a row at an unrelated but existing test file used to pass. The
    named file must mention the symbol or the Python owner module, or the row
    must say in writing which facade stands in for it.
    """

    def mutate(data):
        entry = next(item for item in data["operators"] if item["symbol"] == "fmcw_beat_forward")
        entry["contract_test"] = "tests/test_phase10_diagnostics.py"
        entry.pop("contract_test_note", None)

    completed = _run_gate(_mutated(tmp_path, mutate))
    assert completed.returncode == 1, completed.stdout
    assert "names neither the symbol nor its python_owner module" in completed.stdout


def test_a_facade_row_may_declare_the_gap_but_not_fake_it(tmp_path):
    """The escape hatch is a written note, and the note itself is typed.

    The mutation adds the note rather than emptying an existing one. All three
    rows that carried a ``contract_test_note`` were ``dirichlet_spectrum``
    operators whose contract test drove them through a solver facade, and
    Phase 11 deleted the family; the hatch itself is kept because a future
    facade-driven operator would need it, and an unexercised gate is not a
    gate.
    """

    def mutate(data):
        entry = next(item for item in data["operators"] if item["symbol"] == "fmcw_beat_forward")
        entry["contract_test_note"] = []

    completed = _run_gate(_mutated(tmp_path, mutate))
    assert completed.returncode == 1, completed.stdout
    assert "contract_test_note must be a non-empty string list" in completed.stdout


def test_every_facade_row_that_uses_the_note_says_why(manifest):
    """Only the rows that need the hatch may hold it, and each one explains.

    The list is empty since Phase 11. The loop is kept rather than replaced by
    a bare emptiness assertion because the emptiness is not the claim - the
    claim is that a row holding the note is a row whose contract test really
    cannot name the symbol, and that has to be checked for whichever rows exist.
    """

    noted = []
    for entry in manifest["operators"]:
        note = entry.get("contract_test_note")
        if note is None:
            continue
        text = (REPO_ROOT / entry["contract_test"]).read_text(encoding="utf-8")
        owner_module = Path(entry["python_owner"]).stem
        assert not gate._references(text, symbol=entry["symbol"], owner_module=owner_module), (
            f"{entry['symbol']}: the note is unnecessary, the test names it"
        )
        assert len(" ".join(note)) > 80, entry["symbol"]
        noted.append(entry["symbol"])
    assert noted == [], noted
