"""Gate the radar native binding manifest as an OWNERSHIP REGISTRY.

Schema 1 answered "is every symbol covered?". Schema 2 answers "who owns
this symbol, in which translation unit, in which AD family, at what launch
cost, and what does it read back to the host?" - the questions Phase-10 work
items 3 and 4 ask. A registry that nothing checks is a document, so every
column has an assertion here.

The two assertions worth naming, because they encode a decision rather than a
formatting rule:

* ``numerical_owner`` must be ``radar`` for every row. Item 4 forbids
  registering a RayD-owned or Channel-owned family as a Radar shared
  primitive. Channel and RayD numerics reach Radar as compact typed CUDA
  tensors through the consumer contract, never as linked code (R-ADR-004), so
  a ``rayd`` value in THIS manifest is a boundary violation, not a label.
* when a packaged prebuilt is present, the manifest symbol set must equal the
  ``operator_symbols`` recorded in its build sidecar. That ties three things
  that can otherwise drift independently - the registry, the shipped binary,
  and the loader's own required-symbol check - and it is the machine-checkable
  form of acceptance criterion A4.

Run it directly, or through ``tests/test_phase10_binding_registry.py``, which
also proves the gate FIRES by feeding it a mutated copy.
"""

from __future__ import annotations

import argparse
import collections
import importlib
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "ci" / "native-binding-manifest.json"

# Run as a script, ``sys.path[0]`` is ``ci/``, so ``import witwin.radar`` would
# resolve to whatever is installed in the environment - which on a machine with
# an editable install is a DIFFERENT checkout. A gate must check the tree it
# lives in, so this checkout goes first.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCHEMA_VERSION = 2

TOP_LEVEL_KEYS = frozenset(
    {
        "comment",
        "schema_version",
        "radar_abi_version",
        "library",
        "logical_owner",
        "sources",
        "error_owners",
        "operators",
    }
)

REQUIRED_OPERATOR_COLUMNS = (
    "symbol",
    "family",
    "native_tu",
    "numerical_owner",
    "ad_role",
    "ad_group",
    "launches",
    "fused_stages",
    "host_observations",
    "python_owner",
    "contract_test",
    "end_to_end_caller",
)

OPTIONAL_OPERATOR_COLUMNS = ("caller_status", "caller_note", "contract_test_note")

AD_ROLES = frozenset({"primal", "backward", "jvp", "utility"})

#: The only legal value in the RADAR manifest. See the module docstring.
NUMERICAL_OWNERS = frozenset({"radar"})

ERROR_OWNER_KEYS = frozenset({"domain", "owner_module", "failure_mode"})


class ManifestError(Exception):
    """A registry claim failed."""


def load_manifest(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError(f"{path} is not readable JSON: {error}") from error
    if not isinstance(data, dict):
        raise ManifestError(f"{path} must contain a JSON object")
    return data


def _check_shape(manifest: dict, failures: list[str]) -> None:
    unknown = sorted(set(manifest) - TOP_LEVEL_KEYS)
    if unknown:
        failures.append(f"unknown top-level keys: {unknown}")
    missing = sorted(TOP_LEVEL_KEYS - set(manifest))
    if missing:
        failures.append(f"missing top-level keys: {missing}")
        return
    if manifest["schema_version"] != SCHEMA_VERSION:
        failures.append(
            f"schema_version is {manifest['schema_version']}, expected "
            f"{SCHEMA_VERSION}"
        )
    try:
        from witwin.radar.cuda.runtime import RADAR_ABI_VERSION
    except ImportError as error:  # pragma: no cover - environment problem
        failures.append(f"cannot import the radar ABI constant: {error}")
    else:
        if manifest["radar_abi_version"] != RADAR_ABI_VERSION:
            failures.append(
                f"radar_abi_version is {manifest['radar_abi_version']}, but "
                f"identity.RADAR_ABI_VERSION is {RADAR_ABI_VERSION}"
            )


def _check_sources(manifest: dict, failures: list[str]) -> None:
    """The manifest's source list is the build's source list, exactly.

    Also asserted from pytest. It lives here as well because a release run
    executes the gates and may not execute the suite.
    """

    try:
        from witwin.radar.cuda import runtime as build
    except ImportError as error:  # pragma: no cover - environment problem
        failures.append(f"cannot import the radar build module: {error}")
        return
    if REPO_ROOT not in Path(build.__file__).resolve().parents:
        failures.append(
            f"witwin.radar resolved to {build.__file__}, outside {REPO_ROOT}; "
            "this gate must inspect its own checkout"
        )
        return
    actual = {
        str(path.relative_to(REPO_ROOT)).replace("\\", "/")
        for path in build.extension_sources()
    }
    declared = set(manifest["sources"])
    if actual != declared:
        failures.append(
            f"sources disagree with build.extension_sources(): only in "
            f"manifest {sorted(declared - actual)}, only in build "
            f"{sorted(actual - declared)}"
        )


def _check_operators(manifest: dict, failures: list[str]) -> None:
    operators = manifest["operators"]
    sources = set(manifest["sources"])
    seen: set[str] = set()
    known_columns = set(REQUIRED_OPERATOR_COLUMNS) | set(OPTIONAL_OPERATOR_COLUMNS)

    for entry in operators:
        symbol = entry.get("symbol", "<unnamed>")
        missing = [name for name in REQUIRED_OPERATOR_COLUMNS if name not in entry]
        if missing:
            failures.append(f"{symbol}: missing columns {missing}")
            continue
        unknown = sorted(set(entry) - known_columns)
        if unknown:
            failures.append(f"{symbol}: unknown columns {unknown}")
        if symbol in seen:
            failures.append(f"{symbol}: duplicate symbol")
        seen.add(symbol)

        if entry["native_tu"] not in sources:
            failures.append(
                f"{symbol}: native_tu {entry['native_tu']!r} is not a build source"
            )
        if entry["numerical_owner"] not in NUMERICAL_OWNERS:
            failures.append(
                f"{symbol}: numerical_owner {entry['numerical_owner']!r} is not "
                "'radar'. A RayD-owned or Channel-owned family must not be "
                "registered as a Radar shared primitive; those numerics cross "
                "the boundary as compact typed tensors, never as linked code."
            )
        if entry["ad_role"] not in AD_ROLES:
            failures.append(f"{symbol}: ad_role {entry['ad_role']!r} is unknown")
        if not isinstance(entry["ad_group"], str) or not entry["ad_group"]:
            failures.append(f"{symbol}: ad_group must be a non-empty string")
        launches = entry["launches"]
        if type(launches) is not int or launches < 1:
            failures.append(f"{symbol}: launches must be a positive integer")
        stages = entry["fused_stages"]
        if not isinstance(stages, list) or not stages or not all(
            isinstance(stage, str) and stage for stage in stages
        ):
            failures.append(f"{symbol}: fused_stages must be a non-empty string list")
        observations = entry["host_observations"]
        if type(observations) is not int or observations < 0:
            failures.append(
                f"{symbol}: host_observations must be a non-negative integer"
            )

        owner = REPO_ROOT / entry["python_owner"]
        if not owner.is_file():
            failures.append(f"{symbol}: python_owner {entry['python_owner']} is missing")
        elif symbol not in owner.read_text(encoding="utf-8"):
            failures.append(
                f"{symbol}: python_owner {entry['python_owner']} does not name it"
            )
        contract = REPO_ROOT / entry["contract_test"]
        note = entry.get("contract_test_note")
        if note is not None and (
            not isinstance(note, list)
            or not note
            or not all(isinstance(line, str) and line for line in note)
        ):
            failures.append(
                f"{symbol}: contract_test_note must be a non-empty string list"
            )
            note = None
        if not contract.is_file():
            failures.append(
                f"{symbol}: contract_test {entry['contract_test']} is missing"
            )
        elif note is None and not _references(
            contract.read_text(encoding="utf-8"),
            symbol=symbol,
            owner_module=Path(entry["python_owner"]).stem,
        ):
            failures.append(
                f"{symbol}: contract_test {entry['contract_test']} names neither "
                f"the symbol nor its python_owner module "
                f"{Path(entry['python_owner']).stem!r}, and the row records no "
                "contract_test_note saying which facade exercises it. A test "
                "file that mentions neither is not evidence that it covers "
                "this operator"
            )
        caller = entry["end_to_end_caller"]
        if caller is None:
            if entry.get("caller_status") != "test_only" or not entry.get("caller_note"):
                failures.append(
                    f"{symbol}: a caller-free symbol must record caller_status "
                    "'test_only' and a caller_note"
                )
        elif not _resolves(caller):
            failures.append(f"{symbol}: end_to_end_caller {caller} does not resolve")

    _check_ad_groups(operators, failures)


def _check_ad_groups(operators: list[dict], failures: list[str]) -> None:
    """Exactly one primal per group, and no orphan companion.

    A backward or jvp whose group has no primal is either a stale registration
    or a primal that was renamed without its family, and both are the shape of
    defect this column exists to surface.
    """

    primal = collections.Counter(
        entry["ad_group"]
        for entry in operators
        if entry.get("ad_role") == "primal"
    )
    groups = {entry.get("ad_group") for entry in operators}
    for group in sorted(name for name in groups if isinstance(name, str)):
        count = primal.get(group, 0)
        if count != 1:
            failures.append(f"ad_group {group!r} has {count} primal rows, expected 1")
    for entry in operators:
        if entry.get("ad_role") in {"backward", "jvp"} and entry["ad_group"] not in primal:
            failures.append(
                f"{entry['symbol']}: ad_group {entry['ad_group']!r} has no primal"
            )


def _check_error_owners(manifest: dict, failures: list[str]) -> None:
    domains: set[str] = set()
    for entry in manifest["error_owners"]:
        if not isinstance(entry, dict) or set(entry) != ERROR_OWNER_KEYS:
            failures.append(f"error_owners entry has wrong keys: {entry}")
            continue
        if entry["domain"] in domains:
            failures.append(f"error_owners: duplicate domain {entry['domain']!r}")
        domains.add(entry["domain"])
        module = REPO_ROOT / entry["owner_module"]
        if not module.is_file():
            failures.append(
                f"error_owners: {entry['owner_module']} does not exist"
            )
        if not entry["failure_mode"].strip():
            failures.append(
                f"error_owners: {entry['domain']} has an empty failure_mode"
            )


def _check_sidecar_symbols(manifest: dict, failures: list[str]) -> str:
    """Tie the registry to the shipped binary, when there is one.

    Skipped - reported, never silently passed - when no packaged prebuilt is
    present, because a source checkout that has not built one yet is a normal
    state and this gate must still run in it.
    """

    try:
        from witwin.radar.cuda import runtime as build
        identity = build
    except ImportError as error:  # pragma: no cover - environment problem
        failures.append(f"cannot import the radar loader: {error}")
        return "error"
    binary = build.prebuilt_extension_path()
    if not binary.is_file():
        return "no packaged prebuilt; symbol-set tie not checked"
    try:
        record = identity.read_build_info(binary)
    except identity.RadarExtensionLoadError as error:
        failures.append(f"the packaged prebuilt has no usable build record: {error}")
        return "error"
    recorded = set(record["operator_symbols"])
    manifested = {entry["symbol"] for entry in manifest["operators"]}
    if recorded != manifested:
        failures.append(
            "the packaged prebuilt records a different operator set: only in "
            f"binary {sorted(recorded - manifested)}, only in manifest "
            f"{sorted(manifested - recorded)}"
        )
        return "error"
    return f"symbol-set tie checked against {binary.name} ({len(recorded)} symbols)"


def _references(text: str, *, symbol: str, owner_module: str) -> bool:
    """Does this test file mention the operator it is registered against?

    File existence alone is not coverage: a row re-pointed at any existing
    test passes that check. A contract test earns its column by naming the
    native symbol, or by naming the Python owner module whose facade it drives
    - the same substring rule the ``python_owner`` column is already held to.

    Some tests legitimately do neither - they exercise a solver facade several
    layers above the symbol. Those rows carry a ``contract_test_note`` instead,
    which is a written claim rather than an accident.
    """

    return symbol in text or owner_module in text


def _resolves(dotted: str) -> bool:
    parts = dotted.split(".")
    for split in range(len(parts) - 1, 2, -1):
        try:
            module = importlib.import_module(".".join(parts[:split]))
        except ImportError:
            continue
        target: object = module
        for attribute in parts[split:]:
            if not hasattr(target, attribute):
                return False
            target = getattr(target, attribute)
        return True
    return False


def check_manifest(path: Path = MANIFEST) -> list[str]:
    """Every failure, collected. An empty list is a pass."""

    failures: list[str] = []
    manifest = load_manifest(path)
    _check_shape(manifest, failures)
    if {"sources", "operators", "error_owners"} - set(manifest):
        return failures
    _check_sources(manifest, failures)
    _check_operators(manifest, failures)
    _check_error_owners(manifest, failures)
    _check_sidecar_symbols(manifest, failures)
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    arguments = parser.parse_args(argv)

    try:
        manifest = load_manifest(arguments.manifest)
        failures = check_manifest(arguments.manifest)
    except ManifestError as error:
        print(f"native binding manifest check failed: {error}")
        return 2

    if failures:
        print("native binding manifest FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    operators = manifest["operators"]
    print(
        f"native binding manifest OK: schema {manifest['schema_version']}, "
        f"{len(operators)} operators, "
        f"{len({entry['ad_group'] for entry in operators})} AD groups, "
        f"{len(manifest['error_owners'])} error owners"
    )
    print(f"  {_check_sidecar_symbols(manifest, [])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
