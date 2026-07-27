"""Check the radar workflows against the cross-repository prebuild policy.

`GITHUB_ACTIONS_PREBUILD_MATRIX.md` (platform root, policy version 6) is the
authority. It lives one repository up, so this file is its EXECUTABLE
RESTATEMENT for Radar: every constant below is copied from the policy with the
section it comes from, and `POLICY_VERSION` exists so a policy revision that
nobody mirrored here is visible rather than silent.

This gate is the locally runnable proxy for a remote release run. A workflow
change cannot be validated by dispatching it - that costs a paid full-matrix
CUDA build and a wait - so the invariants that a run would have discovered are
asserted against the checked-in YAML instead:

* the trigger boundary (native wheel work is opt-in, never per-push);
* the complete release SASS set including sm_87, plus compute_120 PTX;
* the reduced pull-request smoke profile, and that it uploads nothing;
* a real manylinux_2_28 image for the Linux wheel;
* an architecture verifier run against the artifact on both platforms;
* exactly one native member in the wheel;
* the Stable ABI compatibility grid, cell for cell;
* GitHub-hosted runners, with one frozen exception that carries its reason.

It reads the YAML, not just the text, so a check cannot be satisfied by a
comment that mentions the right string.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"
PUBLISH_WORKFLOW = WORKFLOW_DIR / "publish-witwin-radar.yml"

#: Bump only together with a review of this file against the policy document.
POLICY_VERSION = 6

# Policy "Required CUDA coverage": the canonical release value and the reduced
# opt-in pull-request set.
FULL_GENCODE_ARCHES = "7.0;7.5;8.0;8.6;8.7;8.9;9.0;10.0;10.1;12.0+PTX"
FULL_EXPECTED_SASS = ("70", "75", "80", "86", "87", "89", "90", "100", "101", "120")
FULL_EXPECTED_PTX = "120"
SMOKE_GENCODE_ARCHES = "8.7;12.0+PTX"
SMOKE_EXPECTED_SASS = ("87", "120")

# Policy "Python and Torch matrix": the eight Stable ABI cells, in order.
STABLE_ABI_CELLS = (
    ("3.10", "2.10.0", "cu128"),
    ("3.11", "2.10.0", "cu128"),
    ("3.12", "2.10.0", "cu128"),
    ("3.13", "2.10.0", "cu128"),
    ("3.14", "2.10.0", "cu128"),
    ("3.14", "2.11.0", "cu128"),
    ("3.14", "2.12.0", "cu126"),
    ("3.14", "2.13.0", "cu126"),
)

# Policy "Required platform coverage".
REQUIRED_OS = ("ubuntu-22.04", "windows-2022")
MANYLINUX_IMAGE = "manylinux_2_28"

# Policy "CI execution policy": native wheel work is opt-in. `push` and
# `schedule` are the two triggers that turn every commit into a paid CUDA build.
ALLOWED_TRIGGERS = frozenset({"release", "workflow_dispatch", "pull_request"})
FORBIDDEN_TRIGGERS = frozenset({"push", "schedule", "create", "issue_comment"})
OPT_IN_LABEL = "run-ci"

# Policy "CI execution policy": "GitHub-hosted runners are mandatory.
# `self-hosted` labels are forbidden." One workflow needs a physical GPU that no
# hosted runner provides. It is frozen here BY NAME with its reason so it cannot
# spread quietly to a wheel-producing workflow, and it is recorded as deviation
# P4 in docs/dev/plans/phase10-deferred-release-matrix.md.
SELF_HOSTED_ALLOWLIST = {
    "gpu-regression.yml": (
        "manually dispatched GPU regression; no GitHub-hosted runner offers a "
        "CUDA device. Produces no wheel and publishes nothing. Deviation P4."
    ),
}


class PolicyFailure(list):
    """A list of failure strings, so every violation is reported at once."""

    def add(self, message: str) -> None:
        self.append(message)


def load_workflow(path: Path) -> dict:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{path} is not a workflow mapping")
    return document


def workflow_triggers(document: dict) -> dict:
    """Return the ``on:`` block.

    YAML 1.1 reads a bare ``on`` key as the boolean ``True``, which is why a
    naive ``document["on"]`` silently finds nothing and a checker written that
    way passes on any workflow at all.
    """

    for key in ("on", True):
        if key in document:
            value = document[key]
            if isinstance(value, dict):
                return value
            if isinstance(value, list):
                return dict.fromkeys(value)
            return {str(value): None}
    return {}


def _steps(document: dict) -> list[tuple[str, str, dict]]:
    collected = []
    for job_id, job in (document.get("jobs") or {}).items():
        for step in job.get("steps") or []:
            collected.append((job_id, step.get("name", "<unnamed>"), step))
    return collected


def _string_leaves(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [leaf for item in value.values() for leaf in _string_leaves(item)]
    if isinstance(value, list):
        return [leaf for item in value for leaf in _string_leaves(item)]
    return []


def _step_text(document: dict) -> str:
    """Every string in every step, RAW.

    Not ``json.dumps`` of the step: that escapes the quotes inside an embedded
    Python program, so a search for ``origin"] == "packaged"`` never matches the
    script that contains exactly that.
    """

    parts: list[str] = []
    for _, _, step in _steps(document):
        parts.extend(_string_leaves(step))
    return "\n".join(parts)


def check_triggers(document: dict, failures: PolicyFailure) -> None:
    triggers = workflow_triggers(document)
    if not triggers:
        failures.add("the workflow declares no triggers at all")
        return
    forbidden = sorted(set(triggers) & FORBIDDEN_TRIGGERS)
    if forbidden:
        failures.add(
            f"forbidden trigger(s) {forbidden}: the policy makes native wheel "
            "work opt-in through a published release, an explicit "
            "workflow_dispatch, or the run-ci label"
        )
    unknown = sorted(set(triggers) - ALLOWED_TRIGGERS - FORBIDDEN_TRIGGERS)
    if unknown:
        failures.add(f"unrecognised trigger(s) {unknown}")
    if "release" not in triggers:
        failures.add("no `release` trigger: a published release must build wheels")
    else:
        types = (triggers.get("release") or {}).get("types")
        if types != ["published"]:
            failures.add(f"release trigger types must be ['published'], found {types!r}")
    if "workflow_dispatch" not in triggers:
        failures.add("no `workflow_dispatch` trigger: a manual full build must exist")
    else:
        inputs = (triggers.get("workflow_dispatch") or {}).get("inputs") or {}
        if "scope" not in inputs:
            failures.add("workflow_dispatch must take a `scope` input")
        else:
            options = inputs["scope"].get("options") or []
            if "full" not in options:
                failures.add(f"workflow_dispatch scope must offer 'full', found {options!r}")
    if "pull_request" in triggers:
        types = (triggers.get("pull_request") or {}).get("types")
        if types != ["labeled"]:
            failures.add(
                "the pull_request trigger must be types: [labeled] so an "
                f"ordinary push to a PR starts nothing; found {types!r}"
            )
        entry_jobs = [
            job_id
            for job_id, job in (document.get("jobs") or {}).items()
            if not job.get("needs")
        ]
        for job_id in entry_jobs:
            condition = str((document["jobs"][job_id]).get("if", ""))
            if OPT_IN_LABEL not in condition:
                failures.add(
                    f"job {job_id!r} has no `needs` and no {OPT_IN_LABEL!r} guard, "
                    "so a label event would start it unconditionally"
                )


def _dotted(arch: str) -> str:
    """``"87" -> "8.7"``, ``"120" -> "12.0"`` - the gencode spelling."""

    return f"{arch[:-1]}.{arch[-1]}"


#: The workflow declares its architecture profiles once, as top-level env, and
#: the jobs reference them. Checking the DECLARATION against the policy - rather
#: than searching the whole file for a substring - is what makes a dropped
#: architecture report itself as "sm_87 is gone" instead of as a diff of two
#: forty-character strings, and stops a mention in a comment from satisfying it.
POLICY_ENV = {
    "FULL_GENCODE_ARCHES": FULL_GENCODE_ARCHES,
    "FULL_EXPECTED_SASS": ",".join(FULL_EXPECTED_SASS),
    "FULL_EXPECTED_PTX": FULL_EXPECTED_PTX,
    "SMOKE_GENCODE_ARCHES": SMOKE_GENCODE_ARCHES,
    "SMOKE_EXPECTED_SASS": ",".join(SMOKE_EXPECTED_SASS),
}


def check_architectures(document: dict, failures: PolicyFailure) -> None:
    env = document.get("env") or {}
    text = _step_text(document)
    for key, expected in POLICY_ENV.items():
        found = env.get(key)
        if found is None:
            failures.add(f"the workflow declares no `{key}` architecture profile")
            continue
        if str(found) != expected:
            detail = ""
            if key.endswith("GENCODE_ARCHES"):
                reference = (
                    FULL_EXPECTED_SASS
                    if key.startswith("FULL")
                    else SMOKE_EXPECTED_SASS
                )
                missing = [
                    f"sm_{arch}"
                    for arch in reference
                    if _dotted(arch) not in str(found)
                ]
                if missing:
                    detail = f"; missing {missing}"
            failures.add(
                f"`{key}` is {found!r}, the policy value is {expected!r}{detail}"
            )
        if key not in text and f"env.{key}" not in text:
            failures.add(
                f"`{key}` is declared but never used, so the profile it names "
                "does not reach any build step"
            )
    if f"{_dotted(FULL_EXPECTED_PTX)}+PTX" not in str(env.get("FULL_GENCODE_ARCHES", "")):
        failures.add(
            f"the release profile requests no compute_{FULL_EXPECTED_PTX} PTX "
            "target; a release wheel without PTX cannot run on a future "
            "architecture"
        )


def check_verifier_and_manylinux(document: dict, failures: PolicyFailure) -> None:
    windows_verified = False
    linux_verified = False
    manylinux = False
    for _, _, step in _steps(document):
        blob = "\n".join(_string_leaves(step))
        if "verify_cuda_binary_arches.py" in blob:
            condition = str(step.get("if", ""))
            if "Linux" in condition or "CIBW" in blob:
                linux_verified = True
            if "Windows" in condition or not condition:
                windows_verified = True
        # The image key specifically, not the string anywhere: `manylinux_2_28`
        # also appears in the auditwheel plat tag and in the publish-time tag
        # assertion, so a substring search would keep passing after the build
        # image itself was swapped for an Ubuntu one.
        image = str(((step.get("env") or {}).get("CIBW_MANYLINUX_X86_64_IMAGE", "")))
        if image.startswith(MANYLINUX_IMAGE):
            manylinux = True
    if not windows_verified:
        failures.add("no architecture verifier runs against the Windows artifact")
    if not linux_verified:
        failures.add("no architecture verifier runs against the Linux artifact")
    if not manylinux:
        failures.add(
            f"the Linux wheel is not built in a {MANYLINUX_IMAGE} image; "
            "relabeling an Ubuntu binary is not compliant"
        )


def check_wheel_shape(document: dict, failures: PolicyFailure) -> None:
    text = _step_text(document)
    if "len(native) == 1" not in text:
        failures.add(
            "no step asserts the wheel carries exactly one native member; a "
            "second binary or a native-free wheel would publish unnoticed"
        )
    if 'origin"] == "packaged"' not in text and "origin'] == 'packaged'" not in text:
        failures.add(
            "no step asserts the installed extension loaded from the packaged "
            "prebuilt (build_info()['origin'] == 'packaged')"
        )
    if "torch.utils.cpp_extension" not in text:
        failures.add(
            "no step asserts torch.utils.cpp_extension stayed unimported, so a "
            "silent JIT would not be detected"
        )
    for sidecar in (".build-info.json", ".build-fingerprint"):
        if sidecar not in text:
            failures.add(f"no step asserts the installed package ships {sidecar}")


def check_stable_abi_matrix(document: dict, failures: PolicyFailure) -> None:
    for job_id, job in (document.get("jobs") or {}).items():
        matrix = ((job.get("strategy") or {}).get("matrix") or {})
        cells = matrix.get("compatibility")
        if not cells:
            continue
        found = tuple(
            (
                str(cell.get("python_version")),
                str(cell.get("torch_version")),
                str(cell.get("cuda_index")),
            )
            for cell in cells
        )
        if found != STABLE_ABI_CELLS:
            failures.add(
                f"job {job_id!r} does not carry the policy's eight Stable ABI "
                f"cells in order: found {found!r}"
            )
        operating_systems = tuple(matrix.get("os") or ())
        if tuple(sorted(operating_systems)) != tuple(sorted(REQUIRED_OS)):
            failures.add(
                f"job {job_id!r} must run the compatibility grid on {REQUIRED_OS}, "
                f"found {operating_systems!r}"
            )
        return
    failures.add("no Stable ABI compatibility job with a `compatibility` matrix")


def check_publish_gating(document: dict, failures: PolicyFailure) -> None:
    jobs = document.get("jobs") or {}
    publish = jobs.get("publish")
    if publish is None:
        failures.add("no `publish` job")
        return
    condition = str(publish.get("if", ""))
    if "github.event_name == 'release'" not in condition:
        failures.add(
            "the publish job is not restricted to `release: published`; a "
            f"manual dispatch must validate without publishing (if: {condition!r})"
        )
    needs = set(publish.get("needs") or ())
    for required in ("build_cuda_wheels", "test_torch_compatibility"):
        if required not in needs:
            failures.add(f"the publish job does not depend on {required!r}")


#: Every policy row this repository cannot satisfy locally must appear in the
#: deferral register with an owner and an executing command. That document is
#: the other half of this gate: what is not asserted here has to be named there,
#: so "we never ran it" cannot quietly become "it passed".
DEFERRAL_REGISTER = REPO_ROOT / "docs" / "dev" / "plans" / "phase10-deferred-release-matrix.md"
DEFERRAL_COLUMNS = 6
REQUIRED_DEFERRALS = ("D1", "D2", "D3", "D4", "D5", "D6")


def check_deferral_register(path: Path, failures: PolicyFailure) -> None:
    if not path.is_file():
        failures.add(f"the deferral register {path.name} does not exist")
        return
    text = path.read_text(encoding="utf-8")
    rows = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if not cells or not cells[0].startswith("D") or not cells[0][1:].isdigit():
            continue
        rows[cells[0]] = cells
    for identifier in REQUIRED_DEFERRALS:
        cells = rows.get(identifier)
        if cells is None:
            failures.add(f"{path.name} has no row for deferral {identifier}")
            continue
        if len(cells) != DEFERRAL_COLUMNS:
            failures.add(
                f"{path.name} row {identifier} has {len(cells)} columns, the "
                f"register schema is {DEFERRAL_COLUMNS} "
                "(id, deferral, why, executed by, evidence, owner)"
            )
            continue
        empty = [
            index for index, cell in enumerate(cells) if not cell or cell in {"-", "TBD"}
        ]
        if empty:
            failures.add(
                f"{path.name} row {identifier} leaves column(s) {empty} unfilled; "
                "a deferral without an owner and an executing command is a gap"
            )
    if "Phase 11 proceeds without waiting" not in text:
        failures.add(
            f"{path.name} must state that these are deferrals rather than gaps "
            "and that Phase 11 does not wait on them"
        )


def check_runners(directory: Path, failures: PolicyFailure) -> None:
    for path in sorted(directory.glob("*.yml")):
        text = path.read_text(encoding="utf-8")
        if "self-hosted" not in text:
            continue
        reason = SELF_HOSTED_ALLOWLIST.get(path.name)
        if reason is None:
            failures.add(
                f"{path.name} uses a self-hosted runner and is not in the frozen "
                "allowlist; GitHub-hosted runners are mandatory"
            )


def check_workflow(
    path: Path,
    *,
    workflow_dir: Path | None = None,
    deferrals: Path | None = None,
) -> PolicyFailure:
    failures = PolicyFailure()
    document = load_workflow(path)
    check_triggers(document, failures)
    check_architectures(document, failures)
    check_verifier_and_manylinux(document, failures)
    check_wheel_shape(document, failures)
    check_stable_abi_matrix(document, failures)
    check_publish_gating(document, failures)
    if workflow_dir is not None:
        check_runners(workflow_dir, failures)
    if deferrals is not None:
        check_deferral_register(deferrals, failures)
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workflow", type=Path, default=PUBLISH_WORKFLOW)
    parser.add_argument(
        "--workflow-dir",
        type=Path,
        default=WORKFLOW_DIR,
        help="directory scanned for forbidden self-hosted runners",
    )
    parser.add_argument("--no-runner-scan", action="store_true")
    parser.add_argument("--deferrals", type=Path, default=DEFERRAL_REGISTER)
    parser.add_argument("--no-deferral-check", action="store_true")
    arguments = parser.parse_args(argv)

    try:
        failures = check_workflow(
            arguments.workflow,
            workflow_dir=None if arguments.no_runner_scan else arguments.workflow_dir,
            deferrals=None if arguments.no_deferral_check else arguments.deferrals,
        )
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(f"workflow policy check failed: {error}", file=sys.stderr)
        return 2

    if failures:
        print(f"workflow policy FAILED for {arguments.workflow.name}:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(
        f"workflow policy OK: {arguments.workflow.name} against prebuild policy "
        f"version {POLICY_VERSION}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
