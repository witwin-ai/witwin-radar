"""Lint the GitHub workflows against the release policy and the repository.

One gate for every statement the repository makes about its workflows. It reads
the parsed YAML, not just text, so a comment that mentions the right string
satisfies nothing.

* **Script references.** Every ``python <path>.py`` a workflow runs exists.
* **Required Channel coverage.** Each workflow in
  ``ci/required-integration-tests.json`` installs the Channel extra, declares
  ``WITWIN_CHANNEL_FINGERPRINT`` and a zero ``WITWIN_REQUIRED_CHANNEL_SKIP_BUDGET``,
  and runs ``ci/check_channel_identity.py``, which reads both.
* **Release claims.** ``README.md`` and ``FEATURE_LIST.md`` name the manylinux
  policy of ``ci/release-policy.json`` and claim no cross-Torch Stable ABI.
* **The torch requirement.** ``pyproject.toml`` pins the exact Torch version the
  loader accepts for a packaged binary (the policy's ``torch_version``).
* **The publish workflow.** Native wheel work is opt-in (published release,
  manual dispatch, or a labelled pull request); the release profile carries the
  complete SASS set plus PTX and the pull-request profile the reduced set; the
  Linux wheel builds in a manylinux image; the architecture verifier runs on
  both platforms; the wheel carries exactly one native member and its sidecars
  and is smoke-tested in isolated mode; the exact-identity matrix is one cell
  per supported Python at the policy's Torch/CUDA identity on both operating
  systems; publishing happens only on a release, after the builds and the
  identity tests, with the artifact set validated by platform tag.
* **Runners.** GitHub-hosted runners only, except the one named workflow below.

Run directly, or as ``quick.workflows`` in ``ci/run_ci_tier.py``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import tomllib
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

FULL_GENCODE_ARCHES = "7.0;7.5;8.0;8.6;8.7;8.9;9.0;10.0;10.1;12.0+PTX"
FULL_EXPECTED_SASS = ("70", "75", "80", "86", "87", "89", "90", "100", "101", "120")
FULL_EXPECTED_PTX = "120"
SMOKE_GENCODE_ARCHES = "8.7;12.0+PTX"
SMOKE_EXPECTED_SASS = ("87", "120")
POLICY_ENV = {
    "FULL_GENCODE_ARCHES": FULL_GENCODE_ARCHES,
    "FULL_EXPECTED_SASS": ",".join(FULL_EXPECTED_SASS),
    "FULL_EXPECTED_PTX": FULL_EXPECTED_PTX,
    "SMOKE_GENCODE_ARCHES": SMOKE_GENCODE_ARCHES,
    "SMOKE_EXPECTED_SASS": ",".join(SMOKE_EXPECTED_SASS),
}

REQUIRED_OS = ("ubuntu-22.04", "windows-2022")
ALLOWED_TRIGGERS = frozenset({"release", "workflow_dispatch", "pull_request"})
FORBIDDEN_TRIGGERS = frozenset({"push", "schedule", "create", "issue_comment"})
OPT_IN_LABEL = "run-ci"

#: The one workflow that needs a physical GPU no hosted runner provides. It
#: builds no wheel and publishes nothing.
SELF_HOSTED_ALLOWLIST = {"gpu-regression.yml": "manually dispatched GPU regression; publishes nothing"}

CHANNEL_IDENTITY_SCRIPT = "ci/check_channel_identity.py"

RUN = re.compile(r"^(?P<spaces>\s*)(?:-\s+)?run:\s*(?P<body>.*)$")
SCRIPT = re.compile(r"(?:^|[\s;&|])(?:python|python3|py)\s+(?!-m\b)([A-Za-z0-9_./\\$\"{}-]+\.py)\b", re.MULTILINE)


def run_text(text: str) -> str:
    """The commands of every ``run:`` step, one per line, comment lines dropped."""

    lines = text.splitlines()
    commands: list[str] = []
    index = 0
    while index < len(lines):
        match = RUN.match(lines[index])
        index += 1
        if match is None:
            continue
        body = match.group("body").strip()
        if body not in {"|", "|-", "|+", ">", ">-", ">+"}:
            if body and not body.startswith("#"):
                commands.append(body)
            continue
        base_indent = len(match.group("spaces"))
        while index < len(lines):
            stripped = lines[index].lstrip()
            if stripped and len(lines[index]) - len(stripped) <= base_indent:
                break
            if stripped and not stripped.startswith("#"):
                commands.append(stripped)
            index += 1
    return "\n".join(commands)


def load_workflow(path: Path) -> dict:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{path} is not a workflow mapping")
    return document


def workflow_triggers(document: dict) -> dict:
    """The ``on:`` block. YAML 1.1 reads a bare ``on`` key as ``True``."""

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
    return [
        (job_id, step.get("name", "<unnamed>"), step)
        for job_id, job in (document.get("jobs") or {}).items()
        for step in job.get("steps") or []
    ]


def _string_leaves(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [leaf for item in value.values() for leaf in _string_leaves(item)]
    if isinstance(value, list):
        return [leaf for item in value for leaf in _string_leaves(item)]
    return []


def _step_text(document: dict) -> str:
    return "\n".join(leaf for _, _, step in _steps(document) for leaf in _string_leaves(step))


def check_script_references(repo: Path, failures: list[str]) -> None:
    for path in sorted((repo / ".github" / "workflows").glob("*.y*ml")):
        for match in SCRIPT.finditer(run_text(path.read_text(encoding="utf-8"))):
            token = match.group(1).replace("\\", "/").strip('"')
            token = re.sub(r'^\$\{?PACKAGE_DIR\}?"?/', "", token).removeprefix("./")
            if "$" not in token and not (repo / token).is_file():
                failures.append(f"{path.name} invokes missing script {token}")


def check_required_channel_coverage(repo: Path, failures: list[str]) -> None:
    policy = json.loads((repo / "ci" / "required-integration-tests.json").read_text(encoding="utf-8"))
    extra = re.escape(str(policy["channel_extra"]))
    install = re.compile(
        rf"(?:pip|python\s+-m\s+pip)\s+install[^\n]*(?:\.\[(?=[^\]]*\b{extra}\b)[^\]]+\]|witwin-{extra})", re.I
    )
    for relative in policy["required_workflows"]:
        path = repo / relative
        if not path.is_file():
            failures.append(f"required workflow missing: {relative}")
            continue
        commands = run_text(path.read_text(encoding="utf-8"))
        env = load_workflow(path).get("env") or {}
        if not install.search(commands):
            failures.append(f"{relative} does not install the Channel extra")
        if "WITWIN_CHANNEL_FINGERPRINT" not in env:
            failures.append(f"{relative} does not declare WITWIN_CHANNEL_FINGERPRINT")
        if str(env.get("WITWIN_REQUIRED_CHANNEL_SKIP_BUDGET")) != "0":
            failures.append(f"{relative} does not pin WITWIN_REQUIRED_CHANNEL_SKIP_BUDGET to 0")
        if CHANNEL_IDENTITY_SCRIPT not in commands:
            failures.append(f"{relative} does not run {CHANNEL_IDENTITY_SCRIPT}")


def check_release_claims(repo: Path, policy: dict, failures: list[str]) -> None:
    manylinux = policy["manylinux_policy"]
    for name in ("README.md", "FEATURE_LIST.md"):
        text = (repo / name).read_text(encoding="utf-8")
        if re.search(r"manylinux_2_\d+", text) and manylinux not in text:
            failures.append(f"{name} names a manylinux policy other than {manylinux}")
        if not policy["stable_abi_cross_torch_claim"] and "stable abi" in text.lower():
            failures.append(f"{name} presents cross-Torch Stable ABI as supported")


def check_torch_requirement(repo: Path, policy: dict, failures: list[str]) -> None:
    dependencies = tomllib.loads((repo / "pyproject.toml").read_text(encoding="utf-8"))["project"]["dependencies"]
    torch = [requirement.replace(" ", "") for requirement in dependencies if re.match(r"torch\b", requirement)]
    expected = f"torch=={policy['torch_version']}"
    if torch != [expected]:
        failures.append(
            f"pyproject.toml requires {torch}, but the loader accepts a packaged binary only on {expected}; "
            "the requirement must say the same"
        )


def check_triggers(document: dict, failures: list[str]) -> None:
    triggers = workflow_triggers(document)
    if not triggers:
        failures.append("the publish workflow declares no triggers")
        return
    forbidden = sorted(set(triggers) & FORBIDDEN_TRIGGERS)
    if forbidden:
        failures.append(f"forbidden trigger(s) {forbidden}: native wheel work is opt-in")
    unknown = sorted(set(triggers) - ALLOWED_TRIGGERS - FORBIDDEN_TRIGGERS)
    if unknown:
        failures.append(f"unrecognised trigger(s) {unknown}")
    if (triggers.get("release") or {}).get("types") != ["published"]:
        failures.append("the release trigger must exist with types ['published']")
    options = (((triggers.get("workflow_dispatch") or {}).get("inputs") or {}).get("scope") or {}).get("options") or []
    if "full" not in options:
        failures.append("workflow_dispatch must exist and offer scope 'full'")
    if "pull_request" in triggers:
        if (triggers.get("pull_request") or {}).get("types") != ["labeled"]:
            failures.append("the pull_request trigger must be types: [labeled]")
        for job_id, job in (document.get("jobs") or {}).items():
            if not job.get("needs") and OPT_IN_LABEL not in str(job.get("if", "")):
                failures.append(f"job {job_id!r} has no `needs` and no {OPT_IN_LABEL!r} guard")


def check_architectures(document: dict, failures: list[str]) -> None:
    env = document.get("env") or {}
    text = _step_text(document)
    for key, expected in POLICY_ENV.items():
        found = env.get(key)
        if found is None:
            failures.append(f"the publish workflow declares no `{key}` architecture profile")
            continue
        if str(found) != expected:
            failures.append(f"`{key}` is {found!r}, the policy value is {expected!r}")
        if key not in text:
            failures.append(f"`{key}` is declared but no build step uses it")


def check_verifier_and_manylinux(document: dict, policy: dict, failures: list[str]) -> None:
    linux = windows = manylinux = False
    for _, _, step in _steps(document):
        blob = "\n".join(_string_leaves(step))
        if "verify_cuda_binary_arches.py" in blob:
            condition = str(step.get("if", ""))
            linux |= "Linux" in condition or "CIBW" in blob
            windows |= "Windows" in condition or not condition
        image = str((step.get("env") or {}).get("CIBW_MANYLINUX_X86_64_IMAGE", ""))
        manylinux |= image.startswith(policy["manylinux_policy"])
    if not windows:
        failures.append("no architecture verifier runs against the Windows artifact")
    if not linux:
        failures.append("no architecture verifier runs against the Linux artifact")
    if not manylinux:
        failures.append(f"the Linux wheel is not built in a {policy['manylinux_policy']} image")


def check_wheel_shape(document: dict, failures: list[str]) -> None:
    text = _step_text(document)
    required = {
        "len(native) == 1": "the wheel carries exactly one native member",
        "torch.utils.cpp_extension": "torch.utils.cpp_extension stayed unimported",
        ".build-info.json": "the installed package ships .build-info.json",
        ".build-fingerprint": "the installed package ships .build-fingerprint",
    }
    for fragment, claim in required.items():
        if fragment not in text:
            failures.append(f"no publish step asserts {claim}")
    if 'origin"] == "packaged"' not in text and "origin'] == 'packaged'" not in text:
        failures.append("no publish step asserts the extension loaded from the packaged prebuilt")
    smoke = [step for _, name, step in _steps(document) if name == "Smoke install prebuilt wheel"]
    if len(smoke) != 1 or "python -I - <<'PY'" not in str(smoke[0].get("run", "")):
        failures.append("exactly one `Smoke install prebuilt wheel` step must run Python in isolated mode")


def check_exact_runtime_matrix(document: dict, policy: dict, failures: list[str]) -> None:
    expected = tuple((python, policy["torch_version"], policy["cuda_index"]) for python in policy["supported_python"])
    for job_id, job in (document.get("jobs") or {}).items():
        matrix = (job.get("strategy") or {}).get("matrix") or {}
        cells = matrix.get("identity")
        if not cells:
            continue
        found = tuple(
            (str(cell.get("python_version")), str(cell.get("torch_version")), str(cell.get("cuda_index")))
            for cell in cells
        )
        if found != expected:
            failures.append(f"job {job_id!r} identity cells are {found!r}, the policy requires {expected!r}")
        if sorted(matrix.get("os") or ()) != sorted(REQUIRED_OS):
            failures.append(f"job {job_id!r} must run the identity grid on {REQUIRED_OS}")
        return
    failures.append("no exact native-identity job with an `identity` matrix")


def check_publish_gating(document: dict, failures: list[str]) -> None:
    publish = (document.get("jobs") or {}).get("publish")
    if publish is None:
        failures.append("no `publish` job")
        return
    if "github.event_name == 'release'" not in str(publish.get("if", "")):
        failures.append("the publish job is not restricted to a published release")
    for required in ("build_cuda_wheels", "test_exact_identity"):
        if required not in set(publish.get("needs") or ()):
            failures.append(f"the publish job does not depend on {required!r}")
    validation = [
        str(step.get("run", ""))
        for step in publish.get("steps") or ()
        if step.get("name") == "Validate the artifact set before publishing"
    ]
    fragments = ('split(".")', '"manylinux_2_28_x86_64" in tags', 'tags == {"win_amd64"}')
    if len(validation) != 1 or not all(fragment in validation[0] for fragment in fragments):
        failures.append("the publish job must validate one Windows and one manylinux wheel by parsed platform tag")


def check_runners(repo: Path, failures: list[str]) -> None:
    for path in sorted((repo / ".github" / "workflows").glob("*.y*ml")):
        if "self-hosted" in path.read_text(encoding="utf-8") and path.name not in SELF_HOSTED_ALLOWLIST:
            failures.append(f"{path.name} uses a self-hosted runner; GitHub-hosted runners are mandatory")


def check_publish_workflow(path: Path, policy: dict) -> list[str]:
    failures: list[str] = []
    document = load_workflow(path)
    check_triggers(document, failures)
    check_architectures(document, failures)
    check_verifier_and_manylinux(document, policy, failures)
    check_wheel_shape(document, failures)
    check_exact_runtime_matrix(document, policy, failures)
    check_publish_gating(document, failures)
    return failures


def audit(repo: Path, *, publish: Path | None = None) -> list[str]:
    policy = json.loads((repo / "ci" / "release-policy.json").read_text(encoding="utf-8"))
    failures = check_publish_workflow(publish or repo / ".github" / "workflows" / "publish-witwin-radar.yml", policy)
    check_runners(repo, failures)
    check_script_references(repo, failures)
    check_required_channel_coverage(repo, failures)
    check_release_claims(repo, policy, failures)
    check_torch_requirement(repo, policy, failures)
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--publish-workflow", type=Path, default=None)
    arguments = parser.parse_args(argv)
    try:
        failures = audit(arguments.root.resolve(), publish=arguments.publish_workflow)
    except (OSError, ValueError, KeyError, yaml.YAMLError) as error:
        print(f"ci/check_workflows.py: cannot read the workflows or the policy: {error}", file=sys.stderr)
        return 2
    if failures:
        for failure in failures:
            print(f"ci/check_workflows.py: {failure}", file=sys.stderr)
        return 1
    print("ci/check_workflows.py: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
