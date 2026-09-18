#!/usr/bin/env python
"""Require integration workflows to install Channel and forbid silent skips."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from _workflow_text import run_text

#: The workflow step that checks the Channel identity. Its source counts as
#: command text: the env vars it reads and the build_info() call it makes are
#: what the workflow consumes by invoking it.
CHANNEL_IDENTITY_SCRIPT = "ci/check_channel_identity.py"

#: The extra name is built into the pattern rather than hard-coded in it, so
#: `channel_extra` in the policy is read by the check it names instead of
#: restating a string the regex already froze.
DEFAULT_CHANNEL_EXTRA = "channel"


def _install_pattern(extra: str) -> re.Pattern[str]:
    quoted = re.escape(extra)
    return re.compile(
        rf"(?:pip|python\s+-m\s+pip)\s+install[^\n]*(?:"
        rf"\.\[(?=[^\]]*\b{quoted}\b)[^\]]+\]|witwin-{quoted})",
        re.IGNORECASE,
    )


def _active_yaml(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))


def _consumed(name: str, *, commands: str, yaml_text: str) -> bool:
    declared = re.search(rf"^\s*{re.escape(name)}\s*:", yaml_text, re.MULTILINE)
    return declared is not None and name in commands


def _audit_test_prefixes(repo: Path, policy: dict) -> list[str]:
    """Require every declared test prefix to match a file that exists.

    `required_test_prefixes` had no reader, so a suite could be renamed or
    deleted and the policy would keep naming it. The property checked is the
    one the key claims: the prefix still selects at least one test file. What
    it deliberately does NOT check is that the workflows pass the prefix on a
    command line - these prefixes describe the suites the Channel budget
    covers, and the workflows run the whole suite rather than each prefix.
    """

    errors: list[str] = []
    for prefix in policy.get("required_test_prefixes", ()):
        parent = repo / prefix
        if parent.is_dir():
            matched = any(parent.rglob("test_*.py"))
        else:
            directory = repo / str(Path(prefix).parent)
            stem = Path(prefix).name
            matched = directory.is_dir() and any(path.name.startswith(stem) for path in directory.glob(f"{stem}*.py"))
        if not matched:
            errors.append(f"required test prefix matches no test file: {prefix}")
    return errors


def audit(repo: Path) -> list[str]:
    policy = json.loads((repo / "ci" / "required-integration-tests.json").read_text(encoding="utf-8"))
    errors: list[str] = _audit_test_prefixes(repo, policy)
    install = _install_pattern(str(policy.get("channel_extra", DEFAULT_CHANNEL_EXTRA)))
    for relative in policy["required_workflows"]:
        path = repo / relative
        if not path.is_file():
            errors.append(f"required workflow missing: {relative}")
            continue
        text = path.read_text(encoding="utf-8")
        commands = run_text(text)
        if CHANNEL_IDENTITY_SCRIPT in commands:
            commands += "\n" + (repo / CHANNEL_IDENTITY_SCRIPT).read_text(encoding="utf-8")
        active_yaml = _active_yaml(text)
        if not install.search(commands):
            errors.append(f"{relative} does not install the Channel dependency")
        fingerprint_consumed = _consumed("WITWIN_CHANNEL_FINGERPRINT", commands=commands, yaml_text=active_yaml)
        fingerprint_observed = all(token in commands for token in ("witwin.channel", "build_info", "build_fingerprint"))
        if not fingerprint_consumed or not fingerprint_observed:
            errors.append(f"{relative} does not record a Channel fingerprint")
        if not _consumed("WITWIN_REQUIRED_CHANNEL_SKIP_BUDGET", commands=commands, yaml_text=active_yaml):
            errors.append(f"{relative} does not enforce a Channel skip budget")
    if int(policy["allowed_channel_skips"]) != 0:
        errors.append("required Channel skip budget must be exactly zero")
    return errors


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    errors = audit(repo)
    if errors:
        for error in errors:
            print(f"ci/check_required_channel_coverage.py: {error}", file=sys.stderr)
        return 1
    print("ci/check_required_channel_coverage.py: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
