#!/usr/bin/env python
"""Require integration workflows to install Channel and forbid silent skips."""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys


INSTALL = re.compile(
    r"(?:pip|python\s+-m\s+pip)\s+install[^\n]*(?:"
    r"\.\[(?=[^\]]*\bchannel\b)[^\]]+\]|witwin-channel)",
    re.IGNORECASE,
)
RUN = re.compile(r"^(?P<spaces>\s*)(?:-\s+)?run:\s*(?P<body>.*)$")


def _run_text(text: str) -> str:
    lines = text.splitlines()
    commands: list[str] = []
    index = 0
    while index < len(lines):
        match = RUN.match(lines[index])
        if match is None:
            index += 1
            continue
        body = match.group("body").strip()
        base_indent = len(match.group("spaces"))
        if body not in {"|", "|-", "|+", ">", ">-", ">+"}:
            if body and not body.startswith("#"):
                commands.append(body)
            index += 1
            continue
        index += 1
        while index < len(lines):
            line = lines[index]
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            if stripped and indent <= base_indent:
                break
            if stripped and not stripped.startswith("#"):
                commands.append(stripped)
            index += 1
    return "\n".join(commands)


def _active_yaml(text: str) -> str:
    return "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )


def _consumed(name: str, *, commands: str, yaml_text: str) -> bool:
    declared = re.search(rf"^\s*{re.escape(name)}\s*:", yaml_text, re.MULTILINE)
    return declared is not None and name in commands


def audit(repo: Path) -> list[str]:
    policy = json.loads(
        (repo / "ci" / "required-integration-tests.json").read_text(encoding="utf-8")
    )
    errors: list[str] = []
    for relative in policy["required_workflows"]:
        path = repo / relative
        if not path.is_file():
            errors.append(f"required workflow missing: {relative}")
            continue
        text = path.read_text(encoding="utf-8")
        commands = _run_text(text)
        active_yaml = _active_yaml(text)
        if not INSTALL.search(commands):
            errors.append(f"{relative} does not install the Channel dependency")
        fingerprint_consumed = _consumed(
            "WITWIN_CHANNEL_FINGERPRINT",
            commands=commands,
            yaml_text=active_yaml,
        )
        fingerprint_observed = all(
            token in commands
            for token in ("witwin.channel", "build_info", "build_fingerprint")
        )
        if not fingerprint_consumed or not fingerprint_observed:
            errors.append(f"{relative} does not record a Channel fingerprint")
        if not _consumed(
            "WITWIN_REQUIRED_CHANNEL_SKIP_BUDGET",
            commands=commands,
            yaml_text=active_yaml,
        ):
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
