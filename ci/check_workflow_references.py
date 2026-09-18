#!/usr/bin/env python
"""Reject workflow commands that invoke missing repository scripts."""

from __future__ import annotations

import re
import sys
from pathlib import Path

from _workflow_text import run_text

SCRIPT = re.compile(r"(?:^|[\s;&|])(?:python|python3|py)\s+(?!-m\b)([A-Za-z0-9_./\\-]+\.py)\b", re.MULTILINE)


def audit(repo: Path) -> list[str]:
    errors = []
    workflows = repo / ".github" / "workflows"
    for path in sorted((*workflows.glob("*.yml"), *workflows.glob("*.yaml"))):
        commands = run_text(path.read_text(encoding="utf-8"))
        for match in SCRIPT.finditer(commands):
            token = match.group(1).replace("\\", "/")
            if token.startswith("./"):
                token = token[2:]
            if not (repo / token).is_file():
                errors.append(f"{path.relative_to(repo).as_posix()} invokes missing script {token}")
    return errors


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    errors = audit(repo)
    if errors:
        for error in errors:
            print(f"ci/check_workflow_references.py: {error}", file=sys.stderr)
        return 1
    print("ci/check_workflow_references.py: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
