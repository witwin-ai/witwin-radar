"""The executable text of a GitHub Actions workflow: every `run:` body, comments dropped."""

from __future__ import annotations

import re

RUN = re.compile(r"^(?P<spaces>\s*)(?:-\s+)?run:\s*(?P<body>.*)$")


def run_text(text: str) -> str:
    """Concatenate the commands of every ``run:`` step, one per line.

    Block scalars (``|``, ``>`` and their chomping variants) are read up to
    the first line indented no deeper than the ``run:`` key. Comment lines are
    dropped so a gate cannot be satisfied by a commented-out command.
    """

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
