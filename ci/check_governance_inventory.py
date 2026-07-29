#!/usr/bin/env python
"""Validate the consolidation debt ledger and fail while any row is open."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROW = re.compile(r"^\|\s*(GOV-\d{3})\s*\|(.+)\|\s*$")


def audit(repo: Path, *, require_closed: bool) -> list[str]:
    path = repo / "docs" / "dev" / "audit" / "radar-governance-debt-and-drift-inventory.md"
    rows = {}
    errors = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        match = ROW.match(line)
        if not match:
            continue
        debt_id = match.group(1)
        columns = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(columns) != 8:
            errors.append(f"{debt_id} line {lineno} has {len(columns)} columns, expected 8")
            continue
        if debt_id in rows:
            errors.append(f"duplicate debt ID {debt_id}")
        rows[debt_id] = columns
        status = columns[6]
        if status not in {"open", "closed", "external blocker"}:
            errors.append(f"{debt_id} has unknown status {status!r}")
        if status == "closed" and columns[7] in {"", "—"}:
            errors.append(f"{debt_id} is closed without evidence")
        if require_closed and status != "closed":
            errors.append(f"{debt_id} is not closed: {status}")
    expected = [f"GOV-{index:03d}" for index in range(1, 28)]
    if sorted(rows) != expected:
        errors.append(f"debt IDs must be contiguous GOV-001..GOV-027, got {sorted(rows)}")
    return errors


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    require_closed = "--schema-only" not in sys.argv[1:]
    errors = audit(repo, require_closed=require_closed)
    if errors:
        for error in errors:
            print(f"ci/check_governance_inventory.py: {error}", file=sys.stderr)
        return 1
    print("ci/check_governance_inventory.py: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
