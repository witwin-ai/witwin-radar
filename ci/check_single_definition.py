#!/usr/bin/env python
"""Reject duplicate concept owners and undeclared public definition targets."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def audit(repo: Path) -> list[str]:
    architecture = json.loads((repo / "ci" / "architecture-manifest.json").read_text(encoding="utf-8"))
    public = json.loads((repo / "ci" / "public-api-manifest.json").read_text(encoding="utf-8"))
    errors = []
    concepts = architecture["concept_owners"]
    duplicate_concepts = [owner for owner, count in Counter(concepts.values()).items() if count > 1]
    errors.extend(
        f"module {owner} owns multiple concept axes without an explicit merge" for owner in sorted(duplicate_concepts)
    )

    targets = []
    for module, exports in public["modules"].items():
        for name, target in exports.items():
            targets.append((f"{module}.{name}", target))
    duplicate_targets = [target for target, count in Counter(target for _, target in targets).items() if count > 1]
    for target in sorted(duplicate_targets):
        exposures = sorted(name for name, candidate in targets if candidate == target)
        errors.append(f"canonical target {target} has multiple exposures: {exposures}")

    target_modules = set(architecture["target_modules"])
    for exposure, target in targets:
        owner = target.rpartition(".")[0]
        if owner not in target_modules:
            errors.append(f"public exposure {exposure} names non-target owner module {owner}")
    return errors


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    errors = audit(repo)
    if errors:
        for error in errors:
            print(f"ci/check_single_definition.py: {error}", file=sys.stderr)
        return 1
    print("ci/check_single_definition.py: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
