#!/usr/bin/env python
"""Keep living release claims equal to the executable release policy."""

from __future__ import annotations

import json
from pathlib import Path
import sys


def audit(repo: Path) -> list[str]:
    policy = json.loads(
        (repo / "ci" / "release-policy.json").read_text(encoding="utf-8")
    )
    errors: list[str] = []
    living = {
        "README.md": (repo / "README.md").read_text(encoding="utf-8"),
        "FEATURE_LIST.md": (repo / "FEATURE_LIST.md").read_text(encoding="utf-8"),
        "phase10-deferred-release-matrix.md": (
            repo / "docs" / "dev" / "plans" / "phase10-deferred-release-matrix.md"
        ).read_text(encoding="utf-8"),
    }
    expected_manylinux = policy["manylinux_policy"]
    for name, text in living.items():
        if "manylinux_2_35" in text:
            errors.append(f"{name} claims retired manylinux_2_35")
        if "manylinux" in text and expected_manylinux not in text:
            errors.append(f"{name} does not name {expected_manylinux}")
    if not policy["stable_abi_cross_torch_claim"]:
        for name in ("README.md", "FEATURE_LIST.md"):
            if "stable abi" in living[name].lower():
                errors.append(f"{name} presents cross-Torch Stable ABI as supported")
    workflow = (
        repo / ".github" / "workflows" / "publish-witwin-radar.yml"
    ).read_text(encoding="utf-8")
    refusal_is_success = (
        "expected_refusal" in workflow and "exit 0" in workflow
    ) or (
        "This cell measures deviation P3, not a passing Stable ABI cell."
        in workflow
        and "except build.RadarExtensionABIError" in workflow
        and "raise SystemExit(0)" in workflow
    )
    if (
        not policy["expected_loader_refusal_is_release_success"]
        and refusal_is_success
    ):
        errors.append(
            "publish workflow treats expected loader refusal as successful release evidence"
        )
    return errors


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    errors = audit(repo)
    if errors:
        for error in errors:
            print(f"ci/check_release_claims.py: {error}", file=sys.stderr)
        return 1
    print("ci/check_release_claims.py: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
