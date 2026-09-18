#!/usr/bin/env python
"""Regenerate ``ci/public-api-snapshot.json`` from the live package.

The builder lives beside the test that pins the snapshot,
``tests/test_public_api_snapshot.py``, so that regenerating and checking read
the same code. Run this after an intended public-surface change, then review
the diff.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from test_public_api_snapshot import SNAPSHOT, build_snapshot  # noqa: E402


def main() -> int:
    SNAPSHOT.write_text(json.dumps(build_snapshot(), indent=2) + "\n", encoding="utf-8")
    print(f"regenerated {SNAPSHOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
