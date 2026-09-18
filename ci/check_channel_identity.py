#!/usr/bin/env python
"""Refuse to run integration tests against any Channel build but the required one.

`WITWIN_CHANNEL_FINGERPRINT` is the repository variable that names the exact
Channel build the suites are calibrated against, and
`WITWIN_REQUIRED_CHANNEL_SKIP_BUDGET` is the number of Channel-backed tests the
workflow may skip, which policy pins at zero. Both are read from the
environment so the workflow YAML declares them once and this script is the one
consumer `ci/check_required_channel_coverage.py` looks for.
"""

from __future__ import annotations

import os

from witwin.channel import build_info


def main() -> int:
    fingerprint = str(build_info()["build_fingerprint"])
    expected = os.environ["WITWIN_CHANNEL_FINGERPRINT"]
    budget = int(os.environ["WITWIN_REQUIRED_CHANNEL_SKIP_BUDGET"])
    assert budget == 0, "required Channel skip budget must remain zero"
    assert expected, "configure repository variable WITWIN_CHANNEL_FINGERPRINT"
    assert fingerprint == expected, f"Channel fingerprint {fingerprint!r} != required {expected!r}"
    print(f"Channel build_fingerprint={fingerprint}; required-skip-budget={budget}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
