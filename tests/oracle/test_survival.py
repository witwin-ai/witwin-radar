"""M0-6: no legal input stops a run, corrupts the CUDA context or fails without a name.

Each case in ``survival_cases.py`` runs in a fresh interpreter under a
timeout, then that same process runs a trivial CUDA reduction and a device
synchronize. Every case must:

1. exit cleanly (no crash, no hang, no unparseable output);
2. leave the CUDA context usable;
3. either return a correct result (finite, and the case's own physical check)
   or raise a refusal that names its cause: a ``ValueError``,
   ``TypeError`` or ``NotImplementedError`` whose message mentions the
   condition. Any other exception class (an ``IndexError`` from a shape, a
   ``RuntimeError`` from a kernel, a ``ZeroDivisionError``) is unnamed by
   definition, whatever its text.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

CHILD = Path(__file__).with_name("survival_cases.py")
#: Per case: interpreter start, imports, Channel compile and one small frame.
TIMEOUT_S = 180

NAMED_REFUSALS = ("ValueError", "TypeError", "NotImplementedError")

_MP2 = (
    "MP-2: a declared site with no path at some discovery makes the two-way join raise ValueError instead of "
    "contributing zero, so a target walking behind an occluder stops the run"
)


_COINCIDENT = (
    "M0-6 (no audit finding yet): a site exactly on an array element returns a finite cube without a refusal, "
    "although the radar equation is singular there (R = 0)"
)

WAVELENGTH_M = 299_792_458.0 / 77e9
WAVENUMBER = 2.0 * math.pi / WAVELENGTH_M


def _radar_equation_power_w(range_m: float) -> float:
    """``P_t lambda^2 sigma / ((4 pi)^3 R^4)`` at 10 dBm, 77 GHz, 1 m^2, unit gains."""

    return 1e-2 * WAVELENGTH_M**2 / ((4.0 * math.pi) ** 3 * range_m**4)


def _relative_power(record, range_m: float) -> float:
    return record["checks"]["mean_power"] / _radar_equation_power_w(range_m)


def _check_occluded(record):
    before, during, after = record["checks"]["frame_power"]
    return before > 0.0 and during == 0.0 and after > 0.0


def _check_near_wall(record):
    """Image method at a perfect conductor, normal incidence, TE (``Gamma = -1``).

    With the site ``delta`` in front of the wall the four round trips sum to
    ``C (1 - exp(-2 j k delta))^2``, so the echo is the free-space echo times
    ``(2 sin(k delta))^4`` (-32 dB at 0.05 mm). ``delta`` is taken from the
    float32 site coordinate production actually holds.
    """

    delta = 5.0 - float(np.float32(5.0 - 5.0e-5))
    expected = (2.0 * math.sin(WAVENUMBER * delta)) ** 4
    return record["checks"]["rows"] == 4 and _relative_power(record, 5.0) == pytest.approx(expected, rel=0.03)


def _check_on_wall(record):
    """On the conductor the total tangential field, and with it the echo, vanishes.

    The four rows cancel to float32 rounding; -60 dB below the free-space echo
    is the bound.
    """

    return _relative_power(record, 5.0) < 1e-6


#: case -> (what a correct result must satisfy, words a refusal must contain, xfail reason or None).
CASES = {
    "occluded_mid_sequence": (_check_occluded, (), _MP2),
    "site_on_wall": (_check_on_wall, ("wall", "surface", "coincid", "distance"), None),
    "site_near_wall": (_check_near_wall, ("wall", "surface", "coincid", "distance"), None),
    "site_at_element": (
        lambda record: False,
        ("coincid", "zero distance", "element", "endpoint", "distance"),
        _COINCIDENT,
    ),
    "empty_scene": (
        lambda record: (
            record["checks"]["rows"] == 1 and _relative_power(record, 5.0) == pytest.approx(1.0, rel=1e-3, abs=0.0)
        ),
        (),
        None,
    ),
    "no_targets": (lambda record: True, ("empty", "no target", "at least one", "site_count"), None),
    "single_triangle": (lambda record: record["checks"]["rows"] == 4, (), None),
    "degenerate_mesh": (lambda record: record["checks"]["rows"] == 4, ("degenerate", "duplicate", "area"), None),
    "very_long_range": (
        lambda record: _relative_power(record, 50_000.0) == pytest.approx(1.0, rel=1e-3, abs=0.0),
        ("range", "distance", "delay"),
        None,
    ),
    "many_sites": (lambda record: record["checks"]["rows"] == 20_000, (), None),
}


def _run_child(case: str) -> dict:
    environment = dict(os.environ)
    root = CHILD.parents[2]
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(root), str(root / "tests"), *filter(None, [environment.get("PYTHONPATH")])]
    )
    completed = subprocess.run(
        [sys.executable, str(CHILD), case],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
        check=False,
    )
    lines = [line for line in completed.stdout.splitlines() if line.startswith("SURVIVAL-RESULT ")]
    assert completed.returncode == 0 and len(lines) == 1, (
        f"{case}: exit {completed.returncode}\nstdout:\n{completed.stdout[-2000:]}\nstderr:\n{completed.stderr[-4000:]}"
    )
    return json.loads(lines[0].removeprefix("SURVIVAL-RESULT "))


def _params():
    for case, (_, _, reason) in CASES.items():
        marks = [pytest.mark.xfail(strict=True, reason=reason)] if reason else []
        yield pytest.param(case, marks=marks, id=case)


@pytest.mark.parametrize("case", list(_params()))
def test_case_survives_with_a_correct_result_or_a_named_refusal(case):
    accepts, refusal_words, _ = CASES[case]
    record = _run_child(case)
    assert record["cuda_ok"], f"{case}: the CUDA context is unusable after the case: {record}"
    if record["outcome"] == "result":
        assert record["finite"], f"{case}: returned a non-finite cube: {record}"
        assert accepts(record), f"{case}: returned a wrong result: {record}"
        return
    message = record["message"].lower()
    assert record["exception"] in NAMED_REFUSALS, f"{case}: unnamed exception {record['exception']}: {record}"
    assert refusal_words and any(word in message for word in refusal_words), (
        f"{case}: the refusal does not name its cause ({refusal_words}): {record['exception']}: {record['message']}"
    )
