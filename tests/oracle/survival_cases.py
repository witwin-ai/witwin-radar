"""The survival cases, each run in a process of its own.

``python tests/oracle/survival_cases.py <case>`` runs one case and prints one
line, ``SURVIVAL-RESULT {json}``, which ``test_survival.py`` parses. The case
runs in a fresh interpreter because a defect this suite looks for is a corrupt
CUDA context, and a corrupt context poisons every later test in the process
that met it.

The record says what happened, never whether it was acceptable; the verdict is
the test's:

* ``outcome``: ``"result"`` when the call returned, ``"raised"`` otherwise.
* ``exception`` / ``message``: the exception's class name and text.
* ``finite``: every cube entry is finite (only for a result).
* ``checks``: case-specific measurements (only for a result).
* ``cuda_ok``: a trivial CUDA reduction and a device synchronize, run AFTER
  the case, both succeeded and returned the right number.
"""

from __future__ import annotations

import json
import os
import sys
import traceback

import torch

# Run as a script: make the worktree's ``witwin`` and the test-support package
# importable the same way ``tests/conftest.py`` does for pytest.
_TESTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (_TESTS, os.path.dirname(_TESTS)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from support import oracle_scenes as sc  # noqa: E402

#: A small beat-domain ramp: every case is about surviving, not about DSP.
_WAVEFORM_SAMPLES = 16


def _radar():
    return sc.radar(sc.beat_fmcw(samples=_WAVEFORM_SAMPLES))


def _pec():
    from witwin.core import PhysicalMaterial

    return PhysicalMaterial.perfect_conductor()


def _wall_at_x(x_m: float, *, half: float = 3.0):
    corners = ((x_m, -half, -half), (x_m, half, -half), (x_m, half, half), (x_m, -half, half))
    return sc.plane_scene(corners=corners, material=_pec())


def _simulate(scene, positions, *, reflections: int):
    from witwin.radar import PointTargets

    return _radar().simulate(
        scene, PointTargets(positions=positions, rcs=1.0), times=(0.0,), los=True, reflections=reflections
    )


def case_occluded_mid_sequence():
    """A target walks behind a small plate and out again across three frames.

    The plate spans ``|y|, |z| <= 0.5`` at ``x = 5``; the target moves along
    ``y`` at ``x = 10`` from ``y = -3`` (visible) through ``y = 0`` (in the
    shadow, whose edge at ``x = 10`` is ``|y| = 1``) to ``y = 3`` (visible).
    Line of sight only, so the occluded frame has no path at all.
    """

    from witwin.radar import Motion, PointTargets

    corners = ((5.0, -0.5, -0.5), (5.0, 0.5, -0.5), (5.0, 0.5, 0.5), (5.0, -0.5, 0.5))
    scene = sc.plane_scene(corners=corners, material=_pec())

    def trajectory(time_s):
        return torch.tensor([[10.0, -3.0 + 6.0 * float(time_s), 0.0]], dtype=torch.float32, device="cuda")

    result = _radar().simulate(
        scene,
        PointTargets(positions=trajectory(0.0), rcs=1.0, trajectory=trajectory),
        times=(0.0, 0.5, 1.0),
        los=True,
        reflections=0,
        motion=Motion.chirp(),
    )
    power = result.cube.abs().pow(2).flatten(1).sum(dim=1)
    return result, {"frame_power": [float(value) for value in power]}


def _mean_power(result) -> dict:
    """Mean ``|x|^2`` of the beat cube and the rows composed, for a physical check."""

    return {
        "mean_power": float(result.cube.abs().double().pow(2).mean()),
        "rows": int(result.last_radar_paths.path_count),
    }


def case_site_on_wall():
    result = _simulate(_wall_at_x(5.0), [[5.0, 0.0, 0.0]], reflections=1)
    return result, _mean_power(result)


def case_site_near_wall():
    """0.05 mm in front of a perfectly conducting wall, on the radar's side."""

    result = _simulate(_wall_at_x(5.0), [[5.0 - 5.0e-5, 0.0, 0.0]], reflections=1)
    return result, _mean_power(result)


def case_site_at_element():
    """The site sits exactly on the (co-located) transmit and receive element."""

    result = _simulate(sc.empty_scene(), [[0.0, 0.0, 0.0]], reflections=0)
    return result, _mean_power(result)


def case_empty_scene():
    result = _simulate(sc.empty_scene(), [[5.0, 0.0, 0.0]], reflections=1)
    return result, _mean_power(result)


def case_no_targets():
    positions = torch.zeros((0, 3), dtype=torch.float32, device="cuda")
    return _simulate(sc.empty_scene(), positions, reflections=0), {}


def case_single_triangle():
    """One triangle, big enough to hold the specular point of a target at 6 m."""

    corners = ((-1.0, -2.0, 2.0), (12.0, -2.0, 2.0), (6.0, 4.0, 2.0))
    scene = sc.plane_scene(corners=corners, material=_pec(), faces=((0, 1, 2),))
    result = _simulate(scene, [[6.0, 0.0, 0.0]], reflections=1)
    return result, {"rows": int(result.last_radar_paths.path_count)}


def case_degenerate_mesh():
    """A valid wall plus a zero-area face and a duplicated face."""

    corners = ((-1.0, -3.0, 2.0), (12.0, -3.0, 2.0), (12.0, 3.0, 2.0), (-1.0, 3.0, 2.0), (5.0, 0.0, 2.0))
    faces = (
        (0, 1, 2),
        (0, 2, 3),
        (0, 2, 4),  # vertex 4 lies on the 0-2 diagonal: zero area
        (0, 1, 2),  # duplicate of face 0
    )
    scene = sc.plane_scene(corners=corners, material=_pec(), faces=faces)
    result = _simulate(scene, [[6.0, 0.0, 0.0]], reflections=1)
    return result, {"rows": int(result.last_radar_paths.path_count)}


def case_very_long_range():
    """50 km: far past the beat Nyquist range of the small ramp."""

    result = _simulate(sc.empty_scene(), [[50_000.0, 0.0, 0.0]], reflections=0)
    return result, _mean_power(result)


def case_many_sites():
    """20 000 sites on a 1 m patch at 20 m, line of sight only."""

    side = torch.linspace(-0.5, 0.5, 200, dtype=torch.float32)
    y, z = torch.meshgrid(side, side[:100], indexing="ij")
    positions = torch.stack((torch.full_like(y, 20.0), y, z), dim=-1).reshape(-1, 3).contiguous().to("cuda")
    result = _simulate(sc.empty_scene(), positions, reflections=0)
    return result, {"rows": int(result.last_radar_paths.path_count)}


CASES = {name.removeprefix("case_"): function for name, function in globals().items() if name.startswith("case_")}


def _cuda_still_usable() -> bool:
    try:
        value = torch.ones(1024, device="cuda").sum().item()
        torch.cuda.synchronize()
        return value == 1024.0
    except Exception:  # noqa: BLE001 - any failure here IS the measurement
        return False


def run(name: str) -> dict:
    record: dict = {"case": name}
    try:
        result, checks = CASES[name]()
        record["outcome"] = "result"
        record["finite"] = bool(torch.isfinite(torch.view_as_real(result.cube)).all())
        record["checks"] = checks
    except Exception as error:  # noqa: BLE001 - the verdict is the parent's
        record["outcome"] = "raised"
        record["exception"] = type(error).__name__
        record["message"] = str(error)
        record["traceback"] = traceback.format_exc(limit=6)
    record["cuda_ok"] = _cuda_still_usable()
    return record


if __name__ == "__main__":
    print("SURVIVAL-RESULT " + json.dumps(run(sys.argv[1])), flush=True)
