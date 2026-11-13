"""M2-3 / D-9: the adaptive phase test compares delays in float64.

The adaptive controller accepts an interval when its interpolated rows match
probed ones within ``Motion.phase_error``. With float32 delays the comparison
carried ``2 pi f_c ulp32(tau)`` of pure quantization - 0.0069 rad at 30 m and
0.0275 rad beyond about 71 m, above the default 0.02 - so whether an interval
was accepted at long range was rounding luck, and the probe count of the same
motion grew with range. Delays are now compared as the float64 pair the
composer publishes (R-ADR-040), less a quantization allowance for what the
representation can still contribute, so the same motion costs the same probes
at 5 m and at 100 m.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from support import oracle_scenes as sc

pytestmark = pytest.mark.gpu

VELOCITY_M_PER_S = 5.0


def _evaluations(range_m: float) -> tuple[int, float]:
    from witwin.radar import Motion, PointTargets

    radar = sc.radar(sc.beat_fmcw(samples=32, chirps=32))

    def trajectory(time_s: float) -> torch.Tensor:
        position = np.array([[range_m + VELOCITY_M_PER_S * float(time_s), 0.0, 0.0]])
        return torch.tensor(position, dtype=torch.float64, device=radar.device)

    result = radar.simulate(
        sc.empty_scene(),
        PointTargets(positions=trajectory(0.0), rcs=1.0, trajectory=trajectory),
        times=(0.0,),
        los=True,
        reflections=0,
        motion=Motion.adaptive(),
    )
    stats = result.adaptive_diagnostics[0]
    return int(stats["evaluations"]), float(stats["max_tested_phase_error_rad"])


def test_the_probe_count_of_one_motion_does_not_grow_with_range():
    near, near_error = _evaluations(5.0)
    far, far_error = _evaluations(100.0)
    assert max(near, far) <= 2 * min(near, far), (near, far)
    assert near_error <= 0.02 and far_error <= 0.02
