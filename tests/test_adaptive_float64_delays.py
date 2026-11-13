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

    radar = sc.radar(sc.beat_fmcw(samples=64, chirps=64))

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
    # A linear motion is interpolated exactly: what the test measures is the
    # representation alone, which the allowance removes at every range.
    assert near_error <= 1e-4 and far_error <= 1e-4, (near_error, far_error)


def test_float32_delays_would_put_the_quantization_into_the_phase_test(monkeypatch):
    """The falsifier: compared as float32, the same motion's tested error grows with range."""

    import witwin.radar.paths as paths

    monkeypatch.setattr(paths, "float64_delay", lambda batch: batch.total_delay_s.double())
    _, near_error = _evaluations(5.0)
    _, far_error = _evaluations(100.0)
    # 2 pi f_c ulp32(tau): up to 1.4e-2 rad at 100 m and 16 times less at 5 m
    # (the float32 ulp of 667 ns over that of 33 ns), against a float64 floor
    # of 1e-4 at both. What a sample of it reaches is rounding luck, so the
    # falsifier asks for an order of magnitude above the floor and growth.
    assert far_error > 2e-3 and far_error > 5 * max(near_error, 1e-4), (near_error, far_error)
