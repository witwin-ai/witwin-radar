"""Public API enums for radar.

The enum subclasses ``StrEnum`` so users can pass either a member or a raw
string; IDEs still complete members and invalid values raise immediately.

``SamplingMode``, ``MotionSampling``, ``TraceSample`` and ``InterpolatorFn``
stood here until Phase 11. All four described the interpolator contract of the
Dirichlet route - a callable that reports scatterer positions at a time - and
that route is deleted. The scene-driven entry takes a Core ``Scene`` and a
declared frame sequence instead, so there is nothing left for them to describe.
"""

from __future__ import annotations

from enum import StrEnum


class DetectorType(StrEnum):
    CFAR = "cfar"
    TOPK = "topk"
