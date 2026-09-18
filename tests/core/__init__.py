"""Radar core-record tests: the configuration surface, the pose, the pattern.

Named owner for the two helpers ``test_antenna_pattern`` and ``test_radar_pose``
both need. Each file used to carry its own byte-identical copy, under two names
in the direction case, which is exactly the drift R-ADR-021 section 5 rejects:
one convention, one place to change it.
"""

from __future__ import annotations

import math

import torch


def local_target_position(x_deg: float, y_deg: float, radius: float = 2.0) -> torch.Tensor:
    """A point ``radius`` metres away at ``(x_deg, y_deg)`` off boresight, in RADAR-LOCAL metres.

    Boresight is ``-z``, so the tangents of the two angles ride on ``x`` and
    ``y`` against a ``-1`` forward component. The vector is normalized before
    scaling, so ``radius`` is a true range and not a ``z`` depth: the angular
    pair alone decides the pattern gain.
    """

    direction = torch.tensor([math.tan(math.radians(x_deg)), math.tan(math.radians(y_deg)), -1.0], dtype=torch.float32)
    direction = direction / torch.linalg.norm(direction)
    return direction * radius


def half_wave_dipole_power(angle_deg: float) -> float:
    """POWER gain of an ideal half-wave dipole at ``angle_deg`` off boresight, normalized to 1 at boresight.

    ``F(theta) = cos(pi/2 sin theta) / cos theta`` is the field pattern; the
    return is ``F**2`` because the tables the tests compare against tabulate
    power. The closed form is the independent oracle for the tabulated dipole
    ``Pattern.dipole()`` builds, so it must not be derived from that table.
    The null at ``theta = +-90 deg`` is the removable singularity of the
    expression, returned as an exact zero rather than left to divide by a
    denominator that has already underflowed.
    """

    angle_rad = math.radians(angle_deg)
    cos_angle = math.cos(angle_rad)
    if abs(cos_angle) < 1e-8:
        return 0.0
    field = math.cos(0.5 * math.pi * math.sin(angle_rad)) / cos_angle
    return field * field
