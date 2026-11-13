"""PH-5: the radar equation off the plane normal to the polarization.

One co-located transmit/receive element at the origin looking along +x with
``polarization="up"`` (world +y), one isotropic point scatterer of
``sigma = 1 m^2`` at 5 m, swept off boresight in two planes:

* the E-plane (toward the polarization, world x-y), where the target leaves
  the plane normal to the polarization;
* the H-plane (world x-z), which stays in that plane and is the control.

The physical statement is the monostatic radar equation with co-polarized
elements and a polarization-preserving scatterer,
``P_r = P_t G(u)^2 lambda^2 sigma / ((4 pi)^3 R^4)``, where ``G(u)`` is the
element's DECLARED power gain toward the target: exactly one for
``Pattern.isotropic()`` and the tabulated cut for ``Pattern.dipole()``. No
other angular factor belongs in it. The received power is read from the beat
cube of a static target, whose every sample has ``|x|^2 = P_r`` in sqrt(W).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from reference import radar_oracles as ref
from support import oracle_scenes as sc

pytestmark = pytest.mark.gpu

RANGE_M = 5.0
RCS_M2 = 1.0
POWER_DBM = 10.0
#: Relative tolerance on received power. A float32 cube of a 5 m static
#: target reproduces the radar equation to ~3e-8 on the polarization plane;
#: 1e-3 (0.004 dB) only absorbs float32 rounding of the pattern lookup.
POWER_RTOL = 1e-3
ANGLES_DEG = (0.0, 30.0, 45.0, 60.0)


def _direction(plane: str, angle_deg: float) -> tuple[float, float, float]:
    angle = math.radians(angle_deg)
    if plane == "E":
        return (math.cos(angle), math.sin(angle), 0.0)
    return (math.cos(angle), 0.0, math.sin(angle))


def _declared_gain(pattern, direction) -> float:
    """The separable pattern's tabulated power gain toward ``direction``.

    The local frame is right = world +z, up = world +y, boresight = world +x,
    so the pattern's two angles are ``atan2(d_z, d_x)`` and ``atan2(d_y, d_x)``.
    Linear interpolation of the declared knots; zero outside the support.
    """

    dx, dy, dz = direction
    ax = math.degrees(math.atan2(dz, dx))
    ay = math.degrees(math.atan2(dy, dx))
    gx = float(np.interp(ax, pattern.x_angles, pattern.x_gain, left=0.0, right=0.0))
    gy = float(np.interp(ay, pattern.y_angles, pattern.y_gain, left=0.0, right=0.0))
    return gx * gy


def _cases():
    for pattern in ("isotropic", "dipole"):
        for plane in ("E", "H"):
            for angle in ANGLES_DEG:
                yield pytest.param(pattern, plane, angle, id=f"{pattern}-{plane}plane-{angle:g}deg")


@pytest.mark.parametrize(("pattern_name", "plane", "angle_deg"), list(_cases()))
def test_received_power_follows_the_radar_equation_off_the_polarization_plane(pattern_name, plane, angle_deg):
    from witwin.radar import Pattern, PointTargets

    pattern = Pattern.isotropic() if pattern_name == "isotropic" else Pattern.dipole()
    radar = sc.radar(pattern=pattern, power_dbm=POWER_DBM)
    direction = _direction(plane, angle_deg)
    position = [[RANGE_M * component for component in direction]]

    result = radar.simulate(
        sc.empty_scene(), PointTargets(positions=position, rcs=RCS_M2), times=(0.0,), los=True, reflections=0
    )
    measured = float(result.cube.abs().double().pow(2).mean())

    gain = _declared_gain(pattern, direction)
    expected = ref.bistatic_received_power_w(
        transmit_power_w=ref.watts_from_dbm(POWER_DBM),
        gain_tx=gain,
        gain_rx=gain,
        wavelength=ref.wavelength_m(sc.CARRIER_HZ),
        rcs_m2=RCS_M2,
        r_in=RANGE_M,
        r_out=RANGE_M,
    )
    # As a ratio: the received power is ~1e-13 W, below pytest.approx's default
    # absolute tolerance, so comparing the powers themselves would pass anything.
    assert measured / expected == pytest.approx(1.0, rel=POWER_RTOL, abs=0.0), (
        f"{pattern_name} {plane}-plane {angle_deg} deg: measured {ref.db(measured / expected):+.3f} dB "
        "against the radar equation"
    )
