"""The scene-driven fixture: one radar, two scatterers, one static world.

``Radar.simulate`` tests share these so that every one of them drives the same
frozen multi-endpoint world ``support.multi_endpoint_driver`` assembles by hand,
and a change to the fixture is one edit rather than one per file.
"""

from __future__ import annotations

import torch

from witwin.radar import PointTargets, Radar

from . import multi_endpoint_driver as drv
from . import multi_endpoint_geometry as geo
from . import multi_endpoint_world as world

#: The radar looks along +x, so its two half-wavelength elements sit along the
#: world z axis. That is not the multi-endpoint fixture's transmitter geometry
#: and it does not have to be: the fixture wall and sites are what make the
#: reflection rows interesting, and the front end is this radar's own.
LOOK_AT_M = (1.0, 0.0, 0.0)

SITE_POSITIONS_M = (geo.SITE_P_POSITION_M, geo.SITE_Q_POSITION_M)


def fixture_radar() -> Radar:
    """The fixture radar. Its pattern is the default isotropic one.

    The stage is then a proven no-op, which is what every assertion wants: a
    dipole roll-off nobody asked for would scale each row by its own bearing.

    ``polarization`` is declared rather than left to the pose-derived default
    because the entry tests are checked against ``multi_endpoint_driver``, whose
    endpoints carry ``geo.POLARIZATION``. Channel projects the material field
    onto it, so two different transverse axes give two different - both correct
    - complex transfers, and the comparison is bitwise. With this pose the
    declared vector IS the frame's ``right`` axis, so it stays transverse to the
    boresight and nothing radiates into a null.
    """

    return Radar.from_dict(
        dict(geo.FIXTURE_RADAR_CONFIG), position=(0.0, 0.0, 0.0), look_at=LOOK_AT_M, polarization=geo.POLARIZATION
    )


def point_targets(positions: torch.Tensor, *, trajectory=None) -> PointTargets:
    """``positions`` as scatterers authored as the dimensionless strength.

    ``amplitude`` rather than ``rcs`` so the leaf a gradient test marks is the
    strength itself: going through the cross-section law would put a square root
    in the gradient that has nothing to do with what is being measured.
    """

    return PointTargets(
        positions=positions, amplitude=drv.FIXTURE_AMPLITUDE, phase=drv.FIXTURE_PHASE_RAD, trajectory=trajectory
    )


def fixture_targets(radar: Radar, *, requires_grad: bool = False) -> PointTargets:
    """The two fixture scatterers, on ``radar.device``."""

    positions = torch.tensor(SITE_POSITIONS_M, dtype=torch.float32, device=radar.device).requires_grad_(requires_grad)
    return point_targets(positions)


def static_scene():
    scene, mesh = world.make_scene()
    world.assert_world_coordinates_survived(mesh)
    return scene


__all__ = ["LOOK_AT_M", "SITE_POSITIONS_M", "fixture_radar", "fixture_targets", "point_targets", "static_scene"]
