"""Radar propagation adapter package.

This package root exports the Radar-shaped contracts and the Core-kinematics
seam only. It deliberately does NOT import
:mod:`witwin.radar.propagation.channel_consumer`, so importing
``witwin.radar.propagation`` never requires ``witwin-channel`` to be installed.
Import the adapter explicitly when you mean to cross that boundary.

:mod:`witwin.radar.propagation.kinematics` names no other witwin package at all:
it is duck typed over Core's ``EndpointState`` / ``StructureState`` shape, so it
adds no import edge in either direction.
"""

from .contracts import (
    EndpointRole,
    RadarEndpointSpec,
    RadarLegBatch,
    require_endpoint_role,
)
from .kinematics import (
    DeformationVelocity,
    Kinematics,
    TwoWayDuals,
    deformation_kinematics,
    endpoint_kinematics,
    replicate_slots,
    rigid_site_velocities,
    rotation_centre_m,
    structure_site_kinematics,
    two_way_duals,
)

__all__ = [
    "DeformationVelocity",
    "EndpointRole",
    "Kinematics",
    "RadarEndpointSpec",
    "RadarLegBatch",
    "TwoWayDuals",
    "deformation_kinematics",
    "endpoint_kinematics",
    "replicate_slots",
    "require_endpoint_role",
    "rigid_site_velocities",
    "rotation_centre_m",
    "structure_site_kinematics",
    "two_way_duals",
]
