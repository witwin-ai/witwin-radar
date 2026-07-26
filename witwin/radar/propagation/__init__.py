"""Radar propagation adapter package.

This package root exports the Radar-shaped contracts and the Core-kinematics
seam only. It deliberately does NOT import
:mod:`witwin.radar.propagation.channel_consumer`, so importing
``witwin.radar.propagation`` never requires ``witwin-channel`` to be installed.
Import the adapter explicitly when you mean to cross that boundary.

:mod:`witwin.radar.propagation.kinematics` and
:mod:`witwin.radar.propagation.epochs` name no witwin package at module scope:
both are duck typed over Core's ``SceneSnapshot`` / ``DynamicScene`` shape and
over the adapter's own surface, so neither adds an import edge in either
direction. ``epochs`` takes its scene compiler as an argument for the same
reason: compiling is a Channel lifecycle operation and the adapter is the only
Radar module allowed to name ``witwin.channel``.
"""

from .contracts import (
    EndpointRole,
    RadarEndpointSpec,
    RadarLegBatch,
    require_endpoint_role,
)
from .epochs import EpochFrame, FrozenEpoch, SceneEpochLoop
from .kinematics import (
    DeformationVelocity,
    Kinematics,
    LinearDeformation,
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
    "EpochFrame",
    "FrozenEpoch",
    "Kinematics",
    "LinearDeformation",
    "RadarEndpointSpec",
    "RadarLegBatch",
    "SceneEpochLoop",
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
