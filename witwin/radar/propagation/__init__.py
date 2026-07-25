"""Radar propagation adapter package.

This package root exports the Radar-shaped contracts only. It deliberately does
NOT import :mod:`witwin.radar.propagation.channel_consumer`, so importing
``witwin.radar.propagation`` never requires ``witwin-channel`` to be installed.
Import the adapter explicitly when you mean to cross that boundary.
"""

from .contracts import (
    EndpointRole,
    RadarEndpointSpec,
    RadarLegBatch,
    require_endpoint_role,
)

__all__ = [
    "EndpointRole",
    "RadarEndpointSpec",
    "RadarLegBatch",
    "require_endpoint_role",
]
