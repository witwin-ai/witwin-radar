"""Radar round-trip path composition.

Two composers, one result contract. :class:`TwoWayComposer` joins an inbound
and an outbound leg through a scatter site; :class:`DirectComposer` publishes a
single source-to-sink leg with no site at all. The mode is chosen explicitly by
the caller and recorded on the batch, so nothing downstream has to infer it and
there is no path by which one silently becomes the other.

:class:`RadarComponentIndex` is a third thing and it is a SIDECAR: it names
what each composed row is - target echo, environment clutter, direct leakage,
multi-interaction - without adding a column to :class:`RadarPathTopology`.
Every component export therefore shares the same topology OBJECT, which is what
makes "processing does not change propagation row identity" a checkable
statement rather than a claim.

The package root exports the contracts and the composers. It does not import
the Channel adapter; the composers duck-type the frozen leg handles they are
given, so this package never crosses the Channel boundary either.
"""

from .components import (
    COMPONENT_NAMES,
    DIRECT_LEAKAGE,
    ENVIRONMENT_CLUTTER,
    MULTI_INTERACTION,
    TARGET,
    ComponentDeclaration,
    RadarComponentIndex,
)
from .contracts import JOIN_MODES, JoinMode, RadarPathBatch, RadarPathTopology
from .direct import DirectComposer
from .two_way import TwoWayComposer

__all__ = [
    "COMPONENT_NAMES",
    "DIRECT_LEAKAGE",
    "ENVIRONMENT_CLUTTER",
    "JOIN_MODES",
    "MULTI_INTERACTION",
    "TARGET",
    "ComponentDeclaration",
    "DirectComposer",
    "JoinMode",
    "RadarComponentIndex",
    "RadarPathBatch",
    "RadarPathTopology",
    "TwoWayComposer",
]
