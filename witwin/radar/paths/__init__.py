"""Radar round-trip path composition.

Two composers, one result contract. :class:`TwoWayComposer` joins an inbound
and an outbound leg through a scatter site; :class:`DirectComposer` publishes a
single source-to-sink leg with no site at all. The mode is chosen explicitly by
the caller and recorded on the batch, so nothing downstream has to infer it and
there is no path by which one silently becomes the other.

The package root exports the contracts and the composers. It does not import
the Channel adapter; the composers duck-type the frozen leg handles they are
given, so this package never crosses the Channel boundary either.
"""

from .contracts import JOIN_MODES, JoinMode, RadarPathBatch, RadarPathTopology
from .direct import DirectComposer
from .two_way import TwoWayComposer

__all__ = [
    "JOIN_MODES",
    "DirectComposer",
    "JoinMode",
    "RadarPathBatch",
    "RadarPathTopology",
    "TwoWayComposer",
]
