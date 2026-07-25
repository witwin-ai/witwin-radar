"""Radar round-trip path composition.

The package root exports the contracts and the composer. It does not import
the Channel adapter; the composer duck-types the frozen leg handles it is
given, so this package never crosses the Channel boundary either.
"""

from .contracts import RadarPathBatch, RadarPathTopology
from .two_way import TwoWayComposer

__all__ = ["RadarPathBatch", "RadarPathTopology", "TwoWayComposer"]
