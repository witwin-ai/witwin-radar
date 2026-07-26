"""Deprecated: the rank-2 CFAR detectors.

Re-exported from :mod:`witwin.radar.processing.adapters`. The detectors
themselves are :mod:`witwin.radar.processing.cfar`, which takes ``[..., D, R]``
with an arbitrary leading batch - so a ``[B, D, R]`` beam cube needs no Python
loop over beams - and returns a :class:`~witwin.radar.processing.cfar.Detections`
record rather than a bare tuple.

``ca_cfar_2d_fast``'s undocumented "~100x faster" docstring claim is gone,
replaced by a measured number in the Phase-8 profiling record.
"""

from __future__ import annotations

from ..processing.adapters import ca_cfar_2d, ca_cfar_2d_fast, os_cfar_2d

__all__ = ["ca_cfar_2d", "ca_cfar_2d_fast", "os_cfar_2d"]
