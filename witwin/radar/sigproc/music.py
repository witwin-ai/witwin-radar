"""Deprecated: the MUSIC imager.

Re-exported from :mod:`witwin.radar.processing.adapters`. The estimator is
:func:`witwin.radar.processing.aoa.music_spectrum`, which reads the element
spacing off an ``ArrayGeometry`` instead of the literal ``spacing = 0.5`` this
class had, builds its spatial smoothing with ``unfold`` instead of an
``(L + 1) ** 2``-way ``torch.stack`` over a list comprehension, and uses no
``numpy`` angle grids.
"""

from __future__ import annotations

from ..processing.adapters import MUSICImager

__all__ = ["MUSICImager"]
