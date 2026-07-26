"""Radar post-processing.

Everything in this package is PyTorch by owner directive: range profiles,
Range-Doppler maps, beam cubes, AoA, CFAR, point clouds and detection handoff
are post-processing, not simulation, and a native DSP kernel needs a measured
dispatch, layout, fusion or tape bottleneck plus its own decision record before
it can exist.

Processing CONSUMES synthesis results. It never mutates a path batch, never
changes composed row identity, and publishes no field that crosses back into
the Channel capability record, its public API, or its native binding manifest.

This package currently owns the component combination laws
(:func:`combine_incoherent` and the coherent law it documents alongside). The
range/Doppler/AoA/CFAR chain lands here as it is built.
"""

from .combination import combine_incoherent

__all__ = ["combine_incoherent"]
