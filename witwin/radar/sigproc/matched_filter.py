"""Deprecated: pulse compression.

Moved to :mod:`witwin.radar.processing.matched_filter` at the Phase-8 cutover,
so that every production ``torch.fft`` expression in the radar processing chain
lives under ``witwin/radar/processing/``. The correlation itself now lives once,
in :mod:`witwin.radar.processing.primitives`, shared with the pulsed
range-profile backend.

One behaviour change, deliberate and named: ``matched_filter`` no longer upcasts
its input to ``complex128`` unconditionally. The working precision is the
``dtype`` argument, defaulting to the input's own; pass ``torch.complex128`` to
reproduce the pre-cutover result exactly.
"""

from __future__ import annotations

from ..processing.matched_filter import lag_axis, matched_filter, pulse_samples

__all__ = ["lag_axis", "matched_filter", "pulse_samples"]
