"""The one test-side mapping from a composed batch to a synthesis batch.

``synthesize_fmcw`` consumes :class:`SynthesisPathBatch`, not
:class:`RadarPathBatch`: the difference between them is the weight's provenance,
and it is what lets ``require_compatible`` refuse a spec that would count the
carrier, the spreading, or the Doppler twice.

``slow_time_mode`` has no default in production for exactly that reason - only
the caller knows whether it froze the weight for the frame or refreshes it per
slot. Every Phase-4/5/6 fixture evaluates the frozen topology ONCE per frame, so
the answer here is always the frozen mode, and writing it in one place keeps a
test from quietly answering it differently.
"""

from __future__ import annotations


def to_synthesis(composed):
    """Wrap a composed round-trip batch as a frozen-weight synthesis batch."""

    from witwin.radar.synthesis import SlowTimeMode, SynthesisPathBatch

    return SynthesisPathBatch.from_radar_paths(
        composed, slow_time_mode=SlowTimeMode.FROZEN_WEIGHT_WITH_CARRIER_RATE
    )


__all__ = ["to_synthesis"]
