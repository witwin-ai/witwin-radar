"""Public Radar system API.

The package root owns only the configured radar system and its configuration.
Scene binding, simulation results, propagation records, processing products,
deployment reporting, geometry and waveform APIs are imported from their
concept-owner modules.
"""

from .radar import Radar, RadarConfig


__all__ = ["Radar", "RadarConfig"]
