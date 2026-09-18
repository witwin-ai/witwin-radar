"""Public Radar API.

The whole happy path is here, in the order a caller meets it: build a
:class:`Radar` out of its waveform, its antennas and its receive chain; say
what it is looking at with :class:`PointTargets` or :class:`StructureTargets`;
choose how often the world is resampled with :class:`Motion`; then call one of
the radar's four verbs. Products go to :mod:`witwin.radar.processing`.

Advanced records stay in their owner modules and are imported from there:
``SensorEndpointIds`` and ``StableIdAllocator`` in
:mod:`witwin.radar.simulation`, ``RadarPathBatch`` in
:mod:`witwin.radar.paths`, the synthesis specs in
:mod:`witwin.radar.synthesis`, the scatter responses in
:mod:`witwin.radar.scattering`.
"""

from . import processing
from .frontend import Adc, Agc, Iq, Noise
from .radar import Fmcw, Leakage, Ofdm, Pulsed, Radar
from .sensors import Pattern
from .simulation import Frame, Motion, Paths, Result
from .targets import Aspect, PointTargets, StructureTargets

__all__ = [
    "Adc",
    "Agc",
    "Aspect",
    "Fmcw",
    "Frame",
    "Iq",
    "Leakage",
    "Motion",
    "Noise",
    "Ofdm",
    "Paths",
    "Pattern",
    "PointTargets",
    "Pulsed",
    "Radar",
    "Result",
    "StructureTargets",
    "processing",
]
