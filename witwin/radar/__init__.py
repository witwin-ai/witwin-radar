"""Public Radar API.

The whole happy path is here, in the order a caller meets it: build a
:class:`Radar` out of its waveform, its antennas and its receive chain; say
what it is looking at with :class:`PointTargets`, :class:`StructureTargets` or
:class:`MeshTargets`, with an :class:`RcsTable` cross section or a
:class:`Fluctuation` where one applies; choose how often the world is
resampled with :class:`Motion`; then call one of the radar's four verbs.
Several radars that share one scene are a :class:`Network`, whose verbs
publish one result per radar. Products go to :mod:`witwin.radar.processing`.

Advanced records stay in their owner modules and are imported from there:
``SensorEndpointIds`` and ``StableIdAllocator`` in
:mod:`witwin.radar.binding`, ``RadarPathBatch`` in
:mod:`witwin.radar.paths`, the synthesis specs in
:mod:`witwin.radar.synthesis`, the scatter responses in
:mod:`witwin.radar.scattering`.
"""

from . import processing
from .frontend import Adc, Agc, Compression, Iq, Noise
from .motion import Motion
from .network import Network
from .radar import Chirp, Fmcw, Leakage, Ofdm, Pulsed, Radar
from .results import Frame, Result
from .sensors import Coupling, Pattern
from .simulation import Paths
from .targets import Aspect, Fluctuation, MeshTargets, PointTargets, RcsTable, ScatteringMatrix, StructureTargets

__all__ = [
    "Adc",
    "Agc",
    "Aspect",
    "Chirp",
    "Compression",
    "Coupling",
    "Fluctuation",
    "Fmcw",
    "Frame",
    "Iq",
    "Leakage",
    "MeshTargets",
    "Motion",
    "Network",
    "Noise",
    "Ofdm",
    "Paths",
    "Pattern",
    "PointTargets",
    "Pulsed",
    "Radar",
    "RcsTable",
    "Result",
    "ScatteringMatrix",
    "StructureTargets",
    "processing",
]
