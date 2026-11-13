"""Public Radar API.

The whole happy path is here, in the order a caller meets it: build a
:class:`Radar` out of its waveform, its antennas and its receive chain; say
what it is looking at with :class:`PointTargets`, :class:`StructureTargets` or
:class:`MeshTargets`; choose how often the world is resampled with
:class:`Motion`; then call one of the radar's four verbs. Products go to :mod:`witwin.radar.processing`.

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
from .radar import Chirp, Fmcw, Leakage, Ofdm, Pulsed, Radar
from .results import Frame, Result
from .sensors import Coupling, Pattern
from .simulation import Paths
from .targets import Aspect, MeshTargets, PointTargets, ScatteringMatrix, StructureTargets

__all__ = [
    "Adc",
    "Agc",
    "Aspect",
    "Chirp",
    "Compression",
    "Coupling",
    "Fmcw",
    "Frame",
    "Iq",
    "Leakage",
    "MeshTargets",
    "Motion",
    "Noise",
    "Ofdm",
    "Paths",
    "Pattern",
    "PointTargets",
    "Pulsed",
    "Radar",
    "Result",
    "ScatteringMatrix",
    "StructureTargets",
    "processing",
]
