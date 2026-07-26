"""Radar target scattering responses."""

from .base import ScatterResponse
from .rcs import RCS_AMPLITUDE_LAW, ScalarRcsResponse, rcs_amplitude

__all__ = [
    "RCS_AMPLITUDE_LAW",
    "ScalarRcsResponse",
    "ScatterResponse",
    "rcs_amplitude",
]
