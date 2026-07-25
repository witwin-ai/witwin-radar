"""Radar target scattering responses."""

from .base import ScatterResponse
from .rcs import ScalarRcsResponse

__all__ = ["ScalarRcsResponse", "ScatterResponse"]
