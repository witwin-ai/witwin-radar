"""Native CUDA runtime for the radar package."""

import importlib

runtime = importlib.import_module("witwin.radar.cuda.runtime")

__all__ = ["runtime"]
