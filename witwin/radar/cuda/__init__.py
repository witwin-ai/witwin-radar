"""Lazy native CUDA boundary for the radar package.

Importing this package does not import the native runtime. Kernel facades bind
:func:`native_ops` and resolve the validated operator table only when they
actually execute native work.
"""

from __future__ import annotations

import importlib


def native_ops():
    """Return the validated native operator table, loading it on first use.

    ``runtime.build_extension`` caches the loaded library for the process, so
    every call after the first is free.
    """

    return importlib.import_module("witwin.radar.cuda.runtime").build_extension()


__all__ = ["native_ops"]
