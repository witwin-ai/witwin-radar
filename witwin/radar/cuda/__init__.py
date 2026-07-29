"""Lazy native CUDA boundary for the radar package.

Importing this package does not import the native runtime. Kernel facades bind
:func:`native_ops` and resolve the validated operator table only when they
actually execute native work.
"""

from __future__ import annotations

import importlib

_NATIVE_OPS = None


def native_ops():
    """Return the validated native operator table, loading it on first use."""

    global _NATIVE_OPS
    if _NATIVE_OPS is None:
        runtime = importlib.import_module("witwin.radar.cuda.runtime")
        _NATIVE_OPS = runtime.build_extension()
    return _NATIVE_OPS


def __getattr__(name: str):
    if name == "runtime":
        runtime = importlib.import_module("witwin.radar.cuda.runtime")
        globals()[name] = runtime
        return runtime
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["native_ops", "runtime"]
