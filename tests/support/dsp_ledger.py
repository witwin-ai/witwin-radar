"""Count vendor DSP launches and host observations while it is active.

This is ``tests/test_phase6_launch_budget.py``'s ``Ledger`` with ONE thing
changed: the patched namespace is ``torch.fft`` instead of the native operator
table. The ``HOST_OBSERVERS`` set is the same four tensor methods plus
``torch.cuda.synchronize``, so a processing budget and a synthesis budget are
counted by the same mechanism and can be added together.

**The honest caveat, restated rather than dropped.** A synchronization inside a
native kernel is invisible from Python. ``torch.fft.fft`` can synchronize inside
cuFFT plan creation and nothing here will see it. This ledger counts DISPATCHES
and HOST-VISIBLE observations; it does not measure time and must never be used
to infer any. Wall time is measured with CUDA events, in
``tools/benchmark_processing.py``.

Two entry shapes, because two callers need it:

* ``DspLedger(monkeypatch, ...)`` inside a test, where pytest restores;
* ``with DspLedger() as ledger:`` inside the benchmark tool, which has no
  ``monkeypatch`` fixture and restores the originals itself.

Both count identically. The self-restoring form exists so the tool and the
budget test report the same integers rather than two counters that can drift.
"""

from __future__ import annotations

import torch


#: The same four tensor methods the Phase-6 ledger watches. Each moves a device
#: value to the host, which is a synchronization whether or not it is written as
#: one.
HOST_OBSERVERS = ("item", "cpu", "tolist", "numpy")

#: Every ``torch.fft`` entry the processing facade is allowed to call. The
#: frozen-surface test asserts this tuple against the source; here it is the set
#: that gets wrapped, so a call to an unlisted transform would go UNCOUNTED and
#: the frozen-surface test is what catches it.
DSP_OPERATORS = (
    "fft",
    "ifft",
    "fft2",
    "fftshift",
    "ifftshift",
    "fftfreq",
)

#: Calls whose OUTPUT SHAPE depends on device data. Each one must read a count
#: back to the host before it can allocate, so each is a synchronization that
#: ``torch.cuda.synchronize`` never appears next to and that none of the four
#: tensor methods above catches.
#:
#: They are counted separately, and this is the concrete case the caveat in the
#: module docstring is about: a Python-level counter sees only what Python
#: names. ``point_cloud`` performs exactly one of these, and it IS the stage - a
#: point cloud has a data-dependent length - so it is attributed rather than
#: hidden inside a pipeline total.
IMPLICIT_SYNCHRONIZERS = ("argwhere", "nonzero")


class DspLedger:
    """Count ``torch.fft`` dispatches and host observations while active."""

    def __init__(self, monkeypatch=None, operators=DSP_OPERATORS) -> None:
        self.launches = dict.fromkeys(operators, 0)
        self.host = dict.fromkeys(
            (*HOST_OBSERVERS, "synchronize", *IMPLICIT_SYNCHRONIZERS), 0
        )
        self._monkeypatch = monkeypatch
        self._restore: list[tuple[object, str, object]] = []
        if monkeypatch is not None:
            self._install()

    # -- installation ------------------------------------------------------

    def _set(self, target, name, value) -> None:
        if self._monkeypatch is None:
            self._restore.append((target, name, getattr(target, name)))
            setattr(target, name, value)
        else:
            self._monkeypatch.setattr(target, name, value)

    def _install(self) -> None:
        for name in self.launches:
            original = getattr(torch.fft, name)

            def counting(*args, _name=name, _original=original, **kwargs):
                self.launches[_name] += 1
                return _original(*args, **kwargs)

            self._set(torch.fft, name, counting)
        for name in HOST_OBSERVERS:
            original_method = getattr(torch.Tensor, name)

            def observing(
                tensor, *args, _name=name, _original=original_method, **kwargs
            ):
                self.host[_name] += 1
                return _original(tensor, *args, **kwargs)

            self._set(torch.Tensor, name, observing)
        for name in IMPLICIT_SYNCHRONIZERS:
            original_call = getattr(torch, name)

            def implicit(*args, _name=name, _original=original_call, **kwargs):
                self.host[_name] += 1
                return _original(*args, **kwargs)

            self._set(torch, name, implicit)
        original_sync = torch.cuda.synchronize

        def counting_sync(*args, **kwargs):
            self.host["synchronize"] += 1
            return original_sync(*args, **kwargs)

        self._set(torch.cuda, "synchronize", counting_sync)

    # -- context-manager form ---------------------------------------------

    def __enter__(self) -> "DspLedger":
        if self._monkeypatch is None and not self._restore:
            self._install()
        return self

    def __exit__(self, *exc) -> None:
        for target, name, value in reversed(self._restore):
            setattr(target, name, value)
        self._restore.clear()
        return None

    # -- reporting ---------------------------------------------------------

    @property
    def transform_count(self) -> int:
        """Total ``torch.fft`` dispatches, across every entry."""

        return sum(self.launches.values())

    @property
    def host_observation_count(self) -> int:
        """Total host observations, including explicit synchronizations."""

        return sum(self.host.values())

    def live(self) -> dict[str, int]:
        """Only the non-zero counters, for a readable report line."""

        return {
            name: value
            for name, value in (*self.launches.items(), *self.host.items())
            if value
        }


__all__ = [
    "DSP_OPERATORS",
    "HOST_OBSERVERS",
    "IMPLICIT_SYNCHRONIZERS",
    "DspLedger",
]
