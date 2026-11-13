"""Count what one scene session costs: Channel calls, native launches, host traffic.

The session budget is a statement about orchestration, so it counts the things
orchestration decides and leaves the kernels' own cost to the per-family
budgets:

* ``channel``: calls into the Channel consumer facade, by entry point, wrapped
  on the module object :mod:`witwin.radar.channel` imported. ``evaluate`` is a
  discovery, ``prepare_fixed_topology`` its partition, ``reevaluate`` one
  replay, ``replicate_over_slots`` a slot table and ``rediscovery_required``
  the free host poll.
* ``native``: Radar native operator launches, by symbol, wrapped on the one
  validated operator table.
* ``synchronizations``: host synchronizations Torch reports through
  ``torch.cuda.set_sync_debug_mode("warn")``. A synchronization inside a
  native library that never returns to Torch is invisible to it; Channel's and
  Radar's own Python-visible ones are not.
* ``d2h`` / ``h2d`` / ``kernels``: device copies and every CUDA kernel, read
  from a separate profiled pass so the profiler cannot perturb the counts
  above. These three depend on what the PROCESS ran before, not only on the
  call: process-global caches (Channel's compile registry rebinds a new
  source's tables to an earlier source with the same geometry, and other
  first-use and address-keyed caches decide some uploads and launches) let
  earlier tests remove a few copies and kernels, so the same budget measured
  inside a long pytest run moved by up to 3 % with the test order. A budget
  that pins them measures in a fresh interpreter (:func:`measure_isolated`)
  and reads the largest of several profiled passes (:func:`profiled_copies`),
  where they are identical from run to run and process to process.
* ``trips``: Python-level per-observation work: world advances
  (``SceneEpochLoop.frame``), composed batches (``TwoWayComposer.compose`` and ``compose_slots``)
  and join builds (``TwoWayComposer.freeze``).

Every count is an exact integer for a fixed fixture. Wall time is not measured
here and must not be inferred from any of these.
"""

from __future__ import annotations

import warnings
from collections import Counter

import torch

#: Channel consumer entry points the adapter reaches.
CHANNEL_ENTRIES = ("evaluate", "prepare_fixed_topology", "reevaluate", "replicate_over_slots", "rediscovery_required")

#: Python-level per-observation trips, as ``(owner, attribute)`` import paths.
TRIP_HOOKS = (
    ("witwin.radar.propagation", "SceneEpochLoop", "frame"),
    ("witwin.radar.paths", "TwoWayComposer", "compose"),
    ("witwin.radar.paths", "TwoWayComposer", "compose_slots"),
    ("witwin.radar.paths", "TwoWayComposer", "freeze"),
)

SYNC_MESSAGE = "called a synchronizing CUDA operation"


class SessionLedger:
    """Install counting wrappers with ``monkeypatch``; read the counters after."""

    def __init__(self, monkeypatch) -> None:
        import importlib

        from witwin.radar import channel as radar_channel
        from witwin.radar.cuda import runtime

        self.channel: Counter[str] = Counter()
        self.native: Counter[str] = Counter()
        self.trips: Counter[str] = Counter()
        self.synchronizations = 0
        consumer = radar_channel.consumer
        for name in CHANNEL_ENTRIES:
            monkeypatch.setattr(consumer, name, self._counting(self.channel, name, getattr(consumer, name)))
        table = runtime.build_extension()
        for name in table.build_info()["native_build"]["operator_symbols"]:
            monkeypatch.setattr(table, name, self._counting(self.native, name, getattr(table, name)), raising=False)
        for module_name, owner, attribute in TRIP_HOOKS:
            cls = getattr(importlib.import_module(module_name), owner)
            label = f"{owner}.{attribute}"
            raw = cls.__dict__[attribute]
            if isinstance(raw, classmethod):
                wrapped: object = classmethod(self._counting(self.trips, label, raw.__func__))
            else:
                wrapped = self._counting(self.trips, label, raw)
            monkeypatch.setattr(cls, attribute, wrapped)

    @staticmethod
    def _counting(counter: Counter, name: str, original):
        def counting(*args, **kwargs):
            counter[name] += 1
            return original(*args, **kwargs)

        return counting

    def run(self, call):
        """Run ``call`` with Torch's synchronization reporting on; return its value."""

        torch.cuda.synchronize()
        previous = torch.cuda.get_sync_debug_mode()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            torch.cuda.set_sync_debug_mode("warn")
            try:
                value = call()
            finally:
                torch.cuda.set_sync_debug_mode(previous)
        self.synchronizations += sum(SYNC_MESSAGE in str(item.message) for item in caught)
        return value


def measure_isolated(module: str, function: str, *args: object, timeout_s: float = 1200.0) -> object:
    """``module.function(*args)`` in a fresh interpreter, returned through JSON.

    ``module`` is imported from ``tests/`` with the worktree first on the path,
    exactly as the suite imports it, and ``function`` must return JSON data. The
    parent holds its GPU slot while the child runs, so the two never compete.
    """

    import json
    import subprocess
    import sys
    from pathlib import Path

    tests = Path(__file__).resolve().parents[1]
    code = (
        "import json, sys\n"
        f"sys.path[:0] = [{str(tests)!r}, {str(tests.parent)!r}]\n"
        f"from {module} import {function} as run\n"
        "value = run(*json.loads(sys.argv[1]))\n"
        "print('MEASURED ' + json.dumps(value))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code, json.dumps(list(args))],
        cwd=tests.parent,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    lines = [line for line in completed.stdout.splitlines() if line.startswith("MEASURED ")]
    if completed.returncode != 0 or len(lines) != 1:
        raise RuntimeError(
            f"{module}.{function}{args} failed in its subprocess (exit {completed.returncode}):\n"
            f"{completed.stdout[-4000:]}\n{completed.stderr[-4000:]}"
        )
    return json.loads(lines[0][len("MEASURED ") :])


def profiled_copies(call, *, passes: int = 3) -> dict[str, int]:
    """D2H and H2D copies and CUDA kernels one call issues, from the profiler.

    The call runs ``passes`` times and each count is the LARGEST pass. A warm
    call does identical work every time, but the profiler's activity buffers
    drop records when a pass is long or the device is contended (seen as one
    missing upload, or a few dozen missing kernels, in one pass of an
    otherwise identical run); a dropped record only ever lowers a count, so the
    largest pass is the complete one. The caller warms the call first.
    """

    from torch.profiler import ProfilerActivity, profile

    best = {"d2h": 0, "h2d": 0, "kernels": 0}
    for _ in range(passes):
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            call()
            torch.cuda.synchronize()
        counts = {"d2h": 0, "h2d": 0, "kernels": 0}
        for event in prof.events():
            name = event.name
            if "Memcpy DtoH" in name:
                counts["d2h"] += 1
            elif "Memcpy HtoD" in name:
                counts["h2d"] += 1
            elif "Memcpy" not in name and "Memset" not in name and event.device_type == torch.autograd.DeviceType.CUDA:
                counts["kernels"] += 1
        best = {name: max(best[name], counts[name]) for name in best}
    return best


__all__ = ["CHANNEL_ENTRIES", "SessionLedger", "TRIP_HOOKS", "measure_isolated", "profiled_copies"]
