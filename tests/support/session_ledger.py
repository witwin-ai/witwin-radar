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
  above.
* ``trips``: Python-level per-observation work: world advances
  (``SceneEpochLoop.frame``) and composed batches (``TwoWayComposer.compose``).

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
            monkeypatch.setattr(cls, attribute, self._counting(self.trips, label, getattr(cls, attribute)))

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


def profiled_copies(call) -> dict[str, int]:
    """D2H and H2D copies and CUDA kernels one call issues, from the profiler."""

    from torch.profiler import ProfilerActivity, profile

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
    return counts


__all__ = ["CHANNEL_ENTRIES", "SessionLedger", "TRIP_HOOKS", "profiled_copies"]
