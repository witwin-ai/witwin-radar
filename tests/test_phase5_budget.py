"""The multipath cardinality budget, measured rather than asserted from a plan.

R-ADR-006 fixes what a frame is allowed to cost in host observations. Phase 5
doubles the rows per leg and quadruples the composed rows, so the question is
whether any of that reached the host. It did not: a two-leg multipath frame
performs exactly the same two host observations as the Phase-4 line-of-sight
frame, one validation copy per leg, and the native join adds none.

Two things are measured here rather than trusted:

* the counters Channel REPORTS, through ``PropagationDiagnostics``;
* what actually happens, by counting every ``.item()``, ``.cpu()``,
  ``.tolist()``, ``.numpy()`` and CUDA synchronize call in the frame.

They agree. That agreement is the assertion; a reported budget nobody measures
is a comment. Note the second measurement can only see PYTHON-level
observations - a ``cudaStreamSynchronize`` inside a native reflection kernel is
invisible from here and is not reported by the consumer either. That gap is
recorded in R-ADR-006 as an upstream observation rather than asserted as a
number this test did not observe.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("witwin.channel")

from support import spike_driver as drv  # noqa: E402


pytestmark = pytest.mark.gpu

MULTIPATH = frozenset({"los", "reflection"})

HOST_OBSERVERS = ("item", "cpu", "tolist", "numpy")


class _Counter:
    """Count every Python-visible host observation while it is active."""

    def __init__(self, monkeypatch) -> None:
        self.counts = dict.fromkeys((*HOST_OBSERVERS, "synchronize"), 0)
        for name in HOST_OBSERVERS:
            original = getattr(torch.Tensor, name)

            def observing(self_, *args, _name=name, _original=original, **kwargs):
                self.counts[_name] += 1
                return _original(self_, *args, **kwargs)

            monkeypatch.setattr(torch.Tensor, name, observing)

        original_sync = torch.cuda.synchronize

        def counting_sync(*args, **kwargs):
            self.counts["synchronize"] += 1
            return original_sync(*args, **kwargs)

        monkeypatch.setattr(torch.cuda, "synchronize", counting_sync)

    def reset(self) -> None:
        for key in self.counts:
            self.counts[key] = 0


@pytest.fixture(scope="module")
def multipath():
    return drv.Phase4Spike(components=MULTIPATH, max_depth=1)


def test_freezing_a_multipath_leg_costs_four_copies_and_four_synchronizations(
    multipath,
):
    """One-time, outside every loop, and reported separately for that reason.

    Line of sight alone reports (3, 17, 3) per leg; adding reflection makes it
    (4, 33, 4). The difference is the reflection bucket's own preparation, paid
    once per frozen topology. Preparing per frame instead would give up the
    entire point of the capability, which is why these counters are published
    apart from the per-frame ones rather than summed into them.
    """

    for frozen in (multipath.inbound, multipath.outbound):
        assert frozen.prepare_d2h_copies == 4
        assert frozen.prepare_d2h_bytes == 33
        assert frozen.prepare_synchronizations == 4
        assert frozen.row_count == 2


def test_a_multipath_frame_costs_exactly_two_host_observations(
    multipath, monkeypatch
):
    tx, site, rx = drv.positions()
    response = drv.make_response()
    multipath.paths(tx, site, rx, response)  # resolve the operator table first

    counter = _Counter(monkeypatch)
    composed, inbound, outbound = multipath.paths(tx, site, rx, response)

    assert composed.path_count == 4
    # Two legs, one validation copy each. Nothing else crosses back.
    assert counter.counts["item"] == 2, counter.counts
    assert counter.counts["cpu"] == 0, counter.counts
    assert counter.counts["tolist"] == 0, counter.counts
    assert counter.counts["numpy"] == 0, counter.counts
    assert counter.counts["synchronize"] == 0, counter.counts

    # And the counters Channel reports agree with what was measured.
    copies = 0
    syncs = 0
    total_bytes = 0
    for legs in (inbound, outbound):
        diagnostics = legs.diagnostics
        assert diagnostics.compact_count_d2h_copies == 0
        assert diagnostics.discovery_launch_count == 0
        copies += diagnostics.validation_d2h_copies
        syncs += diagnostics.validation_sync_count
        total_bytes += diagnostics.validation_d2h_bytes
    assert copies == 2
    assert syncs == 2
    assert total_bytes == 8
    assert copies == counter.counts["item"]


def test_multipath_costs_no_more_per_frame_than_line_of_sight(monkeypatch):
    """Four composed rows instead of one, and the same host traffic.

    This is the property the whole fixed-topology contract exists to provide,
    so it is asserted as a comparison rather than as two independent numbers
    that happen to match.
    """

    tx, site, rx = drv.positions()
    response = drv.make_response()

    measured = {}
    for label, components, depth in (
        ("los", frozenset({"los"}), 0),
        ("multipath", MULTIPATH, 1),
    ):
        spike = drv.Phase4Spike(components=components, max_depth=depth)
        spike.paths(tx, site, rx, response)
        counter = _Counter(monkeypatch)
        composed, _, _ = spike.paths(tx, site, rx, response)
        measured[label] = (composed.path_count, dict(counter.counts))
        monkeypatch.undo()

    assert measured["los"][0] == 1
    assert measured["multipath"][0] == 4
    assert measured["los"][1] == measured["multipath"][1], measured


@pytest.fixture(scope="module")
def slot_spike():
    from support import multi_endpoint_driver

    return multi_endpoint_driver.MultiEndpointSpike()


def test_the_per_frame_host_budget_is_flat_in_slot_count(slot_spike, monkeypatch):
    """A whole frame of slow-time slots costs what one instant costs.

    This is the pin that forbids a Python per-slot loop. A loop publishes the
    same numbers, so nothing downstream can tell the difference; what it cannot
    hide is the budget, because it pays one validation copy and one
    synchronization PER SLOT. Two host observations at T = 256 and two at
    T = 1 is the whole statement.

    The replication of the frozen topology is warmed first on purpose: it is a
    function of the topology and the slot count alone, so it belongs to the
    freeze, not to the frame, and the adapter caches it there. It is measured
    separately below to make sure it is not secretly expensive.
    """

    from support import multi_endpoint_driver as multi

    measured = {}
    for slots in (1, 8, 64, 256):
        times = [index * 1.0e-5 for index in range(slots)]
        stack = multi.slot_site_stack(
            slot_spike.site_tensor(), (0.0, 1.0, 0.0), times
        )
        cold = _Counter(monkeypatch)
        slot_spike.slot_legs(stack, slot_count=slots)
        cold_counts = dict(cold.counts)
        monkeypatch.undo()

        counter = _Counter(monkeypatch)
        inbound, outbound = slot_spike.slot_legs(stack, slot_count=slots)
        measured[slots] = dict(counter.counts)
        monkeypatch.undo()

        assert inbound.slot_count == slots
        # Two legs, one validation copy each, whatever the slot count is.
        assert measured[slots]["item"] == 2, (slots, measured[slots])
        assert measured[slots]["cpu"] == 0, (slots, measured[slots])
        assert measured[slots]["tolist"] == 0, (slots, measured[slots])
        assert measured[slots]["numpy"] == 0, (slots, measured[slots])
        assert measured[slots]["synchronize"] == 0, (slots, measured[slots])
        # Building the block-diagonal replication is index arithmetic, so the
        # first frame at a new slot count costs no more than a later one.
        assert cold_counts == measured[slots], (slots, cold_counts)

        copies = 0
        syncs = 0
        for legs in (inbound, outbound):
            assert legs.diagnostics.compact_count_d2h_copies == 0
            assert legs.diagnostics.discovery_launch_count == 0
            copies += legs.diagnostics.validation_d2h_copies
            syncs += legs.diagnostics.validation_sync_count
        assert copies == 2, slots
        assert syncs == 2, slots

    assert len({tuple(sorted(value.items())) for value in measured.values()}) == 1


def test_the_native_join_adds_no_host_observation_of_its_own(
    multipath, monkeypatch
):
    """Isolated to the join, so the legs cannot mask a regression in it."""

    tx, site, rx = drv.positions()
    response = drv.make_response()
    _, inbound, outbound = multipath.paths(tx, site, rx, response)

    counter = _Counter(monkeypatch)
    composed = multipath.composer.compose(inbound, outbound, response)
    assert composed.path_count == 4
    assert all(value == 0 for value in counter.counts.values()), counter.counts


#: Peak device memory a whole ``T = 1024`` slot frame may allocate, in
#: mebibytes. The Phase-7 survey measured 46.6 MB on this fixture; the gate is
#: set at the plan's 64 MB so it is a budget rather than a fitted value.
PEAK_MEMORY_BUDGET_MB = 64.0

#: How far the per-slot cost may drift between ``T = 64`` and ``T = 1024``.
#: Replay is launch bound and flat in ``T``, so the per-slot cost should FALL
#: as the launch is amortised - the survey measured 0.0495 ms/slot at T = 64
#: against 0.0027 at T = 1024. A factor of two in the wrong direction is the
#: signature of work that grows with the slot count, which is the thing the
#: block-diagonal layout exists to prevent.
PER_SLOT_DRIFT = 2.0


def test_peak_memory_and_per_slot_cost_scale(slot_spike):
    """The two budgets a launch-bound replay must keep as ``T`` grows.

    Measured, not asserted from the plan: peak allocation is read from CUDA's
    own allocator across the whole batched replay, and the per-slot cost is
    wall time after a synchronize, with the first call at each slot count
    discarded so the topology replication and any allocator growth belong to
    the warm-up rather than to the measurement.

    A Python per-slot loop would pass the memory gate and fail this one, and a
    quadratic pair partition would fail both - which is why they are asserted
    together.
    """

    import time

    from support import multi_endpoint_driver as multi

    per_slot_ms = {}
    peak_mb = {}
    for slots in (64, 256, 1024):
        times = [index * 1.0e-5 for index in range(slots)]
        stack = multi.slot_site_stack(
            slot_spike.site_tensor(), (0.0, 1.0, 0.0), times
        )
        slot_spike.slot_legs(stack, slot_count=slots)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()

        start = time.perf_counter()
        inbound, outbound = slot_spike.slot_legs(stack, slot_count=slots)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        peak_mb[slots] = (torch.cuda.max_memory_allocated() - before) / (1024.0 ** 2)
        per_slot_ms[slots] = 1.0e3 * elapsed / slots
        assert inbound.slot_count == slots
        assert inbound.pair_count == slots * inbound.pairs_per_slot
        del inbound, outbound

    assert peak_mb[1024] < PEAK_MEMORY_BUDGET_MB, peak_mb
    assert per_slot_ms[1024] < PER_SLOT_DRIFT * per_slot_ms[64], per_slot_ms
