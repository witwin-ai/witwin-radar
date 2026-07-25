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
