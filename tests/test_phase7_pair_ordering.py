"""The freeze-time pair-ordering gate, wired into production (Phase-6 gap 5).

``synthesis.assembly.validate_pair_ordering`` has existed since Phase 6 with no
production caller at all. The frame path DEPENDS on the sink-major pair rank it
checks - ``pair_tx_index`` reads ``pair % num_tx`` from it and
``assemble_frame_cube`` splits the pair axis with it - so until now that
assertion was empty everywhere except in its own unit test.

The gate belongs at freeze time and nowhere else: it reads its input on the
host, which is free once per topology and would be a per-frame device-to-host
copy anywhere in the frame loop.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("witwin.channel")

from support import multi_endpoint_driver as drv  # noqa: E402
from support import multi_endpoint_geometry as geo  # noqa: E402

pytestmark = pytest.mark.gpu


def test_validate_pair_ordering_runs_in_production(monkeypatch):
    """The plan's Phase-6 gap 5: the layout assertion was empty in production.

    Two halves. First, the freeze path really calls the check, with the
    composer's OWN pair index and the front end it declared - measured by
    wrapping the function, not by reading the source. Second, when the pair
    rank owner produces something the frame path cannot consume, the freeze
    call refuses it by name instead of publishing a composer whose pair axis is
    not a partition.
    """

    import witwin.radar.paths as two_way

    spike = drv.MultiEndpointSpike()
    seen = []
    original = two_way.validate_pair_ordering

    def spying(index, **kwargs):
        seen.append((index, dict(kwargs)))
        return original(index, **kwargs)

    monkeypatch.setattr(two_way, "validate_pair_ordering", spying)

    from witwin.radar.paths import TwoWayComposer

    composer = TwoWayComposer.freeze(
        spike.inbound,
        spike.outbound,
        torch.tensor(list(spike.site_ids), dtype=torch.int64, device="cuda"),
        radar_source_ids=list(spike.transmitter_ids),
        radar_sink_ids=list(spike.receiver_ids),
        reference_frequency_hz=geo.REFERENCE_FREQUENCY_HZ,
    )
    assert len(seen) == 1, seen
    index, kwargs = seen[0]
    assert torch.equal(index, composer.sensor_pair_index)
    assert kwargs == {
        "num_tx": len(spike.transmitter_ids),
        "num_rx": len(spike.receiver_ids),
        "sensor_pair_count": composer.sensor_pair_count,
    }

    # A broken pair-rank owner: every row claims a pair one past the end of the
    # declared array. Nothing downstream would catch it - the offsets table
    # would clamp in the kernel and the cube would still have a plausible shape.
    def escaping(sources, sinks):
        beyond = len(sources) * len(sinks)

        def rank(source: int, sink: int) -> int:
            return beyond

        return rank

    monkeypatch.setattr(two_way, "sink_major_rank", escaping)
    with pytest.raises(ValueError, match="outside the 2 x 2 array's range"):
        TwoWayComposer.freeze(
            spike.inbound,
            spike.outbound,
            torch.tensor(list(spike.site_ids), dtype=torch.int64, device="cuda"),
            radar_source_ids=list(spike.transmitter_ids),
            radar_sink_ids=list(spike.receiver_ids),
            reference_frequency_hz=geo.REFERENCE_FREQUENCY_HZ,
        )


def test_the_direct_composer_is_held_to_the_same_layout(monkeypatch):
    """A direct batch feeds the same cube assembly and the same gate."""

    import witwin.radar.paths as two_way
    from witwin.radar.paths import DirectComposer

    spike = drv.MultiEndpointSpike()
    seen = []
    original = two_way.validate_pair_ordering

    def spying(index, **kwargs):
        seen.append(index)
        return original(index, **kwargs)

    monkeypatch.setattr(two_way, "validate_pair_ordering", spying)
    composer = DirectComposer.freeze(
        spike.inbound,
        radar_source_ids=list(spike.transmitter_ids),
        radar_sink_ids=list(spike.site_ids),
        reference_frequency_hz=geo.REFERENCE_FREQUENCY_HZ,
    )
    assert len(seen) == 1
    assert torch.equal(seen[0], composer.sensor_pair_index)
