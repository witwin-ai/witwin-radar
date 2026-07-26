"""Direct and multipath are two explicit composers publishing one contract.

The acceptance property is that the choice is made once, by the caller, and
RECORDED - never inferred downstream and never silently substituted. So both
composers publish ``RadarPathBatch``, ``synthesize_fmcw_beat`` takes either
without a branch, and the batch says which one it came from.

Scope note, because the words collide: "direct mode" is the direct TX-to-RX
path evaluated through the Channel consumer on the same frozen-topology
contract as every other leg. It is not a Radar-owned native direct-path
evaluator, and nothing here short-cuts that separate work.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("witwin.channel")

from witwin.radar.paths import JOIN_MODES, DirectComposer  # noqa: E402
from witwin.radar.paths.direct import NO_OUTBOUND_ROW, NO_SITE  # noqa: E402
from witwin.radar.synthesis.fmcw_beat import synthesize_fmcw_beat  # noqa: E402

from support import phase4_geometry as geo  # noqa: E402
from support import spike_driver as drv  # noqa: E402


pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def direct():
    return drv.DirectSpike()


@pytest.fixture(scope="module")
def multipath():
    return drv.Phase4Spike()


def test_the_direct_path_delay_is_the_tx_to_rx_distance(direct):
    tx, _, rx = drv.positions()
    composed, leg = direct.paths(tx, rx)

    assert composed.path_count == 1
    assert composed.join_mode == "direct"
    distance = (
        sum(
            (a - b) ** 2
            for a, b in zip(geo.TX_POSITION_M, geo.RX_POSITION_M, strict=True)
        )
        ** 0.5
    )
    assert float(composed.total_delay_s[0]) == pytest.approx(
        distance / geo.C0_M_PER_S, rel=1e-6
    )
    # A gather, not a computation: the leg's transport IS the direct transfer.
    assert torch.equal(composed.complex_transfer_ref, leg.coefficient)


def test_a_direct_row_says_it_has_no_site_and_no_second_leg(direct):
    tx, _, rx = drv.positions()
    composed, _ = direct.paths(tx, rx)
    topology = composed.topology
    assert topology.site_id.tolist() == [NO_SITE]
    assert topology.outbound_row.tolist() == [NO_OUTBOUND_ROW]
    assert topology.radar_source_id.tolist() == [geo.TX_STABLE_ID]
    assert topology.radar_sink_id.tolist() == [geo.RX_STABLE_ID]


def test_both_modes_publish_the_same_contract_to_the_same_synthesis(
    direct, multipath
):
    """One result type, one synthesis entry, no branch anywhere downstream."""

    spec = drv.make_spec(num_chirps=2)
    tx, site, rx = drv.positions()
    direct_batch, _ = direct.paths(tx, rx)
    multipath_batch, _, _ = multipath.paths(tx, site, rx, drv.make_response())

    assert type(direct_batch) is type(multipath_batch)
    assert {direct_batch.join_mode, multipath_batch.join_mode} == JOIN_MODES
    for batch in (direct_batch, multipath_batch):
        assert batch.sensor_pair_count == 1
        assert batch.pair_offsets.tolist() == [0, 1]
        assert batch.reference_frequency_hz == geo.REFERENCE_FREQUENCY_HZ
        iq = synthesize_fmcw_beat(drv.to_synthesis(batch), spec)
        assert iq.shape == (spec.num_chirps, 1, spec.num_samples)
        assert torch.isfinite(iq.real).all() and torch.isfinite(iq.imag).all()

    # And they are genuinely different paths: the direct one is far shorter.
    assert float(direct_batch.total_delay_s[0]) < float(
        multipath_batch.total_delay_s[0]
    )


def test_the_direct_gradient_reaches_both_endpoints(direct):
    """A pass-through composer must still be a pass-through for the tape."""

    tx = torch.tensor(
        [geo.TX_POSITION_M], dtype=torch.float32, device="cuda", requires_grad=True
    )
    rx = torch.tensor(
        [geo.RX_POSITION_M], dtype=torch.float32, device="cuda", requires_grad=True
    )
    composed, _ = direct.paths(tx, rx, ad_mode="vjp")
    (composed.total_delay_s.sum() + composed.complex_transfer_ref.real.sum()).backward()
    for endpoint in (tx, rx):
        assert endpoint.grad is not None
        assert float(endpoint.grad.abs().sum()) > 0.0
    # The endpoints move oppositely along the baseline: translating both changes
    # nothing, so the two gradients cancel.
    torch.testing.assert_close(tx.grad, -rx.grad, rtol=1e-4, atol=1e-9)


def test_an_unknown_join_mode_is_refused_by_the_contract():
    """The mode is validated, not merely stored."""

    from dataclasses import replace

    direct_spike = drv.DirectSpike()
    tx, _, rx = drv.positions()
    composed, _ = direct_spike.paths(tx, rx)
    with pytest.raises(ValueError, match="join_mode must be one of"):
        replace(composed, join_mode="hybrid")


def test_a_leg_endpoint_outside_the_declared_front_end_is_refused(direct):
    with pytest.raises(ValueError, match="not in radar_sink_ids"):
        DirectComposer.freeze(
            direct.leg,
            radar_source_ids=[geo.TX_STABLE_ID],
            radar_sink_ids=[geo.SITE_STABLE_ID],
            reference_frequency_hz=geo.REFERENCE_FREQUENCY_HZ,
        )
