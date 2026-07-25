"""Phase-4 adapter contract: Radar consuming ONLY the Channel consumer facade.

Provisional dependency note (R-ADR-008): Channel is consumed from a source
checkout, not a pinned release wheel. The pin is the recorded follow-up.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("witwin.channel")

from witwin.channel.propagation import consumer  # noqa: E402

from support import phase4_geometry as geo  # noqa: E402
from support import phase4_world as world  # noqa: E402
from witwin.radar.propagation import RadarEndpointSpec, require_endpoint_role  # noqa: E402
from witwin.radar.propagation.channel_consumer import ChannelPropagationAdapter  # noqa: E402


pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def compiled_scene():
    return world.compile_fixture_scene()


@pytest.fixture(scope="module")
def adapter(compiled_scene):
    return ChannelPropagationAdapter(
        compiled_scene,
        reference_frequency_hz=geo.REFERENCE_FREQUENCY_HZ,
        components=frozenset({"los"}),
        max_depth=0,
    )


def _tx():
    return world.endpoint_spec(
        geo.TX_POSITION_M, geo.TX_STABLE_ID, power_w=geo.TX_POWER_W
    )


def _site_sink():
    return world.endpoint_spec(geo.SITE_POSITION_M, geo.SITE_STABLE_ID)


def _site_source():
    return world.endpoint_spec(
        geo.SITE_POSITION_M, geo.SITE_STABLE_ID, power_w=geo.TX_POWER_W
    )


def _rx():
    return world.endpoint_spec(geo.RX_POSITION_M, geo.RX_STABLE_ID)


@pytest.fixture(scope="module")
def frozen_inbound(adapter):
    return adapter.freeze(_tx(), _site_sink())


def test_freeze_discovers_one_los_row_and_reports_prepare_cost(frozen_inbound):
    assert frozen_inbound.row_count == 1
    assert frozen_inbound.components == ("los",)
    # prepare_fixed_topology synchronizes; the cost is reported here, once per
    # frozen topology, and must never be paid inside a per-frame loop.
    assert frozen_inbound.prepare_d2h_copies == 3
    assert frozen_inbound.prepare_synchronizations == 3
    assert frozen_inbound.prepare_d2h_bytes > 0
    assert frozen_inbound.source_id.tolist() == [geo.TX_STABLE_ID]
    assert frozen_inbound.sink_id.tolist() == [geo.SITE_STABLE_ID]


def test_reevaluate_publishes_radar_leg_aliasing_consumer_storage(
    adapter, compiled_scene, frozen_inbound
):
    """delay_s and coefficient must ALIAS the consumer tensors, not copy them."""

    sources, sinks = _tx(), _site_sink()
    legs = adapter.reevaluate(frozen_inbound, sources, sinks, ad_mode="none")

    reference = consumer.reevaluate(
        compiled_scene,
        consumer.FixedTopologyRequest(
            sources=consumer.EndpointBatch(
                stable_ids=sources.stable_ids,
                positions_m=sources.positions_m,
                polarizations=sources.polarizations,
                powers_w=sources.powers_w,
            ),
            sinks=consumer.EndpointBatch(
                stable_ids=sinks.stable_ids,
                positions_m=sinks.positions_m,
                polarizations=sinks.polarizations,
            ),
            reference_frequency_hz=geo.REFERENCE_FREQUENCY_HZ,
            topology=frozen_inbound.prepared,
            response="scalar_transport",
            ad_mode="none",
        ),
    )
    # Same values, and within one call the adapter hands back the very objects
    # the consumer produced.
    torch.testing.assert_close(legs.delay_s, reference.paths.geometry.delay_s)
    assert legs.leg_count == reference.paths.path_count
    assert legs.pair_count == reference.paths.pair_count

    again = adapter.reevaluate(frozen_inbound, sources, sinks, ad_mode="none")
    assert again.delay_s.data_ptr() != 0
    assert again.delay_rate is None


def test_reevaluate_aliases_by_object_identity(adapter, frozen_inbound, monkeypatch):
    captured = {}
    original = consumer.reevaluate

    def spy(compiled, request):
        result = original(compiled, request)
        captured["result"] = result
        return result

    monkeypatch.setattr(consumer, "reevaluate", spy)
    legs = adapter.reevaluate(frozen_inbound, _tx(), _site_sink(), ad_mode="none")
    result = captured["result"]
    assert legs.delay_s is result.paths.geometry.delay_s
    assert legs.coefficient is result.paths.transport.coefficient
    assert legs.pair_index is result.paths.pair_index
    assert legs.diagnostics is result.diagnostics
    assert legs.row_valid is result.row_valid


def test_per_frame_budget_is_one_validation_copy_per_leg(adapter, frozen_inbound):
    legs = adapter.reevaluate(frozen_inbound, _tx(), _site_sink(), ad_mode="none")
    diagnostics = legs.diagnostics
    # The adapter adds nothing: no discovery, no compaction, no second count.
    assert diagnostics.discovery_launch_count == 0
    assert diagnostics.compact_count_d2h_copies == 0
    assert diagnostics.compact_sync_count == 0
    assert diagnostics.validation_d2h_copies == 1
    assert diagnostics.validation_d2h_bytes == 4
    assert diagnostics.validation_sync_count == 1


def test_freeze_is_never_called_per_frame(adapter, monkeypatch):
    calls = {"prepare": 0}
    original = consumer.prepare_fixed_topology

    def counting_prepare(topology):
        calls["prepare"] += 1
        return original(topology)

    monkeypatch.setattr(consumer, "prepare_fixed_topology", counting_prepare)

    inbound = adapter.freeze(_tx(), _site_sink())
    outbound = adapter.freeze(_site_source(), _rx())
    assert calls["prepare"] == 2

    for frame in range(5):
        offset = 0.01 * frame
        site = torch.tensor(
            [[geo.SITE_POSITION_M[0], geo.SITE_POSITION_M[1] + offset, 0.0]],
            dtype=torch.float32,
            device="cuda",
        )
        adapter.reevaluate(
            inbound,
            _tx(),
            world.endpoint_spec(site, geo.SITE_STABLE_ID),
            ad_mode="none",
        )
        adapter.reevaluate(
            outbound,
            world.endpoint_spec(site, geo.SITE_STABLE_ID, power_w=geo.TX_POWER_W),
            _rx(),
            ad_mode="none",
        )
    # Two frozen topologies, five frames, still two preparations.
    assert calls["prepare"] == 2


def test_row_valid_is_passed_through_untouched(adapter, frozen_inbound):
    legs = adapter.reevaluate(frozen_inbound, _tx(), _site_sink(), ad_mode="none")
    # A frozen line-of-sight row is replayed as pure free-space transport and is
    # never re-tested for visibility, so this route reports True by contract.
    # The dead-row semantics themselves are exercised in the composer and
    # synthesis tests, which inject the mask directly.
    assert legs.row_valid is not None
    assert bool(legs.row_valid.all())
    assert legs.row_valid.dtype == torch.bool


def test_reverse_mode_gradients_reach_both_endpoints(adapter, frozen_inbound):
    tx = torch.tensor([geo.TX_POSITION_M], dtype=torch.float32, device="cuda")
    site = torch.tensor([geo.SITE_POSITION_M], dtype=torch.float32, device="cuda")
    tx.requires_grad_(True)
    site.requires_grad_(True)
    legs = adapter.reevaluate(
        frozen_inbound,
        world.endpoint_spec(tx, geo.TX_STABLE_ID, power_w=geo.TX_POWER_W),
        world.endpoint_spec(site, geo.SITE_STABLE_ID),
        ad_mode="vjp",
    )
    legs.delay_s.sum().backward()
    assert tx.grad is not None and site.grad is not None
    # d tau / d site = +u, d tau / d tx = -u for the same unit vector.
    torch.testing.assert_close(site.grad, -tx.grad)
    assert float(site.grad.abs().sum()) > 0.0


def test_jvp_publishes_delay_rate_and_refuses_to_invent_one(adapter, frozen_inbound):
    import torch.autograd.forward_ad as forward_ad

    primal = torch.tensor([geo.SITE_POSITION_M], dtype=torch.float32, device="cuda")
    velocity = torch.tensor([[0.0, 12.0, 0.0]], dtype=torch.float32, device="cuda")
    with forward_ad.dual_level():
        dual = forward_ad.make_dual(primal, velocity)
        legs = adapter.reevaluate(
            frozen_inbound,
            _tx(),
            world.endpoint_spec(dual, geo.SITE_STABLE_ID),
            ad_mode="jvp",
        )
        assert legs.delay_rate is not None
        # Cloned inside the level: still valid after the level exits.
        rate_inside = legs.delay_rate.clone()
    torch.testing.assert_close(legs.delay_rate, rate_inside)

    d_in, _ = geo.leg_distances_m()
    direction = [
        (s - t) / d_in
        for s, t in zip(geo.SITE_POSITION_M, geo.TX_POSITION_M, strict=True)
    ]
    analytic = sum(u * v for u, v in zip(direction, (0.0, 12.0, 0.0), strict=True))
    analytic /= geo.C0_M_PER_S
    assert abs(float(legs.delay_rate[0]) - analytic) < abs(analytic) * 1e-5

    # Without a dual, jvp has no tangent and the adapter refuses rather than
    # publishing a zero rate that is indistinguishable from a static scene.
    with pytest.raises(RuntimeError, match="no delay_s tangent"):
        adapter.reevaluate(frozen_inbound, _tx(), _site_sink(), ad_mode="jvp")


def test_differentiable_power_is_rejected_before_any_native_work(
    adapter, frozen_inbound
):
    powers = torch.ones(1, dtype=torch.float32, device="cuda", requires_grad=True)
    sources = RadarEndpointSpec(
        stable_ids=torch.tensor([geo.TX_STABLE_ID], dtype=torch.int64, device="cuda"),
        positions_m=torch.tensor(
            [geo.TX_POSITION_M], dtype=torch.float32, device="cuda"
        ),
        polarizations=torch.tensor(
            [geo.POLARIZATION], dtype=torch.float32, device="cuda"
        ),
        powers_w=powers,
    )
    with pytest.raises(NotImplementedError, match="primal-only"):
        adapter.reevaluate(frozen_inbound, sources, _site_sink(), ad_mode="vjp")


def test_differentiable_polarization_is_rejected(adapter, frozen_inbound):
    polarizations = torch.tensor(
        [geo.POLARIZATION], dtype=torch.float32, device="cuda"
    ).requires_grad_(True)
    sinks = RadarEndpointSpec(
        stable_ids=torch.tensor([geo.SITE_STABLE_ID], dtype=torch.int64, device="cuda"),
        positions_m=torch.tensor(
            [geo.SITE_POSITION_M], dtype=torch.float32, device="cuda"
        ),
        polarizations=polarizations,
    )
    with pytest.raises(NotImplementedError, match="primal-only"):
        adapter.reevaluate(frozen_inbound, _tx(), sinks, ad_mode="vjp")


def test_frequency_mismatch_fails_before_native_compute(compiled_scene, frozen_inbound):
    mismatched = ChannelPropagationAdapter(
        compiled_scene,
        reference_frequency_hz=24.0e9,
        components=frozenset({"los"}),
        max_depth=0,
    )
    with pytest.raises((ValueError, NotImplementedError, RuntimeError)):
        mismatched.reevaluate(frozen_inbound, _tx(), _site_sink(), ad_mode="none")


def test_adapter_rejects_unfreezable_components(compiled_scene):
    with pytest.raises(NotImplementedError, match="cannot be frozen"):
        ChannelPropagationAdapter(
            compiled_scene,
            reference_frequency_hz=geo.REFERENCE_FREQUENCY_HZ,
            components=frozenset({"diffraction"}),
            max_depth=1,
        )


def test_adapter_rejects_unknown_ad_mode(adapter, frozen_inbound):
    with pytest.raises(NotImplementedError, match="unsupported ad_mode"):
        adapter.reevaluate(frozen_inbound, _tx(), _site_sink(), ad_mode="reverse")


def test_endpoint_role_contract_matches_channel(adapter, frozen_inbound):
    # A source without power and a sink with power are both rejected, and the
    # Radar-side message names the endpoint the caller actually passed.
    with pytest.raises(ValueError, match="source endpoint requires powers_w"):
        adapter.reevaluate(frozen_inbound, _site_sink(), _site_sink(), ad_mode="none")
    with pytest.raises(ValueError, match="sink endpoint must not carry powers_w"):
        adapter.reevaluate(frozen_inbound, _tx(), _site_source(), ad_mode="none")
    require_endpoint_role(_tx(), "source")
    require_endpoint_role(_site_sink(), "sink")
