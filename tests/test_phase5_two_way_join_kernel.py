"""The native two-way join, checked against the retained Torch composition.

``two_way_join.cu`` replaced a Torch composition that issued roughly 17-19
device-side aten ops per frame and measured a flat 0.2-0.6 ms from K = 4 to
K = 24000: launch bound, not bandwidth bound, so one fused launch was the whole
win. The Torch version was not deleted; it lives in
``tests/reference/two_way_torch.py`` and is the lockstep oracle here. Same
arithmetic, same association, so a disagreement is a kernel bug rather than a
precision artefact.

The kernel accumulates in double and rounds once, while the Torch chain rounds
at every complex operation, so the two agree to a few float32 ULPs rather than
bit for bit. The tolerances below say so explicitly instead of using defaults
that would pass for almost any answer.
"""

from __future__ import annotations

import pytest
import torch

from witwin.radar.paths import TwoWayComposer

from reference.two_way_torch import PerSiteResponse, join_reference  # noqa: E402
from support import join_fixture as fx  # noqa: E402


pytestmark = pytest.mark.gpu

SOURCES = [10, 11]
SINKS = [30, 31]
SITES = [20, 21]
COMPONENTS = [0, 1]
REFERENCE_FREQUENCY_HZ = 77.0e9


def _frozen_pair(device: str = "cuda"):
    inbound = fx.frozen_leg(fx.leg_rows(SOURCES, SITES, COMPONENTS), device=device)
    outbound = fx.frozen_leg(fx.leg_rows(SITES, SINKS, COMPONENTS), device=device)
    return inbound, outbound


def _composer(device: str = "cuda") -> TwoWayComposer:
    inbound, outbound = _frozen_pair(device)
    return TwoWayComposer.freeze(
        inbound,
        outbound,
        torch.tensor(SITES, dtype=torch.int64, device=device),
        radar_source_ids=SOURCES,
        radar_sink_ids=SINKS,
        reference_frequency_hz=REFERENCE_FREQUENCY_HZ,
    )


def _frame(composer, *, valid=None, rates=True, seed=(11, 12)):
    rows_in = composer.inbound_row_count
    rows_out = composer.outbound_row_count
    tau_in, rate_in, c_in = fx.payload(rows_in, seed=seed[0])
    tau_out, rate_out, c_out = fx.payload(rows_out, seed=seed[1])
    valid_in, valid_out = valid if valid is not None else (None, None)
    inbound = fx.leg_batch(
        tau_in.float(),
        c_in.to(torch.complex64),
        rate=rate_in.float() if rates else None,
        row_valid=valid_in,
    )
    outbound = fx.leg_batch(
        tau_out.float(),
        c_out.to(torch.complex64),
        rate=rate_out.float() if rates else None,
        row_valid=valid_out,
    )
    _, _, site_value = fx.payload(composer.site_count, seed=99)
    response = PerSiteResponse(site_value.to(torch.complex64))
    return inbound, outbound, response


def _reference(composer, inbound, outbound, response, row_valid):
    return join_reference(
        tau_in=inbound.delay_s,
        tau_out=outbound.delay_s,
        rate_in=(
            torch.zeros_like(inbound.delay_s)
            if inbound.delay_rate is None
            else inbound.delay_rate
        ),
        rate_out=(
            torch.zeros_like(outbound.delay_s)
            if outbound.delay_rate is None
            else outbound.delay_rate
        ),
        c_in=inbound.coefficient,
        c_out=outbound.coefficient,
        response=response.evaluate(composer.site_count, inbound.delay_s.device),
        idx_in=composer.inbound_row,
        idx_out=composer.outbound_row,
        idx_s=composer.response_slot,
        row_valid=row_valid,
    )


def test_the_join_shapes_match_the_declared_front_end():
    composer = _composer()
    assert composer.sensor_pair_count == len(SOURCES) * len(SINKS)
    assert composer.site_count == len(SITES)
    # Per sensor pair and site: 2 inbound components x 2 outbound components.
    assert composer.path_count == 4 * 2 * 4
    offsets = composer.pair_offsets.tolist()
    assert offsets == [0, 8, 16, 24, 32]
    assert composer.inbound_row_count == 8
    assert composer.outbound_row_count == 8


def test_the_csr_tables_permute_every_composed_row_exactly_once():
    """The VJP's no-atomics reduction depends on this being a partition."""

    composer = _composer()
    rows = composer.path_count
    for offsets, table, owner_count in (
        (composer.by_inbound_offsets, composer.by_inbound_rows, 8),
        (composer.by_outbound_offsets, composer.by_outbound_rows, 8),
        (composer.by_response_offsets, composer.by_response_rows, 2),
    ):
        assert offsets.shape == (owner_count + 1,)
        assert offsets.tolist()[0] == 0
        assert offsets.tolist()[-1] == rows
        assert sorted(table.tolist()) == list(range(rows))
    # And each segment really holds the rows that owner produced.
    inbound_row = composer.inbound_row.tolist()
    offsets = composer.by_inbound_offsets.tolist()
    rows_by_owner = composer.by_inbound_rows.tolist()
    for owner in range(8):
        segment = rows_by_owner[offsets[owner] : offsets[owner + 1]]
        assert all(inbound_row[row] == owner for row in segment)
        assert segment == sorted(segment)


def test_the_native_primal_matches_the_retained_torch_composition():
    composer = _composer()
    inbound, outbound, response = _frame(composer)
    composed = composer.compose(inbound, outbound, response)

    tau, rate, transfer = _reference(composer, inbound, outbound, response, None)
    scale = float(transfer.abs().max())
    torch.testing.assert_close(
        composed.total_delay_s, tau, rtol=1e-6, atol=0.0
    )
    torch.testing.assert_close(composed.delay_rate, rate, rtol=1e-6, atol=1e-20)
    torch.testing.assert_close(
        composed.complex_transfer_ref, transfer, rtol=1e-5, atol=scale * 1e-6
    )


def test_a_dead_row_publishes_exactly_zero_in_every_output():
    composer = _composer()
    valid_in = torch.ones(8, dtype=torch.bool, device="cuda")
    valid_in[3] = False
    valid_out = torch.ones(8, dtype=torch.bool, device="cuda")
    valid_out[5] = False
    inbound, outbound, response = _frame(composer, valid=(valid_in, valid_out))
    composed = composer.compose(inbound, outbound, response)

    expected = valid_in.index_select(0, composer.inbound_row) & (
        valid_out.index_select(0, composer.outbound_row)
    )
    assert torch.equal(composed.row_valid, expected)
    assert int((~expected).sum()) > 0
    dead = ~expected
    assert float(composed.total_delay_s[dead].abs().sum()) == 0.0
    assert float(composed.delay_rate[dead].abs().sum()) == 0.0
    assert float(composed.complex_transfer_ref[dead].abs().sum()) == 0.0
    # And the live rows are untouched by their dead neighbours.
    tau, _, transfer = _reference(
        composer, inbound, outbound, response, expected
    )
    torch.testing.assert_close(
        composed.total_delay_s, tau, rtol=1e-6, atol=0.0
    )
    scale = float(transfer.abs().max())
    torch.testing.assert_close(
        composed.complex_transfer_ref, transfer, rtol=1e-5, atol=scale * 1e-6
    )


def test_an_empty_pair_segment_composes_without_a_special_case():
    """A front end wider than the discovered legs is a normal frame.

    The pair partition spans the declared cross product, so the kernel sees an
    offsets table with equal consecutive entries. Nothing about that is special
    to the kernel - it never reads the table - but the composed result has to
    keep the declared pair count so the IQ cube keeps its shape.
    """

    inbound = fx.frozen_leg(fx.leg_rows([10], SITES, COMPONENTS))
    outbound = fx.frozen_leg(fx.leg_rows(SITES, [30], COMPONENTS))
    composer = TwoWayComposer.freeze(
        inbound,
        outbound,
        torch.tensor(SITES, dtype=torch.int64, device="cuda"),
        radar_source_ids=SOURCES,
        radar_sink_ids=SINKS,
        reference_frequency_hz=REFERENCE_FREQUENCY_HZ,
    )
    assert composer.sensor_pair_count == 4
    assert composer.pair_offsets.tolist() == [0, 8, 8, 8, 8]

    frame_inbound, frame_outbound, response = _frame(composer)
    composed = composer.compose(frame_inbound, frame_outbound, response)
    assert composed.path_count == 8
    assert composed.sensor_pair_count == 4

    from witwin.radar.synthesis.fmcw_beat import synthesize_fmcw_beat
    from support import spike_driver as drv

    # The fabricated front end is 2 sources x 2 sinks, so the waveform spec
    # has to describe the SAME array: the TDM slot of a sensor pair is only
    # defined once the pair partition and the array are the same front end.
    from dataclasses import replace

    spec = replace(drv.make_spec(num_chirps=2), num_tx=2, num_rx=2)
    iq = synthesize_fmcw_beat(drv.to_synthesis(composed), spec)
    assert iq.shape[1] == 4
    # The three empty pairs are exactly zero, and the populated one is not.
    assert float(iq[:, 1:, :].abs().sum()) == 0.0
    assert float(iq[:, 0, :].abs().sum()) > 0.0


def test_a_delay_rate_carrying_a_tape_is_refused_rather_than_zeroed():
    """"Returns None" and "silently dropped it" must not look the same.

    ``delay_rate`` is a primal Doppler rate by contract, so the join returns
    None for its gradient and a zero tangent for the composed rate. A rate that
    arrives with a tape would be silently severed, so it is refused.
    """

    composer = _composer()
    inbound, outbound, response = _frame(composer)
    from dataclasses import replace

    live = replace(
        inbound, delay_rate=inbound.delay_rate.clone().requires_grad_(True)
    )
    with pytest.raises(ValueError, match="inbound delay_rate carries requires_grad"):
        composer.compose(live, outbound, response)

    import torch.autograd.forward_ad as forward_ad

    with forward_ad.dual_level():
        dual = forward_ad.make_dual(
            outbound.delay_rate, torch.ones_like(outbound.delay_rate)
        )
        dualled = replace(outbound, delay_rate=dual)
        with pytest.raises(
            ValueError, match="outbound delay_rate carries a forward tangent"
        ):
            composer.compose(inbound, dualled, response)


def test_the_composed_rate_is_absent_unless_both_legs_publish_one():
    composer = _composer()
    inbound, outbound, response = _frame(composer, rates=False)
    composed = composer.compose(inbound, outbound, response)
    assert composed.delay_rate is None

    with_rates = _frame(composer)
    assert (
        composer.compose(with_rates[0], with_rates[1], with_rates[2]).delay_rate
        is not None
    )
    assert (
        composer.compose(
            with_rates[0], with_rates[1], with_rates[2], include_delay_rate=False
        ).delay_rate
        is None
    )
