"""Permuting the frozen leg rows must not change one number of the output.

Channel owns the frozen leg row order, so this permutes fabricated legs: the
same logical rows, published in a different sequence, with their payloads
permuted to match. Everything the join publishes must come back ELEMENTWISE
identical - not as an equal set.

That distinction is the whole point. The join's membership was already by
identity; its ORDER used to be a function of the joined rows' positions, so a
permuted leg order preserved the composed set and permuted the composed
sequence. A set comparison would have passed against that, and every consumer
that reads a composed row by index would still have been wrong.

The gradients are compared BIT for bit. That is a legitimate assertion here
rather than a lucky one: the VJP reduces over the frozen CSR segments, one
thread per gradient slot with no atomics, so the summation order is a property
of the frozen join. If the composed order is genuinely canonical, the segments
are the same segments in the same order and the arithmetic is identical.
"""

from __future__ import annotations

import pytest
import torch
from reference.two_way_torch import PerSiteResponse  # noqa: E402
from support import join_fixture as fx  # noqa: E402

from witwin.radar.paths import TwoWayComposer

pytestmark = pytest.mark.gpu

SOURCES = [10, 11]
SINKS = [30, 31]
SITES = [20, 21]
COMPONENTS = [0, 1]
REFERENCE_FREQUENCY_HZ = 77.0e9


def _permutation(count: int, seed: int) -> list[int]:
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(count, generator=generator).tolist()


def _run(inbound_order, outbound_order, *, requires_grad=False):
    """Freeze and compose with the leg rows in a given order."""

    inbound_rows = fx.leg_rows(SOURCES, SITES, COMPONENTS)
    outbound_rows = fx.leg_rows(SITES, SINKS, COMPONENTS)
    tau_in, rate_in, c_in = fx.payload(len(inbound_rows), seed=31)
    tau_out, rate_out, c_out = fx.payload(len(outbound_rows), seed=32)
    _, _, site_value = fx.payload(len(SITES), seed=33)

    valid_in = torch.ones(len(inbound_rows), dtype=torch.bool, device="cuda")
    valid_in[1] = False
    valid_out = torch.ones(len(outbound_rows), dtype=torch.bool, device="cuda")
    valid_out[4] = False

    def take(values, order):
        index = torch.tensor(order, dtype=torch.int64, device=values.device)
        return values.index_select(0, index).contiguous()

    inbound_frozen = fx.frozen_leg([inbound_rows[row] for row in inbound_order])
    outbound_frozen = fx.frozen_leg([outbound_rows[row] for row in outbound_order])
    composer = TwoWayComposer.freeze(
        inbound_frozen,
        outbound_frozen,
        torch.tensor(SITES, dtype=torch.int64, device="cuda"),
        radar_source_ids=SOURCES,
        radar_sink_ids=SINKS,
        reference_frequency_hz=REFERENCE_FREQUENCY_HZ,
    )

    leaves = {
        "tau_in": take(tau_in, inbound_order).to(torch.float32),
        "tau_out": take(tau_out, outbound_order).to(torch.float32),
        "c_in": take(c_in, inbound_order).to(torch.complex64),
        "c_out": take(c_out, outbound_order).to(torch.complex64),
        "response": site_value.to(torch.complex64),
    }
    if requires_grad:
        leaves = {name: value.clone().requires_grad_(True) for name, value in leaves.items()}
    inbound = fx.leg_batch(
        leaves["tau_in"],
        leaves["c_in"],
        rate=take(rate_in, inbound_order).to(torch.float32),
        row_valid=take(valid_in, inbound_order),
    )
    outbound = fx.leg_batch(
        leaves["tau_out"],
        leaves["c_out"],
        rate=take(rate_out, outbound_order).to(torch.float32),
        row_valid=take(valid_out, outbound_order),
    )
    composed = composer.compose(inbound, outbound, PerSiteResponse(leaves["response"]))
    return composer, composed, leaves, (inbound_order, outbound_order)


def _identity_columns(composed, orders):
    """Composed row identity expressed in ORIGINAL leg row numbering.

    ``inbound_row``/``outbound_row`` are indices into whatever order the legs
    were published in, so they legitimately differ between the two runs. Mapped
    back through the permutation they must agree, which is the statement that
    the join found the same pairs.
    """

    inbound_order, outbound_order = orders
    return (
        composed.topology.radar_source_id.tolist(),
        composed.topology.site_id.tolist(),
        composed.topology.radar_sink_id.tolist(),
        [inbound_order[row] for row in composed.topology.inbound_row.tolist()],
        [outbound_order[row] for row in composed.topology.outbound_row.tolist()],
    )


def test_a_permuted_leg_order_composes_to_an_elementwise_identical_frame():
    straight_order = (list(range(8)), list(range(8)))
    shuffled_order = (_permutation(8, 5), _permutation(8, 6))
    assert shuffled_order != straight_order

    _, straight, _, straight_maps = _run(*straight_order)
    _, permuted, _, permuted_maps = _run(*shuffled_order)

    assert straight.path_count == permuted.path_count
    assert straight.sensor_pair_count == permuted.sensor_pair_count
    assert straight.pair_offsets.tolist() == permuted.pair_offsets.tolist()
    assert straight.sensor_pair_index.tolist() == permuted.sensor_pair_index.tolist()
    assert _identity_columns(straight, straight_maps) == _identity_columns(permuted, permuted_maps)
    # The permutation really did reorder the legs: the raw row indices differ.
    assert straight.topology.inbound_row.tolist() != permuted.topology.inbound_row.tolist()

    for name in ("total_delay_s", "delay_rate", "complex_transfer_ref", "row_valid"):
        assert torch.equal(getattr(straight, name), getattr(permuted, name)), name


def test_a_permuted_leg_order_produces_bit_identical_gradients():
    weights_generator = torch.Generator().manual_seed(808)
    rows = 32
    weight = (torch.rand(rows, generator=weights_generator, dtype=torch.float32) - 0.5).cuda()
    transfer_weight = torch.complex(
        torch.rand(rows, generator=weights_generator, dtype=torch.float32) - 0.5,
        torch.rand(rows, generator=weights_generator, dtype=torch.float32) - 0.5,
    ).cuda()

    def gradients(order):
        _, composed, leaves, maps = _run(*order, requires_grad=True)
        assert composed.path_count == rows
        loss = (weight * composed.total_delay_s * 1.0e8).sum() + (
            torch.conj(transfer_weight) * composed.complex_transfer_ref
        ).real.sum()
        loss.backward()
        return {name: value.grad for name, value in leaves.items()}, maps

    straight, straight_maps = gradients((list(range(8)), list(range(8))))
    permuted, permuted_maps = gradients((_permutation(8, 5), _permutation(8, 6)))

    # The per-site response gradient is a reduction over every round trip
    # through that site, so it is the one that would move if the segments were
    # walked in a different order. Bit identical.
    assert torch.equal(straight["response"], permuted["response"])

    # The leg gradients live in leg row order, so they are compared through the
    # permutation rather than positionally.
    inbound_order = permuted_maps[0]
    outbound_order = permuted_maps[1]
    for name, order in (
        ("tau_in", inbound_order),
        ("c_in", inbound_order),
        ("tau_out", outbound_order),
        ("c_out", outbound_order),
    ):
        index = torch.tensor(order, dtype=torch.int64, device="cuda")
        assert torch.equal(straight[name].index_select(0, index), permuted[name]), name

    # Not vacuous.
    assert float(straight["response"].abs().min()) > 1.0e-6
    assert float(straight["tau_in"].abs().max()) > 1.0e-6
