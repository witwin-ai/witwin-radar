"""Independent complex-arithmetic oracle for native carrier interpolation."""

import math

import numpy as np
import pytest
import torch

from witwin.radar.paths import interpolate_path_rows
from witwin.radar.simulation import _lagrange_weights

pytestmark = pytest.mark.gpu

FC = 77e9


def _basis(query, nodes):
    """Independent Lagrange basis: a product loop, not the production helper."""

    out = []
    for row in range(len(query)):
        row_weights = []
        for near in range(nodes.shape[1]):
            value = 1.0
            for far in range(nodes.shape[1]):
                if far != near:
                    value *= (query[row] - nodes[row, far]) / (nodes[row, near] - nodes[row, far])
            row_weights.append(value)
        out.append(row_weights)
    return np.asarray(out)


def _oracle(delays, transfers, weights):
    """tau = sum w_j tau_j; C = sum w_j C_j exp(-j 2 pi fc (tau - tau_j))."""

    tau = sum(weights[:, index] * delay.double() for index, delay in enumerate(delays))
    field = sum(
        weights[:, index] * transfer.cdouble() * torch.exp(-2j * math.pi * FC * (tau - delay.double()))
        for index, (delay, transfer) in enumerate(zip(delays, transfers, strict=True))
    )
    return tau.float(), field.cfloat()


@pytest.mark.parametrize("nodes", [2, 3, 5])
def test_interpolation_preserves_carrier_wraps_and_native_derivatives(nodes):
    rows = 2
    node_times = np.asarray([[20e-9 + index * 0.7e-9 + row * 1e-9 for index in range(nodes)] for row in range(rows)])
    query = np.asarray([node_times[row, 0] + 0.37 * (node_times[row, -1] - node_times[row, 0]) for row in range(rows)])
    weights = torch.as_tensor(_basis(query, node_times), device="cuda")
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(rows, device="cuda", dtype=torch.float64))

    leaves = []
    for index in range(nodes):
        leaves.append(torch.as_tensor(node_times[:, index], device="cuda", dtype=torch.float32).requires_grad_())
    for index in range(nodes):
        phase = 0.3 + 0.5 * index
        leaves.append(
            torch.tensor([1 + 2j, 0.3 - 0.7j], device="cuda")
            .mul(math.cos(phase) + 1j * math.sin(phase))
            .requires_grad_()
        )
    delays, transfers = leaves[:nodes], leaves[nodes:]

    actual = interpolate_path_rows(delays, transfers, weights, FC)
    expected = _oracle(delays, transfers, weights)
    for produced, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(produced, reference, rtol=2e-6, atol=1e-7)

    def loss(values):
        return values[0].sum() * 1e8 + values[1].real.sum() + 0.3 * values[1].imag.sum()

    # These nodes are 0.7 ns apart at 77 GHz, so the interpolant spans ~54
    # carrier wraps: exactly the cancellation the transport exists to survive,
    # and a badly conditioned place to compare float32 gradients. The delay
    # gradient carries a 2*pi*fc factor of 4.8e11. Three nodes put the worst
    # cancellation in the basis - its weights leave [0, 1] while still summing
    # to one - and need a looser bound than two or five do; that is a property
    # of THIS fixture, not of the kernel, and a production interval is accepted
    # only within 0.02 rad where none of it applies.
    actual_grad = torch.autograd.grad(loss(actual), leaves)
    expected_grad = torch.autograd.grad(loss(expected), leaves)
    tolerance = 2e-5 if nodes == 3 else 3e-6
    for produced, reference in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(produced, reference, rtol=tolerance, atol=1e-6)

    tangents = [torch.full_like(leaf, 1e-12 * (1 + index)) for index, leaf in enumerate(delays)]
    tangents += [torch.full_like(leaf, 0.1j + 0.05 * index) for index, leaf in enumerate(transfers)]
    with torch.autograd.forward_ad.dual_level():
        duals = [torch.autograd.forward_ad.make_dual(a.detach(), t) for a, t in zip(leaves, tangents, strict=True)]
        result = interpolate_path_rows(duals[:nodes], duals[nodes:], weights, FC)
        reference = _oracle(duals[:nodes], duals[nodes:], weights)
        for produced, wanted in zip(result, reference, strict=True):
            torch.testing.assert_close(
                torch.autograd.forward_ad.unpack_dual(produced).tangent,
                torch.autograd.forward_ad.unpack_dual(wanted).tangent,
                rtol=3e-6,
                atol=1e-6,
            )


@pytest.mark.parametrize("nodes", [2, 3, 5])
def test_single_physical_path_has_no_interpolation_fade(nodes):
    """One coherent path must survive the blend at full amplitude."""

    node_times = np.asarray([[20e-9 + index * 0.5e-9 for index in range(nodes)]])
    query = np.asarray([node_times[0, 0] + 0.37 * (node_times[0, -1] - node_times[0, 0])])
    weights = torch.as_tensor(_basis(query, node_times), device="cuda")
    delays = [torch.as_tensor(node_times[:, index], device="cuda", dtype=torch.float32) for index in range(nodes)]
    transfers = [torch.exp(-2j * math.pi * FC * delay.double()).cfloat() for delay in delays]

    tau, weight = interpolate_path_rows(delays, transfers, weights, FC)
    expected_tau = sum(weights[:, index] * delay.double() for index, delay in enumerate(delays))
    torch.testing.assert_close(tau, expected_tau.float(), rtol=1e-6, atol=0)
    torch.testing.assert_close(weight, torch.exp(-2j * math.pi * FC * expected_tau).cfloat(), rtol=1e-6, atol=1e-6)


def test_a_higher_order_basis_tracks_curvature_a_linear_one_cannot():
    """The reason the order is configurable, as a number rather than a claim."""

    # A quadratic delay ramp: the linear rule must miss it and the quadratic
    # rule must reproduce it, both measured against the same true delay.
    def true_delay(t):
        return 20e-9 + 3e-6 * t + 8.0 * t * t

    span = 2e-4
    query = np.asarray([0.37 * span])
    errors = {}
    for nodes in (2, 3):
        node_times = np.asarray([[index * span / (nodes - 1) for index in range(nodes)]])
        weights = torch.as_tensor(_basis(query, node_times), device="cuda")
        delays = [
            torch.as_tensor(true_delay(node_times[:, index]), device="cuda", dtype=torch.float64).float()
            for index in range(nodes)
        ]
        transfers = [torch.exp(-2j * math.pi * FC * delay.double()).cfloat() for delay in delays]
        tau, _ = interpolate_path_rows(delays, transfers, weights, FC)
        errors[nodes] = abs(float(tau[0]) - true_delay(query[0]))

    assert errors[3] < errors[2] / 100, errors


def test_the_production_basis_matches_the_independent_one():
    nodes = np.asarray([[0.0, 1e-4, 2.5e-4, 4e-4, 6e-4], [1e-3, 1.1e-3, 1.25e-3, 1.4e-3, 1.6e-3]])
    query = np.asarray([1.7e-4, 1.32e-3])
    np.testing.assert_allclose(_lagrange_weights(query, nodes), _basis(query, nodes), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(_lagrange_weights(query, nodes).sum(axis=1), 1.0, rtol=0, atol=1e-12)


def test_interpolation_refuses_a_misshaped_node_set():
    delay = torch.zeros(3, device="cuda")
    transfer = torch.zeros(3, device="cuda", dtype=torch.complex64)
    weights = torch.ones((3, 2), device="cuda", dtype=torch.float64) / 2
    with pytest.raises(ValueError, match="at least two"):
        interpolate_path_rows([delay], [transfer], weights, FC)
    with pytest.raises(ValueError, match="nodes for"):
        interpolate_path_rows([delay] * 3, [transfer] * 3, weights, FC)
