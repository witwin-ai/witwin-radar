"""Independent dynamic ADC oracle for compact native fmcw fusion and AD."""

import math

import pytest
import torch
import torch.autograd.forward_ad as ad

from witwin.radar.synthesis.assembly import FmcwSpec
from witwin.radar.synthesis.fmcw import _synthesize_adaptive_fmcw

pytestmark = pytest.mark.gpu
FC = 77e9


def _case(nodes, carrier=0.0):
    spec = FmcwSpec(4, 2, 1e-6, 10e-6, 4e12, 0, FC, num_tx=2, carrier_hz=carrier, output_domain="beat")
    # Two different pair layouts, including empty sensor pairs. Each accepted
    # partition has K probes; no node mixes the two path identities.
    layout = [[0, 2, 3, 3, 6], [0, 1, 1, 3, 4]]
    starts, basis, owners = [], [], []
    for observation in range(16):
        owner = observation // 8
        base, width = (0, 6) if owner == 0 else (nodes * 6, 4)
        starts.append([base + width * j for j in range(nodes)])
        basis.append([1.0 / nodes] * nodes)
        owners.append(owner)
    # Exactly sampled queries: repeated indices and a one-hot basis.
    starts[0] = [0] * nodes
    basis[0] = [1.0] + [0.0] * (nodes - 1)
    # Higher-order Lagrange interpolation can contain negative weights.
    basis[1] = [-0.2, 1.2] + [0.0] * (nodes - 2)
    samples = torch.arange(nodes * 10, dtype=torch.float64, device="cuda")
    delays = (20e-9 + samples * 0.01e-9).float().double()
    coefficient = ((1.0 + samples * 0.01) * torch.exp(-2j * math.pi * FC * delays)).cfloat()
    samples = torch.stack((delays, coefficient.real.double(), coefficient.imag.double()), dim=1)
    valid = [int(i % 6 != 1) for i in range(nodes * 6)] + [int(i % 4 != 1) for i in range(nodes * 4)]
    clock = torch.arange(16, device="cuda", dtype=torch.float64).remainder(4) * spec.sample_period_s
    metadata = (
        torch.tensor(valid, device="cuda", dtype=torch.int32),
        torch.tensor(starts, device="cuda"),
        torch.tensor(basis, device="cuda", dtype=torch.float64),
        torch.tensor(layout, device="cuda"),
        torch.tensor(owners, device="cuda"),
        clock,
    )

    def oracle(x, begin=0, stop=16):
        result = []
        for observation in range(begin, stop):
            for receiver in range(2):
                pair = receiver * 2 + observation // 4 % 2
                bounds = layout[owners[observation]]
                value = x.sum().to(torch.complex128) * 0
                for local in range(bounds[pair], bounds[pair + 1]):
                    indices = [start + local for start in starts[observation]]
                    # Independent complex equation; model's rounding boundary
                    # is explicit, so fusion cannot quietly keep extra precision.
                    tau = sum(basis[observation][j] * x[index, 0] for j, index in enumerate(indices))
                    field = (
                        sum(
                            basis[observation][j]
                            * torch.complex(x[index, 1], x[index, 2])
                            * torch.exp(-2j * math.pi * FC * (tau - x[index, 0]))
                            for j, index in enumerate(indices)
                        )
                        .cfloat()
                        .cdouble()
                    )
                    field = field * valid[indices[0]]
                    tau = tau.float().double()
                    cycles = carrier * tau + spec.slope_hz_per_s * tau * (clock[observation] - tau / 2)
                    value = value + field.conj() * torch.exp(2j * math.pi * cycles)
                result.append(value)
        return torch.stack(result).cfloat()

    def solve(x, begin=0, stop=16):
        return _synthesize_adaptive_fmcw(x, *metadata, spec, FC, begin, stop)

    return samples, metadata, spec, solve, oracle


@pytest.mark.parametrize("nodes", [2, 3, 5, 9])
@pytest.mark.parametrize("carrier", [0.0, FC])
def test_compact_fmcw_matches_complex_oracle_and_first_order_ad(nodes, carrier):
    samples, _, _, solve, oracle = _case(nodes, carrier)
    leaf = samples.clone().requires_grad_()
    actual, expected = solve(leaf), oracle(leaf)
    torch.testing.assert_close(actual, expected, rtol=3e-6, atol=3e-6)
    cotangent = torch.full_like(actual, 0.7 + 0.3j)
    gradient = torch.autograd.grad(actual, leaf, cotangent)[0]
    reference = torch.autograd.grad(expected, leaf, cotangent)[0]
    torch.testing.assert_close(gradient, reference, rtol=3e-4, atol=2e-5)
    direction = torch.ones_like(samples) * 0.1
    direction[:, 0] = 2e-12
    with ad.dual_level():
        dual = ad.make_dual(samples, direction)
        primal, tangent = ad.unpack_dual(solve(dual))
        _, reference_tangent = ad.unpack_dual(oracle(dual))
        assert torch.equal(primal, actual)
        torch.testing.assert_close(tangent, reference_tangent, rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(
        (tangent.conj() * cotangent).real.sum().double(), (gradient * direction).sum(), rtol=3e-5, atol=3e-6
    )


def test_compact_fmcw_batch_cuts_preserve_tdm_and_stream_order():
    samples, _, _, solve, oracle = _case(3)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        retained = solve(samples)
        chunks = torch.cat([solve(samples, 0, 3), solve(samples, 3, 11), solve(samples, 11, 16)])
        later = solve(samples * 1.01)
    torch.cuda.current_stream().wait_stream(stream)
    assert torch.equal(retained, chunks)
    assert not torch.equal(retained, later)
    torch.testing.assert_close(retained, oracle(samples), rtol=3e-6, atol=3e-6)


def test_compact_fmcw_empty_rows_and_derivative_refusals():
    samples, metadata, spec, solve, _ = _case(2)
    leaf = samples.requires_grad_()
    with pytest.raises(RuntimeError, match="first.order|higher.order|create_graph"):
        torch.autograd.grad(solve(leaf).real.sum(), leaf, create_graph=True)
    validity, starts, basis, bounds, owners, clock = metadata
    with pytest.raises(RuntimeError, match="fixed"):
        _synthesize_adaptive_fmcw(
            samples, validity, starts, basis.requires_grad_(), bounds, owners, clock, spec, FC, 0, 16
        )
    empty = samples.new_empty((0, 3), requires_grad=True)
    value = _synthesize_adaptive_fmcw(
        empty, validity[:0], starts * 0, basis.detach(), bounds * 0, owners, clock, spec, FC, 0, 16
    )
    assert torch.equal(value, torch.zeros_like(value))
    assert torch.autograd.grad(value.real.sum(), empty)[0].shape == empty.shape


@pytest.mark.parametrize("column", [2, 5])
@pytest.mark.parametrize("mode", ["vjp", "jvp"])
def test_compact_fmcw_refuses_differentiable_schedule_in_both_modes(column, mode):
    samples, metadata, spec, _, _ = _case(2)
    metadata = list(metadata)
    with ad.dual_level():
        value = metadata[column]
        metadata[column] = value.requires_grad_() if mode == "vjp" else ad.make_dual(value, torch.ones_like(value))
        with pytest.raises(RuntimeError, match="fixed"):
            _synthesize_adaptive_fmcw(samples, *metadata, spec, FC, 0, 16)
