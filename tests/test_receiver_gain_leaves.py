"""Transmit power and LNA gain as leaves (R-ADR-052).

Both are smooth amplitude factors. The transmit power enters every
transmitted wave as ``sqrt(P)``: Channel launches the legs with the host
value and the leaf enters as a factor of value exactly one folded into the
transmit element's peak (and into the leakage rows). The LNA gain scales
everything between the LNA and the AGC except the DC offset, so the chain runs
at the host gain and applies ``s (y - dc) + dc`` with ``s`` exactly one.

Oracles: central differences of the public verb (the noise realisation is
keyed by the seed, so it is the same draw on both sides of the difference),
the closed form ``ln(10)/20`` per dB where the factor is the whole answer, and
bit-identity of a tensor without a derivative against the host value.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.autograd.forward_ad as fwad
from test_pattern_leaves import _radar, _verb_loss

from witwin.radar import Compression, Iq, Leakage, Noise, Pattern

pytestmark = pytest.mark.gpu

PER_DB = math.log(10.0) / 20.0


def _check(build, base: torch.Tensor, *, step: float, rel: float = 2e-3) -> float:
    """VJP = JVP = central difference of the verb loss; returns the gradient."""

    leaf = base.clone().requires_grad_(True)
    (gradient,) = torch.autograd.grad(_verb_loss(build(leaf)), leaf)
    with fwad.dual_level():
        jvp = float(fwad.unpack_dual(_verb_loss(build(fwad.make_dual(base, torch.ones_like(base))))).tangent)
    central = (float(_verb_loss(build(base + step))) - float(_verb_loss(build(base - step)))) / (2.0 * step)
    assert float(gradient) == pytest.approx(jvp, rel=1e-4)
    assert jvp == pytest.approx(central, rel=rel)
    return float(gradient)


@pytest.mark.parametrize("pattern", [None, Pattern.dipole(gain=2.0)], ids=["isotropic", "dipole"])
def test_the_transmit_power_is_a_leaf_with_the_amplitude_law(pattern):
    """d(loss)/d(P_dBm) = loss ln(10)/20: every row carries sqrt(P) once."""

    fields = {} if pattern is None else {"pattern": pattern}
    base = torch.tensor(10.0)
    gradient = _check(lambda power: _radar(power=power, **fields), base, step=1e-2)
    assert gradient == pytest.approx(float(_verb_loss(_radar(power=base, **fields))) * PER_DB, rel=2e-3)


def test_the_transmit_power_reaches_the_leakage_rows():
    """The coupled transmit signal is a transmitted wave too: it carries the same factor."""

    leakage = Leakage(isolation_db=30.0, delay_s=1e-9, phase=0.4)
    base = torch.tensor(12.0)
    gradient = _check(lambda power: _radar(power=power, leakage=leakage), base, step=1e-2)
    assert gradient == pytest.approx(float(_verb_loss(_radar(power=base, leakage=leakage))) * PER_DB, rel=2e-3)


def test_a_tensor_power_without_a_derivative_is_bit_identical_to_the_host_value():
    host = _verb_loss(_radar(power=10.0))
    tensor = _verb_loss(_radar(power=torch.tensor(10.0)))
    assert torch.equal(host, tensor)


CHAINS = {
    "noise": {"noise": Noise(figure=6.0)},
    "compression": {"noise": Noise(figure=6.0), "compression": Compression(iip3=-40.0, am_pm=5.0)},
    "iq_and_dc": {
        "noise": Noise(figure=6.0),
        "compression": Compression(iip3=-40.0),
        "iq": Iq(gain_db=0.5, phase_deg=3.0),
        # Comparable to the echo at the LNA output, so a central difference of
        # the loss is not swamped by the constant DC term.
        "dc_offset": 2e-5 + 1e-5j,
    },
}


@pytest.mark.parametrize("chain", sorted(CHAINS))
def test_the_lna_gain_is_a_leaf_of_the_receive_chain(chain):
    """Thermal noise, compression and the quadrature split are homogeneous in the gain; the DC offset is not."""

    fields = CHAINS[chain]
    base = torch.tensor(20.0)
    gradient = _check(lambda gain: _radar(lna_gain=gain, **fields), base, step=5e-2, rel=5e-3)
    loss = float(_verb_loss(_radar(lna_gain=base, **fields)))
    if "dc_offset" in fields:
        # The DC term does not scale, so the derivative is NOT the whole signal's amplitude law.
        without_dc = float(_verb_loss(_radar(lna_gain=base, **{**fields, "dc_offset": 0.0})))
        assert gradient == pytest.approx(without_dc * PER_DB, rel=5e-3)
        assert gradient != pytest.approx(loss * PER_DB, rel=1e-2)
    else:
        assert gradient == pytest.approx(loss * PER_DB, rel=5e-3)


def test_a_tensor_lna_without_a_derivative_is_bit_identical_to_the_host_value():
    fields = CHAINS["iq_and_dc"]
    host = _verb_loss(_radar(lna_gain=20.0, **fields))
    tensor = _verb_loss(_radar(lna_gain=torch.tensor(20.0), **fields))
    assert torch.equal(host, tensor)


def test_a_non_scalar_gain_tensor_is_refused_by_name():
    with pytest.raises(ValueError, match="Radar.lna_gain"):
        _radar(lna_gain=torch.tensor([20.0, 21.0]))
    with pytest.raises(ValueError, match="Radar.power"):
        _radar(power=torch.tensor([10.0]))


@pytest.mark.parametrize("side", ["tx", "rx"])
def test_a_coupling_matrix_is_a_leaf(side):
    """Two cuBLAS products whose derivative is the transposed products: a complex matrix leaf, both modes."""

    from witwin.radar import Coupling

    generator = torch.Generator().manual_seed(8)
    base = (torch.eye(2) + 0.1 * torch.randn(2, 2, generator=generator)).to(torch.complex64)
    base = base + 0.05j * torch.randn(2, 2, generator=generator)
    direction = torch.complex(torch.randn(2, 2, generator=generator), torch.randn(2, 2, generator=generator))

    def build(matrix):
        return _radar(noise=Noise(figure=6.0), coupling=Coupling(**{side: matrix}))

    leaf = base.clone().requires_grad_(True)
    (gradient,) = torch.autograd.grad(_verb_loss(build(leaf)), leaf)
    vjp = float((gradient.conj() * direction).real.sum())
    with fwad.dual_level():
        jvp = float(fwad.unpack_dual(_verb_loss(build(fwad.make_dual(base, direction)))).tangent)
    step = 1e-3
    central = float(_verb_loss(build(base + step * direction))) - float(_verb_loss(build(base - step * direction)))
    assert vjp == pytest.approx(jvp, rel=1e-4)
    assert jvp == pytest.approx(central / (2.0 * step), rel=5e-3)
