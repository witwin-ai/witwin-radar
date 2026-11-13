"""Antenna pattern tables, peak gains and the pattern frame as leaves (R-ADR-052).

The sensor-weight operator is linear in the table values (the interpolation
weights are constants of each row's direction) and in each element's peak
amplitude, and its pattern-frame derivative is the direction derivative seen
from the frame's side. Each is checked at the operator against central
differences and the VJP/JVP adjoint identity, for every table kind (power and
field, separable and map) on both sides, and through ``Radar.simulate`` against
central differences of the public verb.

The reverse pass is a keyed list reduced in ascending row order, so it is
pinned bit-identical across runs, and its cost is pinned in operator calls: one
``sensor_weight_backward`` and one ``keyed_sum_forward`` with a parameter leaf,
no ``keyed_sum_forward`` without one.
"""

from __future__ import annotations

import pytest
import torch
import torch.autograd.forward_ad as fwad
from test_sensor_weight import C0, _rows, _sample

from witwin.radar import Pattern, PointTargets, Radar
from witwin.radar.sensors import SensorWeightPlan, evaluate_sensor_weights

pytestmark = pytest.mark.gpu

ANGLES = tuple(float(a) for a in range(-90, 91, 10))
KINDS = ("separable", "field_separable", "table", "field_table")


def _values(kind: str, generator: torch.Generator):
    n = len(ANGLES)

    def field(shape):
        magnitude = torch.rand(shape, generator=generator) * 0.5 + 0.2
        return (magnitude * torch.exp(1j * torch.rand(shape, generator=generator))).to(torch.complex64)

    if kind == "separable":
        return [torch.rand(n, generator=generator) * 0.8 + 0.1 for _ in range(2)]
    if kind == "field_separable":
        return [field(n) for _ in range(2)]
    if kind == "table":
        return [torch.rand(n, n, generator=generator) * 0.8 + 0.1]
    return [field((n, n))]


def _pattern(kind: str, values, gain) -> Pattern:
    if kind == "separable":
        return Pattern.separable(ANGLES, values[0], ANGLES, values[1], gain=gain)
    if kind == "field_separable":
        return Pattern.field_separable(ANGLES, values[0], ANGLES, values[1], gain=gain)
    if kind == "table":
        return Pattern.table(ANGLES, ANGLES, values[0], gain=gain)
    return Pattern.field_table(ANGLES, ANGLES, values[0], gain=gain)


def _direction(value: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    real = torch.randn(value.shape, generator=generator)
    if value.is_complex():
        return torch.complex(real, torch.randn(value.shape, generator=generator)).to(value.dtype)
    return real


def _pair(gradient: torch.Tensor, direction: torch.Tensor) -> float:
    """``<g, d>`` with Torch's convention for a complex leaf: ``dL = Re(conj(g) dv)``."""

    return (
        float((gradient.conj() * direction).real.sum())
        if gradient.is_complex()
        else float((gradient * direction).sum())
    )


class _Frame:
    """One operator launch over random rows of the standard radar, with a fixed cotangent."""

    def __init__(self, seed: int = 3):
        from conftest import STANDARD_CONFIG

        self.radar = Radar.from_dict(dict(STANDARD_CONFIG))
        sample = _sample(self.radar, 7, seed=seed)
        self.geometry, self.site_in, self.site_out, self.intensity = _rows(self.radar, sample)
        generator = torch.Generator().manual_seed(seed)
        rows = self.geometry.path_count

        def complex_rows():
            return torch.complex(torch.randn(rows, generator=generator), torch.randn(rows, generator=generator))

        self.weight = complex_rows().to(torch.complex64).cuda()
        self.cotangent = complex_rows().to(torch.complex64).cuda()

    def loss(self, tx: Pattern, rx: Pattern, *, geometry=None) -> torch.Tensor:
        plan = SensorWeightPlan.build(tx, rx_pattern=rx, c0=C0, device="cuda")
        out = evaluate_sensor_weights(
            tx_pos=self.radar.tx_pos,
            rx_pos=self.radar.rx_pos,
            site_in=self.site_in,
            site_out=self.site_out,
            intensity=self.intensity,
            weight=self.weight,
            geometry=self.geometry if geometry is None else geometry,
            plan=plan,
        ).weight
        return (out * self.cotangent.conj()).real.sum()


@pytest.fixture(scope="module")
def frame():
    return _Frame()


@pytest.mark.parametrize("side", ["tx", "rx"])
@pytest.mark.parametrize("kind", KINDS)
def test_table_values_and_gain_match_central_differences_in_both_modes(frame, kind, side):
    """VJP = JVP (adjoint identity) = central difference, for the values and the dBi gain together."""

    generator = torch.Generator().manual_seed(11)
    base = _values(kind, generator)
    directions = [_direction(value, generator) for value in base]
    gain, gain_direction = torch.tensor(2.5), torch.tensor(0.7)
    other = Pattern.dipole(gain=1.0)

    def loss(values, decibels):
        element = _pattern(kind, values, decibels)
        return frame.loss(element, other) if side == "tx" else frame.loss(other, element)

    leaves = [value.clone().requires_grad_(True) for value in base]
    gain_leaf = gain.clone().requires_grad_(True)
    gradients = torch.autograd.grad(loss(leaves, gain_leaf), [*leaves, gain_leaf])
    vjp = sum(_pair(g, d) for g, d in zip(gradients[:-1], directions, strict=True))
    vjp += float(gradients[-1] * gain_direction)

    with fwad.dual_level():
        duals = [fwad.make_dual(value, direction) for value, direction in zip(base, directions, strict=True)]
        jvp = float(fwad.unpack_dual(loss(duals, fwad.make_dual(gain, gain_direction))).tangent)

    # The response is linear in the table values up to a power element's square
    # root, so a wide step keeps the float32 cancellation small.
    step = 2e-2

    def shifted(t):
        values = [value + t * direction for value, direction in zip(base, directions, strict=True)]
        return float(loss(values, gain + t * gain_direction))

    central = (shifted(step) - shifted(-step)) / (2.0 * step)
    assert vjp == pytest.approx(jvp, rel=1e-4)
    assert jvp == pytest.approx(central, rel=1e-2)


def test_the_pattern_frame_matches_central_differences_in_both_modes(frame):
    """The frame's derivative is the direction derivative seen from the frame's side."""

    import dataclasses

    tx = _pattern("field_table", _values("field_table", torch.Generator().manual_seed(5)), 1.0)
    rx = Pattern.dipole(gain=2.0)
    base = frame.geometry.pattern_frame
    direction = torch.randn(3, 3, generator=torch.Generator().manual_seed(6)).cuda()

    def loss(axes):
        return frame.loss(tx, rx, geometry=dataclasses.replace(frame.geometry, pattern_frame=axes))

    leaf = base.clone().requires_grad_(True)
    (gradient,) = torch.autograd.grad(loss(leaf), leaf)
    vjp = float((gradient * direction).sum())
    with fwad.dual_level():
        jvp = float(fwad.unpack_dual(loss(fwad.make_dual(base, direction))).tangent)
    step = 1e-4
    central = (float(loss(base + step * direction)) - float(loss(base - step * direction))) / (2.0 * step)
    assert vjp == pytest.approx(jvp, rel=1e-4)
    assert jvp == pytest.approx(central, rel=1e-2)


def test_the_parameter_reduction_is_deterministic(frame):
    """A keyed list reduced in ascending row order: two backwards are bit-identical."""

    values = [value.cuda() for value in _values("table", torch.Generator().manual_seed(9))]
    gradients = []
    for _ in range(2):
        leaf = values[0].clone().requires_grad_(True)
        gain = torch.tensor(1.5, requires_grad=True)
        gradients.append(
            torch.autograd.grad(frame.loss(_pattern("table", [leaf], gain), Pattern.dipole()), [leaf, gain])
        )
    assert torch.equal(gradients[0][0], gradients[1][0])
    assert torch.equal(gradients[0][1], gradients[1][1])


def test_the_parameter_backward_costs_one_reduction_and_only_when_asked(monkeypatch, frame):
    """Budget: a table leaf adds one ``keyed_sum_forward``; a position-only backward adds none."""

    from support.dsp_ledger import DspLedger

    from witwin.radar.cuda import runtime

    names = ("sensor_weight_forward", "sensor_weight_backward", "sensor_weight_jvp", "keyed_sum_forward")
    table = runtime.build_extension()
    leaf = _values("table", torch.Generator().manual_seed(4))[0].requires_grad_(True)
    with DspLedger(operators=names, table=table) as ledger:
        frame.loss(_pattern("table", [leaf], 0.0), Pattern.dipole()).backward()
    assert dict(ledger.launches) == {
        "sensor_weight_forward": 1,
        "sensor_weight_backward": 1,
        "sensor_weight_jvp": 0,
        "keyed_sum_forward": 1,
    }

    weight = frame.weight.clone().requires_grad_(True)
    plan = SensorWeightPlan.build(Pattern.dipole(), c0=C0, device="cuda")
    with DspLedger(operators=names, table=table) as ledger:
        evaluate_sensor_weights(
            tx_pos=frame.radar.tx_pos,
            rx_pos=frame.radar.rx_pos,
            site_in=frame.site_in,
            site_out=frame.site_out,
            intensity=frame.intensity,
            weight=weight,
            geometry=frame.geometry,
            plan=plan,
        ).weight.abs().sum().backward()
    assert ledger.launches["keyed_sum_forward"] == 0
    assert ledger.launches["sensor_weight_backward"] == 1


def test_a_host_pattern_packs_bit_identically_and_a_tensor_axis_is_refused():
    """A host-authored pattern takes the host route; a leaf pattern packs to the same numbers."""

    values = _values("separable", torch.Generator().manual_seed(1))
    host = Pattern.separable(ANGLES, values[0].tolist(), ANGLES, values[1].tolist(), gain=3.0)
    leaf = Pattern.separable(ANGLES, values[0], ANGLES, values[1], gain=torch.tensor(3.0))
    packed = leaf.packed()
    assert torch.allclose(packed, host.packed(), rtol=1e-6, atol=0.0)
    assert torch.equal(packed[8:], host.packed()[8:])
    with pytest.raises(RuntimeError, match="host declaration"):
        Pattern.separable(torch.tensor(ANGLES, requires_grad=True), values[0], ANGLES, values[1])


# ---------------------------------------------------------------------------
# Through the public verb
# ---------------------------------------------------------------------------


def _fmcw():
    from witwin.radar import Fmcw

    return Fmcw(
        slope=30e12,
        sample_rate=10e6,
        samples_per_chirp=64,
        chirps_per_frame=4,
        adc_start=4e-6,
        idle=10e-6,
        ramp_end=40e-6,
        output="beat",
    )


def _radar(**fields) -> Radar:
    authored = {
        "reference_frequency_hz": 77e9,
        "waveform": _fmcw(),
        "tx": ((0.0, 0.0, 0.0), (0.002, 0.0, 0.0)),
        "rx": ((0.0, 0.0, 0.0), (0.0, 0.002, 0.0)),
        "power": 10.0,
        "look_at": (1.0, 0.0, 0.0),
        "up": (0.0, 1.0, 0.0),
    }
    return Radar(**{**authored, **fields})


def _verb_loss(radar: Radar, ad_mode: str = "none") -> torch.Tensor:
    pytest.importorskip("witwin.channel")
    from witwin.core import Scene

    targets = PointTargets(positions=torch.tensor([[6.0, 0.5, 1.0], [5.0, -0.4, -0.8]], device="cuda"), rcs=1.0)
    cube = radar.simulate(
        Scene(structures=(), endpoints=[]), targets, times=(0.0,), los=True, reflections=0, ad_mode=ad_mode
    )
    weights = torch.randn(cube.cube.shape, generator=torch.Generator().manual_seed(0), dtype=torch.complex64)
    return (cube.cube * weights.to(cube.cube.device).conj()).real.sum() * 1e6


@pytest.mark.parametrize("kind", ["table", "field_table"])
def test_a_pattern_table_leaf_reaches_the_simulated_cube(kind):
    """``Radar(pattern=Pattern.table(..., leaf))``: d(cube loss)/d(table) against central differences."""

    generator = torch.Generator().manual_seed(2)
    base = _values(kind, generator)[0]
    direction = _direction(base, generator)

    def loss(values):
        return _verb_loss(_radar(pattern=_pattern(kind, [values], 0.0)))

    leaf = base.clone().requires_grad_(True)
    (gradient,) = torch.autograd.grad(loss(leaf), leaf)
    with fwad.dual_level():
        jvp = float(fwad.unpack_dual(loss(fwad.make_dual(base, direction))).tangent)
    step = 1e-3
    central = (float(loss(base + step * direction)) - float(loss(base - step * direction))) / (2.0 * step)
    assert _pair(gradient, direction) == pytest.approx(jvp, rel=1e-4)
    assert jvp == pytest.approx(central, rel=5e-3)


def test_a_pattern_gain_leaf_reaches_the_simulated_cube():
    """``Pattern.dipole(gain=leaf)``: the dBi gain's derivative is the amplitude's ``ln(10)/20`` per dB."""

    def loss(gain):
        return _verb_loss(_radar(pattern=Pattern.dipole(gain=gain)))

    base = torch.tensor(3.0)
    leaf = base.clone().requires_grad_(True)
    (gradient,) = torch.autograd.grad(loss(leaf), leaf)
    step = 1e-2
    central = (float(loss(base + step)) - float(loss(base - step))) / (2.0 * step)
    assert float(gradient) == pytest.approx(central, rel=2e-3)
    # The amplitude law: one pattern is both elements, so the cube is proportional to 10^(2 gain / 20).
    assert float(gradient) == pytest.approx(float(loss(base)) * 2.0 * 2.302585092994046 / 20.0, rel=2e-3)
