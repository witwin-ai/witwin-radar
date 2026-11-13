"""Element patterns: absolute gain, separate transmit and receive, complex field.

The oracles are written here. Gain: the radar equation scales by
``G_t G_r`` in power, so a 10 dBi transmit and a 6 dBi receive element put the
echo 16 dB above the isotropic one. Separate patterns: an off-boresight target
sees ``sqrt(G_rx(theta))`` from a receive-only pattern. Field patterns: a
complex table with a linear phase ramp rotates the echo by that ramp's value at
the transmit and the receive angle, and the derivatives agree with central
differences and with each other (the VJP/JVP adjoint identity).
"""

from __future__ import annotations

import math

import pytest
import torch
from test_sensor_weight import C0, _rows, _sample

from witwin.radar import Pattern, PointTargets, Radar
from witwin.radar.sensors import SensorWeightPlan, evaluate_sensor_weights

pytestmark = pytest.mark.gpu

ANGLES = tuple(float(v) for v in range(-90, 91, 5))


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
    return Radar(
        reference_frequency_hz=77e9,
        waveform=_fmcw(),
        tx=((0.0, 0.0, 0.0),),
        rx=((0.0, 0.0, 0.0),),
        power=10.0,
        position=(0.0, 0.0, 0.0),
        look_at=(1.0, 0.0, 0.0),
        up=(0.0, 1.0, 0.0),
        **fields,
    )


def _echo(radar: Radar, position) -> torch.Tensor:
    pytest.importorskip("witwin.channel")
    from witwin.core import Scene

    targets = PointTargets(positions=torch.tensor([position], device=radar.device), rcs=1.0)
    result = radar.simulate(Scene(structures=(), endpoints=[]), targets, times=(0.0,), los=True, reflections=0)
    return result.cube.to(torch.complex128)


def test_the_absolute_gain_is_applied_once_on_each_side():
    """M4-6 / FE-2: ``G_t G_r`` in power, 16 dB for a 10 dBi and a 6 dBi element."""

    reference = _echo(_radar(), (6.0, 0.0, 0.0))
    gained = _echo(
        _radar(pattern=Pattern.isotropic(gain=10.0), rx_pattern=Pattern.isotropic(gain=6.0)), (6.0, 0.0, 0.0)
    )
    ratio_db = 20 * math.log10(float(gained.abs().mean() / reference.abs().mean()))
    assert ratio_db == pytest.approx(16.0, abs=1e-3)
    phase = torch.angle(gained * reference.conj())
    assert float(phase.abs().max()) < 1e-5


def test_a_receive_only_pattern_scales_by_the_receive_gain_alone():
    """The receive element's dipole cut at the target's azimuth, and nothing on transmit."""

    theta = 30.0
    distance = 6.0
    # Local x is world +z for this pose (right = forward x up); off-boresight in the x cut.
    position = (distance * math.cos(math.radians(theta)), 0.0, distance * math.sin(math.radians(theta)))
    reference = _echo(_radar(), position)
    shaped = _echo(_radar(rx_pattern=Pattern.dipole()), position)
    from witwin.radar.sensors import half_wave_dipole_power_cut

    expected = math.sqrt(half_wave_dipole_power_cut(theta))
    assert float(shaped.abs().mean() / reference.abs().mean()) == pytest.approx(expected, rel=2e-3)


def _ramp(slope_rad_per_deg: float) -> Pattern:
    """A unit-magnitude field pattern with a linear phase ramp in the x cut."""

    x_field = [complex(math.cos(slope_rad_per_deg * a), math.sin(slope_rad_per_deg * a)) for a in ANGLES]
    return Pattern.field_separable(ANGLES, x_field, ANGLES, [1.0] * len(ANGLES))


def test_a_complex_field_pattern_rotates_the_echo_by_its_phase_at_each_angle():
    theta = 20.0
    distance = 6.0
    position = (distance * math.cos(math.radians(theta)), 0.0, distance * math.sin(math.radians(theta)))
    slope = 0.01
    reference = _echo(_radar(), position)
    both = _echo(_radar(pattern=_ramp(slope)), position)
    transmit_only = _echo(_radar(pattern=_ramp(slope), rx_pattern=Pattern.isotropic()), position)
    # This pose's local x axis is world +z, so the target sits at x = +theta.
    for cube, sides in ((both, 2), (transmit_only, 1)):
        rotation = torch.angle(cube * reference.conj()).mean()
        # The FMCW beat cube is the conjugated Channel convention.
        assert float(-rotation) == pytest.approx(sides * slope * theta, abs=2e-3)
        assert float(cube.abs().mean() / reference.abs().mean()) == pytest.approx(1.0, abs=1e-4)


def _field_radar():
    from conftest import STANDARD_CONFIG

    amplitude = [0.5 + 0.5 * math.cos(math.radians(a)) for a in ANGLES]
    x_field = [m * complex(math.cos(0.02 * a), math.sin(0.02 * a)) for m, a in zip(amplitude, ANGLES, strict=True)]
    y_field = [complex(math.cos(-0.01 * a), math.sin(-0.01 * a)) * m for m, a in zip(amplitude, ANGLES, strict=True)]
    pattern = Pattern.field_separable(ANGLES, x_field, ANGLES, y_field, gain=3.0)
    return Radar.from_dict(dict(STANDARD_CONFIG), pattern=pattern, rx_pattern=Pattern.dipole(gain=2.0))


def _weights(radar, sample, positions):
    geometry, site_in, site_out, intensity = _rows(radar, sample)
    plan = SensorWeightPlan.build(radar.pattern, rx_pattern=radar.rx_pattern, c0=C0, device=radar.device)
    weight = torch.full((geometry.path_count,), 0.3 - 0.4j, dtype=torch.complex64, device=radar.device)
    tx_pos, rx_pos, site_in = positions(radar.tx_pos, radar.rx_pos, site_in)
    return evaluate_sensor_weights(
        tx_pos=tx_pos,
        rx_pos=rx_pos,
        site_in=site_in,
        site_out=site_out,
        intensity=intensity,
        weight=weight,
        geometry=geometry,
        plan=plan,
    ).weight


def test_field_pattern_derivatives_match_central_differences_and_each_other():
    radar = _field_radar()
    sample = _sample(radar, 6, seed=4)
    generator = torch.Generator().manual_seed(1)
    direction = torch.randn(*_rows(radar, sample)[1].shape, generator=generator).to(radar.device)
    cotangent = torch.randn(_rows(radar, sample)[0].path_count, dtype=torch.complex64, generator=generator).to(
        radar.device
    )

    leaf = _rows(radar, sample)[1].clone().requires_grad_(True)
    weight = _weights(radar, sample, lambda t, r, s: (t, r, leaf))
    (weight * cotangent.conj()).real.sum().backward()
    vjp = float((leaf.grad * direction).sum())

    import torch.autograd.forward_ad as fwad

    base = _rows(radar, sample)[1]
    with fwad.dual_level():
        dual = fwad.make_dual(base, direction)
        tangent = fwad.unpack_dual(_weights(radar, sample, lambda t, r, s: (t, r, dual))).tangent
    jvp = float((tangent * cotangent.conj()).real.sum())
    assert vjp == pytest.approx(jvp, rel=1e-4)

    step = 1e-3
    plus = _weights(radar, sample, lambda t, r, s: (t, r, (base + step * direction).contiguous()))
    minus = _weights(radar, sample, lambda t, r, s: (t, r, (base - step * direction).contiguous()))
    difference = float((((plus - minus) / (2 * step)) * cotangent.conj()).real.sum())
    assert jvp == pytest.approx(difference, rel=2e-2)


def test_power_patterns_are_bit_identical_through_the_packed_tables():
    """A power-by-power radar takes the original real branch: same bits as before."""

    radar = Radar.from_dict(dict(__import__("conftest").STANDARD_CONFIG), pattern=Pattern.dipole())
    sample = _sample(radar, 8, seed=2)
    first = _weights(radar, sample, lambda t, r, s: (t, r, s))
    explicit = Radar.from_dict(
        dict(__import__("conftest").STANDARD_CONFIG), pattern=Pattern.dipole(), rx_pattern=Pattern.dipole()
    )
    assert torch.equal(first, _weights(explicit, sample, lambda t, r, s: (t, r, s)))


def test_from_antenna_samples_a_core_custom_pattern_with_its_peak_as_gain():
    """A TR 38.901-style element sampled from a Core ``AntennaPattern``.

    The callable takes element-frame directions (boresight +x, zenith +z) and
    returns the absolute field; the sampled pattern carries the peak as dBi
    and a normalised table.
    """

    from witwin.core import AntennaPattern

    def element(direction):
        horizontal = torch.sqrt(direction[..., 0] ** 2 + direction[..., 1] ** 2)
        theta = torch.rad2deg(torch.atan2(horizontal, direction[..., 2]))
        phi = torch.rad2deg(torch.atan2(direction[..., 1], direction[..., 0]))
        attenuation = -torch.clamp(12 * ((theta - 90) / 65) ** 2 + 12 * (phi / 65) ** 2, max=30.0)
        return torch.sqrt(10 ** ((8.0 + attenuation) / 10)).to(torch.complex64)

    pattern = Pattern.from_antenna(AntennaPattern(kind="custom", custom=element))
    assert pattern.gain == pytest.approx(8.0, abs=1e-4)
    assert pattern.quantity == "field"
    # Radar-local x is the element's -y, so the x cut is the azimuth cut:
    # 32.5 degrees off boresight is the 3 dB beamwidth edge.
    row = pattern.y_angles.index(0.0)
    column = min(range(len(pattern.x_angles)), key=lambda i: abs(pattern.x_angles[i] - 32.0))
    value = abs(pattern.gain_table[row][column]) ** 2
    expected = 10 ** (-12 * (pattern.x_angles[column] / 65) ** 2 / 10)
    assert value == pytest.approx(expected, rel=1e-4)


def test_channel_presets_sample_through_the_same_route():
    presets = pytest.importorskip("witwin.channel.presets")
    if not hasattr(presets, "tr38901_pattern"):
        pytest.skip("this Channel has no element presets")
    pattern = Pattern.from_antenna(presets.tr38901_pattern(max_gain_dbi=8.0))
    assert pattern.gain == pytest.approx(8.0, abs=1e-3)
    dipole = Pattern.from_antenna(presets.dipole_pattern("half_wave"))
    assert dipole.gain == pytest.approx(2.15, abs=0.02)


def test_unnormalised_tables_are_refused_by_name():
    with pytest.raises(ValueError, match="normalised to at most 1"):
        Pattern.separable((-90, 90), (1.0, 2.0), (-90, 90), (1.0, 1.0))
    with pytest.raises(ValueError, match=r"\|F\| <= 1"):
        Pattern.field_separable((-90, 90), (1.0, 1.5j), (-90, 90), (1.0, 1.0))


def test_the_isotropic_default_skips_the_pattern_launch(monkeypatch):
    """P-8: isotropic elements at 0 dBi are the identity; no kernel runs for them."""

    from support.dsp_ledger import DspLedger

    from witwin.radar.cuda import runtime

    radar = _radar()
    _echo(radar, (6.0, 0.0, 0.0))
    ledger = DspLedger(monkeypatch, ("sensor_weight_forward",), runtime.build_extension())
    _echo(radar, (6.0, 0.0, 0.0))
    assert ledger.launches.get("sensor_weight_forward", 0) == 0
