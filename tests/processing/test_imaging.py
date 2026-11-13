"""Back-projection imaging against the geometry of point targets.

The oracle cubes are written here from the waveforms' own closed forms - the
FMCW de-chirped tone ``exp(+j 2 pi (f tau + S tau (u - tau / 2)))`` at ADC time
``u``, the OFDM CFR ``exp(-j 2 pi (f + n df) tau)`` - with the delay evaluated
from element positions this file places itself (the authored positions plus the
platform track at each chirp's first ADC instant). Nothing of the imaging
module builds them, so a timing, phase or frame error shows up as a defocused
or displaced image. A point target images at its position, at its amplitude
times the interpolation loss, with the textbook -3 dB widths ``0.886 c / 2B`` in
range and ``0.886 lambda R / 2L`` across a straight aperture of ``C`` samples
spaced ``d`` (``L = C d``), or ``0.886 lambda / (2 dtheta)`` across a turn.
"""

from __future__ import annotations

import math

import pytest
import torch
from conftest import PROCESSING_CONFIG

from witwin.radar import Radar
from witwin.radar.processing import (
    Aperture,
    ProcessingCube,
    SarImage,
    backproject,
    element_responses,
    isar_aperture,
    range_profile,
)
from witwin.radar.processing.imaging import _backproject_torch
from witwin.radar.synthesis.assembly import BEAT_PHASOR
from witwin.radar.synthesis.assembly import SPEED_OF_LIGHT_M_PER_S as C0

PAD = 8
#: Worst-case linear-interpolation loss on an 8x padded rectangular-window profile:
#: cos(pi / 16) for the half-bin phasor average times the Dirichlet at 1/16 bin.
INTERPOLATION_FLOOR = 0.975


def _radar(num_tx=1, num_rx=1, *, samples=256, chirps=64, **pose) -> Radar:
    config = {
        **PROCESSING_CONFIG,
        "num_tx": num_tx,
        "num_rx": num_rx,
        "tx_loc": [[4 * t, 0, 0] for t in range(num_tx)],
        "rx_loc": [[r, 0, 0] for r in range(num_rx)],
        "adc_samples": samples,
        "chirp_per_frame": chirps,
        "output_domain": "beat",
    }
    return Radar.from_dict(config, device="cpu", **pose)


def _frame(radar: Radar):
    """``(right, forward)`` world unit vectors of the radar."""

    _, world_from_local = radar._world_from_local_matrix(device="cpu", dtype=torch.float64)
    return world_from_local[:, 0], -world_from_local[:, 2]


def _element_track(radar: Radar, velocity) -> tuple[torch.Tensor, torch.Tensor]:
    """``[C, V, 3]`` element positions at each TDM chirp's first ADC sample, written independently."""

    spec = radar.waveform_spec()
    num_tx, num_rx = radar.num_tx, radar.num_rx
    chirp = torch.arange(spec.num_chirps, dtype=torch.float64).reshape(-1, 1)
    tx_index = torch.arange(num_tx, dtype=torch.float64).reshape(1, -1)
    instants = (chirp * num_tx + tx_index) * spec.chirp_period_s + spec.t_start_s  # [C, T]
    drift = instants.reshape(spec.num_chirps, num_tx, 1, 1) * torch.as_tensor(velocity, dtype=torch.float64)
    tx = radar.tx_pos.to(torch.float64).reshape(1, num_tx, 1, 3) + drift
    rx = radar.rx_pos.to(torch.float64).reshape(1, 1, num_rx, 3) + drift
    shape = (spec.num_chirps, num_tx, num_rx, 3)
    return tx.expand(shape).reshape(spec.num_chirps, -1, 3), rx.expand(shape).reshape(spec.num_chirps, -1, 3)


def _fmcw_cube(radar: Radar, targets, *, velocity=(0.0, 0.0, 0.0), delay_bias_s=0.0, gains=None):
    """``[TX, RX, C, N]`` beat samples of point targets ``[(position, amplitude), ...]`` (start-stop)."""

    spec = radar.waveform_spec()
    tx, rx = _element_track(radar, velocity)
    u = spec.t_start_s + torch.arange(spec.num_samples, dtype=torch.float64) * spec.sample_period_s
    cube = torch.zeros((spec.num_chirps, radar.num_tx * radar.num_rx, spec.num_samples), dtype=torch.complex128)
    for position, amplitude in targets:
        q = torch.as_tensor(position, dtype=torch.float64)
        tau = ((q - tx).norm(dim=-1) + (q - rx).norm(dim=-1)) / C0 + delay_bias_s
        tau = tau.unsqueeze(-1)
        cycles = spec.reference_frequency_hz * tau + spec.slope_hz_per_s * tau * (u - 0.5 * tau)
        cube = cube + amplitude * torch.exp(2j * math.pi * (cycles - torch.floor(cycles)))
    cube = cube.permute(1, 0, 2).reshape(radar.num_tx, radar.num_rx, spec.num_chirps, spec.num_samples)
    if gains is not None:
        cube = cube * gains.reshape(radar.num_tx, radar.num_rx, 1, 1)
    return ProcessingCube.from_tensor(cube, radar)


def _profile(cube: ProcessingCube):
    return range_profile(cube, range_bins=PAD * int(cube.data.shape[-1]))


def _line(center, direction, half_width, count=801):
    steps = torch.linspace(-half_width, half_width, count, dtype=torch.float64)
    return torch.as_tensor(center, dtype=torch.float64) + steps.reshape(-1, 1) * direction, steps


def _half_power_width(power: torch.Tensor, steps: torch.Tensor) -> float:
    """-3 dB full width of the main lobe around the peak, by linear interpolation."""

    peak = int(power.argmax())
    half = 0.5 * float(power[peak])

    def crossing(direction: int) -> float:
        index = peak
        while float(power[index + direction]) > half:
            index += direction
        inner, outer = float(power[index]), float(power[index + direction])
        fraction = (inner - half) / (inner - outer)
        return float(steps[index]) + direction * fraction * float(steps[1] - steps[0])

    return crossing(1) - crossing(-1)


def _bandwidth(radar: Radar) -> float:
    spec = radar.waveform_spec()
    return spec.slope_hz_per_s * spec.num_samples * spec.sample_period_s


def test_two_point_targets_image_at_their_positions_with_the_theoretical_resolution():
    radar = _radar()
    spec = radar.waveform_spec()
    right, forward = _frame(radar)
    speed = 12.0  # m/s along the radar's right: a sample spacing below lambda / 4
    velocity = speed * right
    near = 1.0 * forward
    far = 1.35 * forward + 0.12 * right
    cube = _fmcw_cube(radar, [(near, 1.0), (far, 0.5)], velocity=velocity)
    aperture = Aperture.from_radar(radar, velocity_mps=velocity)
    profile = _profile(cube)

    range_resolution = C0 / (2.0 * _bandwidth(radar))
    spacing = speed * spec.chirp_period_s
    cross_resolution = radar.wavelength * 1.0 / (2.0 * spec.num_chirps * spacing)

    for target, amplitude in ((near, 1.0), (far, 0.5)):
        along, steps = _line(target, forward, 3 * range_resolution)
        across, steps_x = _line(target, right, 3 * cross_resolution)
        image_r = backproject(profile, aperture, along).power
        image_x = backproject(profile, aperture, across).power
        assert abs(float(steps[int(image_r.argmax())])) <= range_resolution / 20
        assert abs(float(steps_x[int(image_x.argmax())])) <= cross_resolution / 20
        level = math.sqrt(float(image_r.max())) / amplitude
        # The other target's sidelobes add a little at this one's position.
        assert INTERPOLATION_FLOOR - 0.01 <= level <= 1.01, level
    along, steps = _line(near, forward, 3 * range_resolution)
    across, steps_x = _line(near, right, 3 * cross_resolution)
    width_r = _half_power_width(backproject(profile, aperture, along).power, steps)
    width_x = _half_power_width(backproject(profile, aperture, across).power, steps_x)
    assert width_r == pytest.approx(0.886 * range_resolution, rel=0.05)
    assert width_x == pytest.approx(0.886 * cross_resolution, rel=0.05)


def test_an_ignored_platform_motion_defocuses_the_same_targets():
    """The negative control: the same cube imaged as if the radar stood still loses its focus."""

    radar = _radar(chirps=128)
    right, forward = _frame(radar)
    velocity = 12.0 * right
    cube = _fmcw_cube(radar, [(1.0 * forward, 1.0)], velocity=velocity)
    still = backproject(_profile(cube), Aperture.from_radar(radar), (1.0 * forward).reshape(1, 3))
    assert float(still.data.abs()) < 0.35


def test_a_tdm_mimo_array_on_a_moving_platform_focuses_with_each_transmitters_own_instant():
    radar = _radar(2, 4, chirps=32)
    right, forward = _frame(radar)
    velocity = 6.0 * right
    target = 1.2 * forward + 0.05 * right
    cube = _fmcw_cube(radar, [(target, 0.7)], velocity=velocity)
    profile = _profile(cube)
    image = backproject(profile, Aperture.from_radar(radar, velocity_mps=velocity), target.reshape(1, 3))
    assert INTERPOLATION_FLOOR * 0.7 <= float(image.data.abs()) <= 0.7 * 1.0001
    # Every element agrees in phase at the target: the per-element responses are one complex number.
    responses = element_responses(profile, Aperture.from_radar(radar, velocity_mps=velocity), target)
    assert responses.shape == (2, 4)
    assert float((responses - responses.mean()).abs().max()) <= 0.04 * 0.7


def test_the_aperture_reads_the_tdm_instants_and_the_pose_convention():
    radar = _radar(2, 3, chirps=4, position=(0.5, -0.2, 0.1), look_at=(1.5, 2.0, 0.3), up=(0.0, 0.0, 1.0))
    velocity = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
    tx, rx = _element_track(radar, velocity)
    aperture = Aperture.from_radar(radar, velocity_mps=velocity)
    torch.testing.assert_close(aperture.tx_m, tx, rtol=0, atol=1e-12)
    torch.testing.assert_close(aperture.rx_m, rx, rtol=0, atol=1e-12)
    # The verbs' pose: a pure translation equals the straight track ...
    posed = Aperture.from_radar(radar, pose=lambda t: (torch.eye(3), velocity * t))
    torch.testing.assert_close(posed.tx_m, tx, rtol=0, atol=1e-12)
    # ... and a rotation turns the array about the authored position.
    turn = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    rotated = Aperture.from_radar(radar, time_s=2.0, pose=lambda t: (turn, torch.zeros(3)))
    origin = torch.tensor(radar.position, dtype=torch.float32).to(torch.float64)
    expected = origin + (radar.tx_pos.to(torch.float64) - origin) @ turn.T
    torch.testing.assert_close(rotated.tx_m[0, :: radar.num_rx], expected, rtol=0, atol=1e-12)
    assert float(rotated.times_s[0, 0]) == pytest.approx(2.0 + radar.waveform_spec().t_start_s)
    with pytest.raises(ValueError, match="not both"):
        Aperture.from_radar(radar, velocity_mps=velocity, pose=lambda t: (torch.eye(3), torch.zeros(3)))


def test_a_turning_target_images_in_its_body_frame_at_the_isar_resolution():
    """ISAR: a still radar, a target spinning about a vertical axis through its centre."""

    radar = _radar(chirps=128)
    spec = radar.waveform_spec()
    right, forward = _frame(radar)
    _, world_from_local = radar._world_from_local_matrix(device="cpu", dtype=torch.float64)
    up = world_from_local[:, 1]
    center = 2.0 * forward
    rate = 0.2 / (spec.num_chirps * spec.chirp_period_s)  # 0.2 rad over the frame
    body = [(center, 1.0), (center - 0.10 * forward + 0.12 * right, 0.6)]

    # The world cube: each body point turned by rate * t about `up` through the centre, at each chirp instant.
    instants = torch.arange(spec.num_chirps, dtype=torch.float64) * spec.chirp_period_s + spec.t_start_s
    tx = radar.tx_pos.to(torch.float64)
    u = spec.t_start_s + torch.arange(spec.num_samples, dtype=torch.float64) * spec.sample_period_s
    cube = torch.zeros((spec.num_chirps, spec.num_samples), dtype=torch.complex128)
    for point, amplitude in body:
        angle = rate * instants
        offset = point - center
        along_up = (offset @ up) * up
        flat = offset - along_up
        turned = (
            center
            + along_up
            + torch.cos(angle).reshape(-1, 1) * flat
            + torch.sin(angle).reshape(-1, 1) * torch.linalg.cross(up.expand(len(angle), 3), flat.expand(len(angle), 3))
        )
        tau = (2.0 * (turned - tx).norm(dim=-1) / C0).reshape(-1, 1)
        cycles = spec.reference_frequency_hz * tau + spec.slope_hz_per_s * tau * (u - 0.5 * tau)
        cube = cube + amplitude * torch.exp(2j * math.pi * (cycles - torch.floor(cycles)))
    profile = _profile(ProcessingCube.from_tensor(cube.reshape(1, 1, *cube.shape), radar))

    aperture = isar_aperture(Aperture.from_radar(radar), center_m=center, angular_velocity_rad_s=rate * up)
    assert aperture.frame == "body"
    range_resolution = C0 / (2.0 * _bandwidth(radar))
    turn = rate * spec.num_chirps * spec.chirp_period_s
    cross_resolution = radar.wavelength / (2.0 * turn)
    for point, amplitude in body:
        image = backproject(profile, aperture, point.reshape(1, 3))
        assert image.frame == "body"
        assert INTERPOLATION_FLOOR - 0.02 <= float(image.data.abs()) / amplitude <= 1.02
    across, steps = _line(center, right, 3 * cross_resolution)
    power = backproject(profile, aperture, across).power
    assert abs(float(steps[int(power.argmax())])) <= cross_resolution / 20
    assert _half_power_width(power, steps) == pytest.approx(0.886 * cross_resolution, rel=0.06)
    # A wide turn widens the range extent of the k-space arc: the far edge sits at f_max and the near
    # one at f_min cos(dtheta / 2), so the effective band is B + f_min (1 - cos(dtheta / 2)).
    lowest = spec.reference_frequency_hz + spec.slope_hz_per_s * spec.t_start_s
    effective = _bandwidth(radar) + lowest * (1.0 - math.cos(0.5 * turn))
    along, steps = _line(center, forward, 3 * range_resolution)
    assert _half_power_width(backproject(profile, aperture, along).power, steps) == pytest.approx(
        0.886 * C0 / (2.0 * effective), rel=0.04
    )
    # The same turn given as a per-sample rotation table images identically.
    angle = rate * aperture.times_s[:, 0]
    cross = torch.tensor([[0, -up[2], up[1]], [up[2], 0, -up[0]], [-up[1], up[0], 0]], dtype=torch.float64)
    table = (
        torch.cos(angle).reshape(-1, 1, 1) * torch.eye(3, dtype=torch.float64)
        + torch.sin(angle).reshape(-1, 1, 1) * cross
        + (1 - torch.cos(angle)).reshape(-1, 1, 1) * torch.outer(up, up)
    )
    tabled = isar_aperture(Aperture.from_radar(radar), center_m=center, rotation=table)
    torch.testing.assert_close(tabled.tx_m, aperture.tx_m, rtol=0, atol=1e-12)


def _ofdm_radar(num_rx=4):
    from witwin.radar import Ofdm

    waveform = Ofdm(
        subcarrier_spacing=2e6, num_subcarriers=256, cyclic_prefix=1e-7, num_symbols=32, max_expected_delay=5e-8
    )
    config = {
        **PROCESSING_CONFIG,
        "num_tx": 1,
        "num_rx": num_rx,
        "tx_loc": [[0, 0, 0]],
        "rx_loc": [[r, 0, 0] for r in range(num_rx)],
    }
    return Radar.from_dict(config, device="cpu", waveform=waveform)


def test_an_ofdm_cfr_images_with_the_channel_phase_law():
    radar = _ofdm_radar()
    spec = radar.waveform_spec()
    right, forward = _frame(radar)
    target = 3.0 * forward + 0.4 * right
    tx = radar.tx_pos.to(torch.float64).reshape(1, 3)
    rx = radar.rx_pos.to(torch.float64)
    tau = (((target - tx).norm(dim=-1) + (target - rx).norm(dim=-1)) / C0).reshape(1, -1, 1)  # [1, 4, 1]
    n = torch.arange(spec.num_subcarriers, dtype=torch.float64)
    cfr = 0.3 * torch.exp(-2j * math.pi * (spec.reference_frequency_hz + n * spec.subcarrier_spacing_hz) * tau)
    cfr = cfr.expand(spec.num_symbols, -1, -1)
    cube = ProcessingCube.from_tensor(cfr.permute(1, 0, 2).reshape(1, 4, spec.num_symbols, -1), radar)
    profile = range_profile(cube, range_bins=PAD * spec.num_subcarriers)
    aperture = Aperture.from_radar(radar)
    assert aperture.phase_sign == -1 and aperture.curvature_hz_per_s == 0.0
    image = backproject(profile, aperture, target.reshape(1, 3))
    assert INTERPOLATION_FLOOR * 0.3 <= float(image.data.abs()) <= 0.3 * 1.0001
    # The beat-phasor law on the same samples sums the elements out of phase.
    beat = Aperture(**{**_fields(aperture), "phasor": BEAT_PHASOR})
    responses = element_responses(profile, aperture, target)
    assert float((responses - responses.mean()).abs().max()) <= 1e-3 * 0.3
    assert float(_backproject_beat(profile, beat, target)) < 0.9 * 0.3


def _backproject_beat(profile, aperture, target):
    from witwin.radar.processing.imaging import _backproject

    samples = profile.data.reshape(1, aperture.element_count, aperture.slow_count, -1)
    return _backproject(samples, aperture, target.reshape(1, 3), profile.axes, groups=1).abs()


def test_the_image_keeps_a_leading_batch_and_differentiates_through_the_cpu_route():
    radar = _radar(1, 2, samples=32, chirps=8)
    _, forward = _frame(radar)
    cube = _fmcw_cube(radar, [(0.8 * forward, 1.0)])
    profile = _profile(cube)
    stacked = type(profile)(
        data=torch.stack((profile.data, 2 * profile.data)), axes=profile.axes, window=profile.window,
        window_coherent_gain=profile.window_coherent_gain,
    )
    aperture = Aperture.from_radar(radar)
    pixels = 0.8 * forward + torch.linspace(-0.1, 0.1, 5, dtype=torch.float64).reshape(5, 1) * forward
    image = backproject(stacked, aperture, pixels.reshape(5, 1, 3))
    assert isinstance(image, SarImage) and image.data.shape == (2, 5, 1)
    torch.testing.assert_close(image.data[1], 2 * image.data[0])

    # The CPU route is linear in the samples: its autograd VJP is the adjoint identity <J t, g> = <t, J^T g>.
    samples = profile.data.reshape(1, 2, 8, -1).detach().requires_grad_(True)
    law = (0.0, profile.axes.range_bin_m, 1 / C0, aperture.carrier_hz, aperture.curvature_hz_per_s, 1.0, 1 / 16)
    out = _backproject_torch(samples, aperture.tx_m, aperture.rx_m, pixels, 1, law)
    generator = torch.Generator().manual_seed(0)
    cotangent = torch.randn(out.shape, dtype=out.dtype, generator=generator)
    (grad,) = torch.autograd.grad(out, samples, cotangent)
    tangent = torch.randn(samples.shape, dtype=samples.dtype, generator=generator)
    forward_t = _backproject_torch(tangent, aperture.tx_m, aperture.rx_m, pixels, 1, law)
    lhs = torch.vdot(cotangent.reshape(-1), forward_t.reshape(-1))
    rhs = torch.vdot(grad.reshape(-1), tangent.reshape(-1))
    torch.testing.assert_close(lhs, rhs, rtol=1e-10, atol=1e-12)


def test_the_geometry_carries_no_derivative_and_mismatches_are_refused():
    radar = _radar(1, 2, samples=32, chirps=8)
    _, forward = _frame(radar)
    profile = _profile(_fmcw_cube(radar, [(0.8 * forward, 1.0)]))
    aperture = Aperture.from_radar(radar)
    with pytest.raises(RuntimeError, match="not differentiable"):
        backproject(profile, aperture, (0.8 * forward).reshape(1, 3).requires_grad_(True))
    with pytest.raises(RuntimeError, match="not differentiable"):
        Aperture(**{**_fields(aperture), "tx_m": aperture.tx_m.clone().requires_grad_(True)})
    with pytest.raises(ValueError, match="slow-time samples"):
        backproject(profile, Aperture.from_radar(_radar(1, 2, samples=32, chirps=4)), forward.reshape(1, 3))
    with pytest.raises(ValueError, match="array"):
        backproject(profile, Aperture.from_radar(_radar(1, 3, samples=32, chirps=8)), forward.reshape(1, 3))
    with pytest.raises(ValueError, match="phase law"):
        backproject(profile, Aperture.from_radar(_ofdm_radar(num_rx=2)), forward.reshape(1, 3))
    with pytest.raises(ValueError, match="exactly one"):
        isar_aperture(aperture, center_m=forward)
    with pytest.raises(TypeError, match="float64"):
        Aperture(**{**_fields(aperture), "tx_m": aperture.tx_m.float()})


def _fields(aperture: Aperture) -> dict:
    return {name: getattr(aperture, name) for name in Aperture.__dataclass_fields__}


# ---------------------------------------------------------------------------
# GPU: the native family against the CPU route, its derivatives, its budget.
# ---------------------------------------------------------------------------


def _random_problem(dtype, seed=0):
    radar = _radar(2, 3, samples=32, chirps=6)
    right, forward = _frame(radar)
    cube = _fmcw_cube(radar, [(0.9 * forward + 0.05 * right, 1.0), (1.3 * forward, 0.4)], velocity=4.0 * right)
    profile = _profile(cube)
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(profile.data.shape, dtype=torch.complex128, generator=generator) * 0.1
    profile = type(profile)(
        data=(profile.data + noise).to(dtype), axes=profile.axes, window=profile.window,
        window_coherent_gain=profile.window_coherent_gain,
    )
    aperture = Aperture.from_radar(radar, velocity_mps=4.0 * right)
    grid_r = torch.linspace(0.6, 1.6, 23, dtype=torch.float64).reshape(-1, 1, 1) * forward
    grid_x = torch.linspace(-0.3, 0.3, 17, dtype=torch.float64).reshape(1, -1, 1) * right
    return profile, aperture, grid_r + grid_x


def _on_cuda(profile):
    return type(profile)(
        data=profile.data.cuda(), axes=profile.axes, window=profile.window,
        window_coherent_gain=profile.window_coherent_gain,
    )


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_the_native_image_equals_the_cpu_route(dtype):
    profile, aperture, pixels = _random_problem(dtype)
    host = backproject(profile, aperture, pixels).data
    device = backproject(_on_cuda(profile), aperture, pixels).data.cpu()
    tolerance = 1e-12 if dtype == torch.complex128 else 2e-5
    torch.testing.assert_close(device, host, rtol=tolerance, atol=tolerance * float(host.abs().max()))
    host_elements = element_responses(profile, aperture, pixels[11, 8])
    device_elements = element_responses(_on_cuda(profile), aperture, pixels[11, 8]).cpu()
    torch.testing.assert_close(device_elements, host_elements, rtol=tolerance, atol=tolerance)


@pytest.mark.gpu
def test_the_native_vjp_and_jvp_are_the_cpu_routes_derivatives_and_match_finite_differences():
    import torch.autograd.forward_ad as fwAD

    profile, aperture, pixels = _random_problem(torch.complex128, seed=1)
    generator = torch.Generator().manual_seed(2)
    cotangent = torch.randn((*pixels.shape[:-1],), dtype=torch.complex128, generator=generator)
    tangent = torch.randn(profile.data.shape, dtype=torch.complex128, generator=generator)

    def image(data):
        return backproject(type(profile)(data=data, axes=profile.axes, window=profile.window,
                                         window_coherent_gain=profile.window_coherent_gain), aperture, pixels).data

    host_samples = profile.data.clone().requires_grad_(True)
    (host_grad,) = torch.autograd.grad(image(host_samples), host_samples, cotangent)
    device_samples = profile.data.cuda().requires_grad_(True)
    (device_grad,) = torch.autograd.grad(image(device_samples), device_samples, cotangent.cuda())
    torch.testing.assert_close(device_grad.cpu(), host_grad, rtol=1e-11, atol=1e-12)

    with fwAD.dual_level():
        dual = fwAD.make_dual(profile.data.cuda(), tangent.cuda())
        jvp = fwAD.unpack_dual(image(dual)).tangent.cpu()
    step = 1e-6
    difference = (image(profile.data + step * tangent) - image(profile.data - step * tangent)) / (2 * step)
    torch.testing.assert_close(jvp, difference, rtol=1e-7, atol=1e-9)
    # The adjoint identity ties the two modes together.
    lhs = torch.vdot(cotangent.reshape(-1), jvp.reshape(-1))
    rhs = torch.vdot(device_grad.cpu().reshape(-1), tangent.reshape(-1))
    torch.testing.assert_close(lhs, rhs, rtol=1e-10, atol=1e-10)

    from witwin.radar.processing.imaging import _Backprojection

    samples = profile.data.cuda().reshape(1, 6, 6, -1).requires_grad_(True)
    out = _Backprojection.apply(
        samples, aperture.tx_m.cuda(), aperture.rx_m.cuda(), pixels.reshape(-1, 3).cuda(), 1,
        (0.0, profile.axes.range_bin_m, 1 / C0, aperture.carrier_hz, aperture.curvature_hz_per_s, 1.0, 1 / 36),
    )
    with pytest.raises(NotImplementedError, match="first-order only"):
        torch.autograd.grad(out.abs().sum(), samples, create_graph=True)


def _profile_names(fn):
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
        fn()
        torch.cuda.synchronize()
    return [event.name for event in trace.events()]


@pytest.mark.gpu
def test_a_device_image_is_one_launch_with_no_host_round_trip():
    profile, aperture, pixels = _random_problem(torch.complex64)
    profile = _on_cuda(profile)
    aperture = aperture.to("cuda")
    pixels = pixels.cuda()
    backproject(profile, aperture, pixels)
    names = _profile_names(lambda: backproject(profile, aperture, pixels))
    kernels = [name for name in names if "backprojection" in name and "Launch" not in name]
    assert sum("backprojection_kernel" in name for name in kernels) == 1, kernels
    assert not any("DtoH" in name or "HtoD" in name for name in names), [n for n in names if "Memcpy" in n]
    assert names.count("cudaStreamSynchronize") == 0
    harness = _profile_names(lambda: None)
    assert names.count("cudaDeviceSynchronize") == harness.count("cudaDeviceSynchronize")


# ---------------------------------------------------------------------------
# GPU, end to end through Radar.simulate and Channel.
# ---------------------------------------------------------------------------


def _scene_config(**overrides):
    return {
        **PROCESSING_CONFIG,
        "num_tx": 1,
        "num_rx": 1,
        "tx_loc": [[0, 0, 0]],
        "rx_loc": [[0, 0, 0]],
        "adc_samples": 256,
        "chirp_per_frame": 64,
        "output_domain": "beat",
        **overrides,
    }


def _simulated_profile(radar, targets, **verb):
    from conftest import empty_world

    result = radar.simulate(empty_world(), targets, times=(0.0,), los=True, reflections=0, **verb)
    return _profile(result.frame(0).processing_cube())


def _peak_level(profile, radar, point, aperture) -> float:
    """The mean over slow time of the compressed peak magnitude near ``point``'s range (the coherent ceiling)."""

    half_path = 0.5 * ((point - aperture.tx_m).norm(dim=-1) + (point - aperture.rx_m).norm(dim=-1))
    centre = int(round(float(half_path.mean()) / profile.axes.range_bin_m))
    window = profile.data[..., centre - 3 * PAD : centre + 3 * PAD + 1].abs()
    return float(window.amax(dim=-1).mean())


def _check_focus(profile, radar, aperture, point, range_resolution, cross_resolution, right, forward):
    """Peak position on both lines, the coherent level, and return the two -3 dB widths."""

    point = point.cpu()
    along, steps = _line(point, forward, 3 * range_resolution)
    across, steps_x = _line(point, right, 3 * cross_resolution)
    power_r = backproject(profile, aperture, along).power.cpu()
    power_x = backproject(profile, aperture, across).power.cpu()
    assert abs(float(steps[int(power_r.argmax())])) <= range_resolution / 10
    assert abs(float(steps_x[int(power_x.argmax())])) <= cross_resolution / 10
    coherence = math.sqrt(float(power_r.max())) / _peak_level(profile, radar, point, aperture)
    assert 0.95 <= coherence <= 1.01, coherence
    return _half_power_width(power_r, steps), _half_power_width(power_x, steps_x)


def _verbs_take_a_pose() -> bool:
    import inspect

    return "pose" in inspect.signature(Radar.simulate).parameters


@pytest.mark.gpu
@pytest.mark.parametrize("motion", ["targets", "pose"])
def test_a_moving_radar_images_two_simulated_point_targets(motion):
    """``Radar.simulate`` -> ``range_profile`` -> ``backproject``: a straight track past two point targets.

    ``motion="targets"`` moves both targets by ``-v`` instead of the radar by
    ``+v``: the same relative geometry, and the target frame is the world at
    ``t = 0``. ``motion="pose"`` moves the radar with the verbs' ``pose=``.
    """

    from conftest import scene_radar

    from witwin.radar import PointTargets

    if motion == "pose" and not _verbs_take_a_pose():
        pytest.skip("Radar.simulate takes no pose= on this branch")
    radar = scene_radar(_scene_config())
    spec = radar.waveform_spec()
    right, forward = _frame(radar)
    speed = 12.0
    velocity = speed * right
    near = torch.as_tensor(radar.position, dtype=torch.float64) + 1.0 * forward
    far = near + 0.35 * forward + 0.12 * right
    positions = torch.stack((near, far)).to(torch.float32).cuda()
    if motion == "targets":
        drift = velocity.to(torch.float32).cuda()
        targets = PointTargets(positions=positions, rcs=1.0, trajectory=lambda t: positions - drift * float(t))
        profile = _simulated_profile(radar, targets)
    else:
        targets = PointTargets(positions=positions, rcs=1.0)
        profile = _simulated_profile(radar, targets, pose=lambda t: (torch.eye(3), velocity * float(t)))
    aperture = Aperture.from_radar(radar, velocity_mps=velocity)

    range_resolution = C0 / (2.0 * _bandwidth(radar))
    cross_resolution = radar.wavelength * 1.0 / (2.0 * spec.num_chirps * speed * spec.chirp_period_s)
    width_r, width_x = _check_focus(profile, radar, aperture, near, range_resolution, cross_resolution, right, forward)
    _check_focus(profile, radar, aperture, far, range_resolution, 1.35 * cross_resolution, right, forward)
    assert width_r == pytest.approx(0.886 * range_resolution, rel=0.06)
    assert width_x == pytest.approx(0.886 * cross_resolution, rel=0.06)


@pytest.mark.gpu
def test_a_simulated_turning_target_images_in_its_body_frame():
    from conftest import scene_radar

    from witwin.radar import PointTargets

    radar = scene_radar(_scene_config(chirp_per_frame=128))
    spec = radar.waveform_spec()
    right, forward = _frame(radar)
    _, world_from_local = radar._world_from_local_matrix(device="cpu", dtype=torch.float64)
    up = world_from_local[:, 1]
    center = torch.as_tensor(radar.position, dtype=torch.float64) + 2.0 * forward
    rate = 0.2 / (spec.num_chirps * spec.chirp_period_s)
    body = torch.stack((center, center - 0.10 * forward + 0.12 * right))
    offset = body - center

    def trajectory(time_s):
        angle = rate * float(time_s)
        turned = math.cos(angle) * offset + math.sin(angle) * torch.linalg.cross(up.expand_as(offset), offset)
        turned = turned + (1 - math.cos(angle)) * (offset @ up).reshape(-1, 1) * up
        return (center + turned).to(torch.float32).cuda()

    profile = _simulated_profile(radar, PointTargets(positions=trajectory(0.0), rcs=1.0, trajectory=trajectory))
    aperture = isar_aperture(Aperture.from_radar(radar), center_m=center, angular_velocity_rad_s=rate * up)
    turn = rate * spec.num_chirps * spec.chirp_period_s
    cross_resolution = radar.wavelength / (2.0 * turn)
    lowest = spec.reference_frequency_hz + spec.slope_hz_per_s * spec.t_start_s
    range_resolution = C0 / (2.0 * (_bandwidth(radar) + lowest * (1.0 - math.cos(0.5 * turn))))
    width_r, width_x = _check_focus(profile, radar, aperture, center, range_resolution, cross_resolution, right, forward)
    _check_focus(profile, radar, aperture, body[1], range_resolution, 1.3 * cross_resolution, right, forward)
    assert width_x == pytest.approx(0.886 * cross_resolution, rel=0.06)
    assert width_r == pytest.approx(0.886 * range_resolution, rel=0.06)
