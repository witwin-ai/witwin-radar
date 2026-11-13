"""The adapter's discovery features and the routes built on them.

* C-17..C-20: Channel's completeness flag reaches ``path_set_complete``, the
  ``discovery=`` source and ``max_paths`` reach Channel, and a straight-segment
  leg runs the endpoint-separation check.
* M3-7: the per-pair round-trip budget is refused by name before the join.
* M3-2: a primal run reads rediscovered rows straight from the discovery, with
  the same numbers the replay route produces, and admits ``transmission`` and
  ``diffraction``; both are refused where rows would be replayed.
* M3-6: the site-less TX->RX leg (environment returns and element coupling).

Levels are compared as ratios; received amplitudes are ~1e-6, far below
``pytest.approx``'s default absolute tolerance.
"""

from __future__ import annotations

import math

import pytest
import torch

pytest.importorskip("witwin.channel")

from support import oracle_scenes as sc

pytestmark = pytest.mark.gpu

C = 299_792_458.0
WAVELENGTH_M = C / sc.CARRIER_HZ
POWER_W = 1e-2  # 10 dBm, oracle_scenes.radar's default


def _pec():
    from witwin.core import PhysicalMaterial

    return PhysicalMaterial.perfect_conductor()


def _wall(x_m: float, *, half: float = 3.0, material=None):
    corners = ((x_m, -half, -half), (x_m, half, -half), (x_m, half, half), (x_m, -half, half))
    return sc.plane_scene(corners=corners, material=_pec() if material is None else material)


def _side_wall_scene():
    return sc.plane_scene(corners=sc.side_wall_corners(z_m=1.5, x_range=(-2.0, 12.0), y_half=4.0), material=_pec())


def _targets(positions, **fields):
    from witwin.radar import PointTargets

    return PointTargets(positions=positions, rcs=1.0, **fields)


def _moving_targets():
    from witwin.radar import PointTargets

    def trajectory(time_s):
        return torch.tensor([[6.0 + 2.0 * float(time_s), 0.3, 0.0]], dtype=torch.float32, device="cuda")

    return PointTargets(positions=trajectory(0.0), rcs=1.0, trajectory=trajectory)


def _replaying(monkeypatch):
    """Force the replay route on a primal run: discover, prepare, replay."""

    from witwin.radar import simulation

    original = simulation.WorldHalf.__init__

    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.discovered_rows = False

    monkeypatch.setattr(simulation.WorldHalf, "__init__", init)


def _channel_calls(monkeypatch):
    from witwin.radar import channel as radar_channel

    consumer = radar_channel.consumer
    calls: dict[str, int] = {}
    for name in ("evaluate", "prepare_fixed_topology", "reevaluate", "check_endpoint_separation"):
        original = getattr(consumer, name)

        def counting(*args, _name=name, _original=original, **kwargs):
            calls[_name] = calls.get(_name, 0) + 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(consumer, name, counting)
    return calls


# ---------------------------------------------------------------------------
# M3-2: discovered rows are the replay's rows
# ---------------------------------------------------------------------------


MOTIONS = ("chirp", "adc", "adaptive")


@pytest.mark.parametrize("kind", MOTIONS)
def test_m3_2_primal_rows_from_discovery_equal_the_replay_route(monkeypatch, kind):
    from witwin.radar import Motion

    radar = sc.radar(sc.beat_fmcw(samples=8, chirps=2))
    motion = getattr(Motion, kind)()
    arguments = {"times": (0.0,), "los": True, "reflections": 1, "motion": motion}
    direct = radar.simulate(_side_wall_scene(), _moving_targets(), **arguments)
    with monkeypatch.context() as patch:
        _replaying(patch)
        replayed = radar.simulate(_side_wall_scene(), _moving_targets(), **arguments)
    assert torch.equal(direct.cube, replayed.cube)
    assert torch.equal(direct.last_radar_paths.total_delay_s, replayed.last_radar_paths.total_delay_s)
    assert torch.equal(direct.last_radar_paths.complex_transfer_ref, replayed.last_radar_paths.complex_transfer_ref)


def test_m3_2_a_static_world_reads_discovery_once_and_never_prepares(monkeypatch):
    calls = _channel_calls(monkeypatch)
    radar = sc.radar(sc.beat_fmcw(samples=8))
    radar.simulate(_side_wall_scene(), _targets([[6.0, 0.3, 0.0]]), times=(0.0, 1e-3, 2e-3), reflections=1)
    assert calls.get("evaluate") == 2
    assert calls.get("prepare_fixed_topology", 0) == 0
    assert calls.get("reevaluate", 0) == 0


def test_m3_2_a_rediscovering_primal_frame_makes_one_channel_call_per_leg(monkeypatch):
    """Per rediscovered observation: two discoveries and nothing else (was 2 + 2 prepares + 2 replays)."""

    from witwin.radar import Motion

    calls = _channel_calls(monkeypatch)
    radar = sc.radar(sc.beat_fmcw(samples=4, chirps=2))
    radar.simulate(_side_wall_scene(), _moving_targets(), times=(0.0,), reflections=1, motion=Motion.chirp())
    observations = 2  # chirps x transmitters
    assert calls.get("evaluate") == 2 * observations
    assert calls.get("prepare_fixed_topology", 0) == 0
    assert calls.get("reevaluate", 0) == 0


def test_m3_2_a_differentiable_run_still_replays(monkeypatch):
    calls = _channel_calls(monkeypatch)
    radar = sc.radar(sc.beat_fmcw(samples=4))
    positions = torch.tensor([[6.0, 0.3, 0.0]], device="cuda", requires_grad=True)
    result = radar.simulate(_side_wall_scene(), _targets(positions), times=(0.0,), reflections=1, ad_mode="vjp")
    result.cube.abs().sum().backward()
    assert positions.grad is not None and torch.isfinite(positions.grad).all()
    assert calls.get("reevaluate") == 2 and calls.get("prepare_fixed_topology") == 2


def _slab():
    from witwin.core import PhysicalMaterial

    return PhysicalMaterial(eps_r=4.0, sigma_e=0.0, thickness_m=0.1)


def _airy_slab_transmission(eps_r: float, thickness_m: float, frequency_hz: float) -> complex:
    """Normal-incidence field transmission of a lossless slab (Airy sum), float64."""

    import cmath

    n = math.sqrt(eps_r)
    r = (1.0 - n) / (1.0 + n)
    delta = 2.0 * math.pi * frequency_hz / C * n * thickness_m
    return (1.0 - r * r) * cmath.exp(-1j * delta) / (1.0 - r * r * cmath.exp(-2j * delta))


def _radar_equation_amplitude(d_in: float, d_out: float, rcs: float = 1.0) -> float:
    """``|C_rt| = sqrt(P_t) lambda sqrt(4 pi sigma) / ((4 pi)^2 d_in d_out)`` at unit gains."""

    return math.sqrt(POWER_W) * WAVELENGTH_M * math.sqrt(4.0 * math.pi * rcs) / ((4.0 * math.pi) ** 2 * d_in * d_out)


def test_m3_2_through_wall_echo_matches_the_analytic_slab_twice():
    radar = sc.radar(sc.beat_fmcw(samples=8))
    paths = radar.trace(
        _wall(5.0, material=_slab()), _targets([[10.0, 0.0, 0.0]]), times=(0.0,), reflections=0, transmission=True
    )
    rows = paths.rows()
    assert rows.path_count == 1
    transmitted = abs(_airy_slab_transmission(4.0, 0.1, sc.CARRIER_HZ)) ** 2
    expected = transmitted * _radar_equation_amplitude(10.0, 10.0)
    assert float(rows.complex_transfer_ref.abs()[0]) / expected == pytest.approx(1.0, rel=1e-4, abs=0.0)
    assert float(rows.total_delay_s[0]) == pytest.approx(20.0 / C, rel=1e-6)
    assert paths.transmission and paths.transmission_dropped_rows == 0


def _utd_knife_edge_amplitude(s_in: float, s_out: float, phi_in: float, phi_out: float) -> float:
    """One-way UTD amplitude per unit source power, PEC half-plane, soft (E parallel to the edge).

    Kouyoumjian-Pathak with ``n = 2`` and the spherical-wave distance
    parameter ``L = s s' / (s + s')``; float64 throughout.
    """

    import cmath

    from scipy.special import fresnel

    k = 2.0 * math.pi / WAVELENGTH_M

    def transition(x: float) -> complex:
        root = math.sqrt(x)
        sine, cosine = fresnel(root * math.sqrt(2.0 / math.pi))
        tail = math.sqrt(math.pi / 2.0) * ((0.5 - cosine) - 1j * (0.5 - sine))
        return 2j * root * cmath.exp(1j * x) * tail

    def a(beta: float, sign: int) -> float:
        winding = round((beta + sign * math.pi) / (4.0 * math.pi))
        return 2.0 * math.cos((4.0 * math.pi * winding - beta) / 2.0) ** 2

    length = s_in * s_out / (s_in + s_out)

    def term(beta: float) -> complex:
        return (1.0 / math.tan((math.pi + beta) / 4.0)) * transition(k * length * a(beta, 1)) + (
            1.0 / math.tan((math.pi - beta) / 4.0)
        ) * transition(k * length * a(beta, -1))

    coefficient = (
        -cmath.exp(-1j * math.pi / 4.0)
        / (4.0 * math.sqrt(2.0 * math.pi * k))
        * (term(phi_out - phi_in) - term(phi_out + phi_in))
    )
    return WAVELENGTH_M / (4.0 * math.pi * s_in) * abs(coefficient) * math.sqrt(s_in / (s_out * (s_in + s_out)))


def test_m3_2_knife_edge_echo_in_the_shadow_matches_utd_twice():
    """Radar at the origin, site at (10, 0, 0), a PEC half-plane z <= 1 at x = 5.

    The line of sight is blocked; the shadow-region echo is the edge-diffracted
    round trip over the edge 1 m above the line. Polarization "up" is world +y,
    parallel to the edge (soft).
    """

    pytest.importorskip("scipy")
    plate = sc.plane_scene(corners=((5.0, -40, -40), (5.0, 40, -40), (5.0, 40, 1.0), (5.0, -40, 1.0)), material=_pec())
    radar = sc.radar(sc.beat_fmcw(samples=8))
    paths = radar.trace(plate, _targets([[10.0, 0.0, 0.0]]), times=(0.0,), reflections=0, diffraction=True)
    rows = paths.rows()
    near = int(torch.argmin(rows.total_delay_s))
    edge = math.sqrt(26.0)
    assert float(rows.total_delay_s[near]) == pytest.approx(4.0 * edge / C, rel=1e-6)
    phi_in = math.atan2(5.0, 1.0)
    one_way = _utd_knife_edge_amplitude(edge, edge, phi_in, 2.0 * math.pi - phi_in)
    expected = math.sqrt(POWER_W) * one_way**2 * math.sqrt(4.0 * math.pi) / WAVELENGTH_M
    assert float(rows.complex_transfer_ref.abs()[near]) / expected == pytest.approx(1.0, rel=0.01, abs=0.0)


def test_m3_2_unfreezable_components_are_refused_where_rows_would_be_replayed():
    from witwin.radar import Motion

    radar = sc.radar(sc.beat_fmcw(samples=4))
    scene = _wall(5.0, material=_slab())
    positions = torch.tensor([[10.0, 0.0, 0.0]], device="cuda", requires_grad=True)
    with pytest.raises(NotImplementedError, match="transmission"):
        radar.simulate(scene, _targets(positions), times=(0.0,), transmission=True, ad_mode="vjp")
    with pytest.raises(NotImplementedError, match="rediscover"):
        radar.simulate(
            scene, _moving_targets(), times=(0.0,), transmission=True, motion=Motion.chirp(rediscover_every_frames=2)
        )


# ---------------------------------------------------------------------------
# C-17..C-20 and M3-7
# ---------------------------------------------------------------------------


def test_c17_c18_discovery_source_reaches_channel_and_its_completeness_is_published():
    radar = sc.radar(sc.beat_fmcw(samples=4))
    targets = _targets([[6.0, 0.3, 0.0]])
    exhaustive = radar.simulate(_side_wall_scene(), targets, times=(0.0,), discovery="exhaustive")
    assert exhaustive.discovery_complete and exhaustive.path_set_complete
    rays = radar.simulate(_side_wall_scene(), targets, times=(0.0,), discovery="rays")
    assert not rays.discovery_complete
    assert not rays.path_set_complete
    with pytest.raises(ValueError, match="discovery"):
        radar.simulate(_side_wall_scene(), targets, times=(0.0,), discovery="guess")


def test_c20_max_paths_is_forwarded_and_channel_refuses_rather_than_truncates():
    radar = sc.radar(sc.beat_fmcw(samples=4))
    with pytest.raises(RuntimeError, match="max_paths"):
        radar.simulate(_side_wall_scene(), _targets([[6.0, 0.3, 0.0]]), times=(0.0,), max_paths=1)


def test_c19_a_site_on_an_element_is_refused_by_the_separation_check():
    radar = sc.radar(sc.beat_fmcw(samples=4))
    with pytest.raises(ValueError, match="coincides"):
        radar.simulate(sc.empty_scene(), _targets([[0.0, 0.0, 0.0]]), times=(0.0,), reflections=0)


def test_m3_7_max_round_trips_names_the_sensor_pair_before_the_join(monkeypatch):
    from witwin.radar import paths as radar_paths

    composed: list[int] = []
    original = radar_paths.TwoWayComposer.compose

    def counting(self, *args, **kwargs):
        composed.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(radar_paths.TwoWayComposer, "compose", counting)
    radar = sc.radar(sc.beat_fmcw(samples=4))
    targets = _targets([[6.0, 0.3, 0.0]])
    # LOS and one side-wall bounce on each leg: 2 x 2 = 4 round trips.
    ok = radar.trace(_side_wall_scene(), targets, times=(0.0,), max_round_trips=4)
    assert ok.rows().path_count == 4
    composed.clear()
    with pytest.raises(ValueError, match=r"max_round_trips=3 .*transmitter 1000000, receiver 2000000.* 4 round-trip"):
        radar.trace(_side_wall_scene(), targets, times=(0.0,), max_round_trips=3)
    assert not composed


# ---------------------------------------------------------------------------
# M3-6: the site-less leg
# ---------------------------------------------------------------------------


def test_m3_6_a_wall_facing_the_radar_echoes_at_2d_over_c_at_the_image_level():
    from witwin.radar.paths import NO_SITE

    radar = sc.radar(sc.beat_fmcw(samples=8))
    paths = radar.trace(
        _wall(4.0), _targets([[2.0, 0.5, 0.3]]), times=(0.0,), los=True, reflections=1, environment=True
    )
    rows = paths.rows()
    site = rows.topology.site_id
    echo = torch.nonzero(site == NO_SITE).reshape(-1)
    assert echo.numel() == 1
    row = int(echo[0])
    assert float(rows.total_delay_s[row]) == pytest.approx(8.0 / C, rel=1e-6)
    # Image method: the PEC wall returns the transmitter's field from its image
    # 2d away with |Gamma| = 1, so |C| = sqrt(P) lambda / (4 pi 2d). Measured
    # agreement is 9e-5; the bound leaves twice that.
    expected = math.sqrt(POWER_W) * WAVELENGTH_M / (4.0 * math.pi * 8.0)
    assert float(rows.complex_transfer_ref.abs()[row]) / expected == pytest.approx(1.0, rel=2e-4, abs=0.0)
    # The site-less row sits in its pair's segment, after that pair's site rows.
    assert row == rows.path_count - 1
    assert paths.environment and not paths.coupling


def test_m3_6_the_environment_echo_is_in_the_cube_and_off_by_default():
    radar = sc.radar(sc.beat_fmcw(samples=16))
    targets = _targets([[2.0, 0.5, 0.3]])
    without = radar.simulate(_wall(4.0), targets, times=(0.0,))
    with_echo = radar.simulate(_wall(4.0), targets, times=(0.0,), environment=True)
    assert without.last_radar_paths.path_count + 1 == with_echo.last_radar_paths.path_count
    assert not torch.equal(without.cube, with_echo.cube)


def test_m3_6_coupling_is_refused_for_co_located_elements_and_published_for_separated_ones():
    radar = sc.radar(sc.beat_fmcw(samples=4))
    targets = _targets([[6.0, 0.3, 0.0]])
    with pytest.raises(ValueError, match="coincides"):
        radar.trace(sc.empty_scene(), targets, times=(0.0,), reflections=0, coupling=True)
    apart = sc.radar(sc.beat_fmcw(samples=4), tx=((0.0, 0.0, 0.0),), rx=((0.0, 0.0, 0.01),))
    rows = apart.trace(sc.empty_scene(), targets, times=(0.0,), reflections=0, coupling=True).rows()
    direct = int(torch.nonzero(rows.topology.site_id == -1).reshape(-1)[0])
    assert float(rows.total_delay_s[direct]) == pytest.approx(0.01 / C, rel=1e-4)
    expected = math.sqrt(POWER_W) * WAVELENGTH_M / (4.0 * math.pi * 0.01)
    assert float(rows.complex_transfer_ref.abs()[direct]) / expected == pytest.approx(1.0, rel=1e-4, abs=0.0)


def test_m3_6_environment_rows_are_classified_and_select_splits_the_echo():
    from witwin.radar.paths import DIRECT_LEAKAGE, ENVIRONMENT_CLUTTER, MULTI_INTERACTION, TARGET

    radar = sc.radar(sc.beat_fmcw(samples=16), tx=((0.0, 0.0, 0.0),), rx=((0.0, 0.0, 0.01),))
    paths = radar.trace(_wall(4.0), _targets([[2.0, 0.5, 0.3]]), times=(0.0,), environment=True, coupling=True)
    index = paths.component_index()
    # Site rows: LOS-LOS is the target; the three that touch the wall are clutter.
    assert index.count(TARGET) == 1
    assert index.count(ENVIRONMENT_CLUTTER) == 3 + 1
    assert index.count(DIRECT_LEAKAGE) == 1
    assert index.count(MULTI_INTERACTION) == 0
    full = radar.echo(paths).cube
    parts = sum(radar.echo(paths.select(name)).cube for name in (TARGET, ENVIRONMENT_CLUTTER, DIRECT_LEAKAGE))
    assert torch.allclose(parts, full, rtol=1e-5, atol=float(full.abs().max()) * 1e-6)


def test_m3_6_direct_rows_survive_the_slot_batched_and_replayed_routes(monkeypatch):
    from witwin.radar import Motion

    radar = sc.radar(sc.beat_fmcw(samples=4, chirps=2), tx=((0.0, 0.0, 0.0),), rx=((0.0, 0.0, 0.01),))
    arguments = {"times": (0.0,), "environment": True, "coupling": True, "motion": Motion.chirp()}
    direct = radar.simulate(_wall(8.0), _moving_targets(), **arguments)
    with monkeypatch.context() as patch:
        _replaying(patch)
        replayed = radar.simulate(_wall(8.0), _moving_targets(), **arguments)
    assert torch.equal(direct.cube, replayed.cube)
    assert direct.last_propagation.direct is not None


# ---------------------------------------------------------------------------
# M3-8: OFDM consumes one transport column per subcarrier
# ---------------------------------------------------------------------------


def _ofdm(subcarriers: int = 64, spacing: float = 10e6, *, symbols: int = 1, **fields):
    from witwin.radar import Ofdm

    return Ofdm(
        subcarrier_spacing=spacing,
        num_subcarriers=subcarriers,
        cyclic_prefix=2e-7,
        num_symbols=symbols,
        max_expected_delay=1e-7,
        **fields,
    )


def _airy_slab_reflection(eps_r: float, thickness_m: float, frequency_hz: float) -> complex:
    """Normal-incidence field reflection of a lossless slab in free space, float64."""

    import cmath

    n = math.sqrt(eps_r)
    r = (1.0 - n) / (1.0 + n)
    phase = cmath.exp(-2j * 2.0 * math.pi * frequency_hz / C * n * thickness_m)
    return r * (1.0 - phase) / (1.0 - r * r * phase)


def test_m3_8_ofdm_subcarrier_response_of_a_slab_bounce_matches_the_analytic_slab():
    """A slab wall 4 m in front of the radar, seen through the site-less leg.

    The 640 MHz band spans most of one Airy fringe of the 0.1 m, eps_r = 4 slab,
    so the reflection magnitude changes by tens of percent across it; the
    narrowband law would hold it at the reference value.
    """

    from witwin.radar.paths import NO_SITE

    radar = sc.radar(_ofdm())
    paths = radar.trace(
        _wall(4.0, material=_slab()),
        _targets([[-3.0, 1.0, 0.5]]),
        times=(0.0,),
        los=True,
        reflections=1,
        environment=True,
        wideband=True,
    )
    rows = paths.rows()
    echo = int(torch.nonzero(rows.topology.site_id == NO_SITE).reshape(-1)[0])
    band = rows.complex_transfer_band
    assert band is not None and tuple(band.shape) == (rows.path_count, 64)
    assert torch.equal(band[:, 0], rows.complex_transfer_ref)
    measured = band[echo].abs().double().cpu()
    frequencies = [sc.CARRIER_HZ + n * 10e6 for n in range(64)]
    expected = torch.tensor(
        [
            math.sqrt(POWER_W) * C / (4.0 * math.pi * f * 8.0) * abs(_airy_slab_reflection(4.0, 0.1, f))
            for f in frequencies
        ],
        dtype=torch.float64,
    )
    assert float(expected.max() / expected.min()) > 1.2
    # Measured worst subcarrier 6.3e-4 (float32 transport); the narrowband law
    # would be off by the full >20 % variation.
    assert torch.allclose(measured / expected, torch.ones_like(expected), rtol=1e-3, atol=0.0)


def test_m3_8_the_wideband_cube_uses_the_band_and_the_narrowband_cube_does_not():
    radar = sc.radar(_ofdm(subcarriers=16))
    scene = _wall(4.0, material=_slab())
    targets = _targets([[2.0, 0.5, 0.3]])
    narrow = radar.simulate(scene, targets, times=(0.0,), environment=True)
    wide = radar.simulate(scene, targets, times=(0.0,), environment=True, wideband=True)
    assert wide.last_radar_paths.complex_transfer_band is not None
    assert narrow.last_radar_paths.complex_transfer_band is None
    # Subcarrier 0 is the reference frequency in both; the rest differ by the
    # material and spreading response the narrowband law freezes.
    assert torch.allclose(wide.cube[..., 0], narrow.cube[..., 0], rtol=1e-5, atol=0.0)
    assert not torch.allclose(wide.cube[..., -1], narrow.cube[..., -1], rtol=1e-3, atol=0.0)


def test_m3_8_wideband_is_refused_for_fmcw_by_name():
    radar = sc.radar(sc.beat_fmcw(samples=4))
    with pytest.raises(NotImplementedError, match="R-ADR-015"):
        radar.simulate(_side_wall_scene(), _targets([[6.0, 0.3, 0.0]]), times=(0.0,), wideband=True)


def test_m3_8_a_slot_batched_moving_scene_carries_the_band_through_slot_rows():
    """Per-symbol observations of a moving target compose as one strided slot group, band included."""

    from witwin.radar import Motion, PointTargets

    def trajectory(time_s):
        return torch.tensor([[2.0 + 50.0 * float(time_s), 0.5, 0.3]], dtype=torch.float32, device="cuda")

    def targets():
        return PointTargets(positions=trajectory(0.0), rcs=1.0, trajectory=trajectory)

    radar = sc.radar(_ofdm(subcarriers=16, symbols=4))
    scene = _wall(4.0, material=_slab())
    arguments = {"times": (0.0,), "environment": True, "motion": Motion.chirp()}
    wide = radar.trace(scene, targets(), wideband=True, **arguments)
    narrow = radar.trace(scene, targets(), **arguments)
    for observation in range(4):
        rows = wide.rows(observation=observation)
        assert rows.complex_transfer_band is not None
        assert torch.equal(rows.complex_transfer_band[:, 0], rows.complex_transfer_ref)
        assert torch.equal(rows.complex_transfer_ref, narrow.rows(observation=observation).complex_transfer_ref)
    assert torch.allclose(radar.echo(wide).cube[..., 0], radar.echo(narrow).cube[..., 0], rtol=1e-5, atol=0.0)


def test_m3_8_wideband_is_refused_for_time_domain_ofdm_by_name():
    radar = sc.radar(_ofdm(subcarriers=16, output="time"))
    with pytest.raises(NotImplementedError, match="time-domain"):
        radar.simulate(_wall(4.0, material=_slab()), _targets([[2.0, 0.5, 0.3]]), times=(0.0,), wideband=True)
