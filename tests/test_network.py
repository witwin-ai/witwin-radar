"""The multi-radar session (R-ADR-051): one shared world half, one result per radar.

* A radar's own rows in a network are the rows its standalone session composes:
  its published cube is ``torch.equal`` to ``Radar.simulate`` in a still world
  and under every radar's own pose trajectory (``adc`` and ``chirp``).
* The world half is paid once: the Channel discoveries and replays of a network
  do not grow with the radar count (count budget against one radar).
* ``trace`` + ``echo`` equals ``simulate``; ``stream`` yields the same frames.
* Refusals name what is unsupported.
"""

from __future__ import annotations

import math

import pytest
import torch

pytest.importorskip("witwin.channel")

from support import oracle_scenes as sc

pytestmark = pytest.mark.gpu


def _waveform(**fields):
    from witwin.radar import Fmcw

    values = {
        "slope": 30e12,
        "sample_rate": 5e6,
        "samples_per_chirp": 32,
        "chirps_per_frame": 4,
        "adc_start": 1e-6,
        "idle": 5e-6,
        "ramp_end": 10e-6,
        "output": "beat",
    }
    values.update(fields)
    return Fmcw(**values)


def _radars(count=2, **waveform):
    from witwin.radar import Radar

    radars = []
    for index in range(count):
        radars.append(
            Radar(
                carrier=sc.CARRIER_HZ,
                waveform=_waveform(**waveform),
                tx=((0.01, 0.0, 0.0), (0.03, 0.0, 0.0)) if index == 0 else ((0.01, 0.0, 0.0),),
                rx=((-0.01, 0.0, 0.0), (-0.03, 0.0, 0.0)),
                power=10.0,
                position=(0.0, 0.6 * index, 0.0),
                look_at=(1.0, 0.6 * index, 0.0),
                up=sc.UP,
            )
        )
    return tuple(radars)


def _targets():
    from witwin.radar import PointTargets

    return PointTargets(positions=[[6.0, 0.2, 0.1], [4.0, -0.5, 0.3]], rcs=1.0)


def _walk(speed):
    def pose(time_s):
        return torch.eye(3), torch.tensor([0.0, 0.0, speed * float(time_s)])

    return pose


def _turn(rate):
    def pose(time_s):
        angle = rate * float(time_s)
        c, s = math.cos(angle), math.sin(angle)
        return torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]), torch.zeros(3)

    return pose


@pytest.mark.parametrize("pattern", ["isotropic", "dipole"])
def test_a_still_network_publishes_each_radars_standalone_cube(pattern):
    from witwin.radar import Pattern
    from witwin.radar.network import Network

    element = Pattern.isotropic() if pattern == "isotropic" else Pattern.dipole()
    radars = tuple(radar.replace(pattern=element) for radar in _radars())
    scene = sc.plane_scene(corners=sc.side_wall_corners(z_m=1.5, x_range=(-2.0, 12.0), y_half=4.0), material=_pec())
    results = Network(radars).simulate(scene, _targets(), times=(0.0, 1e-3), reflections=1)
    for radar, result in zip(radars, results, strict=True):
        alone = radar.simulate(scene, _targets(), times=(0.0, 1e-3), reflections=1)
        assert torch.equal(result.cube, alone.cube)
        assert result.compile_count == 1


def _pec():
    from witwin.core import PhysicalMaterial

    return PhysicalMaterial.perfect_conductor()


@pytest.mark.parametrize("pattern", ["isotropic", "dipole"])
@pytest.mark.parametrize("kind", ["adc", "chirp"])
def test_every_radar_follows_its_own_pose(kind, pattern):
    """Bit-identical with isotropic elements; with a pattern, to float32 rounding.

    A standalone radar turns its pattern directions by the pose and then by its
    authored frame inside the kernel; a network turns each element by its own
    radar's composed frame before the kernel, because one join carries one
    frame (R-ADR-051). The two orders round differently in the last bit.
    """

    from witwin.radar import Motion, Pattern
    from witwin.radar.network import Network

    motion = Motion.adc() if kind == "adc" else Motion.chirp()
    element = Pattern.isotropic() if pattern == "isotropic" else Pattern.dipole()
    radars = tuple(radar.replace(pattern=element) for radar in _radars(chirps_per_frame=2, samples_per_chirp=8))
    poses = (_walk(40.0), _turn(300.0))
    results = Network(radars).simulate(sc.empty_scene(), _targets(), times=(0.0,), poses=poses, motion=motion)
    for radar, pose, result in zip(radars, poses, results, strict=True):
        alone = radar.simulate(sc.empty_scene(), _targets(), times=(0.0,), pose=pose, motion=motion)
        assert result.motion_sampling == kind
        if pattern == "isotropic":
            assert torch.equal(result.cube, alone.cube)
        else:
            torch.testing.assert_close(result.cube, alone.cube, rtol=0, atol=1e-6 * float(alone.cube.abs().max()))


def test_a_radar_without_a_pose_in_a_moving_network_matches_its_standalone_cube():
    from witwin.radar import Motion
    from witwin.radar.network import Network

    radars = _radars(chirps_per_frame=2, samples_per_chirp=8)
    results = Network(radars).simulate(
        sc.empty_scene(), _targets(), times=(0.0,), poses=(None, _walk(30.0)), motion=Motion.chirp()
    )
    alone = radars[0].simulate(sc.empty_scene(), _targets(), times=(0.0,), motion=Motion.static())
    # A still radar observes a still world once per frame alone, and once per
    # chirp in a moving network; its own rows are the same numbers either way.
    assert results[0].motion_sampling == "chirp"
    torch.testing.assert_close(results[0].cube, alone.cube, rtol=0, atol=1e-6 * float(alone.cube.abs().max()))


def _count(monkeypatch, call):
    from support.session_ledger import SessionLedger

    ledger = SessionLedger(monkeypatch)
    ledger.run(call)
    return ledger


@pytest.mark.parametrize("motion", ["still", "chirp"])
def test_the_world_half_is_paid_once_whatever_the_radar_count(monkeypatch, motion):
    """Channel calls of a network equal one radar's on the same merged schedule."""

    from witwin.radar import Motion
    from witwin.radar.network import Network

    kind = Motion.chirp() if motion == "chirp" else None
    poses_for = (lambda n: (_walk(10.0),) * n) if motion == "chirp" else (lambda n: None)
    counts = {}
    for count in (1, 2, 3):
        radars = _radars(count, chirps_per_frame=2, samples_per_chirp=8)
        if count == 3:
            radars = (*radars[:2], radars[1].replace(position=(0.0, -0.6, 0.0), look_at=(1.0, -0.6, 0.0)))
        network = Network(radars, links=tuple((i, j) for i in range(count) for j in range(count) if i != j))
        with monkeypatch.context() as patch:
            ledger = _count(
                patch,
                lambda network=network, count=count: network.simulate(
                    sc.empty_scene(),
                    _targets(),
                    times=(0.0, 1e-3),
                    poses=poses_for(count),
                    motion=kind,
                    coupling=count == 1,
                ),
            )
        counts[count] = (dict(ledger.channel), ledger.trips["TwoWayComposer.freeze"])
    # Identical radar timing, so the merged schedule is one radar's.
    # One radar has no link: its own coupling (coupling=count == 1) asks for the same site-less leg.
    assert counts[1] == counts[2] == counts[3], counts
    evaluations = counts[3][0]["evaluate"]
    # Still: one discovery of each of the three legs (inbound, outbound, direct).
    assert evaluations == (3 if motion == "still" else 3 * 2 * 4), counts


def test_the_cross_family_costs_one_launch_per_replay_group_per_receiver(monkeypatch):
    from witwin.radar.network import Network

    radars = _radars()
    network = Network(radars, links=((0, 1), (1, 0)))
    with monkeypatch.context() as patch:
        ledger = _count(patch, lambda: network.simulate(sc.empty_scene(), _targets(), times=(0.0, 1e-3, 2e-3)))
    assert ledger.native["fmcw_cross_forward"] == 2 * 3
    assert ledger.channel["evaluate"] == 3


def test_trace_then_echo_equals_simulate_and_stream():
    from witwin.radar import Motion
    from witwin.radar.network import Network

    network = Network(_radars(chirps_per_frame=2, samples_per_chirp=8), links=((0, 1),), clock_offsets=(0.0, 0.2e-6))
    arguments = {"times": (0.0, 1e-3), "poses": (_walk(20.0), None), "motion": Motion.chirp()}
    simulated = network.simulate(sc.empty_scene(), _targets(), **arguments)
    echoed = network.echo(network.trace(sc.empty_scene(), _targets(), **arguments))
    streamed = list(network.stream(sc.empty_scene(), _targets(), **arguments))
    for index in range(2):
        assert torch.equal(simulated[index].cube, echoed[index].cube)
        for frame, results in enumerate(streamed):
            assert torch.equal(simulated[index].cube[frame], results[index].cube[0])


def test_a_link_adds_rows_to_the_receiver_only():
    from witwin.radar.network import Network

    radars = _radars()
    alone = Network(radars).simulate(sc.empty_scene(), _targets(), times=(0.0,))
    linked = Network(radars, links=((0, 1),)).simulate(sc.empty_scene(), _targets(), times=(0.0,))
    assert torch.equal(linked[0].cube, alone[0].cube)
    assert not torch.equal(linked[1].cube, alone[1].cube)


def test_network_refusals_name_what_is_unsupported():
    from witwin.radar import Motion, Ofdm, Radar
    from witwin.radar.network import Network

    radars = _radars()
    with pytest.raises(ValueError, match="one carrier"):
        Network((radars[0], radars[1].replace(carrier=78e9)))
    with pytest.raises(ValueError, match="names one radar twice"):
        Network(radars, links=((1, 1),))
    ofdm = Radar(
        carrier=sc.CARRIER_HZ,
        waveform=Ofdm(
            subcarrier_spacing=1e6, num_subcarriers=16, cyclic_prefix=1e-6, num_symbols=4, max_expected_delay=1e-7
        ),
        tx=((0, 0, 0),),
        rx=((0.01, 0, 0),),
        power=10.0,
    )
    with pytest.raises(NotImplementedError, match="FMCW"):
        Network((radars[0], ofdm), links=((0, 1),))
    with pytest.raises(NotImplementedError, match="adaptive"):
        Network(radars).simulate(sc.empty_scene(), _targets(), times=(0.0,), motion=Motion.adaptive())
    with pytest.raises(ValueError, match="poses holds"):
        Network(radars).simulate(sc.empty_scene(), _targets(), times=(0.0,), poses=(None,))


def _bistatic_loss(position, grad, weights=None):
    from witwin.radar import PointTargets
    from witwin.radar.network import Network

    radars = tuple(
        radar.replace(tx=((0.0, 0.0, 0.0),), rx=((0.0, 0.0, 0.0),), position=(0.0, 0.0, z), look_at=(1.0, 0.0, z))
        for radar, z in zip(_radars(), (0.0, 1.0), strict=True)
    )
    network = Network(radars, links=((0, 1),))
    targets = PointTargets(positions=position.reshape(1, 3), rcs=1.0)
    cube = network.simulate(sc.empty_scene(), targets, times=(0.0,), los=True, reflections=0, direct=False, grad=grad)[
        1
    ]
    if weights is None:
        return cube.cube
    return (cube.cube * weights).real.sum()


def test_a_bistatic_link_is_differentiable_in_both_modes():
    """VJP and JVP of a heard target's cube against a central difference (cross family + join + Channel)."""

    from torch.autograd import forward_ad

    start = torch.tensor([3.0, 0.2, 0.4], device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(7)
    shape = _bistatic_loss(start, "none").shape
    weights = torch.complex(
        torch.randn(shape, generator=generator, device="cuda"), torch.randn(shape, generator=generator, device="cuda")
    )
    direction = torch.tensor([0.6, 0.0, 0.8], device="cuda")
    step = 2.0**-16
    difference = (
        float(_bistatic_loss(start + step * direction, "none", weights))
        - float(_bistatic_loss(start - step * direction, "none", weights))
    ) / (2 * step)
    leaf = start.clone().requires_grad_(True)
    (gradient,) = torch.autograd.grad(_bistatic_loss(leaf, "vjp", weights), (leaf,))
    reverse = float((gradient * direction).sum())
    assert reverse == pytest.approx(difference, rel=2e-2)
    with forward_ad.dual_level():
        dual = forward_ad.make_dual(start.clone(), direction)
        forward = float(forward_ad.unpack_dual(_bistatic_loss(dual, "jvp", weights)).tangent)
    assert forward == pytest.approx(reverse, rel=1e-3)


@pytest.mark.parametrize("kind", ["adc", "chirp"])
def test_a_refreshed_schedule_routes_every_link_cell_like_a_still_one(kind):
    """A pose that never moves makes the session sampled; the heard rows are the still world's at every instant."""

    from witwin.radar import Motion
    from witwin.radar.network import Network

    def holding(time_s):
        return torch.eye(3), torch.zeros(3)

    network = Network(
        _radars(chirps_per_frame=2, samples_per_chirp=8), links=((0, 1), (1, 0)), clock_offsets=(0.0, 0.3e-6)
    )
    still = network.simulate(sc.empty_scene(), _targets(), times=(0.0,))
    motion = Motion.adc() if kind == "adc" else Motion.chirp()
    sampled = network.simulate(sc.empty_scene(), _targets(), times=(0.0,), poses=(holding, None), motion=motion)
    for alone, refreshed in zip(still, sampled, strict=True):
        assert refreshed.motion_sampling == kind
        scale = float(alone.cube.abs().max())
        torch.testing.assert_close(refreshed.cube, alone.cube, rtol=0, atol=1e-6 * scale)
