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
                reference_frequency_hz=sc.CARRIER_HZ,
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
        Network((radars[0], radars[1].replace(reference_frequency_hz=78e9)))
    with pytest.raises(ValueError, match="names one radar twice"):
        Network(radars, links=((1, 1),))
    ofdm = Radar(
        reference_frequency_hz=sc.CARRIER_HZ,
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
    cube = network.simulate(
        sc.empty_scene(), targets, times=(0.0,), los=True, reflections=0, direct=False, ad_mode=grad
    )[1]
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


def test_mesh_targets_in_a_network_extract_their_facets_once_for_every_radar(monkeypatch):
    """Each radar's cube is its standalone one, and the facet operator runs once per binding, not per radar."""

    from witwin.core import PhysicalMaterial

    from witwin.radar import MeshTargets
    from witwin.radar.cuda import runtime
    from witwin.radar.network import Network

    vertices, faces = sc.plate_mesh(2, side_m=0.1, range_m=6.0)
    scene = sc.mesh_scene(vertices, faces, material=PhysicalMaterial.perfect_conductor())
    radars = _radars()
    table = runtime.build_extension()
    calls: dict[str, int] = {}
    original = table.facet_geometry_forward

    def counting(*args, **kwargs):
        calls["facet_geometry_forward"] = calls.get("facet_geometry_forward", 0) + 1
        return original(*args, **kwargs)

    monkeypatch.setattr(table, "facet_geometry_forward", counting, raising=False)
    results = Network(radars, links=((0, 1),)).simulate(scene, MeshTargets(), times=(0.0,), reflections=0)
    network_calls = calls.pop("facet_geometry_forward")
    alone = [radar.simulate(scene, MeshTargets(), times=(0.0,), reflections=0) for radar in radars]
    single_calls = calls.pop("facet_geometry_forward") / len(radars)
    assert network_calls == single_calls
    assert torch.equal(results[0].cube, alone[0].cube)
    # Radar 1 also hears radar 0 off the plate (a bistatic facet return).
    assert not torch.equal(results[1].cube, alone[1].cube)
    unlinked = Network(radars).simulate(scene, MeshTargets(), times=(0.0,), reflections=0)
    assert torch.equal(unlinked[1].cube, alone[1].cube)


def _sampled_radars(kind):
    from witwin.radar import Ofdm, Pulsed, Radar

    if kind == "pulsed":
        waveform = Pulsed(
            pulse_kind="lfm",
            pulse_width=1e-6,
            bandwidth=10e6,
            pri=8e-6,
            num_pulses=3,
            sample_rate=20e6,
            num_samples=24,
            range_gate_start=0.0,
        )
    else:
        waveform = Ofdm(
            subcarrier_spacing=2e6,
            num_subcarriers=16,
            cyclic_prefix=0.2e-6,
            num_symbols=3,
            max_expected_delay=0.15e-6,
            modulation="qpsk",
        )
    return tuple(
        Radar(
            reference_frequency_hz=sc.CARRIER_HZ,
            waveform=waveform,
            tx=((0.01, 0.0, 0.0),),
            rx=((-0.01, 0.0, 0.0), (-0.03, 0.0, 0.0)),
            power=10.0,
            position=(0.0, 0.6 * index, 0.0),
            look_at=(1.0, 0.6 * index, 0.0),
            up=sc.UP,
        )
        for index in range(2)
    )


@pytest.mark.parametrize("motion", ["adc", "pulse"])
@pytest.mark.parametrize("kind", ["pulsed", "ofdm"])
def test_a_sampled_receiver_routes_every_cell_like_a_still_one(kind, motion):
    """Pulsed and OFDM receivers: a pose that never moves gives the still world's cube at every instant."""

    from witwin.radar import Emitter, Motion
    from witwin.radar.network import Network

    def holding(time_s):
        return torch.eye(3), torch.zeros(3)

    emitter = Emitter(position=(3.0, 1.0, 0.5), samples=(1.0, 1j, -1.0, -1j), rate=1e6, power=0.0)
    network = Network(_sampled_radars(kind), links=((0, 1), (1, 0)), clock_offsets=(0.0, 0.1e-6), emitters=(emitter,))
    still = network.simulate(sc.empty_scene(), _targets(), times=(0.0,))
    requested = Motion.adc() if motion == "adc" else Motion.chirp()
    sampled = network.simulate(sc.empty_scene(), _targets(), times=(0.0,), poses=(holding, None), motion=requested)
    for alone, refreshed in zip(still, sampled, strict=True):
        scale = float(alone.cube.abs().max())
        torch.testing.assert_close(refreshed.cube, alone.cube, rtol=0, atol=1e-6 * scale)


def test_emitters_and_sampled_links_add_no_channel_call(monkeypatch):
    from witwin.radar import Emitter
    from witwin.radar.network import Network

    radars = _sampled_radars("pulsed")
    emitter = Emitter(position=(3.0, 1.0, 0.5), samples=(1.0,), rate=1e6, power=0.0)
    counts = []
    for network in (
        Network(radars, links=((0, 1),)),
        Network(radars, links=((0, 1), (1, 0)), emitters=(emitter, emitter)),
    ):
        with monkeypatch.context() as patch:
            ledger = _count(patch, lambda network=network: network.simulate(sc.empty_scene(), _targets(), times=(0.0,)))
        counts.append((ledger.channel["evaluate"], ledger.native["pulsed_cross_forward"]))
    # Three legs discovered once; one pulsed-family launch per hearing receiver.
    assert counts == [(3, 1), (3, 2)]


def test_cross_family_refusals_name_what_is_unsupported():
    from witwin.radar import Emitter, Pattern
    from witwin.radar.network import Network

    pulsed = _sampled_radars("pulsed")
    fmcw = _radars()
    with pytest.raises(NotImplementedError, match="FMCW ramp"):
        Network((fmcw[0], pulsed[1]), links=((0, 1),))
    with pytest.raises(NotImplementedError, match="an FMCW receiver dechirps"):
        Network((pulsed[0], fmcw[1]), links=((0, 1),))
    # fmcw radar 0 has two transmitters: as a sampled receiver it would be refused, as FMCW it is fine.
    two = pulsed[1].replace(tx=((0.01, 0.0, 0.0), (0.03, 0.0, 0.0)))
    with pytest.raises(NotImplementedError, match="one transmitter"):
        Network((pulsed[0], two), links=((0, 1),))
    with pytest.raises(ValueError, match="Chirp"):
        Network(fmcw, carrier_offsets=(1e3, 0.0))
    with pytest.raises(NotImplementedError, match="anti-alias"):
        Network(pulsed, emitters=(Emitter(position=(1, 1, 1), samples=(1.0,), rate=100e6, power=0.0),))
    dipole = tuple(radar.replace(pattern=Pattern.dipole()) for radar in fmcw)
    with pytest.raises(NotImplementedError, match="isotropic"):
        Network(dipole, emitters=(Emitter(position=(1, 1, 1), samples=(1.0,), rate=1e6, power=0.0),))


def _pulsed_bistatic_loss(position, grad, weights=None):
    from witwin.radar import PointTargets
    from witwin.radar.network import Network

    radars = tuple(
        radar.replace(rx=((0.0, 0.0, 0.0),), position=(0.0, 0.0, z), look_at=(1.0, 0.0, z))
        for radar, z in zip(_sampled_radars("pulsed"), (0.0, 1.0), strict=True)
    )
    network = Network(radars, links=((0, 1),))
    targets = PointTargets(positions=position.reshape(1, 3), rcs=1.0)
    cube = network.simulate(sc.empty_scene(), targets, times=(0.0,), reflections=0, direct=False, ad_mode=grad)[1]
    if weights is None:
        return cube.cube
    return (cube.cube * weights).real.sum()


def test_a_pulsed_bistatic_link_is_differentiable_in_both_modes():
    from torch.autograd import forward_ad

    start = torch.tensor([3.0, 0.2, 0.4], device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(9)
    shape = _pulsed_bistatic_loss(start, "none").shape
    weights = torch.complex(
        torch.randn(shape, generator=generator, device="cuda"), torch.randn(shape, generator=generator, device="cuda")
    )
    direction = torch.tensor([0.6, 0.0, 0.8], device="cuda")
    step = 2.0**-16
    difference = (
        float(_pulsed_bistatic_loss(start + step * direction, "none", weights))
        - float(_pulsed_bistatic_loss(start - step * direction, "none", weights))
    ) / (2 * step)
    leaf = start.clone().requires_grad_(True)
    (gradient,) = torch.autograd.grad(_pulsed_bistatic_loss(leaf, "vjp", weights), (leaf,))
    reverse = float((gradient * direction).sum())
    assert reverse == pytest.approx(difference, rel=2e-2)
    with forward_ad.dual_level():
        dual = forward_ad.make_dual(start.clone(), direction)
        forward = float(forward_ad.unpack_dual(_pulsed_bistatic_loss(dual, "jvp", weights)).tangent)
    assert forward == pytest.approx(reverse, rel=1e-3)


def _weighted(cube):
    weights = torch.randn(cube.shape, generator=torch.Generator().manual_seed(0), dtype=torch.complex64)
    return (cube * weights.to(cube.device).conj()).real.sum() * 1e6


def test_each_radars_pose_translation_is_a_leaf_inside_a_network():
    """Radar 1's own cube differentiates through its own trajectory exactly as alone (R-ADR-052)."""

    from witwin.radar import Motion
    from witwin.radar.network import Network

    radars = tuple(radar.replace(polarization=(0.0, 1.0, 0.0)) for radar in _radars(chirps_per_frame=2))
    leaf = torch.zeros(3, requires_grad=True)

    def pose(time_s):
        return torch.eye(3), leaf + torch.tensor([0.0, 0.0, 5.0]) * time_s

    arguments = {"times": (0.0,), "motion": Motion.chirp(), "ad_mode": "vjp", "reflections": 0}
    network = Network(radars).simulate(sc.empty_scene(), _targets(), poses=(None, pose), **arguments)
    (inside,) = torch.autograd.grad(_weighted(network[1].cube), leaf)
    alone = radars[1].simulate(sc.empty_scene(), _targets(), pose=pose, **arguments)
    (outside,) = torch.autograd.grad(_weighted(alone.cube), leaf)
    assert float(outside.abs().max()) > 0
    torch.testing.assert_close(inside, outside, rtol=1e-4, atol=0)

    def turning(time_s):
        return torch.eye(3, requires_grad=True), torch.zeros(3)

    with pytest.raises(RuntimeError, match="rotation"):
        Network(radars).simulate(sc.empty_scene(), _targets(), poses=(None, turning), **arguments)


def test_a_shared_pattern_leaf_collects_every_radars_rows_and_a_power_leaf_is_refused():
    from witwin.radar import Pattern
    from witwin.radar.network import Network

    gain = torch.tensor(3.0, requires_grad=True)
    element = Pattern.dipole(gain=gain)
    radars = tuple(radar.replace(pattern=element) for radar in _radars())
    results = Network(radars).simulate(sc.empty_scene(), _targets(), times=(0.0,), ad_mode="vjp", reflections=0)
    (inside,) = torch.autograd.grad(_weighted(results[0].cube), gain)
    alone = radars[0].simulate(sc.empty_scene(), _targets(), times=(0.0,), ad_mode="vjp", reflections=0)
    (outside,) = torch.autograd.grad(_weighted(alone.cube), gain)
    assert float(outside.abs()) > 0
    torch.testing.assert_close(inside, outside, rtol=1e-4, atol=0)
    # Two leaf patterns that are not the same object are two tables.
    other = tuple(radar.replace(pattern=Pattern.dipole(gain=torch.tensor(3.0))) for radar in _radars())
    with pytest.raises(NotImplementedError, match="element pattern"):
        Network((radars[0], other[1]))
    with pytest.raises(NotImplementedError, match="transmit power"):
        Network((radars[0].replace(power=torch.tensor(10.0)), radars[1]))
