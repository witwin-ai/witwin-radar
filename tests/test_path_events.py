"""The per-frame path event log, against the analytic occlusion instants (M2-4 / D-3, M1-1 / MP-2).

A small perfectly conducting plate at ``x = 5`` spans ``|y|, |z| <= 0.5``. A
point target at ``x = 10`` walks along ``y`` at constant speed. Line of sight
from the radar at the origin to ``(10, y, 0)`` crosses the plate's plane at
``y / 2``, so the target is in the shadow exactly while ``|y| < 1``: it dies
when ``y`` reaches ``-1`` and is born again when ``y`` reaches ``+1``. Line of
sight only, so the plate itself returns nothing.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from support import oracle_scenes as sc

from witwin.radar import Fmcw, Motion, PointTargets

pytestmark = pytest.mark.gpu

SPEED = 50.0
START_Y = -1.6
#: Analytic shadow-edge crossings, s.
DEATH_S = (-1.0 - START_Y) / SPEED
BIRTH_S = (1.0 - START_Y) / SPEED
CHIRPS = 64


def _plate():
    from witwin.core import PhysicalMaterial

    corners = ((5.0, -0.5, -0.5), (5.0, 0.5, -0.5), (5.0, 0.5, 0.5), (5.0, -0.5, 0.5))
    return sc.plane_scene(corners=corners, material=PhysicalMaterial.perfect_conductor())


def _radar():
    # 1 ms chirps, so the 64-chirp frame spans both crossings.
    waveform = Fmcw(
        slope=30e12,
        sample_rate=5e6,
        samples_per_chirp=16,
        chirps_per_frame=CHIRPS,
        adc_start=0.0,
        idle=1e-3,
        ramp_end=16 / 5e6 + 1e-6,
        output="beat",
    )
    return sc.radar(waveform)


def _walker(device, speed=SPEED):
    def trajectory(time_s):
        return torch.tensor([[10.0, START_Y + speed * float(time_s), 0.0]], dtype=torch.float32, device=device)

    return PointTargets(positions=trajectory(0.0), rcs=1.0, trajectory=trajectory, ids=(77,))


def _simulate(motion, scene=None):
    radar = _radar()
    return radar.simulate(
        _plate() if scene is None else scene,
        _walker(radar.device),
        times=(0.0,),
        los=True,
        reflections=0,
        motion=motion,
    )


def _assert_brackets(events, expected):
    """Each event's ``(previous_time_s, time_s]`` contains its analytic instant."""

    assert events.dropped == 0
    assert list(events.birth) == [birth for birth, _ in expected]
    for index, (_, instant) in enumerate(expected):
        assert events.previous_time_s[index] < instant <= events.time_s[index], (index, instant, events)
    assert set(events.site_id.tolist()) == {77}
    assert (events.depth == 0).all()
    assert (events.amplitude > 0).all()


@pytest.mark.parametrize(
    "motion",
    [Motion.chirp(), Motion.chirp(rediscover_every_frames=1), Motion.adc(), Motion.adaptive()],
    ids=["chirp-rediscover", "chirp-replay", "adc", "adaptive"],
)
def test_occlusion_events_match_the_analytic_shadow_crossings(motion):
    """One death at the first shadow edge and one birth at the second, and nothing else.

    ``chirp-replay`` discovers once and replays the frozen row, so the events
    are the row's validity flipping; the others rediscover, so the row leaves
    and re-enters the topology and the events come from matching two joins.
    """

    result = _simulate(motion)
    (events,) = result.events
    _assert_brackets(events, [(False, DEATH_S), (True, BIRTH_S)])
    # The occluded stretch contributes exactly zero, the rest does not.
    times = result.sample_times[0].reshape(CHIRPS, -1)[:, 0]
    power = result.cube[0, 0, 0].abs().pow(2).sum(dim=-1).cpu().numpy()
    hidden = (times >= events.time_s[0]) & (times < events.time_s[1])
    assert (power[hidden] == 0).all() and (power[~hidden] > 0).all()


def test_the_birth_amplitude_is_the_live_rows_magnitude():
    """The amplitude at an event is ``|C|`` of the live side, the radar equation's ``R^-2``.

    The two events sit at mirror bearings to within one chirp of travel, so
    the element pattern's gain differs between them by a fraction of a percent.
    """

    result = _simulate(Motion.chirp())
    (events,) = result.events
    ranges = np.hypot(10.0, START_Y + SPEED * events.time_s), np.hypot(10.0, START_Y + SPEED * events.previous_time_s)
    # Death: the last live row is at previous_time_s; birth: the new row at time_s.
    live_range = np.where(events.birth, ranges[0], ranges[1])
    ratio = events.amplitude * live_range**2
    np.testing.assert_allclose(ratio / ratio[0], 1.0, rtol=1e-2)


def test_a_still_world_and_an_unobstructed_walk_log_nothing():
    radar = _radar()
    still = radar.simulate(_plate(), PointTargets(positions=[[10.0, 3.0, 0.0]], rcs=1.0), times=(0.0, 0.1))
    assert [len(events) for events in still.events] == [0, 0]
    assert still.birth_lag_bound == (0.0, 0.0)
    clear = _simulate(Motion.chirp(), scene=sc.empty_scene())
    assert len(clear.events[0]) == 0


def test_the_trace_publishes_the_same_events_as_the_simulation():
    radar = _radar()
    kwargs = {"times": (0.0,), "los": True, "reflections": 0, "motion": Motion.chirp()}
    paths = radar.trace(_plate(), _walker(radar.device), **kwargs)
    fused = radar.simulate(_plate(), _walker(radar.device), **kwargs)
    np.testing.assert_array_equal(paths.events[0].time_s, fused.events[0].time_s)
    np.testing.assert_array_equal(paths.events[0].birth, fused.events[0].birth)
    assert radar.echo(paths).events[0] is paths.events[0]


def _sliding_plate(start_y: float):
    """The plate, sliding along ``+y`` at ``SPEED`` from ``start_y`` (its centre)."""

    from witwin.core.dynamics import DynamicScene, LinearTrajectory

    scene = _plate()
    trajectory = LinearTrajectory(origin=torch.tensor([0.0, start_y, 0.0]), velocity=torch.tensor([0.0, SPEED, 0.0]))
    return DynamicScene(scene, structure_trajectories={1: trajectory})


def test_a_birth_has_the_range_doppler_signature_of_a_gated_echo():
    """M2-4 / D-3: a birth is a slow-time step, so its Doppler profile is the gate's transform.

    A still target at ``(10, 0, 0)`` sits behind the plate until the plate's
    trailing edge passes ``y = 0`` at ``x = 5``, which is a birth at
    ``t = (0.5 - start) / SPEED``. The occluded run equals the unobstructed run
    multiplied, chirp by chirp, by the existence gate the event log reports.
    The unobstructed echo is a zero-Doppler tone whose Hann-windowed Doppler
    cut is the window's transform, below -40 dB away from its main lobe; the
    gated echo's cut is the transform of the window times the gate, which
    leaks the step across every Doppler bin.
    """

    radar = _radar()
    still = PointTargets(positions=[[10.0, 0.0, 0.0]], rcs=1.0, ids=(77,))
    kwargs = {"times": (0.0,), "los": True, "reflections": 0, "motion": Motion.chirp()}
    start = -0.2
    occluded = radar.simulate(_sliding_plate(start), still, **kwargs)
    clear = radar.simulate(sc.empty_scene(), still, **kwargs)
    (events,) = occluded.events
    assert list(events.birth) == [True]
    assert events.previous_time_s[0] < (0.5 - start) / SPEED <= events.time_s[0]
    times = occluded.sample_times[0]
    gate = torch.as_tensor(times >= events.time_s[0], dtype=torch.float32, device=clear.cube.device)
    torch.testing.assert_close(occluded.cube[0, 0, 0], clear.cube[0, 0, 0] * gate[:, None], rtol=1e-6, atol=0)

    def doppler_cut(result):
        magnitude = result.frame(0).range_doppler(window="hann").data.abs()[0, 0]  # [doppler, range]
        bin_ = int(magnitude.max(dim=0).values.argmax())
        cut = magnitude[:, bin_].double()
        return cut / cut.max()

    clear_cut, gated_cut = doppler_cut(clear), doppler_cut(occluded)
    # The still target's tone has the same phase at every chirp, so its
    # Doppler cut is the periodic-Hann window's transform and the gated one is
    # the transform of Hann times the existence gate, both fftshifted.
    index = np.arange(CHIRPS)
    hann = 0.5 - 0.5 * np.cos(2 * np.pi * index / CHIRPS)
    for cut, envelope in ((clear_cut, hann), (gated_cut, hann * gate.cpu().numpy())):
        expected = np.abs(np.fft.fftshift(np.fft.fft(envelope)))
        expected /= expected.max()
        np.testing.assert_allclose(cut.cpu().numpy(), expected, atol=1e-4, rtol=0)
    peak = int(clear_cut.argmax())
    far = torch.ones_like(clear_cut, dtype=torch.bool)
    far[[(peak + offset) % CHIRPS for offset in range(-3, 4)]] = False
    # The ungated tone is confined to the window's main lobe; the step leaks
    # across every Doppler bin.
    assert float(clear_cut[far].max()) < 10 ** (-40 / 20)
    assert float(gated_cut[far].min()) > 10 ** (-50 / 20)


def test_the_birth_lag_bound_follows_the_discovery_cadence():
    every = _simulate(Motion.chirp())
    held = _simulate(Motion.chirp(rediscover_every_frames=1))
    period = float(np.diff(every.sample_times[0])[0])
    assert every.birth_lag_bound[0] == pytest.approx(period, rel=1e-9)
    span = float(held.sample_times[0][-1] - held.sample_times[0][0])
    assert held.birth_lag_bound[0] == pytest.approx(span, rel=1e-9)
