"""Orientation trajectories (M2-8 / D-10): a rotating aspect lobe and a radar that turns with its platform."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from support import oracle_scenes as sc

from witwin.radar import Aspect, Motion, Pattern, PointTargets

pytestmark = pytest.mark.gpu

CHIRPS = 64
EXPONENT = 2.0


def test_a_rotating_lobe_flashes_once_per_turn():
    """``Aspect.axis(t)`` turns in the x-y plane at ``Omega``.

    The target sits on the boresight at ``(3, 0, 0)``, so the wave arrives
    travelling ``+x`` and leaves travelling ``-x`` and the lobe is
    ``max(-cos Omega t, 0)^(2 n)``: one flash per turn, centred where the axis
    points back at the radar. The echo magnitude per chirp is that lobe times
    a constant, and its flashes are one period ``2 pi / Omega`` apart.
    """

    radar = sc.radar(sc.beat_fmcw(samples=8, chirps=CHIRPS))
    period_s = radar.waveform_spec().chirp_period_s
    omega = 2.0 * math.pi / (16.0 * period_s)

    def axis(time_s):
        angle = omega * time_s
        return torch.tensor([[math.cos(angle), math.sin(angle), 0.0]], dtype=torch.float32, device=radar.device)

    targets = PointTargets(
        positions=[[3.0, 0.0, 0.0]], rcs=1.0, aspect=Aspect(axis=axis, exponent=EXPONENT, coherent_interval=1.0)
    )
    result = radar.simulate(sc.empty_scene(), targets, times=(0.0,), reflections=0, motion=Motion.chirp())
    assert result.motion_sampling == "chirp"
    times = result.sample_times[0]
    magnitude = result.cube[0, 0, 0, :, 0].abs().double().cpu().numpy()
    lobe = np.maximum(-np.cos(omega * times), 0.0) ** (2 * EXPONENT)
    lit = lobe > 1e-3
    scale = magnitude[lit] / lobe[lit]
    np.testing.assert_allclose(scale / scale.max(), 1.0, rtol=1e-4)
    assert (magnitude[lobe == 0.0] == 0.0).all()
    peaks = [index for index in range(1, CHIRPS - 1) if magnitude[index - 1] < magnitude[index] >= magnitude[index + 1]]
    assert np.allclose(np.diff(times[peaks]), 2.0 * math.pi / omega, rtol=1e-6)


def _yaw(angle: float) -> np.ndarray:
    return np.array([[math.cos(angle), -math.sin(angle), 0.0], [math.sin(angle), math.cos(angle), 0.0], [0, 0, 1.0]])


def test_the_pattern_turns_with_the_radar_pose():
    """A yawing radar sees what a static radar built at each instant's heading sees.

    The radar's elements sit at its own position, so turning it changes no
    distance and the echo changes only through the dipole pattern and the
    polarization. The oracle at chirp ``c`` is a separate static radar authored
    looking along ``R(theta_c) x`` with up ``R(theta_c) y``.
    """

    chirps = 4
    step = math.radians(12.0)
    waveform = sc.beat_fmcw(samples=8, chirps=chirps)
    probe = sc.radar(waveform)
    period_s = probe.waveform_spec().chirp_period_s

    def heading(time_s):
        return step * time_s / period_s

    def pose(time_s):
        return torch.tensor(_yaw(heading(time_s)), dtype=torch.float32), torch.zeros(3)

    targets = PointTargets(positions=[[4.0, 1.5, 0.3]], rcs=1.0)
    turning = sc.radar(waveform, pattern=Pattern.dipole()).simulate(
        sc.empty_scene(), targets, times=(0.0,), reflections=0, motion=Motion.chirp(), pose=pose
    )
    held = sc.radar(waveform, pattern=Pattern.dipole()).simulate(
        sc.empty_scene(), targets, times=(0.0,), reflections=0, motion=Motion.chirp()
    )
    times = turning.sample_times[0]
    for chirp in range(chirps):
        rotation = _yaw(heading(times[chirp]))
        static = sc.radar(waveform, pattern=Pattern.dipole())
        static = static.replace(look_at=tuple(rotation @ np.array(sc.LOOK_AT)), up=tuple(rotation @ np.array(sc.UP)))
        frozen = static.simulate(sc.empty_scene(), targets, times=(0.0,), reflections=0)
        torch.testing.assert_close(turning.cube[0, 0, 0, chirp], frozen.cube[0, 0, 0, 0], rtol=2e-4, atol=0)
    assert not torch.allclose(turning.cube[0, 0, 0, -1], held.cube[0, 0, 0, -1], rtol=1e-2, atol=0)


def test_a_pose_translation_carries_the_elements():
    """The translation moves every element: a radar closing on a still target is a moving target."""

    waveform = sc.beat_fmcw(samples=8, chirps=4)
    radar = sc.radar(waveform)
    speed = 3.0

    def pose(time_s):
        return torch.eye(3), torch.tensor([speed * time_s, 0.0, 0.0])

    def closing(time_s):
        return torch.tensor([[5.0 - speed * time_s, 0.0, 0.0]], dtype=torch.float32, device=radar.device)

    ego = radar.simulate(
        sc.empty_scene(),
        PointTargets(positions=[[5.0, 0.0, 0.0]], rcs=1.0),
        times=(0.0,),
        reflections=0,
        motion=Motion.adc(),
        pose=pose,
    )
    target = radar.simulate(
        sc.empty_scene(),
        PointTargets(positions=closing(0.0), rcs=1.0, trajectory=closing),
        times=(0.0,),
        reflections=0,
        motion=Motion.adc(),
    )
    # The two routes round the same distance differently in float32 (D-1),
    # a few milliradians of carrier phase at 5 m.
    torch.testing.assert_close(ego.cube, target.cube, rtol=5e-3, atol=0)
