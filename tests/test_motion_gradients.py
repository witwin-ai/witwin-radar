"""Scene-route derivatives of moving targets against finite differences (M2-10 / D-12).

The leaves are a target's initial position and its velocity, entered through
the trajectory ``p(t) = p0 + v t``, and the loss is a fixed random projection
of the cube, ``L = Re sum(w * cube)``. Reverse mode (``ad_mode="vjp"``) and
forward mode (``ad_mode="jvp"``) are each checked against a central difference of
the primal route. The step is a power of two so that the perturbed float32
coordinate is exact, and small enough that the carrier phase moves by a few
hundredths of a radian, where the difference's truncation error is ``1e-4``.
"""

from __future__ import annotations

import pytest
import torch
import torch.autograd.forward_ad as forward_ad
from support import oracle_scenes as sc

from witwin.radar import Fmcw, Motion, PointTargets

pytestmark = pytest.mark.gpu

ORIGIN = (3.0, 0.4, 0.0)
VELOCITY = (6.0, -2.0, 0.0)
POSITION_STEP = 2.0**-16
VELOCITY_STEP = 2.0**-6


def _radar(*, idle=5e-6, chirps=4):
    waveform = Fmcw(
        slope=30e12,
        sample_rate=5e6,
        samples_per_chirp=8,
        chirps_per_frame=chirps,
        adc_start=0.0,
        idle=idle,
        ramp_end=8 / 5e6 + 1e-6,
        output="beat",
    )
    return sc.radar(waveform)


def _loss(radar, scene, origin, velocity, motion, grad, *, weights):
    def trajectory(time_s):
        return (origin + velocity * float(time_s)).reshape(1, 3)

    result = radar.simulate(
        scene,
        PointTargets(positions=trajectory(0.0).detach(), rcs=1.0, trajectory=trajectory),
        times=(0.0,),
        los=True,
        reflections=0,
        motion=motion,
        ad_mode=grad,
    )
    return (result.cube * weights).real.sum(), result


def _weights(radar, scene, motion, start, rate):
    origin = torch.tensor(start, device=radar.device)
    velocity = torch.tensor(rate, device=radar.device)
    _, result = _loss(radar, scene, origin, velocity, motion, "none", weights=1.0)
    generator = torch.Generator(device=radar.device).manual_seed(5)
    real = torch.randn(result.cube.shape, generator=generator, device=radar.device)
    imaginary = torch.randn(result.cube.shape, generator=generator, device=radar.device)
    return torch.complex(real, imaginary), result


def _central(radar, scene, motion, weights, start, rate, leaf: str, direction: torch.Tensor, step: float) -> float:
    base = {"origin": torch.tensor(start, device=radar.device), "velocity": torch.tensor(rate, device=radar.device)}

    def at(sign):
        values = dict(base)
        values[leaf] = values[leaf] + sign * step * direction
        return float(_loss(radar, scene, values["origin"], values["velocity"], motion, "none", weights=weights)[0])

    return (at(1.0) - at(-1.0)) / (2.0 * step)


def _check(radar, scene, motion, *, start=ORIGIN, rate=VELOCITY, rtol, position_step=POSITION_STEP):
    weights, primal = _weights(radar, scene, motion, start, rate)
    direction = torch.tensor([0.6, 0.8, 0.0], device=radar.device)
    origin = torch.tensor(start, device=radar.device, requires_grad=True)
    velocity = torch.tensor(rate, device=radar.device, requires_grad=True)
    loss, _ = _loss(radar, scene, origin, velocity, motion, "vjp", weights=weights)
    gradient = torch.autograd.grad(loss, (origin, velocity))
    for index, (leaf, step) in enumerate((("origin", position_step), ("velocity", VELOCITY_STEP))):
        reverse = float((gradient[index] * direction).sum())
        difference = _central(radar, scene, motion, weights, start, rate, leaf, direction, step)
        assert reverse == pytest.approx(difference, rel=rtol), (leaf, reverse, difference)
        with forward_ad.dual_level():
            tangent = {"origin": torch.zeros(3, device=radar.device), "velocity": torch.zeros(3, device=radar.device)}
            tangent[leaf] = direction
            dual_origin = forward_ad.make_dual(torch.tensor(start, device=radar.device), tangent["origin"])
            dual_velocity = forward_ad.make_dual(torch.tensor(rate, device=radar.device), tangent["velocity"])
            dual_loss, _ = _loss(radar, scene, dual_origin, dual_velocity, motion, "jvp", weights=weights)
            forward = float(forward_ad.unpack_dual(dual_loss).tangent)
        assert forward == pytest.approx(reverse, rel=1e-3), (leaf, forward, reverse)
    return primal


@pytest.mark.parametrize("motion", [Motion.adc(), Motion.auto()], ids=["adc", "auto-adaptive"])
def test_position_and_velocity_gradients_match_finite_differences(motion):
    # Long chirp gaps so the velocity leaf moves the target by tens of microns
    # over the frame, well above the float32 position quantum.
    radar = _radar(idle=1e-3)
    _check(radar, sc.empty_scene(), motion, rtol=2e-2)


def test_gradients_across_a_path_death_match_finite_differences():
    """A frame that contains an occlusion event is still differentiable away from it.

    The target walks into the plate's shadow mid-frame (``tests/test_path_events.py``
    has the geometry). The event's instant moves with the leaves, but by
    ``step / speed``, well under a microsecond, which never carries it across
    an observation; the derivative is the one of the gated echo, and the dead
    observations contribute exactly zero to it.
    """

    from test_path_events import _plate

    radar = _radar(idle=1e-3, chirps=16)
    # At 10 m a float32 range carries ~1e-6 m of rounding (D-1), so the
    # position step is four times the 3 m case's to keep that below 2 %.
    primal = _check(
        radar,
        _plate(),
        Motion.chirp(),
        start=(10.0, -1.6, 0.0),
        rate=(0.0, 50.0, 0.0),
        rtol=2e-2,
        position_step=2.0**-14,
    )
    (events,) = primal.events
    assert len(events) == 1 and not events.birth[0]
