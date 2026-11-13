"""The radar pose as a leaf (R-ADR-052).

``Radar.position``, ``look_at`` and ``up`` place every element and orient the
pattern frame. Both reach the cube natively: the element positions through
Channel's fixed-topology replay and the two-way join, the frame through the
sensor-weight operator's frame slot. A ``pose=`` trajectory's translation is a
leaf the same way.

What cannot be delivered is refused before any work: an ``"up"``/``"right"``
polarization turns with the pose and Channel's replay takes endpoint
polarizations as constants; a pose trajectory's rotation turns every
polarization for the same reason; and a pose leaf under a ``grad`` mode that
does not ask Channel for that derivative would be silently partial.
"""

from __future__ import annotations

import pytest
import torch
import torch.autograd.forward_ad as fwad
from test_pattern_leaves import _radar, _verb_loss

from witwin.radar import Motion, Pattern

pytestmark = pytest.mark.gpu

#: A world-vector polarization: it does not turn with the pose.
FIXED_POLARIZATION = (0.0, 1.0, 0.0)


def _posed(**pose):
    return _radar(polarization=FIXED_POLARIZATION, pattern=Pattern.dipole(gain=1.0), **pose)


@pytest.mark.parametrize(
    ("field", "base", "direction", "step"),
    [
        ("position", (0.0, 0.0, 0.0), (0.3, -0.2, 0.5), 3e-5),
        ("look_at", (1.0, 0.0, 0.0), (0.0, 0.2, 0.3), 1e-3),
        ("up", (0.0, 1.0, 0.0), (0.0, 0.0, 0.4), 1e-3),
    ],
)
def test_a_pose_leaf_matches_central_differences_in_both_modes(field, base, direction, step):
    base = torch.tensor(base)
    direction = torch.tensor(direction)

    def loss(value, grad):
        return _verb_loss(_posed(**{field: value}), grad=grad)

    leaf = base.clone().requires_grad_(True)
    (gradient,) = torch.autograd.grad(loss(leaf, "vjp"), leaf)
    vjp = float((gradient * direction).sum())
    with fwad.dual_level():
        jvp = float(fwad.unpack_dual(loss(fwad.make_dual(base, direction), "jvp")).tangent)
    central = (float(loss(base + step * direction, "none")) - float(loss(base - step * direction, "none"))) / (2 * step)
    assert vjp == pytest.approx(jvp, rel=1e-3)
    assert vjp == pytest.approx(central, rel=1e-2)


def test_a_pose_trajectory_translation_is_a_leaf():
    def loss(offset, grad):
        def pose(t):
            return torch.eye(3), offset + torch.tensor([0.0, 0.0, 5.0]) * t

        return _verb_loss_posed(pose, grad)

    base = torch.zeros(3)
    direction = torch.tensor([0.3, -0.2, 0.5])
    leaf = base.clone().requires_grad_(True)
    (gradient,) = torch.autograd.grad(loss(leaf, "vjp"), leaf)
    # The loss turns through a carrier cycle every half wavelength (about
    # 2 mm at 77 GHz), so the difference needs a step far below that.
    step = 3e-5
    central = (float(loss(base + step * direction, "none")) - float(loss(base - step * direction, "none"))) / (2 * step)
    assert float((gradient * direction).sum()) == pytest.approx(central, rel=2e-2)


def _verb_loss_posed(pose, grad):
    pytest.importorskip("witwin.channel")
    from witwin.core import Scene

    from witwin.radar import PointTargets

    targets = PointTargets(positions=torch.tensor([[6.0, 0.5, 1.0], [5.0, -0.4, -0.8]], device="cuda"), rcs=1.0)
    cube = _posed().simulate(
        Scene(structures=(), endpoints=[]), targets, times=(0.0,), reflections=0, grad=grad, pose=pose,
        motion=Motion.chirp(),
    ).cube  # fmt: skip
    weights = torch.randn(cube.shape, generator=torch.Generator().manual_seed(0), dtype=torch.complex64)
    return (cube * weights.to(cube.device).conj()).real.sum() * 1e6


# ---------------------------------------------------------------------------
# What is refused, before any work
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("polarization", ["up", "right"])
def test_a_pose_leaf_with_a_pose_derived_polarization_is_refused(polarization):
    with pytest.raises(RuntimeError, match="derived from the pose") as raised:
        _radar(position=torch.zeros(3, requires_grad=True), polarization=polarization)
    assert "world vector" in str(raised.value)
    with pytest.raises(RuntimeError, match="derived from the pose"):
        _radar(look_at=torch.tensor([1.0, 0.0, 0.0], requires_grad=True), rx_polarization=polarization,
               polarization=FIXED_POLARIZATION)  # fmt: skip


def test_a_polarization_leaf_is_refused():
    with pytest.raises(RuntimeError, match="primal-only"):
        _radar(polarization=torch.tensor([0.0, 1.0, 0.0], requires_grad=True))


@pytest.mark.parametrize(("marked", "grad"), [("gradient", "none"), ("gradient", "jvp"), ("tangent", "vjp")])
def test_a_pose_leaf_in_the_wrong_ad_mode_is_refused_before_any_work(marked, grad):
    from witwin.radar.cuda import runtime

    calls = []
    table = runtime.build_extension()
    original = table.two_way_join_forward

    with fwad.dual_level():
        position = torch.zeros(3)
        position = position.requires_grad_(True) if marked == "gradient" else fwad.make_dual(position, torch.ones(3))
        radar = _posed(position=position)
        try:
            table.two_way_join_forward = lambda *args: calls.append(args) or original(*args)
            with pytest.raises(RuntimeError, match="grad="):
                _verb_loss(radar, grad=grad)
        finally:
            table.two_way_join_forward = original
    assert calls == []


def test_a_pose_trajectory_rotation_leaf_is_refused():
    angle = torch.tensor(0.1, requires_grad=True)

    def pose(t):
        c, s = torch.cos(angle), torch.sin(angle)
        zero, one = torch.zeros(()), torch.ones(())
        rotation = torch.stack((torch.stack((c, -s, zero)), torch.stack((s, c, zero)), torch.stack((zero, zero, one))))
        return rotation, torch.zeros(3)

    with pytest.raises(RuntimeError, match="polarization"):
        _verb_loss_posed(pose, "vjp")
