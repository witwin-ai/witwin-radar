"""Endpoint positions off a Core snapshot, in the declared batch order.

``witwin.radar.propagation.endpoint_kinematics`` is the one place a Core
``EndpointState`` becomes a Channel endpoint position: the authored antenna
position plus the snapshot's world-frame translation, in the order the caller
declares. That order is the endpoint batch order the frozen leg rows name, so
it is pinned here rather than assumed.
"""

from __future__ import annotations

import pytest
import torch

from witwin.radar import propagation


def _dynamic_endpoint_snapshot(time_s: float):
    from witwin.core import AntennaState, Scene
    from witwin.core.dynamics import DynamicScene, LinearTrajectory
    from witwin.core.identity import reserve_antenna_id

    moving = AntennaState(reserve_antenna_id(77201), "tx", torch.tensor([1.0, 2.0, 3.0]))
    still = AntennaState(reserve_antenna_id(77202), "rx", torch.tensor([-1.0, 0.0, 0.5]))
    scene = Scene(structures=(), endpoints=[moving, still])
    dynamic = DynamicScene(
        scene, endpoint_trajectories={77201: LinearTrajectory(origin=(0.0, 0.0, 0.0), velocity=(4.0, 0.0, -1.0))}
    )
    return dynamic.at(time_s)


def test_endpoint_positions_follow_the_core_composition():
    """Authored position plus the snapshot's translation, per endpoint.

    An endpoint with no trajectory sits exactly at its authored position.
    """

    snapshot = _dynamic_endpoint_snapshot(2.0)
    kinematics = propagation.endpoint_kinematics(snapshot, (77201, 77202), device="cpu")
    torch.testing.assert_close(
        kinematics.positions_m,
        torch.tensor([[9.0, 2.0, 1.0], [-1.0, 0.0, 0.5]], dtype=torch.float32),
        rtol=1e-6,
        atol=1e-7,
    )
    assert kinematics.count == 2
    assert kinematics.positions_m.is_contiguous()

    # The declared order IS the endpoint batch order, so reversing it reverses
    # the positions with it.
    reversed_kinematics = propagation.endpoint_kinematics(snapshot, (77202, 77201), device="cpu")
    assert torch.equal(reversed_kinematics.positions_m, kinematics.positions_m.flip(0))


def test_an_endpoint_the_snapshot_does_not_declare_is_named():
    snapshot = _dynamic_endpoint_snapshot(0.0)
    with pytest.raises(KeyError, match="77999"):
        propagation.endpoint_kinematics(snapshot, (77201, 77999), device="cpu")


def test_kinematics_holds_the_endpoint_contract_at_construction():
    with pytest.raises(ValueError, match="shape"):
        propagation.Kinematics(positions_m=torch.zeros(3, 2))
    with pytest.raises(TypeError, match="float32"):
        propagation.Kinematics(positions_m=torch.zeros(3, 3, dtype=torch.float64))
    with pytest.raises(ValueError, match="contiguous"):
        propagation.Kinematics(positions_m=torch.zeros(3, 6)[:, ::2])
