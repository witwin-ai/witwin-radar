"""The radar pose transforms, and what they mean on the scene-driven route.

``Radar._world_from_local_*`` survives Phase 11 unchanged; what changed is how a
test can OBSERVE that a pose is load bearing. The two observational tests here
used to reach the legacy trace sample
(``solvers.common.normalize_interpolated_sample``) and the float64 path oracle
under ``tests/reference``, both of which the Dirichlet route takes with it. They
now go through ``Radar.simulate``, which is where a pose actually reaches the
world: ``simulation.bind_radar_world`` publishes ``radar.tx_pos`` and
``radar.rx_pos`` - the pose-transformed world positions - as the Channel
endpoints.

``set_pose`` is gone with the mutable radar. ``Radar.replace`` is its
replacement and returns a NEW radar, so the pure-algebra case below also pins
that the original is left alone.

The three pure-algebra tests stay CPU-only. The two observational ones are
``--gpu``, because the thing being observed is a simulated frame.
"""

from __future__ import annotations

import math

import pytest
import torch
from conftest import empty_world, simulate_point_targets

from witwin.radar import Pattern, PointTargets, Radar


def _config() -> dict:
    return {
        "num_tx": 1,
        "num_rx": 1,
        "fc": 77e9,
        "slope": 60.012,
        "adc_samples": 64,
        "adc_start_time": 0,
        "sample_rate": 4400,
        "idle_time": 7,
        "ramp_end_time": 58,
        "chirp_per_frame": 1,
        "power": 12,
        "tx_loc": [[0, 0, 0]],
        "rx_loc": [[0, 0, 0]],
    }


def _local_target(x_deg: float, y_deg: float, radius: float = 2.0) -> torch.Tensor:
    direction = torch.tensor([math.tan(math.radians(x_deg)), math.tan(math.radians(y_deg)), -1.0], dtype=torch.float32)
    direction = direction / torch.linalg.norm(direction)
    return direction * radius


def _half_wave_dipole_power(angle_deg: float) -> float:
    angle_rad = math.radians(angle_deg)
    cos_angle = math.cos(angle_rad)
    if abs(cos_angle) < 1e-8:
        return 0.0
    field = math.cos(0.5 * math.pi * math.sin(angle_rad)) / cos_angle
    return field * field


def _composed_weight(radar: Radar, local_point: torch.Tensor) -> float:
    """``|C_rt|`` of the single composed row for one local target position.

    The composed weight rather than a cube peak: the two-way join publishes one
    row for this 1 x 1 front end, the pattern stage multiplies exactly that row
    by ``sqrt(G_t G_r)``, and reading it directly removes the windowing and the
    transform from a statement about an antenna pattern.
    """

    world = radar._world_from_local_points(local_point.reshape(1, 3).to(radar.device))
    result = radar.simulate(
        empty_world(), PointTargets(positions=world, rcs=1.0), times=(0.0,), los=True, reflections=0
    )
    return float(result.last_radar_paths.complex_transfer_ref.abs().max())


# ---------------------------------------------------------------------------
# The algebra
# ---------------------------------------------------------------------------


def test_radar_transforms_local_points_and_vectors():
    radar = Radar.from_dict(
        _config(), device="cpu", position=(1.0, 2.0, 3.0), look_at=(2.0, 2.0, 3.0), up=(0.0, 1.0, 0.0)
    )
    local_points = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -2.0]], dtype=torch.float32
    )

    world_points = radar._world_from_local_points(local_points)
    expected_points = torch.tensor(
        [[1.0, 2.0, 3.0], [1.0, 2.0, 4.0], [1.0, 3.0, 3.0], [3.0, 2.0, 3.0]], dtype=torch.float32
    )
    assert torch.allclose(world_points, expected_points)

    world_forward = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    local_forward = radar._local_from_world_vectors(world_forward)
    assert torch.allclose(local_forward, torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32), atol=1e-6, rtol=1e-6)


def test_radar_world_positions_follow_pose():
    radar = Radar.from_dict(
        {**_config(), "num_tx": 2, "tx_loc": [[0, 0, 0], [2, 0, 0]]},
        device="cpu",
        position=(1.0, 0.0, 0.0),
        look_at=(2.0, 0.0, 0.0),
        up=(0.0, 1.0, 0.0),
    )
    spacing = radar.wavelength / 2.0

    expected = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 2.0 * spacing]], dtype=torch.float32)
    assert torch.allclose(radar.tx_pos.cpu(), expected, atol=1e-6, rtol=1e-6)


def test_replace_returns_a_repositioned_radar_and_leaves_the_original_alone():
    """``set_pose`` mutated in place; ``replace`` cannot.

    The positional claim is the old one: a new pose moves the world antenna
    positions. What is added is the half the old API could not express - the
    radar the caller still holds is unchanged, which is what lets a radar
    captured in a closure or held by a result stay meaningful.
    """

    radar = Radar.from_dict({**_config(), "num_tx": 2, "tx_loc": [[0, 0, 0], [2, 0, 0]]}, device="cpu")
    moved = radar.replace(position=(1.0, 0.0, 0.0), look_at=(2.0, 0.0, 0.0))

    spacing = radar.wavelength / 2.0
    expected = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 2.0 * spacing]], dtype=torch.float32)
    assert torch.allclose(moved.tx_pos.cpu(), expected, atol=1e-6, rtol=1e-6)
    assert radar.position == (0.0, 0.0, 0.0)
    assert not torch.allclose(radar.tx_pos.cpu(), expected, atol=1e-6, rtol=1e-6)


def test_replace_refuses_a_field_the_radar_does_not_have():
    """``fov`` was a ``set_pose`` keyword and is deleted along with it."""

    radar = Radar.from_dict(_config(), device="cpu")
    with pytest.raises(TypeError, match="unknown fields: fov"):
        radar.replace(fov=42.0)


# ---------------------------------------------------------------------------
# What the pose does to a simulated frame
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_a_rotated_and_translated_radar_simulates_the_same_local_scene():
    """Same local geometry, different world pose, same frame.

    The two radars sit at different places and look along different world axes
    - one along ``+x``, one along ``+y`` - and each is given the SAME target in
    its own local frame. The world round trips are then identical by
    construction, so the published cubes must agree; a pose that leaked into the
    endpoint positions asymmetrically would move one of them.

    Both radars take the default ``polarization="up"``, which is derived from
    the pose and so is transverse to either boresight by construction. A
    hand-written world vector would have to be perpendicular to both.
    """

    identity = Radar.from_dict(_config(), position=(0.0, 0.0, 0.0), look_at=(1.0, 0.0, 0.0), up=(0.0, 1.0, 0.0))
    moved = Radar.from_dict(_config(), position=(1.5, -0.25, 0.5), look_at=(1.5, 0.75, 0.5), up=(0.0, 0.0, 1.0))

    target_local = (0.0, 0.0, -2.0)
    first = simulate_point_targets(identity, [target_local])
    second = simulate_point_targets(moved, [target_local])

    torch.testing.assert_close(first.cube.abs().max(), second.cube.abs().max(), rtol=1e-5, atol=0.0)
    torch.testing.assert_close(
        first.result.last_radar_paths.total_delay_s, second.result.last_radar_paths.total_delay_s, rtol=1e-6, atol=0.0
    )


@pytest.mark.gpu
def test_a_rotated_radar_evaluates_its_pattern_in_the_local_frame():
    """The dipole's 45-degree power gain, measured through the production stage.

    ``RoundTripPatternStage`` multiplies the composed weight by
    ``sqrt(G_t G_r)``. With one transmit and one receive element at the same
    point both lookups see the same angle, so the factor is the POWER gain, and
    for the half-wave dipole at 45 degrees that is the closed form below. The
    radar is posed along ``+x`` while the local target is 45 degrees off its own
    boresight, so a pattern evaluated in WORLD coordinates would read a
    completely different angle.

    ``Radar.pattern`` now defaults to isotropic, so the dipole is named here.
    Without it the stage is a proven no-op and the ratio would be exactly one
    for every angle - a green test that measures nothing.

    The offset is in the local AZIMUTH axis, and that is a physics choice
    rather than a preference. It used to be the elevation axis, because the
    polarization was a fixed world ``+z`` vector that local ``x`` mapped onto.
    ``polarization="up"`` is now the pose's own up vector, so it is local ``y``
    that runs along it and an ELEVATION offset would rotate the target out of
    the transverse plane, leaving the measured ratio as the pattern gain times a
    polarization projection. Local ``x`` maps onto world ``z``, perpendicular to
    both the boresight and the polarization, so the pattern is the only thing
    that changes. The dipole tabulates the same cut on both axes, so the
    expected number is unchanged by the swap.
    """

    radar = Radar.from_dict(
        _config(), position=(0.0, 0.0, 0.0), look_at=(1.0, 0.0, 0.0), up=(0.0, 1.0, 0.0), pattern=Pattern.dipole()
    )
    centre = _composed_weight(radar, _local_target(0.0, 0.0))
    off_axis = _composed_weight(radar, _local_target(45.0, 0.0))

    assert off_axis / centre == pytest.approx(_half_wave_dipole_power(45.0), rel=5e-3, abs=5e-3)
