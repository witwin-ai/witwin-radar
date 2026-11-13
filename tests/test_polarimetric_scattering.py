"""M4-4 / PH-6 / MP-9: Radar owns the element polarization and the site scattering matrix.

Channel replays ``complex3_transport`` (the unprojected field vector at the
site) for both legs, the join divides Channel's element dipole magnitude out
and applies the site's 2x2 scattering matrix in the site's backscatter-aligned
basis (R-ADR-041). These are the acceptance oracles that are not the off-plane
radar equation (``tests/oracle/test_radar_equation.py``):

* a dihedral's HH and VV echoes differ by 180 degrees with equal magnitude;
* a sphere has no cross-polar echo, whether it is the scalar isotropic
  response or the matrix ``diag(1, 1)``;
* the matrix ``diag(1, 1)`` reproduces the scalar isotropic echo on a
  monostatic geometry, which is what makes the scalar the special case;
* a cross-polar matrix element is read by the cross-polar channel only.

Pose: the radar sits at the origin looking along world +x with world +y up, so
``polarization="up"`` is +y and ``"right"`` is +z. The site axis is world +y,
so for the outgoing wave ``k = +x``: ``h = up x k = -z`` and ``v = h x k = -y``;
an ``"up"`` element is V and a ``"right"`` element is H (each up to the same
sign on transmit and receive, which cancels in a co-polar echo).
"""

from __future__ import annotations

import pytest
import torch
from support import oracle_scenes as sc

pytestmark = pytest.mark.gpu

RANGE_M = 5.0
SITE_UP = (0.0, 1.0, 0.0)


def _echo(polarization, rx_polarization=None, scattering=None) -> complex:
    from witwin.radar import PointTargets

    radar = sc.radar(polarization=polarization, rx_polarization=rx_polarization)
    targets = PointTargets(positions=[[RANGE_M, 0.0, 0.0]], rcs=1.0, scattering=scattering)
    result = radar.simulate(sc.empty_scene(), targets, times=(0.0,), los=True, reflections=0)
    return complex(result.cube.reshape(-1)[0].item())


def test_a_dihedral_returns_hh_and_vv_180_degrees_apart():
    from witwin.radar import ScatteringMatrix

    dihedral = ScatteringMatrix.dihedral(up=SITE_UP)
    hh = _echo("right", scattering=dihedral)
    vv = _echo("up", scattering=dihedral)
    assert abs(hh) > 0.0
    ratio = vv / hh
    assert abs(ratio + 1.0) < 1e-5, ratio


def test_a_sphere_has_no_cross_polar_echo():
    from witwin.radar import ScatteringMatrix

    co = _echo("up")
    for scattering in (None, ScatteringMatrix.sphere(up=SITE_UP)):
        cross = _echo("up", rx_polarization="right", scattering=scattering)
        assert abs(cross) <= 1e-6 * abs(co), (scattering, cross, co)


def test_the_identity_matrix_reproduces_the_scalar_isotropic_echo():
    from witwin.radar import ScatteringMatrix

    for polarization in ("up", "right"):
        scalar = _echo(polarization)
        matrix = _echo(polarization, scattering=ScatteringMatrix.sphere(up=SITE_UP))
        assert abs(matrix / scalar - 1.0) < 1e-5, (polarization, matrix, scalar)


def test_a_cross_polar_element_is_read_only_by_the_cross_polar_channel():
    from witwin.radar import ScatteringMatrix

    # HV = receive H, transmit V: a V transmitter and an H receiver.
    coupling = ScatteringMatrix(hh=0.0, hv=0.5, vh=0.0, vv=0.0, up=SITE_UP)
    reference = _echo("up")
    assert abs(_echo("up", scattering=coupling)) <= 1e-6 * abs(reference)
    assert abs(_echo("right", scattering=coupling)) <= 1e-6 * abs(reference)
    cross = _echo("up", rx_polarization="right", scattering=coupling)
    # |HV| = 0.5 of the strength, up to the basis signs of h and v.
    assert abs(abs(cross) / abs(reference) - 0.5) < 1e-5, (cross, reference)


def test_the_matrix_response_refuses_a_derivative_on_its_site_axis():
    from witwin.radar.scattering import ScatteringMatrixResponse

    up = torch.tensor([0.0, 1.0, 0.0], requires_grad=True)
    with pytest.raises(Exception, match="site polarization basis"):
        ScatteringMatrixResponse(matrix=torch.ones(4, dtype=torch.complex64), up=up)


@pytest.mark.parametrize(
    "narrowed",
    [
        {"fixed_topology_responses": frozenset({"scalar_transport"})},
        {"response_components": (("complex3_transport", frozenset({"los"})),)},
    ],
    ids=["no-fixed-topology-field", "no-reflection-field"],
)
def test_a_channel_without_the_field_response_is_refused_by_name(monkeypatch, narrowed):
    """No fallback to scalar_transport: a scalar cannot carry a scattering matrix."""

    import dataclasses

    from witwin.channel.propagation import consumer

    from witwin.radar.channel import ChannelPropagationAdapter

    capabilities = dataclasses.replace(consumer.capabilities(), **narrowed)
    monkeypatch.setattr(consumer, "capabilities", lambda: capabilities)
    with pytest.raises(NotImplementedError, match="complex3_transport"):
        ChannelPropagationAdapter(
            object(), reference_frequency_hz=77e9, components=frozenset({"los", "reflection"}), max_depth=1
        )


def test_the_element_factor_reads_the_polarization_the_pose_turned():
    """A 60 degree yaw turns the y-polarized elements to 30 degrees off the line of sight's normal.

    Channel launches the turned polarization, so its fields at the site carry
    the dipole projection |P_perp(d) p| = cos 60 = 0.5 on each leg; the
    element factor 1/|P_perp(d) p| of each leg's own launch polarization undoes
    it, and with an isotropic pattern the echo is the unposed one. A factor
    read from the authored polarization would leave a quarter of it.
    """

    import math

    from witwin.radar import PointTargets

    angle = math.radians(60.0)
    yaw = torch.tensor(
        [[math.cos(angle), -math.sin(angle), 0.0], [math.sin(angle), math.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )
    radar = sc.radar(polarization=(0.0, 1.0, 0.0))
    targets = PointTargets(positions=[[RANGE_M, 0.0, 0.0]], rcs=1.0)

    def echo(pose):
        result = radar.simulate(sc.empty_scene(), targets, times=(0.0,), los=True, reflections=0, pose=pose)
        return complex(result.cube.reshape(-1)[0].item())

    still = echo(None)
    turned = echo(lambda time_s: (yaw, torch.zeros(3)))
    assert abs(abs(turned) / abs(still) - 1.0) < 1e-5
