"""Mesh targets end to end: materials, occlusion, the site budget and vertex gradients.

Every case runs through ``Radar.trace`` on a Core scene that holds the target
mesh, so the facets' legs are Channel's own. Expected values are independent:
the Airy slab coefficient of ``tests/reference/radar_oracles.py``, and
Channel's own wall-bounce coefficient for the same material read through the
MAT-3 geometry (``tests/oracle/test_material_multipath.py``), which is the
oracle R-ADR-043 names for the facet reflection law.
"""

from __future__ import annotations

import cmath
import math

import numpy as np
import pytest
import torch
from reference import radar_oracles as ref
from support import oracle_scenes as sc

pytestmark = pytest.mark.gpu

WAVELENGTH_M = ref.wavelength_m(sc.CARRIER_HZ)
WAVENUMBER = 2.0 * math.pi / WAVELENGTH_M
RANGE_M = 10.0
#: name -> (eps_r, sigma_e S/m, thickness m): the MAT-3 materials.
MATERIALS = {"concrete-10cm": (5.24, 0.0462, 0.1), "glass-6mm": (6.0, 0.02, 0.006), "board-2mm": (2.5, 0.001, 0.002)}


def _material(name: str | None, **extra):
    from witwin.core import PhysicalMaterial

    if name is None:
        return PhysicalMaterial.perfect_conductor(**extra)
    eps_r, sigma_e, thickness = MATERIALS[name]
    return PhysicalMaterial(eps_r=eps_r, sigma_e=sigma_e, thickness_m=thickness, name=name, **extra)


def _coherent(radar, scene, targets, *, reflections: int = 0) -> complex:
    rows = radar.trace(scene, targets, times=(0.0,), los=True, reflections=reflections).rows()
    return complex(rows.complex_transfer_ref.to(torch.complex128).sum())


def _plate_scene(material, *, n: int = 2, side_m: float = 0.05, range_m: float = RANGE_M, extra=()):
    vertices, faces = sc.plate_mesh(n, side_m=side_m, range_m=range_m)
    return sc.mesh_scene(vertices, faces, material=material, extra=extra)


@pytest.mark.parametrize("material", list(MATERIALS))
def test_metal_and_dielectric_plates_differ_by_the_slab_coefficient(material):
    """M4-3 acceptance: the broadside RCS ratio is |Gamma(normal incidence)|^2, and the phase is Gamma's."""

    from witwin.radar import MeshTargets

    radar = sc.radar()
    metal = _coherent(radar, _plate_scene(_material(None)), MeshTargets())
    other = _coherent(radar, _plate_scene(_material(material)), MeshTargets())
    eps_r, sigma_e, thickness = MATERIALS[material]
    gamma = ref.slab_reflection_coefficient(
        eps_r=eps_r,
        sigma_e=sigma_e,
        thickness_m=thickness,
        frequency_hz=sc.CARRIER_HZ,
        cos_incidence=1.0,
        polarization="te",
    )
    # The conductor's co-polar coefficient is -1.
    ratio = -other / metal
    assert abs(ref.db(abs(ratio) ** 2 / abs(gamma) ** 2)) < 0.01
    assert abs(math.remainder(cmath.phase(ratio) - cmath.phase(gamma), 2.0 * math.pi)) < 5e-3


def _channel_wall_coefficient(material, *, cos_incidence: float, polarization: str) -> complex:
    """Channel's own reflection coefficient for ``material`` from a wall bounce (the MAT-3 geometry).

    A point target on boresight at ``R``, a wall parallel to boresight at
    offset ``W``; the transmit-wall-target row over the direct row is
    ``Gamma (R / L) exp(-j k (L - R))``. A wall in ``z = W`` with the ``"up"``
    (world y) polarization is TE; a wall in ``y = W`` is TM.
    """

    from witwin.radar import PointTargets

    wall = RANGE_M * cos_incidence / (2.0 * math.sqrt(1.0 - cos_incidence**2))
    unfolded = math.hypot(RANGE_M, 2.0 * wall)
    if polarization == "te":
        corners = sc.side_wall_corners(z_m=wall)
    else:
        corners = tuple((x, z, y) for x, y, z in sc.side_wall_corners(z_m=wall))
    scene = sc.plane_scene(corners=corners, material=material)
    rows = (
        sc.radar()
        .trace(scene, PointTargets(positions=[[RANGE_M, 0.0, 0.0]], rcs=1.0), times=(0.0,), reflections=1)
        .rows()
    )
    lengths = rows.total_delay_s.double().cpu().numpy() * ref.C0_M_PER_S
    transfer = rows.complex_transfer_ref.to(torch.complex128).cpu().numpy()
    order = np.argsort(lengths)
    mixed = complex(transfer[order][1] / transfer[order][0])
    coefficient = mixed / ((RANGE_M / unfolded) * cmath.exp(-1j * WAVENUMBER * (unfolded - RANGE_M)))
    if polarization == "tm":
        # The polarization lies in the plane of incidence, along the wall normal.
        # Each end of the bounced leg projects it onto the plane transverse to
        # the ray, which keeps its normal part, sin^2(theta_i); the tangential
        # part the coefficient flips is then projected away. The direct row
        # projects to one.
        coefficient /= -(1.0 - cos_incidence**2)
    return coefficient


def _bistatic_plate_coefficient(material, *, cos_incidence: float, polarization: str) -> complex:
    """The facet route's coefficient: a plate at the bistatic specular point, over the same plate in PEC.

    Transmitter and receiver sit symmetric about the plate normal, across the
    ``"up"`` polarization for TE and along it for TM, so the
    facets see incidence ``cos_incidence`` and the ratio of the two coherent
    returns is the material's co-polar coefficient over the conductor's -1.
    """

    from witwin.radar import MeshTargets

    probe = sc.radar()
    up = np.asarray(probe.polarization_vector, dtype=np.float64)
    boresight = np.array([1.0, 0.0, 0.0])
    across = np.cross(boresight, up) if polarization == "te" else up
    offset = RANGE_M * math.sqrt(1.0 - cos_incidence**2) / cos_incidence
    tx = tuple(float(v) for v in -offset * across)
    rx = tuple(float(v) for v in offset * across)

    def radar_at():
        from witwin.radar import Radar

        world = sc.radar()
        return Radar(
            carrier=sc.CARRIER_HZ,
            waveform=world.waveform,
            tx=(_local(world, tx),),
            rx=(_local(world, rx),),
            power=10.0,
            look_at=sc.LOOK_AT,
            up=sc.UP,
        )

    radar = radar_at()
    assert np.allclose(radar.tx_pos.cpu().numpy()[0], tx, atol=1e-5)
    values = [
        _coherent(radar, _plate_scene(chosen, n=1, side_m=0.2), MeshTargets()) for chosen in (_material(None), material)
    ]
    return -values[1] / values[0]


def _local(radar, world_point) -> tuple[float, float, float]:
    """Radar-local element offset that lands at ``world_point`` for an origin radar."""

    probe_axes = []
    for axis in np.eye(3):
        from witwin.radar import Radar

        moved = Radar(
            carrier=sc.CARRIER_HZ,
            waveform=radar.waveform,
            tx=(tuple(float(v) for v in axis),),
            rx=((0.0, 0.0, 0.0),),
            power=10.0,
            look_at=sc.LOOK_AT,
            up=sc.UP,
        )
        probe_axes.append(moved.tx_pos.cpu().numpy()[0])
    world_from_local = np.stack(probe_axes, axis=1)
    return tuple(float(v) for v in np.linalg.solve(world_from_local, np.asarray(world_point)))


@pytest.mark.parametrize("polarization", ["te", "tm"])
@pytest.mark.parametrize("material", ["concrete-10cm", "glass-6mm"])
def test_the_facet_coefficient_is_channels_wall_coefficient(material, polarization):
    """MAT-1 against MAT-3: the facet law and Channel's leg law agree on the same material."""

    cos_incidence = math.cos(math.radians(35.0))
    facet = _bistatic_plate_coefficient(_material(material), cos_incidence=cos_incidence, polarization=polarization)
    channel = _channel_wall_coefficient(_material(material), cos_incidence=cos_incidence, polarization=polarization)
    assert abs(abs(facet) / abs(channel) - 1.0) < 2e-3, (facet, channel)
    assert abs(math.remainder(cmath.phase(facet / channel), 2.0 * math.pi)) < 1e-2, (facet, channel)


def test_a_plate_behind_a_wall_returns_nothing_and_the_session_completes():
    """Occlusion through the legs: every facet hidden, no refusal, an exactly zero echo."""

    from witwin.core import Mesh, PhysicalMaterial, Structure

    from witwin.radar import MeshTargets

    shield = Structure(
        geometry=Mesh(
            vertices=torch.tensor(((5.0, -1.0, -1.0), (5.0, 1.0, -1.0), (5.0, 1.0, 1.0), (5.0, -1.0, 1.0))),
            faces=torch.tensor(((0, 1, 2), (0, 2, 3)), dtype=torch.int64),
            recenter=False,
            fill_mode="surface",
            topology_diagnostics=False,
        ),
        material=PhysicalMaterial.perfect_conductor(),
        structure_id=2,
        material_id=2,
        assignment_id=2,
        surface_id=2,
    )
    radar = sc.radar()
    hidden = _coherent(radar, _plate_scene(_material(None), extra=(shield,)), MeshTargets(structure_ids=(1,)))
    assert hidden == 0.0


@pytest.mark.parametrize("budget", [8, 64])
def test_a_site_budget_merges_facets_within_the_far_field_of_the_cluster(budget):
    """A merged site keeps every facet's own term; what it gives up is each facet's own wavefront curvature.

    A cluster of extent ``D`` at range ``R`` replaces each facet's exact legs by
    the cluster's legs plus a linear phase, an error of order ``k D^2 / (2 R)``
    per leg: about 0.03 rad for the 2 cm cells of an 8-site 5 cm plate at 10 m.
    """

    from witwin.radar import MeshTargets

    radar = sc.radar()
    scene = _plate_scene(_material(None), n=16)
    exact = _coherent(radar, scene, MeshTargets())
    rows = radar.trace(scene, MeshTargets(max_sites=budget), times=(0.0,), reflections=0).rows()
    assert rows.path_count <= budget
    merged = complex(rows.complex_transfer_ref.to(torch.complex128).sum())
    assert abs(ref.db(abs(merged / exact) ** 2)) < 0.2
    assert abs(cmath.phase(merged / exact)) < 0.05


def test_the_vertex_gradient_matches_finite_differences():
    """Reverse mode from the cube back to the authored mesh vertices, through legs and facets."""

    from witwin.core import Mesh, PhysicalMaterial, Scene, Structure

    from witwin.radar import MeshTargets

    vertices, faces = sc.plate_mesh(2, side_m=0.05, range_m=3.0)
    # A slight tilt so the gradient is not degenerate at the specular point.
    vertices = vertices + np.outer(vertices[:, 1], [0.08, 0.0, 0.0])
    base = torch.tensor(vertices, dtype=torch.float32)
    direction = torch.from_numpy(np.random.default_rng(5).standard_normal(base.shape)).float()

    def power(points: torch.Tensor) -> torch.Tensor:
        mesh = Mesh(
            vertices=points, faces=torch.tensor(faces), recenter=False, fill_mode="surface", topology_diagnostics=False
        )
        structure = Structure(
            geometry=mesh,
            material=PhysicalMaterial.perfect_conductor(),
            structure_id=1,
            material_id=1,
            assignment_id=1,
            surface_id=1,
        )
        scene = Scene(structures=(structure,), endpoints=[])
        rows = sc.radar().trace(scene, MeshTargets(), times=(0.0,), reflections=0, grad="vjp").rows()
        total = rows.complex_transfer_ref.sum()
        return (total.real**2 + total.imag**2) * 1e12

    leaf = base.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(power(leaf), leaf)
    analytic = float((grad.double() * direction.double()).sum())
    step = 2e-5
    with torch.no_grad():
        central = (float(power(base + step * direction)) - float(power(base - step * direction))) / (2.0 * step)
    assert analytic == pytest.approx(central, rel=3e-2, abs=0)


def test_a_still_mesh_target_costs_one_launch_per_composition_and_no_host_traffic(monkeypatch):
    """Budget: the facet route against point targets at the same sites, per marginal frame.

    The facets of a still mesh are placed once and reused, so a later frame
    launches ``facet_geometry`` zero times and ``facet_response`` once per
    composition; its synchronizations and device copies are the point
    targets' own, because nothing on the facet route reads the device.
    """

    from support.session_ledger import SessionLedger, profiled_copies

    from witwin.radar import MeshTargets, Motion, PointTargets
    from witwin.radar.facets import FacetSites

    scene = _plate_scene(_material(None), n=4)
    radar = sc.radar(sc.beat_fmcw(chirps=4))
    frame = FacetSites(structure_ids=None, carrier_hz=sc.CARRIER_HZ, cluster_size_m=None, max_sites=None).resolve(
        scene.snapshot(0.0), device=radar.device
    )
    sites = frame.sites.detach().clone()

    def run(targets, frames):
        times = (0.0, 0.01)[:frames]
        return lambda: radar.simulate(scene, targets, times=times, reflections=0, motion=Motion.chirp())

    def measure(targets):
        counts = {}
        for frames in (1, 2):
            run(targets, frames)()
            copies = profiled_copies(run(targets, frames))
            with monkeypatch.context() as patch:
                ledger = SessionLedger(patch)
                ledger.run(run(targets, frames))
            counts[frames] = (ledger.native.copy(), ledger.synchronizations, copies, ledger.trips.copy())
        return counts

    mesh = measure(MeshTargets())
    point = measure(PointTargets(positions=sites, amplitude=1.0))
    compositions = mesh[2][3]["TwoWayComposer.compose"] - mesh[1][3]["TwoWayComposer.compose"]
    assert compositions > 0
    assert mesh[2][0]["facet_geometry_forward"] - mesh[1][0]["facet_geometry_forward"] == 0
    assert mesh[2][0]["facet_response_forward"] - mesh[1][0]["facet_response_forward"] == compositions
    for frames in (1, 2):
        assert mesh[frames][1] == point[frames][1]
    marginal_mesh = {key: mesh[2][2][key] - mesh[1][2][key] for key in ("d2h", "h2d")}
    marginal_point = {key: point[2][2][key] - point[1][2][key] for key in ("d2h", "h2d")}
    assert marginal_mesh == marginal_point


class _Recede:
    """A deformation that moves every vertex of the plate away from the radar at ``speed`` m/s."""

    def __init__(self, vertex_count: int, speed: float) -> None:
        self.vertex_count = vertex_count
        self.speed = speed

    def at(self, time_s):
        from witwin.core.dynamics import DeformationState

        offsets = torch.zeros((self.vertex_count, 3), dtype=torch.float32)
        offsets[:, 0] = self.speed * float(time_s)
        return DeformationState(offsets=offsets)


def test_a_deformed_mesh_moves_its_facets_with_the_snapshot():
    """The facets are read from each snapshot's deformed vertices, so the delay follows the surface."""

    from witwin.core.dynamics import DynamicScene

    from witwin.radar import MeshTargets, Motion

    vertices, faces = sc.plate_mesh(2, side_m=0.05, range_m=RANGE_M)
    scene = DynamicScene(
        sc.mesh_scene(vertices, faces, material=_material(None)),
        structure_deformations={1: _Recede(len(vertices), speed=50.0)},
    )
    times = (0.0, 2e-3)
    paths = sc.radar().trace(scene, MeshTargets(), times=times, reflections=0, motion=Motion.static())
    delays = [float(paths.rows(index).total_delay_s.double().mean()) for index in range(2)]
    expected = 2.0 * 50.0 * times[1] / ref.C0_M_PER_S
    assert delays[1] - delays[0] == pytest.approx(expected, rel=1e-3)


def test_an_smpl_body_scatters_through_its_own_facets():
    """SMPL: the body is a mesh target of its own dielectric, with no declared cross section."""

    pytest.importorskip("smplpytorch")
    from support.smpl_models import smpl_model_root
    from witwin.core import PhysicalMaterial, Scene

    from witwin.radar.smpl import SMPLBody, SmplPoseDeformation

    model_root = smpl_model_root()
    if model_root is None:
        pytest.skip("no SMPL model files available in this checkout")
    body = SMPLBody(pose=torch.zeros(72), shape=torch.zeros(10), position=(4.0, 0.0, 0.0), model_root=model_root)
    deformation = SmplPoseDeformation(body, pose_rate=torch.zeros(72))
    skin = PhysicalMaterial(eps_r=6.4, sigma_e=42.0, thickness_m=0.01, name="skin")
    scene, targets = deformation.mesh_target(Scene(structures=(), endpoints=[]), structure_id=7, material=skin)
    rows = sc.radar().trace(scene, targets, times=(0.0,), reflections=0).rows()
    assert rows.path_count > 0
    assert float(rows.complex_transfer_ref.abs().sum()) > 0.0
