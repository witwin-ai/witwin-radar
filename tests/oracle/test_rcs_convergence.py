"""PH-3: the cross section of a mesh target against the analytic one.

A surface's echo comes from its facets (``MeshTargets``, R-ADR-042): every
face of the mesh is a physical-optics facet carrying its area, its orientation
to both legs and the exact flat-facet form factor. This oracle asks whether
that converges, as the mesh is refined, to the analytic monostatic RCS of
canonical shapes, each 5 cm in size at 77 GHz and 10 m away (beyond the
far-field distance ``2 D^2 / lambda`` of every shape), all perfect conductors:

* flat square plate at broadside, ``4 pi A^2 / lambda^2``;
* sphere in the optical region (``k a = 81``), ``pi a^2``;
* right-angle dihedral at its symmetric aspect, ``8 pi a^2 b^2 / lambda^2``,
  whose double bounce needs one scene reflection before the facet
  (``reflections=1``) and sees pure TE incidence with the ``"up"`` polarization;
* triangular trihedral on its symmetry axis, ``4 pi L^4 / (3 lambda^2)``, whose
  triple bounce needs two (``reflections=2``): two geometric-optics bounces on
  the inbound leg, physical optics at the third face. The polarization rotation
  of the two bounces reaches the facet through the legs' unprojected fields and
  the facet's 2x2 scattering matrix (R-ADR-041, R-ADR-042).

The measured cross section is the narrowband (CW) one: the coherent sum of the
composed rows' complex transfers at the carrier, inverted through the
monostatic radar equation with unit gains. Pass criterion: within 1 dB of the
analytic value at each of the three finest levels, because a converged
discretisation stays converged when it is refined further and a single
in-tolerance level can be a coincidence of phases.

The isotropic-point recipe this file used to measure (``N`` sites of
``sigma / N``) is not a surface model and does not converge: it reads
``N sigma`` for a broadside plate. It is gone rather than pinned, because
``MeshTargets`` is the route a surface takes now.
"""

from __future__ import annotations

import cmath
import math

import pytest
import torch
from reference import radar_oracles as ref
from support import oracle_scenes as sc

pytestmark = pytest.mark.gpu

RANGE_M = 10.0
SIZE_M = 0.05
POWER_DBM = 10.0
TOLERANCE_DB = 1.0
WAVELENGTH_M = ref.wavelength_m(sc.CARRIER_HZ)
WAVENUMBER = 2.0 * math.pi / WAVELENGTH_M


#: shape -> (mesh generator, analytic RCS m^2, reflections, the three finest levels).
SHAPES = {
    "plate": (
        lambda level: sc.plate_mesh(level, side_m=SIZE_M, range_m=RANGE_M),
        ref.flat_plate_broadside_rcs_m2(SIZE_M**2, WAVELENGTH_M),
        0,
        (1, 4, 16),
    ),
    "sphere": (
        lambda level: sc.sphere_mesh(level, radius_m=SIZE_M, range_m=RANGE_M),
        ref.sphere_optical_rcs_m2(SIZE_M),
        0,
        (32, 64, 96),
    ),
    "dihedral": (
        lambda level: sc.dihedral_mesh(level, edge_m=SIZE_M, face_m=SIZE_M, range_m=RANGE_M),
        8.0 * math.pi * SIZE_M**4 / WAVELENGTH_M**2,
        1,
        (8, 16, 32),
    ),
    "trihedral": (
        lambda level: sc.trihedral_mesh(level, edge_m=SIZE_M, range_m=RANGE_M),
        ref.triangular_trihedral_rcs_m2(SIZE_M, WAVELENGTH_M),
        2,
        (16, 24, 32),
    ),
}


def _coherent_sum(rows) -> complex:
    return complex(rows.complex_transfer_ref.to(torch.complex128).sum())


def _rcs(coherent: complex) -> float:
    return ref.rcs_from_monostatic_power(
        received_power_w=abs(coherent) ** 2,
        transmit_power_w=ref.watts_from_dbm(POWER_DBM),
        wavelength=WAVELENGTH_M,
        range_m=RANGE_M,
    )


def _mesh_rows(vertices, faces, *, reflections: int):
    from witwin.core import PhysicalMaterial

    from witwin.radar import MeshTargets

    radar = sc.radar(power_dbm=POWER_DBM)
    scene = sc.mesh_scene(vertices, faces, material=PhysicalMaterial.perfect_conductor())
    return radar.trace(scene, MeshTargets(), times=(0.0,), los=True, reflections=reflections).rows()


def test_the_measurement_recovers_a_single_point_target():
    """Control: one isotropic site of known sigma reads back as that sigma."""

    from witwin.radar import PointTargets

    radar = sc.radar(power_dbm=POWER_DBM)
    rows = radar.trace(
        sc.empty_scene(), PointTargets(positions=[[RANGE_M, 0.0, 0.0]], rcs=0.37), times=(0.0,), reflections=0
    ).rows()
    assert _rcs(_coherent_sum(rows)) == pytest.approx(0.37, rel=1e-4)


@pytest.mark.parametrize("shape", ["plate", "sphere", "dihedral", "trihedral"])
def test_mesh_target_rcs_converges_to_the_analytic_value(shape):
    generate, analytic, reflections, levels = SHAPES[shape]
    errors_db = {}
    for level in levels:
        vertices, faces = generate(level)
        errors_db[level] = ref.db(_rcs(_coherent_sum(_mesh_rows(vertices, faces, reflections=reflections))) / analytic)
    assert all(abs(error) <= TOLERANCE_DB for error in errors_db.values()), (
        f"{shape}: error against the analytic {ref.db(analytic):.2f} dBsm by mesh level: "
        + ", ".join(f"{level}: {error:+.2f} dB" for level, error in errors_db.items())
    )


@pytest.mark.parametrize(
    ("shape", "level", "front_m", "expected_rad"),
    [("plate", 4, RANGE_M, -0.5 * math.pi), ("sphere", 96, RANGE_M, math.pi)],
)
def test_the_coherent_return_carries_the_physical_optics_phase(shape, level, front_m, expected_rad):
    """The phase reference: a PEC plate returns -j, a PEC sphere -1, against its front point.

    Physical optics carries the Huygens factor ``j``; a stationary-phase
    evaluation over a convex surface contributes ``-j``, so the sphere's
    return is real and the plate's is imaginary, each times the conductor's
    ``-1``. A facet sum with the wrong phase reference reads the right
    magnitude and the wrong phase. The facetted sphere's front facets sit
    about ``a (pi / 2n)^2`` inside the true surface, 0.04 rad of two-way phase
    at ``n = 96``, inside the 0.1 rad tolerance.
    """

    generate, _, reflections, _ = SHAPES[shape]
    coherent = _coherent_sum(_mesh_rows(*generate(level), reflections=reflections))
    residual = coherent * cmath.exp(1j * WAVENUMBER * 2.0 * front_m)
    assert abs(math.remainder(cmath.phase(residual) - expected_rad, 2.0 * math.pi)) < 0.1


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Channel 0.6.0 request: reflection discovery returns no rows off the 45-degree faces of a dihedral meshed "
        "with 52 or more cells per edge (10,816 faces and up; 50 per edge works) and still reports complete=True, "
        "in discovery='auto' and 'exhaustive' alike"
    ),
)
def test_a_finely_meshed_dihedral_keeps_its_double_bounce():
    """The dihedral of the convergence test at 64 cells per edge (16,384 faces).

    Its return is the double bounce, which arrives through the inbound legs'
    reflection rows. Channel's own discovery finds none of them at this mesh
    level (a plain source-sink request off the same mesh reproduces it without
    Radar), so the composed return falls to the single-bounce facets, about
    43 dB below the analytic cross section. A Channel fix flips this test.
    """

    generate, analytic, reflections, _ = SHAPES["dihedral"]
    error_db = ref.db(_rcs(_coherent_sum(_mesh_rows(*generate(64), reflections=reflections))) / analytic)
    assert abs(error_db) <= TOLERANCE_DB, f"dihedral at 64 cells per edge: {error_db:+.2f} dB"
