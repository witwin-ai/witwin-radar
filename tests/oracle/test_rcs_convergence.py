"""PH-3: RCS of sampled surfaces against their analytic cross sections.

A user who wants the echo of a surface today has one recipe: spread ``N``
isotropic point sites over the illuminated surface and give each
``sigma_total / N`` (the SMPL benchmark's recipe). This oracle asks whether
that recipe converges, as the sampling is refined, to the analytic monostatic
RCS of three canonical shapes, each 5 cm in size at 77 GHz and 10 m away
(beyond the far-field distance ``2 D^2 / lambda`` of every shape):

* flat square plate at broadside, ``4 pi A^2 / lambda^2``;
* sphere in the optical region (``k a = 81``), ``pi a^2``;
* triangular trihedral on its symmetry axis, ``4 pi L^4 / (3 lambda^2)``.

The measured cross section is the narrowband (CW) one: the coherent sum of the
composed rows' complex transfers at the carrier, inverted through the
monostatic radar equation with unit gains. Pass criterion: within 1 dB of the
analytic value at each of the three finest sampling levels, because a
converged discretisation stays converged when it is refined further and a
single in-tolerance level can be a coincidence of phases.
"""

from __future__ import annotations

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

_PH3 = (
    "PH-3: N isotropic sites of sigma/N do not converge to a surface's cross section; "
    "they add coherently, so a broadside plate reads N*sigma and curved or multi-face "
    "shapes read a sampling-dependent interference sum"
)

#: shape -> (site generator, analytic RCS m^2, the three finest sampling levels).
#: Levels are per-edge cell counts for the plate and trihedral faces and the
#: point count on the sphere's hemisphere; the finest of each is below lambda/8.
SHAPES = {
    "plate": (
        lambda level: sc.plate_sites(level, side_m=SIZE_M, range_m=RANGE_M),
        ref.flat_plate_broadside_rcs_m2(SIZE_M**2, WAVELENGTH_M),
        (16, 32, 64),
    ),
    "sphere": (
        lambda level: sc.sphere_sites(level, radius_m=SIZE_M, range_m=RANGE_M),
        ref.sphere_optical_rcs_m2(SIZE_M),
        (1024, 4096, 16384),
    ),
    "trihedral": (
        lambda level: sc.trihedral_sites(level, edge_m=SIZE_M, range_m=RANGE_M),
        ref.triangular_trihedral_rcs_m2(SIZE_M, WAVELENGTH_M),
        (16, 32, 64),
    ),
}


def _measured_rcs(sites, *, rcs_per_site: float) -> float:
    from witwin.radar import PointTargets

    radar = sc.radar(power_dbm=POWER_DBM)
    positions = torch.tensor(sites, dtype=torch.float32, device=radar.device)
    paths = radar.trace(
        sc.empty_scene(), PointTargets(positions=positions, rcs=rcs_per_site), times=(0.0,), los=True, reflections=0
    )
    rows = paths.rows()
    assert rows.path_count == int(positions.shape[0])
    coherent = rows.complex_transfer_ref.to(torch.complex128).sum()
    return ref.rcs_from_monostatic_power(
        received_power_w=float(coherent.abs() ** 2),
        transmit_power_w=ref.watts_from_dbm(POWER_DBM),
        wavelength=WAVELENGTH_M,
        range_m=RANGE_M,
    )


def test_the_measurement_recovers_a_single_point_target():
    """Control: one isotropic site of known sigma reads back as that sigma."""

    measured = _measured_rcs([[RANGE_M, 0.0, 0.0]], rcs_per_site=0.37)
    assert measured == pytest.approx(0.37, rel=1e-4)


@pytest.mark.xfail(strict=True, reason=_PH3)
@pytest.mark.parametrize("shape", list(SHAPES))
def test_sampled_surface_rcs_converges_to_the_analytic_value(shape):
    generate, analytic, levels = SHAPES[shape]
    errors_db = {}
    for level in levels:
        sites = generate(level)
        measured = _measured_rcs(sites, rcs_per_site=analytic / len(sites))
        errors_db[level] = ref.db(measured / analytic)
    assert all(abs(error) <= TOLERANCE_DB for error in errors_db.values()), (
        f"{shape}: error against the analytic {ref.db(analytic):.2f} dBsm by sampling level: "
        + ", ".join(f"{level}: {error:+.2f} dB" for level, error in errors_db.items())
    )
