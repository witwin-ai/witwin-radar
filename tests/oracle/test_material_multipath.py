"""MAT-3: wall-bounce path level against the analytic slab reflection coefficient.

The offline MATLAB comparison checked Channel's slab coefficient; this puts
the same physics in CI through the public verbs, on a radar round trip.

One co-located element at the origin looking along +x, one isotropic target
at ``R = 10 m`` on boresight, and one flat slab wall in the plane ``z = W``.
Every ray lies in the ``y = 0`` plane and the polarization is world +y, so the
wall sees pure TE incidence and no polarization projection enters. With
co-located transmit and receive the two bounced legs share one incidence
angle, ``cos(theta_i) = 2 W / L`` with ``L = sqrt(R^2 + 4 W^2)`` the unfolded
one-way length to the mirrored target, and the image method gives the rows
relative to the direct-direct row:

    C_mixed / C_dd = Gamma (R / L) exp(-j k (L - R))
    C_rr    / C_dd = Gamma^2 (R / L)^2 exp(-j k (2 L - 2 R))

``Gamma`` is the two-interface Airy coefficient of a homogeneous slab in
vacuum (``tests/reference/radar_oracles.py``), written independently of
Channel's transfer-matrix implementation. Three materials span the regimes:
a lossy 10 cm slab (effectively a half space), and two thin low-loss slabs
whose coefficient is dominated by the interference between their faces.
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

RANGE_M = 10.0
WAVENUMBER = 2.0 * math.pi / ref.wavelength_m(sc.CARRIER_HZ)
#: name -> (eps_r, sigma_e S/m, thickness m).
MATERIALS = {"concrete-10cm": (5.24, 0.0462, 0.1), "glass-6mm": (6.0, 0.02, 0.006), "board-2mm": (2.5, 0.001, 0.002)}
WALL_OFFSETS_M = (1.0, 2.5, 5.0)
#: |ratio| to 5e-4 (0.004 dB) and phase to 5 mrad. Channel 0.5.0 agrees with
#: the Airy closed form to 3e-5 and 0.3 mrad on every case here; the margin
#: covers float32 path lengths near 28 m (~1e-6 m, 2 mrad per leg at worst)
#: and complex64 coefficients, and is still far below any physics slip (a
#: single-interface coefficient for a 2 mm board is off by 1.6 dB).
MAGNITUDE_RTOL = 5e-4
PHASE_TOL_RAD = 5e-3


@pytest.mark.parametrize("wall_m", WALL_OFFSETS_M, ids=lambda value: f"W{value:g}m")
@pytest.mark.parametrize("material", list(MATERIALS))
def test_wall_bounce_rows_carry_the_slab_coefficient(material, wall_m):
    from witwin.core import PhysicalMaterial

    from witwin.radar import PointTargets

    eps_r, sigma_e, thickness = MATERIALS[material]
    scene = sc.plane_scene(
        corners=sc.side_wall_corners(z_m=wall_m),
        material=PhysicalMaterial(eps_r=eps_r, sigma_e=sigma_e, thickness_m=thickness, name=material),
    )
    radar = sc.radar()
    paths = radar.trace(
        scene, PointTargets(positions=[[RANGE_M, 0.0, 0.0]], rcs=1.0), times=(0.0,), los=True, reflections=1
    )
    rows = paths.rows()
    assert rows.path_count == 4

    lengths = rows.total_delay_s.double().cpu().numpy() * ref.C0_M_PER_S
    transfer = rows.complex_transfer_ref.to(torch.complex128).cpu().numpy()
    order = np.argsort(lengths)
    lengths, transfer = lengths[order], transfer[order]

    unfolded = math.hypot(RANGE_M, 2.0 * wall_m)
    expected_lengths = [2 * RANGE_M, RANGE_M + unfolded, RANGE_M + unfolded, 2 * unfolded]
    np.testing.assert_allclose(lengths, expected_lengths, rtol=0, atol=2e-5)

    gamma = ref.slab_reflection_coefficient(
        eps_r=eps_r,
        sigma_e=sigma_e,
        thickness_m=thickness,
        frequency_hz=sc.CARRIER_HZ,
        cos_incidence=2.0 * wall_m / unfolded,
        polarization="te",
    )
    spread = RANGE_M / unfolded
    expected = {
        "mixed": gamma * spread * cmath.exp(-1j * WAVENUMBER * (unfolded - RANGE_M)),
        "reflected-reflected": gamma**2 * spread**2 * cmath.exp(-2j * WAVENUMBER * (unfolded - RANGE_M)),
    }
    measured = {"mixed": transfer[1:3] / transfer[0], "reflected-reflected": np.array([transfer[3] / transfer[0]])}
    for name, values in measured.items():
        ratio = values / expected[name]
        assert np.max(np.abs(np.abs(ratio) - 1.0)) < MAGNITUDE_RTOL, (name, abs(gamma), np.abs(ratio))
        assert np.max(np.abs(np.angle(ratio))) < PHASE_TOL_RAD, (name, cmath.phase(gamma), np.angle(ratio))
