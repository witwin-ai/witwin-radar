"""M1-6 / PH-4: isotropic targets with their own cross section and phase.

Two isotropic targets on boresight at different ranges, each with its own
``rcs`` and ``phase``. The radar equation fixes each composed row's power, so
the ratio of the two rows is ``(sigma_1 / sigma_2) (d_2 / d_1)^4`` and the
phase difference of each row from its free-space delay phase is the declared
target phase. Before per-site vectors existed this was refused unless an
``Aspect`` was attached, and ``Aspect`` cannot express an isotropic target.
"""

from __future__ import annotations

import math

import pytest
import torch
from reference import radar_oracles as ref
from support import oracle_scenes as sc

pytestmark = pytest.mark.gpu

RANGES_M = (6.0, 9.0)
SIGMAS_M2 = (0.5, 3.0)
PHASES_RAD = (0.3, -1.1)
WAVENUMBER = 2.0 * math.pi / ref.wavelength_m(sc.CARRIER_HZ)


def _rows(targets):
    radar = sc.radar()
    rows = radar.trace(sc.empty_scene(), targets, times=(0.0,), los=True, reflections=0).rows()
    lengths = rows.total_delay_s.double().cpu() * ref.C0_M_PER_S
    order = torch.argsort(lengths)
    return lengths[order], rows.complex_transfer_ref.to(torch.complex128).cpu()[order]


def _positions():
    return [[RANGES_M[0], 0.0, 0.0], [RANGES_M[1], 0.0, 0.0]]


@pytest.mark.parametrize("ids", [None, (3_000_010, 3_000_002)], ids=["allocated", "descending"])
def test_each_target_carries_its_own_cross_section_and_phase(ids):
    from witwin.radar import PointTargets

    lengths, transfer = _rows(PointTargets(positions=_positions(), rcs=SIGMAS_M2, phase=PHASES_RAD, ids=ids))
    assert transfer.shape == (2,)
    power_ratio = float(transfer[0].abs() ** 2 / transfer[1].abs() ** 2)
    expected = (SIGMAS_M2[0] / SIGMAS_M2[1]) * (RANGES_M[1] / RANGES_M[0]) ** 4
    assert ref.db(power_ratio / expected) == pytest.approx(0.0, abs=1e-3)
    for row in range(2):
        residual = transfer[row] * complex(math.cos(WAVENUMBER * lengths[row]), math.sin(WAVENUMBER * lengths[row]))
        # S = A exp(-j phase): the residual after removing exp(-j k L) is exp(-j phase).
        wrapped = math.remainder(-math.atan2(residual.imag, residual.real) - PHASES_RAD[row], 2.0 * math.pi)
        assert abs(wrapped) < 2e-3


def test_a_per_target_cross_section_tensor_is_a_leaf_per_target():
    from witwin.radar import PointTargets

    sigma = torch.tensor(SIGMAS_M2, dtype=torch.float32, device="cuda", requires_grad=True)
    _, transfer = _rows(PointTargets(positions=_positions(), rcs=sigma))
    power = transfer.abs() ** 2
    power[0].backward()
    # |C|^2 is linear in sigma, so d|C_0|^2/d sigma_0 = |C_0|^2 / sigma_0, and the other target takes none.
    assert float(sigma.grad[0]) == pytest.approx(float(power[0].detach()) / SIGMAS_M2[0], rel=1e-4, abs=0)
    assert float(sigma.grad[1]) == 0.0


def test_a_strength_vector_of_the_wrong_length_is_refused():
    from witwin.radar import PointTargets

    with pytest.raises(ValueError, match="names 3 targets but the record declares 2"):
        PointTargets(positions=_positions(), rcs=(1.0, 2.0, 3.0))
