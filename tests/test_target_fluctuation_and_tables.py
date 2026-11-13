"""PH-6: Swerling fluctuation and tabulated cross sections (R-ADR-045).

Swerling statistics are read from the composed rows of many isotropic sites:
the ratio of a fluctuating row's power to the same row without fluctuation is
the draw ``X``, whose first two moments are fixed by the model (exponential:
``E X = 1``, ``E X^2 = 2``; chi-square with 4 degrees of freedom, scaled:
``E X^2 = 3/2``). The decorrelation rule is read from frames inside and across
one coherent interval. A tabulated cross section is read back at known
aspects through the monostatic radar equation, and its native operator is held
to central differences and to its own adjoint.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from reference import radar_oracles as ref
from support import oracle_scenes as sc

pytestmark = pytest.mark.gpu

SITES = 1500
RANGE_M = 10.0
INTERVAL_S = 1e-3


def _positions(count: int = SITES) -> torch.Tensor:
    angle = torch.linspace(-0.3, 0.3, count, dtype=torch.float64)
    points = torch.stack((RANGE_M * torch.cos(angle), RANGE_M * torch.sin(angle), torch.zeros_like(angle)), 1)
    return points.float().cuda()


def _powers(targets, times) -> np.ndarray:
    """Per-frame, per-site row powers, ``(frames, sites)``, in site order."""

    from witwin.radar import Motion

    radar = sc.radar()
    paths = radar.trace(sc.empty_scene(), targets, times=times, reflections=0, motion=Motion.static())
    frames = []
    for index in range(len(times)):
        rows = paths.rows(index)
        order = torch.argsort(rows.topology.site_id)
        frames.append((rows.complex_transfer_ref.abs() ** 2).double()[order].cpu().numpy())
    return np.stack(frames)


@pytest.mark.parametrize(("swerling", "second_moment"), [(1, 2.0), (3, 1.5)])
def test_swerling_draws_have_the_model_moments(swerling, second_moment):
    from witwin.radar import Fluctuation, PointTargets

    positions = _positions()
    steady = _powers(PointTargets(positions=positions, rcs=1.0), (0.0,))[0]
    fluctuating = _powers(
        PointTargets(positions=positions, rcs=1.0, fluctuation=Fluctuation(swerling, INTERVAL_S, seed=4)), (0.0,)
    )[0]
    draws = fluctuating / steady
    assert np.mean(draws) == pytest.approx(1.0, abs=0.08)
    assert np.mean(draws**2) == pytest.approx(second_moment, rel=0.15)


def test_swerling_one_holds_a_scan_and_two_redraws_every_observation():
    from witwin.radar import Fluctuation, PointTargets

    positions = _positions(64)
    times = (0.0, 0.4 * INTERVAL_S, 1.2 * INTERVAL_S)
    scan = _powers(PointTargets(positions=positions, rcs=1.0, fluctuation=Fluctuation(1, INTERVAL_S, seed=9)), times)
    pulse = _powers(PointTargets(positions=positions, rcs=1.0, fluctuation=Fluctuation(2, INTERVAL_S, seed=9)), times)
    steady = _powers(PointTargets(positions=positions, rcs=1.0), times)
    np.testing.assert_allclose(scan[0] / steady[0], scan[1] / steady[1], rtol=1e-5)
    assert not np.allclose(scan[1] / steady[1], scan[2] / steady[2], rtol=1e-2)
    assert not np.allclose(pulse[0] / steady[0], pulse[1] / steady[1], rtol=1e-2)
    again = _powers(PointTargets(positions=positions, rcs=1.0, fluctuation=Fluctuation(2, INTERVAL_S, seed=9)), times)
    np.testing.assert_array_equal(pulse, again)


def test_pulse_to_pulse_fluctuation_refuses_adaptive_sampling():
    from witwin.radar import Fluctuation, Motion, PointTargets

    targets = PointTargets(positions=_positions(4), rcs=1.0, fluctuation=Fluctuation(2, INTERVAL_S))
    scene = sc.empty_scene()
    from witwin.core.dynamics import DynamicScene

    with pytest.raises(ValueError, match="Swerling II or IV"):
        sc.radar().simulate(DynamicScene(scene), targets, times=(0.0,), motion=Motion.adaptive())


def _table(aspects: np.ndarray) -> np.ndarray:
    """A cross section linear in the aspect angle, so linear interpolation is exact."""

    return 0.2 + 0.5 * aspects


def test_a_tabulated_cross_section_reads_back_at_each_target_aspect():
    from witwin.radar import PointTargets, RcsTable

    samples = np.linspace(0.0, math.pi, 37)
    aspects = np.array([0.0, 0.3, 1.1, 2.0])
    # The radar is at the origin and each target on boresight, so the bisector points back along -x.
    axis = np.stack((-np.cos(aspects), np.sin(aspects), np.zeros_like(aspects)), axis=1)
    positions = [[RANGE_M + 2.0 * index, 0.0, 0.0] for index in range(len(aspects))]
    table = RcsTable(
        axis=axis.tolist(), rcs_m2=np.stack((_table(samples), 3.0 * _table(samples))), frequencies_hz=(76e9, 78e9)
    )
    rows = sc.radar().trace(sc.empty_scene(), PointTargets(positions=positions, rcs=table), times=(0.0,), reflections=0)
    rows = rows.rows()
    order = torch.argsort(rows.topology.site_id)
    power = (rows.complex_transfer_ref.abs() ** 2).double()[order].cpu().numpy()
    wavelength = ref.wavelength_m(sc.CARRIER_HZ)
    measured = [
        ref.rcs_from_monostatic_power(
            received_power_w=value, transmit_power_w=ref.watts_from_dbm(10.0), wavelength=wavelength, range_m=r[0]
        )
        for value, r in zip(power, positions, strict=True)
    ]
    # At 77 GHz the complex amplitude is the mean of the two rows': sqrt(sigma) (1 + sqrt(3)) / 2.
    expected = _table(aspects) * ((1.0 + math.sqrt(3.0)) / 2.0) ** 2
    np.testing.assert_allclose(measured, expected, rtol=1e-3)


def test_the_table_operator_matches_finite_differences_and_its_adjoint():
    import torch.autograd.forward_ad as fwad

    from witwin.radar.scattering import _BackwardTables, _TableResponse

    device = "cuda"
    rng = np.random.default_rng(2)

    def unit(rows, mean):
        values = np.asarray(mean)[None, :] + 0.3 * rng.standard_normal((rows, 3))
        return torch.tensor(values / np.linalg.norm(values, axis=1, keepdims=True), dtype=torch.float32, device=device)

    dir_in, dir_out, axis = unit(3, (1.0, 0.0, 0.0)), unit(2, (-1.0, 0.2, 0.0)), unit(2, (-1.0, 0.5, 0.3))
    idx_in = torch.tensor([0, 1, 2, 0], device=device)
    idx_out = torch.tensor([0, 1, 0, 1], device=device)
    idx_site = torch.tensor([0, 0, 1, 1], device=device)
    samples = np.linspace(0.0, math.pi, 25)
    table_re = torch.tensor(np.cos(3 * samples) + 2.0, dtype=torch.float32, device=device)
    table_im = torch.tensor(np.sin(2 * samples), dtype=torch.float32, device=device)
    valid = torch.ones(4, dtype=torch.int32, device=device)

    def csr(keys, owners):
        keys = keys.cpu().numpy()
        offsets = np.concatenate(([0], np.cumsum(np.bincount(keys, minlength=owners))))
        return torch.tensor(offsets, device=device), torch.tensor(np.argsort(keys, kind="stable"), device=device)

    tables = _BackwardTables(*csr(idx_in, 3), *csr(idx_out, 2), *csr(idx_site, 2))

    def evaluate(a, b, c):
        return _TableResponse.apply(a, b, c, idx_in, idx_out, idx_site, valid, table_re, table_im, tables)

    tangents = [torch.randn_like(value) for value in (dir_in, dir_out, axis)]
    weights = torch.randn(2, 4, device=device)
    inputs = [value.clone().requires_grad_(True) for value in (dir_in, dir_out, axis)]
    s_re, s_im = evaluate(*inputs)
    grads = torch.autograd.grad((weights[0] * s_re + weights[1] * s_im).sum(), inputs)
    reverse = sum(float((g.double() * t.double()).sum()) for g, t in zip(grads, tangents, strict=True))
    with fwad.dual_level():
        out = evaluate(*[fwad.make_dual(v, t) for v, t in zip((dir_in, dir_out, axis), tangents, strict=True)])
        forward = float(sum((w * fwad.unpack_dual(o).tangent).sum() for w, o in zip(weights, out, strict=True)))

    def value(step):
        a, b = evaluate(
            *[(v.double() + step * t.double()).float() for v, t in zip((dir_in, dir_out, axis), tangents, strict=True)]
        )
        return float((weights[0].double() * a.double() + weights[1].double() * b.double()).sum())

    central = (value(1e-3) - value(-1e-3)) / 2e-3
    assert forward == pytest.approx(reverse, rel=1e-4, abs=0)
    assert forward == pytest.approx(central, rel=2e-2, abs=0)
