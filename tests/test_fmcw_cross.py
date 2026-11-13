"""The native cross-radar FMCW beat family (R-ADR-051): forward, VJP and JVP.

The oracle is written from the signals, not from the kernel's algebra: the
transmitted ramp ``exp(j 2 pi [(f_ref + d_A) T + S_A w^2 / 2])`` of radar A
(``T`` absolute time, ``w`` time into A's ramp), the receiver's LO
``exp(j 2 pi [(f_ref + d_B) T + S_B v^2 / 2])``, the received field
``C exp(+j 2 pi f_ref tau) tx(T - tau)`` for a Channel coefficient ``C``, and
the dechirped sample ``LO * conj(rx)``. It is evaluated in long double with
the full 77 GHz carrier phase, gated by A's ramp, A's TDM slot and B's IF band,
and differentiated numerically in ``tau``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.gpu

F_REF = np.longdouble(77e9)
FRAME_START = np.longdouble("1.234567e-3")
SAMPLES = 32
SAMPLE_PERIOD = 0.2e-6
ROWS_PER_SLOT = 5

# (tx slope, tx offset, tx ramp end, tx period, tx slots, tx transmitters, frame shift, rx slope, rx offset, IF)
LINKS = (
    (25.5e12, 2.0e6, 20e-6, 25e-6, 8, 2, 0.1e-6, 25e12, 0.5e6, 6e6),
    (20e12, -1.0e6, 20e-6, 25e-6, 8, 2, -0.7e-6, 25e12, 0.5e6, 6e6),
)
ROW_LINK = (0, 1, -1, 0, 1)
ROW_TX = (0, 1, 0, 1, 0)
# (row begin, row end, slot start [s], first ADC instant into the ramp [s])
CELLS = (
    (0, 5, 0.0, 1e-6),
    (5, 10, 25e-6, 1e-6),
    (0, 3, 50e-6, 1e-6),
    (2, 7, 75e-6, 1e-6),
    (5, 10, 100e-6, 1e-6),
    (0, 10, 125e-6, 1e-6),
)


def _phase0(link) -> float:
    cycles = (np.longdouble(link[8]) - np.longdouble(link[1])) * FRAME_START
    return float(cycles - np.floor(cycles))


def _link_table(device):
    rows = [(*link[:9], _phase0(link), link[9], 0.0, 0.0, 0.0) for link in LINKS]
    return torch.tensor(rows, dtype=torch.float64, device=device)


def _routing(device):
    from witwin.radar.synthesis.assembly import CrossRouting

    rows = 2 * ROWS_PER_SLOT
    pairs = [(row, cell) for cell, (begin, end, _, _) in enumerate(CELLS) for row in range(begin, end)]
    pairs.sort()
    offsets = np.zeros(rows + 1, dtype=np.int64)
    for row, _ in pairs:
        offsets[row + 1] += 1
    return CrossRouting(
        row_link=torch.tensor(ROW_LINK, dtype=torch.int32, device=device),
        row_tx=torch.tensor(ROW_TX, dtype=torch.int32, device=device),
        link_table=_link_table(device),
        aux=torch.empty((0, 3), dtype=torch.float64, device=device),
        segment_rows=torch.tensor([cell[:2] for cell in CELLS], dtype=torch.int64, device=device),
        segment_schedule=torch.tensor([(*cell[2:], 0.0) for cell in CELLS], dtype=torch.float64, device=device),
        row_segment_offsets=torch.tensor(np.cumsum(offsets), dtype=torch.int64, device=device),
        row_segments=torch.tensor([cell for _, cell in pairs], dtype=torch.int64, device=device),
        rows_per_slot=ROWS_PER_SLOT,
        num_samples=SAMPLES,
        sample_period_s=SAMPLE_PERIOD,
    )


def _rows(seed: int = 3):
    generator = np.random.default_rng(seed)
    count = 2 * ROWS_PER_SLOT
    delay = generator.uniform(20e-9, 400e-9, count).astype(np.float32)
    residual = (generator.uniform(-1.0, 1.0, count) * 1e-16).astype(np.float32)
    transfer = (generator.normal(size=count) + 1j * generator.normal(size=count)).astype(np.complex64) * 1e-3
    valid = np.ones(count, dtype=bool)
    valid[8] = False
    return transfer, delay, residual, valid


def _cycles(link, tau, t):
    """The sample's phase from the two signals, in long double, and whether it is gated in."""

    s_a, d_a, ramp, period, slots, transmitters, shift, s_b, d_b, band = (np.longdouble(v) for v in link)
    v = t[1] + t[2]
    absolute = FRAME_START + t[0] + v
    emitted = absolute - tau - (FRAME_START + shift)
    slot = np.floor(emitted / period)
    w = emitted - slot * period
    gate = (slot >= 0) and (slot < slots) and (w < ramp)
    beat = d_b - d_a + s_b * v - s_a * w
    gate = gate and abs(beat) <= band
    lo = (F_REF + d_b) * absolute + s_b * v * v / 2
    tx = (F_REF + d_a) * (absolute - tau) + s_a * w * w / 2
    return lo - tx - F_REF * tau, gate, int(slot)


def _oracle(transfer, delay, residual, valid, *, tangent=None):
    """``(out, d out / d tau per row)`` as complex128 arrays ``[cells, samples]`` and ``[rows, cells, samples]``."""

    rows = len(delay)
    out = np.zeros((len(CELLS), SAMPLES), dtype=np.complex128)
    d_tau = np.zeros((rows, len(CELLS), SAMPLES), dtype=np.complex128)
    unit = np.zeros((rows, len(CELLS), SAMPLES), dtype=np.complex128)
    step = np.longdouble(1e-13)
    for cell, (begin, end, start, first) in enumerate(CELLS):
        for n in range(SAMPLES):
            t = (np.longdouble(start), np.longdouble(first), np.longdouble(n) * np.longdouble(SAMPLE_PERIOD))
            for k in range(begin, end):
                link = ROW_LINK[k % ROWS_PER_SLOT]
                if link < 0 or not valid[k]:
                    continue
                tau = np.longdouble(delay[k]) + np.longdouble(residual[k])
                cycles, gate, slot = _cycles(LINKS[link], tau, t)
                if not gate or slot % LINKS[link][5] != ROW_TX[k % ROWS_PER_SLOT]:
                    continue
                phase = np.exp(2j * np.pi * float(cycles - np.floor(cycles)))
                value = np.conj(complex(transfer[k])) * phase
                out[cell, n] += value
                unit[k, cell, n] = phase
                rate = (_cycles(LINKS[link], tau + step, t)[0] - _cycles(LINKS[link], tau - step, t)[0]) / (2 * step)
                d_tau[k, cell, n] = value * 2j * np.pi * float(rate)
    return out, d_tau, unit


@pytest.fixture(scope="module")
def case():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    transfer, delay, residual, valid = _rows()
    return transfer, delay, residual, valid, _oracle(transfer, delay, residual, valid)


def _tensors(transfer, delay, residual, valid):
    return (
        torch.tensor(transfer, device="cuda"),
        torch.tensor(delay, device="cuda"),
        torch.tensor(residual, device="cuda"),
        torch.tensor(valid, device="cuda"),
    )


def test_the_cross_beat_matches_the_signal_oracle_including_every_gate(case):
    from witwin.radar.synthesis.fmcw import synthesize_fmcw_cross

    transfer, delay, residual, valid, (expected, _, unit) = case
    c, tau, lo, flags = _tensors(transfer, delay, residual, valid)
    out = synthesize_fmcw_cross(c, tau, lo, flags, _routing("cuda")).cpu().numpy()
    gated = np.count_nonzero(np.abs(unit).sum(axis=0) == 0)
    assert 0 < gated < unit.shape[1] * unit.shape[2], "the fixture must exercise both gated and live samples"
    scale = np.abs(transfer).max()
    np.testing.assert_allclose(out, expected, rtol=0, atol=2e-6 * scale)
    # Exactly zero where every row is gated out.
    assert np.all(out[np.abs(expected) == 0] == 0)


def test_the_cross_beat_vjp_matches_the_oracle_and_skips_dead_rows(case):
    from witwin.radar.synthesis.fmcw import synthesize_fmcw_cross

    transfer, delay, residual, valid, (_, d_tau, unit) = case
    c, tau, lo, flags = _tensors(transfer, delay, residual, valid)
    c.requires_grad_(True)
    tau.requires_grad_(True)
    out = synthesize_fmcw_cross(c, tau, lo, flags, _routing("cuda"))
    cotangent = torch.randn(out.shape + (2,), generator=torch.Generator().manual_seed(5)).cuda()
    (torch.view_as_real(out) * cotangent).sum().backward()
    g = cotangent.cpu().numpy()
    g = g[..., 0] + 1j * g[..., 1]
    # L = sum Re(conj(g) * out); out is conj(C) * unit, so dL/dRe C + j dL/dIm C = sum conj(g) * unit.
    expected_c = (np.conj(g[None]) * unit).sum(axis=(1, 2))
    expected_tau = np.real(np.conj(g[None]) * d_tau).sum(axis=(1, 2))
    np.testing.assert_allclose(c.grad.cpu().numpy(), expected_c, rtol=0, atol=1e-4 * np.abs(expected_c).max())
    np.testing.assert_allclose(tau.grad.cpu().numpy(), expected_tau, rtol=2e-4, atol=1e-4 * np.abs(expected_tau).max())
    assert c.grad[8] == 0 and tau.grad[8] == 0
    assert c.grad[2] == 0 and tau.grad[7] == 0  # unlinked rows


def test_the_cross_beat_jvp_matches_the_oracle(case):
    import torch.autograd.forward_ad as fwAD

    from witwin.radar.synthesis.fmcw import synthesize_fmcw_cross

    transfer, delay, residual, valid, (_, d_tau, unit) = case
    c, tau, lo, flags = _tensors(transfer, delay, residual, valid)
    generator = np.random.default_rng(9)
    tangent_c = (generator.normal(size=len(delay)) + 1j * generator.normal(size=len(delay))).astype(np.complex64)
    tangent_tau = (generator.normal(size=len(delay)) * 1e-9).astype(np.float32)
    with fwAD.dual_level():
        out = synthesize_fmcw_cross(
            fwAD.make_dual(c, torch.tensor(tangent_c, device="cuda")),
            fwAD.make_dual(tau, torch.tensor(tangent_tau, device="cuda")),
            lo,
            flags,
            _routing("cuda"),
        )
        tangent = fwAD.unpack_dual(out).tangent.cpu().numpy()
    expected = (np.conj(tangent_c)[:, None, None] * unit + tangent_tau[:, None, None] * d_tau).sum(axis=0)
    np.testing.assert_allclose(tangent, expected, rtol=0, atol=2e-4 * np.abs(expected).max())


def test_one_launch_per_direction(case, monkeypatch):
    from witwin.radar.cuda import runtime
    from witwin.radar.synthesis.fmcw import synthesize_fmcw_cross

    table = runtime.build_extension()
    counts: dict[str, int] = {}
    for name in ("fmcw_cross_forward", "fmcw_cross_backward", "fmcw_cross_jvp"):
        original = getattr(table, name)

        def counting(*args, _name=name, _original=original, **kwargs):
            counts[_name] = counts.get(_name, 0) + 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(table, name, counting, raising=False)
    transfer, delay, residual, valid, _ = case
    c, tau, lo, flags = _tensors(transfer, delay, residual, valid)
    c.requires_grad_(True)
    out = synthesize_fmcw_cross(c, tau, lo, flags, _routing("cuda"))
    torch.view_as_real(out).sum().backward()
    assert counts == {"fmcw_cross_forward": 1, "fmcw_cross_backward": 1}
