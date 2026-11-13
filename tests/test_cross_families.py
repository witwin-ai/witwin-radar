"""The sampled-receiver cross families and the FMCW emitter link (R-ADR-051): forward, VJP and JVP.

Each oracle is written from the signals in long double, with the full 77 GHz
carrier phase kept and cancelled numerically: the transmitter radiates its
baseband ``s(T)`` on an oscillator at ``f_c + d_A`` from world time 0, the
field arrives as ``C exp(+j 2 pi f_c tau) s(T - tau) exp(j 2 pi (f_c + d_A)(T - tau))``
for a Channel coefficient ``C``, and

* a sampled receiver (pulsed, OFDM) mixes it down with ``exp(-j 2 pi (f_c + d_B + lo) T)``;
* an FMCW receiver multiplies its LO ``exp(j 2 pi [(f_c + d_B) T + S_B v^2 / 2])``
  by the conjugate, through its IF band.

The baseband ``s`` is built directly: an LFM pulse train with its own PRI and
stepped carrier, a coded (chip) pulse, an OFDM symbol stream from its symbol
table by an explicit subcarrier sum, and an emitter's table with the phase
advancing linearly between samples. ``tau`` derivatives are central
differences of the oracle.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.gpu

F_C = np.longdouble(77e9)
FRAME = np.longdouble("2.5e-4")
ROWS_PER_SLOT = 4
STEP = np.longdouble(1e-13)


def _routing(*, row_link, row_tx, links, aux, cells, samples, period, device="cuda"):
    from witwin.radar.synthesis.assembly import CrossRouting

    rows = 2 * ROWS_PER_SLOT
    pairs = sorted((row, cell) for cell, (begin, end, *_rest) in enumerate(cells) for row in range(begin, end))
    counts = np.zeros(rows + 1, dtype=np.int64)
    for row, _ in pairs:
        counts[row + 1] += 1
    return CrossRouting(
        row_link=torch.tensor(row_link, dtype=torch.int32, device=device),
        row_tx=torch.tensor(row_tx, dtype=torch.int32, device=device),
        link_table=torch.tensor(links, dtype=torch.float64, device=device),
        aux=aux,
        segment_rows=torch.tensor([cell[:2] for cell in cells], dtype=torch.int64, device=device),
        segment_schedule=torch.tensor([cell[2:] for cell in cells], dtype=torch.float64, device=device),
        row_segment_offsets=torch.tensor(np.cumsum(counts), dtype=torch.int64, device=device),
        row_segments=torch.tensor([cell for _, cell in pairs], dtype=torch.int64, device=device),
        rows_per_slot=ROWS_PER_SLOT,
        num_samples=samples,
        sample_period_s=period,
    )


def _rows(seed, low, high):
    generator = np.random.default_rng(seed)
    count = 2 * ROWS_PER_SLOT
    delay = generator.uniform(low, high, count).astype(np.float32)
    residual = (generator.uniform(-1.0, 1.0, count) * 1e-16).astype(np.float32)
    transfer = (generator.normal(size=count) + 1j * generator.normal(size=count)).astype(np.complex64) * 1e-3
    valid = np.ones(count, dtype=bool)
    valid[5] = False
    return transfer, delay, residual, valid


def _oracle(rows, cells, samples, period, unit, *, conjugate=False):
    """``(expected, d expected / d tau per row, unit per row)`` from ``unit(k, tau, t, v, lo) -> complex | None``."""

    transfer, delay, residual, valid, row_link = rows
    expected = np.zeros((len(cells), samples), dtype=np.complex128)
    d_tau = np.zeros((len(delay), len(cells), samples), dtype=np.complex128)
    units = np.zeros((len(delay), len(cells), samples), dtype=np.complex128)
    for cell, (begin, end, start, first, lo) in enumerate(cells):
        for n in range(samples):
            v = np.longdouble(first) + np.longdouble(n) * np.longdouble(period)
            t = np.longdouble(start) + v
            for k in range(begin, end):
                if row_link[k % ROWS_PER_SLOT] < 0 or not valid[k]:
                    continue
                tau = np.longdouble(delay[k]) + np.longdouble(residual[k])
                value = unit(k, tau, t, v, np.longdouble(lo))
                if value is None:
                    continue
                weight = np.conj(complex(transfer[k])) if conjugate else complex(transfer[k])
                expected[cell, n] += weight * value
                units[k, cell, n] = value
                plus, minus = unit(k, tau + STEP, t, v, np.longdouble(lo)), unit(k, tau - STEP, t, v, np.longdouble(lo))
                d_tau[k, cell, n] = weight * (plus - minus) / (2 * float(STEP))
    return expected, d_tau, units


def _phase(cycles):
    return np.exp(2j * np.pi * float(cycles - np.floor(cycles)))


def _check(synthesize, routing, rows, oracle, *, conjugate):
    transfer, delay, residual, valid, _ = rows
    expected, d_tau, units = oracle
    live = np.abs(units).sum(axis=0) > 0
    assert 0 < live.sum() < live.size, "the fixture must exercise both gated and live samples"
    c = torch.tensor(transfer, device="cuda")
    tau = torch.tensor(delay, device="cuda")
    lo = torch.tensor(residual, device="cuda")
    flags = torch.tensor(valid, device="cuda")
    scale = float(np.abs(transfer).max())
    out = synthesize(c, tau, lo, flags, routing).cpu().numpy()
    np.testing.assert_allclose(out, expected, rtol=0, atol=2e-5 * scale)
    # VJP of L = sum Re(conj(g) out).
    leaf_c, leaf_tau = c.clone().requires_grad_(True), tau.clone().requires_grad_(True)
    value = synthesize(leaf_c, leaf_tau, lo, flags, routing)
    cotangent = torch.randn(value.shape + (2,), generator=torch.Generator().manual_seed(4)).cuda()
    (torch.view_as_real(value) * cotangent).sum().backward()
    g = cotangent.cpu().numpy()
    g = g[..., 0] + 1j * g[..., 1]
    # out = W(C) u: dL/dRe C + j dL/dIm C = sum conj(g) u (conjugate W) or g conj(u) (identity W).
    grad_c = (np.conj(g)[None] * units).sum(axis=(1, 2)) if conjugate else (g[None] * np.conj(units)).sum(axis=(1, 2))
    grad_tau = np.real(np.conj(g)[None] * d_tau).sum(axis=(1, 2))
    np.testing.assert_allclose(leaf_c.grad.cpu().numpy(), grad_c, rtol=0, atol=2e-4 * np.abs(grad_c).max())
    np.testing.assert_allclose(leaf_tau.grad.cpu().numpy(), grad_tau, rtol=0, atol=2e-4 * np.abs(grad_tau).max())
    assert leaf_c.grad[5] == 0 and leaf_tau.grad[5] == 0
    # JVP.
    import torch.autograd.forward_ad as fwAD

    generator = np.random.default_rng(8)
    tangent_c = (generator.normal(size=len(delay)) + 1j * generator.normal(size=len(delay))).astype(np.complex64)
    tangent_tau = (generator.normal(size=len(delay)) * 1e-9).astype(np.float32)
    with fwAD.dual_level():
        dual = synthesize(
            fwAD.make_dual(c, torch.tensor(tangent_c, device="cuda")),
            fwAD.make_dual(tau, torch.tensor(tangent_tau, device="cuda")),
            lo,
            flags,
            routing,
        )
        tangent = fwAD.unpack_dual(dual).tangent.cpu().numpy()
    weight_tangent = np.conj(tangent_c) if conjugate else tangent_c
    expected_tangent = (weight_tangent[:, None, None] * units + tangent_tau[:, None, None] * d_tau).sum(axis=0)
    np.testing.assert_allclose(tangent, expected_tangent, rtol=0, atol=3e-4 * np.abs(expected_tangent).max())


# -- pulsed: an LFM train with a stepped carrier, and a coded (emitter-table) pulse ------------------

PULSED_LINKS = (
    # shift, d_A, d_B, F_B, PRI, count, kind, width, B, A, first chip, chips, periodic, step
    (0.15e-6, 2.0e5, -1.0e5, float(FRAME), 4e-6, 3, 1, 1.5e-6, 8e6, 1.0, 0, 0, 0, 3e5),
    (-0.4e-6, -3.0e5, -1.0e5, float(FRAME), 1.0e-6, 0, 2, 1.0e-6, 5e6, 1.0, 0, 5, 0, 0.0),
)
CHIPS = np.array(
    [(1.0, 0.0, 1.0e6), (0.8, 0.25, -2.0e6), (1.0, 0.5, 0.5e6), (0.6, 0.1, 0.0), (0.9, 0.7, 1.5e6)], dtype=np.float64
)
PULSED_CELLS = (
    # begin, end, slot start, first sample offset, LO offset
    (0, 4, 0.0, 0.2e-6, 0.0),
    (4, 8, 4e-6, 0.2e-6, 3e5),
    (0, 8, 8e-6, 0.2e-6, 6e5),
    (2, 6, 12e-6, 0.2e-6, 9e5),
)
PULSED_ROW_LINK = (0, 1, -1, 0)


def _pulsed_unit(k, tau, t, v, lo):
    link = PULSED_LINKS[PULSED_ROW_LINK[k % ROWS_PER_SLOT]]
    shift, d_a, d_b, frame, pri, count, kind, width, bandwidth, amplitude, first, _, _, step = (
        np.longdouble(x) for x in link
    )
    absolute = frame + t
    emitted = absolute - tau - (frame + shift)
    pulse = np.floor(emitted / pri)
    if pulse < 0 or (count > 0 and pulse >= count):
        return None
    w = emitted - pulse * pri
    if not (0 <= w < width):
        return None
    if kind == 1:
        envelope = amplitude * _phase(bandwidth * w * w / (2 * width))
    else:
        chip_width = width / len(CHIPS)
        chip = int(np.floor(w / chip_width))
        magnitude, arg, frequency = (np.longdouble(x) for x in CHIPS[int(first) + chip])
        envelope = amplitude * float(magnitude) * _phase(arg + frequency * (w - chip * chip_width))
    carrier = d_a + pulse * step
    # Received field mixed down: the f_c terms are written out and cancel numerically.
    cycles = (F_C + carrier) * (absolute - tau) - (F_C + d_b + lo) * absolute + F_C * tau
    return envelope * _phase(cycles)


def test_the_pulsed_cross_family_matches_the_signal_oracle():
    from witwin.radar.synthesis.pulsed import synthesize_pulsed_cross

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    transfer, delay, residual, valid = _rows(11, 20e-9, 300e-9)
    row_link = PULSED_ROW_LINK
    samples, period = 40, 0.05e-6
    routing = _routing(
        row_link=row_link,
        row_tx=(0,) * ROWS_PER_SLOT,
        links=PULSED_LINKS,
        aux=torch.tensor(CHIPS, device="cuda"),
        cells=PULSED_CELLS,
        samples=samples,
        period=period,
    )
    rows = (transfer, delay, residual, valid, row_link)
    oracle = _oracle(rows, PULSED_CELLS, samples, period, _pulsed_unit)
    _check(synthesize_pulsed_cross, routing, rows, oracle, conjugate=False)


# -- OFDM: another radar's symbol stream on the receiver's own grid ------------------------------------

N_SC = 16
DF = 1.0e6
CP = 0.25e-6
T_SYM = 1.0 / DF + CP
SYMBOLS = 3


def _symbol_table():
    generator = np.random.default_rng(21)
    values = np.exp(2j * np.pi * generator.integers(0, 4, (SYMBOLS, N_SC)) / 4)
    return values


OFDM_LINKS = (
    # shift, d_A, d_B, F_B, T_sym, symbols, CP, df, N, first entry, 1/sqrt(N)
    (0.1e-6, 0.15 * DF, 0.0, float(FRAME), T_SYM, SYMBOLS, CP, DF, N_SC, 0, 1.0 / math.sqrt(N_SC)),
)
OFDM_ROW_LINK = (0, -1, 0, 0)
OFDM_CELLS = ((0, 4, 0.0, CP, 0.0), (4, 8, T_SYM, CP, 0.0), (0, 8, 2 * T_SYM, CP, 0.0), (2, 6, 3 * T_SYM, CP, 0.0))


def _ofdm_unit(k, tau, t, v, lo):
    shift, d_a, d_b, frame, t_sym, symbols, cp, df, n_sc, _, _ = (np.longdouble(x) for x in OFDM_LINKS[0])
    absolute = frame + t
    emitted = absolute - tau - (frame + shift)
    j = int(np.floor(emitted / t_sym))
    if j < 0 or j >= SYMBOLS:
        return None
    u = emitted - j * t_sym - cp
    table = _symbol_table()[j]
    x = sum(table[n] * _phase(np.longdouble(n) * df * u) for n in range(N_SC)) / math.sqrt(N_SC)
    cycles = (F_C + d_a) * (absolute - tau) - (F_C + d_b + lo) * absolute + F_C * tau
    return x * _phase(cycles)


def test_the_ofdm_cross_family_matches_the_signal_oracle():
    from witwin.radar.synthesis.ofdm import synthesize_ofdm_cross

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    transfer, delay, residual, valid = _rows(13, 50e-9, 900e-9)
    table = _symbol_table().reshape(-1)
    aux = torch.tensor(np.stack((table.real, table.imag), axis=1), dtype=torch.float32, device="cuda")
    samples, period = N_SC, 1.0 / (N_SC * DF)
    routing = _routing(
        row_link=OFDM_ROW_LINK,
        row_tx=(0,) * ROWS_PER_SLOT,
        links=OFDM_LINKS,
        aux=aux,
        cells=OFDM_CELLS,
        samples=samples,
        period=period,
    )
    rows = (transfer, delay, residual, valid, OFDM_ROW_LINK)
    oracle = _oracle(rows, OFDM_CELLS, samples, period, _ofdm_unit)
    _check(synthesize_ofdm_cross, routing, rows, oracle, conjugate=False)


# -- FMCW receiver hearing an emitter's table ---------------------------------------------------------

S_B = 30e12
EMITTER_CHIPS = np.array([(1.0, 0.0, 2.5e8), (0.7, 0.3, 2.52e8), (1.0, 0.6, 2.49e8), (0.9, 0.9, 2.5e8)])
EMITTER_DURATION = 4 * 0.5e-6
FMCW_LINKS = (
    # slope, d_E, ramp end, period, slots, transmitters, shift, S_B, d_B, phase0, IF, kind, first, chips
    (0.0, 1.0e6, EMITTER_DURATION, EMITTER_DURATION, 2.0**52, 1, -3.0e-6, S_B, 0.0, None, 5e6, 1, 0, 4),
)
FMCW_ROW_LINK = (0, 0, -1, 0)
FMCW_CELLS = ((0, 4, 0.0, 8.0e-6, 0.0), (4, 8, 25e-6, 8.0e-6, 0.0), (0, 8, 50e-6, 8.2e-6, 0.0))


def _fmcw_links():
    rows = []
    for link in FMCW_LINKS:
        cycles = (np.longdouble(link[8]) - np.longdouble(link[1])) * FRAME
        rows.append(tuple(float(cycles - np.floor(cycles)) if value is None else value for value in link))
    return rows


def _emitter_unit(k, tau, t, v, lo):
    _, d_e, duration, _, _, _, shift, s_b, d_b, _, band, _, _, _ = (
        np.longdouble(0 if x is None else x) for x in FMCW_LINKS[0]
    )
    absolute = FRAME + t
    emitted = absolute - tau - (FRAME + shift)
    if emitted < 0:
        return None
    w = emitted - np.floor(emitted / duration) * duration
    chip_width = duration / len(EMITTER_CHIPS)
    chip = int(np.floor(w / chip_width))
    magnitude, arg, frequency = (np.longdouble(x) for x in EMITTER_CHIPS[chip])
    if abs(d_b - d_e + s_b * v - frequency) > band:
        return None
    lo_cycles = (F_C + d_b) * absolute + s_b * v * v / 2
    tx_cycles = (F_C + d_e) * (absolute - tau) + arg + frequency * (w - chip * chip_width)
    # LO * conj(received field), with the Channel weight's carrier phase removed.
    return float(magnitude) * _phase(lo_cycles - tx_cycles - F_C * tau)


def test_an_fmcw_receiver_hears_an_emitter_table_through_its_if_band():
    from witwin.radar.synthesis.fmcw import synthesize_fmcw_cross

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    transfer, delay, residual, valid = _rows(17, 20e-9, 300e-9)
    samples, period = 40, 0.1e-6
    routing = _routing(
        row_link=FMCW_ROW_LINK,
        row_tx=(0,) * ROWS_PER_SLOT,
        links=_fmcw_links(),
        aux=torch.tensor(EMITTER_CHIPS, device="cuda"),
        cells=FMCW_CELLS,
        samples=samples,
        period=period,
    )
    rows = (transfer, delay, residual, valid, FMCW_ROW_LINK)
    oracle = _oracle(rows, FMCW_CELLS, samples, period, _emitter_unit, conjugate=True)
    _check(synthesize_fmcw_cross, routing, rows, oracle, conjugate=True)
