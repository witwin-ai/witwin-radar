"""float64 pure-Torch CPU oracle for the OFDM channel frequency response.

TEST-ONLY. CLAUDE.md permits a CPU/Torch reference implementation only under
``tests/``; a production module that imported this would be introducing a Torch
numerical backend, and ``tests/test_phase4_import_boundary.py`` rejects it.

Why an oracle and not finite differences alone: the production cube is float32
and the OFDM subcarrier phase is a small increment sitting on top of a large
frozen carrier phase, so a naive central difference on the production chain
subtracts two nearly equal float32 numbers and can return an exactly zero
derivative that looks like a real answer. The oracle is float64 and is itself
FD-validated in float64, where the conditioning is fine; production AD is then
compared against the oracle.

This file is INDEPENDENT of the beat oracle in ``reference_chain``. It shares no
expression with it: the two waveforms disagree about the phasor sign, about
which quantity the carrier rate multiplies, and about what slow time means, and
a shared helper would quietly make one of those three the other's answer.
"""

from __future__ import annotations

import math

import torch


def cfr_cube(
    total_delay_s: torch.Tensor, delay_rate: torch.Tensor, transfer_ref: torch.Tensor, pair_offsets: torch.Tensor, spec
) -> torch.Tensor:
    """The CFR sum, in the CHANNEL convention, evaluated in float64.

    Mirrors ``ofdm_cfr.cu`` exactly::

        t_l      = l * symbol_period_s
        drift    = tau_rate * t_l
        tau      = tau_rt + drift
        f_sub    = n * subcarrier_spacing_hz
        cycles   = -(f_sub * tau + carrier_rate * drift + carrier * tau)
        H[l][p][n] = sum_k C[k] * exp(+j * 2 * pi * cycles)

    Note which quantity each term multiplies. The subcarrier term takes the FULL
    delay because the ``n * df`` phase is not in the coefficient at any delay;
    the carrier-rate term takes the DRIFT because the ``f_ref`` phase is already
    in the coefficient, frozen at ``tau_rt``. Swapping them is the single most
    likely implementation error in this waveform and it produces a cube that
    still looks like a channel response.

    ``transfer_ref`` is a Channel coefficient and is NOT conjugated here, or
    anywhere, because the OFDM product stays in Channel's convention.
    """

    offsets = [int(value) for value in pair_offsets.tolist()]
    num_segments = len(offsets) - 1
    symbols = torch.arange(spec.num_symbols, dtype=torch.float64)
    subcarriers = torch.arange(spec.num_subcarriers, dtype=torch.float64)
    t_l = (symbols * spec.symbol_period_s).reshape(-1, 1)
    f_sub = (subcarriers * spec.subcarrier_spacing_hz).reshape(1, -1)

    out = torch.zeros((spec.num_symbols, num_segments, spec.num_subcarriers), dtype=torch.complex128)
    for segment in range(num_segments):
        for row in range(offsets[segment], offsets[segment + 1]):
            drift = delay_rate[row].to(torch.float64) * t_l
            tau = total_delay_s[row].to(torch.float64) + drift
            cycles = -(f_sub * tau + spec.carrier_rate_hz * drift + spec.carrier_hz * tau)
            phasor = torch.exp(2j * math.pi * cycles.to(torch.complex128))
            out[:, segment, :] = out[:, segment, :] + transfer_ref[row].to(torch.complex128) * phasor
    return out


def cfr_loss(cube: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Phase-sensitive squared-error loss, accumulated in float64.

    A magnitude-only loss would pass even with the phasor sign inverted, and
    ``.abs()`` would put a kink at zero exactly where a finite difference wants
    smoothness. This has neither problem.
    """

    delta = cube.to(torch.complex128) - reference.to(torch.complex128)
    return (delta.real**2 + delta.imag**2).sum()


__all__ = ["cfr_cube", "cfr_loss"]
