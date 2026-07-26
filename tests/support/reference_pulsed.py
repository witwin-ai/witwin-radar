"""float64 pure-Torch CPU oracle for the pulsed echo train.

TEST-ONLY. CLAUDE.md permits a CPU/Torch reference implementation only under
``tests/``; a production module that imported this would be introducing a Torch
numerical backend, and ``tests/test_phase4_import_boundary.py`` rejects it.

Why an oracle and not finite differences alone: the production train is float32
and the pulse's own phase is a large number - ``pi B T_p`` reaches a hundred
cycles across a 20 MHz, 10 us sweep - sitting on top of an even larger frozen
carrier phase. A naive central difference on the production chain subtracts two
nearly equal float32 numbers and can return an exactly zero derivative that
looks like a real answer. The oracle is float64 and is itself FD-validated in
float64, where the conditioning is fine; production AD is then compared against
the oracle.

This file is INDEPENDENT of both the beat oracle in ``reference_chain`` and the
CFR oracle in ``reference_ofdm``. It shares no expression with either: the three
waveforms disagree about the phasor sign, about which quantity the carrier rate
multiplies, and about what the waveform-specific factor is, and a shared helper
would quietly make one of those the other's answer.

The envelope is written here as a MASK times an analytic phase rather than as a
branch, so that the whole cube is one differentiable expression. The mask's own
derivative is zero almost everywhere, which is exactly the almost-everywhere
convention the kernel implements at a rectangular pulse's two edges; a finite
difference that straddles an edge disagrees with both, and correctly so.
"""

from __future__ import annotations

import math

import torch


def echo_cube(
    total_delay_s: torch.Tensor,
    delay_rate: torch.Tensor,
    transfer_ref: torch.Tensor,
    pair_offsets: torch.Tensor,
    spec,
) -> torch.Tensor:
    """The pulse-train sum, in the CHANNEL convention, evaluated in float64.

    Mirrors ``pulsed_echo.cu`` exactly::

        t_l    = l * pri_s
        drift  = tau_rate * t_l
        tau    = tau_rt + drift
        u      = range_gate_start_s + m * sample_period_s - tau
        cycles = -(carrier_rate * drift + carrier * tau)
                 + [ B u^2 / (2 T_p)   if lfm ]
        y[l][p][m] = sum_k C[k] * A * 1[0 <= u <= T_p]
                     * exp(+j * 2 * pi * cycles)

    Note that ``u`` is CONTINUOUS. Nothing here rounds it to a sample index, for
    the same reason the kernel does not: snapping quantises the delay by half a
    sample period and destroys the closed form the acceptance tests are written
    against. An oracle that snapped would agree with a kernel that snapped and
    both would be wrong together, which is the failure mode an oracle exists to
    prevent.

    ``transfer_ref`` is a Channel coefficient and is NOT conjugated here, or
    anywhere, because the pulsed product stays in Channel's convention.
    """

    offsets = [int(value) for value in pair_offsets.tolist()]
    num_segments = len(offsets) - 1
    pulses = torch.arange(spec.num_pulses, dtype=torch.float64)
    samples = torch.arange(spec.num_samples, dtype=torch.float64)
    t_l = (pulses * spec.pri_s).reshape(-1, 1)
    t_fast = (spec.range_gate_start_s + samples * spec.sample_period_s).reshape(1, -1)

    out = torch.zeros(
        (spec.num_pulses, num_segments, spec.num_samples),
        dtype=torch.complex128,
    )
    for segment in range(num_segments):
        for row in range(offsets[segment], offsets[segment + 1]):
            drift = delay_rate[row].to(torch.float64) * t_l
            tau = total_delay_s[row].to(torch.float64) + drift
            u = t_fast - tau
            cycles = -(
                spec.carrier_rate_hz * drift + spec.carrier_hz * tau
            )
            if spec.is_linear_fm:
                cycles = cycles + 0.5 * spec.bandwidth_hz * u * u / spec.pulse_width_s
            # HALF-OPEN, matching the kernel. A closed support would put one
            # extra sample inside the pulse at an exactly on-grid delay and not
            # otherwise, so the sampled pulse would change length with the
            # delay and the matched filter would see a mismatched tap.
            inside = ((u >= 0.0) & (u < spec.pulse_width_s)).to(torch.float64)
            envelope = spec.pulse_amplitude * inside
            phasor = torch.exp(2j * math.pi * cycles.to(torch.complex128))
            out[:, segment, :] = out[:, segment, :] + transfer_ref[row].to(
                torch.complex128
            ) * envelope.to(torch.complex128) * phasor
    return out


def echo_loss(cube: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Phase-sensitive squared-error loss, accumulated in float64.

    A magnitude-only loss would pass even with the phasor sign inverted, and
    ``.abs()`` would put a kink at zero exactly where a finite difference wants
    smoothness. This has neither problem.
    """

    delta = cube.to(torch.complex128) - reference.to(torch.complex128)
    return (delta.real**2 + delta.imag**2).sum()


def envelope_clearance_s(
    total_delay_s: torch.Tensor, delay_rate: torch.Tensor, spec
) -> float:
    """How far every sample sits from the nearest pulse-support EDGE, in seconds.

    A finite difference on ``tau_rt`` or ``tau_rate`` moves ``u`` by the step. If
    a sample crosses ``u = 0`` or ``u = T_p`` while it does, the difference
    quotient measures the envelope switching on or off - a real discontinuity -
    rather than the derivative, and it disagrees with both the kernel and this
    oracle by design. Every FD test in this suite asserts that its step is
    smaller than this clearance instead of hoping.
    """

    pulses = torch.arange(spec.num_pulses, dtype=torch.float64).reshape(-1, 1, 1)
    samples = torch.arange(spec.num_samples, dtype=torch.float64).reshape(1, 1, -1)
    rows = torch.arange(total_delay_s.numel()).reshape(1, -1, 1)
    tau = total_delay_s.to(torch.float64)[rows] + delay_rate.to(torch.float64)[
        rows
    ] * (pulses * spec.pri_s)
    u = spec.range_gate_start_s + samples * spec.sample_period_s - tau
    return float(torch.minimum(u.abs(), (u - spec.pulse_width_s).abs()).min())


__all__ = ["echo_cube", "echo_loss", "envelope_clearance_s"]
