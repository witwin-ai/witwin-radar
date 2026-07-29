"""float64 CPU/Torch oracles for the receive chain.

TEST-ONLY. CLAUDE.md permits a CPU/Torch reference implementation only under
``tests/``; a production module that imported this would be introducing a Torch
numerical backend, and ``tests/test_phase4_import_boundary.py`` rejects it.

The frontend's three operators are simple enough that a value-for-value oracle
would be a second copy of a one-line expression. What is NOT trivial, and what
this file owns, is the STATISTICS a noise stage has to reproduce and the
quantiser's exact rounding rule, both of which are easy to be subtly wrong about
in ways that no shape or gradient test can see.
"""

from __future__ import annotations

import math

import torch


def quantize(signal: torch.Tensor, *, bits: int, full_scale: float) -> torch.Tensor:
    """The reference mid-tread quantiser, component by component.

    Written out rather than imported from production, because this is the
    expression the kernel is pinned against. The two things that matter and that
    a rewrite loses:

    * the level count is ``2^b`` and the step is ``2 FS / (2^b - 1)``, so the
      grid has a code at exactly ``-FS`` and one at exactly ``+FS``;
    * rounding is half-to-EVEN, which is what ``torch.round`` and ``rintf`` both
      do. Half-away-from-zero would bias every code boundary by half a step in
      the same direction, which shows up as a DC offset rather than as noise.
    """

    levels = 2 ** int(bits)
    step = (2.0 * float(full_scale)) / (levels - 1)

    def _one(component: torch.Tensor) -> torch.Tensor:
        clipped = torch.clamp(component, min=-full_scale, max=full_scale)
        return torch.round((clipped + full_scale) / step) * step - full_scale

    return torch.complex(_one(signal.real), _one(signal.imag))


def thermal_sigma_volts(
    *, noise_figure_db: float, antenna_temperature_k: float, bandwidth_hz: float, reference_impedance_ohm: float
) -> float:
    """``sqrt(k T_sys B R / 2)``, derived here independently of production.

    Independent because this is the number the whole SNR scale rests on. The
    chain from a noise figure to a per-component standard deviation has four
    places to lose a factor of two - the noise factor, the system temperature,
    the impedance, and the split between the real and imaginary components - and
    an oracle that imported the production expression would agree with all four
    mistakes.
    """

    boltzmann = 1.380649e-23
    reference_k = 290.0
    noise_factor = 10.0 ** (float(noise_figure_db) / 10.0)
    system_k = float(antenna_temperature_k) + reference_k * (noise_factor - 1.0)
    noise_power = boltzmann * system_k * float(bandwidth_hz)
    return math.sqrt(0.5 * noise_power * float(reference_impedance_ohm))


def wiener_innovation_sigma_rad(*, level_dbc_per_hz: float, offset_hz: float, sample_rate_hz: float) -> float:
    """``sqrt(10^(L/10) 4 pi^2 f_off^2 / fs)``, derived here independently."""

    level = 10.0 ** (float(level_dbc_per_hz) / 10.0)
    return math.sqrt(level * 4.0 * math.pi**2 * float(offset_hz) ** 2 / float(sample_rate_hz))


def single_sideband_psd(
    phase_rad: torch.Tensor, *, sample_rate_hz: float, segment: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Welch estimate of the single-sideband phase-noise spectrum, in 1/Hz.

    A random walk is not stationary, so each segment is mean-removed and Hann
    windowed before its periodogram; away from DC that recovers the ``1/f^2``
    asymptote the free-running model promises. The factor of two at the end is
    the double-sideband-to-single-sideband conversion: ``L(f) = S_phi(f) / 2``
    for small phase deviations, which is where a factor of two goes missing if
    it is left implicit.
    """

    values = phase_rad.detach().to(torch.float64).cpu()
    count = values.numel() // segment
    if count < 8:
        raise ValueError(
            "a phase-noise spectrum needs at least eight segments to average; "
            f"got {count}. A single periodogram of a random walk has 100 percent "
            "standard error and would make any tolerance meaningless"
        )
    window = torch.hann_window(segment, periodic=True, dtype=torch.float64)
    normalization = (window**2).sum()
    accumulated = torch.zeros(segment // 2 + 1, dtype=torch.float64)
    for index in range(count):
        block = values[index * segment : (index + 1) * segment]
        block = block - block.mean()
        accumulated += torch.fft.rfft(block * window).abs() ** 2
    accumulated /= count
    two_sided = 2.0 * accumulated / (float(sample_rate_hz) * normalization)
    frequencies = torch.fft.rfftfreq(segment, d=1.0 / float(sample_rate_hz)).double()
    return frequencies, two_sided / 2.0


def agc_gain(signal: torch.Tensor, *, target_rms: float, min_gain: float, max_gain: float) -> tuple[float, float]:
    """The reference gain and measured RMS for a whole group, in float64."""

    magnitude_sq = signal.real.double() ** 2 + signal.imag.double() ** 2
    measured = math.sqrt(max(float(magnitude_sq.mean()), 1e-24))
    gain = float(target_rms) / measured
    return min(max(gain, float(min_gain)), float(max_gain)), measured


__all__ = ["agc_gain", "quantize", "single_sideband_psd", "thermal_sigma_volts", "wiener_innovation_sigma_rad"]
