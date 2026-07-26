"""Pulse compression: the matched filter, in DSP glue.

This module is on the Torch/FFT side of the plan's processing exception, and
that placement is the deliberate half of a split. The synthesis kernel
(``witwin/radar/cuda/kernels/pulsed_echo.cu``) produces the received pulse train
- the matched-filter INPUT - and stops there. The filter itself is a correlation
against a chosen replica, and a correlation carries modelling decisions (which
replica, which window, how much oversampling) that belong to a processing chain
a user may replace, not to a propagation model they may not.

The rule this module obeys, and that its test pins: **it re-derives no delay and
no phase.** It reads the pulse shape off the spec, correlates, and returns. There
is no geometry here, no carrier, no Doppler, and no knowledge of a path row. If
a future edit needs one of those, the edit is in the wrong file.

The replica built here and the pulse the kernel evaluates are the same analytic
function, and that is the point rather than a duplication to be factored out:
one is a continuous function evaluated at a fractional delay inside a CUDA
kernel, the other is a discrete sequence on the ADC grid. The matched-filter
peak being exactly the path coefficient is the statement that they agree, and it
is asserted rather than assumed.

Conventions, all three of which change the answer:

* The correlation carries a factor ``T_s`` so that it approximates the
  continuous integral ``integral y(s) conj(p(s - t)) ds`` rather than a bare
  sum. With the unit-ENERGY pulse the spec pins, the peak for a single path row
  is then exactly its complex coefficient, with no sample-count factor.
* The lag axis is ``m * T_s / oversample`` measured from the range-gate start,
  so a peak at lag ``t`` means a round-trip delay of ``range_gate_start_s + t``.
* Correlation, not convolution: the replica is CONJUGATED and NOT reversed.
  Convolving instead would put the peak at ``tau + T_p`` and would look entirely
  plausible.
"""

from __future__ import annotations

import math

import torch


def pulse_samples(spec, device: torch.device | str = "cpu") -> torch.Tensor:
    """The transmitted replica on the ADC grid, as ``complex128[M_p]``.

    ``M_p = round(T_p / T_s)`` samples at ``u = m T_s``, with the analytic
    amplitude ``1 / sqrt(T_p)`` and the analytic phase. In float64 because it is
    a per-frame constant of a few hundred elements whose accuracy sets the
    accuracy of every delay estimate downstream; the saving from float32 here
    would be measured in kilobytes.

    Discrete energy. ``sum_m |p[m]|^2 T_s`` is exactly ``M_p T_s / T_p``, which
    is 1 when the pulse spans a whole number of samples and is reported by
    ``spec.pulse_grid_is_commensurate`` when it does not. The replica is NOT
    renormalised to force the discrete energy to 1: doing so would make this
    sequence a different function from the one the kernel evaluates, and the
    matched-filter peak identity - the thing that ties the two together - would
    then be true by construction instead of being a measurement.

    The LFM phase is accumulated in float64 and wrapped before the trigonometric
    call for the same reason the kernel does it: ``pi B u^2 / T_p`` reaches
    ``pi B T_p`` radians, a hundred cycles at a 20 MHz, 10 us sweep, and a
    single-precision argument there would cost about 1e-5 rad.
    """

    count = spec.pulse_sample_count
    if count < 1:
        raise ValueError(
            f"the pulse spans {count} samples on this grid: pulse_width_s="
            f"{spec.pulse_width_s} is shorter than sample_period_s="
            f"{spec.sample_period_s}, so there is no replica to correlate against"
        )
    u = torch.arange(count, dtype=torch.float64, device=device) * spec.sample_period_s
    amplitude = torch.full((count,), spec.pulse_amplitude, dtype=torch.float64,
                           device=device)
    if spec.is_linear_fm:
        cycles = 0.5 * spec.bandwidth_hz * u * u / spec.pulse_width_s
        phase = math.tau * (cycles - torch.floor(cycles))
    else:
        phase = torch.zeros_like(u)
    return torch.complex(amplitude * torch.cos(phase), amplitude * torch.sin(phase))


def matched_filter(
    signal: torch.Tensor, spec, *, oversample: int = 1
) -> torch.Tensor:
    """Correlate the fast-time axis against ``conj(p)``. Same rank as ``signal``.

    ``signal`` is the received train ``[..., num_samples]`` from the pulsed
    synthesis owner. The result is ``[..., num_samples * oversample]``, indexed
    by lag from the range-gate start.

    The transform length is ``num_samples + M_p`` rather than ``num_samples``, so
    that the negative lags - the correlation's left tail, which is real and
    extends a full pulse width - do not wrap around onto the positive ones. A
    circular correlation over ``num_samples`` would fold that tail onto the far
    end of the gate and invent an echo there.

    ``oversample`` inserts zeros in the middle of the lag spectrum, which is
    exact band-limited interpolation of the sampled correlation rather than a
    smoothing. It exists because the range cell can be a couple of samples wide:
    a three-point parabolic fit on the raw grid then measures its own truncation
    error instead of the peak. It changes only the lag GRID, never the values on
    the original grid.
    """

    if oversample < 1:
        raise ValueError(f"oversample must be at least 1, got {oversample}")
    if signal.shape[-1] != spec.num_samples:
        raise ValueError(
            f"the fast-time axis holds {signal.shape[-1]} samples but the spec "
            f"declares num_samples={spec.num_samples}"
        )
    replica = pulse_samples(spec, device=signal.device)
    length = spec.num_samples + replica.shape[0]
    spectrum = torch.fft.fft(signal.to(torch.complex128), n=length, dim=-1) * torch.conj(
        torch.fft.fft(replica, n=length)
    )
    if oversample > 1:
        # Zero-insert the upper half of the spectrum. The scale keeps the
        # inverse transform an interpolation of the same signal rather than a
        # rescaled copy of it.
        fine = torch.zeros(
            (*spectrum.shape[:-1], length * oversample),
            dtype=spectrum.dtype,
            device=spectrum.device,
        )
        half = length // 2
        fine[..., :half] = spectrum[..., :half]
        fine[..., length * oversample - (length - half):] = spectrum[..., half:]
        spectrum = fine * oversample
        length = length * oversample
    correlation = torch.fft.ifft(spectrum, n=length, dim=-1)
    return correlation[..., : spec.num_samples * oversample] * spec.sample_period_s


def lag_axis(spec, *, oversample: int = 1, device: torch.device | str = "cpu"):
    """The delay each :func:`matched_filter` output sample corresponds to.

    ``range_gate_start_s + m * T_s / oversample``, in seconds. Returned as a
    float64 tensor so that a peak location is read in SECONDS and never as a bin
    index: a bin index is a statement about the sampling grid, and every
    cross-waveform delay comparison in this package is a statement about the
    physics.
    """

    if oversample < 1:
        raise ValueError(f"oversample must be at least 1, got {oversample}")
    steps = torch.arange(
        spec.num_samples * oversample, dtype=torch.float64, device=device
    )
    return spec.range_gate_start_s + steps * (spec.sample_period_s / oversample)


__all__ = ["lag_axis", "matched_filter", "pulse_samples"]
