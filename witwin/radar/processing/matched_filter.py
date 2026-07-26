"""Pulse compression against a ``PulsedEchoSpec``: the spec-facing entry.

This is the thin spec-reading layer over
:mod:`witwin.radar.processing.primitives`. The correlation itself lives there,
once, because the range-profile stage needs the same operation and two matched
filters in one package is exactly the duplication the Phase-8 cutover exists to
remove. What is here is what a spec supplies: the replica's parameters, the lag
axis, and the sample period the correlation is scaled by.

Moved from ``witwin/radar/sigproc/`` at the Phase-8 cutover so that every
production ``torch.fft`` expression in the radar processing chain lives inside
this package. ``witwin.radar.sigproc.matched_filter`` survives as a deprecation
shim that re-exports these three names and computes nothing.

The rule this module obeys, and that its test pins: **it re-derives no delay and
no phase.** It reads the pulse shape off the spec, correlates, and returns.
There is no geometry here, no carrier, no Doppler, and no knowledge of a path
row. If a future edit needs one of those, the edit is in the wrong file.

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

**The working precision is now an explicit argument.** This entry used to upcast
every signal to ``complex128`` unconditionally, which doubled the transform cost
of every pulsed frame to buy precision a ``complex64`` input never had.
``dtype`` defaults to ``None``, meaning the input's own precision; a caller that
wants the pre-cutover behaviour asks for ``torch.complex128`` and can be seen
asking.
"""

from __future__ import annotations

import torch

from .primitives import DEFAULT_WINDOW
from .primitives import matched_filter as _correlate
from .primitives import pulse_replica


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
    """

    return pulse_replica(
        pulse_sample_count=int(spec.pulse_sample_count),
        sample_period_s=float(spec.sample_period_s),
        amplitude=float(spec.pulse_amplitude),
        bandwidth_hz=float(spec.bandwidth_hz),
        pulse_width_s=float(spec.pulse_width_s),
        is_linear_fm=bool(spec.is_linear_fm),
        device=torch.device(device),
    )


def matched_filter(
    signal: torch.Tensor,
    spec,
    *,
    oversample: int = 1,
    dtype: torch.dtype | None = None,
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

    ``dtype`` is the working precision. ``None`` means the input's own, which is
    the honest default; pass ``torch.complex128`` to reproduce the pre-cutover
    behaviour exactly.
    """

    if signal.shape[-1] != spec.num_samples:
        raise ValueError(
            f"the fast-time axis holds {signal.shape[-1]} samples but the spec "
            f"declares num_samples={spec.num_samples}"
        )
    working = signal if dtype is None else signal.to(dtype)
    replica = pulse_samples(spec, device=signal.device)
    return _correlate(
        working,
        replica,
        sample_period_s=float(spec.sample_period_s),
        oversample=oversample,
        window=DEFAULT_WINDOW,
    )


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
