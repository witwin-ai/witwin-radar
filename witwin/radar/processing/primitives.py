"""The small shared Torch pieces every processing stage is built from.

Two things live here and nothing else: the window family, and the pulsed
matched filter's replica plus its correlation. Both are shared by more than one
stage, and both are exactly the sort of thing that ends up written three times
with three slightly different conventions if it has no owner - which is what
happened to the range FFT in ``sigproc``.

**Windows are built from ``arange`` and ``cos``, not from
``torch.hamming_window``.** The design's frozen vendor-DSP primitive list does
not contain the window constructors, and this package is not allowed to grow
that list. Building them explicitly costs three lines, keeps the surface frozen,
and follows what ``sigproc/microdoppler.py`` already does for its Hann window.
The definitions here are PERIODIC, matching ``microdoppler.py``: a symmetric
window makes the transform of a pure tone asymmetric about its bin, which is
exactly the property an exact-bin assertion leans on.

**The matched filter here is the same correlation as
``sigproc/matched_filter.py`` with one deliberate difference: it does not upcast
the signal to ``complex128``.** The upcast doubles the transform cost of every
pulsed frame to buy precision the float32 input never had. The REPLICA is still
built in float64, because it is a per-frame constant of a few hundred elements
whose accuracy sets the accuracy of every delay estimate downstream, and it is
cast to the signal's dtype only at the multiply.
"""

from __future__ import annotations

import math

import torch


#: The window family, named so a caller comparing against an analytic
#: unwindowed spectrum can turn it off by name rather than by passing ``None``
#: and hoping.
WINDOWS = ("rectangular", "hann", "hamming", "blackman")

#: What ``window=None`` means. Named rather than special cased, so that every
#: published :class:`RangeProfile` carries a window string and never a ``None``
#: that a reader has to interpret.
DEFAULT_WINDOW = "rectangular"


def window_values(
    name: str, length: int, *, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """The periodic window ``w[0..length-1]``, as a real tensor.

    ``rectangular`` is exactly ones, so a caller asserting against an analytic
    unwindowed transform gets the analytic answer bit for bit rather than a
    window that is one to within rounding.
    """

    if name not in WINDOWS:
        raise ValueError(f"window must be one of {WINDOWS}, got {name!r}")
    if type(length) is not int or length < 1:
        raise ValueError(f"length must be a positive int, got {length!r}")
    if name == "rectangular":
        return torch.ones(length, dtype=dtype, device=device)
    index = torch.arange(length, dtype=dtype, device=device)
    turn = index * (2.0 * math.pi / length)
    if name == "hann":
        return 0.5 - 0.5 * torch.cos(turn)
    if name == "hamming":
        return 0.54 - 0.46 * torch.cos(turn)
    return 0.42 - 0.5 * torch.cos(turn) + 0.08 * torch.cos(2.0 * turn)


def real_dtype_of(tensor: torch.Tensor) -> torch.dtype:
    """The real dtype a window must be built in to multiply ``tensor``."""

    return torch.float64 if tensor.dtype == torch.complex128 else torch.float32


#: ``mean(w)`` in closed form, per window. Written down rather than reduced on
#: the device: a ``float(w.mean())`` is a device-to-host read, and this package
#: is post-synthesis Torch by the owner directive, not permission to add an
#: unattributed synchronization to every processing call. The identities are
#: exact for a PERIODIC window because ``sum_n cos(2 pi k n / N)`` is exactly
#: zero for every integer ``k`` that is not a multiple of ``N``.
_COHERENT_GAIN = {
    "rectangular": 1.0,
    "hann": 0.5,
    "hamming": 0.54,
    "blackman": 0.42,
}


def window_coherent_gain(name: str, length: int) -> float:
    """``mean(w)``, on the host, with no device reduction.

    The closed form needs the cosine sums to vanish, which they do once the
    window is at least three samples long (the Blackman term is at ``2 n / N``).
    Below that the mean is computed from the two or one values directly, so the
    published gain is never a formula that quietly stops holding at a degenerate
    length.
    """

    if name not in WINDOWS:
        raise ValueError(f"window must be one of {WINDOWS}, got {name!r}")
    if type(length) is not int or length < 1:
        raise ValueError(f"length must be a positive int, got {length!r}")
    if length >= 3 or name == "rectangular":
        return _COHERENT_GAIN[name]
    values = window_values(name, length, dtype=torch.float64, device=torch.device("cpu"))
    return float(values.sum()) / length


def taper(signal: torch.Tensor, name: str, *, dim: int) -> torch.Tensor:
    """Multiply ``signal`` by the window along ``dim``."""

    length = int(signal.shape[dim])
    values = window_values(
        name, length, dtype=real_dtype_of(signal), device=signal.device
    )
    shape = [1] * signal.dim()
    shape[dim] = length
    return signal * values.reshape(shape)


def remove_mean(signal: torch.Tensor, *, dim: int) -> torch.Tensor:
    """Subtract the mean along ``dim``.

    This is the operation ``sigproc``'s ``process_rd_tensor`` applies to fast
    time UNCONDITIONALLY and undocumented. Here it is a named function behind an
    explicit flag that defaults to off, because it is a clutter operation: a
    component-export test that asks for the clutter cube and silently gets the
    clutter removed from it would be comparing two different quantities.
    """

    return signal - signal.mean(dim=dim, keepdim=True)


def pulse_replica(
    *,
    pulse_sample_count: int,
    sample_period_s: float,
    amplitude: float,
    bandwidth_hz: float,
    pulse_width_s: float,
    is_linear_fm: bool,
    device: torch.device,
) -> torch.Tensor:
    """The transmitted replica on the ADC grid, as ``complex128[M_p]``.

    The same analytic function the pulsed synthesis kernel evaluates at a
    continuous fractional delay, sampled at ``u = m T_s``. Deliberately NOT
    renormalised to unit discrete energy: doing so would make this sequence a
    different function from the one the kernel evaluates, and the
    matched-filter peak identity would then be true by construction instead of
    being a measurement.

    The LFM phase is accumulated in float64 and wrapped before the
    trigonometric call, because ``pi B u^2 / T_p`` reaches a hundred cycles at a
    20 MHz, 10 us sweep and a single-precision argument there costs about
    1e-5 rad.

    The arguments are scalars rather than a spec object on purpose: this module
    is shared by three waveforms and must not learn what a ``PulsedEchoSpec``
    is. The axes record reads the spec once and passes the numbers.
    """

    if type(pulse_sample_count) is not int or pulse_sample_count < 1:
        raise ValueError(
            f"the pulse spans {pulse_sample_count} samples on this grid: it is "
            "shorter than one sample period, so there is no replica to "
            "correlate against"
        )
    u = torch.arange(pulse_sample_count, dtype=torch.float64, device=device) * float(
        sample_period_s
    )
    envelope = torch.full(
        (pulse_sample_count,), float(amplitude), dtype=torch.float64, device=device
    )
    if is_linear_fm:
        cycles = 0.5 * float(bandwidth_hz) * u * u / float(pulse_width_s)
        phase = math.tau * (cycles - torch.floor(cycles))
    else:
        phase = torch.zeros_like(u)
    return torch.complex(envelope * torch.cos(phase), envelope * torch.sin(phase))


def matched_filter(
    signal: torch.Tensor,
    replica: torch.Tensor,
    *,
    sample_period_s: float,
    oversample: int,
    window: str,
) -> torch.Tensor:
    """Correlate the trailing axis against ``conj(replica)``.

    ``signal`` is ``[..., M]``; the result is ``[..., M * oversample]``, indexed
    by lag from the range-gate start. Three conventions, all of which change the
    answer:

    * the transform length is ``M + M_p`` so the correlation's negative-lag
      tail - which is real and a full pulse wide - does not wrap onto the
      positive lags and invent an echo at the far end of the gate;
    * correlation, not convolution: the replica is CONJUGATED and NOT reversed,
      because convolving would put the peak at ``tau + T_p`` and would look
      entirely plausible;
    * the ``T_s`` factor makes the sum approximate the continuous integral, so
      with the unit-ENERGY pulse the spec pins, an isolated row's peak is
      exactly its complex coefficient with no sample-count factor.

    The window tapers the correlation SPECTRUM, which is the direct analogue of
    what the FMCW and OFDM backends do: in all three the window tapers the input
    of the final transform. It is applied in the unshifted spectral order via
    ``ifftshift``, so bin zero of the taper lands on DC.
    """

    if oversample < 1:
        raise ValueError(f"oversample must be at least 1, got {oversample}")
    samples = int(signal.shape[-1])
    length = samples + int(replica.shape[0])
    spectrum = torch.fft.fft(signal, n=length, dim=-1) * torch.conj(
        torch.fft.fft(replica.to(signal.dtype), n=length)
    )
    if window != DEFAULT_WINDOW:
        values = window_values(
            window, length, dtype=real_dtype_of(signal), device=signal.device
        )
        spectrum = spectrum * torch.fft.ifftshift(values)
    if oversample > 1:
        # Zero-insert the middle of the lag spectrum: exact band-limited
        # interpolation of the sampled correlation, not a smoothing. The scale
        # keeps the inverse transform an interpolation of the same signal
        # rather than a rescaled copy of it.
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
    return correlation[..., : samples * oversample] * float(sample_period_s)


__all__ = [
    "DEFAULT_WINDOW",
    "WINDOWS",
    "matched_filter",
    "pulse_replica",
    "real_dtype_of",
    "remove_mean",
    "taper",
    "window_coherent_gain",
    "window_values",
]
