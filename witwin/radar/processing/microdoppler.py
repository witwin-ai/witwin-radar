"""Micro-Doppler analysis: slow-time spectra and spectrograms, in Torch.

This module is on the Torch side of the owner directive of 2026-07-25, and the
placement is the whole point rather than an implementation detail. The
SIMULATION of micro-Doppler - a rotor blade's ``omega x r``, a limb's
deformation velocity, an aspect-dependent scatter response - is native: the
velocity reaches ``delay_rate`` through the propagation JVP, the response is
evaluated in ``scatter_response_aspect``, and the slow-time carrier is applied
in the waveform kernel. What this module does is READ the result. A spectrogram
is post-processing, so it stays here and never becomes a kernel.

The rule this module obeys, and that its test pins statically: **it re-derives
no geometry, no delay and no path row.** It takes a slow-time sequence and a
slot period and returns frequencies. If a future edit needs a position, a
velocity or a scatter response, the edit is in the wrong file.

Conventions, both of which change the answer:

* The Doppler axis is ``fftshift``ed and signed, running from
  ``-1/(2 T_slot)`` up to just below ``+1/(2 T_slot)``. A receding target
  gives a NEGATIVE frequency, matching the sign of ``f_D = -f_ref tau_rate``
  that the propagation contract publishes. An unshifted, unsigned axis is the
  usual way a closing target ends up plotted as a receding one.
* The transform is over the LAST axis. A slow-time cube is
  ``[..., slots]`` everywhere in this package, so anything else would need the
  caller to permute and would silently transform range instead of Doppler on a
  cube that happened to be square.
"""

from __future__ import annotations

import torch


#: The window this module applies by default.
#:
#: A rectangular window leaks about -13 dB into the first sidelobe, which for a
#: rotor is the same order as the blade flash it is there to resolve; the
#: periodic Hann window leaks -31 dB and costs 1.5 bins of main-lobe width.
#: Named rather than hard coded because a caller comparing against an analytic
#: unwindowed spectrum has to be able to turn it off.
WINDOWS = ("hann", "rectangular")


def _window(name: str, length: int, *, dtype, device) -> torch.Tensor:
    if name not in WINDOWS:
        raise ValueError(f"window must be one of {WINDOWS}, got {name!r}")
    if name == "rectangular":
        return torch.ones(length, dtype=dtype, device=device)
    # Periodic, not symmetric: a symmetric window makes the DFT of a pure tone
    # asymmetric about its bin, which is exactly the property a micro-Doppler
    # band-edge measurement leans on.
    index = torch.arange(length, dtype=dtype, device=device)
    return 0.5 - 0.5 * torch.cos(2.0 * torch.pi * index / length)


def doppler_frequencies_hz(slot_count: int, slot_period_s: float, *, device=None):
    """The signed, ``fftshift``ed Doppler axis of a ``slot_count`` transform.

    ``slot_period_s`` is the slow-time sample period - the chirp period times
    the transmitter count for a TDM frame, the symbol period for OFDM, the
    pulse repetition interval for a pulsed train. It is the caller's, because
    only the caller knows which of those its samples came from.
    """

    if type(slot_count) is not int or slot_count < 1:
        raise ValueError(f"slot_count must be a positive int, got {slot_count!r}")
    if not float(slot_period_s) > 0.0:
        raise ValueError(f"slot_period_s must be positive, got {slot_period_s}")
    bins = torch.fft.fftshift(
        torch.fft.fftfreq(slot_count, d=float(slot_period_s), device=device)
    )
    return bins


def slow_time_spectrum(samples: torch.Tensor, *, window: str = "hann"):
    """The ``fftshift``ed slow-time spectrum of ``samples[..., slots]``.

    Returns the complex spectrum. Magnitude, power and decibels are the
    caller's, because a caller that wants decibels also wants to choose the
    floor and this module does not get to pick one for it.
    """

    if not isinstance(samples, torch.Tensor):
        raise TypeError(f"samples must be a torch.Tensor, got {type(samples).__name__}")
    if samples.ndim < 1 or samples.shape[-1] < 1:
        raise ValueError("samples must have a non-empty trailing slow-time axis")
    slots = int(samples.shape[-1])
    taper = _window(
        window,
        slots,
        dtype=torch.float64 if samples.dtype == torch.complex128 else torch.float32,
        device=samples.device,
    )
    return torch.fft.fftshift(torch.fft.fft(samples * taper, dim=-1), dim=-1)


def microdoppler_spectrogram(
    samples: torch.Tensor,
    *,
    slot_period_s: float,
    window_slots: int,
    hop_slots: int,
    window: str = "hann",
):
    """A short-time slow-time transform: the micro-Doppler spectrogram.

    Returns ``(times_s, frequencies_hz, spectrum)`` where ``spectrum`` has shape
    ``[..., frames, window_slots]``, complex, with the Doppler axis
    ``fftshift``ed exactly as :func:`slow_time_spectrum` leaves it.

    ``times_s`` is the CENTRE of each window, not its start. A cadence read off
    window starts is late by half a window, which for a rotor whose flash lasts
    a fraction of a window is the difference between a symmetric spectrogram and
    one that looks like it has a lag.

    The framing is explicit rather than a stride trick: ``unfold`` produces a
    view whose windows overlap in storage, and multiplying it by a window in
    place would corrupt the neighbours. This takes the copy on purpose.
    """

    if not isinstance(samples, torch.Tensor):
        raise TypeError(f"samples must be a torch.Tensor, got {type(samples).__name__}")
    if samples.ndim < 1:
        raise ValueError("samples must have a trailing slow-time axis")
    slots = int(samples.shape[-1])
    for name, value in (("window_slots", window_slots), ("hop_slots", hop_slots)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive int, got {value!r}")
    if window_slots > slots:
        raise ValueError(
            f"window_slots={window_slots} exceeds the {slots} available slow-time "
            "samples; a spectrogram cannot be wider than its signal"
        )
    if not float(slot_period_s) > 0.0:
        raise ValueError(f"slot_period_s must be positive, got {slot_period_s}")

    frames = (slots - window_slots) // hop_slots + 1
    framed = samples.unfold(-1, window_slots, hop_slots).contiguous()
    # unfold gives [..., frames, window_slots] already; the transform is over
    # the window axis, which slow_time_spectrum takes as the trailing one.
    spectrum = slow_time_spectrum(framed, window=window)
    frequencies = doppler_frequencies_hz(
        window_slots, slot_period_s, device=samples.device
    )
    centre = (window_slots - 1) / 2.0
    times = (
        torch.arange(frames, dtype=torch.float64, device=samples.device) * hop_slots
        + centre
    ) * float(slot_period_s)
    return times, frequencies, spectrum


def dominant_frequencies_hz(spectrum: torch.Tensor, frequencies_hz: torch.Tensor):
    """The frequency of the largest magnitude bin along the trailing axis.

    The single most useful reduction over a spectrogram and the one every
    cadence estimate starts from. It is a plain ``argmax`` with no interpolation:
    a parabolic peak refinement is a modelling choice, and a caller that wants
    sub-bin accuracy should say so rather than get it silently.
    """

    if int(spectrum.shape[-1]) != int(frequencies_hz.shape[-1]):
        raise ValueError(
            f"the spectrum's trailing axis is {int(spectrum.shape[-1])} bins but "
            f"frequencies_hz holds {int(frequencies_hz.shape[-1])}"
        )
    return frequencies_hz[spectrum.abs().argmax(dim=-1)]


__all__ = [
    "WINDOWS",
    "dominant_frequencies_hz",
    "doppler_frequencies_hz",
    "microdoppler_spectrogram",
    "slow_time_spectrum",
]
