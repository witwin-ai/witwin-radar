"""Range profile, Range-Doppler map, pulse compression and micro-Doppler.

Everything here reads a :class:`~witwin.radar.processing.signal.ProcessingAxes`
record and re-derives no delay, no geometry and no phase: it takes a cube and a
metadata record, transforms, and returns. There is no carrier, no path row and
no knowledge of a scene. If a future edit needs one of those, the edit is in the
wrong file.

The pulsed replica built by :func:`~witwin.radar.processing.signal.pulse_samples`
and the pulse the kernel evaluates are the same analytic function, and that is
the point rather than a duplication to be factored out: one is a continuous
function evaluated at a fractional delay inside a CUDA kernel, the other is a
discrete sequence on the ADC grid. The matched-filter peak being exactly the
path coefficient is the statement that they agree, and it is asserted rather
than assumed.
"""

import math
from dataclasses import dataclass

import torch

from .signal import (
    DEFAULT_WINDOW,
    ProcessingAxes,
    ProcessingCube,
    _doppler_sign_from_phasor,
    _require_complex,
    correlate,
    pulse_samples,
    remove_mean,
    taper,
    window_coherent_gain,
)


@dataclass(frozen=True, slots=True, eq=False)
class RangeProfile:
    """``[..., C, R]`` complex, one range profile per slow-time sample.

    ``R`` is the range-bin count the axes record already published, and the
    profile is asserted against it rather than defining it: a stage that could
    publish its own bin count could publish one that disagrees with the metres
    the axis is in.

    ``window_coherent_gain`` is ``mean(w)``. For FMCW and OFDM the identity
    ``peak == |C_rt| * window_coherent_gain`` is exact for an isolated on-bin
    row. For the pulsed backend the window tapers the correlation SPECTRUM, so
    the gain there is the ``|P(f)|^2``-weighted mean of the same window and the
    identity holds only for the unwindowed case. Published rather than folded
    in, so a caller that wants the raw transform can recover it.
    """

    data: torch.Tensor
    axes: ProcessingAxes
    window: str
    window_coherent_gain: float

    def __post_init__(self) -> None:
        _require_complex("data", self.data)
        if self.data.dim() < 2:
            raise ValueError(f"a range profile is [..., slow_time, range]; got shape {tuple(self.data.shape)}")
        if int(self.data.shape[-1]) != int(self.axes.range_bin_count):
            raise ValueError(
                f"the profile has {int(self.data.shape[-1])} range bins but its "
                f"axes record publishes {int(self.axes.range_bin_count)}"
            )

    @property
    def range_axis(self) -> torch.Tensor:
        """``[R]`` float64 metres, from the axes record and nowhere else."""

        return self.axes.range_m


@dataclass(frozen=True, slots=True, eq=False)
class RangeDopplerMap:
    """``[..., D, R]`` complex, ``fftshift``ed and in the closing-positive sign.

    The Doppler axis is :attr:`ProcessingAxes.velocity_mps`: signed metres per
    second, ascending, with the waveform's phasor reconciliation already
    applied. There is no second place in this package where a sign is decided.
    """

    data: torch.Tensor
    axes: ProcessingAxes
    window: str
    window_coherent_gain: float

    def __post_init__(self) -> None:
        _require_complex("data", self.data)
        if self.data.dim() < 2:
            raise ValueError(f"a Range-Doppler map is [..., doppler, range]; got shape {tuple(self.data.shape)}")
        if int(self.data.shape[-1]) != int(self.axes.range_bin_count):
            raise ValueError(
                f"the map has {int(self.data.shape[-1])} range bins but its axes "
                f"record publishes {int(self.axes.range_bin_count)}"
            )
        if int(self.data.shape[-2]) != int(self.axes.doppler_bin_count):
            raise ValueError(
                f"the map has {int(self.data.shape[-2])} Doppler bins but its "
                f"axes record publishes {int(self.axes.doppler_bin_count)}"
            )

    @property
    def range_axis(self) -> torch.Tensor:
        return self.axes.range_m

    @property
    def doppler_axis(self) -> torch.Tensor:
        """``[D]`` float64 metres per second, positive for a closing target."""

        return self.axes.velocity_mps


def matched_filter(signal: torch.Tensor, spec, *, oversample: int = 1) -> torch.Tensor:
    """Correlate the fast-time axis against ``conj(p)``. Same rank as ``signal``.

    ``signal`` is the received train ``[..., num_samples]`` from the pulsed
    synthesis owner. The result is ``[..., num_samples * oversample]``, indexed
    by lag from the range-gate start, in the input's own precision. The
    conventions - transform length ``num_samples + M_p``, correlation rather
    than convolution, the ``T_s`` factor - are
    :func:`~witwin.radar.processing.signal.correlate`'s.

    ``oversample`` inserts zeros in the middle of the lag spectrum, which is
    exact band-limited interpolation of the sampled correlation rather than a
    smoothing. It exists because the range cell can be a couple of samples
    wide: a three-point parabolic fit on the raw grid then measures its own
    truncation error instead of the peak. It changes only the lag GRID, never
    the values on the original grid.
    """

    if signal.shape[-1] != spec.num_samples:
        raise ValueError(
            f"the fast-time axis holds {signal.shape[-1]} samples but the spec declares num_samples={spec.num_samples}"
        )
    replica = pulse_samples(spec, device=signal.device)
    return correlate(
        signal, replica, sample_period_s=float(spec.sample_period_s), oversample=oversample, window=DEFAULT_WINDOW
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
    steps = torch.arange(spec.num_samples * oversample, dtype=torch.float64, device=device)
    return spec.range_gate_start_s + steps * (spec.sample_period_s / oversample)


def _unpack(cube: ProcessingCube):
    if not isinstance(cube, ProcessingCube):
        raise TypeError(
            "range_profile takes a ProcessingCube; bare tensors have no "
            f"waveform, domain, or axis contract, got {type(cube).__name__}"
        )
    return cube.data, cube.axes


def fmcw_range_fft(samples: torch.Tensor) -> torch.Tensor:
    """Normalized range DFT of explicit ADC samples, including refreshed scenes."""
    return torch.fft.fft(samples, dim=-1, norm="forward")


def range_profile(cube: ProcessingCube, *, window: str | None = None, remove_dc: bool = False) -> RangeProfile:
    """Convert one typed synthesis/processing cube to a range profile.

    `ProcessingCube` carries waveform, output-domain and physical-axis
    metadata with the complex data. Bare tensors are intentionally refused:
    their fast axis cannot distinguish an FMCW spectrum from a beat signal.
    Arbitrary leading batch dimensions remain supported.

    The backend is selected by ``axes.waveform``, a STORED discriminator read
    off the metadata record, not a probe and not an inference:

    * **FMCW** - transform over fast time. The beat tone sits at ``S tau``, so
      bin ``k`` is ``c k f_s / (2 S N)`` metres.
    * **OFDM** - INVERSE transform over subcarriers: the channel impulse
      response. ``H[n] = C exp(-j 2 pi n df tau)`` inverts to a peak at
      ``tau / T_s``.
    * **Pulsed** - matched filter over fast time, correlating against the
      analytic replica the synthesis kernel evaluates.

    ``window`` is one named family, applied to the input of the final
    transform in all three backends, and defaulting to ``rectangular`` so that
    a caller comparing against an analytic unwindowed transform gets the
    analytic answer. ``remove_dc`` is the fast-time mean subtraction; it is a
    clutter operation, so it is a flag and defaults to off.

    Every backend is amplitude normalised, so an isolated on-bin row peaks at
    ``|C_rt| * window_coherent_gain`` in all three.
    """

    data, record = _unpack(cube)
    name = DEFAULT_WINDOW if window is None else str(window)

    if record.waveform == "fmcw":
        if record.output_domain == "spectrum":
            if name != DEFAULT_WINDOW:
                raise ValueError(
                    "an FMCW range spectrum is already transformed; only the "
                    f"{DEFAULT_WINDOW!r} window is valid, got {name!r}"
                )
            if remove_dc:
                data = torch.cat((torch.zeros_like(data[..., :1]), data[..., 1:]), dim=-1)
        elif record.output_domain == "beat":
            if remove_dc:
                data = remove_mean(data, dim=-1)
        else:
            raise ValueError(f"unsupported FMCW output_domain {record.output_domain!r}; expected spectrum or beat")
    elif remove_dc:
        data = remove_mean(data, dim=-1)

    if record.waveform == "pulsed":
        expected = record.range_bin_count // record.range_oversample
        if int(data.shape[-1]) != expected:
            raise ValueError(
                f"the fast-time axis holds {int(data.shape[-1])} samples but the "
                f"metadata record was built for {expected} at oversample "
                f"{record.range_oversample}"
            )
        profile = correlate(
            data,
            record.matched_filter_replica,
            sample_period_s=record.matched_filter_sample_period_s,
            oversample=record.range_oversample,
            window=name,
        )
        taper_length = int(data.shape[-1]) + int(record.matched_filter_replica.shape[0])
    else:
        if int(data.shape[-1]) != record.range_bin_count:
            raise ValueError(
                f"the fast-time axis holds {int(data.shape[-1])} "
                f"{record.fast_time_name}s but the metadata record publishes "
                f"{record.range_bin_count} range bins"
            )
        taper_length = int(data.shape[-1])
        if record.waveform == "fmcw":
            if record.output_domain == "spectrum":
                profile = data
            else:
                windowed = taper(data, name, dim=-1)
                # Amplitude normalised: the unnormalised beat FFT peaks at N |C|.
                profile = fmcw_range_fft(windowed)
        else:
            windowed = taper(data, name, dim=-1)
            # The CIR. The inverse transform already carries the 1 / N_sc that
            # makes the peak the coefficient itself.
            profile = torch.fft.ifft(windowed, dim=-1)

    return RangeProfile(
        data=profile, axes=record, window=name, window_coherent_gain=window_coherent_gain(name, taper_length)
    )


def _reverse_frequency(spectrum: torch.Tensor, dim: int) -> torch.Tensor:
    """``X[k] -> X[(-k) mod N]`` along ``dim`` of an UNSHIFTED spectrum.

    The reconciliation of a conjugated cube to the closing-positive convention.
    An index gather with no arithmetic, so it is exact. It must run in the
    unshifted order: negating a frequency index is a wrap there, and is not the
    same as reversing a shifted axis (for even ``N`` the shifted axis is
    asymmetric about zero, and reversing it would move every bin by one).
    """

    bins = int(spectrum.shape[dim])
    reversed_index = torch.remainder(-torch.arange(bins, device=spectrum.device), bins)
    return spectrum.index_select(dim, reversed_index)


def range_doppler_map(profile: RangeProfile, *, window: str | None = None) -> RangeDopplerMap:
    """``RangeProfile[..., C, R]`` -> ``RangeDopplerMap[..., D, R]``.

    Rank generic with an arbitrary leading batch, so ``[P, C, R]`` and
    ``[TX, RX, C, R]`` both work without a Python loop.

    Amplitude normalised like the range stage: the transform carries ``1 / D``
    so an isolated on-bin row peaks at its own coefficient magnitude rather than
    at the coherent-integration gain times it. The integration gain is
    recoverable exactly from :attr:`RangeDopplerMap.axes.doppler_bin_count`.

    This is the ONE place a Doppler sign is reconciled. FMCW's beat cube is the
    CONJUGATE of Channel's ``exp(-j k d)`` phasor, so its slow-time tone sits at
    ``+f_ref tau_rate`` while the OFDM and pulsed tones sit at
    ``-f_ref tau_rate``. The canonical convention,
    :data:`~witwin.radar.processing.signal.PROCESSING_DOPPLER_CONVENTION`, is
    that a POSITIVE Doppler bin is a CLOSING target: a closing radial speed
    ``v`` gives ``tau_rate = -2 v / c``, so the canonical frequency is
    ``-f_ref tau_rate = +2 v / lambda`` and ``v = lambda f / 2`` is the velocity
    axis :class:`ProcessingAxes` publishes, ascending and ``fftshift``ed, for
    every waveform. The reconciliation is :func:`_reverse_frequency`, applied
    only when ``axes.doppler_sign`` is ``+1`` and BEFORE the shift.
    """

    if not isinstance(profile, RangeProfile):
        raise TypeError(
            "range_doppler consumes a RangeProfile, so that the range axis it "
            "publishes is the one the range stage already decided; got "
            f"{type(profile).__name__}"
        )
    record = profile.axes
    name = DEFAULT_WINDOW if window is None else str(window)
    data = profile.data
    if int(data.shape[-2]) != int(record.doppler_bin_count):
        raise ValueError(
            f"the profile has {int(data.shape[-2])} slow-time samples but the "
            f"metadata record's transform is {int(record.doppler_bin_count)} long"
        )

    spectrum = torch.fft.fft(taper(data, name, dim=-2), dim=-2, norm="forward")
    if record.doppler_sign == 1:
        spectrum = _reverse_frequency(spectrum, -2)
    spectrum = torch.fft.fftshift(spectrum, dim=-2)

    return RangeDopplerMap(
        data=spectrum,
        axes=record,
        window=name,
        window_coherent_gain=window_coherent_gain(name, int(record.doppler_bin_count)),
    )


#: The windows the micro-Doppler stage accepts: a subset of the package's
#: family, not a second one. A rectangular window leaks about -13 dB into the
#: first sidelobe, which for a rotor is the same order as the blade flash it is
#: there to resolve; the periodic Hann window leaks -31 dB and costs 1.5 bins of
#: main-lobe width. The rectangular option stays so a caller comparing against
#: an analytic unwindowed spectrum can turn the taper off.
MICRODOPPLER_WINDOWS = ("hann", "rectangular")


def doppler_frequencies_hz(slot_count: int, slot_period_s: float, *, device=None):
    """The signed, ``fftshift``ed Doppler axis of a ``slot_count`` transform.

    Runs from ``-1/(2 T_slot)`` up to just below ``+1/(2 T_slot)``. In the
    Channel convention a receding target gives a NEGATIVE frequency, matching
    ``f_D = -f_ref tau_rate``; an unshifted, unsigned axis is the usual way a
    closing target ends up plotted as a receding one.

    ``slot_period_s`` is the slow-time sample period - the chirp period times
    the transmitter count for a TDM frame, the symbol period for OFDM, the
    pulse repetition interval for a pulsed train. It is the caller's, because
    only the caller knows which of those its samples came from.
    """

    if type(slot_count) is not int or slot_count < 1:
        raise ValueError(f"slot_count must be a positive int, got {slot_count!r}")
    if not float(slot_period_s) > 0.0:
        raise ValueError(f"slot_period_s must be positive, got {slot_period_s}")
    bins = torch.fft.fftshift(torch.fft.fftfreq(slot_count, d=float(slot_period_s), device=device))
    return bins


def slow_time_spectrum(samples: torch.Tensor, *, window: str = "hann"):
    """The ``fftshift``ed slow-time spectrum of ``samples[..., slots]``.

    The transform is over the LAST axis: a slow-time cube is ``[..., slots]``
    everywhere in this package, so anything else would need the caller to
    permute and would silently transform range instead of Doppler on a cube
    that happened to be square.

    Returns the complex spectrum. Magnitude, power and decibels are the
    caller's, because a caller that wants decibels also wants to choose the
    floor and this module does not get to pick one for it.
    """

    if not isinstance(samples, torch.Tensor):
        raise TypeError(f"samples must be a torch.Tensor, got {type(samples).__name__}")
    if samples.ndim < 1 or samples.shape[-1] < 1:
        raise ValueError("samples must have a non-empty trailing slow-time axis")
    if window not in MICRODOPPLER_WINDOWS:
        raise ValueError(f"window must be one of {MICRODOPPLER_WINDOWS}, got {window!r}")
    return torch.fft.fftshift(torch.fft.fft(taper(samples, window, dim=-1), dim=-1), dim=-1)


@dataclass(frozen=True, slots=True, eq=False)
class SlowTimeSignal:
    """Complex samples [..., time], absolute SI timestamps, and source phasor.

    One sequence represents one sensor pair and one range gate. TDM transmitter
    interleaving and gaps between frames are not uniform slow-time samples.
    Resample or segment those explicitly before asking for an STFT.
    """

    samples: torch.Tensor
    times_s: tuple[float, ...]
    phasor: str

    def __post_init__(self):
        _require_complex("samples", self.samples)
        if self.samples.ndim < 1 or len(self.times_s) != self.samples.shape[-1] or len(self.times_s) < 2:
            raise ValueError("timestamps must match a slow-time axis of at least two samples")
        if not all(math.isfinite(t) for t in self.times_s):
            raise ValueError("slow-time timestamps must be finite")
        period = self.times_s[1] - self.times_s[0]
        if period <= 0 or any(
            not math.isclose(b - a, period, rel_tol=1e-6, abs_tol=1e-12)
            for a, b in zip(self.times_s, self.times_s[1:], strict=False)
        ):
            raise ValueError("slow-time timestamps must be strictly increasing and uniform; split frame gaps")
        _doppler_sign_from_phasor(self.phasor)


def microdoppler_spectrogram(signal: SlowTimeSignal, *, window_slots: int, hop_slots: int, window: str = "hann"):
    """STFT with canonical physical Doppler f_D=-f_ref*d(tau)/dt.

    Return absolute window-centre times, signed frequencies, and complex
    spectra [..., windows, frequency]. Beat-domain bins are reversed using the
    declared phasor, exactly as for range_doppler_map. No time gaps are hidden.
    """
    if not isinstance(signal, SlowTimeSignal):
        raise TypeError("microdoppler_spectrogram requires a SlowTimeSignal with timestamps and phasor")
    samples = signal.samples
    slot_period_s = signal.times_s[1] - signal.times_s[0]
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
    if _doppler_sign_from_phasor(signal.phasor) == 1:
        spectrum = torch.fft.fftshift(_reverse_frequency(torch.fft.ifftshift(spectrum, dim=-1), -1), dim=-1)
    frequencies = doppler_frequencies_hz(window_slots, slot_period_s, device=samples.device)
    centre = (window_slots - 1) / 2.0
    times = (torch.arange(frames, dtype=torch.float64, device=samples.device) * hop_slots + centre) * float(
        slot_period_s
    )
    return times + signal.times_s[0], frequencies, spectrum
