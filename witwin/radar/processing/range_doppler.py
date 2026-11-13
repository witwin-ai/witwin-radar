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
from dataclasses import dataclass, replace
from typing import cast

import torch

from ..display import compact_repr
from ..synthesis.assembly import PulsedSpec
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

    __repr__ = compact_repr

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

    __repr__ = compact_repr

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


def matched_filter(signal: torch.Tensor, spec: PulsedSpec, *, oversample: int = 1) -> torch.Tensor:
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
        signal,
        replica,
        sample_period_s=float(spec.sample_period_s),
        oversample=oversample,
        window=DEFAULT_WINDOW,
        periodic=bool(spec.periodic),
    )


def stepped_frequency_profile(signal: torch.Tensor, spec: PulsedSpec) -> tuple[torch.Tensor, torch.Tensor]:
    """The synthetic range profile of a stepped-frequency train, and its delay axis.

    ``signal`` is ``[..., L, M]``, the received train of ``L`` pulses whose
    carrier steps by ``spec.frequency_step_hz``. At a fast-time sample inside
    the echo, pulse ``l`` holds ``C exp(-j 2 pi l df tau)``, so the inverse
    transform across the pulse axis, normalised by ``1 / L``, peaks at
    ``|C|`` in bin ``L df tau`` (mod ``L``). Returns the profile ``[..., L, M]``
    (synthetic delay bin, fast-time sample) and the delay of each bin,
    ``k / (L |df|)`` in float64 seconds; the unambiguous delay is ``1 / |df|``.
    """

    step = float(spec.frequency_step_hz)
    if step == 0.0:
        raise ValueError("a train with frequency_step_hz = 0 has no synthetic range profile")
    count = int(spec.num_pulses)
    if signal.dim() < 2 or int(signal.shape[-2]) != count:
        raise ValueError(f"the pulse axis must hold num_pulses={count} pulses, got shape {tuple(signal.shape)}")
    profile = torch.fft.ifft(signal, dim=-2) if step > 0.0 else torch.fft.fft(signal, dim=-2) / count
    delays = torch.arange(count, dtype=torch.float64, device=signal.device) / (count * abs(step))
    return profile, delays


def lag_axis(spec: PulsedSpec, *, oversample: int = 1, device: torch.device | str = "cpu") -> torch.Tensor:
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


def _unpack(cube: ProcessingCube) -> tuple[torch.Tensor, ProcessingAxes]:
    if not isinstance(cube, ProcessingCube):
        raise TypeError(
            "range_profile takes a ProcessingCube; bare tensors have no "
            f"waveform, domain, or axis contract, got {type(cube).__name__}"
        )
    return cube.data, cube.axes


def ofdm_cfr_estimate(samples: torch.Tensor, symbols: torch.Tensor) -> torch.Tensor:
    """``[..., L, N]`` received post-prefix samples -> the per-subcarrier estimate ``Y / X``.

    The unitary ``N``-point transform of each symbol's samples, divided by the
    transmitted symbols ``symbols[L, N]``. For a still echo inside the prefix
    this is exactly the CFR ``C exp(-j 2 pi n df tau)``; a walking delay leaves
    the inter-carrier interference in it. Division by a constellation point
    scales that bin's noise by ``1 / |X|^2``.
    """

    if tuple(samples.shape[-2:]) != tuple(symbols.shape):
        raise ValueError(
            f"the samples' trailing (symbol, sample) axes {tuple(samples.shape[-2:])} must match the symbol table "
            f"{tuple(symbols.shape)}"
        )
    return torch.fft.fft(samples, dim=-1, norm="ortho") / symbols.to(device=samples.device, dtype=samples.dtype)


def ofdm_demodulate(cube: ProcessingCube) -> ProcessingCube:
    """A time-domain OFDM cube, demodulated to its CFR estimate, with the matching axes record."""

    if not isinstance(cube, ProcessingCube):
        raise TypeError(f"ofdm_demodulate takes a ProcessingCube, got {type(cube).__name__}")
    axes = cube.axes
    if axes.waveform != "ofdm" or axes.output_domain != "time" or axes.ofdm_symbols is None:
        raise ValueError("ofdm_demodulate takes a time-domain OFDM cube that carries its symbol table")
    data = ofdm_cfr_estimate(cube.data, axes.ofdm_symbols)
    return ProcessingCube(data=data, axes=replace(axes, output_domain="cfr", fast_time_name="subcarrier"))


def fmcw_range_fft(samples: torch.Tensor) -> torch.Tensor:
    """Normalized range DFT of explicit ADC samples, including refreshed scenes."""
    return torch.fft.fft(samples, dim=-1, norm="forward")


def _padded_fmcw_range_fft(samples: torch.Tensor, range_bins: int) -> torch.Tensor:
    """The ``range_bins``-point zero-padded beat transform, still normalised by ``1 / N``."""

    return torch.fft.fft(samples, n=range_bins, dim=-1) * (1.0 / int(samples.shape[-1]))


def range_profile(
    cube: ProcessingCube, *, window: str | None = None, remove_dc: bool = False, range_bins: int | None = None
) -> RangeProfile:
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

    ``range_bins`` zero pads the range transform to that many bins (FMCW beat
    and OFDM only): the window tapers the ``N`` recorded samples, the transform
    is ``range_bins`` long, and the published axes record is
    :meth:`ProcessingAxes.with_bin_counts` of the input's, so the bin shrinks
    and the span is unchanged. Zero padding interpolates the spectrum; it adds
    no resolution. An FMCW spectrum cube is already transformed and a pulsed
    grid is refined with ``range_oversample``, so both refuse it.

    Every backend is amplitude normalised, so an isolated on-bin row peaks at
    ``|C_rt| * window_coherent_gain`` in all three, with or without padding:
    the padded transform is rescaled by ``range_bins / N`` so its normalisation
    stays ``1 / N``.
    """

    data, record = _unpack(cube)
    name = DEFAULT_WINDOW if window is None else str(window)
    if range_bins is not None:
        if type(range_bins) is not int or range_bins < int(data.shape[-1]):
            raise ValueError(
                f"range_bins must be an int no smaller than the {int(data.shape[-1])} recorded fast-time samples, "
                f"got {range_bins!r}"
            )
        if record.waveform == "pulsed" or (record.waveform == "fmcw" and record.output_domain == "spectrum"):
            raise ValueError(
                f"range_bins zero pads a range transform this stage performs; a {record.waveform} "
                f"{record.output_domain} cube "
                + (
                    "is refined with range_oversample on its axes record"
                    if record.waveform == "pulsed"
                    else "is already transformed (use Fmcw(output='beat') to pad)"
                )
            )

    if record.waveform == "ofdm" and record.output_domain == "time":
        raise ValueError(
            "a time-domain OFDM cube is the received samples; demodulate it with ofdm_demodulate (or publish "
            "Ofdm(output='cfr')) before taking its range profile"
        )
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
        # ProcessingAxes holds a replica exactly when the waveform is pulsed.
        replica = cast(torch.Tensor, record.matched_filter_replica)
        profile = correlate(
            data,
            replica,
            sample_period_s=record.matched_filter_sample_period_s,
            oversample=record.range_oversample,
            window=name,
            periodic=record.matched_filter_periodic,
        )
        taper_length = int(data.shape[-1]) + (0 if record.matched_filter_periodic else int(replica.shape[0]))
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
                profile = (
                    fmcw_range_fft(windowed) if range_bins is None else _padded_fmcw_range_fft(windowed, range_bins)
                )
        else:
            windowed = taper(data, name, dim=-1)
            # The CIR. The inverse transform already carries the 1 / N_sc that
            # makes the peak the coefficient itself; a padded inverse carries
            # 1 / range_bins and is rescaled back to 1 / N_sc.
            if range_bins is None:
                profile = torch.fft.ifft(windowed, dim=-1)
            else:
                profile = torch.fft.ifft(windowed, n=range_bins, dim=-1) * (range_bins / taper_length)
        if range_bins is not None:
            record = record.with_bin_counts(range_bins=range_bins)

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


def range_doppler_map(
    profile: RangeProfile, *, window: str | None = None, doppler_bins: int | None = None
) -> RangeDopplerMap:
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

    ``doppler_bins`` zero pads the slow-time transform to that many bins. The
    window tapers the recorded slow-time samples, the published axes record is
    :meth:`ProcessingAxes.with_bin_counts`, and the transform is rescaled so
    its normalisation stays ``1 / C`` over the ``C`` recorded samples.
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

    slow_count = int(record.doppler_bin_count)
    if doppler_bins is None:
        spectrum = torch.fft.fft(taper(data, name, dim=-2), dim=-2, norm="forward")
    else:
        if type(doppler_bins) is not int or doppler_bins < slow_count:
            raise ValueError(
                f"doppler_bins must be an int no smaller than the {slow_count} slow-time samples, got {doppler_bins!r}"
            )
        spectrum = torch.fft.fft(taper(data, name, dim=-2), n=doppler_bins, dim=-2) * (1.0 / slow_count)
        record = record.with_bin_counts(doppler_bins=doppler_bins)
    if record.doppler_sign == 1:
        spectrum = _reverse_frequency(spectrum, -2)
    spectrum = torch.fft.fftshift(spectrum, dim=-2)

    return RangeDopplerMap(
        data=spectrum, axes=record, window=name, window_coherent_gain=window_coherent_gain(name, slow_count)
    )


#: The slow-time static-clutter filters :func:`remove_static_clutter` offers.
#: ``"mean"`` subtracts the slow-time mean per range bin; ``"mti2"`` and
#: ``"mti3"`` are the two- and three-pulse binomial cancellers.
CLUTTER_FILTERS = ("mean", "mti2", "mti3")


def remove_static_clutter(profile: RangeProfile, *, method: str = "mean") -> RangeProfile:
    """Suppress the zero-Doppler return along slow time, per range bin.

    ``profile`` is ``RangeProfile[..., C, R]``; the filter runs along ``C``.

    * ``"mean"`` - ``x[n] - mean_n x[n]``. Zero Doppler is removed exactly and
      every other bin of a ``C``-point transform is untouched, because the
      mean is the DFT's own zero bin. Length is kept.
    * ``"mti2"`` - ``y[n] = x[n+1] - x[n]``, the two-pulse canceller, with
      ``|H(f)| = 2 |sin(pi f T)|``.
    * ``"mti3"`` - ``y[n] = x[n+2] - 2 x[n+1] + x[n]``, ``|H(f)| = 4 sin^2(pi f T)``.

    The cancellers are FIR filters of the stated, un-normalised response: a
    target's amplitude is scaled by ``|H(f_D)|``, and a slow target is
    attenuated, which is the price of a canceller and is not hidden. They are
    computed only where the filter has full support, so the output has
    ``C - 1`` or ``C - 2`` slow-time samples and its axes record is
    :meth:`ProcessingAxes.with_bin_counts` with that count: the slow-time
    period is unchanged, so the velocity axis of a map formed from it is on
    the same closing-positive scale with a coarser bin. Differentiable: every
    filter is a fixed linear map.
    """

    if not isinstance(profile, RangeProfile):
        raise TypeError(f"remove_static_clutter takes a RangeProfile, got {type(profile).__name__}")
    if method not in CLUTTER_FILTERS:
        raise ValueError(f"method must be one of {CLUTTER_FILTERS}, got {method!r}")
    data = profile.data
    slow = int(data.shape[-2])
    if method == "mean":
        return RangeProfile(
            data=remove_mean(data, dim=-2),
            axes=profile.axes,
            window=profile.window,
            window_coherent_gain=profile.window_coherent_gain,
        )
    order = 1 if method == "mti2" else 2
    if slow <= order:
        raise ValueError(f"a {method} canceller needs more than {order} slow-time samples, got {slow}")
    if order == 1:
        filtered = data[..., 1:, :] - data[..., :-1, :]
    else:
        filtered = data[..., 2:, :] - 2.0 * data[..., 1:-1, :] + data[..., :-2, :]
    return RangeProfile(
        data=filtered,
        axes=profile.axes.with_bin_counts(doppler_bins=slow - order),
        window=profile.window,
        window_coherent_gain=profile.window_coherent_gain,
    )


#: The windows the micro-Doppler stage accepts: a subset of the package's
#: family, not a second one. A rectangular window leaks about -13 dB into the
#: first sidelobe, which for a rotor is the same order as the blade flash it is
#: there to resolve; the periodic Hann window leaks -31 dB and costs 1.5 bins of
#: main-lobe width. The rectangular option stays so a caller comparing against
#: an analytic unwindowed spectrum can turn the taper off.
MICRODOPPLER_WINDOWS = ("hann", "rectangular")


def doppler_frequencies_hz(
    slot_count: int, slot_period_s: float, *, device: torch.device | str | None = None
) -> torch.Tensor:
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
    return torch.fft.fftshift(torch.fft.fftfreq(slot_count, d=float(slot_period_s), dtype=torch.float64, device=device))


def slow_time_spectrum(samples: torch.Tensor, *, window: str = "hann") -> torch.Tensor:
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
    """Complex samples ``[..., time]``, absolute SI timestamps, and source phasor.

    One sequence represents one sensor pair and one range gate. TDM transmitter
    interleaving and gaps between frames are not uniform slow-time samples.
    Resample or segment those explicitly before asking for an STFT.

    ``wavelength_m`` is optional. With it the spectrogram also publishes a
    velocity axis, ``v = lambda f / 2``, the same conversion the
    Range-Doppler axis uses; without it only the frequency axis is published.
    """

    __repr__ = compact_repr

    samples: torch.Tensor
    times_s: tuple[float, ...]
    phasor: str
    wavelength_m: float | None = None

    def __post_init__(self) -> None:
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
        if self.wavelength_m is not None and not (math.isfinite(self.wavelength_m) and self.wavelength_m > 0.0):
            raise ValueError(f"wavelength_m must be a positive finite length, got {self.wavelength_m!r}")


@dataclass(frozen=True, slots=True, eq=False)
class MicroDopplerSpectrogram:
    """``[..., W, F]`` complex short-time slow-time spectra, with their axes.

    ``times_s`` is ``[W]`` float64, the absolute centre of each window.
    ``frequency_hz`` is ``[F]`` float64, ascending and ``fftshift``ed, in the
    canonical sign: positive for a CLOSING target, ``f = -f_ref d(tau)/dt``,
    for every phasor. ``velocity_mps`` is ``lambda f / 2`` on the same bins when
    the signal declared a wavelength, else ``None``.

    Amplitude normalised exactly like :class:`RangeDopplerMap`: each window's
    transform carries ``1 / window_slots``, so a unit tone on a bin peaks at
    ``window_coherent_gain``. Differentiable in the samples.
    """

    __repr__ = compact_repr

    data: torch.Tensor
    times_s: torch.Tensor
    frequency_hz: torch.Tensor
    velocity_mps: torch.Tensor | None
    window: str
    window_coherent_gain: float

    def __post_init__(self) -> None:
        _require_complex("data", self.data)
        if self.data.dim() < 2:
            raise ValueError(f"a spectrogram is [..., windows, frequency]; got shape {tuple(self.data.shape)}")
        if tuple(self.times_s.shape) != (int(self.data.shape[-2]),):
            raise ValueError(f"times_s must be [{int(self.data.shape[-2])}], got {tuple(self.times_s.shape)}")
        if tuple(self.frequency_hz.shape) != (int(self.data.shape[-1]),):
            raise ValueError(f"frequency_hz must be [{int(self.data.shape[-1])}], got {tuple(self.frequency_hz.shape)}")
        if self.velocity_mps is not None and tuple(self.velocity_mps.shape) != tuple(self.frequency_hz.shape):
            raise ValueError("velocity_mps must be on the frequency bins")


def microdoppler_spectrogram(
    signal: SlowTimeSignal, *, window_slots: int, hop_slots: int, window: str = "hann"
) -> MicroDopplerSpectrogram:
    """Short-time Fourier transform of one slow-time sequence.

    Frames of ``window_slots`` samples every ``hop_slots``, each tapered by the
    named window and transformed with ``1 / window_slots`` normalisation. A
    conjugated (beat-phasor) sequence has its frequency index reversed before
    the shift, exactly as :func:`range_doppler_map` does, so the published
    frequency is positive for a closing target in every phasor. No time gap is
    hidden: :class:`SlowTimeSignal` refuses non-uniform timestamps.
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
    if window not in MICRODOPPLER_WINDOWS:
        raise ValueError(f"window must be one of {MICRODOPPLER_WINDOWS}, got {window!r}")

    frames = (slots - window_slots) // hop_slots + 1
    # [..., frames, window_slots]; the transform runs over the trailing axis.
    framed = samples.unfold(-1, window_slots, hop_slots)
    spectrum = torch.fft.fft(taper(framed, window, dim=-1), dim=-1, norm="forward")
    if _doppler_sign_from_phasor(signal.phasor) == 1:
        spectrum = _reverse_frequency(spectrum, -1)
    spectrum = torch.fft.fftshift(spectrum, dim=-1)
    frequencies = doppler_frequencies_hz(window_slots, slot_period_s, device=samples.device)
    centre = (window_slots - 1) / 2.0
    times = (torch.arange(frames, dtype=torch.float64, device=samples.device) * hop_slots + centre) * float(
        slot_period_s
    ) + signal.times_s[0]
    velocity = None if signal.wavelength_m is None else frequencies * (signal.wavelength_m / 2.0)
    return MicroDopplerSpectrogram(
        data=spectrum,
        times_s=times,
        frequency_hz=frequencies,
        velocity_mps=velocity,
        window=window,
        window_coherent_gain=window_coherent_gain(window, window_slots),
    )
