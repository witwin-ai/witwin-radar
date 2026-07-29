"""Pulse compression against a ``PulsedEchoSpec``: the spec-facing entry.

This is the thin spec-reading layer over
:mod:`witwin.radar.processing.signal`. The correlation itself lives there,
once, because the range-profile stage needs the same operation and two matched
filters in one package is exactly the duplication the Phase-8 cutover exists to
remove. What is here is what a spec supplies: the replica's parameters, the lag
axis, and the sample period the correlation is scaled by.

This module is the canonical owner of pulse compression, its lag axis, and
the spec-facing matched-filter entry.

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

from dataclasses import dataclass

import torch

from .signal import (
    DEFAULT_WINDOW,
    ProcessingCube,
    _require_complex,
    pulse_replica,
    remove_mean,
    taper,
    window_coherent_gain,
)
from .signal import matched_filter as _correlate


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
    axes: object
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
    axes: object
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
    signal: torch.Tensor, spec, *, oversample: int = 1, dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Correlate the fast-time axis against ``conj(p)``. Same rank as ``signal``.

        ``signal`` is the received train ``[..., num_samples]`` from the pulsed
        synthesis owner. The result is ``[..., num_samples * oversample]``, indexed
        by lag from the range-gate start.

        The transform length is `
    um_samples + M_p`` rather than `
    um_samples``, so
        that the negative lags - the correlation's left tail, which is real and
        extends a full pulse width - do not wrap around onto the positive ones. A
        circular correlation over `
    um_samples`` would fold that tail onto the far
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
            f"the fast-time axis holds {signal.shape[-1]} samples but the spec declares num_samples={spec.num_samples}"
        )
    working = signal if dtype is None else signal.to(dtype)
    replica = pulse_samples(spec, device=signal.device)
    return _correlate(
        working, replica, sample_period_s=float(spec.sample_period_s), oversample=oversample, window=DEFAULT_WINDOW
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


"""One range-profile entry, three waveform backends, one owner.

Range processing previously accumulated multiple FFT implementations with different windowing
choices - Hamming on fast time only, Hamming on both axes plus an unconditional
DC removal, and none at all - and no named range-profile stage at all. The OFDM
equivalent, the inverse transform of the channel frequency response, did not
exist anywhere. This module is the single owner.

The backend is selected by ``axes.waveform``, which is a STORED discriminator
read off the metadata record, not a probe and not an inference:

* **FMCW** - transform over fast time. The beat tone sits at ``S tau``, so bin
  ``k`` is ``c k f_s / (2 S N)`` metres.
* **OFDM** - INVERSE transform over subcarriers: the channel impulse response.
  ``H[n] = C exp(-j 2 pi n df tau)`` inverts to a peak at ``tau / T_s``.
* **Pulsed** - matched filter over fast time, correlating against the analytic
  replica the synthesis kernel evaluates. Same correlation as
  the matched-filter primitive without an unconditional ``complex128``
  upcast, which doubled every pulsed frame's transform cost to buy precision a
  float32 input never had.

Two arguments are explicit here that used to be implicit somewhere:

* ``window`` - one named family, applied to the input of the final transform in
  all three backends, and defaulting to ``rectangular`` so that a caller
  comparing against an analytic unwindowed transform gets the analytic answer.
* ``remove_dc`` - the fast-time mean subtraction that ``process_rd_tensor``
  applies UNCONDITIONALLY and undocumented. It is a clutter operation, so it is
  a flag, and it **defaults to off**: a component-export test that asks for the
  clutter cube and silently gets the clutter removed from it is comparing two
  different quantities.

Every backend is amplitude normalised, so an isolated on-bin row peaks at
``|C_rt| * window_coherent_gain`` in all three. The three native transforms have
three different gains and publishing all three would make a cross-waveform
amplitude comparison a per-waveform bookkeeping exercise.
"""


def _unpack(cube: ProcessingCube):
    if not isinstance(cube, ProcessingCube):
        raise TypeError(
            "range_profile takes a ProcessingCube; bare tensors have no "
            f"waveform, domain, or axis contract, got {type(cube).__name__}"
        )
    return cube.data, cube.axes


def range_profile(cube: ProcessingCube, *, window: str | None = None, remove_dc: bool = False) -> RangeProfile:
    """Convert one typed synthesis/processing cube to a range profile.

    `ProcessingCube` carries waveform, output-domain and physical-axis
    metadata with the complex data. Bare tensors are intentionally refused:
    their fast axis cannot distinguish an FMCW spectrum from a beat signal.
    Arbitrary leading batch dimensions remain supported.
    """

    data, record = _unpack(cube)
    if not data.is_complex():
        raise TypeError(f"a range profile is formed from complex IQ; got a real tensor of dtype {data.dtype}")
    if data.dim() < 2:
        raise ValueError(f"the input is [..., slow_time, fast_time]; got shape {tuple(data.shape)}")
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
        profile = _correlate(
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
                profile = torch.fft.fft(windowed, dim=-1, norm="forward")
        else:
            windowed = taper(data, name, dim=-1)
            # The CIR. The inverse transform already carries the 1 / N_sc that
            # makes the peak the coefficient itself.
            profile = torch.fft.ifft(windowed, dim=-1)

    return RangeProfile(
        data=profile, axes=record, window=name, window_coherent_gain=window_coherent_gain(name, taper_length)
    )


"""The slow-time transform, and the ONE place a Doppler sign is reconciled.

The transform itself is unremarkable: window slow time, transform, ``fftshift``.
What matters is the four lines in the middle.

FMCW's beat cube is the CONJUGATE of Channel's ``exp(-j k d)`` phasor, because
de-chirping multiplies the echo by the conjugate of the transmitted chirp. Its
slow-time tone therefore sits at ``+f_ref tau_rate`` while the OFDM and pulsed
tones sit at ``-f_ref tau_rate``. Nothing in this repository reconciled that:
The micro-Doppler path fixes a Channel-SIGNED axis and is handed FMCW cubes
by callers, so the same receding target read as approaching on one waveform and
receding on another, in a magnitude plot where the difference is invisible.

The canonical convention, stated once in
:data:`~witwin.radar.processing.signal.PROCESSING_DOPPLER_CONVENTION` and
applied once here, is that a POSITIVE Doppler bin is a CLOSING target. A
closing radial speed ``v`` gives ``tau_rate = -2 v / c``, so the canonical
frequency is ``-f_ref tau_rate = +2 v / lambda`` and

    ``v = lambda f / 2``

is the velocity axis :class:`ProcessingAxes` publishes, ascending and
``fftshift``ed, for every waveform. The reconciliation is a FREQUENCY REVERSAL
of the raw spectrum, ``X[k] -> X[(-k) mod D]``, applied only when
``axes.doppler_sign`` is ``+1``. It is an index gather with no arithmetic, so it
is exact, and it happens BEFORE the shift because negating a frequency index is
a wrap in the unshifted order and is not the same as reversing a shifted axis
(for even ``D`` the shifted axis is asymmetric about zero, and reversing it
would move every bin by one).
"""


def range_doppler_map(profile: RangeProfile, *, window: str | None = None) -> RangeDopplerMap:
    """``RangeProfile[..., C, R]`` -> ``RangeDopplerMap[..., D, R]``.

    Rank generic with an arbitrary leading batch, so ``[P, C, R]`` and
    ``[TX, RX, C, R]`` both work without a Python loop.

    Amplitude normalised like the range stage: the transform carries ``1 / D``
    so an isolated on-bin row peaks at its own coefficient magnitude rather than
    at the coherent-integration gain times it. The integration gain is
    recoverable exactly from :attr:`RangeDopplerMap.axes.doppler_bin_count`.
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
        # The cube is conjugated relative to Channel, so its tone sits at the
        # negated canonical frequency. Reverse the frequency index - a gather,
        # not arithmetic - and every waveform below this line is in one sign.
        bins = int(record.doppler_bin_count)
        reversed_index = torch.remainder(-torch.arange(bins, device=spectrum.device), bins)
        spectrum = spectrum.index_select(-2, reversed_index)
    spectrum = torch.fft.fftshift(spectrum, dim=-2)

    return RangeDopplerMap(
        data=spectrum,
        axes=record,
        window=name,
        window_coherent_gain=window_coherent_gain(name, int(record.doppler_bin_count)),
    )


"""Micro-Doppler analysis: slow-time spectra and spectrograms, in Torch.

This module is the canonical owner of slow-time spectra and spectrograms;
its values, axes, and conventions are defined together.

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
    bins = torch.fft.fftshift(torch.fft.fftfreq(slot_count, d=float(slot_period_s), device=device))
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
    samples: torch.Tensor, *, slot_period_s: float, window_slots: int, hop_slots: int, window: str = "hann"
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
    frequencies = doppler_frequencies_hz(window_slots, slot_period_s, device=samples.device)
    centre = (window_slots - 1) / 2.0
    times = (torch.arange(frames, dtype=torch.float64, device=samples.device) * hop_slots + centre) * float(
        slot_period_s
    )
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
