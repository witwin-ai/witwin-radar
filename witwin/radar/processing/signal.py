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

import math
from dataclasses import dataclass
from enum import StrEnum

import torch

from ..synthesis.assembly import BEAT_PHASOR, CHANNEL_PHASOR, SPEED_OF_LIGHT_M_PER_S, assemble_frame_cube

#: The window family, named so a caller comparing against an analytic
#: unwindowed spectrum can turn it off by name rather than by passing ``None``
#: and hoping.
WINDOWS = ("rectangular", "hann", "hamming", "blackman", "hamming_symmetric")

#: The one SYMMETRIC member of the family, and the reason it exists. Every
#: legacy ``sigproc`` transform used ``torch.hamming_window(N, periodic=False)``,
#: which is ``0.54 - 0.46 cos(2 pi n / (N - 1))`` and is a DIFFERENT sequence
#: from the periodic window above. The migration adapters preserve the
#: behaviour of the public names they wrap, and that behaviour includes which
#: window was applied, so the window family carries the symmetric variant rather
#: than the adapters carrying a second window constructor outside this module.
#: New code should use ``"hamming"``: a symmetric window makes the transform of
#: a pure tone asymmetric about its bin, which is the property every exact-bin
#: assertion in this package leans on.
SYMMETRIC_WINDOWS = ("hamming_symmetric",)

#: What ``window=None`` means. Named rather than special cased, so that every
#: published :class:`RangeProfile` carries a window string and never a ``None``
#: that a reader has to interpret.
DEFAULT_WINDOW = "rectangular"


def window_values(name: str, length: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
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
    if name == "hamming_symmetric":
        span = float(length - 1) if length > 1 else 1.0
        return 0.54 - 0.46 * torch.cos(index * (2.0 * math.pi / span))
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
_COHERENT_GAIN = {"rectangular": 1.0, "hann": 0.5, "hamming": 0.54, "blackman": 0.42}


def window_coherent_gain(name: str, length: int) -> float:
    """``mean(w)``, on the host, with no device reduction.

    The closed form needs the cosine sums to vanish, which they do once the
    window is at least three samples long (the Blackman term is at ``2 n / N``).
    Below that the mean is computed from the two or one values directly, so the
    published gain is never a formula that quietly stops holding at a degenerate
    length.

    The SYMMETRIC window has its own closed form and not the periodic one:
    ``sum_{n=0}^{N-1} cos(2 pi n / (N - 1))`` is ``1``, not ``0``, because the
    last sample repeats the first phase, so the mean is ``0.54 - 0.46 / N``.
    """

    if name not in WINDOWS:
        raise ValueError(f"window must be one of {WINDOWS}, got {name!r}")
    if type(length) is not int or length < 1:
        raise ValueError(f"length must be a positive int, got {length!r}")
    if name == "hamming_symmetric":
        if length < 3:
            values = window_values(name, length, dtype=torch.float64, device=torch.device("cpu"))
            return float(values.sum()) / length
        return 0.54 - 0.46 / length
    if length >= 3 or name == "rectangular":
        return _COHERENT_GAIN[name]
    values = window_values(name, length, dtype=torch.float64, device=torch.device("cpu"))
    return float(values.sum()) / length


def taper(signal: torch.Tensor, name: str, *, dim: int) -> torch.Tensor:
    """Multiply ``signal`` by the window along ``dim``."""

    length = int(signal.shape[dim])
    values = window_values(name, length, dtype=real_dtype_of(signal), device=signal.device)
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
    is shared by three waveforms and must not learn what a ``PulsedSpec``
    is. The axes record reads the spec once and passes the numbers.
    """

    if type(pulse_sample_count) is not int or pulse_sample_count < 1:
        raise ValueError(
            f"the pulse spans {pulse_sample_count} samples on this grid: it is "
            "shorter than one sample period, so there is no replica to "
            "correlate against"
        )
    u = torch.arange(pulse_sample_count, dtype=torch.float64, device=device) * float(sample_period_s)
    envelope = torch.full((pulse_sample_count,), float(amplitude), dtype=torch.float64, device=device)
    if is_linear_fm:
        cycles = 0.5 * float(bandwidth_hz) * u * u / float(pulse_width_s)
        phase = math.tau * (cycles - torch.floor(cycles))
    else:
        phase = torch.zeros_like(u)
    return torch.complex(envelope * torch.cos(phase), envelope * torch.sin(phase))


def matched_filter(
    signal: torch.Tensor, replica: torch.Tensor, *, sample_period_s: float, oversample: int, window: str
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
    spectrum = torch.fft.fft(signal, n=length, dim=-1) * torch.conj(torch.fft.fft(replica.to(signal.dtype), n=length))
    if window != DEFAULT_WINDOW:
        values = window_values(window, length, dtype=real_dtype_of(signal), device=signal.device)
        spectrum = spectrum * torch.fft.ifftshift(values)
    if oversample > 1:
        # Zero-insert the middle of the lag spectrum: exact band-limited
        # interpolation of the sampled correlation, not a smoothing. The scale
        # keeps the inverse transform an interpolation of the same signal
        # rather than a rescaled copy of it.
        fine = torch.zeros((*spectrum.shape[:-1], length * oversample), dtype=spectrum.dtype, device=spectrum.device)
        half = length // 2
        fine[..., :half] = spectrum[..., :half]
        fine[..., length * oversample - (length - half) :] = spectrum[..., half:]
        spectrum = fine * oversample
        length = length * oversample
    correlation = torch.fft.ifft(spectrum, n=length, dim=-1)
    return correlation[..., : samples * oversample] * float(sample_period_s)


"""What each processing stage publishes, and the conventions it publishes it in.

Data only. Every record here is a frozen dataclass that a stage CONSTRUCTS
through a ``from_<producer>`` classmethod or that a stage entry returns
directly; none of them evaluates anything. The producer knows which product it
made, so a consumer never has to infer an axis, a unit, or a sign from the name
of the function that returned it.

Three conventions are carried as data rather than as documentation, because all
three change the answer and none of them is visible in a magnitude plot:

* :data:`PROCESSING_DOPPLER_CONVENTION` - a POSITIVE Doppler bin is a CLOSING
  (approaching) target, in every waveform. The FMCW beat cube is the conjugate
  of Channel's phasor, so its raw slow-time tone sits at ``+f_ref tau_rate``
  while the OFDM and pulsed tones sit at ``-f_ref tau_rate``. The two are
  reconciled EXACTLY ONCE, inside :func:`~witwin.radar.processing.range_doppler`,
  driven by :attr:`ProcessingAxes.doppler_sign`. Nothing else in this package
  may look at a phasor.
* :data:`PROCESSING_AMPLITUDE_CONVENTION` - every range profile and every
  Range-Doppler map is an AMPLITUDE estimate, normalised so that an isolated
  path row peaks at ``|C_rt|`` times the window's coherent gain. The three
  waveforms' native transforms have three different gains (``N`` for an
  unnormalised beat FFT, ``1`` for a CIR inverse transform, ``1`` for a
  unit-energy matched filter), and a facade that published all three would make
  a cross-waveform amplitude comparison a per-waveform bookkeeping exercise.
* :data:`PROCESSING_UNITS` - SI throughout, published as a mapping so a test can
  assert it against the values rather than against a docstring.

Nothing in this module reads a path batch, a topology, or a ``row_valid`` mask.
Processing consumes synthesis results; dead rows were already zeroed on the
WEIGHT before the kernel launched, so a cube arrives with their contribution
already exactly zero and there is nothing here for a row mask to do.
"""


class DetectorType(StrEnum):
    CFAR = "cfar"
    TOPK = "topk"


#: A positive Doppler bin is a closing target. Stated once, applied once.
PROCESSING_DOPPLER_CONVENTION = "positive_doppler_bin_is_closing"

#: The amplitude normalisation every stage in this package publishes.
PROCESSING_AMPLITUDE_CONVENTION = "peak = |C_rt| * window_coherent_gain; transforms are amplitude normalised"

#: The name of the fast-time axis per waveform, and of the slow-time axis. Read
#: by :class:`~witwin.radar.processing.signal.ProcessingAxes`; published here so a
#: reader finds the whole vocabulary in one module.
FAST_TIME_NAMES = {"fmcw": "sample", "ofdm": "subcarrier", "pulsed": "sample"}
SLOW_TIME_NAMES = {"fmcw": "chirp", "ofdm": "symbol", "pulsed": "pulse"}

#: Every published scalar and axis, with its SI unit. A quantity that is not in
#: this mapping is not part of the published metadata contract.
PROCESSING_UNITS = {
    "slow_time_period_s": "s",
    "range_bin_m": "m",
    "range_origin_m": "m",
    "max_unambiguous_range_m": "m",
    "velocity_bin_mps": "m/s",
    "max_unambiguous_speed_mps": "m/s",
    "wavelength_m": "m",
    "reference_frequency_hz": "Hz",
    "element_spacing_m": "m",
    "range_m": "m",
    "velocity_mps": "m/s",
}


def _require_complex(name: str, value: object) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}")
    if not value.is_complex():
        raise TypeError(
            f"{name} must be complex; a magnitude has already thrown away the "
            f"phase every later stage needs, got {value.dtype}"
        )
    return value


"""One metadata / axes / units record for all three waveforms.

:class:`ProcessingAxes` is the single physical-axis owner for every waveform.
It is constructed from a typed synthesis result, so the output domain, units,
phasor convention, array layout, and range/Doppler laws remain one coherent
record instead of being reconstructed by callers.

Two rules decide everything below.

**Built from the waveform SPECS, never from the flat ``RadarConfig``.** The flat
configuration is in engineering units - kSPS, microseconds, MHz per microsecond
- and ``to_spec`` is documented as its only conversion site. A second reader of
those units is a second conversion, and a conversion that is wrong once is wrong
everywhere. Everything here comes off an SI spec, off the ``SensorArraySpec``,
or off the ``SynthesisResult``'s own published conventions.

**The Doppler sign is fixed here and reconciled exactly once.** FMCW's beat cube
is the CONJUGATE of Channel's phasor convention, so its slow-time tone sits at
``+f_ref tau_rate`` while the OFDM and pulsed tones sit at ``-f_ref tau_rate``.
:attr:`ProcessingAxes.doppler_sign` is DERIVED from the cube's published
``phasor`` - not from the waveform's name, and not by editing the phasor
constants - and :func:`~witwin.radar.processing.range_doppler.range_doppler` is the
one place it is applied. The canonical convention every stage publishes is
:data:`~witwin.radar.processing.signal.PROCESSING_DOPPLER_CONVENTION`: a
positive Doppler bin is a CLOSING target.

**Scope decision, recorded.** Phase 8 does NOT make the legacy ``Radar``
multi-waveform. Construction is driven by a typed synthesis result, so waveform-specific axis laws remain in one owner.
"""


WAVEFORMS = ("fmcw", "ofdm", "pulsed")


def _doppler_sign_from_phasor(phasor: str) -> int:
    """``+1`` when the cube is conjugated relative to Channel, ``-1`` otherwise.

    This is the whole of the cross-waveform sign trap, written down once. It
    reads the phasor the synthesis owner published as DATA; it does not look at
    the waveform's name, because a waveform's name is not what decides whether
    its product was conjugated.
    """

    if phasor == BEAT_PHASOR:
        return 1
    if phasor == CHANNEL_PHASOR:
        return -1
    raise ValueError(
        f"unknown phasor convention {phasor!r}: this package knows "
        f"{BEAT_PHASOR!r} (the FMCW beat cube, conjugated) and "
        f"{CHANNEL_PHASOR!r} (everything else). A third convention needs its "
        "own Doppler sign decided deliberately, not defaulted"
    )


@dataclass(frozen=True, slots=True, eq=False)
class ProcessingAxes:
    """Everything a processing stage needs to know about a synthesis product.

    The two materialised axes are float64 and are the ONLY place a bin index
    becomes a physical quantity. Every stage reads them; no stage re-derives
    them. That is what makes the cross-waveform criterion - one physical target,
    three waveforms, one range in metres and one signed velocity - a comparison
    of three axis records rather than a comparison of three ad-hoc formulas.
    """

    waveform: str
    output_domain: str
    fast_time_name: str
    slow_time_name: str
    slow_time_period_s: float

    range_bin_count: int
    doppler_bin_count: int
    range_bin_m: float
    range_origin_m: float
    max_unambiguous_range_m: float

    velocity_bin_mps: float
    max_unambiguous_speed_mps: float

    wavelength_m: float
    reference_frequency_hz: float

    phasor: str
    doppler_sign: int

    num_tx: int
    num_rx: int
    element_spacing_m: float
    tx_loc_half_wavelength: tuple[tuple[float, float, float], ...]
    rx_loc_half_wavelength: tuple[tuple[float, float, float], ...]

    range_m: torch.Tensor
    velocity_mps: torch.Tensor

    #: Pulsed only. The matched-filter replica and the fast-time sample period
    #: it was built on travel with the axes because the range-profile entry
    #: takes ONE metadata record and must not learn what a waveform spec is.
    range_oversample: int = 1
    matched_filter_replica: torch.Tensor | None = None
    matched_filter_sample_period_s: float = 0.0

    def __post_init__(self) -> None:
        if self.waveform not in WAVEFORMS:
            raise ValueError(f"waveform must be one of {WAVEFORMS}, got {self.waveform!r}")
        if self.doppler_sign not in (1, -1):
            raise ValueError(f"doppler_sign must be +1 or -1, got {self.doppler_sign!r}")
        if int(self.range_m.shape[0]) != self.range_bin_count:
            raise ValueError(
                f"range_m holds {int(self.range_m.shape[0])} bins but range_bin_count is {self.range_bin_count}"
            )
        if int(self.velocity_mps.shape[0]) != self.doppler_bin_count:
            raise ValueError(
                f"velocity_mps holds {int(self.velocity_mps.shape[0])} bins but "
                f"doppler_bin_count is {self.doppler_bin_count}"
            )
        if self.range_m.dtype != torch.float64:
            raise TypeError("range_m must be float64: it is a coordinate, and a float32 metre at 300 m has a 2 cm ulp")
        if self.velocity_mps.dtype != torch.float64:
            raise TypeError("velocity_mps must be float64, for the same reason")
        if (self.matched_filter_replica is None) != (self.waveform != "pulsed"):
            raise ValueError(
                "a matched-filter replica belongs to the pulsed waveform and to "
                f"no other; this record is {self.waveform!r}"
            )
        if self.waveform != "pulsed" and self.range_oversample != 1:
            raise ValueError(
                f"range_oversample={self.range_oversample} is a matched-filter "
                "lag-grid refinement and means nothing for a transform whose "
                "bin grid is the waveform's own; only the pulsed backend "
                "accepts it"
            )

    # -- the published unit contract ---------------------------------------

    @property
    def units(self) -> dict[str, str]:
        """Every published scalar and axis, with its SI unit."""

        return dict(PROCESSING_UNITS)

    @property
    def sensor_pair_count(self) -> int:
        return self.num_tx * self.num_rx

    @property
    def device(self) -> torch.device:
        return self.range_m.device

    # -- construction -------------------------------------------------------

    @classmethod
    def from_synthesis(cls, result, spec, array, *, range_oversample: int = 1) -> "ProcessingAxes":
        """Read one synthesis result, its waveform spec, and the array.

        The result supplies the CUBE's shape and its published conventions, the
        spec supplies the SI waveform grid, and the array supplies the element
        geometry. All three are checked against each other before anything is
        derived: a cube synthesized from one spec and described by another is a
        configuration error whose only symptom would be a range axis that is
        quietly the wrong scale.
        """

        cube = result.cube
        if cube.dim() != 3:
            raise ValueError(
                "a synthesis result is a rank-3 (slow_time, sensor_pair, "
                f"fast_time) cube; got shape {tuple(cube.shape)}"
            )
        kind = str(result.kind)
        if kind not in WAVEFORMS:
            raise ValueError(f"no processing owner for waveform kind {kind!r}")
        reference = float(result.reference_frequency_hz)
        if float(spec.reference_frequency_hz) != reference:
            raise ValueError(
                f"the cube was synthesized at {reference} Hz but the spec "
                f"declares {float(spec.reference_frequency_hz)} Hz"
            )
        if float(array.reference_frequency_hz) != reference:
            raise ValueError(
                f"the cube was synthesized at {reference} Hz but the array's "
                f"element spacing is defined at "
                f"{float(array.reference_frequency_hz)} Hz; the two are the "
                "same physical quantity"
            )
        if int(cube.shape[1]) != array.sensor_pair_count:
            raise ValueError(
                f"the cube spans {int(cube.shape[1])} sensor pairs but the array "
                f"is {array.num_tx} x {array.num_rx} = "
                f"{array.sensor_pair_count} pairs"
            )
        if type(range_oversample) is not int or range_oversample < 1:
            raise ValueError(f"range_oversample must be a positive int, got {range_oversample!r}")

        device = cube.device
        slow_count = int(cube.shape[0])
        fast_count = int(cube.shape[2])
        wavelength = SPEED_OF_LIGHT_M_PER_S / reference

        replica = None
        replica_period = 0.0
        if kind == "fmcw":
            slow_period = float(spec.slot_period_s)
            max_speed = float(spec.max_unambiguous_speed_mps)
            range_count = fast_count
            range_bin = (
                SPEED_OF_LIGHT_M_PER_S
                * float(spec.sample_rate_hz)
                / (2.0 * float(spec.slope_hz_per_s) * int(spec.num_samples))
            )
            range_origin = 0.0
            max_range = range_bin * range_count
        elif kind == "ofdm":
            slow_period = float(spec.symbol_period_s)
            max_speed = float(spec.max_unambiguous_speed_mps)
            range_count = fast_count
            range_bin = float(spec.range_resolution_m)
            range_origin = 0.0
            max_range = SPEED_OF_LIGHT_M_PER_S * float(spec.max_unambiguous_delay_s) / 2.0
        else:
            slow_period = float(spec.pri_s)
            max_speed = float(spec.max_unambiguous_speed_m_s)
            range_count = fast_count * range_oversample
            range_bin = SPEED_OF_LIGHT_M_PER_S * float(spec.sample_period_s) / (2.0 * range_oversample)
            range_origin = SPEED_OF_LIGHT_M_PER_S * float(spec.range_gate_start_s) / 2.0
            max_range = float(spec.max_unambiguous_range_m)
            replica_period = float(spec.sample_period_s)
            replica = pulse_replica(
                pulse_sample_count=int(spec.pulse_sample_count),
                sample_period_s=replica_period,
                amplitude=float(spec.pulse_amplitude),
                bandwidth_hz=float(spec.bandwidth_hz),
                pulse_width_s=float(spec.pulse_width_s),
                is_linear_fm=bool(spec.is_linear_fm),
                device=device,
            )

        velocity_bin = wavelength / (2.0 * slow_count * slow_period)
        range_m = torch.arange(range_count, dtype=torch.float64, device=device) * range_bin + range_origin
        velocity_mps = torch.fft.fftshift(
            torch.fft.fftfreq(slow_count, d=slow_period, dtype=torch.float64, device=device)
        ) * (wavelength / 2.0)

        return cls(
            waveform=kind,
            output_domain=str(result.output_domain),
            fast_time_name=str(result.axes[-1]),
            slow_time_name=SLOW_TIME_NAMES[kind],
            slow_time_period_s=slow_period,
            range_bin_count=range_count,
            doppler_bin_count=slow_count,
            range_bin_m=range_bin,
            range_origin_m=range_origin,
            max_unambiguous_range_m=max_range,
            velocity_bin_mps=velocity_bin,
            max_unambiguous_speed_mps=max_speed,
            wavelength_m=wavelength,
            reference_frequency_hz=reference,
            phasor=str(result.phasor),
            doppler_sign=_doppler_sign_from_phasor(str(result.phasor)),
            num_tx=int(array.num_tx),
            num_rx=int(array.num_rx),
            element_spacing_m=float(array.element_spacing_m),
            tx_loc_half_wavelength=tuple(tuple(float(value) for value in row) for row in array.tx_loc),
            rx_loc_half_wavelength=tuple(tuple(float(value) for value in row) for row in array.rx_loc),
            range_m=range_m.contiguous(),
            velocity_mps=velocity_mps.contiguous(),
            range_oversample=range_oversample,
            matched_filter_replica=replica,
            matched_filter_sample_period_s=replica_period,
        )


"""The attach point: a synthesis result becomes a processing cube.

``SynthesisResult`` had no consumer outside ``witwin/radar/synthesis/`` and
``radar.py`` at all. This is where the processing chain attaches to it, and the
transpose it needs already exists: ``synthesis/assembly.py::assemble_frame_cube``
converts the SINK-MAJOR composed pair rank into the TX-MAJOR virtual-antenna
index every angle estimator steers. That function is CALLED here rather than
reimplemented, because two packers for one layout is how a TX/RX swap ships
silently on a square array.

Nothing in this module knows about ``row_valid``. A dead row is masked on the
WEIGHT before the waveform kernel launches, so its contribution to the cube is a
literal zero and no downstream stage has anything to mask. That is asserted by
``tests/processing/test_cube.py`` rather than assumed here: an assumption that
nothing executes is an assumption that stops being true.
"""


@dataclass(frozen=True, slots=True, eq=False)
class ProcessingCube:
    """``[TX, RX, C, S]`` complex, with the metadata record it was built with.

    ``C`` is the slow-time axis (chirps, symbols or pulses) and ``S`` the
    fast-time axis (samples or subcarriers); which is which is named on the
    axes record rather than inferred from the waveform.
    """

    data: torch.Tensor
    axes: object

    def __post_init__(self) -> None:
        if not isinstance(self.data, torch.Tensor):
            raise TypeError(f"data must be a torch.Tensor, got {type(self.data).__name__}")
        if not self.data.is_complex():
            raise TypeError(
                "a processing cube is complex IQ; a magnitude has already "
                f"thrown away the phase every later stage needs, got {self.data.dtype}"
            )
        if self.data.dim() != 4:
            raise ValueError(f"a processing cube is [tx, rx, slow_time, fast_time]; got shape {tuple(self.data.shape)}")
        expected = (self.axes.num_tx, self.axes.num_rx)
        if tuple(self.data.shape[:2]) != expected:
            raise ValueError(
                f"the cube's array axes are {tuple(self.data.shape[:2])} but the axes record declares {expected}"
            )
        if int(self.data.shape[2]) != int(self.axes.doppler_bin_count):
            raise ValueError(
                f"the cube has {int(self.data.shape[2])} slow-time samples but "
                f"the axes record's Doppler transform is "
                f"{int(self.axes.doppler_bin_count)} long"
            )

    @property
    def device(self) -> torch.device:
        return self.data.device

    @classmethod
    def from_synthesis(cls, result, axes) -> "ProcessingCube":
        """Pack one rank-3 synthesis cube into the rank-4 array layout.

        The axes record is passed in rather than rebuilt, so that the record
        used to pack the cube and the record every later stage reads are the
        same object. Rebuilding it here would let a caller process a cube
        against a different array than the one it was assembled for.
        """

        if tuple(result.axes) != (axes.slow_time_name, "sensor_pair", axes.fast_time_name):
            raise ValueError(
                f"this result publishes axes {tuple(result.axes)} but the "
                f"metadata record describes ({axes.slow_time_name}, "
                f"sensor_pair, {axes.fast_time_name}); the two describe "
                "different products"
            )
        if str(result.phasor) != axes.phasor:
            raise ValueError(
                f"this result is in the {result.phasor!r} convention but the "
                f"metadata record was built for {axes.phasor!r}; the Doppler "
                "sign is derived from that string and would be reconciled "
                "backwards"
            )
        return cls(data=assemble_frame_cube(result.cube, num_tx=axes.num_tx, num_rx=axes.num_rx), axes=axes)
