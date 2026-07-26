"""One metadata / axes / units record for all three waveforms.

``sigproc`` has three axis conventions that disagree: the synthesis rank-3
tuples, the rank-4 frame cube, and ``FrameConfig``'s own re-derived names. The
Phase-6 ``RadarAxes`` closed part of the gap and is FMCW only - both in its
field names and in fact, because ``RadarSystemConfig.axes()`` raises for
anything else. :class:`ProcessingAxes` is the wider record that supersedes it.

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
constants - and :func:`~witwin.radar.processing.doppler.range_doppler` is the
one place it is applied. The canonical convention every stage publishes is
:data:`~witwin.radar.processing.contracts.PROCESSING_DOPPLER_CONVENTION`: a
positive Doppler bin is a CLOSING target.

**Scope decision, recorded.** Phase 8 does NOT make the legacy ``Radar``
multi-waveform. ``RadarSystemConfig.axes()`` raises for non-FMCW,
``Radar.__init__`` calls ``_init_axes`` unconditionally, and
``from_radar_config`` hard-codes FMCW, so a non-FMCW ``Radar`` is
unconstructible today and making it constructible is a separate change with its
own consequences for every ``Radar`` consumer. This record is built at the
``RadarSystemConfig`` / ``SynthesisResult`` level, where all three waveforms
already exist. :meth:`ProcessingAxes.as_fmcw_axes` is the mechanical migration
for the three existing ``RadarAxes`` consumers.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..synthesis.contracts import (
    BEAT_PHASOR,
    CHANNEL_PHASOR,
    SPEED_OF_LIGHT_M_PER_S,
)
from .contracts import FAST_TIME_NAMES, PROCESSING_UNITS, SLOW_TIME_NAMES
from .primitives import pulse_replica


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
            raise ValueError(
                f"waveform must be one of {WAVEFORMS}, got {self.waveform!r}"
            )
        if self.doppler_sign not in (1, -1):
            raise ValueError(
                f"doppler_sign must be +1 or -1, got {self.doppler_sign!r}"
            )
        if int(self.range_m.shape[0]) != self.range_bin_count:
            raise ValueError(
                f"range_m holds {int(self.range_m.shape[0])} bins but "
                f"range_bin_count is {self.range_bin_count}"
            )
        if int(self.velocity_mps.shape[0]) != self.doppler_bin_count:
            raise ValueError(
                f"velocity_mps holds {int(self.velocity_mps.shape[0])} bins but "
                f"doppler_bin_count is {self.doppler_bin_count}"
            )
        if self.range_m.dtype != torch.float64:
            raise TypeError(
                "range_m must be float64: it is a coordinate, and a float32 "
                "metre at 300 m has a 2 cm ulp"
            )
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
    def from_synthesis(
        cls,
        result,
        spec,
        array,
        *,
        range_oversample: int = 1,
    ) -> "ProcessingAxes":
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
            raise ValueError(
                f"range_oversample must be a positive int, got {range_oversample!r}"
            )

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
            range_bin = SPEED_OF_LIGHT_M_PER_S * float(spec.sample_rate_hz) / (
                2.0 * float(spec.slope_hz_per_s) * int(spec.num_samples)
            )
            range_origin = 0.0
            max_range = range_bin * range_count
        elif kind == "ofdm":
            slow_period = float(spec.symbol_period_s)
            max_speed = float(spec.max_unambiguous_speed_mps)
            range_count = fast_count
            range_bin = float(spec.range_resolution_m)
            range_origin = 0.0
            max_range = (
                SPEED_OF_LIGHT_M_PER_S * float(spec.max_unambiguous_delay_s) / 2.0
            )
        else:
            slow_period = float(spec.pri_s)
            max_speed = float(spec.max_unambiguous_speed_m_s)
            range_count = fast_count * range_oversample
            range_bin = SPEED_OF_LIGHT_M_PER_S * float(spec.sample_period_s) / (
                2.0 * range_oversample
            )
            range_origin = (
                SPEED_OF_LIGHT_M_PER_S * float(spec.range_gate_start_s) / 2.0
            )
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
        range_m = (
            torch.arange(range_count, dtype=torch.float64, device=device) * range_bin
            + range_origin
        )
        velocity_mps = torch.fft.fftshift(
            torch.fft.fftfreq(
                slow_count, d=slow_period, dtype=torch.float64, device=device
            )
        ) * (wavelength / 2.0)

        return cls(
            waveform=kind,
            fast_time_name=FAST_TIME_NAMES[kind],
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
            tx_loc_half_wavelength=tuple(
                tuple(float(value) for value in row) for row in array.tx_loc
            ),
            rx_loc_half_wavelength=tuple(
                tuple(float(value) for value in row) for row in array.rx_loc
            ),
            range_m=range_m.contiguous(),
            velocity_mps=velocity_mps.contiguous(),
            range_oversample=range_oversample,
            matched_filter_replica=replica,
            matched_filter_sample_period_s=replica_period,
        )

    # -- migration ----------------------------------------------------------

    def as_fmcw_axes(self):
        """The Phase-6 ``RadarAxes`` view, for the three ``sigproc`` consumers.

        Mechanical, not a reinterpretation: ``ranges`` is the first half of
        :attr:`range_m` (the half a real-valued range axis is defined on),
        ``velocities`` IS :attr:`velocity_mps`, and the two resolutions are the
        same two bin widths. ``chirp_period_s`` is the RAW chirp period, which
        is what ``RadarAxes.from_fmcw`` publishes and what
        ``_compensate_tdm_phase`` multiplies by a transmitter index; this
        record's :attr:`slow_time_period_s` is the TDM SLOT period, which is
        ``num_tx`` times larger, and confusing the two costs a factor of
        ``num_tx`` in every compensated elevation.

        Refused for anything but FMCW, because ``RadarAxes``'s field names -
        ``doppler_resolution``, ``chirp_period_s`` - are FMCW vocabulary, and a
        record whose names lie about the waveform is worse than no record.
        """

        if self.waveform != "fmcw":
            raise NotImplementedError(
                "RadarAxes is the FMCW-shaped Phase-6 record: its field names "
                "are chirp and Doppler vocabulary. This axes record is "
                f"{self.waveform!r}, and every stage in this package reads "
                "ProcessingAxes directly"
            )
        from ..config import RadarAxes

        return RadarAxes(
            ranges=self.range_m[: self.range_bin_count // 2],
            velocities=self.velocity_mps,
            range_resolution=self.range_bin_m,
            doppler_resolution=self.velocity_bin_mps,
            max_range=self.max_unambiguous_range_m,
            max_doppler=self.max_unambiguous_speed_mps,
            wavelength_m=self.wavelength_m,
            chirp_period_s=self.slow_time_period_s / self.num_tx,
            element_spacing_m=self.element_spacing_m,
        )


__all__ = ["WAVEFORMS", "ProcessingAxes"]
