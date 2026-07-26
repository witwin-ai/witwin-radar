"""The radar configuration, split into five blocks with a stored discriminator.

Work item 6. The flat configuration has 18 required fields, four optional
dicts, and no waveform discriminator at all: every consumer infers "this is
FMCW" by finding a ``slope``. That inference is why waveform, ADC, and receiver
vocabulary has no structural reason to stay out of a propagation request - a
field that nobody groups is a field anybody may read.

Five blocks, one rule each:

    waveform     FmcwWaveformConfig | OfdmWaveformConfig | PulsedWaveformConfig
                 discriminated by a STORED ``kind``, never by an inferred one
    sensors      the array, the antenna pattern, transmit power, polarization
    frontend     the receive chain: one ordered chain, one ADC, one seed base
    propagation  reference frequency, components, max depth
    processing   frame rate and the three bin counts

``PropagationConfig`` is the ONLY block a propagation adapter is given, which is
what makes "a waveform field reaches a propagation request" structurally
impossible rather than merely discouraged. ``reference_frequency_hz`` is the one
legitimate crossing, and it is one number: an OFDM band is a per-subcarrier
offset applied by the Radar synthesis kernel from the narrowband law, never a
set of reference frequencies pushed into requests.

Each waveform block has a ``to_spec`` returning the SI synthesis spec, and that
is the only unit-conversion site for its waveform. The engineering units the
flat configuration uses - kSPS, microseconds, MHz per microsecond - are
converted there and nowhere else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import torch

from .frontend.contracts import FrontendSpec
from .sensors.contracts import (
    AntennaPatternSpec,
    PolarizationSpec,
    SensorArraySpec,
    TxPowerSpec,
)
from .synthesis.contracts import (
    PULSE_NORMALIZATION_UNIT_ENERGY,
    SPEED_OF_LIGHT_M_PER_S,
    SUBCARRIER_ORIGIN_F_REF_AT_N0,
    FmcwBeatSpec,
    OfdmCfrSpec,
    PulsedEchoSpec,
)


WAVEFORM_FMCW = "fmcw"
WAVEFORM_OFDM = "ofdm"
WAVEFORM_PULSED = "pulsed"
WAVEFORM_KINDS = (WAVEFORM_FMCW, WAVEFORM_OFDM, WAVEFORM_PULSED)


@dataclass(frozen=True, slots=True)
class FmcwWaveformConfig:
    """One FMCW ramp and its ADC window, in the configuration's own units.

    Exactly the fields ``FmcwBeatSpec.from_radar_config`` reads today:
    ``sample_rate`` in kSPS, ``adc_start_time`` / ``idle_time`` /
    ``ramp_end_time`` in microseconds, and ``slope`` in MHz per microsecond,
    which is 1e12 Hz per second.
    """

    kind: ClassVar[str] = WAVEFORM_FMCW

    slope: float
    adc_samples: int
    adc_start_time: float
    sample_rate: float
    idle_time: float
    ramp_end_time: float
    chirp_per_frame: int

    def to_spec(
        self,
        *,
        reference_frequency_hz: float,
        num_tx: int = 1,
        num_rx: int = 1,
        carrier_hz: float = 0.0,
    ) -> FmcwBeatSpec:
        carrier = float(carrier_hz)
        return FmcwBeatSpec(
            num_samples=int(self.adc_samples),
            num_chirps=int(self.chirp_per_frame),
            sample_period_s=1.0 / (float(self.sample_rate) * 1e3),
            chirp_period_s=(float(self.idle_time) + float(self.ramp_end_time)) * 1e-6,
            slope_hz_per_s=float(self.slope) * 1e12,
            t_start_s=float(self.adc_start_time) * 1e-6,
            reference_frequency_hz=float(reference_frequency_hz),
            carrier_hz=carrier,
            carrier_rate_hz=0.0 if carrier != 0.0 else float(reference_frequency_hz),
            num_tx=int(num_tx),
            num_rx=int(num_rx),
        )


@dataclass(frozen=True, slots=True)
class OfdmWaveformConfig:
    """One OFDM subcarrier grid and symbol timing, in SI units.

    ``max_expected_delay_s`` is a CONFIGURED bound - the range window the radar
    is set up for - and never a measured maximum, which would be a per-frame
    device-to-host transfer. The cyclic-prefix refusal is written against it.
    """

    kind: ClassVar[str] = WAVEFORM_OFDM

    subcarrier_spacing_hz: float
    num_subcarriers: int
    cyclic_prefix_s: float
    num_symbols: int
    max_expected_delay_s: float
    subcarrier_origin: str = SUBCARRIER_ORIGIN_F_REF_AT_N0

    def to_spec(
        self, *, reference_frequency_hz: float, carrier_hz: float = 0.0
    ) -> OfdmCfrSpec:
        carrier = float(carrier_hz)
        return OfdmCfrSpec(
            num_subcarriers=int(self.num_subcarriers),
            num_symbols=int(self.num_symbols),
            subcarrier_spacing_hz=float(self.subcarrier_spacing_hz),
            cyclic_prefix_s=float(self.cyclic_prefix_s),
            reference_frequency_hz=float(reference_frequency_hz),
            max_expected_delay_s=float(self.max_expected_delay_s),
            carrier_hz=carrier,
            carrier_rate_hz=0.0 if carrier != 0.0 else float(reference_frequency_hz),
            subcarrier_origin=self.subcarrier_origin,
        )


@dataclass(frozen=True, slots=True)
class PulsedWaveformConfig:
    """One pulse train's shape, gate, and repetition interval, in SI units.

    ``max_expected_delay_rate`` is a CONFIGURED bound on ``|d(tau_rt)/dt|`` - the
    velocity window the radar is set up for - and never a measured maximum. It
    is what the range-migration refusal is written against.
    """

    kind: ClassVar[str] = WAVEFORM_PULSED

    pulse_kind: str
    pulse_width_s: float
    bandwidth_hz: float
    pri_s: float
    num_pulses: int
    sample_rate_hz: float
    num_samples: int
    range_gate_start_s: float
    max_expected_delay_rate: float = 0.0
    pulse_normalization: str = PULSE_NORMALIZATION_UNIT_ENERGY

    def to_spec(
        self, *, reference_frequency_hz: float, carrier_hz: float = 0.0
    ) -> PulsedEchoSpec:
        carrier = float(carrier_hz)
        return PulsedEchoSpec(
            num_pulses=int(self.num_pulses),
            num_samples=int(self.num_samples),
            sample_period_s=1.0 / float(self.sample_rate_hz),
            pri_s=float(self.pri_s),
            range_gate_start_s=float(self.range_gate_start_s),
            pulse_kind=self.pulse_kind,
            pulse_width_s=float(self.pulse_width_s),
            bandwidth_hz=float(self.bandwidth_hz),
            reference_frequency_hz=float(reference_frequency_hz),
            max_expected_delay_rate=float(self.max_expected_delay_rate),
            carrier_hz=carrier,
            carrier_rate_hz=0.0 if carrier != 0.0 else float(reference_frequency_hz),
            pulse_normalization=self.pulse_normalization,
        )


WaveformConfig = FmcwWaveformConfig | OfdmWaveformConfig | PulsedWaveformConfig


@dataclass(frozen=True, slots=True)
class SensorConfig:
    """The array, its pattern, its transmit power, and the legacy projection."""

    array: SensorArraySpec
    pattern: AntennaPatternSpec
    tx_power: TxPowerSpec
    polarization: PolarizationSpec | None = None


@dataclass(frozen=True, slots=True)
class PropagationConfig:
    """The ONLY block a propagation adapter is ever handed.

    Folding a waveform field in here is what work item 6 exists to prevent, and
    the boundary test asserts the request keyword set by EQUALITY rather than by
    containment, because a containment check passes when a field is added.
    """

    reference_frequency_hz: float
    components: frozenset[str] = frozenset({"los", "reflection"})
    max_depth: int = 1

    def __post_init__(self) -> None:
        if not self.reference_frequency_hz > 0.0:
            raise ValueError("reference_frequency_hz must be positive")
        if self.max_depth < 0:
            raise ValueError("max_depth must be non-negative")


@dataclass(frozen=True, slots=True)
class ProcessingConfig:
    """Frame rate and the three bin counts the signal processor indexes by."""

    frame_per_second: float
    num_doppler_bins: int
    num_range_bins: int
    num_angle_bins: int


@dataclass(frozen=True, slots=True)
class RadarAxes:
    """The range and velocity axes, as ONE record ``sigproc`` reads.

    ``sigproc`` used to read ``radar.range_resolution``, ``radar._lambda``,
    ``radar.config.idle_time``, and four more raw scalars straight off the radar.
    That is how a signal processor ends up knowing which waveform it is looking
    at. It reads this record instead, and the waveform owner fills it.
    """

    ranges: torch.Tensor
    velocities: torch.Tensor
    range_resolution: float
    doppler_resolution: float
    max_range: float
    max_doppler: float
    #: Three more derived scalars, here for the same reason as the five above:
    #: they are the LAST raw reads ``sigproc`` had left. The TDM phase
    #: compensation needs the wavelength and the chirp period, and the angle
    #: estimator needs the element spacing to recover half-wavelength element
    #: offsets. Leaving those three as ``radar._lambda`` and
    #: ``radar.config.idle_time`` would have kept a private attribute and two
    #: waveform fields inside the signal processor, which is the thing this
    #: record exists to stop.
    wavelength_m: float
    chirp_period_s: float
    element_spacing_m: float

    @classmethod
    def from_fmcw(
        cls,
        waveform: FmcwWaveformConfig,
        processing: ProcessingConfig,
        *,
        reference_frequency_hz: float,
        num_tx: int,
        device: torch.device | str = "cpu",
        c0: float = SPEED_OF_LIGHT_M_PER_S,
    ) -> "RadarAxes":
        target = torch.device(device)
        sample_rate_hz = float(waveform.sample_rate) * 1e3
        slope_hz_per_s = float(waveform.slope) * 1e12
        wavelength = c0 / float(reference_frequency_hz)

        range_resolution = (
            c0 * sample_rate_hz / (2 * slope_hz_per_s * int(waveform.adc_samples))
        )
        max_range = c0 * sample_rate_hz / (2 * slope_hz_per_s)
        chirp_period = (float(waveform.idle_time) + float(waveform.ramp_end_time)) * 1e-6
        effective_period = chirp_period * int(num_tx)
        doppler_resolution = wavelength / (
            2 * int(processing.num_doppler_bins) * effective_period
        )
        max_doppler = wavelength / (4 * chirp_period * int(num_tx))
        ranges = (
            torch.arange(
                0,
                int(processing.num_range_bins) // 2,
                dtype=torch.float64,
                device=target,
            )
            * range_resolution
        )
        velocities = (
            torch.arange(
                -int(processing.num_doppler_bins) // 2,
                int(processing.num_doppler_bins) // 2,
                dtype=torch.float64,
                device=target,
            )
            * doppler_resolution
        )
        return cls(
            ranges=ranges,
            velocities=velocities,
            range_resolution=range_resolution,
            doppler_resolution=doppler_resolution,
            max_range=max_range,
            max_doppler=max_doppler,
            wavelength_m=wavelength,
            chirp_period_s=chirp_period,
            element_spacing_m=wavelength / 2.0,
        )


@dataclass(frozen=True, slots=True)
class RadarSystemConfig:
    """The five blocks, with the waveform discriminator stored rather than read.

    ``waveform.kind`` is the dispatch key. It is a class attribute of the
    waveform block, so a block cannot be built with the wrong one and a caller
    cannot infer a different one by looking for a ``slope``.
    """

    waveform: WaveformConfig
    sensors: SensorConfig
    propagation: PropagationConfig
    processing: ProcessingConfig
    frontend: FrontendSpec | None = None

    def __post_init__(self) -> None:
        if self.waveform.kind not in WAVEFORM_KINDS:
            raise ValueError(
                f"waveform.kind must be one of {list(WAVEFORM_KINDS)}, got "
                f"{self.waveform.kind!r}"
            )
        if (
            self.sensors.array.reference_frequency_hz
            != self.propagation.reference_frequency_hz
        ):
            raise ValueError(
                "the array's reference frequency and the propagation reference "
                "frequency are the same physical quantity and must agree; the "
                "array element spacing is defined in half-wavelengths at that "
                "frequency"
            )

    @property
    def kind(self) -> str:
        return self.waveform.kind

    def waveform_spec(self, *, carrier_hz: float = 0.0):
        """The SI synthesis spec for whichever waveform this configuration is.

        Dispatch is a match on a STORED discriminator, and an unknown kind is a
        hard error rather than a fallback: a waveform with no owner has no
        physics, and returning something plausible would be worse than failing.
        """

        array = self.sensors.array
        reference = self.propagation.reference_frequency_hz
        if self.waveform.kind == WAVEFORM_FMCW:
            return self.waveform.to_spec(
                reference_frequency_hz=reference,
                num_tx=array.num_tx,
                num_rx=array.num_rx,
                carrier_hz=carrier_hz,
            )
        if self.waveform.kind in (WAVEFORM_OFDM, WAVEFORM_PULSED):
            return self.waveform.to_spec(
                reference_frequency_hz=reference, carrier_hz=carrier_hz
            )
        raise ValueError(
            f"no synthesis owner for waveform kind {self.waveform.kind!r}; a "
            "waveform without an owner has no physics and this dispatch has no "
            "fallback"
        )

    @classmethod
    def from_radar_config(cls, config, *, frontend: FrontendSpec | None = None):
        """Split a flat ``RadarConfig`` into the five blocks.

        The flat form remains the file format and the public constructor; this
        is the structural view of it. Blocks are the thing an adapter, a
        synthesis owner, or a signal processor is handed, so that each one sees
        only what it owns.
        """

        return cls(
            waveform=FmcwWaveformConfig(
                slope=float(config.slope),
                adc_samples=int(config.adc_samples),
                adc_start_time=float(config.adc_start_time),
                sample_rate=float(config.sample_rate),
                idle_time=float(config.idle_time),
                ramp_end_time=float(config.ramp_end_time),
                chirp_per_frame=int(config.chirp_per_frame),
            ),
            sensors=SensorConfig(
                array=SensorArraySpec.from_radar_config(config),
                pattern=AntennaPatternSpec.from_config(config.antenna_pattern),
                tx_power=TxPowerSpec.from_radar_config(config),
                polarization=PolarizationSpec.from_config(config.polarization),
            ),
            propagation=PropagationConfig(reference_frequency_hz=float(config.fc)),
            processing=ProcessingConfig(
                frame_per_second=float(config.frame_per_second),
                num_doppler_bins=int(config.num_doppler_bins),
                num_range_bins=int(config.num_range_bins),
                num_angle_bins=int(config.num_angle_bins),
            ),
            frontend=frontend,
        )

    def axes(self, *, device: torch.device | str = "cpu") -> RadarAxes:
        if self.waveform.kind != WAVEFORM_FMCW:
            raise NotImplementedError(
                "range and velocity axes are defined for the FMCW block today; "
                f"waveform kind {self.waveform.kind!r} publishes its resolutions "
                "on its own synthesis spec and has no bin grid yet"
            )
        return RadarAxes.from_fmcw(
            self.waveform,
            self.processing,
            reference_frequency_hz=self.propagation.reference_frequency_hz,
            num_tx=self.sensors.array.num_tx,
            device=device,
        )


__all__ = [
    "WAVEFORM_FMCW",
    "WAVEFORM_KINDS",
    "WAVEFORM_OFDM",
    "WAVEFORM_PULSED",
    "FmcwWaveformConfig",
    "OfdmWaveformConfig",
    "ProcessingConfig",
    "PropagationConfig",
    "PulsedWaveformConfig",
    "RadarAxes",
    "RadarSystemConfig",
    "SensorConfig",
    "WaveformConfig",
]
