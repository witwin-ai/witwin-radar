"""The radar: its parameters, its pose, and the entry points that use them.

One flat immutable record. The fields are what a datasheet or a configuration
file lists, in SI units, one field each. A sub-record appears only where a
field has variants (:class:`Fmcw` / :class:`Ofdm` / :class:`Pulsed`,
:class:`~witwin.radar.sensors.Pattern`) or where several numbers are coupled
(:class:`~witwin.radar.frontend.Noise`, ``Agc``, ``Adc``). Everything else -
the antenna layout, the receive chain's one-number stages, the pose - is a
field here, because a wrapper whose only job is grouping makes the caller
write two constructors to say one thing.

No field name carries a unit suffix; every field's unit is on its docstring
row. That rule covers this surface only. :mod:`witwin.radar.processing` is
frozen by R-ADR-017 and keeps ``range_m`` and its siblings, and the internal
records (``FmcwSpec.sample_period_s``, ``RadarPathBatch.total_delay_s``) keep
theirs, because those are the equation-owning modules where R-ADR-021 requires
the unit to be in the identifier.

This module owns no propagation and no synthesis physics. It holds the
parameter record, the pose transforms every consumer shares, and the
derivation of the internal block specs those consumers are handed.
``Radar.simulate`` delegates to :mod:`witwin.radar.simulation`.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, ClassVar

import torch

from .frontend import Adc, Agc, FrontendSpec, Noise
from .sensors import Pattern, SensorArraySpec, watts_from_dbm
from .synthesis.assembly import (
    PULSE_NORMALIZATION_UNIT_ENERGY,
    SUBCARRIER_ORIGIN_F_REF_AT_N0,
    FmcwSpec,
    OfdmSpec,
    PulsedSpec,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .simulation import Motion, Paths, RadarSimulationResult
    from .synthesis import SynthesisResult
    from .targets import PointTargets, StructureTargets

WAVEFORM_FMCW = "fmcw"
WAVEFORM_OFDM = "ofdm"
WAVEFORM_PULSED = "pulsed"
WAVEFORM_KINDS = (WAVEFORM_FMCW, WAVEFORM_OFDM, WAVEFORM_PULSED)

SPEED_OF_LIGHT_M_PER_S = 299792458.0

#: What ``antenna_unit`` may say. Half-wavelength offsets are the convention
#: every TI-style configuration uses and the one the array keeps internally:
#: the same description then means the same beam pattern at a different
#: carrier, because an array is defined by its electrical size.
ANTENNA_UNITS = ("m", "half_wavelength")

#: What ``polarization`` may name instead of a world vector. Both are derived
#: from the pose, so they are transverse to the boresight by construction - the
#: property a hand-written vector silently loses, producing a cube of exact
#: zeros with nothing raised.
POLARIZATION_ALIASES = ("up", "right")


# ---------------------------------------------------------------------------
# Waveforms
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Fmcw:
    """One FMCW ramp and its ADC window, in SI units.

    The vendor units a millimetre-wave datasheet quotes - MHz/us of slope, kSPS
    of sample rate, microseconds of timing - are read by :meth:`from_ti`, which
    is the one conversion site in the package. Everything downstream of this
    record is SI.
    """

    kind: ClassVar[str] = WAVEFORM_FMCW

    #: Ramp slope, Hz/s. A 60.012 MHz/us ramp is ``60.012e12``.
    slope: float
    #: ADC sample rate, Hz.
    sample_rate: float
    #: ADC samples taken per chirp.
    samples_per_chirp: int
    #: Chirps per frame, per transmitter.
    chirps_per_frame: int
    #: Delay from the chirp start to the first ADC sample, s.
    adc_start: float
    #: Idle time between ramps, s.
    idle: float
    #: Ramp end time, s. The chirp period is ``idle + ramp_end``.
    ramp_end: float
    #: ``"spectrum"`` for the normalized range spectrum, ``"beat"`` for the
    #: synthesized time-domain beat samples. Spectrum is the default and the
    #: beat route is an opt-in output domain, never a fallback.
    output: str = "spectrum"

    def __post_init__(self) -> None:
        _positive("Fmcw.sample_rate", self.sample_rate)
        _positive("Fmcw.ramp_end", self.ramp_end)
        _positive_int("Fmcw.samples_per_chirp", self.samples_per_chirp)
        _positive_int("Fmcw.chirps_per_frame", self.chirps_per_frame)
        _finite("Fmcw.slope", self.slope)
        _finite("Fmcw.adc_start", self.adc_start)
        _non_negative("Fmcw.idle", self.idle)
        if self.output not in ("spectrum", "beat"):
            raise ValueError(f"Fmcw.output must be 'spectrum' or 'beat', got {self.output!r}")

    @property
    def chirp_period(self) -> float:
        """``idle + ramp_end``, s."""

        return float(self.idle) + float(self.ramp_end)

    @property
    def bandwidth(self) -> float:
        """The sampling bandwidth a thermal-noise stage integrates over, Hz."""

        return float(self.sample_rate)

    @classmethod
    def from_ti(
        cls,
        *,
        slope_mhz_per_us: float,
        sample_rate_ksps: float,
        samples_per_chirp: int,
        chirps_per_frame: int,
        adc_start_us: float,
        idle_us: float,
        ramp_end_us: float,
        output: str = "spectrum",
    ) -> Fmcw:
        """Build from the units a TI-style configuration file quotes.

        The parameter names carry their vendor units here, and only here,
        because that is the whole job of this constructor: a caller reading a
        datasheet needs to see which column each number came from.
        """

        return cls(
            slope=float(slope_mhz_per_us) * 1e12,
            sample_rate=float(sample_rate_ksps) * 1e3,
            samples_per_chirp=int(samples_per_chirp),
            chirps_per_frame=int(chirps_per_frame),
            adc_start=float(adc_start_us) * 1e-6,
            idle=float(idle_us) * 1e-6,
            ramp_end=float(ramp_end_us) * 1e-6,
            output=str(output),
        )

    def to_spec(self, *, carrier: float, num_tx: int = 1, num_rx: int = 1, offset: float = 0.0) -> FmcwSpec:
        """The SI synthesis spec. ``offset`` is a wideband band offset, Hz."""

        carrier_hz = float(offset)
        return FmcwSpec(
            num_samples=int(self.samples_per_chirp),
            num_chirps=int(self.chirps_per_frame),
            sample_period_s=1.0 / float(self.sample_rate),
            chirp_period_s=self.chirp_period,
            slope_hz_per_s=float(self.slope),
            t_start_s=float(self.adc_start),
            reference_frequency_hz=float(carrier),
            carrier_hz=carrier_hz,
            carrier_rate_hz=0.0 if carrier_hz != 0.0 else float(carrier),
            num_tx=int(num_tx),
            num_rx=int(num_rx),
            output_domain=self.output,
        )


@dataclass(frozen=True, slots=True)
class Ofdm:
    """One OFDM symbol grid, in SI units."""

    kind: ClassVar[str] = WAVEFORM_OFDM

    #: Subcarrier spacing, Hz.
    subcarrier_spacing: float
    num_subcarriers: int
    #: Cyclic prefix duration, s.
    cyclic_prefix: float
    num_symbols: int
    #: The longest round-trip delay the cyclic prefix must cover, s.
    max_expected_delay: float
    subcarrier_origin: str = SUBCARRIER_ORIGIN_F_REF_AT_N0

    def __post_init__(self) -> None:
        _positive("Ofdm.subcarrier_spacing", self.subcarrier_spacing)
        _positive("Ofdm.cyclic_prefix", self.cyclic_prefix)
        _positive_int("Ofdm.num_subcarriers", self.num_subcarriers)
        _positive_int("Ofdm.num_symbols", self.num_symbols)
        _non_negative("Ofdm.max_expected_delay", self.max_expected_delay)

    @property
    def bandwidth(self) -> float:
        """The occupied band a thermal-noise stage integrates over, Hz."""

        return float(self.subcarrier_spacing) * int(self.num_subcarriers)

    @property
    def sample_rate(self) -> float:
        """The symbol grid's sample rate, Hz."""

        return self.bandwidth

    def to_spec(self, *, carrier: float, offset: float = 0.0) -> OfdmSpec:
        carrier_hz = float(offset)
        return OfdmSpec(
            subcarrier_spacing_hz=float(self.subcarrier_spacing),
            num_subcarriers=int(self.num_subcarriers),
            cyclic_prefix_s=float(self.cyclic_prefix),
            num_symbols=int(self.num_symbols),
            max_expected_delay_s=float(self.max_expected_delay),
            reference_frequency_hz=float(carrier),
            carrier_hz=carrier_hz,
            # Derived, never passed: the weight owns the carrier on the
            # production path, so the rate is the reference frequency there and
            # zero when the caller puts the carrier in the kernel instead.
            # Dropping it understates Doppler by the whole carrier term.
            carrier_rate_hz=0.0 if carrier_hz != 0.0 else float(carrier),
            subcarrier_origin=self.subcarrier_origin,
        )


@dataclass(frozen=True, slots=True)
class Pulsed:
    """One pulse train and its range gate, in SI units."""

    kind: ClassVar[str] = WAVEFORM_PULSED

    #: ``"rect"`` or ``"lfm"``.
    pulse_kind: str
    #: Pulse width, s.
    pulse_width: float
    #: Chirp bandwidth of an LFM pulse, Hz.
    bandwidth: float
    #: Pulse repetition interval, s.
    pri: float
    num_pulses: int
    #: Receive sample rate, Hz.
    sample_rate: float
    num_samples: int
    #: Delay from the pulse start to the first range gate sample, s.
    range_gate_start: float
    #: The largest ``d(tau)/dt`` the matched filter must tolerate,
    #: dimensionless.
    max_expected_delay_rate: float = 0.0
    pulse_normalization: str = PULSE_NORMALIZATION_UNIT_ENERGY

    def __post_init__(self) -> None:
        _positive("Pulsed.pulse_width", self.pulse_width)
        _positive("Pulsed.bandwidth", self.bandwidth)
        _positive("Pulsed.pri", self.pri)
        _positive("Pulsed.sample_rate", self.sample_rate)
        _positive_int("Pulsed.num_pulses", self.num_pulses)
        _positive_int("Pulsed.num_samples", self.num_samples)
        _non_negative("Pulsed.range_gate_start", self.range_gate_start)
        _non_negative("Pulsed.max_expected_delay_rate", self.max_expected_delay_rate)

    def to_spec(self, *, carrier: float, offset: float = 0.0) -> PulsedSpec:
        carrier_hz = float(offset)
        return PulsedSpec(
            pulse_kind=str(self.pulse_kind),
            pulse_width_s=float(self.pulse_width),
            bandwidth_hz=float(self.bandwidth),
            pri_s=float(self.pri),
            num_pulses=int(self.num_pulses),
            sample_period_s=1.0 / float(self.sample_rate),
            num_samples=int(self.num_samples),
            range_gate_start_s=float(self.range_gate_start),
            max_expected_delay_rate=float(self.max_expected_delay_rate),
            reference_frequency_hz=float(carrier),
            carrier_hz=carrier_hz,
            carrier_rate_hz=0.0 if carrier_hz != 0.0 else float(carrier),
            pulse_normalization=self.pulse_normalization,
        )


Waveform = Fmcw | Ofdm | Pulsed


# ---------------------------------------------------------------------------
# Internal block specs
#
# These are what a consumer is HANDED: the propagation adapter sees the
# propagation block and nothing else, the sensor stage sees the array. They are
# derived from the radar rather than authored, which is why they keep their
# unit-suffixed field names and are absent from the public surface.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SensorConfig:
    """The array, its antenna pattern, and its transmit power."""

    array: SensorArraySpec
    pattern: Pattern
    power_dbm: float


@dataclass(frozen=True, slots=True)
class PropagationConfig:
    """The ONLY block a propagation adapter is ever handed.

    Folding a waveform field in here is what the boundary test exists to
    prevent, and it asserts the request keyword set by EQUALITY rather than by
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
class RadarSystemConfig:
    """The four blocks, with the waveform discriminator stored rather than read.

    ``waveform.kind`` is the dispatch key. It is a class attribute of the
    waveform record, so a block cannot be built with the wrong one and a caller
    cannot infer a different one by looking for a ``slope``.
    """

    waveform: Waveform
    sensors: SensorConfig
    propagation: PropagationConfig
    frontend: FrontendSpec | None = None

    def __post_init__(self) -> None:
        if self.waveform.kind not in WAVEFORM_KINDS:
            raise ValueError(f"waveform.kind must be one of {list(WAVEFORM_KINDS)}, got {self.waveform.kind!r}")
        if self.sensors.array.reference_frequency_hz != self.propagation.reference_frequency_hz:
            raise ValueError(
                "the array's reference frequency and the propagation reference "
                "frequency are the same physical quantity and must agree"
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
            return self.waveform.to_spec(carrier=reference, num_tx=array.num_tx, num_rx=array.num_rx, offset=carrier_hz)
        if self.waveform.kind in (WAVEFORM_OFDM, WAVEFORM_PULSED):
            return self.waveform.to_spec(carrier=reference, offset=carrier_hz)
        raise ValueError(
            f"no synthesis owner for waveform kind {self.waveform.kind!r}; a "
            "waveform without an owner has no physics and this dispatch has no fallback"
        )

    def with_propagation(self, *, components=None, max_depth: int | None = None) -> RadarSystemConfig:
        """A copy whose propagation block carries these two knobs.

        A per-solve request returns a new configuration rather than mutating
        this one, because an override that edited the radar's stored
        configuration would silently change every LATER solve as well.

        ``reference_frequency_hz`` is deliberately not overridable: it is tied
        to the array's element spacing and to the compiled scene by Channel, so
        changing it here would produce a configuration that is refused later
        rather than one that means something else.
        """

        if components is None and max_depth is None:
            return self
        current = self.propagation
        return replace(
            self,
            propagation=PropagationConfig(
                reference_frequency_hz=current.reference_frequency_hz,
                components=(current.components if components is None else frozenset(components)),
                max_depth=(current.max_depth if max_depth is None else int(max_depth)),
            ),
        )


# ---------------------------------------------------------------------------
# Primitive validators
# ---------------------------------------------------------------------------


def _finite(name: str, value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite float, got {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be a finite float, got {value!r}")
    return parsed


def _non_negative(name: str, value: Any) -> float:
    parsed = _finite(name, value)
    if parsed < 0.0:
        raise ValueError(f"{name} must be non-negative, got {parsed}")
    return parsed


def _positive(name: str, value: Any) -> float:
    parsed = _finite(name, value)
    if parsed <= 0.0:
        raise ValueError(f"{name} must be positive, got {parsed}")
    return parsed


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive int, got {value!r}")
    return value


def vec3_tensor(value, *, name: str) -> torch.Tensor:
    """Coerce to a CPU float32 tensor of shape (3,)."""

    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    else:
        tensor = torch.tensor(tuple(float(component) for component in value), dtype=torch.float32)
    if tensor.shape != (3,):
        raise ValueError(f"{name} must contain exactly three values.")
    return tensor


def _elements(value, *, name: str) -> tuple[tuple[float, float, float], ...]:
    if isinstance(value, torch.Tensor):
        rows = value.detach().to(device="cpu", dtype=torch.float32).tolist()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        rows = list(value)
    else:
        raise TypeError(f"{name} must be a sequence of 3-element positions, got {type(value).__name__}")
    if not rows:
        raise ValueError(f"{name} must name at least one element")
    out: list[tuple[float, float, float]] = []
    for index, row in enumerate(rows):
        if isinstance(row, torch.Tensor):
            row = row.detach().to(device="cpu", dtype=torch.float32).tolist()
        if not isinstance(row, Sequence) or len(row) != 3:
            raise ValueError(f"{name}[{index}] must be a 3-element position")
        out.append(tuple(_finite(f"{name}[{index}][{axis}]", row[axis]) for axis in range(3)))
    return tuple(out)


# ---------------------------------------------------------------------------
# Radar
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Radar:
    """One radar: its waveform, its array, its receive chain and its pose.

    Immutable. :meth:`replace` returns a new radar rather than editing this
    one, so a radar captured in a closure or held by a result cannot change
    underneath it, and a pose built from a tensor with a tape keeps that tape.

    Four verbs use it, and they are two halves and their fusion.
    :meth:`trace` runs the world half and keeps the composed round trips;
    :meth:`echo` runs the instrument half over those rows. :meth:`simulate`
    fuses the two and stacks every frame; :meth:`stream` fuses them and yields
    one frame at a time, so a sequence too long to hold as a stacked cube is
    still producible. All four share one session loop and one synthesis route,
    so ``echo(trace(...))`` and ``simulate(...)`` are the same numbers and not
    merely the same physics.
    """

    #: Reference frequency, Hz. The carrier the array spacing, the propagation
    #: solve and the synthesis all refer to; they are one physical quantity.
    carrier: float
    waveform: Waveform
    #: Transmit element positions, in ``antenna_unit``.
    tx: tuple[tuple[float, float, float], ...]
    #: Receive element positions, in ``antenna_unit``.
    rx: tuple[tuple[float, float, float], ...]
    #: Transmit power, dBm.
    power: float
    antenna_unit: str = "m"
    #: The element pattern. Isotropic by default: an unchosen dipole attenuates
    #: every off-boresight return by a number nobody asked for.
    pattern: Pattern = field(default_factory=Pattern.isotropic)
    #: Thermal and oscillator noise. ``None`` is an ideal receiver.
    noise: Noise | None = None
    #: Low-noise amplifier voltage gain, dB. ``None`` is no LNA stage.
    lna_gain: float | None = None
    agc: Agc | None = None
    #: ``None`` is no quantisation.
    adc: Adc | None = None
    #: Receive port reference impedance, ohm.
    impedance: float = 50.0
    #: The Philox base seed every receiver stage derives its own stream from.
    seed: int = 0
    #: Radar origin in world coordinates, m. May be a tensor with a tape.
    position: Any = (0.0, 0.0, 0.0)
    #: The point the boresight looks at, world coordinates, m.
    look_at: Any = (0.0, 0.0, -1.0)
    #: World-space up vector, used to complete the frame.
    up: Any = (0.0, 1.0, 0.0)
    #: ``"up"``, ``"right"`` or a world vector. The two aliases are derived
    #: from the pose and are therefore transverse to the boresight; a vector
    #: parallel to the boresight radiates nothing and is refused.
    polarization: Any = "up"
    device: Any = "cuda"

    # -- derived, built once in __post_init__ ------------------------------
    system_config: RadarSystemConfig = field(init=False, repr=False, compare=False)
    frontend: Any = field(init=False, repr=False, compare=False)
    tx_pos: torch.Tensor = field(init=False, repr=False, compare=False)
    rx_pos: torch.Tensor = field(init=False, repr=False, compare=False)
    polarization_vector: tuple[float, float, float] = field(init=False, repr=False, compare=False)

    # A radar holds no run state. The four typed per-frame diagnostics live on
    # the result that produced them, which is the record that can honestly say
    # which frame they describe; a copy here would be a second owner, and a
    # call that raised part way through would leave it describing a world the
    # failed call never simulated.

    def __post_init__(self) -> None:
        set_ = object.__setattr__
        _positive("Radar.carrier", self.carrier)
        _finite("Radar.power", self.power)
        _positive("Radar.impedance", self.impedance)
        if self.antenna_unit not in ANTENNA_UNITS:
            raise ValueError(f"Radar.antenna_unit must be one of {list(ANTENNA_UNITS)}, got {self.antenna_unit!r}")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError(f"Radar.seed must be a non-negative int, got {self.seed!r}")
        if not isinstance(self.pattern, Pattern):
            raise TypeError(f"Radar.pattern must be a Pattern, got {type(self.pattern).__name__}")
        if self.waveform.kind not in WAVEFORM_KINDS:
            raise ValueError(f"Radar.waveform must be Fmcw, Ofdm or Pulsed, got {type(self.waveform).__name__}")

        set_(self, "device", _resolve_device(self.device))
        set_(self, "tx", _elements(self.tx, name="Radar.tx"))
        set_(self, "rx", _elements(self.rx, name="Radar.rx"))

        # The array keeps half-wavelength offsets, so a metre-authored layout is
        # divided by the half wavelength exactly here and nowhere else.
        half_wavelength = SPEED_OF_LIGHT_M_PER_S / float(self.carrier) / 2.0
        scale = 1.0 if self.antenna_unit == "half_wavelength" else 1.0 / half_wavelength
        array = SensorArraySpec(
            num_tx=len(self.tx),
            num_rx=len(self.rx),
            tx_loc=tuple(tuple(v * scale for v in row) for row in self.tx),
            rx_loc=tuple(tuple(v * scale for v in row) for row in self.rx),
            reference_frequency_hz=float(self.carrier),
        )

        set_(
            self,
            "system_config",
            RadarSystemConfig(
                waveform=self.waveform,
                sensors=SensorConfig(array=array, pattern=self.pattern, power_dbm=float(self.power)),
                propagation=PropagationConfig(reference_frequency_hz=float(self.carrier)),
                frontend=self._build_frontend_spec(),
            ),
        )
        self._place_antennas()
        set_(self, "frontend", self._build_frontend_chain())

    # -- construction helpers ---------------------------------------------

    def _build_frontend_spec(self) -> FrontendSpec | None:
        """The internal receive-chain record, or ``None`` for an ideal receiver.

        A chain exists when any stage is configured. ``Noise`` arrives with its
        two waveform-dependent numbers possibly unset and leaves resolved, so
        nothing downstream of here ever has to infer a bandwidth - inferring it
        in three places is how those three quietly disagree.
        """

        if self.noise is None and self.lna_gain is None and self.agc is None and self.adc is None:
            return None
        if self.noise is not None and not isinstance(self.noise, Noise):
            raise TypeError(f"Radar.noise must be a Noise, got {type(self.noise).__name__}")
        if self.agc is not None and not isinstance(self.agc, Agc):
            raise TypeError(f"Radar.agc must be an Agc, got {type(self.agc).__name__}")
        if self.adc is not None and not isinstance(self.adc, Adc):
            raise TypeError(f"Radar.adc must be an Adc, got {type(self.adc).__name__}")
        noise = self.noise
        if noise is not None:
            noise = noise.resolved(bandwidth=self.waveform.bandwidth, sample_rate=self.waveform.sample_rate)
        return FrontendSpec(
            noise=noise,
            lna=None if self.lna_gain is None else float(self.lna_gain),
            agc=self.agc,
            adc=self.adc,
            impedance=float(self.impedance),
            seed=int(self.seed),
        )

    def _build_frontend_chain(self):
        if self.system_config.frontend is None:
            return None
        from .frontend import FrontendChain

        return FrontendChain(self.system_config.frontend)

    def _place_antennas(self) -> None:
        """Validate the pose and put the elements in the world.

        The refusals are all here rather than at first use: a pose whose up is
        collinear with the boresight has no frame, and a polarization parallel
        to the boresight radiates nothing and would otherwise publish a cube of
        exact zeros with nothing raised.
        """

        set_ = object.__setattr__
        position = vec3_tensor(self.position, name="Radar.position")
        look_at = vec3_tensor(self.look_at, name="Radar.look_at")
        up = vec3_tensor(self.up, name="Radar.up")
        forward = look_at - position
        if torch.linalg.norm(forward) <= 1e-12:
            raise ValueError("Radar.look_at must differ from Radar.position.")
        if torch.linalg.norm(up) <= 1e-12:
            raise ValueError("Radar.up must be non-zero.")
        if torch.linalg.norm(torch.cross(forward, up, dim=0)) <= 1e-12:
            raise ValueError("Radar.up must not be collinear with the viewing direction.")

        forward = forward / torch.linalg.norm(forward)
        right = torch.cross(forward, up, dim=0)
        right = right / torch.linalg.norm(right)
        true_up = torch.cross(right, forward, dim=0)
        true_up = true_up / torch.linalg.norm(true_up)

        if self.polarization == "up":
            vector = true_up
        elif self.polarization == "right":
            vector = right
        elif isinstance(self.polarization, str):
            raise ValueError(
                f"Radar.polarization must be one of {list(POLARIZATION_ALIASES)} or a world vector, "
                f"got {self.polarization!r}"
            )
        else:
            vector = vec3_tensor(self.polarization, name="Radar.polarization")
            norm = torch.linalg.norm(vector)
            if norm <= 1e-12:
                raise ValueError("Radar.polarization must be non-zero.")
            vector = vector / norm
            if float(torch.linalg.norm(torch.cross(vector, forward, dim=0))) <= 1e-6:
                raise ValueError(
                    "Radar.polarization is parallel to the boresight, so the field radiates nothing and every "
                    "transport would come back exactly zero. Use 'up' or 'right' for a vector that is transverse "
                    "by construction."
                )
        set_(self, "polarization_vector", tuple(float(v) for v in vector))

        array = self.system_config.sensors.array
        tx_local, rx_local = array.local_offsets_m(device=self.device)
        world_from_local = torch.stack((right, true_up, -forward), dim=1).to(device=self.device)
        origin = position.to(device=self.device)
        set_(self, "tx_pos", (tx_local @ world_from_local.transpose(0, 1) + origin).contiguous())
        set_(self, "rx_pos", (rx_local @ world_from_local.transpose(0, 1) + origin).contiguous())

    # -- derived reads -----------------------------------------------------

    @property
    def num_tx(self) -> int:
        """Transmit element count, which is just how many ``tx`` names."""

        return len(self.tx)

    @property
    def num_rx(self) -> int:
        """Receive element count, which is just how many ``rx`` names."""

        return len(self.rx)

    @property
    def wavelength(self) -> float:
        """``c0 / carrier``, m."""

        return SPEED_OF_LIGHT_M_PER_S / float(self.carrier)

    @property
    def transmit_power_watts(self) -> float:
        """``power`` in watts, which is what a source endpoint's field takes."""

        return watts_from_dbm(self.power)

    def waveform_spec(self, *, offset: float = 0.0):
        """The SI synthesis spec this radar's waveform and array describe."""

        return self.system_config.waveform_spec(carrier_hz=offset)

    # -- rebuilding --------------------------------------------------------

    def replace(self, **fields: Any) -> Radar:
        """A new radar with these fields changed. Nothing here is mutated."""

        unknown = sorted(set(fields) - {f.name for f in self.__dataclass_fields__.values() if f.init})
        if unknown:
            raise TypeError(f"Radar.replace got unknown fields: {', '.join(unknown)}")
        return replace(self, **fields)

    def to(self, device: Any) -> Radar:
        """A new radar on another device."""

        return self.replace(device=device)

    # -- loaders -----------------------------------------------------------

    @classmethod
    def from_dict(cls, config: Mapping[str, Any], **overrides: Any) -> Radar:
        """Build from the flat FMCW configuration file format.

        This is the only place vendor units are read: ``slope`` in MHz/us,
        ``sample_rate`` in kSPS, the three timings in microseconds, ``power`` in
        dBm, and ``tx_loc``/``rx_loc`` in half wavelengths. Keyword overrides
        are applied afterwards and use the SI field names, so a caller attaches
        a receive chain or a pose without editing the mapping.
        """

        return _radar_from_flat_config(config, overrides)

    @classmethod
    def from_json(cls, path: str | os.PathLike[str], **overrides: Any) -> Radar:
        with open(path, encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle), **overrides)

    # -- pose transforms, shared by every consumer -------------------------

    def _world_from_local_matrix(self, *, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        position = vec3_tensor(self.position, name="Radar.position").to(device=device, dtype=dtype)
        look_at = vec3_tensor(self.look_at, name="Radar.look_at").to(device=device, dtype=dtype)
        up = vec3_tensor(self.up, name="Radar.up").to(device=device, dtype=dtype)
        forward = look_at - position
        forward = forward / torch.linalg.norm(forward)
        right = torch.cross(forward, up, dim=0)
        right = right / torch.linalg.norm(right)
        true_up = torch.cross(right, forward, dim=0)
        true_up = true_up / torch.linalg.norm(true_up)
        return position, torch.stack((right, true_up, -forward), dim=1)

    def _world_from_local_points(self, points: torch.Tensor) -> torch.Tensor:
        position, world_from_local = self._world_from_local_matrix(device=points.device, dtype=points.dtype)
        return points @ world_from_local.transpose(0, 1) + position

    def _world_from_local_vectors(self, vectors: torch.Tensor) -> torch.Tensor:
        _, world_from_local = self._world_from_local_matrix(device=vectors.device, dtype=vectors.dtype)
        return vectors @ world_from_local.transpose(0, 1)

    def _local_from_world_points(self, points: torch.Tensor) -> torch.Tensor:
        position, world_from_local = self._world_from_local_matrix(device=points.device, dtype=points.dtype)
        return (points - position) @ world_from_local

    def _local_from_world_vectors(self, vectors: torch.Tensor) -> torch.Tensor:
        _, world_from_local = self._world_from_local_matrix(device=vectors.device, dtype=vectors.dtype)
        return vectors @ world_from_local

    def _evaluate_antenna_pattern_xy(self, x_angles_deg: torch.Tensor, y_angles_deg: torch.Tensor) -> torch.Tensor:
        return self.pattern.evaluate_xy(x_angles_deg, y_angles_deg)

    # -- instrument half ---------------------------------------------------

    def _apply_signal_models(self, signal: torch.Tensor, *, phase_in_signal: bool = False) -> torch.Tensor:
        """Run the receive chain, if one is configured."""

        if self.frontend is None:
            return signal
        return self.frontend.apply(signal, phase_in_signal=phase_in_signal).signal

    def _synthesize(self, paths, *, slow_time_mode, spec=None) -> SynthesisResult:
        """Synthesize one frame with whichever waveform this radar declares.

        Dispatch is a dict lookup on the STORED ``waveform.kind``. It is not a
        ``try``/``except``, not a capability probe, and not an inference from a
        ``slope``: a kind with no owner is a hard error, because a waveform
        without an owner has no physics and returning a plausible cube would be
        worse than failing.

        ``slow_time_mode`` has no default for the reason it has none anywhere
        else: only the caller knows whether it froze the weight for the frame or
        refreshes it per slot, and defaulting it makes the collision a silent
        wrong answer instead of a refusal.
        """

        from .synthesis import SynthesisPathBatch, SynthesisResult, synthesize_fmcw, synthesize_ofdm, synthesize_pulsed

        owners = {
            WAVEFORM_FMCW: (synthesize_fmcw, SynthesisResult.from_fmcw),
            WAVEFORM_OFDM: (synthesize_ofdm, SynthesisResult.from_ofdm),
            WAVEFORM_PULSED: (synthesize_pulsed, SynthesisResult.from_pulsed),
        }
        kind = self.system_config.kind
        if kind not in owners:
            raise ValueError(
                f"no synthesis owner for waveform kind {kind!r}; the supported "
                f"kinds are {sorted(owners)}. This dispatch has no fallback: a "
                "waveform without an owner has no physics."
            )
        batch = (
            paths
            if isinstance(paths, SynthesisPathBatch)
            else SynthesisPathBatch.from_radar_paths(paths, slow_time_mode=slow_time_mode)
        )
        synthesize, build_result = owners[kind]
        spec = self.system_config.waveform_spec() if spec is None else spec
        return build_result(synthesize(batch, spec), spec)

    # -- entry points ------------------------------------------------------

    def trace(
        self,
        scene,
        targets: PointTargets | StructureTargets,
        *,
        times,
        los: bool = True,
        reflections: int = 1,
        motion: Motion | None = None,
        grad: str = "none",
        endpoints=None,
    ) -> Paths:
        """Compose this radar's round trips over a Core world and keep them.

        The world half of the pipeline: sampling the world at the waveform's
        observation instants, compiling or reusing the Channel epoch,
        discovering the topology, joining the two legs through the scatter
        response, and weighting by the array's pattern. It stops there. The
        waveform, the receive chain and the output domain belong to
        :meth:`echo`, which is what lets a receiver sweep re-use one trace.

        Arguments are :meth:`simulate`'s. What differs is the retention:
        :class:`~witwin.radar.simulation.Paths` holds every evaluated
        observation's rows, so an ADC-refreshed sequence costs real device
        memory. Read that record's docstring before tracing a long one.
        """

        from .simulation import trace_scene

        session = self._session(targets, times, los, reflections, motion, grad, endpoints)
        return trace_scene(self, scene, **session)

    def echo(self, paths: Paths) -> RadarSimulationResult:
        """Run this radar's instrument half over already-composed rows.

        Synthesizes the waveform at each traced observation, applies the
        receive chain, lands the frame in the declared output domain and
        assembles the typed result. ``radar.echo(radar.trace(...))`` is
        bit-identical to ``radar.simulate(...)``.

        The radar echoing does not have to be the one that traced. Its receive
        chain, its seed and its FMCW output domain are free; anything the
        observation schedule was derived from is not, and is refused by name
        rather than replayed against a schedule that no longer describes it.
        """

        from .simulation import echo_paths

        return echo_paths(self, paths)

    def simulate(
        self,
        scene,
        targets: PointTargets | StructureTargets,
        *,
        times,
        los: bool = True,
        reflections: int = 1,
        motion: Motion | None = None,
        grad: str = "none",
        endpoints=None,
    ) -> RadarSimulationResult:
        """Simulate this radar over a Core world and return the frame cubes.

        ``scene`` is a ``witwin.core.Scene`` or a
        ``witwin.core.dynamics.DynamicScene``; ``times`` is the sequence of
        frame instants in seconds; ``targets`` names where the scatterers are
        and how strongly they scatter, and it is required because every default
        for it would be an unchosen statement about the world.

        ``los`` and ``reflections`` are the propagation request for THIS call
        and do not edit the radar. ``motion`` selects how often the world is
        resampled inside a frame and defaults to
        :meth:`~witwin.radar.simulation.Motion.auto`. ``grad`` is ``"none"``,
        ``"vjp"`` or ``"jvp"``.

        The whole assembly lives in :mod:`witwin.radar.simulation` and its
        docstring is the contract; this method delegates rather than
        reimplementing so there is one owner of the frame loop.
        """

        from .simulation import simulate_scene

        session = self._session(targets, times, los, reflections, motion, grad, endpoints)
        return simulate_scene(self, scene, **session)

    def stream(
        self,
        scene,
        targets: PointTargets | StructureTargets,
        *,
        times,
        los: bool = True,
        reflections: int = 1,
        motion: Motion | None = None,
        grad: str = "none",
        endpoints=None,
    ) -> Iterator[RadarSimulationResult]:
        """Simulate the same session as :meth:`simulate`, one frame at a time.

        Yields a one-frame result per instant in ``times``, so a sequence long
        enough to exhaust device memory as a single stacked cube can still be
        produced and consumed. The physics, the session state and the per-frame
        cubes are the same; only the retention differs, and a caller that keeps
        every yielded result has spent more memory than :meth:`simulate` would
        have, not less. Arguments are validated when iteration starts rather
        than when this returns, because this is a generator.
        """

        from .simulation import stream_scene

        session = self._session(targets, times, los, reflections, motion, grad, endpoints)
        yield from stream_scene(self, scene, **session)

    def _session(self, targets, times, los, reflections, motion, grad, endpoints) -> dict[str, Any]:
        """Turn the public call arguments into what the frame loop takes."""

        from .simulation import Motion
        from .targets import as_session_targets

        if not isinstance(reflections, int) or isinstance(reflections, bool) or reflections < 0:
            raise ValueError(f"reflections must be a non-negative int, got {reflections!r}")
        if not los and reflections == 0:
            raise ValueError(
                "a solve with neither the line of sight nor any reflection asks for no propagation at all; "
                "set los=True, reflections>0, or both"
            )
        components = set()
        if los:
            components.add("los")
        if reflections > 0:
            components.add("reflection")
        sites, response = as_session_targets(targets, radar=self)
        resolved = Motion.auto() if motion is None else motion
        return {
            "times": times,
            "response": response,
            "sites": sites,
            "components": frozenset(components),
            "max_depth": int(reflections),
            "ad_mode": grad,
            "polarization": self.polarization_vector,
            "antenna_pattern": self.system_config.sensors.pattern,
            "sensor_endpoints": endpoints,
            "motion": resolved,
        }


def _resolve_device(device: Any) -> torch.device:
    resolved = device if isinstance(device, torch.device) else torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Radar defaults to CUDA, but torch.cuda.is_available() is False. "
            "Install a CUDA-enabled PyTorch build and use device='cuda'."
        )
    return resolved


# ---------------------------------------------------------------------------
# The flat configuration file format
# ---------------------------------------------------------------------------

_FLAT_REQUIRED = (
    "num_tx",
    "num_rx",
    "fc",
    "slope",
    "adc_samples",
    "adc_start_time",
    "sample_rate",
    "idle_time",
    "ramp_end_time",
    "chirp_per_frame",
    "power",
    "tx_loc",
    "rx_loc",
)

_FLAT_OPTIONAL = ("antenna_pattern", "output_domain")

#: Keys the flat form used to require and nothing ever read. They described a
#: processing grid the simulator does not use and the processing layer derives
#: from the waveform spec, so carrying them made a caller believe a block was
#: configured. They are named in the refusal rather than dropped.
_FLAT_UNCONSUMED = ("frame_per_second", "num_doppler_bins", "num_range_bins", "num_angle_bins")


def _radar_from_flat_config(config: Mapping[str, Any], overrides: Mapping[str, Any]) -> Radar:
    missing = [key for key in _FLAT_REQUIRED if key not in config]
    if missing:
        raise ValueError(f"Radar config is missing required keys: {', '.join(missing)}")
    unconsumed = sorted(set(config) & set(_FLAT_UNCONSUMED))
    if unconsumed:
        raise ValueError(
            f"Radar config has keys nothing consumes: {', '.join(unconsumed)}. They described a processing grid; "
            "the range, Doppler and angle bin counts come from the waveform spec, and the frame rate is the "
            "caller's own scheduling number. Drop them."
        )
    unknown = sorted(set(config) - set(_FLAT_REQUIRED) - set(_FLAT_OPTIONAL))
    if unknown:
        raise ValueError(
            f"Radar config has unsupported keys: {', '.join(unknown)}. The flat mapping is the FMCW file format "
            f"and accepts only {', '.join(sorted(set(_FLAT_REQUIRED) | set(_FLAT_OPTIONAL)))}; a receive chain, a "
            "pose and a non-FMCW waveform are keyword overrides on Radar.from_dict."
        )

    num_tx = _positive_int("num_tx", config["num_tx"])
    num_rx = _positive_int("num_rx", config["num_rx"])
    tx = _elements(config["tx_loc"], name="tx_loc")
    rx = _elements(config["rx_loc"], name="rx_loc")
    if len(tx) != num_tx:
        raise ValueError(f"tx_loc holds {len(tx)} entries but num_tx is {num_tx}")
    if len(rx) != num_rx:
        raise ValueError(f"rx_loc holds {len(rx)} entries but num_rx is {num_rx}")

    waveform = Fmcw.from_ti(
        slope_mhz_per_us=_finite("slope", config["slope"]),
        sample_rate_ksps=_positive("sample_rate", config["sample_rate"]),
        samples_per_chirp=_positive_int("adc_samples", config["adc_samples"]),
        chirps_per_frame=_positive_int("chirp_per_frame", config["chirp_per_frame"]),
        adc_start_us=_finite("adc_start_time", config["adc_start_time"]),
        idle_us=_non_negative("idle_time", config["idle_time"]),
        ramp_end_us=_positive("ramp_end_time", config["ramp_end_time"]),
        output=str(config.get("output_domain", "spectrum")),
    )
    fields: dict[str, Any] = {
        "carrier": _positive("fc", config["fc"]),
        "waveform": waveform,
        "tx": tx,
        "rx": rx,
        "antenna_unit": "half_wavelength",
        "power": _finite("power", config["power"]),
    }
    if config.get("antenna_pattern") is not None:
        fields["pattern"] = _pattern_from_mapping(config["antenna_pattern"])
    fields.update(overrides)
    return Radar(**fields)


def _pattern_from_mapping(config: Mapping[str, Any]) -> Pattern:
    """Read the ``antenna_pattern`` block of the flat file format."""

    kind = config.get("kind")
    if kind is None:
        kind = "map" if "values" in config else "separable"
    if kind == "separable":
        return Pattern.separable(config["x_angles_deg"], config["x_values"], config["y_angles_deg"], config["y_values"])
    if kind == "map":
        return Pattern.table(config["x_angles_deg"], config["y_angles_deg"], config["values"])
    raise ValueError(f"antenna_pattern kind must be 'separable' or 'map', got {kind!r}")


__all__ = ["Fmcw", "Ofdm", "Pulsed", "Radar"]
