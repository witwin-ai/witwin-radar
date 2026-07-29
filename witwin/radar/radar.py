"""The radar facade: configuration, pose, antenna state and the frame entry.

``Radar.simulate`` is the production entry point and it delegates to
:mod:`witwin.radar.simulation`. This module owns no propagation and no
synthesis physics; what it holds is the configuration record, the pose
transforms every consumer shares, the antenna-pattern state, and the four typed
diagnostics of the last completed frame.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, ClassVar, Iterable

import torch

from .frontend import FrontendSpec
from .sensors import (
    AntennaPatternSpec,
    SensorArraySpec,
    TxPowerSpec,
)
from .sensors import (
    DEFAULT_DIPOLE_ANGLES_DEG,
    DEFAULT_DIPOLE_VALUES,
    evaluate_antenna_pattern_xy,
)
from .synthesis.assembly import (
    PULSE_NORMALIZATION_UNIT_ENERGY,
    SPEED_OF_LIGHT_M_PER_S,
    SUBCARRIER_ORIGIN_F_REF_AT_N0,
    FmcwSpec,
    OfdmSpec,
    PulsedSpec,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .simulation import RadarSimulationResult
    from .synthesis import SynthesisResult

WAVEFORM_FMCW = "fmcw"
WAVEFORM_OFDM = "ofdm"
WAVEFORM_PULSED = "pulsed"
WAVEFORM_KINDS = (WAVEFORM_FMCW, WAVEFORM_OFDM, WAVEFORM_PULSED)


@dataclass(frozen=True, slots=True)
class FmcwWaveformConfig:
    """One FMCW ramp and its ADC window, in the configuration's own units.

    Exactly the fields ``FmcwSpec.from_radar_config`` reads today:
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
    output_domain: str = "spectrum"

    def to_spec(
        self,
        *,
        reference_frequency_hz: float,
        num_tx: int = 1,
        num_rx: int = 1,
        carrier_hz: float = 0.0,
    ) -> FmcwSpec:
        carrier = float(carrier_hz)
        return FmcwSpec(
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
            output_domain=self.output_domain,
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
    ) -> OfdmSpec:
        carrier = float(carrier_hz)
        return OfdmSpec(
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
    ) -> PulsedSpec:
        carrier = float(carrier_hz)
        return PulsedSpec(
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
    """The array, its antenna pattern, and transmit power."""

    array: SensorArraySpec
    pattern: AntennaPatternSpec
    tx_power: TxPowerSpec


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

    def with_propagation(
        self,
        *,
        components: frozenset[str] | None = None,
        max_depth: int | None = None,
    ) -> "RadarSystemConfig":
        """A copy whose propagation block carries these two knobs.

        The scene-driven entry's ``components=`` / ``max_depth=`` keywords land
        here. It returns a new configuration rather than mutating this one
        because a per-solve override that edited the radar's stored
        configuration would silently change every LATER solve as well, and a
        propagation request is a statement about one solve.

        ``reference_frequency_hz`` is deliberately not overridable: it is tied
        to the array's element spacing by ``__post_init__`` and to the compiled
        scene by Channel, so changing it here would produce a configuration that
        is refused later rather than one that means something else.
        """

        if components is None and max_depth is None:
            return self
        current = self.propagation
        replacement = PropagationConfig(
            reference_frequency_hz=current.reference_frequency_hz,
            components=(
                current.components if components is None else frozenset(components)
            ),
            max_depth=(
                current.max_depth if max_depth is None else int(max_depth)
            ),
        )
        return replace(self, propagation=replacement)

    @classmethod
    def from_radar_config(
        cls,
        config,
        *,
        frontend: FrontendSpec | None = None,
        waveform: WaveformConfig | None = None,
        components: frozenset[str] | None = None,
        max_depth: int | None = None,
    ):
        """Split a flat ``RadarConfig`` into the five blocks.

        The flat form remains the file format and the public constructor; this
        is the structural view of it. Blocks are the thing an adapter, a
        synthesis owner, or a signal processor is handed, so that each one sees
        only what it owns.

        ``waveform`` selects the waveform block. ``None`` builds the FMCW block
        out of the flat fields, which is what this classmethod has always done
        and is bit-for-bit unchanged; anything else is used verbatim, which is
        how an OFDM or pulsed radar is configured without hand-assembling all
        five blocks. The flat fields the FMCW block would have read are simply
        not consulted in that case, because an OFDM symbol has no ramp slope.

        ``components`` and ``max_depth`` fill the propagation block. Their
        defaults are :class:`PropagationConfig`'s own, so omitting both is
        exactly the previous behaviour.
        """

        return cls(
            waveform=(
                FmcwWaveformConfig(
                    slope=float(config.slope),
                    adc_samples=int(config.adc_samples),
                    adc_start_time=float(config.adc_start_time),
                    sample_rate=float(config.sample_rate),
                    idle_time=float(config.idle_time),
                    ramp_end_time=float(config.ramp_end_time),
                    chirp_per_frame=int(config.chirp_per_frame),
                )
                if waveform is None
                else waveform
            ),
            sensors=SensorConfig(
                array=SensorArraySpec.from_radar_config(config),
                pattern=AntennaPatternSpec.from_config(config.antenna_pattern),
                tx_power=TxPowerSpec.from_radar_config(config),
            ),
            propagation=PropagationConfig(
                reference_frequency_hz=float(config.fc),
                **(
                    {}
                    if components is None
                    else {"components": frozenset(components)}
                ),
                **({} if max_depth is None else {"max_depth": int(max_depth)}),
            ),
            processing=ProcessingConfig(
                frame_per_second=float(config.frame_per_second),
                num_doppler_bins=int(config.num_doppler_bins),
                num_range_bins=int(config.num_range_bins),
                num_angle_bins=int(config.num_angle_bins),
            ),
            frontend=frontend,
        )

# ---------------------------------------------------------------------------
# Primitive validators
# ---------------------------------------------------------------------------

def _finite_float(name: str, value: Any, prefix: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{prefix} '{name}' must be a finite float.") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{prefix} '{name}' must be a finite float.")
    return parsed


def _non_negative_float(name: str, value: Any, prefix: str) -> float:
    parsed = _finite_float(name, value, prefix)
    if parsed < 0.0:
        raise ValueError(f"{prefix} '{name}' must be non-negative.")
    return parsed


def _positive_float(name: str, value: Any, prefix: str) -> float:
    parsed = _finite_float(name, value, prefix)
    if parsed <= 0.0:
        raise ValueError(f"{prefix} '{name}' must be positive.")
    return parsed


def _positive_int(name: str, value: Any, prefix: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{prefix} '{name}' must be a positive int.")
    return value


def _optional_seed(value: Any, name: str, prefix: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{prefix} '{name}' must be a non-negative int.")
    return value


def _require_keys(config: dict[str, Any], keys: Iterable[str], label: str) -> None:
    missing = [key for key in keys if key not in config]
    if missing:
        raise ValueError(f"{label} is missing required keys: {', '.join(missing)}")


def _parse_vector3(
    name: str,
    value: Any,
    *,
    prefix: str,
    aliases: dict[str, tuple[float, float, float]] | None = None,
) -> tuple[float, float, float]:
    if isinstance(value, str):
        if aliases is None or value.lower() not in aliases:
            raise ValueError(
                f"{prefix} '{name}' must be a 3-element vector"
                + (" or an alias string." if aliases else ".")
            )
        return aliases[value.lower()]
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{prefix} '{name}' must be a 3-element vector.")
    vector = tuple(
        _finite_float(f"{name}[{i}]", component, prefix)
        for i, component in enumerate(value)
    )
    norm_sq = sum(c * c for c in vector)
    if norm_sq <= 1e-24:
        raise ValueError(f"{prefix} '{name}' must be non-zero.")
    return vector


# ---------------------------------------------------------------------------
# Antenna pattern
# ---------------------------------------------------------------------------

_ANTENNA_PREFIX = "Antenna pattern field"


def _validate_axis(name: str, value: Any) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{_ANTENNA_PREFIX} '{name}' must be a sequence of angles in degrees.")
    if len(value) < 2:
        raise ValueError(f"{_ANTENNA_PREFIX} '{name}' must contain at least 2 samples.")
    axis = tuple(_finite_float(f"{name}[{i}]", angle, _ANTENNA_PREFIX) for i, angle in enumerate(value))
    for i in range(1, len(axis)):
        if axis[i] <= axis[i - 1]:
            raise ValueError(f"{_ANTENNA_PREFIX} '{name}' must be strictly increasing.")
    return axis


def _validate_values_1d(name: str, value: Any, expected_count: int) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{_ANTENNA_PREFIX} '{name}' must be a sequence of gain values.")
    if len(value) != expected_count:
        raise ValueError(
            f"{_ANTENNA_PREFIX} '{name}' must contain exactly {expected_count} entries; got {len(value)}."
        )
    gains = []
    for i, item in enumerate(value):
        gain = _finite_float(f"{name}[{i}]", item, _ANTENNA_PREFIX)
        if gain < 0.0:
            raise ValueError(f"{_ANTENNA_PREFIX} '{name}[{i}]' must be non-negative.")
        gains.append(gain)
    return tuple(gains)


def _validate_values_2d(
    name: str, value: Any, expected_rows: int, expected_cols: int
) -> tuple[tuple[float, ...], ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{_ANTENNA_PREFIX} '{name}' must be a 2D sequence of gain values.")
    if len(value) != expected_rows:
        raise ValueError(
            f"{_ANTENNA_PREFIX} '{name}' must contain exactly {expected_rows} rows; got {len(value)}."
        )
    rows = []
    for row_index, row in enumerate(value):
        if not isinstance(row, (list, tuple)):
            raise ValueError(f"{_ANTENNA_PREFIX} '{name}[{row_index}]' must be a sequence of gain values.")
        if len(row) != expected_cols:
            raise ValueError(
                f"{_ANTENNA_PREFIX} '{name}[{row_index}]' must contain exactly {expected_cols} entries; got {len(row)}."
            )
        parsed_row = []
        for col_index, item in enumerate(row):
            gain = _finite_float(f"{name}[{row_index}][{col_index}]", item, _ANTENNA_PREFIX)
            if gain < 0.0:
                raise ValueError(
                    f"{_ANTENNA_PREFIX} '{name}[{row_index}][{col_index}]' must be non-negative."
                )
            parsed_row.append(gain)
        rows.append(tuple(parsed_row))
    return tuple(rows)


def _detect_antenna_kind(data: dict[str, Any]) -> str:
    raw_kind = data.get("kind")
    if raw_kind is None:
        if "values" in data:
            return "map"
        if "x_values" in data or "y_values" in data:
            return "separable"
        raise ValueError("Antenna pattern config must define 'kind' or provide fields for a known pattern type.")
    kind = str(raw_kind)
    if kind not in {"separable", "map"}:
        raise ValueError("Antenna pattern field 'kind' must be 'separable' or 'map'.")
    return kind


def validate_antenna_pattern_config(config: dict[str, Any]) -> dict[str, Any]:
    kind = _detect_antenna_kind(config)
    x_angles_deg = _validate_axis("x_angles_deg", config.get("x_angles_deg"))
    y_angles_deg = _validate_axis("y_angles_deg", config.get("y_angles_deg"))

    if kind == "separable":
        x_values = _validate_values_1d("x_values", config.get("x_values"), len(x_angles_deg))
        y_values = _validate_values_1d("y_values", config.get("y_values"), len(y_angles_deg))
        return {
            "kind": kind,
            "x_angles_deg": list(x_angles_deg),
            "y_angles_deg": list(y_angles_deg),
            "x_values": list(x_values),
            "y_values": list(y_values),
        }

    values = _validate_values_2d("values", config.get("values"), len(y_angles_deg), len(x_angles_deg))
    return {
        "kind": kind,
        "x_angles_deg": list(x_angles_deg),
        "y_angles_deg": list(y_angles_deg),
        "values": [list(row) for row in values],
    }


def default_dipole_antenna_pattern() -> dict[str, Any]:
    return {
        "kind": "separable",
        "x_angles_deg": list(DEFAULT_DIPOLE_ANGLES_DEG),
        "y_angles_deg": list(DEFAULT_DIPOLE_ANGLES_DEG),
        "x_values": list(DEFAULT_DIPOLE_VALUES),
        "y_values": list(DEFAULT_DIPOLE_VALUES),
    }


# ---------------------------------------------------------------------------
# Polarization
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Radar config
# ---------------------------------------------------------------------------

_RADAR_PREFIX = "Radar config field"

_RADAR_REQUIRED_KEYS = (
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
    "frame_per_second",
    "num_doppler_bins",
    "num_range_bins",
    "num_angle_bins",
    "power",
    "tx_loc",
    "rx_loc",
)


_RADAR_OPTIONAL_KEYS = ("antenna_pattern",)

#: What the flat mapping can express. A key outside this set is refused rather
#: than dropped: the flat form is the file format, so an unknown key is a
#: caller who believes a block is configured. ``"waveform"`` and ``"frontend"``
#: are the two that cost real time - a caller who writes
#: ``{"waveform": "ofdm"}`` used to get an FMCW radar with nothing raised, and
#: ``{"frontend": {...}}`` used to get a radar with no receive chain. Neither
#: block is authorable here today (see the migration note); refusing says so.
_RADAR_KNOWN_KEYS = frozenset(_RADAR_REQUIRED_KEYS + _RADAR_OPTIONAL_KEYS)


def _reject_unknown_radar_keys(config: dict[str, Any]) -> None:
    unknown = sorted(set(config) - _RADAR_KNOWN_KEYS)
    if unknown:
        raise ValueError(
            f"Radar config has unsupported keys: {', '.join(unknown)}. The flat "
            f"mapping accepts only {', '.join(sorted(_RADAR_KNOWN_KEYS))}; a "
            "waveform other than FMCW and a frontend chain are not authorable "
            "in it, so attach them to the RadarConfig after validation."
        )


def _validate_antenna_locations(
    name: str, value: Any, expected_count: int
) -> tuple[tuple[float, float, float], ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{_RADAR_PREFIX} '{name}' must be a sequence of 3D coordinates.")
    if len(value) != expected_count:
        raise ValueError(
            f"{_RADAR_PREFIX} '{name}' must contain exactly {expected_count} entries; got {len(value)}."
        )
    coords: list[tuple[float, float, float]] = []
    for index, coord in enumerate(value):
        if not isinstance(coord, (list, tuple)) or len(coord) != 3:
            raise ValueError(f"{_RADAR_PREFIX} '{name}[{index}]' must be a 3-element coordinate.")
        coords.append(
            (
                _finite_float(f"{name}[{index}][0]", coord[0], _RADAR_PREFIX),
                _finite_float(f"{name}[{index}][1]", coord[1], _RADAR_PREFIX),
                _finite_float(f"{name}[{index}][2]", coord[2], _RADAR_PREFIX),
            )
        )
    return tuple(coords)


def validate_radar_config(config: dict[str, Any]) -> RadarConfig:
    _require_keys(config, _RADAR_REQUIRED_KEYS, "Radar config")
    _reject_unknown_radar_keys(config)

    num_tx = _positive_int("num_tx", config["num_tx"], _RADAR_PREFIX)
    num_rx = _positive_int("num_rx", config["num_rx"], _RADAR_PREFIX)

    antenna_pattern = (
        validate_antenna_pattern_config(config["antenna_pattern"])
        if config.get("antenna_pattern") is not None
        else None
    )
    return RadarConfig(
        num_tx=num_tx,
        num_rx=num_rx,
        fc=_finite_float("fc", config["fc"], _RADAR_PREFIX),
        slope=_finite_float("slope", config["slope"], _RADAR_PREFIX),
        adc_samples=_positive_int("adc_samples", config["adc_samples"], _RADAR_PREFIX),
        adc_start_time=_finite_float("adc_start_time", config["adc_start_time"], _RADAR_PREFIX),
        sample_rate=_finite_float("sample_rate", config["sample_rate"], _RADAR_PREFIX),
        idle_time=_finite_float("idle_time", config["idle_time"], _RADAR_PREFIX),
        ramp_end_time=_finite_float("ramp_end_time", config["ramp_end_time"], _RADAR_PREFIX),
        chirp_per_frame=_positive_int("chirp_per_frame", config["chirp_per_frame"], _RADAR_PREFIX),
        frame_per_second=_finite_float("frame_per_second", config["frame_per_second"], _RADAR_PREFIX),
        num_doppler_bins=_positive_int("num_doppler_bins", config["num_doppler_bins"], _RADAR_PREFIX),
        num_range_bins=_positive_int("num_range_bins", config["num_range_bins"], _RADAR_PREFIX),
        num_angle_bins=_positive_int("num_angle_bins", config["num_angle_bins"], _RADAR_PREFIX),
        power=_finite_float("power", config["power"], _RADAR_PREFIX),
        tx_loc=_validate_antenna_locations("tx_loc", config["tx_loc"], num_tx),
        rx_loc=_validate_antenna_locations("rx_loc", config["rx_loc"], num_rx),
        antenna_pattern=antenna_pattern,
    )


# ---------------------------------------------------------------------------
# Block configuration (work item 6)
#
# One validator per block, one ``_REQUIRED`` tuple per block. The flat
# ``validate_radar_config`` above stays the file format; these build the
# structural view of it, and they exist separately because a block is exactly
# the unit a consumer is handed. ``PropagationConfig`` is the only block a
# propagation adapter ever receives, which is what makes a waveform field
# reaching a propagation request impossible to write rather than merely
# discouraged.
# ---------------------------------------------------------------------------

_WAVEFORM_PREFIX = "Waveform config field"
_SENSOR_PREFIX = "Sensor config field"
_FRONTEND_PREFIX = "Frontend config field"
_PROPAGATION_PREFIX = "Propagation config field"
_PROCESSING_PREFIX = "Processing config field"

_FMCW_REQUIRED = (
    "slope",
    "adc_samples",
    "adc_start_time",
    "sample_rate",
    "idle_time",
    "ramp_end_time",
    "chirp_per_frame",
)
_OFDM_REQUIRED = (
    "subcarrier_spacing_hz",
    "num_subcarriers",
    "cyclic_prefix_s",
    "num_symbols",
    "max_expected_delay_s",
)
_PULSED_REQUIRED = (
    "pulse_kind",
    "pulse_width_s",
    "bandwidth_hz",
    "pri_s",
    "num_pulses",
    "sample_rate_hz",
    "num_samples",
    "range_gate_start_s",
)
_PROPAGATION_REQUIRED = ("reference_frequency_hz",)
_PROCESSING_REQUIRED = (
    "frame_per_second",
    "num_doppler_bins",
    "num_range_bins",
    "num_angle_bins",
)
_SENSOR_REQUIRED = ("num_tx", "num_rx", "fc", "tx_loc", "rx_loc", "power")


def validate_waveform_config(config: dict[str, Any]):
    """Build one waveform block from a mapping with a STORED ``kind``.

    The kind is read, never inferred. Inferring "this is FMCW" from the
    presence of a ``slope`` is exactly the habit that lets waveform vocabulary
    leak into places with no business knowing the waveform, so an absent or
    unknown ``kind`` is an error rather than a guess.
    """

    kind = config.get("kind")
    if kind not in WAVEFORM_KINDS:
        raise ValueError(
            f"{_WAVEFORM_PREFIX} 'kind' must be one of {list(WAVEFORM_KINDS)}; "
            f"got {kind!r}. The waveform discriminator is stored, never inferred "
            "from the presence of a slope or a subcarrier spacing."
        )
    if kind == WAVEFORM_FMCW:
        _require_keys(config, _FMCW_REQUIRED, "FMCW waveform config")
        return FmcwWaveformConfig(
            slope=_finite_float("slope", config["slope"], _WAVEFORM_PREFIX),
            adc_samples=_positive_int(
                "adc_samples", config["adc_samples"], _WAVEFORM_PREFIX
            ),
            adc_start_time=_finite_float(
                "adc_start_time", config["adc_start_time"], _WAVEFORM_PREFIX
            ),
            sample_rate=_positive_float(
                "sample_rate", config["sample_rate"], _WAVEFORM_PREFIX
            ),
            idle_time=_non_negative_float(
                "idle_time", config["idle_time"], _WAVEFORM_PREFIX
            ),
            ramp_end_time=_positive_float(
                "ramp_end_time", config["ramp_end_time"], _WAVEFORM_PREFIX
            ),
            chirp_per_frame=_positive_int(
                "chirp_per_frame", config["chirp_per_frame"], _WAVEFORM_PREFIX
            ),
            output_domain=str(config.get("output_domain", "spectrum")),
        )
    if kind == WAVEFORM_OFDM:
        _require_keys(config, _OFDM_REQUIRED, "OFDM waveform config")
        return OfdmWaveformConfig(
            subcarrier_spacing_hz=_positive_float(
                "subcarrier_spacing_hz",
                config["subcarrier_spacing_hz"],
                _WAVEFORM_PREFIX,
            ),
            num_subcarriers=_positive_int(
                "num_subcarriers", config["num_subcarriers"], _WAVEFORM_PREFIX
            ),
            cyclic_prefix_s=_positive_float(
                "cyclic_prefix_s", config["cyclic_prefix_s"], _WAVEFORM_PREFIX
            ),
            num_symbols=_positive_int(
                "num_symbols", config["num_symbols"], _WAVEFORM_PREFIX
            ),
            max_expected_delay_s=_non_negative_float(
                "max_expected_delay_s", config["max_expected_delay_s"], _WAVEFORM_PREFIX
            ),
        )
    _require_keys(config, _PULSED_REQUIRED, "Pulsed waveform config")
    return PulsedWaveformConfig(
        pulse_kind=str(config["pulse_kind"]),
        pulse_width_s=_positive_float(
            "pulse_width_s", config["pulse_width_s"], _WAVEFORM_PREFIX
        ),
        bandwidth_hz=_positive_float(
            "bandwidth_hz", config["bandwidth_hz"], _WAVEFORM_PREFIX
        ),
        pri_s=_positive_float("pri_s", config["pri_s"], _WAVEFORM_PREFIX),
        num_pulses=_positive_int("num_pulses", config["num_pulses"], _WAVEFORM_PREFIX),
        sample_rate_hz=_positive_float(
            "sample_rate_hz", config["sample_rate_hz"], _WAVEFORM_PREFIX
        ),
        num_samples=_positive_int(
            "num_samples", config["num_samples"], _WAVEFORM_PREFIX
        ),
        range_gate_start_s=_non_negative_float(
            "range_gate_start_s", config["range_gate_start_s"], _WAVEFORM_PREFIX
        ),
        max_expected_delay_rate=_non_negative_float(
            "max_expected_delay_rate",
            config.get("max_expected_delay_rate", 0.0),
            _WAVEFORM_PREFIX,
        ),
    )


def validate_sensor_config(config: dict[str, Any]):
    """Build the sensor block: array, pattern, and transmit power.

    ``power`` is in dBm and becomes ``powers_w`` on a source endpoint. There is
    deliberately no transmit-gain output here: a Channel coefficient already
    carries ``sqrt(P_tx)``, so a second one would count the power twice and mix
    sqrt(W) with sqrt(W ohm).
    """
    from .sensors import (
        AntennaPatternSpec,
        SensorArraySpec,
        TxPowerSpec,
    )

    _require_keys(config, _SENSOR_REQUIRED, "Sensor config")
    num_tx = _positive_int("num_tx", config["num_tx"], _SENSOR_PREFIX)
    num_rx = _positive_int("num_rx", config["num_rx"], _SENSOR_PREFIX)
    pattern = (
        validate_antenna_pattern_config(config["antenna_pattern"])
        if config.get("antenna_pattern") is not None
        else None
    )
    return SensorConfig(
        array=SensorArraySpec(
            num_tx=num_tx,
            num_rx=num_rx,
            tx_loc=tuple(_validate_antenna_locations("tx_loc", config["tx_loc"], num_tx)),
            rx_loc=tuple(_validate_antenna_locations("rx_loc", config["rx_loc"], num_rx)),
            reference_frequency_hz=_positive_float("fc", config["fc"], _SENSOR_PREFIX),
        ),
        pattern=AntennaPatternSpec.from_config(pattern),
        tx_power=TxPowerSpec(
            power_dbm=_finite_float("power", config["power"], _SENSOR_PREFIX)
        ),
    )


def validate_propagation_config(config: dict[str, Any]):
    """Build the propagation block, which is the ONLY block an adapter sees."""

    _require_keys(config, _PROPAGATION_REQUIRED, "Propagation config")
    components = config.get("components", ("los", "reflection"))
    if isinstance(components, str):
        raise ValueError(
            f"{_PROPAGATION_PREFIX} 'components' must be a collection of "
            "component names, not a single string"
        )
    return PropagationConfig(
        reference_frequency_hz=_positive_float(
            "reference_frequency_hz",
            config["reference_frequency_hz"],
            _PROPAGATION_PREFIX,
        ),
        components=frozenset(str(name) for name in components),
        max_depth=_positive_int(
            "max_depth", config.get("max_depth", 1), _PROPAGATION_PREFIX
        ),
    )


def validate_processing_config(config: dict[str, Any]):
    """Build the processing block: frame rate and the three bin counts."""

    _require_keys(config, _PROCESSING_REQUIRED, "Processing config")
    return ProcessingConfig(
        frame_per_second=_positive_float(
            "frame_per_second", config["frame_per_second"], _PROCESSING_PREFIX
        ),
        num_doppler_bins=_positive_int(
            "num_doppler_bins", config["num_doppler_bins"], _PROCESSING_PREFIX
        ),
        num_range_bins=_positive_int(
            "num_range_bins", config["num_range_bins"], _PROCESSING_PREFIX
        ),
        num_angle_bins=_positive_int(
            "num_angle_bins", config["num_angle_bins"], _PROCESSING_PREFIX
        ),
    )


def validate_frontend_config(config: dict[str, Any]):
    """Build the receive chain from a mapping. One chain, one ADC, one seed.

    There is deliberately no way to say what order the stages run in. The order
    is a property of the runtime, and the two runtimes this replaces left it to
    whichever caller happened to compose them, which is a difference of
    ``g_lna^2`` in output noise power.

    ``bandwidth_hz`` is required whenever thermal noise is configured and is
    never inferred from a waveform. It is the ADC sample rate for FMCW, the
    matched-filter bandwidth for pulsed, and the subcarrier spacing (or the
    whole occupied band) for OFDM, and inferring it in three places is how those
    three quietly disagree.
    """

    from .frontend import (
        AdcSpec,
        AgcSpec,
        FrontendSpec,
        LnaSpec,
        NoiseSpec,
        PortSpec,
        SeedSpec,
    )

    allowed = {"port", "noise", "lna", "agc", "adc", "seed"}
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise TypeError(f"Unsupported frontend config keys: {', '.join(unknown)}")

    port_config = config.get("port") or {}
    port = PortSpec(
        reference_impedance_ohm=_positive_float(
            "port.reference_impedance_ohm",
            port_config.get("reference_impedance_ohm", 50.0),
            _FRONTEND_PREFIX,
        )
    )

    noise = None
    if config.get("noise") is not None:
        raw = config["noise"]
        thermal = (
            raw.get("noise_figure_db") is not None or raw.get("bandwidth_hz") is not None
        )
        if thermal and raw.get("bandwidth_hz") is None:
            raise ValueError(
                f"{_FRONTEND_PREFIX} 'noise.bandwidth_hz' is required when thermal "
                "noise is configured; it is a per-waveform quantity and inferring "
                "it is a pure SNR scale error"
            )
        noise = NoiseSpec(
            noise_figure_db=_non_negative_float(
                "noise.noise_figure_db",
                raw.get("noise_figure_db", 0.0),
                _FRONTEND_PREFIX,
            ),
            antenna_temperature_k=_non_negative_float(
                "noise.antenna_temperature_k",
                raw.get("antenna_temperature_k", 290.0),
                _FRONTEND_PREFIX,
            ),
            bandwidth_hz=_non_negative_float(
                "noise.bandwidth_hz", raw.get("bandwidth_hz", 0.0), _FRONTEND_PREFIX
            ),
            phase_noise_dbc_per_hz=(
                None
                if raw.get("phase_noise_dbc_per_hz") is None
                else _finite_float(
                    "noise.phase_noise_dbc_per_hz",
                    raw["phase_noise_dbc_per_hz"],
                    _FRONTEND_PREFIX,
                )
            ),
            phase_offset_hz=_non_negative_float(
                "noise.phase_offset_hz",
                raw.get("phase_offset_hz", 0.0),
                _FRONTEND_PREFIX,
            ),
            phase_sample_rate_hz=_non_negative_float(
                "noise.phase_sample_rate_hz",
                raw.get("phase_sample_rate_hz", 0.0),
                _FRONTEND_PREFIX,
            ),
        )

    lna = None
    if config.get("lna") is not None:
        lna = LnaSpec(
            gain_db=_finite_float(
                "lna.gain_db", config["lna"].get("gain_db", 0.0), _FRONTEND_PREFIX
            )
        )

    agc = None
    if config.get("agc") is not None:
        raw = config["agc"]
        agc = AgcSpec(
            target_rms=_positive_float(
                "agc.target_rms", raw.get("target_rms"), _FRONTEND_PREFIX
            ),
            mode=str(raw.get("mode", "per_rx")).lower(),
            min_gain_db=_finite_float(
                "agc.min_gain_db", raw.get("min_gain_db", -60.0), _FRONTEND_PREFIX
            ),
            max_gain_db=_finite_float(
                "agc.max_gain_db", raw.get("max_gain_db", 60.0), _FRONTEND_PREFIX
            ),
        )

    adc = None
    if config.get("adc") is not None:
        raw = config["adc"]
        adc = AdcSpec(
            bits=_positive_int("adc.bits", raw.get("bits"), _FRONTEND_PREFIX),
            full_scale=_positive_float(
                "adc.full_scale", raw.get("full_scale", 1.0), _FRONTEND_PREFIX
            ),
        )

    seed = SeedSpec(
        seed_base=_optional_seed(config.get("seed"), "seed", _FRONTEND_PREFIX) or 0
    )
    return FrontendSpec(port=port, noise=noise, lna=lna, agc=agc, adc=adc, seed=seed)


def validate_radar_system_config(config: dict[str, Any]):
    """Build all five blocks from a block-shaped mapping."""

    _require_keys(
        config,
        ("waveform", "sensors", "propagation", "processing"),
        "Radar system config",
    )
    return RadarSystemConfig(
        waveform=validate_waveform_config(config["waveform"]),
        sensors=validate_sensor_config(config["sensors"]),
        propagation=validate_propagation_config(config["propagation"]),
        processing=validate_processing_config(config["processing"]),
        frontend=(
            validate_frontend_config(config["frontend"])
            if config.get("frontend") is not None
            else None
        ),
    )

def vec3_tensor(value, *, name: str) -> torch.Tensor:
    """Coerce to a CPU float32 tensor of shape (3,)."""
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    else:
        tensor = torch.tensor(tuple(float(component) for component in value), dtype=torch.float32)
    if tensor.shape != (3,):
        raise ValueError(f"{name} must contain exactly three values.")
    return tensor

@dataclass(frozen=True)
class RadarConfig:
    num_tx: int
    num_rx: int
    fc: float
    slope: float
    adc_samples: int
    adc_start_time: float
    sample_rate: float
    idle_time: float
    ramp_end_time: float
    chirp_per_frame: int
    frame_per_second: float
    num_doppler_bins: int
    num_range_bins: int
    num_angle_bins: int
    power: float
    tx_loc: tuple[tuple[float, float, float], ...]
    rx_loc: tuple[tuple[float, float, float], ...]
    antenna_pattern: dict[str, Any] | None = None
    #: The receive chain: ONE ordered chain with ONE ADC and ONE seed base. It
    #: replaced a ``noise_model`` / ``receiver_chain`` pair whose composite
    #: order was the caller's to choose, and since Phase 11 it is the only one
    #: - the pair is deleted, so there is no configuration in which two chains
    #: can disagree about where the LNA sits. It is ``None`` by default: noise
    #: is optional and OFF unless a caller asks for it, and every physics test
    #: runs without it.
    frontend: "FrontendSpec | None" = None

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "RadarConfig":
        return validate_radar_config(config)

    @classmethod
    def from_json(cls, path: str | os.PathLike[str]) -> "RadarConfig":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


def _target_from_position(position: torch.Tensor) -> torch.Tensor:
    return position + torch.tensor((0.0, 0.0, -1.0), dtype=torch.float32)


# `quantize_complex_signal`, `db_to_voltage_gain`, `ReceiverChainRuntime`,
# `NoiseModelRuntime` and `PolarizationRuntime` stood here until Phase 11. The
# first four were the legacy receive chain that `frontend/FrontendChain`
# replaced, and `apply_signal_models` chose between the two owners at runtime -
# a shadow mode, which acceptance criterion 6 forbids. `PolarizationRuntime`
# went with them: its only consumer outside this file was
# `sensors/legacy_paths.py`, on the deleted Dirichlet route.


class Radar:
    #: The one diagnostic retention site, as a CLASS attribute so that the four
    #: ``last_*`` properties answer ``None`` on an instance that has never run -
    #: including one built by ``object.__new__`` for a refusal test - instead of
    #: raising ``AttributeError`` from a half-initialized object.
    _last_result = None

    def __init__(
        self,
        config: RadarConfig | Mapping[str, Any],
        device: str | torch.device = "cuda",
        *,
        position=(0.0, 0.0, 0.0),
        target=None,
        up=(0.0, 1.0, 0.0),
        fov: float = 60.0,
        name: str | None = None,
    ):
        """
        Args:
            config: ``RadarConfig`` or a raw mapping accepted by ``RadarConfig.from_dict``.
            device: CUDA compute device
            position: radar origin in world coordinates
            target: look-at target in world coordinates. Defaults to one meter along -Z from position.
            up: world-space up vector
            fov: perspective field of view in degrees
            name: optional identifier for this radar
        """
        self.c0 = 299792458
        self.device: torch.device = self._resolve_device(device=torch.device(device))
        self.name = None if name is None else str(name)
        self._set_pose_fields(position=position, target=target, up=up, fov=fov)

        self.config: RadarConfig = config if isinstance(config, RadarConfig) else RadarConfig.from_dict(config)
        cfg = self.config

        self._init_system_config(cfg)
        self._init_antenna_locations(cfg)
        self._init_runtime_models(cfg)

    def _init_system_config(self, cfg: RadarConfig) -> None:
        """The five-block structural view of the flat configuration.

        The flat form stays the file format and the public constructor; this is
        what an adapter, a synthesis owner, or a signal processor is handed, so
        each one sees only the block it owns. ``waveform.kind`` is a STORED
        discriminator: nothing downstream infers "this is FMCW" by finding a
        ``slope``.
        """

        self.system_config = RadarSystemConfig.from_radar_config(
            cfg, frontend=cfg.frontend
        )

    def _init_antenna_locations(self, cfg: RadarConfig) -> None:
        self._lambda = self.c0 / cfg.fc
        antenna_spacing = self.c0 / cfg.fc / 2
        self.tx_loc = torch.tensor(cfg.tx_loc, dtype=torch.float32, device=self.device) * antenna_spacing
        self.rx_loc = torch.tensor(cfg.rx_loc, dtype=torch.float32, device=self.device) * antenna_spacing
        self._refresh_pose_dependent_state()

    def _init_runtime_models(self, cfg: RadarConfig) -> None:
        self.antenna_pattern_config = cfg.antenna_pattern or default_dipole_antenna_pattern()
        self._build_antenna_pattern_runtime(self.antenna_pattern_config)
        self.frontend = self._make_frontend(cfg)

    @staticmethod
    def _make_frontend(cfg: RadarConfig):
        if cfg.frontend is None:
            return None
        from .frontend import FrontendChain

        return FrontendChain(cfg.frontend)

    @staticmethod
    def _resolve_device(*, device: torch.device) -> torch.device:
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "Radar defaults to CUDA, but torch.cuda.is_available() is False. "
                "Install a CUDA-enabled PyTorch build and use device='cuda'."
            )
        return device

    def _set_pose_fields(self, *, position, target, up, fov) -> None:
        position_t = vec3_tensor(position, name="Radar.position")
        target_t = _target_from_position(position_t) if target is None else vec3_tensor(target, name="Radar.target")
        up_t = vec3_tensor(up, name="Radar.up")
        forward = target_t - position_t
        if torch.linalg.norm(forward) <= 1e-12:
            raise ValueError("Radar.target must differ from Radar.position.")
        if torch.linalg.norm(up_t) <= 1e-12:
            raise ValueError("Radar.up must be non-zero.")
        if torch.linalg.norm(torch.cross(forward, up_t, dim=0)) <= 1e-12:
            raise ValueError("Radar.up must not be collinear with the viewing direction.")
        self.position = position_t
        self.target = target_t
        self.up = up_t
        self.fov = float(fov)

    def _refresh_pose_dependent_state(self) -> None:
        self.tx_pos = self._world_from_local_points(self.tx_loc).contiguous()
        self.rx_pos = self._world_from_local_points(self.rx_loc).contiguous()
        self.origin = self.position

    def _build_antenna_pattern_runtime(self, config: dict[str, Any]) -> None:
        self.antenna_pattern_kind = config["kind"]
        self.antenna_pattern_x_angles_deg = torch.tensor(config["x_angles_deg"], dtype=torch.float32, device=self.device)
        self.antenna_pattern_y_angles_deg = torch.tensor(config["y_angles_deg"], dtype=torch.float32, device=self.device)
        self.antenna_pattern_x_values = None
        self.antenna_pattern_y_values = None
        self.antenna_pattern_values = None
        if config["kind"] == "separable":
            self.antenna_pattern_x_values = torch.tensor(config["x_values"], dtype=torch.float32, device=self.device)
            self.antenna_pattern_y_values = torch.tensor(config["y_values"], dtype=torch.float32, device=self.device)
        else:
            self.antenna_pattern_values = torch.tensor(config["values"], dtype=torch.float32, device=self.device)

    def _evaluate_antenna_pattern_xy(self, x_angles_deg: torch.Tensor, y_angles_deg: torch.Tensor) -> torch.Tensor:
        return evaluate_antenna_pattern_xy(
            self.antenna_pattern_kind,
            self.antenna_pattern_x_angles_deg,
            self.antenna_pattern_y_angles_deg,
            self.antenna_pattern_x_values,
            self.antenna_pattern_y_values,
            self.antenna_pattern_values,
            x_angles_deg,
            y_angles_deg,
        )

    def set_pose(self, *, position=None, target=None, up=None, fov=None) -> "Radar":
        """Mutate radar pose and refresh pose-dependent antenna state."""
        new_position = self.position if position is None else vec3_tensor(position, name="Radar.position")
        if target is None:
            target_t = self.target if position is None else new_position + (self.target - self.position)
        else:
            target_t = vec3_tensor(target, name="Radar.target")
        up_t = self.up if up is None else vec3_tensor(up, name="Radar.up")
        fov_value = self.fov if fov is None else float(fov)
        self._set_pose_fields(position=new_position, target=target_t, up=up_t, fov=fov_value)
        self._refresh_pose_dependent_state()
        return self

    def _world_from_local_matrix(self, *, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        position = self.position.to(device=device, dtype=dtype)
        target = self.target.to(device=device, dtype=dtype)
        up = self.up.to(device=device, dtype=dtype)

        forward = target - position
        forward = forward / torch.linalg.norm(forward)
        right = torch.cross(forward, up, dim=0)
        right = right / torch.linalg.norm(right)
        true_up = torch.cross(right, forward, dim=0)
        true_up = true_up / torch.linalg.norm(true_up)
        back = -forward
        world_from_local = torch.stack((right, true_up, back), dim=1)
        return position, world_from_local

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

    def _apply_signal_models(self, signal: torch.Tensor) -> torch.Tensor:
        """Run the receive chain, if one is configured.

        This used to CHOOSE between two owners: the frontend block, or the
        legacy `
oise_model`` / ``receiver_chain`` pair, with a constructor
        refusal for the configuration that named both. A refusal is not the
        same as having one owner, and a runtime choice between two chains is
        the shadow mode acceptance criterion 6 forbids. The pair is deleted, so
        the only question left is whether a chain exists.
        """

        if self.frontend is None:
            return signal
        return self.frontend.apply(signal).signal

    def _synthesize(self, paths, *, slow_time_mode) -> "SynthesisResult":
        """Synthesize one frame with whichever waveform this radar declares.

        Dispatch is a dict lookup on the STORED ``waveform.kind``. It is not a
        ``try``/``except``, not a capability probe, and not an inference from a
        ``slope``: a kind with no owner is a hard error, because a waveform
        without an owner has no physics and returning a plausible cube would be
        worse than failing.

        ``paths`` may be a composed :class:`~witwin.radar.paths.RadarPathBatch`
        or an already-wrapped
        :class:`~witwin.radar.synthesis.SynthesisPathBatch`. ``slow_time_mode``
        has no default for the reason it has none anywhere else: only the caller
        knows whether it froze the weight for the frame or refreshes it per
        slot, and defaulting it makes the Phase-7 collision a silent wrong
        answer instead of a refusal.
        """
        from .synthesis import (
            SynthesisPathBatch,
            SynthesisResult,
            synthesize_fmcw,
            synthesize_ofdm,
            synthesize_pulsed,
        )

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
            else SynthesisPathBatch.from_radar_paths(
                paths, slow_time_mode=slow_time_mode
            )
        )
        synthesize, build_result = owners[kind]
        spec = self.system_config.waveform_spec()
        return build_result(synthesize(batch, spec), spec)

    def simulate(
        self,
        scene,
        *,
        times,
        response,
        sites=None,
        components=None,
        max_depth=None,
        ad_mode: str = "none",
        world_motion: str = "frozen_world",
        motion_event_period_frames: int | None = None,
        ids=None,
        polarization=None,
    ) -> "RadarSimulationResult":
        """Simulate this radar over a Core world and return the frame cubes.

        The scene-driven entry point. ``scene`` is a ``witwin.core.Scene`` or a
        ``witwin.core.dynamics.DynamicScene``; ``times`` is the sequence of
        frame instants in seconds; ``response`` is the scatter response the
        two-way join multiplies the round trip by, and it is required because
        every default for it would be an unchosen statement about how strongly
        the target scatters.

        The whole assembly lives in :mod:`witwin.radar.simulation` and its
        docstring is the contract; read it before changing anything here. This
        method exists so that the pipeline is reachable under the name a caller
        looks for, and it delegates rather than reimplementing so there is one
        owner of the frame loop.

        Calling this publishes the four typed diagnostics
        (:attr:`last_snapshot`, :attr:`last_compiled_scene`,
        :attr:`last_propagation`, :attr:`last_radar_paths`). They are cleared
        FIRST, so a call that raises part way through leaves no stale world
        behind claiming to describe this radar.

        Antenna pattern weighting is owned by the stored sensor configuration and
        is applied by the native ``sensor_weight`` family for every solve.

        """

        from .simulation import simulate_scene

        self._last_result = None
        result = simulate_scene(
            self,
            scene,
            times=times,
            response=response,
            sites=sites,
            components=components,
            max_depth=max_depth,
            slow_time_mode=None,
            ad_mode=ad_mode,
            world_motion=world_motion,
            motion_event_period_frames=motion_event_period_frames,
            ids=ids,
            polarization=polarization,
            antenna_pattern=self.system_config.sensors.pattern,
        )
        self._last_result = result
        return result

    # -- the four typed diagnostics (Phase 11 work item 2) ------------------
    #
    # One retention site, four reads of it. The alternative - four independent
    # attributes - can be left describing four different frames by any code
    # path that sets three of them, and "which frame is this" is exactly the
    # question a diagnostic exists to answer. ``None`` before the first
    # ``simulate`` is the pinned answer: a caller may poll these, and raising
    # would make "has this radar run yet" a try/except.

    @property
    def last_result(self) -> "RadarSimulationResult | None":
        """The whole of the last :meth:`simulate` call, or ``None``."""

        return self._last_result

    @property
    def last_snapshot(self):
        """The Core ``SceneSnapshot`` the last simulated frame ran against."""

        return None if self._last_result is None else self._last_result.last_snapshot

    @property
    def last_compiled_scene(self):
        """The Channel ``CompiledScene`` that frame's legs were replayed on."""

        return (
            None
            if self._last_result is None
            else self._last_result.last_compiled_scene
        )

    @property
    def last_propagation(self):
        """That frame's two legs, as a typed
        :class:`~witwin.radar.propagation.RadarPropagationLegs`."""

        return (
            None if self._last_result is None else self._last_result.last_propagation
        )

    @property
    def last_radar_paths(self):
        """That frame's composed
        :class:`~witwin.radar.paths.RadarPathBatch`."""

        return (
            None if self._last_result is None else self._last_result.last_radar_paths
        )

__all__ = ["Radar", "RadarConfig"]
