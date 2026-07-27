"""Centralized validation for radar configuration.

All config validation lives here. Dataclasses are pure data containers;
any parsing/validation of dict-shaped configuration goes through the
``validate_*`` functions exposed by this module.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Iterable

from .sensors.pattern import DEFAULT_DIPOLE_ANGLES_DEG, DEFAULT_DIPOLE_VALUES

if TYPE_CHECKING:
    from .radar import RadarConfig


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

_POLARIZATION_PREFIX = "Polarization field"
_POLARIZATION_ALIASES = {
    "horizontal": (1.0, 0.0, 0.0),
    "h": (1.0, 0.0, 0.0),
    "vertical": (0.0, 1.0, 0.0),
    "v": (0.0, 1.0, 0.0),
}


def _is_single_vector_spec(value: Any) -> bool:
    if isinstance(value, str):
        return True
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return False
    return all(not isinstance(component, (list, tuple)) for component in value)


def _vector_bank(name: str, value: Any, expected_count: int) -> tuple[tuple[float, float, float], ...]:
    if _is_single_vector_spec(value):
        vector = _parse_vector3(name, value, prefix=_POLARIZATION_PREFIX, aliases=_POLARIZATION_ALIASES)
        return tuple(vector for _ in range(expected_count))
    if not isinstance(value, (list, tuple)):
        raise ValueError(
            f"{_POLARIZATION_PREFIX} '{name}' must be a vector or a sequence of {expected_count} vectors."
        )
    if len(value) != expected_count:
        raise ValueError(
            f"{_POLARIZATION_PREFIX} '{name}' must contain exactly {expected_count} entries; got {len(value)}."
        )
    return tuple(
        _parse_vector3(f"{name}[{i}]", entry, prefix=_POLARIZATION_PREFIX, aliases=_POLARIZATION_ALIASES)
        for i, entry in enumerate(value)
    )


def validate_polarization_config(
    config: dict[str, Any], *, num_tx: int, num_rx: int
) -> dict[str, Any]:
    allowed = {"tx", "rx", "reflection_flip"}
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise TypeError(f"Unsupported polarization config keys: {', '.join(unknown)}")

    tx_value = config.get("tx")
    rx_value = config.get("rx")
    if tx_value is None and rx_value is None:
        raise ValueError("Polarization config must define at least one of 'tx' or 'rx'.")
    if tx_value is None:
        tx_value = rx_value
    if rx_value is None:
        rx_value = tx_value

    return {
        "tx": [list(vector) for vector in _vector_bank("tx", tx_value, num_tx)],
        "rx": [list(vector) for vector in _vector_bank("rx", rx_value, num_rx)],
        "reflection_flip": bool(config.get("reflection_flip", True)),
    }


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
    from .radar import RadarConfig

    _require_keys(config, _RADAR_REQUIRED_KEYS, "Radar config")

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

    from .config import (
        WAVEFORM_FMCW,
        WAVEFORM_KINDS,
        WAVEFORM_OFDM,
        FmcwWaveformConfig,
        OfdmWaveformConfig,
        PulsedWaveformConfig,
    )

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
    """Build the sensor block: array, pattern, transmit power, polarization.

    ``power`` is in dBm and becomes ``powers_w`` on a source endpoint. There is
    deliberately no transmit-gain output here: a Channel coefficient already
    carries ``sqrt(P_tx)``, so a second one would count the power twice and mix
    sqrt(W) with sqrt(W ohm).
    """

    from .config import SensorConfig
    from .sensors.contracts import (
        AntennaPatternSpec,
        PolarizationSpec,
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
    polarization = (
        validate_polarization_config(
            config["polarization"], num_tx=num_tx, num_rx=num_rx
        )
        if config.get("polarization") is not None
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
        polarization=PolarizationSpec.from_config(polarization),
    )


def validate_propagation_config(config: dict[str, Any]):
    """Build the propagation block, which is the ONLY block an adapter sees."""

    from .config import PropagationConfig

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

    from .config import ProcessingConfig

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

    from .frontend.contracts import (
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

    from .config import RadarSystemConfig

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
