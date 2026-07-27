"""Sensors: antenna pattern, array geometry, transmit power, and the weight.

Work item 4's "single owner" is this package. What used to be spread between
``utils/antenna.py``, three ``Radar`` attributes, and four Torch expressions in
``solvers/common.py`` was one description and one native kernel.

Two ownership rules are structural here rather than documented:

* **Transmit power reaches physics through ``powers_w`` and nowhere else.**
  :class:`TxPowerSpec` converts dBm to watts and that value fills the source
  endpoint. There is no transmit-gain field to multiply a weight by, because a
  Channel coefficient already carries ``sqrt(P_tx)`` and doing it again both
  double counts the power and mixes sqrt(W) with sqrt(W ohm).
* **The receive polarization projection is Channel's.** It survives here only
  inside the kernel, behind ``legacy_real_polarization``, for the real-amplitude
  route that has a signed scalar instead of a Jones operator.

:class:`RoundTripPatternStage` is the production consumer of all of it on the
scene-driven pipeline: it applies the transmit and receive pattern gain to a
composed round-trip batch through the same native family, with all three mode
flags off because a Channel-sourced weight already carries what they would add.
"""

from .contracts import (
    PATTERN_KINDS,
    PATTERN_KIND_CODE,
    PATTERN_KIND_MAP,
    PATTERN_KIND_SEPARABLE,
    SPEED_OF_LIGHT_M_PER_S,
    AntennaPatternSpec,
    PolarizationSpec,
    SensorArraySpec,
    TxPowerSpec,
)
from .pattern import (
    DEFAULT_DIPOLE_ANGLES_DEG,
    DEFAULT_DIPOLE_VALUES,
    evaluate_antenna_pattern_vectors,
    evaluate_antenna_pattern_xy,
    half_wave_dipole_power_cut,
    interp1d_zero_outside,
    interp2d_zero_outside,
)
from .weights import (
    ROW_KIND_DIRECT,
    ROW_KIND_VIA,
    SensorWeightGeometry,
    SensorWeightModes,
    SensorWeightPlan,
    SensorWeightResult,
    evaluate_sensor_weights,
)
from .round_trip import (
    CHANNEL_SOURCED_MODES,
    ISOTROPIC_PATTERN,
    RoundTripPatternStage,
)

__all__ = [
    "CHANNEL_SOURCED_MODES",
    "DEFAULT_DIPOLE_ANGLES_DEG",
    "DEFAULT_DIPOLE_VALUES",
    "ISOTROPIC_PATTERN",
    "PATTERN_KINDS",
    "PATTERN_KIND_CODE",
    "PATTERN_KIND_MAP",
    "PATTERN_KIND_SEPARABLE",
    "ROW_KIND_DIRECT",
    "ROW_KIND_VIA",
    "SPEED_OF_LIGHT_M_PER_S",
    "AntennaPatternSpec",
    "PolarizationSpec",
    "SensorArraySpec",
    "SensorWeightGeometry",
    "SensorWeightModes",
    "SensorWeightPlan",
    "SensorWeightResult",
    "RoundTripPatternStage",
    "TxPowerSpec",
    "evaluate_antenna_pattern_vectors",
    "evaluate_antenna_pattern_xy",
    "evaluate_sensor_weights",
    "half_wave_dipole_power_cut",
    "interp1d_zero_outside",
    "interp2d_zero_outside",
]
