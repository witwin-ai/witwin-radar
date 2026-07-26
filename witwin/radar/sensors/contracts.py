"""What a radar's antennas are, as three typed descriptions.

Work item 4 names one owner for antenna pattern, array geometry, transmit
power, and the legacy receive projection. This module is the first three; the
fourth is deliberately NOT here, and the reason is the point of the whole
package.

**Transmit power reaches physics through ``powers_w`` and nowhere else.**
:class:`TxPowerSpec` converts dBm to watts and that value fills the source
endpoint's ``powers_w``. Channel's coefficient then already contains
``sqrt(tx_power)``. Applying a separate transmit gain to a Channel-sourced
weight counts the power twice AND mixes units - the old ``radar.gain`` is
``sqrt(P R)`` in sqrt(W ohm) while the weight is in sqrt(W) - so there is no
transmit-gain field on this spec and no Radar-side multiplication by one. The
sqrt(W) to volt conversion is a single explicit factor of ``sqrt(R)`` at the
frontend's port stage.

**The receive polarization projection is Channel's.** Channel publishes
``POLARIZATION = "world_cartesian_complex3_then_receiver_projection"``, so it
has already projected the field onto the declared receive polarization. A
Radar-side projection on a Channel-sourced path is a SECOND projection. It
survives only inside the native kernel, behind the ``legacy_real_polarization``
flag, for the real-amplitude route that has no Jones transport to project with.
:class:`PolarizationSpec` therefore describes that legacy route and says so.

Everything here is pure and CPU-testable. Unit conversion that is wrong once is
wrong everywhere, and it should not need a GPU to check.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch

from .pattern import (
    DEFAULT_DIPOLE_ANGLES_DEG,
    DEFAULT_DIPOLE_VALUES,
    evaluate_antenna_pattern_vectors,
    evaluate_antenna_pattern_xy,
)


#: Exact SI definition, in metres per second. Quoted rather than imported from
#: the synthesis contracts because a sensor package that depended on a waveform
#: package to know the speed of light would be an edge in the wrong direction.
SPEED_OF_LIGHT_M_PER_S = 299792458.0

#: The two supported pattern kinds, named exactly as ``validation.py`` already
#: normalises them. ``separable`` is a product of two one-dimensional cuts;
#: ``map`` is a bilinear two-dimensional table. The kernel's integer selector
#: mirrors this order.
PATTERN_KIND_SEPARABLE = "separable"
PATTERN_KIND_MAP = "map"
PATTERN_KINDS = (PATTERN_KIND_SEPARABLE, PATTERN_KIND_MAP)

#: The kernel's ``pattern_kind`` argument. An integer crosses the ABI because a
#: string would cost an allocation and a comparison per launch to say something
#: the spec already validated once.
PATTERN_KIND_CODE = {PATTERN_KIND_SEPARABLE: 0, PATTERN_KIND_MAP: 1}


def _float_tensor(
    values: Sequence[Any], *, device: torch.device, name: str, shape: tuple[int, ...]
) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32, device=device).contiguous()
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    return tensor


@dataclass(frozen=True, slots=True)
class SensorArraySpec:
    """Element offsets and the wavelength that turns them into metres.

    ``tx_loc`` and ``rx_loc`` are in units of HALF A WAVELENGTH, which is the
    unit the radar configuration has always used, and ``element_spacing_m``
    turns them into metres. Keeping the offsets in half-wavelengths is what lets
    the same array description mean the same beam pattern at a different carrier
    - the array is defined by its electrical size, not by its physical one.
    """

    num_tx: int
    num_rx: int
    tx_loc: tuple[tuple[float, float, float], ...]
    rx_loc: tuple[tuple[float, float, float], ...]
    reference_frequency_hz: float

    def __post_init__(self) -> None:
        if self.num_tx < 1:
            raise ValueError("num_tx must be positive")
        if self.num_rx < 1:
            raise ValueError("num_rx must be positive")
        if not self.reference_frequency_hz > 0.0:
            raise ValueError(
                "reference_frequency_hz must be positive; it is what turns a "
                "half-wavelength element offset into metres"
            )
        if len(self.tx_loc) != self.num_tx:
            raise ValueError(
                f"tx_loc must hold exactly num_tx={self.num_tx} offsets, got "
                f"{len(self.tx_loc)}"
            )
        if len(self.rx_loc) != self.num_rx:
            raise ValueError(
                f"rx_loc must hold exactly num_rx={self.num_rx} offsets, got "
                f"{len(self.rx_loc)}"
            )
        for name, rows in (("tx_loc", self.tx_loc), ("rx_loc", self.rx_loc)):
            for index, row in enumerate(rows):
                if len(row) != 3:
                    raise ValueError(f"{name}[{index}] must be a 3-element offset")

    @property
    def wavelength_m(self) -> float:
        return SPEED_OF_LIGHT_M_PER_S / self.reference_frequency_hz

    @property
    def element_spacing_m(self) -> float:
        """``c0 / f_c / 2``: what one unit of ``tx_loc`` is worth in metres."""

        return self.wavelength_m / 2.0

    @property
    def sensor_pair_count(self) -> int:
        """The virtual array size, ``num_tx * num_rx``."""

        return self.num_tx * self.num_rx

    def local_offsets_m(
        self, *, device: torch.device | str = "cpu"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The element offsets in metres, still in the radar's LOCAL frame.

        Placing them in the world needs a pose, which belongs to the radar and
        not to its array. This spec describes the array; it does not know where
        the radar is pointing.
        """

        target = torch.device(device)
        spacing = self.element_spacing_m
        tx = _float_tensor(
            self.tx_loc, device=target, name="tx_loc", shape=(self.num_tx, 3)
        )
        rx = _float_tensor(
            self.rx_loc, device=target, name="rx_loc", shape=(self.num_rx, 3)
        )
        return (tx * spacing).contiguous(), (rx * spacing).contiguous()

    @classmethod
    def from_radar_config(cls, config) -> "SensorArraySpec":
        return cls(
            num_tx=int(config.num_tx),
            num_rx=int(config.num_rx),
            tx_loc=tuple(tuple(float(v) for v in row) for row in config.tx_loc),
            rx_loc=tuple(tuple(float(v) for v in row) for row in config.rx_loc),
            reference_frequency_hz=float(config.fc),
        )


@dataclass(frozen=True, slots=True)
class AntennaPatternSpec:
    """A tabulated POWER gain versus the two off-boresight angles.

    The table is a CONSTANT. The direction into it is differentiable, and the
    interpolation is piecewise linear, so the gain has an exact
    almost-everywhere derivative that the native kernel carries. A knot and the
    two support edges are genuine non-differentiabilities and the kernel returns
    the almost-everywhere value there, which is what the Torch expression it
    replaces already did.

    The angles are the same two the pattern helpers use: with a direction
    expressed in the radar's LOCAL frame, ``x = atan2(v_x, -v_z)`` and
    ``y = atan2(v_y, -v_z)``, both in degrees. Outside the tabulated support the
    gain is exactly zero rather than the nearest tabulated value, which is a
    modelling choice (an antenna that does not radiate behind itself) rather
    than an extrapolation accident.
    """

    kind: str
    x_angles_deg: tuple[float, ...]
    y_angles_deg: tuple[float, ...]
    x_values: tuple[float, ...] | None = None
    y_values: tuple[float, ...] | None = None
    values: tuple[tuple[float, ...], ...] | None = None

    def __post_init__(self) -> None:
        if self.kind not in PATTERN_KINDS:
            raise ValueError(
                f"kind must be one of {list(PATTERN_KINDS)}, got {self.kind!r}"
            )
        if len(self.x_angles_deg) < 2 or len(self.y_angles_deg) < 2:
            raise ValueError("both pattern axes need at least two samples")
        if self.kind == PATTERN_KIND_SEPARABLE:
            if self.x_values is None or self.y_values is None:
                raise ValueError("a separable pattern needs x_values and y_values")
            if len(self.x_values) != len(self.x_angles_deg):
                raise ValueError("x_values must hold one value per x axis sample")
            if len(self.y_values) != len(self.y_angles_deg):
                raise ValueError("y_values must hold one value per y axis sample")
        else:
            if self.values is None:
                raise ValueError("a map pattern needs values")
            if len(self.values) != len(self.y_angles_deg):
                raise ValueError("values must hold one row per y axis sample")
            for row in self.values:
                if len(row) != len(self.x_angles_deg):
                    raise ValueError("each values row needs one entry per x sample")

    @property
    def kind_code(self) -> int:
        return PATTERN_KIND_CODE[self.kind]

    @classmethod
    def half_wave_dipole(cls) -> "AntennaPatternSpec":
        """The default: a half-wave dipole cut in both planes."""

        return cls(
            kind=PATTERN_KIND_SEPARABLE,
            x_angles_deg=tuple(DEFAULT_DIPOLE_ANGLES_DEG),
            y_angles_deg=tuple(DEFAULT_DIPOLE_ANGLES_DEG),
            x_values=tuple(DEFAULT_DIPOLE_VALUES),
            y_values=tuple(DEFAULT_DIPOLE_VALUES),
        )

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> "AntennaPatternSpec":
        """Adopt a validated antenna-pattern mapping, or the dipole default."""

        if config is None:
            return cls.half_wave_dipole()
        kind = str(config["kind"])
        if kind == PATTERN_KIND_SEPARABLE:
            return cls(
                kind=kind,
                x_angles_deg=tuple(float(v) for v in config["x_angles_deg"]),
                y_angles_deg=tuple(float(v) for v in config["y_angles_deg"]),
                x_values=tuple(float(v) for v in config["x_values"]),
                y_values=tuple(float(v) for v in config["y_values"]),
            )
        return cls(
            kind=kind,
            x_angles_deg=tuple(float(v) for v in config["x_angles_deg"]),
            y_angles_deg=tuple(float(v) for v in config["y_angles_deg"]),
            values=tuple(tuple(float(v) for v in row) for row in config["values"]),
        )

    def tables(
        self, *, device: torch.device | str = "cpu"
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """The five resident tensors the kernel indexes, built once per frame set.

        The unused table for this kind is a one-element placeholder rather than
        an empty tensor: an empty CUDA tensor may carry a null data pointer, and
        a null pointer that is never dereferenced is still a pointer the ABI
        check has to reason about.
        """

        target = torch.device(device)
        num_x = len(self.x_angles_deg)
        num_y = len(self.y_angles_deg)
        x_axis = _float_tensor(
            self.x_angles_deg, device=target, name="x_angles_deg", shape=(num_x,)
        )
        y_axis = _float_tensor(
            self.y_angles_deg, device=target, name="y_angles_deg", shape=(num_y,)
        )
        placeholder = torch.zeros(1, dtype=torch.float32, device=target)
        if self.kind == PATTERN_KIND_SEPARABLE:
            x_values = _float_tensor(
                self.x_values, device=target, name="x_values", shape=(num_x,)
            )
            y_values = _float_tensor(
                self.y_values, device=target, name="y_values", shape=(num_y,)
            )
            return x_axis, y_axis, x_values, y_values, placeholder
        values = _float_tensor(
            self.values, device=target, name="values", shape=(num_y, num_x)
        ).reshape(-1)
        return x_axis, y_axis, placeholder, placeholder, values.contiguous()

    def evaluate_xy(
        self, x_angles_deg: torch.Tensor, y_angles_deg: torch.Tensor
    ) -> torch.Tensor:
        """Torch evaluation, for freeze-time work and as the kernel's oracle."""

        x_axis, y_axis, x_values, y_values, values = self.tables(
            device=x_angles_deg.device
        )
        return evaluate_antenna_pattern_xy(
            self.kind,
            x_axis,
            y_axis,
            x_values,
            y_values,
            None if self.kind == PATTERN_KIND_SEPARABLE else values.reshape(
                len(self.y_angles_deg), len(self.x_angles_deg)
            ),
            x_angles_deg,
            y_angles_deg,
        )

    def evaluate_vectors(self, vectors: torch.Tensor) -> torch.Tensor:
        """Torch evaluation from LOCAL-frame direction vectors."""

        x_axis, y_axis, x_values, y_values, values = self.tables(device=vectors.device)
        return evaluate_antenna_pattern_vectors(
            self.kind,
            x_axis,
            y_axis,
            x_values,
            y_values,
            None if self.kind == PATTERN_KIND_SEPARABLE else values.reshape(
                len(self.y_angles_deg), len(self.x_angles_deg)
            ),
            vectors,
        )


@dataclass(frozen=True, slots=True)
class TxPowerSpec:
    """Transmit power in dBm, and the ONE place it becomes watts.

    ``transmit_power_watts`` is what fills a source endpoint's ``powers_w`` and
    it reaches physics through that field and no other. There is deliberately no
    ``voltage_gain`` here: the old ``radar.gain = sqrt(P R)`` multiplied a weight
    that already carried ``sqrt(P)``, which counts the power twice and leaves the
    result in sqrt(W ohm) while the weight is in sqrt(W).
    """

    power_dbm: float

    @property
    def transmit_power_watts(self) -> float:
        """``1e-3 * 10^(dBm/10)``."""

        return 1e-3 * (10.0 ** (float(self.power_dbm) / 10.0))

    @property
    def transmit_amplitude_sqrt_w(self) -> float:
        """``sqrt(P_tx)``, the amplitude-domain factor a weight would carry.

        Published for the legacy real-amplitude route, whose weight carries no
        transmit power at all and therefore needs the sensor-weight owner to
        apply it. A Channel-sourced weight must NOT be multiplied by this.
        """

        return math.sqrt(self.transmit_power_watts)

    @classmethod
    def from_radar_config(cls, config) -> "TxPowerSpec":
        return cls(power_dbm=float(config.power))


@dataclass(frozen=True, slots=True)
class PolarizationSpec:
    """The LEGACY transmit/receive projection, and why it is legacy.

    Channel already projects the field onto each endpoint's declared
    polarization - its published contract is
    ``world_cartesian_complex3_then_receiver_projection`` - so applying this on
    a Channel-sourced weight is a second projection of the same field. It exists
    for the real-amplitude route, which carries a signed scalar instead of a
    Jones operator and has nothing else to express a reflection's polarization
    flip with.

    ``reflection_flip`` mirrors the transmit polarization about the surface
    normal before the projection. Its SIGN is physics: a mirrored vector can
    point away from the receive polarization, so the factor is signed and a
    mirrored row is exactly ``-1`` times the unmirrored one. Taking an absolute
    value here would be a silent 180-degree error that no magnitude plot shows.
    """

    tx: tuple[tuple[float, float, float], ...]
    rx: tuple[tuple[float, float, float], ...]
    reflection_flip: bool = True

    def __post_init__(self) -> None:
        if not self.tx or not self.rx:
            raise ValueError("a polarization spec needs at least one vector per side")
        for name, rows in (("tx", self.tx), ("rx", self.rx)):
            for index, row in enumerate(rows):
                if len(row) != 3:
                    raise ValueError(f"{name}[{index}] must be a 3-element vector")

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> "PolarizationSpec | None":
        if config is None:
            return None
        return cls(
            tx=tuple(tuple(float(v) for v in row) for row in config["tx"]),
            rx=tuple(tuple(float(v) for v in row) for row in config["rx"]),
            reflection_flip=bool(config.get("reflection_flip", True)),
        )

    def local_vectors(
        self, *, device: torch.device | str = "cpu"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target = torch.device(device)
        tx = _float_tensor(self.tx, device=target, name="tx", shape=(len(self.tx), 3))
        rx = _float_tensor(self.rx, device=target, name="rx", shape=(len(self.rx), 3))
        return tx, rx


__all__ = [
    "PATTERN_KINDS",
    "PATTERN_KIND_CODE",
    "PATTERN_KIND_MAP",
    "PATTERN_KIND_SEPARABLE",
    "SPEED_OF_LIGHT_M_PER_S",
    "AntennaPatternSpec",
    "PolarizationSpec",
    "SensorArraySpec",
    "TxPowerSpec",
]
