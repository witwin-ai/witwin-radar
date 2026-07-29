"""Radar sensor descriptions, antenna patterns, and native row weighting.

This concept-axis module owns the complete sensor response chain: typed array
and power contracts, freeze-time pattern interpolation, the native per-row
weight operator, and the round-trip pattern stage. The former submodule paths
are intentionally not retained.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.autograd.forward_ad as forward_ad

from .cuda import native_ops as _ops
from .paths import RadarPathBatch
from .policy import first_order_only

__all__ = ["AntennaPatternSpec", "ISOTROPIC_PATTERN", "SensorArraySpec", "TxPowerSpec"]

DEFAULT_DIPOLE_ANGLES_DEG = tuple(float(angle) for angle in range(-90, 91))


def half_wave_dipole_power_cut(angle_deg: float) -> float:
    """Normalized half-wave dipole power gain versus off-boresight angle."""
    angle_rad = math.radians(angle_deg)
    cos_angle = math.cos(angle_rad)
    if abs(cos_angle) < 1e-8:
        return 0.0

    field = math.cos(0.5 * math.pi * math.sin(angle_rad)) / cos_angle
    return max(field * field, 0.0)


DEFAULT_DIPOLE_VALUES = tuple(half_wave_dipole_power_cut(angle) for angle in DEFAULT_DIPOLE_ANGLES_DEG)


def interp1d_zero_outside(axis: torch.Tensor, values: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    if query.numel() == 0:
        return torch.empty_like(query, dtype=values.dtype)

    flat_query = query.reshape(-1)
    index_upper = torch.bucketize(flat_query.detach(), axis)
    index_left = torch.clamp(index_upper - 1, 0, axis.numel() - 1)
    index_right = torch.clamp(index_upper, 0, axis.numel() - 1)

    x0 = axis[index_left]
    x1 = axis[index_right]
    y0 = values[index_left]
    y1 = values[index_right]
    denom = torch.clamp(x1 - x0, min=1e-12)
    weight = torch.where(index_left == index_right, torch.zeros_like(flat_query), (flat_query - x0) / denom)
    interpolated = y0 + weight * (y1 - y0)
    inside = (flat_query >= axis[0]) & (flat_query <= axis[-1])
    return torch.where(inside, interpolated, torch.zeros_like(interpolated)).reshape(query.shape)


def interp2d_zero_outside(
    x_axis: torch.Tensor, y_axis: torch.Tensor, values: torch.Tensor, x_query: torch.Tensor, y_query: torch.Tensor
) -> torch.Tensor:
    flat_x = x_query.reshape(-1)
    flat_y = y_query.reshape(-1)

    x_upper = torch.bucketize(flat_x.detach(), x_axis)
    y_upper = torch.bucketize(flat_y.detach(), y_axis)
    x_left = torch.clamp(x_upper - 1, 0, x_axis.numel() - 1)
    x_right = torch.clamp(x_upper, 0, x_axis.numel() - 1)
    y_low = torch.clamp(y_upper - 1, 0, y_axis.numel() - 1)
    y_high = torch.clamp(y_upper, 0, y_axis.numel() - 1)

    x0 = x_axis[x_left]
    x1 = x_axis[x_right]
    y0 = y_axis[y_low]
    y1 = y_axis[y_high]
    tx = torch.where(x_left == x_right, torch.zeros_like(flat_x), (flat_x - x0) / torch.clamp(x1 - x0, min=1e-12))
    ty = torch.where(y_low == y_high, torch.zeros_like(flat_y), (flat_y - y0) / torch.clamp(y1 - y0, min=1e-12))

    v00 = values[y_low, x_left]
    v10 = values[y_low, x_right]
    v01 = values[y_high, x_left]
    v11 = values[y_high, x_right]

    interpolated = (1.0 - tx) * (1.0 - ty) * v00 + tx * (1.0 - ty) * v10 + (1.0 - tx) * ty * v01 + tx * ty * v11
    inside = (flat_x >= x_axis[0]) & (flat_x <= x_axis[-1]) & (flat_y >= y_axis[0]) & (flat_y <= y_axis[-1])
    return torch.where(inside, interpolated, torch.zeros_like(interpolated)).reshape(x_query.shape)


def evaluate_antenna_pattern_xy(
    pattern_kind: str,
    x_axis: torch.Tensor,
    y_axis: torch.Tensor,
    x_values: torch.Tensor | None,
    y_values: torch.Tensor | None,
    values_2d: torch.Tensor | None,
    x_angles_deg: torch.Tensor,
    y_angles_deg: torch.Tensor,
) -> torch.Tensor:
    if pattern_kind == "separable":
        return interp1d_zero_outside(x_axis, x_values, x_angles_deg) * interp1d_zero_outside(
            y_axis, y_values, y_angles_deg
        )
    return interp2d_zero_outside(x_axis, y_axis, values_2d, x_angles_deg, y_angles_deg)


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


def _float_tensor(values: Sequence[Any], *, device: torch.device, name: str, shape: tuple[int, ...]) -> torch.Tensor:
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
                "reference_frequency_hz must be positive; it is what turns a half-wavelength element offset into metres"
            )
        if len(self.tx_loc) != self.num_tx:
            raise ValueError(f"tx_loc must hold exactly num_tx={self.num_tx} offsets, got {len(self.tx_loc)}")
        if len(self.rx_loc) != self.num_rx:
            raise ValueError(f"rx_loc must hold exactly num_rx={self.num_rx} offsets, got {len(self.rx_loc)}")
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

    def local_offsets_m(self, *, device: torch.device | str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
        """The element offsets in metres, still in the radar's LOCAL frame.

        Placing them in the world needs a pose, which belongs to the radar and
        not to its array. This spec describes the array; it does not know where
        the radar is pointing.
        """

        target = torch.device(device)
        spacing = self.element_spacing_m
        tx = _float_tensor(self.tx_loc, device=target, name="tx_loc", shape=(self.num_tx, 3))
        rx = _float_tensor(self.rx_loc, device=target, name="rx_loc", shape=(self.num_rx, 3))
        return (tx * spacing).contiguous(), (rx * spacing).contiguous()

    @classmethod
    def from_radar_config(cls, config) -> SensorArraySpec:
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
            raise ValueError(f"kind must be one of {list(PATTERN_KINDS)}, got {self.kind!r}")
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
    def half_wave_dipole(cls) -> AntennaPatternSpec:
        """The default: a half-wave dipole cut in both planes."""

        return cls(
            kind=PATTERN_KIND_SEPARABLE,
            x_angles_deg=tuple(DEFAULT_DIPOLE_ANGLES_DEG),
            y_angles_deg=tuple(DEFAULT_DIPOLE_ANGLES_DEG),
            x_values=tuple(DEFAULT_DIPOLE_VALUES),
            y_values=tuple(DEFAULT_DIPOLE_VALUES),
        )

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> AntennaPatternSpec:
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
        x_axis = _float_tensor(self.x_angles_deg, device=target, name="x_angles_deg", shape=(num_x,))
        y_axis = _float_tensor(self.y_angles_deg, device=target, name="y_angles_deg", shape=(num_y,))
        placeholder = torch.zeros(1, dtype=torch.float32, device=target)
        if self.kind == PATTERN_KIND_SEPARABLE:
            x_values = _float_tensor(self.x_values, device=target, name="x_values", shape=(num_x,))
            y_values = _float_tensor(self.y_values, device=target, name="y_values", shape=(num_y,))
            return x_axis, y_axis, x_values, y_values, placeholder
        values = _float_tensor(self.values, device=target, name="values", shape=(num_y, num_x)).reshape(-1)
        return x_axis, y_axis, placeholder, placeholder, values.contiguous()

    def evaluate_xy(self, x_angles_deg: torch.Tensor, y_angles_deg: torch.Tensor) -> torch.Tensor:
        """Torch evaluation, for freeze-time work and as the kernel's oracle."""

        x_axis, y_axis, x_values, y_values, values = self.tables(device=x_angles_deg.device)
        return evaluate_antenna_pattern_xy(
            self.kind,
            x_axis,
            y_axis,
            x_values,
            y_values,
            None
            if self.kind == PATTERN_KIND_SEPARABLE
            else values.reshape(len(self.y_angles_deg), len(self.x_angles_deg)),
            x_angles_deg,
            y_angles_deg,
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

    @classmethod
    def from_radar_config(cls, config) -> TxPowerSpec:
        return cls(power_dbm=float(config.power))


__all__ = [
    "PATTERN_KINDS",
    "PATTERN_KIND_CODE",
    "PATTERN_KIND_MAP",
    "PATTERN_KIND_SEPARABLE",
    "SPEED_OF_LIGHT_M_PER_S",
    "AntennaPatternSpec",
    "SensorArraySpec",
    "TxPowerSpec",
]


#: A row that interacts at a site, and a row that goes straight from a
#: transmitter to a receiver. The direct row's length is ``|rx - tx|`` with no
#: site term at all, rather than a via-row with a degenerate site, because a
#: zero-length second leg has no direction and therefore no antenna angle.
ROW_KIND_VIA = 0
ROW_KIND_DIRECT = 1


def _require_frozen_constant(owner: str, name: str, value: torch.Tensor) -> None:
    """Refuse a derivative on an input slot that has no gradient to return.

    Thirteen of the tensors this owner consumes are FROZEN descriptions of the
    array and of the frame's row set: how fast each antenna and site is
    moving, which way each facet faces, how each antenna is polarized, the
    local frame the pattern is tabulated in, the fixed leg length, and the
    pattern tables themselves. Only ``tx_pos``, ``rx_pos``, ``site_in``,
    ``site_out``, ``intensity`` and the complex weight are differentiable
    inputs of the native operator; the rest are either not inputs of the
    autograd ``Function`` at all or sit in slots whose ``backward`` returns
    ``None`` by construction.

    Before Phase 9 a caller who marked one of them got exactly that ``None``
    back, after a full frame had been computed, with nothing anywhere saying
    the slot was not differentiable. That is the failure mode the whole
    capability matrix exists to remove, so this refuses at CONSTRUCTION -
    before ``validate``, before any launch, before a result object exists.

    ``pattern_gain`` is the counter-example and stays as it is: it is a
    published OUTPUT that is correctly ``mark_non_differentiable``, which is a
    declaration rather than a silence.
    """

    tangent = forward_ad.unpack_dual(value).tangent
    if value.requires_grad or tangent is not None:
        raise RuntimeError(
            f"{owner}.{name} carries "
            + ("requires_grad" if value.requires_grad else "a forward tangent")
            + ", and it is a frozen geometric constant of the array rather "
            "than a differentiable input. The native sensor-weight operator "
            "has no gradient or tangent slot for it, so this request would "
            "run the whole frame and return None. The differentiable inputs "
            "here are tx_pos, rx_pos, site_in, site_out, intensity and the "
            "complex weight. A VELOCITY additionally has no leaf semantics at "
            "all under ADR-038: it is a forward-AD tangent direction that "
            "witwin.radar.propagation puts in the tangent slot of "
            "a position, so d(loss)/d(velocity) does not exist in either mode."
        )


def _require(name: str, tensor: object, *, dtype: torch.dtype, shape: tuple[int, ...]):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must use {dtype}, got {tensor.dtype}")
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if not tensor.is_contiguous():
        raise ValueError(
            f"{name} must be contiguous; the kernel indexes it linearly and this "
            "contract will not hide a copy on the hot path"
        )
    return tensor


@dataclass(frozen=True, slots=True)
class SensorWeightGeometry:
    """The constant part of one frame's row set.

    Every field here is a CONSTANT with respect to the derivative except the
    four position tensors, which are passed separately to the autograd entry so
    that they can carry a tangent or a gradient. Velocities and the pattern frame are constants by design: Phase 7
    owns dynamics, and a velocity that carried a gradient would be a different
    contract.

    Since Phase 9 that sentence is ENFORCED rather than merely written down.
    ``__post_init__`` refuses a float tensor here that carries
    ``requires_grad`` or a forward tangent, naming the field, before
    :meth:`validate` and before any launch. The index tensors are deliberately
    not checked: they are ``int64`` / ``int32`` and cannot carry a derivative
    for autograd to lose.
    """

    num_tx: int
    num_rx: int
    tx_velocity: torch.Tensor
    rx_velocity: torch.Tensor
    site_velocity: torch.Tensor
    fixed_length_m: torch.Tensor
    tx_index: torch.Tensor
    rx_index: torch.Tensor
    row_kind: torch.Tensor
    pattern_frame: torch.Tensor

    #: The float tensors this owner declares constant, in declaration order.
    #: Named as data so that a field added to the dataclass without a decision
    #: about its derivative is visible as a gap rather than as silence.
    FROZEN_FIELDS = ("tx_velocity", "rx_velocity", "site_velocity", "fixed_length_m", "pattern_frame")

    def __post_init__(self) -> None:
        for name in SensorWeightGeometry.FROZEN_FIELDS:
            value = getattr(self, name)
            if isinstance(value, torch.Tensor):
                _require_frozen_constant("SensorWeightGeometry", name, value)

    @property
    def path_count(self) -> int:
        return int(self.fixed_length_m.shape[0])

    def validate(self) -> None:
        rows = self.path_count
        _require("tx_velocity", self.tx_velocity, dtype=torch.float32, shape=(self.num_tx, 3))
        _require("rx_velocity", self.rx_velocity, dtype=torch.float32, shape=(self.num_rx, 3))
        _require("site_velocity", self.site_velocity, dtype=torch.float32, shape=(rows, 3))
        _require("fixed_length_m", self.fixed_length_m, dtype=torch.float32, shape=(rows,))
        _require("tx_index", self.tx_index, dtype=torch.int64, shape=(rows,))
        _require("rx_index", self.rx_index, dtype=torch.int64, shape=(rows,))
        _require("row_kind", self.row_kind, dtype=torch.int32, shape=(rows,))
        _require("pattern_frame", self.pattern_frame, dtype=torch.float32, shape=(3, 3))


@dataclass(frozen=True, slots=True)
class SensorWeightPlan:
    """The scalars and the resident pattern tables one launch needs.

    The tables are a resident LOOKUP - a measured or synthesised antenna
    pattern, sampled on a fixed angular grid - and the kernel interpolates
    them. They are not differentiable inputs of the operator, so a marked
    table is refused here for the same reason the geometry's constants are.
    The pattern's contribution to the derivative is real and is carried by the
    WEIGHT, through the positions that decide which angle is looked up.
    """

    pattern_kind: int
    tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    c0: float

    def __post_init__(self) -> None:
        for index, table in enumerate(self.tables):
            if isinstance(table, torch.Tensor):
                _require_frozen_constant("SensorWeightPlan", f"tables[{index}]", table)

    @classmethod
    def build(
        cls, pattern: AntennaPatternSpec, *, c0: float = SPEED_OF_LIGHT_M_PER_S, device: torch.device | str = "cuda"
    ) -> SensorWeightPlan:
        return cls(pattern_kind=pattern.kind_code, tables=pattern.tables(device=device), c0=float(c0))

    def kernel_tail(self, geometry: SensorWeightGeometry) -> tuple:
        """The trailing scalar arguments, in the order all three operators take.

        Written once because the three signatures agree by construction, and a
        divergence between them would be a silently different weight in the
        gradient than in the primal.
        """

        return (geometry.num_tx, geometry.num_rx, self.pattern_kind, self.c0)


@dataclass(frozen=True, slots=True)
class SensorWeightResult:
    """What one launch produced. Data only; the owner above constructs it."""

    weight: torch.Tensor
    total_delay_s: torch.Tensor
    delay_rate: torch.Tensor
    pattern_gain: torch.Tensor

    @classmethod
    def from_components(
        cls,
        out_re: torch.Tensor,
        out_im: torch.Tensor,
        total_delay_s: torch.Tensor,
        delay_rate: torch.Tensor,
        pattern_gain: torch.Tensor,
    ) -> SensorWeightResult:
        return cls(
            weight=torch.complex(out_re, out_im),
            total_delay_s=total_delay_s,
            delay_rate=delay_rate,
            pattern_gain=pattern_gain,
        )


def _row_constants(geometry: SensorWeightGeometry) -> tuple:
    """The constant row tensors that sit BEFORE ``intensity`` in every signature."""

    return (geometry.site_velocity, geometry.fixed_length_m, geometry.tx_index, geometry.rx_index, geometry.row_kind)


def _sensor_constants(geometry: SensorWeightGeometry, plan: SensorWeightPlan) -> tuple:
    """The constant sensor tensors that sit AFTER the weight in every signature."""

    return (geometry.pattern_frame,) + plan.tables


class _SensorWeight(torch.autograd.Function):
    """Autograd bridge for the three native sensor-weight operators."""

    @staticmethod
    def forward(
        tx_pos, rx_pos, site_in, site_out, intensity, weight_re, weight_im, tx_velocity, rx_velocity, geometry, plan
    ):
        rows = int(intensity.shape[0])
        out_re = torch.empty_like(intensity)
        out_im = torch.empty_like(intensity)
        total_delay_s = torch.empty_like(intensity)
        delay_rate = torch.empty_like(intensity)
        pattern_gain = torch.empty_like(intensity)
        _ops().sensor_weight_forward(
            tx_pos,
            rx_pos,
            tx_velocity,
            rx_velocity,
            site_in,
            site_out,
            *_row_constants(geometry),
            intensity,
            weight_re,
            weight_im,
            *_sensor_constants(geometry, plan),
            out_re,
            out_im,
            total_delay_s,
            delay_rate,
            pattern_gain,
            rows,
            *plan.kernel_tail(geometry),
        )
        return out_re, out_im, total_delay_s, delay_rate, pattern_gain

    @staticmethod
    def setup_context(ctx, inputs, output):
        (
            tx_pos,
            rx_pos,
            site_in,
            site_out,
            intensity,
            weight_re,
            weight_im,
            tx_velocity,
            rx_velocity,
            geometry,
            plan,
        ) = inputs
        ctx.geometry = geometry
        ctx.plan = plan
        saved = (tx_pos, rx_pos, site_in, site_out, intensity, weight_re, weight_im, tx_velocity, rx_velocity)
        ctx.save_for_backward(*saved)
        ctx.save_for_forward(*saved)
        # The pattern gain is a diagnostic, not a differentiable product: it is
        # published so a test can pin the interpolation against its Torch
        # oracle, and the weight is where its derivative actually goes.
        ctx.mark_non_differentiable(output[4])

    @staticmethod
    @first_order_only
    def backward(ctx, grad_out_re, grad_out_im, grad_tau_rt, grad_tau_rate, grad_gain):
        (tx_pos, rx_pos, site_in, site_out, intensity, weight_re, weight_im, tx_velocity, rx_velocity) = (
            ctx.saved_tensors
        )
        geometry = ctx.geometry
        plan = ctx.plan
        rows = int(intensity.shape[0])
        zeros = torch.zeros_like(intensity)
        grad_tx_pos = torch.empty_like(tx_pos)
        grad_rx_pos = torch.empty_like(rx_pos)
        grad_site_in = torch.empty_like(site_in)
        grad_site_out = torch.empty_like(site_out)
        grad_intensity = torch.empty_like(intensity)
        grad_weight_re = torch.empty_like(weight_re)
        grad_weight_im = torch.empty_like(weight_im)
        tx_row_scratch = torch.empty_like(site_in)
        rx_row_scratch = torch.empty_like(site_in)
        _ops().sensor_weight_backward(
            tx_pos,
            rx_pos,
            tx_velocity,
            rx_velocity,
            site_in,
            site_out,
            *_row_constants(geometry),
            intensity,
            weight_re,
            weight_im,
            *_sensor_constants(geometry, plan),
            zeros if grad_out_re is None else grad_out_re.contiguous(),
            zeros if grad_out_im is None else grad_out_im.contiguous(),
            zeros if grad_tau_rt is None else grad_tau_rt.contiguous(),
            zeros if grad_tau_rate is None else grad_tau_rate.contiguous(),
            grad_tx_pos,
            grad_rx_pos,
            grad_site_in,
            grad_site_out,
            grad_intensity,
            grad_weight_re,
            grad_weight_im,
            tx_row_scratch,
            rx_row_scratch,
            rows,
            *plan.kernel_tail(geometry),
        )
        return (
            grad_tx_pos,
            grad_rx_pos,
            grad_site_in,
            grad_site_out,
            grad_intensity,
            grad_weight_re,
            grad_weight_im,
            None,
            None,
            None,
            None,
        )

    @staticmethod
    def jvp(
        ctx,
        tan_tx_pos,
        tan_rx_pos,
        tan_site_in,
        tan_site_out,
        tan_intensity,
        tan_weight_re,
        tan_weight_im,
        tan_tx_velocity,
        tan_rx_velocity,
        tan_geometry,
        tan_plan,
    ):
        (tx_pos, rx_pos, site_in, site_out, intensity, weight_re, weight_im, tx_velocity, rx_velocity) = (
            ctx.saved_tensors
        )
        geometry = ctx.geometry
        plan = ctx.plan
        rows = int(intensity.shape[0])
        row_zeros = torch.zeros_like(intensity)
        vector_zeros = torch.zeros_like(site_in)
        tan_tx_pos = torch.zeros_like(tx_pos) if tan_tx_pos is None else tan_tx_pos.contiguous()
        tan_rx_pos = torch.zeros_like(rx_pos) if tan_rx_pos is None else tan_rx_pos.contiguous()
        tan_site_in = vector_zeros if tan_site_in is None else tan_site_in.contiguous()
        tan_site_out = vector_zeros if tan_site_out is None else tan_site_out.contiguous()
        tan_intensity = row_zeros if tan_intensity is None else tan_intensity.contiguous()
        tan_weight_re = row_zeros if tan_weight_re is None else tan_weight_re.contiguous()
        tan_weight_im = row_zeros if tan_weight_im is None else tan_weight_im.contiguous()

        tan_out_re = torch.empty_like(intensity)
        tan_out_im = torch.empty_like(intensity)
        tan_tau_rt = torch.empty_like(intensity)
        tan_tau_rate = torch.empty_like(intensity)
        _ops().sensor_weight_jvp(
            tx_pos,
            rx_pos,
            tx_velocity,
            rx_velocity,
            site_in,
            site_out,
            *_row_constants(geometry),
            intensity,
            weight_re,
            weight_im,
            *_sensor_constants(geometry, plan),
            tan_tx_pos,
            tan_rx_pos,
            tan_site_in,
            tan_site_out,
            tan_intensity,
            tan_weight_re,
            tan_weight_im,
            tan_out_re,
            tan_out_im,
            tan_tau_rt,
            tan_tau_rate,
            rows,
            *plan.kernel_tail(geometry),
        )
        return tan_out_re, tan_out_im, tan_tau_rt, tan_tau_rate, None


def evaluate_sensor_weights(
    *,
    tx_pos: torch.Tensor,
    rx_pos: torch.Tensor,
    site_in: torch.Tensor,
    site_out: torch.Tensor,
    intensity: torch.Tensor,
    weight: torch.Tensor,
    geometry: SensorWeightGeometry,
    plan: SensorWeightPlan,
) -> SensorWeightResult:
    """Apply the sensor description to one frame's rows, natively.

    ``weight`` is complex and is split into its real and imaginary parts with
    Torch's own autograd-aware accessors, so no complex tensor crosses the
    autograd boundary and the conjugate-Wirtinger convention cannot be got wrong
    at the seam. The returned ``total_delay_s`` and ``delay_rate`` are the same
    round-trip quantities the synthesis contract speaks, in seconds and
    dimensionless respectively.
    """

    geometry.validate()
    rows = geometry.path_count
    _require("tx_pos", tx_pos, dtype=torch.float32, shape=(geometry.num_tx, 3))
    _require("rx_pos", rx_pos, dtype=torch.float32, shape=(geometry.num_rx, 3))
    _require("site_in", site_in, dtype=torch.float32, shape=(rows, 3))
    _require("site_out", site_out, dtype=torch.float32, shape=(rows, 3))
    _require("intensity", intensity, dtype=torch.float32, shape=(rows,))
    if weight.dtype != torch.complex64:
        raise TypeError(f"weight must be complex64, got {weight.dtype}")
    if tuple(weight.shape) != (rows,):
        raise ValueError(f"weight must have shape {(rows,)}, got {tuple(weight.shape)}")

    out_re, out_im, total_delay_s, delay_rate, pattern_gain = _SensorWeight.apply(
        tx_pos,
        rx_pos,
        site_in,
        site_out,
        intensity,
        weight.real.contiguous(),
        weight.imag.contiguous(),
        geometry.tx_velocity,
        geometry.rx_velocity,
        geometry,
        plan,
    )
    return SensorWeightResult.from_components(out_re, out_im, total_delay_s, delay_rate, pattern_gain)


__all__ = [
    "ROW_KIND_DIRECT",
    "ROW_KIND_VIA",
    "SensorWeightGeometry",
    "SensorWeightPlan",
    "SensorWeightResult",
    "evaluate_sensor_weights",
]


#: The pattern that changes nothing, published as data so the no-op claim is a
#: value a test can pass rather than a sentence.
#:
#: Both axes span the whole range ``atan2`` can produce, so no direction ever
#: falls outside the support and takes the zero-outside branch, and both values
#: are ``1.0`` at each knot, so the interpolation returns exactly ``1.0``. It is
#: an ISOTROPIC pattern in the only sense this family has: unit power gain in
#: every direction.
ISOTROPIC_PATTERN = AntennaPatternSpec(
    kind=PATTERN_KIND_SEPARABLE,
    x_angles_deg=(-180.0, 180.0),
    y_angles_deg=(-180.0, 180.0),
    x_values=(1.0, 1.0),
    y_values=(1.0, 1.0),
)


def _pattern_plan(
    pattern: AntennaPatternSpec, *, reference_frequency_hz: float, device: torch.device
) -> SensorWeightPlan:
    if not isinstance(pattern, AntennaPatternSpec):
        raise TypeError(
            "antenna_pattern must be a witwin.radar.sensors.AntennaPatternSpec, "
            f"got {type(pattern).__name__}; pass "
            "radar.system_config.sensors for the configured one, or "
            "witwin.radar.sensors.ISOTROPIC_PATTERN for none"
        )
    return SensorWeightPlan.build(pattern, c0=SPEED_OF_LIGHT_M_PER_S, device=device)


def _site_rank_to_array_index(site_ids: tuple[int, ...], *, device: torch.device) -> torch.Tensor:
    """Map a composer response slot back to a row of the site position tensor.

    ``TwoWayComposer.freeze`` sorts the declared site IDs, so its
    ``response_slot`` is a rank in ASCENDING ID order while
    ``RadarWorldBinding.site_positions_m`` is in the order the binding published.
    The two coincide for the default allocator and diverge the moment a caller
    declares its own stable IDs, which is exactly the case where getting it
    wrong would look like a physics bug: every row would take its pattern angle
    from another target.

    Built on the host from the binding's own host tuple, once per epoch, so no
    device tensor is read back to get it.
    """

    listed = [int(value) for value in site_ids]
    if len(set(listed)) != len(listed):
        raise ValueError(f"site_ids must not repeat a stable ID, got {listed}")
    order = sorted(range(len(listed)), key=lambda index: listed[index])
    return torch.tensor(order, dtype=torch.int64, device=device)


@dataclass(frozen=True, slots=True, eq=False)
class RoundTripPatternStage:
    """One frozen topology's antenna-pattern application.

    Every tensor here is a CONSTANT of the frozen join: which transmitter and
    receiver each composed row belongs to, which site it visits, and the two
    zero-filled descriptions for the geometry quantities this route does not model
    (velocities and a fixed leg length). They are
    allocated once, at :meth:`freeze`, so a frame costs one gather and one
    launch.

    The velocities are zero and that is a statement rather than a placeholder:
    the delay and the delay rate this family computes are DISCARDED here.
    Channel owns the round-trip delay and the two-way join owns its rate, and
    recomputing either from the array geometry would put a second owner on a
    number the batch already carries. Only the weight is consumed.
    """

    num_tx: int
    num_rx: int
    row_count: int
    site_count: int
    tx_index: torch.Tensor
    rx_index: torch.Tensor
    site_slot: torch.Tensor
    row_kind: torch.Tensor
    zero_rows: torch.Tensor
    zero_vectors: torch.Tensor
    unit_intensity: torch.Tensor
    tx_velocity: torch.Tensor
    rx_velocity: torch.Tensor
    pattern_frame: torch.Tensor
    plan: SensorWeightPlan

    @classmethod
    def freeze(cls, radar, composer, *, site_ids, pattern: AntennaPatternSpec) -> RoundTripPatternStage:
        """Build the constant tables for one frozen :class:`TwoWayComposer`.

        ``site_ids`` is the binding's host tuple, in the order its site position
        tensor is laid out. Passing it rather than reading the composer's
        ``topology.site_id`` back to the host is deliberate: the host tuple is
        already there, and the device column is not.
        """

        array = radar.system_config.sensors.array
        num_tx = int(array.num_tx)
        num_rx = int(array.num_rx)
        pair_index = composer.sensor_pair_index
        device = pair_index.device
        if composer.sensor_pair_count != num_tx * num_rx:
            raise ValueError(
                f"this join spans {composer.sensor_pair_count} sensor pairs but "
                f"the array is {num_tx} x {num_rx}; the pattern stage looks a "
                "transmitter and a receiver up by pair rank and the two must be "
                "the same front end"
            )
        site_rank_to_index = _site_rank_to_array_index(tuple(site_ids), device=device)
        if int(site_rank_to_index.shape[0]) != composer.site_count:
            raise ValueError(
                f"the binding declares {int(site_rank_to_index.shape[0])} sites "
                f"but this join was frozen against {composer.site_count}"
            )
        rows = int(composer.path_count)
        # PAIR_RANK_LAYOUT is sink major - pair = rx_rank * num_tx + tx_rank -
        # so the transmitter is the REMAINDER and the receiver is the quotient.
        # Getting these two the wrong way round steers every pattern lookup at
        # the wrong element and still produces a plausible cube.
        tx_index = torch.remainder(pair_index, num_tx).contiguous()
        rx_index = torch.div(pair_index, num_tx, rounding_mode="floor").contiguous()
        zero_rows = torch.zeros(rows, dtype=torch.float32, device=device)
        return cls(
            num_tx=num_tx,
            num_rx=num_rx,
            row_count=rows,
            site_count=int(composer.site_count),
            tx_index=tx_index,
            rx_index=rx_index,
            site_slot=site_rank_to_index.index_select(0, composer.response_slot).contiguous(),
            row_kind=torch.full((rows,), ROW_KIND_VIA, dtype=torch.int32, device=device),
            zero_rows=zero_rows,
            zero_vectors=torch.zeros(rows, 3, dtype=torch.float32, device=device),
            unit_intensity=torch.ones(rows, dtype=torch.float32, device=device),
            tx_velocity=torch.zeros(num_tx, 3, dtype=torch.float32, device=device),
            rx_velocity=torch.zeros(num_rx, 3, dtype=torch.float32, device=device),
            # ``local_from_world_vectors`` is ``v @ world_from_local``, so the
            # pattern-frame components are dot products with that matrix's COLUMNS. The
            # kernel takes those columns as its rows; the transpose IS the frame
            # change, and it is the canonical world-to-pattern-frame transform.
            pattern_frame=radar._world_from_local_matrix(device=device, dtype=torch.float32)[1]
            .transpose(0, 1)
            .contiguous(),
            plan=_pattern_plan(pattern, reference_frequency_hz=array.reference_frequency_hz, device=device),
        )

    def _geometry(self) -> SensorWeightGeometry:
        return SensorWeightGeometry(
            num_tx=self.num_tx,
            num_rx=self.num_rx,
            tx_velocity=self.tx_velocity,
            rx_velocity=self.rx_velocity,
            site_velocity=self.zero_vectors,
            fixed_length_m=self.zero_rows,
            tx_index=self.tx_index,
            rx_index=self.rx_index,
            row_kind=self.row_kind,
            pattern_frame=self.pattern_frame,
        )

    def apply(
        self, paths: RadarPathBatch, *, tx_pos: torch.Tensor, rx_pos: torch.Tensor, site_positions_m: torch.Tensor
    ) -> RadarPathBatch:
        """Publish ``paths`` with the transmit and receive pattern gains applied.

        The three position tensors are the binding's own objects and are passed
        through by reference, so a ``requires_grad`` leaf or a forward-AD dual on
        the radar's elements or on a site reaches the native companions.
        ``site_in`` and ``site_out`` are the SAME gathered tensor: a site is one
        point, the transmit pattern reads the direction to it and the receive
        pattern reads the direction from it, and autograd accumulates both
        gradients into the one leaf. Building the gather twice would halve that
        gradient and zero half of a tangent.

        Everything except the weight passes through untouched - the same
        objects, not copies - so row identity, row order, storage aliasing,
        dtype, device and the delay's gradient state all survive.
        """

        if not isinstance(paths, RadarPathBatch):
            raise TypeError(f"the antenna-pattern stage consumes a RadarPathBatch, got {type(paths).__name__}")
        if paths.join_mode != "multipath":
            raise NotImplementedError(
                f"the antenna-pattern stage is frozen against a two-way join and "
                f"these rows declare join_mode {paths.join_mode!r}. A direct row "
                "has no scatter site, so its transmit and receive directions are "
                "the other endpoint rather than a site, and applying this stage's "
                "site-based row kind to it would look up the pattern along a "
                "direction the row does not have. A direct-leakage pattern is a "
                "separate capability with its own row kind"
            )
        if paths.weight_includes_antenna_pattern:
            raise ValueError(
                "these rows already record weight_includes_antenna_pattern; "
                "applying the array pattern twice squares its gain and is "
                "invisible in any magnitude plot, so it is refused here rather "
                "than counted"
            )
        if paths.path_count != self.row_count:
            raise ValueError(
                f"these rows carry {paths.path_count} paths but this stage was "
                f"frozen against {self.row_count}; the batch does not belong to "
                "this frozen topology"
            )
        site = site_positions_m.index_select(0, self.site_slot)
        geometry = self._geometry()
        weight = evaluate_sensor_weights(
            tx_pos=tx_pos,
            rx_pos=rx_pos,
            site_in=site,
            site_out=site,
            intensity=self.unit_intensity,
            weight=paths.complex_transfer_ref,
            geometry=geometry,
            plan=self.plan,
        ).weight
        return RadarPathBatch(
            sensor_pair_count=paths.sensor_pair_count,
            path_count=paths.path_count,
            sensor_pair_index=paths.sensor_pair_index,
            pair_offsets=paths.pair_offsets,
            total_delay_s=paths.total_delay_s,
            delay_rate=paths.delay_rate,
            complex_transfer_ref=weight,
            reference_frequency_hz=paths.reference_frequency_hz,
            row_valid=paths.row_valid,
            topology=paths.topology,
            join_mode=paths.join_mode,
            frequency_response=self._apply_band(paths, geometry, tx_pos, rx_pos, site),
            frequency_offsets_hz=paths.frequency_offsets_hz,
            weight_includes_antenna_pattern=True,
        )

    def _apply_band(
        self,
        paths: RadarPathBatch,
        geometry: SensorWeightGeometry,
        tx_pos: torch.Tensor,
        rx_pos: torch.Tensor,
        site: torch.Tensor,
    ) -> torch.Tensor | None:
        """The same real gain, applied to every column of a composed band.

        The frequency axis is a PYTHON LOOP over the existing ``[K]`` primitive
        rather than a strided ``[K, F]`` kernel, which is the boundary
        ``TwoWayComposer._compose_band`` already draws for the same reason:
        widening a native family means widening its primal, its jvp and its vjp
        together, and that needs a measured reason first.

        The pattern is applied to the band at all - rather than only to the
        reference column - because this family's gain has no frequency axis. A
        band whose reference column carried the pattern and whose columns did not
        would be two different antennas in one batch.
        """

        if paths.frequency_response is None:
            return None
        columns = [
            evaluate_sensor_weights(
                tx_pos=tx_pos,
                rx_pos=rx_pos,
                site_in=site,
                site_out=site,
                intensity=self.unit_intensity,
                weight=paths.frequency_response[:, index],
                geometry=geometry,
                plan=self.plan,
            ).weight
            for index in range(int(paths.frequency_response.shape[1]))
        ]
        return torch.stack(columns, dim=1)


__all__ = ["AntennaPatternSpec", "ISOTROPIC_PATTERN", "SensorArraySpec", "TxPowerSpec"]
