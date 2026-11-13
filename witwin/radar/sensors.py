"""Radar sensor descriptions, antenna patterns, and native row weighting.

This concept-axis module owns the complete sensor response chain: typed array
and power contracts, freeze-time pattern interpolation, the native per-row
weight operator, and the round-trip pattern stage.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, ClassVar

import torch

from .cuda import native_ops as _ops
from .paths import RadarPathBatch, SlotRows
from .policy import SPEED_OF_LIGHT_M_PER_S, first_order_only, refuse_derivative

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


#: The two supported pattern kinds. ``separable`` is a product of two
#: one-dimensional cuts; ``map`` is a bilinear two-dimensional table. The
#: kernel's integer selector mirrors this order.
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


#: What a pattern table holds. ``power`` is a real normalised power gain
#: ``0 <= G <= 1``; ``field`` is a complex normalised field amplitude
#: ``|F| <= 1`` (amplitude AND phase). The kernel's integer selector mirrors
#: this order, and a polarimetric quantity is the next code (R-ADR-030).
PATTERN_QUANTITY_POWER = "power"
PATTERN_QUANTITY_FIELD = "field"
PATTERN_QUANTITIES = (PATTERN_QUANTITY_POWER, PATTERN_QUANTITY_FIELD)
PATTERN_QUANTITY_CODE = {PATTERN_QUANTITY_POWER: 0, PATTERN_QUANTITY_FIELD: 1}

#: The floats of the packed header the kernel reads (see ``sensor_weight.cu``).
PATTERN_HEADER_LENGTH = 8

#: Tolerance on the normalisation bound, for tables sampled in float32.
_NORMALISATION_SLACK = 1e-6


def _axis(values) -> tuple[float, ...]:
    if isinstance(values, torch.Tensor):
        refuse_derivative(
            "witwin.radar.Pattern",
            "an angle axis is a host declaration: a knot decides which interpolation segment a direction lands "
            "in, so moving one is a different table rather than a derivative.",
            axis=values,
        )
        values = values.tolist()
    return tuple(float(v) for v in values)


def _values(values, *, complex_values: bool, rank: int):
    """Cut or table values: a tensor stays a tensor (a possible leaf), anything else becomes host tuples."""

    if isinstance(values, torch.Tensor):
        if values.dim() != rank:
            raise ValueError(f"pattern values must be a rank-{rank} tensor, got shape {tuple(values.shape)}")
        if complex_values != values.is_complex():
            kind = "a complex" if complex_values else "a real floating"
            raise TypeError(f"these pattern values must be {kind} tensor, got {values.dtype}")
        if not complex_values and not values.is_floating_point():
            raise TypeError(f"power pattern values must be a floating tensor, got {values.dtype}")
        return values
    convert = complex if complex_values else float
    if rank == 1:
        return tuple(convert(v) for v in values)
    return tuple(tuple(convert(v) for v in row) for row in values)


def _gain(gain):
    return gain if isinstance(gain, torch.Tensor) else float(gain)


def _host_values(values, *, name: str, complex_values: bool):
    """A host copy for validation only; the field itself is untouched."""

    if not isinstance(values, torch.Tensor):
        return values
    host = values.detach().to("cpu")
    if host.dim() == 0:
        return complex(host.item()) if complex_values else float(host.item())
    if not bool(torch.isfinite(torch.view_as_real(host) if host.is_complex() else host).all()):
        raise ValueError(f"Pattern.{name} must be finite")
    return host.tolist()


@dataclass(frozen=True, slots=True)
class Pattern:
    """One antenna element's pattern versus the two off-boresight angles.

    Build one with :meth:`isotropic`, :meth:`dipole`, :meth:`separable`,
    :meth:`table`, :meth:`field_separable`, :meth:`field_table` or
    :meth:`from_antenna` rather than by naming the fields.

    Angles are in degrees. The table is NORMALISED: a ``power`` pattern holds
    a linear power gain in ``[0, 1]`` and a ``field`` pattern a complex field
    amplitude with ``|F| <= 1``, whose phase is the element's phase response.
    ``gain`` is the element's absolute peak gain in dBi; the sensor stage
    applies it exactly once, as the voltage factor ``10^(gain/20)`` on each
    of the transmit and receive elements, so a radar with 10 dBi elements on
    both sides is 20 dB above the isotropic one and no other stage scales it.

    The direction into the table is differentiable, and the interpolation is
    piecewise linear (in power, or in the field's real and imaginary parts),
    so the response has an exact almost-everywhere derivative that the native
    kernel carries. A knot and the two support edges are genuine
    non-differentiabilities and the kernel returns the almost-everywhere value
    there.

    The table VALUES and ``gain`` may be leaves. Pass a float tensor (power) or
    a complex tensor (field) as the cut or table values, and a 0-dim float
    tensor as ``gain``; either may carry a gradient or a forward tangent. The
    response is LINEAR in the values with the interpolation weights as
    constants of the direction, so the derivative is exact everywhere, and the
    sensor-weight operator carries it natively in both modes (R-ADR-052). The
    angle axes are host floats: a knot position decides which segment a
    direction lands in, and moving one is a different table rather than a
    derivative, so a tensor axis is refused.

    The angles are the same two the pattern helpers use: with a direction
    expressed in the radar's LOCAL frame, ``x = atan2(v_x, -v_z)`` and
    ``y = atan2(v_y, -v_z)``, both in degrees. Outside the tabulated support the
    gain is exactly zero rather than the nearest tabulated value, which is a
    modelling choice (an antenna that does not radiate behind itself) rather
    than an extrapolation accident.
    """

    #: The fields that may be tensors carrying a derivative (their packed
    #: table values and the peak gain), and the AD modes each supports.
    LEAVES: ClassVar[dict[str, tuple[str, ...]]] = dict.fromkeys(
        ("x_gain", "y_gain", "gain_table", "gain"), ("jvp", "vjp")
    )

    kind: str
    #: Off-boresight angles of the x axis, degrees, strictly increasing.
    x_angles: tuple[float, ...]
    #: Off-boresight angles of the y axis, degrees, strictly increasing.
    y_angles: tuple[float, ...]
    #: Separable kind: one normalised value per x angle.
    x_gain: tuple | None = None
    #: Separable kind: one normalised value per y angle.
    y_gain: tuple | None = None
    #: Table kind: ``gain[y][x]``, normalised.
    gain_table: tuple | None = None
    #: ``"power"`` or ``"field"``: what the values are.
    quantity: str = PATTERN_QUANTITY_POWER
    #: Absolute peak gain, dBi.
    gain: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in PATTERN_KINDS:
            raise ValueError(f"kind must be one of {list(PATTERN_KINDS)}, got {self.kind!r}")
        if self.quantity not in PATTERN_QUANTITIES:
            raise ValueError(f"quantity must be one of {list(PATTERN_QUANTITIES)}, got {self.quantity!r}")
        for name in ("x_angles", "y_angles"):
            if any(isinstance(value, torch.Tensor) for value in getattr(self, name)):
                raise TypeError(
                    f"Pattern.{name} must be host floats: a knot decides which interpolation segment a "
                    "direction lands in, so moving one is a different table rather than a derivative"
                )
        if isinstance(self.gain, torch.Tensor) and (self.gain.dim() != 0 or not self.gain.is_floating_point()):
            raise ValueError("a tensor Pattern.gain must be a 0-dim float tensor, dBi")
        gain = _host_values(self.gain, name="Pattern.gain", complex_values=False)
        if not math.isfinite(float(gain)):
            raise ValueError("Pattern.gain must be finite, dBi")
        if len(self.x_angles) < 2 or len(self.y_angles) < 2:
            raise ValueError("both pattern axes need at least two samples")
        for name, axis in (("x_angles", self.x_angles), ("y_angles", self.y_angles)):
            if any(later <= earlier for earlier, later in zip(axis, axis[1:], strict=False)):
                raise ValueError(f"{name} must be strictly increasing")
        field = self.quantity == PATTERN_QUANTITY_FIELD
        if self.kind == PATTERN_KIND_SEPARABLE:
            if self.x_gain is None or self.y_gain is None:
                raise ValueError("a separable pattern needs x_gain and y_gain")
            x_values = _host_values(self.x_gain, name="x_gain", complex_values=field)
            y_values = _host_values(self.y_gain, name="y_gain", complex_values=field)
            if len(x_values) != len(self.x_angles):
                raise ValueError("x_gain must hold one value per x axis sample")
            if len(y_values) != len(self.y_angles):
                raise ValueError("y_gain must hold one value per y axis sample")
            rows: tuple = (x_values, y_values)
        else:
            if self.gain_table is None:
                raise ValueError("a table pattern needs gain_table")
            rows = tuple(_host_values(self.gain_table, name="gain_table", complex_values=field))
            if len(rows) != len(self.y_angles):
                raise ValueError("gain_table must hold one row per y axis sample")
            for row in rows:
                if len(row) != len(self.x_angles):
                    raise ValueError("each gain_table row needs one entry per x sample")
        for row in rows:
            for value in row:
                if self.quantity == PATTERN_QUANTITY_POWER:
                    if isinstance(value, complex):
                        raise ValueError("a power pattern is real; use quantity='field' for amplitude and phase")
                    if value < 0.0:
                        raise ValueError("a power gain must be non-negative")
                    if value > 1.0 + _NORMALISATION_SLACK:
                        raise ValueError(
                            f"a power pattern is normalised to at most 1, got {value}; put the peak in gain (dBi)"
                        )
                elif abs(value) > 1.0 + _NORMALISATION_SLACK:
                    raise ValueError(
                        f"a field pattern is normalised to |F| <= 1, got {abs(value)}; put the peak in gain (dBi)"
                    )

    def _fields(self) -> tuple:
        return (
            self.kind,
            self.x_angles,
            self.y_angles,
            self.x_gain,
            self.y_gain,
            self.gain_table,
            self.quantity,
            self.gain,
        )

    def __eq__(self, other: object) -> bool:
        # A tensor field compares by identity: two leaves holding equal numbers
        # are still two leaves, and elementwise == has no single truth value.
        if not isinstance(other, Pattern):
            return NotImplemented
        return all(
            a is b if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor) else a == b
            for a, b in zip(self._fields(), other._fields(), strict=True)
        )

    def __hash__(self) -> int:
        return hash(tuple(id(value) if isinstance(value, torch.Tensor) else value for value in self._fields()))

    @property
    def kind_code(self) -> int:
        return PATTERN_KIND_CODE[self.kind]

    @property
    def is_leaf_bearing(self) -> bool:
        """Whether a tensor holds the values or the gain: the packed buffer is then built on the tape."""

        return any(isinstance(value, torch.Tensor) for value in (self.x_gain, self.y_gain, self.gain_table, self.gain))

    @property
    def peak_amplitude(self) -> float:
        """``10^(gain/20)``: the element's peak voltage gain the kernel applies once."""

        return 10.0 ** (float(self.gain) / 20.0)

    @property
    def is_isotropic(self) -> bool:
        """Unit response in every direction at 0 dBi: the sensor stage is then the identity."""

        if self.is_leaf_bearing or self.kind != PATTERN_KIND_SEPARABLE or self.gain != 0.0:
            return False
        full = self.x_angles[0] <= -180.0 and self.x_angles[-1] >= 180.0
        full = full and self.y_angles[0] <= -180.0 and self.y_angles[-1] >= 180.0
        return full and all(value == 1.0 for row in (self.x_gain, self.y_gain) for value in row)

    @classmethod
    def isotropic(cls, *, gain: float = 0.0) -> Pattern:
        """Unit power gain in every direction: the stage as a proven no-op.

        Both axes span the whole range ``atan2`` can produce, so no direction
        falls outside the support and takes the zero-outside branch, and both
        values are ``1.0`` at each knot, so the interpolation returns exactly
        ``1.0``. This is the default for a radar that declares no pattern: an
        unchosen dipole attenuates every off-boresight return by a number
        nobody asked for.
        """

        return cls(
            kind=PATTERN_KIND_SEPARABLE,
            x_angles=(-180.0, 180.0),
            y_angles=(-180.0, 180.0),
            x_gain=(1.0, 1.0),
            y_gain=(1.0, 1.0),
            gain=_gain(gain),
        )

    @classmethod
    def dipole(cls, *, gain: float = 0.0) -> Pattern:
        """A half-wave dipole cut in both planes, over -90..90 degrees."""

        return cls(
            kind=PATTERN_KIND_SEPARABLE,
            x_angles=tuple(DEFAULT_DIPOLE_ANGLES_DEG),
            y_angles=tuple(DEFAULT_DIPOLE_ANGLES_DEG),
            x_gain=tuple(DEFAULT_DIPOLE_VALUES),
            y_gain=tuple(DEFAULT_DIPOLE_VALUES),
            gain=_gain(gain),
        )

    @classmethod
    def separable(cls, x_angles, x_gain, y_angles, y_gain, *, gain: float = 0.0) -> Pattern:
        """A power pattern whose two-dimensional gain is the product of two cuts."""

        return cls(
            kind=PATTERN_KIND_SEPARABLE,
            x_angles=_axis(x_angles),
            y_angles=_axis(y_angles),
            x_gain=_values(x_gain, complex_values=False, rank=1),
            y_gain=_values(y_gain, complex_values=False, rank=1),
            gain=_gain(gain),
        )

    @classmethod
    def table(cls, x_angles, y_angles, values, *, gain: float = 0.0) -> Pattern:
        """A power pattern tabulated on the full ``(y, x)`` grid."""

        return cls(
            kind=PATTERN_KIND_MAP,
            x_angles=_axis(x_angles),
            y_angles=_axis(y_angles),
            gain_table=_values(values, complex_values=False, rank=2),
            gain=_gain(gain),
        )

    @classmethod
    def field_separable(cls, x_angles, x_field, y_angles, y_field, *, gain: float = 0.0) -> Pattern:
        """A complex field pattern ``F(x) F(y)``: amplitude and phase per cut."""

        return cls(
            kind=PATTERN_KIND_SEPARABLE,
            x_angles=_axis(x_angles),
            y_angles=_axis(y_angles),
            x_gain=_values(x_field, complex_values=True, rank=1),
            y_gain=_values(y_field, complex_values=True, rank=1),
            quantity=PATTERN_QUANTITY_FIELD,
            gain=_gain(gain),
        )

    @classmethod
    def field_table(cls, x_angles, y_angles, field, *, gain: float = 0.0) -> Pattern:
        """A complex field pattern tabulated on the full ``(y, x)`` grid."""

        return cls(
            kind=PATTERN_KIND_MAP,
            x_angles=_axis(x_angles),
            y_angles=_axis(y_angles),
            gain_table=_values(field, complex_values=True, rank=2),
            quantity=PATTERN_QUANTITY_FIELD,
            gain=_gain(gain),
        )

    @classmethod
    def from_antenna(cls, antenna, *, x_angles=None, y_angles=None) -> Pattern:
        """Sample a ``witwin.core.AntennaPattern`` onto a field table.

        The authoring route for Core's custom patterns, which is what
        Channel's ``presets.tr38901_pattern`` and ``presets.dipole_pattern``
        return. Such a callable takes unit directions in the ELEMENT frame -
        boresight ``+x``, zenith ``+z`` - and returns the absolute complex
        field amplitude, peak gain included. The radar's local frame has
        boresight ``-z`` and up ``+y``, so a radar-local direction ``v`` is
        ``(-v_z, -v_x, v_y)`` in the element frame. The sampled table is
        normalised by its largest magnitude and that magnitude becomes
        ``gain`` (dBi), so the peak is applied once by the sensor stage.

        The default grid is one degree over the front hemisphere,
        ``-90..90`` on both axes; the response is zero outside it.
        """

        from witwin.core import AntennaPattern

        if not isinstance(antenna, AntennaPattern):
            raise TypeError(f"from_antenna needs a witwin.core.AntennaPattern, got {type(antenna).__name__}")
        if antenna.kind == "isotropic":
            return cls.isotropic()
        if antenna.kind != "custom":
            raise ValueError(
                f"AntennaPattern(kind={antenna.kind!r}) is a polarisation shape that Channel's transport "
                "applies itself; only a custom field callable can be sampled into a Radar element pattern"
            )
        xs = tuple(float(v) for v in (x_angles if x_angles is not None else DEFAULT_DIPOLE_ANGLES_DEG))
        ys = tuple(float(v) for v in (y_angles if y_angles is not None else DEFAULT_DIPOLE_ANGLES_DEG))
        x = torch.tensor(xs, dtype=torch.float64).deg2rad()
        y = torch.tensor(ys, dtype=torch.float64).deg2rad()
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        # Radar-local direction with atan2(v_x, -v_z) = x and atan2(v_y, -v_z) = y.
        local = torch.stack((torch.tan(grid_x), torch.tan(grid_y), -torch.ones_like(grid_x)), dim=-1)
        local = local / local.norm(dim=-1, keepdim=True)
        element = torch.stack((-local[..., 2], -local[..., 0], local[..., 1]), dim=-1)
        with torch.no_grad():
            values = torch.as_tensor(antenna.custom(element.to(torch.float32))).to(torch.complex128)
        if tuple(values.shape) != tuple(grid_x.shape):
            raise ValueError(f"the antenna callable returned shape {tuple(values.shape)} for {tuple(grid_x.shape)}")
        peak = float(values.abs().max())
        if not peak > 0.0:
            raise ValueError("the sampled antenna pattern is zero everywhere on the grid")
        normalised = (values / peak).tolist()
        return cls.field_table(xs, ys, normalised, gain=20.0 * math.log10(peak))

    def packed(
        self, *, device: torch.device | str = "cpu", amplitude_scale: torch.Tensor | None = None
    ) -> torch.Tensor:
        """The one float32 buffer the kernel indexes; layout in ``sensor_weight.cu``.

        ``[kind, quantity, num_x, num_y, peak_amplitude, 0, 0, 0, x_axis,
        y_axis, values]``; a power pattern's imaginary block is zero.

        A leaf-bearing pattern, or a transmit element that carries the radar's
        differentiable power as ``amplitude_scale`` (a 0-dim tensor whose value
        is exactly one), is assembled on the tape with a constant number of
        library calls, so the operator's gradient for the buffer reaches the
        leaves. Every other pattern takes the host route and is bit-identical.
        """

        if self.is_leaf_bearing or amplitude_scale is not None:
            return self._packed_on_tape(torch.device(device), amplitude_scale)
        num_x = len(self.x_angles)
        num_y = len(self.y_angles)
        header = [
            float(self.kind_code),
            float(PATTERN_QUANTITY_CODE[self.quantity]),
            float(num_x),
            float(num_y),
            self.peak_amplitude,
            0.0,
            0.0,
            0.0,
        ]
        if self.kind == PATTERN_KIND_SEPARABLE:
            values = [complex(v) for v in self.x_gain] + [complex(v) for v in self.y_gain]
        else:
            values = [complex(v) for row in self.gain_table for v in row]
        if self.kind == PATTERN_KIND_SEPARABLE:
            real = [v.real for v in values]
            imaginary = [v.imag for v in values]
            body = real + imaginary
        else:
            body = [v.real for v in values] + [v.imag for v in values]
        flat = header + list(self.x_angles) + list(self.y_angles) + body
        return torch.tensor(flat, dtype=torch.float32, device=torch.device(device))

    def _packed_on_tape(self, device: torch.device, amplitude_scale: torch.Tensor | None) -> torch.Tensor:
        """:meth:`packed` as a differentiable expression of the leaves (same layout, same numbers)."""

        field = self.quantity == PATTERN_QUANTITY_FIELD
        dtype = torch.complex64 if field else torch.float32

        def block(values) -> torch.Tensor:
            if isinstance(values, torch.Tensor):
                return values.to(device=device, dtype=dtype).reshape(-1)
            flat = [v for row in values for v in row] if self.kind == PATTERN_KIND_MAP else list(values)
            return torch.tensor(flat, dtype=dtype, device=device)

        if self.kind == PATTERN_KIND_SEPARABLE:
            values = torch.cat((block(self.x_gain), block(self.y_gain)))
        else:
            values = block(self.gain_table)
        real = values.real if field else values
        imaginary = values.imag if field else torch.zeros_like(values)
        gain = self.gain
        if isinstance(gain, torch.Tensor):
            peak = torch.pow(10.0, gain.to(device=device, dtype=torch.float32) / 20.0).reshape(1)
        else:
            peak = torch.tensor([self.peak_amplitude], dtype=torch.float32, device=device)
        if amplitude_scale is not None:
            peak = peak * amplitude_scale.to(device=device, dtype=torch.float32).reshape(1)
        head = torch.tensor(
            [
                float(self.kind_code),
                float(PATTERN_QUANTITY_CODE[self.quantity]),
                len(self.x_angles),
                len(self.y_angles),
            ],
            dtype=torch.float32,
            device=device,
        )
        tail = torch.tensor([0.0, 0.0, 0.0, *self.x_angles, *self.y_angles], dtype=torch.float32, device=device)
        return torch.cat((head, peak, tail, real, imaginary)).contiguous()


def _complex_matrix(name: str, rows):
    """A square, finite coupling matrix: host tuples, or a complex tensor kept as given (a possible leaf)."""

    if rows is None:
        return None
    host = rows.detach().cpu().tolist() if isinstance(rows, torch.Tensor) else rows
    matrix = tuple(tuple(complex(value) for value in row) for row in host)
    size = len(matrix)
    if size < 1 or any(len(row) != size for row in matrix):
        raise ValueError(f"Coupling.{name} must be a square matrix, one row and one column per element")
    if not all(math.isfinite(value.real) and math.isfinite(value.imag) for row in matrix for value in row):
        raise ValueError(f"Coupling.{name} must be finite")
    if isinstance(rows, torch.Tensor):
        if not rows.is_complex():
            raise TypeError(f"a tensor Coupling.{name} must be complex, got {rows.dtype}")
        return rows
    return matrix


@dataclass(frozen=True, slots=True)
class Coupling:
    """Mutual coupling between array elements, as port coupling matrices.

    ``tx[i][j]`` is the complex excitation element ``i`` receives when port
    ``j`` is driven with a unit signal, and ``rx[i][j]`` the complex response
    port ``i`` reads from a unit field on element ``j``: the identity is an
    uncoupled array, and an off-diagonal ``0.1`` is a -20 dB neighbour. The
    receive chain applies ``Y = C_tx^T H C_rx^T`` over the cube's (TX, RX) axes
    ahead of every active stage, so coupling mixes the echoes before thermal
    noise enters (R-ADR-031). Either side may be ``None``.

    A complex tensor matrix is a leaf (R-ADR-052): the coupling is two cuBLAS
    products whose derivative in both modes is the transposed products, so the
    chain carries it with no slot of its own.
    """

    #: The port coupling matrices may be complex tensors carrying a derivative.
    LEAVES: ClassVar[dict[str, tuple[str, ...]]] = {"tx": ("jvp", "vjp"), "rx": ("jvp", "vjp")}

    tx: tuple[tuple[complex, ...], ...] | torch.Tensor | None = None
    rx: tuple[tuple[complex, ...], ...] | torch.Tensor | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "tx", _complex_matrix("tx", self.tx))
        object.__setattr__(self, "rx", _complex_matrix("rx", self.rx))
        if self.tx is None and self.rx is None:
            raise ValueError("Coupling needs a tx or an rx matrix; an uncoupled array is coupling=None")

    def require_array(self, num_tx: int, num_rx: int) -> None:
        if self.tx is not None and len(self.tx) != num_tx:
            raise ValueError(f"Coupling.tx is {len(self.tx)} x {len(self.tx)} but the array has {num_tx} transmitters")
        if self.rx is not None and len(self.rx) != num_rx:
            raise ValueError(f"Coupling.rx is {len(self.rx)} x {len(self.rx)} but the array has {num_rx} receivers")


def watts_from_dbm(power_dbm: float) -> float:
    """``1e-3 * 10^(dBm/10)``: the ONE place transmit power becomes watts.

    The result fills a source endpoint's ``powers_w`` and reaches physics
    through that field and no other. There is deliberately no transmit voltage
    gain anywhere: a ``sqrt(P R)`` gain on a weight that already carries
    ``sqrt(P)`` counts the power twice and leaves the result in sqrt(W ohm)
    while the weight is in sqrt(W).
    """

    return 1e-3 * (10.0 ** (float(power_dbm) / 10.0))


#: A row that interacts at a site, which is every row the two-way join
#: publishes. The kernel's selector also knows a direct transmitter-to-receiver
#: row (code 1), whose length is ``|rx - tx|`` with no site term and therefore
#: no antenna angle; no production route builds one.
ROW_KIND_VIA = 0

#: Why the frozen tensors of the sensor-weight family refuse a derivative. They
#: are constants of the frame's row set - velocities and the fixed leg length -
#: and the native operator has no gradient or tangent slot for either, so a
#: marked one would run the whole frame and return ``None``. Refused at
#: CONSTRUCTION, before ``validate`` and before any launch. The pattern frame
#: and the packed pattern tables are differentiable inputs (R-ADR-052).
#: ``pattern_gain`` is a published OUTPUT that is ``mark_non_differentiable``,
#: which is a declaration rather than a silence.
_FROZEN_REASON = (
    "it is a frozen geometric constant of the frame's row set rather than a differentiable "
    "input, and the native sensor-weight operator has no gradient or tangent slot "
    "for it. The differentiable inputs here are tx_pos, rx_pos, site_in, site_out, "
    "intensity, the complex weight, the pattern frame and the packed pattern tables. "
    "A VELOCITY additionally has no leaf "
    "semantics under ADR-038: it is a forward-AD tangent direction that "
    "witwin.radar.propagation puts in the tangent slot of a position, so "
    "d(loss)/d(velocity) does not exist in either mode."
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
    """The row-set part of one frame's sensor description.

    Every field in :attr:`FROZEN_FIELDS` is a CONSTANT with respect to the
    derivative; the four position tensors are passed separately to the
    autograd entry so that they can carry a tangent or a gradient.
    ``__post_init__`` refuses a frozen float tensor that carries
    ``requires_grad`` or a forward tangent, naming the field, before
    :meth:`validate` and before any launch. ``pattern_frame`` is the
    exception: it follows the radar's orientation and the operator carries its
    derivative. The index tensors are deliberately not checked: they are
    ``int64`` / ``int32`` and cannot carry a derivative for autograd to lose.
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
    #: How many slot-major instants the launch carries. The per-row tables and
    #: the velocities describe ONE slot; ``num_tx``/``num_rx`` count every
    #: slot's antennas, and the kernel strides the tables across the slots.
    slots: int = 1

    #: The float tensors this owner declares constant, in declaration order.
    #: Named as data so that a field added to the dataclass without a decision
    #: about its derivative is visible as a gap rather than as silence.
    FROZEN_FIELDS = ("tx_velocity", "rx_velocity", "site_velocity", "fixed_length_m")

    def __post_init__(self) -> None:
        frozen = {name: getattr(self, name) for name in SensorWeightGeometry.FROZEN_FIELDS}
        refuse_derivative("witwin.radar.sensors.SensorWeightGeometry", _FROZEN_REASON, **frozen)

    @property
    def path_count(self) -> int:
        return int(self.fixed_length_m.shape[0]) * self.slots

    def validate(self) -> None:
        if type(self.slots) is not int or self.slots < 1:
            raise ValueError(f"slots must be a positive int, got {self.slots!r}")
        if self.num_tx % self.slots or self.num_rx % self.slots:
            raise ValueError(
                f"{self.num_tx} transmitters and {self.num_rx} receivers do not split into {self.slots} slots"
            )
        rows = int(self.fixed_length_m.shape[0])
        _require("tx_velocity", self.tx_velocity, dtype=torch.float32, shape=(self.num_tx // self.slots, 3))
        _require("rx_velocity", self.rx_velocity, dtype=torch.float32, shape=(self.num_rx // self.slots, 3))
        _require("site_velocity", self.site_velocity, dtype=torch.float32, shape=(rows, 3))
        _require("fixed_length_m", self.fixed_length_m, dtype=torch.float32, shape=(rows,))
        _require("tx_index", self.tx_index, dtype=torch.int64, shape=(rows,))
        _require("rx_index", self.rx_index, dtype=torch.int64, shape=(rows,))
        _require("row_kind", self.row_kind, dtype=torch.int32, shape=(rows,))
        _require("pattern_frame", self.pattern_frame, dtype=torch.float32, shape=(3, 3))


@dataclass(frozen=True, slots=True)
class SensorWeightPlan:
    """The scalars and the packed pattern tables one launch needs.

    The tables are the two packed element patterns the kernel interpolates.
    They are differentiable inputs of the operator: the response is linear in
    the table values (the interpolation weights are constants of the
    direction) and in each element's peak amplitude, and the operator carries
    both derivatives natively (R-ADR-052). A host-authored pattern packs to a
    plain constant buffer, exactly as before.
    """

    #: The packed transmit and receive element patterns, in that order.
    tables: tuple[torch.Tensor, torch.Tensor]
    c0: float
    #: Both elements isotropic at 0 dBi: the stage multiplies by exactly one.
    is_identity: bool = False

    @classmethod
    def build(
        cls,
        pattern: Pattern,
        *,
        rx_pattern: Pattern | None = None,
        c0: float = SPEED_OF_LIGHT_M_PER_S,
        device: torch.device | str = "cuda",
        transmit_scale: torch.Tensor | None = None,
    ) -> SensorWeightPlan:
        """``pattern`` is the transmit element; ``rx_pattern`` the receive one (default: the same).

        ``transmit_scale`` is the radar's differentiable transmit amplitude
        factor ``sqrt(P / P_host)``, exactly one in value, folded into the
        transmit element's peak (see ``Radar._transmit_amplitude_scale``).
        """

        receive = pattern if rx_pattern is None else rx_pattern
        return cls(
            tables=(pattern.packed(device=device, amplitude_scale=transmit_scale), receive.packed(device=device)),
            c0=float(c0),
            is_identity=pattern.is_isotropic and receive.is_isotropic and transmit_scale is None,
        )

    def kernel_tail(self, geometry: SensorWeightGeometry) -> tuple:
        """The trailing scalar arguments, in the order all three operators take.

        Written once because the three signatures agree by construction, and a
        divergence between them would be a silently different weight in the
        gradient than in the primal.
        """

        return (geometry.num_tx, geometry.num_rx, self.c0, geometry.slots)


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


class _SensorWeight(torch.autograd.Function):
    """Autograd bridge for the native sensor-weight operators.

    The pattern frame and the two packed tables are explicit inputs, so a
    caller whose radar pose or element pattern carries a leaf gets their
    derivatives; a launch whose frame and tables are constants runs exactly the
    position-only operators.
    """

    @staticmethod
    def forward(
        tx_pos,
        rx_pos,
        site_in,
        site_out,
        intensity,
        weight_re,
        weight_im,
        pattern_frame,
        tx_table,
        rx_table,
        tx_velocity,
        rx_velocity,
        geometry,
        plan,
    ):
        rows = int(site_in.shape[0])
        out_re = site_in.new_empty(rows)
        out_im = site_in.new_empty(rows)
        total_delay_s = site_in.new_empty(rows)
        delay_rate = site_in.new_empty(rows)
        pattern_gain = site_in.new_empty(rows)
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
            pattern_frame,
            tx_table,
            rx_table,
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
            pattern_frame,
            tx_table,
            rx_table,
            tx_velocity,
            rx_velocity,
            geometry,
            plan,
        ) = inputs
        ctx.geometry = geometry
        ctx.plan = plan
        saved = (
            tx_pos,
            rx_pos,
            site_in,
            site_out,
            intensity,
            weight_re,
            weight_im,
            pattern_frame,
            tx_table,
            rx_table,
            tx_velocity,
            rx_velocity,
        )
        ctx.save_for_backward(*saved)
        ctx.save_for_forward(*saved)
        # The pattern gain is a diagnostic, not a differentiable product: it is
        # published so a test can pin the interpolation against its Torch
        # oracle, and the weight is where its derivative actually goes.
        ctx.mark_non_differentiable(output[4])

    @staticmethod
    @first_order_only
    def backward(ctx, grad_out_re, grad_out_im, grad_tau_rt, grad_tau_rate, grad_gain):
        (
            tx_pos,
            rx_pos,
            site_in,
            site_out,
            intensity,
            weight_re,
            weight_im,
            pattern_frame,
            tx_table,
            rx_table,
            tx_velocity,
            rx_velocity,
        ) = ctx.saved_tensors
        geometry = ctx.geometry
        plan = ctx.plan
        rows = int(site_in.shape[0])
        zeros = site_in.new_zeros(rows)
        constants = (pattern_frame, tx_table, rx_table)
        grad_tx_pos = torch.empty_like(tx_pos)
        grad_rx_pos = torch.empty_like(rx_pos)
        grad_site_in = torch.empty_like(site_in)
        grad_site_out = torch.empty_like(site_out)
        # One value per batched row; the intensity table is one slot's.
        grad_intensity = site_in.new_empty(rows)
        grad_weight_re = torch.empty_like(weight_re)
        grad_weight_im = torch.empty_like(weight_im)
        tx_row_scratch = torch.empty_like(site_in)
        rx_row_scratch = torch.empty_like(site_in)
        want_frame, want_tx, want_rx = ctx.needs_input_grad[7:10]
        want_tables = want_tx or want_rx
        stride = (18 if want_tables else 0) + (9 if want_frame else 0)
        keys = torch.empty(rows * stride, dtype=torch.int64, device=site_in.device)
        values = torch.empty(rows * stride, dtype=torch.float32, device=site_in.device)
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
            *constants,
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
            keys,
            values,
            rows,
            *plan.kernel_tail(geometry),
            int(want_tables),
            int(want_frame),
        )
        if geometry.slots > 1:
            grad_intensity = grad_intensity.view(geometry.slots, -1).sum(dim=0)
        grad_frame = grad_tx_table = grad_rx_table = None
        if stride:
            grad_frame, grad_tx_table, grad_rx_table = _keyed_parameter_gradients(keys, values, tx_table, rx_table)
        return (
            grad_tx_pos,
            grad_rx_pos,
            grad_site_in,
            grad_site_out,
            grad_intensity,
            grad_weight_re,
            grad_weight_im,
            grad_frame if want_frame else None,
            grad_tx_table if want_tx else None,
            grad_rx_table if want_rx else None,
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
        tan_pattern_frame,
        tan_tx_table,
        tan_rx_table,
        tan_tx_velocity,
        tan_rx_velocity,
        tan_geometry,
        tan_plan,
    ):
        (
            tx_pos,
            rx_pos,
            site_in,
            site_out,
            intensity,
            weight_re,
            weight_im,
            pattern_frame,
            tx_table,
            rx_table,
            tx_velocity,
            rx_velocity,
        ) = ctx.saved_tensors
        geometry = ctx.geometry
        plan = ctx.plan
        rows = int(site_in.shape[0])
        row_zeros = site_in.new_zeros(rows)
        vector_zeros = torch.zeros_like(site_in)
        tan_tx_pos = torch.zeros_like(tx_pos) if tan_tx_pos is None else tan_tx_pos.contiguous()
        tan_rx_pos = torch.zeros_like(rx_pos) if tan_rx_pos is None else tan_rx_pos.contiguous()
        tan_site_in = vector_zeros if tan_site_in is None else tan_site_in.contiguous()
        tan_site_out = vector_zeros if tan_site_out is None else tan_site_out.contiguous()
        tan_intensity = torch.zeros_like(intensity) if tan_intensity is None else tan_intensity.contiguous()
        tan_weight_re = row_zeros if tan_weight_re is None else tan_weight_re.contiguous()
        tan_weight_im = row_zeros if tan_weight_im is None else tan_weight_im.contiguous()

        tan_out_re = site_in.new_empty(rows)
        tan_out_im = site_in.new_empty(rows)
        tan_tau_rt = site_in.new_empty(rows)
        tan_tau_rate = site_in.new_empty(rows)
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
            pattern_frame,
            tx_table,
            rx_table,
            tan_tx_pos,
            tan_rx_pos,
            tan_site_in,
            tan_site_out,
            tan_intensity,
            tan_weight_re,
            tan_weight_im,
            # An absent parameter tangent is an empty tensor: the kernel then
            # skips the parameter term, so a position-only jvp costs nothing more.
            _optional_tangent(tan_tx_table, tx_table),
            _optional_tangent(tan_rx_table, rx_table),
            _optional_tangent(tan_pattern_frame, pattern_frame),
            tan_out_re,
            tan_out_im,
            tan_tau_rt,
            tan_tau_rate,
            rows,
            *plan.kernel_tail(geometry),
        )
        return tan_out_re, tan_out_im, tan_tau_rt, tan_tau_rate, None


def _optional_tangent(tangent: torch.Tensor | None, primal: torch.Tensor) -> torch.Tensor:
    if tangent is None:
        return primal.new_empty(0)
    return tangent.to(dtype=torch.float32).contiguous()


def _keyed_parameter_gradients(
    keys: torch.Tensor, values: torch.Tensor, tx_table: torch.Tensor, rx_table: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reduce the backward's keyed contributions into the frame and table gradients.

    The native backward wrote each row's contributions as ``(key, value)``
    pairs over ``[tx_table | rx_table | frame]``; a stable sort groups them by
    parameter in ascending row order and ``keyed_sum`` adds each run in that
    order: deterministic, and no atomics (see ``sensor_weight.cu``). Returns
    ``(grad_frame (3, 3), grad_tx_table, grad_rx_table)``.
    """

    sorted_keys, order = torch.sort(keys, stable=True)
    tx_length = int(tx_table.numel())
    rx_length = int(rx_table.numel())
    total = torch.empty(tx_length + rx_length + 9, dtype=torch.float32, device=keys.device)
    _ops().keyed_sum_forward(sorted_keys, values[order].contiguous(), total)
    grad_tx, grad_rx, grad_frame = torch.split(total, [tx_length, rx_length, 9])
    return grad_frame.view(3, 3), grad_tx, grad_rx


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
    _require("intensity", intensity, dtype=torch.float32, shape=(rows // geometry.slots,))
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
        geometry.pattern_frame,
        plan.tables[0],
        plan.tables[1],
        geometry.tx_velocity,
        geometry.rx_velocity,
        geometry,
        plan,
    )
    return SensorWeightResult.from_components(out_re, out_im, total_delay_s, delay_rate, pattern_gain)


def _pattern_plan(
    pattern: Pattern, rx_pattern: Pattern | None, *, device: torch.device, transmit_scale: torch.Tensor | None
) -> SensorWeightPlan:
    for value in (pattern, rx_pattern):
        if value is not None and not isinstance(value, Pattern):
            raise TypeError(
                f"a sensor weight needs a witwin.radar.Pattern, got {type(value).__name__}; "
                "the radar's own are radar.pattern and radar.rx_pattern, and Pattern.isotropic() is the no-op"
            )
    return SensorWeightPlan.build(
        pattern, rx_pattern=rx_pattern, c0=SPEED_OF_LIGHT_M_PER_S, device=device, transmit_scale=transmit_scale
    )


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
    row_kind: torch.Tensor
    zero_rows: torch.Tensor
    zero_vectors: torch.Tensor
    unit_intensity: torch.Tensor
    tx_velocity: torch.Tensor
    rx_velocity: torch.Tensor
    pattern_frame: torch.Tensor
    plan: SensorWeightPlan

    @classmethod
    def freeze(
        cls, radar, composer, *, site_ids, pattern: Pattern, rx_pattern: Pattern | None = None
    ) -> RoundTripPatternStage:
        """Build the constant tables for one frozen :class:`TwoWayComposer`.

        ``pattern`` is the transmit element and ``rx_pattern`` the receive one
        (``None``: the same as ``pattern``). ``site_ids`` is the binding's host
        tuple of stable site IDs. It is checked against the composer's site
        count here rather than read back from the device column
        ``topology.site_id``: the host tuple is already there, and the device
        column is not.
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
        if len(site_ids) != composer.site_count:
            raise ValueError(
                f"the binding declares {len(site_ids)} sites but this join was frozen against {composer.site_count}"
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
            row_kind=torch.full((rows,), ROW_KIND_VIA, dtype=torch.int32, device=device),
            zero_rows=zero_rows,
            zero_vectors=torch.zeros(rows, 3, dtype=torch.float32, device=device),
            unit_intensity=torch.ones(rows, dtype=torch.float32, device=device),
            tx_velocity=torch.zeros(num_tx, 3, dtype=torch.float32, device=device),
            rx_velocity=torch.zeros(num_rx, 3, dtype=torch.float32, device=device),
            # A local direction is ``v @ world_from_local``, so the pattern-frame
            # components are dot products with that matrix's COLUMNS. The kernel
            # takes those columns as its rows; the transpose IS the frame change.
            pattern_frame=radar._world_from_local_matrix(device=device, dtype=torch.float32)[1]
            .transpose(0, 1)
            .contiguous(),
            plan=_pattern_plan(
                pattern, rx_pattern, device=device, transmit_scale=radar._transmit_amplitude_scale(device=device)
            ),
        )

    def _geometry(self, slots: int = 1) -> SensorWeightGeometry:
        return SensorWeightGeometry(
            num_tx=self.num_tx * slots,
            num_rx=self.num_rx * slots,
            tx_velocity=self.tx_velocity,
            rx_velocity=self.rx_velocity,
            site_velocity=self.zero_vectors,
            fixed_length_m=self.zero_rows,
            tx_index=self.tx_index,
            rx_index=self.rx_index,
            row_kind=self.row_kind,
            pattern_frame=self.pattern_frame,
            slots=slots,
        )

    def for_slots(self, count: int) -> RoundTripPatternStage:
        """Replicate routing for slot-major endpoints, sharing antenna tables."""

        slot = torch.arange(count, device=self.tx_index.device, dtype=torch.int64)[:, None]
        return replace(
            self,
            num_tx=self.num_tx * count,
            num_rx=self.num_rx * count,
            row_count=self.row_count * count,
            tx_index=(self.tx_index[None, :] + slot * self.num_tx).reshape(-1),
            rx_index=(self.rx_index[None, :] + slot * self.num_rx).reshape(-1),
            row_kind=self.row_kind.repeat(count),
            zero_rows=self.zero_rows.repeat(count),
            zero_vectors=self.zero_vectors.repeat(count, 1),
            unit_intensity=self.unit_intensity.repeat(count),
            tx_velocity=self.tx_velocity.repeat(count, 1),
            rx_velocity=self.rx_velocity.repeat(count, 1),
        )

    def apply_slots(
        self,
        rows: SlotRows,
        *,
        tx_pos: torch.Tensor,
        rx_pos: torch.Tensor,
        tx_targets_m: torch.Tensor,
        rx_targets_m: torch.Tensor,
    ) -> SlotRows:
        """:meth:`apply` for a slot-major batch, with this stage's one-slot tables.

        The native operator strides the routing tables across ``rows.slots``
        instants (the transmitter of batch row ``k`` is table row
        ``k % rows_per_slot`` offset by its slot), so no per-slot copy of a
        table is built; the weights are the bit for bit equal of
        :meth:`for_slots` followed by :meth:`apply`. ``tx_pos``/``rx_pos`` are
        the slot-major endpoint stacks and the targets are per batch row.
        """

        if rows.weight_includes_antenna_pattern:
            raise ValueError(
                "these rows already record weight_includes_antenna_pattern; applying the array pattern twice "
                "squares its gain and is refused"
            )
        if rows.rows_per_slot != self.row_count:
            raise ValueError(
                f"these rows carry {rows.rows_per_slot} paths per slot but this stage was frozen against "
                f"{self.row_count}; the batch does not belong to this frozen topology"
            )
        if self.plan.is_identity:
            return replace(rows, weight_includes_antenna_pattern=True)
        weight = evaluate_sensor_weights(
            tx_pos=tx_pos,
            rx_pos=rx_pos,
            site_in=tx_targets_m,
            site_out=rx_targets_m,
            intensity=self.unit_intensity,
            weight=rows.complex_transfer_ref,
            geometry=self._geometry(rows.slots),
            plan=self.plan,
        ).weight
        return replace(rows, complex_transfer_ref=weight, weight_includes_antenna_pattern=True)

    def apply(
        self,
        paths: RadarPathBatch,
        *,
        tx_pos: torch.Tensor,
        rx_pos: torch.Tensor,
        tx_targets_m: torch.Tensor,
        rx_targets_m: torch.Tensor,
    ) -> RadarPathBatch:
        """Publish ``paths`` with the transmit and receive pattern gains applied.

        ``tx_targets_m`` and ``rx_targets_m`` are per-composed-row points on
        the first and last propagation segments. For a reflected leg these are
        interaction points, not the scatter site. Gathering Channel's live
        geometry preserves derivatives through reflection points.

        Everything except the weight passes through untouched - the same
        objects, not copies - so row identity, row order, storage aliasing,
        dtype, device and the delay's gradient state all survive.
        """

        if not isinstance(paths, RadarPathBatch):
            raise TypeError(f"the antenna-pattern stage consumes a RadarPathBatch, got {type(paths).__name__}")
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
        if self.plan.is_identity:
            # Isotropic elements at 0 dBi multiply every row by exactly one and
            # contribute no direction derivative, so the launch is skipped and
            # the weight passes through as the same tensor.
            weight = paths.complex_transfer_ref
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
                weight_includes_antenna_pattern=True,
                total_delay_residual_s=paths.total_delay_residual_s,
            )
        geometry = self._geometry()
        weight = evaluate_sensor_weights(
            tx_pos=tx_pos,
            rx_pos=rx_pos,
            site_in=tx_targets_m,
            site_out=rx_targets_m,
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
            weight_includes_antenna_pattern=True,
            total_delay_residual_s=paths.total_delay_residual_s,
        )


__all__ = ["Coupling", "Pattern"]
