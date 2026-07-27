"""Python owner of the native ``sensor_weight`` family.

The per-path geometry, antenna gain, spreading, transmit power, and legacy
receive projection run entirely inside one CUDA kernel. Torch's role here is
validation, buffer allocation, autograd dispatch, and result assembly; it never
evaluates a distance, an angle, an interpolation, or a projection. Three
registered operators - forward, backward, jvp - have exactly one Python owner,
this module.

**The three mode flags come from the batch's provenance, not from a caller.**
That is the whole mechanism by which the single-count rule becomes a kernel
argument instead of a comment:

* ``spreading`` is set only when the weight does NOT already contain
  ``wavelength / (4 pi d)``. A Channel coefficient does, per leg, so a
  Channel-sourced batch physically cannot have it applied twice.
* ``tx_power`` is set only when the weight does NOT already contain
  ``sqrt(P_tx)``. A Channel coefficient does, from the source endpoint's
  ``powers_w``.
* ``legacy_real_polarization`` is set only when the weight carries no
  reference-frequency phase at all. That is precisely the real-amplitude route,
  which has a signed scalar rather than a Jones operator and therefore needs the
  mirrored projection as its substitute. A Channel-sourced weight has already
  been projected onto both endpoint polarizations, so a second projection here
  would be the same field projected twice.

:meth:`SensorWeightModes.from_provenance` reads those three booleans off the
batch. A caller may not assert its own provenance, because a caller that could
would assert the convenient one.

**The gradient of the antenna positions uses no atomics.** Many rows share one
transmitter and one receiver, so that gradient is a real reduction. The kernel
does it in a second pass over ascending rows with an explicit per-row scratch
buffer, which this module allocates. The summation order is then a property of
the frozen row set rather than of the schedule - the same choice the two-way
join makes for the same reason.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.autograd.forward_ad as forward_ad
from torch.autograd.function import once_differentiable

from .contracts import AntennaPatternSpec, SPEED_OF_LIGHT_M_PER_S

#: A row that interacts at a site, and a row that goes straight from a
#: transmitter to a receiver. The direct row's length is ``|rx - tx|`` with no
#: site term at all, rather than a via-row with a degenerate site, because a
#: zero-length second leg has no direction and therefore no antenna angle.
ROW_KIND_VIA = 0
ROW_KIND_DIRECT = 1


_OPS = None


def _ops():
    """The native operator table, resolved once per process."""

    global _OPS
    if _OPS is None:
        from ..cuda import build

        _OPS = build.build_extension()
    return _OPS


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
            "witwin.radar.propagation.kinematics puts in the tangent slot of "
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
class SensorWeightModes:
    """Which factors this owner applies, decided by what the weight carries."""

    spreading: bool
    tx_power: bool
    legacy_real_polarization: bool
    reflection_flip: bool = True

    @classmethod
    def from_provenance(cls, batch, *, reflection_flip: bool = True) -> "SensorWeightModes":
        """Read the three flags off a :class:`SynthesisPathBatch`'s provenance."""

        return cls(
            spreading=not bool(batch.weight_includes_spreading),
            tx_power=not bool(batch.weight_includes_tx_power),
            legacy_real_polarization=not bool(batch.weight_includes_reference_phase),
            reflection_flip=bool(reflection_flip),
        )


@dataclass(frozen=True, slots=True)
class SensorWeightGeometry:
    """The constant part of one frame's row set.

    Every field here is a CONSTANT with respect to the derivative except the
    four position tensors, which are passed separately to the autograd entry so
    that they can carry a tangent or a gradient. Velocities, normals,
    polarization vectors, and the local frame are constants by design: Phase 7
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
    normals: torch.Tensor
    pol_tx: torch.Tensor
    pol_rx: torch.Tensor
    local_axes: torch.Tensor

    #: The float tensors this owner declares constant, in declaration order.
    #: Named as data so that a field added to the dataclass without a decision
    #: about its derivative is visible as a gap rather than as silence.
    FROZEN_FIELDS = (
        "tx_velocity",
        "rx_velocity",
        "site_velocity",
        "fixed_length_m",
        "normals",
        "pol_tx",
        "pol_rx",
        "local_axes",
    )

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
        _require("normals", self.normals, dtype=torch.float32, shape=(rows, 3))
        _require("pol_tx", self.pol_tx, dtype=torch.float32, shape=(self.num_tx, 3))
        _require("pol_rx", self.pol_rx, dtype=torch.float32, shape=(self.num_rx, 3))
        _require("local_axes", self.local_axes, dtype=torch.float32, shape=(3, 3))


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
    wavelength_m: float
    tx_amplitude: float
    modes: SensorWeightModes

    def __post_init__(self) -> None:
        for index, table in enumerate(self.tables):
            if isinstance(table, torch.Tensor):
                _require_frozen_constant(
                    "SensorWeightPlan", f"tables[{index}]", table
                )

    @classmethod
    def build(
        cls,
        pattern: AntennaPatternSpec,
        *,
        modes: SensorWeightModes,
        wavelength_m: float,
        tx_amplitude: float = 1.0,
        c0: float = SPEED_OF_LIGHT_M_PER_S,
        device: torch.device | str = "cuda",
    ) -> "SensorWeightPlan":
        return cls(
            pattern_kind=pattern.kind_code,
            tables=pattern.tables(device=device),
            c0=float(c0),
            wavelength_m=float(wavelength_m),
            tx_amplitude=float(tx_amplitude),
            modes=modes,
        )

    def kernel_tail(self, geometry: SensorWeightGeometry) -> tuple:
        """The trailing scalar arguments, in the order all three operators take.

        Written once because the three signatures agree by construction, and a
        divergence between them would be a silently different weight in the
        gradient than in the primal.
        """

        return (
            geometry.num_tx,
            geometry.num_rx,
            self.pattern_kind,
            self.c0,
            self.wavelength_m,
            self.tx_amplitude,
            1 if self.modes.spreading else 0,
            1 if self.modes.tx_power else 0,
            1 if self.modes.legacy_real_polarization else 0,
            1 if self.modes.reflection_flip else 0,
        )


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
    ) -> "SensorWeightResult":
        return cls(
            weight=torch.complex(out_re, out_im),
            total_delay_s=total_delay_s,
            delay_rate=delay_rate,
            pattern_gain=pattern_gain,
        )


def _row_constants(geometry: SensorWeightGeometry) -> tuple:
    """The constant row tensors that sit BEFORE ``intensity`` in every signature."""

    return (
        geometry.site_velocity,
        geometry.fixed_length_m,
        geometry.tx_index,
        geometry.rx_index,
        geometry.row_kind,
    )


def _sensor_constants(geometry: SensorWeightGeometry, plan: SensorWeightPlan) -> tuple:
    """The constant sensor tensors that sit AFTER the weight in every signature."""

    return (geometry.normals, geometry.pol_tx, geometry.pol_rx, geometry.local_axes) + plan.tables


class _SensorWeight(torch.autograd.Function):
    """Autograd bridge for the three native sensor-weight operators."""

    @staticmethod
    def forward(
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
        saved = (
            tx_pos,
            rx_pos,
            site_in,
            site_out,
            intensity,
            weight_re,
            weight_im,
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
    @once_differentiable
    def backward(ctx, grad_out_re, grad_out_im, grad_tau_rt, grad_tau_rate, grad_gain):
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
        ) = ctx.saved_tensors
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
        ) = ctx.saved_tensors
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
    return SensorWeightResult.from_components(
        out_re, out_im, total_delay_s, delay_rate, pattern_gain
    )


__all__ = [
    "ROW_KIND_DIRECT",
    "ROW_KIND_VIA",
    "SensorWeightGeometry",
    "SensorWeightModes",
    "SensorWeightPlan",
    "SensorWeightResult",
    "evaluate_sensor_weights",
]
