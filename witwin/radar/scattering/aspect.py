"""Aspect-dependent target response, evaluated per composed row in CUDA.

This is the third ingredient of the plan's micro-Doppler item. Scatter-site
VELOCITY is already carried: a site is an endpoint, so its ``omega x r`` or
deformation velocity reaches ``delay_rate`` through the propagation JVP. Aspect
change and target-specific phase are not, because they vary per PATH, and
``TwoWayComposer.compose`` refuses a geometry-dependent response outright:

    a geometry-dependent scatter response varies per path and must be
    evaluated in a native kernel, not composed here

That refusal is correct and it stays. This module is the route THROUGH it: the
response is evaluated by ``scatter_response_aspect_forward`` and its two AD
companions, from the direction basis the two legs already publish, and the
composer narrows its refusal to responses that are not on the explicit
:data:`~witwin.radar.scattering.base.NATIVE_ROW_RESPONSE_OWNERS` list.

**What a direction means here, and the one thing this response refuses.**
``RadarLegBatch.field_direction`` is the row's FINAL segment direction. For the
inbound leg, ``TX -> ... -> site``, that is the direction the field arrives at
the site travelling in, which is exactly the incidence direction, at any depth.
For the outbound leg, ``site -> ... -> RX``, it is the direction the field
arrives at the RECEIVER travelling in, which equals the departure direction
from the site only when the outbound row is line of sight. A higher-order
outbound row does not carry its own departure direction anywhere in the
published consumer contract, so this response REFUSES an outbound leg whose
frozen rows are not all line of sight, by name, at composition time and from a
host-known depth recorded at freeze. Reading the receiver-side direction as if
it were the departure direction would be a plausible number for a bistatic lobe
and wrong by the reflection angle.

**Aspect-phase rate.** If a response varies with aspect then ``d(arg S)/dt`` is
a second micro-Doppler term, and the join's ``tan_rate_rt = 0`` policy does not
carry it: that policy assumes the whole rate lives in ``tau_rt``. Phase 7 does
not fold it into ``delay_rate``. Following the pulsed spec's own precedent of
REFUSING range migration rather than approximating it, this response takes a
declared aspect phase rate and a coherent interval and refuses the pair when
the phase would walk more than the budget across one coherent interval. Like
``PulsedEchoSpec.max_expected_delay_rate`` the rate is DECLARED, not measured:
reducing over device rows to find a maximum is exactly the hot-path
device-to-host transfer the fixed-topology capability exists to avoid.

The separable law shipped here has a real, non-negative magnitude and a
per-site constant phase, so its OWN aspect phase rate is identically zero. The
declaration exists for the caller who composes a further phase onto it and for
this law's successors, and the guard is a contract rather than a measurement.
Carrying ``d(arg S)/dt`` properly is a separate numerical decision with its own
ADR; see R-ADR-013.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.autograd.function import once_differentiable


#: The law this response evaluates, stated once so a test can quote it.
#:
#: ``ci`` is negated because ``dir_in`` is a PROPAGATION direction and points
#: into the site; ``dir_out`` points away from it and enters directly. The
#: clamp is physical: a negative cosine is a direction on the far side of the
#: aspect plane, which a separable forward lobe does not illuminate.
ASPECT_SCATTER_LAW = (
    "S = amplitude * max(-dot(dir_in, axis), 0)^n "
    "* max(dot(dir_out, axis), 0)^n * exp(-i * phase_rad)"
)

#: The fully qualified name the composer checks against its owner list.
_OWNER = "witwin.radar.scattering.aspect.AspectScatterResponse"

_OPS = None


def _ops():
    global _OPS
    if _OPS is None:
        from ..cuda import build

        _OPS = build.build_extension()
    return _OPS


class _AspectResponse(torch.autograd.Function):
    """Autograd bridge for the three aspect-response operators.

    The same two structural contracts the join carries, for the same reasons:
    the facade always routes through ``Function.apply`` so an ADR-038
    forward-only dual is not swallowed by a ``requires_grad`` shortcut, and no
    complex tensor crosses the boundary - the real and imaginary parts are
    separate outputs.

    ``exponent`` is a host float and takes no gradient. It selects the law.
    """

    @staticmethod
    def forward(
        dir_in,
        dir_out,
        axis,
        amplitude,
        phase_rad,
        idx_in,
        idx_out,
        idx_site,
        row_valid,
        exponent,
        tables,
    ):
        rows = int(idx_in.shape[0])
        s_re = torch.empty(rows, dtype=torch.float32, device=dir_in.device)
        s_im = torch.empty_like(s_re)
        _ops().scatter_response_aspect_forward(
            dir_in,
            dir_out,
            idx_in,
            idx_out,
            idx_site,
            axis,
            amplitude,
            phase_rad,
            row_valid,
            s_re,
            s_im,
            exponent,
            rows,
        )
        return s_re, s_im

    @staticmethod
    def setup_context(ctx, inputs, output):
        (
            dir_in,
            dir_out,
            axis,
            amplitude,
            phase_rad,
            idx_in,
            idx_out,
            idx_site,
            row_valid,
            exponent,
            tables,
        ) = inputs
        ctx.exponent = exponent
        ctx.tables = tables
        saved = (
            dir_in,
            dir_out,
            axis,
            amplitude,
            phase_rad,
            idx_in,
            idx_out,
            idx_site,
            row_valid,
        )
        ctx.save_for_backward(*saved)
        ctx.save_for_forward(*saved)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_s_re, grad_s_im):
        (
            dir_in,
            dir_out,
            axis,
            amplitude,
            phase_rad,
            idx_in,
            idx_out,
            idx_site,
            row_valid,
        ) = ctx.saved_tensors
        tables = ctx.tables
        grad_dir_in = torch.empty_like(dir_in)
        grad_dir_out = torch.empty_like(dir_out)
        grad_axis = torch.empty_like(axis)
        grad_amplitude = torch.empty_like(amplitude)
        grad_phase_rad = torch.empty_like(phase_rad)
        _ops().scatter_response_aspect_backward(
            dir_in,
            dir_out,
            idx_in,
            idx_out,
            idx_site,
            axis,
            amplitude,
            phase_rad,
            row_valid,
            tables.by_in_offsets,
            tables.by_in_rows,
            tables.by_out_offsets,
            tables.by_out_rows,
            tables.by_site_offsets,
            tables.by_site_rows,
            grad_s_re.contiguous(),
            grad_s_im.contiguous(),
            grad_dir_in,
            grad_dir_out,
            grad_axis,
            grad_amplitude,
            grad_phase_rad,
            ctx.exponent,
            int(idx_in.shape[0]),
            int(dir_in.shape[0]),
            int(dir_out.shape[0]),
            int(axis.shape[0]),
        )
        return (
            grad_dir_in,
            grad_dir_out,
            grad_axis,
            grad_amplitude,
            grad_phase_rad,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    @staticmethod
    def jvp(
        ctx,
        tan_dir_in,
        tan_dir_out,
        tan_axis,
        tan_amplitude,
        tan_phase_rad,
        tan_idx_in,
        tan_idx_out,
        tan_idx_site,
        tan_row_valid,
        tan_exponent,
        tan_tables,
    ):
        (
            dir_in,
            dir_out,
            axis,
            amplitude,
            phase_rad,
            idx_in,
            idx_out,
            idx_site,
            row_valid,
        ) = ctx.saved_tensors

        def tangent(value, like):
            return torch.zeros_like(like) if value is None else value.contiguous()

        rows = int(idx_in.shape[0])
        tan_s_re = torch.empty(rows, dtype=torch.float32, device=dir_in.device)
        tan_s_im = torch.empty_like(tan_s_re)
        _ops().scatter_response_aspect_jvp(
            dir_in,
            dir_out,
            idx_in,
            idx_out,
            idx_site,
            axis,
            amplitude,
            phase_rad,
            row_valid,
            tangent(tan_dir_in, dir_in),
            tangent(tan_dir_out, dir_out),
            tangent(tan_axis, axis),
            tangent(tan_amplitude, amplitude),
            tangent(tan_phase_rad, phase_rad),
            tan_s_re,
            tan_s_im,
            ctx.exponent,
            rows,
        )
        return tan_s_re, tan_s_im


@dataclass(frozen=True, slots=True, eq=False)
class _BackwardTables:
    """The frozen CSR the reverse pass sums over, one owner family each.

    They are the JOIN's own tables, passed through rather than rebuilt, so the
    response's gradient sums in the same order the join's does and a
    bit-identical comparison across a permuted leg order stays legitimate.
    """

    by_in_offsets: torch.Tensor
    by_in_rows: torch.Tensor
    by_out_offsets: torch.Tensor
    by_out_rows: torch.Tensor
    by_site_offsets: torch.Tensor
    by_site_rows: torch.Tensor


def _require_parameter(name: str, value: object, *, sites: int, width: int | None):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} must use torch.float32, got {value.dtype}")
    expected = (sites,) if width is None else (sites, width)
    if tuple(value.shape) != expected:
        raise ValueError(f"{name} must have shape {expected}, got {tuple(value.shape)}")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    return value


@dataclass(frozen=True, slots=True, eq=False)
class AspectScatterResponse:
    """A separable bistatic lobe, one parameter set per scatter site.

    Every member but ``exponent`` may carry a gradient or a forward tangent:
    ``axis`` is the aspect the target presents, ``amplitude`` its strength and
    ``phase_rad`` its target-specific phase, authored in the CHANNEL convention
    ``exp(-j ...)`` so that it multiplies transports written there.

    ``axis`` is required to be a unit vector and is NOT normalised here. A
    kernel-side normalisation would add a division to every row and a quotient
    rule to both AD companions to hide a caller error that one check catches
    once, and a silently renormalised axis makes the gradient the caller reads
    back the gradient of a different parameterisation than the one it wrote.

    That unit-norm test is the ONE device read in this module: reducing the
    norms and testing them costs a device-to-host copy and a synchronization
    when ``axis`` is a CUDA tensor. It is paid deliberately HERE - at
    construction, which is freeze time, once per epoch - and never per frame,
    which is why it is a constructor check and not a kernel precondition. The
    per-frame host budget is asserted separately at zero. Everything else this
    class refuses is host state only.

    ``coherent_interval_s`` and ``aspect_phase_rate_rad_per_s`` are the
    aspect-rate guard, checked here at construction - which is freeze time, once
    per epoch, on the host - and refused by name. See the module docstring for
    why the rate is declared rather than measured.
    """

    axis: torch.Tensor
    amplitude: torch.Tensor
    phase_rad: torch.Tensor
    exponent: float
    coherent_interval_s: float
    aspect_phase_rate_rad_per_s: float = 0.0

    #: Read by ``TwoWayComposer.compose``. See ``NATIVE_ROW_RESPONSE_OWNERS``.
    native_row_owner = _OWNER

    def __post_init__(self) -> None:
        from ..synthesis.contracts import require_aspect_phase_rate_bounded

        axis = self.axis
        if not isinstance(axis, torch.Tensor):
            raise TypeError(f"axis must be a torch.Tensor, got {type(axis).__name__}")
        if axis.ndim != 2 or axis.shape[1] != 3:
            raise ValueError(f"axis must have shape (S, 3), got {tuple(axis.shape)}")
        sites = int(axis.shape[0])
        _require_parameter("axis", axis, sites=sites, width=3)
        _require_parameter("amplitude", self.amplitude, sites=sites, width=None)
        _require_parameter("phase_rad", self.phase_rad, sites=sites, width=None)
        for name in ("amplitude", "phase_rad"):
            if getattr(self, name).device != axis.device:
                raise ValueError(f"{name} must share the axis device {axis.device}")
        norm = torch.linalg.vector_norm(axis.detach(), dim=1)
        if not bool(torch.all((norm - 1.0).abs() < 1.0e-5)):
            raise ValueError(
                "axis must hold unit vectors; this response does not normalise "
                "them, because a silently renormalised axis returns the "
                "gradient of a different parameterisation than the caller wrote"
            )
        if not isinstance(self.exponent, float) or not self.exponent >= 1.0:
            raise ValueError(
                f"exponent must be a float of at least 1.0, got {self.exponent!r}"
            )
        require_aspect_phase_rate_bounded(
            self.aspect_phase_rate_rad_per_s, self.coherent_interval_s
        )

    @classmethod
    def from_values(
        cls,
        axis,
        amplitude,
        phase_rad,
        *,
        exponent: float,
        coherent_interval_s: float,
        aspect_phase_rate_rad_per_s: float = 0.0,
        device: torch.device | str = "cuda",
        requires_grad: bool = False,
    ) -> "AspectScatterResponse":
        """Build the response from host sequences, one row per site."""

        def parameter(values, width: int | None) -> torch.Tensor:
            tensor = torch.tensor(
                [list(row) for row in values] if width else list(values),
                dtype=torch.float32,
                device=device,
            )
            return tensor.requires_grad_(requires_grad)

        return cls(
            axis=parameter(axis, 3),
            amplitude=parameter(amplitude, None),
            phase_rad=parameter(phase_rad, None),
            exponent=float(exponent),
            coherent_interval_s=float(coherent_interval_s),
            aspect_phase_rate_rad_per_s=float(aspect_phase_rate_rad_per_s),
        )

    @property
    def site_count(self) -> int:
        return int(self.axis.shape[0])

    @property
    def is_geometry_dependent(self) -> bool:
        return True

    def evaluate(self, row_count: int, device: torch.device) -> torch.Tensor:
        """Refused: this response has no per-site value to broadcast.

        The ``ScatterResponse`` protocol's ``evaluate`` returns one value per
        SITE. An aspect-dependent response has none - its value is a function of
        the two directions of the composed row - so returning anything here
        would be a fabricated per-site average. The composer dispatches
        :meth:`evaluate_rows` instead.
        """

        raise NotImplementedError(
            "AspectScatterResponse is evaluated per composed row by "
            "scatter_response_aspect_forward; TwoWayComposer.compose "
            "dispatches evaluate_rows and there is no per-site value"
        )

    def evaluate_rows(self, composer, inbound, outbound, row_valid):
        """One complex value per composed row, as a real/imaginary pair.

        Host work only: shape and depth validation, then one kernel launch. The
        direction tables are the legs' own aliased tensors, so a gradient taken
        here reaches the endpoint positions the directions were built from.
        """

        if self.site_count != composer.site_count:
            raise ValueError(
                f"this response carries {self.site_count} sites but the join was "
                f"frozen against {composer.site_count}"
            )
        if composer.outbound_max_depth != 0:
            raise NotImplementedError(
                "an aspect-dependent response needs the DEPARTURE direction at "
                "the site, and a leg publishes the direction of its final "
                f"segment; this join's outbound leg reaches depth "
                f"{composer.outbound_max_depth}, whose published direction is "
                "the arrival direction at the receiver. Freeze the outbound leg "
                "with line-of-sight rows only, or use a response that does not "
                "depend on the scattering direction"
            )
        dir_in = _require_direction(inbound, composer.inbound_row_count, "inbound")
        dir_out = _require_direction(outbound, composer.outbound_row_count, "outbound")
        tables = _BackwardTables(
            by_in_offsets=composer.by_inbound_offsets,
            by_in_rows=composer.by_inbound_rows,
            by_out_offsets=composer.by_outbound_offsets,
            by_out_rows=composer.by_outbound_rows,
            by_site_offsets=composer.by_response_offsets,
            by_site_rows=composer.by_response_rows,
        )
        return _AspectResponse.apply(
            dir_in,
            dir_out,
            self.axis,
            self.amplitude,
            self.phase_rad,
            composer.inbound_row,
            composer.outbound_row,
            composer.response_slot,
            row_valid,
            self.exponent,
            tables,
        )


def _require_direction(leg, rows: int, name: str) -> torch.Tensor:
    direction = getattr(leg, "field_direction", None)
    if direction is None:
        raise ValueError(
            f"the {name} leg carries no field_direction, so an aspect-dependent "
            "response has no geometry to evaluate; every batch the Channel "
            "adapter publishes carries one"
        )
    if int(direction.shape[0]) != rows:
        raise ValueError(
            f"the {name} leg carries {int(direction.shape[0])} direction rows "
            f"but this join was frozen against {rows}"
        )
    return direction.contiguous()


__all__ = ["ASPECT_SCATTER_LAW", "AspectScatterResponse"]
