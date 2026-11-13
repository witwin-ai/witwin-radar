"""Radar target scattering responses."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch

from .cuda import native_ops as _ops
from .policy import SPEED_OF_LIGHT_M_PER_S, first_order_only, refuse_derivative

#: Maximum unmodelled aspect-response phase walk over one coherent interval.
ASPECT_PHASE_BUDGET_RAD = 0.1


def require_aspect_phase_rate_bounded(aspect_phase_rate_rad_per_s: float, coherent_interval_s: float) -> None:
    """Refuse an aspect phase whose coherent-interval walk exceeds the budget."""

    rate = float(aspect_phase_rate_rad_per_s)
    interval = float(coherent_interval_s)
    if rate < 0.0:
        raise ValueError(f"aspect_phase_rate_rad_per_s is a magnitude bound and cannot be negative, got {rate}")
    if not interval > 0.0:
        raise ValueError(f"coherent_interval_s must be positive, got {interval}")
    walk = rate * interval
    if walk >= ASPECT_PHASE_BUDGET_RAD:
        raise ValueError(
            "unmodelled aspect Doppler: the scatter response's argument walks "
            f"by |d(arg S)/dt| * T_frame = {walk} rad over the coherent "
            f"interval, which is not below ASPECT_PHASE_BUDGET_RAD="
            f"{ASPECT_PHASE_BUDGET_RAD}. The two-way join publishes no rate "
            "for the response's argument, so that phase would simply be "
            "dropped and the target's Doppler would be "
            "understated by an amount no output reports. Shorten the coherent "
            "interval, slow the aspect change, or accept a response whose "
            "argument is aspect independent - there is no approximated mode"
        )


@runtime_checkable
class ScatterResponse(Protocol):
    """A complex response evaluated for a batch of composed rows.

    The returned factor is authored in the CHANNEL phasor convention,
    ``exp(-j k d)``, because it multiplies transports authored there. The
    conversion to the beat convention happens once, downstream, in the
    synthesis facade.

    A scalar response scatters as ``s`` times the identity dyadic (R-ADR-041).
    A polarimetric response additionally declares ``matrix_width = 4``,
    evaluates to ``complex64[rows, 4]`` in :data:`SCATTERING_MATRIX_LAYOUT`
    order and publishes its sites' up axes through ``site_up``;
    :class:`ScatteringMatrixResponse` is that interface.
    """

    def evaluate(self, site_count: int, device: torch.device) -> torch.Tensor:
        """Return ``complex64[site_count]``, one value per site in the join's site order."""
        ...

    @property
    def is_geometry_dependent(self) -> bool:
        """Whether the response varies per path rather than per target.

        A geometry-dependent response is per-path physics and must be
        evaluated in a native kernel. This flag exists so that a future
        implementation cannot quietly become Torch hot-path physics while
        still satisfying the protocol.
        """
        ...


#: The complete set of geometry-dependent responses the two-way composer will
#: dispatch, named by their fully qualified class path.
#:
#: ``TwoWayComposer.compose`` refuses a geometry-dependent response, because
#: such a response is per-path physics and composing it in Torch is exactly the
#: thing the refusal exists to stop. This list NARROWS that refusal to
#: everything not on it. Membership is a claim that the
#: named class evaluates its rows in a native kernel; a response that merely
#: declares ``is_geometry_dependent`` and grows an ``evaluate_rows`` method is
#: still refused, because a protocol check can only see the method's name and
#: not what runs behind it.
#:
#: The list is deliberately explicit rather than an ``isinstance`` against a
#: base class: a subclass of the native response can override ``evaluate_rows``
#: with a Torch expression and would inherit the permission with it.
NATIVE_ROW_RESPONSE_OWNERS = frozenset(
    {
        "witwin.radar.scattering.AspectScatterResponse",
        "witwin.radar.scattering.FacetScatterResponse",
        "witwin.radar.scattering.FluctuatingResponse",
        "witwin.radar.scattering.TableScatterResponse",
    }
)


# The normalisation that makes ``|C_rt|^2`` the bistatic radar equation.
#
# A composed two-way coefficient is
#
#   |C_rt|^2 = P_in (lam/(4 pi d_in))^2 |S|^2 P_site (lam/(4 pi d_out))^2
#
# and the bistatic radar equation is
#
#   P_r = P_t G_t G_r lam^2 sigma / ((4 pi)^3 d_in^2 d_out^2)
#
# With the site excited at exactly 1 W, matching the two requires
#
#   |S|^2 = 4 pi sigma / lam^2,   i.e.   S = sqrt(4 pi sigma) / lam
#
# Omitting the ``4 pi / lam^2`` factor is a level error of ``lam^2 / (4 pi)``,
# which at 77 GHz is a factor of 6.6e5, or 58 dB.
def rcs_amplitude(
    sigma_m2: float | Sequence[float] | torch.Tensor, wavelength_m: float
) -> float | list[float] | torch.Tensor:
    """``sqrt(4 pi sigma) / lambda``, the dimensionless target strength.

    One value, or one per site: a 1-dim tensor or a host sequence gives the
    per-site strengths of a :class:`ScalarRcsResponse` that is still isotropic
    at every site.

    Dimensionless is the whole content of the normalisation. ``S`` carries no
    propagation phase and no spreading - both belong to Channel transport, once
    per leg - so what is left of a radar cross section after the two
    ``lam/(4 pi d)`` factors have been accounted for is a pure ratio.

    ``sigma_m2`` may be a 0-dim tensor, and then the returned amplitude carries
    its graph. A radar cross section is the canonical inverse-design leaf -
    "how big does this target have to look" - and it is the ONE configuration
    scalar in this package that is genuine scene state rather than a device or
    waveform declaration, which is why it is supported where
    :mod:`witwin.radar.policy` refuses everything else.

    Two things this is NOT. It is not hot-path physics: it runs once per
    response, off the per-path loop, and produces a single number that the
    response broadcasts. And it is not a second numerical owner: the ``sqrt``
    is result CONSTRUCTION, and every per-path product downstream of it is
    still evaluated by a native kernel. The mechanism is recorded as
    ``torch-orchestration`` in the capability matrix for exactly that reason.

    The derivative is the elementary one, and a test asserts it through the
    whole chain rather than only here::

        d(amplitude)/d(sigma) = 0.5 * sqrt(4 pi) / (lambda * sqrt(sigma))
                              = 0.5 * amplitude / sigma

    **It is unbounded at ``sigma = 0`` and that is a property of the
    parameterisation, not a defect to clamp.** The tensor route deliberately
    does NOT range check its input: a value check is a host read, and this
    module is inside the import boundary's no-host-observation scan precisely
    so that a per-frame construction cannot hide a synchronisation. A
    non-positive tensor therefore produces ``nan`` or ``inf``, which
    propagates visibly through the entire cube rather than becoming a
    plausible number. An optimiser that has to reach zero should drive the
    already-supported ``amplitude`` leaf, where the map is linear, or carry
    ``log sigma``. The host-float route refuses a negative value, because
    there is no derivative there to be wrong about.
    """

    if not wavelength_m > 0.0:
        raise ValueError("wavelength_m must be positive")
    if isinstance(sigma_m2, torch.Tensor):
        if sigma_m2.ndim > 1:
            raise ValueError(
                "a tensor sigma_m2 must be a 0-dim scalar or a 1-dim per-site vector, got rank "
                f"{sigma_m2.ndim}; ScalarRcsResponse is one complex number per site, broadcast across that "
                "site's rows"
            )
        return torch.sqrt(4.0 * math.pi * sigma_m2) / float(wavelength_m)
    if not isinstance(sigma_m2, (int, float)):
        values = [float(value) for value in sigma_m2]
        if any(value < 0.0 for value in values):
            raise ValueError("sigma_m2 is a radar cross section in square metres and cannot be negative")
        return [math.sqrt(4.0 * math.pi * value) / float(wavelength_m) for value in values]
    if sigma_m2 < 0.0:
        raise ValueError("sigma_m2 is a radar cross section in square metres and cannot be negative")
    return math.sqrt(4.0 * math.pi * float(sigma_m2)) / float(wavelength_m)


@dataclass(frozen=True, slots=True, eq=False)
class ScalarRcsResponse:
    """Complex target response ``S = amplitude * exp(-j * phase_rad)``.

    Each member is a 0-dim tensor, one value shared by every site, or a 1-dim
    tensor with one value per site in ascending stable-ID order, the join's own
    site slot order. The two may differ in rank: a shared phase with per-site
    strengths is the common case. Both may carry gradients. The response stays
    isotropic at every site; what a vector adds is that two targets of
    different size need no aspect law to say so.

    The phase parameter is here on purpose. An amplitude-only response would
    still pass every magnitude test with the Channel-to-beat conjugation
    inverted; the phase gradient is what actually witnesses that the phase tape
    survives the conjugation boundary and the native synthesis.

    ``S`` is authored in the Channel convention, ``exp(-j ...)``, matching the
    transports it multiplies.
    """

    amplitude: torch.Tensor
    phase_rad: torch.Tensor

    def __post_init__(self) -> None:
        for name in ("amplitude", "phase_rad"):
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if value.ndim > 1:
                raise ValueError(f"{name} must be a 0-dim tensor or a 1-dim per-site tensor, got rank {value.ndim}")
            if value.dtype != torch.float32:
                raise TypeError(f"{name} must use torch.float32, got {value.dtype}")
        if self.amplitude.ndim == 1 and self.phase_rad.ndim == 1 and self.amplitude.shape != self.phase_rad.shape:
            raise ValueError(
                f"per-site amplitude {tuple(self.amplitude.shape)} and phase_rad {tuple(self.phase_rad.shape)} "
                "must name the same sites"
            )

    @property
    def site_count(self) -> int | None:
        """The number of sites a per-site member names, or ``None`` when both are shared."""

        for value in (self.amplitude, self.phase_rad):
            if value.ndim == 1:
                return int(value.shape[0])
        return None

    @classmethod
    def from_values(
        cls,
        amplitude: float | Sequence[float],
        phase_rad: float | Sequence[float],
        *,
        device: torch.device | str = "cpu",
        requires_grad: bool = False,
    ) -> ScalarRcsResponse:
        def parameter(value: float | Sequence[float]) -> torch.Tensor:
            host = float(value) if isinstance(value, (int, float)) else [float(entry) for entry in value]
            tensor = torch.tensor(host, dtype=torch.float32, device=device)
            return tensor.requires_grad_(requires_grad)

        return cls(amplitude=parameter(amplitude), phase_rad=parameter(phase_rad))

    @classmethod
    def from_rcs(
        cls,
        sigma_m2: float | Sequence[float] | torch.Tensor,
        *,
        reference_frequency_hz: float,
        phase_rad: float | Sequence[float] = 0.0,
        device: torch.device | str = "cpu",
        requires_grad: bool = False,
    ) -> ScalarRcsResponse:
        """Build ``S`` from a radar cross section, through the pinned law.

        This is the only constructor that knows what a square metre is worth.
        ``from_values`` still exists because a test or an optimiser may want to
        author the dimensionless strength directly, but a caller that has a
        cross section must come through here rather than guess the
        normalisation: the guess that omits ``4 pi / lam^2`` is 58 dB out at
        77 GHz and looks entirely plausible on a relative plot.

        **A 0-dim ``sigma_m2`` tensor makes the cross section itself a leaf.**
        The amplitude is then ``sqrt(4 pi sigma) / lambda`` with its graph
        intact, so the derivative composes with everything the already-covered
        ``amplitude`` leaf reaches: the join, the waveform kernels, the cube.
        This is the inverse-design question a radar caller actually asks: how
        large does this target have to be.

        Two consequences of the tensor route, both deliberate:

        * ``requires_grad=True`` is REFUSED with a tensor cross section. The
          leaf is ``sigma_m2``, which the caller already marked; marking the
          derived amplitude as well is not expressible - it is not a leaf - and
          Torch's own error for it names neither this constructor nor the law.
        * the placement follows the tensor. ``device`` selects where a
          host-float response is built and cannot move a live one without
          breaking its graph, so the phase is placed beside the amplitude.
        """

        wavelength_m = SPEED_OF_LIGHT_M_PER_S / float(reference_frequency_hz)
        amplitude = rcs_amplitude(sigma_m2, wavelength_m)
        if not isinstance(amplitude, torch.Tensor):
            return cls.from_values(amplitude, phase_rad, device=device, requires_grad=requires_grad)
        if requires_grad:
            raise ValueError(
                "requires_grad=True is not meaningful with a tensor sigma_m2: "
                "the amplitude is derived from it and is not a leaf, so there "
                "is nothing here to mark. Mark sigma_m2 itself - the "
                "derivative then reaches this response through "
                "sqrt(4 pi sigma) / lambda - or use from_values to author the "
                "dimensionless strength as its own leaf."
            )
        phase = float(phase_rad) if isinstance(phase_rad, (int, float)) else [float(value) for value in phase_rad]
        return cls(amplitude=amplitude, phase_rad=torch.tensor(phase, dtype=torch.float32, device=amplitude.device))

    @property
    def is_geometry_dependent(self) -> bool:
        return False

    def evaluate(self, site_count: int, device: torch.device) -> torch.Tensor:
        """One complex value per site: a shared value broadcast, or the per-site vector.

        A per-site member that names a different number of sites than the join
        froze is refused here rather than gathered past its end.

        ``device`` is honoured, not decorative. The composer passes the device
        its composed rows live on; a CPU-authored response left where it was
        would fail with a device-mismatch error several frames of stack away
        from the parameter that caused it. ``Tensor.to`` is autograd-aware, so
        a response whose parameters carry gradients keeps them across the move.
        """

        if site_count < 0:
            raise ValueError("site_count must be non-negative")
        declared = self.site_count
        if declared is not None and declared != site_count:
            raise ValueError(
                f"this response carries {declared} per-site values but the join was frozen against {site_count} sites"
            )
        amplitude = self.amplitude.to(device=device, dtype=torch.complex64)
        phase = self.phase_rad.to(device=device, dtype=torch.complex64)
        return (amplitude * torch.exp(-1j * phase)).expand(site_count)


#: The site-local polarization basis a scattering matrix is authored in.
#:
#: Backscatter alignment (BSA). For a wave propagating INTO the site along the
#: unit ``k`` and the site's unit up axis ``u``:
#:
#:     h = (u x k) / |u x k|        v = h x k
#:
#: The incident basis uses the inbound leg's final-segment direction ``k_i``
#: and the receive basis the outbound leg's ``k_r`` (the outbound leg runs from
#: the receiver to the site, R-ADR-039), so a monostatic pair shares one basis
#: and the round-trip voltage is ``V = sum_ab S_ab (H . b_r,a) (b_i,b . E)``.
#: Where ``u`` is parallel to ``k`` the kernel substitutes ``x`` (or ``y`` when
#: ``k`` is along ``x``) for ``u``.
SITE_BASIS = "bsa:h=normalize(up x k),v=h x k"

#: The column order of a scattering matrix: receive index first.
SCATTERING_MATRIX_LAYOUT = ("HH", "HV", "VH", "VV")


@dataclass(frozen=True, slots=True, eq=False)
class ScatteringMatrixResponse:
    """A per-site 2x2 scattering matrix in the site's local BSA basis.

    ``matrix`` is ``complex64[sites, 4]`` in :data:`SCATTERING_MATRIX_LAYOUT`
    order (``HV`` = receive H, transmit V) or ``complex64[4]`` shared by every
    site; ``up`` is ``float32[sites, 3]`` or ``float32[3]``, each site's up axis
    in world coordinates (:data:`SITE_BASIS`). Each element is the
    dimensionless strength :func:`rcs_amplitude` defines, so
    ``|S_pq|^2 = 4 pi sigma_pq / lambda^2``, authored in the Channel phasor
    convention.

    This is the interface every polarimetric response produces: the composer
    reads ``matrix_width == 4``, :meth:`evaluate` and :meth:`site_up`. The
    scalar responses are its special cases - a ``ScalarRcsResponse`` of
    strength ``s`` scatters as ``s`` times the identity dyadic, which on any
    monostatic geometry equals the matrix ``diag(s, s)`` here.

    ``matrix`` may carry gradients; ``up`` is a frame declaration and is
    refused as a derivative input.
    """

    matrix: torch.Tensor
    up: torch.Tensor

    matrix_width = 4

    def __post_init__(self) -> None:
        if not isinstance(self.matrix, torch.Tensor) or self.matrix.dtype != torch.complex64:
            raise TypeError("matrix must be a complex64 torch.Tensor")
        if self.matrix.shape[-1:] != (4,) or self.matrix.ndim not in (1, 2):
            raise ValueError(f"matrix must have shape (4,) or (sites, 4), got {tuple(self.matrix.shape)}")
        if not isinstance(self.up, torch.Tensor) or self.up.dtype != torch.float32:
            raise TypeError("up must be a float32 torch.Tensor")
        if self.up.shape[-1:] != (3,) or self.up.ndim not in (1, 2):
            raise ValueError(f"up must have shape (3,) or (sites, 3), got {tuple(self.up.shape)}")
        refuse_derivative("site polarization basis", "the site's up axis is a frame declaration", up=self.up)

    @classmethod
    def from_values(
        cls,
        hh: complex,
        hv: complex,
        vh: complex,
        vv: complex,
        *,
        up: tuple[float, float, float] = (0.0, 0.0, 1.0),
        device: torch.device | str = "cpu",
        requires_grad: bool = False,
    ) -> ScatteringMatrixResponse:
        """One matrix of dimensionless strengths, shared by every site."""

        matrix = torch.tensor([hh, hv, vh, vv], dtype=torch.complex64, device=device).requires_grad_(requires_grad)
        return cls(matrix=matrix, up=torch.tensor(up, dtype=torch.float32, device=device))

    @classmethod
    def from_rcs(
        cls,
        sigma_m2: float,
        *,
        reference_frequency_hz: float,
        kind: str = "sphere",
        up: tuple[float, float, float] = (0.0, 0.0, 1.0),
        device: torch.device | str = "cpu",
    ) -> ScatteringMatrixResponse:
        """A canonical scatterer of cross section ``sigma_m2`` in every channel it lights.

        ``kind="sphere"`` (and any odd-bounce scatterer: plate, trihedral) is
        ``diag(a, a)``; ``kind="dihedral"`` (even bounce, fold along h) is
        ``diag(a, -a)``, so its HH and VV differ by 180 degrees. ``a`` is
        :func:`rcs_amplitude` of ``sigma_m2``.
        """

        amplitude = rcs_amplitude(float(sigma_m2), SPEED_OF_LIGHT_M_PER_S / float(reference_frequency_hz))
        signs = {"sphere": 1.0, "dihedral": -1.0}
        if kind not in signs:
            raise ValueError(f"kind must be one of {sorted(signs)}, got {kind!r}")
        return cls.from_values(amplitude, 0.0, 0.0, signs[kind] * amplitude, up=up, device=device)

    @property
    def is_geometry_dependent(self) -> bool:
        return False

    def evaluate(self, row_count: int, device: torch.device) -> torch.Tensor:
        """``complex64[row_count, 4]`` on ``device``."""

        if row_count < 0:
            raise ValueError("row_count must be non-negative")
        matrix = self.matrix.to(device=device)
        return matrix.expand(row_count, 4) if matrix.ndim == 1 else matrix

    def site_up(self, row_count: int, device: torch.device) -> torch.Tensor:
        """``float32[row_count, 3]`` unit up axes on ``device``."""

        up = self.up.to(device=device)
        up = up / torch.linalg.vector_norm(up, dim=-1, keepdim=True)
        return up.expand(row_count, 3).contiguous() if up.ndim == 1 else up.contiguous()


#: The fully qualified name the composer checks against its owner list.
_OWNER = "witwin.radar.scattering.AspectScatterResponse"


class _ScatterDirection(torch.autograd.Function):
    """Native unit departure bearing, with the same Jacobian in both AD modes."""

    @staticmethod
    def forward(origin, target):
        out = torch.empty_like(origin)
        _ops().scatter_direction_forward(origin.contiguous(), target.contiguous(), out)
        return out

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(*inputs)
        ctx.save_for_forward(*inputs)

    @staticmethod
    @first_order_only
    def backward(ctx, gradient):
        origin, target = ctx.saved_tensors
        projected = torch.empty_like(origin)
        _ops().scatter_direction_backward(origin.contiguous(), target.contiguous(), gradient.contiguous(), projected)
        return -projected, projected

    @staticmethod
    def jvp(ctx, d_origin, d_target):
        origin, target = ctx.saved_tensors
        d_origin = torch.zeros_like(origin) if d_origin is None else d_origin
        d_target = torch.zeros_like(target) if d_target is None else d_target
        out = torch.empty_like(origin)
        _ops().scatter_direction_jvp(origin.contiguous(), target.contiguous(), (d_target - d_origin).contiguous(), out)
        return out


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
    def forward(dir_in, dir_out, axis, amplitude, phase_rad, idx_in, idx_out, idx_site, row_valid, exponent, tables):
        rows = int(row_valid.shape[0])
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
            tables.slots,
            tables.slot_local_sites,
        )
        return s_re, s_im

    @staticmethod
    def setup_context(ctx, inputs, output):
        (dir_in, dir_out, axis, amplitude, phase_rad, idx_in, idx_out, idx_site, row_valid, exponent, tables) = inputs
        ctx.exponent = exponent
        ctx.tables = tables
        saved = (dir_in, dir_out, axis, amplitude, phase_rad, idx_in, idx_out, idx_site, row_valid)
        ctx.save_for_backward(*saved)
        ctx.save_for_forward(*saved)

    @staticmethod
    @first_order_only
    def backward(ctx, grad_s_re, grad_s_im):
        (dir_in, dir_out, axis, amplitude, phase_rad, idx_in, idx_out, idx_site, row_valid) = ctx.saved_tensors
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
            int(row_valid.shape[0]),
            int(dir_in.shape[0]),
            int(dir_out.shape[0]),
            int(axis.shape[0]),
            tables.slots,
            tables.slot_local_sites,
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
        (dir_in, dir_out, axis, amplitude, phase_rad, idx_in, idx_out, idx_site, row_valid) = ctx.saved_tensors

        def tangent(value, like):
            return torch.zeros_like(like) if value is None else value.contiguous()

        rows = int(row_valid.shape[0])
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
            ctx.tables.slots,
            ctx.tables.slot_local_sites,
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
    #: The slot stride: how many slots the batch carries, and whether each
    #: slot has its own sites (1) or every slot shares them (0).
    slots: int = 1
    slot_local_sites: int = 0


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
            raise ValueError(f"exponent must be a float of at least 1.0, got {self.exponent!r}")
        require_aspect_phase_rate_bounded(self.aspect_phase_rate_rad_per_s, self.coherent_interval_s)

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
    ) -> AspectScatterResponse:
        """Build the response from host sequences, one row per site."""

        def parameter(values, width: int | None) -> torch.Tensor:
            tensor = torch.tensor(
                [list(row) for row in values] if width else list(values), dtype=torch.float32, device=device
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

    def evaluate(self, site_count: int, device: torch.device) -> torch.Tensor:
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

    def evaluate_rows(self, composer, inbound, outbound, row_valid, *, slots: int = 1):
        """One complex value per composed row, as a real/imaginary pair.

        ``slots`` is the slot count of a slot-major batch: the composer's
        tables describe one slot and the kernel strides them (sites shared by
        every slot, or one set per slot for a time-resolved axis).

        Host work only: shape and depth validation, then one kernel launch. The
        direction tables are the legs' own aliased tensors, so a gradient taken
        here reaches the endpoint positions the directions were built from:
        Channel ADR-043 keeps ``field_direction`` live for ``{los, reflection}``
        under a frozen topology, which covers every component the Radar adapter
        may freeze.
        """

        local = _slot_local_sites(self.site_count, composer, slots, "aspect response")
        dir_in = _require_direction(inbound, composer.inbound_row_count * slots, "inbound")
        return _AspectResponse.apply(
            dir_in,
            _site_departure(outbound),
            self.axis,
            self.amplitude,
            self.phase_rad,
            composer.inbound_row,
            composer.outbound_row,
            composer.response_slot,
            row_valid,
            self.exponent,
            _row_tables(composer, slots=slots, slot_local_sites=local),
        )


class _FacetResponse(torch.autograd.Function):
    """Autograd bridge for the facet response operators.

    The differentiable inputs are the two legs' final-segment directions, the
    facet corners and the float64 site positions; everything in ``law``
    selects the law and takes no derivative. The output is the row's 2x2
    matrix as ``[rows, 4, 2]`` real pairs and the row's site-basis up axis,
    which is a frame declaration and carries no derivative.
    """

    @staticmethod
    def forward(k_in, k_out, corners, sites, idx_in, idx_out, idx_site, row_valid, law, tables):
        rows = int(row_valid.shape[0])
        matrix = torch.empty((rows, 4, 2), dtype=torch.float32, device=k_in.device)
        up = torch.empty((rows, 3), dtype=torch.float32, device=k_in.device)
        _ops().facet_response_forward(
            law.origin_in,
            k_in,
            k_out,
            idx_in,
            idx_out,
            idx_site,
            corners,
            sites,
            *law.tables(),
            row_valid,
            matrix,
            up,
            law.wavenumber,
            law.seed,
            tables.slots,
        )
        return matrix, up

    @staticmethod
    def setup_context(ctx, inputs, output):
        (k_in, k_out, corners, sites, idx_in, idx_out, idx_site, row_valid, law, tables) = inputs
        ctx.law = law
        ctx.tables = tables
        ctx.mark_non_differentiable(output[1])
        saved = (k_in, k_out, corners, sites, idx_in, idx_out, idx_site, row_valid)
        ctx.save_for_backward(*saved)
        ctx.save_for_forward(*saved)

    @staticmethod
    @first_order_only
    def backward(ctx, grad_matrix, grad_up):
        (k_in, k_out, corners, sites, idx_in, idx_out, idx_site, row_valid) = ctx.saved_tensors
        law, tables = ctx.law, ctx.tables
        grads = [torch.empty_like(value) for value in (k_in, k_out, corners, sites)]
        _ops().facet_response_backward(
            law.origin_in, k_in, k_out, idx_in, idx_out, idx_site, corners, sites, *law.tables(), row_valid,
            tables.by_in_offsets, tables.by_in_rows, tables.by_out_offsets, tables.by_out_rows,
            tables.by_site_offsets, tables.by_site_rows, law.facet_site, grad_matrix.contiguous(), *grads,
            law.wavenumber, law.seed, tables.slots,
        )  # fmt: skip
        return (*grads, None, None, None, None, None, None)

    @staticmethod
    def jvp(ctx, tan_k_in, tan_k_out, tan_corners, tan_sites, *_):
        (k_in, k_out, corners, sites, idx_in, idx_out, idx_site, row_valid) = ctx.saved_tensors
        law = ctx.law

        def tangent(value, like):
            return torch.zeros_like(like) if value is None else value.contiguous()

        tan_matrix = torch.empty((int(row_valid.shape[0]), 4, 2), dtype=torch.float32, device=k_in.device)
        _ops().facet_response_jvp(
            law.origin_in, k_in, k_out, idx_in, idx_out, idx_site, corners, sites, *law.tables(), row_valid,
            tangent(tan_k_in, k_in), tangent(tan_k_out, k_out), tangent(tan_corners, corners),
            tangent(tan_sites, sites), tan_matrix, law.wavenumber, law.seed, ctx.tables.slots,
        )  # fmt: skip
        return tan_matrix, None


@dataclass(frozen=True, slots=True, eq=False)
class _FacetLaw:
    """The non-differentiable half of one frame's facet response: tables and law scalars."""

    site_offsets: torch.Tensor
    site_facets: torch.Tensor
    facet_site: torch.Tensor
    facet_material: torch.Tensor
    material_params: torch.Tensor
    layer_offsets: torch.Tensor
    layers: torch.Tensor
    wavenumber: float
    seed: int
    #: Each inbound leg row's last point before the site (the gate against a
    #: facet illuminating itself); a discrete test, so no derivative.
    origin_in: torch.Tensor

    def tables(self) -> tuple[torch.Tensor, ...]:
        return (
            self.site_offsets,
            self.site_facets,
            self.facet_material,
            self.material_params,
            self.layer_offsets,
            self.layers,
        )


@dataclass(frozen=True, slots=True, eq=False)
class FacetScatterResponse:
    """Physical-optics facet scattering of a mesh target (R-ADR-042).

    One 2x2 scattering matrix per composed row (``matrix_width = 4``): the
    projection onto the join's site basis of the coherent sum, over the facets
    of the row's site, of each facet's vector physical-optics dyadic - area,
    exact flat-facet form factor, the facet material's TE and TM reflection
    coefficients and the coherent roughness loss - plus a Lambertian diffuse
    term that carries the power roughness removes from the coherent one. The
    equation, its symbols and units, its normalisation and its validity
    domain are in ``facet_scatter.cu`` and R-ADR-042; this class is
    orchestration only.

    The record built from a target carries the law - the carrier wavenumber
    ``k`` in rad/m and the diffuse seed - and no geometry. The session binds it
    to each frame's facets with :meth:`bind`; an unbound response refuses to
    evaluate.
    """

    wavenumber_rad_per_m: float
    seed: int = 0
    frame: object | None = None

    native_row_owner = "witwin.radar.scattering.FacetScatterResponse"
    matrix_width = 4

    def __post_init__(self) -> None:
        if not float(self.wavenumber_rad_per_m) > 0.0:
            raise ValueError("wavenumber_rad_per_m must be positive")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError(f"seed must be a non-negative int below 2**63, got {self.seed!r}")

    @property
    def is_geometry_dependent(self) -> bool:
        return True

    def bind_observation(self, binding: Any, time_s: float) -> FacetScatterResponse:
        """This response against the binding's facets at one observation."""

        if binding.facets is None:
            raise ValueError("a facet response needs a binding whose sites are a mesh target's facets")
        return self.bind(binding.facets)

    def bind(self, frame: object) -> FacetScatterResponse:
        """This response against one frame's facets (a :class:`witwin.radar.facets.FacetFrame`)."""

        return FacetScatterResponse(wavenumber_rad_per_m=self.wavenumber_rad_per_m, seed=self.seed, frame=frame)

    def evaluate(self, site_count: int, device: torch.device) -> torch.Tensor:
        """Refused: a facet response is a function of each row's two directions."""

        raise NotImplementedError(
            "FacetScatterResponse is evaluated per composed row by facet_response_forward; "
            "TwoWayComposer.compose dispatches evaluate_rows and there is no per-site value"
        )

    def evaluate_rows(self, composer, inbound, outbound, row_valid, *, slots: int = 1):
        """``([rows, 4, 2] matrix pairs, [rows, 3] up axes)`` for every composed row of every slot.

        The directions are the legs' final-segment directions into the site,
        the same tensors the join builds its site basis from. Host work only.
        """

        frame: Any = self.frame
        if frame is None:
            raise ValueError("a facet response must be bound to a frame's facets before it is evaluated")
        model = frame.model
        if model.site_count != composer.site_count:
            raise ValueError(
                f"this frame carries {model.site_count} facet sites but the join was frozen against "
                f"{composer.site_count}"
            )
        law = _FacetLaw(
            site_offsets=model.site_offsets,
            site_facets=model.site_facets,
            facet_site=model.facet_site,
            facet_material=model.facet_material,
            material_params=model.material_params,
            layer_offsets=model.layer_offsets,
            layers=model.layers,
            wavenumber=float(self.wavenumber_rad_per_m),
            seed=int(self.seed),
            origin_in=_present_points(inbound.arrival_origin_m, "arrival_origin_m").detach().contiguous(),
        )
        return _FacetResponse.apply(
            _require_direction(inbound, composer.inbound_row_count * slots, "inbound"),
            _require_direction(outbound, composer.outbound_row_count * slots, "outbound"),
            frame.corners,
            frame.sites,
            composer.inbound_row,
            composer.outbound_row,
            composer.response_slot,
            row_valid,
            law,
            _row_tables(composer, slots=slots, slot_local_sites=0),
        )


class _TableResponse(torch.autograd.Function):
    """Autograd bridge for the tabulated response; the table takes no derivative."""

    @staticmethod
    def forward(dir_in, dir_out, axis, idx_in, idx_out, idx_site, row_valid, table_re, table_im, tables):
        rows = int(row_valid.shape[0])
        s_re = torch.empty(rows, dtype=torch.float32, device=dir_in.device)
        s_im = torch.empty_like(s_re)
        _ops().scatter_response_table_forward(
            dir_in, dir_out, idx_in, idx_out, idx_site, axis, table_re, table_im, row_valid, s_re, s_im, rows,
            tables.slots, tables.slot_local_sites,
        )  # fmt: skip
        return s_re, s_im

    @staticmethod
    def setup_context(ctx, inputs, output):
        (dir_in, dir_out, axis, idx_in, idx_out, idx_site, row_valid, table_re, table_im, tables) = inputs
        ctx.tables = tables
        saved = (dir_in, dir_out, axis, idx_in, idx_out, idx_site, row_valid, table_re, table_im)
        ctx.save_for_backward(*saved)
        ctx.save_for_forward(*saved)

    @staticmethod
    @first_order_only
    def backward(ctx, grad_re, grad_im):
        (dir_in, dir_out, axis, idx_in, idx_out, idx_site, row_valid, table_re, table_im) = ctx.saved_tensors
        tables = ctx.tables
        grads = [torch.empty_like(value) for value in (dir_in, dir_out, axis)]
        _ops().scatter_response_table_backward(
            dir_in,
            dir_out,
            idx_in,
            idx_out,
            idx_site,
            axis,
            table_re,
            table_im,
            row_valid,
            tables.by_in_offsets,
            tables.by_in_rows,
            tables.by_out_offsets,
            tables.by_out_rows,
            tables.by_site_offsets,
            tables.by_site_rows,
            grad_re.contiguous(),
            grad_im.contiguous(),
            *grads,
            int(row_valid.shape[0]),
            tables.slots,
            tables.slot_local_sites,
        )
        return (*grads, None, None, None, None, None, None, None)

    @staticmethod
    def jvp(ctx, tan_dir_in, tan_dir_out, tan_axis, *_):
        (dir_in, dir_out, axis, idx_in, idx_out, idx_site, row_valid, table_re, table_im) = ctx.saved_tensors

        def tangent(value, like):
            return torch.zeros_like(like) if value is None else value.contiguous()

        rows = int(row_valid.shape[0])
        tan_re = torch.empty(rows, dtype=torch.float32, device=dir_in.device)
        tan_im = torch.empty_like(tan_re)
        _ops().scatter_response_table_jvp(
            dir_in,
            dir_out,
            idx_in,
            idx_out,
            idx_site,
            axis,
            table_re,
            table_im,
            row_valid,
            tangent(tan_dir_in, dir_in),
            tangent(tan_dir_out, dir_out),
            tangent(tan_axis, axis),
            tan_re,
            tan_im,
            rows,
            ctx.tables.slots,
            ctx.tables.slot_local_sites,
        )
        return tan_re, tan_im


@dataclass(frozen=True, slots=True, eq=False)
class TableScatterResponse:
    """A tabulated response ``S(theta)`` per composed row (R-ADR-045).

    ``theta`` is the bistatic aspect angle between a site's unit ``axis`` and
    the bisector of its two leg directions; ``table_re``/``table_im`` sample
    ``S = sqrt(4 pi sigma) / lambda exp(-j phi)`` at the carrier uniformly on
    ``[0, pi]``, in the Channel phasor convention. The table is host data; the
    axis and the leg directions carry derivatives. Like
    :class:`AspectScatterResponse`, the unit-norm check of ``axis`` is the one
    construction-time device read.
    """

    axis: torch.Tensor
    table_re: torch.Tensor
    table_im: torch.Tensor

    native_row_owner = "witwin.radar.scattering.TableScatterResponse"

    def __post_init__(self) -> None:
        if not isinstance(self.axis, torch.Tensor) or self.axis.ndim != 2 or self.axis.shape[1] != 3:
            raise ValueError("axis must be an (S, 3) tensor")
        sites = int(self.axis.shape[0])
        _require_parameter("axis", self.axis, sites=sites, width=3)
        entries = int(self.table_re.shape[0]) if isinstance(self.table_re, torch.Tensor) else 0
        if entries < 2:
            raise ValueError("the response table needs at least two aspect samples")
        _require_parameter("table_re", self.table_re, sites=entries, width=None)
        _require_parameter("table_im", self.table_im, sites=entries, width=None)
        if self.table_re.requires_grad or self.table_im.requires_grad:
            raise ValueError("a tabulated response is host data and takes no derivative; differentiate the axis")
        norm = torch.linalg.vector_norm(self.axis.detach(), dim=1)
        if not bool(torch.all((norm - 1.0).abs() < 1.0e-5)):
            raise ValueError("axis must hold unit vectors")

    @property
    def site_count(self) -> int:
        return int(self.axis.shape[0])

    @property
    def is_geometry_dependent(self) -> bool:
        return True

    def evaluate(self, site_count: int, device: torch.device) -> torch.Tensor:
        """Refused: a tabulated response is a function of each row's aspect."""

        raise NotImplementedError(
            "TableScatterResponse is evaluated per composed row by scatter_response_table_forward; "
            "TwoWayComposer.compose dispatches evaluate_rows and there is no per-site value"
        )

    def evaluate_rows(self, composer, inbound, outbound, row_valid, *, slots: int = 1):
        local = _slot_local_sites(self.site_count, composer, slots, "tabulated response")
        return _TableResponse.apply(
            _require_direction(inbound, composer.inbound_row_count * slots, "inbound"),
            _site_departure(outbound),
            self.axis,
            composer.inbound_row,
            composer.outbound_row,
            composer.response_slot,
            row_valid,
            self.table_re,
            self.table_im,
            _row_tables(composer, slots=slots, slot_local_sites=local),
        )


@dataclass(frozen=True, slots=True, eq=False)
class FluctuatingResponse:
    """A response whose sites' power fluctuates by Swerling draws (R-ADR-045).

    ``degrees`` is 2 (Swerling I, II) or 4 (III, IV); ``per_observation``
    redraws at every observation instant (II, IV) instead of once per
    ``coherent_interval_s`` (I, III). The session binds it to each observation
    with :meth:`bind_observation`, which draws the per-site amplitude factor
    ``sqrt(X)`` natively from a key of the seed and the instant (or the
    interval's start), mixed the way the frontend mixes its frame keys. The
    factor multiplies the base response's per-site or per-row value: one
    tensor product, through which the base response keeps its derivatives.
    """

    base: Any
    degrees: int
    per_observation: bool
    coherent_interval_s: float
    seed: int = 0
    factor: torch.Tensor | None = None

    native_row_owner = "witwin.radar.scattering.FluctuatingResponse"
    #: Read by the session: every observation is bound on its own, never slot batched.
    varies_per_observation = True

    @property
    def is_geometry_dependent(self) -> bool:
        return bool(self.base.is_geometry_dependent)

    def key(self, time_s: float) -> int:
        """The draw key of the observation at ``time_s``."""

        from .frontend import frame_seed

        if self.per_observation:
            return frame_seed(self.seed, float(time_s))
        interval = math.floor(float(time_s) / self.coherent_interval_s)
        return frame_seed(self.seed, interval * self.coherent_interval_s)

    def bind_observation(self, binding: Any, time_s: float) -> FluctuatingResponse:
        bind = getattr(self.base, "bind_observation", None)
        base = self.base if bind is None else bind(binding, time_s)
        factor = torch.empty(int(binding.site_count), dtype=torch.float32, device=binding.device)
        _ops().fluctuation_factor_forward(factor, self.key(time_s), self.degrees)
        return FluctuatingResponse(
            base=base,
            degrees=self.degrees,
            per_observation=self.per_observation,
            coherent_interval_s=self.coherent_interval_s,
            seed=self.seed,
            factor=factor,
        )

    def _factor(self) -> torch.Tensor:
        if self.factor is None:
            raise ValueError("a fluctuating response must be bound to an observation before it is evaluated")
        return self.factor

    @property
    def matrix_width(self) -> int:
        return int(getattr(self.base, "matrix_width", 1))

    def site_up(self, site_count: int, device: torch.device) -> torch.Tensor:
        return self.base.site_up(site_count, device)

    def evaluate(self, site_count: int, device: torch.device) -> torch.Tensor:
        value = self.base.evaluate(site_count, device)
        factor = self._factor().to(device=device)
        return value * (factor if value.ndim == 1 else factor[:, None])

    def evaluate_rows(self, composer, inbound, outbound, row_valid, *, slots: int = 1):
        if slots != 1:
            raise ValueError("a fluctuating response is bound per observation and is never slot batched")
        rows_re, rows_im = self.base.evaluate_rows(composer, inbound, outbound, row_valid)
        factor = self._factor().index_select(0, composer.response_slot)
        return rows_re * factor, rows_im * factor


def _row_tables(composer, *, slots: int, slot_local_sites: int) -> _BackwardTables:
    """The join's own single-slot CSR tables and the slot stride a row family reads them with."""

    return _BackwardTables(
        by_in_offsets=composer.by_inbound_offsets,
        by_in_rows=composer.by_inbound_rows,
        by_out_offsets=composer.by_outbound_offsets,
        by_out_rows=composer.by_outbound_rows,
        by_site_offsets=composer.by_response_offsets,
        by_site_rows=composer.by_response_rows,
        slots=int(slots),
        slot_local_sites=int(slot_local_sites),
    )


def _slot_local_sites(sites: int, composer, slots: int, owner: str) -> int:
    """0 when the response's sites are the join's, shared by every slot; 1 when it carries one set per slot."""

    if sites == composer.site_count:
        return 0
    if slots > 1 and sites == composer.site_count * slots:
        return 1
    raise ValueError(
        f"this {owner} carries {sites} sites but the join was frozen against {composer.site_count}"
        + (f" in each of {slots} slots" if slots > 1 else "")
    )


def _site_departure(outbound) -> torch.Tensor:
    """The unit bearing from each outbound row's site to the last point before it.

    The outbound leg runs from the receiver TOWARD the site (R-ADR-039), so
    its last segment arrives at the site and the site's departure bearing is
    that segment reversed, at every depth.
    """

    if outbound.arrival_target_m is None or outbound.arrival_origin_m is None:
        raise ValueError("a per-row response needs the outbound leg's arrival segment endpoints")
    return _ScatterDirection.apply(outbound.arrival_target_m, outbound.arrival_origin_m)


def _present_points(points: torch.Tensor | None, name: str) -> torch.Tensor:
    if points is None:
        raise ValueError(f"a facet response needs the inbound leg's {name}")
    return points


def _require_direction(leg, rows: int, name: str) -> torch.Tensor:
    direction = leg.field_direction
    if int(direction.shape[0]) != rows:
        raise ValueError(
            f"the {name} leg carries {int(direction.shape[0])} direction rows but this join was frozen against {rows}"
        )
    return direction.contiguous()


__all__ = [
    "SCATTERING_MATRIX_LAYOUT",
    "SITE_BASIS",
    "AspectScatterResponse",
    "FacetScatterResponse",
    "FluctuatingResponse",
    "ScalarRcsResponse",
    "ScatterResponse",
    "ScatteringMatrixResponse",
    "TableScatterResponse",
]
