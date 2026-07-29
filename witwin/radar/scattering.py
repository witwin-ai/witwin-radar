"""Radar target scattering responses."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

from .cuda import native_ops as _ops
from .policy import first_order_only

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
            f"{ASPECT_PHASE_BUDGET_RAD}. The two-way join publishes "
            "tan_rate_rt = 0 and carries the whole rate in tau_rt, so that "
            "phase would simply be dropped and the target's Doppler would be "
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
    """

    def evaluate(self, row_count: int, device: torch.device) -> torch.Tensor:
        """Return ``complex64[row_count]``."""
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
#: thing the refusal exists to stop. Phase 7 does not delete that refusal - it
#: NARROWS it, to everything not on this list. Membership is a claim that the
#: named class evaluates its rows in a native kernel; a response that merely
#: declares ``is_geometry_dependent`` and grows an ``evaluate_rows`` method is
#: still refused, because a protocol check can only see the method's name and
#: not what runs behind it.
#:
#: The list is deliberately explicit rather than an ``isinstance`` against a
#: base class: a subclass of the native response can override ``evaluate_rows``
#: with a Torch expression and would inherit the permission with it.
NATIVE_ROW_RESPONSE_OWNERS = frozenset({"witwin.radar.scattering.AspectScatterResponse"})


@runtime_checkable
class NativeRowScatterResponse(Protocol):
    """A geometry-dependent response the composer is allowed to dispatch.

    It publishes one complex value per COMPOSED row rather than one per site,
    and it evaluates them in a native kernel from the direction basis the two
    legs carry. ``native_row_owner`` is its own fully qualified name and must
    appear in :data:`NATIVE_ROW_RESPONSE_OWNERS`; that string, not the protocol,
    is what the composer checks.
    """

    native_row_owner: str

    def evaluate_rows(
        self, composer: object, inbound: object, outbound: object, row_valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the ``float32[composer.path_count]`` real/imaginary pair.

        A PAIR and not a complex tensor, for the reason the join and the beat
        family already give: no complex tensor crosses the autograd boundary,
        so the conjugate-Wirtinger convention cannot be got wrong at the seam.
        It also means the composer hands these straight to the join with no
        intervening ``torch.complex`` and no ``.contiguous()`` copy, so a row
        response costs exactly ONE extra kernel launch per frame.
        """
        ...


SPEED_OF_LIGHT_M_PER_S = 299792458.0

#: The normalisation that makes ``|C_rt|^2`` the bistatic radar equation.
#:
#: A composed two-way coefficient is
#:
#:   |C_rt|^2 = P_in (lam/(4 pi d_in))^2 |S|^2 P_site (lam/(4 pi d_out))^2
#:
#: and the bistatic radar equation is
#:
#:   P_r = P_t G_t G_r lam^2 sigma / ((4 pi)^3 d_in^2 d_out^2)
#:
#: With the site excited at exactly 1 W, matching the two requires
#:
#:   |S|^2 = 4 pi sigma / lam^2,   i.e.   S = sqrt(4 pi sigma) / lam
#:
#: This was unpinned, and an unpinned target strength is not a free parameter:
#: it is a level that is wrong by ``lam^2 / (4 pi)``, which at 77 GHz is a
#: factor of 6.6e5, or 58 dB.
RCS_AMPLITUDE_LAW = "sqrt(4*pi*sigma_m2)/wavelength_m"


def rcs_amplitude(sigma_m2: float | torch.Tensor, wavelength_m: float) -> float | torch.Tensor:
    """``sqrt(4 pi sigma) / lambda``, the dimensionless target strength.

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
    ``log sigma``. The host-float route keeps its exact old behaviour,
    including the negative-value refusal, because there is no derivative there
    to be wrong about.
    """

    if not wavelength_m > 0.0:
        raise ValueError("wavelength_m must be positive")
    if isinstance(sigma_m2, torch.Tensor):
        if sigma_m2.ndim != 0:
            raise ValueError(
                "a tensor sigma_m2 must be a 0-dim scalar, got rank "
                f"{sigma_m2.ndim}; ScalarRcsResponse is one complex number per "
                "target, broadcast across that target's rows"
            )
        return torch.sqrt(4.0 * math.pi * sigma_m2) / float(wavelength_m)
    if sigma_m2 < 0.0:
        raise ValueError("sigma_m2 is a radar cross section in square metres and cannot be negative")
    return math.sqrt(4.0 * math.pi * float(sigma_m2)) / float(wavelength_m)


@dataclass(frozen=True, slots=True, eq=False)
class ScalarRcsResponse:
    """Complex target response ``S = amplitude * exp(-j * phase_rad)``.

    Both members are 0-dim tensors and both may carry gradients.

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
            if value.ndim != 0:
                raise ValueError(f"{name} must be a 0-dim tensor, got rank {value.ndim}")
            if value.dtype != torch.float32:
                raise TypeError(f"{name} must use torch.float32, got {value.dtype}")

    @classmethod
    def from_values(
        cls, amplitude: float, phase_rad: float, *, device: torch.device | str = "cpu", requires_grad: bool = False
    ) -> ScalarRcsResponse:
        def parameter(value: float) -> torch.Tensor:
            tensor = torch.tensor(float(value), dtype=torch.float32, device=device)
            return tensor.requires_grad_(requires_grad)

        return cls(amplitude=parameter(amplitude), phase_rad=parameter(phase_rad))

    @classmethod
    def from_rcs(
        cls,
        sigma_m2: float | torch.Tensor,
        *,
        reference_frequency_hz: float,
        phase_rad: float = 0.0,
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
        This is the inverse-design question a radar caller actually asks - how
        large does this target have to be - and before Phase 9 it could not be
        asked at all, because the amplitude was formed by ``math.sqrt`` on the
        host and no refusal said so.

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
                "RCS_AMPLITUDE_LAW - or use from_values to author the "
                "dimensionless strength as its own leaf."
            )
        return cls(
            amplitude=amplitude, phase_rad=torch.tensor(float(phase_rad), dtype=torch.float32, device=amplitude.device)
        )

    @property
    def is_geometry_dependent(self) -> bool:
        return False

    def evaluate(self, row_count: int, device: torch.device) -> torch.Tensor:
        """Broadcast the response across ``row_count`` composed rows.

        ``device`` is honoured, not decorative. The composer passes the device
        its composed rows live on, and a CPU-authored response used to be
        accepted here and then fail with a device-mismatch error several frames
        of stack away from the parameter that caused it. ``Tensor.to`` is
        autograd-aware, so a response whose parameters carry gradients keeps
        them across the move.
        """

        if row_count < 0:
            raise ValueError("row_count must be non-negative")
        amplitude = self.amplitude.to(device=device, dtype=torch.complex64)
        phase = self.phase_rad.to(device=device, dtype=torch.complex64)
        return (amplitude * torch.exp(-1j * phase)).expand(row_count)


ASPECT_SCATTER_LAW = "S = amplitude * max(-dot(dir_in, axis), 0)^n * max(dot(dir_out, axis), 0)^n * exp(-i * phase_rad)"

#: The fully qualified name the composer checks against its owner list.
_OWNER = "witwin.radar.scattering.AspectScatterResponse"


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
        (dir_in, dir_out, axis, amplitude, phase_rad, idx_in, idx_out, idx_site, row_valid) = ctx.saved_tensors

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

        That last sentence was written in Phase 7 and was not true until
        Channel's ADR-043 (``CONTRACT_VERSION`` 6). Before it,
        ``PropagationGeometry.field_direction`` was marked non-differentiable in
        both field-transport setup contexts, so ``grad_dir_in`` / ``grad_dir_out``
        were computed by the backward kernel below and then discarded, and a
        forward tangent never arrived. It is true now, for ``{los, reflection}``
        under a frozen topology, which is
        ``capabilities().direction_differentiable_components`` and a superset of
        every component the Radar adapter is allowed to freeze. Liveness is
        decided ONCE for a whole propagation result, so there is no result in
        which some of these rows carry a derivative and others silently do not.
        ``tests/test_phase9_aspect_direction_ad.py`` measures the whole chain
        against finite differences, including the falsifier that a detached
        direction takes this gradient to exactly zero.
        """

        if self.site_count != composer.site_count:
            raise ValueError(
                f"this response carries {self.site_count} sites but the join was frozen against {composer.site_count}"
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
            f"the {name} leg carries {int(direction.shape[0])} direction rows but this join was frozen against {rows}"
        )
    return direction.contiguous()


__all__ = ["AspectScatterResponse", "ScalarRcsResponse", "ScatterResponse"]
