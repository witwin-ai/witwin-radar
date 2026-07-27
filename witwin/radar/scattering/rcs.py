"""A scalar, differentiable target response.

This is deliberately the SIMPLEST thing that can carry a gradient: a single
complex number per target, broadcast across the target's rows. It is not per
path and not aspect dependent, so it is a parameter scale rather than
hot-path physics, and evaluating it in Torch does not put a numerical backend
where a native kernel belongs.

The aspect-dependent, material-informed, and polarimetric responses that follow
DO vary per path. Those evaluate inside a native kernel; the protocol's
``is_geometry_dependent`` flag exists so that distinction cannot blur.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

#: Exact SI definition, in metres per second.
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


def rcs_amplitude(
    sigma_m2: float | torch.Tensor, wavelength_m: float
) -> float | torch.Tensor:
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
    :mod:`witwin.radar.host_parameters` refuses everything else.

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
        cls,
        amplitude: float,
        phase_rad: float,
        *,
        device: torch.device | str = "cpu",
        requires_grad: bool = False,
    ) -> "ScalarRcsResponse":
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
    ) -> "ScalarRcsResponse":
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
            return cls.from_values(
                amplitude,
                phase_rad,
                device=device,
                requires_grad=requires_grad,
            )
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
            amplitude=amplitude,
            phase_rad=torch.tensor(
                float(phase_rad), dtype=torch.float32, device=amplitude.device
            ),
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


__all__ = [
    "RCS_AMPLITUDE_LAW",
    "SPEED_OF_LIGHT_M_PER_S",
    "ScalarRcsResponse",
    "rcs_amplitude",
]
