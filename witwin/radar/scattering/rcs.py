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


def rcs_amplitude(sigma_m2: float, wavelength_m: float) -> float:
    """``sqrt(4 pi sigma) / lambda``, the dimensionless target strength.

    Dimensionless is the whole content of the normalisation. ``S`` carries no
    propagation phase and no spreading - both belong to Channel transport, once
    per leg - so what is left of a radar cross section after the two
    ``lam/(4 pi d)`` factors have been accounted for is a pure ratio.
    """

    if sigma_m2 < 0.0:
        raise ValueError("sigma_m2 is a radar cross section in square metres and cannot be negative")
    if not wavelength_m > 0.0:
        raise ValueError("wavelength_m must be positive")
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
        sigma_m2: float,
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
        """

        wavelength_m = SPEED_OF_LIGHT_M_PER_S / float(reference_frequency_hz)
        return cls.from_values(
            rcs_amplitude(sigma_m2, wavelength_m),
            phase_rad,
            device=device,
            requires_grad=requires_grad,
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
