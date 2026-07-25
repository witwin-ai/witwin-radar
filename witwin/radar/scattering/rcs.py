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

from dataclasses import dataclass

import torch


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

    @property
    def is_geometry_dependent(self) -> bool:
        return False

    def evaluate(self, row_count: int, device: torch.device) -> torch.Tensor:
        """Broadcast the response across ``row_count`` composed rows."""

        if row_count < 0:
            raise ValueError("row_count must be non-negative")
        phasor = torch.exp(-1j * self.phase_rad.to(torch.complex64))
        return (self.amplitude.to(torch.complex64) * phasor).expand(row_count)


__all__ = ["ScalarRcsResponse"]
