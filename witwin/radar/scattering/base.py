"""The scatter-response contract.

A scatter response turns a set of composed round-trip rows into the complex
factor that sits between the inbound and outbound transports. Phase 4 ships one
implementation, a per-target broadcast scale; its aspect-dependent,
material-informed, and polarimetric successors evaluate per path and therefore
belong in a native kernel, not here.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


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
NATIVE_ROW_RESPONSE_OWNERS = frozenset(
    {"witwin.radar.scattering.aspect.AspectScatterResponse"}
)


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
        self,
        composer: object,
        inbound: object,
        outbound: object,
        row_valid: torch.Tensor,
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


__all__ = [
    "NATIVE_ROW_RESPONSE_OWNERS",
    "NativeRowScatterResponse",
    "ScatterResponse",
]
