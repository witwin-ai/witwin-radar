"""The retained Torch two-way join, kept as an oracle after the kernel took over.

This is the composition ``TwoWayComposer.compose`` performed before
``two_way_join.cu`` replaced it, moved here verbatim rather than deleted. It has
two jobs and they need different precisions, which is why the dtype is the
caller's:

* in float32 on the device, it is the LOCKSTEP oracle - the same arithmetic in
  the same association, so a disagreement with the kernel is a kernel bug and
  not a precision artefact;
* in float64 on the CPU, it is the AD oracle - Torch autograd differentiates it
  exactly, float64 central differences validate that, and the production
  float32 gradients are then compared against it.

A float32 finite difference on the production join is not a usable oracle. The
composed transfer is a triple product of numbers spanning several orders of
magnitude, so differencing two nearly equal float32 values can return a
confident zero.

The association ``(C_out * S) * C_in`` is load bearing and is stated once, here
and in the kernel. Re-associating a complex product changes the result in the
last bits, so an oracle that associated differently would force a tolerance
that hid real errors.
"""

from __future__ import annotations

import torch


def join_reference(
    *,
    tau_in: torch.Tensor,
    tau_out: torch.Tensor,
    rate_in: torch.Tensor,
    rate_out: torch.Tensor,
    c_in: torch.Tensor,
    c_out: torch.Tensor,
    response: torch.Tensor,
    idx_in: torch.Tensor,
    idx_out: torch.Tensor,
    idx_s: torch.Tensor,
    row_valid: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compose ``(tau_rt, rate_rt, C_rt)`` from two legs and a per-site response.

    ``response`` is indexed per SITE, not per composed row: it is a target
    property, and broadcasting it per row would hide the fact that its gradient
    is a reduction over every round trip through that target.
    """

    total_delay = tau_in.index_select(0, idx_in) + tau_out.index_select(0, idx_out)
    total_rate = rate_in.index_select(0, idx_in) + rate_out.index_select(0, idx_out)
    transfer = (
        c_out.index_select(0, idx_out) * response.index_select(0, idx_s)
    ) * c_in.index_select(0, idx_in)

    if row_valid is not None:
        total_delay = torch.where(
            row_valid, total_delay, torch.zeros_like(total_delay)
        )
        total_rate = torch.where(row_valid, total_rate, torch.zeros_like(total_rate))
        transfer = torch.where(row_valid, transfer, torch.zeros_like(transfer))
    return total_delay, total_rate, transfer


class PerSiteResponse:
    """A scatter response with an independent complex value per site.

    ``ScalarRcsResponse`` broadcasts one complex number over every site, which
    cannot tell a correct per-site gradient reduction from a global sum. This
    satisfies the same protocol with one value per site so that ``grad_S`` is a
    genuine per-site reduce.
    """

    def __init__(self, value: torch.Tensor) -> None:
        self.value = value

    @property
    def is_geometry_dependent(self) -> bool:
        return False

    def evaluate(self, row_count: int, device: torch.device) -> torch.Tensor:
        if int(self.value.shape[0]) != row_count:
            raise ValueError(
                f"this response holds {int(self.value.shape[0])} sites, asked "
                f"for {row_count}"
            )
        return self.value.to(device=device)


__all__ = ["PerSiteResponse", "join_reference"]
