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
    c_in: torch.Tensor,
    c_out: torch.Tensor,
    response: torch.Tensor,
    idx_in: torch.Tensor,
    idx_out: torch.Tensor,
    idx_s: torch.Tensor,
    row_valid: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose ``(tau_rt, C_rt)`` from two legs and a per-site response.

    ``response`` is indexed per SITE, not per composed row: it is a target
    property, and broadcasting it per row would hide the fact that its gradient
    is a reduction over every round trip through that target.
    """

    total_delay = tau_in.index_select(0, idx_in) + tau_out.index_select(0, idx_out)
    transfer = (c_out.index_select(0, idx_out) * response.index_select(0, idx_s)) * c_in.index_select(0, idx_in)

    if row_valid is not None:
        total_delay = torch.where(row_valid, total_delay, torch.zeros_like(total_delay))
        transfer = torch.where(row_valid, transfer, torch.zeros_like(transfer))
    return total_delay, transfer


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
            raise ValueError(f"this response holds {int(self.value.shape[0])} sites, asked for {row_count}")
        return self.value.to(device=device)


def leg_length_f64(source: torch.Tensor, points: torch.Tensor, depth: torch.Tensor, site: torch.Tensor) -> torch.Tensor:
    """The unfolded length [m] of each leg row through its first ``depth`` points."""

    lengths = []
    for row in range(int(source.shape[0])):
        chain = [source[row], *[points[row, j] for j in range(int(depth[row]))], site[row]]
        lengths.append(sum(torch.linalg.vector_norm(b - a) for a, b in zip(chain[:-1], chain[1:], strict=True)))
    return torch.stack(lengths)


def element_factor_f64(source: torch.Tensor, points: torch.Tensor, depth: torch.Tensor, site: torch.Tensor, p):
    """``1 / |P_perp(d) p|`` with ``d`` the unit first-segment direction; ``p`` is one vector or one per row."""

    first = torch.stack([points[row, 0] if int(depth[row]) > 0 else site[row] for row in range(int(source.shape[0]))])
    d = first - source
    d = d / torch.linalg.vector_norm(d, dim=1, keepdim=True)
    p = torch.as_tensor(p, dtype=d.dtype, device=d.device)
    p = p.expand_as(d)
    along = (d * p).sum(-1)
    return 1.0 / torch.sqrt((p * p).sum(-1) - along * along)


def site_basis_f64(k: torch.Tensor, up: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """BSA ``(h, v)`` for propagation ``k`` into the site and site axis ``up``."""

    h = torch.linalg.cross(up.expand_as(k), k, dim=-1)
    h = h / torch.linalg.vector_norm(h, dim=-1, keepdim=True)
    return h, torch.linalg.cross(h, k, dim=-1)


def geometric_join_reference(
    *,
    field_in: torch.Tensor,
    field_out: torch.Tensor,
    length_in: torch.Tensor,
    length_out: torch.Tensor,
    channel_length_in: torch.Tensor,
    channel_length_out: torch.Tensor,
    factor_in: torch.Tensor,
    factor_out: torch.Tensor,
    matrix: torch.Tensor,
    k_in: torch.Tensor | None,
    k_out: torch.Tensor | None,
    up: torch.Tensor | None,
    idx_in: torch.Tensor,
    idx_out: torch.Tensor,
    idx_s: torch.Tensor,
    k64: float,
    k32: float,
    speed_of_light: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(tau, V)`` of R-ADR-040/041 in plain Torch, for float64 autograd.

    ``V = a_t a_r exp(j psi) H^T M E`` with ``psi = -(k64 L64 - k32 L32)`` and
    ``M = s I`` for a scalar ``matrix`` ``[owners]`` or the BSA dyadic for a
    ``[owners, 4]`` one. An independent statement of the kernel's law, written
    from the ADR rather than from the kernel.
    """

    e = field_in.index_select(0, idx_in)
    h = field_out.index_select(0, idx_out)
    if matrix.ndim == 1:
        v0 = matrix.index_select(0, idx_s) * (h * e).sum(dim=1)
    else:
        assert k_in is not None and k_out is not None and up is not None
        axes = up.index_select(0, idx_s)
        hi, vi = site_basis_f64(k_in.index_select(0, idx_in), axes)
        hr, vr = site_basis_f64(k_out.index_select(0, idx_out), axes)
        receive = (h * hr).sum(dim=1), (h * vr).sum(dim=1)
        transmit = (e * hi).sum(dim=1), (e * vi).sum(dim=1)
        s = matrix.index_select(0, idx_s)
        v0 = sum(s[:, 2 * a + b] * receive[a] * transmit[b] for a in range(2) for b in range(2))
    l64 = length_in.index_select(0, idx_in) + length_out.index_select(0, idx_out)
    l32 = channel_length_in.index_select(0, idx_in) + channel_length_out.index_select(0, idx_out)
    psi = -(k64 * l64 - k32 * l32)
    gain = factor_in.index_select(0, idx_in) * factor_out.index_select(0, idx_out)
    value = v0 * gain * torch.exp(1j * psi)
    return l64 / speed_of_light, value


__all__ = [
    "PerSiteResponse",
    "element_factor_f64",
    "geometric_join_reference",
    "join_reference",
    "leg_length_f64",
    "site_basis_f64",
]
