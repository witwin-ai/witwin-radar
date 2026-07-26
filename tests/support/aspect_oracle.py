"""The separable aspect law in float64 on the host, once.

Both the kernel test and the AD test compare against this, so the law is
written down in exactly one place and a disagreement between the two tests is
impossible. It is a REFERENCE and never a production route: nothing under
``witwin/`` imports it.

The two sign conventions it encodes are the ones that would otherwise be
plausible either way:

* ``dir_in`` is a PROPAGATION direction and points INTO the site, so the
  incidence cosine against an outward aspect axis is its NEGATIVE;
* ``dir_out`` points away from the site and enters directly.

Getting either wrong gives a lobe that is exactly backwards and still looks
like a lobe.
"""

from __future__ import annotations

import torch


def aspect_response(
    dir_in: torch.Tensor,
    dir_out: torch.Tensor,
    idx_in,
    idx_out,
    idx_site,
    axis: torch.Tensor,
    amplitude: torch.Tensor,
    phase_rad: torch.Tensor,
    exponent: float,
    row_valid=None,
) -> torch.Tensor:
    """``complex128[K]``: the response of every composed row.

    Every input is taken to the host in float64 first, so the reference rounds
    once at the end rather than at every operation the way a float32 chain
    would.
    """

    def host(value):
        return value.detach().to(device="cpu", dtype=torch.float64)

    d_in = host(dir_in)
    d_out = host(dir_out)
    u = host(axis)
    a = host(amplitude)
    phi = host(phase_rad)
    rows = len(idx_in)
    out = torch.zeros(rows, dtype=torch.complex128)
    for k in range(rows):
        if row_valid is not None and int(row_valid[k]) == 0:
            continue
        i = int(idx_in[k])
        o = int(idx_out[k])
        s = int(idx_site[k])
        ci = float(-torch.dot(d_in[i], u[s]))
        co = float(torch.dot(d_out[o], u[s]))
        gi = ci**exponent if ci > 0.0 else 0.0
        go = co**exponent if co > 0.0 else 0.0
        magnitude = float(a[s]) * gi * go
        angle = float(phi[s])
        out[k] = complex(
            magnitude * torch.cos(torch.tensor(angle, dtype=torch.float64)).item(),
            -magnitude * torch.sin(torch.tensor(angle, dtype=torch.float64)).item(),
        )
    return out


def unit_rows(values) -> torch.Tensor:
    """Normalise a host sequence of vectors to unit length in float64."""

    tensor = torch.tensor([list(row) for row in values], dtype=torch.float64)
    return tensor / torch.linalg.vector_norm(tensor, dim=1, keepdim=True)


__all__ = ["aspect_response", "unit_rows"]
