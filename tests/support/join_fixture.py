"""Synthetic frozen legs for the two-way join, built without a Channel scene.

The real Phase-5 fixture has one sensor pair and one site. That is enough to
check the physics against closed forms and nowhere near enough to check the
JOIN: a permuted leg order, an empty pair segment, a multi-site gradient
reduction, and a per-site response all need shapes the fixture cannot produce.
Channel owns the frozen leg row order, so these legs are fabricated - which is
also the only way to permute them on purpose.

Everything here is deterministic. A join test that fails should fail the same
way twice.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch


def frozen_leg(rows, *, device: str = "cuda"):
    """A duck-typed frozen leg topology from fabricated rows.

    A row is ``(source, sink, component)``, or
    ``(source, sink, component, depth, primitive, material)`` where the last two
    are equal-length tuples of interaction labels.

    The short form ALIASES depth, primitive and material to the component,
    which is what a test that varies only the component wants - and is exactly
    why such a test cannot say which of the four identity-key columns the join
    reads: ``component`` alone always disambiguates. The long form exists so
    ``test_identity_key_columns.py`` can build two rows of one endpoint
    pair that differ in a single tie-break column.
    """

    keyed = [row if len(row) == 6 else (row[0], row[1], row[2], row[2], (row[2],), (row[2],)) for row in rows]
    widths = {(len(row[4]), len(row[5])) for row in keyed}
    if len(widths) != 1:
        raise ValueError(f"every fabricated row must carry the same interaction sequence width, got {sorted(widths)}")

    def column(index: int, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor([row[index] for row in keyed], dtype=dtype, device=device)

    def sequence(index: int) -> torch.Tensor:
        return torch.tensor([list(row[index]) for row in keyed], dtype=torch.int32, device=device)

    return SimpleNamespace(
        source_id=column(0, torch.int64),
        sink_id=column(1, torch.int64),
        component_id=column(2, torch.int32),
        depth=column(3, torch.int32),
        primitive_sequence=sequence(4),
        material_sequence=sequence(5),
    )


def leg_rows(sources, sinks, components) -> list[tuple[int, int, int]]:
    return [(source, sink, component) for source in sources for sink in sinks for component in components]


def outbound_leg_rows(sites, receivers, components) -> list[tuple[int, int, int]]:
    """Outbound rows: the RECEIVER is the source and the site the sink (R-ADR-039).

    Rows come in site-major order, so row ``r`` names the same (site,
    receiver, component) it would in the site-to-receiver orientation.
    """

    return [(receiver, site, component) for site in sites for receiver in receivers for component in components]


def leg_batch(
    delay: torch.Tensor,
    coefficient: torch.Tensor,
    *,
    row_valid: torch.Tensor | None = None,
    direction: torch.Tensor | None = None,
    site_departure: torch.Tensor | None = None,
    field: torch.Tensor | None = None,
    geometry: dict | None = None,
):
    """A ``RadarLegBatch`` around already-built payload tensors.

    The identity columns are zeros: the JOIN reads identity from the FROZEN
    topology, at freeze time, not from the per-frame batch. Filling them with
    anything meaningful here would suggest otherwise.

    ``direction`` is the row's propagation direction. A fabricated leg has no
    geometry behind it, so it defaults to zeros; a test that drives an
    aspect-dependent response passes real unit vectors.

    ``site_departure`` is, for an OUTBOUND leg, the unit bearing from the site
    toward the receiver side. The outbound leg runs from the receiver to the
    site (R-ADR-039), so the response reads that bearing off the leg's arrival
    segment, ``arrival_origin_m - arrival_target_m``; the fixture publishes it
    as a segment from the origin to the unit vector.

    ``field`` replaces the ``coefficient * z_hat`` field with an arbitrary
    complex3 one. ``geometry`` gives the leg Channel's geometry - ``source``,
    ``site`` (float32 or float64), ``points`` ``[rows, W, 3]``, ``depth`` and
    ``length`` - which switches the join to the float64 path length; with
    ``polarization`` (``[rows, 3]``, each row's launch polarization) the join
    also applies the element factor. A site-less leg also carries
    ``sink_polarization``, the receiver's polarization it is read along.
    """

    from witwin.radar.propagation import RadarLegBatch

    rows = int(delay.shape[0])
    device = delay.device

    def zeros(dtype, shape=None):
        return torch.zeros(shape or (rows,), dtype=dtype, device=device)

    return RadarLegBatch(
        leg_count=rows,
        pair_count=1,
        pair_index=zeros(torch.int64),
        pair_offsets=torch.tensor([0, rows], dtype=torch.int64, device=device),
        source_index=zeros(torch.int32),
        sink_index=zeros(torch.int32),
        depth=zeros(torch.int32) if geometry is None else geometry["depth"],
        component_id=zeros(torch.int32),
        source_id=zeros(torch.int64),
        sink_id=zeros(torch.int64),
        primitive_sequence=zeros(torch.int32, (rows, 1)),
        material_sequence=zeros(torch.int32, (rows, 1)),
        interaction_type=zeros(torch.int32, (rows, 1)),
        delay_s=delay,
        field=scalar_field(coefficient) if field is None else field,
        field_direction=zeros(torch.float32, (rows, 3)) if direction is None else direction,
        row_valid=row_valid,
        diagnostics=None,
        arrival_origin_m=site_departure,
        **(
            {"arrival_target_m": None if site_departure is None else torch.zeros_like(site_departure)}
            if geometry is None
            else _geometry_members(geometry)
        ),
    )


def _geometry_members(geometry: dict) -> dict:
    site = geometry["site"]
    members = {
        "departure_origin_m": geometry["source"],
        "interaction_positions_m": geometry["points"],
        "path_length_m": geometry["length"],
        "source_polarization": geometry.get("polarization"),
        "sink_polarization": geometry.get("sink_polarization"),
    }
    if site.dtype == torch.float64:
        members["arrival_target_m"] = site.float()
        members["precise_arrival_target_m"] = site
    else:
        members["arrival_target_m"] = site
    return members


def scalar_field(coefficient: torch.Tensor) -> torch.Tensor:
    """A field whose only component is ``z``: ``coefficient * z_hat``.

    Both legs of a fabricated join carry their field along the same unit
    vector, so the join's isotropic dyadic composes ``(C_out S) C_in`` exactly
    - the scalar algebra the fabricated join tests pin. Built by
    differentiable ops so a leaf on ``coefficient`` keeps its gradient.
    """

    axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=coefficient.device)
    return (coefficient[:, None] * axis).contiguous()


def payload(rows: int, *, seed: int, device: str = "cuda", scale: float = 1.0):
    """Deterministic pseudo-random delays and complex coefficients.

    Delays are nanosecond scale and coefficients are order one, matching what
    the real chain produces closely enough that a tolerance chosen here means
    the same thing there.
    """

    generator = torch.Generator().manual_seed(seed)

    def sample(count: int) -> torch.Tensor:
        return torch.rand(count, generator=generator, dtype=torch.float64)

    delay = (1.0e-8 + 2.0e-8 * sample(rows)).to(device=device)
    real = scale * (sample(rows) - 0.5)
    imag = scale * (sample(rows) - 0.5)
    coefficient = torch.complex(real, imag).to(device=device)
    return delay, coefficient


__all__ = ["frozen_leg", "leg_batch", "leg_rows", "outbound_leg_rows", "payload", "scalar_field"]
