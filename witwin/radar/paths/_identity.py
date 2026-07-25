"""Frozen leg row identity, shared by every composer in this package.

A composer's canonical row order is built from identity, never from array
position. The pieces that read that identity off a frozen leg live here so the
direct and two-way composers cannot drift into two different definitions of
what makes a row the row it is.

The identity key is ``(component, depth, primitive sequence, material
sequence)``. Those sequences are ADR-037 frozen LABELS, not re-validated hits:
Channel keeps the original label when a reevaluated stationary point slides
onto a coplanar twin triangle, which is exactly what makes them stable enough
to key on. They are not a claim about which primitive the ray struck this
frame.
"""

from __future__ import annotations

import torch


LegKey = tuple[int, int, tuple[int, ...], tuple[int, ...]]


def stable_ids(values, name: str) -> list[int]:
    """Normalize a stable-ID sequence to a host list of distinct ints."""

    if isinstance(values, torch.Tensor):
        if values.ndim != 1:
            raise ValueError(f"{name} must be a 1-D sequence of stable IDs")
        listed = [int(value) for value in values.tolist()]
    else:
        listed = [int(value) for value in values]
    if not listed:
        raise ValueError(f"{name} must not be empty")
    if len(set(listed)) != len(listed):
        raise ValueError(f"{name} must not repeat a stable ID, got {listed}")
    return listed


def leg_identity(frozen, name: str) -> tuple[list[int], list[int], list[LegKey]]:
    """Read one frozen leg's row identity to the host, once.

    The key is everything that distinguishes two rows of the SAME leg: which
    multipath component, how deep, and which primitives and materials it
    interacted with. It is frame invariant, so the composed order it induces is
    frame invariant too.
    """

    source = [int(value) for value in frozen.source_id.tolist()]
    sink = [int(value) for value in frozen.sink_id.tolist()]
    component = [int(value) for value in frozen.component_id.tolist()]
    depth = [int(value) for value in frozen.depth.tolist()]
    primitive = [
        tuple(int(value) for value in row)
        for row in frozen.primitive_sequence.tolist()
    ]
    material = [
        tuple(int(value) for value in row)
        for row in frozen.material_sequence.tolist()
    ]
    rows = len(source)
    for label, column in (
        ("sink_id", sink),
        ("component_id", component),
        ("depth", depth),
        ("primitive_sequence", primitive),
        ("material_sequence", material),
    ):
        if len(column) != rows:
            raise ValueError(
                f"{name} leg {label} has {len(column)} rows, expected {rows}"
            )
    keys: list[LegKey] = [
        (component[row], depth[row], primitive[row], material[row])
        for row in range(rows)
    ]
    return source, sink, keys


def group_rows(
    source: list[int], sink: list[int], keys: list[LegKey], name: str
) -> dict[tuple[int, int], list[int]]:
    """Index a leg's rows by its ``(source_id, sink_id)`` endpoint pair.

    Also enforces that the identity key is UNIQUE inside each endpoint pair. A
    collision would make the canonical composed order ambiguous and would
    silently turn the permutation test vacuous, so it is refused here rather
    than tie-broken on row position - which is exactly the positional
    dependence this module exists to remove.
    """

    groups: dict[tuple[int, int], list[int]] = {}
    seen: dict[tuple[int, int], dict[LegKey, int]] = {}
    for row, endpoints in enumerate(zip(source, sink, strict=True)):
        groups.setdefault(endpoints, []).append(row)
        claimed = seen.setdefault(endpoints, {})
        if keys[row] in claimed:
            raise ValueError(
                f"{name} leg rows {claimed[keys[row]]} and {row} share the "
                f"identity key {keys[row]} within endpoint pair {endpoints}; "
                "the composed order would be ambiguous"
            )
        claimed[keys[row]] = row
    return groups


def sink_major_rank(sources: list[int], sinks: list[int]):
    """The sensor-pair index, mirroring the Channel consumer's convention.

    Channel computes ``sink_row_index * source_count + source_row_index`` and
    spans ``source_count * sink_count``. Using anything else here would put a
    second, silently different, virtual-array numbering on the same data.
    """

    source_rank = {value: rank for rank, value in enumerate(sources)}
    sink_rank = {value: rank for rank, value in enumerate(sinks)}

    def rank(source: int, sink: int) -> int:
        return sink_rank[sink] * len(sources) + source_rank[source]

    return rank


def pair_offsets(pair_of_row: list[int], pair_count: int) -> list[int]:
    """A half-open partition of composed rows by sensor pair.

    Empty segments are legal and expected: the partition spans the FRONT END's
    cross product, so a pair that discovered nothing still owns a segment and
    the IQ cube keeps its declared shape.

    The kernel CLAMPS a malformed offsets table rather than failing, because
    reading its values per frame would be exactly the D2H the fixed-topology
    capability exists to avoid, and clamping turns a malformed table into a
    plausible wrong answer. So the gate lives here, at freeze time, where the
    table is still a Python list and checking it is free. What is actually
    checkable is the input: once every row's pair rank is in range, counting
    them can only produce a valid partition.
    """

    if pair_count < 1:
        raise ValueError(f"pair_count must be positive, got {pair_count}")
    counts = [0] * pair_count
    for row, pair in enumerate(pair_of_row):
        if not 0 <= pair < pair_count:
            raise ValueError(
                f"composed row {row} claims sensor pair {pair}, which is "
                f"outside the declared {pair_count} pairs; the offsets table "
                "would not partition all composed rows"
            )
        counts[pair] += 1
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + count)
    return offsets


def csr(owner_of_row: list[int], owner_count: int) -> tuple[list[int], list[int]]:
    """Group composed rows by an owner index, as a CSR offsets/rows pair.

    The VJP needs this: one thread owns one gradient slot and loops its own
    segment, so the reduction needs no atomics and its summation order is fixed
    by the frozen join. That is what makes a bit-identical gradient comparison
    across a leg permutation a legitimate assertion rather than a lucky one.
    """

    buckets: list[list[int]] = [[] for _ in range(owner_count)]
    for composed_row, owner in enumerate(owner_of_row):
        buckets[owner].append(composed_row)
    offsets = [0]
    rows: list[int] = []
    for bucket in buckets:
        rows.extend(bucket)
        offsets.append(len(rows))
    return offsets, rows


__all__ = [
    "LegKey",
    "csr",
    "group_rows",
    "leg_identity",
    "pair_offsets",
    "sink_major_rank",
    "stable_ids",
]
