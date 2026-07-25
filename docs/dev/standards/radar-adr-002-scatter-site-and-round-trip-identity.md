# R-ADR-002: Scatter-site, leg, and round-trip identity

Status: Accepted (Phase 4)

## Context

A radar round trip is two propagation legs joined at a target. The Channel
consumer publishes legs; it has no notion of a round trip, and it should not.
Radar therefore owns the join, and the join has to survive the one thing that
makes it fragile: the two legs are separate compact batches whose row orders are
independent.

## Decision

### Identity tuple

A composed round-trip row is identified by

```
(radar_source_id, site_id, radar_sink_id)
```

all `int64` stable world IDs, plus `(inbound_row, outbound_row)` recording which
frozen leg rows were joined. `site_id` is stable across a frozen sequence.

### The join is by identity, never by position

An inbound row joins an outbound row when the inbound row's `sink_id` equals the
site and the outbound row's `source_id` equals the same site; within a site the
join is the full cross product. Rows are then sorted by sensor pair so that
`pair_offsets` is a valid half-open partition.

Joining by array position, or by truncating to the shorter leg's path count,
is forbidden. Both would produce a plausible, wrong answer the moment a leg
reorders its rows, and the failure would read as a physics bug.

### Where the join is built

Once, at freeze time, from the frozen leg topologies. Host observation is
permitted there because `prepare_fixed_topology` has already synchronized.
`compose()` performs device gathers and arithmetic only.

### Phase-4 scope

One TX/RX pair and one scatter site. The contract is shaped so that batching is
additive: `sensor_pair_count`, `sensor_pair_index`, and `pair_offsets` already
exist and already carry the general case, and `TwoWayComposer.freeze` already
accepts a vector of site IDs. Nothing about the single-site case is special-cased
in a way a batch would have to undo.

### Two-way power: a declared simplification

Each Channel leg independently applies `sqrt(P) * lambda / (4 pi d)`. With unit
power on both legs the site is modelled as a **1 W isotropic re-radiator**, so

```
|C_rt| = amplitude * (lambda / (4 pi d_in)) * (lambda / (4 pi d_out))
```

This is NOT the radar equation. It is a bounded spike simplification, and it is
asserted verbatim by test so that the physically correct normalization cannot
land silently later.

## Consequences

The composer is pure bookkeeping plus device arithmetic and is testable on the
CPU with fabricated legs, which is what makes the permutation and multi-site
cases reachable at all. The single real line-of-sight leg has one row and cannot
distinguish a correct join from a positional one.

## Alternatives rejected

**Join by row index.** Cheaper, and correct exactly until it is not.

**Let Channel publish round trips.** The consumer is solver-neutral and
application-neutral. A round trip is a radar concept; putting it in Channel would
be Radar policy in someone else's contract.

## Acceptance evidence

`tests/test_phase4_two_way.py`:

- `test_join_is_by_identity_not_by_array_position` (permuted outbound frozen
  rows produce an identical composed result, with different outbound row indices)
- `test_delay_is_additive_and_transfer_factorizes`
- `test_rows_are_sorted_into_a_valid_pair_partition`
- `test_a_site_without_a_leg_is_refused_rather_than_dropped`

`tests/test_phase4_spike_e2e.py::test_composed_delay_and_magnitude_match_the_closed_form`
asserts the 1 W isotropic re-radiator magnitude verbatim.
