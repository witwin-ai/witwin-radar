# R-ADR-002: Scatter-site, leg, and round-trip identity

Status: Accepted (Phase 4), amended (Phase 5)

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
join is the full cross product.

Joining by array position, or by truncating to the shorter leg's path count,
is forbidden. Both would produce a plausible, wrong answer the moment a leg
reorders its rows, and the failure would read as a physics bug.

### The composed ORDER is canonical, and it is identity too (Phase 5)

Membership was by identity from the start; the composed row ORDER was not. Rows
were sorted by `(pair, inbound_row_index, outbound_row_index)`, so a permuted
leg order preserved the composed set and permuted the composed sequence. That is
enough to break any elementwise comparison, and it makes "shuffled legs,
identical output" untestable: the permutation test could only assert set
equality, which the positional order already satisfied.

The sort key is now frame-invariant identity:

```
(pair_rank, site_rank, inbound_leg_key, outbound_leg_key)

leg_key(row) = (component_id, depth,
                tuple(primitive_sequence[row]),
                tuple(material_sequence[row]))
```

`leg_key` must be UNIQUE within a leg's `(source_id, sink_id)` endpoint pair. A
collision is refused at freeze rather than tie-broken on row position; a
tie-break there would quietly restore the dependence this removes and make the
permutation test vacuous again.

`primitive_sequence` and `material_sequence` are ADR-037 frozen LABELS, not
re-validated hits. Channel keeps the original label when a reevaluated
stationary point slides onto a coplanar twin triangle, and that is precisely
what makes them stable enough to key on. Using them as identity is not a claim
about which triangle the ray struck on any given frame.

### The pair partition spans the front end, not the survivors (Phase 5)

`freeze` takes explicit `radar_source_ids` and `radar_sink_ids`, and
`sensor_pair_count` is their cross product. Deriving the pair set from surviving
composed rows was a latent correctness bug rather than an optimization:
`synthesize_fmcw_beat` shapes its output `[chirps, sensor_pair_count, samples]`,
so a TX/RX pair whose only site failed discovery silently RENUMBERED and
RESHAPED the IQ cube. A pair that discovered nothing now owns an empty segment,
and the beat kernel already yields an exact-zero accumulator for `start == end`.

The pair index is sink-major, `sink_rank * source_count + source_rank`, matching
Channel's own consumer exactly. Any other order would put a second, silently
different, virtual-array numbering on the same data.

A declared site with no row at all in one of the legs is still refused. A site
absent for ONE endpoint is not: that is discovery reporting that this TX/RX pair
sees nothing there, and it is published as an empty segment. A leg row whose
radar endpoint is outside the declared front end is refused too, since it would
be dropped rather than emptied.

### The response is per site, not per composed row (Phase 5)

`ScatterResponse.evaluate` is called with the SITE count and indexed by a frozen
`response_slot`. Broadcasting per composed row cannot distinguish a correct
per-site gradient reduction from a global sum, and it made `grad_S` look like an
elementwise product rather than the reduce over every round trip through that
target that it actually is.

### A dead composed row's payload is exactly zero (Phase 5)

The join used to compute `0 + tau_out` for a dead row and publish it. It was
harmless downstream, because the synthesis facade zeroes the WEIGHT, but
"validity is the sole authority" is a property of the join, and it is now
enforced there: `tau_rt`, `rate_rt`, and `C_rt` are all exactly zero when
`row_valid` is false.

### Direct rows use sentinels, not fabricated data (Phase 5)

A direct row - radar source straight to radar sink, no scatter site - carries
`site_id = -1` and `outbound_row = -1`, and its batch records
`join_mode = "direct"`. Giving it a fabricated second leg and a unit response
would make it indistinguishable from a real round trip through a target whose
response happens to be one.

### Deferred: polarimetric multipath (Phase 5)

Discovery with `response="polarimetric_transport"` plus `reflection` is refused
by the consumer's capability record; only REEVALUATING an already-prepared
reflection topology under that response succeeds. Supporting polarimetric
multipath therefore needs a two-response split in the adapter -
`scalar_transport` to discover, `polarimetric_transport` to reevaluate, and a
polarization basis on both leg batches. That is not done here, and the boundary
is pinned by test so it cannot be mistaken for support.

### Where the join is built

Once, at freeze time, from the frozen leg topologies. Host observation is
permitted there because `prepare_fixed_topology` has already synchronized.
`compose()` performs device gathers and arithmetic only.

### Scope

Phase 4 exercised one TX/RX pair and one scatter site, with the contract shaped
so that batching would be additive. Phase 5 collected on that: multi-pair,
multi-site, multi-component joins run through the same code with no special
case, and the fixture's single pair is now one instance of the general shape
rather than the only shape tested. Monostatic reciprocity and hybrid
deduplication remain out of scope.

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

## Acceptance evidence (Phase 5)

- `tests/test_phase5_join_identity.py::test_a_permuted_leg_order_composes_to_an_elementwise_identical_frame`
- `tests/test_phase5_join_identity.py::test_a_permuted_leg_order_produces_bit_identical_gradients`
- `tests/test_phase4_two_way.py::test_two_leg_rows_that_share_an_identity_key_are_refused`
- `tests/test_phase4_two_way.py::test_the_pair_partition_spans_the_front_end_not_the_surviving_rows`
- `tests/test_phase4_two_way.py::test_a_leg_endpoint_outside_the_declared_front_end_is_refused`
- `tests/test_phase5_two_way_join_kernel.py::test_an_empty_pair_segment_composes_without_a_special_case`
- `tests/test_phase5_two_way_join_kernel.py::test_a_dead_row_publishes_exactly_zero_in_every_output`
- `tests/test_phase5_multipath_legs.py::test_four_combined_paths_carry_the_analytic_round_trip_delays`
- `tests/test_phase5_multipath_legs.py::test_the_frozen_row_identity_is_the_same_storage_on_every_frame`
- `tests/test_phase5_multipath_legs.py::test_polarimetric_multipath_discovery_is_still_refused_by_the_consumer`
- `tests/test_phase5_join_modes.py` (direct sentinels, one contract, both modes)

## Acceptance evidence (Phase 4)

`tests/test_phase4_two_way.py`:

- `test_join_is_by_identity_not_by_array_position` (permuted outbound frozen
  rows produce an identical composed result, with different outbound row indices)
- `test_delay_is_additive_and_transfer_factorizes`
- `test_rows_are_sorted_into_a_valid_pair_partition`
- `test_a_site_without_a_leg_is_refused_rather_than_dropped`

`tests/test_phase4_spike_e2e.py::test_composed_delay_and_magnitude_match_the_closed_form`
asserts the 1 W isotropic re-radiator magnitude verbatim.
