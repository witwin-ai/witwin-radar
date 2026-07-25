# R-ADR-003: Monostatic reciprocity is opt-in and unused in Phase 4

Status: Accepted (Phase 4)

## Context

When TX and RX coincide, the outbound leg is the reverse of the inbound leg, and
evaluating it looks like wasted work. Skipping it is a real optimization and also
a classic source of quietly wrong results.

## Decision

Reciprocity may be assumed only when ALL of the following hold:

1. The radar source and sink positions are identical, not merely close.
2. The scene is reciprocal: no non-reciprocal material and no moving boundary
   between the two traversals.
3. The two legs use the same component set, depth, and frequency.
4. The transport is polarization-symmetric under the exchange, or the response
   is scalar.
5. The endpoint polarizations and antenna patterns are identical.

The default is explicit outbound evaluation. **Phase 4 declares reciprocity
UNUSED**, and asserts that the outbound leg is actually evaluated: the spike
freezes two topologies and performs two consumer reevaluations per frame.

Reciprocity is never implemented as "scalar path length times two". A round-trip
delay is the sum of two leg delays, which is the same number only when the legs
are geometrically identical, and the transfer coefficient is a product of two
complex transports, not a square, unless conditions 2 through 5 also hold.

## Consequences

Phase 4 pays for one extra leg evaluation. The measured per-frame cost is two
validation D2H copies and two synchronizations, which is the budget frozen in
R-ADR-006. When reciprocity is introduced it will be a measurable change against
that budget rather than an assumption baked into the composer.

## Alternatives rejected

**Assume reciprocity whenever TX and RX are within a tolerance.** A tolerance
turns a structural property into a numerical one; a monostatic array with a
half-wavelength offset is not monostatic.

## Acceptance evidence

`tests/test_phase4_spike_e2e.py::test_per_frame_host_traffic_is_two_copies_and_two_synchronizations`
observes both legs' diagnostics, which exist only because both were evaluated.
`tests/test_phase4_adapter.py::test_freeze_is_never_called_per_frame` reevaluates
both legs across five frames.
