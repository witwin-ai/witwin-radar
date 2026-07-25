# R-ADR-006: Compact contract, cardinality budget, and no partial results

Status: Accepted (Phase 4)

## Context

Channel's ADR-032 buys exact `O(K)` compact results at the price of a small,
audited number of host observations. That budget is only meaningful if consumers
do not spend more on top of it. A consumer that calls `.item()` once per frame
undoes the whole design.

## Decision

### Radar adds zero host observations

`K` is read from `PropagationPathBatch.path_count`, which the compact boundary
already published as a host int. The adapter, the composer's per-frame path, the
scatter response, and the synthesis facade contain no `.cpu()`, `.numpy()`,
`.tolist()`, `.item()`, host iteration, or Boolean compaction.

The one sanctioned exception is named: `TwoWayComposer.freeze` reads leg identity
to the host to build the join. That happens once per frozen topology, after
`prepare_fixed_topology` has already synchronized.

### The frozen-topology cost is paid once and reported separately

`prepare_fixed_topology` costs 3 D2H copies, 17 bytes, and 3 synchronizations per
frozen topology. It is called exactly once per leg, outside every loop, and its
counters are published on `FrozenLegTopology` rather than folded into per-frame
diagnostics -- precisely so it cannot be mistaken for per-frame cost.

`TwoWayComposer.freeze` adds 5 more host reads on top of that, one `.tolist()`
per identity column: both legs' `source_id` and `sink_id`, plus `site_ids`. These
are NOT counted by any Channel diagnostic, because Channel never sees them. They
are recorded here so the one-time total is complete rather than implied:

```
one-time, per two-leg composer: 2 x (3 copies, 17 bytes) from prepare
                              + 5 host reads from the identity join
per frame:                      2 D2H copies, 8 bytes, 2 synchronizations
```

The per-frame figure is the one that matters and the one that is budgeted. The
one-time figure is recorded so that a future composer which moved a `.tolist()`
into `compose` would show up as a changed number rather than as unattributed
host traffic.

### The measured per-frame budget

Two legs, one `validation_d2h_copies` each:

```
per frame: 2 D2H copies, 8 bytes total, 2 synchronizations
```

Any increase is a budget change and requires evidence, not a comment.

### Compact publication

Radar publishes actual rows with native-produced stable pair segmentation:
`sensor_pair_index` and `pair_offsets` over exactly `path_count` rows. Capacity
shapes are not public API.

### No partial results

A contract, ABI, device, or capability failure fails loudly BEFORE any
`RadarPathBatch` or IQ tensor exists. A missing forward tangent in `ad_mode="jvp"`
raises rather than publishing `delay_rate = 0`, which would be indistinguishable
from a static scene.

A dead row is not a failure. `row_valid` is the sole authority on whether a row
means anything; a dead row is the complete, correct answer that the frozen path
does not exist at these endpoints, and it contributes exactly zero. Validity is
never inferred from a zero payload, because a live row may legitimately carry a
zero coefficient.

## Consequences

Dead rows are zeroed on the WEIGHT with `torch.where`, not on the output. Zeroing
the output would leave a live gradient path back through a row that does not
exist.

## Acceptance evidence

- `tests/test_phase4_adapter.py::test_per_frame_budget_is_one_validation_copy_per_leg`
- `tests/test_phase4_adapter.py::test_freeze_is_never_called_per_frame` (a
  counting monkeypatch: two preparations total across five frames)
- `tests/test_phase4_spike_e2e.py::test_per_frame_host_traffic_is_two_copies_and_two_synchronizations`
- `tests/test_phase4_import_boundary.py::test_no_hot_path_host_observation_in_the_new_modules`
- `tests/test_phase4_spike_e2e.py::test_a_dead_row_contributes_exactly_zero_to_loss_and_gradients`
- `tests/test_phase4_adapter.py::test_jvp_publishes_delay_rate_and_refuses_to_invent_one`
