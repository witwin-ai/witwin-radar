# R-ADR-006: Compact contract, cardinality budget, and no partial results

Status: Accepted (Phase 4), amended (Phase 5)

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

The one sanctioned exception has exactly ONE owner: `paths/_identity.py` reads
frozen leg row identity to the host to build a join. That happens once per
frozen topology, after `prepare_fixed_topology` has already synchronized. Naming
one owner rather than one module per composer is what keeps it bounded: neither
`TwoWayComposer` nor `DirectComposer` may grow a read of its own.

### The frozen-topology cost is paid once and reported separately

`prepare_fixed_topology` costs 3 D2H copies, 17 bytes, and 3 synchronizations
per frozen LINE-OF-SIGHT topology. Adding reflection makes it 4 copies, 33
bytes, and 4 synchronizations: the difference is the reflection bucket's own
preparation. It is called exactly once per leg, outside every loop, and its
counters are published on `FrozenLegTopology` rather than folded into per-frame
diagnostics -- precisely so it cannot be mistaken for per-frame cost.

A composer's `freeze` adds host reads on top of that, one `.tolist()` per
identity column: per leg, `source_id`, `sink_id`, `component_id`, `depth`,
`primitive_sequence`, and `material_sequence`. Phase 5 widened this from two
columns to six, because endpoint IDs alone no longer distinguish two rows of the
same leg once a leg carries several multipath components. For a two-leg
composer that is 13 reads with the front-end endpoint IDs passed as Python
lists, or 15 when they are passed as tensors. These are NOT counted by any
Channel diagnostic, because Channel never sees them. They are recorded here so
the one-time total is complete rather than implied:

```
one-time, per two-leg multipath composer:
    2 x (4 copies, 33 bytes, 4 synchronizations) from prepare
  + 13 host reads from the identity join

per frame: 2 D2H copies, 8 bytes, 2 synchronizations
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

Phase 5 does not move this. Four composed rows instead of one, two components
per leg instead of one, and a native join in the middle all cost the same two
host observations, and that equality is asserted as a COMPARISON between the
line-of-sight and multipath frames rather than as two numbers that happen to
match. The native join adds zero, measured in isolation so the legs cannot mask
a regression in it.

Any increase is a budget change and requires evidence, not a comment.

### Reported and measured agree, and the limit of the measurement is stated

The budget is checked twice: against the counters Channel REPORTS through
`PropagationDiagnostics`, and against what actually happens, by counting every
`.item()`, `.cpu()`, `.tolist()`, `.numpy()` and `torch.cuda.synchronize` call
in the frame. Both give 2. A reported budget nobody measures is a comment.

That second measurement can only see PYTHON-level observations. A
`cudaStreamSynchronize` inside a native reflection kernel is invisible from
Python and is not reported by the consumer either, so if one exists this budget
does not cover it. It is recorded as an upstream observation rather than
asserted as a number that was not observed: Channel is read-only here, and
claiming an unmeasured count would be worse than naming the gap.

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

## Acceptance evidence (Phase 5)

- `tests/test_phase5_budget.py::test_a_multipath_frame_costs_exactly_two_host_observations`
- `tests/test_phase5_budget.py::test_multipath_costs_no_more_per_frame_than_line_of_sight`
- `tests/test_phase5_budget.py::test_the_native_join_adds_no_host_observation_of_its_own`
- `tests/test_phase5_budget.py::test_freezing_a_multipath_leg_costs_four_copies_and_four_synchronizations`
- `tests/test_phase4_two_way.py::test_compose_performs_no_host_observation_at_all`
- `tests/test_phase4_two_way.py::test_freeze_host_reads_are_counted`

## Acceptance evidence (Phase 4)

- `tests/test_phase4_adapter.py::test_per_frame_budget_is_one_validation_copy_per_leg`
- `tests/test_phase4_adapter.py::test_freeze_is_never_called_per_frame` (a
  counting monkeypatch: two preparations total across five frames)
- `tests/test_phase4_spike_e2e.py::test_per_frame_host_traffic_is_two_copies_and_two_synchronizations`
- `tests/test_phase4_import_boundary.py::test_no_hot_path_host_observation_in_the_new_modules`
- `tests/test_phase4_spike_e2e.py::test_a_dead_row_contributes_exactly_zero_to_loss_and_gradients`
- `tests/test_phase4_adapter.py::test_jvp_publishes_delay_rate_and_refuses_to_invent_one`

## Amendment (Phase 7): a frame of slow-time slots costs one frame

A radar frame is many slow-time slots. Replaying a frozen topology once per slot
would publish exactly the right numbers and multiply this budget by the slot
count, so the budget is restated as a per-FRAME budget rather than a per-instant
one:

- one batched `ChannelPropagationAdapter.reevaluate_slots` per leg per frame,
  whatever the slot count is;
- two host observations for the whole frame, one validation copy per leg;
- zero synchronizations, zero compact-count copies, zero discovery launches;
- one waveform forward launch per frame.

The slot-major stack is block diagonal, so the pair count grows LINEARLY in the
slot count. Stacking both ends into a plain reevaluation instead would take the
source-by-sink outer product across slots and cost the SQUARE of it, which is
the shape this amendment exists to refuse.

The replication of the frozen topology over slots is index arithmetic and is
cached on the frozen handle, because it is a function of the topology and the
slot count alone. It belongs to the freeze, not to the frame.

Measured on the multi-endpoint fixture, two legs per frame, endpoint specs
hoisted (`ms/slot` is the whole two-leg replay divided by the slot count):

| slots | rows | pairs | ms | ms/slot | peak MB | host reads | syncs |
|---|---|---|---|---|---|---|---|
| 1 | 3 | 4 | 16.9 | 16.86 | 0.0 | 2 | 0 |
| 8 | 24 | 32 | 18.7 | 2.34 | 0.0 | 2 | 0 |
| 64 | 192 | 256 | 21.7 | 0.339 | 0.1 | 2 | 0 |
| 256 | 768 | 1024 | 20.9 | 0.082 | 0.5 | 2 | 0 |
| 1024 | 3072 | 4096 | 21.2 | 0.021 | 2.1 | 2 | 0 |

The Python per-slot loop the batched call replaces costs 589 ms and 128 host
reads at 64 slots, against 21.7 ms and 2. The 27x is not the point; the 128
host observations inside a frame are.

### Amendment (Phase 7): freeze-time host reads

`TwoWayComposer.freeze` and `DirectComposer.freeze` now also run
`synthesis.assembly.validate_pair_ordering`, which reads the composed pair index
on the host by design. That is one additional one-time `tolist` per freeze -
thirteen becomes fourteen - and it is what makes the sink-major layout assertion
non-empty in production (the plan's Phase-6 gap 5). The per-frame budget is
unchanged and still measured at zero.

## Acceptance evidence (Phase 7)

- `tests/test_phase5_budget.py::test_the_per_frame_host_budget_is_flat_in_slot_count`
- `tests/test_phase6_launch_budget.py::test_the_launch_count_is_flat_in_slot_count`
- `tests/test_phase7_slot_batching.py::test_batched_slots_equal_a_per_slot_loop`
- `tests/test_phase7_slot_batching.py::test_pair_count_grows_linearly_not_quadratically`
- `tests/test_phase7_slot_batching.py::test_the_batched_replay_is_exactly_one_consumer_call_per_leg`
