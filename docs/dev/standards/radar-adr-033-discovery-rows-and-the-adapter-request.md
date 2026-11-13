# R-ADR-033: A primal run reads its rows from discovery, and the adapter forwards Channel's discovery contract

Status: Accepted

## Context

Channel 0.6 (consumer contract 7) added four things to discovery that Radar did not use (plan section 2.5):
`PropagationEvaluation.complete`, `PropagationRequest.discovery`, `PropagationRequest.max_paths` and
`consumer.check_endpoint_separation`. Without them `path_set_complete` could be true while Channel's own reflection
discovery had fallen back to rays (C-13, MP-6); a site on an array element produced a finite 7.6e29 W echo instead
of a refusal (survival case `site_at_element`); and nothing bounded the host cross join before it was built (MP-14).

Separately, every rediscovery paid three Channel calls per leg - discover, prepare, replay - for one instant, and
discarded the discovery's own delay and coefficient (P-4). Because a frozen topology may carry only `{los,
reflection}`, the same route refused `transmission` and `diffraction` outright (MP-7, MAT-2): every mesh blocked the
line of sight completely.

## Decision

### 1. Discovery publishes its rows; preparation is lazy

`ChannelPropagationAdapter.discover` runs one `consumer.evaluate` and returns a `FrozenLegTopology` that carries the
discovery `topology` (stamped with its world provenance), the identity columns, Channel's `complete` flag, the
discovery's `transmission_dropped_rows`, and - with `rows=True` - the discovered rows as a `RadarLegBatch` (`row_valid`
all true: a discovered row exists by construction). `FrozenLegTopology.prepared` calls
`consumer.prepare_fixed_topology` on first use and caches it; `freeze()` is `discover(rows=False)` plus that first use.
The rediscovery poll and the epoch check read the discovery topology, which Channel accepts.

### 2. A primal group that rediscovered composes from discovery

`WorldHalf._group_legs` answers a group of instants from their discoveries when the run is `grad="none"`, every
instant in the group rediscovered, and no wideband band is requested. A group of several instants is stacked slot
major by `propagation.stack_slots` (one `torch.cat` per member) and composed by the same slot tables a slot-batched
replay uses. Every other group is replayed exactly as before.

Equality. Discovery and a replay at the same endpoints run the same native field evaluation for `los` and
`reflection` rows: on the wall fixture the delay, coefficient, field direction and interaction positions are
`torch.equal` (checked by hand on 6 and 12-row legs, and end to end by
`tests/test_multipath_discovery.py::test_m3_2_primal_rows_from_discovery_equal_the_replay_route`, which forces the
replay route on the same run and asserts `torch.equal` on the cube, the delays and the coefficients for the chirp, ADC
and adaptive routes, and `test_m3_6_direct_rows_survive_the_slot_batched_and_replayed_routes` with the site-less leg).
The one place the two routes can differ is a row Channel's replay marks invalid at the instant it was discovered
(C-16, a site within ~0.05 mm of a wall); there discovery is the authority, because it is the evaluation that found
the row.

Calls per rediscovered observation, two legs: before 2 `evaluate` + 2 `prepare_fixed_topology` + 2 `reevaluate`;
after 2 `evaluate` (`test_m3_2_a_rediscovering_primal_frame_makes_one_channel_call_per_leg`). A static world: 2
`evaluate`, no preparation, no replay for the whole run. A differentiable run is unchanged (it replays, because only a
replay carries a derivative).

### 3. Transmission and diffraction are admitted where nothing is replayed

The verbs take `transmission=` and `diffraction=` next to `los=`/`reflections=`. Channel cannot freeze either, so a
session that names one opens its adapter with `replay=False` and refuses by name, before any discovery:

- `grad="vjp"`/`"jvp"` (the derivative lives on the replay);
- a discovery cadence longer than one evaluated instant (`Motion.rediscover_every_frames` on a moving or polled
  world), which would replay frozen rows between discoveries;
- a wideband band (R-ADR-035), which is a replay at every subcarrier.

A handle that cannot be frozen also refuses `prepared` by name, as a backstop. Channel's single `max_depth` bounds the
walls a transmission path crosses and the edges a diffracted one meets as well as the reflection order, so a run
with `transmission` or `diffraction` keeps `max_depth >= 1` even at `reflections=0`. `transmission_dropped_rows` is
published on `Paths`, `Result` and each frame.

Oracles (`tests/test_multipath_discovery.py`): a target behind a 0.1 m, `eps_r = 4` slab returns
`|T|^2` times the free-space radar equation, `T` the Airy slab transmission, to 1e-4; a target in the shadow of a
PEC half-plane returns the square of the Kouyoumjian-Pathak UTD soft-polarization amplitude (`n = 2`, spherical `L`)
to 1 %. At the leg level Channel's slab transmission matched the Airy value to 6e-7 and the knife-edge row the UTD
value to 0.16 %.

### 4. The discovery contract is forwarded

- `discovery=` (`"auto"`, `"exhaustive"`, `"rays"`) on the four verbs, forwarded to every discovery.
- `complete` is ANDed into `path_set_complete` per frame (`DiscoveryReport` folds every discovery a frame answered
  from, including the epoch it opened on) and published separately as `discovery_complete`.
- `max_paths=` is forwarded; Channel raises on overflow and never truncates.
- A leg whose request includes a straight-segment component (`los`, `transmission`) calls
  `consumer.check_endpoint_separation` once per discovery, i.e. once per topology epoch. The check is one one-byte
  device read; it turns the `site_at_element` survival case into a named `ValueError` (its strict xfail is removed). A
  reflection-only leg is not checked: a reflection from a point back to itself is well defined.

### 5. `max_round_trips` is checked before the join

`TwoWayComposer.freeze(..., max_round_trips=)` computes each sensor pair's composed row count as
`sum_s A[s, t] B[s, r]` from per-site row counts (one small host matrix product) plus the site-less rows, and refuses
by name - the transmitter and receiver IDs, the count and how many pairs exceed it - before the cross join is
materialised. Default `None` (no bound), so no existing scene changes.

## Measurements and the M3-4 decision

Counts (`tests/test_session_budget.py`, re-pinned; GPU 1, the larger of two processes for profiled columns), first
frame before -> after: `adc_point` (32 observations) Channel calls 64 evaluate + 64 prepare + 2 replay -> 64
evaluate, host synchronizations 2922 -> 358, device-to-host copies 1485 -> 331, CUDA kernels 16839 -> 8204;
`adc_structure` synchronizations 3866 -> 1178; `chirp_point` 402 -> 78; `static` 129 -> 45. These include the new
separation check.

Discovery cost for Radar-shaped requests (M3-4). A 3-wall PEC room, `{los, reflection}`, sources = sites, sinks = 4
receivers; GPU 1 shared with other jobs (contended), minimum of 5 runs, milliseconds per `evaluate`:

| depth | sites | one call | rows | the same sites as one call each |
|---|---|---|---|---|
| 1 | 1 | 27.6 | 16 | 26.0 |
| 1 | 16 | 14.1 | 256 | 339 |
| 1 | 128 | 30.1 | 2048 | 3044 |
| 1 | 1024 | 27.7 | 16384 | - |
| 2 | 1 | 45.9 | 32 | 38.1 |
| 2 | 16 | 46.2 | 512 | 665 |
| 2 | 128 | 45.9 | 4096 | 5482 |
| 2 | 1024 | 46.2 | 32768 | - |

One call's cost does not grow with the site count up to 1024 sites; it is a per-call cost of tens of milliseconds
set by the depth. Splitting a leg into Radar-built partial topologies would multiply exactly that cost, so Radar does
not split handles. What reduces it is fewer calls per rediscovery: this record halves them for primal runs, and the
remaining factor of two (inbound and outbound legs) needs one Channel request - see below.

Channel request (MP-4, M3-4/M3-5). Let one `PropagationRequest` carry several source/sink groups, or a pair mask over
`sources x sinks`, and publish each group's rows in its own pair partition, so a radar discovers TX -> sites and RX ->
sites (the reversed outbound leg, which the `transport` work is introducing for the polarimetric site matrix) in one
discovery with one fixed cost; and let one discovery serve several instants of the same endpoint set (slot-batched
discovery), which is what a moving target's per-observation rediscovery pays for today.

## Consequences

- `FrozenLegTopology` no longer stores `prepared` as a field; `prepared`, `components` and the three prepare counters
  are properties that prepare on first use.
- The session budget counts move: a primal rediscovering observation no longer prepares or replays, and each
  straight-segment discovery adds one separation check (one D2H byte, one synchronization).
- M3-5 (evaluating the outbound leg reversed) is owned by the `transport` work, which needs the reversed orientation
  for the polarimetric site matrix; one Channel request for TX and RX toward the sites builds on that.

## Amendment: the published Channel 0.6.0

The C-16 disagreement near a wall is closed in the published Channel 0.6.0 (tag c287e11): replay
applies the same endpoint-exempt minimum-segment rule as discovery. Discovery remains the authority
for which rows exist; the two no longer differ near a wall.
