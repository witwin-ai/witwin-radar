# AD tape and budget ledger

Every autograd tape in the Radar package, one row per owner, with what it saves,
how many bytes that is as a function of the fixture, how many kernel launches
the forward and the reverse cost, how long the reverse takes, and - the column
this document exists for - **when the context is created and when it is
released**.

A tape is the one part of an AD implementation that is invisible in every
correctness test. A backward that saves twice what it needs produces exactly the
right gradient. So does one that retains sixty-five contexts where one would do.
The only way either shows up is if somebody writes the number down.

Everything below was MEASURED, in the `witwin2` environment, with the packaged
prebuilt loaded (never a JIT rebuild), on an NVIDIA GeForce RTX 5080 that
`nvidia-smi` reported at 2-3 percent utilization and 4.7-5.2 GB in use by
desktop processes throughout. Nothing here is an estimate.

**This document is parsed, not just read.** A mutation run falsified the join
bytes formula, its measured value and a quoted wall-time budget by factors of
two to ten and nothing in the tree noticed: the budgets themselves are pinned by
constants in `tests/test_phase9_backward_budget.py` and by the fixture-derived
band law, so no enforcement was weakened, but the prose was free to drift away
from them. Three tests in that module now read this file -
`::test_the_ledger_bytes_formula_is_the_law_this_module_measures` evaluates the
join formula below at the fixture's own row counts and compares it with bytes
read out of a live context, and
`::test_the_ledger_budget_table_quotes_the_live_constants` /
`::test_the_ledger_channel_accounting_table_quotes_the_live_constants` compare
the two tables against the module's constants. Editing a number here without
editing the constant fails.

## How to read a row

- **saved tensors** are the literal names, in save order, so the row is
  checkable against the `save_for_backward` / `save_for_forward` call rather
  than against somebody's memory of it.
- **bytes formula** is symbolic in the fixture dimensions, so the row PREDICTS.
  A row that only reports a number describes one run and rots on the next
  fixture change. Symbols: `K` composed rows, `R_in` / `R_out` leg rows, `S`
  response slots (sites), `T` transmitters, `R` receivers, `P` pair segments,
  `N` samples or targets, `G` AGC groups, `F` band columns.
- **measured bytes** is the formula evaluated at the pinned fixture, taken by
  reading `ctx.to_save` inside `setup_context` - the only moment the storage is
  both allocated and legal to inspect.
- **fwd / bwd launches** are counted by wrapping the native operator table.
- **bwd ms** is `torch.cuda.Event` timing, 10 warm-up calls then the median of
  50, with the graph rebuilt outside the timed region on every sample so the
  number is the reverse pass alone.
- **lifetime** is when the context comes into existence and what has to happen
  for it to go away.

The fixtures are deliberately small - two rows, two chirps, thirty-two samples -
because a tape row is a contract statement and a production-sized frame answers
it no better. At these sizes the `bwd ms` column is dominated by launch
overhead, not by arithmetic; it is recorded as the floor cost of a companion
launch, and the budget pins that matter are on the whole pipeline further down.

## The eight tape owners

The fixture is `tests/support/ad_boundaries.py`, one boundary per owner. There
are seven boundaries and eight owners: `frontend` runs two contexts in one call.

There were nine boundaries and ten owners until Phase 11 deleted the Dirichlet
route. The two rows it removed - the chunked spectrum and the MIMO-linear frame
- are struck from the table below rather than kept as history, because this
document is read as a description of the live tape and a row for a context that
cannot be created is worse than no row.

| family | tape owner (file:line) | saved tensors | bytes formula | measured bytes @fixture | fwd launches | bwd launches | bwd ms (measured) | lifetime |
|---|---|---|---|---|---|---|---|---|
| two-way join | `witwin/radar/paths/two_way.py:240/241` | `c_in_re`, `c_in_im`, `c_out_re`, `c_out_im`, `s_re`, `s_im`, `row_valid`, `idx_in`, `idx_out`, `idx_s` | `8*R_in + 8*R_out + 8*S + 28*K` | 104 B at `R_in=R_out=S=K=2` | 1 `two_way_join_forward` | 1 `two_way_join_backward` | 0.412 | created in `TwoWayComposer.compose`; released when the composed `RadarPathBatch`'s graph is freed. Under `_compose_band` there is **one context per frequency column plus one**, and all `F+1` live until the band's graph is freed - see the band section below. |
| aspect response | `witwin/radar/scattering/aspect.py:159/160` | `dir_in`, `dir_out`, `axis`, `amplitude`, `phase_rad`, `idx_in`, `idx_out`, `idx_site`, `row_valid` | `12*(R_in + R_out + S) + 8*S + 28*K` | 144 B at `R_in=R_out=S=K=2` | 1 `scatter_response_aspect_forward` | 1 `scatter_response_aspect_backward` | 0.276 | created inside `AspectScatterResponse.evaluate_rows`, which the composer calls once per compose; released with the composed batch. The two direction tables are the legs' own aliased tensors, so this context pins the legs' geometry alive as well as its own. |
| sensor weight | `witwin/radar/sensors/weights.py:405/406` | `tx_pos`, `rx_pos`, `site_in`, `site_out`, `intensity`, `weight_re`, `weight_im`, `tx_velocity`, `rx_velocity` | `24*T + 24*R + 36*K` | 120 B at `T=R=1`, `K=2` | 1 `sensor_weight_forward` | 1 `sensor_weight_backward` | 0.348 | created in `evaluate_sensor_weights`, once per frame; released with the `SensorWeightResult`'s graph. The geometry and the plan are attached to the context as configuration, not saved as tensors, so neither is retained storage. Since Phase 11 the PRODUCTION creator on the scene-driven route is `RoundTripPatternStage.apply` (`witwin/radar/sensors/round_trip.py`), which calls the same facade and therefore reuses this one context rather than defining a `Function` of its own - it added no owner to this table. It exists only when `Radar.simulate` was given an `antenna_pattern`; a composed BAND adds one context per frequency column, exactly as the join's band loop does. |
| FMCW beat | `witwin/radar/synthesis/fmcw_beat.py:141/144` | backward: `tau_rt`, `tau_rate`, `weight_re`, `weight_im`, **`segment`**, `tx_index`; forward: `tau_rt`, `tau_rate`, `weight_re`, `weight_im`, **`offsets`**, `tx_index` | backward `16*K + 8*K + 4*T`; forward `16*K + 8*(P+1) + 4*T` | 52 B at `K=2`, `P=1`, `T=1` | 1 `fmcw_beat_forward` | 1 `fmcw_beat_backward` | 0.313 | created in `synthesize_beat_rows`; released with the cube's graph. **The two lists differ**: the backward needs a per-ROW segment id to reduce into, the jvp needs the per-SEGMENT offsets to walk. That is not an inconsistency, and it means the reverse and forward tapes have different sizes whenever `K != P+1`. |
| OFDM CFR | `witwin/radar/synthesis/ofdm_cfr.py:118/119` | backward: `tau_rt`, `tau_rate`, `weight_re`, `weight_im`, **`segment`**; forward: same four plus **`offsets`** | backward `16*K + 8*K`; forward `16*K + 8*(P+1)` | 48 B at `K=2`, `P=1` | 1 `ofdm_cfr_forward` | 1 `ofdm_cfr_backward` | 0.332 | as FMCW, same asymmetry, no `tx_index`. |
| pulsed echo | `witwin/radar/synthesis/pulsed_echo.py:151/152` | backward: `tau_rt`, `tau_rate`, `weight_re`, `weight_im`, **`segment`**; forward: same four plus **`offsets`** | backward `16*K + 8*K`; forward `16*K + 8*(P+1)` | 48 B at `K=2`, `P=1` | 1 `pulsed_echo_forward` | 1 `pulsed_echo_backward` | 0.341 | as OFDM. |
| frontend noise | `witwin/radar/frontend/chain.py:173/174` | `phase_rad` (the realised phase, taken from the OUTPUT) | `4*N` | 1024 B at `N=256` | 1 `frontend_noise_forward` | 1 `frontend_noise_backward` | 0.585 (both frontend contexts) | created in `FrontendChain.apply`; released with the chain output's graph. Saving the realised phase rather than the generator state is deliberate: the derivative is taken at the phase the primal actually used, so the two are exactly consistent and the backward holds no second copy of the RNG. |
| frontend AGC | `witwin/radar/frontend/chain.py:260/261` | `x_re`, `x_im`, `gain`, `rms` | `8*N + 8*G` | 2056 B at `N=256`, `G=1` | 1 `frontend_agc_forward` | 1 `frontend_agc_backward` | (included above) | created in the same `apply` call, immediately after the noise context; released with the same graph. `gain` and `rms` are outputs marked non-differentiable and saved anyway, because the backward of a normalisation needs the normalisation it actually applied. |

**One backward launch per forward launch, at every one of the eight.** That is
R-ADR-004's shape and it is now pinned at every boundary rather than at the
three synthesis families `tests/test_phase6_launch_budget.py` covers:
`tests/test_phase9_backward_budget.py::test_each_boundary_costs_one_backward_launch_per_forward_launch`
and `::test_the_frontend_costs_one_backward_launch_per_forward_stage`. The set
of budgeted boundaries is itself asserted to equal the set of autograd owners,
so an eleventh `Function` cannot arrive unbudgeted.

## The band loop, which is the lifetime finding

`_compose_band` (`witwin/radar/paths/two_way.py`) calls `_TwoWayJoin.apply` once
per frequency column plus once for the reference column, and each call retains
its own ten-tensor context. The survey read that as "a 64-subcarrier band holds
64 copies of the join tape". That is an **exact statement about contexts and a
five-fold overestimate of the memory**, and the difference is measurable:

```
F      contexts   total tape B   distinct-storage B   live fwd B   peak fwd B
1          2            808              484             79360       104960
2          3           1212              564             96256       119296
4          5           2020              724            151552       169472
8          9           3636             1044            262656       273920
16        17           6868             1684            484864       499712
32        33          13332             2964            929792       953344
64        65          26260             5524           1818624      1859072
```

At the pinned fixture (`R_in=3`, `R_out=7`, `S=2`, `K=11`):

```
contexts   = F + 1
per context= 8*R_in + 8*R_out + 8*S + 28*K            = 404 B
total      = (F + 1) * 404
distinct   = (8*S + 28*K) + 8*(R_in + R_out)*(F + 1)  = 324 + 80*(F + 1)
```

Six of the ten saved tensors are the **same storage in every context** - the
scatter-response pair, the validity mask and the three index tables - because
`_compose_band` evaluates the response once above the loop and the join's index
tables are frozen at freeze time. Only the four per-column coefficient slices
are distinct. So the marginal retained tape per column is
`8*(R_in + R_out) = 80 B`, not 404 B.

**The honest conclusion, and it is not the one the survey expected.** At `F=64`
the tape is 5.5 kB against 1.80 MB of total retained forward allocation. The
tape is not the thing to bound here; the 65 sets of `[K]` complex outputs are.
The tape law is pinned anyway, because it is the thing that would change
silently if the join's save list changed, and because a pinned law catches that
at any band width.

Pinned by
`tests/test_phase9_backward_budget.py::test_the_band_loop_tape_obeys_its_predicted_linear_law`
(widths 1, 2, 4, 8, predicted from the fixture's own row counts rather than
written down as constants) and
`::test_the_band_loop_tape_law_holds_at_a_width_it_was_not_fitted_on` (width
16). The aliasing itself is pinned structurally by
`tests/test_phase9_wideband_join_ad.py::test_the_band_loop_keeps_one_join_context_per_column_and_aliases_its_tables`.

## The Channel half, read-only

Channel's ADR-043 (consumer `CONTRACT_VERSION` 6) added
`ad_companion_launches` and `ad_tape_bytes` to `PropagationDiagnostics`
precisely so the ledger could cover the whole chain rather than the Radar half.
Measured through the Radar adapter at the multi-endpoint fixture, per
`reevaluate` call:

| route | ad_mode | rows | `ad_companion_launches` | `ad_tape_bytes` |
|---|---|---|---|---|
| `reevaluate` inbound leg (line-of-sight only) | `none` | 3 | 0 | 0 |
| `reevaluate` outbound leg (line-of-sight + reflection) | `none` | 7 | 0 | 0 |
| `reevaluate` inbound leg | `vjp` | 3 | 2 | 200 |
| `reevaluate` outbound leg | `vjp` | 7 | 2 | 496 |

Two things this table says that a single number would not. The **vjp-only tape
gate is real**: a primal solve reports an exact zero rather than forwarding the
sidecar's raw count, which is what the ledger's own contract requires. And the
**two legs differ**, because one carries reflections and one does not - an equal
pair would mean the accounting was reporting a constant, which is asserted
directly rather than assumed.

Pinned by
`tests/test_phase9_backward_budget.py::test_the_channel_reevaluate_publishes_its_ad_launches_and_tape_bytes`
and `::test_a_primal_only_reevaluate_builds_no_tape_at_all`.

## The budget pins

Five, and no more than five. A pinned wall-time number is maintenance debt, so
what is pinned is what is budget critical. Every constant is the WORST median
observed over four independent processes, with the headroom applied on top of
that rather than on top of the luckiest run.

| pin | measured | budget | headroom | test |
|---|---|---|---|---|
| full FMCW pipeline, BACKWARD wall time | 2.68 ms (four medians: 1.816, 1.925, 2.153, 2.684) | 3.484 ms | 1.30x | `tests/test_phase9_backward_budget.py::test_the_full_fmcw_pipeline_backward_meets_its_time_budget` |
| full FMCW pipeline, backward peak ALLOCATION | 0.1426 MB (149504 B, identical on all four runs) | 0.1782 MB | 1.25x | `::test_the_full_fmcw_pipeline_backward_meets_its_peak_memory_budget` |
| the same, forward peak, for the ratio | 43008 B forward vs 149504 B forward-plus-backward = 3.48x | reverse > 2x forward | - | `::test_the_backward_peak_is_larger_than_the_forward_peak` |
| Channel `reevaluate`, two legs, reverse cost as a RATIO to the forward | 1.334 to 1.523 over six processes, sampled alternately | 2.0, a structural threshold | - | `::test_the_channel_reevaluate_reverse_pass_is_a_surcharge_not_a_second_solve` |
| `_compose_band` tape law | exact, see above | exact equality | - | `::test_the_band_loop_tape_obeys_its_predicted_linear_law` |

The reverse pass costs about a **fifty-percent surcharge** on the Channel
forward, not a second solve, and about **3.5x the forward's peak allocation**.
Those two ratios are the portable statements; the millisecond constants describe
this machine.

### Why the Channel pin is a ratio and not two wall times

This is a measurement result and not a preference, and it is worth recording
because the obvious pin is the one that does not work.

The absolute medians of the two-leg `reevaluate` call drift over a 1.5x range on
this machine between processes on an idle device: forward 3.636 to 5.542 ms,
forward-plus-backward 5.326 to 7.071 ms, twelve samples in three processes. Both
quantities drift together - allocator state and clock ramp move them the same
way - so an absolute budget set at the tightest observation fails on the first
cold run of a session, and one set at the loosest catches nothing. A first
version of this pin used 3.72 ms and 5.68 ms with a 1.30 factor; it failed on
the first run of the very next session, at 8.14 ms, which is how the problem was
found rather than shipped.

The quotient is stable when the two medians are sampled ALTERNATELY in one
loop - 1.334 to 1.523 over six processes - and noticeably less so when they are
timed one after the other, 1.342 to 1.841 over four. Interleaving puts both
under the same drift.

The budget of 2.0 is a structural threshold rather than a measured number with a
factor bolted on: the reverse pass rides the topology the forward already
solved, so a backward that started re-solving geometry would cost at least a
second forward and could not come in under 2x. The pin also asserts the ratio is
above 1.0, so a backward that stopped running would fail rather than pass
comfortably.

### The cold-clock caveat, measured

An idle RTX 5080 sits at 877 MHz. The first `pytest` invocation of a session can
miss a wall-time budget by about one percent purely on clock ramp: measured
while writing this document, `tests/test_phase8_pipeline_budget.py` reported
2.9225 ms against a 2.8990 ms budget on the first run of a session and 2.7985 /
2.7421 ms on the two immediately following runs, on the same unchanged tree.
Twenty warm-up calls do not boost the clock from idle.

That is a property of the device, not of the code. **The correct response to a
single first-run miss is to re-run.** It is never to widen a factor, and the
existing Phase-8 constants were not changed.

## Tape non-leak

The tape stays inside its owner, and both halves of that are now tested rather
than inspected:

- **No public result carries a tape.** Asserted on real objects produced by the
  production chain with a LIVE graph on them - a `RadarLegBatch`, a
  `RadarPathBatch`, a `SynthesisResult`, a `SensorWeightResult`, a
  `FrontendOutput`, the Channel `PreparedFixedTopology` the adapter holds, a
  `PointCloud` and a `DetectionFrame` - by walking every field transitively.
  A `grad_fn` is not a leak and is not what is being looked for; a field holding
  the saved tensors or the context that owns them is.
  `tests/test_phase9_tape_non_leak.py`, four tests plus a calibration that
  plants a context in a record and checks the walker objects to it.
- **No module outside a tape's own owner reads one.** Every
  `ctx.saved_tensors` / `ctx.to_save` read in the package is located by parsing
  the source: there are exactly **20**, in exactly the **8 owner files**, each
  inside a `backward` or a `jvp`. Ten owners, two reads each. The scan's limit
  is stated in the test: a read through an alias or through `getattr` would not
  be found.
  `::test_every_context_read_sits_inside_a_tape_owner`,
  `::test_the_context_scan_is_not_vacuous`,
  `::test_no_production_module_stores_a_context_on_an_object`.

## What Phase 9 added to the production graph, by category

The phase put roughly a thousand lines of guard and orchestration into the
package. Every new Torch expression in it falls into exactly two categories, and
`tests/test_phase6_no_torch_physics.py` now pins that there is no third:

1. **Refusal predicates.** `torch.is_grad_enabled`,
   `torch.autograd.forward_ad.unpack_dual`,
   `torch.autograd.function.once_differentiable`. They ask Torch a question and
   never construct a value. `witwin/radar/ad_contracts.py` and
   `witwin/radar/host_parameters.py` are allowed nothing else, and are asserted
   to contain no `detach` and no `zeros_like` - the two ways a guard could
   answer a question it cannot answer while looking like it did.
2. **Result construction, one expression.** `rcs_amplitude`'s `torch.sqrt`,
   which is `sqrt(4 pi sigma)/lambda`. It runs once per response rather than
   once per path, off the per-path loop, and every product downstream of it is
   still a native kernel. The capability matrix records its mechanism as
   `torch-orchestration` for exactly that reason. `scattering/rcs.py`'s whole
   matched Torch set is pinned by equality, so a second arithmetic expression
   there fails.

One `requires_grad` route branch survives in the package and is recorded rather
than removed: `SMPLBody._evaluate` nudges a grad-carrying shape by `1e-8` to
keep the SMPL layer's own backward defined. It is a legacy numerical workaround
inside a legacy path that IS driven to a loss, and deleting it is a numerical
decision with its own evidence rather than an architecture cleanup. Pinned in
both directions by
`tests/test_phase6_no_torch_physics.py::test_no_phase9_guarded_package_gates_a_route_on_requires_grad`
and `::test_the_one_recorded_requires_grad_route_still_exists`, so a stale
allowlist entry fails as loudly as a new branch.

## Reproducing every number here

```
conda activate witwin2
cd <radar worktree>
python -m pytest -q tests/test_phase9_backward_budget.py --gpu --basetemp=<short>
python -m pytest -q tests/test_phase9_tape_non_leak.py --gpu --basetemp=<short>
python -m pytest -q tests/test_phase6_no_torch_physics.py --basetemp=<short>
```

The per-owner rows come from wrapping each `Function`'s `setup_context` and
reading `ctx.to_save`; `tests/support/ad_boundaries.py` is the fixture and
`tests/test_phase9_backward_budget.py` is the pinned subset. Run on an idle GPU,
with the packaged prebuilt at
`witwin/radar/cuda/prebuilt/_radar_native.pyd` - never a JIT
rebuild inside a test process.
