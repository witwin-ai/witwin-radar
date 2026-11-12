# R-ADR-024: Scene-session structure and orchestration reuse

Status: Accepted (2026-11-12)

## Context

Every public verb runs one scene session. Before this record the session was
one 499-line function in `witwin/radar/simulation.py` with six nested closures
that passed positional tuples, and the module also held the sampling record,
the scene binding, the adaptive controller and the published records. Its cost
outside the kernels was paid per observation: one join build (four nested Python
loops, six device-to-host reads per leg, fourteen uploads) per rediscovery, two
replays, one composition, one antenna-pattern launch and one synthesis launch
per observation, and for every frame of a world that does not move.

`tests/test_session_budget.py` counts that cost per frame and per `Motion` kind.

## Decision

### 1. One owner per session concept

| Module | Owns |
| --- | --- |
| `motion.py` | `Motion`, including the adaptive tolerances (there is no second spec record) |
| `binding.py` | stable IDs, the scatter-site policy, `SensorEndpointIds`, `bind_radar_world` |
| `adaptive.py` | the host probe controller and the interpolated echo of its partition |
| `results.py` | `Result`, `Frame`, and the per-frame `SceneFrame` hand-off |
| `simulation.py` | `Session`, `WorldHalf` (epoch loop and every Channel call), the instrument half, `Paths`, the verbs |

The waveform dispatch stays the private `Radar._synthesize`. Records
replace positional tuples: `FrozenRoundTrip` is an epoch's payload,
`EvaluatedObservation` and `Observation` are instants, `SlotBatch` is one
grouped replay. `ci/mypy.ini` types these modules strictly.

### 2. Reuse is keyed by identity, never by time

- **Join.** A leg's identity is read to the host in one copy and packed into
  bytes (row count, sequence width, every identity column). A rediscovery whose
  two packed identities and declared site IDs equal the retiring epoch's reuses
  its join and pattern tables, which are functions of exactly those. The join
  is built by vectorized numpy on the host and uploaded in one copy. It is not
  a native kernel, deliberately: the identity is read to the host anyway to
  decide reuse and to name a refused row, the build is integer index
  construction with no derivative, and with reuse it runs once per distinct
  topology rather than per observation, so it is not a hot path.
- **Still world.** A session with no sampled motion and no declared
  rediscovery cadence (`Motion.rediscover_every_frames is None`) replays the
  same endpoints against the same frozen topology every frame. Its first
  frame's rows are published again for every later frame that did not
  rediscover, and its waveform is synthesized once; the receive chain runs per
  frame. A declared cadence opts out, because only the cadence tick can detect
  a world mutated in place.

### 3. Grouped replay and slot-batch synthesis

Non-adaptive sampled sessions evaluate a frame's instants in groups of
`Motion.batch_observations`. Topology-identical instants of a group share one
slot-major replay and one composition; an instant replays the frozen handle of
the first instant of its group when their packed identities are equal. A
refreshed FMCW frame is synthesized per group as one launch over only the pair
segments the frame cube reads - the slot's own transmitter. On the ADC route
every segment is a different ADC instant, so the beat family takes an optional
per-segment window start (`segment_t_start`, float64 seconds) in place of the
uniform `t_start`; it replaces exactly that scalar of the phase expression and
leaves each segment's path order unchanged, so the group is still one launch
and every value, cotangent and tangent is the bit a one-instant launch gives.
Oscillator phase noise and a declared coupling keep one launch per
observation, because they act on or add rows between replay and synthesis.

The fused verbs therefore retain one replay group, at most
`Motion.batch_observations` observations, instead of one observation. That is
the bound `AGENTS.md` states.

### 4. Derivatives are never shared across frames

A held row batch or synthesis is reused only when it carries neither a reverse
tape nor a forward tangent. Sharing a graph across frames would sum every
frame's cotangent at the shared node before the replay's backward instead of
after it, which is the same gradient but not the same bits. Differentiable runs
take the per-frame route, and `simulate` stacks their cubes with `torch.stack`
rather than writing into a preallocated cube.

## Consequences

- Every route is bit-identical to the previous session (`torch.equal` on cubes,
  last composed rows, metadata, adaptive diagnostics and VJP/JVP gradients) on
  the capture set listed in the session report.
- Per-frame counts are pinned in `tests/test_session_budget.py` and only move
  down.
- Not done here: replaying only the active transmitter's inbound rows needs a
  Channel replay of a row subset, and a slot stride in the join and
  sensor-weight kernels (instead of the replicated `for_slots` tables) needs
  native work in both families.

## Verification

```bash
pytest tests/test_session_budget.py tests/test_trace_echo_split.py --gpu
python -m mypy --config-file ci/mypy.ini
python ci/check_architecture.py
```
