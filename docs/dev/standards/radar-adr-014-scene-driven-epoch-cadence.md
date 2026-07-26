# R-ADR-014: The scene-driven epoch cadence, and what a moving world costs

Status: Accepted (Phase 7)

## Context

R-ADR-005 made the adapter hold a `CompiledScene` as an opaque token, and Phase 7
gave it `refreeze` plus a `rediscovery_required` poll. What none of that answers
is the question a moving world actually asks: **for THIS frame, what has to be
rebuilt?**

The prices are not close. Measured on the multi-endpoint fixture, CUDA-synced:

| tier | work | cost |
|---|---|---|
| 0 session freeze | compile + evaluate + prepare | 2.55 + 9.10 + 0.74 ms |
| 1 motion event | compile + refreeze + freeze both legs + composer | 94 ms measured end to end at this fixture |
| 2 inner loop | one batched `reevaluate_slots` per leg, all slots | 18.7 ms for 8 slots, 2.34 ms/slot |

Three facts make the answer non-obvious.

1. **Core's `geometry_version` cannot be used as the compile trigger.**
   `core/witwin/core/scene.py` folds `time_s` and the endpoint states into
   `geometry_version` whenever the snapshot came from a `DynamicScene`, so a
   world with a completely static wall and one moving antenna reports a fresh
   geometry version at every instant. A loop that recompiled on that signal
   would rebuild the RayD scene and its BVH once per frame for nothing. This is
   recorded as Core gap C1 and is not patched here.
2. **A geometry-version mismatch is a legitimate production case, not only an
   error.** Channel's `_fixed_reflection` reads wall vertices from the compiled
   scene it is handed, and rigid motion and deformation both preserve face
   indexing, so replaying a frozen topology against a MOVED wall is numerically
   correct. Making the mismatch always fatal would forbid the moving-environment
   scenario; making it silently allowed is the stale answer the plan forbids.
3. **Replay is subtractive.** A frozen topology re-tested at new geometry
   publishes a row that stopped existing as `row_valid=False` with an exactly
   zero payload. A row that STARTED existing is absent, and nothing on the
   device reports its absence.

## Decision

### 1. `world_motion` is a caller declaration carried by `refreeze`

`ChannelPropagationAdapter.refreeze(compiled_scene, *, world_motion=...)`, and
the value is forwarded verbatim to every later `FixedTopologyRequest`.

- `"frozen_world"` (default) retires every frozen handle by advancing the
  adapter epoch. Unchanged behaviour, and Channel additionally refuses any moved
  version domain.
- `"fixed_winner_replay"` keeps the handles live. It asserts exactly one thing:
  *the discrete winner set is held fixed while the geometry moves*. The caller
  is accepting both consequences - a dead row is published inert, and a born row
  is not published at all.

The declaration lives on the rebind rather than on the replay call because the
rebind IS the moment the caller knows what it did to the world. A per-call
keyword would be a second source of truth for the same statement.

### 2. The compile trigger is the DECLARED descriptors, never a version

`SceneEpochLoop` compiles when `DynamicScene.structure_trajectories` or
`.structure_deformations` is non-empty, and otherwise exactly once for the whole
session no matter how many frames run. Endpoint and target motion goes into the
endpoint tensors, where it costs nothing. This is what routes production around
C1, and it is pinned by counting `scene.compile` calls rather than by argument.

### 3. The birth gap is closed by a declared cadence, not by a detector

`motion_event_period_frames` forces a rediscovery every N frames. The free
per-frame `rediscovery_required` poll compares the frozen rows against the
versions the compiled scene RECORDED, so it catches exactly the drift the
compiled scene already knows about; it cannot catch a born row, because a born
row leaves no trace in any version domain the frozen topology can be compared
against. Every cheap alternative is either a full discovery or a device
reduction plus a device-to-host copy the ADR-032 budget does not have room for.

Under `"fixed_winner_replay"` the poll ignores `geometry_version` and only that
domain, mirroring Channel's own rule: topology, material and assignment changes
respecify the labels the frozen rows carry and are never replayable.

### 3b. The same tick verifies the world the compiled scene was built from

The four version domains are content hashes, so a compiled scene and the rows
discovered on it agree with each other no matter what happens to the world
afterwards. A caller that edits mesh vertices in place therefore moves NO
version domain, the free poll reports `None` forever, and the loop replays a
world that no longer exists at full strength with every row valid. That route
bypasses the declared dynamics API, but "it was a misuse" is not a reason for
the answer to be silent.

`consumer.rediscovery_required(revalidate_source=True)` is the only thing that
can see it: it rehashes the live `witwin.core` world behind the compiled scene.
That is `O(scene)` host work which Channel's own docstring forbids in a replay
loop, so the loop runs it exactly on the motion-event tick - which already pays
a full discovery, where a host hash is invisible - and never per frame. When it
fires, the loop reports `SOURCE_MUTATION` and **recompiles**: the compiled scene
itself is stale, and rediscovering against it would reproduce the stale answer.

The signal is isolated rather than trusted wholesale. `revalidate_source=True`
answers the recorded provenance FIRST, so the loop treats only "the default poll
is silent AND the rehash is not" as a mutation; taking any non-`None` answer
would report a mutation on every ordinary moving-wall frame.

This makes `motion_event_period_frames=None` two declarations rather than one:
no path can be born, AND the authored world is never mutated outside the
`DynamicScene` API. A caller that cannot assert the second must declare a
period. Pinned by
`tests/test_phase7_invalidation.py::test_a_world_mutated_in_place_is_caught_on_the_motion_event_tick`,
`::test_a_declared_trajectory_never_reports_a_source_mutation` and
`::test_no_declared_cadence_means_no_revalidation_at_all`.

### 4. The loop owns *when*, the caller owns *what*

`SceneEpochLoop` never calls `reevaluate`. It takes a `bind` callback that
freezes whatever the caller freezes, and its `compile_scene` is an argument
rather than an import - the adapter stays the only Radar module that names
`witwin.channel`, and the compile count stays observable.

## Consequences

- A moving-structure sequence is T compiles and T replays and is a motion-event
  cadence, never an inner loop. All slots of one batched replay share ONE
  compiled scene, which is what "one frame / one pulse train / one symbol block"
  means physically.
- `world_motion` joins the frozen reevaluation keyword set in
  `tests/test_phase6_config_boundary.py`. It describes scene geometry, which is
  what a propagation request is about; it carries no waveform, ADC or receive
  chain vocabulary and could not, since the vocabulary is Channel's own closed
  set.
- Two host integer comparisons per frozen handle per frame, and nothing else, is
  the whole per-frame cost of the cadence. Measured at zero `.item()`, `.cpu()`,
  `.tolist()`, `.numpy()` and zero synchronizations.
- The `O(scene)` source rehash is charged to the motion-event tick only, and a
  loop with no declared period performs it zero times, so the per-frame budget
  above is unchanged by decision 3b.

## The boundary this stage found, and did not cross

**Structure motion does not reach `delay_rate`.** The published rate is the
propagation JVP of the ENDPOINT position tangents. A wall's vertices reach the
same kernel through the compiled scene and carry no tangent, so a wall moving at
4 m/s produces a real, correct delay evolution - measured against the image
source at 4.4e-5 relative - and an exactly zero `delay_rate`.

That zero is correct for what `delay_rate` is defined to be. It would be a
serious defect if read as "the environment is not moving", so
`test_structure_motion_does_not_reach_the_endpoint_delay_rate` pins it by name.
Carrying environment motion into the tangent channel needs a vertex-tangent
route through the native reflection kernel, which is a numerical change with its
own decision record and is not attempted here.

## Recorded, not patched

- **Core C1** - `geometry_version` folds `time_s` and endpoint states, so
  endpoint-only motion over a static wall reports four distinct geometry
  versions for four snapshots. Minimal repro is in
  `tests/test_phase7_invalidation.py::test_endpoint_only_motion_does_not_recompile`,
  which asserts both halves: the loop compiles once, and the snapshots it
  declined to recompile really would have hashed differently.
- **Core C2** - `DeformationState` carries no velocity. Routed around by
  `witwin.radar.propagation.kinematics.DeformationVelocity` (R-ADR-012) with two
  implementations landed here: `LinearDeformation`, which is simultaneously a
  Core `Deformation` and a `DeformationVelocity`, and
  `witwin.radar.geometry.SmplPoseDeformation`, which differentiates the SMPL
  posing function itself under forward-mode AD with the pose rate as tangent.
- **Channel does not export `WORLD_MOTIONS`** next to `AD_MODES`, `RESPONSES`
  and `TOPOLOGY_MODES`. The adapter reads the same frozen set from
  `capabilities().world_motions`, which is a complete published route, so no
  Channel change was needed.
