# R-ADR-022: Channel 0.6 adoption

Status: Accepted (Radar 0.5.0)

## Context

Radar reaches Channel only through `witwin/radar/channel.py`, which uses the consumer facade
(`capabilities`, `EndpointBatch`, `PropagationRequest`, `evaluate`, `prepare_fixed_topology`,
`replicate_over_slots`, `FixedTopologyRequest`, `reevaluate`, `rediscovery_required`) and the
compile facade `witwin.channel.scene.compile`. Channel 0.6.0 (tag `witwin-channel-v0.6.0`, RayD
0.10.2 `a98e09d` source-linked) moves the consumer contract from version 6 to 7 and changes which
reflection rows exist, when a compiled scene may be evaluated, and how a moving mesh recompiles.
Radar 0.4.1 pinned `witwin-channel>=0.5.0,<0.6`.

## Decision

1. **Requirement.** The `channel` extra is `witwin-channel>=0.6.0,<0.7; python_version < '3.14'`.
   Channel 0.6 publishes CPython 3.10-3.13 only and pins `torch==2.10.0`; the marker keeps the
   extra resolvable on 3.14 (it installs nothing there) and Radar's own matrix in
   `ci/release-policy.json` stays 3.10-3.14. Radar's `torch>=2.10` is unchanged: with the extra,
   the resolver takes Channel's exact pin.
2. **Contract 7.** Version 7 adds `PropagationRequest.discovery` (default `"auto"`),
   `PropagationEvaluation.complete` and `PropagationCapabilities.discovery_modes`; nothing the
   adapter reads changed meaning. The adapter is unchanged. The tests that pin the version, the
   capability field set and the requirement string follow it, and the import-isolation list adds
   Channel's new solver homes `witwin.channel.radiomap` and `witwin.channel.experimental`.
3. **Channel's validity rules are Radar's rows.** Radar does not re-validate or re-add a row
   Channel drops, and it does not refuse a site near a wall: a reflection Channel does not report
   is a reflection the scene does not have.
4. **Freshness is Channel's check.** Channel refuses a compiled scene whose source was edited in
   place on every `evaluate` and `reevaluate`. Radar keeps its motion-event revalidation, because
   that is the only route that RECOVERS (recompiles and rediscovers) rather than refuses.

## Measured behaviour on Channel 0.6.0

Local build `ae8ebb1f59727d72c4aa3502688506a4a3b9f8e1e4b212cc7d142fe3ed587e03` (sm_86), GPU 1.

| item | measured | pinned by |
|---|---|---|
| C-9 replay host cost | each `reevaluate` walks the source world once (`world_state`), visiting every structure; on a wall plus 63 tiles one frame's two replays walk twice, 64 structures each, with the same device-to-host observations as the bare wall. The walk costs about 9 us per structure on the host (11.6 us for 1, 514 us for 64, 4.58 ms for 512 structures; minimum of 4 x 100 calls, contended). `tests/test_join_host_budget.py` and `tests/test_pipeline_budget.py` pass unchanged | `tests/test_replay_host_cost.py` |
| C-10 in-place edit | a replay after an in-place vertex edit raises `ValueError: the compiled scene is stale` (on 0.5 it answered from the old wall). Radar's `source_mutation` tick still detects the edit and recompiles | `tests/test_row_invalidation.py` |
| C-11 moving-structure recompiles | `DynamicScene` snapshots, rigid trajectory or Core deformation, are new geometry objects per instant, so every compile builds a new RayD scene (4 compiles, 4 RayD scenes, generation 0) and no earlier compiled scene goes stale. Channel's in-place refit applies only when the same geometry object is recompiled; Radar's per-observation recompiles do not get it | `tests/test_row_invalidation.py` |
| C-12 reflection validity | a site 0.05 mm from the wall loses that wall's reflection (last segment 50.6 um below the 64.6 um minimum, 1e-5 of the 6.46 m diagonal, Channel ADR-059); 0.1 mm keeps it. A specular point 0.01 mm past the wall edge is no reflection (Channel ADR-068); on 0.5 the slack was 1e-4 of the triangle's 2.4 m altitude, about 0.24 mm (measured: 0.1 mm past accepted, 1 mm past rejected) | `tests/test_reflection_validity.py` |
| C-13 discovery source | the fixture discoveries report `complete=True` under `discovery="auto"`; Radar does not read the flag yet | measured, not pinned |
| C-14 dropped faces | the fixture walls drop no face (`degenerate 0, duplicate 0`). Not measured on SMPL: no SMPL model files on the measuring machine | measured, not pinned |
| C-15 trace backend | with `WITWIN_CHANNEL_TRACE_BACKEND=cuda` the scene reports `trace_backend == "cuda"` and the end-to-end, multipath-leg and reflection-validity tests pass | README |
| C-16 discovery vs replay | at the wall edge they agree: a reflection frozen 1 mm inside the edge is `row_valid=False` on replay 0.01 mm past it. Near a wall they disagree: a reflection frozen 1 cm away stays valid when replayed at 0.05 mm or 0.01 mm, where discovery has no such row. Replay does not apply the minimum segment length | `tests/test_reflection_validity.py` (strict xfail) |

## Values that moved

The full GPU suite was run at the base commit on Channel 0.5.0 and on this branch on Channel 0.6.0
(outcome lists in the release report). No bit-level, tolerance or oracle comparison in the suite
moved: the fixtures place no site within the minimum segment of a wall and no specular point near
a face edge, so the rule changes of Channel ADR-059, ADR-068 and ADR-051 select the same rows and
values there. The differences were:

| test | 0.5 -> 0.6 | class | cause |
|---|---|---|---|
| the consumer-contract, mirrored-row and capability-field tests | fail at the base commit on 0.6 | version / field set | contract 7; fixed by decision 2 |
| `tests/test_scene_leaf_ad.py::test_a_forward_tangent_on_the_wall_matches_the_reverse_gradient` and `::test_vertices_permittivity_and_endpoints_are_live_in_one_call` | pass -> `NotImplementedError` | Channel regression | RayD 0.10's finite-vertex check at scene creation reduces the vertices with `aminmax`, which has no forward-AD formula, so a mesh with dual vertices no longer compiles. Channel's matrix lists vertices for the reverse mode only. Strict xfails; the mirrored row is `REF` |

## Channel requests

1. **Replay applies the minimum segment length (C-16).** `reevaluate` should publish
   `row_valid=False` for a frozen reflection whose segment falls below `1e-5` of the scene
   diagonal, as discovery drops it. Until then a site that moves within 65 um of a wall keeps a
   reflection on replay that rediscovery removes, a discontinuity at the next motion event.
2. **Compile checks the primal vertices.** Running the finite-vertex check on the primal restores
   the vertex JVP that Channel 0.5 delivered.
3. **Refit for snapshots of one dynamic scene (C-11).** A per-observation recompile of a moving
   mesh rebuilds its acceleration structure; a refit keyed on the authored structure rather than
   the snapshot's geometry object would serve Radar's `adc` motion.
4. **A cheaper freshness check (C-9).** The per-replay walk is linear in structures, about 9 us
   each; a scene-level edit counter would make it constant.

## Consequences

- Radar 0.5.0 does not run on Channel 0.5. CI's `WITWIN_CHANNEL_FINGERPRINT` repository variable
  must name the published 0.6.0 wheel's `build_fingerprint`; the local build's fingerprint is not
  that value, and 0.6.0 was not on PyPI when this was written.
- A user who edits a scene in place between motion-event ticks now gets Channel's refusal instead
  of a stale answer; declaring a motion-event period still recovers on the next tick.
- A scatter site closer to a wall than 1e-5 of the scene diagonal has no reflection off that wall,
  and a specular point past a face edge is not a reflection; both follow Channel and are pinned by
  `tests/test_reflection_validity.py`.

## Amendment: the published Channel 0.6.0 (tag 699fd8c)

The measurements above were taken on a build of an earlier 0.6.0 tag that was never published. The
published 0.6.0 (tag 699fd8c, RayD 0.10.3) changes three of them, and Radar 0.5.0 requires it:

- **C-12.** A segment that ends at an endpoint is exempt from the minimum-segment rule, so a site on
  or within 0.05 mm of a wall keeps its reflection off that wall. The edge rule (ADR-068) is
  unchanged.
- **C-16.** Discovery and fixed-topology replay apply the same rule, so a reflection frozen 1 cm from
  the wall replays valid wherever discovery finds it.
- **Forward-mode vertex tangents.** The finite-vertex check reads the primal, so a mesh whose
  vertices carry a forward dual compiles; the two vertex-tangent tests are ordinary tests and the
  mirrored AD-matrix row is `SUP`.

`tests/test_reflection_validity.py`, `tests/oracle/test_survival.py` (`site_on_wall`,
`site_near_wall`) and `tests/test_scene_leaf_ad.py` assert the new behaviour. Requests C-12 and
C-16 above are closed.
