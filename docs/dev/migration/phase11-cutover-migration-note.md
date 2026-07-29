# Phase 11 Cutover - Migration Note

Status: authoritative record of the Phase-11 public API break.

Phase 11 is the one-shot production cutover. `Radar.simulate` stops being a
refusal and becomes the scene-driven entry point; everything the Dr.Jit tracer
route needed - a second logical `Scene`, a trace contract, a path cache, a
timeline, a second FMCW synthesis owner - is deleted rather than deprecated.

The break is approved by the plan (`docs/dev/plans/channel-radar-architecture-plan.md`,
Phase 11 work item 3) and by its acceptance criteria: no compatibility shim, no
shadow mode, no legacy fallback, no orphan binding may survive. A permanent
`NotImplementedError` stub is itself a shim and is therefore not an option.

Read `docs/pipeline_guide.md` first; it is the replacement documentation. This
note is the list of what moved, what died, and what to write instead.

---

## 1. `Radar.simulate` returns instead of raising

**Before.** `Radar.simulate(scene, ...)` raised `NotImplementedError` naming
`ChannelPropagationAdapter`, `TwoWayComposer`, `synthesize_fmcw` and
`mimo_from_paths` as the route a caller had to assemble by hand. Before that it
took a radar `Scene` plus `motion_sampling="per_chirp" | "per_frame"`.

**After.** It runs the whole pipeline and returns a typed record.

```python
result = radar.simulate(
    scene,                     # witwin.core.Scene or DynamicScene
    *,
    times,                     # sequence of frame instants in seconds, non-empty
    response,                  # REQUIRED scatter response
    sites=None,                # ScatterSitePolicy; None -> structure_anchor()
    components=None,           # frozenset override for THIS solve
    max_depth=None,            # int override for THIS solve
    slow_time_mode=None,       # None -> FROZEN_WEIGHT_WITH_CARRIER_RATE
    ad_mode="none",            # "none" | "vjp" | "jvp"
    world_motion="frozen_world",
    motion_event_period_frames=None,
    ids=None,                  # StableIdAllocator
    polarization=None,         # None -> (0, 0, 1)
    antenna_pattern=None,      # AntennaPatternSpec; None applies no pattern
) -> RadarSimulationResult
```

This is a behaviour change under an existing name. A caller that wrapped
`simulate` in `try/except NotImplementedError` now silently gets a result; a
caller that passed the old radar `Scene` gets a type error from Core.

`response` has no default on purpose: every candidate default is an unchosen
statement about target strength. `slow_time_mode` refuses
`REFRESHED_WEIGHT_NO_RATE` by name, because this driver composes once per frame
and a refreshed weight is one that already walked across slow time.

**`Radar.solve` does not exist and never has.** Phase 11 work item 1 names
`Radar.simulate/solve`; there is no `solve` anywhere in the tree's history, and
none was invented. The single scene-driven entry point is `simulate`.

## 2. `Radar.simulate_group` is deleted

**Before.** A classmethod that raised the same `NotImplementedError`.

**After.** The attribute is gone; `hasattr(Radar, "simulate_group")` is `False`,
and so is `hasattr(Radar, "_SIMULATE_REPLACEMENT")`.

Batching several radars over one world is not implemented. Call `simulate` per
radar; the compiled scene is the expensive part and is per-world, so a future
group entry would share it rather than the frame loop.

## 3. `TraceResult` and `empty_trace` are removed

`witwin.radar.TraceResult` and `witwin.radar.trace_result.empty_trace` are gone
with the tracer that produced them. There is no Radar-side trace contract: the
discrete path topology is Channel's, is discovered by
`ChannelPropagationAdapter.freeze`, and is carried by the typed
`PathTopology` / `RadarLegBatch` contracts.

Replace `trace = tracer.trace(time=t); radar.mimo_from_trace(trace)` with one
`radar.simulate(scene, times=(t,), ...)` call.

## 4. `Timeline` and `TransformMotion` are removed

`witwin/radar/timeline.py` is deleted, including `Timeline.from_motion`,
`Timeline.generate_rd` and the `TransformMotion` rigid-motion description.

Motion is Core's. Use `witwin.core.dynamics.DynamicScene` with
`LinearTrajectory` (or a `Deformation`) and hand the dynamic scene to
`simulate` with the frame instants you want:

```python
from witwin.core.dynamics import DynamicScene, LinearTrajectory

dynamic = DynamicScene(scene, structure_trajectories={1: LinearTrajectory(
    origin=torch.tensor((0.0, 0.0, 0.0)), velocity=torch.tensor((4.0, 0.0, 0.0)))})
result = radar.simulate(dynamic, times=tuple(k / 10.0 for k in range(8)), ...)
```

`Timeline.generate_rd` has no single replacement: build the Range-Doppler maps
from `result.cube` with `witwin.radar.processing.range_profile` and
`range_doppler_map` (see `examples/rgbd_range_doppler.py`).

## 5. `MimoPathCache` is removed, and so is `docs/fast_mimo_api.md`

`witwin.radar.MimoPathCache`, `Radar.path_cache_from_trace`,
`Radar.mimo_from_paths` and `Radar.mimo_from_trace` are gone.
`docs/fast_mimo_api.md` described exactly and only those names and is deleted
outright rather than stubbed - a redirect file is a compatibility shim.
`docs/pipeline_guide.md` replaces it.

What replaces the cache is the epoch loop's FROZEN TOPOLOGY REPLAY, which is the
same idea without a cache object: the discrete path set is discovered once per
epoch and every later frame reevaluates those rows at the current endpoint
positions. `world_motion="fixed_winner_replay"` is the declaration that the
discrete winner set stays fixed while the geometry moves; `"frozen_world"`, the
default, retires the frozen handles whenever a structure moves.

The perf table in the deleted document (`mimo_from_trace` / `mimo_from_paths` at
1024 and 6248 targets on an RTX 5080) measured deleted code and is not carried
forward. `PERFORMANCE.md` carries the pipeline's own budgets.

## 6. The radar `Scene`, `SceneModule` and `CompiledMesh` are removed

`witwin/radar/scene.py` is deleted. `witwin.radar.Scene`,
`witwin.radar.SceneModule` and `CompiledMesh` are gone, together with the radar
scene compiler and the geometry helpers that served only it.

There is exactly one logical world in production and it is `witwin.core.Scene`
(plus `witwin.core.dynamics.DynamicScene`), compiled by Channel through
`ChannelPropagationAdapter.compile_scene`. `witwin.radar` re-exports Core's
geometry constructors (`Mesh`, `Box`, `Sphere`, ...) so existing geometry code
keeps working; the container and its mutating methods do not.

Mesh vertices authored in world coordinates need `recenter=False`: Core's `Mesh`
defaults to subtracting the bounding-box centre, silently.

## 7. `SamplingMode` and `MotionSampling` are removed

Both were `witwin/radar/types.py` enums exported from the package root and both
described the deleted `motion_sampling=` argument. `DetectorType` survives and
moves to `witwin/radar/processing/contracts.py` as a pure file move.

Per-chirp versus per-frame sampling is not a knob any more. `simulate` resolves
the world once per FRAME; the within-frame slow-time walk is a named deferral
(section 13).

## 8. `Solver`, `DirichletSolver` and `radar.solver` are removed

`witwin/radar/solvers/` - the `Solver` base, `DirichletSolver`, the native
`dirichlet.cu` translation unit and its nine ABI symbols - is deleted, and the
`Radar.solver` attribute with it. `Radar.chirp`, `Radar.frame` and `Radar.mimo`
went with it; they were thin wrappers over the solver.

**D6, recorded explicitly rather than assumed.** Phase 11 work item 6 says to
MOVE the Dirichlet and receiver/noise files "to their final owners". This phase
resolves both moves as DELETE:

* **The Dirichlet route (item 6, M3).** Moving it would preserve a SECOND FMCW
  synthesis owner beside `witwin.radar.synthesis.synthesize_fmcw`.
  Acceptance criterion 4 requires native duplicate production code to be
  deleted, `witwin/radar/capabilities.py` already labelled the route
  `legacy_solver_route`, and its `backward` symbol was already caller-free in
  `ci/native-binding-manifest.json` (`end_to_end_caller: null`,
  `caller_status: test_only`). A move would have carried an orphan binding
  across the cutover, which criterion 6 forbids. The move is void; the files are
  deleted.
* **The receive-chain runtimes (item 6, M2).** `ReceiverChainRuntime`,
  `NoiseModelRuntime` and `PolarizationRuntime` in `witwin/radar/radar.py` were
  to move to `witwin/radar/frontend/`. `frontend.py` already states that
  those stages are merged into `FrontendChain`, so moving them would relocate a
  shadow rather than remove it, and `Radar._validate_runtime_config` refusing a
  configuration that names both is exactly the compatibility branch criterion 6
  names. The move is void; the runtimes are deleted and `FrontendChain` is the
  single owner.

The only genuinely pure move item 6 leaves is `DetectorType`, and it lands as
its own commit with no behaviour change.

## 9. `quantize_complex_signal` is removed

`witwin.radar.quantize_complex_signal` was the free function behind the legacy
`noise_model["quantization"]` block. Quantization is the ADC stage of the
frontend chain: use `FrontendSpec(adc=AdcSpec(bits=..., full_scale=...))`.

## 10. `noise_model` / `receiver_chain` are replaced by `frontend`

The legacy pair let the CALLER decide the composite order of noise and gain,
which is a difference of `g_lna^2` in output noise power and was never
expressible as a single answer. `FrontendSpec` fixes the order in the runtime:

```text
synthesis output [sqrt(W)] -> port -> phase -> thermal -> LNA -> AGC -> ADC
```

Thermal noise is INPUT REFERRED, so it is added before the gain. Units are
physical - a noise figure, a system temperature, an explicit bandwidth - rather
than a raw standard deviation, and seeds are per stage so toggling one stage
leaves every other bit-identical.

**Known gap, and now a loud one.** The flat mapping accepted by
`RadarConfig.from_dict` cannot express the receive chain: it has no `"frontend"`
key. It used to DROP one silently; it now refuses it by name (section 14).
Attach the block after validation:

```python
config = dataclasses.replace(
    RadarConfig.from_dict(CONFIG),
    frontend=FrontendSpec(noise=NoiseSpec(noise_figure_db=10.0, bandwidth_hz=4.4e6),
                          seed=SeedSpec(20260727)),
)
```

## 11. New: the four typed diagnostics

`Radar.last_trace` is gone. In its place are four read-only properties, all
typed, all describing the LAST completed frame, all `None` before the first
`simulate` and after a failed one:

| property | type |
| --- | --- |
| `radar.last_snapshot` | `witwin.core.SceneSnapshot` |
| `radar.last_compiled_scene` | `witwin.channel.scene.CompiledScene` |
| `radar.last_propagation` | `witwin.radar.RadarPropagationLegs` |
| `radar.last_radar_paths` | `witwin.radar.paths.RadarPathBatch` |

`radar.last_result` returns the `RadarSimulationResult` itself. The four are the
same objects the result carries, so `radar.last_snapshot is result.last_snapshot`.

`RadarPropagationLegs` is a typed two-leg container with `.inbound` / `.outbound`
rather than a tuple or a dict: an untyped pair is exactly the shape that lets a
swapped leg produce a mirrored answer with nothing raised.

These are a real tensor lifetime. Holding the result holds that frame's device
tensors and, under `ad_mode="vjp"`, that frame's autograd graph. None of them
holds a tape.

## 12. What survives, deliberately

* **`witwin.radar.sigproc`.** Its whole public surface survives as deprecation
  adapters over `witwin.radar.processing` (the wrappers live in
  `witwin/radar/processing/adapters.py`). They reproduce the legacy behaviour
  INCLUDING the parts that are wrong - the symmetric unnormalised window, the
  unreconciled Doppler sign whose positive velocities are receding targets on an
  FMCW cube - because a migration adapter that quietly improves the answer
  cannot be trusted. They warn. New code uses the facade; there are stored
  goldens (`tests/goldens/legacy_sigproc.pt`) pinning the adapters bitwise.
* **`witwin/radar/processing/adapters.py`.** The adapters Phase 8's processing
  migration created. They are the one adapter layer this phase does not remove.
* **The Core geometry re-exports** from `witwin.radar` (`Mesh`, `Box`, `Sphere`,
  `Material` = `PhysicalMaterial`, `Structure`, ...). One import path for the
  world model, but the familiar spelling keeps working.

## 13. Named deferrals

Not bugs, and not to be worked around silently.

* **Intra-frame Doppler is zero.** `simulate` composes the round trip once per
  frame, so the chirps of one frame are identical and every return lands in the
  zero-Doppler bin. Frame-to-frame motion is fully modelled. Unpacking a
  `delay_rate` needs a forward-AD velocity dual authored from Core kinematics
  (`witwin.radar.propagation.kinematics.two_way_duals`), and this entry does not
  open one. Consequences: slow-time clutter removal subtracts the whole signal,
  and a slow-time MUSIC covariance is rank deficient.
* **Only `los` and `reflection` can be frozen**, so those are the only
  components `simulate` accepts even though the consumer's vocabulary also has
  `diffraction` and `transmission`.
* **Scatter sites cannot be derived from a mesh** (R-ADR-020). A sampling rule
  is a geometry algorithm and would be new Torch geometry on a production hot
  path. Declare sites explicitly or give the structure a rigid motion so Core
  publishes a world anchor.
* **A non-FMCW `Radar` is unconstructible.** `RadarSystemConfig` supports
  `ofdm` and `pulsed` and `RadarSimulationResult` derives its axes generically,
  but `Radar.__init__` builds the FMCW axes record unconditionally.
* **No `ProcessingAxes` route off a simulation result.** `ProcessingAxes` is
  built from a rank-3 `SynthesisResult` while the result publishes the assembled
  rank-5 cube, so a consumer re-synthesizes the last frame's composed rows to
  obtain one. See `docs/pipeline_guide.md` section 6.

## 14. The constructor loses `pad_factor`, and the flat config refuses what it cannot express

Two breaks in the configuration surface that are easy to miss because neither
one is a deleted name.

**`Radar.__init__(config, pad_factor, device)` loses its second parameter.**

```python
radar = Radar(CONFIG, 16, "cuda")     # before
radar = Radar(CONFIG, "cuda")         # after
```

`pad_factor` was the FFT zero-pad factor of the deleted signal-processing
route; `witwin.radar.processing` takes the padding it needs per call. The
parameter was positional, so an unported call does not raise a missing-argument
error - it binds `16` to `device` and fails somewhere else entirely. Grep for
`Radar(` with three positional arguments.

**An unknown key in the flat mapping is refused.** `RadarConfig.from_dict`
(and therefore `Radar(mapping, ...)` and `RadarConfig.from_json`) used to drop
any key it did not recognize. It now raises and names the keys:

```text
Radar config has unsupported keys: waveform. The flat mapping accepts only ...
```

The two that cost real time were `"waveform"` and `"frontend"`. A caller who
wrote `{"waveform": "ofdm"}` got an FMCW radar and a complete simulation in the
wrong waveform with nothing raised; a caller who wrote `{"frontend": {...}}`
got a radar with no receive chain. Neither block is authorable in the flat form
today - see section 10 for how to attach a frontend after validation, and
section 13 for the FMCW-only deferral - so the refusal is the honest answer.
Every configuration in this repository already passes; only a mapping carrying
a key the validator never read is affected.

## 15. Quick reference

| deleted | write instead |
| --- | --- |
| `Radar.mimo(interpolator, t0=...)` | `radar.simulate(scene, times=(t0,), response=..., sites=...)` |
| `Radar.mimo_from_trace(trace)` | same |
| `Radar.path_cache_from_trace` + `Radar.mimo_from_paths` | frozen topology replay; `world_motion="fixed_winner_replay"` |
| `Radar.frame` / `Radar.chirp` | `result.cube[frame]` / `result.cube[frame, tx, rx]` |
| `Radar.simulate_group` | one `simulate` per radar |
| `TraceResult`, `empty_trace` | `radar.last_propagation`, `radar.last_radar_paths` |
| `witwin.radar.Scene`, `SceneModule`, `CompiledMesh` | `witwin.core.Scene`, `witwin.core.dynamics.DynamicScene` |
| `Timeline`, `TransformMotion` | `DynamicScene` + `LinearTrajectory` + `times=` |
| `SamplingMode`, `MotionSampling` | nothing; sampling is per frame |
| `Solver`, `DirichletSolver`, `radar.solver` | `witwin.radar.synthesis.synthesize_fmcw` via `Radar.synthesize` |
| `quantize_complex_signal` | `FrontendSpec(adc=AdcSpec(...))` |
| `noise_model`, `receiver_chain` | `frontend=FrontendSpec(...)` |
| `radar.last_trace` | `radar.last_snapshot` / `last_compiled_scene` / `last_propagation` / `last_radar_paths` |
| `Radar(config, pad_factor, device)` | `Radar(config, device)`; pad per call in `witwin.radar.processing` |
| an unknown flat config key (dropped) | refused by name; attach `frontend` after validation |
| `witwin.radar.sigproc.*` | `witwin.radar.processing.*` (the adapters still work, and warn) |
| `docs/fast_mimo_api.md` | `docs/pipeline_guide.md` |
