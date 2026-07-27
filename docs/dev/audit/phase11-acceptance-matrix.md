# Phase 11 acceptance matrix

One row per acceptance criterion of the plan's Phase-11 section
(`docs/dev/plans/channel-radar-architecture-plan.md:2306-2313`), each with the
command that was actually run, an excerpt of what it printed, and a verdict.

Nothing here is asserted from a design document. Criteria 2, 3 and 5 were
already passing before this phase and are RE-PROVEN rather than assumed;
criteria 1, 4, 6, 7 and 8 are closed by the cutover.

Recorded 2026-07-27 against:

| tree | branch | commit |
| --- | --- | --- |
| Radar `E:/Code/witwin-platform/.worktrees/radar-phase11` | `claude/stage2-phase11` | `d9c27eb` |
| Channel `E:/Code/witwin-platform/.worktrees/channel-phase11` | `claude/phase11-cutover` | `209cc33` |

Environment: `witwin2` conda env, one RTX 5080, Windows 11. Channel loaded
through the ADR-006 developer override against the Phase-11 rebuild
`E:/p11chbuild/b/_channel.cp311-win_amd64.pyd`
(`f6d81aaa4d1d49b2712f06765a53413c84e262f5130aa0fc2b4a9aadde45ce72`). Radar
loaded from the packaged prebuilt rebuilt in `cf6d836`
(`4d7b7fad26b42c10...`). No remote CI was run; every result below is local.

---

## Summary

| # | criterion | verdict |
| ---: | --- | --- |
| 1 | one logical Core `Scene` and one Channel `CompiledScene` path in production | **PASS** |
| 2 | no RayD scene/BVH owner in Radar, no raw `_channel` access | **PASS** |
| 3 | no solver-to-solver dependency between Channel and Radar | **PASS** |
| 4 | old `Tracer` / `TraceResult` / scene compiler / native duplicate production code deleted | **PASS** |
| 5 | no Dr.Jit in Radar production modules or runtime dependencies | **PASS** |
| 6 | no compatibility shim, shadow mode, legacy fallback or orphan binding | **PASS** |
| 7 | public APIs, docs, examples and package metadata consistent | **PASS** |
| 8 | repository hygiene, import graph, API/ABI manifests and dead-code audit green | **PASS** (Radar `nightly`/`release` deferred, see below) |

---

## 1. One logical Core `Scene`, one Channel `CompiledScene` path

**Commands and output.**

```
$ ls witwin/radar/scene.py witwin/radar/timeline.py
absent  witwin/radar/scene.py
absent  witwin/radar/timeline.py

$ grep -rn "^class Scene" witwin/radar --include=*.py
(no match; the only near-name is propagation/epochs.py:308 class SceneEpochLoop)

$ python -c "import witwin.radar; witwin.radar.Scene"
AttributeError: witwin.radar.Scene has been removed. The one logical world is
witwin.core.Scene; build it there and hand it to Radar.simulate, ...

$ grep -rn "from witwin.channel.scene import" witwin/radar --include=*.py
witwin/radar/propagation/channel_consumer.py:269:    from witwin.channel.scene import compile as channel_compile

$ python -m pytest tests/test_public_api_snapshot.py -q
7 passed
```

`test_the_core_re_exports_are_the_core_objects` additionally pins that the
thirteen world types the radar root re-exports (`Material`, `Structure`,
`GeometryBase`, `Mesh` and the nine primitives) ARE the `witwin.core` objects,
by identity, so a radar-side copy of a world type fails this criterion in CI
rather than in review.

There is exactly one call to the Channel compile facade in the whole package,
and `SceneEpochLoop` takes the compiler as an argument rather than importing it,
so no second compile path can exist without adding a second import.

**Verdict: PASS.**

## 2. No RayD scene/BVH owner in Radar, no raw `_channel` access

**Commands and output.**

```
$ python ci/check_production_dependencies.py
check_production_dependencies: 71 production modules name none of drjit,
mitsuba, rayd, sionna or @dr.wrap, @drjit.wrap; 2 recorded prose occurrence(s);
11 declared distribution(s) clean

$ python ci/check_raw_native_access.py
check_raw_native_access: dispatcher owner witwin/radar/cuda/build.py; 8 recorded
loader consumers; witwin/radar/cuda/identity.py holds no dispatcher access

$ python ci/check_extension_boundary.py      # inside `cuda` tier
[gate passed]

$ python -m pytest tests/test_phase4_import_boundary.py -q      # inside the suite
[part of 782 passed]
```

`ci/check_production_dependencies.py` scans string literals too, so a lazily
built `importlib.import_module("rayd")` is caught; the two prose occurrences are
frozen by equality with their reason. `ci/check_extension_boundary.py` reads the
shipped binary's PE import table and asserts that `_radar_native` imports no
`rayd`/`drjit`/`mitsuba`/`optix` and shares no symbol name with `_channel`.

`test_phase4_import_boundary.py` names exactly one Radar module that may import
`witwin.channel` at all - `propagation/channel_consumer.py` - and pins the
allowed import list by EQUALITY. Radar never touches `witwin.channel._channel`.

**Verdict: PASS.** Re-proven, not assumed.

## 3. No solver-to-solver dependency between Channel and Radar

**Commands and output.**

```
$ (channel) python ci/check_import_graph.py
import graph contract passed (existing_boundary=1, mc_enumerated_dependency=1)

$ (channel) python ci/check_contract_coverage.py
contract coverage passed (60 public exports, 234 native bindings)

$ (radar) grep -rn "witwin.channel" witwin/radar --include=*.py | grep import
witwin/radar/propagation/channel_consumer.py:52:from witwin.channel.propagation import consumer
witwin/radar/propagation/channel_consumer.py:269:    from witwin.channel.scene import compile as channel_compile
```

Radar reaches Channel through the solver-neutral propagation consumer and the
scene compile facade only. It imports no Channel solver (`path`,
`deterministic`, `montecarlo.basic`, `montecarlo.bdpt`) and no internal
propagation module. Channel imports nothing from Radar; its own import-graph
gate keeps the one allowlisted intra-Channel boundary at 1.

**Verdict: PASS.** Re-proven, not assumed.

## 4. Old `Tracer` / `TraceResult` / scene compiler / native duplicates deleted

**Commands and output.**

```
$ ls witwin/radar/{trace.py,trace_result.py,path_cache.py,scene.py,timeline.py,types.py}
$ ls witwin/radar/solvers witwin/radar/sensors/legacy_paths.py
$ ls witwin/radar/synthesis/dirichlet_spectrum.py witwin/radar/cuda/kernels/dirichlet.cu
$ ls witwin/radar/utils/geometry.py tools/benchmark_dirichlet_cuda.py docs/fast_mimo_api.md
absent  (all thirteen)

$ python ci/check_native_bindings.py
native binding manifest OK: schema 2, 25 operators, 9 AD groups, 6 error owners
  symbol-set tie checked against _radar_native.pyd (25 symbols)

$ python ci/check_torch_physics_allowlist.py
check_torch_physics_allowlist: 71 modules scanned under witwin with 0 exclusions;
16 recorded expressions, 27 occurrences; digest c3d85ab84dcfd310
```

| quantity | before Phase 11 | after |
| --- | ---: | ---: |
| registered native operators | 34 | 25 |
| `RADAR_ABI_VERSION` | 1 | 2 |
| CUDA build input sources | 9 | 8 |
| autograd `Function` owners | 10 | 8 |
| AD boundary fixtures | 9 | 7 |
| `ctx.saved_tensors` reads | 20 | 16 |
| torch-physics allowlist entries / occurrences | 23 / 34 | 16 / 27 |
| production modules | 80 | 71 |

The nine deleted operators are the `dirichlet_spectrum` family - a SECOND FMCW
synthesis owner beside `synthesize_fmcw_beat`, which is what made it native
duplicate production code. `_radar_native.pyd` was rebuilt in the same commit
that deleted them (`cf6d836`) so the binary, the manifest and the loader's
required-symbol set moved together; the loader refuses a mismatch.

**Verdict: PASS.**

## 5. No Dr.Jit in Radar production modules or runtime dependencies

**Commands and output.**

```
$ python ci/check_production_dependencies.py
check_production_dependencies: 71 production modules name none of drjit,
mitsuba, rayd, sionna or @dr.wrap, @drjit.wrap; 2 recorded prose occurrence(s);
11 declared distribution(s) clean

$ python -m pytest tests/test_phase10_static_gates.py -q     # inside the suite
[part of 782 passed - 32 tests, each gate proven to FIRE on a planted violation]
```

The gate covers three shapes and each is proven to fail on a mirror of the tree
with one violation planted: a literal `import drjit`, a `@dr.wrap` decorator, and
an `importlib.import_module("drjit")` built from a string. The eleven declared
distributions include the optional `[channel]` extra, which pins
`witwin-channel` and no ray-tracing runtime.

**Verdict: PASS.** Re-proven, not assumed.

## 6. No compatibility shim, shadow mode, legacy fallback or orphan binding

**Commands and output.**

```
$ python -m pytest tests/test_phase4_binding_manifest.py -q  # inside the suite
[part of 782 passed]     CALLER_FREE_SYMBOLS = set()   -> the cap is 0

$ (channel) python ci/check_contract_coverage.py
contract coverage passed (60 public exports, 234 native bindings)
                         DORMANT_SYMBOL_FACADES = {}  -> the allowlist is 0

$ python ci/check_orphan_modules.py
ci/check_orphan_modules.py: OK - 71 production modules, all reachable from 4
declared entry points.

$ python -m pytest tests/test_phase5_removed_entry_points.py -q  # inside the suite
[part of 782 passed]
```

- **Orphan bindings: zero on both sides.** Radar's caller-free symbol set is
  empty and `tests/test_phase4_binding_manifest.py` asserts it by equality, so a
  new caller-free symbol is a decision somebody has to write down. Channel's
  dormant-symbol allowlist is empty for the same reason - and both gates keep
  their "named decision required" branch armed rather than deleting it.
- **No shim.** Every removed public name raises with a message naming its
  replacement instead of resolving to a compatibility object; the twelve names
  are listed in `witwin/radar/__init__.py::_REMOVED` and
  `tests/test_public_api_snapshot.py::test_removed_names_raise_with_a_replacement_rather_than_resolving`
  pins five of them plus their absence from `__all__`.
- **No stub.** `Radar.simulate_group` is DELETED rather than left raising
  `NotImplementedError`: a permanent refusing stub is itself a shim under this
  criterion. Recorded as an approved public break (design D3).
- **No shadow mode.** The `noise_model` / `receiver_chain` configuration blocks
  and their runtimes are deleted rather than moved under `frontend/`, because
  moving them would have relocated the shadow instead of removing it (design D6,
  M2). Same for the Dirichlet route (D6, M3).
- **Deliberate survivors, stated so they are not mistaken for shims:**
  `witwin.radar.sigproc` and `witwin/radar/processing/adapters.py`. These are the
  Phase-8 post-processing migration surface, each name carrying a
  `DeprecationWarning` that points at its replacement; retaining them was decided
  before this phase and is recorded in `PHASE11_ENV.md`.

**Verdict: PASS.**

## 7. Public APIs, docs, examples and package metadata consistent

**Commands and output.**

```
$ python -m pytest tests/test_public_api_snapshot.py tests/test_phase11_repository_gates.py -q
31 passed          (with tests/test_phase10_diagnostics.py)

$ python -m examples.single_point
  Cube: (3, 3, 4, 128, 256) ('frame', 'tx', 'rx', 'chirp', 'sample')  OK
  Epochs: (0, 0, 0)  compiles=1 discoveries=1  OK
  Diagnostics: SceneSnapshot, CompiledScene, RadarPropagationLegs, RadarPathBatch
  |C_rt| = 1.726919e-06 vs radar equation 1.726922e-06  OK
  Range peak: 3.0051 m (target at 3.0000 m)  OK
  Multipath peak near 5.0000 m  OK
PASSED

$ grep -rn "mimo_from_trace|MimoPathCache|radar.mimo|witwin.radar.Scene|add_smpl|
            TransformMotion|SamplingMode|Dirichlet|witwin.radar.solvers|
            last_trace|simulate_group|fast_mimo_api" README.md FEATURE_LIST.md
            PERFORMANCE.md docs/ --include=*.md
            | grep -v docs/dev/migration/phase11-cutover-migration-note.md
```

The grep's surviving hits are all deliberate and were read one by one: the
`_REMOVED` list in `FEATURE_LIST.md`, the "migrating from" pointer in
`README.md`, the dated Phase-10 audit under `docs/dev/audit/` (which carries a
supersession section), and the plan itself. No document claims a deleted API
still works.

What changed here:

- `ci/public-api-snapshot.json` is NEW and freezes the `__all__` of
  `witwin.radar`, `witwin.radar.processing` and `witwin.radar.sigproc` plus
  `Radar`'s public member set, with each name's target and signature. Without
  it, this criterion had no falsifier in the radar repository at all.
- The same test refuses an export that nothing in `witwin/radar/`, `tests/` or
  `examples/` names. `SamplingMode`, `MotionSampling` and `Timeline` sat in
  `__all__` for four phases with zero consumers and were found by hand.
- `README.md:118-133` told the reader that "a scene-driven entry point does not
  exist yet" - the exact paragraph work item 1 invalidates - and `:176-180`
  named two example modules (`examples.mesh_scene`, `examples.humanbody`) that
  do not exist and never did in this tree. Both fixed; the README quick start was
  EXECUTED before it was written down.
- `PERFORMANCE.md` was four fifths tables produced by
  `tools/benchmark_dirichlet_cuda.py`, which is deleted. Those numbers are
  removed rather than carried forward and replaced by measurements taken at this
  commit.
- `FEATURE_LIST.md` operator counts, allowlist counts and AD-owner counts now
  match the gates that print them (25 operators, 16 allowlist entries, 8 tape
  owners, 7 boundaries, 16 saved-tensor reads).

**Two defects were found by building the snapshot**, both fixed in `171d024`:

1. `witwin.radar.capabilities` meant three different things. `capabilities` is
   both a submodule name and the function it exports, and importing the
   submodule binds it onto the package - so the lazy `__getattr__` handed out
   the FUNCTION on first access and the MODULE on every access after, while
   `from witwin.radar import capabilities` got the module from the start,
   because the fromlist `hasattr` probe performed the shadowing import before
   the name was read. Measured, not reasoned. The resolution is now memoised
   into the package globals after the import, and
   `test_a_lazy_export_means_the_same_object_on_every_access` pins it including
   a fresh-process check of the `from ... import` path.
2. `witwin/radar/config.py` lost its `PolarizationSpec` import in `06e689f`
   while keeping the annotation, so `quick.ruff` was red at the deletion tip
   (F821) and `typing.get_type_hints(SensorConfig)` would raise. Restored.

**Verdict: PASS.**

## 8. Repository hygiene, import graph, API/ABI manifests, dead-code audit green

### Radar

| gate | command | result |
| --- | --- | --- |
| ruff | `python -m ruff check witwin/radar tests tools ci scripts` | All checks passed |
| native bindings | `python ci/check_native_bindings.py` | 25 operators, 9 AD groups, 6 error owners; symbol set tied to the pyd |
| production dependencies | `python ci/check_production_dependencies.py` | 71 modules clean, 11 distributions clean |
| test-oracle isolation | `python ci/check_test_oracle_isolation.py` | 71 modules import no test module; wheel packages `['witwin']` |
| raw native access | `python ci/check_raw_native_access.py` | one dispatcher owner, 8 recorded consumers |
| torch-physics allowlist | `python ci/check_torch_physics_allowlist.py` | 16 expressions, 27 occurrences, digest `c3d85ab84dcfd310` |
| workflow policy | `python ci/check_workflow_policy.py` | prebuild policy version 6 |
| **dead code (NEW)** | `python ci/check_orphan_modules.py` | 71 modules, all reachable from 4 entry points |
| **public API (NEW)** | `pytest tests/test_public_api_snapshot.py` | 7 passed |
| extension boundary | `python ci/check_extension_boundary.py` | passed inside the `cuda` tier |
| tier `quick` | `python ci/run_ci_tier.py quick` | **exit 0**; 782 passed / 792 skipped; coverage 65% (floor 50) |
| tier `cuda` | `python ci/run_ci_tier.py cuda` | **exit 0**; 1573 passed / 1 skipped; coverage 83% (floor 75) |
| full suite | `pytest tests -q` | 782 passed / 792 skipped |
| full suite, GPU | `pytest tests --gpu -q` | 1573 passed / 1 skipped / 0 failed |

### Channel

| gate | command | result |
| --- | --- | --- |
| tier `quick` | `python ci/run_ci_tier.py quick` | **exit 0**, 13 stages |
| tier `cuda` | `python ci/run_ci_tier.py cuda` | **exit 0**; 2143 passed / 3 skipped |
| full suite | `pytest tests -q` | 2628 passed / 9 skipped / 1 xfailed |
| import graph | `python ci/check_import_graph.py` | passed (`existing_boundary=1`, `mc_enumerated_dependency=1`) |
| contract coverage | `python ci/check_contract_coverage.py` | 60 public exports, 234 native bindings, dormant allowlist empty |
| repository hygiene | `python ci/check_repository_hygiene.py` | passed for 895 tracked files |
| maintenance budgets | `python ci/check_maintenance_budgets.py` | passed |
| product identity | `python ci/check_product_identity.py` | passed |
| secret scan | `python ci/check_secrets.py` | passed |
| `CLAUDE.md` == `AGENTS.md` | sha256 of both files | identical, `bb9e38d8c8f89fc3...`, 30737 bytes each |

### Deltas from the recorded baselines, by name

Radar baseline at `222c0c7`: default 843 passed / 807 skipped, `--gpu` 1650
passed. Now 782 / 792 and 1573 / 1.

| delta | cause |
| ---: | --- |
| default `-61` passed, `-15` skipped | 12 test files deleted with the routes they covered (`legacy-deletion`, commits `8fda971` and `cf6d836`), partly offset by the migrated coverage and the 15 new tests below |
| `--gpu` `-77` passed | same, plus the GPU half of the deleted Dirichlet and legacy-path suites |
| `+15` in both | this stage: `tests/test_public_api_snapshot.py` (7) and `tests/test_phase11_repository_gates.py` (8) |

Channel baseline at `f9a9444`: targeted 1572 passed / 9 skipped / 1 xfailed. The
comparable full-suite figure now is 2628 / 9 / 1; `channel-dormant` measured the
same targeted set at 1459 / 8 / 1 after deleting 12 test files (182 cases, 3
added).

### Named deferrals

| item | owner | reason |
| --- | --- | --- |
| Radar `nightly` tier | release | builds the Core and Radar wheels, runs the wheel smoke, and needs a Channel wheel artifact this repository cannot build. Also blocked here by host disk: `C:` sat at 100 percent during this run and two Channel tests failed on `No space left on device` until 3 GB was freed. |
| Radar `release` tier | release | `release.arch-verification` demands the complete release SASS set plus `compute_120` PTX. The prebuilt in this tree is a deliberate single-architecture developer build, so the gate SHOULD fail here; only a release-matrix binary can pass it. |
| Channel `nightly` / `release` tiers | release | same shape: manylinux_2_28 cells and the SM87 SASS matrix are the release workflow's, and the standing owner directive forbids remote CI runs from this session. |
| any remote CI (GitHub Actions) | owner | standing directive for this phase: local and static verification only. |
| a Radar contract-coverage manifest | radar | Channel's `ci/check_contract_coverage.py` ties public exports to native bindings; Radar has no equivalent. The interim equivalent is `ci/native-binding-manifest.json`'s non-null `end_to_end_caller` requirement plus the caller-free cap of 0, which `ci/check_native_bindings.py` and `tests/test_phase4_binding_manifest.py` enforce together. Adding the Channel-shaped gate is a separate change. |
| mesh-derived scatter sites | radar | design D2 / R-ADR-020. Only Core-owned structure anchors and explicit site lists are supported; mesh sampling is refused by name rather than approximated. |
| intra-frame (slow-time) Doppler | radar | identically zero unless the caller drives `propagation.kinematics.two_way_duals` around `simulate`. Named in the module docstring, the feature list and the migration note. |

**Verdict: PASS**, with the release-tier cells deferred to the release workflow
by the standing no-remote-CI directive.

---

## Open escalations from the design, and how each was finally resolved

`wf13/01-design.md` section 5 listed five items that no stage was allowed to
resolve silently. What actually happened:

| id | question | resolution |
| --- | --- | --- |
| **D5** | `sensor_weight` orphaned by deleting `sensors/legacy_paths.py`: (a) rewire, (b) delete the family, (c) defer | **(a) rewire, taken on the design DEFAULT, not on an owner ruling** - no D5 decision was found anywhere in the session scratch. `witwin/radar/sensors/round_trip.py` gives the family a production end-to-end caller on the scene-driven route, so cluster L could be deleted without the family ever passing through a caller-free state. `git revert 37aef3b` is a clean single revert if the owner prefers (b). |
| **D6** | item 6 says MOVE the Dirichlet (M3) and receiver/noise (M2) files; criteria 4 and 6 argue DELETE | **Both resolved as DELETE**, with the reasoning written into the migration note section 8 rather than assumed. M2: `frontend/chain.py` already stated those stages were merged into `FrontendChain`, so a move relocates a shadow. M3: the route was a second FMCW synthesis owner and its `backward` symbol was already caller-free. The only genuine pure move, `DetectorType` -> `processing/contracts.py` (M1), landed as its own commit `3dc843c`. |
| **D3** | `Radar.simulate_group` - delete or leave refusing? | **Deleted.** A permanent `NotImplementedError` stub is a shim under criterion 6. Recorded as an approved public break in the migration note section 2. |
| **D2** | scatter-site policy scope | **Deferred as designed.** `ScatterSitePolicy` supports `explicit(positions)` and `structure_anchor()`; mesh sampling is refused by name (R-ADR-020). Inventing a sampling algorithm would have been new Torch geometry in a production hot path. |
| naming | `STABLE_TORCH_LIBRARY(witwin_radar_dirichlet_cuda, ...)` is named after the deleted route - rename it? | **No rename, because there is nothing to rename.** That symbol does not exist: `extension.cpp:3` opens `STABLE_TORCH_LIBRARY(_radar_native, m)` and all eight `..._IMPL` blocks name `_radar_native`. The claim came from pre-Phase-10 prose left in `docs/dev/audit/phase10-extension-boundary.md`, which stage `legacy-deletion` corrected in place with a supersession section. |

## Findings this stage raises rather than fixes

1. **The per-frame wall-clock budget sits at ~87 percent of a threshold derived
   for a smaller quantity.** 5.044 ms is `3.88 x 1.30`, and 3.88 ms described two
   leg replays plus one composition without synthesis; the pin now measures the
   marginal cost of the whole production frame (4.16 to 4.54 ms depending on
   instrumentation). It passes everywhere it was run and no number was raised.
   Re-deriving it (`MEASURED_SIMULATION_FRAME_MS = 4.25`, same 1.30 factor,
   5.53 ms) is proposed in `PERFORMANCE.md` and deliberately NOT taken.
2. **`validate_radar_config` silently drops a `"frontend"` key.** Two independent
   stages hit this: the flat mapping accepted by `RadarConfig.from_dict` cannot
   express the Phase-6 receive chain at all and does not raise on the unknown
   key either, so both examples work around it with `dataclasses.replace`. A
   one-line `validate_frontend_config(config["frontend"])` closes it; the
   validator already exists.
3. **There is no route from a `RadarSimulationResult` to a `ProcessingAxes`.**
   Every consumer re-synthesizes the last frame purely to obtain a metadata
   record - an extra waveform launch for metadata. A `axes_record` field built
   once in `simulate_scene`, or a `ProcessingAxes.from_simulation`, closes it.
4. **A polarization parallel to the boresight publishes an exactly zero
   transport with nothing raised.** `DEFAULT_POLARIZATION = (0, 0, 1)` is
   transverse for a radar looking along `x` and parallel for one looking along
   `-z`. Three stages independently paid a bisection to find this. A warning or
   refusal in `bind_radar_world` when every endpoint-to-site direction is within
   a few degrees of the declared polarization would close it.
5. **The `DIFFRACTION_PAIR_REDUCER` NVTX name survives in Channel
   `runtime/profiling.py`** although ADR-030 is Removed. It is a profiling range
   name, not a binding, and four files outside the `channel-dormant` write set
   pin it as frozen comparative evidence. Retiring that evidence group is a
   separate change with its own justification.
