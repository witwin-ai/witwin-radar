# Phase 10 acceptance matrix

Every row of the plan's eight acceptance criteria
(`docs/dev/plans/channel-radar-architecture-plan.md`, "#### Phase 10"), mapped
to the artifact that satisfies it, the command that produced the artifact, and
the measured result.

**The rule this document is written under: no criterion is marked satisfied
without a command that was actually run.** Where a criterion has a part that
cannot be run on this machine, that part is named and points at
`docs/dev/plans/phase10-deferred-release-matrix.md`, which the Phase-10 owner
directive makes a DEFERRAL register rather than a gap list. A row is never
marked satisfied on the strength of a deferral.

Environment for every measurement below: Windows 11, CPython 3.11.14,
Torch 2.10.0, nvcc 12.9.41, RTX 5080 (sm_120), `witwin2` conda environment,
radar worktree `E:/Code/witwin-platform/.worktrees/radar-phase10`.

**No GitHub Actions run was triggered, dispatched, re-run or waited on in any
stage of this phase.**

---

## The eight criteria

### A1 - supported wheels fresh-install, coexist, import, and run a full smoke

> Windows/Linux supported wheels can fresh install, coexist, import and run a
> complete smoke.

**Satisfied on Windows. The Linux half is D1.**

| what | command | result |
|---|---|---|
| Core wheel | `python ci/build_core_wheel.py` (channel repo) | `witwin-0.4.0-py3-none-win_amd64.whl` |
| Channel wheel | `python -m build --wheel --no-isolation` with `RAYD_SOURCE_DIR=E:/ch9rayd`, `CMAKE_CUDA_ARCHITECTURES=120-real` | `witwin_channel-0.4.0-cp311-cp311-win_amd64.whl`, 8.4 MB |
| Radar wheel | `python scripts/build_radar_cuda_prebuilt.py --developer` then `python -m build --wheel --no-isolation` | `witwin_radar-0.3.0-py3-none-win_amd64.whl` |
| Radar wheel audit + fresh install | `python ci/wheel_smoke.py artifacts/nightly/wheels/radar --core-wheel artifacts/nightly/wheels/core` | EXIT 0, `artifacts/nightly/radar-wheel-smoke.v1.json` |
| Channel wheel audit + fresh install | `python ci/wheel_smoke.py artifacts/nightly/wheel --core-wheel artifacts/nightly/core-wheel` (channel repo) | EXIT 0 |
| Three-wheel coexistence, nine scenarios | `python ci/coexistence_smoke.py --core-wheel ... --channel-wheel ... --radar-wheel ...` | EXIT 0, `artifacts/nightly/coexistence.v1.json` |

The coexistence smoke installs all three wheels into one disposable `--target`
directory and runs nine independently attributable `python -I` subprocesses:
A (core alone), B (radar alone), C (the adapter), D (`radar.build_info()`),
E (`channel.build_info()`), F (both extensions in ONE process, running a
two-extension compute), G (JIT refusal), H (dependency closure), I (source
fingerprint tie).

Scenario F is the "run a full smoke" clause: with both binaries loaded in the
same interpreter, the dispatcher namespaces are disjoint
(`['_radar_native']` against Channel's `[]`, because `_channel` is a Python
extension module rather than a dispatcher library), and one crossing compute
completes with the expected delay and beat phase increment.

The archived Radar wheel under `artifacts/phase10/wheels/` was REBUILT at the
final branch tip and both smokes re-run against it, because the first archive
predated a lint commit that touched four shipped modules and
`ci/wheel_smoke.py` correctly refused it (`wheel checked-in source bytes
differ: ['witwin/radar/radar.py', 'witwin/radar/sensors/legacy_paths.py',
'witwin/radar/synthesis/__init__.py', 'witwin/radar/validation.py']`, exit 2).
That refusal is the currency check working, and it is worth stating as a
standing rule: an archived evidence wheel is only evidence for the source it
was built from, so any later commit that changes a shipped `.py` invalidates
it. `ci/coexistence_smoke.py` does not repeat that check - it validates
whichever artifacts it is handed - so the two smokes are complementary rather
than interchangeable, and its docstring now says so.

Deferred: the Linux / `manylinux_2_28` cells for all three packages, **D1**.

### A2 - a Core world-contract import loads no RayD or Channel propagation runtime, and the mesh-SDF CUDA wheel keeps its independent owner

> A Core world-contract import does not load the RayD/Channel propagation
> runtime; the existing mesh-SDF CUDA wheel capability keeps its independent
> owner and validation.

**Satisfied, measured, no deferral.**

| what | command | result |
|---|---|---|
| coexistence scenario A | `ci/coexistence_smoke.py` | from an installed Core wheel: `channel_modules == []`, `forbidden_modules == []`, `mesh_sdf_modules == []`, `cuda_initialized == False`; the `drjit` / `rayd.torch` detector probe confirms the check is not vacuous |
| coexistence scenario B | same | `import witwin.radar`: zero `witwin.channel` modules, `witwin.radar.cuda.build` not loaded |
| coexistence scenario C | same | the adapter loads Channel but NOT `witwin.channel._channel` |
| `quick.import-no-native` | `python ci/run_ci_tier.py quick` | EXIT 0: after `import witwin.radar`, no `witwin.radar.cuda.build`, no `torch.utils.cpp_extension`, `torch.cuda.is_initialized()` false |
| static boundary | `pytest tests/test_phase4_import_boundary.py` | inside the 1644-passing `--gpu` run |

Scenario A carries the mesh-SDF half explicitly: `witwin.core.geometry.mesh_sdf`
is not loaded by a world-contract import, so the Core CUDA capability keeps its
own owner and its own validation rather than being pulled in by Radar's.

### A3 - the Channel and Radar extensions each fail loudly and report full build identity

> Channel/Radar extensions each fail loudly and report complete build identity.

**Satisfied, measured, no deferral.**

Channel already had this (`build_info()`, 28 validated keys, a
`.build-fingerprint` sidecar, `CHANNEL_ABI_VERSION`). Radar had NONE of it: the
loaded object exposed one attribute, `is_available`.

| what | command | result |
|---|---|---|
| the loader contract, in subprocesses | `python ci/run_ci_tier.py cuda` -> `cuda.loader-contract` | **22 passed**. Hidden prebuilt raises instead of compiling; a partial override raises; a corrupted `.build-fingerprint` raises; a mutated source fails the `source_fingerprint`; a byte-flipped binary fails `binary_sha256` |
| the record itself | `python -c "import witwin.radar as r; print(r.build_info())"` | the validated identity plus `origin`, `extension_path` |
| Channel's half | `ci/wheel_smoke.py` (channel) | the 28 `_BUILD_INFO_KEYS`, `__file__` inside the install target |
| diagnostics never raise | `pytest tests/test_phase10_diagnostics.py` | inside the passing suite |
| coexistence scenario G | `ci/coexistence_smoke.py` | with the packaged binary renamed: `RadarExtensionLoadError`, `torch.utils.cpp_extension` never imported, no new directory under the JIT build root |

Scenario G is the sharp end of "fail loudly": before this phase the same
situation silently fell through to `torch.utils.cpp_extension.load` inside the
calling process, which is the `vcvars`/`DllMain` hazard the environment notes
document.

### A4 - every native symbol has a unique owner, a manifest entry, a direct contract test, and an end-to-end caller

> Every native symbol has a unique owner, manifest entry, direct contract test
> and end-to-end caller.

**Satisfied for 33 of the 34 symbols. One symbol has no end-to-end caller and
the manifest says so; no deferral.**

The exception, stated before the evidence rather than after it: the
`dirichlet_spectrum` symbol `backward` (the single-block VJP) carries
`end_to_end_caller: null` and `caller_status: "test_only"`. Its Python owner,
`spectrum_vjp_single_block`, is called only by its contract test, because
`DirichletSolver.backward` dispatches the parallel-bin path instead. The
manifest records that with a nine-line `caller_note` rather than naming a
caller that does not exist, and `tests/test_phase10_binding_registry.py` caps
caller-free symbols at exactly one so the exception cannot spread. Under the
platform guardrail a caller-free ABI symbol is cleanup debt: the honest close
is to delete the symbol - kernel, registration, load probe and coverage row
together - which is a deliberate native change and belongs in Phase 11, not in
a packaging phase.

| what | command | result |
|---|---|---|
| the registry gate | `python ci/check_native_bindings.py` | EXIT 0: "schema 2, 34 operators, 11 AD groups, 6 error owners; symbol-set tie checked against `_radar_native.pyd` (34 symbols)" |
| the gate fires | `pytest tests/test_phase10_binding_registry.py` | passes on the tree, non-zero on mutated copies |
| in the tier | `run_ci_tier.py quick` -> `quick.native-bindings` | EXIT 0 |

Schema 2 turned the manifest from a coverage list into an ownership registry.
Each of the 34 rows carries `python_owner`, `contract_test`,
`end_to_end_caller`, `native_tu`, `numerical_owner`, `ad_role`, `ad_group`,
`launches`, `fused_stages` and `host_observations`, and the gate resolves the
owner/test/caller values against real files and importable dotted paths rather
than accepting them as strings.

`contract_test` is held to a reference rule, not to file existence. The named
test must mention the symbol or the `python_owner` module - the same substring
rule `python_owner` itself is held to - so a row re-pointed at any other
existing test fails. Three rows drive their operator through a solver facade
that names neither (`forward_chunked`, `forward_mimo_linear_chunked` and
`backward_batched`, all reached through `Radar.chirp` / `Radar.mimo`); each one
carries a written `contract_test_note` saying which facade stands in, and the
suite asserts that only those three hold the hatch and that none of them could
have satisfied the rule directly.

The machine-checkable form of "unique": when a packaged prebuilt is present,
the manifest symbol set must EQUAL the `operator_symbols` recorded in its build
sidecar. Registry, shipped binary and loader are tied together, so a symbol
that exists in two of the three fails.

### A5 - no shared RF/geometry binary, no third Python binding, no second RayD registry, no cross-extension private call

> No shared RF/geometry binary, third Python binding, second RayD registry or
> cross-extension private call; only per-helper-audited non-numerical
> validation/schema utilities may share source.

**Satisfied, measured at the BINARY level, no deferral.**

| what | command | result |
|---|---|---|
| the boundary gate | `python ci/check_extension_boundary.py` | EXIT 0, "extension boundary OK" |
| in the tier | `run_ci_tier.py cuda` -> `cuda.extension-boundary` | EXIT 0 |
| the measured evidence | `docs/dev/audit/phase10-extension-boundary.md` | PE import tables of both shipped binaries, with sizes and dates |

The gate reads the SHIPPED BINARY - the PE import table on Windows, `DT_NEEDED`
on Linux - rather than the source. The radar library imports only the CRT,
`KERNEL32`, `MSVCP140`, `VCRUNTIME140`, `cudart64_12`, `torch_cpu` and
`torch_cuda`; it does not import `python311.dll` at all, consistent with
`is_python_module=False` and the Stable ABI target. Neither binary names the
other's stem in either direction, and no `rayd*`, `drjit*`, `mitsuba*` or
`optix*` import appears in the radar binary. `_channel` source-links RayD as a
build-tree CMake target and absorbs it statically, so RayD never becomes a
runtime dependency of anything.

Reading source would have proved less: a shared static archive or a copied
`extern "C"` handshake leaves no import in a header and a very obvious one in a
link closure.

### A6 - architecture-only native moves keep exact outputs, the launch ledger, and performance

> Architecture-only native moves preserve exact outputs, launch ledger and
> performance.

**Satisfied, measured, no deferral.**

The phase's one architecture-only native move is the `_radar_native` rename,
which changes how every `_OPS.<x>` call resolves.

| what | command | result |
|---|---|---|
| exact outputs | a 22-digest probe: sha256 over the RAW BYTES of each operator family's output | **identical before and after the rename** |
| the suite across the rename | `pytest tests --gpu` | 1570 passed / 0 failed on both sides |
| launch ledger, forward | `pytest tests/test_phase6_launch_budget.py --gpu` | passes; `sensor_weight {sensor_weight_forward: 1}`, `frontend {noise: 1, agc: 1, quantize: 1}` |
| launch ledger, backward | `pytest tests/test_phase9_backward_budget.py --gpu` | passes |
| the two together, this stage | `pytest tests/test_phase6_launch_budget.py tests/test_phase9_backward_budget.py -q --gpu` | **27 passed** |
| performance pins | the same two files | backward peak **0.1426 MB** against the 0.1782 MB budget; Channel reevaluate forward 4.1843 ms, forward+backward 6.1340 ms, **ratio 1.4660** against the 2.00 budget and inside the recorded (1.334, 1.523) range |
| the ledger itself | `docs/dev/ad-tape-and-budget-ledger.md` | ten tape owners, unchanged by this phase |

**No budget was raised.** The two ledgers hold at their recorded values and the
measured numbers sit inside their recorded ranges.

The manifest now DECLARES the launch ledger as well - `launches` is 1 on 29
operators and 2 on 5, with `fused_stages` naming what each launch does. The
declaration and the measurement are separate on purpose: the manifest is what a
refactor would have to change deliberately, the tests are what would catch it
if the refactor changed reality instead.

### A7 - release artifacts come from a clean locked build, with no silent fallback

> Release artifacts come from a clean locked build, no silent fallback.

**The no-silent-fallback half is satisfied and measured. The "release artifact
from a clean locked build" half is D4 and D5.**

| what | command | result |
|---|---|---|
| no silent fallback, subprocess evidence | `run_ci_tier.py cuda` -> `cuda.loader-contract` | 22 passed |
| no silent fallback, from an installed wheel | `ci/coexistence_smoke.py` scenario G | `RadarExtensionLoadError`, `cpp_extension` never imported, no new JIT directory |
| no silent fallback, statically | `python ci/check_raw_native_access.py` | one dispatcher owner, ten recorded consumers |
| clean worktree at every measured build | `git status --porcelain` before each wheel build | clean |
| locked RayD for the Channel wheel | `RAYD_SOURCE_DIR=E:/ch9rayd`, a clean clone | the platform `RayD/` checkout is dirty at `4f0e953` and fails `dependencies/rayd.lock.json`, so it was not used |
| honesty about the local build | `scripts/build_radar_cuda_prebuilt.py` without `--release` | every locally built artifact is stamped `build_type="developer"`; `coexistence.v1.json` records `radar_build_type: developer` |

Deferred: **D4** (local nvcc is 12.9.41 against the locked 12.8.1, so no local
artifact may claim "clean locked build"), and **D5** (the still-pending Stage-I
release full build, folded in under the same owner directive). The `--release`
stamp is reachable only from a `release: published` event, so the provenance
claim cannot be made by accident.

### A8 - the Radar wheel needs no Dr.Jit or RayD runtime dependency, and RayD is introduced, locked and fingerprinted only by Channel

> The Radar wheel needs no Dr.Jit or RayD runtime dependency; RayD is
> introduced, locked and reported by fingerprint only by the Channel
> build/runtime package owner.

**Satisfied, measured three independent ways, no deferral.**

| what | command | result |
|---|---|---|
| source | `python ci/check_production_dependencies.py` | EXIT 0: 79 production modules name none of `drjit`, `mitsuba`, `rayd`, `sionna`, no `@dr.wrap` / `@drjit.wrap` decorator, and exactly 2 recorded prose occurrences |
| declared metadata | the same gate, fourth scan | EXIT 0: 11 declared distributions - base list, both extras and `build-system.requires` - carry no ray-tracing runtime. Added after a mutation audit showed a bare `rayd>=0.1` requirement had exactly one catcher in the default test set; a frozen property belongs in a gate that runs in `quick` |
| binary | `python ci/check_extension_boundary.py` | no `rayd*` or `drjit*` import in the radar library |
| dependency resolution | `ci/coexistence_smoke.py` scenario H | `pip install --dry-run --report witwin-radar[channel]` resolves `witwin`, `witwin-channel`, `witwin-radar` and no ray-tracing distribution |
| process | `ci/coexistence_smoke.py` scenarios A, B, C | zero `drjit` / `rayd` modules in any of the three |
| Channel owns the lock | `witwin.channel.build_info()` | carries the RayD repository URL, commit, integration-ABI kind/path/sha256 and source-manifest sha256; Radar's `build_info()` carries no RayD field at all, because it has nothing to report |

The `witwin-radar[channel]` extra is what turns this from an omission into a
constraint. Before Phase 10 the extra did not exist, so "Radar does not require
Channel" was true only because Radar declared no Channel dependency of any
kind. R-ADR-008 records the closure.

---

## The four production static gates (work item 7)

Each runs in `quick`, and each is proven to FAIL on a planted violation in
`tests/test_phase10_static_gates.py` (**32 passed**). A gate that cannot be
shown to fail is a comment with an exit code.

| gate | what it freezes | measured today |
|---|---|---|
| `ci/check_production_dependencies.py` | imports, `@dr.wrap` decorators, string-literal tokens, and DECLARED distributions for `drjit` / `rayd` / `mitsuba` / `sionna` | 79 modules clean; 2 prose occurrences frozen by equality with their reason; 11 declared distributions clean |
| `ci/check_test_oracle_isolation.py` | no production import of `tests`; `packages == ["witwin"]`; no `tests/` member in a built wheel | clean; the wheel half also runs as `nightly.oracle-isolation-wheel` |
| `ci/check_raw_native_access.py` | the dispatcher owner set, by equality; the loader consumer set, by equality; `cuda/identity.py` holds no dispatcher access | 1 owner, 10 consumers |
| `ci/check_torch_physics_allowlist.py` | the whole tree with an empty frozen exclusion list, against `ci/torch-physics-allowlist.json`, under a `FROZEN_BASELINE_DIGEST` | 79 modules, 23 recorded expressions, 34 occurrences |

The fourth gate's widened scope found one expression the old one-package scan
could not see: `witwin/radar/sensors/pattern.py`'s `torch.atan2` pair, recorded
as debt with its reason rather than repaired. Moving it is a numerical change
and belongs in its own commit with its own evidence.

---

## Tier evidence

Both radar tiers were run end to end in this worktree.

```
python ci/run_ci_tier.py quick    EXIT 0   (10 gates)
python ci/run_ci_tier.py cuda     EXIT 0   (14 gates)
```

| gate | result |
|---|---|
| `quick.ruff` | All checks passed (18 pre-existing errors fixed in this stage) |
| `quick.native-bindings` | EXIT 0 |
| `quick.production-dependencies` | EXIT 0 |
| `quick.oracle-isolation` | EXIT 0 |
| `quick.raw-native-access` | EXIT 0 |
| `quick.torch-physics-allowlist` | EXIT 0 |
| `quick.workflow-policy` | EXIT 0 |
| `quick.import-no-native` | EXIT 0 |
| `quick.cpu-tests` | **843 passed, 807 skipped** |
| `quick.cpu-coverage` | TOTAL 64%, floor 50 |
| `cuda.gpu-tests` | **1644 passed, 0 failed** (see the wall-clock note below) |
| `cuda.gpu-coverage` | TOTAL 81%, floor 75 |
| `cuda.loader-contract` | 22 passed |
| `cuda.extension-boundary` | EXIT 0, "extension boundary OK" |

Suite counts against the phase baseline (`28b5360`): default 699 passed / 807
skipped becomes **843 passed / 807 skipped**; `--gpu` 1506 passed / 0 failed
becomes **1650 passed / 0 failed**. Every added test is a Phase-10 test; no
existing test was weakened, skipped or removed.

**Wall-clock note, measured during remediation on the same tip.** The bare
`pytest -q tests --gpu` run is green (1650 passed, 0 failed), but two
consecutive `run_ci_tier.py cuda` runs failed on
`tests/test_phase8_pipeline_budget.py` (`2 failed / 1648 passed`, then
`1 failed / 1649 passed`), with medians 3.384 ms against the 2.899 ms pipeline
budget and 6.530 ms against the 5.044 ms frame budget. Isolated file reruns
alternate pass and fail both with and without `coverage`. The failure
reproduces at the phase baseline `28b5360`, the GPU is idle when it happens,
and the diff touches no budget, tolerance or anything on that test's path: it
is a host wall-clock flake on this machine, made likelier by running the suite
under `coverage run`. **No budget was raised.** The correct fix is to make the
measurement robust - more repetitions, a rejected-outlier median, or moving the
pin to a device-time measurement - which is a numerical/perf change with its
own evidence and belongs in Phase 11, not in a packaging phase.

`nightly` and `release` were exercised gate by gate rather than as whole tiers,
because `nightly.coexistence-smoke` needs a Channel wheel this repository
cannot build and `release.arch-verification` expects the ten-image release SASS
set that only a release runner produces (**D3**). Every individual `nightly`
gate was run and returned EXIT 0.

---

## Named deferrals

`docs/dev/plans/phase10-deferred-release-matrix.md` is the register. Under the
Phase-10 owner directive these are deferrals with named owners and named
executing workflows, not gaps, and Phase 11 proceeds without waiting on them.

| id | what | which criterion it touches |
|---|---|---|
| D1 | Linux / `manylinux_2_28` cells for all three packages | A1 |
| D2 | SM87 runtime validation | A1 |
| D3 | the full ten-image SASS set | A1, A7 |
| D4 | the "clean locked build" claim | A7 |
| D5 | the Stage-I release full build | A7 |
| D6 | the 8-cell Stable ABI compatibility matrix for Radar | A3 |

**D6 is not merely unrun.** As measured in stage S4, R-ADR-019's runtime
identity check pins `torch_version` and `cuda_version`, so six of the sixteen
configured cells will report a `RadarExtensionABIError` rather than a load.
The workflow asserts the contract AS BUILT in both directions and emits a
warning; resolving it needs an architecture decision, not a workflow edit.
