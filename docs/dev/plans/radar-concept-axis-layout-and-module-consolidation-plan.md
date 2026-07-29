# Radar concept-axis layout and module consolidation plan

Status: **completed — all consolidation, API reset, native flattening, governance and documentation gates passed.**

Binding adversarial corrections are recorded in
`docs/dev/audit/radar-consolidation-execution-amendments.md`.

Recorded: 2026-07-28.

## Execution outcome

Completed on 2026-07-28. The production Python tree is 26 modules (down from
71) behind 10 declared public facades. FMCW now synthesizes the native direct
Dirichlet range spectrum by default; explicit `output_domain="beat"` selects
time-domain beat synthesis. Compatibility facades, aliases and overloads were
deleted, the governance inventory has zero open rows, and living documentation
resolves to current owners.

Final evidence: 1483 tests collected; CPU 687 passed / 796 skipped; GPU 1470
passed / 13 skipped. All architecture, API, native-boundary, Torch-policy,
documentation, release, workflow and governance gates pass without widening
the recorded performance budgets.

Reference:
`channel/docs/dev/plans/15-concept-axis-layout-and-module-consolidation-plan.md`.
This is the Radar counterpart, not a copy of Channel's target tree. Channel is
organized around propagation interactions and solvers; Radar is organized
around the round-trip signal chain and the products each stage owns.

The proposal assumes the Phase-11 cutover is complete: the production route is
already Core `Scene` -> Channel compact propagation consumer -> Radar two-way
composition/scattering/sensor weighting -> waveform synthesis -> frontend ->
processing. It does not reopen the deleted Radar `Scene`, tracer, Dirichlet
solver, legacy receiver chain, or any shadow route.

## Decision log

| Date | Decision |
| --- | --- |
| 2026-07-28 | Radar has no maximum Python file-line limit. File length is an observation, not an architectural constraint. |
| 2026-07-28 | Concept ownership is the primary layout axis. `contracts`, `utils`, `adapters`, `base`, and `components` are not allowed to become top-level architectural axes merely because they are artifact categories. |
| 2026-07-28 | The target is approximately 26 Python files, down from 71, while keeping the native numerical families separate. |
| 2026-07-28 | `processing/`, `synthesis/`, and `cuda/` remain packages because each contains several independently changing concepts. Most other packages collapse to top-level concept modules. |
| 2026-07-28 | The Channel boundary remains explicit and unique. Flattening must not broaden the set of Radar modules allowed to import `witwin.channel`. |
| 2026-07-28 | `witwin.radar.sigproc`, `processing/adapters.py`, root compatibility behavior and compatibility-only overloads are deleted before concept consolidation. This is an intentional API reset, not an incidental file move. |
| 2026-07-28 | Native CUDA source flattening is a separate late phase because it changes build manifests, source fingerprints, and the packaged binary even when the arithmetic is unchanged. |
| 2026-07-28 | FMCW direct Dirichlet range-spectrum synthesis is restored as the default output domain. The current time-domain beat synthesis remains available only by explicit selection. Both live under one `fmcw.py` owner; the deleted solver/Dirichlet route is not restored as a second pipeline. |
| 2026-07-28 | No backward-compatibility layer is required or permitted. Stable public APIs may change when the concept layout or owner boundary calls for it. Removed names disappear normally; no deprecated alias, migration adapter, `_REMOVED` error proxy, legacy overload, or compatibility re-export survives. |
| 2026-07-28 | Existing governance debt and code/document drift are part of this plan's blocking scope. The consolidation is not complete while a recorded debt, stale living document, policy contradiction, path-keyed stale owner, or compatibility-only route remains. |

---

## 1. Context and measured baseline

At `ff2d9cc`, the Radar package contains:

| Area | Python files | Python lines |
| --- | ---: | ---: |
| package root | 10 | 3,748 |
| `cuda/` | 3 | 1,092 |
| `frontend/` | 3 | 1,056 |
| `geometry/` | 2 | 516 |
| `paths/` | 6 | 1,939 |
| `processing/` | 17 | 4,586 |
| `propagation/` | 5 | 2,620 |
| `scattering/` | 4 | 871 |
| `sensors/` | 5 | 1,686 |
| `sigproc/` | 6 | 204 |
| `synthesis/` | 7 | 3,378 |
| `utils/` | 3 | 117 |
| **Total** | **71** | **21,813** |

The native side contains eight translation units (`extension.cpp` plus seven
CUDA files), 6,330 source lines. Those translation units already follow useful
numerical ownership boundaries:

- FMCW beat synthesis;
- OFDM CFR synthesis;
- pulsed echo synthesis;
- two-way join;
- aspect-dependent scatter response;
- sensor weighting;
- receiver frontend.

The Python tree is only two directories deep, so depth alone is not the main
problem. The problem is horizontal fragmentation:

- one concept is split into an `__init__.py`, `contracts.py`, implementation
  file, and one or more helpers;
- `config.py` and `validation.py` separately describe the same Radar-authored
  configuration model;
- the Channel consumer, propagation contracts, kinematics, epoch policy, scene
  binding, and frame loop form one production chain but are spread over seven
  files in three locations;
- path composition is six files although it has only two public composition
  modes;
- sensor weighting is five files and scattering response is four files;
- `processing/` has seventeen files, several under 200 lines, indexed partly by
  artifact kind rather than by output product;
- `sigproc/` is six compatibility modules backed by an eighth, 848-line
  migration adapter in `processing/adapters.py`;
- `utils/` is three modules, but only `utils.vector.vec3_tensor` has a production
  importer outside the package initializer.

The target is not “one huge file.” It is “one obvious owner per concept, with a
package only where several concepts genuinely evolve independently.”

---

## 2. Architectural rules

### 2.1 No file-line ceiling

Radar currently has no `file_lines` maintenance gate. This plan makes the policy
explicit: **do not add one as part of this consolidation.**

A 2,000-4,000 line concept module is acceptable when:

- its sections belong to one change axis;
- functions remain individually testable;
- public and native ownership remains unambiguous;
- no import cycle is hidden by function-local imports;
- a second definition of the same concept is not created elsewhere.

Removing a file limit does not authorize:

- growing function complexity without tests;
- mixing unrelated numerical owners;
- weakening input validation;
- adding mutable module-global state;
- hiding a second pipeline in a large file;
- merging native translation units merely to reduce the file count.

### 2.2 Concept axis

The primary Radar concepts are:

1. Radar system definition and pose;
2. simulation/session orchestration;
3. Channel propagation consumption;
4. propagation epoch and motion policy;
5. round-trip path identity and composition;
6. target scattering response;
7. sensor/array weighting;
8. waveform synthesis;
9. receiver frontend;
10. post-processing products;
11. optional SMPL-to-Core authoring;
12. native runtime/deployment.

`contracts`, `config`, `validation`, `base`, `components`, `selection`,
`assembly`, `utils`, and `adapters` are implementation roles. They may exist as
sections within a concept owner, but they do not justify a package by
themselves.

### 2.3 Ownership boundaries that must survive

- Core remains the owner of the logical scene, geometry, material, structure,
  dynamics, and snapshots.
- Channel remains the owner of compiled scenes, RayD resources, propagation
  discovery/evaluation, and the compact propagation consumer schema.
- Radar imports Channel in exactly one production module.
- Radar owns two-way composition, target response, sensor weighting, waveform
  synthesis, frontend, and signal processing.
- `_channel` and `_radar_native` remain separate native artifacts. Radar never
  links Channel or RayD.
- A move cannot introduce a Torch propagation fallback, host observation,
  finite difference, compatibility shadow, or partial result.
- Discrete processing remains behind the R-ADR-018 differentiability wall.

### 2.4 Public surface policy

There is no backward-compatibility requirement. The current public snapshot is
baseline evidence, not a promise the target layout must preserve.

The target public surface is rebuilt from ownership:

- `witwin.radar` exports Radar-owned primary entry points only;
- Core scene, geometry, material and structure types are imported from
  `witwin.core`, not re-exported by Radar;
- processing algorithms and records are imported from
  `witwin.radar.processing`, not duplicated at the root;
- synthesis algorithms and records are imported from
  `witwin.radar.synthesis`;
- SMPL integration, while Radar owns it, is imported from `witwin.radar.smpl`;
- scene-binding utilities live under `witwin.radar.simulation` unless they are
  required to construct the primary `Radar.simulate` call;
- removed symbols raise normal `AttributeError` / `ImportError`. The root
  `_REMOVED` dictionary and its replacement prose are deleted.

`ci/public-api-snapshot.json` is regenerated after the new surface is accepted.
It freezes the result against accidental drift; it does not force deprecated
aliases, old signatures, compatibility overloads, or old definition targets
into the new architecture.

No moved or removed path receives:

- a deprecated alias;
- a module-level `__getattr__` compatibility branch;
- a wrapper preserving an old signature or output convention;
- an old submodule re-export;
- a runtime fallback;
- a “helpful” replacement object under an old name.

Repository callers, examples and living documentation move in the same phase.
Historical audits may name old paths only when clearly marked historical or
superseded.

---

## 3. Prerequisites

### P0 — Establish an immutable integration baseline

Do not execute this plan against the current mutable sibling checkouts. At the
time this document was written:

- Radar is clean at `ff2d9cc`;
- Channel is on `refactor/plan15-concept-axis` with a large uncommitted native
  translation-unit consolidation;
- Core also has uncommitted changes.

Execution begins only after selecting and recording:

- one clean Core commit;
- one clean Channel commit;
- one matching Channel native build and fingerprint;
- the Radar base commit;
- Python, Torch, CUDA, compiler, driver, GPU, and operating system identities.

The baseline records both the default and `--gpu` pytest node-id sets. A later
phase is not accepted merely because “tests are green”; the collected node set
must not shrink.

### P1 — Ratify the no-file-limit policy

No code gate needs removal in Radar, but the owner decision still needs to be
recorded in the repository guidance:

- file size is not capped;
- function-level clarity and testability still matter;
- concept ownership, single definition, import direction, and test coverage
  replace file size as the structural controls.

This prevents a later reviewer from splitting the target modules back into
artifact-category files solely because a line count looks large.

### P2 — Delete all compatibility surfaces

The owner decision is final: remove, without a deprecation period:

- `witwin/radar/sigproc/` (six files);
- `witwin/radar/processing/adapters.py`;
- the legacy golden used only by those adapters;
- the `sigproc` entries in the orphan gate and public API snapshot.
- root `_REMOVED`, its `__getattr__` branches and tests that require replacement
  prose for deleted names;
- Radar-root re-exports of Core geometry/material/structure types;
- Radar-root convenience re-exports whose owner is `processing`, `simulation`,
  `synthesis` or `smpl`;
- old internal submodule imports and re-export packages.

The same phase audits behavior-bearing compatibility candidates and deletes
each one that has no current owner-approved use:

- `SynthesisPathBatch.from_real_amplitudes`;
- sensor-weight modes that only reconstruct the deleted real-amplitude route;
- the legacy receive-polarization projection;
- bare-tensor-plus-axes processing overloads retained only beside the typed
  `ProcessingCube` route;
- NumPy-returning wrappers beside tensor-native processing results;
- old configuration keys, aliases or defaults whose only justification is a
  removed route;
- SMPL behaviors or exports documented as serving the deleted Radar `Scene`.

This list is a starting census, not an allowlist. A repository-wide semantic
scan must prove what remains is a current feature with a named owner and current
caller. “A test imports it” is not evidence that it is production API.

There is no migration adapter and no deprecation release. A short breaking
change note may state the new API, but it must not provide or preserve an
executable old path.

### P3 — Decide the SMPL owner, without blocking layout

`SMPLBody` and `SmplPoseDeformation` author Core geometry and deformation but
currently live in Radar. Cross-repository ownership may eventually move them to
Core or a body-model integration package.

That decision does not block this plan. Until it is made, collapse
`geometry/__init__.py + geometry/smpl.py` to top-level `smpl.py`. The root
`SMPLBody` re-export is not preserved; callers use the owning module directly.
Do not perform a cross-repository ownership move as part of file cleanup.

### P4 — Capture the old surface, then ratify the new one

Before moving files, record:

- the exact old `ci/public-api-snapshot.json` as before-evidence;
- the one allowed Channel-importing module;
- `ci/check_orphan_modules.py::ENTRY_POINTS`;
- native operator -> Python owner mapping;
- the Torch-physics allowlist;
- AD owner and tape-owner paths;
- package import closure in a fresh process.

Then approve a new public inventory by concept. Every retained export must name
its owner, primary caller and reason to be public. The new snapshot is generated
from that inventory after incompatible removals and signature cleanup; it is
not obtained by editing the old snapshot until tests pass.

This makes both removals and path-keyed governance migration reviewable rather
than a bulk search-and-replace with no independent reference.

### P5 — Ratify the FMCW output-domain contract

This is an approved product direction but not a move-only change. It needs a
dedicated ADR and acceptance record before implementation.

The public setting is an output domain, not an implementation hint:

```python
fmcw_output_domain: Literal["range_spectrum", "beat"] = "range_spectrum"
```

- `"range_spectrum"` is the default. The native kernel evaluates the Dirichlet
  spectrum directly from compact round-trip path rows.
- `"beat"` explicitly selects the current per-chirp, per-sample time-domain
  synthesis path.
- An unknown value fails during configuration validation.
- There is no `"auto"` value and no runtime fallback from spectrum to beat.

The setting belongs to the stored FMCW waveform configuration and reaches the
FMCW spec as data. It is not inferred from a consumer, a requested processing
stage, `requires_grad`, or the presence of a frontend.

The direct-spectrum contract is:

1. **One FMCW owner.** `synthesis/fmcw.py` owns both modes and their dispatch.
   Do not resurrect `solvers/`, `DirichletSolver`,
   `synthesis/dirichlet_spectrum.py`, or a second public FMCW facade.
2. **Current compact input.** The spectrum kernel consumes
   `SynthesisPathBatch`: total round-trip delay/rate, complex Channel-domain
   transfer, row validity, pair segmentation and slow-time mode. It must not
   restore the old tracer distances, legacy amplitude normalization, path
   cache, sensor loss, or radar-owned scene sampling.
3. **Observable domain metadata.** `SynthesisResult` and the assembled
   processing cube carry `domain="range_spectrum" | "beat"`. FMCW spectrum axes
   are `("chirp", "sensor_pair", "range_bin")`; beat axes remain
   `("chirp", "sensor_pair", "sample")`. A consumer never infers the domain
   from shape.
4. **No second FFT.** `processing.range_profile` treats a direct spectrum as an
   already formed range profile. It must not apply another fast-time FFT.
5. **Window ownership is explicit.** Direct spectrum initially represents the
   rectangular-window DFT, matching the existing processing default. A request
   for a non-rectangular range window fails before computation and tells the
   caller to select `beat`; it must not silently IFFT, window, and FFT. A later
   native windowed-Dirichlet family would be a separate numerical extension.
6. **DC removal stays exact.** For a rectangular direct spectrum,
   `remove_dc=True` is implemented by the exact spectral equivalent of
   subtracting the fast-time mean (zeroing the DC range bin), without returning
   to time domain.
7. **Frontend domain is enforced.** LNA/noise/ADC/quantization are time-domain
   receiver operations. A configured frontend is incompatible with
   `range_spectrum` and fails configuration validation; the caller must
   explicitly select `beat`. There is no automatic mode change.
8. **Default behavior is an intentional break.** `Radar.simulate` will return a
   range-bin FMCW cube by default instead of a sample-domain beat cube. The
   public snapshot, examples, pipeline guide and breaking-change note must say
   so.
9. **The initial grid is exact, not inferred.** The direct spectrum has
   `num_samples` bins and equals the length-`num_samples`, `norm="forward"` DFT
   of the retained beat path. The deleted solver's `pad_factor`, alternate
   `N_fft`, truncated-bin and IFFT orchestration are not restored. Optional
   zero-padding is a later processing/config feature with its own axis and
   normalization contract.

The old `dirichlet.cu` is evidence and a mathematical oracle, not a file to
copy back wholesale. The new native owner should start with the minimum public
operator family needed by the current AD contract:

- `fmcw_spectrum_forward`;
- `fmcw_spectrum_backward`;
- `fmcw_spectrum_jvp`;
- the existing beat forward/backward/JVP family.

Chunking and alternative backward strategies may be internal CUDA helpers. They
become additional registered ABI operators only if measurement proves that a
single entry cannot meet the accepted memory/performance budget.

### P6 — Complete a governance-debt and documentation-drift audit

Before consolidation, create
`docs/dev/audit/radar-governance-debt-and-drift-inventory.md`. Every row has:

- a stable debt ID;
- category;
- concrete file/symbol;
- why it is debt rather than a current feature;
- owning phase;
- falsifying gate/test;
- status;
- closing commit and evidence.

The inventory starts with the debts already observed while drafting this plan:

| ID | Observed debt | Required disposition |
| --- | --- | --- |
| GOV-001 | `witwin/radar/__init__.py::_REMOVED` preserves old names through custom errors | delete dictionary, compatibility `__getattr__` branch and tests that require it |
| GOV-002 | `sigproc/` plus `processing/adapters.py` preserve the pre-Phase-8 API and numerical conventions | delete completely |
| GOV-003 | Radar root re-exports Core world types and widens `Geometry` with `SMPLBody` | remove; import from owning packages |
| GOV-004 | compatibility candidates remain in synthesis/sensor/processing contracts (`from_real_amplitudes`, legacy weight/projection modes, NumPy and bare-tensor overloads) | semantic caller audit; delete every compatibility-only path |
| GOV-005 | `ci/torch-physics-allowlist.json` explicitly records `freeze_time_pattern_oracle` as debt | eliminate the Torch production accessor or move the necessary operation to its approved owner; remove the debt entry, do not reclassify it |
| GOV-006 | Radar has no standalone layer/import-graph gate comparable to Channel's | add an exact import-layer/cross-domain gate with planted-violation tests |
| GOV-007 | owner/module inventories are repeated manually in tests (`SPIKE_MODULES`, entry points and owner lists) | create one architecture manifest or generated inventory and make all checks consume it |
| GOV-008 | `AGENTS.md` / `CLAUDE.md` still describe the deleted Radar Scene, Tracer, Timeline and Dirichlet solver architecture | rewrite both to the current/final concept architecture and enforce equality/currentness |
| GOV-009 | living documents contain stale statements such as SMPL publishing into the legacy Radar Scene and compatibility adapters as current features | correct all living docs and executable examples |
| GOV-010 | many governance documents pin fragile `file.py:line` owners that become false after consolidation | migrate to `module::qualified_symbol` where possible; generate line links only as derived evidence |
| GOV-011 | Phase-10 Stable-ABI policy conflicts with strict runtime Torch/CUDA identity equality | resolve by ADR: define a real compatibility rule or abandon the cross-version Stable-ABI claim; no contradictory release matrix may remain |
| GOV-012 | workflow policy exceptions/deferred release rows can be mistaken for satisfied governance | classify each as resolved, externally blocked release evidence, or approved policy exception with owner and expiry/review trigger |
| GOV-013 | no machine gate forbids compatibility shims from regrowing | add `ci/check_no_compatibility.py` and planted violations |
| GOV-014 | no machine gate checks living documentation for deleted paths, wrong defaults and stale owner names | add a documentation-surface/drift gate with an explicit historical-document boundary |

The audit is not limited to this seed table. It scans:

- production imports and module reachability;
- public exports, overloads, aliases and lazy attributes;
- native dispatcher owners and callers;
- AD owners, tape owners and refused tangents;
- every allowlist, frozen digest, exception and deferred row;
- configuration keys/defaults and result metadata;
- tests kept alive only to preserve removed behavior;
- examples and all living documentation;
- build, wheel, workflow and release-policy claims.

No new permanent allowlist may be introduced to make consolidation pass. A
temporary phase-local relocation list is allowed only on the working branch,
must shrink monotonically, and is deleted before the phase closes.

#### Living versus historical documentation

Documentation is classified, not searched as one undifferentiated tree:

- **Living:** `AGENTS.md`, `CLAUDE.md`, README, FEATURE_LIST, PERFORMANCE,
  pipeline guide, examples, current API/AD matrices, build instructions and
  release policy. These must contain zero stale production path, signature,
  default, output domain, owner or command.
- **Historical:** dated audits, accepted/superseded ADR text, migration evidence
  and old baselines. Historical facts remain intact, but each document that
  names a superseded architecture receives a visible status banner and a link
  to the current owner/decision. Historical code blocks must not be presented
  as current commands.

`ci/check_documentation_surface.py` enforces at minimum:

- every path claimed as current exists;
- every public symbol claimed as current resolves from its documented owner;
- documented FMCW default/domain matches the config default;
- deleted compatibility names appear in no living document;
- examples import only the new public surface;
- repository guidance describes the same target architecture;
- historical exceptions are confined to explicitly classified paths.

---

## 4. Target layout

```text
witwin/radar/
├── __init__.py              minimal public root: Radar-owned primary entry points
├── radar.py              ★  Radar, RadarConfig, system blocks, validation,
│                            axes construction, pose and array placement
├── simulation.py         ★  Core scene binding, stable IDs, site policy,
│                            frame/session orchestration and diagnostics
├── channel.py            ★  the ONLY Channel import boundary; compile facade,
│                            endpoint conversion, freeze and slot reevaluation
├── propagation.py        ★  leg contracts, kinematics, epoch/motion policy
├── paths.py              ★  identity, direct and two-way composition,
│                            component partition/export
├── scattering.py         ★  response protocol, scalar RCS and aspect response
├── sensors.py            ★  array/polarization/pattern contracts, native sensor
│                            weights and round-trip pattern stage
├── frontend.py           ★  frontend spec, stages, RNG and native execution
├── smpl.py                  optional SMPL body/deformation authoring bridge
├── policy.py                first-order AD wall and host-parameter admission
├── capabilities.py          Radar and already-loaded Channel capability record
├── deployment.py            build identity and runtime diagnostics
│
├── synthesis/
│   ├── __init__.py          public facade, shared path/result records, dispatch
│   ├── assembly.py          pair/segment indexing and frame cube assembly
│   ├── fmcw.py           ★  one FMCW owner: default direct Dirichlet spectrum,
│   │                        optional time-domain beat, shared AD and provenance
│   ├── ofdm.py           ★  OFDM spec, validation and native CFR synthesis
│   └── pulsed.py         ★  pulsed spec, validation and native echo synthesis
│
├── processing/
│   ├── __init__.py          public facade
│   ├── signal.py         ★  axes, cube/result records, windows and common signal
│                            primitives
│   ├── range_doppler.py  ★  range profile, Doppler, matched filter and
│                            micro-Doppler
│   ├── angle.py          ★  AoA, steering, beamforming and beam cube
│   ├── detection.py      ★  CFAR, point cloud and incoherent combination
│   └── tracking.py       ★  detection frames and tracks
│
└── cuda/
    ├── __init__.py
    ├── runtime.py        ★  packaged load, identity, ABI validation and operator
    │                        lookup (`build.py + identity.py`)
    ├── extension.cpp
    ├── fmcw.cu             default Dirichlet spectrum + optional beat kernels
    ├── frontend.cu
    ├── ofdm_cfr.cu
    ├── pulsed_echo.cu
    ├── scatter_response.cu
    ├── sensor_weight.cu
    ├── two_way_join.cu
    └── prebuilt/            packaged artifact and identity sidecars
```

Expected result after the mandatory P2 removal:

| Measure | Current | Target | Change |
| --- | ---: | ---: | ---: |
| Python files | 71 | about 26 | about -63% |
| Python subpackages below root | 11 | 3 | about -73% |
| compatibility-only Python files | 7 | 0 | -100% |
| native translation units | 8 | 8 | unchanged |
| native source directory levels below `radar/` | 2 | 1 | flatter |

The target count is a forecast, not a gate. If preserving a real owner requires
27 or 29 files, do that. The plan rejects both “one file per artifact kind” and
“one file for the entire package.”

---

## 5. Consolidation rationale by concept

### 5.1 `radar.py`: authored Radar state

Absorb:

- current `radar.py`;
- `config.py`;
- `validation.py`;
- `utils/vector.py`.

These files change together whenever a field, waveform discriminator, sensor
layout, pose rule, or validation rule changes. Keeping validation in a separate
770-line file makes the same configuration concept have two owners.

`RadarConfig`, its system block records, validation, axes construction, and pose
transforms become sections of one module. After Phase 1 ratifies the new
contract, validation order and exact error messages are frozen behavior and must
not change during the move.

`policy.py` separately absorbs all of `ad_contracts.py` and
`host_parameters.py`: first-order AD admission and host-parameter admission are
cross-cutting pipeline policy, not Radar configuration fields.

`utils/` receives no replacement package. `vec3_tensor`, the one helper with a
production consumer, moves into `radar.py`. The remaining tensor/vector helpers
first receive a repository-wide consumer inventory; an actually used helper
moves to its concept owner, and an internal helper kept alive only by
`utils/__init__.py` is deleted rather than re-exported from a new grab-bag.

### 5.2 `simulation.py`: world-to-frame session

Absorb current `scene_binding.py` into `simulation.py`.

Stable ID allocation, scatter-site policy, endpoint/site binding, epoch loop
construction, frame iteration, result assembly, and last-frame diagnostics are
one session-level concept. They may remain separate classes and functions
inside the file; consolidation does not mean inlining them into one function.

### 5.3 `channel.py` and `propagation.py`: keep the cross-domain seam visible

Do **not** collapse all five propagation files into a module that casually
imports Channel.

- `channel.py` owns the only allowed `witwin.channel` imports and the adapter's
  compile/freeze/reevaluate lifecycle.
- `propagation.py` owns Radar-side endpoint/leg contracts, kinematics, epoch
  policy, and motion declarations without importing Channel.

This is two files rather than the current five, but more importantly it retains
an auditable cross-domain seam. The canonical architecture manifest changes its
sole Channel importer from `propagation/channel_consumer.py` to `channel.py`,
and the import-boundary gate continues to require a set of size one.

### 5.4 `paths.py`: round-trip identity and composition

Collapse the current `paths/` package:

- `_identity.py`;
- `contracts.py`;
- `components.py`;
- `direct.py`;
- `two_way.py`;
- `__init__.py`.

The file is organized in this order:

1. identity and freeze-time host offsets;
2. leg/path/topology records;
3. direct composition;
4. two-way native autograd owner;
5. component partition/export;
6. public facade.

This keeps the `witwin.radar.paths` import path while removing internal
artifact-category paths. The native binding manifest still names the exact
two-way functions inside the collapsed owner.

### 5.5 `scattering.py`, `sensors.py`, and `frontend.py`

Each current package becomes one top-level concept module:

- `scattering.py`: response protocol, scalar RCS and aspect response;
- `sensors.py`: sensor contracts, pattern evaluation, native weights and the
  round-trip pattern stage;
- `frontend.py`: frontend spec, stages, Philox realization and native chain.

The native numerical families remain separate CUDA translation units. A shared
Python file does not imply shared native arithmetic or a merged AD group.

### 5.6 `synthesis/`: one file per waveform

Keep `synthesis/` because FMCW, OFDM and pulsed waveforms change independently.
Remove artifact-category fragmentation instead:

- move each waveform's spec, validation rules and output contract beside its
  implementation;
- keep only cross-waveform path/result records and dispatch in `__init__.py`;
- keep assembly/indexing in `assembly.py`;
- absorb `selection.py` into the facade or the consumer that owns the
  selection.

The result is five files instead of seven. More importantly, changing an OFDM
contract no longer requires editing a 1,784-line cross-waveform
`contracts.py`.

`fmcw.py` is deliberately broader than the current `fmcw.py`: it is the
single owner of the FMCW waveform and exposes two output domains. The direct
Dirichlet range spectrum is the stored default; the existing beat cube is an
explicit option. Shared carrier-home, Channel-to-beat phasor, TDM slot, delay
rate, row-validity and AD rules are defined once and consumed by both paths.

The two modes must satisfy the defining equivalence for the rectangular window:

```text
direct_dirichlet_spectrum(paths)
    == FFT(synthesize_beat(paths), norm="forward")
```

under the repository's existing tolerances, across stationary/moving,
single/multi-path, TDM-MIMO, narrowband provenance, primal, JVP and VJP
cases, while spectrum and beat preserve the same explicit refusal of discrete
wideband `frequency_response`. This equivalence is the reason the two implementations are modes of one
owner rather than two FMCW pipelines.

### 5.7 `processing/`: product axis

Seventeen files become six:

- `signal.py`: axes, typed products, cube wrapper, windows and shared transforms;
- `range_doppler.py`: fast-time/slow-time estimation and pulse compression;
- `angle.py`: array geometry, AoA and beamforming;
- `detection.py`: CFAR, point clouds and response combination;
- `tracking.py`: temporal association;
- `__init__.py`: public facade only.

This preserves genuine algorithmic owners while eliminating tiny files that
exist only because “contract,” “axis,” “cube,” “primitive,” and “product” were
treated as parallel architectural categories.

P2 deletes `processing/adapters.py` and `sigproc/` before this merge. No legacy
window, Doppler-sign, NumPy-return or old dispatch behavior is folded into the
new concept owners merely to keep an obsolete call working.

### 5.8 `cuda/`: flatten sources, preserve numerical owners

Merge `cuda/build.py` and `cuda/identity.py` into `cuda/runtime.py`. Move the
seven `.cu` files from `cuda/kernels/` directly under `cuda/`.

Do not merge the seven numerical translation units in this plan. Radar's native
surface already has one useful file per numerical family, unlike the Channel
translation-unit layout that motivated ADR-044.

Because source paths participate in the build identity, this phase requires:

- a new source digest;
- a rebuilt `_radar_native`;
- updated prebuilt sidecars;
- updated native binding source paths;
- wheel and coexistence checks.

It cannot be reviewed as a Python-only rename.

---

## 6. Migration phases

Move-only phases preserve arithmetic, evaluation order, AD support, tensor
shapes, dtypes, devices, phase convention, path ordering and RNG realization.
They do **not** preserve a public name, signature, overload or default whose
retention would be compatibility work: intentional API cleanup is declared in
the phase and the new snapshot moves with it. Phase **F** is the numerical
behavior-bearing exception authorized by P5; it has its own numerical, AD,
performance and breaking-change evidence and must not be hidden inside a layout
commit.

| Phase | Content | Depends on | Acceptance gate |
| --- | --- | --- | --- |
| **0 — baseline** | Pin the Core commit/tree plus Channel/Radar commits and native fingerprints; capture file inventory, old public snapshot, import closure, pytest node IDs, quick/cuda results and numerical smoke outputs | P0-P4 | immutable before-evidence committed |
| **G — governance census/gates** | Complete P6 inventory; add import-graph, single-definition, no-compatibility and documentation-drift gates; classify every exception/deferred row; correct AGENTS/CLAUDE enough to describe the execution baseline | 0, P6 | every debt has owner/phase/falsifier; each new gate fails on planted violations |
| **1 — API reset and compatibility deletion** | Delete `sigproc`, adapters, `_REMOVED`, root convenience/Core re-exports and every audited compatibility-only overload/mode/golden; absorb or delete `utils`; ratify and snapshot the new public inventory | G, P2, P4 | Phase-1 retired-surface scan is zero; the terminal gate remainder is exactly the scheduled typed-handoff/native/FMCW rows; new public snapshot is intentional |
| **2 — system/session** | Consolidate `radar.py`; merge scene binding into `simulation.py`; flatten `geometry/` to `smpl.py`; create `policy.py`; simplify signatures/defaults where concept ownership requires | 1 | config/pose/scene-binding suites; snapshot matches the approved new surface |
| **3 — propagation/path** | Create `channel.py` and `propagation.py`; collapse `paths/` to `paths.py` | 2 | exactly one Channel importer; path/AD/epoch targeted suites; node IDs preserved |
| **4 — response chain** | Collapse `scattering/`, `sensors/`, and `frontend/` to top-level files | 3 | native owner/AD manifest paths updated; response, weights and frontend tests exact |
| **5 — synthesis** | Reorganize `synthesis/` by waveform; move specs beside implementations; delete empty artifact modules | 4 | all three waveform analytic and AD suites; cross-waveform frozen-batch agreement |
| **6 — processing** | Reorganize `processing/` by output product; expose typed tensor-native products only | 5 | new processing snapshot; exact-bin/false-alarm/angle/tracking suites; no compatibility golden |
| **7 — native layout** | Merge Python loader modules; flatten CUDA source directory; rebuild packaged native extension | 6 | binding registry, ABI identity, wheel, coexistence, quick and cuda tiers |
| **F — FMCW spectrum default** | Add direct Dirichlet spectrum forward/backward/JVP to the consolidated `fmcw.py` / `fmcw.cu`; add stored output-domain config and result metadata; make range processing domain-aware; default to spectrum and keep beat explicit | 7, P5 | spectrum == normalized FFT(beat); primal/JVP/VJP; breaking-default/API evidence; frontend/window refusals; performance and memory budgets |
| **8 — governance/document closure** | Delete empty packages; close every P6 debt; regenerate owner ledgers; correct all living docs/examples; mark historical records; resolve Radar policy contradictions; run orphan/stale-path/compatibility/drift scans | F | debt inventory has zero open rows; living docs have zero stale claim; full default/GPU suites and every gate green |
| **A — adversarial audit** | Independent declared-scope audit after every phase: exact approved API breaks, behavior preservation everywhere else | each phase | no unresolved finding |

The phase order is load-bearing:

- governance gates land before deletion/movement so debt cannot disappear by
  path churn without being closed;
- public compatibility is deleted before architecture work so shims are not
  merged into permanent owners;
- the system/session layer moves before its downstream imports;
- Channel and path identity move together because the adapter produces the leg
  contract the composer consumes;
- native source paths move before the FMCW numerical addition so the new
  spectrum family lands directly in the final `fmcw.cu` owner;
- Phase F lands after every move-only phase so its default/output-domain change
  has a clean numerical diff and cannot be rationalized as file motion.

Parallel work is allowed only between disjoint concepts inside a phase:

- Phase 4: scattering, sensors, and frontend may be separate tasks;
- Phase 5: FMCW, OFDM, and pulsed moves may be separate tasks after the shared
  result contract is frozen;
- Phase 6: range/Doppler, angle, detection, and tracking may be separate tasks.

No two tasks may edit the same facade, manifest, or public snapshot. One
integration owner updates those files after the concept moves land.

---

## 7. Verification protocol for execution

No verification command was run while drafting this plan. The following is the
required future harness.

### 7.1 Baseline capture

Record:

```powershell
python -m pytest tests --collect-only -q
python -m pytest tests --gpu --collect-only -q
python ci/run_ci_tier.py quick
python ci/run_ci_tier.py cuda
python -m examples.single_point
```

Use the approved `witwin2` environment, an isolated writable `--basetemp`, the
pinned Core/Channel `PYTHONPATH`, and the pinned Channel developer override and
fingerprint. The exact environment belongs in the execution record.

Phase 1 is allowed to remove tests whose sole purpose is preserving a deleted
compatibility surface. Each removed node ID must map to a P6 debt row and a
deleted production symbol; no bulk test-file deletion is accepted without this
mapping. At Phase-1 exit, capture the new target node-id baseline. Phases 2-8
must not shrink that set except for a separately declared feature removal.

### 7.2 Per-phase invariants

Every phase proves:

1. collected test node IDs match the active baseline; every Phase-1 removal is
   individually accounted for against a deleted compatibility contract;
2. `ci/public-api-snapshot.json` changes only from an approved target-surface
   inventory, never as a reaction to a failing snapshot test;
3. every retained public export has one owner and current caller; no removed
   export, alias, overload or old default remains reachable;
4. only one Radar module names Channel;
5. the Radar import closure loads no Channel solver and no raw native module;
6. native symbol set, AD group, launch count and host-observation count are
   unchanged before Phase 7 except for the approved removal of legacy sensor
   compatibility modes; that exception preserves the Channel-sourced numerical
   slice and is governed by the execution amendments;
7. Torch-physics allowlist entries either move with their owner or are removed
   by paying the debt; none is added/reclassified to absorb a violation;
8. no `ctx.saved_tensors` read leaves its owning AD function;
9. no test oracle is imported by production;
10. the orphan-module gate reaches every production module;
11. no old internal path remains in code, manifests, docs, examples or tests;
12. the no-compatibility gate finds no shim, deprecated alias, adapter,
    old-name `__getattr__`, re-export facade or fallback;
13. the documentation-drift gate finds no stale claim in living documentation;
14. every P6 debt touched by the phase is closed with evidence rather than
    renamed, reclassified or moved to an allowlist.

### 7.3 Numerical invariants

Move-only phases require exact equality where the existing tests use exact
equality, and the existing tolerance unchanged everywhere else:

- Channel leg row count/order and stable IDs;
- composed path count/order, `pair_offsets`, total delay/rate and complex
  transfer;
- FMCW, OFDM and pulsed synthesis outputs;
- TDM slot phase;
- antenna-pattern weighting;
- frontend seeded realization;
- range/Doppler/angle exact-bin results;
- AD primal/JVP/VJP and refused tangents;
- row-validity zeroing and no-partial-result behavior.

The plan does not permit widening a tolerance, increasing a budget, changing a
seed, or regenerating a golden because a move failed.

### 7.4 Phase F FMCW acceptance

Phase F is judged against both an independent analytic oracle and the retained
beat implementation:

1. direct spectrum equals `torch.fft.fft(beat, norm="forward")` for a
   rectangular window at every tested bin, not merely at the peak;
2. fractional-bin targets reproduce the analytic Dirichlet kernel, including
   phase and sidelobes;
3. empty rows, dead rows, pair offsets and component partitions preserve exact
   zero/ordering behavior;
4. Channel carrier phase is conjugated exactly once and carrier-rate phase is
   present exactly once;
5. TDM slot timing matches the beat path for moving targets;
6. monostatic, bistatic, multi-site and multipath delays retain the compact
   round-trip convention, with no restored factor-of-two distance assumption;
7. forward-mode and reverse-mode derivatives agree with the beat-FFT reference
   and finite differences at the existing AD-capability tolerances;
8. default FMCW synthesis publishes `domain="range_spectrum"` and
   `range_bin`; explicit `"beat"` publishes `domain="beat"` and `sample`;
9. `range_profile` is an identity-plus-metadata operation on a direct spectrum,
   apart from the exact optional DC-bin zeroing, and never launches a second
   FFT;
10. non-rectangular windows and configured frontend stages refuse spectrum mode
    before a synthesis launch and name `"beat"` as the required explicit mode;
11. the direct path meets an accepted forward/backward memory and wall-time
    budget relative to beat synthesis plus FFT;
12. both modes use the same public FMCW owner, config, result type and
    processing facade; no legacy solver or fallback is reachable.

The old Dirichlet tests may be recovered as reference evidence, but they must be
rewritten over the current compact Channel-sourced path contract. Restoring a
test fixture that constructs tracer distances, legacy amplitudes or
`MimoPathCache` would validate the deleted architecture rather than the new
default.

### 7.5 Final gates

The final phase runs:

- Ruff over production, tests, tools, CI and scripts;
- native binding, raw native access, production dependency, Torch physics,
  test-oracle isolation, workflow policy and orphan-module gates;
- import-graph, single-definition, no-compatibility and documentation-surface
  gates, each with its planted-violation calibration tests;
- public API snapshot tests;
- default full suite;
- GPU full suite;
- `quick` and `cuda` CI tiers;
- extension boundary and wheel smoke;
- Core/Channel/Radar coexistence;
- documented examples;
- the governance inventory parser, which fails unless every row is closed and
  points to live evidence;
- an executable living-document snippet/symbol audit.

Release-only Linux, manylinux, multi-SASS and Stable-ABI evidence remains subject
to `phase10-deferred-release-matrix.md`; layout completion must not be described
as release completion.

---

## 8. Governance files that move with code

| Artifact | Required update |
| --- | --- |
| `ci/public-api-snapshot.json` | Replace the old snapshot with the ratified minimal target inventory, new definition targets and derived hashes; incompatible removals and signature/default changes are expected |
| `ci/architecture-manifest.json` | Canonical production modules, concepts, owners, allowed dependency edges, public facades and the sole Channel importer |
| `ci/check_architecture.py` | Read the architecture manifest and reject undeclared modules/edges, cycles, duplicate Channel importers and path-local exception lists |
| `ci/check_single_definition.py` | Reject duplicate public definition targets, alias owners and concept contracts defined in more than one production module |
| `ci/check_no_compatibility.py` | Reject deprecated aliases, `_REMOVED`/compatibility `__getattr__`, legacy adapters, old-path re-exports, signature-preserving wrappers and compatibility fallbacks |
| `ci/check_documentation_surface.py` | Resolve living-document paths, symbols and executable snippets against the new public inventory; reject unmarked historical claims |
| `ci/native-binding-manifest.json` | `python_owner`, `owner_module`, source paths and source digest; numerical owner/AD/launch fields unchanged |
| `ci/torch-physics-allowlist.json` | Exact module/function owner paths; remove `freeze_time_pattern_oracle` and any paid debt; never add or reclassify a row to make consolidation pass |
| `ci/check_orphan_modules.py` | Consume canonical graph roots from the architecture manifest; remove all `sigproc` and deleted-module entries |
| `ci/check_raw_native_access.py` | New `cuda/runtime.py` dispatcher owner and consumers |
| `ci/check_production_dependencies.py` | Production module inventory/path strings |
| `ci/check_test_oracle_isolation.py` | Production inventory and path references |
| `tests/test_phase4_import_boundary.py` | Remove its hand-maintained `SPIKE_MODULES`; test the canonical architecture manifest and sole Channel importer |
| `tests/test_phase4_binding_manifest.py` | Native Python owners |
| `tests/test_phase9_tape_non_leak.py` | Autograd owner paths and saved-tensor scan |
| `tests/support/ad_boundaries.py` | Boundary imports after concept collapse |
| `docs/dev/audit/radar-governance-debt-and-drift-inventory.md` | Close every P6 row with live evidence; zero unresolved or deferred rows at completion |
| `docs/dev/radar-ad-capability-matrix.md` | Owner paths and line references |
| `docs/dev/ad-tape-and-budget-ledger.md` | Tape owners, formulas and path references |
| `AGENTS.md` / `CLAUDE.md` | Replace deleted Scene/Tracer/Timeline/Dirichlet descriptions with the actual Core → Channel → Radar architecture, commands and ownership rules |
| build scripts / hatch hook / wheel smoke | Flattened CUDA source paths and rebuilt prebuilt identity |
| README / FEATURE_LIST / PERFORMANCE / pipeline guide | Only current target paths and APIs; no stale module, compatibility-adapter or deleted-Scene claims |
| `docs/dev/plans/phase10-deferred-release-matrix.md` | Resolve the Stable-ABI versus strict Torch/CUDA identity contradiction by an explicit ADR and make its release claim internally consistent |

Path-keyed artifacts are updated in the same commit as the owning move. A later
“manifest cleanup” commit is not acceptable because the intermediate commit
would describe a false architecture.

API-reset phases and Phase F intentionally change the public snapshot. Phase F
is also the declared numerical exception to “path updates only”:

- the public snapshot records the new stored FMCW output-domain field and any
  affected method/config signatures;
- the native binding manifest adds the spectrum primal/backward/JVP roles and
  their measured launch counts, while retaining the beat roles;
- the capability record states that both domains support the same approved
  first-order AD inputs;
- the processing metadata contract adds the domain and fast-axis name;
- the breaking-change note states that the default FMCW cube changed from
  samples to range bins and shows how to request the optional beat product
  explicitly; it does not preserve the old default or an old call signature.

---

## 9. Adversarial audit brief

Each phase auditor tries to refute the phase's declared scope. Layout phases
must be behavior-preserving; API-reset and Phase-F changes must be exactly the
approved breaks and no more. The auditor looks for:

1. validation or an exact error message dropped during merge;
2. module-level initialization, registration, cache construction, or lazy import
   firing at a different time;
3. a dataclass/enum/function being recreated so identity or `isinstance`
   behavior changes;
4. a new circular import hidden by a function-local import;
5. a public name, method, signature or default changing without an approved
   target-inventory decision;
6. a test node disappearing without a one-to-one mapping to a deleted contract;
7. an AD owner losing `setup_context`, `backward`, `jvp`, or a refusal;
8. a Channel import appearing outside `channel.py`;
9. a host tensor observation entering a per-frame or per-path route;
10. path, pair, site, component or slot ordering changing;
11. RNG draw order changing in the frontend;
12. an internal or public compatibility shim, alias, fallback, deprecated path
    or duplicate owner surviving or being added;
13. a native source move changing the registered symbol set or build identity
    without the required rebuild;
14. a direct FMCW spectrum receiving a second FFT in processing;
15. a time-domain frontend stage being applied to a range spectrum;
16. a non-rectangular window being silently ignored or approximated in spectrum
    mode;
17. the direct and beat modes counting Channel carrier phase, path loss, antenna
    gain or TDM motion differently;
18. an old root re-export recreating ambiguous ownership after its source module
    was consolidated;
19. a governance debt being renamed, reclassified, moved to an allowlist or
    hidden in a new manifest instead of closed;
20. a stale living document, executable example, owner path or line-number claim
    surviving because it was not part of the code diff;
21. a historical plan being presented as current guidance without an explicit
    historical/superseded marker;
22. the public API snapshot being edited reactively to accept an accidental
    implementation surface rather than generated from the ratified inventory.

A finding names the old and new owner, the concrete input that differs, and the
test or gate that should catch it. General style preferences are not audit
findings.

---

## 10. Non-goals

- No propagation physics change.
- No Channel solver or compact-consumer change.
- No native arithmetic change in the layout phases. Phase F adds only the
  approved FMCW Dirichlet-spectrum family; it does not authorize unrelated
  reduction-order, launch-geometry, fast-math or ABI redesign.
- No new waveform, scatter law, antenna model, frontend stage, detector or
  tracker. Direct spectrum and beat are two output domains of the existing FMCW
  waveform.
- No performance “optimization” mixed with code motion.
- Public API redesign is in scope where concept ownership, debt removal or the
  FMCW domain contract requires it; accidental or unrecorded API drift is not.
- No Core/Channel/Radar ownership transfer for SMPL in this plan.
- No compatibility preservation for old public names, signatures, defaults,
  output conventions or internal submodule paths.
- No reopening of deleted Phase-11 routes.
- No claim that a locally green layout is release-ready.

---

## 11. Open decisions

1. **Ratify the target tree.** In particular, confirm that top-level
   `channel.py` is preferred over keeping a one-file `propagation/` package
   that has no independent concept axis.
2. **Ratify the minimal root API inventory.** The proposal keeps only
   Radar-owned primary construction/simulation entry points at `witwin.radar`;
   Core types, processing, synthesis, SMPL, capability and deployment details
   are imported from their owner modules. This choice does not retain old
   re-exports.
3. **Choose the immutable Core/Channel baseline.** The current sibling
   worktrees are not suitable while their consolidations are uncommitted.
4. **Confirm `processing/` at six files.** The alternative is a single
   `processing.py` around 4,000 lines; this plan rejects that because
   range/Doppler, angle, detection and tracking are genuine independent change
   axes.
5. **Confirm `synthesis/` at five files.** The alternative is one
   `synthesis.py` around 3,300 lines; this plan keeps one file per waveform
   because their specs, validation contracts, native operators and AD tests
   move together.
6. **Decide whether `capabilities.py` and `deployment.py` remain separate.**
   They total about 445 lines but answer different questions: what Radar can do
   versus what artifact/runtime is loaded. The proposal keeps both.
7. **Decide the future SMPL owner separately.** The temporary top-level
   `smpl.py` is layout cleanup, not an ownership endorsement.
8. **Ratify the Phase-10 ABI policy outcome.** Either define and test a real
   cross-version Stable-ABI compatibility rule or delete that release claim and
   retain strict runtime identity matching. A contradictory matrix is not an
   acceptable open-ended deferral.

---

## 12. Definition of done

This plan is complete only when:

- the accepted target tree is present and empty packages are deleted;
- Python file count is materially reduced without making the count itself a
  gate;
- every surviving module is a concept owner or a deliberate public facade;
- only `processing/`, `synthesis/`, and `cuda/` remain as substantive
  subpackages;
- the Channel import boundary still has cardinality one;
- FMCW defaults to a directly generated, rectangular, normalized Dirichlet
  range spectrum with explicit domain/axis metadata;
- the retained time-domain beat path is reachable only by the explicit
  `fmcw_output_domain="beat"` setting, and spectrum mode never silently falls
  back to it;
- direct spectrum and normalized FFT of beat agree across primal, JVP, VJP,
  motion, TDM and multipath acceptance cases;
- spectrum mode cannot receive a second range FFT, a non-rectangular window or
  a time-domain frontend stage;
- the ratified minimal public inventory is the only public surface: removed
  names, old signatures/defaults, `sigproc`, adapters, root Core/convenience
  re-exports and `_REMOVED` behavior are unreachable;
- `ci/public-api-snapshot.json` is generated from that target inventory and
  guards only against subsequent accidental drift;
- every pre-existing native numerical family, symbol, AD role, launch count and
  host observation is unchanged; the only additions are the Phase F spectrum
  primal/backward/JVP entries and their recorded budgets;
- all path-keyed governance artifacts agree with the new owners and consume one
  canonical architecture manifest rather than parallel hand-maintained lists;
- every removed test node maps to a deleted compatibility contract, and the
  post-reset default/GPU node sets have not silently shrunk;
- default and GPU suites, quick/cuda tiers, static gates, wheel/coexistence
  checks and examples pass against pinned clean dependencies;
- the no-compatibility, import-graph, single-definition and documentation-surface
  gates pass, including their planted-violation calibration tests;
- the P6 governance inventory has zero open/deferred rows, every closed row has
  live evidence, and no debt was hidden by renaming, reclassification or a new
  allowlist;
- `freeze_time_pattern_oracle` and every other recorded Torch-policy debt is
  removed rather than grandfathered;
- `AGENTS.md`, `CLAUDE.md`, README, FEATURE_LIST, PERFORMANCE, pipeline guides
  and executable examples describe the new owners and APIs; historical records
  are explicitly marked and are not treated as living guidance;
- the Phase-10 Stable-ABI/runtime-identity policy is internally consistent and
  executable, or the unsupported release claim has been removed;
- no old public or internal import path, compatibility shim, shadow route,
  fallback, orphan module, stale owner/line reference or stale living document
  remains;
- an adversarial audit finds no behavior change hidden inside the consolidation;
- the completion record names exact commits, native fingerprints, environment,
  commands and outcomes.
