# Radar governance debt and documentation drift inventory

Status: active; zero open rows is a completion gate.

Recorded: 2026-07-28.

Every row is closed only by a concrete commit plus executable evidence. Moving,
renaming, reclassifying, grandfathering, or adding an allowlist is not closure.

| ID | Category | Concrete owner | Debt | Phase | Falsifier | Status | Closing evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GOV-001 | compatibility | `witwin/radar/__init__.py::_REMOVED`, compatibility `__getattr__` | deleted names remain behavior-bearing API | 1 | `ci/check_no_compatibility.py` | closed | compatibility gate passes; root has no removed-name fallback |
| GOV-002 | compatibility | `witwin/radar/sigproc/**`, `witwin/radar/processing/adapters.py`, `tests/sigproc/**`, `tests/processing/test_adapters.py`, `tests/goldens/legacy_sigproc.pt` | pre-Phase-8 API and numerical conventions remain executable | 1 | absence scan and post-reset collection | closed | retired sigproc/adapters/golden files absent; current processing suite retained |
| GOV-003 | ownership | `witwin/radar/__init__.py` | Radar root re-exports Core world types and widens `Geometry` with `SMPLBody` | 1 | public API snapshot and single-definition gate | closed | public snapshot pins root to Radar and RadarConfig only |
| GOV-004 | compatibility | `SynthesisPathBatch.from_real_amplitudes`; sensor legacy modes/projection; processing tensor/NumPy overloads | compatibility-only behavior survives outside the named adapter package | 1/4/6 | semantic caller inventory plus compatibility gate | closed | compatibility gate passes; real-amplitude, legacy sensor and bare-tensor success paths absent |
| GOV-005 | Torch policy | `witwin/radar/sensors.py::evaluate_antenna_pattern_vectors` | `freeze_time_pattern_oracle` is explicitly recorded debt | 4 | Torch-physics allowlist equality with no debt category | closed | Torch policy now records 14 accepted expressions and no vector-pattern oracle entry |
| GOV-006 | architecture | Radar CI | no standalone import-layer graph gate | G | planted forbidden edge and second Channel importer | closed | `ci/check_architecture.py`; second-importer and false-string-edge tests |
| GOV-007 | governance duplication | `ci/check_orphan_modules.py::ENTRY_POINTS`, `tests/test_phase4_import_boundary.py::SPIKE_MODULES`, owner lists | architecture inventories drift independently | G/8 | all gates consume one manifest | closed | import-boundary hot paths and all owner gates consume ci/architecture-manifest.json |
| GOV-008 | documentation | `AGENTS.md`, `CLAUDE.md` | guidance describes deleted Scene/Tracer/Timeline/Dirichlet architecture | 8 | documentation-surface gate | closed | rewritten to concept-axis/current FMCW policy; `python ci/check_documentation_surface.py` |
| GOV-009 | documentation | README, FEATURE_LIST, pipeline/AD docs and examples | living documentation presents retired processing/adapters and deleted Scene behavior as current | 1/8 | documentation-surface gate | closed | current living surface and historical-plan exemptions; `python ci/check_documentation_surface.py` |
| GOV-010 | documentation governance | current API/AD/ledger documents | fragile line-number owners drift after layout changes | 8 | current-owner symbol resolver | closed | AD matrix and ledger use `path.py::symbol`; AST resolver in `ci/check_documentation_surface.py`; 27 governance/doc tests passed |
| GOV-011 | release policy | `docs/dev/plans/phase10-deferred-release-matrix.md` | cross-Torch claim conflicts with strict Torch/CUDA runtime identity | 8 | accepted ADR and executable matrix | closed | exact-runtime-identity resolution in `ci/release-policy.json`; `python ci/check_workflow_policy.py` |
| GOV-012 | policy exceptions | workflow policy and deferred release rows | exceptions can be mistaken for satisfied evidence | G/8 | every row has resolution, external blocker, or expiring policy owner | closed | D1-D5 retain executor/evidence/owner; P3 resolved; named P4 exception; `python ci/check_workflow_policy.py` |
| GOV-013 | compatibility governance | Radar CI | no machine gate prevents compatibility shims from regrowing | G | planted alias, adapter, `_REMOVED`, fallback and old-path re-export | closed | `ci/check_no_compatibility.py`; planted Python/native shim test |
| GOV-014 | documentation governance | Radar CI | no machine gate checks living docs against current paths, symbols and defaults | G/8 | planted stale path/symbol/default | closed | `ci/check_documentation_surface.py`; planted retired/missing path test |
| GOV-015 | baseline integrity | Core/Channel source worktrees | mutable siblings must not define the execution baseline | 0/8 | clean pinned snapshots collect the full suite | closed | clean Core `7791ce2` and Channel `c07b489` exports; 1579 tests collected |
| GOV-016 | test governance | Phase-1 compatibility tests | deleting tests could hide non-compatibility coverage | 1 | one-to-one removed-file disposition below and post-reset node baseline | closed | deletion disposition retained below; current suite collection and processing coverage are executable |
| GOV-017 | configuration ownership | `radar.py::RadarConfig`, `config.py::RadarSystemConfig`, waveform blocks and `FmcwSpec.from_radar_config` | configuration and SI conversion have two owners | 2/F | one config owner and one config-to-spec conversion | closed | config.py removed; radar.py owns RadarConfig and waveform-spec conversion |
| GOV-018 | axes ownership | `config.py::RadarAxes`, Radar convenience properties, `processing.axes::ProcessingAxes` | physical axes have two owners | 2/6 | only `ProcessingAxes` remains | closed | RadarAxes removed; ProcessingAxes is the sole physical-axis record |
| GOV-019 | typed handoff | `RadarSimulationResult` to `ProcessingCube` | formal pipeline needs a bare tensor plus axes | 2/6 | typed simulation-frame processing entry and no bare-tensor overload | closed | range_profile requires ProcessingCube and refuses bare tensors |
| GOV-020 | native compatibility | sensor weight mode flags, `PolarizationSpec`, `from_real_amplitudes` | deleted real-amplitude route remains in Python and native ABI | 1/7 | Channel-sourced parity with retired flags/schemas absent | closed | native manifest has 28 current operators; real-amplitude schemas and sensor mode flags absent |
| GOV-021 | API governance | public facade packages | old snapshot omits public owner facades and target inventory | G/1 | generated snapshot equals `ci/public-api-manifest.json` | closed | generated schema-v2 snapshot covers every symbol in all 10 public facades |
| GOV-022 | API semantics | `Radar.simulate::slow_time_mode` | public argument has one accepted value and no choice | 2 | argument absent; driver fixes the mode internally | closed | Radar.simulate signature has no slow_time_mode parameter |
| GOV-023 | configuration semantics | stored antenna pattern versus `simulate(antenna_pattern=None)` | `None` means both default dipole and no pattern | 2/4 | one owner and one meaning | closed | Radar sensor configuration is the only pattern owner; every simulation applies it |
| GOV-024 | CI coverage | quality/GPU workflows and required Channel imports | required main chain can be skipped while CI is green | G/8 | pinned Channel installed/fingerprinted and required-chain skip budget is zero | closed | Channel extra, consumed build fingerprint, zero skip budget; `python ci/check_required_channel_coverage.py` |
| GOV-025 | workflow drift | `.github/workflows/gpu-regression.yml` | workflow invokes a deleted benchmark | G | workflow entry existence gate and replacement benchmark | closed | processing pipeline benchmark substituted; `python ci/check_workflow_references.py` |
| GOV-026 | release evidence | README/FEATURE_LIST/workflow/deferred matrix | cross-Torch and manylinux claims contradict executable policy | 8 | one consistent ADR, policy and publish gate | closed | exact identity plus `manylinux_2_28`; `python ci/check_release_claims.py`; `python ci/check_workflow_policy.py` |
| GOV-027 | geometry compatibility | `geometry/smpl.py` legacy Scene docstring and epsilon gradient workaround | deleted Scene route and compatibility workaround remain production behavior | 3/8 | owner decision, current docs, targeted derivative evidence | closed | stale Scene prose and shape-gradient epsilon workaround removed; SMPL boundary docs are current |

## Phase-1 test deletion disposition

The following tests exist solely to preserve the API being deleted:

- `tests/sigproc/**`: tests the removed `witwin.radar.sigproc` surface;
- `tests/processing/test_adapters.py`: bitwise legacy-adapter golden;
- compatibility-only assertions in `tests/processing/test_cutover.py`;
- removed-name replacement-message assertions in
  `tests/test_phase5_removed_entry_points.py`,
  `tests/test_phase11_repository_gates.py`, and
  `tests/test_public_api_snapshot.py`.

Numerical processing coverage under `tests/processing/` remains. Any deleted
test that also covers a current owner must first be rewritten against that
owner; deleting the compatibility import is not permission to lose the
underlying numerical invariant.
