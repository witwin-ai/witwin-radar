# R-ADR-026: Generated governance records, trimmed gates, and the packaged-binary identity

Status: Accepted (2026-11-12, witwin-radar 0.5.0)

## Context

By 0.4.1 the governance surface (docs, `ci/`, `tools/`) was larger than the product, 1.30 lines to 1. Adding one
native AD family needed hand edits in seven places outside its code and tests: a 56-line block of the binding
manifest, count literals in four test files (operators 38 -> 41, AD rows 7 -> 9, Functions 11 -> 12 and 2 -> 3 per
module, tape reads 22 -> 24) and two ledger documents. The full GPU suite of that release failed on exactly those
counts. About 4,200 lines of tests asserted only source text, layout, counts or milestone history, and took most of
the CPU tier's time. The documentation gate failed on every Linux clone because a ledger named the Windows-only
`_radar_native.pyd`. Finished plans and phase records sat beside living documents, some still describing
themselves as active. The packaging metadata promised more than the loader accepted: `torch>=2.10` against an
exact-version identity check, no `.gitattributes` against a source fingerprint over raw bytes, and no check that the
current device can run the binary's architecture list.

## Decision

### 1. What governance tests and gates are for

Behaviour tests and the public-API snapshot are never weakened. A test or gate that only pins source text, a
layout, line numbers, a hash, a count or milestone history is replaced by a check derived from the code, or
deleted. A count that must hold is stated as the property it stands for ("every registered operator has a backward
or a declared reason not to", "every `Function` reads its tape in `backward` and `jvp`"), and the scan that checks
it is kept non-vacuous by a second, independent answer rather than by a frozen number.

Absolute wall-clock budgets are machine-dependent; they run only with `pytest --gpu --benchmark`. Launch,
host-observation, synchronization, tape-byte and peak-allocation budgets stay in the GPU suite.

### 2. The native binding manifest is generated

`ci/native-binding-manifest.json` is written by `python ci/check_native_bindings.py --write`. Every operator row
is derived: the symbol from `m.def` in `witwin/radar/cuda/extension.cpp`, its translation unit from the one
`m.impl` that implements it, its AD group and role from the `_forward`/`_backward`/`_jvp` suffix, and its Python
owner and calling functions from the production attribute reads. Two sections cannot be derived and are written
by hand: `no_backward` (a forward-only AD group, with the reason its derivative is refused) and `error_owners`.

The gate fails when the committed manifest is stale, when a declaration has no implementation, when an AD group has
one companion but not the other, when a forward-only group is not declared, when a `no_backward` entry is stale,
and when no production module calls an operator. The hand-transcribed `launches`, `fused_stages`,
`host_observations`, `contract_test`, `numerical_owner`, `family` and `end_to_end_caller` columns are gone: the
budget tests measure launches and host observations, and a second, unmeasured copy of those numbers is what had
to be edited by hand. `scripts/build_radar_cuda_prebuilt.py` refuses a manifest that disagrees with
`extension.cpp`, so a binary's recorded `operator_symbols` cannot omit a new family.

### 3. Gates kept, merged and deleted

Kept: the public-API manifest and snapshot, architecture, orphan modules, duplicate code, production
dependencies, test-oracle isolation, extension boundary, Channel identity, wheel and coexistence smoke, and the CI
tier runner. The single dispatcher-owner rule of the former `check_raw_native_access` is now part of
`ci/check_architecture.py` (`dispatcher_owner` in `ci/architecture-manifest.json`): only
`witwin.radar.cuda.runtime` may name `torch.ops` or `torch.utils.cpp_extension`.

Merged: the workflow-policy, required-Channel-coverage, release-claims and workflow-reference gates are one lint,
`ci/check_workflows.py`. It also checks that `pyproject.toml` pins the Torch version in
`ci/release-policy.json`.

Deleted: the Torch-physics allowlist (occurrence counts and a frozen SHA-256 over AST matches), the
no-compatibility gate (the consolidation is complete; `AGENTS.md` still forbids compatibility modules), and the
test files that asserted source text, layout or plan history. Their behavioural halves moved to behaviour test
files. The rule that hot-path numerics are native is enforced by the launch-count budgets and by review, not by
an AST allowlist.

### 4. Documents

Living documents are listed in `ci/documentation-manifest.json`. Finished plans, phase records, the closed
governance-debt register and superseded or retracted performance sections live in `docs/dev/archive/`, each
labelled historical and indexed by `docs/dev/archive/README.md`; the dated reports in `docs/dev/audit/` are
historical as well. The documentation gate checks every path a living document names, except a path the
repository's `.gitignore` excludes: a build or run product's existence depends on the machine, not on the tree.
Evidence a living document relies on is committed beside it; a pointer into the gitignored `output/` run directory
is not evidence and is not kept in a living document.

### 5. The packaged binary's identity

- **Line endings.** `.gitattributes` checks every text file out with LF endings. The source fingerprint stays a
  hash of raw bytes - which is what the wheel smoke and the loader both compute - and the bytes are now the same on
  every checkout. Normalizing line endings inside the digest was rejected: it would need two implementations kept
  in step and would make the fingerprint disagree with the bytes actually compiled.
- **Torch.** The loader accepts a packaged binary only on the exact Torch version it was built against, so
  `pyproject.toml` requires `torch==2.10.0` rather than `torch>=2.10`.
- **Device architecture.** Before `torch.ops.load_library`, the loader refuses a binary whose recorded
  `cuda_architectures` cannot run on the current CUDA device, naming the device, its `sm_XY` and the
  `TORCH_CUDA_ARCH_LIST` value that would fix it. The rule is CUDA binary compatibility: SASS for `sm_XY` runs on
  the same major version with minor at least `Y`; PTX for `compute_XY` runs on any device of capability `XY` or
  higher.
- **The `channel` extra** carries `python_version < '3.14'`, because Channel's wheels cover CPython 3.10-3.13;
  Radar's own Python matrix is unchanged.

## Consequences

- Adding a native AD family edits its sources, its Python owner and its tests, then regenerates the manifest. No
  count, ledger row or frozen inventory changes by hand.
- `python ci/run_ci_tier.py quick` passes on a clean Linux checkout without a GPU, as far as the gates are
  concerned.
- Deleting the processing-surface freeze test removed its frozen list of Channel capability fields; Channel
  consistency is still checked by the mirrored rows of `docs/dev/radar-ad-capability-matrix.md` and by
  `tests/test_import_boundary.py::test_the_consumer_contract_is_the_version_this_adapter_was_built_against`.

## Verification

```bash
python ci/check_native_bindings.py
python ci/check_workflows.py
python ci/check_architecture.py
python ci/check_documentation_surface.py
pytest tests/test_native_binding_manifest.py tests/test_ci_gates.py tests/test_native_loader_contract.py --gpu -q
```
