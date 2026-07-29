# Phase 10 deferred release matrix

Status: accepted, 2026-07-27.

Everything in Phase 10 that cannot be executed on the development machine is
listed here. Each entry says what is deferred, why it cannot run locally, the
exact command or workflow that will execute it, the evidence that run will
produce, and who owns it.

## These are deferrals, not gaps

The owner directive for this phase is explicit: **Phase 10 must not wait for a
real GitHub Actions run.** The deliverable for CI configuration is the
configuration itself, validated as far as it can be locally - YAML syntax and
schema, job and matrix logic reviewed against
`GITHUB_ACTIONS_PREBUILD_MATRIX.md`, and the machine-checkable invariants
asserted by `ci/check_workflow_policy.py`. Wheel builds, fresh installs,
coexistence and import isolation run locally for the cells this machine can
build; the Linux and `manylinux_2_28` cells are configured, reviewed, and
recorded here rather than executed.

So: **Phase 11 proceeds without waiting on any entry below.** None of them
blocks it. What each one does is name the evidence that a later real CI run
must produce before a release claim is made, so that "we never ran it" cannot
later be mistaken for "it passed".

The one thing this register must never become is a place to move an inconvenient requirement. D6/P3 was a real policy contradiction when recorded; it is now resolved by the executable exact-runtime-identity policy described below. The remaining rows are still deferred evidence, not passing claims.

## Deferral register

| id | deferral | why it cannot run here | executed by | evidence it will produce | owner |
|---|---|---|---|---|---|
| D1 | Linux / `manylinux_2_28` wheels for Core, Channel and Radar | Docker 29.6.1 and WSL Ubuntu-24.04 exist on this machine, but a compliant run means the hosted-runner environment the release gates describe, not a local container: the manylinux image, the CUDA 12.8 subset installed inside it, and cibuildwheel's `/host` mount semantics | `publish-witwin-radar.yml` (scope `full`) and `publish-witwin-channel.yml` (scope `full`) | a `manylinux_2_28_x86_64` wheel per package, `auditwheel` repair output, and the Linux half of the architecture verifier | platform release owner |
| D2 | SM87 **runtime** validation | no Orin or Jetson hardware is connected; the local GPU is sm_120. The policy already separates "an `sm_87` SASS image is present", which any binary can be checked for, from runtime validation, which it says must not be claimed until such a runner exists | a connected SM87 runner, running the radar `cuda` tier | a GPU suite result on sm_87 hardware | platform release owner |
| D3 | the full ten-image SASS set | local builds use `-DCMAKE_CUDA_ARCHITECTURES=120-real` / `WITWIN_CUDA_GENCODE_ARCHES=12.0`, because compiling ten architectures locally costs the same hours it costs a runner and produces an artifact this machine cannot run more of | `publish-witwin-radar.yml` scope `full`, then `scripts/verify_cuda_binary_arches.py` with its default expectation | `Verified CUDA architectures ...: SASS 70,75,80,86,87,89,90,100,101,120 plus sm_120 PTX` | platform release owner |
| D4 | the "clean locked build" claim | local nvcc is 12.9.41 against the locked 12.8.1. Every locally produced binary is therefore stamped `build_type="developer"` by `scripts/build_radar_cuda_prebuilt.py`, and only a published release run passes `--release` | `publish-witwin-radar.yml` on a `release: published` event | a wheel whose `build_info()["build_type"] == "release"` and whose `cuda_compiler_version` is 12.8.1 | platform release owner |
| D5 | the Stage-I release full build, still pending from that stage | folded here under the same owner directive rather than tracked separately, because it is the same run as D1/D3/D4 | `publish-witwin-channel.yml` scope `full` | the two-wheel Channel artifact set with its `validate-wheels` job green | Channel release owner |

### D4, restated after the adversarial loader audit: the sidecar is self-signed

`build_type` is a DECLARATION inside the identity record, not proof of
provenance. The audit re-signed a tampered record - flipping `build_type` to
`release`, recomputing both sidecars, pinning the new fingerprint - and it
validated and loaded. That crosses no security boundary: anyone who can rewrite
the sidecar can rewrite the loader beside it, and the digests still cover the
defect class the record exists for, which is a stale, swapped or mismatched
artifact. What it does mean is that a release claim may never be made by
reading `build_type` out of an artifact. It comes from the release pipeline
that produced it, which is precisely what D4 defers.

### Closed since it was first recorded

The `witwin-radar[channel]` dependency-closure dry run was deferred at one point
because `witwin-channel` was on no index this machine could reach. It is
**closed**, and not with a stub: the real 8.4 MB Channel wheel built in this
phase is what `pip install --dry-run --report` resolves against, and the
resolution names `witwin`, `witwin-channel`, `witwin-radar` and no
ray-tracing distribution. Do not carry it forward into D1 or D5.

## Policy deviations

Recorded against `GITHUB_ACTIONS_PREBUILD_MATRIX.md`, policy version 6.

### P1 - trigger boundary: FIXED for Radar, recorded for Core

`publish-witwin-radar.yml` used to run on every push to `main`, `master` and
`codex/**`, and on every pull request, building the complete ten-image CUDA
matrix on both platforms each time. The policy reserves that work for a
published release, an explicit `workflow_dispatch`, or the exact `run-ci` label.
The workflow now carries `release: [published]`, `workflow_dispatch` with a
`scope` input, and `pull_request: types: [labeled]` behind a single
`github.event.label.name == 'run-ci'` guard on the one entry job. Every other
job reaches that job through `needs`, so the guard is the whole boundary.
`ci/check_workflow_policy.py` fails if a `push` or `schedule` trigger returns.

`core/.github/workflows/publish-witwin.yml` has the same violation and Core is
read-only this phase. **Recorded, not fixed.** Owner: Core maintainer. The same
applies to Core's build hook, which returns silently when no prebuilt exists and
lets hatchling emit a valid-looking pure-Python wheel; Radar's hook now raises
instead.

### P2 - grouped `--generate-code` families: cost, not correctness

Channel passes the policy's five grouped `--generate-code` families; Core and
Radar still expand one `-gencode` per architecture through
`_cuda_gencode_flags`. Their verifiers still prove complete SASS coverage, so
the artifacts are equally correct - the deviation costs compile time, which the
RayD measurements put at roughly 43% of the Windows native build step.

Radar may adopt grouped families. That is a build-shape change with its own
measurement, not an architecture-cleanup edit, so it is not made here. Core is
read-only: recorded only. Owner: whoever next opens the Radar build for a
performance change.

### P3 - cross-Torch release claim: RESOLVED by exact runtime identity

The old policy combined two incompatible statements: one binary should load
across several Torch/CUDA versions, while the native loader required exact
Torch, CUDA, C++ ABI, and platform identity. The previous workflow treated an
expected loader refusal as a successful compatibility cell. That branch could
never prove the advertised compatibility.

Radar now chooses the strict identity contract. ci/release-policy.json sets
stable_abi_cross_torch_claim to false,

untime_identity_policy to xact_torch_cuda_and_abi_identity, and
xpected_loader_refusal_is_release_success to false. The release matrix
therefore exercises Torch 2.10/CUDA 12.8 across CPython 3.10-3.14 on Linux and
Windows. Every cell must load the packaged binary, match the recorded Torch
identity, and avoid JIT. A refusal fails the cell.

This resolves former D6: there is no deferred cross-Torch evidence and no Radar
release may advertise it. Adding another Torch/CUDA line requires a separately
built artifact, an explicit identity/indexing policy, and a real successful
load matrix.

Owner: Radar release-policy owner. Executable enforcement:
ci/check_release_claims.py and ci/check_workflow_policy.py.
### P4 - `gpu-regression.yml` uses a self-hosted runner

The policy states that GitHub-hosted runners are mandatory and `self-hosted`
labels are forbidden. `gpu-regression.yml` runs on `[self-hosted, windows, x64,
gpu]` because no GitHub-hosted runner offers a CUDA device, and the radar GPU
suite is the phase's main numerical evidence.

It is manually dispatched only, produces no wheel, and publishes nothing, so it
is outside the paid-build and release-artifact concerns the policy rule protects.
`ci/check_workflow_policy.py` freezes the exception **by filename with its
reason**: any other radar workflow that grows a `self-hosted` runner fails the
gate. Owner: platform policy owner, to either grant the exception in the policy
document or provide a hosted GPU lane.

## What was executed locally instead

Recorded so the deferrals above are read against what is already proven, not
against nothing:

- three Windows wheels built and installed together into one disposable target;
- nine coexistence and import-isolation scenarios, including
  `import witwin.core` loading no Channel module, no `rayd`, no `drjit`, and
  leaving CUDA uninitialized;
- both native extensions loaded in one process with disjoint dispatcher
  namespaces and one numerical compute crossing the boundary;
- the packaged loader refusing to JIT with its binary hidden, with
  `torch.utils.cpp_extension` never imported;
- the architecture verifier correctly FAILING a local `120-real` build against
  the release expectation, which is what makes D3 a deferral rather than a
  claim;
- `ci/check_workflow_policy.py` passing on the checked-in workflow and failing
  on seven separately mutated copies of it.
