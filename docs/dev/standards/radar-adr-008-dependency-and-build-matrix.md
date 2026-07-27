# R-ADR-008: Dependency, Python/Torch/CUDA matrix, and the provisional pin

Status: Accepted (Phase 4), follow-ups closed (Phase 10)

## Context

Radar now optionally depends on Channel. Channel's release artifacts were still
building when Phase 4 started, so the intended pin to released wheels could not
be applied.

## Decision

### Declared dependencies

- `witwin>=0.4,<0.5` (Stage-I Core) remains a hard dependency.
- Channel becomes an OPTIONAL extra, `witwin-radar[channel]`. Radar must remain
  installable and importable without it.
- Torch `>=2.10`, with the Stable ABI target pinned at
  `TORCH_TARGET_VERSION=0x020a000000000000`. Official LibTorch Stable ABI
  coverage starts at Torch 2.10; earlier versions are not labelled Stable ABI.
- Python `>=3.10,<3.15`.

### The optional dependency has to be real

`witwin/radar/propagation/__init__.py` deliberately does not import the adapter,
so `import witwin.radar.propagation` works without Channel installed. Every test
that imports Channel carries a module-scope `pytest.importorskip`. The ubuntu CPU
quality job therefore stays green without adding the dependency.

### Provisional consumption: an owner-approved deviation

**Phase 4 consumes Channel and Core from SOURCE CHECKOUTS, not pinned release
wheels.** This is a deviation from the intended Phase-4 item 1 and is recorded as
provisional in this ADR and in the docstring of every affected test.

The follow-up, not done in Phase 4:

1. Pin `witwin-channel` to a released version in the `channel` extra.
2. Add a required-consumer CI job that installs the pinned artifacts and runs the
   Phase-4 tests against them.

### Both follow-ups landed in Phase 10

**Follow-up 1 is closed and verified locally.** Until Phase 10 the `channel`
extra did not exist at all: `pyproject.toml` declared only `dev`, so
"Radar does not require Channel" was true by OMISSION rather than by
constraint, and this ADR described an extra nobody could install. The extra now
exists and pins `witwin-channel>=0.4,<0.5` - a range rather than an open
dependency, because Channel's propagation consumer contract is versioned and
Radar reads it, so a new minor Channel is a contract review and not an
automatic upgrade.

It is verified rather than declared. `ci/coexistence_smoke.py` scenario H runs
`pip install --dry-run --report` for `witwin-radar[channel]` against the real
Channel wheel built in this phase, and the resolution names `witwin`,
`witwin-channel`, `witwin-radar` and no ray-tracing distribution. That is the
same statement as acceptance criterion A8, made by resolution rather than by
absence.

**Follow-up 2 is landed as CONFIGURATION.** `publish-witwin-radar.yml` carries a
consumer job that installs `witwin-radar[channel]` on a CPU Linux runner,
asserts that no `rayd*`, `drjit*`, `mitsuba*` or `sionna*` distribution is
resolved and that `witwin-channel` IS, then runs the import-isolation subset.
The job's REMOTE execution is a named deferral (D1 in
`docs/dev/plans/phase10-deferred-release-matrix.md`) under the Phase-10 owner
directive, which makes the workflow configuration the deliverable and defers
hosted-runner runs. It was validated locally: the YAML parses, its embedded
Python programs parse, and `ci/check_workflow_policy.py` asserts the job's
presence and is proven to fail when it is removed.

So the Phase-4 closing sentence - "until both land, no claim about
released-artifact compatibility is supported by this work" - is **retired**.
What replaces it is narrower and true: the dependency closure of
`witwin-radar[channel]` is checked against real built wheels, and the
consumer job that will check it against PUBLISHED wheels is configured and
gated, with its first run recorded as D1 rather than as a gap.

### Loading

SUPERSEDED by R-ADR-019 (Phase 10). This section stated the intent - the
packaged prebuilt is the normal load source, a developer override must be
explicit, nothing silently loads a stale global extension - while the Phase-4
code still fell through to a just-in-time build on any packaged failure and
validated nothing but eight operator names. R-ADR-019 is the contract the loader
now implements, and it names the three developer-override variables, the
`WITWIN_RADAR_NATIVE_BUILD=1` gate on the compiler, and the sidecar identity
chain.

## Acceptance evidence

- `ci/coexistence_smoke.py` scenario H (the `witwin-radar[channel]` dependency
  closure resolves no ray-tracing distribution), Phase 10
- `ci/check_workflow_policy.py` (the consumer job is present, and the gate is
  proven to fail when it is deleted), Phase 10
- `tests/test_phase10_wheel_packaging.py` (the `channel` extra exists and pins
  the versioned range), Phase 10
- `tests/test_phase4_contracts.py::test_propagation_package_does_not_import_the_channel_adapter`
- `tests/test_phase4_import_boundary.py::test_the_propagation_package_alone_does_not_require_channel`
- `tests/test_phase4_import_boundary.py::test_synthesis_scattering_and_paths_do_not_require_channel`
- `tests/test_phase4_binding_manifest.py::test_the_load_check_covers_every_operator_family`
