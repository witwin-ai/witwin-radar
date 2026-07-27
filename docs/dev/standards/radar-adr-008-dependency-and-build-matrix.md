# R-ADR-008: Dependency, Python/Torch/CUDA matrix, and the provisional pin

Status: Accepted (Phase 4)

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

The follow-up, not done here:

1. Pin `witwin-channel` to a released version in the `channel` extra.
2. Add a required-consumer CI job that installs the pinned artifacts and runs the
   Phase-4 tests against them.

Until both land, no claim about released-artifact compatibility is supported by
this work.

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

- `tests/test_phase4_contracts.py::test_propagation_package_does_not_import_the_channel_adapter`
- `tests/test_phase4_import_boundary.py::test_the_propagation_package_alone_does_not_require_channel`
- `tests/test_phase4_import_boundary.py::test_synthesis_scattering_and_paths_do_not_require_channel`
- `tests/test_phase4_binding_manifest.py::test_the_load_check_covers_every_operator_family`
