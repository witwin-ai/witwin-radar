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

The packaged prebuilt extension is the normal load source. A developer override
must be explicit. Nothing silently searches for or loads a stale global
extension, and the load-time presence check names one operator per family so a
stale binary fails at load.

## Acceptance evidence

- `tests/test_phase4_contracts.py::test_propagation_package_does_not_import_the_channel_adapter`
- `tests/test_phase4_import_boundary.py::test_the_propagation_package_alone_does_not_require_channel`
- `tests/test_phase4_import_boundary.py::test_synthesis_scattering_and_paths_do_not_require_channel`
- `tests/test_phase4_binding_manifest.py::test_the_load_check_covers_every_operator_family`
