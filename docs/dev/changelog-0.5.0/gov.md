## Breaking changes

- `witwin-radar` now requires `torch==2.10.0` instead of `torch>=2.10`. The loader only ever accepted a packaged binary on the exact Torch version it was built against; the metadata now says so. Migration: install Torch 2.10.0 (the CUDA 12.8 build matches the release wheels), or build the native library for your Torch with `python scripts/build_radar_cuda_prebuilt.py` from a source checkout.
- `ci/native-binding-manifest.json` is generated (schema 3). The `launches`, `fused_stages`, `host_observations`, `contract_test`, `numerical_owner`, `family`, `end_to_end_caller` and `logical_owner` fields are gone; operator rows carry `symbol`, `ad_group`, `ad_role`, `native_tu`, `python_owner` and `python_callers`. Migration for contributors: after changing a native operator, run `python ci/check_native_bindings.py --write`.

## Added

- The native loader refuses, before loading, a binary whose recorded CUDA architectures cannot run on the current device, with a message naming the device, its `sm_XY` and the `TORCH_CUDA_ARCH_LIST` value that fixes it. Previously the first kernel launch failed with the driver's "no kernel image is available".
- `pytest --benchmark` runs the absolute wall-clock budgets, which are now skipped by default; launch, synchronization, host-observation and memory budgets still run with `--gpu`.

## Fixed

- `python ci/run_ci_tier.py quick`'s documentation gate no longer fails on Linux because a document names the Windows-only `_radar_native.pyd`: paths the repository ignores (build and run products) are not checked for existence.
- A checkout made with `core.autocrlf=true` no longer refuses every prebuilt binary: `.gitattributes` checks text out with LF endings, so the source fingerprint is the same on every checkout.
- `README.md` and `FEATURE_LIST.md` list all eighteen root exports (they omitted `Iq` and `Leakage`), and `docs/pipeline_guide.md` states that `Frame.points()` combines sensor pairs coherently.

## Changed

- `witwin-radar[channel]` installs Channel only on Python 3.10-3.13, which Channel's wheels cover; on Python 3.14 the extra installs Radar alone.
- Finished plans, phase records and superseded performance measurements moved to `docs/dev/archive/`; `PERFORMANCE.md` keeps the current conclusions and known limits. R-ADR-002, R-ADR-012, R-ADR-014 and R-ADR-015 are marked superseded or historical where they describe deleted mechanisms.
