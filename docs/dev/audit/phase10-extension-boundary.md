# Phase-10 audit: the `_radar_native` / `_channel` source and link boundary

This is the evidence behind acceptance criterion A5 - "no shared RF/geometry
binary, no third Python binding, no second RayD registry, no cross-extension
private call" - and the Radar half of A8, "the Radar wheel needs no Dr.Jit or
RayD runtime dependency".

The claim was already true before Phase 10. What it lacked was evidence that
survives a refactor. So this document records MEASURED binary facts rather than
a reading of the build files, and `ci/check_extension_boundary.py` re-measures
them on demand. Reading is how a boundary is designed; measuring is how it stays
designed.

Measured 2026-07-27 on Windows 11 / Python 3.11.14 / Torch 2.10.0 / nvcc 12.9.41
by `ci/check_extension_boundary.py`, in the Phase-10 radar worktree
(`claude/stage2-phase10`) and against the Channel ADR-043 developer build.

---

## 1. The two binaries

| | Radar | Channel |
|---|---|---|
| file | `witwin/radar/cuda/prebuilt/witwin_radar_dirichlet_cuda.pyd` | `channel/.codex_tmp/adr043-objbuild/_channel.cp311-win_amd64.pyd` |
| bytes | 1,244,160 | 37,206,528 |
| built (UTC) | 2026-07-27T11:26:51Z | 2026-07-27T10:49:57Z |
| sha256 (16) | `9705f4593653fc8c` | `581ae947f057c12f` |
| format | PE32+ | PE32+ |
| build system | `torch.utils.cpp_extension.load(..., is_python_module=False)` | CMake + `Python_add_library(_channel MODULE WITH_SOABI)` |
| kind | Torch dispatcher library | CPython extension module |

### Radar import table (10 entries, complete)

```
api-ms-win-crt-heap-l1-1-0.dll
api-ms-win-crt-runtime-l1-1-0.dll
api-ms-win-crt-string-l1-1-0.dll
cudart64_12.dll
KERNEL32.dll
MSVCP140.dll
torch_cpu.dll
torch_cuda.dll
VCRUNTIME140.dll
VCRUNTIME140_1.dll
```

### Channel import table (21 entries, complete)

```
ADVAPI32.dll                        c10.dll            KERNEL32.dll
api-ms-win-crt-environment-l1-1-0   c10_cuda.dll       MSVCP140.dll
api-ms-win-crt-heap-l1-1-0          CFGMGR32.dll       nvcuda.dll
api-ms-win-crt-math-l1-1-0          cudart64_12.dll    python311.dll
api-ms-win-crt-runtime-l1-1-0       cusolver64_11.dll  torch_cpu.dll
api-ms-win-crt-stdio-l1-1-0                            torch_cuda.dll
api-ms-win-crt-string-l1-1-0                           torch_python.dll
                                                       VCRUNTIME140.dll
                                                       VCRUNTIME140_1.dll
```

### What the tables say

- **The intersection is the intended shared substrate and nothing else**: the
  platform CRT, `cudart64_12`, `torch_cpu`, `torch_cuda`. There is no shared
  RF or geometry library, static archive, or object file. Both sides were
  compiled against Torch 2.10, which is the only third-party surface they have
  in common.
- **No `rayd*` DLL in either table.** RayD enters Channel by SOURCE, as a CMake
  build-tree target, and is statically absorbed into `_channel`. It never
  becomes a Radar runtime dependency, and it never becomes a separate
  redistributable that a second consumer could introduce unlocked.
- **No `drjit*`, `mitsuba*`, `optix*` or `sionna*` import anywhere.** Channel
  reaches OptiX through `nvcuda.dll` (the driver), which is the RayD-owned
  path, not a Channel-owned second one.
- **The radar library does not import `python311.dll`.** It is built with
  `is_python_module=False` against `TORCH_TARGET_VERSION=0x020a000000000000`,
  so it holds no Python C API references at all. A `python3xx` import appearing
  here would mean the Stable-ABI property had silently broken, which is why the
  gate asserts its absence rather than assuming it.
- **Channel imports `python311.dll` and `torch_python.dll`; Radar imports
  neither.** The two extensions are different KINDS of artifact, and the audit
  records that rather than flattening it: `_channel` is a CPython module whose
  ABI is version-locked, `_radar_native` is a dispatcher library whose ABI is
  not.

## 2. Neither binary names the other

Checked over the WHOLE FILE, not just the import directory, because a
cross-extension private call that resolved the other library by name at runtime
would leave the string in `.rdata` and nothing in the import table.

| scan | result |
|---|---|
| `witwin_radar_dirichlet_cuda` in the radar binary | present (its own dispatcher namespace) |
| `_channel` in the radar binary | **absent** |
| `_channel` in the Channel binary | present (its own module name) |
| `witwin_radar_dirichlet_cuda` in the Channel binary | **absent** |

The two positive rows are the calibration. A scan that found neither stem would
be indistinguishable from a scan that does not work, so the audit records that
each binary DOES name itself before concluding that neither names the other.

## 3. Source and link facts

### `_channel` source-links the locked RayD

- `channel/CMakeLists.txt:382-384` forces `RAYD_TORCH_BUILD_NATIVE ON`,
  `RAYD_TORCH_BUILD_PYTHON_MODULE OFF`,
  `RAYD_TORCH_INSTALL_SOURCE_BUNDLE OFF`.
- `channel/CMakeLists.txt:386` `add_subdirectory("${RAYD_SOURCE_DIR}/backends/torch" ...)`,
  with a hard `FATAL_ERROR` at `:389-391` if `rayd_torch_native_core` is absent.
  RayD arrives as a build-tree CMake target from source; there is no installed
  RayD library and no RayD Python extension in the picture.
- `channel/CMakeLists.txt:549` builds the single `_channel` module.
- `channel/CMakeLists.txt:678-687` links `rayd_torch_native_core`, the Torch
  libraries, the Python library and `CUDA::cusolver`. That is the complete link
  line; nothing Radar-owned appears in it.
- The only other CMake target in the file is `channel_legacy_slab_lockstep`
  (`:723-735`), a `BUILD_TESTING` lockstep binary that is not shipped.

Measured at Channel `1606f0d`.

### `_radar_native` compiles nine Radar-owned sources

`witwin/radar/cuda/build.py:214-226` lists exactly nine files and they are the
whole build input:

```
witwin/radar/cuda/extension.cpp
witwin/radar/cuda/kernels/dirichlet.cu
witwin/radar/cuda/kernels/fmcw_beat.cu
witwin/radar/cuda/kernels/frontend.cu
witwin/radar/cuda/kernels/ofdm_cfr.cu
witwin/radar/cuda/kernels/pulsed_echo.cu
witwin/radar/cuda/kernels/scatter_response.cu
witwin/radar/cuda/kernels/sensor_weight.cu
witwin/radar/cuda/kernels/two_way_join.cu
```

No CMake, no `add_subdirectory`, no external target, no third-party source.
`ci/native-binding-manifest.json` pins the same list and
`tests/test_phase4_binding_manifest.py::test_every_manifested_source_is_a_build_input`
asserts the equality; `ci/check_native_bindings.py` re-asserts it outside pytest
so a release run that does not execute the suite still checks it.

`witwin/radar/cuda/extension.cpp:3` opens a single
`STABLE_TORCH_LIBRARY(witwin_radar_dirichlet_cuda, m)` block with 34 `m.def`
entries. One dispatcher namespace, one registry, one binary.

### No second RayD registry, no `extern "C"` handshake

RayD's typed C++ API is consumed by Channel through
`rayd/torch/integration.h` inside the Channel build. Radar names neither RayD
nor `_channel` in any source: `grep -rn "torch.ops" witwin/` returns only
`witwin/radar/cuda/build.py`, and the static AST closure in
`tests/test_phase4_import_boundary.py` asserts `witwin.channel._channel` is
never named. There is no function-pointer getter, no copied `extern "C"`
signature and no dynamic symbol lookup between the two extensions - which the
byte scan in section 2 now backs at the binary level rather than at the source
level.

## 4. Exactly one native member

`witwin/radar/cuda/prebuilt/` contains one `.pyd`/`.so` and its two identity
sidecars. `ci/check_extension_boundary.py::discover_radar_binary` asserts the
count is exactly one and finds it by SUFFIX rather than by stem, so the
Phase-10 physical rename to `_radar_native` does not require this gate to be
edited in the same commit.

## 5. The Channel half of work items 3 and 4: read-only audit

Channel already carries every artifact item 3 asks for. Recorded here, changed
nowhere:

| artifact | Channel | Radar before Phase 10 | Radar after S0/S1 |
|---|---|---|---|
| binding manifest | `ci/native-binding-manifest.json`, 253 symbols | 34 symbols, no schema version | schema 2, ownership registry |
| contract-coverage manifest | `ci/contract-coverage-manifest.json` | none | coverage lives in the binding manifest rows |
| public API snapshot | `ci/public-api-snapshot.json` | none | none (Phase-11 work) |
| import-graph gate | `ci/check_import_graph.py` + frozen digest | pytest AST scans | unchanged (S5 owns the static gates) |
| duplication ledger | `ci/check_duplication.py` | none | none |
| maintenance budgets | `ci/maintenance-budgets.json` | none | none |
| ABI version constant | `CHANNEL_ABI_VERSION = 1` | none | `RADAR_ABI_VERSION = 1` |
| runtime build identity | `_channel.build_info()`, 22 fields + sidecar | none | 18-field sidecar record + `build_info()` |
| runtime diagnostics | `deployment.runtime_diagnostics()` | none | `witwin/radar/deployment.py` |
| capability record | `capabilities.py` | reads Channel's only | `witwin/radar/capabilities.py` |
| error owners | one `CapacityFailureState` + terminal check | none | `error_owners` in the binding manifest |

No Channel file was edited for this audit. The Channel-side Phase-10 defects
found by the design pass (the wheel-smoke `CONTRACT_VERSION` literal, the
workflow's stale assertion, and the nightly tier's missing `--core-wheel`) are
attached to the stages that execute the failing gate, not to this document.

## 6. What this audit does NOT prove

Recorded so the evidence is not read as broader than it is.

- **Linux.** Both measurements are PE. `ci/check_extension_boundary.py` also
  reads ELF64 `DT_NEEDED`, and that reader was verified against two real ELF
  libraries (`libstdc++.so.6` -> `libm.so.6, libc.so.6, ld-linux-x86-64.so.2,
  libgcc_s.so.1`; `libm.so.6` -> `libc.so.6, ld-linux-x86-64.so.2`), matching
  `readelf -d` exactly. What is NOT verified is the Linux ALLOWLIST, because no
  Linux radar binary exists here to check it against; the manylinux cells of
  the release matrix are the named deferral that will.
- **Dynamic loading through a computed name.** The byte scan finds a stem that
  is present as a literal. A name assembled at runtime from fragments would
  evade it. Nothing in either tree does that, and a call graph is not what this
  gate is.
- **Symbol-level cross-calls through Torch.** Both libraries register into the
  Torch dispatcher, which is a shared registry by design. What the boundary
  forbids is a PRIVATE ABI between them, and the disjoint namespaces
  (`witwin_radar_dirichlet_cuda::*` versus Channel's pybind surface) plus the
  absence of any cross-naming are what is measured here.
