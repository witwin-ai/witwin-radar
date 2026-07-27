# R-ADR-019: The packaged native load contract and its identity chain

Status: Accepted (Phase 10)

Supersedes the "Loading" paragraph of R-ADR-008, which asserted this contract
before the code implemented it.

## Context

`witwin/radar/cuda/build.py` resolved four load routes and the DEFAULT one was
the unsafe one:

| route | trigger | failure behaviour |
|---|---|---|
| packaged prebuilt | default | missing file returned `None`; ANY load exception was swallowed by `except Exception: module = None` |
| env prebuilt | `WITWIN_RADAR_DIRICHLET_CUDA_PREBUILT=1` | loud |
| JIT build | plain fallthrough, no env var | the normal-import fallback |
| build dir | `WITWIN_RADAR_DIRICHLET_CUDA_BUILD_DIR` | redirected routes 2 and 3 |

Load-time validation was `hasattr` over eight family names, one per operator
family. Nothing checked Torch, CUDA, the C++ ABI, the platform, the compiled SM
list, the Git revision, the sources, or the binary. `source_fingerprint()`
existed but only keyed the shared JIT build directory; no load route ever
compared it to anything.

Three consequences followed, and all three were observed:

1. **The silent JIT was reachable from an ordinary import.** A missing or
   ABI-stale prebuilt made `import witwin.radar.paths.two_way` call
   `torch.utils.cpp_extension.load` inside the importing process. On Windows
   that runs `_ensure_windows_build_tools_on_path()`, which copies the entire
   `vcvars64` environment over `os.environ` including a wholesale `PATH`
   replacement. A library built after that mutation fails `DllMain` with an
   access violation in the same process, and the unbounded `PATH` growth
   surfaces much later as unrelated CUDA tests failing partway through a
   session.
2. **A stale or swapped binary loaded clean.** Two revisions of the same
   operator set register the same names, so the family check cannot tell them
   apart. R-ADR-004 already names that as the silent-wrong-numerics case.
3. **The shared JIT temp directory was a cross-checkout global load source.**
   Two worktrees at different revisions keyed the same directory and the binary
   on disk was whichever one linked last.

The release wheel also ships the CUDA sources, so an end user's machine could
reach the JIT route too. Shipping sources while the JIT is the silent fallback
is the worst of the available combinations.

## Decision

### The load order

```
0. read the developer override configuration; a PARTIAL set of its three
   variables raises immediately, whatever else is available
1. WITWIN_RADAR_NATIVE_BUILD=1        -> compile from source (build script only)
2. packaged prebuilt present          -> validate, load, origin="packaged"
3. complete developer override        -> validate, load, origin="developer"
4. otherwise                          -> RadarExtensionLoadError naming the
                                         three override variables
```

There is no branch that returns `None`, no branch that answers a validation
failure with a rebuild, and no branch that catches a load error broadly.

A partial override is read and rejected FIRST, before the packaged prebuilt is
even looked for. Ignoring two of three variables because the default happened to
work is how a developer ends up measuring a binary they did not select.

`WITWIN_RADAR_NATIVE_BUILD=1` deliberately outranks the packaged artifact: it
means "compile from these sources", and a stale prebuilt that the loader refuses
must not be able to block the one command that replaces it.

A present packaged prebuilt outranks the developer override, matching Channel's
ADR-006 precedence. In a source checkout the packaged prebuilt IS the developer
artifact, and refreshing it is the supported dev flow.

### The environment variables

| variable | meaning |
|---|---|
| `WITWIN_RADAR_NATIVE_DEVELOPER_OVERRIDE=1` | enable the override; all three required together |
| `WITWIN_RADAR_NATIVE_EXTENSION_PATH` | absolute path to a `.pyd`/`.so`; its two sidecars must sit beside it |
| `WITWIN_RADAR_NATIVE_EXPECTED_FINGERPRINT` | SHA-256; must equal the validated `build_fingerprint` |
| `WITWIN_RADAR_NATIVE_BUILD=1` | the ONLY way to reach `cpp_extension.load`; set by `scripts/build_radar_cuda_prebuilt.py` |
| `WITWIN_RADAR_NATIVE_BUILD_DIR` | honoured only under `WITWIN_RADAR_NATIVE_BUILD=1` |

Deleted: `WITWIN_RADAR_DIRICHLET_CUDA_PREBUILT` (route 2 no longer exists; the
override replaces it), `WITWIN_RADAR_DIRICHLET_CUDA_SKIP_PREBUILT` (replaced by
`WITWIN_RADAR_NATIVE_BUILD`), `WITWIN_RADAR_DIRICHLET_CUDA_BUILD_DIR` (replaced
by `WITWIN_RADAR_NATIVE_BUILD_DIR`).

`_ensure_windows_build_tools_on_path()` and `_ensure_cuda_home_from_nvcc()` are
guarded a second time at the point of damage: reaching either without
`WITWIN_RADAR_NATIVE_BUILD=1` raises rather than mutating `os.environ`.
`torch.utils.cpp_extension` is imported inside `_jit_build_extension` and not at
module scope, so an ordinary import of the loader cannot pull the compiler
machinery into the process, and a test can assert exactly that.

### The identity chain: sidecars plus a binary digest

`_radar_native` is a Torch *dispatcher* library (`is_python_module=False`,
`STABLE_TORCH_LIBRARY`), not a Python extension module. It cannot expose a
`build_info()` Python symbol the way `witwin.channel._channel` does without
adding a native ABI symbol. The identity therefore travels in two files written
next to the binary:

```
witwin/radar/cuda/prebuilt/<stem>.pyd            (or .so)
witwin/radar/cuda/prebuilt/<stem>.build-info.json
witwin/radar/cuda/prebuilt/<stem>.build-fingerprint    one ASCII sha256 line
```

`<stem>` is derived from the binary's own filename, so the physical stem stays
written in exactly one place (`build.EXTENSION_NAME`).

`build-info.json` schema, `radar_abi_version = 1`:

| field | type | source |
|---|---|---|
| `radar_abi_version` | int | `identity.RADAR_ABI_VERSION` |
| `extension_name` | str | `build.EXTENSION_NAME` |
| `build_type` | str | `release` or `developer` |
| `torch_version` | str | `torch.__version__` without `+local` |
| `torch_target_version` | str | the `TORCH_TARGET_VERSION` macro compiled in |
| `cuda_version` | str | `torch.version.cuda` |
| `cuda_compiler_version` | str | `nvcc --version` |
| `compiler` | str | MSVC / GCC version string |
| `cxx_abi` | str | `msvc` / `cxx11` / `pre-cxx11` |
| `cuda_architectures` | list[str] | normalized from `WITWIN_CUDA_GENCODE_ARCHES` or `TORCH_CUDA_ARCH_LIST` |
| `platform_tag` | str | `sysconfig.get_platform()` normalized |
| `python_abi` | str | `stable-abi-v1` |
| `radar_git_sha` | str | 40 hex, or `unknown` |
| `radar_git_dirty` | bool | |
| `source_fingerprint` | str | sha256 over the nine-file source set, names + content |
| `operator_symbols` | list[str] | sorted, read from `ci/native-binding-manifest.json` at build time |
| `binary_sha256` | str | sha256 of the produced binary |
| `build_fingerprint` | str | sha256 over canonical JSON of every field above |

`build_fingerprint` uses the same recipe as Channel: `json.dumps(payload,
sort_keys=True, separators=(",",":"), ensure_ascii=True)` then sha256. The
`.build-fingerprint` sidecar holds the same value, so the chain is three-way -
the JSON field, the recomputed value, and the sidecar must all agree.

`validate_identity()` runs BEFORE `torch.ops.load_library`, because loading a
shared library is irreversible within a process: a check afterwards would report
a problem the process can no longer avoid. In order it verifies both sidecars
exist, the record parses with every field at its exact type and no unknown
field, the ABI version, `build_type`, non-empty architecture and sorted-unique
symbol lists, the Git SHA shape, the three-way fingerprint, the binary's actual
SHA-256, the source fingerprint recomputed from the sources shipped beside the
loader, and finally the live `torch_version` / `cuda_version` / `cxx_abi` /
`platform_tag`. Only then does it load, and then every recorded symbol - all 34,
not an eight-family sample - must resolve on `torch.ops.<namespace>`.

`_REQUIRED_OPERATORS`, the eight-family sample, remains as the fallback for the
JIT route, which has no record to consult.

### Considered and rejected: a compiled-in `build_info` ABI symbol

Mirroring Channel exactly would mean adding a `build_info` dispatcher operator
that returns the record from inside the binary. Rejected, for two reasons.

**It is not free in an architecture-only phase.** A new ABI symbol means a new
row in `ci/native-binding-manifest.json`, a direct contract test, an
end-to-end production caller, a `STABLE_TORCH_LIBRARY` schema change for string
returns (returning strings through `StableIValue` is not proven in this
codebase), and an immediate forced rebuild of every packaged prebuilt on every
platform. Phase 10's own acceptance criterion A6 requires architecture-only
native moves to keep exact outputs, the launch ledger and performance; touching
the ABI to describe the ABI works against that.

**It is also strictly weaker at the job.** The defect class this contract exists
to kill is *a stale, swapped or ABI-mismatched binary loaded silently*. A
self-report is regenerated by the same rebuild that produces the swap, so it
agrees with whatever binary is present. A recorded digest of the bytes, compared
against the bytes actually on disk, does not - and it additionally catches
corruption and truncation that a self-report cannot see. Channel's own chain has
the same shape from the other side: `build_info()` from the binary, plus a
`_channel.build-fingerprint` sidecar holding the expected value. Radar inverts
which half is authoritative and adds the binary digest to close the gap.

Named follow-up, not Phase 10: if a later phase already reopens the radar ABI
for numerical reasons, embedding the fingerprint as a native constant becomes
cheap and should be reconsidered then. It is recorded here so a later reviewer
reads the sidecar as a decision rather than as laziness.

### Shipping the CUDA sources is now part of the contract

`source_fingerprint` is validated against the sources that sit beside the
loader. That is the check which closes the "two revisions of the same operator
set are indistinguishable" hole, and it is the reason the wheel must keep
shipping `*.cu` / `*.cpp` / `*.cuh` / `*.h`. They are identity material, not a
leftover. The JIT route they used to enable is now unreachable without
`WITWIN_RADAR_NATIVE_BUILD=1`.

### The refresh procedure

```
# throwaway process, NEVER inside a test process
conda run -n witwin2 python scripts/build_radar_cuda_prebuilt.py
#   sets WITWIN_RADAR_NATIVE_BUILD=1 internally
#   builds into a private build directory
#   refuses to publish unless every manifest operator resolves on the
#     freshly built library
#   copies the binary and writes both sidecars into witwin/radar/cuda/prebuilt/
#   re-validates the published artifact with the loader's own function

# clean process, verifies the identity chain end to end
conda run -n witwin2 python -c \
  "from witwin.radar.cuda import build; print(build.build_extension().build_info())"
```

`--release` stamps `build_type="release"`; the DEFAULT is `developer`, so a
locally built artifact can never be mistaken for a locked release build by
omission. On this development machine every local artifact is `developer`
regardless: the local nvcc is 12.9.41 against a locked 12.8.1.

An existing prebuilt without sidecars now fails loudly with a message naming the
rebuild command. That is the entire behavioural change a user sees, and it is
intended.

## Consequences

- A missing, stale, swapped, corrupted, mis-targeted or wrong-Torch artifact
  fails at load with the full recorded identity, and never compiles a
  replacement.
- The vcvars/`DllMain` hazard cannot occur in a test or user process.
- The identity chain costs one 1.4 MB SHA-256 plus nine source digests per
  process, measured at 3.6 ms, once, behind the existing `_LOADED_MODULE` memo.
  Total validated load is 6.3 ms against a 50 ms ceiling.
- `witwin/radar/cuda/identity.py` has no `torch.ops` access and is importable
  without CUDA, so an artifact can be validated on a machine that cannot run it.

## Acceptance evidence

- `tests/test_phase10_loader_contract.py` (22 cases, every load case in a fresh
  subprocess with an explicit environment, asserting on exception type and
  message)
- `tests/test_phase4_binding_manifest.py::test_every_load_route_validates_the_required_operators`
- `tests/test_phase4_binding_manifest.py::test_the_jit_build_directory_is_keyed_by_the_source_set`
- radar suite: `721 passed / 807 skipped` default, `1528 passed / 0 failed`
  with `--gpu` (Phase-9 baseline 699 / 1506, plus the 22 cases above)
