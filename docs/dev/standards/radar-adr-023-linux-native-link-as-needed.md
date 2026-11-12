# R-ADR-023: The Linux native library records only the libraries it references

Status: Accepted (Radar 0.5.0)

## Context

`ci/check_extension_boundary.py` asserts that the radar library's imports are a subset of a frozen
allowlist, one for the PE import table and one for the ELF `DT_NEEDED` list. Neither list contains
the Python binding library, because the library is a Torch dispatcher library built with
`is_python_module=False` and must not depend on the Python C API (R-ADR-004).

The Linux developer build failed that gate: its `DT_NEEDED` named `libtorch_python.so`.
`torch.utils.cpp_extension` appends `-ltorch_python` to every build that is not a standalone
executable, whatever `is_python_module` says. MSVC's linker records a DLL only when a symbol is
taken from it, so the Windows library never imported `torch_python.dll`; GNU ld records every `-l`
it is given. `ldd -u -r` on the Linux library listed `libtorch_python.so`, `libtorch.so`,
`libc10_cuda.so` and `libm.so.6` as unused direct dependencies: no symbol of the library resolves
to any of them.

## Decision

`witwin/radar/cuda/runtime.py` passes `-Wl,--as-needed` as the first extra linker flag on every
non-Windows build. It precedes every `-l` Torch appends, so the linker records exactly the libraries
the object code references, which is the set the PE import table already holds. The allowlist is
unchanged: the evidence says the extra entry was a link artefact, not a dependency, and widening
the allowlist would have accepted it.

## Consequences

- The Linux library needs `libc10.so`, `libtorch_cpu.so`, `libtorch_cuda.so`, `libcudart.so.12`,
  `libstdc++.so.6`, `libgcc_s.so.1` and `libc.so.6`, all on the allowlist, and
  `tests/test_native_boundary.py` passes on it.
- The source fingerprint covers the native sources only, so the flag moves no identity; the
  library is relinked, and an existing build directory relinks once.
