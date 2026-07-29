"""Freeze the Radar/Channel native boundary as a machine-checked fact.

Phase-10 acceptance criterion A5 claims that there is no shared RF/geometry
binary, no third Python binding, no second RayD registry and no cross-extension
private call. That claim was true when it was written by reading CMake and
``build.py``. Reading is not evidence that survives a refactor, so this gate
reads the SHIPPED BINARIES instead: a PE import table on Windows, the ELF
``DT_NEEDED`` list on Linux.

What it asserts, and why each one is worth a check rather than a comment:

* the radar library's imports are a subset of a frozen allowlist. Linking one
  new third-party runtime into the radar extension is exactly how a shared
  RF/geometry library would arrive, and it would arrive silently;
* no ``rayd``/``drjit``/``mitsuba``/``optix`` import anywhere in the radar
  library. A8 says the Radar wheel needs no Dr.Jit or RayD runtime, and RayD
  reaches Channel by SOURCE (``add_subdirectory``), so a RayD DLL appearing on
  the Radar side would mean a second, unlocked introduction path;
* neither binary names the other's stem, checked over the whole file rather
  than over the import table alone, so a ``LoadLibrary``-style late binding is
  caught as well as a link-time one;
* the radar library does not import ``python3*.dll``. It is a Torch dispatcher
  library built with ``is_python_module=False`` against the Stable ABI target;
  a python3xx import would mean that property had quietly broken;
* exactly one native member sits under ``witwin/radar/cuda/prebuilt/``.

The allowlist is deliberately a list and not a pattern over "system-looking"
names. A gate whose allowlist can absorb a new entry by resembling an old one
is a gate that grows.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PREBUILT_DIR = REPO_ROOT / "witwin" / "radar" / "cuda" / "prebuilt"
NATIVE_SUFFIXES = (".pyd", ".so")

#: Every DLL the radar dispatcher library may import on Windows. CRT plus the
#: CUDA runtime plus the two Torch libraries it registers into - nothing else.
WINDOWS_ALLOWLIST = frozenset(
    {
        "api-ms-win-crt-heap-l1-1-0.dll",
        "api-ms-win-crt-math-l1-1-0.dll",
        "api-ms-win-crt-runtime-l1-1-0.dll",
        "api-ms-win-crt-stdio-l1-1-0.dll",
        "api-ms-win-crt-string-l1-1-0.dll",
        "api-ms-win-crt-utility-l1-1-0.dll",
        "kernel32.dll",
        "msvcp140.dll",
        "vcruntime140.dll",
        "vcruntime140_1.dll",
        "cudart64_12.dll",
        "torch_cpu.dll",
        "torch_cuda.dll",
    }
)

#: The ELF equivalent. Not exercised on Windows; the Linux cells of the release
#: matrix are the named deferral that runs it (D1), and it is written out here
#: rather than left as a TODO so that run has a gate to fail against.
LINUX_ALLOWLIST = frozenset(
    {
        "ld-linux-x86-64.so.2",
        "libc.so.6",
        "libm.so.6",
        "libdl.so.2",
        "libpthread.so.0",
        "librt.so.1",
        "libgcc_s.so.1",
        "libstdc++.so.6",
        "libcudart.so.12",
        "libtorch.so",
        "libtorch_cpu.so",
        "libtorch_cuda.so",
        "libc10.so",
        "libc10_cuda.so",
    }
)

#: Substrings that must not appear in any radar import name, in either format.
FORBIDDEN_IMPORT_TOKENS = ("rayd", "drjit", "mitsuba", "optix", "sionna")

#: The radar library is not a Python module. A python3xx import means the
#: Stable-ABI dispatcher-library property broke.
PYTHON_IMPORT_TOKENS = ("python3", "libpython3")


class BoundaryError(Exception):
    """A boundary claim failed against a real binary."""


def read_pe_imports(path: Path) -> list[str]:
    """The DLL names in a PE32+ import directory, in table order."""

    data = path.read_bytes()
    pe_offset = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe_offset : pe_offset + 4] != b"PE\0\0":
        raise BoundaryError(f"{path} is not a PE image")
    coff = pe_offset + 4
    section_count = struct.unpack_from("<H", data, coff + 2)[0]
    optional_size = struct.unpack_from("<H", data, coff + 16)[0]
    optional = coff + 20
    magic = struct.unpack_from("<H", data, optional)[0]
    if magic != 0x20B:
        raise BoundaryError(f"{path} is not PE32+ (magic {magic:#x})")
    directories = optional + 112
    import_rva = struct.unpack_from("<I", data, directories + 8)[0]

    sections = []
    section_table = optional + optional_size
    for index in range(section_count):
        base = section_table + index * 40
        virtual_size, virtual_address, raw_size, raw_pointer = struct.unpack_from("<IIII", data, base + 8)
        sections.append((virtual_address, max(virtual_size, raw_size), raw_pointer))

    def to_offset(rva: int) -> int:
        for virtual_address, span, raw_pointer in sections:
            if virtual_address <= rva < virtual_address + span:
                return raw_pointer + (rva - virtual_address)
        raise BoundaryError(f"{path} has an unmapped RVA {rva:#x}")

    if import_rva == 0:
        return []
    offset = to_offset(import_rva)
    names: list[str] = []
    while True:
        entry = data[offset : offset + 20]
        if len(entry) < 20 or entry == b"\0" * 20:
            break
        name_rva = struct.unpack_from("<I", entry, 12)[0]
        if name_rva == 0:
            break
        name_offset = to_offset(name_rva)
        end = data.index(b"\0", name_offset)
        names.append(data[name_offset:end].decode("ascii", "replace"))
        offset += 20
    return names


def read_elf_needed(path: Path) -> list[str]:
    """The ``DT_NEEDED`` entries of an ELF64 shared object, in table order."""

    data = path.read_bytes()
    if data[:4] != b"\x7fELF":
        raise BoundaryError(f"{path} is not an ELF image")
    if data[4] != 2:
        raise BoundaryError(f"{path} is not ELF64")
    little_endian = data[5] == 1
    endian = "<" if little_endian else ">"
    program_header_offset = struct.unpack_from(f"{endian}Q", data, 0x20)[0]
    program_header_size = struct.unpack_from(f"{endian}H", data, 0x36)[0]
    program_header_count = struct.unpack_from(f"{endian}H", data, 0x38)[0]

    dynamic_offset = 0
    dynamic_size = 0
    for index in range(program_header_count):
        base = program_header_offset + index * program_header_size
        segment_type = struct.unpack_from(f"{endian}I", data, base)[0]
        if segment_type == 2:  # PT_DYNAMIC
            dynamic_offset = struct.unpack_from(f"{endian}Q", data, base + 0x08)[0]
            dynamic_size = struct.unpack_from(f"{endian}Q", data, base + 0x20)[0]
            break
    if dynamic_size == 0:
        return []

    string_table_address = 0
    needed_offsets: list[int] = []
    cursor = dynamic_offset
    end = dynamic_offset + dynamic_size
    while cursor + 16 <= end:
        tag, value = struct.unpack_from(f"{endian}QQ", data, cursor)
        if tag == 0:  # DT_NULL
            break
        if tag == 1:  # DT_NEEDED
            needed_offsets.append(value)
        elif tag == 5:  # DT_STRTAB
            string_table_address = value
        cursor += 16
    if not needed_offsets:
        return []
    if string_table_address == 0:
        raise BoundaryError(f"{path} has DT_NEEDED entries but no DT_STRTAB")

    string_table_offset = _elf_address_to_offset(
        data, endian, program_header_offset, program_header_size, program_header_count, string_table_address, path
    )
    names: list[str] = []
    for name_offset in needed_offsets:
        start = string_table_offset + name_offset
        stop = data.index(b"\0", start)
        names.append(data[start:stop].decode("ascii", "replace"))
    return names


def _elf_address_to_offset(
    data: bytes,
    endian: str,
    program_header_offset: int,
    program_header_size: int,
    program_header_count: int,
    address: int,
    path: Path,
) -> int:
    for index in range(program_header_count):
        base = program_header_offset + index * program_header_size
        segment_type = struct.unpack_from(f"{endian}I", data, base)[0]
        if segment_type != 1:  # PT_LOAD
            continue
        offset = struct.unpack_from(f"{endian}Q", data, base + 0x08)[0]
        virtual_address = struct.unpack_from(f"{endian}Q", data, base + 0x10)[0]
        file_size = struct.unpack_from(f"{endian}Q", data, base + 0x20)[0]
        if virtual_address <= address < virtual_address + file_size:
            return offset + (address - virtual_address)
    raise BoundaryError(f"{path} has an unmapped virtual address {address:#x}")


def binary_format(path: Path) -> str:
    with path.open("rb") as handle:
        magic = handle.read(4)
    if magic[:2] == b"MZ":
        return "pe"
    if magic == b"\x7fELF":
        return "elf"
    raise BoundaryError(f"{path} is neither a PE nor an ELF image")


def read_imports(path: Path) -> list[str]:
    if binary_format(path) == "pe":
        return read_pe_imports(path)
    return read_elf_needed(path)


def allowlist_for(path: Path) -> frozenset[str]:
    return WINDOWS_ALLOWLIST if binary_format(path) == "pe" else LINUX_ALLOWLIST


def names_binary(path: Path, token: str) -> bool:
    """Does the binary contain ``token`` as an ASCII string anywhere?

    Stronger than an import-table check on purpose: a cross-extension private
    call that resolved the other library by name at runtime would leave the
    name in ``.rdata`` and nowhere in the import directory.
    """

    return token.encode("ascii") in path.read_bytes()


def discover_radar_binary() -> Path:
    """The one native member under the packaged prebuilt directory.

    Discovered by suffix rather than by stem so the Phase-10 physical rename
    does not need this gate edited in the same commit.
    """

    if not PREBUILT_DIR.is_dir():
        raise BoundaryError(
            f"{PREBUILT_DIR} is not a directory; build the packaged prebuilt "
            "with `python scripts/build_radar_cuda_prebuilt.py`"
        )
    members = sorted(entry for entry in PREBUILT_DIR.iterdir() if entry.is_file() and entry.suffix in NATIVE_SUFFIXES)
    if len(members) != 1:
        raise BoundaryError(
            f"expected exactly one native member under {PREBUILT_DIR}, found {[member.name for member in members]}"
        )
    return members[0]


def stem_of(path: Path) -> str:
    """``_channel.cp311-win_amd64.pyd`` -> ``_channel``."""

    return path.name.split(".", maxsplit=1)[0]


def check_radar(path: Path) -> dict[str, object]:
    imports = read_imports(path)
    allowed = allowlist_for(path)
    failures: list[str] = []

    unexpected = sorted({name for name in imports if name.lower() not in allowed})
    if unexpected:
        failures.append(f"{path.name} imports outside the frozen allowlist: {unexpected}")
    for name in imports:
        lowered = name.lower()
        for token in FORBIDDEN_IMPORT_TOKENS:
            if token in lowered:
                failures.append(f"{path.name} imports {name}, which names {token!r}")
        for token in PYTHON_IMPORT_TOKENS:
            if lowered.startswith(token):
                failures.append(
                    f"{path.name} imports {name}: the radar library is a Torch "
                    "dispatcher library and must stay free of the Python C API"
                )
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "format": binary_format(path),
        "imports": sorted(imports, key=str.lower),
        "failures": failures,
    }


def check_channel(path: Path) -> dict[str, object]:
    """Channel is audited, not policed: it legitimately links more than Radar.

    The only Channel assertion here is the one that belongs to the BOUNDARY -
    no Dr.Jit runtime, and no reference to the Radar library.
    """

    imports = read_imports(path)
    failures = [f"{path.name} imports {name}, which names 'drjit'" for name in imports if "drjit" in name.lower()]
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "format": binary_format(path),
        "imports": sorted(imports, key=str.lower),
        "failures": failures,
    }


def check_boundary(radar_binary: Path, channel_binary: Path | None = None) -> dict[str, object]:
    report: dict[str, object] = {"radar": check_radar(radar_binary)}
    failures = list(report["radar"]["failures"])  # type: ignore[index]

    radar_stem = stem_of(radar_binary)
    if channel_binary is not None:
        channel = check_channel(channel_binary)
        report["channel"] = channel
        failures.extend(channel["failures"])  # type: ignore[index]
        channel_stem = stem_of(channel_binary)
        if names_binary(radar_binary, channel_stem):
            failures.append(
                f"{radar_binary.name} names the Channel extension stem "
                f"{channel_stem!r}: no cross-extension private ABI is allowed"
            )
        if names_binary(channel_binary, radar_stem):
            failures.append(
                f"{channel_binary.name} names the Radar extension stem "
                f"{radar_stem!r}: no cross-extension private ABI is allowed"
            )
    report["failures"] = failures
    report["ok"] = not failures
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--radar-binary", type=Path, default=None, help="the radar native library; defaults to the packaged prebuilt"
    )
    parser.add_argument(
        "--channel-binary",
        type=Path,
        default=None,
        help="an optional _channel binary, to check the pair for cross-naming",
    )
    parser.add_argument("--json", type=Path, default=None, help="write the report")
    arguments = parser.parse_args(argv)

    try:
        radar_binary = arguments.radar_binary or discover_radar_binary()
        if not radar_binary.is_file():
            raise BoundaryError(f"{radar_binary} does not exist")
        if arguments.channel_binary is not None and not arguments.channel_binary.is_file():
            raise BoundaryError(f"{arguments.channel_binary} does not exist")
        report = check_boundary(radar_binary, arguments.channel_binary)
    except BoundaryError as error:
        print(f"extension boundary check failed: {error}")
        return 2

    if arguments.json is not None:
        arguments.json.parent.mkdir(parents=True, exist_ok=True)
        arguments.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for key in ("radar", "channel"):
        entry = report.get(key)
        if entry is None:
            continue
        print(f"{key}: {entry['path']} ({entry['bytes']} bytes, {entry['format']})")
        for name in entry["imports"]:
            print(f"    {name}")
    failures = report["failures"]
    if failures:
        print("\nFAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nextension boundary OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
