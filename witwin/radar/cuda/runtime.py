"""Build identity for the packaged radar native library.

The radar native library is a Torch *dispatcher* library
(``is_python_module=False``), not a Python extension module, so it cannot hand
back a ``build_info()`` Python symbol the way ``witwin.channel._channel`` does
without growing a new native ABI symbol. R-ADR-019 records the decision: the
identity travels in two sidecar files written next to the binary, and the
record names the binary's own SHA-256 so a swapped binary is detected by the
bytes rather than by a self-report that the swap would have regenerated.

Nothing here touches ``torch.ops`` or requires CUDA. The module is importable
on a machine with no GPU, which is what lets the loader validate an artifact
before it hands it to ``torch.ops.load_library``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sysconfig


#: Bumped whenever the sidecar schema or the loader contract changes shape.
#: A packaged artifact whose record carries a different value is rejected; it is
#: never silently upgraded, and it never triggers a rebuild.
#:
#: 2 - Phase 11 deleted the nine ``dirichlet_spectrum`` operators together with
#:     their translation unit. The registered operator set went from 34 symbols
#:     and changed the sensor-weight schema, both observable by a consumer.
#:     fails. That is an ABI change even though the sidecar schema is unchanged.
RADAR_ABI_VERSION = 3

BUILD_INFO_SUFFIX = ".build-info.json"
FINGERPRINT_SUFFIX = ".build-fingerprint"

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")

#: The sidecar schema: field name -> exact required Python type. ``type(value)
#: is not expected`` rather than ``isinstance`` so a bool cannot pass as an int.
BUILD_INFO_FIELDS: tuple[tuple[str, type], ...] = (
    ("radar_abi_version", int),
    ("extension_name", str),
    ("build_type", str),
    ("torch_version", str),
    ("torch_target_version", str),
    ("cuda_version", str),
    ("cuda_compiler_version", str),
    ("compiler", str),
    ("cxx_abi", str),
    ("cuda_architectures", list),
    ("platform_tag", str),
    ("python_abi", str),
    ("radar_git_sha", str),
    ("radar_git_dirty", bool),
    ("source_fingerprint", str),
    ("operator_symbols", list),
    ("binary_sha256", str),
    ("build_fingerprint", str),
)

#: Everything except ``build_fingerprint`` itself, which is the digest over
#: these. Sorted canonical JSON, identical recipe to Channel's
#: ``runtime/extension.py::_expected_fingerprint``.
FINGERPRINT_FIELDS: tuple[str, ...] = tuple(
    name for name, _ in BUILD_INFO_FIELDS if name != "build_fingerprint"
)

BUILD_TYPES = ("release", "developer")


class RadarExtensionLoadError(ImportError):
    """The radar native library could not be selected or loaded safely."""


class RadarExtensionSymbolError(RadarExtensionLoadError):
    """The loaded library does not register every operator it must register."""


class RadarExtensionABIError(RadarExtensionLoadError):
    """The library's recorded identity does not match this checkout or runtime."""


def is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256_PATTERN.fullmatch(value) is not None


def build_info_sidecar_path(binary_path: Path) -> Path:
    """``<dir>/<stem>.build-info.json`` beside the binary.

    Derived from the binary's own name rather than from a constant so the
    physical stem stays written in exactly one place (``build.EXTENSION_NAME``).
    """

    return _sidecar(binary_path, BUILD_INFO_SUFFIX)


def fingerprint_sidecar_path(binary_path: Path) -> Path:
    """``<dir>/<stem>.build-fingerprint`` beside the binary: one ASCII line."""

    return _sidecar(binary_path, FINGERPRINT_SUFFIX)


def _sidecar(binary_path: Path, suffix: str) -> Path:
    name = binary_path.name
    extension = binary_path.suffix
    stem = name[: -len(extension)] if extension else name
    return binary_path.parent / f"{stem}{suffix}"


def source_digest(paths: Iterable[Path]) -> str:
    """SHA-256 over the source set: which files, and what is in them.

    File CONTENT and not just paths, because two worktrees of one branch have
    different absolute paths but the same sources, while one path with an edited
    kernel is a different build input.
    """

    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_build_fingerprint(info: Mapping[str, object]) -> str:
    """SHA-256 over canonical JSON of every identity field but the digest."""

    missing = [name for name in FINGERPRINT_FIELDS if name not in info]
    if missing:
        raise RadarExtensionABIError(
            "the radar build record is missing " + ", ".join(sorted(missing))
        )
    payload = {name: info[name] for name in FINGERPRINT_FIELDS}
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def platform_tag() -> str:
    return sysconfig.get_platform().replace("-", "_").replace(".", "_")


def cxx_abi() -> str:
    if os.name == "nt":
        return "msvc"
    import torch

    return "cxx11" if torch._C._GLIBCXX_USE_CXX11_ABI else "pre-cxx11"


def runtime_identity() -> dict[str, str]:
    """The live values a packaged artifact has to agree with.

    A mismatch is an ABI error and never a reason to rebuild: a wheel built for
    another Torch is simply not usable here, and quietly compiling a
    replacement is the silent path this contract exists to remove.
    """

    import torch

    return {
        "torch_version": str(torch.__version__).split("+", maxsplit=1)[0],
        "cuda_version": str(torch.version.cuda or ""),
        "cxx_abi": cxx_abi(),
        "platform_tag": platform_tag(),
    }


def normalize_cuda_architectures(raw: str) -> list[str]:
    """``"8.7+PTX;12.0"`` -> ``["87+PTX", "120"]``.

    Accepts the ``;``, ``,`` and whitespace separators that
    ``WITWIN_CUDA_GENCODE_ARCHES`` and ``TORCH_CUDA_ARCH_LIST`` each use.
    """

    entries: list[str] = []
    for chunk in re.split(r"[;,\s]+", raw.strip()):
        if not chunk:
            continue
        include_ptx = chunk.endswith("+PTX")
        number = chunk.removesuffix("+PTX").replace(".", "")
        if not number.isdigit():
            raise ValueError(f"Invalid CUDA architecture {chunk!r}.")
        entries.append(f"{number}+PTX" if include_ptx else number)
    return entries


def _run(command: Sequence[str], *, env_overrides: Mapping[str, str] | None = None) -> str:
    env = None
    if env_overrides:
        env = dict(os.environ)
        env.update(env_overrides)
    try:
        completed = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            encoding="mbcs" if os.name == "nt" else "utf-8",
            errors="replace",
            check=False,
            env=env,
        )
    except OSError:
        return ""
    return f"{completed.stdout}\n{completed.stderr}"


def detect_cuda_compiler_version() -> str:
    output = _run(["nvcc", "--version"])
    match = re.search(r"release\s+[\d.]+,\s*V([\d.]+)", output)
    return match.group(1) if match else "unknown"


def detect_compiler() -> str:
    """``"msvc 19.44.35207.1"`` / ``"g++ (GCC) 13.2.0"``, or ``"unknown"``.

    The MSVC banner is localized, so the version is matched as a dotted number
    rather than by an English keyword: forcing ``VSLANG=1033`` fixes the common
    case, but a machine that ignores it must still produce a usable record
    instead of a mis-parsed one.
    """

    if os.name == "nt":
        output = _run(["cl"], env_overrides={"VSLANG": "1033"})
        match = re.search(r"\b(\d+\.\d+\.\d+(?:\.\d+)?)\b", output)
        return f"msvc {match.group(1)}" if match else "unknown"
    output = _run(["c++", "--version"])
    first = output.strip().splitlines()
    return first[0].strip() if first else "unknown"


def detect_git_identity(repo_root: Path) -> tuple[str, bool]:
    sha = _run(["git", "-C", str(repo_root), "rev-parse", "HEAD"]).strip().splitlines()
    revision = sha[0].strip() if sha else ""
    if _GIT_SHA_PATTERN.fullmatch(revision) is None:
        return "unknown", False
    status = _run(["git", "-C", str(repo_root), "status", "--porcelain"])
    return revision, bool(status.strip())


def _require_field(info: Mapping[str, object], name: str, expected: type) -> object:
    if name not in info:
        raise RadarExtensionABIError(f"the radar build record is missing {name!r}")
    value = info[name]
    if type(value) is not expected:
        raise RadarExtensionABIError(
            f"the radar build record field {name!r} must be {expected.__name__}"
        )
    return value


def read_build_info(binary_path: Path) -> dict[str, object]:
    """Parse and type-check the sidecar record. No cross-checks yet."""

    sidecar = build_info_sidecar_path(binary_path)
    if not sidecar.is_file():
        raise RadarExtensionLoadError(
            f"the radar native library {binary_path} has no build record at "
            f"{sidecar}; rebuild it with "
            "`python scripts/build_radar_cuda_prebuilt.py`"
        )
    try:
        raw = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RadarExtensionABIError(f"{sidecar} is not readable JSON") from exc
    if not isinstance(raw, Mapping):
        raise RadarExtensionABIError(f"{sidecar} must contain a JSON object")

    info = dict(raw)
    known = {name for name, _ in BUILD_INFO_FIELDS}
    unknown = sorted(set(info) - known)
    if unknown:
        raise RadarExtensionABIError(
            f"{sidecar} carries unknown fields: " + ", ".join(unknown)
        )
    for name, expected in BUILD_INFO_FIELDS:
        _require_field(info, name, expected)
    return info


def validate_identity(
    binary_path: Path,
    source_paths: Sequence[Path],
    *,
    expected_fingerprint: str | None = None,
) -> dict[str, object]:
    """Validate a radar native artifact before anything loads it.

    Every failure raises. There is no path through this function that returns a
    partially validated record, and no path that answers a mismatch with a
    rebuild.
    """

    if not binary_path.is_file():
        raise RadarExtensionLoadError(
            f"the radar native library {binary_path} does not exist"
        )
    info = read_build_info(binary_path)

    abi_version = info["radar_abi_version"]
    if abi_version != RADAR_ABI_VERSION:
        raise RadarExtensionABIError(
            "radar native ABI mismatch: expected "
            f"{RADAR_ABI_VERSION}, {binary_path} reports {abi_version}"
        )
    if info["build_type"] not in BUILD_TYPES:
        raise RadarExtensionABIError(
            f"{binary_path} reports an unknown build_type {info['build_type']!r}"
        )

    architectures = info["cuda_architectures"]
    if not architectures or not all(
        isinstance(entry, str) and entry for entry in architectures
    ):
        raise RadarExtensionABIError(
            f"{binary_path} must record a non-empty list of CUDA architectures"
        )
    symbols = info["operator_symbols"]
    if not symbols or not all(isinstance(entry, str) and entry for entry in symbols):
        raise RadarExtensionABIError(
            f"{binary_path} must record a non-empty list of operator symbols"
        )
    if list(symbols) != sorted(symbols) or len(set(symbols)) != len(symbols):
        raise RadarExtensionABIError(
            f"{binary_path} must record operator_symbols sorted and unique"
        )

    git_sha = str(info["radar_git_sha"])
    if git_sha != "unknown" and _GIT_SHA_PATTERN.fullmatch(git_sha) is None:
        raise RadarExtensionABIError(
            "the recorded radar Git SHA must be 40 lowercase hex digits"
        )

    fingerprint = str(info["build_fingerprint"])
    if _SHA256_PATTERN.fullmatch(fingerprint) is None:
        raise RadarExtensionABIError(
            "the recorded build_fingerprint must be a SHA-256 digest"
        )
    recomputed = compute_build_fingerprint(info)
    if fingerprint != recomputed:
        raise RadarExtensionABIError(
            f"{binary_path} carries an invalid build_fingerprint: the record "
            "does not hash to the value it declares"
        )

    sidecar = fingerprint_sidecar_path(binary_path)
    if not sidecar.is_file():
        raise RadarExtensionLoadError(
            f"the radar native library {binary_path} has no build fingerprint "
            f"at {sidecar}; rebuild it with "
            "`python scripts/build_radar_cuda_prebuilt.py`"
        )
    try:
        declared = sidecar.read_text(encoding="ascii").strip()
    except (OSError, UnicodeDecodeError) as exc:
        raise RadarExtensionABIError(f"{sidecar} is not one ASCII digest") from exc
    if _SHA256_PATTERN.fullmatch(declared) is None:
        raise RadarExtensionABIError(f"{sidecar} must contain one SHA-256 digest")
    if declared != fingerprint:
        raise RadarExtensionABIError(
            f"{sidecar} disagrees with the build_fingerprint recorded in "
            f"{build_info_sidecar_path(binary_path)}"
        )

    binary_sha256 = str(info["binary_sha256"])
    if _SHA256_PATTERN.fullmatch(binary_sha256) is None:
        raise RadarExtensionABIError(
            "the recorded binary_sha256 must be a SHA-256 digest"
        )
    actual_binary = file_digest(binary_path)
    if actual_binary != binary_sha256:
        raise RadarExtensionABIError(
            f"{binary_path} does not match its recorded binary_sha256: "
            f"expected {binary_sha256}, found {actual_binary}"
        )

    recorded_sources = str(info["source_fingerprint"])
    if _SHA256_PATTERN.fullmatch(recorded_sources) is None:
        raise RadarExtensionABIError(
            "the recorded source_fingerprint must be a SHA-256 digest"
        )
    actual_sources = source_digest(source_paths)
    if actual_sources != recorded_sources:
        raise RadarExtensionABIError(
            f"{binary_path} was not built from the sources shipped beside it: "
            f"source_fingerprint expected {recorded_sources}, found "
            f"{actual_sources}"
        )

    live = runtime_identity()
    mismatched = sorted(
        name for name, value in live.items() if info.get(name) != value
    )
    if mismatched:
        detail = ", ".join(
            f"{name}: record {info.get(name)!r} vs runtime {live[name]!r}"
            for name in mismatched
        )
        raise RadarExtensionABIError(
            f"{binary_path} does not match the active runtime ({detail})"
        )

    if expected_fingerprint is not None and fingerprint != expected_fingerprint:
        raise RadarExtensionABIError(
            f"{binary_path} has build_fingerprint {fingerprint}, but the "
            f"expected fingerprint is {expected_fingerprint}"
        )
    return info


def collect_build_info(
    *,
    extension_name: str,
    build_type: str,
    torch_target_version: str,
    cuda_architectures: list[str],
    source_paths: Sequence[Path],
    operator_symbols: Sequence[str],
    binary_path: Path,
    repo_root: Path,
) -> dict[str, object]:
    """Assemble the identity record for a freshly built artifact.

    Called by ``scripts/build_radar_cuda_prebuilt.py`` only. The loader never
    constructs a record; it only validates one.
    """

    if build_type not in BUILD_TYPES:
        raise ValueError(f"build_type must be one of {BUILD_TYPES}, got {build_type!r}")
    git_sha, git_dirty = detect_git_identity(repo_root)
    live = runtime_identity()
    info: dict[str, object] = {
        "radar_abi_version": RADAR_ABI_VERSION,
        "extension_name": extension_name,
        "build_type": build_type,
        "torch_version": live["torch_version"],
        "torch_target_version": torch_target_version,
        "cuda_version": live["cuda_version"],
        "cuda_compiler_version": detect_cuda_compiler_version(),
        "compiler": detect_compiler(),
        "cxx_abi": live["cxx_abi"],
        "cuda_architectures": list(cuda_architectures),
        "platform_tag": live["platform_tag"],
        "python_abi": "stable-abi-v1",
        "radar_git_sha": git_sha,
        "radar_git_dirty": git_dirty,
        "source_fingerprint": source_digest(source_paths),
        "operator_symbols": sorted(operator_symbols),
        "binary_sha256": file_digest(binary_path),
    }
    info["build_fingerprint"] = compute_build_fingerprint(info)
    return info


def write_sidecars(binary_path: Path, info: Mapping[str, object]) -> tuple[Path, Path]:
    """Write both sidecars atomically enough for a build script: json then line."""

    record = {name: info[name] for name, _ in BUILD_INFO_FIELDS}
    info_path = build_info_sidecar_path(binary_path)
    info_path.write_text(
        json.dumps(record, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    fingerprint_path = fingerprint_sidecar_path(binary_path)
    fingerprint_path.write_text(
        str(record["build_fingerprint"]) + "\n", encoding="ascii"
    )
    return info_path, fingerprint_path



"""Select, validate and load the radar native library.

R-ADR-019 is the contract this file implements. In one sentence: the packaged
prebuilt is the only normal load source, every failure is loud and names the
full build identity, and ``torch.utils.cpp_extension`` is reachable only when
the build script explicitly asks for it.

That last clause is not tidiness. The just-in-time route calls
``_ensure_windows_build_tools_on_path()``, which copies the whole ``vcvars64``
environment over ``os.environ`` including ``PATH``; a library built after that
mutation fails ``DllMain`` with an access violation in the same process, and the
unbounded ``PATH`` growth surfaces much later as unrelated CUDA failures. While
the JIT route was the silent fallback for a missing or stale prebuilt, an
ordinary ``import witwin.radar.paths`` could reach it.
"""


import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch



EXTENSION_NAME = "_radar_native"

#: The Stable ABI target compiled into every translation unit. Recorded in the
#: build record so a binary built against a different target is visible.
TORCH_TARGET_VERSION = "0x020a000000000000"

_BUILD_ENV = "WITWIN_RADAR_NATIVE_BUILD"
_BUILD_DIR_ENV = "WITWIN_RADAR_NATIVE_BUILD_DIR"
_OVERRIDE_ENABLE_ENV = "WITWIN_RADAR_NATIVE_DEVELOPER_OVERRIDE"
_OVERRIDE_PATH_ENV = "WITWIN_RADAR_NATIVE_EXTENSION_PATH"
_OVERRIDE_FINGERPRINT_ENV = "WITWIN_RADAR_NATIVE_EXPECTED_FINGERPRINT"


def _candidate_vcvars64_paths() -> list[Path]:
    paths: list[Path] = []
    program_files_x86 = os.environ.get("ProgramFiles(x86)")
    if program_files_x86:
        vswhere = Path(program_files_x86) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
        if vswhere.exists():
            try:
                install_root = subprocess.check_output(
                    [
                        str(vswhere),
                        "-latest",
                        "-products",
                        "*",
                        "-requires",
                        "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                        "-property",
                        "installationPath",
                    ],
                    text=True,
                    encoding="mbcs",
                    errors="replace",
                ).strip()
            except (OSError, subprocess.CalledProcessError):
                install_root = ""
            if install_root:
                paths.append(Path(install_root) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat")

    vs_install = os.environ.get("VSINSTALLDIR")
    if vs_install:
        paths.append(Path(vs_install) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat")

    program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
    for edition in ("Community", "Professional", "Enterprise", "BuildTools"):
        paths.append(
            Path(program_files)
            / "Microsoft Visual Studio"
            / "2022"
            / edition
            / "VC"
            / "Auxiliary"
            / "Build"
            / "vcvars64.bat"
        )

    seen: set[Path] = set()
    unique_paths: list[Path] = []
    for path in paths:
        resolved = path.resolve() if path.exists() else path
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_paths.append(path)
    return unique_paths


def _load_vcvars64_environment() -> bool:
    for vcvars in _candidate_vcvars64_paths():
        if not vcvars.exists():
            continue
        fd, probe_name = tempfile.mkstemp(prefix="witwin_radar_vcvars_probe_", suffix=".cmd")
        os.close(fd)
        probe = Path(probe_name)
        probe.write_text(f'@echo off\ncall "{vcvars}" >nul\nset\n', encoding="utf-8")
        try:
            output = subprocess.check_output(
                ["cmd.exe", "/d", "/c", str(probe)],
                text=True,
                encoding="mbcs",
                errors="replace",
            )
        except (OSError, subprocess.CalledProcessError):
            continue
        finally:
            try:
                probe.unlink()
            except OSError:
                pass
        updates: dict[str, str] = {}
        for line in output.splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            updates[key] = value
        for key, value in updates.items():
            os.environ[key] = value
        path_value = updates.get("PATH", updates.get("Path"))
        if path_value is not None:
            os.environ["PATH"] = path_value
            os.environ["Path"] = path_value
        return True
    return False


def _build_requested() -> bool:
    return os.environ.get(_BUILD_ENV) == "1"


def _ensure_windows_build_tools_on_path() -> None:
    # Guarded a second time, at the point of damage. This function replaces the
    # process PATH wholesale; a caller that reaches it without asking for a
    # build has made a mistake that must not silently mutate the environment.
    if not _build_requested():
        raise RadarExtensionLoadError(
            f"preparing MSVC build tools requires {_BUILD_ENV}=1"
        )
    if os.name != "nt":
        return
    # Keep MSVC diagnostics ASCII/English so PyTorch's compiler-version probe
    # does not fail to decode localized `cl` output under a different code page.
    os.environ.setdefault("VSLANG", "1033")
    if shutil.which("cl") is None:
        _load_vcvars64_environment()
    if shutil.which("cl") is None:
        return

    current_path = os.environ.get("PATH") or os.environ.get("Path") or ""
    prefixes: list[str] = []
    vc_tools = os.environ.get("VCToolsInstallDir")
    if vc_tools:
        prefixes.append(str(Path(vc_tools) / "bin" / "Hostx64" / "x64"))
    vs_install = os.environ.get("VSINSTALLDIR")
    if vs_install:
        prefixes.append(
            str(Path(vs_install) / "Common7" / "IDE" / "CommonExtensions" / "Microsoft" / "CMake" / "Ninja")
        )
    if not prefixes:
        return
    merged_path = os.pathsep.join([*prefixes, current_path])
    os.environ["PATH"] = merged_path
    os.environ["Path"] = merged_path


def _ensure_cuda_home_from_nvcc() -> None:
    if not _build_requested():
        raise RadarExtensionLoadError(
            f"locating CUDA_HOME for a build requires {_BUILD_ENV}=1"
        )
    if os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH"):
        return
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        return
    cuda_home = Path(nvcc).resolve().parents[1]
    os.environ["CUDA_HOME"] = str(cuda_home)
    if os.name == "nt":
        os.environ["CUDA_PATH"] = str(cuda_home)


def source_root() -> Path:
    return Path(__file__).resolve().parent


def prebuilt_root() -> Path:
    return source_root() / "prebuilt"


def extension_suffix() -> str:
    return ".pyd" if os.name == "nt" else ".so"


def prebuilt_extension_path() -> Path:
    return prebuilt_root() / f"{EXTENSION_NAME}{extension_suffix()}"


def extension_sources() -> list[Path]:
    root = source_root()
    return [
        root / "extension.cpp",
        root / "fmcw_beat.cu",
        root / "fmcw_spectrum.cu",
        root / "frontend.cu",
        root / "ofdm_cfr.cu",
        root / "pulsed_echo.cu",
        root / "scatter_response.cu",
        root / "sensor_weight.cu",
        root / "two_way_join.cu",
    ]


def _cuda_gencode_flags() -> list[str]:
    """Translate the release architecture list directly into nvcc flags."""
    value = os.environ.get("WITWIN_CUDA_GENCODE_ARCHES")
    if not value:
        return []
    flags: list[str] = []
    for entry in value.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        include_ptx = entry.endswith("+PTX")
        number = entry.removesuffix("+PTX").replace(".", "")
        if not number.isdigit():
            raise ValueError(f"Invalid CUDA architecture {entry!r} in WITWIN_CUDA_GENCODE_ARCHES.")
        flags.append(f"-gencode=arch=compute_{number},code=sm_{number}")
        if include_ptx:
            flags.append(f"-gencode=arch=compute_{number},code=compute_{number}")
    return flags


def _conda_torch_ldflags() -> list[str]:
    if os.name != "nt":
        return []
    library_lib = Path(sys.prefix) / "Library" / "lib"
    if (library_lib / "c10.lib").exists():
        return [f"/LIBPATH:{library_lib}"]
    return []


class _StableOpsModule:
    """Attribute-compatible view of the dispatcher operators."""

    def __init__(
        self,
        library_path: Path,
        *,
        origin: str,
        info: dict[str, object] | None = None,
    ) -> None:
        self.__file__ = str(library_path)
        self._origin = origin
        self._info = dict(info) if info is not None else None

    def is_available(self) -> bool:
        return bool(torch.cuda.is_available())

    def build_info(self) -> dict[str, object]:
        """The validated identity record plus where the library came from.

        ``origin`` is ``packaged``, ``developer`` or ``jit``. A JIT build has no
        validated record - it was compiled in this process from these sources -
        so ``native_build`` is ``None`` there and the caller can tell the two
        situations apart instead of guessing from a missing key.
        """

        return {
            "origin": self._origin,
            "extension_path": self.__file__,
            "radar_abi_version": RADAR_ABI_VERSION,
            "native_build": dict(self._info) if self._info is not None else None,
        }

    def __getattr__(self, name: str):
        return getattr(torch.ops._radar_native, name)


# Every operator family the library is required to register. A stale binary
# that predates a family loads fine and then fails deep inside a kernel call,
# so the presence check names one operator per family and fails at load.
#
# `forward_chunked` stood first until Phase 11 deleted the `dirichlet_spectrum`
# family with its route. Leaving it here would have made every load reject the
# correct binary, which is the same defect in the other direction.
_REQUIRED_OPERATORS = (
    "fmcw_beat_forward",
    "fmcw_spectrum_forward",
    "frontend_noise_forward",
    "ofdm_cfr_forward",
    "pulsed_echo_forward",
    "scatter_response_aspect_forward",
    "sensor_weight_forward",
    "two_way_join_forward",
)


def _require_operators(
    library_path: Path,
    symbols: tuple[str, ...] | list[str] | None = None,
    *,
    origin: str = "jit",
    info: dict[str, object] | None = None,
) -> _StableOpsModule:
    """Reject a library that does not register every required operator.

    Applied to EVERY load route, including the just-in-time build. A JIT build
    can hand back a stale library too: `torch.utils.cpp_extension.load` reuses
    whatever is already linked in its build directory, so if another checkout of
    this package compiled a different source list into the same directory, the
    operators this process needs may simply not be there. Skipping the check on
    the JIT route turns that into a failure deep inside a kernel call.

    A validated artifact records the full symbol list it registers, so the
    packaged and developer routes check all of them. The seven-family sample
    above is the fallback for the JIT route, which has no record to consult.
    """

    required = tuple(symbols) if symbols is not None else _REQUIRED_OPERATORS
    missing = [
        name
        for name in required
        if not hasattr(torch.ops._radar_native, name)
    ]
    if missing:
        raise RadarExtensionSymbolError(
            f"{library_path} does not register the Stable ABI radar operators "
            f"{missing}; the binary is stale."
        )
    return _StableOpsModule(library_path, origin=origin, info=info)


def _load_validated_extension(
    library_path: Path,
    *,
    origin: str,
    expected_fingerprint: str | None = None,
) -> _StableOpsModule:
    """Validate the identity chain, then load. Never the other way round.

    Validation happens before `torch.ops.load_library` because loading a shared
    library is irreversible within the process: a mismatched binary that has
    already run its initializers cannot be unloaded, so a check afterwards would
    report a problem the process can no longer avoid.
    """

    info = validate_identity(
        library_path,
        extension_sources(),
        expected_fingerprint=expected_fingerprint,
    )
    torch.ops.load_library(str(library_path))
    return _require_operators(
        library_path,
        list(info["operator_symbols"]),
        origin=origin,
        info=info,
    )


def _developer_override_config() -> tuple[Path, str] | None:
    """All three variables together, or none of them. Never a partial set.

    A partial set is always a mistake and never a request for the default
    behaviour, so it raises even when the packaged prebuilt is present and would
    otherwise have won: silently ignoring two of three variables is how a
    developer ends up measuring the wrong binary.
    """

    enabled = os.environ.get(_OVERRIDE_ENABLE_ENV)
    raw_path = os.environ.get(_OVERRIDE_PATH_ENV)
    fingerprint = os.environ.get(_OVERRIDE_FINGERPRINT_ENV)
    if enabled is None and raw_path is None and fingerprint is None:
        return None
    if enabled != "1" or not raw_path or not fingerprint:
        raise RadarExtensionLoadError(
            "loading a developer radar native library requires all three of "
            f"{_OVERRIDE_ENABLE_ENV}=1, an absolute {_OVERRIDE_PATH_ENV}, and a "
            f"SHA-256 {_OVERRIDE_FINGERPRINT_ENV}"
        )
    if not is_sha256(fingerprint):
        raise RadarExtensionLoadError(
            f"{_OVERRIDE_FINGERPRINT_ENV} must be a SHA-256 digest"
        )
    path = Path(raw_path)
    if not path.is_absolute():
        raise RadarExtensionLoadError(f"{_OVERRIDE_PATH_ENV} must be an absolute path")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise RadarExtensionLoadError(
            f"the developer radar native library does not exist: {path}"
        ) from exc
    if not resolved.is_file() or resolved.suffix != extension_suffix():
        raise RadarExtensionLoadError(
            "the developer radar native library must be a "
            f"{extension_suffix()} file: {resolved}"
        )
    return resolved, fingerprint


# The loaded library, cached for the process.
#
# Loading is idempotent in effect but NOT in side effects: on Windows,
# _ensure_windows_build_tools_on_path() prepends the MSVC tool directories to
# PATH every time it runs, and _load_vcvars64_environment() copies the whole
# vcvars environment over os.environ. Calling build_extension() in a loop
# therefore grows PATH without bound until Windows rejects it with
# "the environment variable is longer than 32767 characters", which surfaces
# far away from the cause -- as unrelated CUDA tests failing partway through a
# long session. Caching the module makes the second and later calls free and
# side-effect free.
_LOADED_MODULE = None


def build_extension(*, verbose: bool = False):
    global _LOADED_MODULE
    if _LOADED_MODULE is not None:
        return _LOADED_MODULE
    _LOADED_MODULE = _build_extension(verbose=verbose)
    return _LOADED_MODULE


def source_fingerprint() -> str:
    """Short digest of the source set: which files, and what is in them.

    The JIT build directory is keyed by this. Two checkouts of this package that
    compile different sources - or the same sources at different revisions - into
    one shared directory make ninja relink on every alternating run, and the
    binary that happens to be on disk is whichever checkout ran last. That is a
    silent wrong-numerics path, not just wasted work: a stale link only fails
    loudly when the missing operator is missing entirely, and two revisions of
    the SAME operator set register the same names with different kernel code.

    File CONTENT, not just paths, because two worktrees of one branch have
    different absolute paths but should share a build, while one path with an
    edited kernel must not.
    """

    return source_digest(extension_sources())[:16]


def default_build_directory() -> Path:
    return (
        Path(tempfile.gettempdir())
        / EXTENSION_NAME
        / f"stable_abi_v1_{source_fingerprint()}"
    )


def _build_extension(*, verbose: bool = False):
    """Resolve exactly one load source, or fail naming what would fix it.

    Order, and the reason for it:

    1. the developer override configuration is READ first, because a partial set
       of its three variables is a mistake in every situation and must not be
       silently ignored just because a packaged prebuilt happens to be present;
    2. ``WITWIN_RADAR_NATIVE_BUILD=1`` means "compile from these sources", so it
       bypasses the packaged artifact entirely. That is how
       ``scripts/build_radar_cuda_prebuilt.py`` replaces a prebuilt that the
       loader would otherwise refuse - a stale artifact must not be able to
       block the command that fixes it;
    3. a present packaged prebuilt wins over the override, matching Channel's
       ADR-006 precedence - in a source checkout the packaged prebuilt IS the
       developer artifact, and refreshing it is the supported dev flow;
    4. otherwise the fully specified developer override;
    5. otherwise a loud failure naming the three override variables.

    There is no branch that answers a validation failure with a rebuild, and no
    branch that returns ``None``.
    """

    override = _developer_override_config()

    if _build_requested():
        return _jit_build_extension(verbose=verbose)

    packaged = prebuilt_extension_path()
    if packaged.exists():
        return _load_validated_extension(packaged, origin="packaged")

    if override is not None:
        override_path, expected_fingerprint = override
        return _load_validated_extension(
            override_path,
            origin="developer",
            expected_fingerprint=expected_fingerprint,
        )

    raise RadarExtensionLoadError(
        f"no radar native library is available: {packaged} does not exist. "
        "Build the packaged prebuilt with "
        "`python scripts/build_radar_cuda_prebuilt.py`, or point at an existing "
        f"one with all three of {_OVERRIDE_ENABLE_ENV}=1, an absolute "
        f"{_OVERRIDE_PATH_ENV}, and a SHA-256 {_OVERRIDE_FINGERPRINT_ENV}. "
        f"Compiling from source requires {_BUILD_ENV}=1 and must never happen "
        "inside a test or user process."
    )


def _jit_build_extension(*, verbose: bool = False):
    """Compile from source. Reachable only from the build script.

    ``torch.utils.cpp_extension`` is imported HERE and not at module scope, so
    an ordinary import of this module cannot pull the compiler machinery into
    the process, and a test can assert that by looking at ``sys.modules``.
    """

    from torch.utils.cpp_extension import load

    root = source_root()
    build_directory = Path(
        os.environ.get(_BUILD_DIR_ENV, default_build_directory())
    )
    _ensure_windows_build_tools_on_path()
    _ensure_cuda_home_from_nvcc()
    build_directory.mkdir(parents=True, exist_ok=True)
    target_flag = f"TORCH_TARGET_VERSION={TORCH_TARGET_VERSION}"
    library_path = load(
        name=EXTENSION_NAME,
        sources=[str(path) for path in extension_sources()],
        build_directory=str(build_directory),
        extra_include_paths=[str(root)],
        extra_cflags=(
            ["/O2", f"/D{target_flag}"]
            if os.name == "nt"
            else ["-O3", f"-D{target_flag}"]
        ),
        extra_cuda_cflags=[
            "-O3",
            f"-D{target_flag}",
            "-DUSE_CUDA",
            *_cuda_gencode_flags(),
        ],
        extra_ldflags=_conda_torch_ldflags(),
        is_python_module=False,
        verbose=verbose,
    )
    return _require_operators(Path(library_path))


