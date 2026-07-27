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
ordinary ``import witwin.radar.paths.two_way`` could reach it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from . import identity
from .identity import (
    RADAR_ABI_VERSION,
    RadarExtensionABIError,
    RadarExtensionLoadError,
    RadarExtensionSymbolError,
)


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
        root / "kernels" / "dirichlet.cu",
        root / "kernels" / "fmcw_beat.cu",
        root / "kernels" / "frontend.cu",
        root / "kernels" / "ofdm_cfr.cu",
        root / "kernels" / "pulsed_echo.cu",
        root / "kernels" / "scatter_response.cu",
        root / "kernels" / "sensor_weight.cu",
        root / "kernels" / "two_way_join.cu",
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
_REQUIRED_OPERATORS = (
    "forward_chunked",
    "fmcw_beat_forward",
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
    packaged and developer routes check all of them. The eight-family sample
    below is the fallback for the JIT route, which has no record to consult.
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

    info = identity.validate_identity(
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
    if not identity.is_sha256(fingerprint):
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

    return identity.source_digest(extension_sources())[:16]


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
        extra_include_paths=[str(root / "kernels")],
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


__all__ = [
    "EXTENSION_NAME",
    "RADAR_ABI_VERSION",
    "TORCH_TARGET_VERSION",
    "RadarExtensionABIError",
    "RadarExtensionLoadError",
    "RadarExtensionSymbolError",
    "build_extension",
    "default_build_directory",
    "extension_sources",
    "extension_suffix",
    "prebuilt_extension_path",
    "prebuilt_root",
    "source_fingerprint",
    "source_root",
]
