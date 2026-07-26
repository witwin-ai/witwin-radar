from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


EXTENSION_NAME = "witwin_radar_dirichlet_cuda"


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


def _ensure_windows_build_tools_on_path() -> None:
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

    def __init__(self, library_path: Path) -> None:
        self.__file__ = str(library_path)

    def is_available(self) -> bool:
        return bool(torch.cuda.is_available())

    def __getattr__(self, name: str):
        return getattr(torch.ops.witwin_radar_dirichlet_cuda, name)


# Every operator family the library is required to register. A stale binary
# that predates a family loads fine and then fails deep inside a kernel call,
# so the presence check names one operator per family and fails at load.
_REQUIRED_OPERATORS = (
    "forward_chunked",
    "fmcw_beat_forward",
    "frontend_noise_forward",
    "ofdm_cfr_forward",
    "pulsed_echo_forward",
    "sensor_weight_forward",
    "two_way_join_forward",
)


def _require_operators(library_path: Path) -> _StableOpsModule:
    """Reject a library that does not register every required family.

    Applied to EVERY load route, including the just-in-time build. A JIT build
    can hand back a stale library too: `torch.utils.cpp_extension.load` reuses
    whatever is already linked in its build directory, so if another checkout of
    this package compiled a different source list into the same directory, the
    operators this process needs may simply not be there. Skipping the check on
    the JIT route turns that into a failure deep inside a kernel call.
    """

    missing = [
        name
        for name in _REQUIRED_OPERATORS
        if not hasattr(torch.ops.witwin_radar_dirichlet_cuda, name)
    ]
    if missing:
        raise ImportError(
            f"{library_path} does not register the Stable ABI radar operators "
            f"{missing}; the binary is stale."
        )
    return _StableOpsModule(library_path)


def _load_extension_file(library_path: Path) -> _StableOpsModule:
    torch.ops.load_library(str(library_path))
    return _require_operators(library_path)


def _load_packaged_prebuilt_extension():
    module_path = prebuilt_extension_path()
    if not module_path.exists():
        return None
    return _load_extension_file(module_path)


def _load_prebuilt_extension(build_directory: Path):
    module_path = build_directory / f"{EXTENSION_NAME}{extension_suffix()}"
    if not module_path.exists():
        raise FileNotFoundError(
            f"WITWIN_RADAR_DIRICHLET_CUDA_PREBUILT=1 but {module_path} does not exist; "
            "run a normal build first."
        )
    return _load_extension_file(module_path)


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

    import hashlib

    digest = hashlib.sha256()
    for path in extension_sources():
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def default_build_directory() -> Path:
    return (
        Path(tempfile.gettempdir())
        / "witwin_radar_dirichlet_cuda"
        / f"stable_abi_v1_{source_fingerprint()}"
    )


def _build_extension(*, verbose: bool = False):
    root = source_root()
    build_directory = Path(
        os.environ.get(
            "WITWIN_RADAR_DIRICHLET_CUDA_BUILD_DIR", default_build_directory()
        )
    )
    if os.environ.get("WITWIN_RADAR_DIRICHLET_CUDA_SKIP_PREBUILT") != "1":
        try:
            module = _load_packaged_prebuilt_extension()
        except Exception:  # noqa: BLE001 - stale/ABI-mismatched prebuilt, rebuild instead
            module = None
        if module is not None:
            return module
    if os.environ.get("WITWIN_RADAR_DIRICHLET_CUDA_PREBUILT") == "1":
        return _load_prebuilt_extension(build_directory)

    _ensure_windows_build_tools_on_path()
    _ensure_cuda_home_from_nvcc()
    build_directory.mkdir(parents=True, exist_ok=True)
    library_path = load(
        name=EXTENSION_NAME,
        sources=[str(path) for path in extension_sources()],
        build_directory=str(build_directory),
        extra_include_paths=[str(root / "kernels")],
        extra_cflags=(
            ["/O2", "/DTORCH_TARGET_VERSION=0x020a000000000000"]
            if os.name == "nt"
            else ["-O3", "-DTORCH_TARGET_VERSION=0x020a000000000000"]
        ),
        extra_cuda_cflags=[
            "-O3",
            "-DTORCH_TARGET_VERSION=0x020a000000000000",
            "-DUSE_CUDA",
            *_cuda_gencode_flags(),
        ],
        extra_ldflags=_conda_torch_ldflags(),
        is_python_module=False,
        verbose=verbose,
    )
    return _require_operators(Path(library_path))
