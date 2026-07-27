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
RADAR_ABI_VERSION = 1

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


__all__ = [
    "BUILD_INFO_FIELDS",
    "BUILD_INFO_SUFFIX",
    "BUILD_TYPES",
    "FINGERPRINT_FIELDS",
    "FINGERPRINT_SUFFIX",
    "RADAR_ABI_VERSION",
    "RadarExtensionABIError",
    "RadarExtensionLoadError",
    "RadarExtensionSymbolError",
    "build_info_sidecar_path",
    "collect_build_info",
    "compute_build_fingerprint",
    "cxx_abi",
    "detect_compiler",
    "detect_cuda_compiler_version",
    "detect_git_identity",
    "file_digest",
    "is_sha256",
    "fingerprint_sidecar_path",
    "normalize_cuda_architectures",
    "platform_tag",
    "read_build_info",
    "runtime_identity",
    "source_digest",
    "validate_identity",
    "write_sidecars",
]
