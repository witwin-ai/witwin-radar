"""Build the packaged radar native library and stamp its identity.

This script is the ONLY supported way to reach
``torch.utils.cpp_extension.load``: it is what sets
``WITWIN_RADAR_NATIVE_BUILD=1``. Run it in a throwaway process, never inside a
test or user process - the MSVC environment it prepares breaks a library loaded
later in the same process (R-ADR-019).

It refuses to publish an artifact it cannot vouch for: every operator in
``ci/native-binding-manifest.json`` must resolve on the freshly built library
before either sidecar is written.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import types

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "ci" / "native-binding-manifest.json"


def _ensure_current_device_arch() -> None:
    if not torch.cuda.is_available():
        return
    major, minor = torch.cuda.get_device_capability()
    current_arch = f"{major}.{minor}"
    arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "")
    entries = [entry.strip() for entry in arch_list.split(";") if entry.strip()]
    normalized = {entry.removesuffix("+PTX") for entry in entries}
    if current_arch not in normalized:
        entries.append(current_arch)
        os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(entries)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_cuda_build_module():
    """Load ``build.py`` as a standalone module, exactly as before.

    It imports ``identity`` as a sibling, so ``identity`` is registered under
    the package name ``build`` expects before ``build`` executes.
    """

    cuda_dir = REPO_ROOT / "witwin" / "radar" / "cuda"
    package = types.ModuleType("witwin_radar_cuda_build_pkg")
    package.__path__ = [str(cuda_dir)]
    sys.modules["witwin_radar_cuda_build_pkg"] = package
    package.identity = _load_module(
        "witwin_radar_cuda_build_pkg.identity", cuda_dir / "identity.py"
    )
    package.build = _load_module(
        "witwin_radar_cuda_build_pkg.build", cuda_dir / "build.py"
    )
    return package.build


def _manifest_symbols() -> list[str]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    symbols = sorted({entry["symbol"] for entry in manifest["operators"]})
    if not symbols:
        raise SystemExit(f"{MANIFEST} declares no operators.")
    return symbols


def _resolved_cuda_architectures(identity) -> list[str]:
    raw = os.environ.get("WITWIN_CUDA_GENCODE_ARCHES") or os.environ.get(
        "TORCH_CUDA_ARCH_LIST", ""
    )
    architectures = identity.normalize_cuda_architectures(raw)
    if not architectures:
        raise SystemExit(
            "No CUDA architectures to record. Set WITWIN_CUDA_GENCODE_ARCHES "
            "(release list) or TORCH_CUDA_ARCH_LIST, or run on a CUDA device so "
            "the current architecture can be detected."
        )
    return architectures


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the packaged radar native library and stamp its identity."
    )
    parser.add_argument("--verbose", action="store_true")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--developer",
        dest="build_type",
        action="store_const",
        const="developer",
        help="stamp build_type=developer (the default)",
    )
    group.add_argument(
        "--release",
        dest="build_type",
        action="store_const",
        const="release",
        help="stamp build_type=release; only a locked release build may claim this",
    )
    parser.set_defaults(build_type="developer")
    args = parser.parse_args()

    cuda_build = _load_cuda_build_module()
    identity = cuda_build.identity

    build_dir = Path(
        os.environ.get(
            "WITWIN_RADAR_NATIVE_BUILD_DIR",
            Path(tempfile.gettempdir())
            / f"{cuda_build.EXTENSION_NAME}_wheel"
            / "stable_abi_v1",
        )
    )
    os.environ["WITWIN_RADAR_NATIVE_BUILD_DIR"] = str(build_dir)
    os.environ["WITWIN_RADAR_NATIVE_BUILD"] = "1"
    _ensure_current_device_arch()

    symbols = _manifest_symbols()
    architectures = _resolved_cuda_architectures(identity)

    module = cuda_build.build_extension(verbose=args.verbose)
    module_file = Path(module.__file__).resolve()

    # Refuse to publish a library the manifest does not describe. Doing this on
    # the freshly built object, before anything is copied, keeps a partial
    # artifact out of the package directory entirely.
    missing = [
        name
        for name in symbols
        if not hasattr(torch.ops.witwin_radar_dirichlet_cuda, name)
    ]
    if missing:
        raise SystemExit(
            f"The freshly built library {module_file} does not register "
            f"{missing}; refusing to publish it. Update "
            f"{MANIFEST.relative_to(REPO_ROOT)} or fix the sources."
        )

    target_dir = cuda_build.prebuilt_root()
    target_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (".pyd", ".so"):
        stale = target_dir / f"{cuda_build.EXTENSION_NAME}{suffix}"
        for path in (
            stale,
            identity.build_info_sidecar_path(stale),
            identity.fingerprint_sidecar_path(stale),
        ):
            if path.exists():
                path.unlink()

    target = cuda_build.prebuilt_extension_path()
    shutil.copy2(module_file, target)

    info = identity.collect_build_info(
        extension_name=cuda_build.EXTENSION_NAME,
        build_type=args.build_type,
        torch_target_version=cuda_build.TORCH_TARGET_VERSION,
        cuda_architectures=architectures,
        source_paths=cuda_build.extension_sources(),
        operator_symbols=symbols,
        binary_path=target,
        repo_root=REPO_ROOT,
    )
    info_path, fingerprint_path = identity.write_sidecars(target, info)

    # Validate what was just written, with the same function the loader uses.
    identity.validate_identity(target, cuda_build.extension_sources())

    print(f"Built radar native library: {target}")
    print(f"  build_type        {info['build_type']}")
    print(f"  build_fingerprint {info['build_fingerprint']}")
    print(f"  binary_sha256     {info['binary_sha256']}")
    print(f"  operators         {len(symbols)}")
    print(f"  record            {info_path}")
    print(f"  fingerprint       {fingerprint_path}")


if __name__ == "__main__":
    main()
