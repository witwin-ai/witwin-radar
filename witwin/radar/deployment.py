"""Build identity and runtime diagnostics for the packaged radar extension.

Phase-10 acceptance criterion A3 asks that the Channel and Radar extensions
each fail loudly AND report their full build identity. Channel has had this
since ADR-006 (``witwin/channel/deployment.py``); this is the Radar half, in
the same shape so the two records can be read side by side.

Three entry points, and the difference between them is the whole design:

* :func:`build_info` LOADS the extension and therefore fails loudly. It is the
  answer to "what exactly is installed here", and there is no such answer if
  nothing can be loaded.
* :func:`runtime_diagnostics` never raises. It is what a bug report pastes,
  and a diagnostic that dies on a broken runtime is useless precisely when it
  is needed. Every failure becomes an entry in ``errors``.
* :func:`require_supported_runtime` is the loud precondition: CUDA present, an
  active device whose SM is declared, and SASS for that SM actually compiled
  into the installed binary.

R-ADR-019 owns the identity chain itself; this module only reports it.
"""

from __future__ import annotations

import importlib.metadata
import platform
import sys
from typing import Any


DEPLOYMENT_ABI = "witwin.radar.deployment.v1"

#: The architectures the release build compiles SASS for, mirroring
#: ``scripts/verify_cuda_binary_arches.py::EXPECTED_SASS`` and the
#: ``WITWIN_CUDA_GENCODE_ARCHES`` value in ``publish-witwin-radar.yml``. Kept
#: as one list here and pinned against the verifier by the Phase-10 tests, so
#: a release that changed its gencode list without telling the runtime record
#: fails rather than reporting a stale matrix.
DECLARED_SM_ARCHITECTURES = (70, 75, 80, 86, 87, 89, 90, 100, 101, 120)

#: The architectures a Radar test run has actually executed on. Everything else
#: is declared-and-unverified, which is a different claim and is reported as
#: one. SM87 runtime validation needs Orin/Jetson hardware and is a named
#: Phase-10 deferral, not a silent gap.
VERIFIED_SM_ARCHITECTURES = (120,)

#: The one architecture whose PTX is embedded for forward compatibility.
PTX_FORWARD_COMPATIBILITY_SM = 120

SM120_EVIDENCE = (
    "tests --gpu on sm_120 via .github/workflows/gpu-regression.yml; "
    "SASS presence for every declared architecture is checked by "
    "scripts/verify_cuda_binary_arches.py"
)


def _package_version() -> str:
    try:
        return importlib.metadata.version("witwin-radar")
    except importlib.metadata.PackageNotFoundError:
        return "source-tree"


def sm_support(sm: int) -> dict[str, Any]:
    """Declared / verified / unavailable for one architecture."""

    sm = int(sm)
    declared = sm in DECLARED_SM_ARCHITECTURES
    verified = sm in VERIFIED_SM_ARCHITECTURES
    return {
        "sm": sm,
        "declared_supported": declared,
        "runtime_verified": verified,
        "status": (
            "runtime_verified"
            if verified
            else "declared_unverified"
            if declared
            else "not_declared"
        ),
        "evidence": [SM120_EVIDENCE] if verified else [],
        "mode": (
            "sass+ptx"
            if sm == PTX_FORWARD_COMPATIBILITY_SM
            else "sass"
            if declared
            else "not_available"
        ),
    }


def build_info() -> dict[str, Any]:
    """The validated identity of the radar native library, or a loud failure.

    Loads the extension through the one owner that is allowed to
    (:mod:`witwin.radar.cuda.runtime`), so the record returned is the record that
    was validated before ``torch.ops.load_library`` ran - not a re-read of the
    sidecar, which would answer a different question.
    """

    from .cuda import runtime as build

    return build.build_extension().build_info()


def _torch_diagnostics(diagnostics: dict[str, Any]) -> None:
    import torch

    diagnostics["torch"] = torch.__version__
    diagnostics["cuda_runtime"] = torch.version.cuda
    cuda_available = bool(torch.cuda.is_available())
    diagnostics["cuda_available"] = cuda_available
    if not cuda_available:
        return
    index = int(torch.cuda.current_device())
    if index < 0:
        raise RuntimeError(
            f"CUDA reports available but has invalid active device index {index}"
        )
    major, minor = torch.cuda.get_device_capability(index)
    support = sm_support(major * 10 + minor)
    diagnostics["device"] = {
        "name": torch.cuda.get_device_name(index),
        "total_memory_bytes": int(
            torch.cuda.get_device_properties(index).total_memory
        ),
        **support,
    }
    diagnostics["sm_matrix_status"] = support["status"]


def runtime_diagnostics() -> dict[str, Any]:
    """Report the runtime. Never raise, whatever is broken.

    The two collectors below catch ``Exception`` rather than a narrow tuple on
    purpose. A missing extension raises ``RadarExtensionLoadError`` (an
    ``ImportError``), but a mismatched driver raises ``RuntimeError`` from
    inside Torch and a corrupted record raises ``ValueError``; a diagnostics
    function that reports the first two and dies on the third has the failure
    mode it exists to remove.
    """

    from .cuda.runtime import RADAR_ABI_VERSION

    diagnostics: dict[str, Any] = {
        "deployment_abi": DEPLOYMENT_ABI,
        "radar_abi_version": RADAR_ABI_VERSION,
        "package_version": _package_version(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "declared_sm_architectures": list(DECLARED_SM_ARCHITECTURES),
        "verified_sm_architectures": list(VERIFIED_SM_ARCHITECTURES),
        "sm_matrix_status": "declared_unverified",
        "ptx_forward_compatibility_sm": PTX_FORWARD_COMPATIBILITY_SM,
        "native_build": None,
        "errors": [],
    }
    try:
        _torch_diagnostics(diagnostics)
    except Exception as error:  # noqa: BLE001 - see the docstring
        diagnostics["errors"].append(
            f"PyTorch runtime unavailable ({type(error).__name__}): {error}"
        )
    try:
        diagnostics["native_build"] = build_info()
    except Exception as error:  # noqa: BLE001 - see the docstring
        diagnostics["errors"].append(
            "radar native extension unavailable; install a matching "
            "witwin-radar wheel or rebuild the packaged prebuilt with "
            "`python scripts/build_radar_cuda_prebuilt.py`; reason "
            f"({type(error).__name__}): {error}"
        )
    return diagnostics


def require_supported_runtime() -> dict[str, Any]:
    """Require CUDA, a declared device architecture, and SASS for it."""

    diagnostics = runtime_diagnostics()
    errors = list(diagnostics["errors"])
    if not diagnostics.get("cuda_available", False):
        errors.append("CUDA is unavailable; Radar has no CPU backend")
    device = diagnostics.get("device")
    if diagnostics.get("cuda_available", False) and not isinstance(device, dict):
        errors.append(
            "CUDA is available but runtime diagnostics has no valid active device"
        )
    if isinstance(device, dict):
        if not device.get("declared_supported", False):
            errors.append(
                f"GPU SM {device.get('sm')} is outside the declared build SM "
                f"values {list(DECLARED_SM_ARCHITECTURES)}"
            )
        else:
            native_build = diagnostics.get("native_build")
            record = (
                native_build.get("native_build")
                if isinstance(native_build, dict)
                else None
            )
            architectures = (
                record.get("cuda_architectures") if isinstance(record, dict) else None
            )
            sm = device.get("sm")
            if not isinstance(architectures, list) or not any(
                entry.split("+", maxsplit=1)[0] == str(sm) for entry in architectures
            ):
                errors.append(
                    f"the installed radar native library contains no sm_{sm} "
                    f"image; compiled CUDA architectures are {architectures!r}"
                )
    if errors:
        raise RuntimeError("Radar runtime requirements failed: " + "; ".join(errors))
    return diagnostics


__all__ = ["build_info", "require_supported_runtime", "runtime_diagnostics"]
