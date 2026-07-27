from __future__ import annotations

import os
import sysconfig
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


#: The one native artifact. Matched by exact name plus a known suffix rather
#: than by ``_radar_native.*``, which also matches the two identity sidecars
#: and would let a binary-free directory look populated.
EXTENSION_NAME = "_radar_native"

#: Explicit opt-out for the two builds that legitimately have no binary: an
#: sdist-driven or documentation build. It is a variable and not a silent
#: fallback because "no prebuilt" and "no prebuilt on purpose" are different
#: states and only one of them may publish.
ALLOW_PURE_WHEEL_ENV = "WITWIN_RADAR_ALLOW_PURE_WHEEL"


class CustomBuildHook(BuildHookInterface):
    """Tag the wheel for this platform, and refuse to build a native-free one.

    Before Phase 10 this hook returned silently when no prebuilt existed, and
    hatchling then emitted a perfectly valid-looking ``py3-none-any`` wheel with
    no native member in it. A release run whose prebuilt step failed without
    stopping the job would publish that wheel, and every install of it would
    fail at first import rather than at build time. The failure now happens
    where the artifact is made.
    """

    def initialize(self, version: str, build_data: dict) -> None:
        del version
        if self.target_name != "wheel":
            return
        prebuilt_dir = Path(self.root) / "witwin" / "radar" / "cuda" / "prebuilt"
        binaries = sorted(
            path
            for suffix in (".pyd", ".so")
            for path in prebuilt_dir.glob(f"{EXTENSION_NAME}{suffix}")
        )
        if not binaries:
            if os.environ.get(ALLOW_PURE_WHEEL_ENV) == "1":
                return
            raise RuntimeError(
                f"no packaged radar extension in {prebuilt_dir}: expected "
                f"{EXTENSION_NAME}.pyd or {EXTENSION_NAME}.so. Build it with "
                "`python scripts/build_radar_cuda_prebuilt.py` before building "
                f"the wheel, or set {ALLOW_PURE_WHEEL_ENV}=1 to deliberately "
                "produce a py3-none-any wheel that cannot load the native "
                "library."
            )
        if len(binaries) != 1:
            raise RuntimeError(
                f"multiple packaged radar extensions in {prebuilt_dir}: "
                f"{[path.name for path in binaries]}. The wheel must contain "
                "exactly one native member."
            )
        binary = binaries[0]
        stem = binary.name[: -len(binary.suffix)]
        missing = [
            name
            for name in (f"{stem}.build-info.json", f"{stem}.build-fingerprint")
            if not (prebuilt_dir / name).exists()
        ]
        if missing:
            raise RuntimeError(
                f"packaged radar extension {binary.name} has no build identity: "
                f"missing {missing}. R-ADR-019 validates the binary against both "
                "sidecars before loading it, so a wheel without them cannot "
                "import. Rebuild with `python scripts/build_radar_cuda_prebuilt.py`."
            )
        platform_tag = sysconfig.get_platform().replace("-", "_").replace(".", "_")
        build_data["tag"] = f"py3-none-{platform_tag}"
        build_data["pure_python"] = False
