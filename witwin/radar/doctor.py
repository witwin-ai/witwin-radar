"""``witwin-radar-doctor``: can this machine run witwin-radar, and if not, what to do.

``python -m witwin.radar doctor`` and the ``witwin-radar-doctor`` script run
:func:`main`. It reports, and gives a verdict on, four things in order:

1. the Python, PyTorch, CUDA runtime and NVIDIA driver identity;
2. the radar native library: its build identity (the loader's own validated
   record, :func:`witwin.radar.deployment.build_info`) and whether it carries
   SASS for the active GPU (:func:`witwin.radar.deployment.runtime_failures`);
3. the optional Channel installation: its version against the range this
   Radar reads (``witwin-channel>=0.6.0,<0.7``) and Channel's own doctor
   verdict, which runs in a child process so a broken Channel cannot take the
   radar report down with it;
4. a tiny end-to-end :meth:`~witwin.radar.Radar.simulate` on the GPU - one
   point target in an empty world - checked against its own range bin.

Every failure carries the step that fixes it. The exit status is 0 when Radar
can run here (warnings allowed) and 1 when it cannot.

The report is assembled by :func:`doctor` (a JSON-ready record, never raises)
and formatted by :func:`doctor_report`, the same split Channel's doctor uses.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import platform
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .deployment import package_version, runtime_diagnostics, runtime_failures

#: The Channel releases this Radar reads, ``[minimum, bound)``: the ``channel`` extra's range.
CHANNEL_SUPPORTED = ((0, 6, 0), (0, 7, 0))

#: How long Channel's own doctor may run in its child process, s.
CHANNEL_DOCTOR_TIMEOUT_S = 180.0

_TORCH_INSTALL = "pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128"

#: The smoke target's range, m, and the waveform it is measured with: 64 samples
#: of a 30 MHz/us ramp at 10 MS/s, 0.78 m range bins, one transmitter and receiver.
_SMOKE_RANGE_M = 5.0


def _nvidia_driver_release() -> str | None:
    """The NVIDIA kernel driver release (for example ``570.86``), when it can be read."""

    try:
        match = re.search(r"\s(\d+\.\d+(?:\.\d+)?)\s", Path("/proc/driver/nvidia/version").read_text(encoding="utf-8"))
    except OSError:
        match = None
    if match is not None:
        return match.group(1)
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    lines = completed.stdout.strip().splitlines()
    return lines[0].strip() if completed.returncode == 0 and lines else None


def _release(text: str) -> tuple[int, ...]:
    """The leading numeric release of a version string: ``0.6.0rc1`` is ``(0, 6, 0)``."""

    parts = []
    for piece in text.split("+", 1)[0].split("."):
        digits = re.match(r"\d+", piece)
        if digits is None:
            break
        parts.append(int(digits.group(0)))
        if digits.group(0) != piece:
            break
    return tuple(parts)


def channel_diagnostics(*, run_doctor: bool = True) -> dict[str, Any]:
    """The optional Channel installation and its doctor verdict. Never raises.

    Channel is located without importing it (``importlib.util.find_spec``) and
    its doctor runs as ``python -m witwin.channel doctor`` in a child process:
    this module is not Channel's consumer boundary (:mod:`witwin.radar.channel`
    is), and a Channel whose native library aborts the process must still leave
    a radar report behind.
    """

    record: dict[str, Any] = {"installed": False}
    try:
        found = importlib.util.find_spec("witwin.channel") is not None
    except (ImportError, ValueError):
        found = False
    if not found:
        return record
    record["installed"] = True
    try:
        version = importlib.metadata.version("witwin-channel")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    record["version"] = version
    low, high = CHANNEL_SUPPORTED
    record["supported"] = low <= _release(version) < high
    if not run_doctor:
        return record
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "witwin.channel", "doctor"],
            capture_output=True,
            text=True,
            timeout=CHANNEL_DOCTOR_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        record.update(status=None, verdict=f"Channel's doctor did not finish ({type(error).__name__}): {error}")
        return record
    lines = completed.stdout.splitlines()
    verdict = next((line for line in lines if line.startswith("Verdict:")), None)
    record["status"] = completed.returncode
    record["verdict"] = verdict or (completed.stderr.strip().splitlines() or ["no verdict printed"])[-1]
    record["findings"] = [line.strip()[2:] for line in lines if line.startswith(("  - ", "  ! "))]
    return record


def smoke_test() -> dict[str, Any]:
    """Simulate one point target in an empty world on the GPU and find it. Never raises.

    The radar looks down ``-z`` at a 1 m^2 target ``_SMOKE_RANGE_M`` away; the
    published range spectrum must peak within one range bin of it. This runs
    every layer once - Core, Channel's discovery and replay, the join, the FMCW
    spectrum kernel and the processing axes - in well under a second of GPU time.
    """

    try:
        import torch
        from witwin.core import Scene

        from .radar import Fmcw, Radar
        from .targets import PointTargets

        radar = Radar(
            reference_frequency_hz=77e9,
            waveform=Fmcw(
                slope=30e12,
                sample_rate=10e6,
                samples_per_chirp=64,
                chirps_per_frame=8,
                adc_start=1e-6,
                idle=5e-6,
                ramp_end=10e-6,
            ),
            tx=((0.0, 0.0, 0.0),),
            rx=((0.0, 0.0, 0.0),),
            power=12.0,
            antenna_unit="half_wavelength",
        )
        targets = PointTargets(positions=[(0.0, 0.0, -_SMOKE_RANGE_M)], rcs=1.0)
        result = radar.simulate(Scene(), targets, times=[0.0], los=True, reflections=0)
        cube = result.cube
        if not bool(torch.isfinite(torch.view_as_real(cube)).all()):
            return {"ran": True, "passed": False, "detail": "the simulated cube holds a non-finite sample"}
        profile = cube.abs().reshape(-1, cube.shape[-1]).sum(dim=0)
        peak = int(profile.argmax())
        measured = float(result.axes.range_m[peak])
        passed = abs(measured - _SMOKE_RANGE_M) <= result.axes.range_bin_m and float(profile[peak]) > 0.0
        detail = (
            f"target at {_SMOKE_RANGE_M:.2f} m, range peak at {measured:.2f} m "
            f"({result.axes.range_bin_m:.2f} m bins) on {cube.device}"
        )
        return {"ran": True, "passed": passed, "detail": detail}
    except Exception as error:  # noqa: BLE001 - the doctor reports every failure it meets
        return {"ran": True, "passed": False, "detail": f"{type(error).__name__}: {error}"}


def doctor(*, smoke: bool = True, channel_doctor: bool = True) -> dict[str, Any]:
    """Collect every check into one JSON-ready record. Never raises.

    ``failures`` and ``warnings`` are the verdict's inputs: each entry is a
    sentence that says what is wrong and how to fix it.
    """

    try:
        diagnostics = runtime_diagnostics()
    except Exception as error:  # noqa: BLE001 - the doctor reports every failure it meets
        diagnostics = {
            "package_version": package_version(),
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "errors": [f"runtime diagnostics failed ({type(error).__name__}): {error}"],
        }
    failures = _runtime_findings(diagnostics)
    warnings: list[str] = []
    device = diagnostics.get("device")
    if isinstance(device, dict) and device.get("status") == "declared_unverified":
        warnings.append(
            f"sm_{device.get('sm')} is a declared build target that no Radar test run has executed on; "
            "results are expected to hold but are not release-verified on this architecture."
        )

    channel = channel_diagnostics(run_doctor=channel_doctor)
    if not channel["installed"]:
        warnings.append(
            "witwin-channel is not installed, so Radar cannot simulate a scene (trace, simulate, stream). "
            "Install it with: pip install 'witwin-radar[channel]'"
        )
    else:
        if not channel["supported"]:
            failures.append(
                f"witwin-channel {channel['version']} is outside the range this Radar reads (>=0.6.0,<0.7). "
                "Install a matching release: pip install 'witwin-channel>=0.6.0,<0.7'"
            )
        if channel_doctor and channel.get("status") != 0:
            failures.append(
                f"Channel's own doctor fails: {channel.get('verdict')}"
                + "".join(f"; {finding}" for finding in channel.get("findings", []))
                + ". Run `python -m witwin.channel doctor` and follow its fixes."
            )

    smoke_record: dict[str, Any] = {"ran": False, "passed": False, "detail": "not run"}
    if not smoke:
        smoke_record["detail"] = "skipped by --no-smoke"
    elif failures:
        smoke_record["detail"] = "skipped: the checks above already fail"
    elif not channel["installed"]:
        smoke_record["detail"] = "skipped: a scene simulation needs witwin-channel"
    else:
        smoke_record = smoke_test()
        if not smoke_record["passed"]:
            failures.append(
                f"the end-to-end simulate smoke test failed ({smoke_record['detail']}). Every component check "
                "passed, so this is a defect: report it with the output of `witwin-radar-doctor --json`."
            )
    if smoke and not smoke_record["ran"] and not failures:
        warnings.append(f"the end-to-end smoke test did not run ({smoke_record['detail']}).")

    return {
        **diagnostics,
        "driver_release": _nvidia_driver_release(),
        "channel": channel,
        "smoke": smoke_record,
        "failures": failures,
        "warnings": warnings,
    }


def _runtime_findings(diagnostics: dict[str, Any]) -> list[str]:
    """The runtime refusals, each with the step that fixes it."""

    torch_version = diagnostics.get("torch")
    if torch_version is None:
        return [f"PyTorch is not importable. Install the CUDA build: {_TORCH_INSTALL}", *diagnostics.get("errors", [])]
    if diagnostics.get("cuda_runtime") is None:
        return [f"PyTorch {torch_version} is a CPU-only build. Install the CUDA build: {_TORCH_INSTALL}"]
    findings = []
    for failure in runtime_failures(diagnostics):
        if failure.startswith("CUDA is unavailable"):
            failure += (
                ". PyTorch sees no GPU: check `nvidia-smi`, CUDA_VISIBLE_DEVICES, and in a container run "
                "with `--gpus all`."
            )
        elif "outside the declared build SM values" in failure or "contains no sm_" in failure:
            sm = (diagnostics.get("device") or {}).get("sm")
            failure += (
                f". Build the native library for this GPU from a source checkout: TORCH_CUDA_ARCH_LIST="
                f"{_arch_text(sm)} python scripts/build_radar_cuda_prebuilt.py"
            )
        findings.append(failure)
    return findings


def _arch_text(sm: object) -> str:
    return f"{int(sm) // 10}.{int(sm) % 10}" if isinstance(sm, int) else "<major.minor>"


def _report_rows(record: dict[str, Any]) -> list[tuple[str, object]]:
    rows: list[tuple[str, object]] = [
        ("witwin-radar", record.get("package_version")),
        ("Python", f"{record.get('python')} ({record.get('python_executable')})"),
        ("Platform", record.get("platform")),
        ("PyTorch", record.get("torch") or "not importable"),
        ("PyTorch CUDA", record.get("cuda_runtime") or "none (CPU-only build)"),
        ("NVIDIA driver", record.get("driver_release") or "unknown"),
    ]
    device = record.get("device")
    rows.append(
        (
            "GPU",
            f"{device['name']} (sm_{device['sm']}, {device['status']})" if isinstance(device, dict) else "none visible",
        )
    )
    native = record.get("native_build")
    build = native.get("native_build") if isinstance(native, dict) else None
    if isinstance(build, dict):
        rows += [
            ("Native build torch", f"{build.get('torch_version')} / CUDA {build.get('cuda_version')}"),
            ("Native SASS", ", ".join(f"sm_{entry}" for entry in build.get("cuda_architectures", [])) or "none"),
            ("Native fingerprint", build.get("build_fingerprint")),
            ("Radar ABI", record.get("radar_abi_version")),
        ]
    else:
        rows.append(("Native library", "not loaded"))
    channel = record.get("channel", {})
    if channel.get("installed"):
        supported = "supported" if channel.get("supported") else "UNSUPPORTED"
        rows.append(("witwin-channel", f"{channel.get('version')} ({supported})"))
        if "verdict" in channel:
            rows.append(("Channel doctor", channel.get("verdict")))
    else:
        rows.append(("witwin-channel", "not installed (optional)"))
    smoke = record.get("smoke", {})
    state = "passed" if smoke.get("passed") else ("FAILED" if smoke.get("ran") else "not run")
    rows.append(("Simulate smoke test", f"{state}: {smoke.get('detail')}"))
    return rows


def doctor_report(record: dict[str, Any]) -> tuple[str, int]:
    """Format a :func:`doctor` record; return the text and the exit status (0 ok, 1 failure)."""

    failures = list(record.get("failures", []))
    warnings = list(record.get("warnings", []))
    lines = ["witwin-radar doctor", "", *(f"  {label:<22} {value}" for label, value in _report_rows(record)), ""]
    if failures:
        lines.append("Verdict: FAIL - Radar cannot run here.")
        lines += [f"  - {item}" for item in failures]
    elif warnings:
        lines.append("Verdict: OK with warnings - Radar runs here.")
    else:
        lines.append("Verdict: OK - Radar runs here.")
    lines += [f"  ! {item}" for item in warnings]
    return "\n".join(lines), 1 if failures else 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``witwin-radar-doctor`` / ``python -m witwin.radar doctor``."""

    parser = argparse.ArgumentParser(
        prog="witwin-radar-doctor", description="Check that this machine can run witwin-radar, and say how to fix it."
    )
    parser.add_argument("command", nargs="?", default="doctor", choices=["doctor"])
    parser.add_argument("--json", action="store_true", help="print the whole record as JSON")
    parser.add_argument("--no-smoke", action="store_true", help="skip the end-to-end simulate on the GPU")
    parser.add_argument("--no-channel-doctor", action="store_true", help="do not run Channel's own doctor")
    arguments = parser.parse_args(argv)
    record = doctor(smoke=not arguments.no_smoke, channel_doctor=not arguments.no_channel_doctor)
    text, status = doctor_report(record)
    if arguments.json:
        print(json.dumps(record, indent=2, sort_keys=True, default=str))
    else:
        print(text)
    return status


__all__ = ["channel_diagnostics", "doctor", "doctor_report", "main", "smoke_test"]
