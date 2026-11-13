"""``witwin-radar-doctor`` (F-8, M5-9): the verdict, the fix for each failure, and a live run on the GPU."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from witwin.radar import doctor as doctor_module
from witwin.radar.doctor import _release, _runtime_findings, doctor_report

REPO = Path(__file__).resolve().parents[1]


def _healthy() -> dict:
    return {
        "package_version": "0.5.0",
        "python": "3.11.0",
        "python_executable": sys.executable,
        "platform": "Linux",
        "torch": "2.10.0+cu128",
        "cuda_runtime": "12.8",
        "cuda_available": True,
        "device": {"name": "GPU", "sm": 120, "status": "runtime_verified", "declared_supported": True},
        "native_build": {"native_build": {"cuda_architectures": ["120"], "torch_version": "2.10.0"}},
        "radar_abi_version": 11,
        "driver_release": "570.86",
        "channel": {"installed": True, "version": "0.6.0", "supported": True, "status": 0, "verdict": "Verdict: OK"},
        "smoke": {"ran": True, "passed": True, "detail": "target at 5.00 m"},
        "errors": [],
        "failures": [],
        "warnings": [],
    }


def test_a_healthy_record_is_ok_and_exits_zero():
    text, status = doctor_report(_healthy())
    assert status == 0 and "Verdict: OK - Radar runs here." in text
    assert "sm_120" in text and "0.6.0 (supported)" in text


def test_a_failure_prints_its_fix_and_exits_one():
    record = {**_healthy(), "failures": ["CUDA is unavailable; check `nvidia-smi`"]}
    text, status = doctor_report(record)
    assert status == 1 and "Verdict: FAIL" in text and "  - CUDA is unavailable; check `nvidia-smi`" in text


def test_runtime_findings_name_the_fix():
    cpu_build = {**_healthy(), "cuda_runtime": None}
    assert "Install the CUDA build: pip install torch==2.10.0" in _runtime_findings(cpu_build)[0]
    no_gpu = {**_healthy(), "cuda_available": False, "device": None}
    assert any("CUDA_VISIBLE_DEVICES" in finding for finding in _runtime_findings(no_gpu))
    other_sm = {**_healthy(), "device": {**_healthy()["device"], "sm": 86}}
    assert any("TORCH_CUDA_ARCH_LIST=8.6" in finding for finding in _runtime_findings(other_sm))
    assert _runtime_findings(_healthy()) == []


def test_the_channel_range_is_the_channel_extra(monkeypatch):
    pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(r'(?m)^\s*"witwin-channel>=0\.6\.0,<0\.7;', pyproject)
    assert doctor_module.CHANNEL_SUPPORTED == ((0, 6, 0), (0, 7, 0))
    assert _release("0.6.0") == (0, 6, 0) and _release("0.6.1rc1") == (0, 6, 1) and _release("0.7.0") == (0, 7, 0)


def test_a_missing_channel_is_a_warning_and_skips_the_smoke(monkeypatch):
    monkeypatch.setattr(doctor_module, "runtime_diagnostics", _healthy)
    monkeypatch.setattr(doctor_module, "channel_diagnostics", lambda run_doctor: {"installed": False})
    record = doctor_module.doctor()
    assert record["failures"] == []
    assert any("pip install 'witwin-radar[channel]'" in warning for warning in record["warnings"])
    assert record["smoke"]["ran"] is False


def test_an_unsupported_channel_fails_with_the_install_line(monkeypatch):
    monkeypatch.setattr(doctor_module, "runtime_diagnostics", _healthy)
    monkeypatch.setattr(
        doctor_module,
        "channel_diagnostics",
        lambda run_doctor: {"installed": True, "version": "0.5.0", "supported": False},
    )
    record = doctor_module.doctor(channel_doctor=False)
    assert any("pip install 'witwin-channel>=0.6.0,<0.7'" in failure for failure in record["failures"])
    assert doctor_report(record)[1] == 1


@pytest.mark.gpu
def test_the_doctor_runs_end_to_end_on_this_machine():
    """`python -m witwin.radar doctor --json` in a fresh process: verdict OK and the smoke test passed."""

    environment = {**os.environ, "PYTHONPATH": str(REPO) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    completed = subprocess.run(
        [sys.executable, "-m", "witwin.radar", "doctor", "--json"],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=REPO,
        env=environment,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    record = json.loads(completed.stdout)
    assert record["failures"] == []
    assert record["smoke"]["passed"], record["smoke"]
    assert record["channel"]["installed"] and record["channel"]["status"] == 0
