"""Failure-boundary tests for the external comparison harness, using synthetic files."""

import json

import numpy as np
import pytest
from scipy.io import savemat

from tools.compare_matlab_radar import analyze


def test_external_comparison_requires_actual_output(tmp_path):
    with pytest.raises(ValueError, match="No exported"):
        analyze(tmp_path)
    savemat(tmp_path / "static-input.mat", {"witwin_iq": np.ones((64, 32))})
    with pytest.raises(OSError):
        analyze(tmp_path)


def test_external_comparison_rejects_wrong_doppler(tmp_path):
    samples, chirps = 64, 32
    reference = np.ones((samples, chirps), dtype=complex)
    savemat(
        tmp_path / "static-input.mat",
        {
            "witwin_iq": reference,
            "fs": 1e6,
            "fc": 77e9,
            "slope": 1e12,
            "samples": samples,
            "chirps": chirps,
            "period": 1e-3,
            "lengths": 0.0,
            "rates": 0.0,
            "gains": 1.0,
            "witwin_seconds": 0.0,
        },
    )
    result_path = tmp_path / "static-matlab.mat"
    savemat(result_path, {"matlab_iq": reference, "matlab_seconds": 0.0})
    analyze(tmp_path)
    assert json.loads((tmp_path / "status.json").read_text())["failures"] == []
    corrupted = reference * np.exp(2j * np.pi * np.arange(chirps)[None, :] * 5 / chirps)
    savemat(result_path, {"matlab_iq": corrupted, "matlab_seconds": 0.0})
    with pytest.raises(AssertionError, match="raw IQ exceeds"):
        analyze(tmp_path)
    assert json.loads((tmp_path / "status.json").read_text())["status"] == "failed"
