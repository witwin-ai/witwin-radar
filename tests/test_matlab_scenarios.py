"""Physical invariants and raw phase sensitivity of the comparison diagnostics."""

import numpy as np

from tools.compare_matlab_scenarios import independent_free_space_oracle, independent_ground_oracle, iq_metrics


def test_air_boundary_disappears_from_four_path_oracle():
    source = {
        "fs": 4e6,
        "fc": 77e9,
        "samples": 32,
        "chirps": 16,
        "period": 32e-6,
        "slope": 0.015625e12,
        "eps": 1.0,
        "sigma": 0.0,
    }
    free = independent_free_space_oracle(source, "acceleration")
    ground = independent_ground_oracle(source)
    np.testing.assert_allclose(ground, free, rtol=1e-12, atol=1e-18)


def test_iq_metrics_cannot_hide_a_phase_sign_error_with_power_or_alignment():
    reference = np.exp(1j * np.arange(64)[:, None] * 0.13) * np.ones((1, 16))
    result = iq_metrics(reference, -reference, 4)
    assert abs(result["raw_iq_relative_l2"] - 2) < 1e-12
    assert result["rd_power_relative_l2"] == 0
    assert result["aligned_iq_relative_l2"] < 1e-12
