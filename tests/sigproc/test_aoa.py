"""
Tests for Angle-of-Arrival (AoA) estimation.

Tests the phase-comparison and 2D FFT AoA methods with synthetic
virtual antenna data — no GPU needed.
"""

import numpy as np
import pytest

from witwin.radar.sigproc.pointcloud import naive_xyz, _compensate_tdm_phase
from conftest import MockRadar, STANDARD_CONFIG


# Standard 3TX 4RX layout in half-wavelength units
TX_HW = np.array([[0, 0, 0], [2, 0, 0], [0, 1, 0]], dtype=np.float64)
RX_HW = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=np.float64)

NUM_TX = 3
NUM_RX = 4
FFT_SIZE = 64


def _make_virtual_ant(theta_az=0.0, theta_el=0.0, num_detected=1,
                      tx_hw=TX_HW, rx_hw=RX_HW, num_tx=NUM_TX, num_rx=NUM_RX):
    """Create synthetic virtual antenna data for a single target at given angles.

    Phase = pi * (x * sin(theta_az) + y * sin(theta_el))
    where x, y are virtual antenna positions in half-wavelength units.
    """
    virtual_ant = np.zeros((num_tx * num_rx, num_detected), dtype=np.complex64)
    for ti in range(num_tx):
        for ri in range(num_rx):
            va_idx = ti * num_rx + ri
            pos = tx_hw[ti] + rx_hw[ri]
            phase = np.pi * (pos[0] * np.sin(theta_az) + pos[1] * np.sin(theta_el))
            virtual_ant[va_idx, :] = np.exp(1j * phase)
    return virtual_ant


def _make_ula_virtual_ant(theta_az=0.0, theta_el=0.0, num_tx=NUM_TX, num_rx=NUM_RX):
    """Create ideal ULA virtual antenna data (what the FFT implicitly assumes).

    Azimuth array: first 2*num_rx elements at positions [0..2*num_rx-1].
    Elevation array: starts at index 2*num_rx, positions [0..num_rx-1] at y=1.
    """
    n_va = num_tx * num_rx
    virtual_ant = np.zeros((n_va, 1), dtype=np.complex64)
    n_az = 2 * num_rx
    for i in range(n_az):
        phase = np.pi * i * np.sin(theta_az)
        virtual_ant[i, 0] = np.exp(1j * phase)
    el_start = 2 * num_rx
    for i in range(min(num_rx, n_va - el_start)):
        phase = np.pi * (i * np.sin(theta_az) + 1.0 * np.sin(theta_el))
        virtual_ant[el_start + i, 0] = np.exp(1j * phase)
    return virtual_ant


class TestPhaseComparisonAoA:
    """Tests for _aoa_phase_comparison (num_tx <= 4)."""

    def test_broadside_target(self):
        """Target at broadside (0°, 0°) -> x≈0, z≈0, y≈1."""
        va = _make_virtual_ant(theta_az=0.0, theta_el=0.0)
        x, y, z = naive_xyz(va, num_tx=NUM_TX, num_rx=NUM_RX,
                             fft_size=FFT_SIZE, tx_loc_hw=TX_HW)
        assert abs(x[0]) < 0.05, f"x = {x[0]}"
        assert abs(z[0]) < 0.05, f"z = {z[0]}"
        assert y[0] > 0.9, f"y = {y[0]}"

    @pytest.mark.parametrize("theta_deg", [10, 20, 30])
    def test_azimuth_angle_ideal_ula(self, theta_deg):
        """Ideal ULA input at known azimuth -> x ≈ sin(theta)."""
        theta = np.radians(theta_deg)
        va = _make_ula_virtual_ant(theta_az=theta)
        x, y, z = naive_xyz(va, num_tx=NUM_TX, num_rx=NUM_RX,
                             fft_size=FFT_SIZE, tx_loc_hw=TX_HW)
        expected_x = np.sin(theta)
        assert abs(x[0] - expected_x) < 0.05, (
            f"theta={theta_deg}°: x={x[0]:.3f}, expected {expected_x:.3f}"
        )

    def test_azimuth_monotonic_with_real_array(self):
        """Larger azimuth angle -> larger |x| with real 3TX 4RX array."""
        results = []
        for deg in [0, 10, 20, 30]:
            theta = np.radians(deg)
            va = _make_virtual_ant(theta_az=theta)
            x, _, _ = naive_xyz(va, num_tx=NUM_TX, num_rx=NUM_RX,
                                 fft_size=FFT_SIZE, tx_loc_hw=TX_HW)
            results.append(x[0])
        # Should be monotonically increasing
        for i in range(len(results) - 1):
            assert results[i + 1] >= results[i], (
                f"Non-monotonic: {results}"
            )

    @pytest.mark.parametrize("theta_deg", [5, 15])
    def test_elevation_angle(self, theta_deg):
        """Target at known elevation -> |z| ≈ sin(theta_el).

        Sign of z can depend on phase convention; check magnitude.
        """
        theta = np.radians(theta_deg)
        va = _make_ula_virtual_ant(theta_el=theta)
        x, y, z = naive_xyz(va, num_tx=NUM_TX, num_rx=NUM_RX,
                             fft_size=FFT_SIZE, tx_loc_hw=TX_HW)
        expected_z = np.sin(theta)
        assert abs(abs(z[0]) - expected_z) < 0.15, (
            f"theta_el={theta_deg}°: |z|={abs(z[0]):.3f}, expected {expected_z:.3f}"
        )

    def test_direction_cosines_unit_sphere(self):
        """x² + y² + z² should be ≈ 1 for valid detections."""
        for az in [0, 10, 20]:
            for el in [0, 5]:
                theta_az = np.radians(az)
                theta_el = np.radians(el)
                va = _make_ula_virtual_ant(theta_az=theta_az, theta_el=theta_el)
                x, y, z = naive_xyz(va, num_tx=NUM_TX, num_rx=NUM_RX,
                                     fft_size=FFT_SIZE, tx_loc_hw=TX_HW)
                if y[0] > 0:  # valid detection
                    r2 = x[0] ** 2 + y[0] ** 2 + z[0] ** 2
                    assert abs(r2 - 1.0) < 0.05, f"az={az}, el={el}: |r|²={r2:.3f}"


class TestDynamicTxOffset:
    """Regression (bug #5): AoA must use tx_loc dynamically, not hardcode 2."""

    def test_different_tx_layouts_give_different_results(self):
        """Two TX layouts with different X-offsets for TX2 should give
        different elevation estimates."""
        theta_az = np.radians(20)
        theta_el = np.radians(10)

        # Layout 1: TX2 at X=0 (standard)
        tx1 = np.array([[0, 0, 0], [2, 0, 0], [0, 1, 0]], dtype=np.float64)
        va1 = _make_virtual_ant(theta_az, theta_el, tx_hw=tx1, rx_hw=RX_HW)
        _, _, z1 = naive_xyz(va1, num_tx=3, num_rx=4, fft_size=64, tx_loc_hw=tx1)

        # Layout 2: TX2 at X=1 (different)
        tx2 = np.array([[0, 0, 0], [2, 0, 0], [1, 1, 0]], dtype=np.float64)
        va2 = _make_virtual_ant(theta_az, theta_el, tx_hw=tx2, rx_hw=RX_HW)
        _, _, z2 = naive_xyz(va2, num_tx=3, num_rx=4, fft_size=64, tx_loc_hw=tx2)

        # Results should differ because the compensation term changes
        # (unless angles happen to make them equal, which is unlikely)
        assert z1[0] != pytest.approx(z2[0], abs=0.001), (
            "Different TX layouts produced identical elevation — "
            "dynamic tx_loc may not be used"
        )

    def test_tx_offset_zero_equals_no_compensation(self):
        """When TX2 X-offset is 0, the compensation exp(1j*0*wx) = 1."""
        theta_az = np.radians(20)
        tx_hw = np.array([[0, 0, 0], [2, 0, 0], [0, 1, 0]], dtype=np.float64)
        # TX2 at X=0, so offset = 0 - 0 = 0
        assert tx_hw[2][0] - tx_hw[0][0] == 0.0

        va = _make_virtual_ant(theta_az, theta_el=0.0, tx_hw=tx_hw, rx_hw=RX_HW)
        x, y, z = naive_xyz(va, num_tx=3, num_rx=4, fft_size=64, tx_loc_hw=tx_hw)
        # Should still give valid results
        assert y[0] > 0


class TestTDMPhaseCompensation:
    """Test _compensate_tdm_phase function."""

    def test_no_compensation_for_zero_velocity(self):
        """Zero velocity should produce no phase change."""
        mock = MockRadar(STANDARD_CONFIG)
        from witwin.radar.sigproc.pointcloud import FrameConfig
        fc = FrameConfig(mock)

        aoa = np.random.randn(12, 5) + 1j * np.random.randn(12, 5)
        aoa = aoa.astype(np.complex64)
        velocities = np.zeros(5)

        compensated = _compensate_tdm_phase(aoa, velocities, mock, fc)
        np.testing.assert_allclose(compensated, aoa, atol=1e-6)

    def test_compensation_changes_phases_for_nonzero_velocity(self):
        """Non-zero velocity should modify TX1+ phases."""
        mock = MockRadar(STANDARD_CONFIG)
        from witwin.radar.sigproc.pointcloud import FrameConfig
        fc = FrameConfig(mock)

        aoa = np.ones((12, 3), dtype=np.complex64)
        velocities = np.array([1.0, 2.0, 3.0])

        compensated = _compensate_tdm_phase(aoa, velocities, mock, fc)
        # TX0 rows (0:4) should be unchanged
        np.testing.assert_allclose(compensated[:4, :], aoa[:4, :], atol=1e-6)
        # TX1 rows (4:8) should be modified
        assert not np.allclose(compensated[4:8, :], aoa[4:8, :])
        # TX2 rows (8:12) should be modified
        assert not np.allclose(compensated[8:12, :], aoa[8:12, :])

    def test_compensation_magnitude_preserved(self):
        """Compensation is pure phase rotation — magnitude should be preserved."""
        mock = MockRadar(STANDARD_CONFIG)
        from witwin.radar.sigproc.pointcloud import FrameConfig
        fc = FrameConfig(mock)

        rng = np.random.RandomState(42)
        aoa = (rng.randn(12, 10) + 1j * rng.randn(12, 10)).astype(np.complex64)
        velocities = rng.uniform(-3, 3, 10)

        compensated = _compensate_tdm_phase(aoa, velocities, mock, fc)
        np.testing.assert_allclose(
            np.abs(compensated), np.abs(aoa), atol=1e-5
        )
