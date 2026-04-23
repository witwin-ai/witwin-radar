"""
Tests for Angle-of-Arrival (AoA) estimation.

Tests the phase-comparison and 2D FFT AoA methods with synthetic
virtual antenna data — no GPU needed. Inputs are torch tensors.
"""

import numpy as np
import pytest
import torch

from witwin.radar.sigproc.pointcloud import naive_xyz, _compensate_tdm_phase
from conftest import MockRadar, STANDARD_CONFIG


TX_HW = np.array([[0, 0, 0], [2, 0, 0], [0, 1, 0]], dtype=np.float64)
RX_HW = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=np.float64)

NUM_TX = 3
NUM_RX = 4
FFT_SIZE = 64


def _make_virtual_ant(theta_az=0.0, theta_el=0.0, num_detected=1,
                      tx_hw=TX_HW, rx_hw=RX_HW, num_tx=NUM_TX, num_rx=NUM_RX):
    """Create synthetic virtual antenna data for a single target at given angles.

    Returns a torch.complex64 tensor of shape (num_tx*num_rx, num_detected).
    """
    virtual_ant = np.zeros((num_tx * num_rx, num_detected), dtype=np.complex64)
    for ti in range(num_tx):
        for ri in range(num_rx):
            va_idx = ti * num_rx + ri
            pos = tx_hw[ti] + rx_hw[ri]
            phase = np.pi * (pos[0] * np.sin(theta_az) + pos[1] * np.sin(theta_el))
            virtual_ant[va_idx, :] = np.exp(1j * phase)
    return torch.from_numpy(virtual_ant)


def _make_ula_virtual_ant(theta_az=0.0, theta_el=0.0, num_tx=NUM_TX, num_rx=NUM_RX):
    """Create ideal ULA virtual antenna data (what the FFT implicitly assumes)."""
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
    return torch.from_numpy(virtual_ant)


def _scalar(t):
    return float(t.item() if isinstance(t, torch.Tensor) else t)


class TestPhaseComparisonAoA:
    """Tests for _aoa_phase_comparison (num_tx <= 4)."""

    def test_broadside_target(self):
        va = _make_virtual_ant(theta_az=0.0, theta_el=0.0)
        x, y, z = naive_xyz(va, num_tx=NUM_TX, num_rx=NUM_RX,
                             fft_size=FFT_SIZE, tx_loc_hw=TX_HW)
        assert abs(_scalar(x[0])) < 0.05
        assert abs(_scalar(z[0])) < 0.05
        assert _scalar(y[0]) > 0.9

    @pytest.mark.parametrize("theta_deg", [10, 20, 30])
    def test_azimuth_angle_ideal_ula(self, theta_deg):
        theta = np.radians(theta_deg)
        va = _make_ula_virtual_ant(theta_az=theta)
        x, y, z = naive_xyz(va, num_tx=NUM_TX, num_rx=NUM_RX,
                             fft_size=FFT_SIZE, tx_loc_hw=TX_HW)
        expected_x = np.sin(theta)
        assert abs(_scalar(x[0]) - expected_x) < 0.05

    def test_azimuth_monotonic_with_real_array(self):
        results = []
        for deg in [0, 10, 20, 30]:
            theta = np.radians(deg)
            va = _make_virtual_ant(theta_az=theta)
            x, _, _ = naive_xyz(va, num_tx=NUM_TX, num_rx=NUM_RX,
                                 fft_size=FFT_SIZE, tx_loc_hw=TX_HW)
            results.append(_scalar(x[0]))
        for i in range(len(results) - 1):
            assert results[i + 1] >= results[i], f"Non-monotonic: {results}"

    @pytest.mark.parametrize("theta_deg", [5, 15])
    def test_elevation_angle(self, theta_deg):
        theta = np.radians(theta_deg)
        va = _make_ula_virtual_ant(theta_el=theta)
        x, y, z = naive_xyz(va, num_tx=NUM_TX, num_rx=NUM_RX,
                             fft_size=FFT_SIZE, tx_loc_hw=TX_HW)
        expected_z = np.sin(theta)
        assert abs(abs(_scalar(z[0])) - expected_z) < 0.15

    def test_direction_cosines_unit_sphere(self):
        for az in [0, 10, 20]:
            for el in [0, 5]:
                theta_az = np.radians(az)
                theta_el = np.radians(el)
                va = _make_ula_virtual_ant(theta_az=theta_az, theta_el=theta_el)
                x, y, z = naive_xyz(va, num_tx=NUM_TX, num_rx=NUM_RX,
                                     fft_size=FFT_SIZE, tx_loc_hw=TX_HW)
                if _scalar(y[0]) > 0:
                    r2 = _scalar(x[0]) ** 2 + _scalar(y[0]) ** 2 + _scalar(z[0]) ** 2
                    assert abs(r2 - 1.0) < 0.05


class TestDynamicTxOffset:
    """Regression (bug #5): AoA must use tx_loc dynamically, not hardcode 2."""

    def test_different_tx_layouts_give_different_results(self):
        theta_az = np.radians(20)
        theta_el = np.radians(10)

        tx1 = np.array([[0, 0, 0], [2, 0, 0], [0, 1, 0]], dtype=np.float64)
        va1 = _make_virtual_ant(theta_az, theta_el, tx_hw=tx1, rx_hw=RX_HW)
        _, _, z1 = naive_xyz(va1, num_tx=3, num_rx=4, fft_size=64, tx_loc_hw=tx1)

        tx2 = np.array([[0, 0, 0], [2, 0, 0], [1, 1, 0]], dtype=np.float64)
        va2 = _make_virtual_ant(theta_az, theta_el, tx_hw=tx2, rx_hw=RX_HW)
        _, _, z2 = naive_xyz(va2, num_tx=3, num_rx=4, fft_size=64, tx_loc_hw=tx2)

        assert _scalar(z1[0]) != pytest.approx(_scalar(z2[0]), abs=0.001)

    def test_tx_offset_zero_equals_no_compensation(self):
        theta_az = np.radians(20)
        tx_hw = np.array([[0, 0, 0], [2, 0, 0], [0, 1, 0]], dtype=np.float64)
        assert tx_hw[2][0] - tx_hw[0][0] == 0.0

        va = _make_virtual_ant(theta_az, theta_el=0.0, tx_hw=tx_hw, rx_hw=RX_HW)
        x, y, z = naive_xyz(va, num_tx=3, num_rx=4, fft_size=64, tx_loc_hw=tx_hw)
        assert _scalar(y[0]) > 0


class TestTDMPhaseCompensation:
    """Test _compensate_tdm_phase function."""

    def test_no_compensation_for_zero_velocity(self):
        mock = MockRadar(STANDARD_CONFIG)
        from witwin.radar.sigproc.pointcloud import FrameConfig
        fc = FrameConfig(mock)

        g = torch.Generator().manual_seed(0)
        aoa = (torch.randn(12, 5, generator=g) + 1j * torch.randn(12, 5, generator=g)).to(torch.complex64)
        velocities = torch.zeros(5, dtype=torch.float64)

        compensated = _compensate_tdm_phase(aoa, velocities, mock, fc)
        torch.testing.assert_close(compensated, aoa, atol=1e-6, rtol=1e-6)

    def test_compensation_changes_phases_for_nonzero_velocity(self):
        mock = MockRadar(STANDARD_CONFIG)
        from witwin.radar.sigproc.pointcloud import FrameConfig
        fc = FrameConfig(mock)

        aoa = torch.ones((12, 3), dtype=torch.complex64)
        velocities = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

        compensated = _compensate_tdm_phase(aoa, velocities, mock, fc)
        torch.testing.assert_close(compensated[:4, :], aoa[:4, :], atol=1e-6, rtol=1e-6)
        assert not torch.allclose(compensated[4:8, :], aoa[4:8, :])
        assert not torch.allclose(compensated[8:12, :], aoa[8:12, :])

    def test_compensation_magnitude_preserved(self):
        mock = MockRadar(STANDARD_CONFIG)
        from witwin.radar.sigproc.pointcloud import FrameConfig
        fc = FrameConfig(mock)

        g = torch.Generator().manual_seed(42)
        aoa = (torch.randn(12, 10, generator=g) + 1j * torch.randn(12, 10, generator=g)).to(torch.complex64)
        velocities = (torch.rand(10, generator=g) * 6 - 3).to(torch.float64)

        compensated = _compensate_tdm_phase(aoa, velocities, mock, fc)
        torch.testing.assert_close(torch.abs(compensated), torch.abs(aoa), atol=1e-5, rtol=1e-5)
