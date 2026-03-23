"""
Tests for CFAR detectors: CA-CFAR (reference + fast) and OS-CFAR.

All tests use synthetic Range-Doppler magnitude maps — no GPU needed.
"""

import numpy as np
import pytest
import torch

from conftest import MINIMAL_CONFIG, MockRadar
from witwin.radar.sigproc.cfar import ca_cfar_2d, ca_cfar_2d_fast, os_cfar_2d
from witwin.radar.sigproc.pointcloud import process_pc


def _make_rd_map_with_peak(Nd=64, Nr=64, noise_level=1.0, peak_val=50.0,
                            peak_pos=(32, 32), seed=42):
    """Create a synthetic RD map with Gaussian noise and a single peak."""
    rng = np.random.RandomState(seed)
    rd_map = np.abs(rng.randn(Nd, Nr)) * noise_level
    rd_map[peak_pos] = peak_val
    return rd_map


class TestCACFAR:

    def test_detects_strong_peak(self):
        """CA-CFAR should detect a clear peak above noise."""
        rd_map = _make_rd_map_with_peak(peak_val=100.0)
        detections, _ = ca_cfar_2d(rd_map, guard_cells=(2, 2), training_cells=(4, 4))
        assert detections[32, 32], "Failed to detect strong peak"

    def test_no_detection_in_noise(self):
        """Pure noise should have very few false alarms."""
        rng = np.random.RandomState(123)
        rd_map = np.abs(rng.randn(64, 64))
        detections, _ = ca_cfar_2d(rd_map, guard_cells=(2, 2),
                                    training_cells=(4, 4), pfa=1e-4)
        fa_rate = detections.sum() / detections.size
        # With pfa=1e-4, should have very low false alarm rate
        assert fa_rate < 0.01, f"False alarm rate too high: {fa_rate:.4f}"

    def test_detects_multiple_peaks(self):
        """CA-CFAR should detect multiple well-separated peaks."""
        rd_map = np.abs(np.random.RandomState(0).randn(128, 128))
        rd_map[30, 30] = 80.0
        rd_map[30, 100] = 80.0
        rd_map[90, 60] = 80.0

        detections, _ = ca_cfar_2d(rd_map, guard_cells=(2, 3),
                                    training_cells=(4, 6))
        assert detections[30, 30], "Peak 1 not detected"
        assert detections[30, 100], "Peak 2 not detected"
        assert detections[90, 60], "Peak 3 not detected"

    def test_torch_matches_numpy_reference(self):
        rd_map = _make_rd_map_with_peak(peak_val=80.0, seed=9)

        det_np, thr_np = ca_cfar_2d(rd_map, guard_cells=(2, 3), training_cells=(4, 6), pfa=1e-3)
        det_t, thr_t = ca_cfar_2d(torch.from_numpy(rd_map), guard_cells=(2, 3), training_cells=(4, 6), pfa=1e-3)

        np.testing.assert_array_equal(det_t.numpy(), det_np)
        np.testing.assert_allclose(thr_t.numpy(), thr_np, atol=1e-5, rtol=1e-6)


class TestCACFARFast:

    def test_detects_peak_same_as_reference(self):
        """Fast CA-CFAR should detect the same peaks as reference."""
        rd_map = _make_rd_map_with_peak(peak_val=80.0)
        guard = (2, 3)
        train = (4, 6)
        pfa = 1e-3

        det_ref, _ = ca_cfar_2d(rd_map, guard_cells=guard,
                                 training_cells=train, pfa=pfa)
        det_fast, _ = ca_cfar_2d_fast(rd_map, guard_cells=guard,
                                       training_cells=train, pfa=pfa)

        # Both should detect the peak
        assert det_ref[32, 32] and det_fast[32, 32], (
            "Both should detect the main peak"
        )

    def test_fast_matches_reference_on_realistic_map(self):
        """Detection masks should be very similar (small edge differences allowed)."""
        rng = np.random.RandomState(99)
        rd_map = np.abs(rng.randn(64, 64)) + 0.1
        # Add some peaks
        for r, c in [(10, 20), (30, 30), (50, 40)]:
            rd_map[r, c] = 40.0

        guard = (2, 3)
        train = (4, 6)
        pfa = 1e-3

        det_ref, _ = ca_cfar_2d(rd_map, guard_cells=guard,
                                 training_cells=train, pfa=pfa)
        det_fast, _ = ca_cfar_2d_fast(rd_map, guard_cells=guard,
                                       training_cells=train, pfa=pfa)

        # Allow small boundary differences due to padding mode
        agreement = (det_ref == det_fast).mean()
        assert agreement > 0.95, f"Agreement = {agreement:.3f}"


class TestOSCFAR:

    def test_detects_strong_peak(self):
        """OS-CFAR should detect a clear peak."""
        rd_map = _make_rd_map_with_peak(peak_val=100.0, Nd=32, Nr=32,
                                         peak_pos=(16, 16))
        detections, threshold_map = os_cfar_2d(rd_map, guard_cells=(1, 1),
                                               training_cells=(3, 3))
        assert detections[16, 16], "OS-CFAR failed to detect strong peak"
        assert threshold_map.shape == rd_map.shape

    def test_position_mask_not_value_based(self):
        """Regression (bug #2): uniform background must not confuse OS-CFAR.

        With the old setdiff1d approach, if all training cells have the same
        value as guard cells, they would all be removed from the training set.
        The position mask approach handles this correctly.
        """
        Nd, Nr = 32, 32
        # ALL cells have identical value except one peak
        rd_map = np.full((Nd, Nr), 5.0)
        rd_map[16, 16] = 200.0

        detections, threshold_map = os_cfar_2d(rd_map, guard_cells=(1, 1),
                                               training_cells=(3, 3), pfa=1e-3)
        assert detections[16, 16], (
            "OS-CFAR failed on uniform background (position mask regression)"
        )
        assert threshold_map.shape == rd_map.shape

    def test_robust_to_interferer(self):
        """OS-CFAR should still detect a target even with a nearby interferer.

        OS-CFAR uses ordered statistics so one strong interferer in the
        training band should not raise the threshold as much as in CA-CFAR.
        """
        Nd, Nr = 64, 64
        rng = np.random.RandomState(42)
        rd_map = np.abs(rng.randn(Nd, Nr)) * 0.5
        # Target
        rd_map[32, 32] = 50.0
        # Strong interferer within training region
        rd_map[32, 38] = 50.0

        detections, threshold_map = os_cfar_2d(rd_map, guard_cells=(2, 2),
                                               training_cells=(4, 4),
                                               rank_fraction=0.75)
        assert detections[32, 32], "OS-CFAR failed with nearby interferer"
        assert threshold_map.shape == rd_map.shape

    def test_torch_matches_numpy_reference(self):
        rd_map = _make_rd_map_with_peak(peak_val=70.0, seed=21)

        det_np, thr_np = os_cfar_2d(rd_map, guard_cells=(2, 3), training_cells=(4, 6), rank_fraction=0.75, pfa=1e-3)
        det_t, thr_t = os_cfar_2d(
            torch.from_numpy(rd_map),
            guard_cells=(2, 3),
            training_cells=(4, 6),
            rank_fraction=0.75,
            pfa=1e-3,
        )

        np.testing.assert_array_equal(det_t.numpy(), det_np)
        np.testing.assert_allclose(thr_t.numpy(), thr_np, atol=1e-5, rtol=1e-6)


class TestCFARParameterRanges:
    """Verify CFAR works with various guard/training cell sizes."""

    @pytest.mark.parametrize("guard,train", [
        ((1, 1), (2, 2)),
        ((2, 3), (4, 6)),
        ((3, 4), (6, 8)),
        ((1, 2), (8, 10)),
    ])
    def test_ca_cfar_various_cell_sizes(self, guard, train):
        rd_map = _make_rd_map_with_peak(Nd=64, Nr=64, peak_val=80.0)
        detections, threshold = ca_cfar_2d_fast(rd_map, guard_cells=guard,
                                                 training_cells=train)
        assert detections[32, 32], f"guard={guard}, train={train}: peak not detected"
        assert threshold.shape == rd_map.shape

    @pytest.mark.parametrize("pfa", [1e-2, 1e-3, 1e-4, 1e-5])
    def test_ca_cfar_pfa_range(self, pfa):
        rd_map = _make_rd_map_with_peak(Nd=64, Nr=64, peak_val=80.0)
        detections, _ = ca_cfar_2d_fast(rd_map, guard_cells=(2, 3),
                                         training_cells=(4, 6), pfa=pfa)
        # Strong peak should always be detected regardless of Pfa
        assert detections[32, 32]


def test_process_pc_rejects_unknown_detector():
    radar = MockRadar(MINIMAL_CONFIG)
    frame = np.zeros((radar.num_tx, radar.num_rx, radar.chirp_per_frame, radar.adc_samples), dtype=np.complex64)

    with pytest.raises(ValueError, match="Unsupported detector"):
        process_pc(radar, frame, detector="unknown")


@pytest.mark.gpu
def test_torch_reference_cfar_paths_preserve_cuda_device():
    rd_map = torch.from_numpy(
        _make_rd_map_with_peak(Nd=32, Nr=32, peak_val=60.0, peak_pos=(16, 16))
    ).to("cuda")

    det_ca, thr_ca = ca_cfar_2d(rd_map, guard_cells=(1, 1), training_cells=(3, 3))
    det_os, thr_os = os_cfar_2d(rd_map, guard_cells=(1, 1), training_cells=(3, 3))

    assert det_ca.device.type == "cuda"
    assert thr_ca.device.type == "cuda"
    assert det_os.device.type == "cuda"
    assert thr_os.device.type == "cuda"
