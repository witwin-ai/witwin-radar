"""
Tests for CFAR detectors: CA-CFAR (reference + fast) and OS-CFAR.

All CFAR helpers accept torch tensors only.
"""

import pytest
import torch

from conftest import MINIMAL_CONFIG, MockRadar
from witwin.radar.sigproc.cfar import ca_cfar_2d, ca_cfar_2d_fast, os_cfar_2d
from witwin.radar.sigproc.pointcloud import process_pc


def _make_rd_map_with_peak(Nd=64, Nr=64, noise_level=1.0, peak_val=50.0,
                           peak_pos=(32, 32), seed=42):
    """Create a synthetic RD map (torch tensor) with noise and a single peak."""
    g = torch.Generator().manual_seed(seed)
    rd_map = torch.abs(torch.randn(Nd, Nr, generator=g, dtype=torch.float32)) * noise_level
    rd_map[peak_pos] = peak_val
    return rd_map


class TestCACFAR:

    def test_detects_strong_peak(self):
        rd_map = _make_rd_map_with_peak(peak_val=100.0)
        detections, _ = ca_cfar_2d(rd_map, guard_cells=(2, 2), training_cells=(4, 4))
        assert bool(detections[32, 32].item())

    def test_no_detection_in_noise(self):
        g = torch.Generator().manual_seed(123)
        rd_map = torch.abs(torch.randn(64, 64, generator=g, dtype=torch.float32))
        detections, _ = ca_cfar_2d(rd_map, guard_cells=(2, 2),
                                    training_cells=(4, 4), pfa=1e-4)
        fa_rate = detections.to(torch.float32).mean().item()
        assert fa_rate < 0.01

    def test_detects_multiple_peaks(self):
        g = torch.Generator().manual_seed(0)
        rd_map = torch.abs(torch.randn(128, 128, generator=g, dtype=torch.float32))
        rd_map[30, 30] = 80.0
        rd_map[30, 100] = 80.0
        rd_map[90, 60] = 80.0

        detections, _ = ca_cfar_2d(rd_map, guard_cells=(2, 3), training_cells=(4, 6))
        assert bool(detections[30, 30].item())
        assert bool(detections[30, 100].item())
        assert bool(detections[90, 60].item())


class TestCACFARFast:

    def test_detects_peak_same_as_reference(self):
        rd_map = _make_rd_map_with_peak(peak_val=80.0)
        guard = (2, 3)
        train = (4, 6)
        pfa = 1e-3

        det_ref, _ = ca_cfar_2d(rd_map, guard_cells=guard, training_cells=train, pfa=pfa)
        det_fast, _ = ca_cfar_2d_fast(rd_map, guard_cells=guard, training_cells=train, pfa=pfa)

        assert bool(det_ref[32, 32].item()) and bool(det_fast[32, 32].item())

    def test_fast_matches_reference_on_realistic_map(self):
        g = torch.Generator().manual_seed(99)
        rd_map = torch.abs(torch.randn(64, 64, generator=g, dtype=torch.float32)) + 0.1
        for r, c in [(10, 20), (30, 30), (50, 40)]:
            rd_map[r, c] = 40.0

        guard = (2, 3)
        train = (4, 6)
        pfa = 1e-3

        det_ref, _ = ca_cfar_2d(rd_map, guard_cells=guard, training_cells=train, pfa=pfa)
        det_fast, _ = ca_cfar_2d_fast(rd_map, guard_cells=guard, training_cells=train, pfa=pfa)

        agreement = (det_ref == det_fast).to(torch.float32).mean().item()
        assert agreement > 0.95


class TestOSCFAR:

    def test_detects_strong_peak(self):
        rd_map = _make_rd_map_with_peak(peak_val=100.0, Nd=32, Nr=32, peak_pos=(16, 16))
        detections, threshold_map = os_cfar_2d(rd_map, guard_cells=(1, 1), training_cells=(3, 3))
        assert bool(detections[16, 16].item())
        assert threshold_map.shape == rd_map.shape

    def test_position_mask_not_value_based(self):
        """Regression (bug #2): uniform background must not confuse OS-CFAR."""
        Nd, Nr = 32, 32
        rd_map = torch.full((Nd, Nr), 5.0, dtype=torch.float32)
        rd_map[16, 16] = 200.0

        detections, threshold_map = os_cfar_2d(rd_map, guard_cells=(1, 1),
                                               training_cells=(3, 3), pfa=1e-3)
        assert bool(detections[16, 16].item())
        assert threshold_map.shape == rd_map.shape

    def test_robust_to_interferer(self):
        Nd, Nr = 64, 64
        g = torch.Generator().manual_seed(42)
        rd_map = torch.abs(torch.randn(Nd, Nr, generator=g, dtype=torch.float32)) * 0.5
        rd_map[32, 32] = 50.0
        rd_map[32, 38] = 50.0

        detections, threshold_map = os_cfar_2d(rd_map, guard_cells=(2, 2),
                                               training_cells=(4, 4),
                                               rank_fraction=0.75)
        assert bool(detections[32, 32].item())
        assert threshold_map.shape == rd_map.shape


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
        detections, threshold = ca_cfar_2d_fast(rd_map, guard_cells=guard, training_cells=train)
        assert bool(detections[32, 32].item())
        assert threshold.shape == rd_map.shape

    @pytest.mark.parametrize("pfa", [1e-2, 1e-3, 1e-4, 1e-5])
    def test_ca_cfar_pfa_range(self, pfa):
        rd_map = _make_rd_map_with_peak(Nd=64, Nr=64, peak_val=80.0)
        detections, _ = ca_cfar_2d_fast(rd_map, guard_cells=(2, 3),
                                         training_cells=(4, 6), pfa=pfa)
        assert bool(detections[32, 32].item())


def test_process_pc_rejects_unknown_detector():
    radar = MockRadar(MINIMAL_CONFIG)
    frame = torch.zeros(
        (radar.num_tx, radar.num_rx, radar.chirp_per_frame, radar.adc_samples),
        dtype=torch.complex64,
    )

    with pytest.raises(ValueError, match="not a valid DetectorType"):
        process_pc(radar, frame, detector="unknown")


@pytest.mark.gpu
def test_cfar_paths_preserve_cuda_device():
    rd_map = _make_rd_map_with_peak(Nd=32, Nr=32, peak_val=60.0, peak_pos=(16, 16)).to("cuda")

    det_ca, thr_ca = ca_cfar_2d(rd_map, guard_cells=(1, 1), training_cells=(3, 3))
    det_os, thr_os = os_cfar_2d(rd_map, guard_cells=(1, 1), training_cells=(3, 3))

    assert det_ca.device.type == "cuda"
    assert thr_ca.device.type == "cuda"
    assert det_os.device.type == "cuda"
    assert thr_os.device.type == "cuda"
