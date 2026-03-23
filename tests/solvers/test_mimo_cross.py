"""
MIMO-level cross-validation aligned with tests/verify_mimo.py.
"""

import numpy as np
import pytest
import torch

from conftest import complex_correlation, mag_correlation, peak_ratio, make_static_interpolator

pytestmark = pytest.mark.gpu


def _mimo_config(**overrides):
    """MIMO validation config derived from verify_mimo.py."""
    cfg = {
        "num_tx": 3,
        "num_rx": 4,
        "fc": 77e9,
        "slope": 60.012,
        "adc_samples": 256,
        "adc_start_time": 0,
        "sample_rate": 4400,
        "idle_time": 7,
        "ramp_end_time": 65,
        "chirp_per_frame": 2,
        "frame_per_second": 10,
        "num_doppler_bins": 2,
        "num_range_bins": 256,
        "num_angle_bins": 64,
        "power": 15,
        "tx_loc": [[0, 0, 0], [4, 0, 0], [2, 1, 0]],
        "rx_loc": [[-6, 0, 0], [-5, 0, 0], [-4, 0, 0], [-3, 0, 0]],
    }
    cfg.update(overrides)
    return cfg


def _random_static_scene(n_targets=50, seed=42):
    """Create the same style of random static scene used by verify_mimo.py."""
    rng = np.random.RandomState(seed)
    positions = rng.randn(n_targets, 3).astype(np.float32)
    positions[:, 2] -= 3
    intensities = rng.uniform(0.5, 1.5, n_targets).astype(np.float32)

    pos_t = torch.tensor(positions, device="cuda")
    sigma_t = torch.tensor(intensities, device="cuda")

    def interp(t):
        return sigma_t, pos_t

    return interp


class TestMIMOCrossValidation:

    @pytest.mark.parametrize(
        ("backend_a", "backend_b"),
        [
            ("pytorch", "dirichlet"),
            ("pytorch", "slang"),
            ("slang", "dirichlet"),
        ],
    )
    def test_backend_pairs_match_verify_metrics(self, backend_a, backend_b):
        """All solver backends should agree on the same verify_mimo-style scene."""
        from witwin.radar import Radar

        cfg = _mimo_config()
        interp = _random_static_scene(50)

        try:
            frame_a = Radar(cfg, backend=backend_a).mimo(interp).detach().cpu().numpy()
            frame_b = Radar(cfg, backend=backend_b).mimo(interp).detach().cpu().numpy()
        except (FileNotFoundError, OSError, RuntimeError) as exc:
            pytest.skip(f"backend unavailable: {exc}")

        mag_corr = mag_correlation(frame_a, frame_b)
        cx_corr = complex_correlation(frame_a, frame_b)
        ratio = peak_ratio(frame_a, frame_b)
        assert mag_corr > 0.99, f"{backend_a} vs {backend_b} magnitude correlation = {mag_corr:.4f}"
        assert cx_corr > 0.99, f"{backend_a} vs {backend_b} complex correlation = {cx_corr:.4f}"
        assert 0.98 < ratio < 1.02, f"{backend_a} vs {backend_b} peak ratio = {ratio:.4f}"

    def test_slang_vs_dirichlet_per_chirp(self):
        """Per-chirp TX0/RX0 agreement should remain high."""
        from witwin.radar import Radar

        cfg = _mimo_config()
        interp = _random_static_scene(50)

        try:
            radar_slang = Radar(cfg, backend="slang")
            radar_dirichlet = Radar(cfg, backend="dirichlet")
        except (FileNotFoundError, OSError, RuntimeError) as exc:
            pytest.skip(f"backend unavailable: {exc}")

        frame_slang = radar_slang.mimo(interp).detach().cpu().numpy()
        frame_dirichlet = radar_dirichlet.mimo(interp).detach().cpu().numpy()

        for chirp_id in range(cfg["chirp_per_frame"]):
            mag_corr = mag_correlation(frame_slang[0, 0, chirp_id, :], frame_dirichlet[0, 0, chirp_id, :])
            cx_corr = complex_correlation(frame_slang[0, 0, chirp_id, :], frame_dirichlet[0, 0, chirp_id, :])
            assert mag_corr > 0.99, f"chirp {chirp_id} magnitude correlation = {mag_corr:.4f}"
            assert cx_corr > 0.99, f"chirp {chirp_id} complex correlation = {cx_corr:.4f}"

    def test_slang_vs_dirichlet_nonzero_adc_start(self):
        """Non-zero adc_start_time should preserve verify_mimo-style agreement."""
        from witwin.radar import Radar

        cfg = _mimo_config(adc_start_time=6)
        interp = _random_static_scene(30)

        try:
            radar_slang = Radar(cfg, backend="slang")
            radar_dirichlet = Radar(cfg, backend="dirichlet")
        except (FileNotFoundError, OSError, RuntimeError) as exc:
            pytest.skip(f"backend unavailable: {exc}")

        frame_slang = radar_slang.mimo(interp).detach().cpu().numpy()
        frame_dirichlet = radar_dirichlet.mimo(interp).detach().cpu().numpy()

        mag_corr = mag_correlation(frame_slang, frame_dirichlet)
        cx_corr = complex_correlation(frame_slang, frame_dirichlet)
        assert mag_corr > 0.99, f"non-zero adc_start magnitude correlation = {mag_corr:.4f}"
        assert cx_corr > 0.99, f"non-zero adc_start complex correlation = {cx_corr:.4f}"


class TestMIMOOutputShape:

    def test_output_shape(self):
        """MIMO output should be (TX, RX, chirps, ADC)."""
        from witwin.radar import Radar

        cfg = _mimo_config()
        interp = make_static_interpolator([0, 0, -3])

        for backend in ("pytorch", "slang", "dirichlet"):
            try:
                radar = Radar(cfg, backend=backend)
                frame = radar.mimo(interp)
            except (FileNotFoundError, OSError, RuntimeError) as exc:
                pytest.skip(f"backend unavailable: {exc}")
            assert frame.shape == (
                cfg["num_tx"],
                cfg["num_rx"],
                cfg["chirp_per_frame"],
                cfg["adc_samples"],
            ), f"backend={backend}: shape={frame.shape}"

    def test_output_not_all_zeros(self):
        """MIMO output with a target should contain non-zero values."""
        from witwin.radar import Radar

        cfg = _mimo_config()
        interp = make_static_interpolator([0, 0, -3])

        for backend in ("pytorch", "slang", "dirichlet"):
            try:
                radar = Radar(cfg, backend=backend)
                frame = radar.mimo(interp)
            except (FileNotFoundError, OSError, RuntimeError) as exc:
                pytest.skip(f"backend unavailable: {exc}")
            assert frame.abs().max().item() > 0, f"{backend}: all zeros"

    def test_zero_targets_gives_zero_output(self):
        """Empty scene should produce all-zero frame."""
        from witwin.radar import Radar

        cfg = _mimo_config()
        positions = torch.zeros((0, 3), device="cuda")
        intensities = torch.zeros(0, device="cuda")

        def empty_interp(t):
            return intensities, positions

        for backend in ("pytorch", "slang", "dirichlet"):
            try:
                radar = Radar(cfg, backend=backend)
                frame = radar.mimo(empty_interp)
            except (FileNotFoundError, OSError, RuntimeError) as exc:
                pytest.skip(f"backend unavailable: {exc}")
            assert frame.abs().max().item() == 0, f"{backend}: not zero for empty scene"
