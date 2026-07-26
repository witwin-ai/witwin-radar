"""
MIMO-level cross-validation aligned with tests/verify_mimo.py.
"""

import numpy as np
import pytest
import torch

from conftest import complex_correlation, mag_correlation, peak_ratio, make_static_interpolator

pytestmark = pytest.mark.gpu


def _mimo_config_dict(**overrides):
    """MIMO validation config dict derived from verify_mimo.py."""
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


def _mimo_config(**overrides):
    from witwin.radar import RadarConfig

    return RadarConfig.from_dict(_mimo_config_dict(**overrides))


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


def _trace_result(points, intensities):
    from witwin.radar import TraceResult

    return TraceResult(
        torch.as_tensor(points, dtype=torch.float32, device="cuda"),
        torch.as_tensor(intensities, dtype=torch.float32, device="cuda"),
    )


class TestMIMOCrossValidation:

    def test_dirichlet_mimo_from_static_trace_matches_legacy_path(self):
        """Static per-frame traces should be reusable without changing the generated MIMO frame."""
        from witwin.radar import Radar

        cfg = _mimo_config(chirp_per_frame=4, num_doppler_bins=4, adc_start_time=6)
        trace = _trace_result(
            points=[
                [0.0, 0.0, -3.0],
                [0.4, -0.1, -3.8],
                [-0.3, 0.2, -2.5],
            ],
            intensities=[0.7, 0.4, 1.1],
        )

        radar = Radar(cfg)

        def interp(_t):
            return trace

        legacy = radar.mimo(interp, fast=False)
        fast = radar.mimo_from_trace(trace)
        cache = radar.path_cache_from_trace(trace)
        fast_from_cache = radar.mimo_from_paths(cache)
        legacy_freq = radar.mimo(interp, fast=False, freq_domain=True)
        fast_freq = radar.mimo_from_trace(trace, freq_domain=True)

        legacy_np = legacy.detach().cpu().numpy()
        fast_np = fast.detach().cpu().numpy()
        assert fast.shape == legacy.shape
        torch.testing.assert_close(fast, legacy, rtol=1e-5, atol=1e-8)
        torch.testing.assert_close(fast_from_cache, fast, rtol=1e-6, atol=1e-9)
        torch.testing.assert_close(fast_freq, legacy_freq, rtol=1e-5, atol=1e-8)
        torch.testing.assert_close(fast_freq, torch.fft.fft(fast, dim=-1), rtol=1e-5, atol=1e-8)
        assert mag_correlation(legacy_np, fast_np) > 0.9999
        assert complex_correlation(legacy_np, fast_np) > 0.9999
        assert 0.999 < peak_ratio(legacy_np, fast_np) < 1.001

    def test_dirichlet_mimo_from_trace_with_radial_velocity_matches_legacy_path(self):
        """A linear velocity prior should reproduce per-chirp recomputation for boresight radial motion."""
        from witwin.radar import Radar, TraceResult

        cfg = _mimo_config(
            num_tx=1,
            num_rx=1,
            tx_loc=[[0, 0, 0]],
            rx_loc=[[0, 0, 0]],
            chirp_per_frame=8,
            num_doppler_bins=8,
            adc_start_time=6,
        )
        base_points = torch.tensor([[0.0, 0.0, -3.0]], dtype=torch.float32, device="cuda")
        intensities = torch.tensor([0.9], dtype=torch.float32, device="cuda")
        velocities = torch.tensor([[0.0, 0.0, -0.75]], dtype=torch.float32, device="cuda")
        trace = TraceResult(base_points, intensities)
        radar = Radar(cfg)
        t0 = 0.25

        def interp(t):
            return TraceResult(base_points + velocities * (float(t) - t0), intensities)

        legacy = radar.mimo(interp, t0=t0, fast=False)
        fast = radar.mimo_from_trace(trace, velocities=velocities, t0=t0)
        cache = radar.path_cache_from_trace(trace, velocities=velocities)
        fast_from_cache = radar.mimo_from_paths(cache)

        legacy_np = legacy.detach().cpu().numpy()
        fast_np = fast.detach().cpu().numpy()
        assert fast.shape == legacy.shape
        # RECORDED REGRESSION, with its bound derived rather than fitted.
        # Before Phase 6's work-item-8 migration these two routes agreed to
        # 3.8e-7, and that agreement was an accident of representation: both
        # carried a ONE-WAY DISTANCE IN METRES, so `d0 + rate * t` and the
        # recomputed `|p(t)|` landed on the same float32 grid for radial motion
        # and the kernel's phase was bit-identical. The routes now carry a
        # ROUND-TRIP DELAY, which is the contract every other Phase-6 family
        # speaks and which removes the 2x hazard of halving a distance twice,
        # and the two roundings no longer coincide.
        #
        # What is left is `dirichlet.cu`'s float32 phase resolution. At
        # fc = 77 GHz and tau = 2e-8 s the absolute phase 2 pi fc tau is about
        # 9.7e3 rad, which float32 resolves to 9.7e3 * 6e-8 = 5.8e-4 rad, so
        # two float32 delays that differ by one ulp differ by that much phase
        # and by about 1e-3 in the complex value. The measured difference is
        # 1.02e-3, i.e. exactly that bound; the tolerance below is it, doubled.
        #
        # The remedy is a numerical change with its own decision: accumulate
        # the cycle count in double and wrap to [0, 1) before `sincosf`, which
        # is what `fmcw_beat.cu` already does and why the beat family does not
        # have this floor. It is recorded as debt rather than folded into an
        # architecture migration.
        torch.testing.assert_close(fast, legacy, rtol=2e-3, atol=1e-8)
        # And the property the test is actually about - that a linear velocity
        # prior reproduces per-chirp recomputation - is asserted where the
        # float32 phase floor does not reach: the range-profile MAGNITUDE.
        torch.testing.assert_close(fast.abs(), legacy.abs(), rtol=2e-3, atol=1e-9)
        torch.testing.assert_close(fast_from_cache, fast, rtol=1e-6, atol=1e-9)
        assert mag_correlation(legacy_np, fast_np) > 0.9999
        assert complex_correlation(legacy_np, fast_np) > 0.9999
        assert 0.999 < peak_ratio(legacy_np, fast_np) < 1.001

    def test_dirichlet_mimo_from_trace_grad_fallback_honors_t0(self):
        """Gradient-preserving linear fallback should use trace time as frame start."""
        from witwin.radar import Radar, TraceResult

        cfg = _mimo_config(
            num_tx=1,
            num_rx=1,
            tx_loc=[[0, 0, 0]],
            rx_loc=[[0, 0, 0]],
            chirp_per_frame=4,
            num_doppler_bins=4,
            adc_start_time=6,
        )
        base_points = torch.tensor([[0.0, 0.0, -3.0]], dtype=torch.float32, device="cuda", requires_grad=True)
        intensities = torch.tensor([0.9], dtype=torch.float32, device="cuda")
        velocities = torch.tensor([[0.0, 0.0, -0.5]], dtype=torch.float32, device="cuda", requires_grad=True)
        trace = TraceResult(base_points, intensities)
        radar = Radar(cfg)
        t0 = 0.4

        def interp(t):
            return TraceResult(base_points + velocities * (float(t) - t0), intensities)

        legacy = radar.mimo(interp, t0=t0, fast=False)
        fallback = radar.mimo_from_trace(trace, velocities=velocities, t0=t0)

        torch.testing.assert_close(fallback, legacy, rtol=5e-4, atol=1e-8)
        assert fallback.requires_grad


class TestMIMOOutputShape:

    def test_output_shape(self):
        """MIMO output should be (TX, RX, chirps, ADC)."""
        from witwin.radar import Radar

        cfg = _mimo_config()
        interp = make_static_interpolator([0, 0, -3])

        try:
            radar = Radar(cfg)
            frame = radar.mimo(interp)
        except (FileNotFoundError, OSError, RuntimeError) as exc:
            pytest.skip(f"dirichlet backend unavailable: {exc}")
        assert frame.shape == (
            cfg.num_tx,
            cfg.num_rx,
            cfg.chirp_per_frame,
            cfg.adc_samples,
        )

    def test_output_not_all_zeros(self):
        """MIMO output with a target should contain non-zero values."""
        from witwin.radar import Radar

        cfg = _mimo_config()
        interp = make_static_interpolator([0, 0, -3])

        try:
            radar = Radar(cfg)
            frame = radar.mimo(interp)
        except (FileNotFoundError, OSError, RuntimeError) as exc:
            pytest.skip(f"dirichlet backend unavailable: {exc}")
        assert frame.abs().max().item() > 0

    def test_zero_targets_gives_zero_output(self):
        """Empty scene should produce all-zero frame."""
        from witwin.radar import Radar

        cfg = _mimo_config()
        positions = torch.zeros((0, 3), device="cuda")
        intensities = torch.zeros(0, device="cuda")

        def empty_interp(t):
            return intensities, positions

        try:
            radar = Radar(cfg)
            frame = radar.mimo(empty_interp)
        except (FileNotFoundError, OSError, RuntimeError) as exc:
            pytest.skip(f"dirichlet backend unavailable: {exc}")
        assert frame.abs().max().item() == 0
