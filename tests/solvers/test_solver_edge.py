"""Solver edge-case and regression tests."""

import numpy as np
import pytest
import torch

from conftest import make_radar_or_skip, make_static_interpolator

pytestmark = pytest.mark.gpu


def _single_pair_config(*, adc_samples=256, slope=60.012, sample_rate=4400):
    return {
        "num_tx": 1,
        "num_rx": 1,
        "fc": 77e9,
        "slope": slope,
        "adc_samples": adc_samples,
        "adc_start_time": 0,
        "sample_rate": sample_rate,
        "idle_time": 7,
        "ramp_end_time": 58,
        "chirp_per_frame": 1,
        "frame_per_second": 10,
        "num_doppler_bins": 1,
        "num_range_bins": adc_samples,
        "num_angle_bins": 64,
        "power": 12,
        "tx_loc": [[0, 0, 0]],
        "rx_loc": [[0, 0, 0]],
    }


class TestDirichletLargeADC:
    def test_dirichlet_mimo_512_samples(self):
        cfg = _single_pair_config(adc_samples=512)
        r = make_radar_or_skip(cfg, backend="dirichlet")
        frame = r.mimo(make_static_interpolator([0, 0, -3]))

        second_half = frame[0, 0, 0, 256:].abs().cpu()
        assert second_half.max().item() > 0

    def test_dirichlet_mimo_640_samples(self):
        cfg = _single_pair_config(adc_samples=640, slope=128.0, sample_rate=12500)
        r = make_radar_or_skip(cfg, backend="dirichlet")
        frame = r.mimo(make_static_interpolator([0, 0, -3]))

        last_quarter = frame[0, 0, 0, 480:].abs().cpu()
        assert last_quarter.max().item() > 0

    def test_dirichlet_512_produces_valid_spectrum(self):
        cfg = _single_pair_config(adc_samples=512, slope=128.0, sample_rate=12500)
        r = make_radar_or_skip(cfg, backend="dirichlet")
        frame = r.mimo(make_static_interpolator([0, 0, -2]))
        spectrum = torch.fft.fft(frame[0, 0, 0, :])
        mag = spectrum.abs().cpu().numpy()
        assert mag[:256].max() > mag[:256].mean() * 5


class TestChirpEdgeCases:
    def test_very_close_target(self):
        r = make_radar_or_skip(_single_pair_config(), backend="dirichlet")
        d = torch.tensor([0.1], dtype=torch.float32, device="cuda")
        a = torch.tensor([1.0], dtype=torch.float32, device="cuda")
        spectrum = r.chirp(d, a)
        assert not torch.isnan(spectrum).any()
        assert spectrum.abs().max().item() > 0

    def test_large_target_count(self):
        rng = np.random.RandomState(0)
        d = torch.tensor(rng.uniform(0.5, 8, 4096), dtype=torch.float32, device="cuda")
        a = torch.tensor(rng.uniform(0.01, 1, 4096), dtype=torch.float32, device="cuda")

        r = make_radar_or_skip(_single_pair_config(), backend="dirichlet")
        result = r.chirp(d, a)
        assert not torch.isnan(result).any()
