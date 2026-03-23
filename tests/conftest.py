"""
Pytest configuration and shared fixtures for the radar test suite.

Run:
    cd radar
    pytest tests/                         # CPU-only tests
    pytest tests/ --gpu                   # include GPU tests (needs CUDA)
    pytest tests/sigproc/ -v              # single subfolder
"""

import sys
import os

import numpy as np
import pytest

# Ensure witwin.radar is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# pytest plugins
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    try:
        parser.addoption(
            "--gpu", action="store_true", default=False,
            help="Run GPU-only tests (solver cross-validation, end-to-end validation)",
        )
    except ValueError:
        pass


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: test requires CUDA GPU")


def pytest_collection_modifyitems(config, items):
    import torch

    run_gpu = config.getoption("--gpu") and torch.cuda.is_available()
    if run_gpu:
        return

    skip_gpu = pytest.mark.skip(reason="needs --gpu flag and CUDA device")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)


# ---------------------------------------------------------------------------
# Standard radar configurations
# ---------------------------------------------------------------------------

STANDARD_CONFIG = {
    "num_tx": 3, "num_rx": 4,
    "fc": 77e9, "slope": 60.012,
    "adc_samples": 256, "adc_start_time": 6,
    "sample_rate": 4400, "idle_time": 7, "ramp_end_time": 58,
    "chirp_per_frame": 128, "frame_per_second": 10,
    "num_doppler_bins": 128, "num_range_bins": 256,
    "num_angle_bins": 64, "power": 12,
    "tx_loc": [[0, 0, 0], [2, 0, 0], [0, 1, 0]],
    "rx_loc": [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]],
}

FAST_CONFIG = {
    **STANDARD_CONFIG,
    "chirp_per_frame": 32,
    "num_doppler_bins": 32,
}

MINIMAL_CONFIG = {
    "num_tx": 1, "num_rx": 1,
    "fc": 77e9, "slope": 60.012,
    "adc_samples": 256, "adc_start_time": 0,
    "sample_rate": 4400, "idle_time": 7, "ramp_end_time": 58,
    "chirp_per_frame": 2, "frame_per_second": 10,
    "num_doppler_bins": 2, "num_range_bins": 256,
    "num_angle_bins": 64, "power": 12,
    "tx_loc": [[0, 0, 0]],
    "rx_loc": [[0, 0, 0]],
}


# ---------------------------------------------------------------------------
# CPU-only mock for sigproc tests that need FrameConfig / PointCloudProcessConfig
# ---------------------------------------------------------------------------

class MockRadar:
    """Lightweight CPU-only mock providing the attributes needed by sigproc code."""

    def __init__(self, config=None):
        config = config or STANDARD_CONFIG
        self.c0 = 299792458
        self.num_tx = config["num_tx"]
        self.num_rx = config["num_rx"]
        self.fc = config["fc"]
        self.slope = config["slope"]
        self.adc_samples = config["adc_samples"]
        self.adc_start_time = config.get("adc_start_time", 0)
        self.sample_rate = config["sample_rate"]
        self.idle_time = config["idle_time"]
        self.ramp_end_time = config["ramp_end_time"]
        self.chirp_per_frame = config["chirp_per_frame"]
        self.num_angle_bins = config["num_angle_bins"]
        self.power = config.get("power", 12)

        self._lambda = self.c0 / self.fc
        antenna_spacing = self._lambda / 2
        self.tx_loc = np.array(config["tx_loc"], dtype=np.float32) * antenna_spacing
        self.rx_loc = np.array(config["rx_loc"], dtype=np.float32) * antenna_spacing

        fs = self.sample_rate * 1e3
        S = self.slope * 1e12
        self.range_resolution = self.c0 * fs / (2 * S * self.adc_samples)
        self.max_range = self.c0 * fs / (2 * S)

        T_chirp = (self.idle_time + self.ramp_end_time) * 1e-6
        T_eff = T_chirp * self.num_tx
        Nd = self.chirp_per_frame
        self.doppler_resolution = self._lambda / (2 * Nd * T_eff)
        self.max_doppler = self._lambda / (4 * T_chirp * self.num_tx)

        self.gain = 1.0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def standard_config():
    return STANDARD_CONFIG.copy()


@pytest.fixture
def fast_config():
    return FAST_CONFIG.copy()


@pytest.fixture
def minimal_config():
    return MINIMAL_CONFIG.copy()


@pytest.fixture
def mock_radar():
    return MockRadar(STANDARD_CONFIG)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mag_correlation(a, b):
    """Pearson correlation of magnitudes (works for numpy and torch)."""
    a_m = np.abs(np.asarray(a).ravel()).astype(np.float64)
    b_m = np.abs(np.asarray(b).ravel()).astype(np.float64)
    a_c = a_m - a_m.mean()
    b_c = b_m - b_m.mean()
    denom = np.linalg.norm(a_c) * np.linalg.norm(b_c)
    if denom < 1e-30:
        return 1.0 if np.linalg.norm(a_c) < 1e-30 and np.linalg.norm(b_c) < 1e-30 else 0.0
    return float(np.dot(a_c, b_c) / denom)


def complex_correlation(a, b):
    """Normalized complex inner-product correlation."""
    a_c = np.asarray(a).ravel().astype(np.complex128)
    b_c = np.asarray(b).ravel().astype(np.complex128)
    denom = np.linalg.norm(a_c) * np.linalg.norm(b_c)
    if denom < 1e-30:
        return 1.0 if np.linalg.norm(a_c) < 1e-30 and np.linalg.norm(b_c) < 1e-30 else 0.0
    return float(np.abs(np.vdot(a_c, b_c)) / denom)


def peak_ratio(a, b):
    """Ratio of maximum magnitudes, matching verify.py style checks."""
    a_peak = float(np.abs(np.asarray(a)).max())
    b_peak = float(np.abs(np.asarray(b)).max())
    if a_peak < 1e-30 or b_peak < 1e-30:
        return 1.0 if max(a_peak, b_peak) < 1e-30 else 0.0
    return b_peak / a_peak


def make_static_interpolator(pos, sigma=1.0):
    """Create an interpolator for a static target (GPU tensors)."""
    import torch
    pos_t = torch.tensor([pos], dtype=torch.float32, device="cuda")
    sigma_t = torch.tensor([sigma], dtype=torch.float32, device="cuda")

    def interp(t):
        return sigma_t, pos_t

    return interp


def make_moving_interpolator(pos0, velocity, sigma=1.0):
    """Create an interpolator for a linearly moving target (GPU tensors)."""
    import torch
    pos0_t = torch.tensor(pos0, dtype=torch.float32, device="cuda")
    vel_t = torch.tensor(velocity, dtype=torch.float32, device="cuda")
    sigma_t = torch.tensor([sigma], dtype=torch.float32, device="cuda")

    def interp(t):
        pos = (pos0_t + vel_t * t).unsqueeze(0)
        return sigma_t, pos

    return interp


def make_radar_or_skip(config, *, backend):
    """Construct a Radar backend or skip when the local toolchain is missing."""
    from witwin.radar import Radar

    try:
        return Radar(config, backend=backend)
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        pytest.skip(f"{backend} backend unavailable: {exc}")
