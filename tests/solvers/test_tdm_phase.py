"""TDM-MIMO per-TX motion phase: the solver must simulate the phase that
sigproc's _compensate_tdm_phase removes.

Regression for the bug where all TX antennas shared one scene sample per chirp
loop, so the velocity-dependent TDM phase was missing and the downstream
compensation corrupted AoA for moving targets.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.gpu

_CONFIG = {
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

_SPEED = 1.5  # m/s, receding along -Z


def _make_radar():
    from witwin.radar import Radar, RadarConfig

    try:
        return Radar(RadarConfig.from_dict(_CONFIG))
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        pytest.skip(f"radar runtime unavailable: {exc}")


def _rd_cube(frame):
    rng = torch.fft.fft(frame, dim=-1)
    return torch.fft.fftshift(torch.fft.fft(rng, dim=2), dim=2)


def _peak_bin(dop):
    mag = dop.abs().sum(dim=(0, 1))
    mag[:, 0] = 0  # skip DC range bin
    idx = int(torch.argmax(mag))
    return idx // mag.shape[1], idx % mag.shape[1]


def _per_tx_motion_phase(radar, frame_static, frame_moving):
    """Slow-time phase of each TX row relative to TX0, geometry cancelled
    by referencing the static frame."""
    dop_s, dop_m = _rd_cube(frame_static), _rd_cube(frame_moving)
    ds, rs = _peak_bin(dop_s)
    dm, rm = _peak_bin(dop_m)
    va_s = dop_s[:, :, ds, rs]
    va_m = dop_m[:, :, dm, rm]
    dphase = torch.angle(va_m * va_s.conj())
    per_tx = torch.angle(torch.exp(1j * dphase.to(torch.complex64)).mean(dim=1))
    return (per_tx - per_tx[0]).cpu().numpy()


def _tdm_phase_per_slot(radar):
    t_chirp = (radar.config.idle_time + radar.config.ramp_end_time) * 1e-6
    return 4 * np.pi * _SPEED * t_chirp / radar._lambda


class TestTDMMotionPhase:

    def test_interpolator_path_simulates_per_tx_phase(self):
        radar = _make_radar()
        pos0 = torch.tensor([0.0, 0.0, -3.0], device="cuda")
        vel = torch.tensor([0.0, 0.0, -_SPEED], device="cuda")
        sigma = torch.tensor([1.0], device="cuda")

        frame_static = radar.mimo(lambda t: (sigma, pos0.unsqueeze(0)))
        frame_moving = radar.mimo(lambda t: (sigma, (pos0 + vel * t).unsqueeze(0)))

        rel = _per_tx_motion_phase(radar, frame_static, frame_moving)
        theta = _tdm_phase_per_slot(radar)
        expected = np.array([0.0, theta, 2 * theta])
        # The Doppler phase slope tracks the effective carrier (fc + slope*t_mid),
        # ~3% above fc; allow that plus estimation noise.
        assert np.abs(rel - expected).max() < 0.15 * theta + 0.02, (
            f"per-TX phase {rel} does not match TDM expectation {expected}"
        )

    def test_linear_rate_path_simulates_per_tx_phase(self):
        from witwin.radar import TraceResult

        radar = _make_radar()
        pos0 = torch.tensor([[0.0, 0.0, -3.0]], device="cuda")
        vel = torch.tensor([[0.0, 0.0, -_SPEED]], device="cuda")
        sigma = torch.tensor([1.0], device="cuda")
        trace = TraceResult(pos0, sigma)

        frame_static = radar.mimo_from_trace(trace)
        frame_moving = radar.mimo_from_trace(trace, velocities=vel)

        rel = _per_tx_motion_phase(radar, frame_static, frame_moving)
        theta = _tdm_phase_per_slot(radar)
        expected = np.array([0.0, theta, 2 * theta])
        assert np.abs(rel - expected).max() < 0.15 * theta + 0.02, (
            f"per-TX phase {rel} does not match TDM expectation {expected}"
        )

    def test_interpolator_and_linear_paths_agree_for_moving_target(self):
        from witwin.radar import TraceResult

        radar = _make_radar()
        pos0 = torch.tensor([[0.2, -0.1, -3.0]], device="cuda")
        vel = torch.tensor([[0.0, 0.0, -_SPEED]], device="cuda")
        sigma = torch.tensor([1.0], device="cuda")

        frame_loop = radar.mimo(lambda t: (sigma, pos0 + vel * t))
        frame_linear = radar.mimo_from_trace(TraceResult(pos0, sigma), velocities=vel)

        # The linear path freezes antenna-pattern/incidence terms and uses a
        # first-order range-rate model, so a small model difference remains;
        # a per-TX timing mismatch between the paths would show up as ~0.3.
        rel_diff = (frame_loop - frame_linear).abs().max() / frame_loop.abs().max()
        assert rel_diff < 2e-2, f"loop vs linear path relative diff {rel_diff:.2e}"
