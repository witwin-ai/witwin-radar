"""Differentiable MIMO on a real (multi-antenna, multi-target) configuration.

Regression for the gradient-reference broadcast bug: amplitudes were flattened
across antenna pairs, which crashed for N > 1 targets and silently mixed
per-pair amplitudes for N == 1.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.gpu

_CONFIG = {
    "num_tx": 3, "num_rx": 4,
    "fc": 77e9, "slope": 60.012,
    "adc_samples": 32, "adc_start_time": 6,
    "sample_rate": 4400, "idle_time": 7, "ramp_end_time": 58,
    "chirp_per_frame": 2, "frame_per_second": 10,
    "num_doppler_bins": 2, "num_range_bins": 32,
    "num_angle_bins": 8, "power": 12,
    "tx_loc": [[0, 0, 0], [2, 0, 0], [0, 1, 0]],
    "rx_loc": [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]],
}


def _make_radar():
    from witwin.radar import Radar, RadarConfig

    try:
        return Radar(RadarConfig.from_dict(_CONFIG))
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        pytest.skip(f"radar runtime unavailable: {exc}")


def _loss(frame):
    return frame.abs().square().sum()


def test_multi_target_mimo_backward_runs():
    radar = _make_radar()
    positions = torch.tensor(
        [[0.0, 0.0, -3.0], [0.5, -0.2, -4.0], [-0.4, 0.3, -2.5]],
        device="cuda",
        requires_grad=True,
    )
    intensities = torch.tensor([1.0, 0.6, 0.8], device="cuda", requires_grad=True)

    frame = radar.mimo(lambda t: (intensities, positions))
    _loss(frame).backward()

    assert positions.grad is not None and torch.isfinite(positions.grad).all()
    assert intensities.grad is not None and torch.isfinite(intensities.grad).all()
    assert positions.grad.abs().sum() > 0
    assert intensities.grad.abs().sum() > 0


def test_mimo_reference_gradient_matches_finite_difference():
    """AD gradient of the float64 reference (the MIMO autograd source) vs FD."""
    from witwin.radar.solvers.common import collect_interpolated_samples, pytorch_mimo_from_samples

    radar = _make_radar()
    base = torch.tensor([[0.0, 0.0, -3.0], [0.5, -0.2, -4.0]], device="cuda")
    intensities = torch.tensor([1.0, 0.6], device="cuda")

    def reference_loss(positions):
        samples = collect_interpolated_samples(radar, lambda t: (intensities, positions))
        return _loss(pytorch_mimo_from_samples(radar, samples))

    positions = base.clone().requires_grad_(True)
    reference_loss(positions).backward()
    ad_grad = float(positions.grad[0, 2].item())

    eps = 2.0 ** -14  # power of two: exactly representable around float32 coordinates
    delta = torch.zeros_like(base)
    delta[0, 2] = eps
    fd_grad = float((reference_loss(base + delta) - reference_loss(base - delta)).item()) / (2 * eps)

    assert ad_grad == pytest.approx(fd_grad, rel=0.03), (
        f"AD gradient {ad_grad:.6e} vs finite difference {fd_grad:.6e}"
    )
