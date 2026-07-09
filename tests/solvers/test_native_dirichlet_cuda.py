"""Native CUDA Dirichlet kernel correctness tests."""

from __future__ import annotations

import pytest
import torch

from conftest import MINIMAL_CONFIG

pytestmark = pytest.mark.gpu


def _requires_cuda():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")


def _make_radar():
    from witwin.radar import Radar, RadarConfig

    cfg = RadarConfig.from_dict(
        {
            **MINIMAL_CONFIG,
            "adc_samples": 64,
            "num_range_bins": 64,
            "chirp_per_frame": 2,
            "num_doppler_bins": 2,
        }
    )
    return Radar(cfg, backend="dirichlet", device="cuda")


def test_native_dirichlet_chirp_matches_pytorch_fft_reference():
    _requires_cuda()
    from witwin.radar.solvers.common import pytorch_chirp_reference

    radar = _make_radar()
    distances = torch.tensor([0.8, 1.7, 3.2, 4.5], dtype=torch.float32, device="cuda")
    amplitudes = torch.tensor([0.7, 1.0, 0.4, 0.9], dtype=torch.float32, device="cuda")

    actual = radar.chirp(distances, amplitudes)
    reference = pytorch_chirp_reference(radar, distances.to(torch.float64), amplitudes.to(torch.float64))
    reference = torch.fft.fft(reference, n=radar.solver.N_fft)[: radar.solver.N_fft // 2]

    max_error = (actual.to(reference.dtype) - reference).abs().max()
    peak = reference.abs().max()
    assert max_error / peak < 2e-3


def test_native_dirichlet_mimo_from_paths_static_matches_mimo_reference():
    _requires_cuda()
    from witwin.radar import TraceResult

    radar = _make_radar()
    trace = TraceResult(
        torch.tensor([[0.0, 0.0, -2.0], [0.35, 0.1, -3.5]], dtype=torch.float32, device="cuda"),
        torch.tensor([0.8, 0.4], dtype=torch.float32, device="cuda"),
    )

    reference = radar.mimo(lambda _t: trace, fast=False)
    fast = radar.mimo_from_trace(trace)

    torch.testing.assert_close(fast, reference, rtol=2e-4, atol=1e-6)


def test_native_dirichlet_backward_matches_pytorch_reference_gradients():
    _requires_cuda()
    from witwin.radar.solvers.common import pytorch_chirp_reference

    radar = _make_radar()
    distances = torch.tensor([1.1, 2.4, 3.7], dtype=torch.float32, device="cuda")
    amplitudes = torch.tensor([0.9, 0.6, 0.3], dtype=torch.float32, device="cuda")
    grad_re = torch.linspace(0.2, 1.3, radar.solver.num_bins, dtype=torch.float32, device="cuda")
    grad_im = torch.linspace(-0.7, 0.4, radar.solver.num_bins, dtype=torch.float32, device="cuda")

    actual_d, actual_a = radar.solver.backward(distances, amplitudes, grad_re, grad_im)
    bin_d, bin_a = radar.solver.backward_per_bin(distances, amplitudes, grad_re, grad_im)
    torch.testing.assert_close(bin_d, actual_d, rtol=1e-5, atol=1e-3)
    torch.testing.assert_close(bin_a, actual_a, rtol=1e-5, atol=1e-3)

    ref_d = distances.detach().to(torch.float64).requires_grad_(True)
    ref_a = amplitudes.detach().to(torch.float64).requires_grad_(True)
    reference = pytorch_chirp_reference(radar, ref_d, ref_a)
    reference = torch.fft.fft(reference, n=radar.solver.N_fft)[: radar.solver.N_fft // 2]
    real_dtype = reference.real.dtype
    loss = (reference.real * grad_re.to(real_dtype) + reference.imag * grad_im.to(real_dtype)).sum()
    loss.backward()

    torch.testing.assert_close(actual_d.to(ref_d.grad.dtype), ref_d.grad, rtol=2e-3, atol=2e-2)
    torch.testing.assert_close(actual_a.to(ref_a.grad.dtype), ref_a.grad, rtol=1e-3, atol=2e-3)
