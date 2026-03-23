from __future__ import annotations

import math

import pytest
import torch

from witwin.radar import Radar
from witwin.radar.noise import quantize_complex_signal


def _base_config() -> dict:
    return {
        "num_tx": 2,
        "num_rx": 3,
        "fc": 77e9,
        "slope": 60.012,
        "adc_samples": 16,
        "adc_start_time": 0,
        "sample_rate": 4400,
        "idle_time": 7,
        "ramp_end_time": 58,
        "chirp_per_frame": 4,
        "frame_per_second": 10,
        "num_doppler_bins": 4,
        "num_range_bins": 16,
        "num_angle_bins": 16,
        "power": 12,
        "tx_loc": [[0, 0, 0], [2, 0, 0]],
        "rx_loc": [[0, 0, 0], [1, 0, 0], [2, 0, 0]],
    }


def _make_radar(*, receiver_chain=None, noise_model=None) -> Radar:
    config = _base_config()
    if receiver_chain is not None:
        config["receiver_chain"] = receiver_chain
    if noise_model is not None:
        config["noise_model"] = noise_model
    return Radar(config, backend="pytorch", device="cpu")


def _static_interpolator(radar: Radar, position=(0.0, 0.0, -3.0), intensity=1.0):
    position_tensor = torch.tensor([position], dtype=torch.float32, device=radar.device)
    intensity_tensor = torch.tensor([intensity], dtype=torch.float32, device=radar.device)
    normals = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32, device=radar.device)

    def interp(_t):
        return intensity_tensor, position_tensor

    def interp_with_normals(_t):
        return type("Sample", (), {"intensities": intensity_tensor, "points": position_tensor, "normals": normals})()

    return interp, interp_with_normals


def test_missing_receiver_chain_leaves_signal_unchanged():
    radar = _make_radar()
    signal = torch.tensor([1.0 + 2.0j, -0.5 + 0.25j], dtype=torch.complex64)

    observed = radar.apply_receiver_chain(signal)

    torch.testing.assert_close(observed, signal, atol=0, rtol=0)


def test_lna_gain_runtime_matches_voltage_gain():
    radar = _make_radar(receiver_chain={"lna": {"gain_db": 20.0}})
    signal = torch.tensor([0.1 + 0.2j, -0.3 + 0.05j], dtype=torch.complex64)

    observed = radar.apply_receiver_chain(signal)

    torch.testing.assert_close(observed, signal * 10.0, atol=1e-6, rtol=0)


def test_agc_targets_per_rx_rms():
    radar = _make_radar(receiver_chain={"agc": {"target_rms": 0.25, "mode": "per_rx"}})
    signal = torch.zeros((2, 3, 4, 8), dtype=torch.complex64)
    signal[:, 0] = 0.05 + 0.0j
    signal[:, 1] = 0.20 + 0.0j
    signal[:, 2] = 0.50 + 0.0j

    observed = radar.apply_receiver_chain(signal)
    rms = torch.sqrt((observed.real.square() + observed.imag.square()).mean(dim=(0, 2, 3)))

    torch.testing.assert_close(rms, torch.full((3,), 0.25, dtype=rms.dtype), atol=1e-5, rtol=1e-5)


def test_adc_quantization_runtime_matches_shared_helper():
    radar = _make_radar(receiver_chain={"adc": {"bits": 3, "full_scale": 1.0}})
    signal = torch.tensor([0.2 + 0.6j, 1.4 - 1.2j, -0.8 + 0.1j], dtype=torch.complex64)

    observed = radar.apply_receiver_chain(signal)
    expected = quantize_complex_signal(signal, bits=3, full_scale=1.0)

    torch.testing.assert_close(observed, expected, atol=1e-6, rtol=0)


def test_receiver_chain_uses_power_for_absolute_gain_scale():
    clean_radar = _make_radar()
    physical_radar = _make_radar(receiver_chain={"lna": {"gain_db": 0.0}})
    interp, _ = _static_interpolator(clean_radar)

    clean = clean_radar.mimo(interp)
    physical = physical_radar.mimo(interp)
    expected_scale = physical_radar.tx_voltage_rms

    torch.testing.assert_close(physical / clean, torch.full_like(physical, expected_scale), atol=1e-5, rtol=1e-5)


def test_receiver_chain_full_path_matches_measured_signal_transform():
    clean_radar = _make_radar()
    chain = {
        "lna": {"gain_db": 20.0},
        "agc": {"target_rms": 0.2, "mode": "global", "max_gain_db": 40.0, "min_gain_db": -40.0},
        "adc": {"bits": 10, "full_scale": 1.0},
    }
    chained_radar = _make_radar(receiver_chain=chain)
    interp, _ = _static_interpolator(clean_radar)

    clean = clean_radar.mimo(interp) * chained_radar.tx_voltage_rms
    analog = clean * 10.0
    rms = torch.sqrt(torch.clamp((analog.real.square() + analog.imag.square()).mean(), min=1e-24))
    agc_gain = 0.2 / rms
    agc_gain = torch.clamp(agc_gain, min=10 ** (-40.0 / 20.0), max=10 ** (40.0 / 20.0))
    expected = quantize_complex_signal(analog * agc_gain, bits=10, full_scale=1.0)
    observed = chained_radar.mimo(interp)

    torch.testing.assert_close(observed, expected, atol=1e-5, rtol=1e-5)


def test_receiver_chain_can_follow_noise_model_without_adc_conflict():
    radar = _make_radar(
        receiver_chain={"lna": {"gain_db": 6.0}, "adc": {"bits": 12, "full_scale": 1.0}},
        noise_model={"thermal": {"std": 0.01}, "seed": 11},
    )
    signal = torch.zeros((4, 8), dtype=torch.complex64)

    observed = radar.apply_signal_models(signal)

    assert observed.abs().max() > 0
    assert torch.is_complex(observed)


def test_mimo_freq_domain_output_is_rejected_when_receiver_chain_is_enabled():
    radar = _make_radar(receiver_chain={"lna": {"gain_db": 12.0}})

    with pytest.raises(ValueError, match="time-domain mimo output"):
        radar.mimo(lambda t: None, freq_domain=True)
