from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from witwin.radar import Radar


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


def _make_radar(*, noise_model=None) -> Radar:
    config = _base_config()
    if noise_model is not None:
        config["noise_model"] = noise_model
    return Radar(config, backend="pytorch", device="cpu")


def _static_interpolator(radar: Radar, position=(0.0, 0.0, -3.0), intensity=1.0):
    position_tensor = torch.tensor([position], dtype=torch.float32, device=radar.device)
    intensity_tensor = torch.tensor([intensity], dtype=torch.float32, device=radar.device)

    def interp(_t):
        return intensity_tensor, position_tensor

    return interp


def _expected_phase(signal: torch.Tensor, *, std: float, seed: int) -> torch.Tensor:
    real_dtype = signal.real.dtype
    if signal.ndim == 4:
        phase_shape = signal.shape[-2:]
        broadcast_shape = (1, 1, *phase_shape)
    elif signal.ndim == 2:
        phase_shape = signal.shape
        broadcast_shape = phase_shape
    elif signal.ndim == 1:
        phase_shape = signal.shape
        broadcast_shape = phase_shape
    else:
        raise ValueError("Unsupported test tensor shape.")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    innovations = torch.randn(phase_shape, generator=generator, dtype=real_dtype) * std
    phase = torch.cumsum(innovations.reshape(-1), dim=0).reshape(phase_shape).reshape(broadcast_shape)
    factor = torch.polar(torch.ones_like(phase, dtype=real_dtype), phase).to(dtype=signal.dtype)
    return signal * factor


def _quantize_complex(signal: torch.Tensor, *, bits: int, full_scale: float) -> torch.Tensor:
    levels = 2 ** bits
    step = (2.0 * full_scale) / (levels - 1)

    def _quantize(component: torch.Tensor) -> torch.Tensor:
        clipped = torch.clamp(component, min=-full_scale, max=full_scale)
        code = torch.round((clipped + full_scale) / step)
        return code * step - full_scale

    return torch.complex(_quantize(signal.real), _quantize(signal.imag)).to(dtype=signal.dtype)


def test_missing_noise_model_leaves_signal_unchanged():
    radar = _make_radar()
    signal = torch.tensor([1.0 + 2.0j, -0.5 + 0.25j], dtype=torch.complex64)

    observed = radar.apply_noise(signal)

    torch.testing.assert_close(observed, signal, atol=0, rtol=0)


def test_thermal_noise_runtime_matches_seeded_reference():
    signal = torch.zeros((4, 8), dtype=torch.complex64)
    radar = _make_radar(
        noise_model={
            "thermal": {"std": 0.05},
            "seed": 17,
        }
    )

    observed = radar.apply_noise(signal)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(17)
    real = torch.randn(signal.shape, generator=generator, dtype=signal.real.dtype) * 0.05
    imag = torch.randn(signal.shape, generator=generator, dtype=signal.real.dtype) * 0.05
    expected = torch.complex(real, imag)

    torch.testing.assert_close(observed, expected, atol=0, rtol=0)


def test_quantization_runtime_matches_expected_levels():
    radar = _make_radar(
        noise_model={
            "quantization": {"bits": 2, "full_scale": 1.0},
        }
    )
    signal = torch.tensor(
        [
            -0.9 + 0.4j,
            -0.1 - 0.4j,
            0.2 + 1.2j,
            0.9 - 1.2j,
        ],
        dtype=torch.complex64,
    )

    observed = radar.apply_noise(signal)
    expected = _quantize_complex(signal, bits=2, full_scale=1.0)

    torch.testing.assert_close(observed, expected, atol=1e-6, rtol=0)


def test_phase_noise_runtime_preserves_magnitude_and_is_shared_across_channels():
    signal = torch.ones((2, 3, 4, 8), dtype=torch.complex64)
    radar = _make_radar(
        noise_model={
            "phase": {"std": 0.02},
            "seed": 23,
        }
    )

    observed = radar.apply_noise(signal)
    expected = _expected_phase(signal, std=0.02, seed=23)

    torch.testing.assert_close(observed, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(observed.abs(), signal.abs(), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(observed[0, 0], observed[1, 2], atol=1e-6, rtol=1e-6)


def test_chirp_and_frame_time_domain_outputs_apply_noise():
    radar = _make_radar(
        noise_model={
            "quantization": {"bits": 3, "full_scale": 1.0},
        }
    )
    chirp_signal = torch.tensor([0.1 + 0.6j, 0.8 - 0.2j], dtype=torch.complex64)
    frame_signal = torch.tensor(
        [
            [0.1 + 0.6j, 0.8 - 0.2j],
            [-0.7 + 0.9j, 1.2 - 1.4j],
        ],
        dtype=torch.complex64,
    )
    radar.solver = SimpleNamespace(
        chirp=lambda distances, amplitudes: chirp_signal,
        frame=lambda interpolator, t0: frame_signal,
        mimo=lambda interpolator, t0, **options: frame_signal.unsqueeze(0).unsqueeze(0),
    )

    chirp_observed = radar.chirp(None, None)
    frame_observed = radar.frame(lambda t: None)

    torch.testing.assert_close(chirp_observed, _quantize_complex(chirp_signal, bits=3, full_scale=1.0), atol=1e-6, rtol=0)
    torch.testing.assert_close(frame_observed, _quantize_complex(frame_signal, bits=3, full_scale=1.0), atol=1e-6, rtol=0)


def test_mimo_thermal_noise_matches_measured_signal_plus_seeded_noise():
    clean_radar = _make_radar()
    noisy_radar = _make_radar(
        noise_model={
            "thermal": {"std": 0.03},
            "seed": 31,
        }
    )
    interp = _static_interpolator(clean_radar)

    clean = clean_radar.mimo(interp)
    noisy = noisy_radar.mimo(_static_interpolator(noisy_radar))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(31)
    real = torch.randn(clean.shape, generator=generator, dtype=clean.real.dtype) * 0.03
    imag = torch.randn(clean.shape, generator=generator, dtype=clean.real.dtype) * 0.03
    expected = clean + torch.complex(real, imag).to(dtype=clean.dtype)

    torch.testing.assert_close(noisy, expected, atol=1e-6, rtol=1e-6)


def test_mimo_quantization_matches_quantized_measured_signal():
    clean_radar = _make_radar()
    noisy_radar = _make_radar(
        noise_model={
            "quantization": {"bits": 4, "full_scale": 1.0},
        }
    )

    clean = clean_radar.mimo(_static_interpolator(clean_radar))
    noisy = noisy_radar.mimo(_static_interpolator(noisy_radar))
    expected = _quantize_complex(clean, bits=4, full_scale=1.0)

    torch.testing.assert_close(noisy, expected, atol=1e-6, rtol=1e-6)


def test_mimo_phase_noise_matches_measured_signal_and_keeps_channel_ratio():
    clean_radar = _make_radar()
    noisy_radar = _make_radar(
        noise_model={
            "phase": {"std": 0.01},
            "seed": 43,
        }
    )

    clean = clean_radar.mimo(_static_interpolator(clean_radar))
    noisy = noisy_radar.mimo(_static_interpolator(noisy_radar))
    expected = _expected_phase(clean, std=0.01, seed=43)

    torch.testing.assert_close(noisy, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(noisy.abs(), clean.abs(), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        noisy[0, 0] * clean[1, 2],
        noisy[1, 2] * clean[0, 0],
        atol=1e-6,
        rtol=1e-6,
    )


def test_mimo_freq_domain_output_is_rejected_when_noise_model_is_enabled():
    radar = _make_radar(
        noise_model={
            "thermal": {"std": 0.01},
        }
    )

    with pytest.raises(ValueError, match="time-domain mimo output"):
        radar.mimo(lambda t: None, freq_domain=True)


def test_process_rd_with_noise_model_changes_rd_map_but_preserves_shape():
    from witwin.radar.sigproc import process_rd

    clean_radar = _make_radar()
    noisy_radar = _make_radar(
        noise_model={
            "thermal": {"std": 0.02},
            "seed": 19,
        }
    )

    clean_frame = clean_radar.mimo(_static_interpolator(clean_radar))
    noisy_frame = noisy_radar.mimo(_static_interpolator(noisy_radar))

    clean_rd, clean_map, clean_ranges, clean_velocities = process_rd(clean_radar, clean_frame)
    noisy_rd, noisy_map, noisy_ranges, noisy_velocities = process_rd(noisy_radar, noisy_frame)

    assert clean_rd.shape == noisy_rd.shape
    assert clean_map.shape == noisy_map.shape
    assert clean_ranges.shape == noisy_ranges.shape
    assert clean_velocities.shape == noisy_velocities.shape
    assert not np.allclose(clean_rd, noisy_rd)
