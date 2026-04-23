"""Tests for Radar configuration validation and derived parameters."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import MockRadar, STANDARD_CONFIG
from witwin.radar import RadarConfig


C0 = 299792458


class TestRadarConfigSchema:
    def test_config_round_trip_from_dict(self):
        config = RadarConfig.from_dict(STANDARD_CONFIG)
        assert config.num_tx == STANDARD_CONFIG["num_tx"]
        assert config.tx_loc[1] == tuple(STANDARD_CONFIG["tx_loc"][1])
        assert config.to_dict()["rx_loc"][0] == list(STANDARD_CONFIG["rx_loc"][0])

    def test_antenna_pattern_round_trip_from_dict(self):
        radar_config = RadarConfig.from_dict(
            {
                **STANDARD_CONFIG,
                "antenna_pattern": {
                    "kind": "separable",
                    "x_angles_deg": [-60, 0, 60],
                    "x_values": [0.2, 1.0, 0.2],
                    "y_angles_deg": [-30, 0, 30],
                    "y_values": [0.5, 1.0, 0.5],
                },
            }
        )

        assert radar_config.antenna_pattern is not None
        assert radar_config.antenna_pattern.kind == "separable"
        assert radar_config.antenna_pattern.x_values[1] == pytest.approx(1.0)
        assert radar_config.to_dict()["antenna_pattern"]["y_values"] == [0.5, 1.0, 0.5]

    def test_noise_model_round_trip_from_dict(self):
        radar_config = RadarConfig.from_dict(
            {
                **STANDARD_CONFIG,
                "noise_model": {
                    "thermal": {"std": 0.01},
                    "quantization": {"bits": 12, "full_scale": 0.8},
                    "phase": {"std": 1e-3},
                    "seed": 7,
                },
            }
        )

        assert radar_config.noise_model is not None
        assert radar_config.noise_model.thermal is not None
        assert radar_config.noise_model.thermal.std == pytest.approx(0.01)
        assert radar_config.noise_model.quantization is not None
        assert radar_config.noise_model.quantization.bits == 12
        assert radar_config.noise_model.phase is not None
        assert radar_config.noise_model.phase.std == pytest.approx(1e-3)
        assert radar_config.to_dict()["noise_model"] == {
            "thermal": {"std": 0.01},
            "quantization": {"bits": 12, "full_scale": 0.8},
            "phase": {"std": 0.001},
            "seed": 7,
        }

    def test_polarization_round_trip_from_dict(self):
        radar_config = RadarConfig.from_dict(
            {
                **STANDARD_CONFIG,
                "polarization": {
                    "tx": "horizontal",
                    "rx": ["vertical", "horizontal", "vertical", "horizontal"],
                    "reflection_flip": False,
                },
            }
        )

        assert radar_config.polarization is not None
        assert radar_config.polarization.tx[0] == pytest.approx((1.0, 0.0, 0.0))
        assert radar_config.polarization.rx[0] == pytest.approx((0.0, 1.0, 0.0))
        assert radar_config.polarization.rx[1] == pytest.approx((1.0, 0.0, 0.0))
        assert radar_config.to_dict()["polarization"]["reflection_flip"] is False

    def test_receiver_chain_round_trip_from_dict(self):
        radar_config = RadarConfig.from_dict(
            {
                **STANDARD_CONFIG,
                "receiver_chain": {
                    "reference_impedance_ohm": 75.0,
                    "lna": {"gain_db": 24.0},
                    "agc": {"target_rms": 0.2, "max_gain_db": 20.0, "min_gain_db": -10.0},
                    "adc": {"bits": 12, "full_scale": 0.8},
                },
            }
        )

        assert radar_config.receiver_chain is not None
        assert radar_config.receiver_chain.reference_impedance_ohm == pytest.approx(75.0)
        assert radar_config.receiver_chain.lna is not None
        assert radar_config.receiver_chain.lna.gain_db == pytest.approx(24.0)
        assert radar_config.receiver_chain.agc is not None
        assert radar_config.receiver_chain.agc.target_rms == pytest.approx(0.2)
        assert radar_config.receiver_chain.adc is not None
        assert radar_config.receiver_chain.to_dict()["adc"]["bits"] == 12

    def test_missing_required_key_raises(self):
        broken = dict(STANDARD_CONFIG)
        broken.pop("num_tx")
        with pytest.raises(ValueError, match="missing required keys"):
            RadarConfig.from_dict(broken)

    def test_antenna_count_mismatch_raises(self):
        broken = dict(STANDARD_CONFIG)
        broken["tx_loc"] = [[0, 0, 0]]
        with pytest.raises(ValueError, match="must contain exactly 3 entries"):
            RadarConfig.from_dict(broken)

    def test_antenna_pattern_map_shape_mismatch_raises(self):
        broken = {
            **STANDARD_CONFIG,
            "antenna_pattern": {
                "kind": "map",
                "x_angles_deg": [-60, 0, 60],
                "y_angles_deg": [-30, 0, 30],
                "values": [
                    [0.1, 0.2, 0.1],
                    [0.5, 1.0],
                    [0.1, 0.2, 0.1],
                ],
            },
        }
        with pytest.raises(ValueError, match="must contain exactly 3 entries"):
            RadarConfig.from_dict(broken)

    def test_noise_model_requires_enabled_component(self):
        broken = {
            **STANDARD_CONFIG,
            "noise_model": {
                "seed": 3,
            },
        }
        with pytest.raises(ValueError, match="must enable at least one"):
            RadarConfig.from_dict(broken)

    def test_quantization_full_scale_must_be_positive(self):
        broken = {
            **STANDARD_CONFIG,
            "noise_model": {
                "quantization": {"bits": 10, "full_scale": 0.0},
            },
        }
        with pytest.raises(ValueError, match="must be positive"):
            RadarConfig.from_dict(broken)

    def test_receiver_chain_requires_enabled_stage(self):
        broken = {
            **STANDARD_CONFIG,
            "receiver_chain": {},
        }
        with pytest.raises(ValueError, match="must enable at least one"):
            RadarConfig.from_dict(broken)

    def test_receiver_chain_agc_bounds_must_be_ordered(self):
        broken = {
            **STANDARD_CONFIG,
            "receiver_chain": {
                "agc": {
                    "target_rms": 0.2,
                    "min_gain_db": 10.0,
                    "max_gain_db": 0.0,
                }
            },
        }
        with pytest.raises(ValueError, match="min_gain_db <= max_gain_db"):
            RadarConfig.from_dict(broken)

    def test_polarization_requires_matching_rx_count(self):
        broken = {
            **STANDARD_CONFIG,
            "polarization": {
                "tx": "horizontal",
                "rx": ["horizontal", "vertical"],
            },
        }
        with pytest.raises(ValueError, match="must contain exactly 4 entries"):
            RadarConfig.from_dict(broken)


class TestParameterFormulas:
    """Verify derived parameter formulas against expected values."""

    def test_range_resolution(self):
        cfg = STANDARD_CONFIG
        fs = cfg["sample_rate"] * 1e3
        slope_hz = cfg["slope"] * 1e12
        expected = C0 * fs / (2 * slope_hz * cfg["adc_samples"])
        mock = MockRadar(cfg)
        assert mock.range_resolution == pytest.approx(expected, rel=1e-10)
        assert 0.03 < mock.range_resolution < 0.06

    def test_doppler_resolution(self):
        cfg = STANDARD_CONFIG
        lam = C0 / cfg["fc"]
        chirp_period = (cfg["idle_time"] + cfg["ramp_end_time"]) * 1e-6
        effective_period = chirp_period * cfg["num_tx"]
        expected = lam / (2 * cfg["chirp_per_frame"] * effective_period)
        mock = MockRadar(cfg)
        assert mock.doppler_resolution == pytest.approx(expected, rel=1e-10)
        assert 0.05 < mock.doppler_resolution < 0.15

    def test_max_range_uses_precise_c0(self):
        cfg = STANDARD_CONFIG
        fs = cfg["sample_rate"] * 1e3
        slope_hz = cfg["slope"] * 1e12
        expected = C0 * fs / (2 * slope_hz)
        mock = MockRadar(cfg)
        assert mock.max_range == pytest.approx(expected, rel=1e-10)

    def test_max_range_equals_resolution_times_adc(self):
        mock = MockRadar(STANDARD_CONFIG)
        assert mock.max_range == pytest.approx(
            mock.range_resolution * STANDARD_CONFIG["adc_samples"],
            rel=1e-10,
        )

    def test_max_doppler(self):
        cfg = STANDARD_CONFIG
        lam = C0 / cfg["fc"]
        chirp_period = (cfg["idle_time"] + cfg["ramp_end_time"]) * 1e-6
        expected = lam / (4 * chirp_period * cfg["num_tx"])
        mock = MockRadar(cfg)
        assert mock.max_doppler == pytest.approx(expected, rel=1e-10)

    def test_wavelength(self):
        mock = MockRadar(STANDARD_CONFIG)
        assert mock._lambda == pytest.approx(C0 / 77e9, rel=1e-10)
        assert 3.8e-3 < mock._lambda < 4.0e-3

    def test_antenna_positions_scaled(self):
        cfg = STANDARD_CONFIG
        mock = MockRadar(cfg)
        spacing = mock._lambda / 2
        np.testing.assert_allclose(mock.tx_loc, np.array(cfg["tx_loc"], dtype=np.float32) * spacing)
        np.testing.assert_allclose(mock.rx_loc, np.array(cfg["rx_loc"], dtype=np.float32) * spacing)


class TestConfigVariations:
    @pytest.mark.parametrize("adc_samples", [128, 256, 512, 640])
    def test_range_resolution_scales_with_adc(self, adc_samples):
        cfg = {**STANDARD_CONFIG, "adc_samples": adc_samples, "num_range_bins": adc_samples}
        mock = MockRadar(cfg)
        fs = cfg["sample_rate"] * 1e3
        slope_hz = cfg["slope"] * 1e12
        expected = C0 * fs / (2 * slope_hz * adc_samples)
        assert mock.range_resolution == pytest.approx(expected, rel=1e-10)

    @pytest.mark.parametrize("chirps", [8, 32, 64, 128, 256])
    def test_doppler_resolution_scales_with_chirps(self, chirps):
        cfg = {**STANDARD_CONFIG, "chirp_per_frame": chirps, "num_doppler_bins": chirps}
        mock = MockRadar(cfg)
        lam = C0 / cfg["fc"]
        chirp_period = (cfg["idle_time"] + cfg["ramp_end_time"]) * 1e-6
        effective_period = chirp_period * cfg["num_tx"]
        expected = lam / (2 * chirps * effective_period)
        assert mock.doppler_resolution == pytest.approx(expected, rel=1e-10)

    @pytest.mark.parametrize("num_tx", [1, 2, 3, 4, 8])
    def test_max_doppler_scales_with_num_tx(self, num_tx):
        cfg = {**STANDARD_CONFIG, "num_tx": num_tx, "tx_loc": [[0, 0, 0]] * num_tx}
        mock = MockRadar(cfg)
        lam = C0 / cfg["fc"]
        chirp_period = (cfg["idle_time"] + cfg["ramp_end_time"]) * 1e-6
        expected = lam / (4 * chirp_period * num_tx)
        assert mock.max_doppler == pytest.approx(expected, rel=1e-10)


def test_pytorch_radar_can_target_cpu_device(standard_config):
    from witwin.radar import Radar

    radar = Radar(standard_config, backend="pytorch", device="cpu")
    assert radar.device == "cpu"
    assert radar.tx_pos.device.type == "cpu"
    assert radar.ranges.device.type == "cpu"


def test_radar_rejects_unknown_backend(standard_config):
    from witwin.radar import Radar

    with pytest.raises(ValueError, match="not a valid SolverBackend"):
        Radar(standard_config, backend="unknown")


def test_radar_builds_runtime_antenna_pattern(standard_config):
    from witwin.radar import Radar

    radar = Radar(
        RadarConfig.from_dict({
            **STANDARD_CONFIG,
            "antenna_pattern": {
                "x_angles_deg": [-60, 0, 60],
                "x_values": [0.25, 1.0, 0.25],
                "y_angles_deg": [-30, 0, 30],
                "y_values": [0.5, 1.0, 0.5],
            },
        }),
        backend="pytorch",
        device="cpu",
    )
    assert radar.antenna_pattern is not None
    assert radar.antenna_pattern.kind == "separable"


def test_radar_builds_runtime_noise_model(standard_config):
    from witwin.radar import Radar

    radar = Radar(
        RadarConfig.from_dict({
            **STANDARD_CONFIG,
            "noise_model": {
                "thermal": {"std": 0.01},
                "seed": 5,
            },
        }),
        backend="pytorch",
        device="cpu",
    )
    assert radar.noise_model_config is not None
    assert radar.noise_model is not None


def test_radar_builds_runtime_polarization(standard_config):
    from witwin.radar import Radar

    radar = Radar(
        RadarConfig.from_dict({
            **STANDARD_CONFIG,
            "polarization": {
                "tx": "horizontal",
                "rx": "vertical",
            },
        }),
        backend="pytorch",
        device="cpu",
    )
    assert radar.polarization_config is not None
    assert radar.polarization is not None
    assert radar.polarization.tx_world.shape == (radar.num_tx, 3)
    assert radar.polarization.rx_world.shape == (radar.num_rx, 3)


def test_radar_builds_runtime_receiver_chain(standard_config):
    from witwin.radar import Radar

    radar = Radar(
        RadarConfig.from_dict({
            **STANDARD_CONFIG,
            "receiver_chain": {
                "lna": {"gain_db": 20.0},
                "adc": {"bits": 10, "full_scale": 1.0},
            },
        }),
        backend="pytorch",
        device="cpu",
    )
    assert radar.receiver_chain_config is not None
    assert radar.receiver_chain is not None
    assert radar.gain == pytest.approx(radar.tx_voltage_rms)


def test_radar_rejects_double_quantization(standard_config):
    from witwin.radar import Radar

    with pytest.raises(ValueError, match="cannot both be enabled"):
        Radar(
            RadarConfig.from_dict({
                **STANDARD_CONFIG,
                "noise_model": {"quantization": {"bits": 8, "full_scale": 1.0}},
                "receiver_chain": {"adc": {"bits": 10, "full_scale": 1.0}},
            }),
            backend="pytorch",
            device="cpu",
        )


@pytest.mark.gpu
class TestRadarConstruction:
    def test_radar_creates_with_all_backends(self, standard_config):
        from witwin.radar import Radar

        for backend in ("pytorch", "slang", "dirichlet"):
            try:
                radar = Radar(standard_config, backend=backend)
            except (FileNotFoundError, OSError, RuntimeError) as exc:
                pytest.skip(f"backend unavailable: {exc}")
            assert radar.adc_samples == 256
            assert radar.num_tx == 3
            assert radar.num_rx == 4

    def test_radar_accepts_schema_object(self, standard_config):
        from witwin.radar import Radar

        radar = Radar(standard_config, backend="pytorch")
        assert radar.config is standard_config

    def test_radar_matches_formula(self, standard_config):
        from witwin.radar import Radar

        radar = Radar(standard_config, backend="pytorch")
        mock = MockRadar(standard_config)
        assert radar.range_resolution == pytest.approx(mock.range_resolution, rel=1e-10)
        assert radar.doppler_resolution == pytest.approx(mock.doppler_resolution, rel=1e-10)
        assert radar.max_range == pytest.approx(mock.max_range, rel=1e-10)

    def test_radar_axes_shapes(self, standard_config):
        from witwin.radar import Radar

        radar = Radar(standard_config, backend="pytorch")
        assert radar.ranges.shape[0] == radar.num_range_bins // 2
        assert radar.velocities.shape[0] == radar.num_doppler_bins

    def test_dirichlet_solver_owns_fft_state(self, standard_config):
        from witwin.radar import Radar

        try:
            radar = Radar(standard_config, backend="dirichlet")
        except (FileNotFoundError, OSError, RuntimeError) as exc:
            pytest.skip(f"backend unavailable: {exc}")

        assert not hasattr(radar, "N_fft")
        assert not hasattr(radar, "pad_factor")
        assert radar.solver.N_fft == standard_config.adc_samples * radar.solver.pad_factor
