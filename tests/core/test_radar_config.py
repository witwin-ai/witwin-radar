"""Tests for Radar configuration validation and derived parameters."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from conftest import MockRadar, STANDARD_CONFIG
from witwin.radar import RadarConfig


C0 = 299792458


class TestRadarConfigSchema:
    def test_config_round_trip_from_dict(self):
        config = RadarConfig.from_dict(STANDARD_CONFIG)
        assert config.num_tx == STANDARD_CONFIG["num_tx"]
        assert config.tx_loc[1] == tuple(STANDARD_CONFIG["tx_loc"][1])
        assert config.rx_loc[0] == tuple(STANDARD_CONFIG["rx_loc"][0])

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
        assert radar_config.antenna_pattern["kind"] == "separable"
        assert radar_config.antenna_pattern["x_values"][1] == pytest.approx(1.0)
        assert radar_config.antenna_pattern["y_values"] == [0.5, 1.0, 0.5]

    def test_polarization_round_trip_through_the_sensor_block(self):
        """The named directions still resolve; the FLAT key no longer exists.

        Phase 11 deleted ``RadarConfig.polarization`` with the runtime that read
        it - a second projection of a field Channel has already projected onto
        each endpoint's declared polarization. ``validate_polarization_config``
        survives because the sensor BLOCK still declares one, for the native
        kernel mode that implements the legacy projection, so the round trip is
        asserted where it still happens.
        """

        from witwin.radar.validation import validate_polarization_config

        polarization = validate_polarization_config(
            {
                "tx": "horizontal",
                "rx": ["vertical", "horizontal", "vertical", "horizontal"],
                "reflection_flip": False,
            },
            num_tx=STANDARD_CONFIG["num_tx"],
            num_rx=STANDARD_CONFIG["num_rx"],
        )

        assert polarization["tx"][0] == pytest.approx((1.0, 0.0, 0.0))
        assert polarization["rx"][0] == pytest.approx((0.0, 1.0, 0.0))
        assert polarization["rx"][1] == pytest.approx((1.0, 0.0, 0.0))
        assert polarization["reflection_flip"] is False

    def test_the_flat_record_no_longer_carries_the_three_deleted_blocks(self):
        """A dataclass field is the claim; asserting its absence is the check."""

        import dataclasses

        fields = {field.name for field in dataclasses.fields(RadarConfig)}
        assert "noise_model" not in fields
        assert "receiver_chain" not in fields
        assert "polarization" not in fields
        assert "frontend" in fields

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

    def test_polarization_requires_matching_rx_count(self):
        from witwin.radar.validation import validate_polarization_config

        with pytest.raises(ValueError, match="must contain exactly 4 entries"):
            validate_polarization_config(
                {"tx": "horizontal", "rx": ["horizontal", "vertical"]},
                num_tx=STANDARD_CONFIG["num_tx"],
                num_rx=STANDARD_CONFIG["num_rx"],
            )


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


def test_a_radar_can_be_constructed_on_cpu_for_configuration_workflows(standard_config):
    from witwin.radar import Radar

    radar = Radar(standard_config, device="cpu")
    assert radar.device == torch.device("cpu")
    assert radar.tx_pos.device.type == "cpu"
    assert radar.ranges.device.type == "cpu"


def test_radar_rejects_backend_keyword(standard_config):
    from witwin.radar import Radar

    with pytest.raises(TypeError, match="backend"):
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
        device="cpu",
    )
    assert radar.antenna_pattern_kind == "separable"


@pytest.mark.gpu
class TestRadarConstruction:
    def test_radar_creates_from_a_validated_config(self, standard_config):
        from witwin.radar import Radar

        try:
            radar = Radar(standard_config)
        except (FileNotFoundError, OSError, RuntimeError) as exc:
            pytest.skip(f"backend unavailable: {exc}")
        assert radar.config.adc_samples == 256
        assert radar.config.num_tx == 3
        assert radar.config.num_rx == 4

    def test_radar_accepts_schema_object(self, standard_config):
        from witwin.radar import Radar

        radar = Radar(standard_config)
        assert radar.config is standard_config

    def test_radar_matches_formula(self, standard_config):
        from witwin.radar import Radar

        radar = Radar(standard_config)
        mock = MockRadar(standard_config)
        assert radar.range_resolution == pytest.approx(mock.range_resolution, rel=1e-10)
        assert radar.doppler_resolution == pytest.approx(mock.doppler_resolution, rel=1e-10)
        assert radar.max_range == pytest.approx(mock.max_range, rel=1e-10)

    def test_radar_axes_shapes(self, standard_config):
        from witwin.radar import Radar

        radar = Radar(standard_config)
        assert radar.ranges.shape[0] == radar.config.num_range_bins // 2
        assert radar.velocities.shape[0] == radar.config.num_doppler_bins

    def test_no_solver_and_no_fft_state_hang_off_the_radar(self, standard_config):
        """This used to assert where the FFT state LIVED; now there is none.

        The claim was that ``N_fft`` and ``pad_factor`` belonged to the solver
        rather than to the radar. Phase 11 deleted the solver, so the radar
        carries neither the state nor the owner, and ``pad_factor`` is not a
        constructor argument any more - an accepted-but-ignored parameter is
        indistinguishable from one that works.
        """

        import inspect

        from witwin.radar import Radar

        try:
            radar = Radar(standard_config)
        except (FileNotFoundError, OSError, RuntimeError) as exc:
            pytest.skip(f"backend unavailable: {exc}")

        assert not hasattr(radar, "N_fft")
        assert not hasattr(radar, "pad_factor")
        assert not hasattr(radar, "solver")
        assert "pad_factor" not in inspect.signature(Radar.__init__).parameters
