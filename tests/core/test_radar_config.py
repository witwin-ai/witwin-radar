"""Tests for Radar configuration validation and derived parameters.

``RadarConfig`` is gone: the ``Radar`` record IS the configuration, and the flat
FMCW file format is read by ``Radar.from_dict``. Every schema case below moved
with it, so what used to be asserted about the intermediate record is now
asserted about the loader's refusals and about the fields of ``Radar`` itself.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch
from conftest import STANDARD_CONFIG, MockRadar

from witwin.radar import Radar

C0 = 299792458


class TestRadarConfigSchema:
    def test_config_round_trip_from_dict(self):
        radar = Radar.from_dict(STANDARD_CONFIG, device="cpu")
        assert radar.num_tx == STANDARD_CONFIG["num_tx"]
        assert radar.tx[1] == tuple(STANDARD_CONFIG["tx_loc"][1])
        assert radar.rx[0] == tuple(STANDARD_CONFIG["rx_loc"][0])

    def test_antenna_pattern_round_trip_from_dict(self):
        radar = Radar.from_dict(
            {
                **STANDARD_CONFIG,
                "antenna_pattern": {
                    "kind": "separable",
                    "x_angles_deg": [-60, 0, 60],
                    "x_values": [0.2, 1.0, 0.2],
                    "y_angles_deg": [-30, 0, 30],
                    "y_values": [0.5, 1.0, 0.5],
                },
            },
            device="cpu",
        )

        assert radar.pattern.kind == "separable"
        assert radar.pattern.x_gain[1] == pytest.approx(1.0)
        assert radar.pattern.y_gain == (0.5, 1.0, 0.5)

    def test_the_record_carries_the_receive_chain_as_flat_fields(self):
        """A dataclass field is the claim; asserting its shape is the check.

        The old record was checked for the ABSENCE of three grouping blocks.
        Two of them, ``noise_model`` and ``receiver_chain``, are still absent
        and the stages they wrapped are one field each. The third,
        ``polarization``, is now a real field of ``Radar`` - that is the decided
        behaviour change, not a regression, and it is pinned here so the change
        cannot happen again silently.
        """

        fields = {field.name for field in dataclasses.fields(Radar)}
        assert "noise_model" not in fields
        assert "receiver_chain" not in fields
        assert {"noise", "lna_gain", "agc", "adc", "impedance", "seed"} <= fields
        assert "polarization" in fields
        assert "frontend" in fields

    def test_an_unknown_key_is_refused_rather_than_dropped(self):
        """The flat mapping used to swallow anything it did not recognize."""

        broken = {**STANDARD_CONFIG, "receiver_chain": {"lna_gain_db": 30.0}}
        with pytest.raises(ValueError, match="unsupported keys: receiver_chain"):
            Radar.from_dict(broken, device="cpu")

    def test_a_waveform_selector_cannot_be_silently_ignored(self):
        """`{"waveform": "ofdm"}` used to build an FMCW radar with no error.

        That is the worst shape the swallow had: the one key a caller reaches
        for to choose a waveform, dropped, with a full simulation returned in
        the wrong waveform. The flat form is the FMCW file format; a non-FMCW
        radar is a ``waveform=`` keyword override, and this pins that the
        refusal says so instead of the result implying otherwise.
        """

        with pytest.raises(ValueError, match="unsupported keys: waveform"):
            Radar.from_dict({**STANDARD_CONFIG, "waveform": "ofdm"}, device="cpu")

    def test_a_frontend_block_is_refused_instead_of_dropped(self):
        """The receive chain is attached as keyword overrides, not authored here."""

        with pytest.raises(ValueError, match="unsupported keys: frontend"):
            Radar.from_dict({**STANDARD_CONFIG, "frontend": {"seed": 7}}, device="cpu")

    @pytest.mark.parametrize("key", ["frame_per_second", "num_doppler_bins", "num_range_bins", "num_angle_bins"])
    def test_a_key_nothing_consumes_is_refused_by_name(self, key: str):
        """These four were required by the old form and read by nobody.

        They described a processing grid: the bin counts are derived from the
        waveform spec and the frame rate is the caller's own scheduling number.
        Accepting them made a caller believe a block was configured, so the
        loader names them in its refusal rather than dropping them - a separate
        message from the unsupported-key one, because "you configured nothing"
        and "we do not know this key" are different mistakes.
        """

        with pytest.raises(ValueError, match=f"keys nothing consumes: {key}"):
            Radar.from_dict({**STANDARD_CONFIG, key: 10}, device="cpu")

    def test_missing_required_key_raises(self):
        broken = dict(STANDARD_CONFIG)
        broken.pop("num_tx")
        with pytest.raises(ValueError, match="missing required keys"):
            Radar.from_dict(broken, device="cpu")

    def test_antenna_count_mismatch_raises(self):
        broken = dict(STANDARD_CONFIG)
        broken["tx_loc"] = [[0, 0, 0]]
        with pytest.raises(ValueError, match="tx_loc holds 1 entries but num_tx is 3"):
            Radar.from_dict(broken, device="cpu")

    def test_antenna_pattern_map_shape_mismatch_raises(self):
        broken = {
            **STANDARD_CONFIG,
            "antenna_pattern": {
                "kind": "map",
                "x_angles_deg": [-60, 0, 60],
                "y_angles_deg": [-30, 0, 30],
                "values": [[0.1, 0.2, 0.1], [0.5, 1.0], [0.1, 0.2, 0.1]],
            },
        }
        with pytest.raises(ValueError, match="each gain row needs one entry per x sample"):
            Radar.from_dict(broken, device="cpu")


class TestParameterFormulas:
    """Verify derived parameter formulas against expected values."""

    def test_range_resolution(self):
        cfg = STANDARD_CONFIG
        fs = cfg["sample_rate"] * 1e3
        slope_hz = cfg["slope"] * 1e12
        expected = C0 * fs / (2 * slope_hz * cfg["adc_samples"])
        mock = MockRadar(cfg)
        assert mock.axes.range_bin_m == pytest.approx(expected, rel=1e-10)
        assert 0.03 < mock.axes.range_bin_m < 0.06

    def test_doppler_resolution(self):
        cfg = STANDARD_CONFIG
        lam = C0 / cfg["fc"]
        chirp_period = (cfg["idle_time"] + cfg["ramp_end_time"]) * 1e-6
        effective_period = chirp_period * cfg["num_tx"]
        expected = lam / (2 * cfg["chirp_per_frame"] * effective_period)
        mock = MockRadar(cfg)
        assert mock.axes.velocity_bin_mps == pytest.approx(expected, rel=1e-10)
        assert 0.05 < mock.axes.velocity_bin_mps < 0.15

    def test_max_range_uses_precise_c0(self):
        cfg = STANDARD_CONFIG
        fs = cfg["sample_rate"] * 1e3
        slope_hz = cfg["slope"] * 1e12
        expected = C0 * fs / (2 * slope_hz)
        mock = MockRadar(cfg)
        assert mock.axes.max_unambiguous_range_m == pytest.approx(expected, rel=1e-10)

    def test_max_range_equals_resolution_times_adc(self):
        mock = MockRadar(STANDARD_CONFIG)
        assert mock.axes.max_unambiguous_range_m == pytest.approx(
            mock.axes.range_bin_m * STANDARD_CONFIG["adc_samples"], rel=1e-10
        )

    def test_max_doppler(self):
        cfg = STANDARD_CONFIG
        lam = C0 / cfg["fc"]
        chirp_period = (cfg["idle_time"] + cfg["ramp_end_time"]) * 1e-6
        expected = lam / (4 * chirp_period * cfg["num_tx"])
        mock = MockRadar(cfg)
        assert mock.axes.max_unambiguous_speed_mps == pytest.approx(expected, rel=1e-10)

    def test_wavelength(self):
        mock = MockRadar(STANDARD_CONFIG)
        assert mock.wavelength_m == pytest.approx(C0 / 77e9, rel=1e-10)
        assert 3.8e-3 < mock.wavelength_m < 4.0e-3

    def test_antenna_positions_scaled(self):
        cfg = STANDARD_CONFIG
        mock = MockRadar(cfg)
        spacing = mock.wavelength_m / 2
        np.testing.assert_allclose(mock.tx_loc, np.array(cfg["tx_loc"], dtype=np.float32) * spacing)
        np.testing.assert_allclose(mock.rx_loc, np.array(cfg["rx_loc"], dtype=np.float32) * spacing)


class TestConfigVariations:
    @pytest.mark.parametrize("adc_samples", [128, 256, 512, 640])
    def test_range_resolution_scales_with_adc(self, adc_samples):
        cfg = {**STANDARD_CONFIG, "adc_samples": adc_samples}
        mock = MockRadar(cfg)
        fs = cfg["sample_rate"] * 1e3
        slope_hz = cfg["slope"] * 1e12
        expected = C0 * fs / (2 * slope_hz * adc_samples)
        assert mock.axes.range_bin_m == pytest.approx(expected, rel=1e-10)

    @pytest.mark.parametrize("chirps", [8, 32, 64, 128, 256])
    def test_doppler_resolution_scales_with_chirps(self, chirps):
        cfg = {**STANDARD_CONFIG, "chirp_per_frame": chirps}
        mock = MockRadar(cfg)
        lam = C0 / cfg["fc"]
        chirp_period = (cfg["idle_time"] + cfg["ramp_end_time"]) * 1e-6
        effective_period = chirp_period * cfg["num_tx"]
        expected = lam / (2 * chirps * effective_period)
        assert mock.axes.velocity_bin_mps == pytest.approx(expected, rel=1e-10)

    @pytest.mark.parametrize("num_tx", [1, 2, 3, 4, 8])
    def test_max_doppler_scales_with_num_tx(self, num_tx):
        cfg = {**STANDARD_CONFIG, "num_tx": num_tx, "tx_loc": [[0, 0, 0]] * num_tx}
        mock = MockRadar(cfg)
        lam = C0 / cfg["fc"]
        chirp_period = (cfg["idle_time"] + cfg["ramp_end_time"]) * 1e-6
        expected = lam / (4 * chirp_period * num_tx)
        assert mock.axes.max_unambiguous_speed_mps == pytest.approx(expected, rel=1e-10)


def test_a_radar_can_be_constructed_on_cpu_for_configuration_workflows(standard_config):
    radar = Radar.from_dict(standard_config, device="cpu")
    assert radar.device == torch.device("cpu")
    assert radar.tx_pos.device.type == "cpu"
    assert not hasattr(radar, "axes")


def test_radar_rejects_backend_keyword(standard_config):
    with pytest.raises(TypeError, match="backend"):
        Radar.from_dict(standard_config, backend="unknown", device="cpu")


def test_radar_builds_runtime_antenna_pattern(standard_config):
    radar = Radar.from_dict(
        {
            **standard_config,
            "antenna_pattern": {
                "x_angles_deg": [-60, 0, 60],
                "x_values": [0.25, 1.0, 0.25],
                "y_angles_deg": [-30, 0, 30],
                "y_values": [0.5, 1.0, 0.5],
            },
        },
        device="cpu",
    )
    assert radar.pattern.kind == "separable"


@pytest.mark.gpu
class TestRadarConstruction:
    def test_radar_creates_from_a_validated_config(self, standard_config):
        try:
            radar = Radar.from_dict(standard_config)
        except (FileNotFoundError, OSError, RuntimeError) as exc:
            pytest.skip(f"backend unavailable: {exc}")
        assert radar.waveform.samples_per_chirp == 256
        assert radar.num_tx == 3
        assert radar.num_rx == 4

    def test_the_radar_record_is_the_configuration(self, standard_config):
        """There is no separate config object to hand around any more.

        The old case checked that ``Radar`` stored the very ``RadarConfig`` it
        was given rather than re-parsing it. The record and the radar are now
        one frozen dataclass, so the equivalent claim is that reading the same
        mapping twice yields equal radars and that neither can be edited in
        place afterwards.
        """

        radar = Radar.from_dict(standard_config)
        assert radar == Radar.from_dict(standard_config)
        with pytest.raises(dataclasses.FrozenInstanceError):
            radar.carrier = 24e9

    def test_radar_matches_formula(self, standard_config):
        radar = Radar.from_dict(standard_config)
        mock = MockRadar(standard_config)
        spec = radar.waveform_spec()
        assert spec.max_unambiguous_speed_mps == pytest.approx(mock.axes.max_unambiguous_speed_mps, rel=1e-10)

    def test_radar_has_no_processing_axis_state(self, standard_config):
        radar = Radar.from_dict(standard_config)
        assert not hasattr(radar, "axes")
        assert not hasattr(radar, "ranges")
        assert not hasattr(radar, "velocities")

    def test_no_solver_and_no_fft_state_hang_off_the_radar(self, standard_config):
        """This used to assert where the FFT state LIVED; now there is none.

        The claim was that ``N_fft`` and ``pad_factor`` belonged to the solver
        rather than to the radar. Phase 11 deleted the solver, so the radar
        carries neither the state nor the owner, and ``pad_factor`` is not a
        constructor argument any more - an accepted-but-ignored parameter is
        indistinguishable from one that works.
        """

        import inspect

        try:
            radar = Radar.from_dict(standard_config)
        except (FileNotFoundError, OSError, RuntimeError) as exc:
            pytest.skip(f"backend unavailable: {exc}")

        assert not hasattr(radar, "N_fft")
        assert not hasattr(radar, "pad_factor")
        assert not hasattr(radar, "solver")
        assert "pad_factor" not in inspect.signature(Radar.__init__).parameters
