"""Configuration I/O: ``Radar.to_dict`` / ``from_dict`` round trips, the schema, TI profiles and presets (U-13, M5-9)."""

from __future__ import annotations

import json
import pathlib

import pytest
from conftest import STANDARD_CONFIG

from witwin.radar import Adc, Agc, Chirp, Compression, Coupling, Fmcw, Iq, Leakage, Noise, Ofdm, Pattern, Pulsed, Radar
from witwin.radar.config import PRESETS, config_schema, preset, read_ti_cfg

REPO = pathlib.Path(__file__).resolve().parents[1]


def _round_trip(radar: Radar) -> Radar:
    """Through JSON text, so nothing survives that a file could not carry."""

    return Radar.from_dict(json.loads(json.dumps(radar.to_dict())), device=radar.device)


def _every_field_radar() -> Radar:
    return Radar(
        carrier=60e9,
        waveform=Fmcw(
            slope=30e12,
            sample_rate=10e6,
            samples_per_chirp=64,
            chirps_per_frame=8,
            adc_start=1e-6,
            idle=5e-6,
            ramp_end=10e-6,
            sequence=(Chirp(slope=20e12), Chirp(start_offset=1e8, idle=6e-6)),
            phase_error=(1e9,),
            sampling="real",
        ),
        tx=[(0.0, 0.0, 0.0), (0.004, 0.0, 0.0)],
        rx=[(0.0, 0.0, 0.0), (0.0025, 0.0, 0.0)],
        antenna_unit="m",
        power=10.0,
        pattern=Pattern.field_separable([-90, 0, 90], [0.1, 1.0, 0.2j], [-90, 90], [1.0, 1.0], gain=3.0),
        rx_pattern=Pattern.table([-60, 60], [-30, 30], [[0.5, 1.0], [0.25, 0.5]], gain=2.0),
        noise=Noise(figure=8.0, phase_density=-90.0, phase_offset=1e5),
        lna_gain=20.0,
        agc=Agc(target_rms=0.1, mode="global"),
        adc=Adc(bits=12, full_scale=1.0),
        iq=Iq(gain_db=0.5, phase_deg=2.0),
        leakage=Leakage(isolation_db=30.0, delay_s=[[1e-10, 2e-10], [3e-10, 4e-10]], phase=0.25),
        compression=Compression(iip3=0.0, p1db=-12.0, am_pm=3.0),
        dc_offset=(1e-3 + 2e-3j, -5e-4),
        coupling=Coupling(rx=((1.0, 0.1j), (0.1j, 1.0))),
        impedance=75.0,
        seed=9,
        position=(1.0, 2.0, 3.0),
        look_at=(1.0, 2.0, 0.0),
        up=(0.0, 1.0, 0.0),
        polarization="right",
        rx_polarization="up",
        device="cpu",
    )


class TestRoundTrip:
    def test_the_flat_ti_file_round_trips_to_the_same_radar(self):
        radar = Radar.from_dict(STANDARD_CONFIG, device="cpu")
        assert _round_trip(radar) == radar

    def test_every_radar_field_round_trips_exactly(self):
        radar = _every_field_radar()
        again = _round_trip(radar)
        assert again == radar
        assert again.to_dict() == radar.to_dict()

    @pytest.mark.parametrize(
        "waveform",
        [
            Ofdm(
                subcarrier_spacing=120e3,
                num_subcarriers=64,
                cyclic_prefix=1e-6,
                num_symbols=16,
                max_expected_delay=5e-7,
                modulation="qpsk",
                output="time",
            ),
            Pulsed.pmcw(Pulsed.m_sequence(5), chip_rate=1e8, repetitions=8, sample_rate=1e8),
            Pulsed.fsk((1e6, -1e6, 2e6), chip_width=1e-7, pri=2e-5, num_pulses=4, sample_rate=4e7, num_samples=128),
            Fmcw(
                slope=60e12,
                sample_rate=5e6,
                samples_per_chirp=128,
                chirps_per_frame=16,
                adc_start=6e-6,
                idle=7e-6,
                ramp_end=60e-6,
                mimo="ddm",
                ddm_bands=4,
            ),
        ],
        ids=["ofdm", "pmcw", "fsk", "ddm"],
    )
    def test_every_waveform_kind_round_trips(self, waveform):
        radar = Radar(
            carrier=77e9, waveform=waveform, tx=[(0, 0, 0), (2, 0, 0)], rx=[(0, 0, 0)], power=10.0, device="cpu"
        )
        assert _round_trip(radar) == radar

    def test_to_json_and_from_json_round_trip(self, tmp_path):
        radar = _every_field_radar()
        radar.to_json(tmp_path / "radar.json")
        assert Radar.from_json(tmp_path / "radar.json", device="cpu") == radar

    def test_the_output_is_json_text_and_omits_defaults(self):
        written = Radar.from_dict(STANDARD_CONFIG, device="cpu").to_dict()
        json.dumps(written, allow_nan=False)
        assert written["format"] == "witwin.radar.config/1"
        assert written["waveform"]["kind"] == "fmcw"
        assert "noise" not in written and "position" not in written and "device" not in written

    def test_a_pose_given_as_a_tensor_is_written_as_its_values(self):
        import torch

        radar = Radar.from_dict(STANDARD_CONFIG, device="cpu", position=torch.tensor([1.0, 2.0, 3.0]))
        assert radar.to_dict()["position"] == [1.0, 2.0, 3.0]


class TestRefusals:
    def test_a_device_is_a_keyword_not_configuration(self):
        with pytest.raises(ValueError, match="unsupported keys: device"):
            Radar.from_dict({**STANDARD_CONFIG, "device": "cpu"})

    def test_an_unknown_record_field_is_refused_by_name(self):
        written = Radar.from_dict(STANDARD_CONFIG, device="cpu", noise=Noise(figure=3.0)).to_dict()
        written["noise"]["figure_db"] = 3.0
        with pytest.raises(ValueError, match="noise has unsupported keys: figure_db"):
            Radar.from_dict(written, device="cpu")

    def test_a_foreign_format_tag_is_refused(self):
        with pytest.raises(ValueError, match="format"):
            Radar.from_dict({**STANDARD_CONFIG, "format": "witwin.radar.config/9"}, device="cpu")

    def test_a_sequence_on_a_non_fmcw_waveform_is_refused(self):
        written = Radar(
            carrier=77e9,
            waveform=Ofdm(
                subcarrier_spacing=120e3, num_subcarriers=8, cyclic_prefix=1e-6, num_symbols=4, max_expected_delay=0
            ),
            tx=[(0, 0, 0)],
            rx=[(0, 0, 0)],
            power=0.0,
            device="cpu",
        ).to_dict()
        written["waveform"]["sequence"] = [{"slope": 1e12}]
        with pytest.raises(ValueError, match="only an fmcw waveform"):
            Radar.from_dict(written, device="cpu")


class TestSchema:
    def test_every_written_key_is_a_schema_property(self):
        schema = config_schema()
        written = _every_field_radar().to_dict()
        assert set(written) <= set(schema["properties"])
        for key in ("noise", "agc", "adc", "iq", "leakage", "compression", "coupling"):
            assert set(written[key]) <= set(schema["properties"][key]["properties"]), key
        kinds = {
            variant["properties"]["kind"]["const"]: variant for variant in schema["properties"]["waveform"]["oneOf"]
        }
        assert set(written["waveform"]) <= set(kinds["fmcw"]["properties"])

    def test_the_schema_is_json_and_names_the_ti_keys(self):
        schema = json.loads(json.dumps(config_schema()))
        assert schema["$schema"].endswith("2020-12/schema")
        assert {"slope", "sample_rate", "adc_samples"} <= set(schema["properties"])

    def test_the_documented_field_table_lists_every_key(self):
        text = (REPO / "docs" / "configuration.md").read_text(encoding="utf-8")
        missing = [key for key in config_schema()["properties"] if f"`{key}`" not in text]
        assert not missing, missing


# A TI SDK profile: TX0, TX2, TX1 slot order, real ADC, a slope variation on slot 1.
TI_PROFILE = """\
% a comment line
sensorStop
flushCfg
dfeDataOutputMode 1
channelCfg 15 7 0
adcCfg 2 0
adcbufCfg -1 0 1 1 1
profileCfg 0 77 7 6 60 0 0 60 1 256 5000 0 0 30
chirpCfg 0 0 0 0 0 0 0 1
chirpCfg 1 1 0 0 10 0 0 4
chirpCfg 2 2 0 0 0 0 0 2
frameCfg 0 2 32 8 50 1 0
cfarCfg -1 0 2 8 4 3 0 15 1
sensorStart
"""

BOARD_TX = ((0.0, 0.0, 0.0), (2.0, 1.0, 0.0), (4.0, 0.0, 0.0))
BOARD_RX = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0))


class TestTiProfile:
    def test_a_tdm_profile_maps_slots_to_transmitters_in_slot_order(self):
        config = read_ti_cfg(TI_PROFILE)
        radar = config.to_radar(tx=BOARD_TX, rx=BOARD_RX, device="cpu")
        assert radar.carrier == 77e9
        assert radar.tx == (BOARD_TX[0], BOARD_TX[2], BOARD_TX[1])
        assert radar.rx == BOARD_RX
        waveform = radar.waveform
        assert waveform.slope == pytest.approx(60e12, rel=0, abs=0)
        assert waveform.sample_rate == 5e6
        assert (waveform.samples_per_chirp, waveform.chirps_per_frame) == (256, 32)
        assert (waveform.adc_start, waveform.idle, waveform.ramp_end) == pytest.approx((6e-6, 7e-6, 60e-6), rel=1e-12)
        assert waveform.sampling == "real" and waveform.mimo == "tdm"
        # Slot 1 carries a 10 kHz/us slope variation; the others the profile's slope.
        assert [chirp.slope for chirp in waveform.sequence] == pytest.approx([60e12, 60.01e12, 60e12], rel=1e-12)
        assert config.frame_times(3) == pytest.approx((0.0, 0.05, 0.1))

    def test_a_uniform_profile_has_no_sequence(self):
        uniform = TI_PROFILE.replace("chirpCfg 1 1 0 0 10 0 0 4", "chirpCfg 1 1 0 0 0 0 0 4")
        radar = read_ti_cfg(uniform).to_radar(tx=BOARD_TX, rx=BOARD_RX, device="cpu")
        assert radar.waveform.sequence == ()

    def test_a_bpm_profile_fires_every_transmitter_every_chirp(self):
        bpm = """\
channelCfg 15 5 0
adcCfg 2 1
profileCfg 0 60 7 6 60 0 0 60 1 128 5000 0 0 30
chirpCfg 0 1 0 0 0 0 0 5
frameCfg 0 1 16 0 100 1 0
bpmCfg -1 1 0 1
"""
        radar = read_ti_cfg(bpm).to_radar(tx=BOARD_TX, rx=BOARD_RX, device="cpu")
        assert radar.waveform.mimo == "bpm" and radar.waveform.chirps_per_frame == 32
        assert radar.tx == (BOARD_TX[0], BOARD_TX[2])
        assert radar.waveform.sampling == "complex"

    def test_a_reads_from_a_file_path(self, tmp_path):
        path = tmp_path / "profile.cfg"
        path.write_text(TI_PROFILE, encoding="utf-8")
        assert read_ti_cfg(path) == read_ti_cfg(TI_PROFILE)

    def test_two_transmitters_in_a_tdm_slot_are_refused(self):
        both = TI_PROFILE.replace("chirpCfg 0 0 0 0 0 0 0 1", "chirpCfg 0 0 0 0 0 0 0 3")
        with pytest.raises(ValueError, match="fires exactly one per chirp"):
            read_ti_cfg(both).to_radar(tx=BOARD_TX, rx=BOARD_RX, device="cpu")

    def test_a_frame_over_an_undeclared_chirp_is_refused(self):
        with pytest.raises(ValueError, match="declares none for \\[3\\]"):
            read_ti_cfg(TI_PROFILE.replace("frameCfg 0 2", "frameCfg 0 3"))

    def test_an_advanced_frame_profile_is_refused(self):
        with pytest.raises(NotImplementedError, match="advanced"):
            read_ti_cfg(TI_PROFILE.replace("dfeDataOutputMode 1", "dfeDataOutputMode 3"))


class TestPresets:
    @pytest.mark.parametrize("name", sorted(PRESETS))
    def test_a_preset_builds_the_board_and_round_trips(self, name):
        radar = preset(name, device="cpu")
        assert (radar.num_tx, radar.num_rx) == (3, 4)
        assert _round_trip(radar) == radar

    def test_the_boost_virtual_array_is_eight_wide_with_four_elevated(self):
        radar = preset("awr1843boost", device="cpu")
        virtual = {(t[0] + r[0], t[1] + r[1]) for t in radar.tx for r in radar.rx}
        assert {x for x, y in virtual if y == 0.0} == set(range(8))
        assert {x for x, y in virtual if y == 1.0} == {2.0, 3.0, 4.0, 5.0}

    def test_an_unknown_preset_names_the_known_ones(self):
        with pytest.raises(ValueError, match="awr1843boost"):
            preset("iwr9999")
