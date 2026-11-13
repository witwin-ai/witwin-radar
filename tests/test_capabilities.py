"""``witwin.radar.capabilities()``: the record is read from its owners, so it says what the code accepts.

Each test ties one section of the record to the owner it claims to report:
a vocabulary the record lists is accepted by the record that owns it and a
word outside it is refused, a declared AD leaf is a real field of its owner,
and the mirrored Channel vocabularies match the live Channel contract.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

import witwin.radar as wr
from witwin.radar import Fmcw, Motion, Ofdm, Pulsed

FMCW = {
    "slope": 60e12,
    "sample_rate": 10e6,
    "samples_per_chirp": 64,
    "chirps_per_frame": 8,
    "adc_start": 1e-6,
    "idle": 5e-6,
    "ramp_end": 20e-6,
}
OFDM = {
    "subcarrier_spacing": 1e5,
    "num_subcarriers": 64,
    "cyclic_prefix": 1e-6,
    "num_symbols": 8,
    "max_expected_delay": 1e-7,
}
PULSED = {
    "pulse_width": 1e-6,
    "bandwidth": 1e6,
    "pri": 1e-4,
    "num_pulses": 8,
    "sample_rate": 1e7,
    "num_samples": 64,
    "range_gate_start": 0.0,
}


def test_the_record_is_versioned_json_and_a_fresh_copy():
    record = wr.capabilities()
    assert record["schema_version"] == 1
    assert record["package_version"] == wr.__version__
    json.dumps(record)
    record["waveforms"]["kinds"].append("mutated")
    assert "mutated" not in wr.capabilities()["waveforms"]["kinds"]


def test_every_listed_waveform_option_is_accepted_and_an_unlisted_one_refused():
    waveforms = wr.capabilities()["waveforms"]
    assert waveforms["kinds"] == [Fmcw.kind, Ofdm.kind, Pulsed.kind]
    for name, values in (("output", waveforms["fmcw"]["outputs"]), ("sampling", waveforms["fmcw"]["sampling"])):
        for value in values:
            Fmcw(**FMCW, **{name: value})
        with pytest.raises(ValueError, match=f"Fmcw.{name} must be one of"):
            Fmcw(**FMCW, **{name: "unlisted"})
    for value in waveforms["fmcw"]["mimo"]:
        Fmcw(**FMCW, mimo=value)
    with pytest.raises(ValueError, match="Fmcw.mimo must be one of"):
        Fmcw(**FMCW, mimo="cdm")
    for value in waveforms["ofdm"]["modulations"]:
        Ofdm(**OFDM, modulation=value)
    for value in waveforms["ofdm"]["outputs"]:
        Ofdm(**OFDM, output=value)
    with pytest.raises(ValueError, match="Ofdm.modulation must be one of"):
        Ofdm(**OFDM, modulation="qam8")
    for value in set(waveforms["pulsed"]["pulse_kinds"]) - {"coded"}:
        Pulsed(pulse_kind=value, **PULSED)
    Pulsed(pulse_kind="coded", chips=(1.0, -1.0), **PULSED)
    with pytest.raises(ValueError, match="Pulsed.pulse_kind must be one of"):
        Pulsed(pulse_kind="barker", **PULSED)


def test_every_listed_motion_kind_and_world_motion_is_accepted():
    motion = wr.capabilities()["motion"]
    for kind in motion["kinds"]:
        Motion(kind=kind)
    for world in motion["world_motions"]:
        Motion(world=world)
    with pytest.raises(ValueError, match="Motion.world must be one of"):
        Motion(world="moving")
    assert motion["cadences"] == [field for field in motion["cadences"] if hasattr(Motion(), field)]


def test_every_declared_leaf_is_a_field_of_its_owner():
    from witwin.radar.sensors import Coupling, Pattern
    from witwin.radar.smpl import SMPLBody

    owners = {"Radar": wr.Radar, "PointTargets": wr.PointTargets, "Pattern": Pattern, "Coupling": Coupling}
    leaves = wr.capabilities()["ad_leaves"]
    for name, owner in owners.items():
        fields = {field.name for field in dataclasses.fields(owner)}
        assert set(leaves[name]) <= fields, name
        assert all(set(modes) <= {"jvp", "vjp"} and modes for modes in leaves[name].values())
    assert set(leaves["SMPLBody"]) <= set(SMPLBody.__init__.__code__.co_varnames)


def test_the_mirrored_channel_vocabularies_match_the_live_contract():
    pytest.importorskip("witwin.channel")
    record = wr.capabilities()
    channel = record["channel"]
    assert channel is not None
    assert sorted(record["motion"]["world_motions"]) == channel["world_motions"]
    assert set(record["propagation"]["components"]) <= set(channel["components"])


def test_the_native_section_is_the_prebuilt_build_record():
    native = wr.capabilities()["native"]
    from witwin.radar.cuda import runtime

    assert native["abi_version"] == runtime.RADAR_ABI_VERSION
    if not native["available"]:
        assert native["operators"] == [] and native["reason"]
        return
    info = runtime.read_build_info(runtime.prebuilt_extension_path())
    assert native["operators"] == sorted(info["operator_symbols"])
