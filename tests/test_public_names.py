"""The naming rule of R-ADR-049, checked over the whole root surface.

A public field or verb keyword that names a quantity Channel also publishes,
in Channel's unit, takes Channel's name (:data:`CHANNEL_NAMES`); every other
one carries no unit suffix. Processing products, labels and equation-owning
records are outside the rule and are not root exports.
"""

from __future__ import annotations

import dataclasses
import inspect
import re

import witwin.radar as wr

UNIT_SUFFIX = re.compile(r"_(s|hz|m|m2|db|dbm|dbi|deg|rad|w|watts|mps|k)$")

#: Channel's names for the quantities Radar shares with it, verbatim.
CHANNEL_NAMES = frozenset({"reference_frequency_hz", "power_w"})

#: Derived read-outs an equation owner computes (R-ADR-021 keeps the unit in
#: their identifier); they are not fields a caller sets.
DERIVED = frozenset(
    {
        ("Iq", "image_rejection_db"),
        ("Noise", "noise_power_watts"),
        ("Noise", "system_noise_temperature_k"),
        ("Noise", "phase_innovation_sigma_rad"),
        ("Noise", "phase_floor_sigma_rad"),
        ("Noise", "phase_flicker_scale_rad"),
    }
)


def _public_names(owner: type) -> set[str]:
    names = {field.name for field in dataclasses.fields(owner)} if dataclasses.is_dataclass(owner) else set()
    names |= {name for name, value in vars(owner).items() if isinstance(value, property)}
    return {name for name in names if not name.startswith("_")}


def test_no_root_record_field_carries_a_unit_suffix_unless_it_is_channels_name():
    offenders = []
    for export in wr.__all__:
        owner = getattr(wr, export)
        if not inspect.isclass(owner):
            continue
        for name in _public_names(owner):
            if UNIT_SUFFIX.search(name) and name not in CHANNEL_NAMES and (export, name) not in DERIVED:
                offenders.append(f"{export}.{name}")
    assert offenders == []


def test_the_verbs_take_channels_ad_mode_and_the_radar_channels_frequency():
    for verb in (wr.Radar.trace, wr.Radar.simulate, wr.Radar.stream, wr.Network.simulate):
        parameters = inspect.signature(verb).parameters
        assert "ad_mode" in parameters and "grad" not in parameters, verb
    assert "reference_frequency_hz" in {field.name for field in dataclasses.fields(wr.Radar)}
    assert "reference_frequency_hz" in {field.name for field in dataclasses.fields(wr.Paths)}
    assert "carrier" not in {field.name for field in dataclasses.fields(wr.Radar)}
