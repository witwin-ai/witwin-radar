"""Radar configurations from outside this package: TI mmWave profiles, device presets, the schema.

The flat configuration format itself - both directions and its field table -
is owned by :mod:`witwin.radar.radar` (``Radar.from_dict`` / ``Radar.to_dict``);
:func:`config_schema` here publishes its JSON Schema.

:func:`read_ti_cfg` reads a TI mmWave SDK ``.cfg`` profile into a
:class:`TiConfig`, and :meth:`TiConfig.to_radar` builds the
:class:`~witwin.radar.Radar` it describes once the board's antenna layout is
given. :func:`preset` builds the radar of a named evaluation board
(:data:`PRESETS`): an antenna layout plus a representative profile.

Conversion is in one place per unit: the ``.cfg``'s GHz, MHz/us, kHz/us, ksps,
us and ms become SI in :meth:`TiConfig.to_radar` and :func:`read_ti_cfg` and
nowhere else.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .radar import Chirp, Fmcw, Radar, config_schema

# ---------------------------------------------------------------------------
# TI mmWave SDK profiles
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TiProfile:
    """One ``profileCfg`` line, in the SDK's units."""

    profile_id: int
    start_ghz: float
    idle_us: float
    adc_start_us: float
    ramp_end_us: float
    slope_mhz_per_us: float
    adc_samples: int
    sample_rate_ksps: float


@dataclass(frozen=True, slots=True)
class TiChirp:
    """One chirp slot from ``chirpCfg``: its profile, its variations and its transmit mask."""

    profile_id: int
    start_offset_mhz: float
    slope_offset_khz_per_us: float
    idle_offset_us: float
    adc_start_offset_us: float
    tx_mask: int


@dataclass(frozen=True, slots=True)
class TiConfig:
    """The radar-relevant content of a TI mmWave SDK ``.cfg`` profile.

    Read by :func:`read_ti_cfg`. :meth:`to_radar` builds the
    :class:`~witwin.radar.Radar` it describes once the board's antenna layout
    is given; the ``.cfg`` names channels, not positions.
    """

    rx_mask: int
    tx_mask: int
    profiles: tuple[TiProfile, ...]
    #: The frame's chirp slots, ``frameCfg``'s start to end index, in order.
    chirps: tuple[TiChirp, ...]
    loops: int
    frame_count: int
    #: Frame periodicity, s.
    frame_period: float
    #: ``adcCfg``'s output format: 0 real, 1 complex 1x, 2 complex 2x.
    adc_output_format: int = 1
    bpm: bool = False

    def frame_times(self, count: int | None = None) -> tuple[float, ...]:
        """The first ``count`` frame instants, s (``frame_count`` when not given; 0 means one)."""

        frames = self.frame_count if count is None else int(count)
        return tuple(index * self.frame_period for index in range(max(frames, 1)))

    def to_radar(
        self,
        *,
        tx: Sequence[Sequence[float]],
        rx: Sequence[Sequence[float]],
        antenna_unit: str = "half_wavelength",
        power: float = 12.0,
        **overrides: Any,
    ) -> Radar:
        """The radar this profile describes on a board whose elements are ``tx``/``rx``.

        ``tx[i]`` and ``rx[j]`` are the positions of the board's transmit channel
        ``i`` and receive channel ``j`` (bit ``i``/``j`` of the ``.cfg`` masks).
        The radar's receivers are the enabled receive channels in channel order.
        Under TDM each chirp slot enables one transmitter and the radar's
        transmitters are those, in SLOT order, so a slot order ``TX0, TX2, TX1``
        becomes ``tx = (tx[0], tx[2], tx[1])``: the radar's slot ``s`` is its
        transmitter ``s``. Under BPM every slot enables the same transmitters,
        whose count is a power of two, and they are taken in channel order.
        ``power`` is the transmit power, dBm, which the ``.cfg`` does not state
        (its ``txOutPower`` is a back-off code).
        """

        profiles = {profile.profile_id: profile for profile in self.profiles}
        missing = sorted({chirp.profile_id for chirp in self.chirps} - set(profiles))
        if missing:
            raise ValueError(f"chirpCfg names profiles {missing} that no profileCfg declares")
        receivers = _bits(self.rx_mask)
        if not receivers or max(receivers) >= len(rx):
            raise ValueError(f"channelCfg enables receive channels {receivers}, but the board names {len(rx)}")
        for chirp in self.chirps:
            if chirp.tx_mask & ~self.tx_mask:
                raise ValueError(
                    f"a chirp enables transmit mask {chirp.tx_mask:#x} outside channelCfg's {self.tx_mask:#x}"
                )
        if self.bpm:
            masks = {chirp.tx_mask for chirp in self.chirps}
            transmitters = _bits(next(iter(masks)))
            if len(masks) != 1 or len(transmitters) < 2 or len(transmitters) & (len(transmitters) - 1):
                raise ValueError(
                    "a BPM frame enables the same two or four transmitters on every chirp; this profile enables "
                    f"masks {sorted(masks)}"
                )
            if len(self.chirps) % len(transmitters):
                raise ValueError("a BPM frame's chirp count per loop must be a multiple of its transmitter count")
            chirps_per_frame, mimo = self.loops * len(self.chirps), "bpm"
        else:
            transmitters = []
            for index, chirp in enumerate(self.chirps):
                enabled = _bits(chirp.tx_mask)
                if len(enabled) != 1:
                    raise ValueError(
                        f"chirp slot {index} enables transmitters {enabled}; a TDM frame fires exactly one per "
                        "chirp (bpmCfg selects BPM; simultaneous transmit beamforming is not modelled)"
                    )
                transmitters.append(enabled[0])
            if len(set(transmitters)) != len(transmitters):
                raise ValueError(
                    f"the chirp slots fire transmitters {transmitters}; a TDM loop fires each transmitter once"
                )
            chirps_per_frame, mimo = self.loops, "tdm"
        if max(transmitters) >= len(tx):
            raise ValueError(f"the profile fires transmit channels {transmitters}, but the board names {len(tx)}")

        base = profiles[self.chirps[0].profile_id]
        carrier = base.start_ghz * 1e9
        for chirp in self.chirps:
            profile = profiles[chirp.profile_id]
            if profile.adc_samples != base.adc_samples or profile.sample_rate_ksps != base.sample_rate_ksps:
                raise ValueError("every chirp of one frame keeps the sample count and sample rate: one cube")
        sequence = tuple(
            Chirp(
                slope=(profiles[c.profile_id].slope_mhz_per_us * 1e12 + c.slope_offset_khz_per_us * 1e9),
                start_offset=(profiles[c.profile_id].start_ghz * 1e9 - carrier + c.start_offset_mhz * 1e6),
                idle=(profiles[c.profile_id].idle_us + c.idle_offset_us) * 1e-6,
                adc_start=(profiles[c.profile_id].adc_start_us + c.adc_start_offset_us) * 1e-6,
                ramp_end=profiles[c.profile_id].ramp_end_us * 1e-6,
            )
            for c in self.chirps
        )
        uniform = all(
            c.profile_id == base.profile_id
            and c.start_offset_mhz == 0.0
            and c.slope_offset_khz_per_us == 0.0
            and c.idle_offset_us == 0.0
            and c.adc_start_offset_us == 0.0
            for c in self.chirps
        )
        waveform = Fmcw(
            slope=base.slope_mhz_per_us * 1e12,
            sample_rate=base.sample_rate_ksps * 1e3,
            samples_per_chirp=base.adc_samples,
            chirps_per_frame=chirps_per_frame,
            adc_start=base.adc_start_us * 1e-6,
            idle=base.idle_us * 1e-6,
            ramp_end=base.ramp_end_us * 1e-6,
            sequence=() if uniform else sequence,
            sampling="real" if self.adc_output_format == 0 else "complex",
            mimo=mimo,
        )
        fields: dict[str, Any] = {
            "carrier": carrier,
            "waveform": waveform,
            "tx": tuple(tuple(tx[index]) for index in transmitters),
            "rx": tuple(tuple(rx[index]) for index in receivers),
            "power": float(power),
            "antenna_unit": antenna_unit,
        }
        fields.update(overrides)
        return Radar(**fields)


def _bits(mask: int) -> list[int]:
    return [index for index in range(32) if mask >> index & 1]


def _numbers(command: str, tokens: Sequence[str], count: int) -> list[float]:
    if len(tokens) < count:
        raise ValueError(f"{command} takes at least {count} arguments, got {len(tokens)}: {' '.join(tokens)}")
    try:
        return [float(token) for token in tokens[:count]]
    except ValueError as error:
        raise ValueError(f"{command} has a non-numeric argument: {' '.join(tokens)}") from error


def read_ti_cfg(source: str | os.PathLike[str]) -> TiConfig:
    """Read a TI mmWave SDK ``.cfg`` profile: a file path, or the profile's text itself.

    Read commands, with the SDK's argument order and units:

    * ``channelCfg <rxMask> <txMask> <cascading>``
    * ``adcCfg <numADCBits> <adcOutputFmt>`` (0 real, 1 complex 1x, 2 complex 2x)
    * ``profileCfg <id> <startFreq GHz> <idle us> <adcStart us> <rampEnd us> <txOutPower>
      <txPhaseShifter> <slope MHz/us> <txStart us> <numAdcSamples> <sampleRate ksps> ...``
    * ``chirpCfg <start> <end> <profileId> <startFreqVar MHz> <slopeVar kHz/us> <idleVar us>
      <adcStartVar us> <txMask>``
    * ``frameCfg <chirpStart> <chirpEnd> <loops> <frames> <period ms> <trigger> <delay ms>``
    * ``bpmCfg <subFrameIdx> <enabled> <chirp0Idx> <chirp1Idx>``

    Every other command (``sensorStop``, ``cfarCfg``, ``guiMonitor``, ...) is
    a processing or GUI setting and is ignored. ``%`` starts a comment. An
    advanced-frame profile (``dfeDataOutputMode 3``, ``advFrameCfg``) is refused.
    """

    text = str(source)
    if "\n" not in text and Path(text).is_file():
        text = Path(text).read_text(encoding="utf-8")
    rx_mask = tx_mask = None
    profiles: list[TiProfile] = []
    chirp_table: dict[int, TiChirp] = {}
    frame: list[float] | None = None
    adc_format, bpm = 1, False
    for raw in text.splitlines():
        line = raw.split("%", 1)[0].strip()
        if not line:
            continue
        command, *tokens = line.split()
        if command == "channelCfg":
            rx_mask, tx_mask = (int(value) for value in _numbers(command, tokens, 2))
        elif command == "adcCfg":
            adc_format = int(_numbers(command, tokens, 2)[1])
        elif command == "profileCfg":
            values = _numbers(command, tokens, 11)
            profiles.append(
                TiProfile(
                    profile_id=int(values[0]),
                    start_ghz=values[1],
                    idle_us=values[2],
                    adc_start_us=values[3],
                    ramp_end_us=values[4],
                    slope_mhz_per_us=values[7],
                    adc_samples=int(values[9]),
                    sample_rate_ksps=values[10],
                )
            )
        elif command == "chirpCfg":
            values = _numbers(command, tokens, 8)
            record = TiChirp(
                profile_id=int(values[2]),
                start_offset_mhz=values[3],
                slope_offset_khz_per_us=values[4],
                idle_offset_us=values[5],
                adc_start_offset_us=values[6],
                tx_mask=int(values[7]),
            )
            for index in range(int(values[0]), int(values[1]) + 1):
                chirp_table[index] = record
        elif command == "frameCfg":
            frame = _numbers(command, tokens, 5)
        elif command == "bpmCfg":
            bpm = bool(int(_numbers(command, tokens, 2)[1]))
        elif command == "advFrameCfg" or (command == "dfeDataOutputMode" and tokens[:1] == ["3"]):
            raise NotImplementedError("advanced (sub-frame) TI profiles are not read; give a legacy frameCfg profile")
    if rx_mask is None or tx_mask is None:
        raise ValueError("the profile has no channelCfg line")
    if not profiles:
        raise ValueError("the profile has no profileCfg line")
    if frame is None:
        raise ValueError("the profile has no frameCfg line")
    start, end = int(frame[0]), int(frame[1])
    absent = [index for index in range(start, end + 1) if index not in chirp_table]
    if absent:
        raise ValueError(f"frameCfg loops over chirps {start}..{end} but chirpCfg declares none for {absent}")
    return TiConfig(
        rx_mask=rx_mask,
        tx_mask=tx_mask,
        profiles=tuple(profiles),
        chirps=tuple(chirp_table[index] for index in range(start, end + 1)),
        loops=int(frame[2]),
        frame_count=int(frame[3]),
        frame_period=frame[4] * 1e-3,
        adc_output_format=adc_format,
        bpm=bpm,
    )


# ---------------------------------------------------------------------------
# Device presets
# ---------------------------------------------------------------------------

#: Element positions in half wavelengths, radar-local ``(right, up, -boresight)``,
#: indexed by the board's channel number. The xWR1843BOOST and IWR6843ISK
#: antennas place RX0..RX3 half a wavelength apart along x, TX0 and TX2 two
#: wavelengths apart along x, and TX1 midway between them one half wavelength
#: up, which is what makes the 8-element azimuth and 4-element elevated virtual
#: rows of the TI demos. Taken from TI's antenna descriptions of those boards;
#: a board revision with another layout is described by passing tx/rx.
_BOOST_TX = ((0.0, 0.0, 0.0), (2.0, 1.0, 0.0), (4.0, 0.0, 0.0))
_BOOST_RX = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0))

#: A representative TDM profile per band: three transmitters fired TX0, TX2,
#: TX1 (azimuth pair first), four receivers, 256 complex samples at 5 MS/s,
#: 60 MHz/us over a 60 us ramp, 64 loops, 100 ms frames. Not a TI demo file.
_PROFILE = """\
channelCfg 15 7 0
adcCfg 2 1
profileCfg 0 {start} 7 6 60 0 0 60 1 256 5000 0 0 30
chirpCfg 0 0 0 0 0 0 0 1
chirpCfg 1 1 0 0 0 0 0 4
chirpCfg 2 2 0 0 0 0 0 2
frameCfg 0 2 64 0 100 1 0
"""

#: Board name -> (transmit channel positions, receive channel positions, profile text, transmit power dBm).
PRESETS: Mapping[str, tuple[tuple[tuple[float, float, float], ...], tuple[tuple[float, float, float], ...], str, float]]
PRESETS = {
    "awr1843boost": (_BOOST_TX, _BOOST_RX, _PROFILE.format(start=77), 12.0),
    "iwr6843isk": (_BOOST_TX, _BOOST_RX, _PROFILE.format(start=60), 12.0),
}


def preset(name: str, **overrides: Any) -> Radar:
    """The radar of evaluation board ``name`` (a key of :data:`PRESETS`), with SI field overrides."""

    key = str(name).lower()
    if key not in PRESETS:
        raise ValueError(f"no preset {name!r}; presets are {sorted(PRESETS)}")
    tx, rx, profile, power = PRESETS[key]
    return read_ti_cfg(profile).to_radar(tx=tx, rx=rx, power=power, **overrides)


__all__ = ["PRESETS", "TiConfig", "config_schema", "preset", "read_ti_cfg"]
