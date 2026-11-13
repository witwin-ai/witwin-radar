"""Several radars in one scene and one call (R-ADR-051).

A :class:`Network` is a set of radars, and optionally declared emitters, that
share one world. Its session compiles the scene once, binds every radar's
elements and every emitter as ONE endpoint set (each radar's elements carried
by its own pose), discovers each leg family once per topology epoch for all of
them, replays each family once per observation instant, and composes every
transmitter-receiver pair in one join. The world half is therefore paid once,
whatever the radar count: the Channel calls per epoch are the single-radar
ones, with longer endpoint lists.

What a radar hears is then split per receiving radar:

* its OWN transmitters' rows (the monostatic rows) are gathered into the
  composed batch a standalone session of that radar would have built, in the
  same order, and run through that radar's own instrument half unchanged;
* rows from ANOTHER radar's transmitters reach it only for a declared link
  ``(transmitting radar, receiving radar)``, and rows from every emitter reach
  every radar: site rows (bistatic target returns), and site-less rows of the
  direct leg (the line of sight with ``direct=True``, the environment's
  returns with ``environment=True``). They are synthesized by the native
  cross-radar families, each receiver with its own reference: an FMCW
  receiver dechirps them against its own ramp through its IF band
  (:func:`~witwin.radar.synthesis.fmcw.synthesize_fmcw_cross`); a pulsed or
  OFDM receiver samples them on its own grid and its own LO
  (:func:`~witwin.radar.synthesis.pulsed.synthesize_pulsed_cross`,
  :func:`~witwin.radar.synthesis.ofdm.synthesize_ofdm_cross`), and demodulates
  them downstream against its own replica or symbols. The result is added to
  the receiver's frame before its receive chain, so its noise, gains and ADC
  act on the sum.

The phase conventions, the gates and what is refused are R-ADR-051's.
Adaptive motion is refused: the radars' observation instants are merged into
one schedule, which the adaptive controller's per-radar probe grid does not
share.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Any

import numpy as np
import torch

from .binding import (
    DEFAULT_RECEIVER_ID_BASE,
    DEFAULT_TRANSMITTER_ID_BASE,
    RadarWorldBinding,
    ScatterSitePolicy,
    bind_radar_world,
)
from .events import PathEvents
from .motion import Motion
from .paths import NO_SITE, RadarPathBatch, RadarPathTopology, SlotRows, TwoWayComposer
from .propagation import DiscoveryReport, RadarEndpointSpec
from .radar import WAVEFORM_FMCW, WAVEFORM_OFDM, WAVEFORM_PULSED, Fmcw, Radar, vec3_tensor
from .results import Result, SceneFrame, assemble
from .sensors import RoundTripPatternStage
from .simulation import (
    UNFREEZABLE_COMPONENTS,
    FrameTrace,
    Observation,
    Session,
    SlotBatch,
    WorldHalf,
    _Bounded,
    _constant_report,
    _dynamic_scene,
    _PendingObservation,
    _round_trip,
    _uploaded,
    carry_binding,
    echo_frame,
    moves_within_frame,
    pose_at,
    prepare_session,
)

#: How many frozen joins (topology epochs) each receiving radar keeps routing
#: tables for. A still or slowly changing world reuses one; the bound only
#: limits a world whose topology keeps changing.
ROUTING_CACHE = 8


def _finite(name: str, value: Any) -> float:
    if isinstance(value, bool) or isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a host float, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return number


def _fmcw_profile(radar: Radar, role: str) -> tuple[float, float, float, float]:
    """``(slope [Hz/s], start offset [Hz], ramp end [s], chirp period [s])`` of an FMCW radar on a link.

    A link reads one uniform TDM ramp or a one-profile sequence (whose
    ``Chirp.start_offset`` is a start-frequency offset); anything else is
    refused by name here, at construction.
    """

    waveform = radar.waveform
    assert isinstance(waveform, Fmcw)
    if waveform.mimo != "tdm":
        raise NotImplementedError(
            f"a network link's {role} uses mimo={waveform.mimo!r}; the cross-radar beat gates each transmitter on "
            "its own TDM slot, and DDM/BPM codes are not synthesized across radars"
        )
    if len(waveform.sequence) > 1:
        raise NotImplementedError(
            f"a network link's {role} declares a {len(waveform.sequence)}-profile chirp sequence; a link reads one "
            "ramp per radar (a one-profile sequence carries a start-frequency offset)"
        )
    if waveform.phase_error:
        raise NotImplementedError(f"a network link's {role} declares a ramp phase error, which links do not model")
    if waveform.sequence:
        slope, offset, idle, _, ramp_end = waveform._profile(waveform.sequence[0])
        return slope, offset, ramp_end, idle + ramp_end
    return float(waveform.slope), 0.0, float(waveform.ramp_end), waveform.chirp_period


@dataclass(frozen=True, slots=True)
class Emitter:
    """A non-radar transmitter: a declared complex baseband at a fixed position (R-ADR-051).

    ``samples`` is the baseband, relative to the full transmit amplitude (each
    magnitude at most one), at ``rate`` samples per second; ``power`` is the
    full-scale transmit power, dBm. The emitter's oscillator runs at the
    network's carrier plus ``carrier_offset`` [Hz] from world time zero, and
    sample 0 leaves at world time ``start`` [s]; with ``repeat`` the table
    repeats with no gap from then on, else it is sent once. Between samples
    the magnitude is held and the phase advances linearly to the next sample
    (a sampled tone stays a tone), which is the arbitrary pulse family's chip
    table with each chip's frequency set to its phase step. ``position`` is in
    world metres and ``polarization`` a world vector. The emitter radiates
    isotropically and is heard by every radar of the network.
    """

    position: Any
    samples: Any
    rate: float
    power: float
    carrier_offset: float = 0.0
    start: float = 0.0
    repeat: bool = True
    polarization: Any = (0.0, 1.0, 0.0)

    def __post_init__(self) -> None:
        set_ = object.__setattr__
        samples = tuple(complex(value) for value in self.samples)
        if not samples:
            raise ValueError("an Emitter needs at least one sample")
        if any(abs(value) > 1.0 + 1e-9 for value in samples):
            raise ValueError("Emitter.samples are relative to the full transmit amplitude; each magnitude is at most 1")
        set_(self, "samples", samples)
        set_(self, "position", tuple(float(v) for v in vec3_tensor(self.position, name="Emitter.position").tolist()))
        polarization = tuple(float(v) for v in vec3_tensor(self.polarization, name="Emitter.polarization").tolist())
        norm = math.sqrt(sum(v * v for v in polarization))
        if norm <= 1e-12:
            raise ValueError("Emitter.polarization must be non-zero")
        set_(self, "polarization", tuple(v / norm for v in polarization))
        for name in ("rate", "power", "carrier_offset", "start"):
            set_(self, name, _finite(f"Emitter.{name}", getattr(self, name)))
        if not self.rate > 0.0:
            raise ValueError("Emitter.rate must be positive")
        if type(self.repeat) is not bool:
            raise TypeError("Emitter.repeat must be a bool")

    @property
    def duration(self) -> float:
        """One pass of the table, s."""

        return len(self.samples) / self.rate

    def chip_table(self) -> np.ndarray:
        """``float64[samples, 3]``: ``|a_i|``, ``arg(a_i) / 2 pi`` and the phase-step frequency ``f_i`` [Hz]."""

        values = np.asarray(self.samples, dtype=np.complex128)
        following = np.roll(values, -1)
        if not self.repeat and len(values) > 1:
            # A single pass holds its last sample's phase rate.
            following[-1] = values[-1] * (values[-1] / values[-2] if values[-2] != 0 else 1.0)
        step = np.angle(following * np.conj(values))
        step[(values == 0) | (following == 0)] = 0.0
        return np.stack((np.abs(values), np.angle(values) / (2 * math.pi), step / (2 * math.pi) * self.rate), axis=1)


def _family(radar: Radar) -> str:
    return radar.system_config.kind


def _occupied_band(radar: Radar) -> float:
    """A transmitting radar's occupied bandwidth, Hz, for the sampled-receiver anti-alias refusal."""

    spec = radar.waveform_spec()
    if _family(radar) == WAVEFORM_OFDM:
        return float(spec.occupied_bandwidth_hz)
    return float(spec.bandwidth_hz)


def _sample_rate(radar: Radar) -> float:
    spec = radar.waveform_spec()
    if _family(radar) == WAVEFORM_OFDM:
        return 1.0 / float(spec.waveform_sample_period_s)
    return 1.0 / float(spec.sample_period_s)


def _require_hearing(
    network: Network, receiver: int, transmitter_family: str, band_hz: float, offset_hz: float, name: str
) -> None:
    """Refuse, by name, a transmitter this receiver's cross families do not synthesize."""

    radar = network.radars[receiver]
    family = _family(radar)
    if family == WAVEFORM_FMCW:
        if transmitter_family not in (WAVEFORM_FMCW, "emitter"):
            raise NotImplementedError(
                f"radar {receiver} is FMCW and {name} is {transmitter_family}: an FMCW receiver dechirps its ramp "
                "against an FMCW ramp or an emitter's table, whose instantaneous frequency its IF band gates; another "
                "waveform's beat through that band is not modelled (R-ADR-051)"
            )
        _fmcw_profile(radar, "receiver")
        return
    if transmitter_family == WAVEFORM_FMCW:
        raise NotImplementedError(
            f"radar {receiver} samples {family} and {name} is an FMCW ramp: sweeping through a directly sampled "
            "receiver needs that receiver's anti-alias filter, which is not modelled (R-ADR-051)"
        )
    if radar.num_tx != 1:
        raise NotImplementedError(
            f"radar {receiver} is a {family} receiver with {radar.num_tx} transmitters, whose channels are modelled "
            "as ideally orthogonal; another transmitter's signal is not, so a hearing sampled radar has one transmitter"
        )
    rate = _sample_rate(radar)
    if band_hz > rate * (1.0 + 1e-9) or abs(offset_hz) > 0.5 * rate:
        raise NotImplementedError(
            f"{name} occupies {band_hz:.6g} Hz at {offset_hz:.6g} Hz from radar {receiver}'s LO, against its complex "
            f"sample rate {rate:.6g} Hz: a band wider than the sample rate, or offset by more than half of it, needs "
            "the anti-alias filter, which is not modelled (R-ADR-051)"
        )


@dataclass(frozen=True, slots=True, eq=False)
class Network:
    """Several radars in one scene, simulated by one session (R-ADR-051).

    ``radars`` share one world: the scene is compiled once, every leg family is
    discovered once per topology epoch for all their elements, and each radar
    publishes its own :class:`~witwin.radar.Result`. Every radar hears its own
    transmitters as it would alone.

    ``links`` declares which other radars a radar hears: ``(i, j)`` puts radar
    ``i``'s transmissions into radar ``j``'s receiver, through the targets
    (bistatic returns), along the straight segment between them (``direct``
    on the verbs) and off the environment (``environment``). Each link is
    synthesized with radar ``j``'s own reference: an FMCW receiver dechirps
    radar ``i``'s FMCW ramp (slope mismatch, start-time and start-frequency
    offsets all appear in the beat); a pulsed or OFDM receiver samples radar
    ``i``'s pulse train or OFDM stream on its own grid and LO and matches or
    demodulates it with its own replica or symbols. ``emitters``
    (:class:`Emitter`) are heard by every radar the same way.

    ``clock_offsets`` is each radar's frame start relative to the frame times
    the verbs take, s (default all zero): radar ``k``'s frame ``f`` starts at
    ``times[f] + clock_offsets[k]``, and its own observation instants follow
    from there. Every oscillator runs from world time zero.

    ``carrier_offsets`` is each pulsed or OFDM radar's oscillator offset from
    the shared carrier, Hz (an FMCW radar declares its start frequency as a
    one-profile ``Chirp(start_offset=...)`` instead, and a nonzero entry for it
    is refused). It acts on what other radars hear and on what this radar
    hears from them; the radar's own rows keep its nominal carrier (their
    phase error is ``offset * tau``).

    ``if_bandwidths`` is each FMCW receiving radar's IF (anti-alias) bandwidth,
    Hz: a cross-radar row reaches the samples only while its instantaneous
    beat frequency is within it. ``None`` (the default) is half the radar's
    ADC sample rate. A pulsed or OFDM receiver has no IF model, so a
    transmitter whose band exceeds its complex sample rate, or whose carrier
    is offset by more than half of it, is refused.

    Every radar shares one carrier (``Radar.reference_frequency_hz``), the frequency the
    shared world is evaluated at, and one element pattern pair
    (``pattern``/``rx_pattern``), because one join applies one pattern table;
    with emitters the pattern must be isotropic. A pattern carrying a leaf is
    shared by being the same object on every radar, and its gradient collects
    every radar's rows; a pose trajectory's translation is a leaf per radar; a
    transmit-power leaf is refused (one table has no per-radar power).
    """

    radars: tuple[Radar, ...]
    links: tuple[tuple[int, int], ...] = ()
    clock_offsets: tuple[float, ...] = ()
    if_bandwidths: tuple[float | None, ...] = ()
    emitters: tuple[Emitter, ...] = ()
    carrier_offsets: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        from .sensors import Pattern

        set_ = object.__setattr__
        radars = tuple(self.radars)
        if not radars or not all(isinstance(radar, Radar) for radar in radars):
            raise TypeError("Network.radars must be a non-empty sequence of witwin.radar.Radar")
        set_(self, "radars", radars)
        count = len(radars)
        first = radars[0]
        for index, radar in enumerate(radars[1:], start=1):
            if float(radar.reference_frequency_hz) != float(first.reference_frequency_hz):
                raise ValueError(
                    f"radar {index} is at {radar.reference_frequency_hz} Hz and radar 0 at {first.reference_frequency_hz} Hz; a network evaluates "
                    "one world at one carrier. Declare a different start frequency as a one-profile chirp sequence "
                    "(FMCW) or a carrier offset (pulsed, OFDM)"
                )
            if radar.device != first.device:
                raise ValueError(
                    f"radar {index} is on {radar.device} and radar 0 on {first.device}; one network is one device"
                )
            if (
                radar.pattern != first.pattern
                or radar.system_config.sensors.rx_pattern != first.system_config.sensors.rx_pattern
            ):
                raise NotImplementedError(
                    f"radar {index} declares a different element pattern from radar 0; a network composes every "
                    "radar's rows in one join, which applies one transmit and one receive pattern table"
                )
        for index, radar in enumerate(radars):
            if isinstance(radar.power, torch.Tensor):
                raise NotImplementedError(
                    f"radar {index} declares its transmit power as a tensor (a leaf); a network applies one transmit "
                    "pattern table to every radar's transmitters, which carries no per-radar power factor. Give the "
                    "radars host powers inside a network"
                )
        emitters = tuple(self.emitters)
        if not all(isinstance(emitter, Emitter) for emitter in emitters):
            raise TypeError("Network.emitters holds witwin.radar.Emitter records")
        set_(self, "emitters", emitters)
        isotropic = Pattern.isotropic()
        if emitters and (first.pattern != isotropic or first.system_config.sensors.rx_pattern not in (None, isotropic)):
            raise NotImplementedError(
                "a network with emitters applies one transmit pattern table to every transmitter, emitters included, "
                "and an emitter radiates isotropically; give the radars isotropic elements"
            )
        offsets = tuple(self.clock_offsets) or (0.0,) * count
        if len(offsets) != count:
            raise ValueError(f"clock_offsets holds {len(offsets)} values for {count} radars")
        set_(self, "clock_offsets", tuple(_finite("clock_offsets", value) for value in offsets))
        carriers = tuple(self.carrier_offsets) or (0.0,) * count
        if len(carriers) != count:
            raise ValueError(f"carrier_offsets holds {len(carriers)} values for {count} radars")
        carriers = tuple(_finite("carrier_offsets", value) for value in carriers)
        for index, value in enumerate(carriers):
            if value != 0.0 and _family(radars[index]) == WAVEFORM_FMCW:
                raise ValueError(
                    f"radar {index} is FMCW; declare its start-frequency offset as Fmcw(sequence=(Chirp("
                    "start_offset=...),)), which its own rows carry too, not as a network carrier offset"
                )
        set_(self, "carrier_offsets", carriers)
        bandwidths = tuple(self.if_bandwidths) or (None,) * count
        if len(bandwidths) != count:
            raise ValueError(f"if_bandwidths holds {len(bandwidths)} values for {count} radars")
        checked: list[float | None] = []
        for bandwidth in bandwidths:
            if bandwidth is not None and not _finite("if_bandwidths", bandwidth) > 0.0:
                raise ValueError(f"an IF bandwidth must be positive, got {bandwidth!r}")
            checked.append(None if bandwidth is None else float(bandwidth))
        set_(self, "if_bandwidths", tuple(checked))
        links: list[tuple[int, int]] = []
        for link in tuple(self.links):
            if (
                not isinstance(link, tuple)
                or len(link) != 2
                or not all(type(value) is int and 0 <= value < count for value in link)
            ):
                raise ValueError(
                    f"a link is a pair (transmitting radar, receiving radar) of indices below {count}, got {link!r}"
                )
            if link[0] == link[1]:
                raise ValueError(f"link {link} names one radar twice; every radar always hears its own transmitters")
            if link in links:
                raise ValueError(f"link {link} is declared twice")
            links.append(link)
        set_(self, "links", tuple(links))
        for transmitter, receiver in links:
            tx = radars[transmitter]
            if _family(tx) == WAVEFORM_FMCW:
                _fmcw_profile(tx, "transmitter")
                band = 0.0
            else:
                band = _occupied_band(tx)
            offset = carriers[transmitter] - carriers[receiver]
            _require_hearing(self, receiver, _family(tx), band, offset, f"radar {transmitter}")
        for index, emitter in enumerate(emitters):
            for receiver in range(count):
                offset = emitter.carrier_offset - carriers[receiver]
                _require_hearing(self, receiver, "emitter", emitter.rate, offset, f"emitter {index}")
        for receiver in self.hearing():
            frontend = radars[receiver].frontend
            if frontend is not None and frontend.has_phase_noise:
                raise NotImplementedError(
                    f"radar {receiver} hears another transmitter and carries oscillator phase noise; the oscillators "
                    "are independent and the cross-radar families do not model either one's noise"
                )

    def hearing(self) -> tuple[int, ...]:
        """The radars that hear another transmitter: a link's receiver, or every radar when there are emitters."""

        if self.emitters:
            return tuple(range(len(self.radars)))
        return tuple(sorted({receiver for _, receiver in self.links}))

    def _same(self, other: Network) -> bool:
        return (
            len(self.radars) == len(other.radars)
            and all(a == b for a, b in zip(self.radars, other.radars, strict=True))
            and self.links == other.links
            and self.clock_offsets == other.clock_offsets
            and self.if_bandwidths == other.if_bandwidths
            and self.emitters == other.emitters
            and self.carrier_offsets == other.carrier_offsets
        )

    # -- the four verbs ------------------------------------------------------

    def trace(self, scene: Any, targets: Any, *, times: Any, poses: Any = None, **request: Any) -> NetworkPaths:
        """Run the shared world half and keep every composed row.

        Arguments are :meth:`simulate`'s. The record holds every frame's
        observations for all radars, so it costs what tracing each radar would,
        once.
        """

        session = _open(self, scene, targets, times=times, poses=poses, **request)
        frames = tuple(_traced_frames(session))
        return NetworkPaths(network=self, _session=session, _frames=frames)

    def echo(self, paths: NetworkPaths) -> tuple[Result, ...]:
        """Run every radar's instrument half over traced rows: one result per radar, in ``radars`` order."""

        if not isinstance(paths, NetworkPaths):
            raise TypeError(
                f"Network.echo takes the NetworkPaths that Network.trace returned, got {type(paths).__name__}"
            )
        if not self._same(paths.network):
            raise ValueError(
                "these paths were traced by a different network; echo them with the network that traced them"
            )
        session = paths._session
        frames: list[list[SceneFrame]] = [[] for _ in self.radars]
        for frame in paths._frames:
            for index, scene_frame in enumerate(_echo(session, frame)):
                frames[index].append(scene_frame)
        return tuple(assemble(member) for member in frames)

    def simulate(
        self,
        scene: Any,
        targets: Any,
        *,
        times: Any,
        poses: Any = None,
        los: bool = True,
        reflections: int = 1,
        transmission: bool = False,
        diffraction: bool = False,
        environment: bool = False,
        coupling: bool = False,
        direct: bool = True,
        discovery: str = "auto",
        max_paths: int | None = None,
        max_round_trips: int | None = None,
        motion: Motion | None = None,
        ad_mode: str = "none",
    ) -> tuple[Result, ...]:
        """Simulate every radar over one shared world: one result per radar, in ``radars`` order.

        The arguments are :meth:`Radar.simulate`'s, for the whole network.
        ``poses`` is one ``pose(t)`` (or ``None``) per radar, each the same
        ego-motion contract as ``Radar.simulate(pose=...)``. ``environment``
        adds the site-less environment returns to every pair (a radar's own
        and every declared link), ``coupling`` the straight segment between a
        radar's own transmit and receive elements, and ``direct`` the straight
        segment between the radars of each declared link (the direct
        interference path). ``motion`` is shared; ``Motion.auto()`` resolves to
        the exhaustive ``adc`` schedule when anything moves, because adaptive
        motion is refused for a network.
        """

        arguments: dict[str, Any] = {
            "los": los,
            "reflections": reflections,
            "transmission": transmission,
            "diffraction": diffraction,
            "environment": environment,
            "coupling": coupling,
            "direct": direct,
            "discovery": discovery,
            "max_paths": max_paths,
            "max_round_trips": max_round_trips,
            "motion": motion,
            "ad_mode": ad_mode,
        }
        session = _open(self, scene, targets, times=times, poses=poses, **arguments)
        frames: list[list[SceneFrame]] = [[] for _ in self.radars]
        for traced in _traced_frames(session):
            for index, scene_frame in enumerate(_echo(session, traced)):
                if frames[index]:
                    frames[index][-1] = frames[index][-1].released()
                frames[index].append(scene_frame)
        return tuple(assemble(member) for member in frames)

    def stream(
        self, scene: Any, targets: Any, *, times: Any, poses: Any = None, **request: Any
    ) -> Iterator[tuple[Result, ...]]:
        """The same session as :meth:`simulate`, one frame at a time: one result per radar per frame."""

        session = _open(self, scene, targets, times=times, poses=poses, **request)
        for traced in _traced_frames(session):
            yield tuple(assemble([scene_frame]) for scene_frame in _echo(session, traced))


@dataclass(frozen=True, slots=True, eq=False)
class NetworkPaths:
    """A network's traced frames: every observation's composed rows for every radar pair.

    Retention is the union of what each radar's :class:`~witwin.radar.Paths`
    would hold, stored once. :meth:`Network.echo` runs each radar's
    instrument half over it.
    """

    network: Network
    _session: _NetworkSession = field(repr=False)
    _frames: tuple[_TracedFrame, ...] = field(repr=False)

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    @property
    def observation_count(self) -> int:
        """Shared world instants evaluated across every frame (the radars' schedules merged)."""

        return sum(len(frame.observations) for frame in self._frames)

    @property
    def compile_count(self) -> int:
        return int(self._session.world.loop.compile_count)

    @property
    def discovery_count(self) -> int:
        return int(self._session.world.loop.discovery_count)


# ---------------------------------------------------------------------------
# The shared world half
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, eq=False)
class _Member:
    """One radar of the session and where its elements sit in the shared endpoint lists."""

    radar: Radar
    pose: Callable[[float], Any] | None
    tx_offset: int
    rx_offset: int
    #: ``float32[3, 3]`` local-to-world matrix of the authored orientation
    #: (columns: right, up, -boresight), on the radar's device.
    frame: torch.Tensor

    @property
    def num_tx(self) -> int:
        return self.radar.num_tx

    @property
    def num_rx(self) -> int:
        return self.radar.num_rx


def _rows(spec: RadarEndpointSpec, start: int, count: int) -> RadarEndpointSpec:
    """Rows ``[start, start + count)`` of an endpoint spec."""

    stop = start + count
    return RadarEndpointSpec(
        stable_ids=spec.stable_ids[start:stop],
        positions_m=spec.positions_m[start:stop],
        polarizations=spec.polarizations[start:stop],
        powers_w=None if spec.powers_w is None else spec.powers_w[start:stop],
    )


class NetworkWorldHalf(WorldHalf):
    """The world half of a network: one binding over every radar's elements.

    Transmitters and receivers are concatenated in radar order, with stable
    IDs ``DEFAULT_TRANSMITTER_ID_BASE + rank`` and
    ``DEFAULT_RECEIVER_ID_BASE + rank`` over the network's lists, so one join
    composes every pair. Each radar's elements follow its own pose. The
    element pattern sees each direction in its own radar's frame: the stage's
    frame is the identity and every point is turned into its element's radar
    frame before the stage reads it.
    """

    def __init__(
        self,
        members: tuple[_Member, ...],
        *,
        emitters: tuple[Emitter, ...],
        straight_groups: tuple[tuple[int, int, int, int], ...],
        **world: Any,
    ) -> None:
        from .sensors import watts_from_dbm

        super().__init__(**world)
        self.members = members
        self.emitters = emitters
        #: ``(first transmitter, count, first receiver, count)`` groups whose
        #: straight-segment rows are kept, so whose endpoints must be apart.
        self.straight_groups = straight_groups
        self.radar_tx = sum(member.num_tx for member in members)
        self.num_tx = self.radar_tx + len(emitters)
        self.num_rx = sum(member.num_rx for member in members)
        device = members[0].radar.device
        self.transmitter_ids = tuple(range(DEFAULT_TRANSMITTER_ID_BASE, DEFAULT_TRANSMITTER_ID_BASE + self.num_tx))
        self.receiver_ids = tuple(range(DEFAULT_RECEIVER_ID_BASE, DEFAULT_RECEIVER_ID_BASE + self.num_rx))
        self._transmitter_ids = torch.tensor(self.transmitter_ids, dtype=torch.int64, device=device)
        self._receiver_ids = torch.tensor(self.receiver_ids, dtype=torch.int64, device=device)
        self._emitters: RadarEndpointSpec | None = None
        if emitters:
            # Fixed endpoints, uploaded once: an emitter has no pose.
            self._emitters = RadarEndpointSpec(
                stable_ids=self._transmitter_ids[self.radar_tx :],
                positions_m=torch.tensor([e.position for e in emitters], dtype=torch.float32, device=device),
                polarizations=torch.tensor([e.polarization for e in emitters], dtype=torch.float32, device=device),
                powers_w=torch.tensor([watts_from_dbm(e.power) for e in emitters], dtype=torch.float32, device=device),
            )
        self._tx_owner = torch.tensor(
            [k for k, member in enumerate(members) for _ in range(member.num_tx)], dtype=torch.int64, device=device
        )
        self._rx_owner = torch.tensor(
            [k for k, member in enumerate(members) for _ in range(member.num_rx)], dtype=torch.int64, device=device
        )
        self._static_frames = torch.stack([member.frame for member in members])
        self._frame_index: dict[tuple[int, str], torch.Tensor] = {}

    def _bind_world(self, snapshot: object, time_s: float | None = None) -> RadarWorldBinding:
        instant = self._clock if time_s is None else time_s
        parts: list[RadarWorldBinding] = []
        sites = self.sites
        for member in self.members:
            radar = member.radar
            binding = bind_radar_world(
                radar,
                snapshot,
                sites=sites,
                polarization=radar.polarization_vector,
                rx_polarization=radar.rx_polarization_vector,
            )
            if not parts:
                # The sites (a mesh target's facet frame included) are resolved
                # once per binding, by the first radar; the others bind their
                # elements against those positions and resolve nothing.
                sites = ScatterSitePolicy.explicit(binding.site_positions_m, stable_ids=binding.site_ids)
            if member.pose is not None:
                rotation, translation = pose_at(member.pose, instant, binding.device, ad_mode=self.ad_mode)
                binding = carry_binding(binding, radar.position, rotation, translation)
            parts.append(binding)
        return self._union(parts)

    def _union(self, parts: list[RadarWorldBinding]) -> RadarWorldBinding:
        overlap = set(parts[0].site_ids) & (set(self.transmitter_ids) | set(self.receiver_ids))
        if overlap:
            raise ValueError(
                f"site stable IDs {sorted(overlap)} collide with the network's transmitter or receiver blocks; "
                "two endpoints cannot share one stable world ID"
            )

        def joined(specs: list[RadarEndpointSpec], ids: torch.Tensor) -> RadarEndpointSpec:
            powers = [spec.powers_w for spec in specs]
            return RadarEndpointSpec(
                stable_ids=ids,
                positions_m=torch.cat([spec.positions_m for spec in specs]),
                polarizations=torch.cat([spec.polarizations for spec in specs]),
                powers_w=None if powers[0] is None else torch.cat([value for value in powers if value is not None]),
            )

        receivers = joined([part.receivers for part in parts], self._receiver_ids)
        sources = torch.cat(
            [part.receiver_sources.powers_w for part in parts if part.receiver_sources.powers_w is not None]
        )
        transmitters = [part.transmitters for part in parts]
        if self._emitters is not None:
            transmitters.append(self._emitters)
        return RadarWorldBinding(
            transmitters=joined(transmitters, self._transmitter_ids),
            receivers=receivers,
            receiver_sources=replace(receivers, powers_w=sources),
            sites=parts[0].sites,
            transmitter_ids=self.transmitter_ids,
            receiver_ids=self.receiver_ids,
            site_ids=parts[0].site_ids,
            facets=parts[0].facets,
        )

    def _direct_separation(
        self, binding: RadarWorldBinding
    ) -> list[tuple[RadarEndpointSpec, RadarEndpointSpec]] | None:
        """The ``(transmitters, receivers)`` groups whose straight-segment rows are consumed.

        Only those pairs must be apart; a radar's transmitter beside its own
        receiver is evaluated by the shared direct leg but its row is dropped
        unless ``coupling`` asked for it.
        """

        return [
            (_rows(binding.transmitters, tx_first, tx_count), _rows(binding.receivers, rx_first, rx_count))
            for tx_first, tx_count, rx_first, rx_count in self.straight_groups
        ]

    def _freeze_pattern(self, composer: TwoWayComposer, binding: RadarWorldBinding) -> RoundTripPatternStage:
        return RoundTripPatternStage.freeze(
            self.radar,
            composer,
            site_ids=binding.site_ids,
            pattern=self.antenna_pattern,
            rx_pattern=self.radar.system_config.sensors.rx_pattern,
            array_size=(self.num_tx, self.num_rx),
            pattern_frame=torch.eye(3, dtype=torch.float32, device=binding.device),
        )

    def _frames_at(self, time_s: float, device: torch.device) -> torch.Tensor:
        """Every radar's local-to-world matrix at ``time_s``, ``float32[radars, 3, 3]``."""

        if all(member.pose is None for member in self.members):
            return self._static_frames
        frames = []
        for member in self.members:
            if member.pose is None:
                frames.append(member.frame)
            else:
                frames.append(pose_at(member.pose, time_s, device, ad_mode=self.ad_mode)[0] @ member.frame)
        return torch.stack(frames)

    def _pending(self, time_s: float, frame: Any) -> _PendingObservation:
        binding = self._binding_for(frame)
        identity_pattern = _round_trip(frame.frozen).pattern.plan.is_identity
        return _PendingObservation(
            time_s=time_s,
            epoch_frame=frame,
            binding=binding,
            identity=self._identity(frame),
            rotation=None if identity_pattern else self._frames_at(time_s, binding.device),
        )

    def _owner_frames(self, slots: int, role: str) -> torch.Tensor:
        """``int64[slots * elements]``: each slot-major element's row in the ``[slots * radars]`` frame stack."""

        key = (slots, role)
        index = self._frame_index.get(key)
        if index is None:
            owner = self._tx_owner if role == "tx" else self._rx_owner
            slot = torch.arange(slots, dtype=torch.int64, device=owner.device)[:, None]
            index = (slot * len(self.members) + owner[None, :]).reshape(-1)
            self._frame_index[key] = index
        return index

    def _body_frame(
        self, group: list[_PendingObservation], pattern: RoundTripPatternStage, *points: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """Every point seen in its element's own radar frame: one gather and one batched product per point set."""

        if pattern.plan.is_identity:
            return points
        frames = torch.stack([entry.rotation for entry in group if entry.rotation is not None]).reshape(-1, 3, 3)
        slots = len(group)
        tx_pos, rx_pos, tx_targets, rx_targets = points
        tx_frames = frames.index_select(0, self._owner_frames(slots, "tx"))
        rx_frames = frames.index_select(0, self._owner_frames(slots, "rx"))

        def turned(values: torch.Tensor, per_row: torch.Tensor) -> torch.Tensor:
            # A row vector times M is M^T v: the components along M's columns.
            return torch.bmm(values.unsqueeze(1), per_row).squeeze(1).contiguous()

        return (
            turned(tx_pos, tx_frames),
            turned(rx_pos, rx_frames),
            turned(tx_targets, tx_frames.index_select(0, pattern.tx_index)),
            turned(rx_targets, rx_frames.index_select(0, pattern.rx_index)),
        )

    def _strided(self, round_trip: Any) -> bool:
        # Every response strides; a directional pattern needs each element's
        # own radar frame, which only the per-row route turns points into.
        return bool(round_trip.pattern.plan.is_identity)


# ---------------------------------------------------------------------------
# Opening the session
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, eq=False)
class _NetworkSession:
    network: Network
    members: tuple[_Member, ...]
    world: NetworkWorldHalf
    #: Each radar's own instrument half.
    sessions: tuple[Session, ...]
    times: tuple[float, ...]
    #: The merged schedule's offsets from a frame time, s, and each radar's
    #: observations as indices into it.
    offsets: np.ndarray
    schedule: tuple[np.ndarray, ...]
    path_set_complete: bool
    exhaustive: bool
    coupling: bool
    environment: bool
    direct: bool
    routes: tuple[_Receiver, ...]


@dataclass(frozen=True, slots=True, eq=False)
class _TracedFrame:
    time_s: float
    observations: tuple[Observation, ...]
    events: PathEvents
    discovery: DiscoveryReport


def _poses(poses: Any, count: int) -> tuple[Callable[[float], Any] | None, ...]:
    if poses is None:
        return (None,) * count
    values = tuple(poses)
    if len(values) != count:
        raise ValueError(
            f"poses holds {len(values)} entries for {count} radars; pass None for a radar that holds still"
        )
    for value in values:
        if value is not None and not callable(value):
            raise TypeError("each pose must be None or callable as pose(t) -> (rotation (3, 3), translation (3,))")
    return values


def _open(
    network: Network,
    scene: object,
    targets: Any,
    *,
    times: Sequence[float],
    poses: Any = None,
    los: bool = True,
    reflections: int = 1,
    transmission: bool = False,
    diffraction: bool = False,
    environment: bool = False,
    coupling: bool = False,
    direct: bool = True,
    discovery: str = "auto",
    max_paths: int | None = None,
    max_round_trips: int | None = None,
    motion: Motion | None = None,
    ad_mode: str = "none",
) -> _NetworkSession:
    radars = network.radars
    member_poses = _poses(poses, len(radars))
    for name, value in (("environment", environment), ("coupling", coupling), ("direct", direct)):
        if type(value) is not bool:
            raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
    requested = Motion.auto() if motion is None else motion
    if not isinstance(requested, Motion):
        raise TypeError(f"motion must be a Motion, got {type(requested).__name__}")
    if requested.kind == "adaptive":
        raise NotImplementedError(
            "a network merges every radar's observation instants into one schedule, and adaptive motion probes one "
            "radar's ramp; use Motion.adc() (the exhaustive reference) or Motion.chirp()"
        )
    if requested.rediscover_every_chirps is not None:
        raise NotImplementedError(
            "Motion.rediscover_every_chirps counts one radar's chirps; a network's radars have their own, so declare "
            "a frame, time or displacement cadence"
        )
    arguments = [
        radar._session(
            targets,
            times,
            los,
            reflections,
            requested,
            ad_mode,
            None,
            transmission=transmission,
            diffraction=diffraction,
            discovery=discovery,
            max_paths=max_paths,
            max_round_trips=max_round_trips,
            pose=pose,
        )
        for radar, pose in zip(radars, member_poses, strict=True)
    ]
    dynamic = _dynamic_scene(scene)
    sampled = any(moves_within_frame(dynamic, entry["sites"], entry["response"], entry["pose"]) for entry in arguments)
    if sampled and requested.kind == "auto":
        requested = replace(requested, kind="adc")
    components = arguments[0]["components"]
    straight_pairs: list[tuple[int, int]] = []
    direct_components: set[str] = set()
    if environment:
        interacting = components & {"reflection", "diffraction"}
        if not interacting:
            raise ValueError(
                "environment=True asks for the environment's own returns, which are reflections or diffraction; "
                "set reflections>0 or diffraction=True"
            )
        direct_components |= interacting
    if coupling:
        straight_pairs += [(k, k) for k in range(len(radars))]
    if direct:
        straight_pairs += list(network.links)
    if direct:
        straight_pairs += [(len(radars) + e, k) for e in range(len(network.emitters)) for k in range(len(radars))]
    if straight_pairs:
        direct_components.add("los")
    if transmission and direct_components:
        raise NotImplementedError(
            "a network's site-less leg serves every radar pair and its rows are kept per pair by depth (a straight "
            "segment has none); transmission rows have depth too, so transmission is not combined with "
            "environment, coupling or direct links in a network"
        )
    for radar in radars:
        radar._require_cuda()
    receivers = set(network.hearing())
    prepared = [
        prepare_session(radar, scene, **{**entry, "motion": requested}, world_moves=sampled, beat=index in receivers)
        for index, (radar, entry) in enumerate(zip(radars, arguments, strict=True))
    ]
    first = prepared[0]
    kinds = {entry.session["kind"] for entry in prepared}
    if len(kinds) != 1:
        raise NotImplementedError(f"the network's radars resolved to different motion schedules {sorted(kinds)}")
    member_offsets = [
        np.asarray(entry.session["offsets"], dtype=np.float64) + network.clock_offsets[index]
        for index, entry in enumerate(prepared)
    ]
    merged, inverse = np.unique(np.concatenate(member_offsets), return_inverse=True)
    bounds = np.cumsum([0] + [len(values) for values in member_offsets])
    schedule = tuple(inverse[bounds[k] : bounds[k + 1]] for k in range(len(radars)))
    instants = first.session["times"]
    span = float(merged[-1] - merged[0])
    if len(merged) > 1 and any(b - a <= span for a, b in zip(instants, instants[1:], strict=False)):
        raise ValueError(
            "the network's frames overlap: every radar's observations of a frame must end before the next frame's begin"
        )

    world_arguments = dict(first.world)
    request = replace(world_arguments["request"], direct_components=frozenset(direct_components))
    unfrozen = (frozenset(components) | request.direct_components) & UNFREEZABLE_COMPONENTS
    request = replace(request, replays=not unfrozen)
    cadence = world_arguments["cadence"]
    if sampled and requested.rediscover_every_frames is not None:
        cadence = requested.rediscover_every_frames * len(merged)
    world_arguments.update(
        request=request, cadence=cadence, pose=next((pose for pose in member_poses if pose is not None), None)
    )
    device = radars[0].device
    members: list[_Member] = []
    tx_offset = rx_offset = 0
    for radar, pose in zip(radars, member_poses, strict=True):
        frame = radar._world_from_local_matrix(device=device, dtype=torch.float32)[1].contiguous()
        members.append(_Member(radar=radar, pose=pose, tx_offset=tx_offset, rx_offset=rx_offset, frame=frame))
        tx_offset += radar.num_tx
        rx_offset += radar.num_rx
    radar_tx = tx_offset

    def transmitters(owner: int) -> tuple[int, int]:
        if owner < len(members):
            return members[owner].tx_offset, members[owner].num_tx
        return radar_tx + owner - len(members), 1

    straight_groups = tuple(
        (*transmitters(owner), members[receiver].rx_offset, members[receiver].num_rx)
        for owner, receiver in straight_pairs
    )
    world = NetworkWorldHalf(
        tuple(members), emitters=network.emitters, straight_groups=straight_groups, **world_arguments
    )
    sessions = tuple(
        Session(loop=world.loop, **{**entry.session, "environment": environment, "coupling": coupling})
        for entry in prepared
    )
    routes = tuple(
        _Receiver(
            index=index,
            network=network,
            members=tuple(members),
            session=sessions[index],
            coupling=coupling,
            environment=environment,
            direct=direct,
        )
        for index in range(len(radars))
    )
    return _NetworkSession(
        network=network,
        members=tuple(members),
        world=world,
        sessions=sessions,
        times=tuple(instants),
        offsets=merged,
        schedule=schedule,
        path_set_complete=first.complete,
        exhaustive=first.exhaustive,
        coupling=coupling,
        environment=environment,
        direct=direct,
        routes=routes,
    )


def _traced_frames(session: _NetworkSession) -> Iterator[_TracedFrame]:
    """Every frame's shared observations, drained so each radar can read its own instants."""

    world = session.world
    for start in session.times:
        events = PathEvents()
        observations = tuple(world.observations(start, session.offsets, events))
        yield _TracedFrame(time_s=float(start), observations=observations, events=events, discovery=world.report())


def _echo(session: _NetworkSession, frame: _TracedFrame) -> list[SceneFrame]:
    """Each radar's frame: its own rows through its own instrument half, plus what it hears from its links."""

    published: list[SceneFrame] = []
    for index, route in enumerate(session.routes):
        member_session = session.sessions[index]
        start = frame.time_s + session.network.clock_offsets[index]
        selected = [frame.observations[int(position)] for position in session.schedule[index]]
        trace = FrameTrace(
            time_s=start,
            sample_times_s=start + member_session.offsets,
            observations=tuple(route.observation(observation) for observation in selected),
            path_set_complete=session.path_set_complete,
            motion_sampling_exhaustive=session.exhaustive,
            events=frame.events,
            discovery=partial(_constant_report, frame.discovery),
        )
        interference = None
        if route.links:
            interference = partial(route.interference, selected, start)
        published.append(echo_frame(member_session, trace, interference))
        route.end_frame()
    return published


# ---------------------------------------------------------------------------
# One receiving radar's view of the shared rows
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, eq=False)
class _OwnRows:
    """A receiving radar's own transmitters' rows of one frozen join, gathered once per epoch."""

    composer: TwoWayComposer
    index: torch.Tensor
    host_index: np.ndarray
    offsets: np.ndarray
    sensor_pair_index: torch.Tensor
    pair_offsets: torch.Tensor
    topology: RadarPathTopology


@dataclass(frozen=True, slots=True, eq=False)
class _HeardRows:
    """Which rows of one frozen join a receiving radar hears from its links."""

    composer: TwoWayComposer
    row_link: torch.Tensor
    row_tx: torch.Tensor


@dataclass(frozen=True, slots=True, eq=False)
class _Source:
    """A transmitter owner a receiver hears through a cross family: a radar, or an emitter."""

    owner: int
    radar: Radar | None = None
    emitter: Emitter | None = None


@dataclass(frozen=True, slots=True, eq=False)
class _Family:
    """One cross family a receiver runs: its sources (position = link id), its tables and its entry."""

    name: str
    sources: tuple[_Source, ...]
    #: Each source's first entry in ``aux`` (chips or OFDM symbols).
    first: tuple[int, ...]
    aux: torch.Tensor
    synthesize: Callable[..., torch.Tensor]


def _family_tables(
    name: str, sources: tuple[_Source, ...], device: torch.device
) -> tuple[tuple[int, ...], torch.Tensor]:
    """The transmitters' constant tables, concatenated: chips ``float64[n, 3]`` or OFDM symbols ``float32[n, 2]``."""

    first: list[int] = []
    pieces: list[np.ndarray] = []
    entries = 0
    for source in sources:
        first.append(entries)
        if source.emitter is not None:
            table = source.emitter.chip_table()
        elif name == WAVEFORM_OFDM:
            assert source.radar is not None
            symbols = source.radar.waveform_spec().symbol_table().reshape(-1)
            table = np.stack((symbols.real, symbols.imag), axis=1)
        else:
            assert source.radar is not None
            spec = source.radar.waveform_spec()
            table = np.asarray(spec.chip_table() if name == WAVEFORM_PULSED and spec.is_coded else (), dtype=np.float64)
        table = np.asarray(table, dtype=np.float64).reshape(-1, 2 if name == WAVEFORM_OFDM else 3)
        pieces.append(table)
        entries += len(table)
    width = 2 if name == WAVEFORM_OFDM else 3
    joined = np.concatenate(pieces) if pieces else np.zeros((0, width))
    dtype = torch.float32 if name == WAVEFORM_OFDM else torch.float64
    return tuple(first), torch.from_numpy(np.ascontiguousarray(joined)).to(device=device, dtype=dtype)


class _Receiver:
    """One radar as a receiver: its own rows for its instrument half, and the routing of what else it hears."""

    def __init__(
        self,
        *,
        index: int,
        network: Network,
        members: tuple[_Member, ...],
        session: Session,
        coupling: bool,
        environment: bool,
        direct: bool,
    ) -> None:
        from .synthesis.fmcw import synthesize_fmcw_cross
        from .synthesis.ofdm import synthesize_ofdm_cross
        from .synthesis.pulsed import synthesize_pulsed_cross

        self.index = index
        self.network = network
        self.members = members
        self.member = members[index]
        self.session = session
        self.coupling = coupling
        self.environment = environment
        self.direct = direct
        radar_tx = sum(member.num_tx for member in members)
        self.num_tx_total = radar_tx + len(network.emitters)
        #: Each shared transmitter's owner (radar index, or radars + emitter
        #: index) and its index within that owner.
        self._tx_owner = np.concatenate(
            [np.full(member.num_tx, k) for k, member in enumerate(members)]
            + [np.arange(len(members), len(members) + len(network.emitters))]
        ).astype(np.int64)
        self._tx_local = np.concatenate(
            [np.arange(member.num_tx) for member in members] + [np.zeros(len(network.emitters))]
        ).astype(np.int64)
        self._rx_radar = np.concatenate([np.full(member.num_rx, k) for k, member in enumerate(members)])
        self._own: _Bounded[int, _OwnRows] = _Bounded(ROUTING_CACHE)
        self._heard: _Bounded[tuple[int, str], _HeardRows] = _Bounded(ROUTING_CACHE)
        self._batches: dict[int, tuple[SlotBatch, SlotBatch]] = {}
        self._views: dict[tuple[int, int], tuple[SlotBatch, RadarPathBatch]] = {}
        self._previous_batches: dict[int, tuple[SlotBatch, SlotBatch]] = {}
        self._previous_views: dict[tuple[int, int], tuple[SlotBatch, RadarPathBatch]] = {}
        radar = self.member.radar
        self.kind = _family(radar)
        sources = [_Source(owner=a, radar=network.radars[a]) for a, b in network.links if b == index]
        sources += [_Source(owner=len(members) + e, emitter=emitter) for e, emitter in enumerate(network.emitters)]
        if self.kind == WAVEFORM_FMCW:
            grouped = {WAVEFORM_FMCW: sources}
        else:
            kinds = [WAVEFORM_PULSED if s.radar is None else _family(s.radar) for s in sources]
            grouped = {
                WAVEFORM_PULSED: [s for s, kind in zip(sources, kinds, strict=True) if kind == WAVEFORM_PULSED],
                WAVEFORM_OFDM: [s for s, kind in zip(sources, kinds, strict=True) if kind == WAVEFORM_OFDM],
            }
        entries = {
            WAVEFORM_FMCW: synthesize_fmcw_cross,
            WAVEFORM_PULSED: synthesize_pulsed_cross,
            WAVEFORM_OFDM: synthesize_ofdm_cross,
        }
        self.families: list[_Family] = []
        for name, members_of in grouped.items():
            if not members_of:
                continue
            first, aux = _family_tables(name, tuple(members_of), radar.device)
            self.families.append(
                _Family(name=name, sources=tuple(members_of), first=first, aux=aux, synthesize=entries[name])
            )
        self.links = bool(self.families)
        if not self.links:
            return
        spec: Any = session.specs.working
        self.spec = spec
        # The receive grid: each slot's start in the frame, its first sample's
        # offset from it and the LO offset of the cell, plus the cube's shape.
        if self.kind == WAVEFORM_FMCW:
            profiles = spec.slot_profiles()
            self.slot_start, self.first_sample = profiles[:, 0], profiles[:, 3]
            self.lo = np.zeros(len(profiles))
            self.slow, self.fast, self.period = spec.num_chirps, spec.num_samples, spec.sample_period_s
            self.rx_slope, self.rx_offset, _, _ = _fmcw_profile(radar, "receiver")
            bandwidth = network.if_bandwidths[index]
            self.if_bandwidth = 0.5 / spec.sample_period_s if bandwidth is None else bandwidth
        elif self.kind == WAVEFORM_PULSED:
            slots = np.arange(spec.num_pulses, dtype=np.float64)
            self.slot_start = slots * spec.pri_s
            self.first_sample = np.full(len(slots), spec.range_gate_start_s)
            self.lo = slots * spec.frequency_step_hz
            self.slow, self.fast, self.period = spec.num_pulses, spec.num_samples, spec.sample_period_s
        else:
            slots = np.arange(spec.num_symbols, dtype=np.float64)
            self.slot_start = slots * spec.symbol_period_s
            self.first_sample = np.full(len(slots), spec.cyclic_prefix_s)
            self.lo = np.zeros(len(slots))
            self.slow, self.fast = spec.num_symbols, spec.samples_per_symbol
            self.period = spec.waveform_sample_period_s
        self.slot_count = len(self.slot_start)

    def end_frame(self) -> None:
        """Keep this frame's gathered batches for the next one, which a still world re-publishes."""

        self._previous_batches, self._batches = self._batches, {}
        self._previous_views, self._views = self._views, {}

    # -- own rows ---------------------------------------------------------------

    def _kept_direct(self, composer: TwoWayComposer, rows: np.ndarray, *, own: bool) -> np.ndarray:
        """Which of ``rows`` are kept: every site row, and the site-less rows this pair class asked for."""

        identity = composer.identity_host[rows]
        site_less = identity[:, 1] == NO_SITE
        straight = identity[:, 5] == 0
        wanted_straight = self.coupling if own else self.direct
        keep = ~site_less | (straight & wanted_straight) | (~straight & self.environment)
        return rows[keep]

    def _own_rows(self, composer: TwoWayComposer) -> _OwnRows:
        cached = self._own.get(id(composer))
        if cached is not None and cached.composer is composer:
            return cached
        member = self.member
        offsets = composer.pair_offsets_host
        total_tx = self.num_tx_total
        pieces: list[np.ndarray] = []
        counts: list[int] = []
        for r in range(member.num_rx):
            for t in range(member.num_tx):
                pair = (member.rx_offset + r) * total_tx + member.tx_offset + t
                rows = self._kept_direct(composer, np.arange(offsets[pair], offsets[pair + 1]), own=True)
                pieces.append(rows)
                counts.append(len(rows))
        host = np.concatenate(pieces).astype(np.int64) if pieces else np.zeros(0, dtype=np.int64)
        pair_offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
        pair_index = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
        device = composer.pair_offsets.device
        uploaded = torch.from_numpy(np.concatenate((host, pair_offsets, pair_index))).to(device)
        index, pair_offsets_device, pair_index_device = torch.split(uploaded, [len(host), len(pair_offsets), len(host)])
        union = composer.topology
        topology = RadarPathTopology(
            radar_source_id=union.radar_source_id.index_select(0, index),
            site_id=union.site_id.index_select(0, index),
            radar_sink_id=union.radar_sink_id.index_select(0, index),
            inbound_row=union.inbound_row.index_select(0, index),
            outbound_row=union.outbound_row.index_select(0, index),
            direct_row=None if union.direct_row is None else union.direct_row.index_select(0, index),
        )
        own = _OwnRows(
            composer=composer,
            index=index,
            host_index=host,
            offsets=pair_offsets,
            sensor_pair_index=pair_index_device,
            pair_offsets=pair_offsets_device,
            topology=topology,
        )
        self._own.put(id(composer), own)
        return own

    def _own_batch(self, batch: SlotBatch, composer: TwoWayComposer) -> SlotBatch:
        key = id(batch)
        for store in (self._batches, self._previous_batches):
            held = store.get(key)
            if held is not None and held[0] is batch:
                self._batches[key] = held
                return held[1]
        own = self._own_rows(composer)
        rows = batch.rows
        per_slot = composer.path_count
        index = own.index
        if batch.slots > 1:
            slot = torch.arange(batch.slots, dtype=torch.int64, device=index.device)[:, None] * per_slot
            index = (slot + index[None, :]).reshape(-1)

        def gathered(value: torch.Tensor | None) -> torch.Tensor | None:
            return None if value is None else value.index_select(0, index)

        delay = gathered(rows.total_delay_s)
        transfer = gathered(rows.complex_transfer_ref)
        assert delay is not None and transfer is not None
        member_batch = SlotBatch(
            rows=SlotRows(
                total_delay_s=delay,
                complex_transfer_ref=transfer,
                row_valid=gathered(rows.row_valid),
                slots=batch.slots,
                rows_per_slot=len(own.host_index),
                weight_includes_antenna_pattern=rows.weight_includes_antenna_pattern,
                total_delay_residual_s=gathered(rows.total_delay_residual_s),
            ),
            offsets=own.offsets,
            slots=batch.slots,
        )
        self._batches[key] = (batch, member_batch)
        return member_batch

    def observation(self, observation: Observation) -> Observation:
        """``observation`` as this radar alone would have composed it: its own pairs' rows only."""

        batch = observation.batch
        assert batch is not None
        key = (id(batch), observation.slot)
        for store in (self._views, self._previous_views):
            held = store.get(key)
            if held is not None and held[0] is batch:
                self._views[key] = held
                return replace(observation, composed=held[1], batch=self._own_batch(batch, self._composer(observation)))
        composer = self._composer(observation)
        own = self._own_rows(composer)
        member_batch = self._own_batch(batch, composer)
        rows = member_batch.rows
        count = rows.rows_per_slot
        selection = slice(observation.slot * count, (observation.slot + 1) * count)
        radar = self.member.radar
        composed = RadarPathBatch(
            sensor_pair_count=radar.num_tx * radar.num_rx,
            path_count=count,
            sensor_pair_index=own.sensor_pair_index,
            pair_offsets=own.pair_offsets,
            total_delay_s=rows.total_delay_s[selection],
            delay_rate=None,
            complex_transfer_ref=rows.complex_transfer_ref[selection],
            reference_frequency_hz=composer.reference_frequency_hz,
            row_valid=None if rows.row_valid is None else rows.row_valid[selection],
            topology=own.topology,
            weight_includes_antenna_pattern=rows.weight_includes_antenna_pattern,
            total_delay_residual_s=(
                None if rows.total_delay_residual_s is None else rows.total_delay_residual_s[selection]
            ),
        )
        self._views[key] = (batch, composed)
        return replace(observation, composed=composed, batch=member_batch)

    @staticmethod
    def _composer(observation: Observation) -> TwoWayComposer:
        return _round_trip(observation.epoch_frame.frozen).composer

    # -- heard rows ---------------------------------------------------------------

    def _heard_rows(self, composer: TwoWayComposer, family: _Family) -> _HeardRows:
        key = (id(composer), family.name)
        cached = self._heard.get(key)
        if cached is not None and cached.composer is composer:
            return cached
        identity = composer.identity_host
        tx_rank = identity[:, 0] - DEFAULT_TRANSMITTER_ID_BASE
        rx_rank = identity[:, 2] - DEFAULT_RECEIVER_ID_BASE
        owner = self._tx_owner[tx_rank]
        mine = self._rx_radar[rx_rank] == self.index
        link = np.full(len(identity), -1, dtype=np.int64)
        for position, source in enumerate(family.sources):
            link[(owner == source.owner) & mine] = position
        heard = np.flatnonzero(link >= 0)
        dropped = np.setdiff1d(heard, self._kept_direct(composer, heard, own=False))
        link[dropped] = -1
        device = composer.pair_offsets.device
        uploaded = torch.from_numpy(np.concatenate((link, self._tx_local[tx_rank])).astype(np.int32)).to(device)
        row_link, row_tx = torch.split(uploaded, [len(identity), len(identity)])
        heard_rows = _HeardRows(composer=composer, row_link=row_link, row_tx=row_tx)
        self._heard.put(key, heard_rows)
        return heard_rows

    def _link_row(self, family: _Family, position: int, frame_start_s: float) -> tuple[float, ...]:
        """One link row of ``family``'s table for the frame starting at ``frame_start_s`` (world time, s)."""

        source = family.sources[position]
        first = float(family.first[position])
        clock = self.network.clock_offsets
        carriers = self.network.carrier_offsets
        emitter = source.emitter
        # The transmitter's frame start (an emitter's first sample) relative to this receiver's.
        shift = emitter.start - frame_start_s if emitter is not None else clock[source.owner] - clock[self.index]
        if family.name == WAVEFORM_FMCW:
            if emitter is not None:
                tx_slope, tx_offset, ramp, period = 0.0, emitter.carrier_offset, emitter.duration, emitter.duration
                slots, count, kind, chips = (1.0 if not emitter.repeat else 2.0**52), 1.0, 1.0, len(emitter.samples)
            else:
                assert source.radar is not None
                tx_slope, tx_offset, ramp, period = _fmcw_profile(source.radar, "transmitter")
                slots, count = float(source.radar.waveform_spec().slot_count), float(source.radar.num_tx)
                kind, chips = 0.0, 0
            cycles = (self.rx_offset - tx_offset) * frame_start_s
            return (
                tx_slope,
                tx_offset,
                ramp,
                period,
                slots,
                count,
                shift,
                self.rx_slope,
                self.rx_offset,
                cycles - math.floor(cycles),
                self.if_bandwidth,
                kind,
                first,
                float(chips),
            )
        rx_offset = carriers[self.index]
        if family.name == WAVEFORM_PULSED:
            if emitter is not None:
                width = emitter.duration
                return (shift, emitter.carrier_offset, rx_offset, frame_start_s, width,
                        0.0 if emitter.repeat else 1.0, 2.0, width, emitter.rate, 1.0, first,
                        float(len(emitter.samples)), 0.0, 0.0)  # fmt: skip
            assert source.radar is not None
            from .synthesis.pulsed import pulse_kind_code

            spec = source.radar.waveform_spec()
            chips = len(spec.chip_table()) if spec.is_coded else 0
            return (shift, carriers[source.owner], rx_offset, frame_start_s, spec.pri_s, float(spec.num_pulses),
                    float(pulse_kind_code(spec)), spec.pulse_width_s, spec.bandwidth_hz, spec.pulse_amplitude, first,
                    float(chips), float(spec.periodic), spec.frequency_step_hz)  # fmt: skip
        assert source.radar is not None
        spec = source.radar.waveform_spec()
        return (shift, carriers[source.owner], rx_offset, frame_start_s, spec.symbol_period_s, float(spec.num_symbols),
                spec.cyclic_prefix_s, spec.subcarrier_spacing_hz, float(spec.num_subcarriers), first,
                1.0 / math.sqrt(spec.num_subcarriers))  # fmt: skip

    def _link_table(self, family: _Family, frame_start_s: float, device: torch.device) -> torch.Tensor:
        rows = [self._link_row(family, position, frame_start_s) for position in range(len(family.sources))]
        table = np.asarray(rows, dtype=np.float64)
        return _uploaded(self.session, table.view(np.int64), device).view(torch.float64).reshape(table.shape)

    def interference(self, observations: list[Observation], frame_start_s: float) -> torch.Tensor:
        """What this radar hears from other transmitters over one frame, ``[slow, pairs, fast]`` in its working domain.

        One native launch per cross family per replay group of the frame. Each
        launch reads the group's shared rows in place; every receive cell names
        its row range (the receive element's whole pair block, all
        transmitters) and the per-row link table skips the rows this family
        does not carry.
        """

        from .synthesis.assembly import CrossRouting

        session = self.session
        member = self.member
        num_tx, num_rx = member.num_tx, member.num_rx
        pairs = num_tx * num_rx
        samples = self.fast
        width = samples if not session.adc_sampled else 1
        values: list[torch.Tensor] = []
        targets: list[np.ndarray] = []
        device = observations[0].composed.total_delay_s.device
        tables = {family.name: self._link_table(family, frame_start_s, device) for family in self.families}
        position = 0
        while position < len(observations):
            batch = observations[position].batch
            assert batch is not None
            stop = position
            while stop < len(observations) and observations[stop].batch is batch:
                stop += 1
            group = observations[position:stop]
            composer = self._composer(group[0])
            # Each observation's receive cells: slot, receive element, first sample.
            numbers = np.arange(position, stop)
            if not session.sampled:
                slot = np.arange(self.slot_count)
                first = np.zeros(self.slot_count, dtype=np.int64)
                owner = np.zeros(self.slot_count, dtype=np.int64)
            elif session.adc_sampled:
                slot, first, owner = numbers // samples, numbers % samples, np.arange(len(group))
            else:
                slot, first, owner = numbers, np.zeros(len(group), dtype=np.int64), np.arange(len(group))
            group_slot = np.asarray([observation.slot for observation in group], dtype=np.int64)[owner]
            cell_slot = np.repeat(slot, num_rx)
            cell_first = np.repeat(first, num_rx)
            cell_group_slot = np.repeat(group_slot, num_rx)
            receiver = np.tile(np.arange(num_rx), len(slot))
            offsets = composer.pair_offsets_host
            block = (member.rx_offset + receiver) * self.num_tx_total
            base = cell_group_slot * composer.path_count
            segment_rows = np.stack((base + offsets[block], base + offsets[block + self.num_tx_total]), axis=1)
            schedule = np.stack(
                (
                    self.slot_start[cell_slot],
                    self.first_sample[cell_slot] + cell_first * self.period,
                    self.lo[cell_slot],
                ),
                axis=1,
            )
            counts = segment_rows[:, 1] - segment_rows[:, 0]
            cell_of_entry = np.repeat(np.arange(len(counts)), counts)
            row_of_entry = np.repeat(segment_rows[:, 0] - np.cumsum(counts) + counts, counts) + np.arange(
                int(counts.sum())
            )
            order = np.argsort(row_of_entry, kind="stable")
            row_count = batch.slots * composer.path_count
            row_offsets = np.concatenate(([0], np.cumsum(np.bincount(row_of_entry, minlength=row_count))))
            host = (
                segment_rows.ravel(),
                row_offsets,
                cell_of_entry[order],
                np.ascontiguousarray(schedule).view(np.int64).ravel(),
            )
            routing_table = _uploaded(session, np.concatenate(host), device)
            parts = torch.split(routing_table, [len(table) for table in host])
            rows = batch.rows
            heard: torch.Tensor | None = None
            for family in self.families:
                selected = self._heard_rows(composer, family)
                routing = CrossRouting(
                    row_link=selected.row_link,
                    row_tx=selected.row_tx,
                    link_table=tables[family.name],
                    aux=family.aux,
                    segment_rows=parts[0].reshape(-1, 2),
                    segment_schedule=parts[3].view(torch.float64).reshape(-1, 3),
                    row_segment_offsets=parts[1],
                    row_segments=parts[2],
                    rows_per_slot=composer.path_count,
                    num_samples=width,
                    sample_period_s=self.period,
                )
                value = family.synthesize(
                    rows.complex_transfer_ref, rows.total_delay_s, rows.total_delay_residual_s, rows.row_valid, routing
                )
                heard = value if heard is None else heard + value
            assert heard is not None
            values.append(heard)
            slow, transmitter = cell_slot // num_tx, cell_slot % num_tx
            targets.append(((slow * pairs + receiver * num_tx + transmitter) * samples + cell_first) // width)
            position = stop
        destination = np.concatenate(targets)
        order = np.argsort(destination, kind="stable")
        cells = self.slow * pairs * samples // width
        if not np.array_equal(destination[order], np.arange(cells)):
            raise ValueError("a receiving radar's observations must cover every slot, pair and sample exactly once")
        cube = torch.cat(values).index_select(0, _uploaded(session, order, device))
        return cube.reshape(self.slow, pairs, samples)


__all__ = ["Emitter", "Network", "NetworkPaths"]
