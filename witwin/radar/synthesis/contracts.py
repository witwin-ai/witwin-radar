"""The input contract every waveform synthesis kernel consumes, and the
waveform descriptions that go with it.

This module is pure and CPU-testable on purpose: the unit conversions between
the radar config's engineering units and SI are exactly the kind of thing that
is wrong once and then wrong everywhere, and they should not require a GPU to
check. The same is true of the provenance rules below, which decide whether a
weight and a waveform spec may be used together at all.

Two contracts live here and they are not the same statement:

* :class:`~witwin.radar.paths.contracts.RadarPathBatch` is what the two-way
  composer PRODUCED.
* :class:`SynthesisPathBatch` is what a waveform kernel is ALLOWED TO ASSUME
  about a weight.

The difference is provenance. Every double-count hazard the Phase-6 physics
survey found is a combination of a weight and a spec that nobody validates
against each other: a Channel coefficient already carries
``exp(-j 2 pi f_ref tau_rt)``, ``lambda/(4 pi d)`` per leg, and
``sqrt(tx_power)``, so a kernel that applies any of them again is silently
wrong by a factor nobody notices. Recording that on the batch and validating
the spec against it at construction turns six documented hazards into four
impossible states, which is the difference between a rule and a comment.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Protocol, runtime_checkable

import torch

from ..paths.contracts import JOIN_MODES, JoinMode, RadarPathBatch, RadarPathTopology


#: Exact SI definition, in metres per second. Named here because the FMCW spec
#: derives its unambiguous-velocity bound from a wavelength and must agree with
#: ``Radar.max_doppler`` to the last bit.
SPEED_OF_LIGHT_M_PER_S = 299792458.0


@dataclass(frozen=True, slots=True)
class FmcwBeatSpec:
    """One chirp frame's sampling grid and ramp, in SI units.

    The carrier phase ``2 * pi * f_c * tau`` has two legitimate homes, and the
    two carrier parameters together say which one. Exactly one of them is
    nonzero:

    * ``carrier_hz = fc``, ``carrier_rate_hz = 0``  -  the kernel owns the whole
      carrier phase. This reproduces the Dirichlet solver's phase structure
      exactly, which is what the equivalence test uses.
    * ``carrier_hz = 0``, ``carrier_rate_hz = fc``  -  the production path for
      Channel-sourced weights, where the absolute carrier phase already sits
      inside the natively computed coefficient. That placement is more accurate,
      because the coefficient's phase was formed against a float64 delay inside
      the native kernel, while a float32 ``tau`` re-multiplied by 77 GHz loses
      roughly 2e-4 rad at 2 m and 1e-2 rad at 100 m.

    ``carrier_rate_hz`` is not a second copy of the carrier and not a tuning
    knob. A Channel coefficient is frozen at the per-frame ``tau_rt``, so the
    carrier phase it holds does NOT advance across chirps. Without this term the
    slow-time phase walk keeps only ``slope * (t_start - tau + t_m) * tau_rate``
    and understates intra-frame Doppler by 21x to 215x across the fast-time axis
    - silently, because the primal still looks like a plausible radar cube.
    ``carrier_rate_hz`` applies the carrier to the delay CHANGE
    ``(tau - tau_rt)`` only, which is exactly the missing term.

    Setting both to ``fc`` double counts the carrier and is refused. Both
    supported settings are exact; neither is a fallback for the other.

    ``num_tx`` and ``num_rx`` describe the TDM-MIMO array. They belong on the
    spec rather than on the batch because they are a property of the WAVEFORM's
    time structure: TDM fires the transmitters sequentially, so the slow-time
    coordinate of a sensor pair is its slot ``chirp * num_tx + tx``, not the
    chirp index. ``num_tx = 1`` is the degenerate single-transmitter case where
    slot and chirp coincide.
    """

    #: The beat kernel has no ``lambda / (4 pi d)`` term at all: free-space
    #: spreading is Channel transport's, per leg, once. This is a statement
    #: about the kernel and not a setting, so it is a class attribute rather
    #: than a field nobody may change.
    applies_spreading: ClassVar[bool] = False

    num_samples: int
    num_chirps: int
    sample_period_s: float
    chirp_period_s: float
    slope_hz_per_s: float
    t_start_s: float
    reference_frequency_hz: float
    carrier_hz: float = 0.0
    carrier_rate_hz: float = 0.0
    num_tx: int = 1
    num_rx: int = 1

    def __post_init__(self) -> None:
        if self.num_samples < 1:
            raise ValueError("num_samples must be positive")
        if self.num_chirps < 1:
            raise ValueError("num_chirps must be positive")
        if self.sample_period_s <= 0.0:
            raise ValueError("sample_period_s must be positive")
        if self.chirp_period_s <= 0.0:
            raise ValueError("chirp_period_s must be positive")
        if not self.reference_frequency_hz > 0.0:
            raise ValueError(
                "reference_frequency_hz must be positive; it is the frequency "
                "the weight this spec will consume was evaluated at, and "
                "require_compatible refuses a mismatch"
            )
        if self.num_tx < 1:
            raise ValueError("num_tx must be positive")
        if self.num_rx < 1:
            raise ValueError("num_rx must be positive")
        if self.carrier_hz != 0.0 and self.carrier_rate_hz != 0.0:
            raise ValueError(
                "carrier_hz and carrier_rate_hz name the same carrier in two "
                "different homes; setting both double counts it. Use "
                "carrier_hz=fc with carrier_rate_hz=0 when the kernel owns the "
                "carrier phase, or carrier_hz=0 with carrier_rate_hz=fc when a "
                "Channel-sourced weight already carries it."
            )

    @classmethod
    def from_radar_config(cls, config, *, carrier_hz: float = 0.0) -> "FmcwBeatSpec":
        """Convert a :class:`witwin.radar.RadarConfig` into SI units.

        The config carries engineering units: ``sample_rate`` in kSPS,
        ``idle_time`` / ``ramp_end_time`` / ``adc_start_time`` in microseconds,
        and ``slope`` in MHz per microsecond, which is 1e12 Hz per second.

        ``carrier_rate_hz`` is derived, not passed: it is ``config.fc`` on the
        production path (``carrier_hz = 0``, weight owns the carrier) and zero
        when the caller puts the carrier in the kernel. Deriving it here is what
        makes the default configuration Doppler-correct; a caller that overrides
        ``carrier_hz`` through ``dataclasses.replace`` will hit the both-nonzero
        error rather than silently losing the rate term.
        """

        carrier = float(carrier_hz)
        return cls(
            num_samples=int(config.adc_samples),
            num_chirps=int(config.chirp_per_frame),
            sample_period_s=1.0 / (float(config.sample_rate) * 1e3),
            chirp_period_s=(float(config.idle_time) + float(config.ramp_end_time))
            * 1e-6,
            slope_hz_per_s=float(config.slope) * 1e12,
            t_start_s=float(config.adc_start_time) * 1e-6,
            reference_frequency_hz=float(config.fc),
            carrier_hz=carrier,
            carrier_rate_hz=0.0 if carrier != 0.0 else float(config.fc),
            num_tx=int(config.num_tx),
            num_rx=int(config.num_rx),
        )

    @property
    def sample_rate_hz(self) -> float:
        return 1.0 / self.sample_period_s

    @property
    def sensor_pair_count(self) -> int:
        """The TDM-MIMO virtual array size the pair partition must span."""

        return self.num_tx * self.num_rx

    @property
    def wavelength_m(self) -> float:
        return SPEED_OF_LIGHT_M_PER_S / self.reference_frequency_hz

    @property
    def slot_period_s(self) -> float:
        """Slow-time spacing between two chirps of the SAME transmitter.

        With ``num_tx`` transmitters sharing the frame in TDM, a given
        transmitter revisits its slot once every ``num_tx`` chirp periods. This
        is the period the Doppler FFT actually samples at, and it is why TDM
        costs a factor ``num_tx`` of unambiguous velocity.
        """

        return self.chirp_period_s * self.num_tx

    @property
    def max_unambiguous_speed_mps(self) -> float:
        """``lambda / (4 * T_chirp * num_tx)``, the aliasing bound on ``|v_r|``.

        Half a wavelength of two-way path change per slow-time sample is half a
        cycle of Doppler phase; beyond it the sign of the velocity is not
        recoverable.
        """

        return self.wavelength_m / (4.0 * self.slot_period_s)

    def beat_frequency_hz(self, round_trip_delay_s: float) -> float:
        """``f_beat = slope * tau``, with ``tau`` the ROUND-TRIP delay.

        There is no factor of two here. A two-leg round trip already knows its
        own total delay; doubling it would be a monostatic assumption that this
        contract does not make.
        """

        return self.slope_hz_per_s * float(round_trip_delay_s)

    def beat_bin(self, round_trip_delay_s: float) -> float:
        """Fractional FFT bin of the beat tone over ``num_samples``."""

        return (
            self.beat_frequency_hz(round_trip_delay_s)
            * self.num_samples
            / self.sample_rate_hz
        )


class SlowTimeMode(str, Enum):
    """How the weight and the slow-time axis divide the Doppler phase.

    These two are mutually exclusive, and they are ONE enum rather than two
    independently-settable fields because the combination "the caller refreshed
    the weight at every slot AND the kernel still applies a carrier rate"
    applies Doppler twice and looks like a plausible radar cube while doing it.
    Phase 6 always uses the frozen mode; Phase 7 owns dynamics and is the reason
    the refreshed mode is named now.
    """

    #: The weight was computed once, at the frame's ``tau_rt``, and does not
    #: walk across chirps/symbols/pulses. The slow-time carrier phase is the
    #: waveform kernel's job.
    FROZEN_WEIGHT_WITH_CARRIER_RATE = "frozen_weight_with_carrier_rate"

    #: The weight is re-evaluated at every slow-time slot, so it already walked.
    #: A carrier-rate term on top of it would double the Doppler.
    REFRESHED_WEIGHT_NO_RATE = "refreshed_weight_no_rate"


@runtime_checkable
class WaveformSpecProtocol(Protocol):
    """Exactly the attributes :func:`require_compatible` reads.

    Declared as a Protocol rather than a base class because the three waveform
    specs have nothing else in common: an FMCW ramp, an OFDM subcarrier grid,
    and a pulse envelope share no fields. What they DO share is a position on
    the four questions that decide whether a weight may be handed to them.

    ``tx_power_mode`` is deliberately absent: it belongs to the sensor-weight
    owner rather than to a waveform, and :func:`require_compatible` reads it
    only when a spec chooses to declare it.
    """

    #: Absolute reference-frequency carrier the KERNEL applies, in Hz. Zero
    #: means the kernel applies none because the weight already carries it.
    carrier_hz: float

    #: Reference frequency applied to the delay CHANGE only, in Hz. Zero means
    #: the kernel applies none.
    carrier_rate_hz: float

    #: The frequency the weight was evaluated at. Must equal the batch's.
    reference_frequency_hz: float

    #: Whether the waveform owner multiplies by ``lambda/(4 pi d)`` itself.
    applies_spreading: bool


def _require_tensor(
    name: str,
    value: object,
    *,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}")
    if value.dtype != dtype:
        raise TypeError(f"{name} must use {dtype}, got {value.dtype}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
    if not value.is_contiguous():
        raise ValueError(
            f"{name} must be contiguous; a synthesis kernel indexes it linearly "
            "and the contract will not hide a copy on the hot path"
        )
    if value.device != device:
        raise ValueError(
            f"{name} is on {value.device} but the batch is on {device}; a "
            "synthesis batch is single-device by contract"
        )
    return value


@dataclass(frozen=True, slots=True, eq=False)
class SynthesisPathBatch:
    """What a waveform synthesis kernel may assume about a set of path rows.

    Geometry is ``total_delay_s`` (the ROUND-TRIP delay ``tau_rt``, in seconds,
    never a one-way distance) and ``delay_rate`` (``d(tau_rt)/dt``,
    dimensionless). A kernel consumes those two and nothing else about the
    geometry: it may never reconstruct a distance, and it may never re-apply a
    ``1/(4 pi d)``.

    ``complex_transfer_ref`` is in the CHANNEL phasor convention,
    ``exp(-j k d)`` under ``exp(+j 2 pi f t)`` time dependence, evaluated at
    ``reference_frequency_hz``. It is NOT a beat weight; converting to one is
    the FMCW owner's single call site.

    The four provenance fields are the whole reason this type exists. They say
    what is ALREADY inside the weight, so :func:`require_compatible` can refuse
    a spec that would apply it a second time. They are set by the two
    classmethods below rather than by a caller, because a caller that could
    assert its own provenance could assert the convenient one.

    Validation is host-only: shapes, dtypes, contiguity, device, and flags. It
    reads no tensor VALUE, so constructing this contract costs no
    device-to-host transfer and no synchronization. In particular
    ``pair_offsets[0] == 0`` and ``pair_offsets[-1] == path_count`` are a
    documented producer obligation, exactly as in ``RadarPathBatch``, not a
    device read.
    """

    # ---- cardinality (host ints, already published by the compact contract) --
    sensor_pair_count: int
    path_count: int

    # ---- row -> segment partition -------------------------------------------
    sensor_pair_index: torch.Tensor
    pair_offsets: torch.Tensor

    # ---- geometry ------------------------------------------------------------
    total_delay_s: torch.Tensor
    delay_rate: torch.Tensor | None

    # ---- transfer ------------------------------------------------------------
    complex_transfer_ref: torch.Tensor
    reference_frequency_hz: float
    frequency_response: torch.Tensor | None
    frequency_offsets_hz: torch.Tensor | None

    # ---- identity ------------------------------------------------------------
    topology: RadarPathTopology
    row_valid: torch.Tensor | None
    join_mode: JoinMode

    # ---- provenance ----------------------------------------------------------
    weight_includes_reference_phase: bool
    weight_includes_spreading: bool
    weight_includes_tx_power: bool
    slow_time_mode: SlowTimeMode

    def __post_init__(self) -> None:
        if self.join_mode not in JOIN_MODES:
            raise ValueError(
                f"join_mode must be one of {sorted(JOIN_MODES)}, got "
                f"{self.join_mode!r}"
            )
        if not isinstance(self.slow_time_mode, SlowTimeMode):
            raise TypeError(
                "slow_time_mode must be a SlowTimeMode member, got "
                f"{self.slow_time_mode!r}"
            )
        if type(self.sensor_pair_count) is not int or self.sensor_pair_count < 1:
            raise ValueError("sensor_pair_count must be a positive int")
        if type(self.path_count) is not int or self.path_count < 0:
            raise ValueError("path_count must be a non-negative int")
        for name in (
            "weight_includes_reference_phase",
            "weight_includes_spreading",
            "weight_includes_tx_power",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool")
        if (
            type(self.reference_frequency_hz) is not float
            or not self.reference_frequency_hz > 0.0
        ):
            raise ValueError(
                "reference_frequency_hz must be a positive float; it is the "
                "frequency the weight was evaluated at, not an optional label"
            )

        device = self.total_delay_s.device
        rows = (self.path_count,)
        _require_tensor(
            "total_delay_s",
            self.total_delay_s,
            dtype=torch.float32,
            shape=rows,
            device=device,
        )
        _require_tensor(
            "sensor_pair_index",
            self.sensor_pair_index,
            dtype=torch.int64,
            shape=rows,
            device=device,
        )
        _require_tensor(
            "pair_offsets",
            self.pair_offsets,
            dtype=torch.int64,
            shape=(self.sensor_pair_count + 1,),
            device=device,
        )
        _require_tensor(
            "complex_transfer_ref",
            self.complex_transfer_ref,
            dtype=torch.complex64,
            shape=rows,
            device=device,
        )
        if self.delay_rate is not None:
            _require_tensor(
                "delay_rate",
                self.delay_rate,
                dtype=torch.float32,
                shape=rows,
                device=device,
            )
        if self.row_valid is not None:
            _require_tensor(
                "row_valid",
                self.row_valid,
                dtype=torch.bool,
                shape=rows,
                device=device,
            )
        if (self.frequency_response is None) != (self.frequency_offsets_hz is None):
            raise ValueError(
                "frequency_response and frequency_offsets_hz are one statement "
                "and must be supplied together; a response without its "
                "frequency grid says nothing"
            )
        if self.frequency_response is not None:
            if self.frequency_offsets_hz.dim() != 1:
                raise ValueError("frequency_offsets_hz must have shape (F,)")
            bands = (self.path_count, int(self.frequency_offsets_hz.shape[0]))
            _require_tensor(
                "frequency_response",
                self.frequency_response,
                dtype=torch.complex64,
                shape=bands,
                device=device,
            )
            _require_tensor(
                "frequency_offsets_hz",
                self.frequency_offsets_hz,
                dtype=torch.float32,
                shape=(bands[1],),
                device=device,
            )
        if self.topology.row_count != self.path_count:
            raise ValueError("topology must have exactly path_count rows")

    @property
    def device(self) -> torch.device:
        return self.total_delay_s.device

    @classmethod
    def from_radar_paths(
        cls,
        paths: RadarPathBatch,
        *,
        slow_time_mode: SlowTimeMode,
    ) -> "SynthesisPathBatch":
        """Wrap a composed round-trip batch, zero-copy, with Channel provenance.

        Every tensor passes through by reference. Nothing is cloned, made
        contiguous, or moved: row identity, row order, storage aliasing, stride,
        dtype, device, and gradient state are all preserved, and a test asserts
        object identity rather than value equality.

        The three provenance booleans are Channel's published contract, not a
        caller's opinion, which is why they are written here:

        * ``coefficient_reference = "includes_reference_frequency_phase"``
        * ``FREE_SPACE_AMPLITUDE = "sqrt(tx_power)*wavelength/(4*pi*distance)"``

        ``slow_time_mode`` is the one thing the caller must say, because only
        the caller knows whether it froze the weight for the frame or refreshes
        it per slot. It has no default: defaulting it would make the Phase-7
        collision a silent wrong answer instead of a refusal.
        """

        if not isinstance(paths, RadarPathBatch):
            raise TypeError(
                f"from_radar_paths needs a RadarPathBatch, got {type(paths).__name__}"
            )
        return cls(
            sensor_pair_count=paths.sensor_pair_count,
            path_count=paths.path_count,
            sensor_pair_index=paths.sensor_pair_index,
            pair_offsets=paths.pair_offsets,
            total_delay_s=paths.total_delay_s,
            delay_rate=paths.delay_rate,
            complex_transfer_ref=paths.complex_transfer_ref,
            reference_frequency_hz=float(paths.reference_frequency_hz),
            frequency_response=None,
            frequency_offsets_hz=None,
            topology=paths.topology,
            row_valid=paths.row_valid,
            join_mode=paths.join_mode,
            weight_includes_reference_phase=True,
            weight_includes_spreading=True,
            weight_includes_tx_power=True,
            slow_time_mode=slow_time_mode,
        )

    @classmethod
    def from_real_amplitudes(
        cls,
        one_way_distances_m: torch.Tensor,
        amplitudes: torch.Tensor,
        *,
        pair_offsets: torch.Tensor,
        topology: RadarPathTopology,
        c0: float,
        reference_frequency_hz: float,
        delay_rate: torch.Tensor | None = None,
        join_mode: JoinMode = "multipath",
    ) -> "SynthesisPathBatch":
        """Embed the legacy real-amplitude path as the complex special case.

        This is the whole of the real-compatibility criterion: the existing
        Radar baseline is not a second code path, it is ``C = amp + 0j`` with
        the monostatic delay written down once, here, in Python, explicitly.

        Two traps are encoded rather than commented:

        * ``torch.complex(amplitudes, zeros)``, never
          ``complex(abs(amplitudes), 0)``. The SIGN of a legacy amplitude is the
          only phase a real amplitude can carry - it is the reflection flip -
          and discarding it is a silent 180-degree error that no magnitude plot
          shows.
        * ``weight_includes_reference_phase = False``. A real amplitude carries
          no phase at all, so rule R2 forces the spec to own the carrier, which
          is exactly the legacy Dirichlet phase structure. The complex-weight
          switch and the carrier-home switch are therefore the same act.

        ``one_way_distances_m`` is doubled here because the legacy input is a
        one-way distance and every contract downstream of this one speaks
        round-trip delay. Making that conversion visible at the boundary is the
        point: the legacy kernel did it internally, where a caller that already
        had a round-trip delay could not tell.
        """

        if one_way_distances_m.shape != amplitudes.shape:
            raise ValueError(
                "one_way_distances_m and amplitudes must have the same shape"
            )
        if amplitudes.dtype != torch.float32:
            raise TypeError(
                f"amplitudes must be float32, got {amplitudes.dtype}"
            )
        if not c0 > 0.0:
            raise ValueError("c0 must be positive")
        total_delay_s = one_way_distances_m * (2.0 / float(c0))
        complex_transfer_ref = torch.complex(
            amplitudes, torch.zeros_like(amplitudes)
        ).to(torch.complex64)
        path_count = int(amplitudes.shape[0])
        sensor_pair_count = int(pair_offsets.shape[0]) - 1
        rows = torch.arange(
            path_count, device=pair_offsets.device, dtype=torch.int64
        )
        sensor_pair_index = torch.bucketize(rows, pair_offsets[1:], right=True)
        return cls(
            sensor_pair_count=sensor_pair_count,
            path_count=path_count,
            sensor_pair_index=sensor_pair_index.contiguous(),
            pair_offsets=pair_offsets.contiguous(),
            total_delay_s=total_delay_s.contiguous(),
            delay_rate=None if delay_rate is None else delay_rate.contiguous(),
            complex_transfer_ref=complex_transfer_ref.contiguous(),
            reference_frequency_hz=float(reference_frequency_hz),
            frequency_response=None,
            frequency_offsets_hz=None,
            topology=topology,
            row_valid=None,
            join_mode=join_mode,
            weight_includes_reference_phase=False,
            weight_includes_spreading=True,
            weight_includes_tx_power=True,
            slow_time_mode=SlowTimeMode.FROZEN_WEIGHT_WITH_CARRIER_RATE,
        )


def require_compatible(batch: SynthesisPathBatch, spec: WaveformSpecProtocol) -> None:
    """Refuse any weight/spec pair that would count a factor twice.

    Called by every waveform entry point before any kernel launch. Each rule
    below names the hazard it prevents, because the failure it prevents is
    always a plausible-looking number rather than a crash, and an error message
    that only says "invalid configuration" would send the reader looking for a
    bug in the physics.

    One deviation from the Phase-6 design document is recorded here rather than
    buried: the design states R3 as "the frozen mode requires
    ``carrier_rate_hz == reference_frequency_hz``", full stop. That is
    unsatisfiable for the legacy real-amplitude batch, which is frozen AND has
    ``weight_includes_reference_phase = False``: R2 then forces
    ``carrier_hz = f_ref``, and a spec with both carrier parameters nonzero
    double counts the carrier and is refused by the spec itself. The physics
    resolves it - differentiating the FMCW phase with respect to slow time gives
    the same bracket for ``(f_ref, 0)`` and ``(0, f_ref)``, because a
    kernel-owned carrier multiplies the FULL ``tau(t)`` and therefore already
    walks. So R3 is enforced as "the delay change has exactly one owner, chosen
    by the provenance": the weight's carrier home decides which of the two
    parameters must equal ``f_ref``.
    """

    if not isinstance(batch, SynthesisPathBatch):
        raise TypeError(
            f"require_compatible needs a SynthesisPathBatch, got {type(batch).__name__}"
        )
    for attribute in (
        "carrier_hz",
        "carrier_rate_hz",
        "reference_frequency_hz",
        "applies_spreading",
    ):
        if not hasattr(spec, attribute):
            raise TypeError(
                f"{type(spec).__name__} does not declare {attribute!r}, so it "
                "cannot be checked against a weight's provenance; a waveform "
                "spec must satisfy WaveformSpecProtocol"
            )

    carrier_hz = float(spec.carrier_hz)
    carrier_rate_hz = float(spec.carrier_rate_hz)
    f_ref = batch.reference_frequency_hz

    # R1 - hazard H1: the Channel coefficient already holds
    # exp(-j 2 pi f_ref tau_rt).
    if batch.weight_includes_reference_phase and carrier_hz != 0.0:
        raise ValueError(
            "double-counted carrier phase: the weight already carries "
            "exp(-j*2*pi*f_ref*tau_rt) (coefficient_reference = "
            "'includes_reference_frequency_phase'), so carrier_hz must be 0; "
            f"got carrier_hz={carrier_hz}"
        )

    # R2 - the mirror image: nobody owns the absolute carrier at all.
    if (
        not batch.weight_includes_reference_phase
        and carrier_hz == 0.0
        and carrier_rate_hz == 0.0
    ):
        raise ValueError(
            "missing carrier phase: this weight carries no reference-frequency "
            "phase, and neither carrier_hz nor carrier_rate_hz is set, so the "
            "absolute carrier has no owner and the synthesized IQ would have no "
            "range phase at all"
        )

    # R3 - hazard H4, first half: a frozen weight does not walk across slow
    # time, so the delay CHANGE needs exactly one owner and the provenance says
    # which parameter it is.
    if batch.slow_time_mode is SlowTimeMode.FROZEN_WEIGHT_WITH_CARRIER_RATE:
        if batch.weight_includes_reference_phase:
            if carrier_rate_hz != f_ref:
                raise ValueError(
                    "understated Doppler: the weight is frozen at the frame's "
                    "tau_rt and carries the reference phase, so the delay "
                    f"change has no other home; carrier_rate_hz must be {f_ref}, "
                    f"got {carrier_rate_hz}. Dropping it understates intra-frame "
                    "Doppler by one to two orders of magnitude while still "
                    "producing a plausible-looking cube"
                )
        elif carrier_hz != f_ref:
            raise ValueError(
                "understated Doppler: the weight carries no reference phase, so "
                "the kernel must own the absolute carrier and thereby the delay "
                f"change; carrier_hz must be {f_ref}, got {carrier_hz}"
            )

    # R4 - hazard H4, second half: a refreshed weight already walked.
    if batch.slow_time_mode is SlowTimeMode.REFRESHED_WEIGHT_NO_RATE:
        if carrier_rate_hz != 0.0:
            raise ValueError(
                "double-counted Doppler: this weight is re-evaluated at every "
                "slow-time slot, so it already carries the delay change; "
                f"carrier_rate_hz must be 0, got {carrier_rate_hz}"
            )
        if batch.delay_rate is not None:
            raise ValueError(
                "double-counted Doppler: a refreshed weight walks by itself, so "
                "the batch must not also publish delay_rate for a kernel to "
                "apply"
            )

    # R5 - hazard F1: free-space spreading is Channel transport's, per leg,
    # once.
    if batch.weight_includes_spreading and bool(spec.applies_spreading):
        raise ValueError(
            "double-counted free-space spreading: the weight already contains "
            "wavelength/(4*pi*distance) per leg (FREE_SPACE_AMPLITUDE), so the "
            "waveform owner must not apply it again; set applies_spreading=False"
        )

    # R6 - hazard F4: TX power reaches physics through powers_w and nowhere
    # else. tx_power_mode belongs to the sensor-weight owner, so it is checked
    # only when a spec declares it.
    tx_power_mode = getattr(spec, "tx_power_mode", None)
    if (
        batch.weight_includes_tx_power
        and tx_power_mode is not None
        and tx_power_mode != "already_in_weight"
    ):
        raise ValueError(
            "double-counted transmit power: the weight already contains "
            "sqrt(tx_power) from the source endpoint's powers_w, so the sensor "
            "weight owner must run with tx_power_mode='already_in_weight'; got "
            f"{tx_power_mode!r}"
        )

    # R7 - the weight was evaluated at one frequency and means nothing at
    # another. This mirrors Channel's own request/compile frequency rule.
    if float(spec.reference_frequency_hz) != f_ref:
        raise ValueError(
            "reference frequency mismatch: the weight was evaluated at "
            f"{f_ref} Hz but the waveform spec declares "
            f"{float(spec.reference_frequency_hz)} Hz; a narrowband coefficient "
            "is not transferable between reference frequencies"
        )

    # R8 - wideband material response is Phase 8.
    if batch.frequency_response is not None:
        raise ValueError(
            "wideband material response is Phase 8 work: this contract declares "
            "frequency_response/frequency_offsets_hz so that the Phase-6 "
            "narrowband assumption is explicit, and refuses a non-None value so "
            "that it cannot be silently ignored by a kernel that only knows the "
            "narrowband law H(f_ref+df) = C(f_ref)*exp(-j*2*pi*df*delay_s)"
        )


__all__ = [
    "FmcwBeatSpec",
    "SlowTimeMode",
    "SynthesisPathBatch",
    "WaveformSpecProtocol",
    "require_compatible",
]
