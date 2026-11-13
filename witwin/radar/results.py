"""The records the instrument half publishes: :class:`Result` and :class:`Frame`.

Data only. A session produces one :class:`SceneFrame` per frame and
:func:`assemble` stacks them into the published :class:`Result`; the frame
facade delegates every product to :mod:`witwin.radar.processing`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    import numpy as np

    from .paths import RadarPathBatch
    from .processing import ArrayGeometry, PointCloud, ProcessingAxes, ProcessingCube, RangeDopplerMap, RangeProfile
    from .propagation import EpochFrame, RadarPropagationLegs
    from .sensors import SensorArraySpec
    from .synthesis.assembly import FmcwSpec, OfdmSpec, PulsedSpec, SynthesisResult

#: The axis names of the published multi-frame cube. The last two are the
#: waveform's own slow and fast axes and are filled in from the synthesis
#: result, so an OFDM run publishes ``("frame", "tx", "rx", "symbol",
#: "subcarrier")`` without this module knowing what a symbol is.
SIMULATION_CUBE_LEADING_AXES = ("frame", "tx", "rx")


@dataclass(frozen=True, slots=True, eq=False)
class Result:
    """What one :meth:`witwin.radar.Radar.simulate` call produced.

    Data only. The driver builds it through :meth:`from_frames`, which is this
    repository's standing shape for a result: the producer knows what it made,
    and a consumer never has to infer a phasor convention from a method name.

    ``cube`` is ``[frame, TX, RX, slow, fast]`` and is the product. It is one
    stacked tensor rather than a list because a frame sequence is what every
    downstream consumer - a range-Doppler map, a tracker, a loss - indexes, and
    the stack is a single differentiable op outside the frame loop.

    The four ``last_*`` members are the LAST frame's typed state, and this
    record is their only home: a radar holds no run state, so there is no
    second copy to disagree with these and no way for a call that raised part
    way through to leave one behind describing a world it never simulated.
    They describe one frame, not the sequence: a compiled scene and a leg pair
    are per-epoch and per-frame objects, and stacking them would either
    misrepresent the epochs or retain every frame's device memory for the life
    of the result. Keeping the last one is the smallest retention that answers
    "what did the closing frame see".

    ``path_set_complete`` and ``motion_sampling_exhaustive`` are TWO
    statements, and a consumer that needs to know how much to trust a run
    needs both. The first says no path birth can have been missed - because
    every observation was evaluated, or because the candidate family was
    certified complete for all time. The second says no observation's transport
    was interpolated between probes. Adaptive sampling routinely gives the
    first and not the second, which is exactly the trade it exists to make;
    collapsing them into one flag would report a proven-complete adaptive run
    as if it might have lost a path.

    RETENTION, stated because these members are a real tensor lifetime. The
    ``last_*`` members alias the frame's own batches, so holding this result
    holds that frame's device tensors - and, when ``ad_mode`` asked for a graph,
    that frame's autograd graph. None of them holds a tape: an autograd context
    or a ``saved_tensors`` tuple in any of these fields would be a data record
    turned into a handle on somebody else's memory, and
    ``tests/test_tape_containment.py`` walks all four to keep it that way.
    """

    cube: torch.Tensor
    times_s: tuple[float, ...]
    kind: str
    #: The cube's named axes, outermost first. ``axes`` is the metadata record
    #: below, not this.
    axis_names: tuple[str, ...]
    #: Everything a processing stage reads: the SI range and velocity axes, the
    #: phasor convention, the Doppler sign and the array layout. Built once,
    #: here, from the waveform spec and the array that produced the cube, so no
    #: caller assembles it from a re-viewed synthesis result.
    axes: ProcessingAxes
    phasor: str
    time_dependence: str
    reference_frequency_hz: float
    epochs: tuple[int, ...]
    rediscovery_reasons: tuple[str | None, ...]
    compile_count: int
    discovery_count: int
    #: The closing frame's Core ``SceneSnapshot``.
    last_snapshot: object
    #: The closing frame's Channel ``CompiledScene``.
    last_compiled_scene: object
    last_propagation: RadarPropagationLegs
    last_radar_paths: RadarPathBatch
    #: Every observation instant of every frame, ``float64[observations]``
    #: per frame. An ADC-refreshed frame schedules one per chirp,
    #: transmitter and sample, so this is an array rather than a tuple of
    #: Python floats: materialising those costs more than the frame does.
    sample_times_s: tuple[np.ndarray, ...]
    path_set_complete: bool
    motion_sampling_exhaustive: bool
    motion_sampling: str
    #: The synthesis result's own output domain.
    output_domain: str
    adaptive_diagnostics: tuple[dict[str, Any], ...]
    #: When the closing frame's synthesis observed the world: ``frame``,
    #: ``slow_time``, ``fast_time`` or ``adaptive``, as the synthesis owner
    #: published it (``SynthesisResult.observation_schedule``). It is what ran,
    #: which can be coarser than the requested ``Motion``: an OFDM CFR symbol
    #: has no fast-time grid of its own.
    observation_schedule: str = "frame"

    def __post_init__(self) -> None:
        if self.cube.dim() != len(self.axis_names):
            raise ValueError(
                f"a {self.kind} simulation cube has {len(self.axis_names)} axes "
                f"{self.axis_names}, got shape {tuple(self.cube.shape)}"
            )
        frames = int(self.cube.shape[0])
        for name in ("times_s", "epochs", "rediscovery_reasons"):
            values = getattr(self, name)
            if len(values) != frames:
                raise ValueError(
                    f"{name} carries {len(values)} entries for {frames} frames; "
                    "the per-frame records and the cube's frame axis are the "
                    "same sequence"
                )

    @property
    def frame_count(self) -> int:
        return int(self.cube.shape[0])

    def frame(self, index: int = 0) -> Frame:
        """One frame of this result, with the metadata a processing stage reads.

        A view, not a copy: the cube is indexed, not cloned. The returned
        record's methods are facades over :mod:`witwin.radar.processing` and
        compute nothing themselves.
        """

        return Frame(
            cube=self.cube[index],
            axes=self.axes,
            time_s=float(self.times_s[index]),
            index=int(index),
            synthesis=self.frame_synthesis(index),
        )

    def frame_synthesis(self, frame_index: int = 0) -> SynthesisResult:
        """Expose a simulated frame and its recorded axes without resynthesis."""
        from .synthesis.assembly import SynthesisResult

        frame = self.cube[frame_index]
        tx, rx, slow, fast = frame.shape
        packed = frame.permute(2, 1, 0, 3).reshape(slow, rx * tx, fast)
        return SynthesisResult(
            cube=packed,
            kind=self.kind,
            axes=(self.axis_names[-2], "sensor_pair", self.axis_names[-1]),
            phasor=self.phasor,
            time_dependence=self.time_dependence,
            reference_frequency_hz=self.reference_frequency_hz,
            output_domain=self.output_domain,
            observation_schedule=self.observation_schedule,
        )

    @classmethod
    def from_frames(
        cls,
        cubes: Sequence[torch.Tensor] | torch.Tensor,
        *,
        times_s: Sequence[float],
        synthesis: SynthesisResult,
        axes: ProcessingAxes,
        epochs: Sequence[int],
        rediscovery_reasons: Sequence[str | None],
        compile_count: int,
        discovery_count: int,
        last_snapshot: object,
        last_compiled_scene: object,
        last_propagation: RadarPropagationLegs,
        last_radar_paths: RadarPathBatch,
        sample_times_s: Sequence[np.ndarray],
        path_set_complete: bool,
        motion_sampling_exhaustive: bool,
        motion_sampling: str,
        adaptive_diagnostics: Sequence[dict[str, Any]],
    ) -> Result:
        """Stack the per-frame cubes and carry the waveform's conventions.

        ``cubes`` is either the per-frame cubes or the already stacked
        ``[frame, ...]`` cube a session filled in place; the second is taken
        as it stands.

        ``synthesis`` is the LAST frame's
        :class:`~witwin.radar.synthesis.assembly.SynthesisResult`. Its
        conventions are properties of the waveform spec, which is the radar's
        stored configuration and therefore the same for every frame; taking them
        from one frame rather than re-deriving them is what keeps this result
        from becoming a second owner of the phasor convention.
        """

        stacked = cubes if isinstance(cubes, torch.Tensor) else torch.stack(tuple(cubes), dim=0)
        return cls(
            cube=stacked,
            times_s=tuple(float(value) for value in times_s),
            kind=synthesis.kind,
            axis_names=(SIMULATION_CUBE_LEADING_AXES + (synthesis.axes[0], synthesis.axes[2])),
            axes=axes,
            phasor=synthesis.phasor,
            time_dependence=synthesis.time_dependence,
            reference_frequency_hz=float(synthesis.reference_frequency_hz),
            epochs=tuple(int(value) for value in epochs),
            rediscovery_reasons=tuple(rediscovery_reasons),
            compile_count=int(compile_count),
            discovery_count=int(discovery_count),
            last_snapshot=last_snapshot,
            last_compiled_scene=last_compiled_scene,
            last_propagation=last_propagation,
            last_radar_paths=last_radar_paths,
            sample_times_s=tuple(sample_times_s),
            path_set_complete=path_set_complete,
            motion_sampling_exhaustive=motion_sampling_exhaustive,
            motion_sampling=motion_sampling,
            output_domain=synthesis.output_domain,
            adaptive_diagnostics=tuple(adaptive_diagnostics),
            observation_schedule=synthesis.observation_schedule,
        )


@dataclass(frozen=True, slots=True, eq=False)
class Frame:
    """One frame of a :class:`Result`, and the products it can become.

    Every method here delegates to :mod:`witwin.radar.processing` and adds no
    arithmetic of its own. What it adds is the pairing: a cube and the metadata
    record that describes it, built by the echo from the array that produced
    the cube, so no caller can pair the cube with a different array.
    """

    #: ``[tx, rx, slow, fast]``, a view into the result's stacked cube.
    cube: torch.Tensor
    #: The processing metadata, shared with the result this came from.
    axes: ProcessingAxes
    #: This frame's instant, s.
    time_s: float
    #: Its position in the result.
    index: int
    #: The rank-3 synthesis view, for a stage that wants the waveform's own
    #: ``(slow, sensor_pair, fast)`` layout.
    synthesis: SynthesisResult

    def processing_cube(self) -> ProcessingCube:
        """The cube and its axes as the typed record every stage takes."""

        from .processing import ProcessingCube

        return ProcessingCube(data=self.cube, axes=self.axes)

    def range_profile(self, *, window: str | None = None, remove_dc: bool = False) -> RangeProfile:
        """Range profile.

        ``window`` defaults to rectangular, and that default is load-bearing
        for FMCW: the spectrum output domain has already run the range
        transform, so a fast-time taper here would be applied to the wrong
        domain and the stage refuses it. Pass a taper only with
        ``Fmcw(output="beat")``.
        """

        from .processing import range_profile

        return range_profile(self.processing_cube(), window=window, remove_dc=remove_dc)

    def range_doppler(
        self, *, window: str | None = "hann", range_window: str | None = None, remove_dc: bool = False
    ) -> RangeDopplerMap:
        """Range-Doppler map.

        ``window`` tapers the slow axis, which is never pre-transformed, so
        Hann is a safe default there. ``range_window`` goes to the range stage
        and keeps its rectangular default for the reason above.
        """

        from .processing import range_doppler_map

        return range_doppler_map(self.range_profile(window=range_window, remove_dc=remove_dc), window=window)

    def array(self) -> ArrayGeometry:
        """The element geometry this frame's axes describe."""

        from .processing import ArrayGeometry

        return ArrayGeometry.from_axes(self.axes)

    def points(
        self,
        *,
        pfa: float = 1e-4,
        guard_cells: tuple[int, int] = (2, 4),
        training_cells: tuple[int, int] = (4, 8),
        route: str = "scan",
        fft_size: int = 64,
        max_points: int | None = None,
        window: str | None = "hann",
    ) -> PointCloud:
        """Range-Doppler, then CA-CFAR, then a point cloud, in one call.

        The sensor pairs are combined coherently - the boresight complex sum -
        before the detector, which is what makes one threshold meaningful
        across the virtual array. A caller who wants a different detector
        takes :meth:`range_doppler` and runs one from
        :mod:`witwin.radar.processing` on it.

        Named for the product rather than for the stage: the processing fence
        forbids any of the detector, angle-estimator and beamformer names
        outside that package, by name and without an allowance list, and a
        facade is not a reason to blunt it.
        """

        from .processing import ca_cfar, combine_coherent, point_cloud

        rd = self.range_doppler(window=window)
        found = ca_cfar(combine_coherent(rd.data), guard_cells=guard_cells, training_cells=training_cells, pfa=pfa)
        return point_cloud(found, rd, self.array(), route=route, fft_size=fft_size, max_points=max_points)


@dataclass(frozen=True, slots=True, eq=False)
class SceneFrame:
    """One produced frame and the session state as of that frame.

    The unit a scene session yields, so that stacking the sequence and
    streaming it read the same record. Everything a one-frame
    :class:`Result` needs is here; nothing that only makes sense
    for a whole run is. ``diagnostics`` is ``None`` for a route that does not
    adapt, which is what distinguishes "this route publishes no probe record"
    from "the probes found nothing".

    The tensor members ALIAS the frame's own device storage. This record is a
    hand-off, not a retention point.
    """

    cube: torch.Tensor
    #: The four members below are the frame's own device state. A stacked run
    #: publishes only its closing frame's, so it releases them from every
    #: earlier frame (:meth:`released`) rather than holding a run of them.
    synthesis: SynthesisResult | None
    time_s: float
    sample_times_s: np.ndarray
    epoch: int
    reason: str | None
    path_set_complete: bool
    motion_sampling_exhaustive: bool
    compile_count: int
    discovery_count: int
    epoch_frame: EpochFrame | None
    legs: RadarPropagationLegs | None
    composed: RadarPathBatch | None
    diagnostics: dict[str, Any] | None
    motion_sampling: str
    #: The declared waveform spec and the array, so that assembling the result
    #: builds its processing metadata from the same two records the cube came
    #: from rather than from whatever a caller has to hand.
    output_spec: FmcwSpec | OfdmSpec | PulsedSpec
    array: SensorArraySpec

    def released(self) -> SceneFrame:
        """This frame without the device state only a closing frame publishes."""

        return replace(self, synthesis=None, epoch_frame=None, legs=None, composed=None)


def assemble(frames: Sequence[SceneFrame], *, stacked: torch.Tensor | None = None) -> Result:
    """Stack one or more session frames into the published result.

    ``stacked`` is the ``[frame, ...]`` cube when the caller already wrote
    every frame into one allocation; the frames' own cubes are then views of
    it and are not stacked a second time.

    The run-level completeness statements are conjunctions: one frame that
    could have missed a path birth makes the sequence one that could have
    missed a path birth.
    """

    last = frames[-1]
    from .processing import ProcessingAxes

    if last.synthesis is None or last.epoch_frame is None or last.legs is None or last.composed is None:
        raise ValueError("the closing frame of a result must keep its synthesis and its world state")

    return Result.from_frames(
        [frame.cube for frame in frames] if stacked is None else stacked,
        axes=ProcessingAxes.from_synthesis(last.synthesis, last.output_spec, last.array),
        times_s=[frame.time_s for frame in frames],
        synthesis=last.synthesis,
        epochs=[frame.epoch for frame in frames],
        rediscovery_reasons=[frame.reason for frame in frames],
        compile_count=last.compile_count,
        discovery_count=last.discovery_count,
        last_snapshot=last.epoch_frame.snapshot,
        last_compiled_scene=last.epoch_frame.compiled,
        last_propagation=last.legs,
        last_radar_paths=last.composed,
        sample_times_s=[frame.sample_times_s for frame in frames],
        path_set_complete=all(frame.path_set_complete for frame in frames),
        motion_sampling_exhaustive=all(frame.motion_sampling_exhaustive for frame in frames),
        motion_sampling=last.motion_sampling,
        adaptive_diagnostics=[frame.diagnostics for frame in frames if frame.diagnostics is not None],
    )


__all__ = ["Frame", "Result"]
