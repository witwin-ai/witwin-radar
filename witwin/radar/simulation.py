"""The scene-driven entry point: one Core world in, one frame cube out.

Phase 11 work item 1. Until this module existed ``Radar.simulate`` was a
refusal, and the only thing that assembled the pipeline end to end was fixture
orchestration under ``tests/support``. Everything that orchestration called was
already a production owner - the compile facade, the epoch loop, the propagation
adapter, the two-way join, the waveform kernels, the frame assembly - so this
module invents no physics and no geometry. It is the ASSEMBLY, and the assembly
is the thing that was missing.

The chain, once per frame, in the order it runs:

    SceneEpochLoop.frame(t)          which world, and what does it cost
      -> bind_radar_world             endpoints, sites and their stable IDs
      -> reevaluate_slots x 2         one consumer call per leg, no discovery
      -> TwoWayComposer.compose       the round trip, on the device
      -> RoundTripPatternStage.apply  the array pattern, when one was declared
      -> Radar.synthesize             the waveform this radar declares
      -> assemble_frame_cube          [chirp, pair, sample] -> [TX, RX, ...]
      -> Radar.apply_signal_models    the receive chain, if one is configured

Four decisions are written here rather than left to a reader:

**The binding is rebuilt every frame, the topology is not.** ``bind_radar_world``
reads the CURRENT snapshot, so a site riding a Core rigid motion moves between
frames; the frozen leg topologies and the join are built once per topology epoch
and replayed. Rebuilding the binding is what a moving world costs, and it is the
same three small constant tensors per endpoint set that the fixture already
built per frame - it adds no discovery, no preparation and no host observation.

**The frozen slow-time mode is the only mode this driver can honestly declare.**
It composes ONCE per frame, so the weight does not walk across chirps and the
waveform kernel owns the slow-time carrier. Declaring
``REFRESHED_WEIGHT_NO_RATE`` here would tell the kernel that a weight which
never walked has already walked, which drops the intra-frame Doppler while still
producing a plausible cube. It is refused by name.

**A scatter response has no default.** ``response`` is required. The two-way
join multiplies the round trip by the target's complex response, and every
possible default - unit amplitude, unit RCS, zero phase - is a statement about
how strongly the target scatters. Guessing it would put a number nobody chose
into every result.

**Intra-frame Doppler needs a velocity dual and this entry does not open one.**
``delay_rate`` is unpacked from a forward-AD tangent, and the tangent has to be
authored from Core kinematics - :func:`witwin.radar.propagation.kinematics.two_way_duals`
is that owner. Frame-to-frame motion is fully modelled here, because every frame
re-resolves the world at its own instant; the slow-time walk WITHIN one frame is
zero unless a caller drives the kinematics seam itself. That is a named Phase-11
scope boundary, not an approximation hidden in a default.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


#: What this driver declares to the waveform kernel about the composed weight.
#:
#: Named rather than inlined so the refusal below and the value it enforces are
#: the same statement.
DRIVER_SLOW_TIME_MODE = "frozen_weight_with_carrier_rate"

#: The axis names of the published multi-frame cube. The last two are the
#: waveform's own slow and fast axes and are filled in from the synthesis
#: result, so an OFDM run publishes ``("frame", "tx", "rx", "symbol",
#: "subcarrier")`` without this module knowing what a symbol is.
SIMULATION_CUBE_LEADING_AXES = ("frame", "tx", "rx")


@dataclass(frozen=True, slots=True, eq=False)
class RadarSimulationResult:
    """What one :meth:`witwin.radar.Radar.simulate` call produced.

    Data only. The driver builds it through :meth:`from_frames`, which is this
    repository's standing shape for a result: the producer knows what it made,
    and a consumer never has to infer a phasor convention from a method name.

    ``cube`` is ``[frame, TX, RX, slow, fast]`` and is the product. It is one
    stacked tensor rather than a list because a frame sequence is what every
    downstream consumer - a range-Doppler map, a tracker, a loss - indexes, and
    the stack is a single differentiable op outside the frame loop.

    The four ``last_*`` members are the LAST frame's typed state, and they are
    what :attr:`witwin.radar.Radar.last_snapshot` and its three siblings read.
    They describe one frame, not the sequence: a compiled scene and a leg pair
    are per-epoch and per-frame objects, and stacking them would either
    misrepresent the epochs or retain every frame's device memory for the life
    of the result. Keeping the last one is the diagnostic the plan asked for and
    the smallest retention that answers it.

    RETENTION, stated because these members are a real tensor lifetime. The
    ``last_*`` members alias the frame's own batches, so holding this result
    holds that frame's device tensors - and, when ``ad_mode`` asked for a graph,
    that frame's autograd graph. None of them holds a tape: an autograd context
    or a ``saved_tensors`` tuple in any of these fields would be a data record
    turned into a handle on somebody else's memory, and
    ``tests/test_phase9_tape_non_leak.py`` walks all four to keep it that way.
    """

    cube: torch.Tensor
    times_s: tuple[float, ...]
    kind: str
    axes: tuple[str, ...]
    phasor: str
    time_dependence: str
    reference_frequency_hz: float
    epochs: tuple[int, ...]
    rediscovery_reasons: tuple[str | None, ...]
    compile_count: int
    discovery_count: int
    last_snapshot: object
    last_compiled_scene: object
    last_propagation: object
    last_radar_paths: object

    def __post_init__(self) -> None:
        if self.cube.dim() != len(self.axes):
            raise ValueError(
                f"a {self.kind} simulation cube has {len(self.axes)} axes "
                f"{self.axes}, got shape {tuple(self.cube.shape)}"
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

    @classmethod
    def from_frames(
        cls,
        cubes,
        *,
        times_s,
        synthesis,
        epochs,
        rediscovery_reasons,
        compile_count: int,
        discovery_count: int,
        last_snapshot: object,
        last_compiled_scene: object,
        last_propagation: object,
        last_radar_paths: object,
    ) -> "RadarSimulationResult":
        """Stack the per-frame cubes and carry the waveform's conventions.

        ``synthesis`` is the LAST frame's
        :class:`~witwin.radar.synthesis.contracts.SynthesisResult`. Its
        conventions are properties of the waveform spec, which is the radar's
        stored configuration and therefore the same for every frame; taking them
        from one frame rather than re-deriving them is what keeps this result
        from becoming a second owner of the phasor convention.
        """

        stacked = torch.stack(tuple(cubes), dim=0)
        return cls(
            cube=stacked,
            times_s=tuple(float(value) for value in times_s),
            kind=synthesis.kind,
            axes=(
                SIMULATION_CUBE_LEADING_AXES
                + (synthesis.axes[0], synthesis.axes[2])
            ),
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
        )


def _dynamic_scene(scene: object) -> object:
    """A ``DynamicScene`` for whichever of the two Core worlds was passed.

    A static ``Scene`` is wrapped rather than refused: it IS a dynamic scene
    with no declared motion, the loop then reports ``structures_move = False``,
    and the compiled scene is built exactly once for the whole run. Refusing it
    would force every caller with a still world to write the wrapper themselves.
    """

    if all(
        hasattr(scene, name)
        for name in ("at", "structure_trajectories", "structure_deformations")
    ):
        return scene
    from witwin.core.dynamics import DynamicScene

    return DynamicScene(scene)


def _slow_time_mode(declared: object):
    from .synthesis import SlowTimeMode

    if declared is None:
        return SlowTimeMode.FROZEN_WEIGHT_WITH_CARRIER_RATE
    mode = SlowTimeMode(declared)
    if mode is not SlowTimeMode.FROZEN_WEIGHT_WITH_CARRIER_RATE:
        raise ValueError(
            f"this entry point composes ONCE per frame, so its weight is "
            f"{DRIVER_SLOW_TIME_MODE!r} and cannot be declared {mode.value!r}. "
            "A refreshed weight is one that already walked across slow time; "
            "declaring it for a weight frozen at the frame's tau_rt drops the "
            "kernel's carrier-rate term and understates intra-frame Doppler "
            "while still producing a plausible cube. Drive the refreshed mode "
            "from a slot-batched replay instead"
        )
    return mode


def _times(times: object) -> tuple[float, ...]:
    values = tuple(float(value) for value in times)
    if not values:
        raise ValueError(
            "times must name at least one frame instant; an empty sequence "
            "asks for a simulation of nothing"
        )
    return values


def simulate_scene(
    radar: object,
    scene: object,
    *,
    times,
    response: object,
    sites: object = None,
    components: frozenset[str] | None = None,
    max_depth: int | None = None,
    slow_time_mode: object = None,
    ad_mode: str = "none",
    world_motion: str = "frozen_world",
    motion_event_period_frames: int | None = None,
    ids: object = None,
    polarization: object = None,
    antenna_pattern: object = None,
) -> RadarSimulationResult:
    """Run ``radar`` over ``scene`` at ``times`` and publish the frame cubes.

    This is the whole of :meth:`witwin.radar.Radar.simulate`; the method is a
    delegation so that the assembly lives next to the contracts it assembles
    rather than inside the radar's own configuration and pose module.

    ``sites`` is a :class:`~witwin.radar.scene_binding.ScatterSitePolicy` and
    defaults to ``ScatterSitePolicy.structure_anchor()`` - one site at every
    Core-owned structure world anchor. A structure with no rigid motion has no
    such anchor and the policy refuses it by name; that refusal is the design
    (R-ADR-020), because the alternative is a mesh-sampling rule, which is a
    geometry algorithm and does not belong in a Torch expression here.

    ``components`` and ``max_depth`` override the radar's propagation block for
    THIS call through
    :meth:`~witwin.radar.config.RadarSystemConfig.with_propagation`, which
    returns a new configuration rather than mutating the radar's stored one.

    ``world_motion`` and ``motion_event_period_frames`` are
    :class:`~witwin.radar.propagation.epochs.SceneEpochLoop`'s own two
    arguments, forwarded verbatim. A caller that would rather describe its scene
    by its parts resolves
    :func:`~witwin.radar.propagation.epochs.epoch_policy` first and passes the
    two fields it produces; the loop never reads a component declaration, and
    keeping that one-way is what stops "which parts does this scene have" and
    "when does the pipeline pay" from becoming one question.

    ``ad_mode`` is forwarded to every replay. ``"none"`` is the default and
    builds no graph; ``"vjp"`` makes the published cube differentiable with
    respect to the endpoint and site positions the binding passed through by
    identity.

    ``antenna_pattern`` is an
    :class:`~witwin.radar.sensors.contracts.AntennaPatternSpec` and defaults to
    ``None``, which applies no pattern and launches no extra kernel. It does NOT
    default to ``radar.system_config.sensors.pattern``: that spec falls back to
    a half-wave dipole, so adopting it here would attenuate every result by a
    number nobody chose. Pass that spec to use it,
    :data:`~witwin.radar.sensors.round_trip.ISOTROPIC_PATTERN` to run the stage
    as a proven no-op, or leave it ``None``.
    """

    from .paths import TwoWayComposer
    from .propagation.channel_consumer import (
        ChannelPropagationAdapter,
        compile_scene,
    )
    from .propagation.contracts import RadarPropagationLegs
    from .propagation.epochs import FrozenEpoch, SceneEpochLoop
    from .sensors.round_trip import RoundTripPatternStage
    from .scene_binding import (
        DEFAULT_POLARIZATION,
        ScatterSitePolicy,
        bind_radar_world,
    )
    from .synthesis import assemble_frame_cube, validate_pair_ordering

    instants = _times(times)
    mode = _slow_time_mode(slow_time_mode)
    policy = ScatterSitePolicy.structure_anchor() if sites is None else sites
    if not isinstance(policy, ScatterSitePolicy):
        raise TypeError(
            f"sites must be a ScatterSitePolicy, got {type(policy).__name__}; "
            "where the scatter sites come from is a declaration, not a search"
        )
    orientation = DEFAULT_POLARIZATION if polarization is None else polarization

    solve_config = radar.system_config.with_propagation(
        components=components, max_depth=max_depth
    )
    propagation = solve_config.propagation
    array = solve_config.sensors.array
    reference_frequency_hz = propagation.reference_frequency_hz

    def bind(compiled, snapshot, previous):
        binding = bind_radar_world(
            radar, snapshot, sites=policy, ids=ids, polarization=orientation
        )
        adapter = (
            ChannelPropagationAdapter(
                compiled,
                reference_frequency_hz=reference_frequency_hz,
                components=propagation.components,
                max_depth=propagation.max_depth,
            )
            if previous is None
            else previous.adapter
        )
        inbound = adapter.freeze(binding.transmitters, binding.site_sinks)
        outbound = adapter.freeze(binding.site_sources, binding.receivers)
        composer = TwoWayComposer.freeze(
            inbound,
            outbound,
            torch.tensor(
                binding.site_ids, dtype=torch.int64, device=binding.device
            ),
            radar_source_ids=list(binding.transmitter_ids),
            radar_sink_ids=list(binding.receiver_ids),
            reference_frequency_hz=reference_frequency_hz,
        )
        # Once per topology epoch, where the topology is decided and the host
        # read is free. It must never move into the frame loop: the same read
        # there is a per-frame device-to-host transfer.
        validate_pair_ordering(
            composer.sensor_pair_index,
            num_tx=array.num_tx,
            num_rx=array.num_rx,
            sensor_pair_count=composer.sensor_pair_count,
        )
        # The pattern tables are a property of the frozen join - which pair each
        # row belongs to and which site it visits - so they are built here, once
        # per epoch, and the frame loop only gathers positions and launches.
        stage = (
            None
            if antenna_pattern is None
            else RoundTripPatternStage.freeze(
                radar,
                composer,
                site_ids=binding.site_ids,
                pattern=antenna_pattern,
            )
        )
        # The binding travels with the epoch so the frame that just froze does
        # not build a second one from the same snapshot. It is deterministic, so
        # the two would agree - which is exactly why building both is waste.
        return FrozenEpoch(
            adapter=adapter,
            handles=(inbound, outbound),
            payload=(composer, binding, stage),
        )

    loop = SceneEpochLoop(
        _dynamic_scene(scene),
        reference_frequency_hz=reference_frequency_hz,
        bind=bind,
        compile_scene=compile_scene,
        motion_event_period_frames=motion_event_period_frames,
        world_motion=world_motion,
    )

    cubes: list[torch.Tensor] = []
    epochs: list[int] = []
    reasons: list[str | None] = []
    synthesis = None
    legs = None
    composed = None
    epoch_frame = None
    for time_s in instants:
        epoch_frame = loop.frame(time_s)
        frozen = epoch_frame.frozen
        inbound_handle, outbound_handle = frozen.handles
        composer, epoch_binding, pattern_stage = frozen.payload
        # Rebound at every frame that did not just freeze, because the site
        # positions and the radar pose are read from the CURRENT world: a site
        # riding a Core rigid motion moves between frames while the frozen
        # topology and the join do not.
        binding = (
            epoch_binding
            if epoch_frame.rediscovered
            else bind_radar_world(
                radar,
                epoch_frame.snapshot,
                sites=policy,
                ids=ids,
                polarization=orientation,
            )
        )
        legs = RadarPropagationLegs(
            inbound=frozen.adapter.reevaluate_slots(
                inbound_handle,
                binding.transmitters,
                binding.site_sinks,
                slot_count=1,
                ad_mode=ad_mode,
            ),
            outbound=frozen.adapter.reevaluate_slots(
                outbound_handle,
                binding.site_sources,
                binding.receivers,
                slot_count=1,
                ad_mode=ad_mode,
            ),
        )
        composed = composer.compose(legs.inbound, legs.outbound, response)
        if pattern_stage is not None:
            composed = pattern_stage.apply(
                composed,
                tx_pos=binding.transmitters.positions_m,
                rx_pos=binding.receivers.positions_m,
                site_positions_m=binding.site_positions_m,
            )
        synthesis = radar.synthesize(composed, slow_time_mode=mode)
        cubes.append(
            radar.apply_signal_models(
                assemble_frame_cube(
                    synthesis.cube, num_tx=array.num_tx, num_rx=array.num_rx
                )
            )
        )
        epochs.append(epoch_frame.epoch)
        reasons.append(epoch_frame.reason)

    return RadarSimulationResult.from_frames(
        cubes,
        times_s=instants,
        synthesis=synthesis,
        epochs=epochs,
        rediscovery_reasons=reasons,
        compile_count=loop.compile_count,
        discovery_count=loop.discovery_count,
        last_snapshot=epoch_frame.snapshot,
        last_compiled_scene=epoch_frame.compiled,
        last_propagation=legs,
        last_radar_paths=composed,
    )


__all__ = [
    "DRIVER_SLOW_TIME_MODE",
    "SIMULATION_CUBE_LEADING_AXES",
    "RadarSimulationResult",
    "simulate_scene",
]
