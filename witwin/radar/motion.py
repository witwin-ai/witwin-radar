"""How often a scene session resamples the world inside one frame.

:class:`Motion` is the one sampling record. Its tolerance fields belong to the
adaptive kind; every other kind ignores them, and :mod:`witwin.radar.adaptive`
reads them straight off this record rather than off a second copy of it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

MOTION_KINDS = ("auto", "static", "chirp", "adc", "adaptive")


def _positive_host_float(value: float, *, name: str) -> float:
    if isinstance(value, torch.Tensor) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive host value")
    return float(value)


def _positive_int(value: object, *, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive int, got {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class Motion:
    """How often the world is resampled inside one frame.

    Build one with :meth:`auto`, :meth:`static`, :meth:`chirp`, :meth:`adc` or
    :meth:`adaptive`. The constructors are the only way to spell a sampling
    choice, so a tolerance without adaptive sampling, or a discovery cadence
    with it, is unwritable rather than refused.

    The tolerance fields belong to :meth:`adaptive` and are ignored by every
    other kind. They are fields of this one record rather than a second object
    because a sampling choice and the error budget that justifies it are one
    statement. Every field is validated at construction, whatever the kind.

    The adaptive tolerances are sampled error control, not a proof of absent
    events between probes. The phase tolerance is radians per path before
    coherent summation and the amplitude tolerance is relative per path; near
    coherent nulls a relative IQ bound cannot be inferred from these path
    bounds. Exhausting the discovery budget raises.

    ``max_interval`` is the MAXIMUM PROBE SPACING in seconds and it always
    applies: no accepted interval spans more than this, so the probe grid is
    never coarser than ``max_interval / (2 * (nodes - 1))``. Lowering it
    refines the grid. Every tolerance is checked by sampling, so it bounds only
    what it samples: motion periodic at the grid's step, or at a divisor of it,
    sits at a zero of every probe and is invisible to all of the tests. This
    bound is what sets that step, so it is an accuracy control as much as a
    topology one, and it is NOT relaxed for a topologically certified family:
    the certification says no path can be born, not that no path moves fast.
    Within the bound the run starts from the coarsest partition it allows
    rather than bisecting down to it. Set it from the motion's bandwidth; the
    default 2 ms resolves roughly 250 Hz at two nodes.

    ``nodes`` is how many sampled instants carry one accepted interval, and
    therefore the polynomial order of the delay it interpolates: 2 is the
    linear rule and 5 a quartic. Each interval probes a grid of
    ``2 * (nodes - 1) + 1`` instants, spends the even ones as interpolation
    nodes and tests the error at the odd ones. Raise it when the phase test is
    what shortens the intervals (micro-Doppler at a high carrier); leave it at 2
    when ``max_interval`` decides the length, because the order then cannot
    reduce the interval count and only multiplies each interval's grid.
    Raising it never makes a coarse grid safe: the grid still has to resolve
    the motion.
    """

    kind: str = "auto"
    #: Adaptive: the largest tested round-trip phase error an accepted interval
    #: may carry, rad.
    phase_error: float = 0.02
    #: Adaptive: the largest tested relative amplitude error, dimensionless.
    relative_amplitude_error: float = 0.02
    #: Adaptive: the maximum probe spacing, s.
    max_interval: float = 0.002
    #: Adaptive: how many sampled instants one accepted interval interpolates
    #: through. Two is the linear rule and five is a quartic.
    nodes: int = 2
    #: Adaptive: the discovery budget. Exhausting it raises rather than
    #: returning an unchecked cube.
    max_evaluations: int = 8192
    #: Adaptive: observations per batched replay, and per synthesis batch.
    batch_observations: int = 256
    #: Topology rediscovery cadence in frames. ``None`` rediscovers at every
    #: observation, which is the only cadence that cannot miss a path birth.
    rediscover_every_frames: int | None = None
    #: Channel's replay vocabulary for compiled geometry.
    world: str = "frozen_world"

    def __post_init__(self) -> None:
        if self.kind not in MOTION_KINDS:
            raise ValueError(f"Motion.kind must be one of {list(MOTION_KINDS)}, got {self.kind!r}")
        for name in ("phase_error", "relative_amplitude_error", "max_interval"):
            _positive_host_float(getattr(self, name), name=name)
        for name in ("max_evaluations", "batch_observations"):
            _positive_int(getattr(self, name), name=name)
        if _positive_int(self.nodes, name="nodes") < 2:
            raise ValueError("nodes must be at least 2; one node cannot interpolate")

    @classmethod
    def auto(cls, *, rediscover_every_frames: int | None = None, world: str = "frozen_world") -> Motion:
        """Resample at every ADC instant when anything moves, else once a frame.

        "Anything moves" is a property of the session, not of this record:
        structure trajectories or deformations, endpoint trajectories, a target
        trajectory, or a receiver with oscillator phase noise, which needs
        ADC-time observations to place its delayed phase difference.
        """

        return cls(kind="auto", rediscover_every_frames=rediscover_every_frames, world=world)

    @classmethod
    def static(cls, *, rediscover_every_frames: int | None = None, world: str = "frozen_world") -> Motion:
        """One observation per frame. Refused for a world that moves."""

        return cls(kind="static", rediscover_every_frames=rediscover_every_frames, world=world)

    @classmethod
    def chirp(cls, *, rediscover_every_frames: int | None = None, world: str = "frozen_world") -> Motion:
        """Stop and hop: geometry frozen within each chirp, symbol or pulse."""

        return cls(kind="chirp", rediscover_every_frames=rediscover_every_frames, world=world)

    @classmethod
    def adc(cls, *, rediscover_every_frames: int | None = None, world: str = "frozen_world") -> Motion:
        """Resample at every ADC instant. The exhaustive reference."""

        return cls(kind="adc", rediscover_every_frames=rediscover_every_frames, world=world)

    @classmethod
    def adaptive(
        cls,
        *,
        phase_error: float = 0.02,
        relative_amplitude_error: float = 0.02,
        max_interval: float = 0.002,
        nodes: int = 2,
        max_evaluations: int = 8192,
        batch_observations: int = 256,
        rediscover_every_frames: int | None = None,
        world: str = "frozen_world",
    ) -> Motion:
        """Interpolate between error-controlled probes. FMCW only.

        It certifies nothing between its probes: an arbitrarily brief path
        birth or oscillation is invisible to a sampled test. The run publishes
        ``path_set_complete`` and ``motion_sampling_exhaustive`` separately so
        that limit is readable rather than implied.
        """

        return cls(
            kind="adaptive",
            phase_error=phase_error,
            relative_amplitude_error=relative_amplitude_error,
            max_interval=max_interval,
            nodes=nodes,
            max_evaluations=max_evaluations,
            batch_observations=batch_observations,
            rediscover_every_frames=rediscover_every_frames,
            world=world,
        )


__all__ = ["MOTION_KINDS", "Motion"]
