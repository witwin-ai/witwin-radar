"""The radar facade: configuration, pose, antenna state and the frame entry.

``Radar.simulate`` is the production entry point and it delegates to
:mod:`witwin.radar.simulation`. This module owns no propagation and no
synthesis physics; what it holds is the configuration record, the pose
transforms every consumer shares, the antenna-pattern state, and the four typed
diagnostics of the last completed frame.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from .sensors.pattern import evaluate_antenna_pattern_vectors, evaluate_antenna_pattern_xy
from .utils.vector import vec3_tensor

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .frontend import FrontendSpec
    from .simulation import RadarSimulationResult
    from .synthesis import SynthesisResult


@dataclass(frozen=True)
class RadarConfig:
    num_tx: int
    num_rx: int
    fc: float
    slope: float
    adc_samples: int
    adc_start_time: float
    sample_rate: float
    idle_time: float
    ramp_end_time: float
    chirp_per_frame: int
    frame_per_second: float
    num_doppler_bins: int
    num_range_bins: int
    num_angle_bins: int
    power: float
    tx_loc: tuple[tuple[float, float, float], ...]
    rx_loc: tuple[tuple[float, float, float], ...]
    antenna_pattern: dict[str, Any] | None = None
    #: The receive chain: ONE ordered chain with ONE ADC and ONE seed base. It
    #: replaced a ``noise_model`` / ``receiver_chain`` pair whose composite
    #: order was the caller's to choose, and since Phase 11 it is the only one
    #: - the pair is deleted, so there is no configuration in which two chains
    #: can disagree about where the LNA sits. It is ``None`` by default: noise
    #: is optional and OFF unless a caller asks for it, and every physics test
    #: runs without it.
    #:
    #: ``polarization`` left this record with them. It described a second
    #: projection of a field Channel has already projected onto each endpoint's
    #: declared polarization, and its only reader was the deleted Dirichlet
    #: route. The sensor block still carries a ``PolarizationSpec`` for the
    #: kernel mode that implements it, which no production route enables.
    frontend: "FrontendSpec | None" = None

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "RadarConfig":
        from .validation import validate_radar_config

        return validate_radar_config(config)

    @classmethod
    def from_json(cls, path: str | os.PathLike[str]) -> "RadarConfig":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


def _target_from_position(position: torch.Tensor) -> torch.Tensor:
    return position + torch.tensor((0.0, 0.0, -1.0), dtype=torch.float32)


# `quantize_complex_signal`, `db_to_voltage_gain`, `ReceiverChainRuntime`,
# `NoiseModelRuntime` and `PolarizationRuntime` stood here until Phase 11. The
# first four were the legacy receive chain that `frontend/FrontendChain`
# replaced, and `apply_signal_models` chose between the two owners at runtime -
# a shadow mode, which acceptance criterion 6 forbids. `PolarizationRuntime`
# went with them: its only consumer outside this file was
# `sensors/legacy_paths.py`, on the deleted Dirichlet route.


class Radar:
    #: The one diagnostic retention site, as a CLASS attribute so that the four
    #: ``last_*`` properties answer ``None`` on an instance that has never run -
    #: including one built by ``object.__new__`` for a refusal test - instead of
    #: raising ``AttributeError`` from a half-initialized object.
    _last_result = None

    def __init__(
        self,
        config: RadarConfig | Mapping[str, Any],
        device: str | torch.device = "cuda",
        *,
        position=(0.0, 0.0, 0.0),
        target=None,
        up=(0.0, 1.0, 0.0),
        fov: float = 60.0,
        name: str | None = None,
    ):
        """
        Args:
            config: ``RadarConfig`` or a raw mapping accepted by ``RadarConfig.from_dict``.
            device: CUDA compute device
            position: radar origin in world coordinates
            target: look-at target in world coordinates. Defaults to one meter along -Z from position.
            up: world-space up vector
            fov: perspective field of view in degrees
            name: optional identifier for this radar
        """
        self.c0 = 299792458
        self.device: torch.device = self._resolve_device(device=torch.device(device))
        self.name = None if name is None else str(name)
        self._set_pose_fields(position=position, target=target, up=up, fov=fov)

        self.config: RadarConfig = config if isinstance(config, RadarConfig) else RadarConfig.from_dict(config)
        cfg = self.config

        self._init_system_config(cfg)
        self._init_antenna_locations(cfg)
        self._init_runtime_models(cfg)
        self._init_axes(cfg)

    def _init_system_config(self, cfg: RadarConfig) -> None:
        """The five-block structural view of the flat configuration.

        The flat form stays the file format and the public constructor; this is
        what an adapter, a synthesis owner, or a signal processor is handed, so
        each one sees only the block it owns. ``waveform.kind`` is a STORED
        discriminator: nothing downstream infers "this is FMCW" by finding a
        ``slope``.
        """

        from .config import RadarSystemConfig

        self.system_config = RadarSystemConfig.from_radar_config(
            cfg, frontend=cfg.frontend
        )

    def _init_antenna_locations(self, cfg: RadarConfig) -> None:
        self._lambda = self.c0 / cfg.fc
        antenna_spacing = self.c0 / cfg.fc / 2
        self.tx_loc = torch.tensor(cfg.tx_loc, dtype=torch.float32, device=self.device) * antenna_spacing
        self.rx_loc = torch.tensor(cfg.rx_loc, dtype=torch.float32, device=self.device) * antenna_spacing
        self._refresh_pose_dependent_state()

    def _init_runtime_models(self, cfg: RadarConfig) -> None:
        from .validation import default_dipole_antenna_pattern

        self.antenna_pattern_config = cfg.antenna_pattern or default_dipole_antenna_pattern()
        self._build_antenna_pattern_runtime(self.antenna_pattern_config)
        self.frontend = self._make_frontend(cfg)

    @staticmethod
    def _make_frontend(cfg: RadarConfig):
        if cfg.frontend is None:
            return None
        from .frontend import FrontendChain

        return FrontendChain(cfg.frontend)

    def _init_axes(self, cfg: RadarConfig) -> None:
        """Build the one axes record, and keep the flat reads as views of it.

        ``sigproc`` used to read ``radar.range_resolution``, ``radar._lambda``,
        ``radar.config.idle_time``, and four more raw scalars straight off the
        radar, which is how a signal processor ends up knowing which waveform it
        is looking at. There is now a single :class:`~witwin.radar.config.RadarAxes`
        record and the flat attributes are reads of it, so the two cannot
        disagree.
        """

        self.axes = self.system_config.axes(device=self.device)
        self.range_resolution = self.axes.range_resolution
        self.max_range = self.axes.max_range
        self.ranges = self.axes.ranges
        self.doppler_resolution = self.axes.doppler_resolution
        self.max_doppler = self.axes.max_doppler
        self.velocities = self.axes.velocities

    @staticmethod
    def _resolve_device(*, device: torch.device) -> torch.device:
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "Radar defaults to CUDA, but torch.cuda.is_available() is False. "
                "Install a CUDA-enabled PyTorch build and use device='cuda'."
            )
        return device

    def _set_pose_fields(self, *, position, target, up, fov) -> None:
        position_t = vec3_tensor(position, name="Radar.position")
        target_t = _target_from_position(position_t) if target is None else vec3_tensor(target, name="Radar.target")
        up_t = vec3_tensor(up, name="Radar.up")
        forward = target_t - position_t
        if torch.linalg.norm(forward) <= 1e-12:
            raise ValueError("Radar.target must differ from Radar.position.")
        if torch.linalg.norm(up_t) <= 1e-12:
            raise ValueError("Radar.up must be non-zero.")
        if torch.linalg.norm(torch.cross(forward, up_t, dim=0)) <= 1e-12:
            raise ValueError("Radar.up must not be collinear with the viewing direction.")
        self.position = position_t
        self.target = target_t
        self.up = up_t
        self.fov = float(fov)

    def _refresh_pose_dependent_state(self) -> None:
        self.tx_pos = self.world_from_local_points(self.tx_loc).contiguous()
        self.rx_pos = self.world_from_local_points(self.rx_loc).contiguous()
        self.origin = self.position

    def _build_antenna_pattern_runtime(self, config: dict[str, Any]) -> None:
        self.antenna_pattern_kind = config["kind"]
        self.antenna_pattern_x_angles_deg = torch.tensor(config["x_angles_deg"], dtype=torch.float32, device=self.device)
        self.antenna_pattern_y_angles_deg = torch.tensor(config["y_angles_deg"], dtype=torch.float32, device=self.device)
        self.antenna_pattern_x_values = None
        self.antenna_pattern_y_values = None
        self.antenna_pattern_values = None
        if config["kind"] == "separable":
            self.antenna_pattern_x_values = torch.tensor(config["x_values"], dtype=torch.float32, device=self.device)
            self.antenna_pattern_y_values = torch.tensor(config["y_values"], dtype=torch.float32, device=self.device)
        else:
            self.antenna_pattern_values = torch.tensor(config["values"], dtype=torch.float32, device=self.device)

    def evaluate_antenna_pattern_xy(self, x_angles_deg: torch.Tensor, y_angles_deg: torch.Tensor) -> torch.Tensor:
        return evaluate_antenna_pattern_xy(
            self.antenna_pattern_kind,
            self.antenna_pattern_x_angles_deg,
            self.antenna_pattern_y_angles_deg,
            self.antenna_pattern_x_values,
            self.antenna_pattern_y_values,
            self.antenna_pattern_values,
            x_angles_deg,
            y_angles_deg,
        )

    def evaluate_antenna_pattern_vectors(self, vectors: torch.Tensor) -> torch.Tensor:
        return evaluate_antenna_pattern_vectors(
            self.antenna_pattern_kind,
            self.antenna_pattern_x_angles_deg,
            self.antenna_pattern_y_angles_deg,
            self.antenna_pattern_x_values,
            self.antenna_pattern_y_values,
            self.antenna_pattern_values,
            vectors,
        )

    def set_pose(self, *, position=None, target=None, up=None, fov=None) -> "Radar":
        """Mutate radar pose and refresh pose-dependent antenna state."""
        new_position = self.position if position is None else vec3_tensor(position, name="Radar.position")
        if target is None:
            target_t = self.target if position is None else new_position + (self.target - self.position)
        else:
            target_t = vec3_tensor(target, name="Radar.target")
        up_t = self.up if up is None else vec3_tensor(up, name="Radar.up")
        fov_value = self.fov if fov is None else float(fov)
        self._set_pose_fields(position=new_position, target=target_t, up=up_t, fov=fov_value)
        self._refresh_pose_dependent_state()
        return self

    def _world_from_local_matrix(self, *, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        position = self.position.to(device=device, dtype=dtype)
        target = self.target.to(device=device, dtype=dtype)
        up = self.up.to(device=device, dtype=dtype)

        forward = target - position
        forward = forward / torch.linalg.norm(forward)
        right = torch.cross(forward, up, dim=0)
        right = right / torch.linalg.norm(right)
        true_up = torch.cross(right, forward, dim=0)
        true_up = true_up / torch.linalg.norm(true_up)
        back = -forward
        world_from_local = torch.stack((right, true_up, back), dim=1)
        return position, world_from_local

    def world_from_local_points(self, points: torch.Tensor) -> torch.Tensor:
        position, world_from_local = self._world_from_local_matrix(device=points.device, dtype=points.dtype)
        return points @ world_from_local.transpose(0, 1) + position

    def world_from_local_vectors(self, vectors: torch.Tensor) -> torch.Tensor:
        _, world_from_local = self._world_from_local_matrix(device=vectors.device, dtype=vectors.dtype)
        return vectors @ world_from_local.transpose(0, 1)

    def local_from_world_points(self, points: torch.Tensor) -> torch.Tensor:
        position, world_from_local = self._world_from_local_matrix(device=points.device, dtype=points.dtype)
        return (points - position) @ world_from_local

    def local_from_world_vectors(self, vectors: torch.Tensor) -> torch.Tensor:
        _, world_from_local = self._world_from_local_matrix(device=vectors.device, dtype=vectors.dtype)
        return vectors @ world_from_local

    def apply_signal_models(self, signal: torch.Tensor) -> torch.Tensor:
        """Run the receive chain, if one is configured.

        This used to CHOOSE between two owners: the frontend block, or the
        legacy ``noise_model`` / ``receiver_chain`` pair, with a constructor
        refusal for the configuration that named both. A refusal is not the
        same as having one owner, and a runtime choice between two chains is
        the shadow mode acceptance criterion 6 forbids. The pair is deleted, so
        the only question left is whether a chain exists.
        """

        if self.frontend is None:
            return signal
        return self.frontend.apply(signal).signal

    def synthesize(self, paths, *, slow_time_mode) -> "SynthesisResult":
        """Synthesize one frame with whichever waveform this radar declares.

        Dispatch is a dict lookup on the STORED ``waveform.kind``. It is not a
        ``try``/``except``, not a capability probe, and not an inference from a
        ``slope``: a kind with no owner is a hard error, because a waveform
        without an owner has no physics and returning a plausible cube would be
        worse than failing.

        ``paths`` may be a composed :class:`~witwin.radar.paths.RadarPathBatch`
        or an already-wrapped
        :class:`~witwin.radar.synthesis.SynthesisPathBatch`. ``slow_time_mode``
        has no default for the reason it has none anywhere else: only the caller
        knows whether it froze the weight for the frame or refreshes it per
        slot, and defaulting it makes the Phase-7 collision a silent wrong
        answer instead of a refusal.
        """

        from .config import WAVEFORM_FMCW, WAVEFORM_OFDM, WAVEFORM_PULSED
        from .synthesis import (
            SynthesisPathBatch,
            SynthesisResult,
            synthesize_fmcw_beat,
            synthesize_ofdm_cfr,
            synthesize_pulsed_echo,
        )

        owners = {
            WAVEFORM_FMCW: (synthesize_fmcw_beat, SynthesisResult.from_fmcw_beat),
            WAVEFORM_OFDM: (synthesize_ofdm_cfr, SynthesisResult.from_ofdm_cfr),
            WAVEFORM_PULSED: (synthesize_pulsed_echo, SynthesisResult.from_pulsed_echo),
        }
        kind = self.system_config.kind
        if kind not in owners:
            raise ValueError(
                f"no synthesis owner for waveform kind {kind!r}; the supported "
                f"kinds are {sorted(owners)}. This dispatch has no fallback: a "
                "waveform without an owner has no physics."
            )
        batch = (
            paths
            if isinstance(paths, SynthesisPathBatch)
            else SynthesisPathBatch.from_radar_paths(
                paths, slow_time_mode=slow_time_mode
            )
        )
        synthesize, build_result = owners[kind]
        spec = self.system_config.waveform_spec()
        return build_result(synthesize(batch, spec), spec)

    def simulate(
        self,
        scene,
        *,
        times,
        response,
        sites=None,
        components=None,
        max_depth=None,
        slow_time_mode=None,
        ad_mode: str = "none",
        world_motion: str = "frozen_world",
        motion_event_period_frames: int | None = None,
        ids=None,
        polarization=None,
        antenna_pattern=None,
    ) -> "RadarSimulationResult":
        """Simulate this radar over a Core world and return the frame cubes.

        The scene-driven entry point. ``scene`` is a ``witwin.core.Scene`` or a
        ``witwin.core.dynamics.DynamicScene``; ``times`` is the sequence of
        frame instants in seconds; ``response`` is the scatter response the
        two-way join multiplies the round trip by, and it is required because
        every default for it would be an unchosen statement about how strongly
        the target scatters.

        The whole assembly lives in :mod:`witwin.radar.simulation` and its
        docstring is the contract; read it before changing anything here. This
        method exists so that the pipeline is reachable under the name a caller
        looks for, and it delegates rather than reimplementing so there is one
        owner of the frame loop.

        Calling this publishes the four typed diagnostics
        (:attr:`last_snapshot`, :attr:`last_compiled_scene`,
        :attr:`last_propagation`, :attr:`last_radar_paths`). They are cleared
        FIRST, so a call that raises part way through leaves no stale world
        behind claiming to describe this radar.

        ``antenna_pattern`` opts this solve into the array's transmit and
        receive pattern gain, applied by the native ``sensor_weight`` family
        between the two-way join and synthesis. It is ``None`` by default and
        deliberately does not fall back to
        ``self.system_config.sensors.pattern``, whose own default is a half-wave
        dipole: silently attenuating every result by an unrequested pattern is
        the failure this keyword exists to avoid. Pass that spec to use it.

        ``simulate_group`` is gone. It was a permanently refusing classmethod
        and a permanent refusal is itself a legacy shim; simulating several
        radars over one world is a loop over this method, and no Radar-owned
        batching of it exists to hide.
        """

        from .simulation import simulate_scene

        self._last_result = None
        result = simulate_scene(
            self,
            scene,
            times=times,
            response=response,
            sites=sites,
            components=components,
            max_depth=max_depth,
            slow_time_mode=slow_time_mode,
            ad_mode=ad_mode,
            world_motion=world_motion,
            motion_event_period_frames=motion_event_period_frames,
            ids=ids,
            polarization=polarization,
            antenna_pattern=antenna_pattern,
        )
        self._last_result = result
        return result

    # -- the four typed diagnostics (Phase 11 work item 2) ------------------
    #
    # One retention site, four reads of it. The alternative - four independent
    # attributes - can be left describing four different frames by any code
    # path that sets three of them, and "which frame is this" is exactly the
    # question a diagnostic exists to answer. ``None`` before the first
    # ``simulate`` is the pinned answer: a caller may poll these, and raising
    # would make "has this radar run yet" a try/except.

    @property
    def last_result(self) -> "RadarSimulationResult | None":
        """The whole of the last :meth:`simulate` call, or ``None``."""

        return self._last_result

    @property
    def last_snapshot(self):
        """The Core ``SceneSnapshot`` the last simulated frame ran against."""

        return None if self._last_result is None else self._last_result.last_snapshot

    @property
    def last_compiled_scene(self):
        """The Channel ``CompiledScene`` that frame's legs were replayed on."""

        return (
            None
            if self._last_result is None
            else self._last_result.last_compiled_scene
        )

    @property
    def last_propagation(self):
        """That frame's two legs, as a typed
        :class:`~witwin.radar.propagation.contracts.RadarPropagationLegs`."""

        return (
            None if self._last_result is None else self._last_result.last_propagation
        )

    @property
    def last_radar_paths(self):
        """That frame's composed
        :class:`~witwin.radar.paths.RadarPathBatch`."""

        return (
            None if self._last_result is None else self._last_result.last_radar_paths
        )
