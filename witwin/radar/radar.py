"""Unified FMCW radar using the native Dirichlet CUDA solver."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from .types import MotionSampling, SamplingMode
from .utils.antenna import evaluate_antenna_pattern_vectors, evaluate_antenna_pattern_xy
from .utils.vector import vec3_tensor


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
    noise_model: dict[str, Any] | None = None
    polarization: dict[str, Any] | None = None
    receiver_chain: dict[str, Any] | None = None

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


def _component_dtype(signal: torch.Tensor) -> torch.dtype:
    return torch.float64 if signal.dtype == torch.complex128 else torch.float32


def _randn(shape, *, device, dtype, generator: torch.Generator | None) -> torch.Tensor:
    if generator is None:
        return torch.randn(shape, device=device, dtype=dtype)
    return torch.randn(shape, device=device, dtype=dtype, generator=generator)


def _normalize_rows(vectors: torch.Tensor) -> torch.Tensor:
    return vectors / torch.clamp(torch.linalg.norm(vectors, dim=-1, keepdim=True), min=1e-12)


def quantize_complex_signal(signal: torch.Tensor, *, bits: int, full_scale: float) -> torch.Tensor:
    levels = 2 ** bits
    step = (2.0 * full_scale) / (levels - 1)

    def _quantize(component: torch.Tensor) -> torch.Tensor:
        clipped = torch.clamp(component, min=-full_scale, max=full_scale)
        code = torch.round((clipped + full_scale) / step)
        return code * step - full_scale

    real = _quantize(signal.real)
    imag = _quantize(signal.imag)
    return torch.complex(real, imag).to(dtype=signal.dtype)


def db_to_voltage_gain(gain_db: float) -> float:
    return 10.0 ** (float(gain_db) / 20.0)


class ReceiverChainRuntime:
    def __init__(self, config: dict[str, Any], *, device: str | torch.device):
        self.config = config
        self.device = device

    @classmethod
    def from_config(cls, config: dict[str, Any], *, device: str | torch.device) -> "ReceiverChainRuntime":
        return cls(config=config, device=device)

    def apply(self, signal: torch.Tensor) -> torch.Tensor:
        processed = signal
        lna = self.config.get("lna")
        agc = self.config.get("agc")
        adc = self.config.get("adc")
        if lna is not None:
            processed = processed * db_to_voltage_gain(lna["gain_db"])
        if agc is not None:
            processed = self._apply_agc(processed, agc)
        if adc is not None:
            processed = quantize_complex_signal(
                processed,
                bits=adc["bits"],
                full_scale=adc["full_scale"],
            )
        return processed

    def _apply_agc(self, signal: torch.Tensor, config: dict) -> torch.Tensor:
        real_dtype = signal.real.dtype
        magnitude_sq = signal.real.square() + signal.imag.square()

        if signal.ndim == 4 and config["mode"] == "per_rx":
            rms = torch.sqrt(torch.clamp(magnitude_sq.mean(dim=(0, 2, 3), keepdim=True), min=1e-24))
            target = torch.tensor(config["target_rms"], dtype=real_dtype, device=signal.device).view(1, 1, 1, 1)
        else:
            rms = torch.sqrt(torch.clamp(magnitude_sq.mean(), min=1e-24))
            target = torch.tensor(config["target_rms"], dtype=real_dtype, device=signal.device)

        gain = target / rms
        min_gain = db_to_voltage_gain(config["min_gain_db"])
        max_gain = db_to_voltage_gain(config["max_gain_db"])
        gain = torch.clamp(gain, min=min_gain, max=max_gain)
        return signal * gain.to(dtype=signal.dtype)


class NoiseModelRuntime:
    def __init__(self, config: dict[str, Any], *, device: str | torch.device):
        self.config = config
        self.device = device

    @classmethod
    def from_config(cls, config: dict[str, Any], *, device: str | torch.device) -> "NoiseModelRuntime":
        return cls(config=config, device=device)

    def apply(self, signal: torch.Tensor, *, generator: torch.Generator | None = None) -> torch.Tensor:
        noisy = signal
        phase = self.config.get("phase")
        thermal = self.config.get("thermal")
        quantization = self.config.get("quantization")
        if phase is not None and phase["std"] > 0.0:
            noisy = self._apply_phase_noise(noisy, std=phase["std"], generator=generator)
        if thermal is not None and thermal["std"] > 0.0:
            noisy = self._apply_thermal_noise(noisy, std=thermal["std"], generator=generator)
        if quantization is not None:
            noisy = quantize_complex_signal(
                noisy,
                bits=quantization["bits"],
                full_scale=quantization["full_scale"],
            )
        return noisy

    def _apply_phase_noise(
        self,
        signal: torch.Tensor,
        *,
        std: float,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        real = _component_dtype(signal)
        if signal.ndim == 4:
            phase_shape = signal.shape[-2:]
            broadcast_shape = (1, 1, *phase_shape)
        elif signal.ndim in (1, 2):
            phase_shape = signal.shape
            broadcast_shape = phase_shape
        else:
            raise ValueError("Phase noise currently supports chirp (T,), frame (F, T), or mimo (TX, RX, F, T) tensors.")

        innovations = _randn(phase_shape, device=signal.device, dtype=real, generator=generator) * std
        phase = torch.cumsum(innovations.reshape(-1), dim=0).reshape(phase_shape).reshape(broadcast_shape)
        phase_factor = torch.polar(torch.ones_like(phase, dtype=real), phase)
        return signal * phase_factor.to(dtype=signal.dtype)

    def _apply_thermal_noise(
        self,
        signal: torch.Tensor,
        *,
        std: float,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        real = _component_dtype(signal)
        real_part = _randn(signal.shape, device=signal.device, dtype=real, generator=generator) * std
        imag_part = _randn(signal.shape, device=signal.device, dtype=real, generator=generator) * std
        noise = torch.complex(real_part, imag_part).to(dtype=signal.dtype)
        return signal + noise


class PolarizationRuntime:
    def __init__(
        self,
        *,
        tx_world: torch.Tensor,
        rx_world: torch.Tensor,
        reflection_flip: bool = True,
    ) -> None:
        self.tx_world = tx_world
        self.rx_world = rx_world
        self.reflection_flip = bool(reflection_flip)

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        device: str | torch.device,
        radar,
    ) -> "PolarizationRuntime":
        tx_local = _normalize_rows(torch.tensor(config["tx"], dtype=torch.float32, device=device))
        rx_local = _normalize_rows(torch.tensor(config["rx"], dtype=torch.float32, device=device))
        tx_world = _normalize_rows(radar.world_from_local_vectors(tx_local))
        rx_world = _normalize_rows(radar.world_from_local_vectors(rx_local))
        return cls(
            tx_world=tx_world.contiguous(),
            rx_world=rx_world.contiguous(),
            reflection_flip=bool(config.get("reflection_flip", True)),
        )


class Radar:
    def __init__(
        self,
        config: RadarConfig | Mapping[str, Any],
        pad_factor: int = 16,
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
            pad_factor: FFT zero-padding factor for the Dirichlet backend
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

        self._validate_runtime_config(cfg)
        self._init_antenna_locations(cfg)
        self._init_waveform_state(cfg)
        self._init_rf_state(cfg)
        self._init_runtime_models(cfg)
        self._init_axes(cfg)
        self.solver = self._make_solver(pad_factor)

    @staticmethod
    def _validate_runtime_config(cfg: RadarConfig) -> None:
        if (
            cfg.receiver_chain is not None
            and cfg.receiver_chain.get("adc") is not None
            and cfg.noise_model is not None
            and cfg.noise_model.get("quantization") is not None
        ):
            raise ValueError(
                "Radar receiver_chain.adc and noise_model.quantization cannot both be enabled; use one quantizer."
            )

    def _init_antenna_locations(self, cfg: RadarConfig) -> None:
        antenna_spacing = self.c0 / cfg.fc / 2
        self.tx_loc = torch.tensor(cfg.tx_loc, dtype=torch.float32, device=self.device) * antenna_spacing
        self.rx_loc = torch.tensor(cfg.rx_loc, dtype=torch.float32, device=self.device) * antenna_spacing
        self._refresh_pose_dependent_state()

    def _init_waveform_state(self, cfg: RadarConfig) -> None:
        self.t_sample = (
            torch.arange(0, cfg.adc_samples, dtype=torch.float64, device=self.device)
            / (cfg.sample_rate * 1e3)
            + cfg.adc_start_time * 1e-6
        )
        self.tx_waveform = self.waveform(self.t_sample)
        self._lambda = self.c0 / cfg.fc

    def _init_rf_state(self, cfg: RadarConfig) -> None:
        self.transmit_power_watts = 1e-3 * (10.0 ** (cfg.power / 10.0))
        reference_impedance = (
            cfg.receiver_chain["reference_impedance_ohm"] if cfg.receiver_chain is not None else 50.0
        )
        self.tx_voltage_rms = math.sqrt(self.transmit_power_watts * reference_impedance)
        self.gain = self.tx_voltage_rms if cfg.receiver_chain is not None else 1.0

    def _init_runtime_models(self, cfg: RadarConfig) -> None:
        from .validation import default_dipole_antenna_pattern

        self.antenna_pattern_config = cfg.antenna_pattern or default_dipole_antenna_pattern()
        self._build_antenna_pattern_runtime(self.antenna_pattern_config)
        self.noise_model_config = cfg.noise_model
        self.noise_model = (
            NoiseModelRuntime.from_config(cfg.noise_model, device=self.device)
            if cfg.noise_model is not None
            else None
        )
        self.polarization_config = cfg.polarization
        self.polarization = (
            PolarizationRuntime.from_config(cfg.polarization, device=self.device, radar=self)
            if cfg.polarization is not None
            else None
        )
        self.receiver_chain_config = cfg.receiver_chain
        self.receiver_chain = (
            ReceiverChainRuntime.from_config(cfg.receiver_chain, device=self.device)
            if cfg.receiver_chain is not None
            else None
        )
        self._noise_generator = self._make_noise_generator()

    def _init_axes(self, cfg: RadarConfig) -> None:
        fs = cfg.sample_rate * 1e3
        slope_hz = cfg.slope * 1e12

        self.range_resolution = self.c0 * fs / (2 * slope_hz * cfg.adc_samples)
        self.max_range = self.c0 * fs / (2 * slope_hz)
        self.ranges = (
            torch.arange(0, cfg.num_range_bins // 2, dtype=torch.float64, device=self.device)
            * self.range_resolution
        )

        chirp_period = (cfg.idle_time + cfg.ramp_end_time) * 1e-6
        effective_period = chirp_period * cfg.num_tx
        self.doppler_resolution = self._lambda / (2 * cfg.num_doppler_bins * effective_period)
        self.max_doppler = self._lambda / (4 * chirp_period * cfg.num_tx)
        self.velocities = (
            torch.arange(
                -cfg.num_doppler_bins // 2,
                cfg.num_doppler_bins // 2,
                dtype=torch.float64,
                device=self.device,
            )
            * self.doppler_resolution
        )

    def _make_solver(self, pad_factor: int):
        from .solvers.solver_dirichlet import DirichletSolver

        return DirichletSolver(self, pad_factor)

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
        if self.polarization_config is not None:
            self.polarization = PolarizationRuntime.from_config(
                self.polarization_config,
                device=self.device,
                radar=self,
            )
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

    def _make_noise_generator(self) -> torch.Generator | None:
        if self.config.noise_model is None or self.config.noise_model.get("seed") is None:
            return None
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.config.noise_model["seed"])
        return generator

    def waveform(self, t, phi=0):
        """FMCW chirp waveform: exp(j * 2pi * (fc*t + 0.5*slope*t^2))."""
        phase = self.config.fc * t + 0.5 * (self.config.slope * 1e12) * t * t
        return torch.exp(1j * (2 * torch.pi * phase + phi))

    def apply_noise(self, signal: torch.Tensor) -> torch.Tensor:
        if self.noise_model is None:
            return signal
        return self.noise_model.apply(signal, generator=self._noise_generator)

    def apply_receiver_chain(self, signal: torch.Tensor) -> torch.Tensor:
        if self.receiver_chain is None:
            return signal
        return self.receiver_chain.apply(signal)

    def apply_signal_models(self, signal: torch.Tensor) -> torch.Tensor:
        modeled = self.apply_noise(signal)
        modeled = self.apply_receiver_chain(modeled)
        return modeled

    def chirp(self, distances, amplitudes):
        """Compute one chirp. distances/amplitudes: (N,) one-way range."""
        signal = self.solver.chirp(distances, amplitudes)
        return self.apply_signal_models(signal)

    def frame(self, interpolator, t0=0):
        """Single TX-RX frame. Returns: (chirps, N_fft // 2) complex range spectra."""
        signal = self.solver.frame(interpolator, t0)
        return self.apply_signal_models(signal)

    def mimo(self, interpolator, t0=0, **options):
        """Full MIMO data cube. Returns: (TX, RX, chirps, adc_samples) complex."""
        if bool(options.get("freq_domain", False)) and (
            self.noise_model is not None or self.receiver_chain is not None
        ):
            raise ValueError(
                "Radar noise_model and receiver_chain only support time-domain mimo output; "
                "omit freq_domain=True."
            )
        signal = self.solver.mimo(interpolator, t0, **options)
        return self.apply_signal_models(signal)

    def mimo_from_trace(self, trace, *, velocities=None, t0=0, **options):
        """Full MIMO data cube from one pre-traced frame.

        This is the fixed-trace fast path: it does not call ray tracing or a
        per-chirp interpolator inside the frame. When ``velocities`` is passed,
        the Dirichlet backend uses a first-order per-path range-rate model; it
        does not update incidence, antenna-pattern, polarization, or occlusion
        terms within the frame.

        Args:
            trace: ``TraceResult`` or legacy ``(intensities, points)`` sample at frame start.
            velocities: optional per-path world velocity tensor with shape ``(N, 3)``.
            t0: frame start time in seconds, used only by fallback backends.
            **options: backend options, including ``freq_domain=True`` for Dirichlet.
        """
        if bool(options.get("freq_domain", False)) and (
            self.noise_model is not None or self.receiver_chain is not None
        ):
            raise ValueError(
                "Radar noise_model and receiver_chain only support time-domain mimo output; "
                "omit freq_domain=True."
            )
        if hasattr(self.solver, "mimo_from_trace"):
            signal = self.solver.mimo_from_trace(
                trace,
                velocities=velocities,
                t0=t0,
                **options,
            )
            return self.apply_signal_models(signal)

        if velocities is None:
            return self.mimo(lambda _t: trace, t0=t0, **options)

        if not hasattr(trace, "points"):
            raise TypeError("mimo_from_trace with velocities requires a TraceResult-like trace object.")

        from .trace_result import TraceResult

        velocity_t = torch.as_tensor(velocities, dtype=torch.float32, device=self.device)
        base_points = trace.points.to(dtype=torch.float32, device=self.device)
        base_entry = trace.entry_points.to(dtype=torch.float32, device=self.device)

        def interpolator(t):
            dt = float(t) - float(t0)
            return TraceResult(
                base_points + velocity_t * dt,
                trace.intensities,
                entry_points=base_entry + velocity_t * dt,
                fixed_path_lengths=trace.fixed_path_lengths,
                depths=trace.depths,
                normals=trace.normals,
            )

        return self.mimo(interpolator, t0=t0, **options)

    def path_cache_from_trace(self, trace, *, velocities=None):
        """Precompute fixed-trace path distances and amplitudes for fast MIMO.

        Optional velocities are converted to first-order one-way range rates at
        the trace pose. The cache intentionally freezes material, antenna, and
        polarization terms within the frame.
        """
        if not hasattr(self.solver, "path_cache_from_trace"):
            raise NotImplementedError("path_cache_from_trace is currently implemented by the Dirichlet backend.")
        return self.solver.path_cache_from_trace(trace, velocities=velocities)

    def mimo_from_paths(self, cache, **options):
        """Full MIMO data cube from a ``MimoPathCache``."""
        if bool(options.get("freq_domain", False)) and (
            self.noise_model is not None or self.receiver_chain is not None
        ):
            raise ValueError(
                "Radar noise_model and receiver_chain only support time-domain mimo output; "
                "omit freq_domain=True."
            )
        if not hasattr(self.solver, "mimo_from_paths"):
            raise NotImplementedError("mimo_from_paths is currently implemented by the Dirichlet backend.")
        signal = self.solver.mimo_from_paths(cache, **options)
        return self.apply_signal_models(signal)

    # The Dr.Jit ray tracer that backed simulate() and simulate_group() has
    # been removed, and so has the scene-driven entry that wrapped it. There is
    # no in-scope replacement with the same signature: propagation is now a
    # frozen-topology contract with the Channel consumer rather than a
    # per-frame retrace, so a shim that quietly picked some other route would
    # return numbers from a different model under the old name.
    _SIMULATE_REPLACEMENT = (
        "Radar.simulate and Radar.simulate_group have been removed with the "
        "Dr.Jit ray tracer. Propagation now goes through the Channel "
        "consumer: build a "
        "witwin.radar.propagation.ChannelPropagationAdapter, freeze each leg "
        "once, reevaluate it per frame, compose the legs with "
        "witwin.radar.paths.TwoWayComposer or DirectComposer, and synthesize "
        "with witwin.radar.synthesis.synthesize_fmcw_beat. A scene-driven "
        "entry point that assembles those steps for a whole Scene is separate "
        "work and does not exist yet; Radar.mimo, mimo_from_trace, "
        "mimo_from_paths, path_cache_from_trace, chirp and frame are "
        "unaffected."
    )

    def simulate(self, *args, **kwargs):
        raise NotImplementedError(self._SIMULATE_REPLACEMENT)

    @classmethod
    def simulate_group(cls, *args, **kwargs):
        raise NotImplementedError(cls._SIMULATE_REPLACEMENT)
