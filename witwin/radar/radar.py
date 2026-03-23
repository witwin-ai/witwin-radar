"""Unified FMCW radar with selectable compute backend."""

from __future__ import annotations

import numpy as np
import torch

from .antenna_pattern import AntennaPatternConfig, AntennaPatternRuntime
from .config import RadarConfig
from .noise import NoiseModelRuntime
from .polarization import PolarizationRuntime
from .receiver_chain import ReceiverChainRuntime
from .sensor import Sensor
from .types import SolverBackend, normalize_solver_backend


class Radar:
    def __init__(
        self,
        config: RadarConfig | dict | str = "config.json",
        backend: SolverBackend = "dirichlet",
        pad_factor: int = 16,
        device: str = "cuda",
        sensor: Sensor | None = None,
    ):
        """
        Args:
            config: RadarConfig, JSON path, or dict
            backend: "dirichlet" | "slang" | "pytorch"
            pad_factor: FFT zero-padding factor for the Dirichlet backend
            device: compute device for public tensors and PyTorch execution
            sensor: radar pose in world coordinates
        """
        self.c0 = 299792458
        self.backend = normalize_solver_backend(backend)
        self.device = self._resolve_device(device=self._normalize_device(device), backend=self.backend)
        self.sensor = sensor or Sensor.identity()

        resolved_config = RadarConfig.coerce(config)
        self.config = resolved_config

        self.num_tx = resolved_config.num_tx
        self.num_rx = resolved_config.num_rx
        self.fc = resolved_config.fc
        self.slope = resolved_config.slope
        self.adc_samples = resolved_config.adc_samples
        self.adc_start_time = resolved_config.adc_start_time
        self.sample_rate = resolved_config.sample_rate
        self.idle_time = resolved_config.idle_time
        self.ramp_end_time = resolved_config.ramp_end_time
        self.chirp_per_frame = resolved_config.chirp_per_frame
        self.frame_per_second = resolved_config.frame_per_second
        self.num_doppler_bins = resolved_config.num_doppler_bins
        self.num_range_bins = resolved_config.num_range_bins
        self.num_angle_bins = resolved_config.num_angle_bins
        self.power = resolved_config.power
        self.antenna_pattern_config = resolved_config.antenna_pattern or AntennaPatternConfig.default_dipole()
        self.noise_model_config = resolved_config.noise_model
        self.polarization_config = resolved_config.polarization
        self.receiver_chain_config = resolved_config.receiver_chain

        if (
            self.receiver_chain_config is not None
            and self.receiver_chain_config.adc is not None
            and self.noise_model_config is not None
            and self.noise_model_config.quantization is not None
        ):
            raise ValueError(
                "Radar receiver_chain.adc and noise_model.quantization cannot both be enabled; use one quantizer."
            )

        antenna_spacing = self.c0 / self.fc / 2
        self.tx_loc = np.array(resolved_config.tx_loc, dtype=np.float32) * antenna_spacing
        self.rx_loc = np.array(resolved_config.rx_loc, dtype=np.float32) * antenna_spacing
        self.tx_pos = self.world_from_local_points(
            torch.tensor(self.tx_loc, dtype=torch.float32, device=self.device)
        ).contiguous()
        self.rx_pos = self.world_from_local_points(
            torch.tensor(self.rx_loc, dtype=torch.float32, device=self.device)
        ).contiguous()

        self.sensor_origin = torch.tensor(self.sensor.origin, dtype=torch.float32, device=self.device)

        self.t_sample = (
            torch.arange(0, self.adc_samples, dtype=torch.float64, device=self.device)
            / (self.sample_rate * 1e3)
            + self.adc_start_time * 1e-6
        )
        self.tx_waveform = self.waveform(self.t_sample)
        self._lambda = self.c0 / self.fc
        self.transmit_power_watts = 1e-3 * (10.0 ** (self.power / 10.0))
        reference_impedance = (
            self.receiver_chain_config.reference_impedance_ohm
            if self.receiver_chain_config is not None
            else 50.0
        )
        self.tx_voltage_rms = float(np.sqrt(self.transmit_power_watts * reference_impedance))
        self.gain = self.tx_voltage_rms if self.receiver_chain_config is not None else 1.0
        self.antenna_pattern = AntennaPatternRuntime.from_config(self.antenna_pattern_config, device=self.device)
        self.noise_model = (
            NoiseModelRuntime.from_config(self.noise_model_config, device=self.device)
            if self.noise_model_config is not None
            else None
        )
        self.polarization = (
            PolarizationRuntime.from_config(self.polarization_config, device=self.device, sensor=self.sensor)
            if self.polarization_config is not None
            else None
        )
        self.receiver_chain = (
            ReceiverChainRuntime.from_config(self.receiver_chain_config, device=self.device)
            if self.receiver_chain_config is not None
            else None
        )
        self._noise_generator = self._make_noise_generator()

        fs = self.sample_rate * 1e3
        slope_hz = self.slope * 1e12

        self.range_resolution = self.c0 * fs / (2 * slope_hz * self.adc_samples)
        self.max_range = self.c0 * fs / (2 * slope_hz)
        self.ranges = (
            torch.arange(0, self.num_range_bins // 2, dtype=torch.float64, device=self.device)
            * self.range_resolution
        )

        chirp_period = (self.idle_time + self.ramp_end_time) * 1e-6
        effective_period = chirp_period * self.num_tx
        self.doppler_resolution = self._lambda / (2 * self.num_doppler_bins * effective_period)
        self.max_doppler = self._lambda / (4 * chirp_period * self.num_tx)
        self.velocities = (
            torch.arange(-self.num_doppler_bins // 2, self.num_doppler_bins // 2, dtype=torch.float64, device=self.device)
            * self.doppler_resolution
        )

        if self.backend == "pytorch":
            from .solvers.solver_pytorch import PytorchSolver

            self.solver = PytorchSolver(self)
        elif self.backend == "slang":
            from .solvers.solver_slang import SlangSolver

            self.solver = SlangSolver(self)
        elif self.backend == "dirichlet":
            from .solvers.solver_dirichlet import DirichletSolver

            self.solver = DirichletSolver(self, pad_factor)

    @staticmethod
    def _normalize_device(device: str) -> torch.device:
        return torch.device(device)

    @staticmethod
    def _resolve_device(*, device: torch.device, backend: SolverBackend) -> str:
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "Radar defaults to CUDA, but torch.cuda.is_available() is False. "
                "Pass device='cpu' only when using the PyTorch backend without CUDA."
            )
        if device.type != "cuda" and backend in {"slang", "dirichlet"}:
            raise ValueError(f"Radar backend '{backend}' requires device='cuda'.")
        return str(device)

    def world_from_local_points(self, points: torch.Tensor) -> torch.Tensor:
        return self.sensor.world_from_local_points(points)

    def world_from_local_vectors(self, vectors: torch.Tensor) -> torch.Tensor:
        return self.sensor.world_from_local_vectors(vectors)

    def local_from_world_points(self, points: torch.Tensor) -> torch.Tensor:
        return self.sensor.local_from_world_points(points)

    def local_from_world_vectors(self, vectors: torch.Tensor) -> torch.Tensor:
        return self.sensor.local_from_world_vectors(vectors)

    def _make_noise_generator(self) -> torch.Generator | None:
        if self.noise_model_config is None or self.noise_model_config.seed is None:
            return None
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.noise_model_config.seed)
        return generator

    def waveform(self, t, phi=0):
        """FMCW chirp waveform: exp(j * 2pi * (fc*t + 0.5*slope*t^2))."""
        phase = self.fc * t + 0.5 * (self.slope * 1e12) * t * t
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
        """Single TX-RX frame. Returns: (chirps, adc_samples) complex."""
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
