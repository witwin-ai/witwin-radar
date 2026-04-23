"""Unified FMCW radar with selectable compute backend."""

from __future__ import annotations

import math

import torch

from .antenna_pattern import AntennaPatternRuntime
from .config import RadarConfig
from .noise import NoiseModelRuntime
from .polarization import PolarizationRuntime
from .receiver_chain import ReceiverChainRuntime
from .sensor import Sensor
from .types import SolverBackend
from .validation import default_dipole_antenna_pattern


class Radar:
    def __init__(
        self,
        config: RadarConfig,
        backend: SolverBackend = "dirichlet",
        pad_factor: int = 16,
        device: str = "cuda",
        sensor: Sensor | None = None,
    ):
        """
        Args:
            config: ``RadarConfig``. Use ``RadarConfig.from_dict`` or ``RadarConfig.from_json``
                to build one from raw sources.
            backend: "dirichlet" | "slang" | "pytorch"
            pad_factor: FFT zero-padding factor for the Dirichlet backend
            device: compute device for public tensors and PyTorch execution
            sensor: radar pose in world coordinates
        """
        self.c0 = 299792458
        self.backend = SolverBackend(backend)
        self.device = self._resolve_device(device=self._normalize_device(device), backend=self.backend)
        self.sensor = sensor or Sensor.identity()

        self.config: RadarConfig = config
        cfg = self.config

        if (
            cfg.receiver_chain is not None
            and cfg.receiver_chain.adc is not None
            and cfg.noise_model is not None
            and cfg.noise_model.quantization is not None
        ):
            raise ValueError(
                "Radar receiver_chain.adc and noise_model.quantization cannot both be enabled; use one quantizer."
            )

        antenna_spacing = self.c0 / cfg.fc / 2
        tx_loc = torch.tensor(cfg.tx_loc, dtype=torch.float32, device=self.device) * antenna_spacing
        rx_loc = torch.tensor(cfg.rx_loc, dtype=torch.float32, device=self.device) * antenna_spacing
        self.tx_loc = tx_loc
        self.rx_loc = rx_loc
        self.tx_pos = self.world_from_local_points(tx_loc).contiguous()
        self.rx_pos = self.world_from_local_points(rx_loc).contiguous()

        self.sensor_origin = torch.tensor(self.sensor.origin, dtype=torch.float32, device=self.device)

        self.t_sample = (
            torch.arange(0, cfg.adc_samples, dtype=torch.float64, device=self.device)
            / (cfg.sample_rate * 1e3)
            + cfg.adc_start_time * 1e-6
        )
        self.tx_waveform = self.waveform(self.t_sample)
        self._lambda = self.c0 / cfg.fc

        self.transmit_power_watts = 1e-3 * (10.0 ** (cfg.power / 10.0))
        reference_impedance = (
            cfg.receiver_chain.reference_impedance_ohm if cfg.receiver_chain is not None else 50.0
        )
        self.tx_voltage_rms = math.sqrt(self.transmit_power_watts * reference_impedance)
        self.gain = self.tx_voltage_rms if cfg.receiver_chain is not None else 1.0

        self.antenna_pattern_config = cfg.antenna_pattern or default_dipole_antenna_pattern()
        self.antenna_pattern = AntennaPatternRuntime.from_config(self.antenna_pattern_config, device=self.device)
        self.noise_model_config = cfg.noise_model
        self.noise_model = (
            NoiseModelRuntime.from_config(cfg.noise_model, device=self.device)
            if cfg.noise_model is not None
            else None
        )
        self.polarization_config = cfg.polarization
        self.polarization = (
            PolarizationRuntime.from_config(cfg.polarization, device=self.device, sensor=self.sensor)
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

        if self.backend == SolverBackend.PYTORCH:
            from .solvers.solver_pytorch import PytorchSolver

            self.solver = PytorchSolver(self)
        elif self.backend == SolverBackend.SLANG:
            from .solvers.solver_slang import SlangSolver

            self.solver = SlangSolver(self)
        else:
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
        if device.type != "cuda" and backend in {SolverBackend.SLANG, SolverBackend.DIRICHLET}:
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
        if self.config.noise_model is None or self.config.noise_model.seed is None:
            return None
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.config.noise_model.seed)
        return generator

    # ---- Convenience accessors mirroring config fields (no duplicated state) ----

    @property
    def num_tx(self) -> int:
        return self.config.num_tx

    @property
    def num_rx(self) -> int:
        return self.config.num_rx

    @property
    def fc(self) -> float:
        return self.config.fc

    @property
    def slope(self) -> float:
        return self.config.slope

    @property
    def adc_samples(self) -> int:
        return self.config.adc_samples

    @property
    def adc_start_time(self) -> float:
        return self.config.adc_start_time

    @property
    def sample_rate(self) -> float:
        return self.config.sample_rate

    @property
    def idle_time(self) -> float:
        return self.config.idle_time

    @property
    def ramp_end_time(self) -> float:
        return self.config.ramp_end_time

    @property
    def chirp_per_frame(self) -> int:
        return self.config.chirp_per_frame

    @property
    def frame_per_second(self) -> float:
        return self.config.frame_per_second

    @property
    def num_doppler_bins(self) -> int:
        return self.config.num_doppler_bins

    @property
    def num_range_bins(self) -> int:
        return self.config.num_range_bins

    @property
    def num_angle_bins(self) -> int:
        return self.config.num_angle_bins

    @property
    def power(self) -> float:
        return self.config.power

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
