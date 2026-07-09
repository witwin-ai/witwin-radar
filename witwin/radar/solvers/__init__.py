"""Solver base class and Dirichlet implementation for radar signal computation."""


class Solver:
    """Abstract base class for radar chirp/frame/MIMO computation.

    Subclasses implement the actual signal generation.
    """

    def __init__(self, radar):
        self.radar = radar
        self.device = radar.device

    def _pop_bool_option(self, options, name: str, default: bool = False) -> bool:
        value = options.pop(name, default)
        if not isinstance(value, bool):
            raise TypeError(f"Solver option '{name}' must be a bool.")
        return value

    def _ensure_no_options(self, options) -> None:
        if options:
            unsupported = ", ".join(sorted(options))
            raise TypeError(f"Unsupported solver options: {unsupported}")

    def chirp(self, distances, amplitudes):
        """Compute one chirp sweep.

        Args:
            distances: (N,) one-way range to each target (meters)
            amplitudes: (N,) reflectance / intensity per target

        Returns:
            Beat signal or spectrum, shape depends on backend.
        """
        raise NotImplementedError

    def frame(self, interpolator, t0=0):
        """Compute a single TX-RX pair frame.

        Args:
            interpolator: callable(t) -> (intensities, positions)
            t0: frame start time (seconds)

        Returns:
            (chirps_per_frame, adc_samples) complex tensor
        """
        raise NotImplementedError

    def mimo(self, interpolator, t0=0, **options):
        """Compute a full MIMO data cube.

        Args:
            interpolator: callable(t) -> (intensities, positions)
            t0: frame start time (seconds)

        Returns:
            (TX, RX, chirps_per_frame, adc_samples) complex tensor
        """
        raise NotImplementedError


# Import after Solver is defined to avoid circular imports
from .solver_dirichlet import DirichletSolver

__all__ = [
    'Solver',
    'DirichletSolver',
]
