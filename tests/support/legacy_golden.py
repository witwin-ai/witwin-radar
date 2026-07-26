"""The inputs every legacy ``sigproc`` entry is measured on, built once.

This module is the SAFETY NET of the Phase-8 cutover. It defines - and nothing
else - the exact inputs each public ``sigproc`` name was fed before the old
internal paths were deleted. The outputs were captured from the pre-cutover
tree and committed as ``tests/goldens/legacy_sigproc.pt``;
``tests/processing/test_adapters.py`` replays these inputs through the migration
adapters and compares.

It imports NO production module, so the same file runs against the pre-cutover
tree and against the post-cutover one. Every tensor is built from a seeded
generator on the CPU, so the goldens are reproducible.
"""

from __future__ import annotations

import torch

from conftest import STANDARD_CONFIG


#: Small enough to keep the golden file in kilobytes, large enough that a CFAR
#: ring and a 64-point angle FFT are exercised on real data.
GOLDEN_CONFIG = {
    **STANDARD_CONFIG,
    "num_tx": 3,
    "num_rx": 4,
    "adc_start_time": 0,
    "adc_samples": 64,
    "chirp_per_frame": 16,
    "num_doppler_bins": 16,
    "num_range_bins": 64,
    "num_angle_bins": 64,
    "tx_loc": [[0, 0, 0], [2, 0, 0], [0, 1, 0]],
    "rx_loc": [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]],
}

#: The second front end. ``naive_xyz`` dispatches to the two-dimensional route
#: only when ``num_tx > 4``, so the 3 x 4 array above cannot reach it at all.
GOLDEN_CONFIG_2D = {
    **GOLDEN_CONFIG,
    "num_tx": 6,
    "tx_loc": [[0, 0, 0], [2, 0, 0], [0, 1, 0], [2, 1, 0], [0, 2, 0], [2, 2, 0]],
}

SEED = 20260726


def _complex(shape, *, seed: int, dtype=torch.complex64) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    real = torch.randn(shape, generator=generator, dtype=real_dtype)
    imaginary = torch.randn(shape, generator=generator, dtype=real_dtype)
    return torch.complex(real, imaginary)


def frame(config=GOLDEN_CONFIG) -> torch.Tensor:
    """``[TX, RX, chirp, sample]`` complex64, with one strong on-bin target.

    Noise alone would make every detector's mask a coin flip on the last bit of
    a threshold; a dominant target makes the golden stable while the noise floor
    still exercises the ring.
    """

    shape = (
        int(config["num_tx"]),
        int(config["num_rx"]),
        int(config["chirp_per_frame"]),
        int(config["adc_samples"]),
    )
    data = _complex(shape, seed=SEED) * 0.05
    samples = torch.arange(shape[3], dtype=torch.float32)
    chirps = torch.arange(shape[2], dtype=torch.float32)
    tone = torch.exp(2j * torch.pi * 11.0 * samples / shape[3]).to(torch.complex64)
    walk = torch.exp(2j * torch.pi * 3.0 * chirps / shape[2]).to(torch.complex64)
    return data + walk.reshape(1, 1, -1, 1) * tone.reshape(1, 1, 1, -1)


def virtual_antenna(*, num_tx: int, num_rx: int, seed: int = SEED + 1) -> torch.Tensor:
    """``[P, N]`` complex64 in the TX-major virtual-antenna order."""

    return _complex((num_tx * num_rx, 5), seed=seed)


def velocities(count: int = 5) -> torch.Tensor:
    """Non-zero on purpose: TDM compensation is the identity at ``v = 0``."""

    return torch.tensor([-3.0, -1.25, 0.0, 1.25, 3.0], dtype=torch.float64)[:count]


def rd_magnitude(seed: int = SEED + 2) -> torch.Tensor:
    """``[32, 48]`` real, noise plus three peaks: the CFAR input."""

    generator = torch.Generator().manual_seed(seed)
    values = torch.abs(torch.randn((32, 48), generator=generator, dtype=torch.float32))
    for row, column in ((6, 9), (16, 24), (25, 40)):
        values[row, column] = 40.0
    return values


def angle_data(*, rows: int = 6, columns: int = 6, bins: int = 3, snapshots: int = 4):
    """``[B, M, N, T]`` complex64: the MUSIC pseudo-spectrum input."""

    return _complex((bins, rows, columns, snapshots), seed=SEED + 3)


def music_frame(*, rows: int = 6, columns: int = 6) -> torch.Tensor:
    """``[TX, RX, chirps, samples]`` complex64: the MUSIC image input."""

    return _complex((rows, columns, 8, 32), seed=SEED + 4)


def music_range_bins() -> torch.Tensor:
    """Explicit, so the golden does not depend on an auto-detected peak."""

    return torch.tensor([5, 6, 7], dtype=torch.int64)


def slow_time(seed: int = SEED + 5) -> torch.Tensor:
    """``[2, 64]`` complex64: the micro-Doppler input."""

    return _complex((2, 64), seed=seed)


def point_columns(seed: int = SEED + 6) -> torch.Tensor:
    """``[7, 6]`` float32: the ``reg_data`` input, in the legacy column order."""

    generator = torch.Generator().manual_seed(seed)
    directions = torch.randn((7, 3), generator=generator, dtype=torch.float64)
    directions = directions / directions.square().sum(dim=1, keepdim=True).sqrt()
    ranges = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=torch.float64)
    return torch.cat(
        (
            directions * ranges.reshape(-1, 1),
            torch.linspace(-2.0, 2.0, 7, dtype=torch.float64).reshape(-1, 1),
            torch.linspace(10.0, 40.0, 7, dtype=torch.float64).reshape(-1, 1),
            ranges.reshape(-1, 1),
        ),
        dim=1,
    ).to(torch.float32)


__all__ = [
    "GOLDEN_CONFIG",
    "GOLDEN_CONFIG_2D",
    "SEED",
    "angle_data",
    "frame",
    "music_frame",
    "music_range_bins",
    "point_columns",
    "rd_magnitude",
    "slow_time",
    "velocities",
    "virtual_antenna",
]
