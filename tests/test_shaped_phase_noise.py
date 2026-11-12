"""The three-region oscillator: 1/f^3 flicker FM, 1/f^2 white FM, white floor.

Each region is declared by its own single-sideband level at its own offset and
measured here with the independent Welch estimator in
``support.reference_frontend`` on a uniform grid of the standalone oscillator.
The asserted levels are the declared ``L(f)`` read off the textbook power
laws; nothing is imported from the generator's own calibration.

Also here: the invalid-query counter that replaced a device assertion. A bad
query must raise on the host and leave the CUDA context usable.
"""

from __future__ import annotations

import math
import subprocess
import sys
import textwrap

import pytest
import torch
from support.reference_frontend import single_sideband_psd

from witwin.radar.frontend import Noise

pytestmark = pytest.mark.gpu

RATE = 10e6


def _phase(noise: Noise, count: int, *, seed: int = 5) -> torch.Tensor:
    times = torch.arange(count, dtype=torch.float64, device="cuda") / RATE + 0.25
    return noise.phase_difference(times, times - times[0], seed_base=seed)


def _band_level_dbc(frequencies, psd, low: float, exponent: float) -> float:
    """The octave-band average of ``psd * f^exponent``, in dB: a level referred to 1 Hz."""

    band = (frequencies >= low) & (frequencies < 2 * low)
    return 10 * math.log10(float((psd[band] * frequencies[band] ** exponent).mean()))


def test_the_flicker_region_falls_at_thirty_db_per_decade_at_the_declared_level():
    """M4-8 / FE-4: ``L(f) = L3 (f3 / f)^3``, within 1 dB per octave band."""

    level, offset = -60.0, 1e4
    noise = Noise(phase_flicker_density=level, phase_flicker_offset=offset).resolved(bandwidth=RATE, sample_rate=RATE)
    frequencies, psd = single_sideband_psd(_phase(noise, 1 << 20), sample_rate_hz=RATE, segment=1 << 16)
    expected = level + 30 * math.log10(offset)
    for low in (2e3, 8e3, 3.2e4, 1.28e5):
        assert _band_level_dbc(frequencies, psd, low, 3.0) == pytest.approx(expected, abs=1.0), low


def test_the_floor_is_flat_at_the_declared_level():
    floor = -130.0
    noise = Noise(phase_floor=floor).resolved(bandwidth=RATE, sample_rate=RATE)
    frequencies, psd = single_sideband_psd(_phase(noise, 1 << 18), sample_rate_hz=RATE, segment=1 << 12)
    for low in (1e4, 1e5, 1e6):
        assert _band_level_dbc(frequencies, psd, low, 0.0) == pytest.approx(floor, abs=0.5), low


def test_the_composite_mask_is_the_sum_of_the_three_regions():
    """Each region dominates where the mask says it should, at its own level."""

    noise = Noise(
        phase_density=-100.0,
        phase_offset=1e5,
        phase_flicker_density=-60.0,
        phase_flicker_offset=1e4,
        phase_floor=-125.0,
    ).resolved(bandwidth=RATE, sample_rate=RATE)
    frequencies, psd = single_sideband_psd(_phase(noise, 1 << 20), sample_rate_hz=RATE, segment=1 << 16)

    def mask(f):
        # The white-FM term is the SAMPLED random walk's exact density, which
        # rises above q / (4 pi^2 f^2) by (pi f T / sin(pi f T))^2 near Nyquist.
        aliasing = (math.pi * f / RATE / torch.sin(math.pi * f / RATE)) ** 2
        return 10 ** (-6.0) * (1e4 / f) ** 3 + 10 ** (-10.0) * (1e5 / f) ** 2 * aliasing + 10 ** (-12.5)

    for low in (2e3, 5e4, 2e6):
        band = (frequencies >= low) & (frequencies < 2 * low)
        measured = float(psd[band].mean())
        expected = float(mask(frequencies[band]).mean())
        assert 10 * math.log10(measured / expected) == pytest.approx(0.0, abs=1.0), low


def test_the_white_fm_region_is_bit_identical_with_or_without_the_other_regions_absent():
    """Declaring no flicker and no floor reproduces the Wiener-only realisation exactly."""

    plain = Noise(phase_density=-90.0, phase_offset=1e5).resolved(bandwidth=RATE, sample_rate=RATE)
    explicit = Noise(phase_density=-90.0, phase_offset=1e5, phase_floor=None).resolved(bandwidth=RATE, sample_rate=RATE)
    assert torch.equal(_phase(plain, 4096), _phase(explicit, 4096))


def test_the_floor_is_range_correlated_only_within_one_hold_cell():
    """``var[phi(t) - phi(t - tau)] = 2 s^2 min(tau f_cell, 1)`` for the white floor."""

    noise = Noise(phase_floor=-120.0).resolved(bandwidth=RATE, sample_rate=RATE)
    sigma_sq = 10 ** (-12.0) * RATE
    # An irrational spacing in cells, so the query's position inside its cell is equidistributed.
    times = torch.arange(1 << 18, dtype=torch.float64, device="cuda") / RATE * math.pi + 1.0
    for cells in (0.25, 0.5, 2.0):
        delay = torch.full_like(times, cells / RATE)
        variance = float(noise.phase_difference(times, delay, seed_base=3).double().var())
        assert variance == pytest.approx(2 * sigma_sq * min(cells, 1.0), rel=0.05), cells


def test_an_invalid_oscillator_query_raises_on_the_host_and_leaves_the_context_usable():
    """A-7: the old device assertion poisoned the CUDA context; the counter does not."""

    script = textwrap.dedent(
        """
        import torch
        from witwin.radar.frontend import Noise

        noise = Noise(phase_density=-90.0, phase_offset=1e5).resolved(bandwidth=1e6, sample_rate=1e6)
        times = torch.tensor([1e-3, 2e-3, float("nan")], dtype=torch.float64, device="cuda")
        delays = torch.tensor([1e-7, -1e-7, 1e-7], dtype=torch.float64, device="cuda")
        try:
            noise.phase_difference(times, delays, seed_base=1)
        except ValueError as error:
            assert "2 oscillator queries" in str(error), error
        else:
            raise AssertionError("an invalid query was accepted")
        # The context is intact: ordinary work and a valid query still run.
        assert float((torch.ones(4, device="cuda") * 2).sum()) == 8.0
        good = noise.phase_difference(times[:1], delays[:1], seed_base=1)
        assert bool(torch.isfinite(good).all())
        print("ok")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=300, cwd=str(_repository_root())
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")


def _repository_root():
    from pathlib import Path

    return Path(__file__).resolve().parents[1]
