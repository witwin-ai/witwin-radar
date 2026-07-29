"""The one pulsed reference grid, shared by the three pulsed test files.

Stated once because the twelve numbers below are not independent: four
inequalities tie them together (Nyquist for the correlation, the gate inside
the PRI, the whole echo inside the gate, and the speed inside the unambiguous
bound), and restating them per file is how one of the four quietly stops
holding. ``test_phase6_pulsed_spec.py`` asserts all four.

This grid is NOT the one the Phase-6 brief sketched, and the reasons are
recorded rather than quietly fixed:

* ``fs = 5 MSPS`` cannot carry a ``B = 20 MHz`` LFM. Complex baseband needs
  ``fs > B`` merely to represent the pulse, and ``fs > 2 B`` for the DISCRETE
  matched-filter sum to equal the continuous integral, because the integrand
  ``y conj(p)`` occupies ``[-B, B]``. At 5 MSPS the pulse is aliased four times
  over and no delay estimate means anything. This grid uses
  ``fs = 50 MSPS = 2.5 B``.
* ``t_g + M T_s`` at 512 samples and 5 MSPS is 102.4 us, which OVERRUNS the
  100 us PRI: the sketch fails its own gate check. At 50 MSPS the gate needs
  ``M >= 601`` to hold a 10 us pulse arriving at 2 us, so ``M`` is 1024 and the
  gate closes at 20.48 us.
* 12 m/s exceeds ``lambda / (4 T_pri) = 9.73 m/s`` at this PRI, so it aliases
  and no slow-time slope assertion could survive it. The radial speed is 5 m/s
  and the aliasing case is tested on purpose, at 1.05x the bound.
"""

from __future__ import annotations

import torch

from witwin.radar.synthesis.assembly import (
    PULSE_KIND_LFM,
    PULSE_KIND_RECT,
    SPEED_OF_LIGHT_M_PER_S,
    PulsedSpec,
)


C0 = SPEED_OF_LIGHT_M_PER_S

F_REF_HZ = 77.0e9
SAMPLE_RATE_HZ = 50.0e6
SAMPLE_PERIOD_S = 1.0 / SAMPLE_RATE_HZ
NUM_SAMPLES = 1024
PRI_S = 100.0e-6
NUM_PULSES = 32
PULSE_WIDTH_S = 10.0e-6
BANDWIDTH_HZ = 20.0e6
RANGE_GATE_START_S = 0.0

RANGE_M = 300.0
TAU_RT_S = 2.0 * RANGE_M / C0
RADIAL_SPEED_MPS = 5.0
TAU_RATE = 2.0 * RADIAL_SPEED_MPS / C0

#: A delay that lands EXACTLY on the sample grid. The sampled matched filter is
#: exact there - the peak is the coefficient to float precision - so it is where
#: identities are asserted. Off the grid the correlation loses O(T_s / T_p) to
#: the partial samples at the pulse's two ends, which is the straddle cost of a
#: sampled receiver and is asserted separately.
ON_GRID_SAMPLE = 100
ON_GRID_TAU_S = ON_GRID_SAMPLE * SAMPLE_PERIOD_S


def reference_spec(**overrides) -> PulsedSpec:
    """The shared grid at the PRODUCTION carrier placement, LFM pulse."""

    fields = dict(
        num_pulses=NUM_PULSES,
        num_samples=NUM_SAMPLES,
        sample_period_s=SAMPLE_PERIOD_S,
        pri_s=PRI_S,
        range_gate_start_s=RANGE_GATE_START_S,
        pulse_kind=PULSE_KIND_LFM,
        pulse_width_s=PULSE_WIDTH_S,
        bandwidth_hz=BANDWIDTH_HZ,
        reference_frequency_hz=F_REF_HZ,
        max_expected_delay_rate=abs(TAU_RATE),
        carrier_hz=0.0,
        carrier_rate_hz=F_REF_HZ,
    )
    fields.update(overrides)
    return PulsedSpec(**fields)


def rect_spec(**overrides) -> PulsedSpec:
    """The rectangular pulse on the same grid.

    ``bandwidth_hz = 1 / T_p`` is the rectangle's own matched-filter bandwidth
    and is what this spec requires a rectangular pulse to declare, so that the
    range cell and the migration bound mean the same thing for both kinds.
    """

    fields = dict(pulse_kind=PULSE_KIND_RECT, bandwidth_hz=1.0 / PULSE_WIDTH_S)
    fields.update(overrides)
    return reference_spec(**fields)


def stored(value: float) -> float:
    """The float32 value a kernel actually sees.

    ``total_delay_s`` is float32 by contract, so rounding it costs up to half an
    ulp, about 6e-8 relative. That is a property of the CONTRACT, not of the
    kernel, and separating the two is what lets the kernel be asserted tighter
    than its input can be represented.
    """

    return float(torch.tensor([value], dtype=torch.float32))


def peak_estimate(cube_row: torch.Tensor, spec, *, oversample: int = 64):
    """Matched-filter peak of one ``[num_samples]`` row: (delay, value, magnitude).

    The delay comes from a three-point parabolic fit on the band-limited
    interpolated magnitude, and it is returned IN SECONDS measured from time
    zero, not as a bin index. Its measured accuracy on this grid is 1.1e-10 s -
    about 0.002 of a range cell - and that floor is a property of fitting a
    parabola to a cusped ``|R|``: it does not shrink with more oversampling and
    it does not shrink with a faster ADC. Both were measured.
    """

    from witwin.radar.processing.range_doppler import lag_axis, matched_filter

    compressed = matched_filter(cube_row, spec, oversample=oversample)
    lags = lag_axis(spec, oversample=oversample)
    magnitude = compressed.abs()
    peak = int(magnitude.argmax())
    left = float(magnitude[peak - 1])
    centre = float(magnitude[peak])
    right = float(magnitude[peak + 1])
    step = float(lags[1] - lags[0])
    offset = 0.5 * (left - right) / (left - 2.0 * centre + right)
    return float(lags[peak]) + offset * step, complex(compressed[peak]), centre


__all__ = [
    "BANDWIDTH_HZ",
    "C0",
    "F_REF_HZ",
    "NUM_PULSES",
    "NUM_SAMPLES",
    "ON_GRID_SAMPLE",
    "ON_GRID_TAU_S",
    "PRI_S",
    "PULSE_WIDTH_S",
    "RADIAL_SPEED_MPS",
    "RANGE_GATE_START_S",
    "RANGE_M",
    "SAMPLE_PERIOD_S",
    "SAMPLE_RATE_HZ",
    "TAU_RATE",
    "TAU_RT_S",
    "peak_estimate",
    "rect_spec",
    "reference_spec",
    "stored",
]
