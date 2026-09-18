"""One physical target, three waveforms, one range and one signed velocity.

Plan Phase-8 acceptance criterion: the FMCW, OFDM and pulsed range and Doppler
outputs must agree on an analytic target. This file is that agreement, stated
where it can actually be wrong - in METRES and in METRES PER SECOND, read off
each waveform's own :class:`ProcessingAxes`, never off a bin index.

The setup is one composed row of the real multi-endpoint fixture, placed on an
exact bin in all three waveforms by solving the ellipse and the three waveform
grids in closed form (``tests/support/exact_bin_grid.py``). One reevaluation
per frame drives all three, so this is genuinely one target rather than three
fixtures that happen to agree.

The Doppler half is the sharp one. FMCW's beat cube is the CONJUGATE of
Channel's phasor, so its raw slow-time tone sits at ``+f_ref tau_rate`` and the
other two at ``-f_ref tau_rate``. Without the reconciliation the same closing
target reads as approaching on one waveform and receding on another, in a
magnitude plot where the difference is invisible. The target here is
unambiguously CLOSING, so a sign error cannot pass.
"""

from __future__ import annotations

import pytest
import torch
from support import exact_bin_grid as grid
from support import multi_endpoint_driver as drv

from witwin.radar.processing import ProcessingAxes, ProcessingCube, range_doppler_map, range_profile
from witwin.radar.processing.signal import PROCESSING_DOPPLER_CONVENTION
from witwin.radar.synthesis.assembly import SynthesisResult
from witwin.radar.synthesis.fmcw import synthesize_fmcw
from witwin.radar.synthesis.ofdm import synthesize_ofdm
from witwin.radar.synthesis.pulsed import synthesize_pulsed

pytestmark = pytest.mark.gpu


WAVEFORMS = (
    ("fmcw", grid.fmcw_spec, synthesize_fmcw, SynthesisResult.from_fmcw),
    ("ofdm", grid.ofdm_spec, synthesize_ofdm, SynthesisResult.from_ofdm),
    ("pulsed", grid.pulsed_spec, synthesize_pulsed, SynthesisResult.from_pulsed),
)


@pytest.fixture(scope="module")
def spike():
    pytest.importorskip("witwin.channel")
    return grid.make_spike()


@pytest.fixture(scope="module")
def moving(spike):
    """ONE closing target, composed once, synthesized three ways below."""

    composed = grid.moving_frame(spike)
    row = grid.target_row(spike, composed)
    return (grid.isolate(drv.to_synthesis(composed), row), row, int(composed.sensor_pair_index[row]))


def _measure(batch, segment, spec_of, synthesize, maker):
    spec = spec_of()
    result = maker(synthesize(batch, spec), spec)
    axes = ProcessingAxes.from_synthesis(result, spec, grid.array_spec())
    rd = range_doppler_map(range_profile(ProcessingCube.from_synthesis(result, axes)))
    tx = segment % grid.FMCW_NUM_TX
    rx = segment // grid.FMCW_NUM_TX
    magnitude = rd.data[tx, rx].abs()
    flat = int(magnitude.argmax())
    doppler = flat // magnitude.shape[1]
    range_bin = flat % magnitude.shape[1]
    return axes, float(axes.range_m[range_bin]), float(axes.velocity_mps[doppler])


def test_one_target_reads_as_one_range_and_one_velocity_in_all_three_waveforms(moving):
    """The criterion, in SI, with every waveform's own axis doing the conversion.

    The tolerance is half of EACH waveform's own bin, which is the largest error
    a correct peak can have. The measured agreement is far tighter than that -
    the delay and the rate were solved onto exact bins - and both numbers are in
    the assertion message so a run shows how much of the bound is used.
    """

    batch, row, segment = moving
    truth_range = grid.RANGE_M
    truth_speed = grid.CLOSING_SPEED_MPS
    reported = []
    for name, spec_of, synthesize, maker in WAVEFORMS:
        axes, measured_range, measured_speed = _measure(batch, segment, spec_of, synthesize, maker)
        reported.append((name, axes, measured_range, measured_speed))

        assert abs(measured_range - truth_range) <= 0.5 * axes.range_bin_m, (
            f"{name}: range {measured_range:.9f} m, error {measured_range - truth_range:.3e} m "
            f"against {truth_range:.9f} m and a {axes.range_bin_m:.6f} m bin"
        )
        assert abs(measured_speed - truth_speed) <= 0.5 * axes.velocity_bin_mps, (
            f"{name}: speed {measured_speed:.9f} m/s, error {measured_speed - truth_speed:.3e} m/s "
            f"against {truth_speed:.9f} m/s and a {axes.velocity_bin_mps:.6f} m/s bin"
        )

    # And the three agree with EACH OTHER, not only with the oracle: the
    # velocity comes back bit-identical because the three grids were solved to
    # share one coherent processing interval.
    speeds = {round(value, 12) for _, _, _, value in reported}
    assert len(speeds) == 1, reported
    ranges = [value for _, _, value, _ in reported]
    assert max(ranges) - min(ranges) <= 0.5 * min(axes.range_bin_m for _, axes, _, _ in reported)


def test_the_closing_target_is_positive_in_every_waveform(moving):
    """The isolated sign test, on a target that closes on both legs.

    The site sits at ``+y`` between two endpoints on the ``x`` axis and moves
    along ``-y``, so BOTH legs shorten and there is no geometry in which this
    could read as receding. Two of the three waveforms would report a negative
    velocity without the reconciliation.
    """

    batch, row, segment = moving
    assert float(batch.delay_rate[row]) < 0.0
    assert PROCESSING_DOPPLER_CONVENTION == "positive_doppler_bin_is_closing"

    for name, spec_of, synthesize, maker in WAVEFORMS:
        _, _, measured_speed = _measure(batch, segment, spec_of, synthesize, maker)
        assert measured_speed > 0.0, name


def test_every_waveform_publishes_the_same_units_and_the_same_conventions(moving):
    """One vocabulary. A per-waveform unit mapping would be three contracts."""

    batch, _, segment = moving
    records = [_measure(batch, segment, spec_of, synthesize, maker)[0] for _, spec_of, synthesize, maker in WAVEFORMS]
    reference = records[0].units
    for record in records[1:]:
        assert record.units == reference, record.waveform
    for record in records:
        assert record.reference_frequency_hz == grid.F_REF_HZ
        assert record.wavelength_m == pytest.approx(grid.C0 / grid.F_REF_HZ, rel=1e-15)
        assert record.range_m.dtype is torch.float64
        assert record.velocity_mps.dtype is torch.float64
