"""The slow-time transform, and the one place a Doppler sign is reconciled.

One closing target, driven through the production forward-AD seam so that
``delay_rate`` is what the propagation consumer produced and not a number this
fixture invented. The three waveform grids were solved so that their coherent
processing intervals MATCH, which is why all three land on the same signed bin
with the same velocity resolution and why the comparison below is an equality
rather than a tolerance.

The reconciliation is the subject. If ``doppler_sign`` were stubbed to ``+1``
everywhere the OFDM and pulsed assertions fail; if the reversal were removed
altogether the FMCW one fails; and
``test_the_unreconciled_beat_spectrum_peaks_on_the_opposite_bin`` measures the
raw spectrum directly so that "the reconciliation did something" is a
measurement rather than an inference.
"""

from __future__ import annotations

import pytest
import torch

from support import exact_bin_grid as grid
from support import multi_endpoint_driver as drv
from witwin.radar.processing import (
    ProcessingAxes,
    ProcessingCube,
    range_doppler_map,
    range_profile,
)
from witwin.radar.synthesis import (
    synthesize_fmcw,
    synthesize_ofdm,
    synthesize_pulsed,
)
from witwin.radar.synthesis.assembly import SynthesisResult

pytestmark = pytest.mark.gpu


WAVEFORMS = (
    ("fmcw", grid.fmcw_spec, synthesize_fmcw, SynthesisResult.from_fmcw),
    ("ofdm", grid.ofdm_spec, synthesize_ofdm, SynthesisResult.from_ofdm),
    (
        "pulsed",
        grid.pulsed_spec,
        synthesize_pulsed,
        SynthesisResult.from_pulsed,
    ),
)


@pytest.fixture(scope="module")
def spike():
    pytest.importorskip("witwin.channel")
    return grid.make_spike()


@pytest.fixture(scope="module")
def closing(spike):
    """The isolated target row, moving toward the front end at the solved speed."""

    composed = grid.moving_frame(spike)
    row = grid.target_row(spike, composed)
    batch = grid.isolate(drv.to_synthesis(composed), row)
    return batch, row, int(composed.sensor_pair_index[row])


def _map(batch, spec, maker, synthesize):
    result = maker(synthesize(batch, spec), spec)
    axes = ProcessingAxes.from_synthesis(result, spec, grid.array_spec())
    profile = range_profile(ProcessingCube.from_synthesis(result, axes))
    return range_doppler_map(profile), axes


def _peak(rd, segment):
    tx = segment % grid.FMCW_NUM_TX
    rx = segment // grid.FMCW_NUM_TX
    magnitude = rd.data[tx, rx].abs()
    flat = int(magnitude.argmax())
    return flat // magnitude.shape[1], flat % magnitude.shape[1]


def test_the_frozen_row_carries_the_delay_rate_the_geometry_was_solved_for(closing):
    """The seam, before the transform: 3.744 m/s of closing, to 1e-7 relative."""

    batch, row, _ = closing
    assert batch.delay_rate is not None
    measured = float(batch.delay_rate[row])
    assert measured == pytest.approx(grid.DELAY_RATE, rel=1e-6), measured
    assert measured < 0.0, "a closing target shortens the round trip"


# ---------------------------------------------------------------------------
# 4 and 5 - the exact Doppler bin, in all three waveforms
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,spec_of,synthesize,maker", WAVEFORMS)
def test_the_closing_target_lands_on_the_solved_doppler_bin(
    closing, name, spec_of, synthesize, maker, capsys
):
    """Bin ``centre + 2`` in every waveform, and the same signed velocity.

    The bin is POSITIVE for a closing target in all three, which is the
    canonical processing convention. The FMCW raw tone sits at
    ``+f_ref tau_rate`` and the other two at ``-f_ref tau_rate``, so two of the
    three would land on ``centre - 2`` without the reconciliation.
    """

    batch, _, segment = closing
    rd, axes = _map(batch, spec_of(), maker, synthesize)
    doppler, range_bin = _peak(rd, segment)
    centre = axes.doppler_bin_count // 2
    assert doppler == centre + grid.DOPPLER_BIN, (name, doppler, centre)

    velocity = float(axes.velocity_mps[doppler])
    assert velocity > 0.0, name
    assert velocity == pytest.approx(grid.CLOSING_SPEED_MPS, rel=1e-9), name
    with capsys.disabled():
        print(
            f"\n{name}: doppler bin {doppler} of {axes.doppler_bin_count} "
            f"(centre {centre}), v = {velocity:.9f} m/s, "
            f"bin = {axes.velocity_bin_mps:.9f} m/s, range bin {range_bin}"
        )


def test_all_three_waveforms_share_one_velocity_resolution_by_construction(closing):
    """The three grids were solved to have the same coherent interval.

    Asserted rather than assumed, because it is what makes the exact-bin
    comparison above an equality between three waveforms instead of three
    independent single-waveform checks.
    """

    batch, _, _ = closing
    bins = []
    for _, spec_of, synthesize, maker in WAVEFORMS:
        _, axes = _map(batch, spec_of(), maker, synthesize)
        bins.append(axes.velocity_bin_mps)
    for value in bins[1:]:
        assert value == pytest.approx(bins[0], rel=1e-12), bins


# ---------------------------------------------------------------------------
# 7 - the sign, isolated
# ---------------------------------------------------------------------------


def test_a_receding_target_lands_on_the_mirrored_negative_bin(spike):
    """Reverse the velocity, and every waveform's bin reflects about the centre.

    The one test that cannot pass with a stubbed sign: a stub makes two of the
    three waveforms report a receding target as approaching, and here the same
    physical reversal has to move all three the same way.
    """

    receding = tuple(-value for value in grid.SITE_VELOCITY_M_PER_S)
    composed = grid.moving_frame(spike, receding)
    row = grid.target_row(spike, composed)
    batch = grid.isolate(drv.to_synthesis(composed), row)
    segment = int(composed.sensor_pair_index[row])
    assert float(batch.delay_rate[row]) > 0.0

    for name, spec_of, synthesize, maker in WAVEFORMS:
        rd, axes = _map(batch, spec_of(), maker, synthesize)
        doppler, _ = _peak(rd, segment)
        centre = axes.doppler_bin_count // 2
        assert doppler == centre - grid.DOPPLER_BIN, (name, doppler, centre)
        velocity = float(axes.velocity_mps[doppler])
        assert velocity < 0.0, name
        assert velocity == pytest.approx(-grid.CLOSING_SPEED_MPS, rel=1e-9), name


def test_the_unreconciled_beat_spectrum_peaks_on_the_opposite_bin(closing):
    """The reconciliation is measured, not inferred.

    Transforming the beat cube's slow-time axis by hand - no reversal, the
    Phase-6 ``doppler_fft`` behaviour - puts the same closing target on
    ``centre - 2``. That is the defect this stage exists to close, and it is
    reproduced here so the fix cannot silently become a no-op.
    """

    batch, _, segment = closing
    spec = grid.fmcw_spec()
    result = SynthesisResult.from_fmcw(synthesize_fmcw(batch, spec), spec)
    axes = ProcessingAxes.from_synthesis(result, spec, grid.array_spec())
    processing = ProcessingCube.from_synthesis(result, axes)
    profile = range_profile(processing)

    tx, rx = segment % grid.FMCW_NUM_TX, segment // grid.FMCW_NUM_TX
    raw = torch.fft.fftshift(
        torch.fft.fft(profile.data[tx, rx], dim=-2), dim=-2
    ).abs()
    flat = int(raw.argmax())
    centre = axes.doppler_bin_count // 2
    assert flat // raw.shape[1] == centre - grid.DOPPLER_BIN

    reconciled, _ = _map(batch, spec, SynthesisResult.from_fmcw, synthesize_fmcw)
    assert _peak(reconciled, segment)[0] == centre + grid.DOPPLER_BIN


# ---------------------------------------------------------------------------
# Shape and plumbing
# ---------------------------------------------------------------------------


def test_the_map_is_rank_generic_and_publishes_the_axes_it_was_built_with(closing):
    """A ``[TX, RX, C, R]`` map and a ``[C, R]`` slice agree on the shared slice.

    RECORDED DEVIATION from the range stage, which IS asserted bitwise. The
    range transform runs over the trailing, contiguous axis and a leading batch
    does not change its arithmetic. The Doppler transform runs over a STRIDED
    axis, and cuFFT picks its plan from the whole shape, so the two calls do not
    execute the same reduction order. The difference measured here is at the
    float32 rounding floor; asserting it bitwise would be asserting a plan
    choice rather than the stage's semantics.
    """

    batch, _, segment = closing
    spec = grid.fmcw_spec()
    result = SynthesisResult.from_fmcw(synthesize_fmcw(batch, spec), spec)
    axes = ProcessingAxes.from_synthesis(result, spec, grid.array_spec())
    processing = ProcessingCube.from_synthesis(result, axes)

    tx, rx = segment % grid.FMCW_NUM_TX, segment // grid.FMCW_NUM_TX
    full = range_doppler_map(range_profile(processing))
    reference = full.data[tx, rx]
    assert tuple(reference.shape) == (
        axes.doppler_bin_count,
        axes.range_bin_count,
    )
    assert full.range_axis is axes.range_m
    assert full.doppler_axis is axes.velocity_mps
    assert tuple(full.data.shape) == (
        axes.num_tx,
        axes.num_rx,
        axes.doppler_bin_count,
        axes.range_bin_count,
    )


def test_the_doppler_stage_refuses_a_bare_tensor(closing):
    batch, _, _ = closing
    spec = grid.fmcw_spec()
    result = SynthesisResult.from_fmcw(synthesize_fmcw(batch, spec), spec)
    axes = ProcessingAxes.from_synthesis(result, spec, grid.array_spec())
    cube = ProcessingCube.from_synthesis(result, axes)
    with pytest.raises(TypeError, match="consumes a RangeProfile"):
        range_doppler_map(cube.data)
