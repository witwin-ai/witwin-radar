"""One entry, three waveform backends, three exact range bins.

Every assertion below is made about ONE composed row of the real multi-endpoint
fixture - the reflection-free round trip ``TX_A -> site -> RX_A`` - whose delay
was placed on an exact bin by solving the ellipse in closed form
(``support/exact_bin_grid.py``). Nothing is searched and nothing is fitted.

The three exact-bin conditions:

* FMCW  ``bin = S tau N / f_s``      -> bin 50 of 256
* OFDM  ``sample = tau / T_s``       -> CIR sample 4 of 64
* pulsed ``lag = (tau - t_g) / T_s`` -> lag 4 of 128

and the same physical range, ``c tau / 2``, comes back out of all three range
axes.
"""

from __future__ import annotations

import math

import pytest
import torch
from support import exact_bin_grid as grid
from support import multi_endpoint_driver as drv

from witwin.radar.processing import ProcessingAxes, ProcessingCube, range_profile
from witwin.radar.synthesis import synthesize_fmcw, synthesize_ofdm, synthesize_pulsed
from witwin.radar.synthesis.assembly import PULSE_KIND_RECT, SynthesisResult

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def target():
    """One frozen composition, one isolated row, and where it sits."""

    pytest.importorskip("witwin.channel")
    spike = grid.make_spike()
    composed, _, _ = spike.frame()
    row = grid.target_row(spike, composed)
    batch = grid.isolate(drv.to_synthesis(composed), row)
    segment = int(composed.sensor_pair_index[row])
    return batch, row, segment


def _profile(batch, spec, maker, synthesize, *, window=None, remove_dc=False):
    cube = synthesize(batch, spec)
    result = maker(cube, spec)
    axes = ProcessingAxes.from_synthesis(result, spec, grid.array_spec())
    processing = ProcessingCube.from_synthesis(result, axes)
    return range_profile(processing, window=window, remove_dc=remove_dc), axes


def _row_of(profile, segment):
    tx = segment % grid.FMCW_NUM_TX
    rx = segment // grid.FMCW_NUM_TX
    return profile.data[tx, rx, 0]


# ---------------------------------------------------------------------------
# 1 - FMCW exact range bin
# ---------------------------------------------------------------------------


def test_the_fmcw_beat_spectrum_peaks_on_the_solved_bin(target, capsys):
    """Bin 50, with both neighbours below -40 dB and no window.

    A pure beat tone whose frequency is exactly ``k f_s / N`` puts every other
    DFT bin at zero, so the sidelobe level measures how far the fixture's
    float32 delay is from the solved one rather than any windowing choice. The
    measured level is printed, because a number that is quoted in a report and
    never produced by a run is a number nobody checked.
    """

    batch, row, segment = target
    profile, axes = _profile(batch, grid.fmcw_spec(1), SynthesisResult.from_fmcw, synthesize_fmcw)
    magnitude = _row_of(profile, segment).abs()
    peak = int(magnitude.argmax())
    assert peak == grid.FMCW_RANGE_BIN, (peak, grid.FMCW_RANGE_BIN)

    top = float(magnitude[peak])
    sidelobes = [20.0 * math.log10(float(magnitude[peak + offset]) / top) for offset in (-1, 1)]
    assert max(sidelobes) < -40.0, sidelobes
    with capsys.disabled():
        print(f"\nFMCW bin {peak}, neighbours {sidelobes[0]:.1f} / {sidelobes[1]:.1f} dB")

    # And the bin means the right number of metres, through the axis and not
    # through a formula restated here.
    assert float(axes.range_m[peak]) == pytest.approx(grid.RANGE_M, abs=1e-9)


# ---------------------------------------------------------------------------
# 2 - OFDM exact CIR sample, and the amplitude anchor
# ---------------------------------------------------------------------------


def test_the_ofdm_cir_peaks_on_the_solved_sample_and_anchors_the_amplitude(target):
    """CIR sample 4, and ``H[0][p][0] == C_rt`` EXACTLY.

       The subcarrier origin is pinned at `
    = 0 -> f_ref``, so subcarrier zero of
       a stationary row is the Channel coefficient itself with no phase offset at
       all. That identity is the cross-waveform amplitude anchor and it is asserted
       bitwise, not to a tolerance: anything else would mean the CFR kernel applied
       something at `
    = 0``.
    """

    batch, row, segment = target
    spec = grid.ofdm_spec(num_symbols=1)
    cube = synthesize_ofdm(batch, spec)
    assert cube[0, segment, 0] == batch.complex_transfer_ref[row]

    result = SynthesisResult.from_ofdm(cube, spec)
    axes = ProcessingAxes.from_synthesis(result, spec, grid.array_spec())
    profile = range_profile(ProcessingCube.from_synthesis(result, axes))
    magnitude = _row_of(profile, segment).abs()
    peak = int(magnitude.argmax())
    assert peak == grid.OFDM_CIR_SAMPLE, (peak, grid.OFDM_CIR_SAMPLE)
    assert float(axes.range_m[peak]) == pytest.approx(grid.RANGE_M, abs=1e-9)

    # The inverse transform carries the 1 / N_sc that makes the peak the
    # coefficient rather than N times it.
    coefficient = float(batch.complex_transfer_ref[row].abs())
    assert float(magnitude[peak]) == pytest.approx(coefficient, rel=1e-5)


# ---------------------------------------------------------------------------
# 3 - pulsed exact matched-filter lag
# ---------------------------------------------------------------------------


def test_the_pulsed_matched_filter_peaks_on_the_solved_lag(target, capsys):
    """Lag 4, with the peak within the one-sample straddle the support costs.

    RECORDED DEVIATION. The design asks for a rectangular pulse here. Measured
    on this fixture, a rect pulse cannot carry an exact-lag assertion: the
    correlation is a triangle whose adjacent bins differ by ``1 / M_p`` -
    0.2757 dB at ``M_p = 32``, measured below - and the pulse support is HALF
    OPEN, so a delay that float32 rounds a part in 1e8 above ``m T_s`` drops the
    first received sample and moves the apex to ``m + 1``. That is a property of
    the rectangular pulse, and the second assertion here reproduces it on
    purpose rather than working around it.

    The LFM's compressed main lobe is ``1 / B`` wide - 2.5 samples on this grid
    - so its argmax is decided by the delay. The same one missing sample costs
    ``31 / 32`` of the peak, which is asserted rather than hidden.
    """

    batch, row, segment = target
    spec = grid.pulsed_spec(num_pulses=1)
    assert spec.pulse_grid_is_commensurate
    assert spec.range_gate_end_s <= spec.pri_s
    assert spec.range_migration_delay_s < spec.range_cell_delay_s

    profile, axes = _profile(batch, spec, SynthesisResult.from_pulsed, synthesize_pulsed)
    magnitude = _row_of(profile, segment).abs()
    peak = int(magnitude.argmax())
    assert peak == grid.PULSED_LAG_SAMPLE, (peak, grid.PULSED_LAG_SAMPLE)
    assert float(axes.range_m[peak]) == pytest.approx(grid.RANGE_M, abs=1e-9)

    coefficient = float(batch.complex_transfer_ref[row].abs())
    ratio = float(magnitude[peak]) / coefficient
    assert 0.95 < ratio <= 1.0, ratio
    with capsys.disabled():
        print(f"\npulsed LFM lag {peak}, peak / |C_rt| = {ratio:.6f} (31/32 = {31 / 32:.6f})")

    # The rect pulse on the same grid: a triangle whose neighbours are
    # 20 log10(31/32) = -0.2757 dB down, which is why it is not the exact-bin
    # fixture.
    rect_profile, _ = _profile(
        batch,
        grid.pulsed_spec(num_pulses=1, pulse_kind=PULSE_KIND_RECT),
        SynthesisResult.from_pulsed,
        synthesize_pulsed,
    )
    rect = _row_of(rect_profile, segment).abs()
    apex = int(rect.argmax())
    assert abs(apex - grid.PULSED_LAG_SAMPLE) <= 1, apex
    neighbour = 20.0 * math.log10(float(rect[apex - 1]) / float(rect[apex]))
    assert neighbour == pytest.approx(20.0 * math.log10(31.0 / 32.0), abs=0.02), neighbour


# ---------------------------------------------------------------------------
# 9 - the DC removal is a flag, and it is off
# ---------------------------------------------------------------------------


def test_remove_dc_defaults_to_off_and_removes_the_fast_time_mean_when_asked(target):
    """``process_rd_tensor`` does this unconditionally; here it is a choice.

    A constant fast-time offset is exactly what a clutter export or a leakage
    component looks like, and silently deleting it would make a component sum
    disagree with the whole-scene cube by an amount nothing reports.
    """

    batch, _, segment = target
    spec = grid.fmcw_spec(1)
    cube = synthesize_fmcw(batch, spec)
    offset = torch.full_like(cube, 3.0 + 1.0j)
    result = SynthesisResult.from_fmcw(cube + offset, spec)
    axes = ProcessingAxes.from_synthesis(result, spec, grid.array_spec())
    processing = ProcessingCube.from_synthesis(result, axes)

    kept = range_profile(processing)
    assert kept.data is not None
    dropped = range_profile(processing, remove_dc=True)

    # Bin zero of an unwindowed transform IS the fast-time mean, so the flag is
    # visible exactly there and the target bin is untouched by it.
    tx, rx = segment % grid.FMCW_NUM_TX, segment // grid.FMCW_NUM_TX
    assert float(kept.data[tx, rx, 0, 0].abs()) > 3.0
    assert float(dropped.data[tx, rx, 0, 0].abs()) < 1e-4
    torch.testing.assert_close(
        kept.data[tx, rx, 0, grid.FMCW_RANGE_BIN], dropped.data[tx, rx, 0, grid.FMCW_RANGE_BIN], rtol=1e-5, atol=0.0
    )

    # The default is off. Asserted through the entry point rather than by
    # reading the signature, because a default is what callers get.
    baseline = range_profile(processing, remove_dc=False)
    assert torch.equal(kept.data, baseline.data)


# ---------------------------------------------------------------------------
# 10 - rank genericity
# ---------------------------------------------------------------------------


def test_a_batched_cube_and_one_of_its_slices_give_the_same_profile(target):
    """``[TX, RX, C, S]`` and ``[C, S]`` agree bitwise on the shared slice.

    Bitwise, not close: the transform is over the trailing axis and a leading
    batch does not change the arithmetic. Anything looser would let a reshape
    that reorders the fast-time axis pass.
    """

    batch, _, segment = target
    spec = grid.fmcw_spec(2)
    result = SynthesisResult.from_fmcw(synthesize_fmcw(batch, spec), spec)
    axes = ProcessingAxes.from_synthesis(result, spec, grid.array_spec())
    processing = ProcessingCube.from_synthesis(result, axes)

    tx, rx = segment % grid.FMCW_NUM_TX, segment // grid.FMCW_NUM_TX
    full = range_profile(processing, window="hann")
    assert tuple(full.data[tx, rx].shape) == (2, axes.range_bin_count)


def test_a_window_scales_the_peak_by_its_published_coherent_gain(target):
    """``peak == |C_rt| * window_coherent_gain``, for an on-bin row.

    The gain is computed on the HOST from the window's closed form, so this also
    pins that the published number is the same one the taper applied - the two
    would otherwise drift the first time a window definition moved.
    """

    batch, row, segment = target
    coefficient = float(batch.complex_transfer_ref[row].abs())
    for window, gain in (("rectangular", 1.0), ("hann", 0.5), ("hamming", 0.54)):
        profile, _ = _profile(batch, grid.fmcw_spec(1), SynthesisResult.from_fmcw, synthesize_fmcw, window=window)
        assert profile.window == window
        assert profile.window_coherent_gain == pytest.approx(gain, rel=1e-12)
        peak = float(_row_of(profile, segment).abs()[grid.FMCW_RANGE_BIN])
        assert peak == pytest.approx(coefficient * gain, rel=1e-4), window


def test_the_entry_refuses_a_bare_tensor_with_no_metadata(target):
    batch, _, _ = target
    spec = grid.fmcw_spec(1)
    result = SynthesisResult.from_fmcw(synthesize_fmcw(batch, spec), spec)
    axes = ProcessingAxes.from_synthesis(result, spec, grid.array_spec())
    processing = ProcessingCube.from_synthesis(result, axes)

    with pytest.raises(TypeError, match="ProcessingCube"):
        range_profile(processing.data)
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        range_profile(processing, axes=axes)
