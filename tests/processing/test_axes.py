"""The unified metadata / axes / units record, on the host.

No GPU and no propagation: every claim here is about a record built from three
waveform SPECS and a cube's published conventions, and the whole point of that
record is that it can be checked without running anything. The unit conversions
it wraps are exactly the kind of thing that is wrong once and then wrong
everywhere.

Four claims:

* the Doppler sign is DERIVED from the cube's phasor, not from the waveform's
  name, and it is ``+1`` for the conjugated beat cube and ``-1`` for the other
  two;
* every published axis is SI, float64, and equal to ``index * bin + origin``;
* a round trip from the spec through the record reproduces the spec's own
  scalars EXACTLY, not to a tolerance - both sides are the same float;
* the record refuses a cube, a spec and an array that do not describe the same
  front end.
"""

from __future__ import annotations

import pytest
import torch

from support import exact_bin_grid as grid
from witwin.radar.processing.signal import PROCESSING_UNITS, ProcessingAxes
from witwin.radar.synthesis.assembly import (
    BEAT_PHASOR,
    CHANNEL_PHASOR,
    SPEED_OF_LIGHT_M_PER_S,
    SynthesisResult,
)


PAIRS = grid.FMCW_NUM_TX * grid.FMCW_NUM_RX


def _cube(slow: int, fast: int) -> torch.Tensor:
    return torch.zeros((slow, PAIRS, fast), dtype=torch.complex64)


def fmcw_axes(**kwargs) -> ProcessingAxes:
    spec = grid.fmcw_spec()
    result = SynthesisResult.from_fmcw(
        _cube(spec.num_chirps, spec.num_samples), spec
    )
    return ProcessingAxes.from_synthesis(result, spec, grid.array_spec(), **kwargs)


def ofdm_axes(**kwargs) -> ProcessingAxes:
    spec = grid.ofdm_spec(num_symbols=8)
    result = SynthesisResult.from_ofdm(
        _cube(spec.num_symbols, spec.num_subcarriers), spec
    )
    return ProcessingAxes.from_synthesis(result, spec, grid.array_spec(), **kwargs)


def pulsed_axes(**kwargs) -> ProcessingAxes:
    spec = grid.pulsed_spec(num_pulses=8)
    result = SynthesisResult.from_pulsed(
        _cube(spec.num_pulses, spec.num_samples), spec
    )
    return ProcessingAxes.from_synthesis(result, spec, grid.array_spec(), **kwargs)


# ---------------------------------------------------------------------------
# The Doppler sign
# ---------------------------------------------------------------------------


def test_the_doppler_sign_is_derived_from_the_phasor_not_from_the_waveform():
    """``+1`` for the conjugated beat cube, ``-1`` for the two Channel ones.

    The FMCW beat cube is the conjugate of Channel's ``exp(-j k d)`` product, so
    its slow-time tone sits at ``+f_ref tau_rate`` while the OFDM and pulsed
    tones sit at ``-f_ref tau_rate``. That is a fact about the PRODUCT, and the
    record reads the product's own published convention string to decide it.
    """

    assert fmcw_axes().phasor == BEAT_PHASOR
    assert fmcw_axes().doppler_sign == 1
    for record in (ofdm_axes(), pulsed_axes()):
        assert record.phasor == CHANNEL_PHASOR, record.waveform
        assert record.doppler_sign == -1, record.waveform


def test_an_unknown_phasor_is_refused_rather_than_defaulted():
    """A third convention needs its sign decided, not inherited."""

    from dataclasses import replace

    spec = grid.fmcw_spec()
    result = replace(
        SynthesisResult.from_fmcw(_cube(spec.num_chirps, spec.num_samples), spec),
        phasor="exp(+j*k*d)",
    )
    with pytest.raises(ValueError, match="unknown phasor convention"):
        ProcessingAxes.from_synthesis(result, spec, grid.array_spec())


# ---------------------------------------------------------------------------
# Units and axis values
# ---------------------------------------------------------------------------


def test_every_published_axis_is_si_float64_and_matches_its_bin_width():
    """``range_m[k] == k * range_bin_m + range_origin_m``, exactly.

    Not to a tolerance: both sides are float64 and the axis is built by that
    expression, so any difference would mean the record published a bin width
    that is not the one its own axis uses.
    """

    for record in (fmcw_axes(), ofdm_axes(), pulsed_axes()):
        assert record.range_m.dtype is torch.float64, record.waveform
        assert record.velocity_mps.dtype is torch.float64, record.waveform
        index = torch.arange(record.range_bin_count, dtype=torch.float64)
        expected = index * record.range_bin_m + record.range_origin_m
        assert torch.equal(record.range_m, expected), record.waveform

        # The Doppler axis is fftshifted, so the zero bin sits at D // 2 and the
        # next one up is exactly one velocity bin away.
        centre = record.doppler_bin_count // 2
        assert float(record.velocity_mps[centre]) == 0.0, record.waveform
        assert float(record.velocity_mps[centre + 1]) == pytest.approx(
            record.velocity_bin_mps, rel=1e-12
        ), record.waveform
        # And the whole axis spans exactly twice the unambiguous speed.
        assert record.velocity_bin_mps * record.doppler_bin_count == pytest.approx(
            2.0 * record.max_unambiguous_speed_mps, rel=1e-12
        ), record.waveform


def test_the_units_mapping_names_every_published_quantity():
    """SI throughout, and the mapping is the contract rather than a docstring."""

    record = fmcw_axes()
    published = record.units
    assert published == PROCESSING_UNITS
    for name, unit in published.items():
        assert hasattr(record, name), name
        value = getattr(record, name)
        if isinstance(value, torch.Tensor):
            assert value.dtype is torch.float64, name
        else:
            assert type(value) is float, name
        assert unit in ("s", "m", "m/s", "Hz"), (name, unit)
    # The mapping is a copy: a caller that mutates it cannot edit the contract.
    published["range_bin_m"] = "furlongs"
    assert record.units["range_bin_m"] == "m"


# ---------------------------------------------------------------------------
# The spec round trip
# ---------------------------------------------------------------------------


def test_the_record_reproduces_the_spec_scalars_exactly():
    """``==`` on floats, because both sides come from the same spec.

    This is the statement that the record is built from the waveform SPECS and
    never from the flat engineering-unit configuration: a second conversion from
    kSPS and microseconds would land a part in 1e16 away and this would fail.
    """

    fmcw = grid.fmcw_spec()
    record = fmcw_axes()
    assert record.slow_time_period_s == fmcw.slot_period_s
    assert record.wavelength_m == fmcw.wavelength_m
    assert record.reference_frequency_hz == fmcw.reference_frequency_hz
    assert record.max_unambiguous_speed_mps == fmcw.max_unambiguous_speed_mps
    assert record.range_origin_m == 0.0

    ofdm = grid.ofdm_spec(num_symbols=8)
    record = ofdm_axes()
    assert record.slow_time_period_s == ofdm.symbol_period_s
    assert record.range_bin_m == ofdm.range_resolution_m
    assert record.max_unambiguous_speed_mps == ofdm.max_unambiguous_speed_mps
    assert record.max_unambiguous_range_m == (
        SPEED_OF_LIGHT_M_PER_S * ofdm.max_unambiguous_delay_s / 2.0
    )

    pulsed = grid.pulsed_spec(num_pulses=8)
    record = pulsed_axes()
    assert record.slow_time_period_s == pulsed.pri_s
    assert record.max_unambiguous_speed_mps == pulsed.max_unambiguous_speed_m_s
    assert record.max_unambiguous_range_m == pulsed.max_unambiguous_range_m
    assert record.range_bin_m == (
        SPEED_OF_LIGHT_M_PER_S * pulsed.sample_period_s / 2.0
    )


def test_the_axis_names_follow_the_waveform_and_not_the_other_way_round():
    assert (fmcw_axes().slow_time_name, fmcw_axes().fast_time_name) == (
        "chirp",
        "sample",
    )
    assert (ofdm_axes().slow_time_name, ofdm_axes().fast_time_name) == (
        "symbol",
        "subcarrier",
    )
    assert (pulsed_axes().slow_time_name, pulsed_axes().fast_time_name) == (
        "pulse",
        "sample",
    )


def test_a_pulsed_range_gate_moves_the_axis_origin_and_nothing_else():
    """``range_origin_m = c t_g / 2``: the gate start is a range, not a bin."""

    from dataclasses import replace

    spec = replace(grid.pulsed_spec(num_pulses=8), range_gate_start_s=2.0e-8)
    result = SynthesisResult.from_pulsed(
        _cube(spec.num_pulses, spec.num_samples), spec
    )
    record = ProcessingAxes.from_synthesis(result, spec, grid.array_spec())
    assert record.range_origin_m == pytest.approx(
        SPEED_OF_LIGHT_M_PER_S * 2.0e-8 / 2.0, rel=1e-15
    )
    assert float(record.range_m[0]) == record.range_origin_m
    assert record.range_bin_m == pulsed_axes().range_bin_m


def test_range_oversample_refines_the_lag_grid_only_for_the_pulsed_backend():
    record = pulsed_axes(range_oversample=4)
    plain = pulsed_axes()
    assert record.range_bin_count == 4 * plain.range_bin_count
    assert record.range_bin_m == pytest.approx(plain.range_bin_m / 4.0, rel=1e-15)
    assert record.matched_filter_replica is not None

    for builder in (fmcw_axes, ofdm_axes):
        with pytest.raises(ValueError, match="only the pulsed backend"):
            builder(range_oversample=2)


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_a_cube_a_spec_and_an_array_must_describe_one_front_end():
    from dataclasses import replace

    spec = grid.fmcw_spec()
    result = SynthesisResult.from_fmcw(
        _cube(spec.num_chirps, spec.num_samples), spec
    )
    array = grid.array_spec()

    with pytest.raises(ValueError, match="but the spec declares"):
        ProcessingAxes.from_synthesis(
            replace(result, reference_frequency_hz=76.0e9), spec, array
        )
    with pytest.raises(ValueError, match="element spacing is defined"):
        ProcessingAxes.from_synthesis(
            result, spec, replace(array, reference_frequency_hz=76.0e9)
        )
    with pytest.raises(ValueError, match="sensor pairs but the array"):
        ProcessingAxes.from_synthesis(
            result,
            spec,
            replace(
                array,
                num_rx=1,
                rx_loc=(array.rx_loc[0],),
            ),
        )
