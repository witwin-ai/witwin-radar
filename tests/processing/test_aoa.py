"""Angle of arrival: exact bins, both routes, the TDM sign, and MUSIC's spacing.

Nothing here is searched and no tolerance is chosen to pass. The estimators
publish a closed relation between a DFT bin and a direction cosine, so a
wavefront can be placed EXACTLY on a bin and the recovered angle asserted to
1e-4 degrees.

Design R4, honoured: ``naive_xyz`` requires more than two transmitters and
dispatches to the two-dimensional route only above four, so the fixture's
2 TX x 2 RX front end cannot exercise either. Two array configurations are used
below - 3 TX x 4 RX for phase comparison and 6 TX x 4 RX for the 2-D FFT - and
both routes are covered.
"""

from __future__ import annotations

import math

import pytest
import torch

from support import exact_bin_grid as grid
from witwin.radar.processing import (
    ArrayGeometry,
    fft2_aoa,
    music_spectrum,
    phase_comparison_aoa,
    tdm_compensate,
)
from conftest import PROCESSING_CONFIG, make_processing_axes
from witwin.radar.synthesis.assembly import SPEED_OF_LIGHT_M_PER_S


#: Read from the same constant the axes record is built on. A hand-typed
#: wavelength that is a part in 1e5 off puts a part in 1e5 into every phase this
#: file asserts to 1e-6 radians.
WAVELENGTH_M = SPEED_OF_LIGHT_M_PER_S / 77e9
FFT_SIZE = 64

#: Transmitters at 0 and 4 half wavelengths with receivers at 0..3 make the
#: first ``2 * num_rx`` virtual elements a uniform line, which is the array the
#: phase-comparison relation is written against. The third transmitter is
#: displaced in z only, so ``el_tx_dx`` is exactly zero and the elevation
#: estimate carries no azimuth-walk correction to get wrong.
TX_3 = ((0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (0.0, 0.0, 1.0))
TX_6 = (
    (0.0, 0.0, 0.0),
    (4.0, 0.0, 0.0),
    (0.0, 0.0, 1.0),
    (4.0, 0.0, 1.0),
    (0.0, 0.0, 2.0),
    (4.0, 0.0, 2.0),
)
RX_4 = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0))


def _array(transmitters, *, phase_sign: int = -1) -> ArrayGeometry:
    return ArrayGeometry.from_offsets(
        transmitters,
        RX_4,
        element_spacing_m=WAVELENGTH_M / 2.0,
        wavelength_m=WAVELENGTH_M,
        phase_sign=phase_sign,
    )


def _exact_cosine(bin_index: int, fft_size: int = FFT_SIZE) -> float:
    """``2 k / fft_size``: the direction cosine bin ``k`` is exactly on."""

    return 2.0 * bin_index / fft_size


# ---------------------------------------------------------------------------
# Exact bins
# ---------------------------------------------------------------------------


def test_phase_comparison_recovers_an_exactly_on_bin_azimuth():
    """``sin(theta) = 2 k / fft_size``; at ``k = 8, N = 64``, 14.4775 degrees.

    The relation is ``wx = (2 pi / fft_size) signed_k`` and ``x = wx / pi``, so
    on a half-wavelength line array ``x`` IS ``sin(theta_az)`` and the bin is
    exact. The wavefront below is built in the Channel convention, which is the
    array response ``exp(+j k <r, u>)``.
    """

    array = _array(TX_3)
    expected = _exact_cosine(8)
    assert expected == 0.25
    theta = math.degrees(math.asin(expected))
    assert theta == pytest.approx(14.4775, abs=1e-4)

    x_positions = array.element_positions_m[:, 0] / array.element_spacing_m
    signal = torch.polar(
        torch.ones(12, dtype=torch.float64),
        torch.pi * x_positions * expected,
    ).reshape(-1, 1).to(torch.complex64)

    cosines = phase_comparison_aoa(signal, array, fft_size=FFT_SIZE)
    recovered = math.degrees(math.asin(float(cosines[0, 0])))
    assert recovered == pytest.approx(theta, abs=1e-4)


def test_phase_comparison_recovers_an_exactly_on_bin_elevation():
    """Same relation on the z axis, at broadside azimuth so it stands alone.

    The elevation sub-aperture sits one half wavelength up in z, so its phase
    lead is ``pi uz`` and the estimator's angle difference reads it directly.
    The recovered cosine points ALONG the array's z axis, which the deleted
    original did not: it took the reciprocal ratio and published the negative,
    which is why every legacy elevation assertion was written on an absolute
    value.
    """

    array = _array(TX_3)
    expected = _exact_cosine(8)
    z_positions = array.element_positions_m[:, 2] / array.element_spacing_m
    signal = torch.polar(
        torch.ones(12, dtype=torch.float64), torch.pi * z_positions * expected
    ).reshape(-1, 1).to(torch.complex64)

    cosines = phase_comparison_aoa(signal, array, fft_size=FFT_SIZE)
    assert float(cosines[0, 0]) == pytest.approx(0.0, abs=1e-6)
    recovered = math.degrees(math.asin(float(cosines[2, 0])))
    assert recovered == pytest.approx(math.degrees(math.asin(expected)), abs=1e-4)
    assert float(cosines[2, 0]) > 0.0


def test_the_two_dimensional_route_recovers_both_cosines_on_their_own_bins():
    """The second array configuration, and the second route.

    The interleaved grid the route builds is ``rows = num_tx // 2`` by
    ``2 * num_rx``, with the even transmitter rows on the left half and the odd
    ones on the right. A wavefront placed exactly on ``(k_el, k_az)`` of that
    grid comes back at exactly those two cosines.
    """

    array = _array(TX_6)
    k_az, k_el = 8, 4
    x_expected = _exact_cosine(k_az)
    z_expected = _exact_cosine(k_el)

    values = torch.zeros((24, 1), dtype=torch.complex128)
    for tx in range(6):
        for rx in range(4):
            row = tx // 2
            column = rx + (tx % 2) * 4
            phase = 2.0 * math.pi * (k_el * row + k_az * column) / FFT_SIZE
            values[tx * 4 + rx, 0] = complex(math.cos(phase), math.sin(phase))

    cosines = fft2_aoa(values.to(torch.complex64), array, fft_size=FFT_SIZE)
    assert float(cosines[0, 0]) == pytest.approx(x_expected, abs=1e-7)
    assert float(cosines[2, 0]) == pytest.approx(z_expected, abs=1e-7)
    assert float(cosines[1, 0]) == pytest.approx(
        math.sqrt(1.0 - x_expected**2 - z_expected**2), abs=1e-6
    )


def test_a_conjugated_beat_cube_is_reconciled_rather_than_mirrored():
    """The third and last appearance of the conjugation trap.

    A DFT peak measures the progression of the data it was handed. The same
    physical target in a conjugated beat cube progresses the other way, so an
    unreconciled estimator reports its mirror image. Both conventions are driven
    here from one wavefront and both recover the SAME direction.
    """

    channel = _array(TX_3, phase_sign=-1)
    beat = _array(TX_3, phase_sign=1)
    expected = _exact_cosine(8)
    x_positions = channel.element_positions_m[:, 0] / channel.element_spacing_m
    wave = torch.polar(
        torch.ones(12, dtype=torch.float64), torch.pi * x_positions * expected
    ).reshape(-1, 1).to(torch.complex64)

    from_channel = phase_comparison_aoa(wave, channel, fft_size=FFT_SIZE)
    from_beat = phase_comparison_aoa(wave.conj(), beat, fft_size=FFT_SIZE)
    torch.testing.assert_close(from_channel, from_beat, rtol=1e-6, atol=1e-7)
    assert float(from_channel[0, 0]) == pytest.approx(expected, abs=1e-7)


def test_a_direction_that_cannot_close_is_published_as_an_exact_zero():
    array = _array(TX_3)
    # Two cosines whose squares exceed one: the boresight component would be
    # imaginary, so the row describes no direction at all.
    x_positions = array.element_positions_m[:, 0] / array.element_spacing_m
    z_positions = array.element_positions_m[:, 2] / array.element_spacing_m
    signal = torch.polar(
        torch.ones(12, dtype=torch.float64),
        torch.pi * (x_positions * 0.875 + z_positions * 0.875),
    ).reshape(-1, 1).to(torch.complex64)
    cosines = phase_comparison_aoa(signal, array, fft_size=FFT_SIZE)
    assert float(cosines[0, 0]) == 0.0
    assert float(cosines[1, 0]) == 0.0
    assert float(cosines[2, 0]) == 0.0


# ---------------------------------------------------------------------------
# TDM compensation
# ---------------------------------------------------------------------------


def test_the_tdm_compensation_is_the_analytic_slot_phase_at_a_nonzero_velocity():
    """``exp(-j 2 pi f_ref tau_rate m T_chirp)``, to 1e-6 radians.

    Driven from the closed-form closing speed and delay rate the exact-bin
    fixture solves - the SAME pair that drives ``delay_rate`` through the
    forward-AD seam, measured there to a relative 6e-8 - rather than from an
    invented velocity, so the number that compensates the phase is the number
    that produced it.

    It is exercised at a NONZERO velocity on purpose: the compensation is the
    identity at ``v = 0``, so a zero-velocity test asserts nothing.
    """

    array = _array(TX_3, phase_sign=1)
    axes = make_processing_axes(PROCESSING_CONFIG)
    chirp_period_s = axes.slow_time_period_s / axes.num_tx

    velocity = torch.tensor([grid.CLOSING_SPEED_MPS], dtype=torch.float64)
    assert float(velocity) != 0.0
    ones = torch.ones((12, 1), dtype=torch.complex64)
    compensated = tdm_compensate(ones, velocity, array, axes)

    analytic = -2.0 * math.pi * grid.F_REF_HZ * grid.DELAY_RATE * chirp_period_s
    for pair in range(12):
        transmitter = pair // array.num_rx
        expected = analytic * transmitter
        measured = float(torch.angle(compensated[pair, 0]))
        wrapped = (measured - expected + math.pi) % (2 * math.pi) - math.pi
        assert abs(wrapped) < 1e-6, (pair, measured, expected)


def test_the_compensation_reads_the_raw_chirp_period_not_the_slot_period():
    """A factor of ``num_tx`` in every compensated elevation, if confused.

    ``ProcessingAxes.slow_time_period_s`` is the TDM SLOT period, ``num_tx``
    times the chirp period. The compensation multiplies a TRANSMITTER index by
    the chirp period, and this pins which of the two it uses by comparing two
    arrays that differ only in transmitter count.
    """

    axes = make_processing_axes(PROCESSING_CONFIG)
    array = _array(TX_3, phase_sign=1)
    velocity = torch.tensor([grid.CLOSING_SPEED_MPS], dtype=torch.float64)
    ones = torch.ones((12, 1), dtype=torch.complex64)
    phase = float(torch.angle(tdm_compensate(ones, velocity, array, axes)[4, 0]))
    chirp_period_s = axes.slow_time_period_s / axes.num_tx
    slot_phase = (
        -2.0 * math.pi * grid.F_REF_HZ * grid.DELAY_RATE * axes.slow_time_period_s
    )
    chirp_phase = -2.0 * math.pi * grid.F_REF_HZ * grid.DELAY_RATE * chirp_period_s
    assert abs((phase - chirp_phase + math.pi) % (2 * math.pi) - math.pi) < 1e-6
    assert abs((phase - slot_phase + math.pi) % (2 * math.pi) - math.pi) > 1e-3


def test_the_compensation_preserves_magnitude_and_is_one_multiply():
    array = _array(TX_3, phase_sign=1)
    axes = make_processing_axes(PROCESSING_CONFIG)
    generator = torch.Generator().manual_seed(11)
    signal = torch.complex(
        torch.randn((12, 7), generator=generator),
        torch.randn((12, 7), generator=generator),
    ).to(torch.complex64)
    velocities = torch.linspace(-4.0, 4.0, 7, dtype=torch.float64)
    compensated = tdm_compensate(signal, velocities, array, axes)
    assert compensated.dtype == signal.dtype
    torch.testing.assert_close(
        compensated.abs(), signal.abs(), rtol=1e-6, atol=1e-6
    )
    # Transmitter zero is untouched, bitwise.
    assert torch.equal(compensated[: array.num_rx], signal[: array.num_rx])


# ---------------------------------------------------------------------------
# MUSIC
# ---------------------------------------------------------------------------


def _upa_snapshots(
    *, rows: int, columns: int, spacing_wavelengths: float, angle_rad: float, count: int
) -> torch.Tensor:
    """``[1, rows, columns, count]``: one plane wave on a planar array.

    The wave is placed on the COLUMN axis only, so a one-dimensional peak in
    azimuth is what MUSIC has to find, and the phase step per column is
    ``2 pi d sin(theta)`` with ``d`` in wavelengths - which is the number the
    estimator has to read off the array rather than assume.
    """

    generator = torch.Generator().manual_seed(909)
    column = torch.arange(columns, dtype=torch.float64)
    step = 2.0 * math.pi * spacing_wavelengths * math.sin(angle_rad)
    manifold = torch.polar(torch.ones(columns, dtype=torch.float64), column * step)
    amplitude = torch.complex(
        torch.randn((count,), generator=generator, dtype=torch.float64),
        torch.randn((count,), generator=generator, dtype=torch.float64),
    )
    wave = manifold.reshape(1, 1, columns, 1) * amplitude.reshape(1, 1, 1, count)
    noise = torch.complex(
        torch.randn((1, rows, columns, count), generator=generator, dtype=torch.float64),
        torch.randn((1, rows, columns, count), generator=generator, dtype=torch.float64),
    ) * 0.02
    return (wave + noise).to(torch.complex64)


@pytest.mark.parametrize("spacing_wavelengths", [0.5, 0.25])
def test_the_music_peak_lands_on_the_true_angle_at_any_element_spacing(
    spacing_wavelengths,
):
    """This is the test the hard-coded ``spacing = 0.5`` could not pass.

    ``MUSICImager._build_steering_vectors`` took ``spacing = 0.5`` as a default
    that nothing ever overrode, so a quarter-wave array was scanned with a
    half-wave manifold and reported an angle that was wrong by a factor of two -
    with no symptom other than being wrong. The spacing is data now, and the
    same wavefront is recovered at both.
    """

    rows, columns = 6, 6
    array = ArrayGeometry.from_offsets(
        [[0.0, 0.0, float(index)] for index in range(rows)],
        [[float(index), 0.0, 0.0] for index in range(columns)],
        element_spacing_m=WAVELENGTH_M * spacing_wavelengths,
        wavelength_m=WAVELENGTH_M,
    )
    assert array.spacing_wavelengths == pytest.approx(spacing_wavelengths)

    truth = 0.35
    data = _upa_snapshots(
        rows=rows,
        columns=columns,
        spacing_wavelengths=spacing_wavelengths,
        angle_rad=truth,
        count=64,
    )
    azimuth = torch.linspace(-math.pi / 3, math.pi / 3, 241, dtype=torch.float32)
    elevation = torch.zeros(1, dtype=torch.float32)
    spectrum = music_spectrum(
        data,
        array,
        elevation_rad=elevation,
        azimuth_rad=azimuth,
        num_signals=1,
        spatial_smooth=2,
    )
    peak = int(spectrum[0, 0].abs().argmax())
    recovered = float(azimuth[peak])
    assert recovered == pytest.approx(truth, abs=float(azimuth[1] - azimuth[0]))


def test_music_refuses_a_smoothing_or_a_signal_count_that_leaves_no_subspace():
    array = ArrayGeometry.from_offsets(
        [[0.0, 0.0, float(i)] for i in range(4)],
        [[float(i), 0.0, 0.0] for i in range(4)],
        element_spacing_m=WAVELENGTH_M / 2.0,
        wavelength_m=WAVELENGTH_M,
    )
    data = torch.ones((1, 4, 4, 3), dtype=torch.complex64)
    angles = torch.zeros(2, dtype=torch.float32)
    with pytest.raises(ValueError, match="spatial_smooth"):
        music_spectrum(
            data, array, elevation_rad=angles, azimuth_rad=angles, spatial_smooth=4
        )
    with pytest.raises(ValueError, match="noise subspace"):
        music_spectrum(
            data,
            array,
            elevation_rad=angles,
            azimuth_rad=angles,
            num_signals=9,
            spatial_smooth=1,
        )
