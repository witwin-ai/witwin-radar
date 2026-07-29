"""The array record and the two weight families.

Host only and synthetic on purpose: what is under test is where an element is
and what a weight does to a wavefront, and both are exactly checkable against a
wavefront this file builds itself.
"""

from __future__ import annotations

import math

import pytest
import torch

from witwin.radar.processing import ArrayGeometry, conventional_steering, mvdr_weights
from witwin.radar.synthesis.assembly import SPEED_OF_LIGHT_M_PER_S


TX = ((0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (0.0, 0.0, 1.0))
RX = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0))
WAVELENGTH_M = SPEED_OF_LIGHT_M_PER_S / 77e9


def _array(*, spacing_m: float | None = None, phase_sign: int = -1) -> ArrayGeometry:
    return ArrayGeometry.from_offsets(
        TX,
        RX,
        element_spacing_m=(
            WAVELENGTH_M / 2.0 if spacing_m is None else spacing_m
        ),
        wavelength_m=WAVELENGTH_M,
        phase_sign=phase_sign,
    )


def _direction(azimuth_rad: float, elevation_rad: float = 0.0) -> torch.Tensor:
    return torch.tensor(
        [
            [
                math.sin(azimuth_rad) * math.cos(elevation_rad),
                math.cos(azimuth_rad) * math.cos(elevation_rad),
                math.sin(elevation_rad),
            ]
        ],
        dtype=torch.float64,
    )


# ---------------------------------------------------------------------------
# ArrayGeometry
# ---------------------------------------------------------------------------


def test_the_element_table_is_tx_major_and_reads_its_spacing_as_data():
    """No half wavelength is written into any source file. That is the point.

    ``SensorArraySpec.element_spacing_m`` is a half wavelength, and for a
    conventional array so is this - but it arrives as a NUMBER. A quarter-wave
    array is expressible, which is what ``MUSICImager``'s literal
    ``spacing = 0.5`` made impossible.
    """

    array = _array()
    assert array.spacing_wavelengths == 0.5
    assert tuple(array.element_positions_m.shape) == (12, 3)
    for pair in range(12):
        tx = TX[pair // 4]
        rx = RX[pair % 4]
        expected = [(a + b) * array.element_spacing_m for a, b in zip(tx, rx)]
        assert [float(v) for v in array.element_positions_m[pair]] == pytest.approx(
            expected, rel=1e-12
        )

    quarter = _array(spacing_m=WAVELENGTH_M / 4.0)
    assert quarter.spacing_wavelengths == 0.25
    torch.testing.assert_close(
        quarter.element_positions_m * 2.0,
        array.element_positions_m,
        rtol=1e-12,
        atol=0.0,
    )


def test_the_first_eight_virtual_elements_of_this_array_are_a_uniform_line():
    """The precondition the phase-comparison estimator is written against.

    Transmitters at 0 and 4 half wavelengths with four receivers at 0..3 give
    virtual elements at 0..7 with no gap and no repeat. The default fixture
    array - transmitters at 0 and 2 - gives 0,1,2,3,2,3,4,5, which is why every
    legacy exact-angle test used a synthetic array instead of the real one.
    """

    array = _array()
    x = array.element_positions_m[:8, 0] / array.element_spacing_m
    assert x.tolist() == pytest.approx([float(index) for index in range(8)], rel=1e-15)


def test_the_two_phasor_conventions_give_conjugate_manifolds():
    array = _array(phase_sign=-1)
    beat = _array(phase_sign=1)
    directions = _direction(0.3)
    torch.testing.assert_close(
        conventional_steering(beat, directions),
        conventional_steering(array, directions).conj(),
        rtol=1e-6,
        atol=1e-7,
    )


# ---------------------------------------------------------------------------
# Conventional weights
# ---------------------------------------------------------------------------


def test_normalized_weights_give_a_unit_response_to_a_matched_wavefront():
    array = _array()
    directions = torch.cat([_direction(a) for a in (-0.3, 0.0, 0.25)], dim=0)
    weights = conventional_steering(array, directions)
    manifold = conventional_steering(array, directions, normalize=False)
    response = (weights.conj() * manifold).sum(dim=0)
    torch.testing.assert_close(
        response, torch.ones_like(response), rtol=1e-6, atol=1e-6
    )


def test_the_manifold_matches_a_wavefront_built_from_the_delay_convention():
    """The ABSOLUTE spatial phase sign, against a wavefront this test derives.

    The test above cannot see a global conjugation: both of its sides come out
    of :func:`conventional_steering`, so conjugating the manifold conjugates the
    weights with it and the response stays one. Here the wavefront is built from
    the propagation convention instead - a far-field source at ``u`` reaches the
    element at ``r`` over ``d0 - <r, u>``, and Channel publishes ``exp(-j k d)``,
    so the element carries ``exp(+j k <r, u>)`` relative to the origin - and the
    unnormalised manifold must BE that wavefront for ``phase_sign = -1``.

    The conjugate wavefront is asserted NOT to form, which is what makes this an
    absolute statement: a steering vector that agreed with both would be saying
    nothing about which way the array looks.
    """

    array = _array(phase_sign=-1)
    direction = _direction(0.3)
    wavenumber = 2.0 * math.pi / array.wavelength_m

    hand_built = []
    for position in array.element_positions_m.tolist():
        projection = sum(p * d for p, d in zip(position, direction[0].tolist()))
        # exp(-j k d) at d = -projection, the far-field path relative to origin.
        phase = -wavenumber * (-projection)
        hand_built.append(complex(math.cos(phase), math.sin(phase)))
    wavefront = torch.tensor(hand_built, dtype=torch.complex128).reshape(-1, 1)

    manifold = conventional_steering(
        array, direction, normalize=False, dtype=torch.complex128
    )
    torch.testing.assert_close(manifold, wavefront, rtol=1e-9, atol=1e-9)

    weights = conventional_steering(array, direction, dtype=torch.complex128)
    matched = (weights.conj() * wavefront).sum(dim=0)
    torch.testing.assert_close(
        matched, torch.ones_like(matched), rtol=1e-9, atol=1e-9
    )

    # A conjugated array response is a DIFFERENT look direction, and this
    # off-broadside one does not form: the sign is pinned, not free.
    reversed_response = (weights.conj() * wavefront.conj()).sum(dim=0)
    assert float(reversed_response.abs().max()) < 0.5


def test_a_steering_grid_needs_three_components_and_an_array_record():
    array = _array()
    with pytest.raises(ValueError, match=r"\[\*beam, 3\]"):
        conventional_steering(array, torch.zeros(4, 2, dtype=torch.float64))
    with pytest.raises(TypeError, match="ArrayGeometry"):
        conventional_steering(object(), _direction(0.0))


# ---------------------------------------------------------------------------
# MVDR
# ---------------------------------------------------------------------------


def _snapshots(array: ArrayGeometry, angles, powers, *, noise: float, count: int):
    """``[P, T]``: a sum of plane waves in independent complex Gaussian noise."""

    generator = torch.Generator().manual_seed(4242)
    pairs = array.sensor_pair_count
    data = torch.complex(
        torch.randn((pairs, count), generator=generator, dtype=torch.float64),
        torch.randn((pairs, count), generator=generator, dtype=torch.float64),
    ) * math.sqrt(noise / 2.0)
    for angle, power in zip(angles, powers, strict=True):
        manifold = conventional_steering(
            array, _direction(angle), normalize=False, dtype=torch.complex128
        )
        amplitude = torch.complex(
            torch.randn((1, count), generator=generator, dtype=torch.float64),
            torch.randn((1, count), generator=generator, dtype=torch.float64),
        ) * math.sqrt(power / 2.0)
        data = data + manifold * amplitude
    return data


def test_mvdr_passes_its_own_look_direction_and_nulls_an_interferer():
    """``w^H a = 1`` exactly, and the interferer's output power collapses.

    The distortionless constraint is an identity, so it is asserted to float
    precision rather than to a tolerance chosen to pass. The null is a
    comparison against the conventional weights on the SAME snapshots: MVDR
    exists to do better than delay-and-sum in exactly this case, and if it does
    not, the solve is wrong.
    """

    array = _array()
    look, interferer = 0.0, 0.6
    snapshots = _snapshots(
        array, (look, interferer), (1.0, 100.0), noise=1e-3, count=512
    )
    covariance = (snapshots @ snapshots.conj().transpose(0, 1)) / snapshots.shape[1]

    steering = conventional_steering(
        array, _direction(look), normalize=False, dtype=torch.complex128
    )
    weights = mvdr_weights(covariance, steering, diagonal_loading=1e-6)
    assert tuple(weights.shape) == (12, 1)

    response = (weights.conj() * steering).sum(dim=0)
    torch.testing.assert_close(
        response, torch.ones_like(response), rtol=1e-9, atol=1e-9
    )

    # The null, measured where it matters: how much of the interferer's own
    # manifold each weight vector lets through. Total output power would be the
    # wrong statistic, because the distortionless constraint means MVDR CANNOT
    # suppress the desired signal and the comparison would be dominated by it.
    conventional = conventional_steering(
        array, _direction(look), dtype=torch.complex128
    )
    interfering = conventional_steering(
        array, _direction(interferer), normalize=False, dtype=torch.complex128
    )
    mvdr_leak = float((weights.conj() * interfering).sum().abs() ** 2)
    conventional_leak = float((conventional.conj() * interfering).sum().abs() ** 2)
    assert mvdr_leak < conventional_leak / 100.0, (mvdr_leak, conventional_leak)

    # And the total output power is then the desired signal alone, to the noise
    # floor: MVDR has removed a hundred-fold interferer entirely.
    mvdr_power = float(
        (weights.conj().transpose(0, 1) @ covariance @ weights).real.squeeze()
    )
    assert mvdr_power == pytest.approx(1.0, rel=0.05)


def test_mvdr_carries_a_batch_and_demands_an_explicit_loading():
    array = _array()
    directions = torch.cat([_direction(a) for a in (-0.2, 0.0, 0.2)], dim=0)
    steering = conventional_steering(
        array, directions, normalize=False, dtype=torch.complex128
    )
    snapshots = _snapshots(array, (0.0,), (1.0,), noise=1.0, count=64)
    covariance = (snapshots @ snapshots.conj().transpose(0, 1)) / 64
    batched = torch.stack((covariance, covariance * 2.0), dim=0)

    weights = mvdr_weights(batched, steering, diagonal_loading=1e-3)
    assert tuple(weights.shape) == (2, 12, 3)
    # The loading is a FRACTION of trace(R) / P, so it is scale free and a
    # covariance scaled by two gives the same weights.
    torch.testing.assert_close(weights[0], weights[1], rtol=1e-9, atol=1e-12)

    with pytest.raises(TypeError):
        mvdr_weights(covariance.real, steering, diagonal_loading=1e-3)
    with pytest.raises(ValueError, match="same front end"):
        mvdr_weights(covariance[:4, :4], steering, diagonal_loading=1e-3)
    with pytest.raises(TypeError):
        mvdr_weights(covariance, steering)


def test_diagonal_loading_makes_a_rank_deficient_covariance_solvable():
    """Fewer snapshots than elements is exactly singular, and it is common."""

    array = _array()
    snapshots = _snapshots(array, (0.0,), (1.0,), noise=0.0, count=1)
    covariance = snapshots @ snapshots.conj().transpose(0, 1)
    steering = conventional_steering(
        array, _direction(0.3), normalize=False, dtype=torch.complex128
    )
    weights = mvdr_weights(covariance, steering, diagonal_loading=1e-2)
    assert torch.isfinite(weights).all()
    response = (weights.conj() * steering).sum(dim=0)
    torch.testing.assert_close(
        response, torch.ones_like(response), rtol=1e-9, atol=1e-9
    )
