"""The beam / velocity / range cube former, and its conventional steering.

Host only and synthetic on purpose. What is under test is the CONTRACTION and
the array manifold, and both are exactly checkable against a wavefront this file
builds itself: a cube that is one steering vector reproduces a unit response on
the beam it was steered from and less everywhere else. Driving that through a
propagation solve would test the solve, not the former.

The conjugation trap appears here for the second time. An FMCW beat cube is the
conjugate of Channel's product in SPACE as well as in slow time, so its manifold
runs the other way; both are covered below, from the same ``doppler_sign`` the
Doppler stage reads.
"""

from __future__ import annotations

import math

import pytest
import torch

from support import exact_bin_grid as grid
from witwin.radar.processing import (
    ProcessingAxes,
    RangeDopplerMap,
    beam_cube,
    conventional_steering,
    virtual_element_offsets_m,
)
from witwin.radar.synthesis.contracts import SynthesisResult


PAIRS = grid.FMCW_NUM_TX * grid.FMCW_NUM_RX
DOPPLER = 8
RANGES = 16


def _axes(waveform: str = "fmcw") -> ProcessingAxes:
    array = grid.array_spec()
    if waveform == "fmcw":
        spec = grid.fmcw_spec(DOPPLER)
        cube = torch.zeros((DOPPLER, PAIRS, RANGES), dtype=torch.complex64)
        result = SynthesisResult.from_fmcw_beat(cube, spec)
    else:
        spec = grid.ofdm_spec(num_symbols=DOPPLER)
        cube = torch.zeros(
            (DOPPLER, PAIRS, spec.num_subcarriers), dtype=torch.complex64
        )
        result = SynthesisResult.from_ofdm_cfr(cube, spec)
    return ProcessingAxes.from_synthesis(result, spec, array)


def _directions(angles_rad) -> torch.Tensor:
    return torch.tensor(
        [[math.sin(a), 0.0, math.cos(a)] for a in angles_rad], dtype=torch.float64
    )


def _map(axes, data) -> RangeDopplerMap:
    return RangeDopplerMap(
        data=data, axes=axes, window="rectangular", window_coherent_gain=1.0
    )


# ---------------------------------------------------------------------------
# The array manifold
# ---------------------------------------------------------------------------


def test_the_virtual_element_is_the_sum_of_its_transmitter_and_receiver_offsets():
    """And the pair rank is SINK MAJOR, which is what decides which is which.

    Under ``PAIR_RANK_LAYOUT`` the transmitter index of pair ``p`` is
    ``p % num_tx``, not ``p // num_rx``. On a square array the two are a
    transpose of each other, so getting it backwards mis-steers every angle
    without changing a single shape.
    """

    axes = _axes()
    offsets = virtual_element_offsets_m(axes)
    assert tuple(offsets.shape) == (PAIRS, 3)
    spacing = axes.element_spacing_m
    for pair in range(PAIRS):
        tx = axes.tx_loc_half_wavelength[pair % axes.num_tx]
        rx = axes.rx_loc_half_wavelength[pair // axes.num_tx]
        expected = [(a + b) * spacing for a, b in zip(tx, rx, strict=True)]
        assert [float(v) for v in offsets[pair]] == pytest.approx(expected, rel=1e-12)


def test_the_manifold_runs_the_other_way_for_a_conjugated_beat_cube():
    """One derived quantity, two reconciliations, no second sign decision."""

    directions = _directions((0.3,))
    beat = conventional_steering(_axes("fmcw"), directions)
    channel = conventional_steering(_axes("ofdm"), directions)
    assert _axes("fmcw").doppler_sign == 1
    assert _axes("ofdm").doppler_sign == -1
    torch.testing.assert_close(beat, channel.conj(), rtol=1e-6, atol=1e-7)


def test_normalized_weights_give_a_unit_response_to_a_matched_wavefront():
    """``w^H a = 1``: the amplitude convention the range and Doppler stages use.

    It is also the constraint an MVDR weight satisfies, so stage S4's weight
    owner is interchangeable with this one at :func:`beam_cube` without a scale
    factor appearing between them.
    """

    axes = _axes()
    directions = _directions((-0.3, 0.0, 0.25))
    weights = conventional_steering(axes, directions)
    manifold = conventional_steering(axes, directions, normalize=False)
    response = (weights.conj() * manifold).sum(dim=0)
    torch.testing.assert_close(
        response,
        torch.ones_like(response),
        rtol=1e-6,
        atol=1e-6,
    )


# ---------------------------------------------------------------------------
# The former
# ---------------------------------------------------------------------------


def test_a_wavefront_from_one_beam_peaks_on_that_beam():
    """A cube that IS one steering vector reproduces unit amplitude on its beam."""

    axes = _axes()
    angles = (-0.4, -0.2, 0.0, 0.2, 0.4)
    directions = _directions(angles)
    weights = conventional_steering(axes, directions)
    manifold = conventional_steering(axes, directions, normalize=False)

    data = torch.zeros((PAIRS, DOPPLER, RANGES), dtype=torch.complex64)
    steered = 3
    data[:, 4, 7] = manifold[:, steered]
    cube = beam_cube(_map(axes, data), weights, directions=directions)

    assert tuple(cube.data.shape) == (len(angles), DOPPLER, RANGES)
    response = cube.data[:, 4, 7].abs()
    assert int(response.argmax()) == steered
    assert float(response[steered]) == pytest.approx(1.0, rel=1e-5)
    for beam in range(len(angles)):
        if beam != steered:
            assert float(response[beam]) < 1.0
    # Everything the wavefront did not occupy stays exactly zero.
    assert torch.equal(cube.data[:, 0, 0], torch.zeros_like(cube.data[:, 0, 0]))
    assert cube.directions is directions
    assert cube.range_axis is axes.range_m
    assert cube.doppler_axis is axes.velocity_mps


def test_a_tx_rx_map_and_a_flat_pair_map_form_the_same_cube():
    """``[TX, RX]`` IS ``[P]`` reshaped, in the published sink-major order."""

    axes = _axes()
    directions = _directions((-0.2, 0.1))
    weights = conventional_steering(axes, directions)
    flat = torch.randn(PAIRS, DOPPLER, RANGES, dtype=torch.complex64)
    grid_shaped = flat.reshape(axes.num_rx, axes.num_tx, DOPPLER, RANGES)
    # The cube's array axes are [tx, rx]; the pair rank is rx-major, so the
    # reshape above is [rx, tx] and has to be transposed to match.
    grid_shaped = grid_shaped.permute(1, 0, 2, 3).contiguous()

    from_flat = beam_cube(_map(axes, flat), weights, directions=directions)
    from_grid = beam_cube(
        _map(axes, grid_shaped.permute(1, 0, 2, 3).contiguous()),
        weights,
        directions=directions,
    )
    torch.testing.assert_close(from_flat.data, from_grid.data, rtol=1e-6, atol=1e-7)


def test_a_planar_beam_grid_keeps_both_of_its_axes():
    axes = _axes()
    azimuth = _directions((-0.2, 0.0, 0.2))
    elevation = torch.tensor(
        [[0.0, math.sin(a), math.cos(a)] for a in (-0.1, 0.1)], dtype=torch.float64
    )
    planar = torch.nn.functional.normalize(
        azimuth.reshape(3, 1, 3) + elevation.reshape(1, 2, 3), dim=-1
    )
    weights = conventional_steering(axes, planar)
    assert tuple(weights.shape) == (PAIRS, 3, 2)

    data = torch.zeros((PAIRS, DOPPLER, RANGES), dtype=torch.complex64)
    data[:, 1, 2] = 1.0
    cube = beam_cube(_map(axes, data), weights, directions=planar)
    assert tuple(cube.data.shape) == (3, 2, DOPPLER, RANGES)


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_the_former_refuses_a_front_end_mismatch_and_a_missing_direction_grid():
    axes = _axes()
    directions = _directions((0.0, 0.2))
    weights = conventional_steering(axes, directions)
    data = torch.zeros((PAIRS, DOPPLER, RANGES), dtype=torch.complex64)

    with pytest.raises(ValueError, match="same front end"):
        beam_cube(_map(axes, data), weights[:2], directions=directions)
    with pytest.raises(ValueError, match="one statement"):
        beam_cube(_map(axes, data), weights, directions=directions[:1])
    with pytest.raises(TypeError):
        beam_cube(data, weights, directions=directions)
    with pytest.raises(ValueError, match=r"\[\*beam, 3\]"):
        conventional_steering(axes, torch.zeros(4, 2, dtype=torch.float64))
