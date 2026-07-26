"""CFAR: the false-alarm law measured, a known target detected, batching exact.

The nominal false-alarm rate is not a docstring here. On complex Gaussian noise
of any variance, a cell-averaging detector with ``N`` training cells and scale
``alpha`` has

    ``P_fa = (1 + alpha / N) ** (-N)``

and the scale the module computes is ``alpha = N (P_fa ** (-1/N) - 1)``, which
inverts it exactly. The test below measures the EMPIRICAL rate over a large
sample and compares to three standard errors, so neither the constant nor the
tolerance can be moved without the measurement moving with it.

Two conditions the identity needs, both honoured below rather than assumed: the
detector must be fed POWER, because the exponential distribution is the
distribution of ``|x| ** 2``; and the border cells must be excluded, because
replicate padding correlates a border cell's ring with itself.
"""

from __future__ import annotations

import math

import pytest
import torch

from witwin.radar.processing import (
    Detections,
    ca_cfar,
    ca_cfar_1d,
    ca_cfar_fast,
    os_cfar,
)


def _complex_gaussian_power(shape, *, variance: float, seed: int) -> torch.Tensor:
    """``|x| ** 2`` for circular complex Gaussian ``x``: exponential, mean ``variance``."""

    generator = torch.Generator().manual_seed(seed)
    scale = math.sqrt(variance / 2.0)
    real = torch.randn(shape, generator=generator, dtype=torch.float64) * scale
    imaginary = torch.randn(shape, generator=generator, dtype=torch.float64) * scale
    return real.square() + imaginary.square()


# ---------------------------------------------------------------------------
# The false-alarm law
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pfa", [1e-2, 3e-3])
def test_the_measured_false_alarm_rate_matches_the_analytic_one(pfa):
    """``P_fa = (1 + alpha / N) ** (-N)``, to three standard errors."""

    guard, training = (1, 1), (2, 2)
    outer_d = guard[0] + training[0]
    outer_r = guard[1] + training[1]
    n_train = (2 * outer_d + 1) * (2 * outer_r + 1) - (2 * guard[0] + 1) * (
        2 * guard[1] + 1
    )
    alpha = n_train * (pfa ** (-1.0 / n_train) - 1.0)
    analytic = (1.0 + alpha / n_train) ** (-n_train)
    assert analytic == pytest.approx(pfa, rel=1e-12)

    noise = _complex_gaussian_power((320, 320), variance=7.0, seed=31337)
    detected = ca_cfar(noise, guard_cells=guard, training_cells=training, pfa=pfa)

    # Interior only: a border cell's replicate-padded ring repeats its own row,
    # which correlates the estimate with the cell under test.
    interior = detected.mask[outer_d:-outer_d, outer_r:-outer_r]
    count = int(interior.sum())
    total = int(interior.numel())
    measured = count / total
    standard_error = math.sqrt(analytic * (1.0 - analytic) / total)
    assert abs(measured - analytic) <= 3.0 * standard_error, (
        measured,
        analytic,
        standard_error,
    )


@pytest.mark.parametrize("pfa", [1e-2, 3e-3])
def test_the_ordered_statistic_rate_follows_rohlings_law_and_not_its_pfa(pfa):
    """``os_cfar`` misses the ``pfa`` it is handed, and this is what it hits instead.

    It scales the ``k``-th smallest training sample with the constant that
    inverts the false-alarm law of their MEAN, so the declared number is a
    design constant rather than a prediction. The rate it actually achieves has
    its own exact law on exponential power, Rohling's

        ``P_fa = prod_{i=0}^{k-1} (N - i) / (N - i + alpha)``

    which is measured here, so the gap is pinned rather than merely written
    down: neither the constant nor the rank can move without this failing.

    Tolerance is five standard errors of an iid binomial, and that is a LOWER
    bound on the true spread - neighbouring cells share training rings, so their
    detections are correlated. Observed deviations over four independent seeds
    spanned -0.6 to +2.5 of these units.
    """

    guard, training, rank_fraction = (1, 1), (2, 2), 0.75
    outer_d = guard[0] + training[0]
    outer_r = guard[1] + training[1]
    n_train = (2 * outer_d + 1) * (2 * outer_r + 1) - (2 * guard[0] + 1) * (
        2 * guard[1] + 1
    )
    rank = min(int(rank_fraction * n_train), n_train - 1) + 1
    alpha = n_train * (pfa ** (-1.0 / n_train) - 1.0)

    analytic = 1.0
    for offset in range(rank):
        analytic *= (n_train - offset) / (n_train - offset + alpha)

    # The declared rate is NOT achieved, and the miss is in the conservative
    # direction: fewer false alarms than asked for, at the cost of sensitivity.
    assert analytic < 0.25 * pfa

    noise = _complex_gaussian_power((900, 900), variance=7.0, seed=31337)
    detected = os_cfar(
        noise,
        guard_cells=guard,
        training_cells=training,
        rank_fraction=rank_fraction,
        pfa=pfa,
    )
    interior = detected.mask[outer_d:-outer_d, outer_r:-outer_r]
    total = int(interior.numel())
    measured = int(interior.sum()) / total
    standard_error = math.sqrt(analytic * (1.0 - analytic) / total)
    assert abs(measured - analytic) <= 5.0 * standard_error, (
        measured,
        analytic,
        standard_error,
    )


def test_the_rate_does_not_depend_on_the_noise_level():
    """It is CONSTANT false alarm rate, and that is the whole claim."""

    guard, training, pfa = (1, 1), (2, 2), 1e-2
    rates = []
    for variance in (1e-4, 1.0, 1e4):
        noise = _complex_gaussian_power((256, 256), variance=variance, seed=99)
        detected = ca_cfar(noise, guard_cells=guard, training_cells=training, pfa=pfa)
        rates.append(float(detected.mask[3:-3, 3:-3].to(torch.float64).mean()))
    assert rates[0] == pytest.approx(rates[1], rel=1e-12)
    assert rates[1] == pytest.approx(rates[2], rel=1e-12)


def test_a_target_at_a_known_signal_to_noise_ratio_is_detected():
    """20 dB over the local noise floor clears a 1e-4 threshold comfortably."""

    noise = _complex_gaussian_power((64, 96), variance=2.0, seed=7)
    cell = (30, 61)
    noise[cell] = 2.0 * 100.0
    for detector in (ca_cfar, ca_cfar_fast, os_cfar):
        detected = detector(
            noise, guard_cells=(2, 3), training_cells=(4, 6), pfa=1e-4
        )
        assert bool(detected.mask[cell]), detector.__name__
        assert float(noise[cell]) > float(detected.threshold[cell])


def test_the_ordered_statistic_detector_survives_a_second_target_in_its_ring():
    """The reason it exists: the mean is dragged up by an interferer, the rank is not."""

    values = torch.full((48, 48), 1.0, dtype=torch.float64)
    values[24, 24] = 60.0
    values[24, 30] = 60.0
    detected = os_cfar(
        values, guard_cells=(2, 2), training_cells=(4, 4), rank_fraction=0.75
    )
    assert bool(detected.mask[24, 24])
    assert bool(detected.mask[24, 30])


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("detector", [ca_cfar, ca_cfar_fast, os_cfar])
def test_a_batched_map_equals_a_python_loop_over_its_slices_bitwise(detector):
    """The whole point of the rewrite: a beam cube needs no loop over beams.

    Bitwise, not close: the batch axis is the OUTER one in every kernel here, so
    a slice of the batched result is the same reduction over the same elements
    in the same order as the standalone call.
    """

    generator = torch.Generator().manual_seed(2024)
    cube = torch.rand((5, 32, 48), generator=generator, dtype=torch.float64) + 0.1
    cube[2, 10, 20] = 30.0
    batched = detector(cube, guard_cells=(1, 2), training_cells=(3, 4), pfa=1e-3)
    assert tuple(batched.mask.shape) == (5, 32, 48)
    for index in range(5):
        one = detector(
            cube[index], guard_cells=(1, 2), training_cells=(3, 4), pfa=1e-3
        )
        assert torch.equal(batched.mask[index], one.mask)
        assert torch.equal(batched.threshold[index], one.threshold)


def test_a_rank_four_beam_cube_carries_both_of_its_leading_axes():
    generator = torch.Generator().manual_seed(5)
    cube = torch.rand((3, 4, 24, 24), generator=generator, dtype=torch.float32) + 0.1
    detected = ca_cfar_fast(cube, guard_cells=(1, 1), training_cells=(2, 2))
    assert tuple(detected.mask.shape) == (3, 4, 24, 24)
    flat = ca_cfar_fast(
        cube.reshape(12, 24, 24), guard_cells=(1, 1), training_cells=(2, 2)
    )
    assert torch.equal(detected.mask.reshape(12, 24, 24), flat.mask)


# ---------------------------------------------------------------------------
# The one-dimensional detector
# ---------------------------------------------------------------------------


def test_the_range_only_detector_finds_a_target_in_a_profile_with_no_doppler_axis():
    """A range profile had no detector anywhere in this repository."""

    profile = _complex_gaussian_power((7, 512), variance=3.0, seed=808)
    profile[4, 200] = 3.0 * 200.0
    detected = ca_cfar_1d(profile, guard_cells=2, training_cells=8, pfa=1e-4)
    assert tuple(detected.mask.shape) == (7, 512)
    assert bool(detected.mask[4, 200])
    interior = detected.mask[:, 10:-10]
    interior[4, 190] = False
    assert float(interior.to(torch.float64).mean()) < 5e-3


def test_the_one_dimensional_false_alarm_rate_matches_its_own_law():
    pfa = 1e-2
    guard, training = 2, 12
    n_train = 2 * training
    alpha = n_train * (pfa ** (-1.0 / n_train) - 1.0)
    analytic = (1.0 + alpha / n_train) ** (-n_train)
    assert analytic == pytest.approx(pfa, rel=1e-12)

    noise = _complex_gaussian_power((400, 400), variance=5.0, seed=4242)
    detected = ca_cfar_1d(noise, guard_cells=guard, training_cells=training, pfa=pfa)
    margin = guard + training
    interior = detected.mask[:, margin:-margin]
    total = int(interior.numel())
    measured = int(interior.sum()) / total
    standard_error = math.sqrt(analytic * (1.0 - analytic) / total)
    assert abs(measured - analytic) <= 3.0 * standard_error, (measured, analytic)


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------


def test_the_detection_record_pairs_a_mask_with_the_level_it_beat():
    values = torch.rand((16, 16), dtype=torch.float64) + 0.5
    detected = ca_cfar(values, guard_cells=(1, 1), training_cells=(2, 2))
    assert isinstance(detected, Detections)
    assert detected.mask.dtype == torch.bool
    assert detected.threshold.shape == detected.mask.shape
    # The count is a DEVICE scalar: reading it is a synchronization and this
    # package refuses to add one implicitly.
    assert isinstance(detected.count, torch.Tensor)
    assert detected.count.dim() == 0

    with pytest.raises(ValueError, match="same map"):
        Detections(mask=detected.mask, threshold=detected.threshold[:8])
    with pytest.raises(TypeError, match="bool"):
        Detections(mask=detected.threshold, threshold=detected.threshold)


def test_a_ring_with_no_training_cells_is_refused():
    values = torch.ones((8, 8), dtype=torch.float64)
    for detector in (ca_cfar, ca_cfar_fast, os_cfar):
        with pytest.raises(ValueError, match="no training cells"):
            detector(values, guard_cells=(2, 2), training_cells=(0, 0))
    with pytest.raises(ValueError, match="no training cells"):
        ca_cfar_1d(values, guard_cells=1, training_cells=0)
