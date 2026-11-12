"""CFAR: the false-alarm law measured, the oracle matched, a known target detected.

Every detector is square law and solves its threshold constant from its own
exact false-alarm law, for the training count each cell actually has. The
claims below are therefore measurements against ``pfa`` itself, not against a
second law the detector happens to follow:

* the laws agree with independent forms (Gandhi-Kassam's closed form for equal
  halves, a numerical integral from the definition otherwise, Rohling's
  product for the ordered statistic);
* on complex Gaussian noise the measured rate of every detector is within
  20 % of ``pfa`` at 1e-2, 1e-3 and 1e-4, over more than a million cells and
  over the whole map, edges included;
* every detector's threshold equals an independent shifted-sum oracle that
  wraps Doppler and truncates range (``tests/support/reference_cfar.py``).

Before the square-law change the facade's own point-cloud path fed magnitudes
to a power-law constant and measured zero false alarms in 4.2 million cells at
all three rates, and the ordered-statistic detector achieved 0.18, 0.08 and
0.05 of its declared rate.
"""

from __future__ import annotations

import math

import pytest
import torch
from support.reference_cfar import reference_cfar, ring_offsets, solve_alpha

from witwin.radar.processing import (
    Detections,
    ca_cfar,
    ca_cfar_1d,
    cfar_false_alarm_probability,
    combine_coherent,
    go_cfar,
    group_peaks,
    os_cfar,
    so_cfar,
)

DETECTORS = {"ca": ca_cfar, "go": go_cfar, "so": so_cfar, "os": os_cfar}


def _complex_noise(shape, *, variance: float = 1.0, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    scale = math.sqrt(variance / 2.0)
    real = torch.randn(shape, generator=generator, dtype=torch.float64) * scale
    imaginary = torch.randn(shape, generator=generator, dtype=torch.float64) * scale
    return torch.complex(real, imaginary)


def _power(values: torch.Tensor) -> torch.Tensor:
    return values.real.square() + values.imag.square()


# ---------------------------------------------------------------------------
# The laws
# ---------------------------------------------------------------------------


def test_the_two_half_laws_match_gandhi_kassam_for_equal_halves():
    """``P_so = 2 sum_k C(n-1+k, k) (2 + a/n)^-(n+k)`` and ``P_go = 2 (1 + a/n)^-n - P_so``."""

    for n, alpha in ((4, 2.5), (12, 7.0), (40, 30.0)):
        q = 2.0 + alpha / n
        smaller = 2.0 * sum(math.comb(n - 1 + k, k) * q ** (-(n + k)) for k in range(n))
        greater = 2.0 * (1.0 + alpha / n) ** (-n) - smaller
        counts = torch.tensor([n])
        a = torch.tensor([alpha], dtype=torch.float64)
        assert float(cfar_false_alarm_probability("so", a, counts, counts)) == pytest.approx(smaller, rel=1e-10)
        assert float(cfar_false_alarm_probability("go", a, counts, counts)) == pytest.approx(greater, rel=1e-10)


@pytest.mark.parametrize("kind", ["ca", "go", "so", "os"])
def test_the_solved_constants_match_an_independent_solve_for_unequal_halves(kind):
    """Truncated rings at the range edges have unequal halves; their constants are solved, not borrowed."""

    from witwin.radar.processing.detection import _ring, _threshold_constants

    ring = _ring((1, 2), (2, 3), rank=2)
    alpha, _, _ = _threshold_constants(kind, 40, ring, 1e-3, 0.75)
    lag, lead = ring_offsets((1, 2), (2, 3))
    for column in (0, 1, 3, 20, 39):
        n1 = sum(1 for _, r in lag if 0 <= column + r < 40)
        n2 = sum(1 for _, r in lead if 0 <= column + r < 40)
        rank = min(int(0.75 * (n1 + n2)), n1 + n2 - 1) + 1
        expected = solve_alpha(kind, 1e-3, n1, n2, rank if kind == "os" else 0)
        assert float(alpha[column]) == pytest.approx(expected, rel=1e-9), (kind, column, n1, n2)


# ---------------------------------------------------------------------------
# The measured rate: M1-4 acceptance
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def noise_maps():
    """2.1 million complex Gaussian cells in 32 maps of ``[256, 256]``."""

    return _complex_noise((32, 256, 256), seed=20261112).to(torch.complex64)


@pytest.mark.parametrize("pfa", [1e-2, 1e-3, 1e-4])
@pytest.mark.parametrize("kind", ["ca", "os", "go", "so"])
def test_m1_4_the_measured_false_alarm_rate_is_the_declared_one(noise_maps, kind, pfa):
    """Within 20 % of ``pfa`` over every cell of 2.1e6 noise cells, at the point-cloud default windows.

    At 1e-4 that is 210 expected false alarms, so 20 % is about three binomial
    standard errors; the measured ratios are printed by the failure message.
    """

    detector = DETECTORS[kind]
    count = 0
    for chunk in noise_maps.split(4):
        count += int(detector(chunk, guard_cells=(2, 4), training_cells=(4, 8), pfa=pfa).mask.sum())
    measured = count / noise_maps.numel()
    assert abs(measured / pfa - 1.0) <= 0.2, (kind, pfa, measured / pfa)


def test_the_edge_columns_hold_the_rate_too(noise_maps):
    """The truncated rings at both ends of the range axis, alone, at 1e-2."""

    for kind, detector in DETECTORS.items():
        mask = detector(noise_maps[:16], guard_cells=(2, 4), training_cells=(4, 8), pfa=1e-2).mask
        edges = torch.cat((mask[..., :12], mask[..., -12:]), dim=-1)
        measured = float(edges.to(torch.float64).mean())
        assert abs(measured / 1e-2 - 1.0) <= 0.15, (kind, measured)


def test_the_rate_does_not_depend_on_the_noise_level():
    """It is CONSTANT false alarm rate, and that is the whole claim."""

    for detector in DETECTORS.values():
        rates = []
        for variance in (1e-4, 1.0, 1e4):
            noise = _complex_noise((64, 128), variance=variance, seed=99)
            detected = detector(noise, guard_cells=(1, 1), training_cells=(2, 2), pfa=1e-2)
            rates.append(int(detected.mask.sum()))
        assert max(rates) - min(rates) <= 1, (detector.__name__, rates)


# ---------------------------------------------------------------------------
# The oracle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["ca", "go", "so", "os"])
def test_every_detector_matches_the_shifted_sum_oracle(kind):
    """Wrap along Doppler, truncation along range, the solved constant, the statistic."""

    noise = _power(_complex_noise((2, 24, 40), seed=77))
    noise[1, 0, 5] = 300.0  # a return near the Doppler wrap and the range edge
    detected = DETECTORS[kind](noise, guard_cells=(1, 2), training_cells=(2, 3), pfa=1e-3)
    expected = reference_cfar(kind, noise, guard_cells=(1, 2), training_cells=(2, 3), pfa=1e-3)
    torch.testing.assert_close(detected.threshold, expected, rtol=1e-9, atol=1e-12)
    assert torch.equal(detected.mask, noise > expected)


def test_the_one_dimensional_detector_matches_the_oracle_on_a_one_bin_doppler_axis():
    profile = _power(_complex_noise((3, 64), seed=5))
    detected = ca_cfar_1d(profile, guard_cells=2, training_cells=5, pfa=1e-3)
    expected = reference_cfar("ca", profile.unsqueeze(-2), guard_cells=(0, 2), training_cells=(0, 5), pfa=1e-3)
    torch.testing.assert_close(detected.threshold, expected.squeeze(-2), rtol=1e-9, atol=1e-12)


def test_the_doppler_ring_wraps_instead_of_replicating():
    """A return in Doppler bin 0 sits in the training ring of a cell in bin D - 1."""

    values = torch.ones((16, 32), dtype=torch.float64)
    values[0, 10] = 1e3
    raised = ca_cfar(values, guard_cells=(0, 0), training_cells=(1, 1)).threshold
    assert float(raised[15, 10]) > float(raised[7, 10]) * 10.0


def test_complex_input_is_detected_on_its_squared_magnitude():
    """Square law by default: complex IQ and its ``|x|^2`` are the same detection."""

    cube = _complex_noise((4, 32, 48), seed=3)
    for detector in DETECTORS.values():
        from_complex = detector(cube, guard_cells=(1, 2), training_cells=(2, 3))
        from_power = detector(_power(cube), guard_cells=(1, 2), training_cells=(2, 3))
        assert torch.equal(from_complex.mask, from_power.mask)
        assert torch.equal(from_complex.threshold, from_power.threshold)


def test_the_point_cloud_input_is_a_power_map():
    """``Frame.points`` runs ``ca_cfar(combine_coherent(rd))``: the combined map is ``|sum|^2``."""

    cube = _complex_noise((3, 4, 16, 24), seed=8)
    combined = cube.reshape(12, 16, 24).sum(dim=0)
    torch.testing.assert_close(combine_coherent(cube), _power(combined), rtol=1e-12, atol=0.0)
    assert torch.equal(ca_cfar(combine_coherent(cube)).mask, ca_cfar(combined).mask)


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------


def test_a_target_at_a_known_signal_to_noise_ratio_is_detected():
    noise = _power(_complex_noise((64, 96), variance=2.0, seed=7))
    cell = (30, 61)
    noise[cell] = 2.0 * 100.0
    for detector in DETECTORS.values():
        detected = detector(noise, guard_cells=(2, 3), training_cells=(4, 6), pfa=1e-4)
        assert bool(detected.mask[cell]), detector.__name__


def test_the_ordered_statistic_detector_survives_a_second_target_in_its_ring():
    """The reason it exists: the mean is dragged up by an interferer, the rank is not."""

    values = torch.full((48, 48), 1.0, dtype=torch.float64)
    values[24, 24] = 60.0
    values[24, 30] = 60.0
    detected = os_cfar(values, guard_cells=(2, 2), training_cells=(4, 4), rank_fraction=0.75)
    assert bool(detected.mask[24, 24])
    assert bool(detected.mask[24, 30])


def test_greatest_of_holds_down_false_alarms_at_a_clutter_edge():
    """A step in the noise level along range: the low-side mean of CA lets the high side through."""

    noise = _power(_complex_noise((128, 128), seed=17))
    noise[:, 64:] *= 100.0
    window = {"guard_cells": (1, 2), "training_cells": (2, 6), "pfa": 1e-3}
    edge = slice(60, 72)
    ca_alarms = int(ca_cfar(noise, **window).mask[:, edge].sum())
    go_alarms = int(go_cfar(noise, **window).mask[:, edge].sum())
    assert go_alarms < ca_alarms


def test_smallest_of_resolves_a_target_beside_a_stronger_one():
    """An interferer in one half raises CA's level; SO reads the clean half."""

    values = torch.full((32, 64), 1.0, dtype=torch.float64)
    values[16, 30] = 30.0
    values[16, 36] = 3000.0
    window = {"guard_cells": (1, 1), "training_cells": (2, 6), "pfa": 1e-4}
    assert not bool(ca_cfar(values, **window).mask[16, 30])
    assert bool(so_cfar(values, **window).mask[16, 30])


# ---------------------------------------------------------------------------
# Peak grouping
# ---------------------------------------------------------------------------


def _windowed_tone_map(*, seed: int, snr_db: float):
    """A Hann-windowed off-bin tone in a ``[64, 128]`` Range-Doppler map, with unit-power noise."""

    doppler, ranges = 64, 128
    slow = torch.arange(doppler, dtype=torch.float64).reshape(-1, 1)
    fast = torch.arange(ranges, dtype=torch.float64).reshape(1, -1)
    amplitude = 10 ** (snr_db / 20.0)
    tone = amplitude * torch.exp(2j * math.pi * (10.37 * slow / doppler + 41.61 * fast / ranges))
    signal = tone + _complex_noise((doppler, ranges), seed=seed)
    hann_d = 0.5 - 0.5 * torch.cos(2 * math.pi * torch.arange(doppler, dtype=torch.float64) / doppler)
    hann_r = 0.5 - 0.5 * torch.cos(2 * math.pi * torch.arange(ranges, dtype=torch.float64) / ranges)
    windowed = signal * hann_d.reshape(-1, 1) * hann_r.reshape(1, -1)
    return torch.fft.fft2(windowed, norm="forward")


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_peak_grouping_turns_one_tone_into_one_detection(seed):
    """F-7: the main lobe and leakage of one tone light many cells; grouping leaves its peak."""

    rd = _windowed_tone_map(seed=seed, snr_db=40.0)
    found = ca_cfar(rd, guard_cells=(2, 4), training_cells=(4, 8), pfa=1e-4)
    near = torch.zeros_like(found.mask)
    near[2:19, 30:55] = True  # the tone's main lobe and window leakage, well past the CFAR guard band
    assert int((found.mask & near).sum()) > 1
    grouped = group_peaks(found, rd, neighbourhood=(5, 5))
    cells = torch.argwhere(grouped.mask & near)
    assert cells.tolist() == [[10, 42]], cells.tolist()


def test_peak_grouping_breaks_a_tie_by_position():
    values = torch.zeros((8, 8), dtype=torch.float64)
    values[3, 3] = values[3, 4] = 5.0
    everything = Detections(mask=torch.ones((8, 8), dtype=torch.bool), threshold=torch.zeros((8, 8)))
    grouped = group_peaks(everything, values)
    assert torch.argwhere(grouped.mask & (values > 0)).tolist() == [[3, 3]]


# ---------------------------------------------------------------------------
# Batching and the record
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["ca", "go", "so", "os"])
def test_a_batched_map_equals_a_python_loop_over_its_slices_bitwise(kind):
    generator = torch.Generator().manual_seed(2024)
    cube = torch.rand((5, 32, 48), generator=generator, dtype=torch.float64) + 0.1
    cube[2, 10, 20] = 30.0
    detector = DETECTORS[kind]
    batched = detector(cube, guard_cells=(1, 2), training_cells=(3, 4), pfa=1e-3)
    assert tuple(batched.mask.shape) == (5, 32, 48)
    for index in range(5):
        one = detector(cube[index], guard_cells=(1, 2), training_cells=(3, 4), pfa=1e-3)
        assert torch.equal(batched.mask[index], one.mask)
        assert torch.equal(batched.threshold[index], one.threshold)


def test_a_rank_four_beam_cube_carries_both_of_its_leading_axes():
    generator = torch.Generator().manual_seed(5)
    cube = torch.rand((3, 4, 24, 24), generator=generator, dtype=torch.float32) + 0.1
    detected = ca_cfar(cube, guard_cells=(1, 1), training_cells=(2, 2))
    assert tuple(detected.mask.shape) == (3, 4, 24, 24)
    flat = ca_cfar(cube.reshape(12, 24, 24), guard_cells=(1, 1), training_cells=(2, 2))
    assert torch.equal(detected.mask.reshape(12, 24, 24), flat.mask)


def test_the_range_only_detector_finds_a_target_in_a_profile_with_no_doppler_axis():
    profile = _power(_complex_noise((7, 512), variance=3.0, seed=808))
    profile[4, 200] = 3.0 * 200.0
    detected = ca_cfar_1d(profile, guard_cells=2, training_cells=8, pfa=1e-4)
    assert tuple(detected.mask.shape) == (7, 512)
    assert bool(detected.mask[4, 200])


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


def test_invalid_windows_and_rates_are_refused():
    values = torch.ones((8, 8), dtype=torch.float64)
    for detector in DETECTORS.values():
        with pytest.raises(ValueError, match="no training cells"):
            detector(values, guard_cells=(2, 2), training_cells=(0, 0))
        with pytest.raises(ValueError, match="pfa"):
            detector(values, guard_cells=(1, 1), training_cells=(1, 1), pfa=1.5)
        with pytest.raises(ValueError, match=r"count a cell twice. Use guard_cells\[0\] \+ training_cells\[0\] <= 3"):
            detector(values, guard_cells=(2, 1), training_cells=(3, 1))
    with pytest.raises(ValueError, match="no training cells"):
        ca_cfar_1d(values, guard_cells=1, training_cells=0)
    with pytest.raises(TypeError, match="power map"):
        ca_cfar(torch.ones((8, 8), dtype=torch.int64))
