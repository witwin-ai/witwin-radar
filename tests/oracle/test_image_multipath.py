"""MP-15: ghost range, Doppler and angle of arrival against the image method.

World: one perfectly conducting wall in the plane ``z = 3 m`` (normal along
the receive array axis), one transmitter and eight receivers spaced
``lambda / 2`` along world +z, and one point target at ``(12, 0, -2) m``
moving at ``(-2, 0, 4) m/s``. Everything sits in the ``y = 0`` plane, so the
``"up"`` polarization is perpendicular to every plane of incidence and no
polarization factor enters.

With one bounce there are four round-trip classes. By the image method a leg
that bounces has the length of the straight leg to the mirrored target
``T' = (12, 0, 8)`` and arrives from it, so each class is "transmit leg sees
copy A, receive leg sees copy B":

==================== ========== ========== ==========================
class                tx copy    rx copy    arrival direction at RX
==================== ========== ========== ==========================
direct-direct        T          T          target
direct-reflected     T          T'         mirrored target
reflected-direct     T'         T          target
reflected-reflected  T'         T'         mirrored target (the ghost)
==================== ========== ========== ==========================

The two mixed classes land in one range-Doppler cell (their lengths differ by
1.4 cm here) but arrive from different directions: those are the mixed-path
angles. Two oracles:

* rows, via ``Radar.trace``: every class at every receiver, its path length,
  its phase across the array (angle of arrival) and its phase advance over the
  frame (Doppler), against the float64 image-method closed form;
* the processed cube, via ``Radar.simulate``: the range-Doppler peaks of the
  three resolvable cells, and the beam-scan angles of the targets in them,
  through the public processing stages.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from reference import radar_oracles as ref
from support import oracle_scenes as sc

pytestmark = pytest.mark.gpu

WALL_Z_M = 3.0
TARGET_M = np.array([12.0, 0.0, -2.0])
VELOCITY_M_PER_S = np.array([-2.0, 0.0, 4.0])
WAVELENGTH_M = ref.wavelength_m(sc.CARRIER_HZ)
WAVENUMBER = 2.0 * math.pi / WAVELENGTH_M
RECEIVERS = 8

#: float32 path lengths near 28 m carry ~2e-6 m of rounding.
LENGTH_TOL_M = 2e-5
#: Phase residual after removing one constant per class (the wall's
#: reflection phase): ~0.01 rad of float32 range rounding at 28 m, doubled.
PHASE_TOL_RAD = 0.03
#: Relative least-squares residual of a cell snapshot on its predicted
#: steering vectors. Range-Doppler leakage from the neighbouring cells, 2-3
#: range bins away under Hann windows, is the only thing outside the span.
SPAN_RESIDUAL = 0.05
#: The same residual on a wrong set of directions: an 8-element array cannot
#: fit a source 43 degrees away with any coefficient.
WRONG_RESIDUAL = 0.3


def _radar(waveform):
    from witwin.radar import Pattern

    rx = tuple((index * WAVELENGTH_M / 2.0, 0.0, 0.0) for index in range(RECEIVERS))
    # Local x is world +z here, so the transmitter sits 2 cm below the array
    # along the wall normal; this separates the two mixed classes' lengths.
    return sc.radar(waveform, tx=((-0.02, 0.0, 0.0),), rx=rx, pattern=Pattern.isotropic())


def _world():
    from witwin.core import PhysicalMaterial

    return sc.plane_scene(corners=sc.side_wall_corners(z_m=WALL_Z_M), material=PhysicalMaterial.perfect_conductor())


def _targets(device):
    from witwin.radar import PointTargets

    def trajectory(time_s):
        position = TARGET_M + VELOCITY_M_PER_S * float(time_s)
        return torch.tensor(position[None], dtype=torch.float32, device=device)

    return PointTargets(positions=trajectory(0.0), rcs=1.0, trajectory=trajectory)


def _predictions(radar, time_s: float):
    tx = radar.tx_pos[0].double().cpu().numpy()
    out = []
    for index in range(RECEIVERS):
        out.append(
            ref.image_round_trip(
                transmitter=tx,
                receiver=radar.rx_pos[index].double().cpu().numpy(),
                target=TARGET_M + VELOCITY_M_PER_S * time_s,
                target_velocity=VELOCITY_M_PER_S,
                plane_point=(0.0, 0.0, WALL_Z_M),
                normal=(0.0, 0.0, 1.0),
            )
        )
    return out


def _wrap(angle):
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def _classified_rows(rows, predictions):
    """``{class: complex128[RX]}`` and the worst length mismatch.

    A row is assigned the class whose image-method length it is nearest to;
    each receiver must then hold exactly one row of each class.
    """

    lengths = rows.total_delay_s.double().cpu().numpy() * ref.C0_M_PER_S
    transfer = rows.complex_transfer_ref.to(torch.complex128).cpu().numpy()
    receiver = rows.sensor_pair_index.cpu().numpy()  # one transmitter: pair == receiver
    table = {name: np.full(RECEIVERS, np.nan + 0j) for name in ref.ROUND_TRIP_CLASSES}
    worst = 0.0
    for length, value, rx in zip(lengths, transfer, receiver, strict=True):
        expected = {name: predictions[rx][name]["length_m"] for name in ref.ROUND_TRIP_CLASSES}
        name = min(expected, key=lambda key: abs(expected[key] - length))
        assert np.isnan(table[name][rx]), f"receiver {rx} holds two {name} rows"
        table[name][rx] = value
        worst = max(worst, abs(expected[name] - length))
    for name, values in table.items():
        assert not np.isnan(values).any(), f"{name} is missing at receivers {np.flatnonzero(np.isnan(values)).tolist()}"
    return table, worst


@pytest.fixture(scope="module")
def traced():
    from witwin.radar import Motion

    waveform = sc.beat_fmcw(samples=64, chirps=16, sample_rate=10e6)
    radar = _radar(waveform)
    paths = radar.trace(_world(), _targets(radar.device), times=(0.0,), los=True, reflections=1, motion=Motion.chirp())
    return radar, paths


def test_every_round_trip_class_has_its_image_method_length_at_every_receiver(traced):
    radar, paths = traced
    rows = paths.rows(0, 0)
    assert rows.path_count == 4 * RECEIVERS
    _, worst = _classified_rows(rows, _predictions(radar, float(paths.sample_times[0][0])))
    assert worst < LENGTH_TOL_M


@pytest.mark.parametrize("name", ref.ROUND_TRIP_CLASSES)
def test_angle_of_arrival_of_each_class_is_the_image_methods(traced, name):
    """The phase across the receive array follows the class's own geometry.

    ``arg C_i = -k L_i + const``: one constant per class (the wall's reflection
    phase and the scatterer's) and the per-receiver lengths from the image
    method, which encode the arrival direction - from the target for the
    classes whose receive leg is direct, from the mirrored target otherwise.
    The fitted direction cosine along the array is checked as well, so a wrong
    copy of the target (``0.55`` against ``-0.16`` here) cannot hide.
    """

    radar, paths = traced
    predictions = _predictions(radar, float(paths.sample_times[0][0]))
    table, _ = _classified_rows(paths.rows(0, 0), predictions)
    lengths = np.array([predictions[rx][name]["length_m"] for rx in range(RECEIVERS)])
    residual = _wrap(np.angle(table[name]) + WAVENUMBER * lengths)
    assert np.max(np.abs(_wrap(residual - residual[0]))) < PHASE_TOL_RAD

    # Direction cosine along world +z from the inter-element phase slope.
    spacing = WAVELENGTH_M / 2.0
    slope = np.angle(np.sum(table[name][1:] * np.conj(table[name][:-1])))
    measured = slope / (WAVENUMBER * spacing)
    expected = float(predictions[0][name]["arrival_unit"][2])
    # The arrival unit points from the receiver to the source; a source toward
    # +z shortens the path to the higher receivers, so the phase -k L rises.
    assert measured == pytest.approx(expected, abs=0.01), (name, measured, expected)


@pytest.mark.parametrize("name", ref.ROUND_TRIP_CLASSES)
def test_doppler_of_each_class_is_the_image_methods(traced, name):
    """Phase advance from the first to the last chirp, per class and receiver."""

    radar, paths = traced
    times = np.asarray(paths.sample_times[0], dtype=np.float64)
    first_predictions = _predictions(radar, float(times[0]))
    last_predictions = _predictions(radar, float(times[-1]))
    first, _ = _classified_rows(paths.rows(0, 0), first_predictions)
    last, _ = _classified_rows(paths.rows(0, len(times) - 1), last_predictions)

    advance = np.angle(last[name] * np.conj(first[name]))
    expected = np.array(
        [
            -WAVENUMBER * (last_predictions[rx][name]["length_m"] - first_predictions[rx][name]["length_m"])
            for rx in range(RECEIVERS)
        ]
    )
    assert np.max(np.abs(_wrap(advance - expected))) < PHASE_TOL_RAD

    # And as a Doppler frequency, f_D = -(dL/dt) / lambda.
    elapsed = float(times[-1] - times[0])
    rate = first_predictions[0][name]["length_rate_m_per_s"]
    unwrapped = expected[0] + _wrap(advance[0] - expected[0])
    assert unwrapped / (2.0 * math.pi * elapsed) == pytest.approx(-rate / WAVELENGTH_M, rel=2e-3)


def test_processed_cube_puts_each_ghost_at_its_mirrored_range_doppler_and_angle():
    """Range-Doppler peaks and beam-scan angles of the three resolvable cells.

    Ranges are half the round-trip length and velocities are closing-positive
    half the length rate, which is how the processing axes define them for
    any round trip. Angles are checked as a subspace statement: each cell's
    array snapshot is fitted by the steering vectors of its predicted arrival
    directions and not by a wrong set. The mixed cell holds two classes from
    two directions, the target's and the mirrored target's, and needs both; a
    beam-scan peak is biased there by the other source's sidelobe, so the peak
    is only checked in the single-source cells.
    """

    from witwin.radar import Fmcw, Motion
    from witwin.radar.processing import conventional_steering

    waveform = Fmcw(
        slope=30e12,
        sample_rate=10e6,
        samples_per_chirp=256,
        chirps_per_frame=32,
        adc_start=0.0,
        idle=5e-6,
        ramp_end=26.6e-6,
        output="spectrum",
    )
    radar = _radar(waveform)
    result = radar.simulate(
        _world(), _targets(radar.device), times=(0.0,), los=True, reflections=1, motion=Motion.chirp()
    )
    frame = result.frame(0)
    rd = frame.range_doppler(window="hann")
    data = rd.data.reshape(RECEIVERS, *rd.data.shape[-2:]).to(torch.complex128)
    power = data.abs().pow(2).sum(dim=0).cpu().numpy()
    ranges = rd.range_axis.cpu().numpy()
    velocities = rd.doppler_axis.cpu().numpy()
    range_bin = float(result.axes.range_bin_m)
    velocity_bin = float(result.axes.velocity_bin_mps)

    times = np.asarray(result.sample_times_s[0], dtype=np.float64)
    middle = _predictions(radar, 0.5 * float(times[0] + times[-1]))
    origin = radar.rx_pos.double().mean(dim=0).cpu().numpy()

    azimuth_deg = np.arange(-70.0, 70.0, 0.25)
    directions = torch.tensor(
        np.stack((np.sin(np.radians(azimuth_deg)), np.cos(np.radians(azimuth_deg)), np.zeros_like(azimuth_deg)), -1)
    )
    steering = conventional_steering(frame.array(), directions, dtype=torch.complex128).cpu()

    reference = middle[RECEIVERS // 2]
    target_azimuth = round(_azimuth_deg(reference["direct-direct"]["receive_copy"] - origin), 6)
    image_azimuth = round(_azimuth_deg(reference["reflected-reflected"]["receive_copy"] - origin), 6)

    cells = {
        "direct": ("direct-direct",),
        "mixed": ("direct-reflected", "reflected-direct"),
        "ghost": ("reflected-reflected",),
    }
    for cell, members in cells.items():
        length = np.mean([middle[RECEIVERS // 2][name]["length_m"] for name in members])
        rate = np.mean([middle[RECEIVERS // 2][name]["length_rate_m_per_s"] for name in members])
        expected_range, expected_velocity = 0.5 * length, -0.5 * rate
        r0 = int(np.argmin(np.abs(ranges - expected_range)))
        d0 = int(np.argmin(np.abs(velocities - expected_velocity)))
        window = power[max(d0 - 3, 0) : d0 + 4, max(r0 - 4, 0) : r0 + 5]
        dd, rr = np.unravel_index(int(np.argmax(window)), window.shape)
        d_peak, r_peak = max(d0 - 3, 0) + dd, max(r0 - 4, 0) + rr
        assert abs(ranges[r_peak] - expected_range) <= range_bin, (cell, ranges[r_peak], expected_range)
        assert abs(velocities[d_peak] - expected_velocity) <= velocity_bin, (
            cell,
            velocities[d_peak],
            expected_velocity,
        )

        snapshot = data[:, d_peak, r_peak].cpu()
        expected = sorted(
            {round(_azimuth_deg(middle[RECEIVERS // 2][name]["receive_copy"] - origin), 6) for name in members}
        )
        # The cell's snapshot lies in the span of the predicted arrival
        # directions' steering vectors, and in no smaller set of them.
        assert _fit_residual(frame, snapshot, expected) < SPAN_RESIDUAL, (cell, expected)
        for wrong in _wrong_hypotheses(expected, (target_azimuth, image_azimuth)):
            assert _fit_residual(frame, snapshot, wrong) > WRONG_RESIDUAL, (cell, wrong)

        if len(expected) == 1:
            beam = (steering.conj() * snapshot[:, None]).sum(dim=0).abs().pow(2).numpy()
            assert float(azimuth_deg[int(np.argmax(beam))]) == pytest.approx(expected[0], abs=1.0), (cell, expected)


def _fit_residual(frame, snapshot, azimuths_deg) -> float:
    from witwin.radar.processing import conventional_steering

    radians = np.radians(np.asarray(azimuths_deg, dtype=np.float64))
    directions = torch.tensor(np.stack((np.sin(radians), np.cos(radians), np.zeros_like(radians)), -1))
    basis = conventional_steering(frame.array(), directions, dtype=torch.complex128).cpu()
    coefficients = torch.linalg.lstsq(basis, snapshot[:, None]).solution
    return float((basis @ coefficients - snapshot[:, None]).norm() / snapshot.norm())


def _wrong_hypotheses(expected, both):
    """Direction sets a confused join would produce: the other copy, or one copy short."""

    if len(expected) == 2:
        return [[expected[0]], [expected[1]]]
    other = [value for value in both if abs(value - expected[0]) > 1e-6]
    return [other]


def _source(name: str) -> str:
    return name.split("-")[1]


def _azimuth_deg(vector) -> float:
    """Azimuth toward local right (world +z) from boresight (world +x)."""

    return math.degrees(math.atan2(float(vector[2]), float(vector[0])))
