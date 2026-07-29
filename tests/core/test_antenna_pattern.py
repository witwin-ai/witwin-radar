"""Antenna-pattern interpolation, measured through the production route.

This file used to drive the pattern through ``solvers.common`` and compare it
with ``tests/reference/path_math.py``. Both belong to the legacy Dirichlet
route that Phase 11 deletes, so the same four questions are now asked of the
route that survives: :class:`witwin.radar.sensors.RoundTripPatternStage`, which
applies the transmit and receive pattern gain to a composed round-trip batch
through the native ``sensor_weight`` family.

The measured quantity is unchanged and so are the expected numbers. With one
transmitter and one receiver co-located at the radar origin, the transmit and
receive directions to a target are the same vector, so the stage's amplitude
factor ``sqrt(G_t * G_r)`` is exactly ``G``, and the ratio of an off-axis row to
a boresight row is the POWER gain the tables tabulate. That is why the dipole
and bilinear expectations below are the same as before the migration.

These are now GPU tests. The interpolation they exercise lives in a CUDA kernel;
its Torch oracle is pinned separately, over random directions, by
``tests/test_phase6_sensor_weight.py``.
"""

from __future__ import annotations

import math
import types

import pytest
import torch

from witwin.radar import Radar, RadarConfig
from witwin.radar.paths import RadarPathBatch, RadarPathTopology
from witwin.radar.sensors import AntennaPatternSpec
from witwin.radar.sensors import RoundTripPatternStage


pytestmark = pytest.mark.gpu

#: One site, one transmitter, one receiver: one composed row.
_SITE_STABLE_ID = 3_000_000


def _base_config() -> dict:
    return {
        "num_tx": 1,
        "num_rx": 1,
        "fc": 77e9,
        "slope": 60.012,
        "adc_samples": 128,
        "adc_start_time": 0,
        "sample_rate": 4400,
        "idle_time": 7,
        "ramp_end_time": 58,
        "chirp_per_frame": 1,
        "frame_per_second": 10,
        "num_doppler_bins": 1,
        "num_range_bins": 128,
        "num_angle_bins": 16,
        "power": 12,
        "tx_loc": [[0, 0, 0]],
        "rx_loc": [[0, 0, 0]],
    }


def _make_radar(*, antenna_pattern=None) -> Radar:
    config = _base_config()
    if antenna_pattern is not None:
        config["antenna_pattern"] = antenna_pattern
    return Radar(RadarConfig.from_dict(config))


def _target_position(x_deg: float, y_deg: float, radius: float = 2.0) -> torch.Tensor:
    direction = torch.tensor(
        [
            math.tan(math.radians(x_deg)),
            math.tan(math.radians(y_deg)),
            -1.0,
        ],
        dtype=torch.float32,
    )
    direction = direction / torch.linalg.norm(direction)
    return direction * radius


def _one_row_stage(radar: Radar) -> tuple[RoundTripPatternStage, RadarPathBatch]:
    """The stage and a unit-weight batch for a single-element single-site array.

    The join is duck-typed: the stage reads a pair index, a pair count, a site
    count and a response slot off it, and a real Channel round trip would only
    supply four one-element tensors at the cost of a compiled scene.
    """

    device = radar.device
    zeros = torch.zeros(1, dtype=torch.int64, device=device)
    join = types.SimpleNamespace(
        sensor_pair_index=zeros,
        sensor_pair_count=1,
        site_count=1,
        path_count=1,
        response_slot=zeros,
    )
    stage = RoundTripPatternStage.freeze(
        radar,
        join,
        site_ids=(_SITE_STABLE_ID,),
        pattern=AntennaPatternSpec.from_config(radar.config.antenna_pattern),
    )
    batch = RadarPathBatch(
        sensor_pair_count=1,
        path_count=1,
        sensor_pair_index=zeros,
        pair_offsets=torch.tensor([0, 1], dtype=torch.int64, device=device),
        total_delay_s=torch.zeros(1, dtype=torch.float32, device=device),
        delay_rate=None,
        complex_transfer_ref=torch.ones(1, dtype=torch.complex64, device=device),
        reference_frequency_hz=float(radar.config.fc),
        row_valid=None,
        topology=RadarPathTopology(
            radar_source_id=zeros,
            site_id=zeros + _SITE_STABLE_ID,
            radar_sink_id=zeros,
            inbound_row=zeros,
            outbound_row=zeros,
        ),
        join_mode="multipath",
    )
    return stage, batch


def _signal_peak(
    radar: Radar, *, x_deg: float, y_deg: float, radius: float = 2.0
) -> torch.Tensor:
    """The composed weight's magnitude for a target at that off-boresight angle."""

    stage, batch = _one_row_stage(radar)
    site = _target_position(x_deg, y_deg, radius).to(device=radar.device).unsqueeze(0)
    published = stage.apply(
        batch, tx_pos=radar.tx_pos, rx_pos=radar.rx_pos, site_positions_m=site
    )
    return published.complex_transfer_ref.abs().max()


def _half_wave_dipole_power(angle_deg: float) -> float:
    angle_rad = math.radians(angle_deg)
    cos_angle = math.cos(angle_rad)
    if abs(cos_angle) < 1e-8:
        return 0.0
    field = math.cos(0.5 * math.pi * math.sin(angle_rad)) / cos_angle
    return field * field


def _bilinear_value(
    *,
    x_deg: float,
    y_deg: float,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    v00: float,
    v10: float,
    v01: float,
    v11: float,
) -> float:
    tx = (x_deg - x0) / (x1 - x0)
    ty = (y_deg - y0) / (y1 - y0)
    return (
        (1.0 - tx) * (1.0 - ty) * v00
        + tx * (1.0 - ty) * v10
        + (1.0 - tx) * ty * v01
        + tx * ty * v11
    )


def test_missing_antenna_pattern_uses_default_dipole_runtime():
    radar = _make_radar()

    assert radar.config.antenna_pattern is None
    assert radar.antenna_pattern_config["kind"] == "separable"

    center_gain = radar._evaluate_antenna_pattern_xy(
        torch.tensor([0.0], dtype=torch.float32, device=radar.device),
        torch.tensor([0.0], dtype=torch.float32, device=radar.device),
    )
    edge_gain = radar._evaluate_antenna_pattern_xy(
        torch.tensor([85.0], dtype=torch.float32, device=radar.device),
        torch.tensor([0.0], dtype=torch.float32, device=radar.device),
    )

    assert torch.allclose(
        center_gain,
        torch.tensor([1.0], dtype=torch.float32, device=radar.device),
        atol=1e-6,
        rtol=1e-6,
    )
    assert edge_gain.item() < 0.05


@pytest.mark.parametrize("angle_deg", [0.0, 30.0, 60.0])
def test_default_dipole_signal_matches_expected_gain(angle_deg: float):
    """The configured default is a dipole, and the stage applies exactly it.

    ``AntennaPatternSpec.from_config(None)`` is the half-wave dipole, which is
    also what ``radar.antenna_pattern_config`` falls back to, so passing the
    spec through the production stage measures the same cut the runtime
    declares.
    """

    radar = _make_radar()
    center_peak = _signal_peak(radar, x_deg=0.0, y_deg=0.0)
    off_axis_peak = _signal_peak(radar, x_deg=angle_deg, y_deg=0.0)
    measured_ratio = (off_axis_peak / center_peak).item()

    assert measured_ratio == pytest.approx(
        _half_wave_dipole_power(angle_deg), rel=5e-3, abs=5e-3
    )


def test_flat_custom_pattern_keeps_signal_constant():
    radar = _make_radar(
        antenna_pattern={
            "kind": "separable",
            "x_angles_deg": [-90, 0, 90],
            "x_values": [1.0, 1.0, 1.0],
            "y_angles_deg": [-90, 0, 90],
            "y_values": [1.0, 1.0, 1.0],
        }
    )

    center_peak = _signal_peak(radar, x_deg=0.0, y_deg=0.0)
    for angle_deg in (15.0, 45.0, 70.0):
        off_axis_peak = _signal_peak(radar, x_deg=angle_deg, y_deg=0.0)
        assert (off_axis_peak / center_peak).item() == pytest.approx(
            1.0, rel=5e-3, abs=5e-3
        )


def test_2d_map_signal_matches_bilinear_gain():
    radar = _make_radar(
        antenna_pattern={
            "kind": "map",
            "x_angles_deg": [0, 40],
            "y_angles_deg": [0, 20],
            "values": [
                [1.0, 0.8],
                [0.6, 0.2],
            ],
        }
    )

    x_deg = 20.0
    y_deg = 10.0
    center_peak = _signal_peak(radar, x_deg=0.0, y_deg=0.0)
    query_peak = _signal_peak(radar, x_deg=x_deg, y_deg=y_deg)
    measured_ratio = (query_peak / center_peak).item()

    expected = _bilinear_value(
        x_deg=x_deg,
        y_deg=y_deg,
        x0=0.0,
        x1=40.0,
        y0=0.0,
        y1=20.0,
        v00=1.0,
        v10=0.8,
        v01=0.6,
        v11=0.2,
    )
    assert measured_ratio == pytest.approx(expected, rel=5e-3, abs=5e-3)
