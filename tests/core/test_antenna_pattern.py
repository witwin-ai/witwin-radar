"""Antenna-pattern interpolation, measured through the production route.

This file used to drive the pattern through ``solvers.common`` and compare it
with ``tests/reference/path_math.py``. ``solvers.common`` belonged to the legacy
Dirichlet route and went with it when Phase 11 deleted that route; ``path_math``
stayed, as the independent oracle for the live ``sensor_weight`` family. So the
same four questions are now asked of the route that survives:
:class:`witwin.radar.sensors.RoundTripPatternStage`, which applies the transmit
and receive pattern gain to a composed round-trip batch through the native
``sensor_weight`` family.

The measured quantity is unchanged and so are the expected numbers. With one
transmitter and one receiver co-located at the radar origin, the transmit and
receive directions to a target are the same vector, so the stage's amplitude
factor ``sqrt(G_t * G_r)`` is exactly ``G``, and the ratio of an off-axis row to
a boresight row is the POWER gain the tables tabulate. That is why the dipole
and bilinear expectations below are the same as before the migration.

What did change is the DEFAULT. ``Radar.pattern`` is now
``Pattern.isotropic()``, so the two dipole cases name ``Pattern.dipole()``
themselves and the default has a case of its own asserting unit gain.

These are now GPU tests. The interpolation they exercise lives in a CUDA kernel;
its Torch oracle is pinned separately, over random directions, by
``tests/test_phase6_sensor_weight.py``.
"""

from __future__ import annotations

import types

import pytest
import torch

from core import half_wave_dipole_power, local_target_position
from witwin.radar import Radar
from witwin.radar.paths import RadarPathBatch, RadarPathTopology
from witwin.radar.sensors import Pattern, RoundTripPatternStage

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
        "power": 12,
        "tx_loc": [[0, 0, 0]],
        "rx_loc": [[0, 0, 0]],
    }


def _make_radar(*, antenna_pattern=None, **overrides) -> Radar:
    """A radar whose pattern comes from the file format or from a keyword.

    Both routes exist in production: ``antenna_pattern`` is the flat mapping's
    optional block, and ``pattern=`` is the SI keyword override. Keeping both
    here is what lets the map cases below stay authored as file content while
    the dipole cases name the record.
    """

    config = _base_config()
    if antenna_pattern is not None:
        config["antenna_pattern"] = antenna_pattern
    return Radar.from_dict(config, **overrides)


def _one_row_stage(radar: Radar) -> tuple[RoundTripPatternStage, RadarPathBatch]:
    """The stage and a unit-weight batch for a single-element single-site array.

    The join is duck-typed: the stage reads a pair index, a pair count, a site
    count and a response slot off it, and a real Channel round trip would only
    supply four one-element tensors at the cost of a compiled scene.
    """

    device = radar.device
    zeros = torch.zeros(1, dtype=torch.int64, device=device)
    join = types.SimpleNamespace(
        sensor_pair_index=zeros, sensor_pair_count=1, site_count=1, path_count=1, response_slot=zeros
    )
    stage = RoundTripPatternStage.freeze(radar, join, site_ids=(_SITE_STABLE_ID,), pattern=radar.pattern)
    batch = RadarPathBatch(
        sensor_pair_count=1,
        path_count=1,
        sensor_pair_index=zeros,
        pair_offsets=torch.tensor([0, 1], dtype=torch.int64, device=device),
        total_delay_s=torch.zeros(1, dtype=torch.float32, device=device),
        delay_rate=None,
        complex_transfer_ref=torch.ones(1, dtype=torch.complex64, device=device),
        reference_frequency_hz=float(radar.carrier),
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


def _signal_peak(radar: Radar, *, x_deg: float, y_deg: float, radius: float = 2.0) -> torch.Tensor:
    """The composed weight's magnitude for a target at that off-boresight angle."""

    stage, batch = _one_row_stage(radar)
    site = local_target_position(x_deg, y_deg, radius).to(device=radar.device).unsqueeze(0)
    published = stage.apply(
        batch,
        tx_pos=radar.tx_pos,
        rx_pos=radar.rx_pos,
        tx_targets_m=(site).index_select(0, stage.site_slot),
        rx_targets_m=(site).index_select(0, stage.site_slot),
    )
    return published.complex_transfer_ref.abs().max()


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
    return (1.0 - tx) * (1.0 - ty) * v00 + tx * (1.0 - ty) * v10 + (1.0 - tx) * ty * v01 + tx * ty * v11


def test_missing_antenna_pattern_is_isotropic_at_runtime():
    """The default is unit gain everywhere, and 85 degrees is the proof.

    This case used to pin the opposite: a radar that declared no pattern got a
    half-wave dipole, and the edge gain was asserted to be BELOW 0.05. The
    default is now ``Pattern.isotropic()`` - an unchosen dipole attenuates
    every off-boresight return by a number nobody asked for - so the same
    edge angle is asserted to be exactly unity instead.
    """

    radar = _make_radar()

    assert radar.pattern == Pattern.isotropic()

    center_gain = radar._evaluate_antenna_pattern_xy(
        torch.tensor([0.0], dtype=torch.float32, device=radar.device),
        torch.tensor([0.0], dtype=torch.float32, device=radar.device),
    )
    edge_gain = radar._evaluate_antenna_pattern_xy(
        torch.tensor([85.0], dtype=torch.float32, device=radar.device),
        torch.tensor([0.0], dtype=torch.float32, device=radar.device),
    )

    unit = torch.tensor([1.0], dtype=torch.float32, device=radar.device)
    assert torch.allclose(center_gain, unit, atol=1e-6, rtol=1e-6)
    assert torch.allclose(edge_gain, unit, atol=1e-6, rtol=1e-6)


def test_a_declared_dipole_rolls_off_at_the_edge():
    """The roll-off the default used to supply, now named by the caller."""

    radar = _make_radar(pattern=Pattern.dipole())

    assert radar.pattern.kind == "separable"

    center_gain = radar._evaluate_antenna_pattern_xy(
        torch.tensor([0.0], dtype=torch.float32, device=radar.device),
        torch.tensor([0.0], dtype=torch.float32, device=radar.device),
    )
    edge_gain = radar._evaluate_antenna_pattern_xy(
        torch.tensor([85.0], dtype=torch.float32, device=radar.device),
        torch.tensor([0.0], dtype=torch.float32, device=radar.device),
    )

    assert torch.allclose(
        center_gain, torch.tensor([1.0], dtype=torch.float32, device=radar.device), atol=1e-6, rtol=1e-6
    )
    assert edge_gain.item() < 0.05


@pytest.mark.parametrize("angle_deg", [0.0, 30.0, 60.0])
def test_dipole_signal_matches_expected_gain(angle_deg: float):
    """``Pattern.dipole()`` reaches the stage as exactly the cut it tabulates.

    The radar names the dipole, the stage is frozen against ``radar.pattern``,
    and the measured ratio is the closed-form power gain. Against the isotropic
    default the stage is a proven no-op and every ratio would be one, which is
    why this case names the pattern rather than relying on a default.
    """

    radar = _make_radar(pattern=Pattern.dipole())
    center_peak = _signal_peak(radar, x_deg=0.0, y_deg=0.0)
    off_axis_peak = _signal_peak(radar, x_deg=angle_deg, y_deg=0.0)
    measured_ratio = (off_axis_peak / center_peak).item()

    assert measured_ratio == pytest.approx(half_wave_dipole_power(angle_deg), rel=5e-3, abs=5e-3)


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
        assert (off_axis_peak / center_peak).item() == pytest.approx(1.0, rel=5e-3, abs=5e-3)


def test_2d_map_signal_matches_bilinear_gain():
    radar = _make_radar(
        antenna_pattern={
            "kind": "map",
            "x_angles_deg": [0, 40],
            "y_angles_deg": [0, 20],
            "values": [[1.0, 0.8], [0.6, 0.2]],
        }
    )

    x_deg = 20.0
    y_deg = 10.0
    center_peak = _signal_peak(radar, x_deg=0.0, y_deg=0.0)
    query_peak = _signal_peak(radar, x_deg=x_deg, y_deg=y_deg)
    measured_ratio = (query_peak / center_peak).item()

    expected = _bilinear_value(
        x_deg=x_deg, y_deg=y_deg, x0=0.0, x1=40.0, y0=0.0, y1=20.0, v00=1.0, v10=0.8, v01=0.6, v11=0.2
    )
    assert measured_ratio == pytest.approx(expected, rel=5e-3, abs=5e-3)
