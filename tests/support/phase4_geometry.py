"""The frozen Phase-4 fixture geometry and waveform.

Every analytic expectation in the spike tests is computed in float64 from these
constants, so they are stated once here and never duplicated in a test file.

The wall at ``x = 4`` is part of the fixture scene but produces no rows: the
spike requests only line-of-sight legs. It is present because a compiled scene
with real geometry is the realistic case, and because a later phase that turns
reflection on should not have to change the fixture world.
"""

from __future__ import annotations

C0_M_PER_S = 299792458.0

# World coordinates in metres. The site sits off the TX/RX baseline so that the
# in-plane gradient components are all nonzero and the out-of-plane component
# is exactly zero, which is a checkable structure rather than an accident.
TX_POSITION_M = (0.0, 0.0, 0.0)
RX_POSITION_M = (0.15, 0.0, 0.0)
SITE_POSITION_M = (2.0, 0.6, 0.0)

# Endpoint polarization: z, i.e. transverse to every leg in the z = 0 plane.
POLARIZATION = (0.0, 0.0, 1.0)

TX_POWER_W = 1.0

# Stable world IDs. They are arbitrary but must be distinct and must stay
# stable across a frozen sequence, because the two-way join is by identity.
TX_STABLE_ID = 10
SITE_STABLE_ID = 20
RX_STABLE_ID = 30

REFERENCE_FREQUENCY_HZ = 77.0e9

WALL_PLANE_X_M = 4.0
WALL_VERTICES_M = (
    (WALL_PLANE_X_M, -3.0, -3.0),
    (WALL_PLANE_X_M, 3.0, -3.0),
    (WALL_PLANE_X_M, 3.0, 3.0),
    (WALL_PLANE_X_M, -3.0, 3.0),
)
WALL_FACES = ((0, 1, 2), (0, 2, 3))
WALL_EPS_R = 5.24
WALL_SIGMA_E = 0.0462

# A single-TX single-RX FMCW front end. The waveform numbers are the ones the
# rest of the radar suite already uses; only the antenna count is reduced.
FIXTURE_RADAR_CONFIG = {
    "num_tx": 1,
    "num_rx": 1,
    "fc": REFERENCE_FREQUENCY_HZ,
    "slope": 60.012,
    "adc_samples": 256,
    "adc_start_time": 6,
    "sample_rate": 4400,
    "idle_time": 7,
    "ramp_end_time": 58,
    "chirp_per_frame": 8,
    "frame_per_second": 10,
    "num_doppler_bins": 8,
    "num_range_bins": 256,
    "num_angle_bins": 64,
    "power": 12,
    "tx_loc": [[0, 0, 0]],
    "rx_loc": [[0, 0, 0]],
}


def leg_distances_m() -> tuple[float, float]:
    """Closed-form inbound and outbound leg lengths for the fixture."""

    def distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
        return sum((x - y) ** 2 for x, y in zip(a, b, strict=True)) ** 0.5

    return (
        distance(SITE_POSITION_M, TX_POSITION_M),
        distance(RX_POSITION_M, SITE_POSITION_M),
    )


def round_trip_delay_s() -> float:
    """Closed-form two-way delay. This is NOT ``2 * d / c0``."""

    d_in, d_out = leg_distances_m()
    return (d_in + d_out) / C0_M_PER_S


__all__ = [
    "C0_M_PER_S",
    "FIXTURE_RADAR_CONFIG",
    "POLARIZATION",
    "REFERENCE_FREQUENCY_HZ",
    "RX_POSITION_M",
    "RX_STABLE_ID",
    "SITE_POSITION_M",
    "SITE_STABLE_ID",
    "TX_POSITION_M",
    "TX_POWER_W",
    "TX_STABLE_ID",
    "WALL_EPS_R",
    "WALL_FACES",
    "WALL_PLANE_X_M",
    "WALL_SIGMA_E",
    "WALL_VERTICES_M",
    "leg_distances_m",
    "round_trip_delay_s",
]
