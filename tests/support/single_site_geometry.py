"""The single-site fixture geometry and waveform: one TX, one site, one RX.

Every analytic expectation in the single-site tests is computed in float64
from these constants, so they are stated once here and never duplicated in a
test file. The closed forms are ``multi_endpoint_geometry``'s, specialised to
this endpoint triple; only the endpoints, the wall extent and the front end
differ from the multi-endpoint fixture.

The wall at ``x = 4`` was authored for a spike that requested only
line-of-sight legs, on the bet that turning reflection on later should not
require a new fixture world. The SAME wall, at
``components={los, reflection}, max_depth=1``, yields exactly two rows per leg
and four combined round trips, all of which have closed forms below.

One trap is encoded here rather than left to be rediscovered. The two MIXED
combined paths - inbound reflection with outbound line of sight, and its
mirror - differ by only 6.04 mm of two-way path length, because the TX/RX
baseline is 0.15 m. That is 20.15 ps, about 1e4 float32 ULPs at 2.66e-8 s, so
comparing ``total_delay_s`` directly is safe; but the two are NOT separable in
an FMCW range bin, so a test that compares only the SORTED set of combined
delays cannot tell them apart. Identify a combined row by its
``inbound_row``/``outbound_row``, never by its position in a sorted delay list.
Widening ``RX_POSITION_M`` to make them separable would move every pinned
number and is not an option.
"""

from __future__ import annotations

from .multi_endpoint_geometry import (
    C0_M_PER_S,
    LOS_COMPONENT_ID,
    POLARIZATION,
    REFERENCE_FREQUENCY_HZ,
    REFLECTION_COMPONENT_ID,
    WALL_EPS_R,
    WALL_FACES,
    WALL_PLANE_X_M,
    WALL_SIGMA_E,
    Point,
    distance_m,
    image_position_m,
    leg_delay_rate_s_per_s,
    reflection_length_m,
    specular_point_m,
)

# World coordinates in metres. The site sits off the TX/RX baseline so that the
# in-plane gradient components are all nonzero and the out-of-plane component
# is exactly zero, which is a checkable structure rather than an accident.
TX_POSITION_M: Point = (0.0, 0.0, 0.0)
RX_POSITION_M: Point = (0.15, 0.0, 0.0)
SITE_POSITION_M: Point = (2.0, 0.6, 0.0)

TX_POWER_W = 1.0

# Stable world IDs. They are arbitrary but must be distinct and must stay
# stable across a frozen sequence, because the two-way join is by identity.
TX_STABLE_ID = 10
SITE_STABLE_ID = 20
RX_STABLE_ID = 30

# The wall is wide enough that every specular point in this fixture lands on
# it; the multi-endpoint fixture's narrower facet is what makes rows go
# missing there, and this fixture must not inherit that.
WALL_VERTICES_M = (
    (WALL_PLANE_X_M, -3.0, -3.0),
    (WALL_PLANE_X_M, 3.0, -3.0),
    (WALL_PLANE_X_M, 3.0, 3.0),
    (WALL_PLANE_X_M, -3.0, 3.0),
)

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
    "power": 12,
    "tx_loc": [[0, 0, 0]],
    "rx_loc": [[0, 0, 0]],
}

# The site velocity the Doppler fixtures use. Purely lateral, so no combined
# path is stationary and no two of the four share a Doppler shift.
SITE_VELOCITY_M_PER_S: Point = (0.0, 12.0, 0.0)


def leg_distances_m() -> tuple[float, float]:
    """Closed-form inbound and outbound line-of-sight leg lengths."""

    return (distance_m(SITE_POSITION_M, TX_POSITION_M), distance_m(RX_POSITION_M, SITE_POSITION_M))


def round_trip_delay_s() -> float:
    """Closed-form two-way delay. This is NOT ``2 * d / c0``."""

    d_in, d_out = leg_distances_m()
    return (d_in + d_out) / C0_M_PER_S


# --------------------------------------------------------------------------
# Multipath: the image-source closed forms for the single wall
# --------------------------------------------------------------------------

#: The fixed endpoint of each leg, keyed by ``(leg, component)``.
_LEG_ENDPOINTS: dict[tuple[str, str], Point] = {
    ("inbound", "los"): TX_POSITION_M,
    ("inbound", "reflection"): TX_POSITION_M,
    ("outbound", "los"): RX_POSITION_M,
    ("outbound", "reflection"): RX_POSITION_M,
}


def leg_lengths_m() -> dict[tuple[str, str], float]:
    """Every leg length in the multipath fixture, keyed by (leg, component)."""

    return {
        (leg, component): (
            distance_m(endpoint, SITE_POSITION_M)
            if component == "los"
            else reflection_length_m(endpoint, SITE_POSITION_M)
        )
        for (leg, component), endpoint in _LEG_ENDPOINTS.items()
    }


def leg_delays_s() -> dict[tuple[str, str], float]:
    return {key: value / C0_M_PER_S for key, value in leg_lengths_m().items()}


def leg_delay_rates_s_per_s(velocity: Point = SITE_VELOCITY_M_PER_S) -> dict[tuple[str, str], float]:
    """``d(delay)/dt`` per leg for a moving SITE, both endpoints static."""

    return {
        (leg, component): leg_delay_rate_s_per_s(SITE_POSITION_M, endpoint, component, velocity)
        for (leg, component), endpoint in _LEG_ENDPOINTS.items()
    }


def combined_delays_s() -> dict[tuple[str, str], float]:
    """Two-way delay of each of the four combined paths.

    Keyed by ``(inbound component, outbound component)``. The two mixed keys
    are 20.15 ps apart; see the module docstring.
    """

    legs = leg_delays_s()
    return {
        (inbound, outbound): legs[("inbound", inbound)] + legs[("outbound", outbound)]
        for inbound in ("los", "reflection")
        for outbound in ("los", "reflection")
    }


def combined_doppler_hz(velocity: Point = SITE_VELOCITY_M_PER_S) -> dict[tuple[str, str], float]:
    """Doppler shift of each combined path at the reference frequency.

    ``f_D = -f_c * d(tau_rt)/dt``: the sign follows Channel's
    ``exp(-j k d)`` phasor, so a receding site gives a negative shift.
    """

    rates = leg_delay_rates_s_per_s(velocity)
    return {
        (inbound, outbound): -REFERENCE_FREQUENCY_HZ * (rates[("inbound", inbound)] + rates[("outbound", outbound)])
        for inbound in ("los", "reflection")
        for outbound in ("los", "reflection")
    }


__all__ = [
    "C0_M_PER_S",
    "FIXTURE_RADAR_CONFIG",
    "LOS_COMPONENT_ID",
    "POLARIZATION",
    "REFERENCE_FREQUENCY_HZ",
    "REFLECTION_COMPONENT_ID",
    "RX_POSITION_M",
    "RX_STABLE_ID",
    "SITE_POSITION_M",
    "SITE_STABLE_ID",
    "SITE_VELOCITY_M_PER_S",
    "TX_POSITION_M",
    "TX_POWER_W",
    "TX_STABLE_ID",
    "WALL_EPS_R",
    "WALL_FACES",
    "WALL_PLANE_X_M",
    "WALL_SIGMA_E",
    "WALL_VERTICES_M",
    "combined_delays_s",
    "combined_doppler_hz",
    "image_position_m",
    "leg_delay_rates_s_per_s",
    "leg_delays_s",
    "leg_distances_m",
    "leg_lengths_m",
    "reflection_length_m",
    "round_trip_delay_s",
    "specular_point_m",
]
