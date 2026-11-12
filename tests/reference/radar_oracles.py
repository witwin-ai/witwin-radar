"""Closed-form float64 references for the analytic oracle suite.

TEST-ONLY, and deliberately free of any ``witwin`` import: every expression
here is written from the textbook statement, in Python floats or NumPy
float64, so that an oracle built on it cannot agree with production by
construction. ``tests/test_oracle_independence.py`` scans this package, and
``ci/check_test_oracle_isolation.py`` keeps production from importing it.

Conventions, stated once:

* SI units throughout: metres, seconds, hertz, watts, kelvin, square metres.
* Phasors use ``exp(+j 2 pi f t)`` time dependence, so a propagation delay
  ``tau`` contributes ``exp(-j 2 pi f tau)`` and a lossy permittivity is
  ``eps_r - j sigma / (omega eps_0)``.
* The FMCW beat phase is the transmitted minus the received chirp phase,
  ``2 pi (f_c tau + K tau (t_local - tau / 2))``, with ``t_local`` the time
  since the ramp start and ``tau`` the round-trip delay evaluated at the sample
  instant (the quasi-static observation model; light-time retardation is a
  ``v / c`` correction this suite does not claim).
"""

from __future__ import annotations

import cmath
import math
from collections.abc import Sequence

import numpy as np

#: Speed of light in vacuum, m/s (exact by SI definition).
C0_M_PER_S = 299_792_458.0
#: Vacuum permittivity, F/m (CODATA 2018).
EPS0_F_PER_M = 8.8541878128e-12
#: Boltzmann constant, J/K (exact by SI definition).
BOLTZMANN_J_PER_K = 1.380649e-23
#: The IEEE reference temperature a noise figure is quoted against, K.
REFERENCE_TEMPERATURE_K = 290.0


def wavelength_m(frequency_hz: float) -> float:
    return C0_M_PER_S / float(frequency_hz)


def watts_from_dbm(power_dbm: float) -> float:
    return 1e-3 * 10.0 ** (float(power_dbm) / 10.0)


def db(ratio: float) -> float:
    """``10 log10`` of a power ratio."""

    return 10.0 * math.log10(ratio)


# ---------------------------------------------------------------------------
# Radar equation and analytic cross sections
# ---------------------------------------------------------------------------


def bistatic_received_power_w(
    *,
    transmit_power_w: float,
    gain_tx: float,
    gain_rx: float,
    wavelength: float,
    rcs_m2: float,
    r_in: float,
    r_out: float,
) -> float:
    """``P_r = P_t G_t G_r lambda^2 sigma / ((4 pi)^3 R_in^2 R_out^2)``.

    Gains are linear power gains of the two elements toward the target, with
    co-polarized elements and a polarization-preserving (isotropic) scatterer.
    """

    return transmit_power_w * gain_tx * gain_rx * wavelength**2 * rcs_m2 / ((4.0 * math.pi) ** 3 * r_in**2 * r_out**2)


def rcs_from_monostatic_power(*, received_power_w: float, transmit_power_w: float, wavelength: float, range_m: float):
    """The radar equation solved for ``sigma`` with unit gains."""

    return received_power_w * (4.0 * math.pi) ** 3 * range_m**4 / (transmit_power_w * wavelength**2)


def flat_plate_broadside_rcs_m2(area_m2: float, wavelength: float) -> float:
    """Physical-optics broadside RCS of a flat plate, ``4 pi A^2 / lambda^2``."""

    return 4.0 * math.pi * area_m2**2 / wavelength**2


def sphere_optical_rcs_m2(radius_m: float) -> float:
    """Optical-region sphere RCS, ``pi a^2``, valid for ``k a >> 1``."""

    return math.pi * radius_m**2


def triangular_trihedral_rcs_m2(edge_m: float, wavelength: float) -> float:
    """Peak (symmetry-axis) RCS of a triangular trihedral, ``4 pi L^4 / (3 lambda^2)``.

    ``L`` is the length of each of the three mutually orthogonal edges.
    """

    return 4.0 * math.pi * edge_m**4 / (3.0 * wavelength**2)


# ---------------------------------------------------------------------------
# Thermal noise and the post-integration SNR
# ---------------------------------------------------------------------------


def system_noise_temperature_k(*, antenna_temperature_k: float, noise_figure_db: float) -> float:
    """``T_sys = T_ant + T_0 (F - 1)``."""

    return antenna_temperature_k + REFERENCE_TEMPERATURE_K * (10.0 ** (noise_figure_db / 10.0) - 1.0)


def coherent_integration_snr(*, received_power_w: float, observed_time_s: float, system_temperature_k: float) -> float:
    """Matched-filter SNR ``E / N_0 = P_r T_obs / (k T_sys)``.

    ``T_obs`` is the total time the echo is actually observed: samples times
    the sample period for FMCW, pulse width times pulse count for a pulse
    train, useful symbol time (cyclic prefix excluded) times symbol count for
    OFDM. The statement is independent of how a cube is normalised, which is
    what makes it an oracle for the normalisation.
    """

    return received_power_w * observed_time_s / (BOLTZMANN_J_PER_K * system_temperature_k)


# ---------------------------------------------------------------------------
# Image method
# ---------------------------------------------------------------------------


def mirror_point(point: Sequence[float], *, plane_point: Sequence[float], normal: Sequence[float]) -> np.ndarray:
    """Reflection of ``point`` through the plane, float64."""

    p = np.asarray(point, dtype=np.float64)
    o = np.asarray(plane_point, dtype=np.float64)
    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    return p - 2.0 * np.dot(p - o, n) * n


def mirror_vector(vector: Sequence[float], *, normal: Sequence[float]) -> np.ndarray:
    v = np.asarray(vector, dtype=np.float64)
    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    return v - 2.0 * np.dot(v, n) * n


#: The four round-trip classes of a one-wall, one-bounce world. The first word
#: is the transmit leg (transmitter to target), the second the receive leg.
ROUND_TRIP_CLASSES = ("direct-direct", "direct-reflected", "reflected-direct", "reflected-reflected")


def image_round_trip(
    *,
    transmitter: Sequence[float],
    receiver: Sequence[float],
    target: Sequence[float],
    target_velocity: Sequence[float],
    plane_point: Sequence[float],
    normal: Sequence[float],
) -> dict[str, dict[str, object]]:
    """Path length, its rate and the arrival direction of each round-trip class.

    By the image method a leg that bounces off the plane has the length of the
    straight leg to the mirrored target, and arrives from the mirrored target.
    So each class is described by which copy of the target the transmit leg
    sees and which copy the receive leg sees:

    * ``length_m``: ``|T_tx - tx| + |T_rx - rx|``.
    * ``length_rate_m_per_s``: its time derivative under the target velocity
      (the mirrored copy moves with the mirrored velocity).
    * ``arrival_unit``: the unit vector from the receiver toward ``T_rx``,
      the direction the receive array sees the echo arrive from.
    """

    tx = np.asarray(transmitter, dtype=np.float64)
    rx = np.asarray(receiver, dtype=np.float64)
    target_m = np.asarray(target, dtype=np.float64)
    velocity = np.asarray(target_velocity, dtype=np.float64)
    image = mirror_point(target_m, plane_point=plane_point, normal=normal)
    image_velocity = mirror_vector(velocity, normal=normal)
    copies = {"direct": (target_m, velocity), "reflected": (image, image_velocity)}
    out: dict[str, dict[str, object]] = {}
    for name in ROUND_TRIP_CLASSES:
        leg_in, leg_out = name.split("-")
        p_in, v_in = copies[leg_in]
        p_out, v_out = copies[leg_out]
        u_in = (p_in - tx) / np.linalg.norm(p_in - tx)
        u_out = (p_out - rx) / np.linalg.norm(p_out - rx)
        out[name] = {
            "length_m": float(np.linalg.norm(p_in - tx) + np.linalg.norm(p_out - rx)),
            "length_rate_m_per_s": float(np.dot(u_in, v_in) + np.dot(u_out, v_out)),
            "arrival_unit": u_out,
            "transmit_copy": p_in,
            "receive_copy": p_out,
        }
    return out


# ---------------------------------------------------------------------------
# Planar slab reflection (Airy summation)
# ---------------------------------------------------------------------------


def slab_reflection_coefficient(
    *, eps_r: float, sigma_e: float, thickness_m: float, frequency_hz: float, cos_incidence: float, polarization: str
) -> complex:
    """Reflection coefficient of a homogeneous slab in vacuum.

    The two-interface Airy sum
    ``r = r12 (1 - e^{-2 j delta}) / (1 - r12^2 e^{-2 j delta})``,
    with ``delta = k_z2 d`` the one-way phase thickness and ``r12`` the
    vacuum-to-slab Fresnel coefficient of the requested polarization:

    * ``"te"`` (E perpendicular to the plane of incidence):
      ``r12 = (cos_i - sqrt(eps - sin^2_i)) / (cos_i + sqrt(eps - sin^2_i))``;
    * ``"tm"`` (E in the plane of incidence), stated for the tangential
      magnetic field: ``r12 = (eps cos_i - sqrt(eps - sin^2_i)) / (eps cos_i + sqrt(eps - sin^2_i))``.

    ``eps = eps_r - j sigma / (omega eps_0)``; the square root takes the
    passive branch (non-positive imaginary part under ``exp(+j omega t)``).
    """

    omega = 2.0 * math.pi * float(frequency_hz)
    eps = complex(eps_r, -float(sigma_e) / (omega * EPS0_F_PER_M))
    sin2 = 1.0 - float(cos_incidence) ** 2
    root = cmath.sqrt(eps - sin2)
    if root.imag > 0.0:
        root = -root
    cos_i = float(cos_incidence)
    if polarization == "te":
        r12 = (cos_i - root) / (cos_i + root)
    elif polarization == "tm":
        r12 = (eps * cos_i - root) / (eps * cos_i + root)
    else:
        raise ValueError(f"polarization must be 'te' or 'tm', got {polarization!r}")
    k0 = omega / C0_M_PER_S
    phase = cmath.exp(-2j * k0 * root * float(thickness_m))
    return r12 * (1.0 - phase) / (1.0 - r12 * r12 * phase)


# ---------------------------------------------------------------------------
# Closed-form trajectories and the FMCW beat they produce
# ---------------------------------------------------------------------------


def ballistic_position(
    time_s: np.ndarray, *, origin: Sequence[float], velocity: Sequence[float], acceleration: Sequence[float]
) -> np.ndarray:
    """``p(t) = p0 + v t + a t^2 / 2`` in float64, shape ``(..., 3)``."""

    t = np.asarray(time_s, dtype=np.float64)[..., None]
    p0 = np.asarray(origin, dtype=np.float64)
    v = np.asarray(velocity, dtype=np.float64)
    a = np.asarray(acceleration, dtype=np.float64)
    return p0 + v * t + 0.5 * a * t * t


def monostatic_beat(
    *,
    positions_m: np.ndarray,
    transmitter: Sequence[float],
    receiver: Sequence[float],
    local_time_s: np.ndarray,
    carrier_hz: float,
    slope_hz_per_s: float,
) -> np.ndarray:
    """Unit-amplitude float64 beat phasor at each sample, complex128.

    ``positions_m`` is the target at each sample instant, ``local_time_s`` the
    time since the ramp start at the same instants.
    """

    tx = np.asarray(transmitter, dtype=np.float64)
    rx = np.asarray(receiver, dtype=np.float64)
    tau = (np.linalg.norm(positions_m - tx, axis=-1) + np.linalg.norm(positions_m - rx, axis=-1)) / C0_M_PER_S
    cycles = carrier_hz * tau + slope_hz_per_s * tau * (local_time_s - 0.5 * tau)
    # Reduce the integer cycles in float64 before the exponential so the phase
    # keeps its sub-cycle precision at every range this suite uses.
    return np.exp(2j * np.pi * (cycles - np.floor(cycles)))


__all__ = [
    "BOLTZMANN_J_PER_K",
    "C0_M_PER_S",
    "EPS0_F_PER_M",
    "REFERENCE_TEMPERATURE_K",
    "ROUND_TRIP_CLASSES",
    "ballistic_position",
    "bistatic_received_power_w",
    "coherent_integration_snr",
    "db",
    "flat_plate_broadside_rcs_m2",
    "image_round_trip",
    "mirror_point",
    "mirror_vector",
    "monostatic_beat",
    "rcs_from_monostatic_power",
    "slab_reflection_coefficient",
    "sphere_optical_rcs_m2",
    "system_noise_temperature_k",
    "triangular_trihedral_rcs_m2",
    "wavelength_m",
]
