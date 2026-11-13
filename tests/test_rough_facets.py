"""MAT-2: rough-surface backscatter of a mesh target (R-ADR-044).

A perfectly conducting 5 cm plate of 512 facets at 10 m, with a Gaussian
surface roughness of rms height ``h`` on its material. The facet law splits
each facet's return into a coherent part, reduced by the Ament factor
``exp(-(2 k h cos)^2 / 2)``, and a Lambertian diffuse part of cross section
``4 A (1 - exp(-(2 k h cos)^2)) |Gamma|^2 cos^2`` with a frozen random phase per
facet. Over many diffuse seeds the mean complex return is the coherent part
and the variance about it is the diffuse power. Expected values are the
closed forms above, summed over the plate; nothing is read from the kernel.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from reference import radar_oracles as ref
from support import oracle_scenes as sc

pytestmark = pytest.mark.gpu

RANGE_M = 10.0
SIDE_M = 0.05
WAVELENGTH_M = ref.wavelength_m(sc.CARRIER_HZ)
WAVENUMBER = 2.0 * math.pi / WAVELENGTH_M
ROUGHNESS_M = (0.0, WAVELENGTH_M / 16.0, WAVELENGTH_M / 8.0)
#: The diffuse power is a variance estimate over the seeds; with 48 its relative
#: standard error is about 15%, so the 45% tolerance below is three of them.
SEEDS = 48


def _scene(rms_m: float, tilt_rad: float):
    from witwin.core import PhysicalMaterial, SurfaceRoughness

    vertices, faces = sc.plate_mesh(16, side_m=SIDE_M, range_m=RANGE_M)
    centre = np.array([RANGE_M, 0.0, 0.0])
    rotation = np.array(
        [[math.cos(tilt_rad), 0.0, math.sin(tilt_rad)], [0.0, 1.0, 0.0], [-math.sin(tilt_rad), 0.0, math.cos(tilt_rad)]]
    )
    vertices = (vertices - centre) @ rotation.T + centre
    roughness = (
        None
        if rms_m == 0.0
        else SurfaceRoughness(rms_height_m=rms_m, correlation_length_x_m=0.01, correlation_length_y_m=0.01)
    )
    return sc.mesh_scene(vertices, faces, material=PhysicalMaterial.perfect_conductor(roughness_front=roughness))


def _rcs(power: float) -> float:
    return ref.rcs_from_monostatic_power(
        received_power_w=power, transmit_power_w=ref.watts_from_dbm(10.0), wavelength=WAVELENGTH_M, range_m=RANGE_M
    )


def _split(rms_m: float, tilt_rad: float) -> tuple[float, float]:
    """(coherent RCS, diffuse RCS) over the diffuse seeds, m^2."""

    from witwin.radar import MeshTargets

    radar = sc.radar()
    scene = _scene(rms_m, tilt_rad)
    returns = []
    for seed in range(SEEDS):
        rows = radar.trace(scene, MeshTargets(seed=seed), times=(0.0,), reflections=0).rows()
        returns.append(complex(rows.complex_transfer_ref.to(torch.complex128).sum()))
    values = np.asarray(returns)
    mean = values.mean()
    return _rcs(abs(mean) ** 2), _rcs(float(np.mean(np.abs(values - mean) ** 2)) * SEEDS / (SEEDS - 1))


def _diffuse_expected(rms_m: float, tilt_rad: float) -> float:
    cosine = math.cos(tilt_rad)
    return 4.0 * SIDE_M**2 * (1.0 - math.exp(-((2.0 * WAVENUMBER * rms_m * cosine) ** 2))) * cosine**2


def test_roughness_moves_power_from_the_specular_peak_to_the_diffuse_floor():
    smooth = ref.flat_plate_broadside_rcs_m2(SIDE_M**2, WAVELENGTH_M)
    peaks, floors, off_floors = [], [], []
    for rms in ROUGHNESS_M:
        coherent, diffuse = _split(rms, 0.0)
        expected_peak = smooth * math.exp(-((2.0 * WAVENUMBER * rms) ** 2))
        assert abs(ref.db(coherent / expected_peak)) < 0.2, (rms, coherent, expected_peak)
        if rms > 0.0:
            assert diffuse == pytest.approx(_diffuse_expected(rms, 0.0), rel=0.45)
        # Bounded: the rough plate never returns more than the smooth one.
        assert coherent + diffuse <= smooth * 1.01
        _, off = _split(rms, math.radians(40.0))
        if rms > 0.0:
            assert off == pytest.approx(_diffuse_expected(rms, math.radians(40.0)), rel=0.45)
        peaks.append(coherent)
        floors.append(diffuse)
        off_floors.append(off)
    assert peaks[0] > peaks[1] > peaks[2]
    assert floors[1] < floors[2] and off_floors[0] < off_floors[1] < off_floors[2]


def test_the_diffuse_term_carries_exactly_the_power_the_coherent_term_loses():
    """Energy: hemisphere-integrated, a facet's Lambertian term is (1 - exp(-g_i)) |Gamma|^2 of the incident power.

    With ``sigma_d = 4 A f cos_i cos_s``, the scattered power is
    ``(I / 4 pi) int sigma_d dOmega = I A cos_i f``, and the coherent specular
    power is ``I A cos_i exp(-g_i) |Gamma|^2``; their sum is the smooth facet's
    ``I A cos_i |Gamma|^2`` for every roughness. This checks that the closed
    form the kernel implements integrates to that, on a fine angular grid.
    """

    for rms in ROUGHNESS_M[1:]:
        for incidence in (0.0, math.radians(30.0), math.radians(60.0)):
            g = (2.0 * WAVENUMBER * rms * math.cos(incidence)) ** 2
            fraction = 1.0 - math.exp(-g)
            polar = (np.arange(4000) + 0.5) * (0.5 * math.pi / 4000)
            integral = np.sum(4.0 * fraction * math.cos(incidence) * np.cos(polar) * np.sin(polar)) * (
                0.5 * math.pi / 4000
            )
            scattered = 2.0 * math.pi * integral / (4.0 * math.pi)
            assert scattered + math.exp(-g) * math.cos(incidence) == pytest.approx(math.cos(incidence), rel=1e-6)
