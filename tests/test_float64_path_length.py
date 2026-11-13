"""M2-2 / D-1: the round-trip length is recomputed in float64 (R-ADR-040).

Channel computes a leg's phase as ``-k32 L32`` from float32 positions, a
float32 length and a float32 wave number. The join replaces it with
``-k64 L64``, ``L64`` the unfolded length from float64 endpoints and
Channel's float32 interaction points. Four facts make that correct, and each
is pinned here against Channel itself or a closed form:

* Channel's phase really is ``-k32 L32`` with ``k32 = float32(2 pi f / c)``
  evaluated in float32 (the replica the join removes);
* a specular length is stationary in its interaction point, so the float32
  point's error enters ``L64`` at second order (Fermat);
* the published float32 delay pair carries the float64 round-trip delay;
* the outcome: one micrometre of motion at 30 m moves the carrier phase by
  ``4 pi dR / lambda``, where a float32 position would quantize it away.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
import torch
from support import multi_endpoint_geometry as mgeo
from support import oracle_scenes as sc
from support import single_site_geometry as geo
from support import single_site_world as world
from support import spike_driver as drv

from witwin.radar.paths import SPEED_OF_LIGHT_M_PER_S, channel_wave_number

pytestmark = pytest.mark.gpu

K64 = 2.0 * math.pi * geo.REFERENCE_FREQUENCY_HZ / SPEED_OF_LIGHT_M_PER_S


def _precise_site(position) -> object:
    """The site as a float64-authored endpoint: float32 for Channel, float64 kept."""

    spec = world.endpoint_spec(position, geo.SITE_STABLE_ID)
    precise = torch.tensor([position], dtype=torch.float64, device="cuda")
    return dataclasses.replace(spec, precise_positions_m=precise)


def _legs(spike, site):
    tx = world.endpoint_spec(geo.TX_POSITION_M, geo.TX_STABLE_ID, power_w=geo.TX_POWER_W)
    rx = world.endpoint_spec(geo.RX_POSITION_M, geo.RX_STABLE_ID, power_w=geo.TX_POWER_W)
    inbound = spike.adapter.reevaluate_slots(spike.inbound, tx, site, slot_count=1, ad_mode="none")
    outbound = spike.adapter.reevaluate_slots(spike.outbound, rx, site, slot_count=1, ad_mode="none")
    return inbound, outbound


def test_channel_evaluates_a_legs_phase_as_minus_k32_times_its_float32_length():
    """The float32 wave number the join removes is Channel's own, to 1e-6 rad at 100 m.

    Using the float64 wave number instead would leave ``(k64 - k32) L32``,
    7.7e-4 rad at 100 m, which this test also shows is resolvable.
    """

    far = (-100.0, 0.6, 0.0)  # behind the transmitter, away from the wall
    spike = drv.Phase4Spike()
    tx = world.endpoint_spec(geo.TX_POSITION_M, geo.TX_STABLE_ID, power_w=geo.TX_POWER_W)
    inbound = spike.adapter.reevaluate_slots(
        spike.inbound, tx, world.endpoint_spec(far, geo.SITE_STABLE_ID), slot_count=1, ad_mode="none"
    )
    field = inbound.field[0].cpu().to(torch.complex128)
    length = float(inbound.path_length_m[0])
    component = field[int(field.abs().argmax())]
    k32 = channel_wave_number(geo.REFERENCE_FREQUENCY_HZ)
    # A line-of-sight field is a real vector times exp(-j k32 L32): the residual is 0 or pi.
    residual = complex(component * np.exp(1j * k32 * length))
    assert abs(math.sin(math.atan2(residual.imag, residual.real))) < 1e-6
    wrong = complex(component * np.exp(1j * K64 * length))
    assert abs(math.sin(math.atan2(wrong.imag, wrong.real))) > 1e-4


def test_a_specular_length_is_stationary_along_the_surface_and_not_across_it():
    """Fermat, measured on Channel's own float32 specular points.

    Moving the point within the reflecting plane changes the unfolded length
    at second order, so the point's float32 error ALONG the wall costs nothing:
    with the point's normal component put back on the wall (``x = 4``), the
    length equals the image-method length to 1e-12 m. ACROSS the wall the
    length changes at first order, and Channel's float32 point is off the
    plane by up to half a float32 ulp of its coordinate; that is the residual
    the correction cannot remove (R-ADR-040), bounded by
    ``(cos theta_i + cos theta_r) ulp32(x) / 2``.
    """

    spike = drv.Phase4Spike(components=frozenset({"los", "reflection"}), max_depth=1)
    inbound, outbound = _legs(spike, _precise_site(geo.SITE_POSITION_M))
    site = np.asarray(geo.SITE_POSITION_M)
    checked = 0
    for leg, endpoint in ((inbound, geo.TX_POSITION_M), (outbound, geo.RX_POSITION_M)):
        source = np.asarray(endpoint, dtype=np.float32).astype(np.float64)
        for row in range(leg.leg_count):
            if int(leg.depth[row]) != 1:
                continue
            point = leg.interaction_positions_m[row, 0].cpu().double().numpy()
            exact = mgeo.reflection_length_m(tuple(source), geo.SITE_POSITION_M)

            def unfolded(x, source=source):
                return np.linalg.norm(x - source) + np.linalg.norm(site - x)

            on_wall = point.copy()
            on_wall[0] = mgeo.WALL_PLANE_X_M
            assert abs(unfolded(on_wall) - exact) < 1e-12, (row, unfolded(on_wall) - exact)
            bound = 2.0 * float(np.spacing(np.float32(mgeo.WALL_PLANE_X_M))) / 2.0
            assert abs(unfolded(point) - exact) <= bound, (row, unfolded(point) - exact, bound)
            checked += 1
    assert checked == 2


def test_the_composed_delay_pair_is_the_float64_round_trip_delay():
    """Line-of-sight rows to 1e-20 s; reflected rows to the across-the-wall bound above."""

    spike = drv.Phase4Spike(components=frozenset({"los", "reflection"}), max_depth=1)
    inbound, outbound = _legs(spike, _precise_site(geo.SITE_POSITION_M))
    composed = spike.composer.compose(inbound, outbound, drv.make_response())
    assert composed.total_delay_residual_s is not None
    produced = composed.total_delay_s.double() + composed.total_delay_residual_s.double()
    # The array elements are the radar's float32 positions; the site is exact.
    elements = {
        "inbound": tuple(np.float32(geo.TX_POSITION_M).astype(np.float64)),
        "outbound": tuple(np.float32(geo.RX_POSITION_M).astype(np.float64)),
    }

    def length(leg: str, component_id: int) -> float:
        if component_id == geo.LOS_COMPONENT_ID:
            return mgeo.distance_m(elements[leg], geo.SITE_POSITION_M)
        return mgeo.reflection_length_m(elements[leg], geo.SITE_POSITION_M)

    reflection_bound = float(np.spacing(np.float32(mgeo.WALL_PLANE_X_M))) / SPEED_OF_LIGHT_M_PER_S
    for row in range(composed.path_count):
        i, o = int(composed.topology.inbound_row[row]), int(composed.topology.outbound_row[row])
        components = (int(inbound.component_id[i]), int(outbound.component_id[o]))
        expected = (length("inbound", components[0]) + length("outbound", components[1])) / SPEED_OF_LIGHT_M_PER_S
        reflections = sum(component == geo.REFLECTION_COMPONENT_ID for component in components)
        tolerance = 1e-20 if reflections == 0 else reflections * reflection_bound
        assert abs(float(produced[row]) - expected) <= tolerance, (row, components, float(produced[row]) - expected)


def _carrier_phase(position: list[list[float]], dtype: torch.dtype) -> float:
    from witwin.radar import PointTargets

    radar = sc.radar()
    targets = PointTargets(positions=torch.tensor(position, dtype=dtype, device=radar.device), rcs=1.0)
    result = radar.simulate(sc.empty_scene(), targets, times=(0.0,), los=True, reflections=0)
    return float(torch.angle(result.cube.reshape(-1)[0]))


def test_one_micrometre_of_motion_is_visible_at_30_m():
    """``4 pi dR / lambda`` = 3.23 mrad for 1 um at 77 GHz, to 1%.

    The float32 ulp at 30 m is 1.9 um, so a float32-authored position either
    does not move or moves by a whole ulp; the float64 one moves by 1 um.
    """

    step = 1.0e-6
    expected = 4.0 * math.pi * step * sc.CARRIER_HZ / SPEED_OF_LIGHT_M_PER_S
    near, far = [[30.0, 0.0, 0.0]], [[30.0 + step, 0.0, 0.0]]
    # The cube is the beat (conjugated) convention: a longer path advances the phase.
    moved = _carrier_phase(far, torch.float64) - _carrier_phase(near, torch.float64)
    moved = math.remainder(moved, 2.0 * math.pi)
    assert abs(abs(moved) - expected) < 0.01 * expected, (moved, expected)
    quantized = math.remainder(_carrier_phase(far, torch.float32) - _carrier_phase(near, torch.float32), 2 * math.pi)
    assert abs(abs(quantized) - expected) > 0.3 * expected, (quantized, expected)


def test_every_wideband_column_carries_the_float64_phase_at_its_own_frequency():
    """Column n of a band is ``exp(-j 2 pi (f_ref + n df) tau64)`` relative to column 0.

    The join removes Channel's float32 phase per column with that column's own
    float32 wave number (Channel launches column n at float32(f_ref + n df)) and
    applies the float64 one. A 10 MHz spacing is not a multiple of the float32
    ulp of 77 GHz (8192 Hz), so a replica of the wrong column frequency would
    leave up to k32 ulp x L = 1e-2 rad here; the float32 field leaves 1e-6.
    """

    from witwin.radar import Ofdm, PointTargets

    spacing, subcarriers, range_m = 10.0e6, 64, 37.000000123
    radar = sc.radar(
        Ofdm(
            subcarrier_spacing=spacing,
            num_subcarriers=subcarriers,
            cyclic_prefix=3e-7,
            num_symbols=1,
            max_expected_delay=2.6e-7,
        )
    )
    targets = PointTargets(
        positions=torch.tensor([[range_m, 0.0, 0.0]], dtype=torch.float64, device=radar.device), rcs=1.0
    )
    rows = radar.trace(sc.empty_scene(), targets, times=(0.0,), los=True, reflections=0, wideband=True).rows()
    band = rows.complex_transfer_band
    assert band is not None and tuple(band.shape) == (1, subcarriers)
    tau = 2.0 * range_m / SPEED_OF_LIGHT_M_PER_S
    produced = torch.angle(band[0] * band[0, :1].conj()).double().cpu().numpy()
    expected = -2.0 * math.pi * spacing * np.arange(subcarriers) * tau
    error = np.angle(np.exp(1j * (produced - expected)))
    assert float(np.abs(error).max()) < 1e-5, float(np.abs(error).max())
