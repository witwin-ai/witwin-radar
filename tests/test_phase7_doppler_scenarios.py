"""The Phase-7 Doppler acceptance scenarios, end to end through the seam.

Every scenario here drives the PRODUCTION chain - Core-shaped kinematics into
``two_way_duals``, into ``ChannelPropagationAdapter.reevaluate``, into the
native two-way join - and checks the answer against the float64 image-source
closed form in ``support.multi_endpoint_geometry``. The oracle never imports
anything from ``witwin.radar.paths``; it derives which rows exist from the facet
geometry and each row's rate from the projection of the moving endpoint's
velocity onto the (possibly mirrored) leg direction.

Sign convention, once: ``f_D = -f_ref * d(tau_rt)/dt``. Channel publishes
``exp(-j k d)``, so an APPROACHING row has a shrinking delay, a negative rate,
and a POSITIVE Doppler.

Two things every scenario carries deliberately.

* A non-zero RADIAL component wherever the answer is meant to be non-zero. A
  purely transverse fixture cannot distinguish a correct near-zero rate from a
  dead forward-AD tangent, which also publishes zero. R-ADR-012 states the rule;
  ``SITE_P_RADIAL_VELOCITY_M_PER_S`` and the moving front end satisfy it.
* Rows that must be EXACTLY zero are asserted with ``torch.equal``, not with a
  tolerance. "Small" would be satisfied by a broken chain that happened to be
  quiet.

Deviation from the brief, recorded rather than hidden: the reference values
``-1378.4 Hz`` (line of sight) and ``-219.6 Hz`` (reflection) quoted for the
tangential scenario belong to the PHASE-4 single-endpoint geometry, whose
transmitter, site and receiver sit at different places from this fixture's. The
scenario is run on the multi-endpoint fixture as the brief's own fixture
instruction requires, so the reference is that fixture's closed form; the
structural claim - that the reflection row and the line-of-sight row of the same
endpoint triple carry DIFFERENT shifts because the reflection sees the image
source - is asserted directly and is the part that was ever load bearing.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import pytest
import torch
import torch.autograd.forward_ad as forward_ad

pytest.importorskip("witwin.channel")

from support import multi_endpoint_driver as drv  # noqa: E402
from support import multi_endpoint_geometry as geo  # noqa: E402
from witwin.radar.propagation import kinematics as kin  # noqa: E402


pytestmark = pytest.mark.gpu


ZERO3 = geo.STATIONARY

#: Tolerance against the float64 closed form. The measured worst case over
#: every scenario in this file is 6.9e-6 relative, so this is three orders of
#: margin and it is a bound on the float32 chain, not a fitted number.
RATE_RTOL = 2.0e-3


@pytest.fixture(scope="module")
def spike():
    return drv.MultiEndpointSpike()


@dataclass(frozen=True, slots=True, eq=False)
class Measured:
    """One frame's primal payload, lifted out of the dual level.

    ``delay_rate`` is a PRIMAL Doppler value by contract, so it leaves the level
    detached exactly as synthesis consumes it. Everything else is carried along
    so a test does not have to re-run the frame to look at a delay.
    """

    composed: object
    total_delay_s: torch.Tensor
    delay_rate: torch.Tensor
    transfer: torch.Tensor
    inbound_rate: torch.Tensor
    outbound_rate: torch.Tensor

    def frame(self) -> object:
        """The composed batch with primal payloads, ready for synthesis."""

        return replace(
            self.composed,
            total_delay_s=self.total_delay_s,
            delay_rate=self.delay_rate,
            complex_transfer_ref=self.transfer,
        )

    @property
    def doppler_hz(self) -> list[float]:
        return [
            -geo.REFERENCE_FREQUENCY_HZ * value
            for value in self.delay_rate.tolist()
        ]


def _track(positions: torch.Tensor, velocities) -> kin.Kinematics:
    return kin.Kinematics(
        positions_m=positions,
        velocities_m_per_s=torch.tensor(
            list(velocities), dtype=torch.float32, device="cuda"
        ),
    )


def _measure(
    spike,
    *,
    site_velocity=None,
    transmitter_velocity=None,
    receiver_velocity=None,
) -> Measured:
    """One frame with ALL THREE endpoint tensors dualised in one level.

    The transmitters and receivers are always dualised, even when they are
    stationary, because a zero tangent and a MISSING tangent are different
    things: the second is the trap R-ADR-012 exists to close, and covering the
    tensors unconditionally is what keeps a scenario from silently becoming a
    single-tensor test when its velocity happens to be zero.
    """

    sites = _track(
        spike.site_tensor(), site_velocity or [ZERO3] * len(spike.sites)
    )
    transmitters = _track(
        spike.transmitter_tensor(),
        transmitter_velocity or [ZERO3] * len(spike.transmitters),
    )
    receivers = _track(
        spike.receiver_tensor(),
        receiver_velocity or [ZERO3] * len(spike.receivers),
    )
    with kin.two_way_duals(
        sites=sites, transmitters=transmitters, receivers=receivers
    ) as duals:
        for tensor in (duals.transmitters, duals.sites, duals.receivers):
            assert forward_ad.unpack_dual(tensor).tangent is not None
        composed, inbound, outbound = spike.frame(
            duals.sites,
            transmitters=duals.transmitters,
            receivers=duals.receivers,
            ad_mode="jvp",
        )
        return Measured(
            composed=composed,
            total_delay_s=composed.total_delay_s.detach().clone(),
            delay_rate=composed.delay_rate.detach().clone(),
            transfer=composed.complex_transfer_ref.detach().clone(),
            inbound_rate=inbound.delay_rate.detach().clone(),
            outbound_rate=outbound.delay_rate.detach().clone(),
        )


def _assert_matches_oracle(spike, measured: Measured, velocities: dict) -> None:
    expected = geo.combined_delay_rate_s_per_s(
        spike.predicted_combined_rows(), velocities
    )
    for index, (value, reference) in enumerate(
        zip(measured.delay_rate.tolist(), expected, strict=True)
    ):
        if reference == 0.0:
            assert value == 0.0, index
        else:
            assert value == pytest.approx(reference, rel=RATE_RTOL), index


def _row_named(spike, key) -> int:
    rows = spike.predicted_combined_rows()
    return next(index for index, row in enumerate(rows) if row.key == key)


# --------------------------------------------------------------------------
# S1  static
# --------------------------------------------------------------------------


def test_a_static_scene_has_exactly_zero_delay_rate(spike):
    """Not "small". EXACTLY zero, on every one of the eleven rows.

    A stationary world is the one case where the correct answer and the
    dead-tangent answer coincide, so the value alone proves nothing. What makes
    this test worth having is that it is asserted with ``torch.equal`` and that
    the whole chain still ran in ``jvp`` mode: every tangent was live (the
    helper checks that), the legs were reevaluated, the join composed, and the
    result was zero because the velocities were zero.
    """

    measured = _measure(spike)
    assert measured.delay_rate.shape[0] == 11
    assert torch.equal(
        measured.delay_rate, torch.zeros_like(measured.delay_rate)
    )
    assert torch.equal(
        measured.inbound_rate, torch.zeros_like(measured.inbound_rate)
    )
    assert torch.equal(
        measured.outbound_rate, torch.zeros_like(measured.outbound_rate)
    )


def test_a_static_scene_has_no_slow_time_phase_slope(spike):
    """The frozen-mode cube repeats chirp for chirp, bitwise.

    The frozen weight is computed once at the frame's ``tau_rt`` and the kernel
    owns the slow-time carrier through ``carrier_rate_hz * tau_rate * t_slot``.
    With ``tau_rate`` exactly zero that term vanishes and every chirp of the
    frame must be the SAME complex sample - not merely close. Any residual
    slow-time slope here would be a carrier the kernel is advancing on its own.
    """

    from witwin.radar.synthesis.fmcw_beat import synthesize_fmcw_beat

    spec = drv.make_spec(num_chirps=8)
    cube = synthesize_fmcw_beat(drv.to_synthesis(_measure(spike).frame()), spec)
    assert cube.shape[0] == 8
    assert float(cube.abs().max()) > 0.0
    for chirp in range(1, cube.shape[0]):
        assert torch.equal(cube[chirp], cube[0])


def test_a_static_scene_repeats_its_weight_in_every_slot(spike):
    """The refreshed-weight producer, on a world that is not moving.

    ``reevaluate_slots`` replays the frozen topology once per slot from the
    slot's own endpoint positions, so it is the mode that WOULD see intra-frame
    motion. On a static world the per-slot weights must be bit-identical, which
    is the statement that the slot axis carries no phase of its own: any drift
    here would be Doppler the refreshed mode invented, and it would then be
    counted twice against a kernel carrier.
    """

    slots = 8
    inbound, outbound = spike.slot_legs(
        spike.stacked(spike.site_tensor(), slots), slot_count=slots
    )
    for leg in (inbound, outbound):
        assert leg.slot_count == slots
        first = leg.slot(0)
        for slot in range(1, slots):
            later = leg.slot(slot)
            assert torch.equal(later.coefficient, first.coefficient)
            assert torch.equal(later.delay_s, first.delay_s)


# --------------------------------------------------------------------------
# S2  radial
# --------------------------------------------------------------------------


def test_a_radially_moving_site_matches_the_projection_formula(spike):
    """The scenario that makes the dead-tangent trap detectable.

    ``SITE_P`` closes on ``TX_A`` along ``-x`` at 12 m/s, which is radial to
    every leg that reaches it, so no row's rate is accidentally near zero. The
    per-LEG reference is the projection of the velocity onto the unit vector
    from the (mirrored, for a reflection) fixed endpoint to the site, and the
    round trip is the sum of two of them.

    ``SITE_Q`` is stationary in the same frame, so its rows must come back
    EXACTLY zero while ``SITE_P``'s do not. One tensor, two answers: that is
    what rules out a chain that lost the tangent and one that broadcast a
    single velocity over every site.
    """

    velocity = geo.SITE_P_RADIAL_VELOCITY_M_PER_S
    velocities = {geo.SITE_P_STABLE_ID: velocity}
    measured = _measure(spike, site_velocity=[velocity, ZERO3])
    _assert_matches_oracle(spike, measured, velocities)

    # Per LEG, against the projection formula directly.
    for name, rates, rows in (
        ("inbound", measured.inbound_rate, spike.predicted_inbound_rows()),
        ("outbound", measured.outbound_rate, spike.predicted_outbound_rows()),
    ):
        expected = geo.leg_delay_rates_s_per_s(rows, velocities)
        for index, (value, reference) in enumerate(
            zip(rates.tolist(), expected, strict=True)
        ):
            if reference == 0.0:
                assert value == 0.0, (name, index)
            else:
                assert value == pytest.approx(reference, rel=RATE_RTOL), (
                    name,
                    index,
                )

    # The stationary site is exactly stationary, and the moving one is not.
    rows = spike.predicted_combined_rows()
    moving = [
        index
        for index, row in enumerate(rows)
        if row.site_id == geo.SITE_P_STABLE_ID
    ]
    still = [
        index
        for index, row in enumerate(rows)
        if row.site_id == geo.SITE_Q_STABLE_ID
    ]
    assert moving and still
    values = measured.delay_rate
    assert torch.equal(values[still], torch.zeros_like(values[still]))
    assert float(values[moving].abs().min()) > 0.0


def test_the_doppler_sign_follows_the_channel_phasor(spike):
    """Approaching is POSITIVE, receding is NEGATIVE, and the two are negatives.

    A sign error here is invisible in a magnitude-only range-Doppler map and
    surfaces much later as a target that approaches when it should recede.
    """

    approaching = _measure(
        spike, site_velocity=[geo.SITE_P_RADIAL_VELOCITY_M_PER_S, ZERO3]
    )
    receding = _measure(
        spike,
        site_velocity=[
            tuple(-value for value in geo.SITE_P_RADIAL_VELOCITY_M_PER_S),
            ZERO3,
        ],
    )
    row = _row_named(spike, (geo.TX_A_STABLE_ID, geo.SITE_P_STABLE_ID,
                             geo.RX_A_STABLE_ID, "los", "los"))
    assert approaching.doppler_hz[row] > 1000.0
    assert receding.doppler_hz[row] < -1000.0
    torch.testing.assert_close(
        approaching.delay_rate, -receding.delay_rate, rtol=1e-6, atol=0.0
    )


# --------------------------------------------------------------------------
# S3  tangential
# --------------------------------------------------------------------------


def test_the_los_and_reflection_rows_differ(spike):
    """The image source is what makes the two components disagree.

    ``SITE_P`` moves along ``+y`` at 12 m/s. The line-of-sight leg sees that
    velocity projected onto ``TX_A -> P``; the reflection leg sees it projected
    onto ``image(TX_A) -> P``, a different direction, so the two rows of the
    SAME endpoint triple carry different shifts. Equality would mean the
    reflection row was being computed from the direct geometry, which is a
    failure this fixture is built to catch and which a magnitude check cannot.
    """

    velocities = {geo.SITE_P_STABLE_ID: geo.SITE_P_VELOCITY_M_PER_S}
    measured = _measure(
        spike, site_velocity=[geo.SITE_P_VELOCITY_M_PER_S, ZERO3]
    )
    _assert_matches_oracle(spike, measured, velocities)

    triple = (geo.TX_A_STABLE_ID, geo.SITE_P_STABLE_ID, geo.RX_A_STABLE_ID)
    los = measured.doppler_hz[_row_named(spike, (*triple, "los", "los"))]
    reflection = measured.doppler_hz[
        _row_named(spike, (*triple, "reflection", "reflection"))
    ]
    mixed = measured.doppler_hz[
        _row_named(spike, (*triple, "los", "reflection"))
    ]

    assert los < 0.0 and reflection < 0.0
    assert abs(los - reflection) > 100.0
    # The mixed round trip sits strictly between them, which is what "one leg
    # each" means and what a reflection row copied from the direct geometry
    # could not produce.
    assert reflection > mixed > los


# --------------------------------------------------------------------------
# S6  moving transmitter and receiver
# --------------------------------------------------------------------------


def test_moving_transmitter_and_receiver(spike):
    """The half of item 4 that had no fixture until the driver took tensors.

    ``TX_A`` and ``RX_A`` cross in opposite directions and every site is
    STATIC, so the entire round-trip rate comes from the front end. Before the
    driver accepted live transmitter and receiver tensors this scenario could
    not be expressed at all: both batches were rebuilt from Python tuples on
    every call and neither could carry a tangent.
    """

    velocities = {
        geo.TX_A_STABLE_ID: geo.TX_A_VELOCITY_M_PER_S,
        geo.RX_A_STABLE_ID: geo.RX_A_VELOCITY_M_PER_S,
    }
    measured = _measure(
        spike,
        transmitter_velocity=[geo.TX_A_VELOCITY_M_PER_S, ZERO3],
        receiver_velocity=[geo.RX_A_VELOCITY_M_PER_S, ZERO3],
    )
    _assert_matches_oracle(spike, measured, velocities)

    # Not vacuous: with the sites frozen the answer is entirely the front end's,
    # and it is nowhere near zero.
    assert float(measured.delay_rate.abs().max()) > 0.0
    assert max(abs(value) for value in measured.doppler_hz) > 100.0

    # A row reaching the STATIONARY receiver still moves, because its
    # transmitter does; a chain that only dualised one end would zero one of
    # these two groups.
    rows = spike.predicted_combined_rows()
    to_rx_b = [
        index for index, row in enumerate(rows) if row.sink_id == geo.RX_B_STABLE_ID
    ]
    to_rx_a = [
        index for index, row in enumerate(rows) if row.sink_id == geo.RX_A_STABLE_ID
    ]
    assert float(measured.delay_rate[to_rx_b].abs().min()) > 0.0
    assert float(measured.delay_rate[to_rx_a].abs().min()) > 0.0


def test_moving_endpoints_equal_the_reciprocal_moving_site(spike):
    """Galilean equivalence, and the exact limit of it.

    Boosting the whole front end by ``u`` with a static target must give the
    same round-trip rate as holding the front end still and moving the target
    by ``-u``. That is the strongest available check that the seam built its
    tangents for the tensors it claims: getting it wrong requires making the
    same mistake twice, in opposite directions.

    The equivalence is EXACT for a line-of-sight row at any ``u``. It is exact
    for a reflection row only when ``u`` lies in the reflecting plane, because
    the WALL does not move under the boost. For a wall-normal ``u`` the mirror
    turns the reflection leg's contribution around, so a double-reflection row's
    rate in the two cases is not equal but exactly OPPOSITE. Both halves are
    asserted; the second is a sharper statement than the first and would be lost
    if the scenario only ever used a wall-parallel velocity.
    """

    parallel = (0.0, 3.0, 0.0)
    boosted = _measure(
        spike,
        transmitter_velocity=[parallel] * len(spike.transmitters),
        receiver_velocity=[parallel] * len(spike.receivers),
    )
    reciprocal = _measure(
        spike,
        site_velocity=[tuple(-value for value in parallel)] * len(spike.sites),
    )
    assert float(boosted.delay_rate.abs().min()) > 0.0
    torch.testing.assert_close(
        boosted.delay_rate, reciprocal.delay_rate, rtol=1e-4, atol=0.0
    )

    normal = (-3.0, 0.0, 0.0)
    boosted = _measure(
        spike,
        transmitter_velocity=[normal] * len(spike.transmitters),
        receiver_velocity=[normal] * len(spike.receivers),
    )
    reciprocal = _measure(
        spike,
        site_velocity=[tuple(-value for value in normal)] * len(spike.sites),
    )
    rows = spike.predicted_combined_rows()
    line_of_sight = [
        index
        for index, row in enumerate(rows)
        if row.inbound.component == "los" and row.outbound.component == "los"
    ]
    two_bounce = [
        index
        for index, row in enumerate(rows)
        if row.inbound.component == "reflection"
        and row.outbound.component == "reflection"
    ]
    assert line_of_sight and two_bounce
    torch.testing.assert_close(
        boosted.delay_rate[line_of_sight],
        reciprocal.delay_rate[line_of_sight],
        rtol=1e-4,
        atol=0.0,
    )
    torch.testing.assert_close(
        boosted.delay_rate[two_bounce],
        -reciprocal.delay_rate[two_bounce],
        rtol=1e-4,
        atol=0.0,
    )
    assert float(boosted.delay_rate[two_bounce].abs().min()) > 0.0


# --------------------------------------------------------------------------
# The round trip is the sum of its legs
# --------------------------------------------------------------------------


def test_the_round_trip_rate_is_the_sum_of_the_two_legs(spike):
    """Bitwise, on a geometry where every endpoint moves.

    The join accumulates ``rate_in + rate_out`` in double and rounds once to
    float32, and IEEE float32 addition is correctly rounded, so the two agree to
    the last bit rather than merely to a tolerance. Asserting the tolerance
    instead would hide a join that dropped one leg's contribution on the rows
    where that contribution happened to be small.

    Both sites, one transmitter and both receivers carry different velocities in
    different directions, so no two rows share a rate and no cancellation can
    make a wrong sum look right.
    """

    velocities = {
        geo.SITE_P_STABLE_ID: (-9.0, 4.0, 0.0),
        geo.SITE_Q_STABLE_ID: (2.0, -3.0, 1.0),
        geo.TX_A_STABLE_ID: (1.0, 2.0, 0.0),
        geo.RX_A_STABLE_ID: (0.0, -2.5, 0.5),
        geo.RX_B_STABLE_ID: (1.5, 0.0, 0.0),
    }
    measured = _measure(
        spike,
        site_velocity=[
            velocities[geo.SITE_P_STABLE_ID],
            velocities[geo.SITE_Q_STABLE_ID],
        ],
        transmitter_velocity=[velocities[geo.TX_A_STABLE_ID], ZERO3],
        receiver_velocity=[
            velocities[geo.RX_A_STABLE_ID],
            velocities[geo.RX_B_STABLE_ID],
        ],
    )
    topology = measured.composed.topology
    summed = (
        measured.inbound_rate[topology.inbound_row]
        + measured.outbound_rate[topology.outbound_row]
    )
    summed = torch.where(
        measured.composed.row_valid, summed, torch.zeros_like(summed)
    )
    assert torch.equal(measured.delay_rate, summed)

    _assert_matches_oracle(spike, measured, velocities)
    shifts = sorted(measured.doppler_hz)
    assert min(b - a for a, b in zip(shifts[:-1], shifts[1:], strict=True)) > 1.0


# --------------------------------------------------------------------------
# Aliasing
# --------------------------------------------------------------------------


def _slow_time_peak_hz(frame, spec, row: int):
    """The slow-time tone of ONE composed row, isolated through ``row_valid``.

    Isolating by validity rather than by rebuilding the batch is the contract's
    own way of saying a row contributes nothing, and it keeps the pair
    partition - and therefore the TDM slot table - exactly as the frame
    published it.
    """

    from witwin.radar.synthesis.fmcw_beat import synthesize_fmcw_beat

    alone = torch.zeros(frame.path_count, dtype=torch.bool, device=frame.device)
    alone[row] = True
    cube = synthesize_fmcw_beat(
        drv.to_synthesis(replace(frame, row_valid=alone)), spec
    ).cpu()
    pair = int(frame.sensor_pair_index[row])
    slow = cube[:, pair, 0].to(torch.complex128)
    spectrum = torch.fft.fftshift(torch.fft.fft(slow)).abs()
    frequencies = torch.fft.fftshift(
        torch.fft.fftfreq(spec.num_chirps, d=spec.slot_period_s)
    )
    peak = float(frequencies[int(spectrum.argmax())])
    return peak, float(frequencies[1] - frequencies[0])


@pytest.mark.parametrize("speed_mps", [6.0, 12.0])
def test_doppler_aliasing_folds_as_predicted(spike, speed_mps):
    """In limit it reports the tone; over limit it folds to the PREDICTED alias.

    ``max_unambiguous_speed_mps = lambda / (4 T_chirp num_tx)`` is 7.487 m/s on
    this two-transmitter front end. A site closing at 6 m/s implies 5.73 m/s of
    round-trip radial rate and stays inside it; at 12 m/s the implied rate is
    11.45 m/s and the tone must WRAP. What the test buys is that it wraps to
    ``tone - k * PRF_eff`` for the integer ``k`` the bound predicts, and not to
    an arbitrary number: an aliased measurement is still a measurement, and the
    difference between "folded" and "garbage" is the whole reason the bound is
    published.
    """

    spec = drv.make_spec(num_chirps=64)
    measured = _measure(
        spike, site_velocity=[(-float(speed_mps), 0.0, 0.0), ZERO3]
    )
    row = _row_named(spike, (geo.TX_A_STABLE_ID, geo.SITE_P_STABLE_ID,
                             geo.RX_A_STABLE_ID, "los", "los"))
    rate = float(measured.delay_rate[row])
    delay = float(measured.total_delay_s[row])
    implied_mps = geo.C0_M_PER_S * abs(rate) / 2.0
    in_limit = implied_mps < spec.max_unambiguous_speed_mps
    assert in_limit == (speed_mps == 6.0), implied_mps

    # The beat cube is conjugated once, so its slow-time tone sits at
    # +tau_rate * (f_ref + slope * (t_start - tau_rt)) - the carrier plus the
    # ramp's own contribution at the sample being read.
    tone_hz = rate * (
        spec.reference_frequency_hz
        + spec.slope_hz_per_s * (spec.t_start_s - delay)
    )
    prf_hz = 1.0 / spec.slot_period_s
    folds = round(tone_hz / prf_hz)
    folded_hz = tone_hz - folds * prf_hz
    assert folds == (0 if in_limit else -1)

    peak_hz, bin_hz = _slow_time_peak_hz(measured.frame(), spec, row)
    assert abs(peak_hz - folded_hz) <= 0.5 * bin_hz

    if in_limit:
        assert peak_hz == pytest.approx(tone_hz, abs=0.5 * bin_hz)
    else:
        # It really folded: the unaliased tone is many bins away and on the
        # other side of the band.
        assert abs(peak_hz - tone_hz) > 20.0 * bin_hz
        assert peak_hz * tone_hz < 0.0
