"""TDM slot time in the beat family: the slow-time axis is slots, not chirps.

TDM-MIMO fires the transmitters sequentially, so the slow-time coordinate of a
(chirp, sensor pair) cell is its slot ``chirp * num_tx + tx``, one chirp period
apart. Two things have to be true at once and they pull in opposite directions:

* at ``num_tx > 1`` the per-TX phase walk must be REAL, so that the phase
  ``sigproc/pointcloud.py::_compensate_tdm_phase`` removes downstream is the
  phase the kernel put there rather than a downstream reinterpretation;
* at ``num_tx = 1`` the slot must collapse to the chirp index EXACTLY, so that
  every pre-TDM result is reproduced bit for bit.

The geometry is the physics survey's probe geometry, as in
``test_phase6_fmcw_analytic``.
"""

from __future__ import annotations

import math

import pytest
import torch

from support import fd  # noqa: E402
from support import reference_chain as ref  # noqa: E402
from witwin.radar.synthesis.contracts import (  # noqa: E402
    SPEED_OF_LIGHT_M_PER_S,
    FmcwBeatSpec,
)
from witwin.radar.synthesis.fmcw_beat import synthesize_beat_rows  # noqa: E402


pytestmark = pytest.mark.gpu


C0 = SPEED_OF_LIGHT_M_PER_S
FC_HZ = 77.0e9
SLOPE_HZ_PER_S = 6.0e13
SAMPLE_RATE_HZ = 5.0e6
NUM_SAMPLES = 256
T_START_S = 6.0e-6
CHIRP_PERIOD_S = 60.0e-6

TAU_RT_S = 2.0 * 3.7 / C0
TAU_RATE = 2.0 * 12.0 / C0


def _spec(**overrides) -> FmcwBeatSpec:
    fields = dict(
        num_samples=NUM_SAMPLES,
        num_chirps=8,
        sample_period_s=1.0 / SAMPLE_RATE_HZ,
        chirp_period_s=CHIRP_PERIOD_S,
        slope_hz_per_s=SLOPE_HZ_PER_S,
        t_start_s=T_START_S,
        reference_frequency_hz=FC_HZ,
        carrier_hz=0.0,
        carrier_rate_hz=FC_HZ,
    )
    fields.update(overrides)
    return FmcwBeatSpec(**fields)


def _frozen_channel_weight() -> complex:
    phase = 2.0 * math.pi * FC_HZ * TAU_RT_S
    return complex(math.cos(phase), math.sin(phase))


def _two_transmitters(weight: complex, rate: float = TAU_RATE):
    """One row per sensor pair, same target, 2 TX x 1 RX.

    Under the composed pair numbering (``pair = rx * num_tx + tx``) with one
    receiver, pair ``p`` is transmitter ``p``. The two rows are physically
    identical, so any difference between the two pairs in the output is slot
    time and nothing else.
    """

    tau = torch.tensor([TAU_RT_S, TAU_RT_S], dtype=torch.float32, device="cuda")
    tau_rate = torch.tensor([rate, rate], dtype=torch.float32, device="cuda")
    w = torch.tensor([weight, weight], dtype=torch.complex64, device="cuda")
    offsets = torch.tensor([0, 1, 2], dtype=torch.int64, device="cuda")
    tx_index = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    return tau, tau_rate, w, offsets, tx_index


def _slot_phase_difference(cube: torch.Tensor, sample: int) -> float:
    """Mean phase of pair 1 relative to pair 0, at one fast-time sample."""

    later = cube[:, 1, sample].to(torch.complex128)
    earlier = cube[:, 0, sample].to(torch.complex128)
    return float(torch.angle(later * torch.conj(earlier)).mean())


def _analytic_one_slot_walk(sample: int) -> float:
    """One chirp period of delay walk, in radians.

    Identical in form to the per-chirp slope of the single-transmitter case,
    because a TDM slot IS one chirp period: transmitter 1 fires exactly
    ``Tc`` after transmitter 0 within the same chirp index.
    """

    t_m = sample / SAMPLE_RATE_HZ
    ramp = SLOPE_HZ_PER_S * (T_START_S - TAU_RT_S + t_m)
    return 2.0 * math.pi * TAU_RATE * CHIRP_PERIOD_S * (FC_HZ + ramp)


# --------------------------------------------------------------------------
# T1.6  the slot phase
# --------------------------------------------------------------------------


@pytest.mark.parametrize("sample", [0, NUM_SAMPLES - 1])
def test_a_second_transmitter_is_exactly_one_chirp_period_later(sample):
    """The whole content of TDM slow time, in one number.

    Two physically identical rows differing only in which transmitter drives
    them must differ in phase by exactly one chirp period of delay walk. A
    kernel that kept using the chirp index would put them at the same slow time
    and the difference would be zero, which is also what a stationary target
    looks like - so the non-vacuity assertion below is not decoration.
    """

    spec = _spec(num_tx=2, num_rx=1)
    tau, rate, weight, offsets, tx_index = _two_transmitters(
        _frozen_channel_weight()
    )
    cube = synthesize_beat_rows(
        tau, rate, weight, offsets, spec, segment_tx_index=tx_index
    ).cpu()

    measured = _slot_phase_difference(cube, sample)
    analytic = _analytic_one_slot_walk(sample)
    assert measured == pytest.approx(analytic, rel=1e-5)
    assert abs(analytic) > 1.0  # the two pairs are genuinely far apart

    # A stationary target has no slot phase at all, which is what makes the
    # measurement above attributable to motion rather than to the table.
    still = synthesize_beat_rows(
        tau,
        torch.zeros_like(rate),
        weight,
        offsets,
        spec,
        segment_tx_index=tx_index,
    ).cpu()
    assert abs(_slot_phase_difference(still, sample)) < 1e-6


def test_the_transmitter_table_is_what_decides_the_slot_not_the_pair_rank():
    """Relabelling the transmitters relabels the phase, and nothing else.

    ``pair = rx * num_tx + tx`` is sink major, so the transmitter index is
    ``pair % num_tx`` and NOT ``pair // num_rx``. Feeding the reversed table
    must exchange the two pairs' outputs exactly; if the kernel derived the slot
    from the segment rank instead of reading the table, the reversal would
    change nothing.
    """

    spec = _spec(num_tx=2, num_rx=1)
    tau, rate, weight, offsets, tx_index = _two_transmitters(
        _frozen_channel_weight()
    )
    forward = synthesize_beat_rows(
        tau, rate, weight, offsets, spec, segment_tx_index=tx_index
    )
    reversed_table = torch.tensor([1, 0], dtype=torch.int32, device="cuda")
    swapped = synthesize_beat_rows(
        tau, rate, weight, offsets, spec, segment_tx_index=reversed_table
    )
    assert torch.equal(forward[:, 0, :], swapped[:, 1, :])
    assert torch.equal(forward[:, 1, :], swapped[:, 0, :])
    assert not torch.equal(forward, swapped)


@pytest.mark.parametrize("sample", [0, NUM_SAMPLES - 1])
def test_the_sigproc_tdm_compensation_removes_exactly_the_carrier_slot_phase(
    sample,
):
    """Cross-check against the consumer that has to undo this phase.

    ``sigproc/pointcloud.py::_compensate_tdm_phase`` multiplies virtual antenna
    block ``tx_i`` by ``exp(-j 4 pi v tx_i Tc / lambda)``. Written in the
    contract's own variables that is ``exp(-j 2 pi fc tau_rate tx_i Tc)``, i.e.
    exactly the CARRIER part of one slot's walk. It does not know about the ramp
    term, so on a real chirp it leaves a residual, and the residual is asserted
    here rather than tolerated: it is 1/216 of the total at sample 0 and 1/24 at
    the last sample, which is small but is not zero and is not noise.

    Both the magnitude and the SIGN are pinned. A compensation with the wrong
    sign doubles the very phase it is supposed to remove.
    """

    spec = _spec(num_tx=2, num_rx=1)
    tau, rate, weight, offsets, tx_index = _two_transmitters(
        _frozen_channel_weight()
    )
    cube = synthesize_beat_rows(
        tau, rate, weight, offsets, spec, segment_tx_index=tx_index
    ).cpu()

    wavelength = C0 / FC_HZ
    radial_speed = C0 * TAU_RATE / 2.0
    compensation = 4.0 * math.pi * radial_speed * 1 * CHIRP_PERIOD_S / wavelength
    assert compensation == pytest.approx(
        2.0 * math.pi * FC_HZ * TAU_RATE * CHIRP_PERIOD_S, rel=1e-12
    )

    corrected = cube[:, 1, sample].to(torch.complex128) * complex(
        math.cos(-compensation), math.sin(-compensation)
    )
    residual = float(
        torch.angle(corrected * torch.conj(cube[:, 0, sample].to(torch.complex128)))
        .mean()
    )
    t_m = sample / SAMPLE_RATE_HZ
    ramp_residual = (
        2.0
        * math.pi
        * TAU_RATE
        * CHIRP_PERIOD_S
        * SLOPE_HZ_PER_S
        * (T_START_S - TAU_RT_S + t_m)
    )
    assert residual == pytest.approx(ramp_residual, rel=2e-3)
    assert abs(residual) < abs(_slot_phase_difference(cube, sample)) / 20.0


def test_a_pure_carrier_slot_phase_is_compensated_to_nothing():
    """With no ramp, the sigproc compensation is exact - down to 1e-4 rad.

    Isolating the carrier is what makes the residual above attributable to the
    ramp rather than to a sign or a factor-of-two error in either side.
    """

    spec = _spec(num_samples=4, slope_hz_per_s=0.0, t_start_s=0.0, num_tx=2, num_rx=1)
    tau, rate, weight, offsets, tx_index = _two_transmitters(
        _frozen_channel_weight()
    )
    cube = synthesize_beat_rows(
        tau, rate, weight, offsets, spec, segment_tx_index=tx_index
    ).cpu()

    raw = _slot_phase_difference(cube, 0)
    assert raw == pytest.approx(
        2.0 * math.pi * FC_HZ * TAU_RATE * CHIRP_PERIOD_S, rel=1e-6
    )

    wavelength = C0 / FC_HZ
    radial_speed = C0 * TAU_RATE / 2.0
    compensation = 4.0 * math.pi * radial_speed * CHIRP_PERIOD_S / wavelength
    corrected = cube[:, 1, 0].to(torch.complex128) * complex(
        math.cos(-compensation), math.sin(-compensation)
    )
    residual = float(
        torch.angle(corrected * torch.conj(cube[:, 0, 0].to(torch.complex128))).mean()
    )
    assert abs(residual) < 1e-4


# --------------------------------------------------------------------------
# T1.6b  the same statement at a GENERAL delay rate and every transmitter
# --------------------------------------------------------------------------


#: A round-trip delay rate that is NOT ``2 v / c`` for any tidy monostatic ``v``.
#:
#: Measured by the Phase-7 kinematics seam on the multi-endpoint fixture, for
#: the genuinely bistatic round trip ``TX_A -> SITE_P -> RX_B`` with both sites,
#: the transmitter and both receivers carrying different velocities. Quoted here
#: rather than recomputed so that this file keeps no Channel dependency; what it
#: buys is that the identity below is exercised at a rate no monostatic
#: derivation could have produced.
TAU_RATE_BISTATIC = -3.417386e-08


def _n_transmitters(weight: complex, rate: float, num_tx: int):
    """One row per sensor pair, same target, ``num_tx`` TX x 1 RX.

    With one receiver the composed pair numbering ``pair = rx * num_tx + tx``
    makes pair ``p`` transmitter ``p``, so the whole difference between the
    output pairs is slot time.
    """

    tau = torch.full((num_tx,), TAU_RT_S, dtype=torch.float32, device="cuda")
    tau_rate = torch.full((num_tx,), rate, dtype=torch.float32, device="cuda")
    w = torch.full((num_tx,), weight, dtype=torch.complex64, device="cuda")
    offsets = torch.arange(num_tx + 1, dtype=torch.int64, device="cuda")
    tx_index = torch.arange(num_tx, dtype=torch.int32, device="cuda")
    return tau, tau_rate, w, offsets, tx_index


def _wrapped(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


@pytest.mark.parametrize("num_tx", [2, 3, 4])
def test_tdm_phase_matches_downstream_compensation(num_tx):
    """``2 pi f_ref tau_rate Tc tx_i`` IS ``4 pi v tx_i Tc / lambda`` at
    ``tau_rate = 2 v / c``, for every transmitter and at a general rate.

    The pin above checks ONE transmitter at a rate that was constructed as
    ``2 v / c`` for a round radial speed, so it cannot distinguish the identity
    from the construction. Two things are generalised here:

    * every ``tx_i``, not only ``tx_i = 1``, because the compensation is linear
      in the virtual-antenna block index and an off-by-one there is exactly the
      kind of error that leaves block 1 correct;
    * a bistatic ``tau_rate`` that no monostatic ``v`` produced. ``sigproc``
      speaks in radial velocity because that is what a Doppler bin reports; the
      kernel speaks in ``tau_rate`` because a bistatic round trip has no single
      radial velocity. ``v = c tau_rate / 2`` is the whole translation between
      the two vocabularies and it has to hold for a rate that was never a
      monostatic ``v`` in the first place.

    The ramp is switched off so the compensation is exact rather than leaving
    the ramp residual the parametrised test above measures.
    """

    spec = _spec(
        num_samples=4,
        slope_hz_per_s=0.0,
        t_start_s=0.0,
        num_tx=num_tx,
        num_rx=1,
    )
    tau, rate, weight, offsets, tx_index = _n_transmitters(
        _frozen_channel_weight(), TAU_RATE_BISTATIC, num_tx
    )
    cube = synthesize_beat_rows(
        tau, rate, weight, offsets, spec, segment_tx_index=tx_index
    ).cpu()

    wavelength = C0 / FC_HZ
    radial_speed = C0 * TAU_RATE_BISTATIC / 2.0
    reference = cube[:, 0, 0].to(torch.complex128)
    walks = []

    for tx_i in range(num_tx):
        contract_phase = (
            2.0 * math.pi * FC_HZ * TAU_RATE_BISTATIC * CHIRP_PERIOD_S * tx_i
        )
        sigproc_phase = (
            4.0 * math.pi * radial_speed * tx_i * CHIRP_PERIOD_S / wavelength
        )
        assert abs(sigproc_phase - contract_phase) < 1e-4

        measured = float(
            torch.angle(
                cube[:, tx_i, 0].to(torch.complex128) * torch.conj(reference)
            ).mean()
        )
        walks.append(abs(measured))
        assert abs(_wrapped(measured - contract_phase)) < 1e-4

        # And the compensation removes it: the corrected block lands on top of
        # block 0 with nothing left over.
        corrected = cube[:, tx_i, 0].to(torch.complex128) * complex(
            math.cos(-sigproc_phase), math.sin(-sigproc_phase)
        )
        residual = float(torch.angle(corrected * torch.conj(reference)).mean())
        assert abs(residual) < 1e-4

    # Non-vacuous: the blocks really are far apart before compensation, and the
    # walk grows with the block index rather than being one constant offset.
    assert walks[0] < 1e-9
    assert all(
        later > earlier + 0.5
        for earlier, later in zip(walks[:-1], walks[1:], strict=True)
    )


# --------------------------------------------------------------------------
# T1.7  the same statement through the PRODUCTION slot table
# --------------------------------------------------------------------------


def _identical_row_per_pair(*, num_tx: int, num_rx: int, rate: float):
    """One physically identical row per sensor pair of a ``num_tx x num_rx``
    array, as the batch ``synthesize_fmcw_beat`` consumes.

    Every row carries the same delay, the same rate and the same weight, so any
    difference between two pairs in the synthesized cube is slot time and
    nothing else - and which slot a pair sits in is decided entirely by
    ``assembly.pair_tx_index``, which this route calls rather than being told.

    ``num_rx > 1`` is load bearing. With one receiver the sink-major rank makes
    pair ``p`` transmitter ``p``, so ``pair % num_tx`` is the identity and every
    hand-built table in this file happens to agree with the production one; the
    table's arithmetic only becomes observable when the pair axis is longer
    than the transmitter axis.
    """

    from witwin.radar.paths.contracts import RadarPathTopology
    from witwin.radar.synthesis.contracts import SlowTimeMode, SynthesisPathBatch

    pairs = num_tx * num_rx
    rows = torch.arange(pairs, dtype=torch.int64, device="cuda")
    weight = _frozen_channel_weight()
    return SynthesisPathBatch(
        sensor_pair_count=pairs,
        path_count=pairs,
        sensor_pair_index=rows,
        pair_offsets=torch.arange(pairs + 1, dtype=torch.int64, device="cuda"),
        total_delay_s=torch.full(
            (pairs,), TAU_RT_S, dtype=torch.float32, device="cuda"
        ),
        delay_rate=torch.full(
            (pairs,), rate, dtype=torch.float32, device="cuda"
        ),
        complex_transfer_ref=torch.full(
            (pairs,), weight, dtype=torch.complex64, device="cuda"
        ),
        reference_frequency_hz=FC_HZ,
        frequency_response=None,
        frequency_offsets_hz=None,
        topology=RadarPathTopology(
            radar_source_id=torch.remainder(rows, num_tx),
            site_id=torch.zeros(pairs, dtype=torch.int64, device="cuda"),
            radar_sink_id=torch.div(rows, num_tx, rounding_mode="floor"),
            inbound_row=rows,
            outbound_row=rows,
        ),
        row_valid=None,
        join_mode="multipath",
        weight_includes_reference_phase=True,
        weight_includes_spreading=True,
        weight_includes_tx_power=True,
        slow_time_mode=SlowTimeMode.FROZEN_WEIGHT_WITH_CARRIER_RATE,
    )


@pytest.mark.parametrize("num_tx", [2, 4])
def test_the_production_slot_table_survives_the_downstream_compensation(num_tx):
    """The acceptance criterion, driven by the table production actually uses.

    Every other TDM test in this file hands the kernel a table it built itself
    (``torch.tensor([0, 1])``, ``torch.arange(num_tx)``), so they pin the
    kernel's phase LAW and are blind to the table that feeds it in production.
    This one drives the whole chain instead - ``synthesize_fmcw_beat`` ->
    ``assembly.pair_tx_index`` -> ``segment_tx_index`` -> the beat kernel ->
    ``assemble_frame_cube`` -> ``processing.tdm_compensate`` - and asserts that
    the compensation lands every virtual-antenna block back on top of block 0. A
    slot table off by one transmitter passes every other test in this file and
    fails here, because the cube's TX axis comes from the sink-major layout
    while the phase in it came from the table.

    The compensation is called for real. Only the scalars it reads off the two
    records are stubbed - the wavelength, the slot period and the array shape -
    because building a full ``ProcessingAxes`` would drag a whole processing
    configuration in to supply numbers this test already owns.

    The VELOCITY sign is negated at the call. The Phase-8 owner takes the
    canonical closing-positive velocity, and ``C0 * tau_rate / 2`` is its
    negative; the compensation is the same phase either way, and that negation
    is asserted bitwise against the pre-cutover golden in
    ``tests/processing/test_adapters.py``.

    The ramp is switched off so the compensation is exact; its residual on a
    real chirp is measured by
    ``test_the_sigproc_tdm_compensation_removes_exactly_the_carrier_slot_phase``.
    """

    from types import SimpleNamespace

    from witwin.radar.processing import ArrayGeometry, tdm_compensate
    from witwin.radar.synthesis.assembly import assemble_frame_cube
    from witwin.radar.synthesis.fmcw_beat import synthesize_fmcw_beat

    num_rx = 2
    num_chirps = 4
    spec = _spec(
        num_samples=4,
        num_chirps=num_chirps,
        slope_hz_per_s=0.0,
        t_start_s=0.0,
        num_tx=num_tx,
        num_rx=num_rx,
    )
    batch = _identical_row_per_pair(
        num_tx=num_tx, num_rx=num_rx, rate=TAU_RATE_BISTATIC
    )
    cube = synthesize_fmcw_beat(batch, spec)
    frame = assemble_frame_cube(cube, num_tx=num_tx, num_rx=num_rx)
    assert tuple(frame.shape) == (num_tx, num_rx, num_chirps, spec.num_samples)

    # The sigproc layout: virtual antenna va = tx * num_rx + rx, one column per
    # chirp, which is exactly what the AoA input carries per detection.
    aoa_input = frame[:, :, :, 0].reshape(num_tx * num_rx, num_chirps)
    radial_speed = C0 * TAU_RATE_BISTATIC / 2.0
    velocities = torch.full(
        (num_chirps,), radial_speed, dtype=torch.float64, device=frame.device
    )
    array = ArrayGeometry.from_offsets(
        [[float(index), 0.0, 0.0] for index in range(num_tx)],
        [[float(index), 0.0, 0.0] for index in range(num_rx)],
        element_spacing_m=(C0 / FC_HZ) / 2.0,
        wavelength_m=C0 / FC_HZ,
        phase_sign=1,
        device=frame.device,
    )
    axes = SimpleNamespace(
        slow_time_period_s=CHIRP_PERIOD_S * num_tx, num_tx=num_tx
    )

    raw = aoa_input.to(torch.complex128).cpu()
    compensated = (
        tdm_compensate(aoa_input, -velocities, array, axes)
        .to(torch.complex128)
        .cpu()
    )

    walks = []
    for tx_i in range(num_tx):
        for rx_i in range(num_rx):
            va = tx_i * num_rx + rx_i
            reference = raw[rx_i]  # va of (tx=0, rx=rx_i)
            walks.append(
                abs(float(torch.angle(raw[va] * torch.conj(reference)).mean()))
            )
            residual = float(
                torch.angle(compensated[va] * torch.conj(reference)).mean()
            )
            assert abs(residual) < 1e-4, (tx_i, rx_i, residual)

    # Non-vacuity: the blocks really were far apart before compensation, and the
    # walk grows one slot per transmitter rather than being one constant offset.
    per_tx = [walks[tx_i * num_rx] for tx_i in range(num_tx)]
    assert per_tx[0] < 1e-9
    assert all(
        later > earlier + 0.5
        for earlier, later in zip(per_tx[:-1], per_tx[1:], strict=True)
    )


# --------------------------------------------------------------------------
# T1.8  the num_tx = 1 compatibility pin
# --------------------------------------------------------------------------


def _single_row(spec_period: float = CHIRP_PERIOD_S):
    tau = torch.tensor([TAU_RT_S], dtype=torch.float32, device="cuda")
    rate = torch.tensor([TAU_RATE], dtype=torch.float32, device="cuda")
    weight = torch.tensor(
        [_frozen_channel_weight()], dtype=torch.complex64, device="cuda"
    )
    offsets = torch.tensor([0, 1], dtype=torch.int64, device="cuda")
    return tau, rate, weight, offsets


def test_a_single_transmitter_makes_the_slot_the_chirp_index_exactly():
    """``(c * 1 + 0) * Tc == c * Tc``, bit for bit.

    The cross-revision half of this pin - the pre-TDM binary against the
    post-TDM one at ``num_tx = 1`` - cannot live in the tree, because the
    pre-TDM binary is not rebuildable from this source. It was captured and is
    reported with the stage's evidence: primal and all four gradients came back
    ``np.array_equal`` True, maximum absolute difference 0.0. What IS pinnable
    here is the property that made the equality possible, in both directions.
    """

    spec = _spec(num_chirps=6)
    tau, rate, weight, offsets = _single_row()
    assert spec.num_tx == 1

    implicit = synthesize_beat_rows(tau, rate, weight, offsets, spec)
    explicit = synthesize_beat_rows(
        tau,
        rate,
        weight,
        offsets,
        spec,
        segment_tx_index=torch.zeros(1, dtype=torch.int32, device="cuda"),
    )
    assert torch.equal(implicit, explicit)


def test_the_slot_is_num_tx_chirp_periods_wide():
    """``num_tx = 2`` at ``tx = 0`` samples slow time every ``2 Tc``.

    That is the whole arithmetic content of the slot, and it is checkable
    without any pre-TDM binary: the same rows, with the transmitter count folded
    into the chirp period instead, must come out bit-identical. It also states
    the cost of TDM in the one place a reader will look for it - a transmitter
    revisits its slot ``num_tx`` times less often, which is exactly why
    ``max_unambiguous_speed_mps`` divides by ``num_tx``.
    """

    tau, rate, weight, offsets = _single_row()
    tdm = synthesize_beat_rows(
        tau,
        rate,
        weight,
        offsets,
        _spec(num_chirps=6, num_tx=2, num_rx=1),
        segment_tx_index=torch.zeros(1, dtype=torch.int32, device="cuda"),
    )
    widened = synthesize_beat_rows(
        tau,
        rate,
        weight,
        offsets,
        _spec(num_chirps=6, chirp_period_s=2.0 * CHIRP_PERIOD_S),
    )
    assert torch.equal(tdm, widened)


def test_a_multi_transmitter_spec_refuses_to_guess_the_slot_table():
    spec = _spec(num_tx=2, num_rx=2)
    tau, rate, weight, offsets = _single_row()
    with pytest.raises(ValueError, match="must name the transmitter"):
        synthesize_beat_rows(tau, rate, weight, offsets, spec)


def test_the_transmitter_table_must_span_the_segments():
    spec = _spec(num_tx=2, num_rx=1)
    tau, rate, weight, offsets, _ = _two_transmitters(_frozen_channel_weight())
    with pytest.raises(ValueError, match="one transmitter index per sensor-pair"):
        synthesize_beat_rows(
            tau,
            rate,
            weight,
            offsets,
            spec,
            segment_tx_index=torch.zeros(5, dtype=torch.int32, device="cuda"),
        )


# --------------------------------------------------------------------------
# T1.10  AD across the slot axis
# --------------------------------------------------------------------------


AD_SPEC = FmcwBeatSpec(
    num_samples=24,
    num_chirps=4,
    sample_period_s=1.0 / SAMPLE_RATE_HZ,
    chirp_period_s=CHIRP_PERIOD_S,
    slope_hz_per_s=SLOPE_HZ_PER_S,
    t_start_s=T_START_S,
    reference_frequency_hz=FC_HZ,
    carrier_hz=0.0,
    carrier_rate_hz=FC_HZ,
    num_tx=2,
    num_rx=1,
)
AD_DELAYS = (TAU_RT_S, 2.4e-8, 1.7e-8)
AD_RATES = (TAU_RATE, -1.1e-8, 4.0e-9)
AD_WEIGHTS = (0.6 - 0.3j, -0.2 + 0.45j, 0.15 + 0.8j)
# Two segments, unequal, so the backward kernel's per-path segment mapping
# actually has to choose a transmitter rather than always reading slot zero.
AD_OFFSETS = (0, 2, 3)
AD_TX_INDEX = (0, 1)


@pytest.fixture(scope="module")
def ad_target():
    torch.manual_seed(20260726)
    return torch.randn(
        (AD_SPEC.num_chirps, len(AD_OFFSETS) - 1, AD_SPEC.num_samples),
        dtype=torch.complex128,
    )


def _oracle_loss(tau, rate, weight, target):
    iq = ref.beat_samples(
        tau,
        rate,
        weight,
        torch.tensor(AD_OFFSETS, dtype=torch.int64),
        AD_SPEC,
        AD_TX_INDEX,
    )
    return ref.radar_loss(iq, target)


def _cuda_rows():
    return (
        torch.tensor(AD_DELAYS, dtype=torch.float32, device="cuda"),
        torch.tensor(AD_RATES, dtype=torch.float32, device="cuda"),
        torch.tensor(AD_WEIGHTS, dtype=torch.complex64, device="cuda"),
        torch.tensor(AD_OFFSETS, dtype=torch.int64, device="cuda"),
        torch.tensor(AD_TX_INDEX, dtype=torch.int32, device="cuda"),
    )


def _cpu_rows():
    return (
        torch.tensor(AD_DELAYS, dtype=torch.float64),
        torch.tensor(AD_RATES, dtype=torch.float64),
        torch.tensor(AD_WEIGHTS, dtype=torch.complex128),
    )


def test_the_float64_oracle_knows_about_slot_time():
    """The oracle has to be independently right before it can judge anything.

    ``beat_samples`` gained the slot term in the same change as the kernel, so
    the first thing to check is that it is the term the physics asks for and not
    a transcription of the kernel: two identical rows in two segments driven by
    different transmitters must differ by exactly one chirp period of delay
    walk, the same closed form the CUDA path is held to above.
    """

    tau = torch.tensor([TAU_RT_S, TAU_RT_S], dtype=torch.float64)
    rate = torch.tensor([TAU_RATE, TAU_RATE], dtype=torch.float64)
    weight = torch.tensor(
        [_frozen_channel_weight()] * 2, dtype=torch.complex128
    )
    offsets = torch.tensor([0, 1, 2], dtype=torch.int64)

    cube = ref.beat_samples(tau, rate, weight, offsets, AD_SPEC, (0, 1))
    single_tx = ref.beat_samples(tau, rate, weight, offsets, AD_SPEC, (0, 0))

    for sample in (0, AD_SPEC.num_samples - 1):
        walk = float(
            torch.angle(
                cube[:, 1, sample] * torch.conj(cube[:, 0, sample])
            ).mean()
        )
        t_m = sample * AD_SPEC.sample_period_s
        analytic = (
            2.0
            * math.pi
            * TAU_RATE
            * CHIRP_PERIOD_S
            * (FC_HZ + SLOPE_HZ_PER_S * (T_START_S - TAU_RT_S + t_m))
        )
        # 1e-6 rather than machine precision because the closed form is
        # evaluated at tau_rt while the measurement averages over slots, where
        # tau has already drifted by rate * slot * Tc. That is a real O(drift)
        # term of relative size 1.3e-8 here, not float64 noise.
        assert walk == pytest.approx(analytic, rel=1e-6)

        # And with both segments on transmitter 0 there is no walk at all, so
        # the number above is the transmitter table and not the segment rank.
        assert (
            abs(
                float(
                    torch.angle(
                        single_tx[:, 1, sample]
                        * torch.conj(single_tx[:, 0, sample])
                    ).mean()
                )
            )
            < 1e-12
        )


def test_native_vjp_matches_the_oracle_across_the_slot_axis(ad_target):
    """The backward kernel reads the transmitter of the segment each row is in.

    Every pre-TDM gradient test is single-transmitter, where the slot table is
    the constant zero and cannot be wrong. Here the two segments sit in
    different slots, so a backward that fell back on the chirp index would
    produce gradients that are wrong only for the second segment - which is the
    half of a MIMO frame nobody looks at first.
    """

    tau, rate, weight, offsets, tx_index = _cuda_rows()
    tau = tau.clone().requires_grad_(True)
    rate = rate.clone().requires_grad_(True)
    weight = weight.clone().requires_grad_(True)
    iq = synthesize_beat_rows(
        tau, rate, weight, offsets, AD_SPEC, segment_tx_index=tx_index
    )
    ref.radar_loss(iq.cpu(), ad_target).backward()

    o_tau, o_rate, o_weight = _cpu_rows()
    o_tau = o_tau.clone().requires_grad_(True)
    o_rate = o_rate.clone().requires_grad_(True)
    o_weight = o_weight.clone().requires_grad_(True)
    _oracle_loss(o_tau, o_rate, o_weight, ad_target).backward()

    for index in range(len(AD_DELAYS)):
        assert fd.relative_error(
            float(tau.grad[index]), float(o_tau.grad[index]), floor=1e-6
        ) < 2e-3
        assert fd.relative_error(
            float(rate.grad[index]), float(o_rate.grad[index]), floor=1e-6
        ) < 2e-3
        assert fd.relative_error(
            float(weight.grad[index].real),
            float(o_weight.grad[index].real),
            floor=1e-9,
        ) < 2e-3
        assert fd.relative_error(
            float(weight.grad[index].imag),
            float(o_weight.grad[index].imag),
            floor=1e-9,
        ) < 2e-3


def test_the_oracle_gradients_agree_with_float64_finite_differences(ad_target):
    """Validate the oracle itself, in float64, before trusting it above.

    Steps are chosen per variable and stated: a delay of order 1e-8 s needs
    1e-14 s to stay far from the float64 noise floor while keeping the
    truncation error of a central difference below the tolerance, and a
    dimensionless weight of order 1 needs 1e-6.
    """

    tau, rate, weight = _cpu_rows()
    tau = tau.clone().requires_grad_(True)
    rate = rate.clone().requires_grad_(True)
    weight = weight.clone().requires_grad_(True)
    _oracle_loss(tau, rate, weight, ad_target).backward()

    for index in range(len(AD_DELAYS)):
        measured_tau = fd.central_difference(
            lambda value: _oracle_loss(
                value, rate.detach(), weight.detach(), ad_target
            ),
            tau.detach(),
            index,
            1e-14,
        )
        assert fd.relative_error(
            measured_tau, float(tau.grad[index]), floor=1e-6
        ) < 1e-4

        measured_rate = fd.central_difference(
            lambda value: _oracle_loss(
                tau.detach(), value, weight.detach(), ad_target
            ),
            rate.detach(),
            index,
            1e-14,
        )
        assert fd.relative_error(
            measured_rate, float(rate.grad[index]), floor=1e-6
        ) < 1e-4

        measured_weight = fd.central_difference(
            lambda value: _oracle_loss(
                tau.detach(), rate.detach(), value, ad_target
            ),
            weight.detach(),
            index,
            1e-6,
        )
        assert fd.relative_error(
            measured_weight, float(weight.grad[index].real), floor=1e-9
        ) < 1e-5


def test_native_jvp_matches_a_central_difference_of_the_primal():
    """Forward mode across the slot axis, against the primal it differentiates.

    The dual is FORWARD ONLY - no ``requires_grad`` anywhere - which is the
    ADR-038 shape the facade must not swallow. The comparison is a central
    difference of the float64 oracle along the same direction, at a step chosen
    per variable and recorded in the code.
    """

    import torch.autograd.forward_ad as forward_ad

    tau, rate, weight, offsets, tx_index = _cuda_rows()
    directions = {
        "tau": (torch.tensor([1.0, -0.5, 0.25]), 1e-14),
        "rate": (torch.tensor([0.5, 1.0, -0.75]), 1e-14),
        "weight_re": (torch.tensor([1.0, 0.5, -0.25]), 1e-6),
        "weight_im": (torch.tensor([-0.5, 1.0, 0.75]), 1e-6),
    }
    o_tau, o_rate, o_weight = _cpu_rows()

    for name, (direction, step) in directions.items():
        with forward_ad.dual_level():
            dual_tau = tau
            dual_rate = rate
            dual_weight = weight
            tangent = direction.to(dtype=torch.float32, device="cuda")
            if name == "tau":
                dual_tau = forward_ad.make_dual(tau, tangent)
            elif name == "rate":
                dual_rate = forward_ad.make_dual(rate, tangent)
            elif name == "weight_re":
                dual_weight = forward_ad.make_dual(
                    weight, torch.complex(tangent, torch.zeros_like(tangent))
                )
            else:
                dual_weight = forward_ad.make_dual(
                    weight, torch.complex(torch.zeros_like(tangent), tangent)
                )
            assert not dual_tau.requires_grad
            iq = synthesize_beat_rows(
                dual_tau,
                dual_rate,
                dual_weight,
                offsets,
                AD_SPEC,
                segment_tx_index=tx_index,
            )
            jvp = forward_ad.unpack_dual(iq).tangent
            assert jvp is not None
            jvp = jvp.cpu().to(torch.complex128)

        def evaluate(scale: float, name=name, direction=direction):
            shift = direction.to(torch.float64) * scale
            moved_tau = o_tau + (shift if name == "tau" else 0.0)
            moved_rate = o_rate + (shift if name == "rate" else 0.0)
            moved_weight = o_weight
            if name == "weight_re":
                moved_weight = o_weight + shift.to(torch.complex128)
            elif name == "weight_im":
                moved_weight = o_weight + 1j * shift.to(torch.complex128)
            return ref.beat_samples(
                moved_tau,
                moved_rate,
                moved_weight,
                torch.tensor(AD_OFFSETS, dtype=torch.int64),
                AD_SPEC,
                AD_TX_INDEX,
            )

        expected = (evaluate(step) - evaluate(-step)) / (2.0 * step)
        scale = float(expected.abs().max())
        assert scale > 0.0
        torch.testing.assert_close(
            jvp, expected, rtol=2e-3, atol=2e-3 * scale, msg=lambda text: f"{name}: {text}"
        )


# --------------------------------------------------------------------------
# One real TDM frame, end to end, with its launch and host budget
# --------------------------------------------------------------------------


HOST_OBSERVERS = ("item", "cpu", "tolist", "numpy")


class _FrameLedger:
    """Count native launches and host observations while it is active."""

    def __init__(self, monkeypatch, operators) -> None:
        self.launches = dict.fromkeys(
            ("fmcw_beat_forward", "fmcw_beat_backward", "fmcw_beat_jvp"), 0
        )
        self.host = dict.fromkeys((*HOST_OBSERVERS, "synchronize"), 0)
        for name in self.launches:
            original = getattr(operators, name)

            def counting(*args, _name=name, _original=original, **kwargs):
                self.launches[_name] += 1
                return _original(*args, **kwargs)

            monkeypatch.setattr(operators, name, counting)
        for name in HOST_OBSERVERS:
            original_method = getattr(torch.Tensor, name)

            def observing(
                tensor, *args, _name=name, _original=original_method, **kwargs
            ):
                self.host[_name] += 1
                return _original(tensor, *args, **kwargs)

            monkeypatch.setattr(torch.Tensor, name, observing)
        original_sync = torch.cuda.synchronize

        def counting_sync(*args, **kwargs):
            self.host["synchronize"] += 1
            return original_sync(*args, **kwargs)

        monkeypatch.setattr(torch.cuda, "synchronize", counting_sync)


@pytest.fixture(scope="module")
def multi_endpoint_spike():
    pytest.importorskip("witwin.channel")
    from support import multi_endpoint_driver as drv

    return drv.MultiEndpointSpike()


def test_a_real_tdm_frame_assembles_into_the_sigproc_layout(multi_endpoint_spike):
    """2 TX x 2 RX, eleven composed rows, four pairs of which two are empty.

    The whole S1 path in one statement: a frozen multi-endpoint topology, one
    reevaluate, one composition, one synthesis launch, and one structural
    packing into the rank-4 cube ``sigproc`` consumes. The empty pairs survive
    the packing as empty CHANNELS rather than being renumbered away, which is
    the thing that would silently mis-steer an angle estimate.
    """

    from support import multi_endpoint_driver as drv
    from witwin.radar.synthesis.assembly import assemble_frame_cube
    from witwin.radar.synthesis.fmcw_beat import synthesize_fmcw_beat

    composed, _, _ = multi_endpoint_spike.frame()
    spec = drv.make_spec(num_chirps=2)
    assert (spec.num_tx, spec.num_rx) == (2, 2)
    assert composed.sensor_pair_count == spec.sensor_pair_count

    cube = synthesize_fmcw_beat(drv.to_synthesis(composed), spec)
    assert tuple(cube.shape) == (2, 4, spec.num_samples)

    frame = assemble_frame_cube(cube, num_tx=spec.num_tx, num_rx=spec.num_rx)
    assert tuple(frame.shape) == (2, 2, 2, spec.num_samples)

    # pair = rx * num_tx + tx, and transmitter B discovers nothing, so pairs 1
    # and 3 are the empty ones - which is (tx=1, rx=0) and (tx=1, rx=1).
    for rx in range(2):
        assert float(frame[0, rx].abs().sum()) > 0.0
        assert float(frame[1, rx].abs().sum()) == 0.0
        assert torch.equal(frame[0, rx], cube[:, rx * 2 + 0, :])
        assert torch.equal(frame[1, rx], cube[:, rx * 2 + 1, :])


def test_one_tdm_frame_is_one_launch_and_no_host_observation(
    multi_endpoint_spike, monkeypatch
):
    """Acceptance: exactly one ``fmcw_beat_forward`` per frame, and no D2H.

    TDM is a kernel ARGUMENT, so adding transmitters must not add launches; a
    per-transmitter pass would be a budget regression that no numerical test
    would notice. The host budget is measured over synthesis and assembly only,
    because the frame's two sanctioned ``.item()`` copies belong to the two
    frozen legs and are already pinned by ``test_phase5_budget``.
    """

    from support import multi_endpoint_driver as drv
    from witwin.radar.synthesis import fmcw_beat
    from witwin.radar.synthesis.assembly import assemble_frame_cube

    composed, _, _ = multi_endpoint_spike.frame()
    spec = drv.make_spec(num_chirps=2)
    batch = drv.to_synthesis(composed)
    operators = fmcw_beat._ops()

    ledger = _FrameLedger(monkeypatch, operators)
    cube = fmcw_beat.synthesize_fmcw_beat(batch, spec)
    frame = assemble_frame_cube(cube, num_tx=spec.num_tx, num_rx=spec.num_rx)
    assert frame.shape[0] == spec.num_tx

    assert ledger.launches == {
        "fmcw_beat_forward": 1,
        "fmcw_beat_backward": 0,
        "fmcw_beat_jvp": 0,
    }, ledger.launches
    assert ledger.host == dict.fromkeys((*HOST_OBSERVERS, "synchronize"), 0), (
        ledger.host
    )


def test_one_backward_launch_per_forward_launch(multi_endpoint_spike, monkeypatch):
    from support import multi_endpoint_driver as drv
    from witwin.radar.synthesis import fmcw_beat

    composed, _, _ = multi_endpoint_spike.frame(
        response=drv.make_response(requires_grad=True)
    )
    spec = drv.make_spec(num_chirps=2)
    batch = drv.to_synthesis(composed)
    operators = fmcw_beat._ops()

    ledger = _FrameLedger(monkeypatch, operators)
    cube = fmcw_beat.synthesize_fmcw_beat(batch, spec)
    (cube.real.sum() + cube.imag.sum()).backward()
    assert ledger.launches["fmcw_beat_forward"] == 1
    assert ledger.launches["fmcw_beat_backward"] == 1
    assert ledger.launches["fmcw_beat_jvp"] == 0
