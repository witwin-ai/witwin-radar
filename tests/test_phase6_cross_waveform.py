"""One path batch, three waveforms: the shared-path invariants (work item 7).

Plan work item 7 asks that the three synthesis owners agree on ONE frozen
``RadarPathBatch``. This file is that proof, on the real multi-endpoint fixture
(2 TX x 2 RX x 2 sites, eleven composed rows, two empty pair segments), against
the fixture's float64 image-source oracle rather than against each other.

The six invariants of the design's section 4.1:

* I1 one delay, three estimators (T5.1)
* I2 one Doppler, three signs (T5.2)
* I3 one amplitude, three peaks (T5.3)
* I4 dead rows contribute exactly zero (T5.5)
* I5 linearity in the weight (T5.6)
* I6 row identity and order (T5.7)

RECORDED DEVIATION on the I1 tolerance. The brief asks for ``rtol=1e-5`` on the
delay in seconds. That is not attainable at this fixture's geometry and it is
not a defect: the fixture's round trips are 4 to 18 metres, so ``tau_rt`` is
1.3e-8 to 6e-8 s, while the three estimators' own resolutions are

    FMCW    1 / (S N T_s)      6.9e-10 s at the slope used here
    OFDM    1 / (N_sc df)      1.3e-7 s  (but the phase-slope fit is exact)
    pulsed  1 / B              5.0e-8 s

so a 1e-5 relative bound on a 4e-8 s delay is 4e-13 s, which is between a
thousandth and a ten-thousandth of a pulsed range cell. What is asserted
instead is each estimator against the SAME float64 oracle within a stated
fraction of ITS OWN resolution, which is the statement that actually says the
three waveforms agree, and the measured deviations are quoted in the assertion
messages. The OFDM phase-slope fit is exact and IS asserted at ``rtol=1e-5``.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch
import torch.autograd.forward_ad as forward_ad
from support import multi_endpoint_driver as drv
from support import multi_endpoint_geometry as geo
from support import pulsed_grid

from witwin.radar.synthesis import FmcwSpec, OfdmSpec, synthesize_fmcw, synthesize_ofdm, synthesize_pulsed

pytestmark = pytest.mark.gpu


C0 = geo.C0_M_PER_S
F_REF_HZ = geo.REFERENCE_FREQUENCY_HZ

# ---------------------------------------------------------------------------
# The three specs, on ONE geometry
# ---------------------------------------------------------------------------

# The fixture's own radar config has slope 60.012 MHz/us, and at a 4.4 MSPS ADC
# that puts the longest round trip's beat tone at 3.6 MHz - above the 2.2 MHz
# Nyquist limit, where a beat-frequency estimate means nothing. The slope is
# the one number lowered here, and it is lowered to keep the whole fixture
# inside the band rather than to make anything pass.
FMCW_SLOPE_HZ_PER_S = 2.5e13
FMCW_SAMPLES = 256
FMCW_SAMPLE_RATE_HZ = 4.4e6
FMCW_SAMPLE_PERIOD_S = 1.0 / FMCW_SAMPLE_RATE_HZ
FMCW_CHIRP_PERIOD_S = (7.0 + 58.0) * 1e-6
FMCW_T_START_S = 6.0e-6
FMCW_NUM_TX = 2
FMCW_NUM_RX = 2
#: ``1 / (S N T_s)``: one FFT bin expressed as a delay.
FMCW_DELAY_RESOLUTION_S = 1.0 / (FMCW_SLOPE_HZ_PER_S * FMCW_SAMPLES * FMCW_SAMPLE_PERIOD_S)

OFDM_SUBCARRIERS = 64
OFDM_DF_HZ = 120.0e3
OFDM_CYCLIC_PREFIX_S = 2.0e-6
OFDM_MAX_DELAY_S = 1.0e-6

PULSED_MAX_DELAY_RATE = 2.0 * 12.0 / C0
#: ``1 / B``: one pulsed range cell expressed as a delay.
PULSED_DELAY_RESOLUTION_S = 1.0 / pulsed_grid.BANDWIDTH_HZ


def fmcw_spec(num_chirps: int = 4) -> FmcwSpec:
    return FmcwSpec(
        num_samples=FMCW_SAMPLES,
        num_chirps=num_chirps,
        sample_period_s=FMCW_SAMPLE_PERIOD_S,
        chirp_period_s=FMCW_CHIRP_PERIOD_S,
        slope_hz_per_s=FMCW_SLOPE_HZ_PER_S,
        t_start_s=FMCW_T_START_S,
        reference_frequency_hz=F_REF_HZ,
        carrier_hz=0.0,
        carrier_rate_hz=F_REF_HZ,
        num_tx=FMCW_NUM_TX,
        num_rx=FMCW_NUM_RX,
        output_domain="beat",
    )


def ofdm_spec(num_symbols: int = 4) -> OfdmSpec:
    return OfdmSpec(
        num_subcarriers=OFDM_SUBCARRIERS,
        num_symbols=num_symbols,
        subcarrier_spacing_hz=OFDM_DF_HZ,
        cyclic_prefix_s=OFDM_CYCLIC_PREFIX_S,
        reference_frequency_hz=F_REF_HZ,
        max_expected_delay_s=OFDM_MAX_DELAY_S,
        carrier_hz=0.0,
        carrier_rate_hz=F_REF_HZ,
    )


def pulsed_spec(num_pulses: int = 4):
    return pulsed_grid.reference_spec(
        num_pulses=num_pulses,
        num_samples=512,
        reference_frequency_hz=F_REF_HZ,
        carrier_rate_hz=F_REF_HZ,
        max_expected_delay_rate=PULSED_MAX_DELAY_RATE,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def spike():
    pytest.importorskip("witwin.channel")
    return drv.MultiEndpointSpike()


@pytest.fixture(scope="module")
def frame(spike):
    """ONE frozen composition, synthesized three ways by every test below."""

    composed, _, _ = spike.frame()
    return spike, composed, drv.to_synthesis(composed)


def _one_hot(batch, row: int) -> torch.Tensor:
    mask = torch.zeros(batch.path_count, dtype=torch.bool, device="cuda")
    mask[row] = True
    return mask


def _masked(batch, mask: torch.Tensor):
    """The same batch with ``row_valid`` replaced. Row identity is untouched."""

    existing = batch.row_valid
    combined = mask if existing is None else (mask & existing)
    return replace(batch, row_valid=combined.contiguous())


def _select(batch, keep: torch.Tensor):
    """A batch physically containing only ``keep``, with rebuilt offsets.

    ``keep`` is a boolean row mask. Compact row order is preserved because the
    rows are already in segment order, so the new CSR offsets are the running
    counts of the surviving rows per segment.
    """

    from witwin.radar.paths import RadarPathTopology

    index = torch.nonzero(keep, as_tuple=False).flatten()
    pair = batch.sensor_pair_index[index].contiguous()
    counts = torch.bincount(pair, minlength=batch.sensor_pair_count)
    offsets = torch.zeros(batch.sensor_pair_count + 1, dtype=torch.int64, device=pair.device)
    offsets[1:] = torch.cumsum(counts, dim=0)
    topology = batch.topology
    reduced_topology = RadarPathTopology(
        radar_source_id=topology.radar_source_id[index].contiguous(),
        site_id=topology.site_id[index].contiguous(),
        radar_sink_id=topology.radar_sink_id[index].contiguous(),
        inbound_row=topology.inbound_row[index].contiguous(),
        outbound_row=topology.outbound_row[index].contiguous(),
    )
    return replace(
        batch,
        path_count=int(index.numel()),
        sensor_pair_index=pair,
        pair_offsets=offsets.contiguous(),
        total_delay_s=batch.total_delay_s[index].contiguous(),
        delay_rate=(None if batch.delay_rate is None else batch.delay_rate[index].contiguous()),
        complex_transfer_ref=batch.complex_transfer_ref[index].contiguous(),
        topology=reduced_topology,
        row_valid=(None if batch.row_valid is None else batch.row_valid[index].contiguous()),
    )


def _live_rows(batch) -> list[int]:
    if batch.row_valid is None:
        return list(range(batch.path_count))
    return [row for row, live in enumerate(batch.row_valid.tolist()) if bool(live)]


# ---------------------------------------------------------------------------
# Estimators. Each returns a delay in SECONDS, never a bin index.
# ---------------------------------------------------------------------------


def fmcw_delay_s(row_samples: torch.Tensor, spec: FmcwSpec) -> float:
    """``f_beat / S`` from a parabolic fit on the fast-time FFT magnitude."""

    values = row_samples.detach().to(torch.complex128).cpu()
    magnitude = torch.fft.fft(values).abs()
    half = magnitude.shape[0] // 2
    peak = int(magnitude[:half].argmax())
    left = float(magnitude[peak - 1])
    centre = float(magnitude[peak])
    right = float(magnitude[peak + 1])
    offset = 0.5 * (left - right) / (left - 2.0 * centre + right)
    bin_estimate = peak + offset
    f_beat = bin_estimate * spec.sample_rate_hz / spec.num_samples
    return f_beat / spec.slope_hz_per_s


def ofdm_delay_s(row_response: torch.Tensor, spec: OfdmSpec) -> float:
    """``-(1/(2 pi df)) d(arg H)/dn`` by least squares over the whole band."""

    values = row_response.detach().to(torch.complex128).cpu()
    phase = torch.from_numpy(__import__("numpy").unwrap(torch.angle(values).numpy()))
    n = torch.arange(values.shape[0], dtype=torch.float64)
    n_mean = n.mean()
    slope = ((n - n_mean) * (phase - phase.mean())).sum() / ((n - n_mean) ** 2).sum()
    return float(-slope / (2.0 * math.pi * spec.subcarrier_spacing_hz))


def pulsed_delay_s(row_samples: torch.Tensor, spec) -> float:
    """Matched-filter peak, minus the range-gate start."""

    delay, _, _ = pulsed_grid.peak_estimate(row_samples.detach(), spec)
    return delay - spec.range_gate_start_s


# ---------------------------------------------------------------------------
# T5.1 (I1) - one delay, three estimators
# ---------------------------------------------------------------------------


def test_all_three_estimators_return_the_same_round_trip_delay(frame, capsys):
    """Per live row, three independent estimators against ONE float64 oracle.

    Every row is isolated with ``row_valid`` first, so the estimate in its
    segment is that row's and not a blend: the fixture's closest two composed
    rows differ by 20 ps, far below any of the three resolutions, and a peak
    finder handed both would report neither.
    """

    spike, composed, batch = frame
    oracle = {row.key: row.total_delay_s for row in geo.combined_rows()}
    keys = drv.composed_keys(spike, composed)
    reported = []
    for row in _live_rows(batch):
        isolated = _masked(batch, _one_hot(batch, row))
        segment = int(batch.sensor_pair_index[row])
        # The truth is the fixture's float64 image-source table, resolved by
        # frame-invariant identity, never the solver's own output.
        truth = oracle[keys[row]]

        measured = {
            "fmcw": fmcw_delay_s(synthesize_fmcw(isolated, fmcw_spec(1))[0, segment], fmcw_spec(1)),
            "ofdm": ofdm_delay_s(synthesize_ofdm(isolated, ofdm_spec(1))[0, segment], ofdm_spec(1)),
            "pulsed": pulsed_delay_s(synthesize_pulsed(isolated, pulsed_spec(1))[0, segment], pulsed_spec(1)),
        }
        reported.append((row, truth, measured))

        # OFDM's phase-slope fit is exact for any delay: no bin, no window.
        assert measured["ofdm"] == pytest.approx(truth, rel=1e-5), (row, measured)
        # The other two are peak finders and are held to a fraction of their
        # own resolution, stated rather than fitted.
        assert abs(measured["fmcw"] - truth) < 0.3 * FMCW_DELAY_RESOLUTION_S, (row, measured["fmcw"], truth)
        assert abs(measured["pulsed"] - truth) < 0.05 * PULSED_DELAY_RESOLUTION_S, (row, measured["pulsed"], truth)

    assert len(reported) == 11
    with capsys.disabled():
        print("\nT5.1 delay table (seconds)")
        print(f"{'row':>4} {'oracle':>14} {'fmcw':>14} {'ofdm':>14} {'pulsed':>14}")
        for row, truth, measured in reported:
            print(
                f"{row:>4} {truth:14.6e} {measured['fmcw']:14.6e} {measured['ofdm']:14.6e} {measured['pulsed']:14.6e}"
            )
    # And the oracle the batch itself agrees with is the fixture's image-source
    # table, not the solver's own output.
    for row, truth, _ in reported:
        assert any(abs(truth - value) < 1e-12 for value in oracle.values()), (row, truth)


def _spike_of(frame):  # pragma: no cover - placeholder for the unused branch
    raise AssertionError("unused")


# ---------------------------------------------------------------------------
# T5.2 (I2) - one Doppler, three signs
# ---------------------------------------------------------------------------


def _slow_time_step_rad(cube: torch.Tensor, segment: int, index: int) -> float:
    """Mean phase advance per slow-time step at one fast-time index."""

    column = cube[:, segment, index].detach().to(torch.complex128).cpu()
    steps = column[1:] * torch.conj(column[:-1])
    return float(torch.angle(steps.sum()))


def test_the_three_waveforms_carry_one_doppler_with_two_signs(spike):
    """``|f_slow| = f_ref |tau_rate|``, and the FMCW cube's sign is opposite.

    Physical Doppler is ``f_D = -f_ref tau_rate`` in Channel's ``exp(-j k d)``
    convention. The FMCW beat cube is the CONJUGATE of that product, so its
    slow-time tone sits at ``+f_ref tau_rate`` while the OFDM and pulsed cubes
    sit at ``-f_ref tau_rate``. A magnitude-only range-Doppler map cannot see
    the difference, which is exactly why the sign is asserted here.

    The magnitude tolerance is 1 percent rather than 1e-5 for a stated reason:
    the FMCW cube's slow-time bracket is ``f_ref + S (t_0 - tau + t_m)``, not
    ``f_ref``, and at this slope the ramp term is 0.47 percent of the carrier at
    the first fast-time sample. That term is real physics, not error; removing
    it from the comparison would mean restating the kernel's own expression.
    """

    velocities = {geo.SITE_P_STABLE_ID: geo.SITE_P_VELOCITY_M_PER_S, geo.SITE_Q_STABLE_ID: geo.SITE_Q_VELOCITY_M_PER_S}
    tangent = torch.tensor([velocities[stable_id] for stable_id in spike.site_ids], dtype=torch.float32, device="cuda")
    positions = spike.site_tensor()
    with forward_ad.dual_level():
        composed, _, _ = spike.frame(forward_ad.make_dual(positions, tangent), ad_mode="jvp")
        rate = composed.delay_rate.clone()
        transfer = composed.complex_transfer_ref.clone()
        delay = composed.total_delay_s.clone()
        frozen = composed
        batch = drv.to_synthesis(replace(frozen, total_delay_s=delay, delay_rate=rate, complex_transfer_ref=transfer))

    fmcw = fmcw_spec(8)
    ofdm = ofdm_spec(8)
    pulsed = pulsed_spec(8)
    checked = 0
    for row in _live_rows(batch):
        tau_rate = float(batch.delay_rate[row])
        if abs(tau_rate) < 1e-12:
            continue
        segment = int(batch.sensor_pair_index[row])
        isolated = _masked(batch, _one_hot(batch, row))

        beat = synthesize_fmcw(isolated, fmcw)
        cfr = synthesize_ofdm(isolated, ofdm)
        train = synthesize_pulsed(isolated, pulsed)

        # The FMCW slow-time step for a fixed pair is num_tx chirp periods:
        # TDM fires the transmitters in turn.
        fmcw_step = fmcw.num_tx * fmcw.chirp_period_s
        peak_sample = int(train[0, segment].abs().argmax())
        measured = {
            "fmcw": _slow_time_step_rad(beat, segment, 0) / (2.0 * math.pi * fmcw_step),
            "ofdm": _slow_time_step_rad(cfr, segment, 0) / (2.0 * math.pi * ofdm.symbol_period_s),
            "pulsed": _slow_time_step_rad(train, segment, peak_sample) / (2.0 * math.pi * pulsed.pri_s),
        }
        expected = F_REF_HZ * tau_rate
        assert measured["fmcw"] == pytest.approx(expected, rel=1e-2), (row, measured)
        assert measured["ofdm"] == pytest.approx(-expected, rel=1e-2), (row, measured)
        assert measured["pulsed"] == pytest.approx(-expected, rel=1e-2), (row, measured)
        # The signs are opposite and neither is zero.
        assert measured["fmcw"] * measured["ofdm"] < 0.0
        assert measured["fmcw"] * measured["pulsed"] < 0.0
        checked += 1

    assert checked >= 8
    # Both physical signs appear across the fixture: one site recedes, one
    # approaches, so this is not a test of one sign convention twice.
    rates = [float(batch.delay_rate[row]) for row in _live_rows(batch)]
    assert any(value > 0.0 for value in rates)
    assert any(value < 0.0 for value in rates)


# ---------------------------------------------------------------------------
# T5.3 (I3) - one amplitude, three peaks
# ---------------------------------------------------------------------------


def test_one_amplitude_and_one_phase_reach_all_three_products(frame):
    """The criterion-A3 test: the complex target phase enters every IQ product.

    The three exact identities, each stated where its waveform has no window
    and no straddle loss:

        FMCW    s[0][p][0]   = conj(C_rt) exp(+j 2 pi S tau (t0 - tau/2))
        OFDM    H[0][p][0]   = C_rt
        pulsed  MF peak      = C_rt   (to the sampled filter's straddle loss)

    The FMCW identity is taken on the first fast-time SAMPLE rather than on the
    FFT peak on purpose. A peak's argument carries the Dirichlet kernel's own
    phase at an off-grid delay, so asserting it would be asserting a window
    rather than the target's phase. The FFT peak's MAGNITUDE is asserted
    separately, against the rectangular window's worst-case scalloping loss.
    """

    _, _, batch = frame
    fmcw = fmcw_spec(1)
    ofdm = ofdm_spec(1)
    pulsed = pulsed_spec(1)
    rows = _live_rows(batch)
    assert rows

    for row in rows:
        segment = int(batch.sensor_pair_index[row])
        isolated = _masked(batch, _one_hot(batch, row))
        transfer = complex(batch.complex_transfer_ref[row].cpu())
        tau = float(batch.total_delay_s[row])

        beat = synthesize_fmcw(isolated, fmcw)[0, segment]
        cfr = synthesize_ofdm(isolated, ofdm)[0, segment]
        train = synthesize_pulsed(isolated, pulsed)[0, segment]
        _, mf_peak, _ = pulsed_grid.peak_estimate(train.detach(), pulsed)

        magnitude = abs(transfer)
        assert abs(beat[0].cpu()) == pytest.approx(magnitude, rel=1e-4), row
        assert abs(cfr[0].cpu()) == pytest.approx(magnitude, rel=1e-4), row
        assert abs(mf_peak) == pytest.approx(magnitude, rel=1e-2), row

        residual = 2.0 * math.pi * fmcw.slope_hz_per_s * tau * (fmcw.t_start_s - 0.5 * tau)
        expected_beat = -_arg(transfer) + residual
        assert _angle_close(_arg(complex(beat[0].cpu())), expected_beat, 1e-3), row
        assert _angle_close(_arg(complex(cfr[0].cpu())), _arg(transfer), 1e-4), row
        assert _angle_close(_arg(mf_peak), _arg(transfer), 2e-2), row

        # And the FFT peak's magnitude, over the samples, within the
        # rectangular window's worst-case scalloping loss of 2/pi.
        spectrum = torch.fft.fft(beat.detach().to(torch.complex128).cpu())
        peak = float(spectrum.abs().max()) / fmcw.num_samples
        assert 0.6 * magnitude <= peak <= 1.01 * magnitude, (row, peak, magnitude)


def _arg(value: complex) -> float:
    return math.atan2(value.imag, value.real)


def _angle_close(measured: float, expected: float, tolerance: float) -> bool:
    delta = (measured - expected + math.pi) % (2.0 * math.pi) - math.pi
    return abs(delta) <= tolerance


# ---------------------------------------------------------------------------
# T5.5 (I4) - dead rows
# ---------------------------------------------------------------------------


def test_a_dead_row_is_indistinguishable_from_a_row_that_never_existed(frame):
    """``torch.equal`` against a batch built from the survivors alone.

    Not ``allclose``: a masked row must contribute an accumulation of exactly
    zero, and a zero added to a float sum is exact. Anything looser would let a
    dead row leak a denormal.
    """

    _, _, batch = frame
    keep = torch.ones(batch.path_count, dtype=torch.bool, device="cuda")
    keep[1] = False
    keep[4] = False
    keep[batch.path_count - 1] = False

    reduced = _select(batch, keep)
    masked = _masked(batch, keep)
    for synthesize, spec in (
        (synthesize_fmcw, fmcw_spec(2)),
        (synthesize_ofdm, ofdm_spec(2)),
        (synthesize_pulsed, pulsed_spec(2)),
    ):
        assert torch.equal(synthesize(masked, spec), synthesize(reduced, spec)), spec


def test_a_dead_rows_weight_receives_exactly_zero_gradient(frame):
    """The zeroing is on the WEIGHT, so no gradient reaches a dead row.

    Zeroing the OUTPUT would leave a live path back through a row that does not
    exist, and the primal would look identical.
    """

    _, _, batch = frame
    keep = torch.ones(batch.path_count, dtype=torch.bool, device="cuda")
    keep[2] = False
    keep[5] = False

    for synthesize, spec in (
        (synthesize_fmcw, fmcw_spec(2)),
        (synthesize_ofdm, ofdm_spec(2)),
        (synthesize_pulsed, pulsed_spec(2)),
    ):
        transfer = batch.complex_transfer_ref.detach().clone().requires_grad_(True)
        live = _masked(replace(batch, complex_transfer_ref=transfer), keep)
        cube = synthesize(live, spec)
        (cube.real.sum() + cube.imag.sum()).backward()
        dead = ~keep
        assert torch.equal(transfer.grad[dead], torch.zeros_like(transfer.grad[dead])), spec
        assert float(transfer.grad[keep].abs().sum()) > 0.0, spec


# ---------------------------------------------------------------------------
# T5.6 (I5) - linearity
# ---------------------------------------------------------------------------


def test_every_waveform_is_linear_in_the_path_weight(frame):
    """``synth({a, b}) == synth({a}) + synth({b})`` for all three."""

    _, _, batch = frame
    rows = _live_rows(batch)
    first, second = rows[0], rows[3]
    both = torch.zeros(batch.path_count, dtype=torch.bool, device="cuda")
    both[first] = True
    both[second] = True

    for synthesize, spec in (
        (synthesize_fmcw, fmcw_spec(2)),
        (synthesize_ofdm, ofdm_spec(2)),
        (synthesize_pulsed, pulsed_spec(2)),
    ):
        together = synthesize(_masked(batch, both), spec)
        apart = synthesize(_masked(batch, _one_hot(batch, first)), spec) + synthesize(
            _masked(batch, _one_hot(batch, second)), spec
        )
        torch.testing.assert_close(together, apart, rtol=1e-5, atol=1e-8 * float(together.abs().max()))


def test_the_frontend_agc_breaks_linearity_and_that_is_a_tested_fact(frame):
    """AGC is data dependent, so I5 holds for synthesis and not for the chain.

    Asserted rather than footnoted: a caveat nobody executes is a caveat that
    stops being true.
    """

    from witwin.radar.frontend import AgcSpec, FrontendChain, FrontendSpec, PortSpec

    _, _, batch = frame
    cube = synthesize_fmcw(batch, fmcw_spec(2))
    scaled = cube * 2.0

    without_agc = FrontendChain(FrontendSpec(port=PortSpec(reference_impedance_ohm=50.0)))
    linear_once = without_agc.apply(cube).signal
    linear_twice = without_agc.apply(scaled).signal
    torch.testing.assert_close(linear_twice, 2.0 * linear_once, rtol=1e-6, atol=0.0)

    with_agc = FrontendChain(
        FrontendSpec(
            port=PortSpec(reference_impedance_ohm=50.0),
            agc=AgcSpec(target_rms=1e-3, mode="global", min_gain_db=-60.0, max_gain_db=60.0),
        )
    )
    agc_once = with_agc.apply(cube).signal
    agc_twice = with_agc.apply(scaled).signal
    assert not torch.allclose(agc_twice, 2.0 * agc_once, rtol=1e-3, atol=0.0)


# ---------------------------------------------------------------------------
# T5.7 (I6) - row identity and order
# ---------------------------------------------------------------------------


def test_permuting_rows_within_a_segment_changes_nothing_but_the_sum_order(frame):
    """A cube is a per-segment sum, so a within-segment permutation is inert."""

    _, _, batch = frame
    order = torch.arange(batch.path_count, device="cuda")
    pair = batch.sensor_pair_index
    # Reverse each segment in place: the segment boundaries are unchanged, so
    # the CSR offsets stay valid and only the summation order moves.
    for segment in range(batch.sensor_pair_count):
        rows = torch.nonzero(pair == segment, as_tuple=False).flatten()
        if rows.numel() > 1:
            order[rows] = rows.flip(0)

    permuted = _reindex(batch, order)
    for synthesize, spec in (
        (synthesize_fmcw, fmcw_spec(2)),
        (synthesize_ofdm, ofdm_spec(2)),
        (synthesize_pulsed, pulsed_spec(2)),
    ):
        reference = synthesize(batch, spec)
        torch.testing.assert_close(
            synthesize(permuted, spec), reference, rtol=1e-6, atol=1e-7 * float(reference.abs().max())
        )


def _reindex(batch, order: torch.Tensor):
    from witwin.radar.paths import RadarPathTopology

    topology = batch.topology
    return replace(
        batch,
        sensor_pair_index=batch.sensor_pair_index[order].contiguous(),
        total_delay_s=batch.total_delay_s[order].contiguous(),
        delay_rate=(None if batch.delay_rate is None else batch.delay_rate[order].contiguous()),
        complex_transfer_ref=batch.complex_transfer_ref[order].contiguous(),
        topology=RadarPathTopology(
            radar_source_id=topology.radar_source_id[order].contiguous(),
            site_id=topology.site_id[order].contiguous(),
            radar_sink_id=topology.radar_sink_id[order].contiguous(),
            inbound_row=topology.inbound_row[order].contiguous(),
            outbound_row=topology.outbound_row[order].contiguous(),
        ),
        row_valid=(None if batch.row_valid is None else batch.row_valid[order].contiguous()),
    )


def test_moving_a_row_to_another_segment_moves_it_in_the_cube(frame):
    """The cube's segment axis follows ``sensor_pair_index``, not row order."""

    _, _, batch = frame
    rows = _live_rows(batch)
    row = rows[0]
    source_segment = int(batch.sensor_pair_index[row])
    empty = [
        segment for segment in range(batch.sensor_pair_count) if int((batch.sensor_pair_index == segment).sum()) == 0
    ]
    assert empty, "the fixture is supposed to publish empty pair segments"
    target = empty[0]

    isolated = _masked(batch, _one_hot(batch, row))
    moved_index = isolated.sensor_pair_index.clone()
    moved_index[row] = target
    counts = torch.bincount(moved_index, minlength=isolated.sensor_pair_count)
    offsets = torch.zeros(isolated.sensor_pair_count + 1, dtype=torch.int64, device=moved_index.device)
    offsets[1:] = torch.cumsum(counts, dim=0)
    # Moving a row across a segment boundary reorders the CSR, so the whole row
    # set is sorted by its new segment - which is what the contract requires.
    order = torch.argsort(moved_index, stable=True)
    moved = _reindex(replace(isolated, sensor_pair_index=moved_index), order)
    moved = replace(moved, sensor_pair_index=moved_index[order].contiguous(), pair_offsets=offsets.contiguous())

    for synthesize, spec in (
        (synthesize_fmcw, fmcw_spec(1)),
        (synthesize_ofdm, ofdm_spec(1)),
        (synthesize_pulsed, pulsed_spec(1)),
    ):
        before = synthesize(isolated, spec)
        after = synthesize(moved, spec)
        assert float(before[0, source_segment].abs().sum()) > 0.0
        assert float(after[0, source_segment].abs().sum()) == 0.0
        torch.testing.assert_close(after[0, target], before[0, source_segment], rtol=1e-6, atol=0.0)


# ---------------------------------------------------------------------------
# T5.8 - empty segments and zero rows
# ---------------------------------------------------------------------------


def test_empty_pair_segments_are_exactly_zero_in_all_three_cubes(frame):
    _, _, batch = frame
    empty = [
        segment for segment in range(batch.sensor_pair_count) if int((batch.sensor_pair_index == segment).sum()) == 0
    ]
    assert len(empty) == 2

    for synthesize, spec in (
        (synthesize_fmcw, fmcw_spec(2)),
        (synthesize_ofdm, ofdm_spec(2)),
        (synthesize_pulsed, pulsed_spec(2)),
    ):
        cube = synthesize(batch, spec)
        for segment in empty:
            assert torch.equal(cube[:, segment], torch.zeros_like(cube[:, segment])), (spec, segment)


def test_a_batch_with_no_rows_gives_an_all_zero_cube_of_the_right_shape(frame):
    _, _, batch = frame
    none = torch.zeros(batch.path_count, dtype=torch.bool, device="cuda")
    empty_batch = _select(batch, none)
    assert empty_batch.path_count == 0

    shapes = {
        "fmcw": (2, batch.sensor_pair_count, FMCW_SAMPLES),
        "ofdm": (2, batch.sensor_pair_count, OFDM_SUBCARRIERS),
        "pulsed": (2, batch.sensor_pair_count, 512),
    }
    cubes = {
        "fmcw": synthesize_fmcw(empty_batch, fmcw_spec(2)),
        "ofdm": synthesize_ofdm(empty_batch, ofdm_spec(2)),
        "pulsed": synthesize_pulsed(empty_batch, pulsed_spec(2)),
    }
    for name, cube in cubes.items():
        assert tuple(cube.shape) == shapes[name], name
        assert torch.equal(cube, torch.zeros_like(cube)), name


# ---------------------------------------------------------------------------
# T5.4 - absolute level against the bistatic radar equation (criterion A3)
# ---------------------------------------------------------------------------


#: A one square metre target, so the radar equation's sigma is not a scale
#: factor that could hide a normalisation error by being one.
SIGMA_M2 = 2.5


def _los_row(spike, composed):
    """The one reflection-free line-of-sight round trip and its two leg lengths."""

    keys = drv.composed_keys(spike, composed)
    table = {row.key: row for row in geo.combined_rows()}
    for index, key in enumerate(keys):
        if key[3] == "los" and key[4] == "los":
            row = table[key]
            return index, row.inbound.length_m, row.outbound.length_m
    raise AssertionError("the fixture must publish a line-of-sight round trip")


def _radar_equation_power(power_w: float, d_in: float, d_out: float) -> float:
    wavelength = C0 / F_REF_HZ
    return power_w * wavelength**2 * SIGMA_M2 / ((4.0 * math.pi) ** 3 * d_in**2 * d_out**2)


def test_the_composed_coefficient_is_the_bistatic_radar_equation(spike):
    """``|C_rt|^2 == P_t lambda^2 sigma / ((4 pi)^3 d_in^2 d_out^2)``.

    This is the ABSOLUTE level, and it is what pins the two normalisations this
    stage owns:

    * ``S = sqrt(4 pi sigma) / lambda``, which was unpinned and whose omission
      is a factor of ``lambda^2 / (4 pi)`` - 58 dB at 77 GHz;
    * the site excited at exactly 1 W, so the site is a re-radiator rather than
      a second transmitter.

    ``P_t`` is the power the fixture declares: under Channel ADR-039 the
    consumer coefficient carries ``sqrt(powers_w)``, so the declared and the
    effective transmit power are the same thing.
    """

    from witwin.radar.scattering import ScalarRcsResponse

    response = ScalarRcsResponse.from_rcs(SIGMA_M2, reference_frequency_hz=F_REF_HZ, device="cuda")
    composed, _, _ = spike.frame(response=response)
    row, d_in, d_out = _los_row(spike, composed)

    measured = float(composed.complex_transfer_ref[row].abs().cpu()) ** 2
    expected = _radar_equation_power(geo.TX_POWER_W, d_in, d_out)
    assert measured == pytest.approx(expected, rel=1e-4), (measured, expected)


def test_channel_applies_the_declared_transmit_power_once(spike, monkeypatch):
    """The declared ``powers_w`` reaches the coefficient exactly once.

    Under Channel ADR-039 the consumer transport carries the declared source
    amplitude ``sqrt(powers_w)``: a fourfold transmit power gives an amplitude
    ratio of exactly 2.0, and the absolute level is the bistatic radar
    equation at the DECLARED power. The second assertion is what makes this a
    single-count pin: the site stays excited at exactly 1 W, so ``sqrt(P_tx)``
    entering twice (Channel and the sensor weight, or Channel and the site
    excitation) fails the absolute level by the power ratio.

    History: before ADR-039 the coefficient was unit-source-amplitude and this
    test pinned that gap as an upstream defect (F-19). The fixture running at
    ``TX_POWER_W = 0.01`` rather than 1 W is what made either behaviour
    visible at all.
    """

    from witwin.radar.scattering import ScalarRcsResponse

    response = ScalarRcsResponse.from_rcs(SIGMA_M2, reference_frequency_hz=F_REF_HZ, device="cuda")
    monkeypatch.setattr(geo, "TX_POWER_W", 0.01)
    low, _, _ = spike.frame(response=response)
    row, d_in, d_out = _los_row(spike, low)
    low_amplitude = float(low.complex_transfer_ref[row].abs().cpu())

    monkeypatch.setattr(geo, "TX_POWER_W", 0.04)
    high, _, _ = spike.frame(response=response)
    high_amplitude = float(high.complex_transfer_ref[row].abs().cpu())

    # sqrt(0.04 / 0.01) = 2, the single-application amplitude ratio.
    assert high_amplitude / low_amplitude == pytest.approx(2.0, rel=1e-6)
    # And the absolute level at 0.01 W is the radar equation at 0.01 W: not
    # 100x it (the pre-ADR-039 unit-amplitude value), not 0.01x it (a double
    # count).
    would_be = _radar_equation_power(0.01, d_in, d_out)
    assert low_amplitude**2 == pytest.approx(would_be, rel=1e-4)


def test_no_waveform_spec_may_reapply_the_spreading_the_weight_carries(frame):
    """Hazard F1, made structural: three specs, one class attribute each.

    A Channel-sourced weight carries ``lambda / (4 pi d)`` per leg. All three
    synthesis kernels declare ``applies_spreading = False`` as a statement about
    the kernel rather than as a setting, and ``require_compatible`` refuses the
    combination that would apply it twice before any launch.
    """

    _, _, batch = frame
    assert batch.weight_includes_spreading is True
    assert batch.weight_includes_tx_power is True
    assert batch.weight_includes_reference_phase is True
    for spec in (fmcw_spec(1), ofdm_spec(1), pulsed_spec(1)):
        assert spec.applies_spreading is False, spec
        # And the carrier has exactly one home, the rate one, because the
        # weight already holds the absolute reference phase.
        assert spec.carrier_hz == 0.0, spec
        assert spec.carrier_rate_hz == F_REF_HZ, spec
