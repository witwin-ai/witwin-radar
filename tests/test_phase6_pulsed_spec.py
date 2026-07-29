"""The pulsed spec, its guards, and the matched filter - all on the CPU.

Everything here is closed form or pure Python, so it runs in the default suite
without a GPU. That is deliberate rather than incidental: the unit conversions,
the range-migration bound, and the pulse normalisation are exactly the kind of
thing that is wrong once and then wrong everywhere, and gating them behind
``--gpu`` would mean the wrong-once is discovered by a kernel test that is
already assuming they are right.

The reference grid lives in ``tests/support/pulsed_grid`` and is shared with the
two GPU files. It is NOT the grid the Phase-6 brief sketched; that module records
the three reasons, and the first test below asserts the four inequalities the
grid has to satisfy for any of the assertions in this suite to mean anything.
"""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch
from support.pulsed_grid import (  # noqa: E402
    BANDWIDTH_HZ,
    C0,
    F_REF_HZ,
    NUM_PULSES,
    PRI_S,
    PULSE_WIDTH_S,
    RADIAL_SPEED_MPS,
    SAMPLE_PERIOD_S,
    SAMPLE_RATE_HZ,
    TAU_RATE,
    TAU_RT_S,
    rect_spec,
    reference_spec,
)

from witwin.radar.processing.range_doppler import lag_axis, matched_filter, pulse_samples  # noqa: E402
from witwin.radar.synthesis.assembly import (  # noqa: E402
    PulsedSpec,
    SlowTimeMode,
    SynthesisPathBatch,
    require_pulsed_compatible,
)

# --------------------------------------------------------------------------
# The grid is self-consistent, and every reason it has to be is asserted
# --------------------------------------------------------------------------


def test_the_reference_grid_is_internally_consistent():
    """Four inequalities, each of which the sketched grid violated.

    Written as assertions rather than as a comment because a later edit to any
    one of the five numbers can break a different one of the four, silently.
    """

    spec = reference_spec()

    # 1. Nyquist for the DISCRETE correlation, not merely for the pulse. The
    #    matched filter's integrand y * conj(p) occupies [-B, B], so the
    #    rectangle-rule sum equals the continuous integral only above 2 B.
    assert spec.sample_rate_hz > 2.0 * spec.bandwidth_hz

    # 2. The gate closes before the next pulse fires.
    assert spec.range_gate_end_s == pytest.approx(20.48e-6, rel=1e-12)
    assert spec.range_gate_end_s <= spec.pri_s

    # 3. The WHOLE echo is inside the gate, not just its leading edge.
    assert TAU_RT_S + spec.pulse_width_s < spec.range_gate_end_s

    # 4. The test speed is inside the unambiguous bound, so a slow-time slope
    #    means something. 12 m/s would not be.
    assert spec.max_unambiguous_speed_m_s == pytest.approx(9.733521363636363, rel=1e-12)
    assert RADIAL_SPEED_MPS < spec.max_unambiguous_speed_m_s
    assert 12.0 > spec.max_unambiguous_speed_m_s


def test_the_derived_quantities_are_the_closed_forms():
    spec = reference_spec()
    assert spec.sample_rate_hz == pytest.approx(SAMPLE_RATE_HZ, rel=1e-12)
    assert spec.wavelength_m == pytest.approx(C0 / F_REF_HZ, rel=1e-12)
    assert spec.duty_cycle == pytest.approx(0.1, rel=1e-12)
    assert spec.coherent_processing_interval_s == pytest.approx(3.2e-3, rel=1e-12)

    # LFM resolution comes from the SWEEP and is independent of the length.
    assert spec.range_resolution_m == pytest.approx(7.49481145, rel=1e-12)
    assert spec.range_resolution_m == pytest.approx(C0 / (2.0 * BANDWIDTH_HZ), rel=1e-12)
    # The rectangle's comes from its LENGTH alone, and the two disagree by the
    # time-bandwidth product - 200 here - which is the whole reason an LFM is
    # worth transmitting.
    assert rect_spec().range_resolution_m == pytest.approx(1498.96229, rel=1e-12)
    assert rect_spec().range_resolution_m / spec.range_resolution_m == pytest.approx(
        BANDWIDTH_HZ * PULSE_WIDTH_S, rel=1e-12
    )

    assert spec.max_unambiguous_range_m == pytest.approx(14989.6229, rel=1e-12)
    assert spec.max_unambiguous_range_m == pytest.approx(C0 * PRI_S / 2.0, rel=1e-12)
    assert spec.max_unambiguous_speed_m_s == pytest.approx(spec.wavelength_m / (4.0 * PRI_S), rel=1e-12)
    assert spec.range_cell_delay_s == pytest.approx(5.0e-8, rel=1e-12)
    assert spec.range_migration_delay_s == pytest.approx(abs(TAU_RATE) * NUM_PULSES * PRI_S, rel=1e-12)
    assert spec.pulse_amplitude == pytest.approx(1.0 / math.sqrt(PULSE_WIDTH_S), rel=1e-12)
    assert spec.pulse_sample_count == 500
    assert spec.pulse_grid_is_commensurate


def test_the_doppler_and_slow_time_closed_forms():
    """The sign, and the pulse-position-dependent correction.

    ``f_D = -f_ref tau_rate`` is negative for a receding row, in Channel's
    convention, which this waveform publishes unchanged. The per-pulse phase step
    additionally carries ``B u / T_p``: the LFM's own phase moving with the
    drifting envelope position. It is the pulsed analogue of OFDM's ``n df`` and
    is about 2.6e-4 of the step at the trailing edge of this sweep - small, but
    two orders of magnitude larger than the tolerance the slope is asserted to.
    """

    spec = reference_spec()
    assert spec.doppler_frequency_hz(TAU_RATE) == pytest.approx(-2568.443533, rel=1e-8)
    assert spec.doppler_frequency_hz(TAU_RATE) < 0.0
    assert spec.doppler_frequency_hz(-TAU_RATE) > 0.0
    assert abs(spec.doppler_frequency_hz(TAU_RATE)) < 0.5 / PRI_S

    # At the leading edge the correction is exactly zero, so the step is the
    # pure carrier term.
    at_edge = spec.slow_time_phase_step_rad(TAU_RATE, 0.0)
    assert at_edge == pytest.approx(-math.tau * F_REF_HZ * TAU_RATE * PRI_S, rel=1e-12)
    assert spec.instantaneous_pulse_frequency_hz(0.0) == 0.0

    # At the trailing edge it is the largest it ever gets, and it is B.
    assert spec.instantaneous_pulse_frequency_hz(PULSE_WIDTH_S) == pytest.approx(BANDWIDTH_HZ, rel=1e-12)
    at_tail = spec.slow_time_phase_step_rad(TAU_RATE, PULSE_WIDTH_S)
    assert at_tail / at_edge == pytest.approx(1.0 + BANDWIDTH_HZ / F_REF_HZ, rel=1e-12)

    # A rectangular pulse has no such correction at all: its envelope carries no
    # phase, so the whole slow-time step is the carrier's.
    assert rect_spec().instantaneous_pulse_frequency_hz(PULSE_WIDTH_S) == 0.0
    assert rect_spec().slow_time_phase_step_rad(TAU_RATE, PULSE_WIDTH_S) == pytest.approx(at_edge, rel=1e-12)


# --------------------------------------------------------------------------
# Spec-level refusals
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "override, message",
    [
        ({"num_pulses": 0}, "num_pulses"),
        ({"num_samples": 0}, "num_samples"),
        ({"sample_period_s": 0.0}, "sample_period_s"),
        ({"pri_s": -1.0}, "pri_s"),
        ({"range_gate_start_s": -1.0e-9}, "range_gate_start_s"),
        ({"pulse_width_s": 0.0}, "pulse_width_s"),
        ({"bandwidth_hz": 0.0}, "bandwidth_hz"),
        ({"reference_frequency_hz": 0.0}, "reference_frequency_hz"),
        ({"max_expected_delay_rate": -1.0e-9}, "max_expected_delay_rate"),
        ({"pulse_kind": "gaussian"}, "pulse_kind"),
        ({"pulse_normalization": "unit_amplitude"}, "pulse_normalization"),
    ],
)
def test_the_spec_refuses_an_impossible_grid(override, message):
    with pytest.raises(ValueError, match=message):
        reference_spec(**override)


def test_a_tabulated_pulse_is_refused_by_name_not_by_accident():
    """The catalogue is two ANALYTIC kinds and the refusal says why.

    A sampled or tabulated pulse would need a lookup inside the kernel at a
    fractional argument, which is the interpolation this family exists to avoid.
    Adding one is a design decision, not a configuration value.
    """

    with pytest.raises(ValueError, match="ANALYTIC"):
        reference_spec(pulse_kind="tabulated")


def test_both_carrier_homes_may_not_be_named_at_once():
    """The shared refusal, reached through the shared helper.

    The same rule, the same message, and the same helper as the FMCW and OFDM
    specs: the absolute carrier lives in the weight or in the kernel, and naming
    both counts it twice.
    """

    with pytest.raises(ValueError, match="two"):
        reference_spec(carrier_hz=F_REF_HZ, carrier_rate_hz=F_REF_HZ)

    from witwin.radar.synthesis.assembly import require_single_carrier_home

    with pytest.raises(ValueError) as spec_error:
        reference_spec(carrier_hz=F_REF_HZ, carrier_rate_hz=F_REF_HZ)
    with pytest.raises(ValueError) as helper_error:
        require_single_carrier_home(F_REF_HZ, F_REF_HZ)
    assert str(spec_error.value) == str(helper_error.value)


# --------------------------------------------------------------------------
# T3.8  the compatibility guards, on CONFIGURED values only
# --------------------------------------------------------------------------


def _cpu_batch(spec: PulsedSpec) -> SynthesisPathBatch:
    from witwin.radar.paths import RadarPathTopology

    zeros = torch.zeros(1, dtype=torch.int64)
    return SynthesisPathBatch(
        sensor_pair_count=1,
        path_count=1,
        sensor_pair_index=zeros.clone(),
        pair_offsets=torch.tensor([0, 1], dtype=torch.int64),
        total_delay_s=torch.tensor([TAU_RT_S], dtype=torch.float32),
        delay_rate=torch.tensor([TAU_RATE], dtype=torch.float32),
        complex_transfer_ref=torch.tensor([1.0 + 0.0j], dtype=torch.complex64),
        reference_frequency_hz=spec.reference_frequency_hz,
        frequency_response=None,
        frequency_offsets_hz=None,
        topology=RadarPathTopology(zeros.clone(), zeros.clone(), zeros.clone(), zeros.clone(), zeros.clone()),
        row_valid=None,
        join_mode="multipath",
        weight_includes_reference_phase=True,
        weight_includes_spreading=True,
        weight_includes_tx_power=True,
        slow_time_mode=SlowTimeMode.FROZEN_WEIGHT_WITH_CARRIER_RATE,
    )


def test_the_reference_grid_passes_every_pulsed_guard():
    spec = reference_spec()
    require_pulsed_compatible(_cpu_batch(spec), spec)
    require_pulsed_compatible(_cpu_batch(spec), rect_spec())


def test_a_pulse_longer_than_the_repetition_interval_is_refused():
    spec = reference_spec(pulse_width_s=PRI_S)
    with pytest.raises(ValueError, match="pulse_width_s"):
        require_pulsed_compatible(_cpu_batch(spec), spec)


def test_a_gate_that_overruns_the_repetition_interval_is_refused():
    """Exactly the failure the sketched 512-sample, 5 MSPS grid had."""

    spec = reference_spec(num_samples=8192)
    assert spec.range_gate_end_s > spec.pri_s
    with pytest.raises(ValueError, match="range gate overruns"):
        require_pulsed_compatible(_cpu_batch(spec), spec)


def test_range_migration_past_one_cell_is_refused_and_names_itself():
    """T3.8. The bound is ``tau_rate L T_pri < 1 / B``, on CONFIGURED values.

    The failure it prevents is not a crash: the echo walks between range cells
    inside one coherent processing interval, the compressed peak smears, and the
    result reads as a defocused target rather than as an error. Named in the
    message so that a reader is not sent looking for a bug in the physics.
    """

    spec = reference_spec()
    assert spec.range_migration_delay_s < spec.range_cell_delay_s

    # 1000 m/s at this PRI and coherent processing interval walks 2.13e-8 s,
    # which is 0.43 of a cell: legal, and close enough that the bound is not
    # vacuous at any speed a radar would care about.
    near = reference_spec(max_expected_delay_rate=2.0 * 1000.0 / C0)
    assert near.range_migration_delay_s == pytest.approx(2.1348102e-08, rel=1e-6)
    assert near.range_migration_delay_s / near.range_cell_delay_s == pytest.approx(0.4270, rel=1e-3)
    require_pulsed_compatible(_cpu_batch(near), near)

    # 5000 m/s walks 2.1 cells and is refused.
    fast = reference_spec(max_expected_delay_rate=2.0 * 5000.0 / C0)
    assert fast.range_migration_delay_s > fast.range_cell_delay_s
    with pytest.raises(ValueError, match="range migration"):
        require_pulsed_compatible(_cpu_batch(fast), fast)

    # The bound is a THREE-way statement and every term is load bearing: the
    # same delay rate is legal at a shorter coherent processing interval.
    short = dataclasses.replace(fast, num_pulses=1)
    assert short.range_migration_delay_s < short.range_cell_delay_s
    require_pulsed_compatible(_cpu_batch(short), short)


def test_the_guards_read_no_tensor_value(monkeypatch):
    """CONFIGURED values only: no reduction over the device delays.

    A measured maximum delay or delay rate would be a per-frame device-to-host
    transfer, which is exactly what the fixed-topology capability exists to
    avoid. Asserted by making every host-observation method on a Tensor raise.
    """

    spec = reference_spec()
    batch = _cpu_batch(spec)
    for name in ("item", "cpu", "tolist", "numpy", "max", "min", "amax", "amin"):
        monkeypatch.setattr(
            torch.Tensor,
            name,
            lambda *args, _n=name, **kwargs: pytest.fail(
                f"require_pulsed_compatible observed a tensor through .{_n}()"
            ),
        )
    require_pulsed_compatible(batch, spec)


def test_the_shared_provenance_rules_still_apply():
    """R1 and R3 reach the pulsed entry through the shared function.

    The pulsed guard adds three checks; it does not replace the eight that
    decide whether a weight and a spec may be used together at all.
    """

    spec = reference_spec()
    with pytest.raises(ValueError, match="double-counted carrier phase"):
        require_pulsed_compatible(_cpu_batch(spec), reference_spec(carrier_hz=F_REF_HZ, carrier_rate_hz=0.0))
    with pytest.raises(ValueError, match="understated Doppler"):
        require_pulsed_compatible(_cpu_batch(spec), reference_spec(carrier_rate_hz=0.5 * F_REF_HZ))
    # R7. The carrier rate still matches the BATCH's reference frequency, so
    # R3 is satisfied and the mismatch this catches is the spec's own declared
    # frequency: a narrowband coefficient is not transferable between them.
    with pytest.raises(ValueError, match="reference frequency mismatch"):
        require_pulsed_compatible(
            _cpu_batch(spec), reference_spec(reference_frequency_hz=24.0e9, carrier_rate_hz=F_REF_HZ)
        )


def test_the_pulsed_guard_refuses_a_spec_of_another_waveform():
    from witwin.radar.synthesis.assembly import OfdmSpec

    other = OfdmSpec(
        num_subcarriers=4,
        num_symbols=1,
        subcarrier_spacing_hz=120.0e3,
        cyclic_prefix_s=2.0e-6,
        reference_frequency_hz=F_REF_HZ,
        max_expected_delay_s=1.0e-6,
        carrier_rate_hz=F_REF_HZ,
    )
    with pytest.raises(TypeError, match="PulsedSpec"):
        require_pulsed_compatible(_cpu_batch(reference_spec()), other)


# --------------------------------------------------------------------------
# T3.3 (first half)  the unit-energy replica
# --------------------------------------------------------------------------


@pytest.mark.parametrize("spec_factory", [reference_spec, rect_spec])
def test_the_replica_carries_exactly_unit_energy(spec_factory):
    """``sum_m |p[m]|^2 T_s == 1`` to ``rtol=1e-12``, for BOTH pulse kinds.

    Exact rather than approximate because the pulse spans a whole number of
    samples on this grid and its support is HALF-OPEN. With a closed support the
    replica would need 501 samples to match a received pulse of 501, but only at
    delays that land exactly on the grid; the resulting one-tap mismatch costs
    0.2 percent of the peak and biases the delay estimate.

    The LFM and the rectangle share an amplitude and differ only by a phase,
    which is why one assertion covers both: a phase cannot change ``|p|``.
    """

    spec = spec_factory()
    replica = pulse_samples(spec)
    assert replica.shape == (500,)
    assert replica.dtype == torch.complex128
    energy = float((replica.abs() ** 2).sum()) * spec.sample_period_s
    assert energy == pytest.approx(1.0, rel=1e-12)
    assert float(replica.abs().max()) == pytest.approx(spec.pulse_amplitude, rel=1e-12)
    assert float(replica.abs().min()) == pytest.approx(spec.pulse_amplitude, rel=1e-12)


def test_the_two_replicas_differ_only_by_phase():
    lfm = pulse_samples(reference_spec())
    rect = pulse_samples(rect_spec())
    torch.testing.assert_close(lfm.abs(), rect.abs(), rtol=1e-15, atol=0.0)
    assert float(rect.imag.abs().max()) == 0.0
    assert float(lfm.imag.abs().max()) > 0.1 * float(lfm.abs().max())

    # The LFM's instantaneous frequency sweeps from 0 to B across the pulse:
    # the phase step between adjacent samples is 2 pi B u T_s / T_p.
    spec = reference_spec()
    steps = torch.angle(lfm[1:] * torch.conj(lfm[:-1]))
    analytic_last = (
        math.tau * spec.bandwidth_hz * (498.5 * spec.sample_period_s) * spec.sample_period_s / spec.pulse_width_s
    )
    assert float(steps[-1]) == pytest.approx(analytic_last, rel=1e-9)


def test_a_pulse_shorter_than_one_sample_is_refused():
    spec = reference_spec(pulse_width_s=SAMPLE_PERIOD_S / 4.0)
    with pytest.raises(ValueError, match="no replica"):
        pulse_samples(spec)


# --------------------------------------------------------------------------
# The matched filter's own conventions, on a synthetic CPU signal
# --------------------------------------------------------------------------


def _delayed_replica(spec: PulsedSpec, shift_samples: int) -> torch.Tensor:
    """The received train for one on-grid row, built without the kernel."""

    replica = pulse_samples(spec)
    signal = torch.zeros((1, 1, spec.num_samples), dtype=torch.complex128)
    signal[0, 0, shift_samples : shift_samples + replica.shape[0]] = replica
    return signal


def test_the_matched_filter_is_a_correlation_not_a_convolution():
    """The peak sits at ``tau``, not at ``tau + T_p``.

    Convolving instead of correlating - forgetting to conjugate, or reversing
    the replica - moves the peak by a whole pulse width, which at this grid is
    1500 m of range. It still looks like a compressed peak.
    """

    spec = reference_spec(num_pulses=1)
    shift = 100
    z = matched_filter(_delayed_replica(spec, shift), spec)[0, 0]
    lag = lag_axis(spec)
    assert int(z.abs().argmax()) == shift
    assert float(lag[int(z.abs().argmax())]) == pytest.approx(shift * SAMPLE_PERIOD_S, rel=1e-12)


def test_the_matched_filter_peak_is_the_coefficient_at_an_on_grid_delay():
    """T3.3. ``MF peak == C_rt``, magnitude AND argument, exactly.

    The unit-ENERGY normalisation plus the ``T_s`` factor in the correlation is
    what removes every sample-count factor: with a unit-amplitude pulse the peak
    would be ``C_rt T_p / T_s``, 500x larger here, and every cross-waveform
    amplitude comparison would have to carry it.

    Asserted at an ON-GRID delay because that is where the identity is exact.
    Off the grid the sampled correlation loses ``O(T_s / T_p)`` to the partial
    samples at the pulse's two ends - measured 2.0e-3 in magnitude and 8.5e-3
    rad in argument at this grid - which is the straddle cost of a sampled
    receiver rather than a defect, and which the GPU sweep asserts separately.
    """

    spec = reference_spec(num_pulses=1)
    weight = 0.6 - 0.3j
    signal = _delayed_replica(spec, 100) * weight
    z = matched_filter(signal, spec)[0, 0]
    peak = complex(z[100])
    assert abs(peak) == pytest.approx(abs(weight), rel=1e-12)
    assert math.atan2(peak.imag, peak.real) == pytest.approx(math.atan2(weight.imag, weight.real), abs=1e-12)


def test_oversampling_interpolates_and_changes_nothing_on_the_original_grid():
    """Band-limited interpolation, not smoothing.

    ``oversample`` exists because a range cell can be a couple of samples wide,
    and a three-point parabolic fit on the raw grid then measures its own
    truncation error rather than the peak. It must not move the values it
    already had: every original lag reappears, bit for bit within float64.
    """

    spec = reference_spec(num_pulses=1)
    signal = _delayed_replica(spec, 100)
    coarse = matched_filter(signal, spec)[0, 0]
    fine = matched_filter(signal, spec, oversample=8)[0, 0]
    assert fine.shape[0] == 8 * coarse.shape[0]
    torch.testing.assert_close(fine[::8], coarse, rtol=1e-10, atol=1e-12 * float(coarse.abs().max()))
    assert float(lag_axis(spec, oversample=8)[8]) == pytest.approx(float(lag_axis(spec)[1]), rel=1e-12)


def test_the_matched_filter_refuses_a_signal_of_the_wrong_length():
    spec = reference_spec(num_pulses=1)
    with pytest.raises(ValueError, match="num_samples"):
        matched_filter(torch.zeros((1, 1, 7), dtype=torch.complex128), spec)
    with pytest.raises(ValueError, match="oversample"):
        matched_filter(_delayed_replica(spec, 10), spec, oversample=0)


def test_the_negative_lag_tail_does_not_wrap_onto_the_gate():
    """The transform is longer than the gate on purpose.

    The correlation's left tail is a full pulse width long. A circular
    correlation over ``num_samples`` alone would fold it onto the far end of the
    gate and invent an echo at long range that no target produced.
    """

    spec = reference_spec(num_pulses=1)
    signal = _delayed_replica(spec, 100)
    z = matched_filter(signal, spec)[0, 0].abs()
    # Everything beyond the peak plus one pulse width is structurally zero.
    beyond = z[100 + spec.pulse_sample_count + 2 :]
    assert float(beyond.max()) < 1e-9 * float(z.max())
