"""Launch and host budgets of the R-ADR-046/047/048 waveform families.

Every new synthesis route is one native forward launch per call and one
backward launch per backward, with no host observation; a refreshed pulsed
frame is one launch per replay group, not one per observation.
"""

from __future__ import annotations

import math

import pytest
import torch
from support import oracle_scenes as sc
from support.dsp_ledger import DspLedger

from witwin.radar import Chirp, Fmcw, Motion, Ofdm, PointTargets, Pulsed

pytestmark = pytest.mark.gpu

OPERATORS = (
    "fmcw_beat_forward",
    "fmcw_beat_backward",
    "fmcw_spectrum_forward",
    "ofdm_cfr_forward",
    "ofdm_time_forward",
    "ofdm_time_backward",
    "pulsed_echo_forward",
    "pulsed_echo_backward",
)


def _table():
    from witwin.radar.cuda import runtime

    return runtime.build_extension()


def _rows(paths: int = 3):
    tau = torch.linspace(1e-7, 2e-7, paths, device="cuda").requires_grad_(True)
    rate = torch.full((paths,), 1e-7, device="cuda").requires_grad_(True)
    weight = torch.ones(paths, dtype=torch.complex64, device="cuda").requires_grad_(True)
    offsets = torch.tensor((0, paths), dtype=torch.int64, device="cuda")
    return tau, rate, weight, offsets


def _specs():
    fmcw = Fmcw(
        slope=30e12,
        sample_rate=10e6,
        samples_per_chirp=32,
        chirps_per_frame=4,
        adc_start=1e-6,
        idle=5e-6,
        ramp_end=8e-6,
        output="beat",
        sequence=(Chirp(), Chirp(slope=45e12, start_offset=1e8)),
        phase_error=(0.0, 1e14),
    ).to_spec(carrier=sc.CARRIER_HZ)
    ofdm = Ofdm(
        subcarrier_spacing=1e6,
        num_subcarriers=32,
        cyclic_prefix=0.3e-6,
        num_symbols=4,
        max_expected_delay=0.25e-6,
        modulation="qam16",
        pilot_spacing=4,
    ).to_spec(carrier=sc.CARRIER_HZ)
    pmcw = Pulsed.pmcw(Pulsed.m_sequence(5), chip_rate=20e6, repetitions=4, sample_rate=20e6).to_spec(
        carrier=sc.CARRIER_HZ
    )
    return fmcw, ofdm, pmcw


def test_every_new_family_is_one_forward_and_one_backward_launch_with_no_host_read(monkeypatch):
    from witwin.radar.synthesis.fmcw import synthesize_fmcw_rows
    from witwin.radar.synthesis.ofdm import synthesize_ofdm_time_rows
    from witwin.radar.synthesis.pulsed import synthesize_echo_rows

    fmcw, ofdm, pmcw = _specs()
    table = _table()
    calls = (
        ("fmcw_beat", lambda t, r, w, o: synthesize_fmcw_rows(t, r, w, o, fmcw)),
        ("ofdm_time", lambda t, r, w, o: synthesize_ofdm_time_rows(t, r, w, o, ofdm)),
        ("pulsed_echo", lambda t, r, w, o: synthesize_echo_rows(t, r, w, o, pmcw)),
    )
    for family, call in calls:
        tau, rate, weight, offsets = _rows()
        call(tau, rate, weight, offsets)  # the per-spec tables upload once, here
        ledger = DspLedger(monkeypatch, OPERATORS, table)
        cube = call(tau, rate, weight, offsets)
        cube.abs().sum().backward()
        assert ledger.launches[f"{family}_forward"] == 1, ledger.launches
        assert ledger.launches[f"{family}_backward"] == 1, ledger.launches
        assert sum(ledger.launches.values()) == 2, ledger.launches
        assert ledger.host_observation_count == 0, ledger.host
        monkeypatch.undo()


def test_a_fast_time_pulsed_frame_is_one_launch_per_replay_group(monkeypatch):
    pytest.importorskip("witwin.channel")
    waveform = Pulsed(
        pulse_kind="lfm",
        pulse_width=2e-6,
        bandwidth=10e6,
        pri=10e-6,
        num_pulses=2,
        sample_rate=25e6,
        num_samples=100,
        range_gate_start=0.0,
        max_expected_delay_rate=1e-6,
    )
    radar = sc.radar(waveform)
    origin = torch.tensor([[3.0, 0.0, 0.0]], device=radar.device)
    targets = PointTargets(positions=origin, rcs=1.0, trajectory=lambda t: origin + 30.0 * t)
    table = _table()
    ledger = DspLedger(monkeypatch, OPERATORS, table)
    motion = Motion(kind="adc", batch_observations=64)
    result = radar.simulate(sc.empty_scene(), targets, times=(0.0,), los=True, reflections=0, motion=motion)
    observations = waveform.num_pulses * waveform.num_samples
    assert len(result.sample_times_s[0]) == observations
    assert ledger.launches["pulsed_echo_forward"] == math.ceil(observations / 64), ledger.launches
