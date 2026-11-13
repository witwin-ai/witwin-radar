"""Every public record refuses a bad input at construction, by the field's name (M1-7, U-7, U-2).

Each case was accepted by its constructor before and failed, if at all, deep
inside a run with an error that named neither the record nor the field.
"""

from __future__ import annotations

import pytest
import torch

from witwin.radar import (
    Aspect,
    Chirp,
    Fluctuation,
    Fmcw,
    Iq,
    MeshTargets,
    Motion,
    Ofdm,
    PointTargets,
    Pulsed,
    Radar,
    ScatteringMatrix,
    StructureTargets,
)

RAMP = {
    "slope": 60e12,
    "sample_rate": 10e6,
    "samples_per_chirp": 64,
    "chirps_per_frame": 8,
    "adc_start": 1e-6,
    "idle": 5e-6,
    "ramp_end": 20e-6,
}
OFDM = {
    "subcarrier_spacing": 1e5,
    "num_subcarriers": 64,
    "cyclic_prefix": 1e-6,
    "num_symbols": 8,
    "max_expected_delay": 1e-7,
}
PULSE = {
    "pulse_width": 1e-6,
    "bandwidth": 1e6,
    "pri": 1e-4,
    "num_pulses": 8,
    "sample_rate": 1e7,
    "num_samples": 64,
    "range_gate_start": 0.0,
}
ONE = [(0.0, 0.0, -2.0)]
TWO = [(0.0, 0.0, -2.0), (0.0, 1.0, -2.0)]


def _radar(**fields):
    base = {
        "reference_frequency_hz": 77e9,
        "waveform": Fmcw(**RAMP),
        "tx": [(0, 0, 0)],
        "rx": [(0, 0, 0)],
        "power": 10.0,
    }
    return Radar(**{**base, "device": "cpu", **fields})


CASES = {
    # The plan's list (U-7).
    "rcs per target, wrong count": (lambda: PointTargets(positions=ONE, rcs=[1.0, 2.0]), "PointTargets.rcs"),
    "negative rcs": (lambda: PointTargets(positions=ONE, rcs=-1.0), "PointTargets.rcs"),
    "2-D position": (lambda: PointTargets(positions=[(0.0, 0.0)], rcs=1.0), "PointTargets.positions"),
    "slope as a string": (lambda: Fmcw(**{**RAMP, "slope": "60e12"}), "Fmcw.slope"),
    "ADC window past the ramp": (
        lambda: Fmcw(**{**RAMP, "sample_rate": 1e6, "samples_per_chirp": 4096, "ramp_end": 65e-6}),
        "Fmcw: the ADC window",
    ),
    "unknown pulse kind": (lambda: Pulsed(pulse_kind="barker", **PULSE), "Pulsed.pulse_kind"),
    "one adaptive node": (lambda: Motion.adaptive(nodes=1), "Motion.nodes"),
    # The rest of the surface.
    "negative rcs in a list": (lambda: PointTargets(positions=TWO, rcs=[1.0, -2.0]), r"PointTargets.rcs\[1\]"),
    "rcs as a string": (lambda: PointTargets(positions=ONE, rcs="1"), "PointTargets.rcs"),
    "positions tensor of the wrong shape": (
        lambda: PointTargets(positions=torch.zeros(2, 2), rcs=1.0),
        "PointTargets.positions",
    ),
    "no positions": (lambda: PointTargets(positions=[], rcs=1.0), "PointTargets.positions"),
    "repeated ids": (lambda: PointTargets(positions=TWO, rcs=1.0, ids=(3, 3)), "PointTargets.ids"),
    "too few ids": (lambda: PointTargets(positions=TWO, rcs=1.0, ids=(3,)), "PointTargets.ids"),
    "aspect axis count": (
        lambda: PointTargets(positions=TWO, rcs=1.0, aspect=Aspect(axis=[(0, 0, 1)], exponent=2, coherent_interval=1)),
        "Aspect.axis",
    ),
    "aspect exponent as a string": (
        lambda: Aspect(axis=[(0, 0, 1)], exponent="2", coherent_interval=1.0),
        "Aspect.exponent",
    ),
    "negative structure id": (lambda: StructureTargets(rcs=1.0, structure_ids=(1, -1)), "StructureTargets"),
    "fractional mesh structure id": (lambda: MeshTargets(structure_ids=(1.5,)), "MeshTargets.structure_ids"),
    "scattering factor as a string": (lambda: ScatteringMatrix(hh="1"), "ScatteringMatrix.hh"),
    "non-finite scattering axis": (lambda: ScatteringMatrix(up=(0, 0, float("nan"))), "ScatteringMatrix.up"),
    "fluctuation interval as a string": (
        lambda: Fluctuation(swerling=1, coherent_interval="1"),
        "Fluctuation.coherent_interval",
    ),
    "unknown MIMO scheme": (lambda: Fmcw(**{**RAMP, "mimo": "cdm"}), "Fmcw.mimo"),
    "negative DDM band count": (lambda: Fmcw(**{**RAMP, "ddm_bands": -1}), "Fmcw.ddm_bands"),
    "negative ADC start": (lambda: Fmcw(**{**RAMP, "adc_start": -1e-6}), "Fmcw.adc_start"),
    "chirp slope as a string": (lambda: Chirp(slope="1e12"), "Chirp.slope"),
    "negative chirp ramp": (lambda: Chirp(ramp_end=-1.0), "Chirp.ramp_end"),
    "negative chirp idle": (lambda: Chirp(idle=-1.0), "Chirp.idle"),
    "unknown OFDM modulation": (lambda: Ofdm(**OFDM, modulation="qam8"), "Ofdm.modulation"),
    "unknown OFDM output": (lambda: Ofdm(**OFDM, output="x"), "Ofdm.output"),
    "negative pilot spacing": (lambda: Ofdm(**OFDM, pilot_spacing=-1), "Ofdm.pilot_spacing"),
    "chips on a rectangular pulse": (lambda: Pulsed(pulse_kind="rect", chips=(1.0,), **PULSE), "Pulsed: only a coded"),
    "non-finite IQ gain": (lambda: Iq(gain=float("nan")), "Iq.gain"),
    "unknown world motion": (lambda: Motion(world="moving"), "Motion.world"),
    "waveform of the wrong type": (lambda: _radar(waveform="fmcw"), "Radar.waveform"),
    "BPM on three transmitters": (
        lambda: _radar(waveform=Fmcw(**{**RAMP, "mimo": "bpm"}), tx=[(0, 0, 0)] * 3),
        "Radar.waveform: BPM",
    ),
    "noise of the wrong type": (lambda: _radar(noise=3.0), "Radar.noise"),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_a_bad_record_is_refused_at_construction_by_name(case):
    build, name = CASES[case]
    with pytest.raises((TypeError, ValueError), match=name):
        build()


def test_a_radar_off_cuda_is_refused_by_its_device_before_any_work(monkeypatch):
    """U-2: a CPU radar used to reach Channel and fail on 'endpoint tensors must be CUDA tensors'."""

    from witwin.core import Scene

    def no_session(*args, **kwargs):
        raise AssertionError("the session was opened before the device was checked")

    monkeypatch.setattr("witwin.radar.simulation.simulate_scene", no_session)
    monkeypatch.setattr("witwin.radar.simulation.trace_scene", no_session)
    radar = _radar()
    for verb in (radar.simulate, radar.trace):
        with pytest.raises(RuntimeError, match="this Radar is on device 'cpu'"):
            verb(Scene(), PointTargets(positions=ONE, rcs=1.0), times=(0.0,))
    with pytest.raises(RuntimeError, match="this Radar is on device 'cpu'"):
        radar.echo(object())


def test_the_fmcw_derived_quantities_follow_their_definitions():
    from witwin.radar.policy import SPEED_OF_LIGHT_M_PER_S as c

    ramp = Fmcw(**RAMP)
    sweep = 60e12 * 64 / 10e6
    assert ramp.sweep_bandwidth == pytest.approx(sweep, rel=1e-15)
    assert ramp.range_resolution == pytest.approx(c / (2.0 * sweep), rel=1e-15)
    assert ramp.max_range == pytest.approx(10e6 * c / (2.0 * 60e12), rel=1e-15)
    wavelength = c / 77e9
    assert ramp.max_velocity(reference_frequency_hz=77e9, num_tx=2) == pytest.approx(
        wavelength / (4.0 * 2 * 25e-6), rel=1e-15
    )


def test_a_network_of_cpu_radars_is_refused_by_device_after_its_own_argument_checks():
    from witwin.core import Scene

    from witwin.radar import Network

    network = Network((_radar(), _radar(position=(1.0, 0.0, 0.0))))
    with pytest.raises(RuntimeError, match="this Radar is on device 'cpu'"):
        network.simulate(Scene(), PointTargets(positions=ONE, rcs=1.0), times=(0.0,))
