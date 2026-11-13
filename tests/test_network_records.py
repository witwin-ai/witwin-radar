"""The multi-radar records on the host (R-ADR-051): ``Network`` and ``Emitter`` validation and refusals.

Everything here is decided at construction or before any world work, so it
runs on the CPU: each unsupported combination is refused by name, and the
emitter's sample table becomes the phase-linear chip table the native
families read.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from support import oracle_scenes as sc


def _fmcw(**fields):
    from witwin.radar import Fmcw

    values = {
        "slope": 30e12,
        "sample_rate": 10e6,
        "samples_per_chirp": 16,
        "chirps_per_frame": 2,
        "adc_start": 1e-6,
        "idle": 5e-6,
        "ramp_end": 10e-6,
        "output": "beat",
    }
    values.update(fields)
    return Fmcw(**values)


def _pulsed(**fields):
    from witwin.radar import Pulsed

    values = {
        "pulse_kind": "lfm",
        "pulse_width": 1e-6,
        "bandwidth": 10e6,
        "pri": 8e-6,
        "num_pulses": 2,
        "sample_rate": 20e6,
        "num_samples": 16,
        "range_gate_start": 0.0,
    }
    values.update(fields)
    return Pulsed(**values)


def _ofdm(**fields):
    from witwin.radar import Ofdm

    values = {
        "subcarrier_spacing": 1e6,
        "num_subcarriers": 16,
        "cyclic_prefix": 0.2e-6,
        "num_symbols": 2,
        "max_expected_delay": 0.1e-6,
    }
    values.update(fields)
    return Ofdm(**values)


def _radar(waveform, *, index=0, **fields):
    from witwin.radar import Radar

    values = {
        "reference_frequency_hz": sc.CARRIER_HZ,
        "waveform": waveform,
        "tx": ((0.01, 0.0, 0.0),),
        "rx": ((-0.01, 0.0, 0.0),),
        "power": 10.0,
        "position": (0.0, 0.5 * index, 0.0),
        "look_at": (1.0, 0.5 * index, 0.0),
        "up": sc.UP,
        "device": "cpu",
    }
    values.update(fields)
    return Radar(**values)


def _emitter(**fields):
    from witwin.radar import Emitter

    values = {"position": (5.0, 0.0, 0.0), "samples": (1.0, 1j), "rate": 1e6, "power": 0.0}
    values.update(fields)
    return Emitter(**values)


def test_a_network_normalises_its_declarations():
    from witwin.radar import Network

    radars = (_radar(_fmcw()), _radar(_fmcw(), index=1))
    network = Network(list(radars), links=[(0, 1)])
    assert network.radars == radars
    assert network.links == ((0, 1),)
    assert network.clock_offsets == (0.0, 0.0)
    assert network.carrier_offsets == (0.0, 0.0)
    assert network.if_bandwidths == (None, None)
    assert network.hearing() == (1,)
    assert Network(radars, emitters=(_emitter(),)).hearing() == (0, 1)
    assert network._same(Network(radars, links=((0, 1),)))
    assert not network._same(Network(radars))


@pytest.mark.parametrize(
    ("arguments", "error", "match"),
    [
        ({"radars": ()}, TypeError, "non-empty"),
        ({"links": ((0, 2),)}, ValueError, "indices below"),
        ({"links": ((0, 0),)}, ValueError, "names one radar twice"),
        ({"links": ((0, 1), (0, 1))}, ValueError, "declared twice"),
        ({"clock_offsets": (0.0,)}, ValueError, "clock_offsets holds"),
        ({"clock_offsets": (0.0, math.inf)}, ValueError, "finite"),
        ({"carrier_offsets": (0.0,)}, ValueError, "carrier_offsets holds"),
        ({"carrier_offsets": (1e3, 0.0)}, ValueError, "Chirp"),
        ({"if_bandwidths": (None,)}, ValueError, "if_bandwidths holds"),
        ({"if_bandwidths": (None, -1.0)}, ValueError, "positive"),
        ({"emitters": ("not an emitter",)}, TypeError, "Emitter"),
    ],
)
def test_a_network_refuses_a_malformed_declaration_by_name(arguments, error, match):
    from witwin.radar import Network

    values = {"radars": (_radar(_fmcw()), _radar(_fmcw(), index=1))}
    values.update(arguments)
    with pytest.raises(error, match=match):
        Network(**values)


def test_radars_of_a_network_share_a_carrier_a_device_and_a_pattern():
    from witwin.radar import Network, Pattern

    first = _radar(_fmcw())
    with pytest.raises(ValueError, match="one carrier"):
        Network((first, _radar(_fmcw(), index=1, reference_frequency_hz=78e9)))
    with pytest.raises(NotImplementedError, match="element pattern"):
        Network((first, _radar(_fmcw(), index=1, pattern=Pattern.dipole())))
    dipoles = (_radar(_fmcw(), pattern=Pattern.dipole()), _radar(_fmcw(), index=1, pattern=Pattern.dipole()))
    with pytest.raises(NotImplementedError, match="isotropic"):
        Network(dipoles, emitters=(_emitter(),))


@pytest.mark.parametrize(
    ("transmitter", "receiver", "match"),
    [
        ("fmcw", "pulsed", "FMCW ramp"),
        ("fmcw", "ofdm", "FMCW ramp"),
        ("pulsed", "fmcw", "dechirps"),
        ("ofdm", "fmcw", "dechirps"),
        ("ddm", "fmcw", "DDM/BPM"),
        ("sequence", "fmcw", "profile chirp sequence"),
        ("phase_error", "fmcw", "phase error"),
        ("wide", "pulsed", "anti-alias"),
        ("pulsed", "two_tx", "one transmitter"),
    ],
)
def test_a_link_the_cross_families_do_not_synthesize_is_refused_by_name(transmitter, receiver, match):
    from witwin.radar import Chirp, Network

    waveforms = {
        "fmcw": _fmcw(),
        "pulsed": _pulsed(),
        "ofdm": _ofdm(),
        "ddm": _fmcw(mimo="ddm"),
        "sequence": _fmcw(sequence=(Chirp(), Chirp(slope=20e12))),
        "phase_error": _fmcw(phase_error=(1e9,)),
        "wide": _pulsed(bandwidth=40e6),
        "two_tx": _pulsed(),
    }
    tx_fields = {"tx": ((0.01, 0.0, 0.0), (0.03, 0.0, 0.0))} if transmitter == "ddm" else {}
    tx = _radar(waveforms[transmitter], **tx_fields)
    rx_fields = {"tx": ((0.01, 0.0, 0.0), (0.03, 0.0, 0.0))} if receiver == "two_tx" else {}
    rx = _radar(waveforms[receiver], index=1, **rx_fields)
    with pytest.raises(NotImplementedError, match=match):
        Network((tx, rx), links=((0, 1),))


def test_links_the_families_synthesize_are_accepted():
    from witwin.radar import Chirp, Network

    Network((_radar(_fmcw(sequence=(Chirp(start_offset=2e6),))), _radar(_fmcw(), index=1)), links=((0, 1), (1, 0)))
    Network((_radar(_pulsed()), _radar(_ofdm(), index=1)), links=((0, 1), (1, 0)), carrier_offsets=(1e5, -1e5))
    Network((_radar(_pulsed()), _radar(_fmcw(), index=1)), emitters=(_emitter(),))
    with pytest.raises(NotImplementedError, match="anti-alias"):
        Network((_radar(_pulsed()),), emitters=(_emitter(carrier_offset=15e6),))


def test_a_receiver_with_oscillator_phase_noise_is_refused_when_it_hears_another_transmitter():
    from witwin.radar import Network, Noise

    noisy = _radar(_fmcw(), index=1, noise=Noise(figure=6.0, phase_density=-90.0, phase_offset=1e6))
    if not (noisy.frontend is not None and noisy.frontend.has_phase_noise):
        pytest.skip("this Noise spelling carries no phase noise")
    with pytest.raises(NotImplementedError, match="phase noise"):
        Network((_radar(_fmcw()), noisy), links=((0, 1),))


def test_an_emitter_validates_and_normalises_its_declaration():
    emitter = _emitter(polarization=(0.0, 3.0, 4.0), samples=[0.5, 0.5j])
    assert emitter.polarization == pytest.approx((0.0, 0.6, 0.8))
    assert emitter.samples == (0.5 + 0j, 0.5j)
    assert emitter.duration == pytest.approx(2e-6)
    for fields, error in (
        ({"samples": ()}, ValueError),
        ({"samples": (1.5,)}, ValueError),
        ({"rate": 0.0}, ValueError),
        ({"rate": math.nan}, ValueError),
        ({"polarization": (0.0, 0.0, 0.0)}, ValueError),
        ({"repeat": 1}, TypeError),
        ({"power": torch.tensor(0.0)}, TypeError),
    ):
        with pytest.raises(error):
            _emitter(**fields)


@pytest.mark.parametrize("repeat", [True, False])
def test_an_emitter_table_is_phase_linear_between_samples(repeat):
    rate, tone = 10e6, 1.3e6
    samples = 0.8 * np.exp(2j * math.pi * tone * np.arange(12) / rate)
    table = _emitter(samples=samples, rate=rate, repeat=repeat).chip_table()
    np.testing.assert_allclose(table[:, 0], 0.8)
    np.testing.assert_allclose(np.exp(2j * math.pi * table[:, 1]), samples / 0.8, atol=1e-12)
    # Each chip advances at the tone, so the held phase reaches the next sample exactly.
    np.testing.assert_allclose(table[:-1, 2], tone, rtol=1e-9)
    if not repeat:
        assert table[-1, 2] == pytest.approx(tone, rel=1e-9)
    silent = _emitter(samples=(0.0, 1.0)).chip_table()
    assert silent[0, 2] == 0.0


def test_a_session_refuses_what_a_network_does_not_schedule_before_any_world_work():
    from witwin.radar import Motion, Network, PointTargets

    network = Network((_radar(_fmcw()), _radar(_fmcw(), index=1)), links=((0, 1),))
    targets = PointTargets(positions=[[4.0, 0.0, 0.0]], rcs=1.0)
    scene = sc.empty_scene()
    with pytest.raises(NotImplementedError, match="adaptive"):
        network.simulate(scene, targets, times=(0.0,), motion=Motion.adaptive())
    with pytest.raises(NotImplementedError, match="rediscover_every_chirps"):
        network.simulate(scene, targets, times=(0.0,), motion=Motion.chirp(rediscover_every_chirps=2))
    with pytest.raises(ValueError, match="poses holds"):
        network.simulate(scene, targets, times=(0.0,), poses=(None,))
    with pytest.raises(TypeError, match="callable"):
        network.simulate(scene, targets, times=(0.0,), poses=(None, "pose"))
    with pytest.raises(TypeError, match="direct"):
        network.simulate(scene, targets, times=(0.0,), direct=1)
    with pytest.raises(TypeError, match="Motion"):
        network.simulate(scene, targets, times=(0.0,), motion="adc")
    with pytest.raises(ValueError, match="environment"):
        network.simulate(scene, targets, times=(0.0,), reflections=0, environment=True)
    with pytest.raises(NotImplementedError, match="transmission"):
        network.simulate(scene, targets, times=(0.0,), transmission=True)
    with pytest.raises(TypeError, match="NetworkPaths"):
        network.echo("paths")
